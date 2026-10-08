"""The rendering backends behind ``/v1/images/generations``.

Two engines implement one small interface:

``DiffusersEngine``
    Runs Qwen-Image-2.1 through ``QwenImage21Pipeline`` from diffusers, with the
    text encoder and the tokenizer coming from transformers. This is the CPU
    path the service is built around, and the same code runs on a CUDA device.

``StubEngine``
    Draws a deterministic placeholder. It exists so the HTTP contract, the
    gateway integration and the documentation can be exercised on a machine
    that cannot hold 33 GB of weights; it is never a production backend.

Both are single-flight on purpose: one diffusion pipeline saturates an 8-core
CPU, and the pipeline object is not safe to call from two threads at once.
"""

from __future__ import annotations

import io
import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .config import Settings

log = logging.getLogger("imageint.image.engine")


class EngineError(RuntimeError):
    """Raised when the engine cannot load or cannot render."""


@dataclass(frozen=True)
class RenderRequest:
    """One normalised render job."""

    prompt: str
    width: int
    height: int
    steps: int
    true_cfg_scale: float
    negative_prompt: str = ""
    seed: Optional[int] = None

    def as_meta(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "true_cfg_scale": self.true_cfg_scale,
        }


class BaseEngine:
    """Loading state, locking and status reporting shared by both engines."""

    #: Value of the ``engine`` field in every response.
    name = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._load_lock = threading.Lock()
        self._render_lock = threading.Lock()
        # idle -> loading -> ready | error. Read without the lock from the
        # health endpoint; a plain attribute read is atomic enough here.
        self._state = "idle"
        self._error = ""
        self._load_seconds = 0.0
        self._loaded_at = 0.0
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        """Begin loading in the background so ``/health`` can report progress."""

        if not self.settings.preload:
            log.info("[image] preload disabled, loading on first request")
            return
        self._thread = threading.Thread(
            target=self.ensure_loaded, name="imageint-image-load", daemon=True
        )
        self._thread.start()

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until the engine is ready or the timeout expires."""

        if self._thread is None:
            return self._state == "ready"
        self._thread.join(timeout)
        return self._state == "ready"

    def ensure_loaded(self) -> None:
        """Load the weights once, no matter how many callers ask for it."""

        if self._state == "ready":
            return
        with self._load_lock:
            if self._state == "ready":
                return
            self._state = "loading"
            self._error = ""
            started = time.monotonic()
            log.info("[image] loading %s on %s", self.settings.model, self.settings.device)
            try:
                self._load()
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                self._state = "error"
                self._error = f"{type(exc).__name__}: {exc}"
                log.exception("[image] loading %s failed", self.settings.model)
                return
            self._load_seconds = time.monotonic() - started
            self._loaded_at = time.time()
            self._state = "ready"
            log.info(
                "[image] %s ready after %.1f s", self.settings.model, self._load_seconds
            )

    # -- rendering --------------------------------------------------------- #

    def generate(self, request: RenderRequest) -> tuple[bytes, dict]:
        """Render one picture and return ``(png_bytes, meta)``."""

        self.ensure_loaded()
        if self._state != "ready":
            raise EngineError(self._error or "Das Bildmodell ist nicht geladen.")
        with self._render_lock:
            return self._render(request)

    def _load(self) -> None:
        raise NotImplementedError

    def _render(self, request: RenderRequest) -> tuple[bytes, dict]:
        raise NotImplementedError

    def close(self) -> None:
        """Release whatever the engine holds. Called on application shutdown."""

        self._state = "idle"
        self._error = ""

    # -- reporting --------------------------------------------------------- #

    def status(self) -> dict:
        return {
            "engine": self.name,
            "model": self.settings.model,
            "device": self.settings.device,
            "dtype": self.settings.dtype,
            "quant": self.settings.quant,
            "state": self._state,
            "loaded": self._state == "ready",
            "loading": self._state in ("idle", "loading"),
            "error": self._error,
            "load_seconds": round(self._load_seconds, 1),
            "loaded_at": self._loaded_at,
        }

    # -- helpers ----------------------------------------------------------- #

    def _pick_seed(self, requested: Optional[int]) -> int:
        if requested is not None:
            return int(requested) % (2**31)
        return random.SystemRandom().randrange(0, 2**31)


class DiffusersEngine(BaseEngine):
    """Qwen-Image-2.1 through diffusers + transformers."""

    name = "diffusers"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._pipe = None

    def _torch(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - container always has it
            raise EngineError(
                "torch ist nicht installiert; dieser Server braucht torch, "
                "diffusers und transformers."
            ) from exc
        return torch

    def _load(self) -> None:
        torch = self._torch()
        try:
            import diffusers
        except ImportError as exc:  # pragma: no cover - container always has it
            raise EngineError("diffusers ist nicht installiert.") from exc

        pipeline_cls = getattr(diffusers, "QwenImage21Pipeline", None)
        if pipeline_cls is None:
            raise EngineError(
                "Diese diffusers-Version kennt QwenImage21Pipeline nicht. "
                "Qwen-Image-2.1 braucht diffusers>=0.41.0."
            )

        if self.settings.cpu_threads:
            torch.set_num_threads(self.settings.cpu_threads)

        dtype = getattr(torch, self.settings.dtype, None)
        if dtype is None:
            raise EngineError(f"Unbekannter dtype: {self.settings.dtype}")

        kwargs: dict = {"torch_dtype": dtype}
        if self.settings.revision:
            kwargs["revision"] = self.settings.revision
        if self.settings.cache_dir:
            kwargs["cache_dir"] = self.settings.cache_dir

        quant_config = self._quantization_config(diffusers)
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config

        self._pipe = pipeline_cls.from_pretrained(self.settings.model, **kwargs)
        self._pipe.to(self.settings.device)
        # The progress bar writes to stderr and the gateway never reads it.
        self._pipe.set_progress_bar_config(disable=True)

    def close(self) -> None:
        # bf16 weights are tens of gigabytes; dropping the reference is the
        # fastest way to hand the memory back on shutdown.
        self._pipe = None
        super().close()

    def _quantization_config(self, diffusers):
        """Translate ``IMAGEINT_IMAGE_QUANT`` into a diffusers config."""

        quant = self.settings.quant
        if quant == "none":
            return None
        try:
            from diffusers.quantizers.quantization_config import QuantoConfig
        except ImportError as exc:
            raise EngineError(
                f"Quantisierung {quant} braucht optimum-quanto "
                "(pip install optimum-quanto)."
            ) from exc
        return QuantoConfig(weights_dtype=quant)

    def _render(self, request: RenderRequest) -> tuple[bytes, dict]:
        torch = self._torch()
        seed = self._pick_seed(request.seed)
        generator_device = "cuda" if self.settings.device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        kwargs: dict = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.steps,
            "true_cfg_scale": request.true_cfg_scale,
            "generator": generator,
            "use_kv_cache": self.settings.use_kv_cache,
        }
        # The pipeline ignores a negative prompt while guidance is off, so it is
        # only passed when it can actually change the result.
        negative = (request.negative_prompt or self.settings.negative_prompt).strip()
        if negative and request.true_cfg_scale > 1.0:
            kwargs["negative_prompt"] = negative

        started = time.monotonic()
        with torch.inference_mode():
            result = self._pipe(**kwargs)
        latency_ms = int(round((time.monotonic() - started) * 1000))

        images = getattr(result, "images", None) or []
        if not images:
            raise EngineError("Die Pipeline hat kein Bild zurückgegeben.")

        buffer = io.BytesIO()
        images[0].save(buffer, format="PNG")
        return buffer.getvalue(), {"seed": seed, "latency_ms": latency_ms}


class StubEngine(BaseEngine):
    """Deterministic placeholder renderer for tests and documentation."""

    name = "stub"

    def _load(self) -> None:
        # Imported here so a production container without Pillow still starts.
        try:
            import PIL  # noqa: F401
        except ImportError as exc:  # pragma: no cover - Pillow ships with diffusers
            raise EngineError("Pillow ist nicht installiert.") from exc

    def _render(self, request: RenderRequest) -> tuple[bytes, dict]:
        from PIL import Image, ImageDraw

        seed = self._pick_seed(request.seed)
        rng = random.Random(seed)

        if self.settings.stub_delay:
            time.sleep(self.settings.stub_delay)

        started = time.monotonic()
        top = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        bottom = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
        image = Image.new("RGB", (request.width, request.height))
        draw = ImageDraw.Draw(image)

        height = max(1, request.height)
        for y in range(height):
            blend = y / height
            colour = tuple(
                int(top[i] + (bottom[i] - top[i]) * blend) for i in range(3)
            )
            draw.line([(0, y), (request.width, y)], fill=colour)

        # A few blocks so the placeholder is visibly a rendering and not a
        # broken image; the prompt decides the layout.
        for _ in range(12):
            x0 = rng.randrange(request.width)
            y0 = rng.randrange(height)
            size = rng.randrange(40, max(41, request.width // 6))
            draw.rectangle(
                [x0, y0, x0 + size, y0 + size],
                outline=(255, 255, 255),
                width=max(2, request.width // 512),
            )

        label = f"ImageInt stub {request.width}x{request.height} seed {seed}"
        draw.text((16, 16), label, fill=(255, 255, 255))
        draw.text((16, 40), request.prompt[:120], fill=(255, 255, 255))

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        latency_ms = int(round((time.monotonic() - started) * 1000))
        return buffer.getvalue(), {"seed": seed, "latency_ms": latency_ms}


def build_engine(settings: Settings) -> BaseEngine:
    """Instantiate the engine named by ``IMAGEINT_IMAGE_ENGINE``."""

    if settings.engine == "stub":
        return StubEngine(settings)
    return DiffusersEngine(settings)
