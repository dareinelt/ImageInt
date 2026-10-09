"""Configuration of the image server.

Like the gateway, the server is configured entirely through the environment, so
one ``.env`` file describes a whole deployment. The names are the same ones the
gateway reports under ``GET /v1/config``; where a value has to be identical on
both sides (``IMAGEINT_IMAGE_MODEL``, ``IMAGEINT_IMAGE_QUANT``) it is read from
the same variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: The checkpoint. Qwen-Image-2.1 is a unified text-to-image / image-editing
#: model whose visual component is a 7B DiT; the pipeline that drives it is
#: ``QwenImage21Pipeline``.
DEFAULT_MODEL = "Qwen/Qwen-Image-2.1"

#: Weight formats. bfloat16 is what the Qwen-Image-2.1 model card documents and
#: therefore the default. float32 is the escape hatch for a CPU without bf16
#: support; it doubles the resident size to roughly 66 GB and is not the
#: reference configuration. float16 is accepted because diffusers offers it, but
#: is a poor fit for a CPU denoiser.
DTYPES = ("bfloat16", "float32", "float16")

#: Devices the pipeline can be placed on. ``cpu`` is the requirement of this
#: service, ``cuda`` is the prepared NVIDIA path, ``mps`` exists so the server
#: can be run natively on an Apple Silicon development machine.
DEVICES = ("cpu", "cuda", "mps")

#: Quantisation of the denoiser. ``none`` keeps the dtype above and is the
#: required production setting; ``int8`` and ``int4`` are the small-footprint
#: modes for a test machine, applied through optimum-quanto, which is the
#: quantisation backend that works on a CPU.
QUANTS = ("none", "int8", "int4")

#: Rendering backends. ``diffusers`` runs the real checkpoint. ``stub`` renders a
#: deterministic placeholder and exists so the API, the gateway and the
#: documentation can be exercised without a 33 GB download.
ENGINES = ("diffusers", "stub")

#: Largest canvas in pixels. The default is the largest of the seven aspect
#: ratios the model card documents (2400x1792); 16:9 (2752x1536) fits under it.
DEFAULT_MAX_PIXELS = 4300800

#: Side length every width and height is snapped to. The largest common divisor
#: of the seven aspect ratios the model card documents is 32 (2400 and 1696 are
#: 75 and 53 times 32), so a multiple of 32 is the coarsest grid that leaves
#: every documented canvas untouched. It also covers the 16x16 patch grid of
#: the DiT and the 8x8 one of the VAE.
MULTIPLE = 32

#: Smallest and largest accepted side, in pixels.
MIN_SIDE = 256
MAX_SIDE = 8192


def _text(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


def _number(name: str, default: int, low: int, high: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(float(raw))
    except ValueError:
        return default
    return max(low, min(high, value))


def _decimal(name: str, default: float, low: float, high: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(low, min(high, value))


def _flag(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off", "nein")


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = (os.environ.get(name) or "").strip().lower()
    return value if value in allowed else default


def _sizes() -> tuple[tuple[int, int], ...]:
    """The aspect ratios the model card documents, as (width, height) pairs."""

    return (
        (2048, 2048),
        (2400, 1792),
        (1792, 2400),
        (2528, 1696),
        (1696, 2528),
        (2752, 1536),
        (1536, 2752),
    )


DOCUMENTED_SIZES = _sizes()


@dataclass(frozen=True)
class Settings:
    """Everything the server reads from the environment."""

    model: str = DEFAULT_MODEL
    #: Optional commit, tag or branch of the checkpoint repository.
    revision: str = ""
    dtype: str = "bfloat16"
    device: str = "cpu"
    quant: str = "none"
    engine: str = "diffusers"
    #: Shared secret. When set, every request except ``/health`` must carry it as
    #: ``X-Auth-Token`` or ``Authorization: Bearer``.
    token: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    #: Load the weights during startup instead of on the first request. Keeping
    #: this true is what makes the gateway's loading contract meaningful: the
    #: process is up, ``/health`` answers 503, and the model becomes available
    #: minutes later without a request ever timing out on a cold cache.
    preload: bool = True
    #: Torch intra-op threads. 0 leaves torch's own default, which is one thread
    #: per core -- the right value on the 8-core reference machine.
    cpu_threads: int = 0
    #: Denoising steps and guidance. The model card documents 40 steps and
    #: sampling *without* guidance, hence the true-CFG default of 1.0.
    steps: int = 40
    true_cfg_scale: float = 1.0
    #: Negative prompt applied when a request carries none. The Qwen-Image-2.1
    #: enhancer only produces a positive prompt, so this is normally empty.
    negative_prompt: str = ""
    #: Largest canvas in pixels; a larger request is rejected rather than
    #: silently scaled, because the gateway already applies its own budget.
    max_pixels: int = DEFAULT_MAX_PIXELS
    #: Prefix KV cache of the DiT. Saves a lot of work on a CPU host and is on by
    #: default in the pipeline as well.
    use_kv_cache: bool = True
    #: Upper length of an accepted prompt.
    max_prompt_chars: int = 4000
    #: Artificial delay of the stub engine, in seconds, so a slow render can be
    #: demonstrated without a model.
    stub_delay: float = 0.0
    #: Path of the model cache. ``HF_HOME`` is the usual way to set this; the
    #: variable exists so the container can be pointed at a mounted volume.
    cache_dir: str = ""

    @property
    def auth_required(self) -> bool:
        return self.token != ""

    @property
    def torch_dtype(self) -> str:
        """The dtype actually handed to ``from_pretrained``."""

        return self.dtype


def load_settings() -> Settings:
    """Build a :class:`Settings` from the current environment."""

    return Settings(
        model=_text("IMAGEINT_IMAGE_MODEL", DEFAULT_MODEL),
        revision=_text("IMAGEINT_IMAGE_REVISION", ""),
        dtype=_choice("IMAGEINT_IMAGE_DTYPE", "bfloat16", DTYPES),
        device=_choice("IMAGEINT_IMAGE_DEVICE", "cpu", DEVICES),
        quant=_choice("IMAGEINT_IMAGE_QUANT", "none", QUANTS),
        engine=_choice("IMAGEINT_IMAGE_ENGINE", "diffusers", ENGINES),
        token=_text("IMAGEINT_IMAGE_TOKEN", ""),
        host=_text("IMAGEINT_IMAGE_HOST", "0.0.0.0"),
        port=_number("IMAGEINT_IMAGE_PORT", 8000, 1, 65535),
        preload=_flag("IMAGEINT_IMAGE_PRELOAD", True),
        cpu_threads=_number("IMAGEINT_IMAGE_CPU_THREADS", 0, 0, 1024),
        steps=_number("IMAGEINT_IMAGE_STEPS", 40, 1, 200),
        true_cfg_scale=_decimal("IMAGEINT_IMAGE_TRUE_CFG_SCALE", 1.0, 0.0, 20.0),
        negative_prompt=_text("IMAGEINT_IMAGE_NEGATIVE_PROMPT", ""),
        max_pixels=_number(
            "IMAGEINT_IMAGE_MAX_PIXELS", DEFAULT_MAX_PIXELS, MIN_SIDE * MIN_SIDE, 10**9
        ),
        use_kv_cache=_flag("IMAGEINT_IMAGE_KV_CACHE", True),
        max_prompt_chars=_number("IMAGEINT_IMAGE_MAX_PROMPT_CHARS", 4000, 1, 100000),
        stub_delay=_decimal("IMAGEINT_IMAGE_STUB_DELAY", 0.0, 0.0, 3600.0),
        cache_dir=_text("IMAGEINT_IMAGE_CACHE_DIR", ""),
    )


def snap(value: int) -> int:
    """Round a side length to the patch grid the DiT and the VAE expect.

    The grid is 32, which is the largest common divisor of the documented
    aspect ratios -- so every canvas from the model card survives this
    unchanged and only off-grid requests are moved.
    """

    snapped = int(round(value / MULTIPLE)) * MULTIPLE
    return max(MIN_SIDE, min(MAX_SIDE, snapped))
