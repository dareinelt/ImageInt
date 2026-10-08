"""One generation request, end to end.

The pipeline is deliberately small and linear, because every step is observable
from outside:

1. **enhance** -- the documented Qwen-Image-2.1 prompt enhancer rewrites the
   request and picks the canvas ratio. Skipped only when the caller asks for it
   (``enhance: false``) or the enhancer is disabled; the model card treats it as
   part of the pipeline, so it is on by default.
2. **render** -- Qwen-Image-2.1 draws the picture, addressed through the OpenAI
   Images API of the image server (diffusers + transformers on the CPU).
3. **store** -- the PNG is written once under its job id, so a chat can fetch it
   more than once without paying for another render.

Every stage writes into the job before it finishes, so a client polling
``GET /v1/jobs/{id}`` can see which half of the wait it is in -- the enhancer
alone can think for minutes on a CPU host, and "still enhancing" is a much more
useful answer than "still running".
"""

from __future__ import annotations

import logging
import time
from typing import Any, Mapping

from . import enhancer as enhancer_module
from . import ratios, vllm
from .config import DEFAULT_IMAGE_SIZE, Settings
from .errors import ImageIntError
from .jobs import DONE, ERROR, Job
from .storage import ImageStore

log = logging.getLogger("imageint.pipeline")


async def run_generation(job: Job, settings: Settings, store: ImageStore) -> None:
    """Fill in one job. Never raises: failures land in the job itself."""

    started = time.monotonic()
    request = job.request

    try:
        prompt, width, height, negative_prompt, seed = await _enhance(job, settings)

        job.enhanced_prompt = prompt
        job.negative_prompt = negative_prompt
        job.width = width
        job.height = height
        job.seed = seed
        job.stage = "rendering"

        payload = vllm.build_image_payload(
            settings.image,
            prompt,
            width,
            height,
            negative_prompt=negative_prompt,
            steps=int(request.get("steps") or 0) or None,
            seed=seed,
        )

        result = await vllm.generate_image(settings.image, payload)
        job.timings["render_ms"] = result["latency_ms"]
        job.model = result["model"]

        path = store.write(job.id, result["image"], result["content_type"])
        if path is None:
            raise ImageIntError(500, "storage_failed", "Das Bild konnte nicht gespeichert werden.")

        job.image_bytes = len(result["image"])
        job.content_type = result["content_type"]
        # The model reports its own canvas; prefer it over the requested one, so
        # the metadata describes the file that actually exists.
        if result["width"]:
            job.width = result["width"]
        if result["height"]:
            job.height = result["height"]
        if result["seed"] is not None:
            job.seed = result["seed"]

        job.status = DONE
        job.stage = "done"
    except ImageIntError as exc:
        job.status = ERROR
        job.stage = "error"
        job.error_code = exc.code
        job.error_message = exc.message
        log.warning("Bildauftrag %s fehlgeschlagen: %s (%s)", job.id, exc.code, exc.message)
    except Exception as exc:  # noqa: BLE001 - the registry logs and reports it
        job.status = ERROR
        job.stage = "error"
        job.error_code = "internal_error"
        job.error_message = str(exc) or exc.__class__.__name__
        log.exception("Bildauftrag %s ist unerwartet gescheitert.", job.id)
    finally:
        job.timings["total_ms"] = int(round((time.monotonic() - started) * 1000))


def wants_enhancer(request: Mapping[str, Any], settings: Settings) -> bool:
    """Whether this request should run the documented prompt enhancer.

    ``enhance`` is tri-state on purpose. Left out, the request follows the
    operator's ``IMAGEINT_PE_ENABLED``; set explicitly, it overrides it. A
    client that only ever sends a prompt -- LLMInt does exactly that -- then
    gets the prompt used verbatim on a host where the enhancer was switched off,
    instead of an error it cannot do anything about.
    """

    explicit = request.get("enhance")
    if explicit is None:
        return settings.enhancer.enabled
    return bool(explicit)


async def _enhance(job: Job, settings: Settings) -> tuple:
    """Return ``(prompt, width, height, negative_prompt, seed)`` for one job."""

    request = job.request
    seed = request.get("seed")
    negative_prompt = str(request.get("negative_prompt") or "")
    requested = _requested_size(request, settings)

    if not wants_enhancer(request, settings):
        # Without the enhancer nothing picks a ratio, so the caller's size or the
        # documented native canvas is used.
        size = requested if requested[0] else DEFAULT_IMAGE_SIZE
        job.stage = "rendering"
        job.parse_ok = None
        return job.prompt, size[0], size[1], negative_prompt, seed

    job.stage = "enhancing"
    result = await enhancer_module.enhance(
        settings.enhancer, job.prompt, max_pixels=settings.image.max_pixels
    )
    job.timings["enhance_ms"] = result["latency_ms"]
    job.enhancer_model = result["model"]
    job.parse_ok = result["parse_ok"]
    job.wh_ratio = result["wh_ratio"]

    # An explicit canvas always wins over the ratio the enhancer picked; a
    # caller that names a size means it.
    width, height = (result["width"], result["height"]) if not requested[0] else requested

    if not negative_prompt:
        negative_prompt = result["negative_prompt"]

    return result["prompt"], width, height, negative_prompt, seed


def _requested_size(request: dict, settings: Settings) -> tuple:
    """Explicit canvas from the request: ``width``/``height``, ``size`` or ``ratio``.

    Returns ``(0, 0)`` when the request names no canvas, which lets the enhancer
    decide the ratio.
    """

    width = int(request.get("width") or 0)
    height = int(request.get("height") or 0)
    if width > 0 and height > 0:
        return ratios.parse_size(
            f"{width}x{height}", settings.image.max_pixels
        ) or ratios.size_for(f"{width}:{height}", settings.image.max_pixels)

    size = str(request.get("size") or "").strip()
    if size:
        return ratios.parse_size(size, settings.image.max_pixels) or ratios.size_for(
            size, settings.image.max_pixels
        )

    ratio = str(request.get("ratio") or "").strip()
    if ratio:
        return ratios.size_for(ratio, settings.image.max_pixels)

    return (0, 0)
