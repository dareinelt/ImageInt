"""The ImageInt HTTP API.

ImageInt is the image endpoint of the LLMInt chat: it takes a prompt from the
text model's ``generate_image`` tool call and returns a picture. The API is
deliberately small and shaped like the OpenAI Images API, so the LLMInt client
can keep its existing request builder.

Endpoints
---------
``GET  /``                       service banner
``GET  /health``                 liveness for the container health check
``GET  /v1/ready``               readiness, unauthenticated (200 or 503)
``GET  /v1/health``              readiness, authenticated; same body as /v1/ready
``GET  /v1/config``              effective configuration, tokens masked
``GET  /v1/models``              the two model servers and their state
``GET  /v1/ratios``              the canvas table of the model card
``POST /v1/enhance``             only the prompt enhancer (debugging, admin test)
``POST /v1/images/generations``  the actual endpoint: generate an image
``GET  /v1/jobs``                recent jobs
``GET  /v1/jobs/{id}``           state of one job
``GET  /v1/jobs/{id}/image``     the finished image
``GET  /v1/images/{id}``         the finished image (the URL reported to clients)

Everything except ``/health`` and ``/v1/ready`` requires the shared token, as
``X-Auth-Token`` or ``Authorization: Bearer``.

Because a CPU render takes minutes, ``POST /v1/images/generations`` waits only
``IMAGEINT_SYNC_TIMEOUT`` seconds and then answers **202** with a job id; the
client polls ``GET /v1/jobs/{id}`` and finally fetches ``GET /v1/images/{id}``.
On a fast (or GPU) host the same call returns the finished image directly, so a
simple client needs no polling at all.
"""

from __future__ import annotations

import base64
import hmac
import logging
import time
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from . import __version__, enhancer as enhancer_module, host, pipeline, ratios
from .config import IMAGE_ROUTE_PATHS, Settings, get_settings
from .errors import ImageIntError, bad_request, not_found, payload_too_large, unauthorized
from .health import COMPONENTS, HealthMonitor
from .jobs import DONE, ERROR, JobRegistry
from .storage import ImageStore

log = logging.getLogger("imageint")

#: ``Retry-After`` sent with a 202: how long the client should wait before asking
#: for the job again. Same value as the loading hint -- both mean "come back in a
#: moment", and a CPU render takes minutes, so a slower poll costs nothing.
POLL_AFTER = 15


class Runtime:
    """Everything the request handlers need, built once at startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = ImageStore(settings.storage_dir)
        self.health = HealthMonitor(settings)
        self.registry = JobRegistry(
            settings,
            self.store,
            partial(pipeline.run_generation, settings=settings, store=self.store),
        )
        self.started_at = time.time()


_runtime: Optional[Runtime] = None


def runtime() -> Runtime:
    """The process-wide runtime; the lifespan hook must have run first."""

    if _runtime is None:  # pragma: no cover - only before startup
        raise RuntimeError("ImageInt runtime is not initialised")
    return _runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the runtime, report the host verdict, and clean up on shutdown."""

    global _runtime
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("ImageInt %s startet (Gateway auf Port 8080).", __version__)

    # The host verdict comes first: on an unsupported machine nothing else here
    # is worth trying, and the administrator should see it before the long model
    # download rather than after it.
    host.log_sizing()
    for blocker in host.blocking():
        log.error(
            "Dieser Host kann keine Bilder erzeugen. Der Dienst antwortet auf "
            "Generierungsanfragen mit HTTP 503 host_unsupported. Grund: %s",
            blocker,
        )

    _runtime = Runtime(settings)
    _runtime.store.ensure()
    # A restart loses the in-memory registry, so files from the previous run are
    # orphans by definition; removing them keeps the volume from growing.
    removed = _runtime.registry.cleanup_store()
    log.info(
        "Bildspeicher %s bereit (%d verwaiste Datei(en) entfernt).",
        settings.storage_dir,
        removed,
    )
    log.info(
        "Generierung: %d gleichzeitig, %d in der Warteschlange, Sync-Timeout %d s, "
        "Aufträge %d h / %d Stück.",
        settings.max_concurrent_jobs,
        settings.max_queued_jobs,
        settings.sync_timeout_seconds,
        settings.job_retention_seconds // 3600,
        settings.max_jobs,
    )

    try:
        yield
    finally:
        if _runtime is not None:
            await _runtime.registry.shutdown()
            log.info("ImageInt beendet sich.")
        _runtime = None


app = FastAPI(
    title="ImageInt",
    version=__version__,
    summary="Bilderzeugung für LLMInt (vLLM + Qwen-Image-2.1)",
    description=(
        "Eigenständiger Bild-Endpunkt für LLMInt. Ersetzt die ComfyUI- und "
        "AUTOMATIC1111-Integration und übernimmt deren Tool-Namen generate_image. "
        "Prompt-Enhancement und Bilderzeugung laufen auf Qwen-Image-2.1, "
        "serviert von vLLM im CPU-Betrieb."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
async def require_token(request: Request) -> None:
    """Accept the shared secret as ``X-Auth-Token`` or ``Authorization: Bearer``.

    An empty configured token disables the check, which is what a deployment on
    a private compose network wants; the public URL must always set one.
    """

    token = runtime().settings.token
    if not token:
        return

    provided = request.headers.get("x-auth-token") or ""
    if not provided:
        header = request.headers.get("authorization") or ""
        if header.lower().startswith("bearer "):
            provided = header[7:].strip()

    # Constant-time: the token is a shared secret and a timing side channel here
    # would be free to exploit.
    if not provided or not hmac.compare_digest(provided, token):
        raise unauthorized()


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class GenerationRequest(BaseModel):
    """Body of ``POST /v1/images/generations``.

    ``prompt`` is the only required field. Everything else exists so the caller
    can override a decision the enhancer would otherwise make -- and so the
    LLMInt client, which only ever sends a prompt, needs no changes.
    """

    prompt: str = Field(..., description="Der Bildwunsch, z. B. 'Ein roter Panda im Schnee'.")
    size: str = Field("", description="Ausdrückliche Größe wie '1024x1024' oder ein Verhältnis wie '16:9'.")
    ratio: str = Field("", description="Seitenverhältnis, falls kein size angegeben ist.")
    width: int = Field(0, ge=0, le=8192, description="Breite in Pixeln (0 = vom Enhancer bestimmen).")
    height: int = Field(0, ge=0, le=8192, description="Höhe in Pixeln (0 = vom Enhancer bestimmen).")
    enhance: Optional[bool] = Field(
        None,
        description=(
            "Den dokumentierten Prompt-Enhancer verwenden. Weggelassen gilt die "
            "Einstellung des Betreibers (IMAGEINT_PE_ENABLED)."
        ),
    )
    steps: int = Field(0, ge=0, le=200, description="Denoising-Schritte (0 = Standard).")
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1, description="Seed für reproduzierbare Bilder.")
    negative_prompt: str = Field("", description="Negativ-Prompt; überschreibt den des Enhancers.")
    wait: bool = Field(True, description="Bis IMAGEINT_SYNC_TIMEOUT auf das Bild warten, sonst sofort 202.")
    include_image: bool = Field(True, description="Das fertige Bild als base64 in die Antwort legen.")


class EnhanceRequest(BaseModel):
    """Body of ``POST /v1/enhance``."""

    prompt: str = Field(..., description="Der umzuschreibende Bildwunsch.")


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
#: Job error codes mapped to the HTTP status the client should see when the
#: failure is reported back through the synchronous path.
_STATUS_BY_CODE = {
    "empty_prompt": 400,
    "bad_request": 400,
    "storage_failed": 500,
    "not_configured": 503,
    "service_loading": 503,
    "busy": 503,
    "enhancer_timeout": 504,
    "image_timeout": 504,
    "enhancer_error": 502,
    "image_error": 502,
    "enhancer_unavailable": 502,
    "image_unavailable": 502,
    "image_empty": 502,
    "cancelled": 503,
}


@app.exception_handler(ImageIntError)
async def imageint_error_handler(request: Request, exc: ImageIntError) -> JSONResponse:
    """Answer with the uniform ``{ok, error, message}`` envelope."""

    return JSONResponse(status_code=exc.status_code, content=exc.detail, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Report a malformed body in the same envelope as everything else."""

    first = (exc.errors() or [{}])[0]
    location = ".".join(str(part) for part in first.get("loc") or [] if part != "body")
    message = f"Ungültige Anfrage: {location or 'body'} – {first.get('msg') or 'unbrauchbar'}"
    return JSONResponse(
        status_code=400,
        content={"ok": False, "error": "bad_request", "message": message},
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: log the traceback, answer in the documented envelope."""

    log.exception("Unbehandelter Fehler in %s.", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": "internal_error",
            "message": "Unerwarteter Fehler im ImageInt-Gateway. Details stehen im Log.",
        },
    )


# --------------------------------------------------------------------------- #
# Service and health
# --------------------------------------------------------------------------- #
@app.get("/", tags=["service"])
async def root() -> dict:
    """Banner with the version and the endpoint list."""

    return {
        "ok": True,
        "service": "imageint",
        "version": __version__,
        "description": "Bilderzeugung für LLMInt auf Basis von Qwen-Image-2.1 (diffusers, CPU).",
        "tool": "generate_image",
        "endpoints": [
            "/health",
            "/v1/ready",
            "/v1/health",
            "/v1/config",
            "/v1/models",
            "/v1/ratios",
            "/v1/enhance",
            "/v1/images/generations",
            "/v1/jobs",
            "/v1/jobs/{job_id}",
            "/v1/images/{job_id}",
        ],
    }


@app.get("/health", tags=["service"])
async def health() -> dict:
    """Liveness. Always 200 while the process runs; no upstream is contacted.

    This is what the container health check polls: the gateway being up and the
    model servers being ready are two different questions, and only the second
    one belongs in ``/v1/ready``.
    """

    return {
        "ok": True,
        "status": "alive",
        "service": "imageint",
        "version": __version__,
        "uptime_seconds": round(time.time() - runtime().started_at, 1),
    }


def _readiness_body(snapshot: dict) -> dict:
    """Add the pieces of context a client needs to decide what to do next."""

    settings = runtime().settings
    body: dict[str, Any] = {
        "ok": snapshot["ready"],
        "service": "imageint",
        "version": __version__,
        "state": snapshot["state"],
        "transient": snapshot["transient"],
        "host": snapshot["host"],
        "components": snapshot["components"],
        "jobs": runtime().registry.stats(),
    }
    if not snapshot["ready"]:
        body["retry_after"] = settings.loading_retry_after
    return body


async def _readiness(request: Request) -> JSONResponse:
    snapshot = await runtime().health.snapshot()
    body = _readiness_body(snapshot)
    headers = {}
    if not snapshot["ready"]:
        # The distinction that matters to a client: keep the request and come
        # back, rather than showing the user an error.
        headers["Retry-After"] = str(runtime().settings.loading_retry_after)
    return JSONResponse(status_code=200 if snapshot["ready"] else 503, content=body, headers=headers)


@app.get("/v1/ready", tags=["service"])
async def ready(request: Request) -> JSONResponse:
    """Readiness without authentication, for the compose health check."""

    return await _readiness(request)


@app.get("/v1/health", tags=["service"], dependencies=[Depends(require_token)])
async def health_authenticated(request: Request) -> JSONResponse:
    """Readiness with authentication; same body as ``/v1/ready``."""

    return await _readiness(request)


# --------------------------------------------------------------------------- #
# Configuration and models
# --------------------------------------------------------------------------- #
def _mask(token: str) -> str:
    """Show whether a token is set without disclosing it."""

    if not token:
        return ""
    if len(token) <= 4:
        return "****"
    return f"{token[:2]}****{token[-2:]}"


@app.get("/v1/config", tags=["service"], dependencies=[Depends(require_token)])
async def config() -> dict:
    """The effective configuration, so the admin area can show what is running."""

    settings = runtime().settings
    return {
        "ok": True,
        "service": "imageint",
        "version": __version__,
        "public_url": settings.public_url,
        "auth_required": settings.auth_required,
        "token_masked": _mask(settings.token),
        "limits": {
            "max_prompt_chars": settings.max_prompt_chars,
            "sync_timeout_seconds": settings.sync_timeout_seconds,
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "max_queued_jobs": settings.max_queued_jobs,
            "max_jobs": settings.max_jobs,
            "job_retention_seconds": settings.job_retention_seconds,
            "poll_after_seconds": POLL_AFTER,
        },
        "storage_dir": settings.storage_dir,
        "enhancer": {
            **enhancer_module.profile(),
            "token_masked": _mask(settings.enhancer.token),
        },
        "image": {
            "url": settings.image.url,
            "model": settings.image.model,
            "quant": settings.image.quant,
            "route": settings.image.route,
            "route_path": IMAGE_ROUTE_PATHS.get(
                settings.image.route, IMAGE_ROUTE_PATHS["images"]
            ),
            "token_masked": _mask(settings.image.token),
            "steps": settings.image.steps,
            "true_cfg_scale": settings.image.true_cfg_scale,
            "negative_prompt": settings.image.negative_prompt,
            "max_pixels": settings.image.max_pixels,
            "timeout": settings.image.timeout,
        },
        "host": host.inspect(),
    }


@app.get("/v1/models", tags=["service"], dependencies=[Depends(require_token)])
async def models() -> dict:
    """Both model servers with their current state."""

    snapshot = await runtime().health.snapshot(force=True)
    return {
        "ok": True,
        "service": "imageint",
        "ready": snapshot["ready"],
        "models": {
            name: {
                "url": snapshot["components"][name]["url"],
                "model": snapshot["components"][name]["model"],
                "state": snapshot["components"][name]["state"],
                "ready": snapshot["components"][name]["ready"],
                "http": snapshot["components"][name]["http"],
                "message": snapshot["components"][name]["message"],
                "role": "prompt-enhancer" if name == "enhancer" else "image-generation",
            }
            for name in COMPONENTS
        },
        "quant": runtime().settings.image.quant,
    }


@app.get("/v1/ratios", tags=["service"], dependencies=[Depends(require_token)])
async def ratio_table() -> dict:
    """The canvas sizes of the model card, plus how an unknown ratio is derived."""

    settings = runtime().settings
    return {
        "ok": True,
        "documented": {
            name: {"width": size[0], "height": size[1]}
            for name, size in ratios.DOCUMENTED.items()
        },
        "default": {"width": ratios.NATIVE_PIXELS**0.5, "height": ratios.NATIVE_PIXELS**0.5},
        "max_pixels": settings.image.max_pixels,
        "multiple": ratios.MULTIPLE,
        "note": (
            "Nicht dokumentierte Verhältnisse werden auf dieselbe Fläche wie das "
            "native 2K-Bild abgebildet, auf ein Vielfaches von 64 gerundet und auf "
            "IMAGEINT_IMAGE_MAX_PIXELS begrenzt."
        ),
    }


# --------------------------------------------------------------------------- #
# Prompt enhancement
# --------------------------------------------------------------------------- #
@app.post("/v1/enhance", tags=["generation"], dependencies=[Depends(require_token)])
async def enhance(payload: EnhanceRequest) -> dict:
    """Run only the documented prompt enhancer.

    Used by the LLMInt admin area as a connection test: it exercises the
    enhancer end to end without paying for a diffusion render.
    """

    settings = runtime().settings
    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise bad_request("Es wurde kein Prompt übergeben.")
    if len(prompt) > settings.max_prompt_chars:
        raise payload_too_large(
            f"Der Prompt ist länger als die erlaubten {settings.max_prompt_chars} Zeichen."
        )

    await runtime().health.require_ready("enhancer")
    result = await enhancer_module.enhance(
        settings.enhancer, prompt, max_pixels=settings.image.max_pixels
    )
    return {"ok": True, **result}


# --------------------------------------------------------------------------- #
# Image generation
# --------------------------------------------------------------------------- #
def _job_body(job, base_url: str) -> dict:
    body = job.public(base_url)
    body["ok"] = job.status == DONE
    return body


def _sync_body(job, base_url: str, include_image: bool) -> dict:
    """Response of a generation that finished inside the sync window."""

    body = _job_body(job, base_url)
    body["status_url"] = job.image_url(base_url).replace("/v1/images/", "/v1/jobs/")
    if include_image:
        data = runtime().registry.image(job.id) or b""
        body["b64_json"] = base64.b64encode(data).decode("ascii")
        body["content_type"] = job.content_type or "image/png"
    return body


def _pending_body(job, base_url: str) -> dict:
    """Response of a generation that is still running: the 202 body."""

    return {
        "ok": True,
        "job_id": job.id,
        "status": job.status,
        "stage": job.stage,
        "status_url": job.image_url(base_url).replace("/v1/images/", "/v1/jobs/"),
        "image_url": job.image_url(base_url),
        "poll_after_seconds": POLL_AFTER,
        "message": (
            "Das Bild wird noch erzeugt. Der Auftrag bleibt unter status_url abrufbar; "
            f"der nächste Abruf sollte nach etwa {POLL_AFTER} Sekunden erfolgen."
        ),
    }


@app.post("/v1/images/generations", tags=["generation"], dependencies=[Depends(require_token)])
async def generate_image(payload: GenerationRequest, request: Request) -> JSONResponse:
    """Generate one image, waiting at most ``IMAGEINT_SYNC_TIMEOUT`` seconds.

    Answers 200 with the finished image, or 202 with a job id when the render
    needs longer -- which on a CPU host is the normal case. A failed job is
    reported with the HTTP status that belongs to its error code.
    """

    settings = runtime().settings
    registry = runtime().registry

    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise bad_request("Es wurde kein Prompt übergeben.")
    if len(prompt) > settings.max_prompt_chars:
        raise payload_too_large(
            f"Der Prompt ist länger als die erlaubten {settings.max_prompt_chars} Zeichen."
        )

    # Fail before queuing a job when a model server cannot serve at all: a client
    # that is told "loading, retry in 15 s" keeps its request, one that waits two
    # minutes for a 502 has wasted the user's time.
    #
    # Only probe the enhancer when the request will actually use it, and never
    # when the operator switched it off: an explicit "enhance": true against a
    # disabled enhancer must reach the enhancer module, which explains that in
    # one sentence instead of a generic loading error.
    if settings.enhancer.enabled and pipeline.wants_enhancer(payload.model_dump(), settings):
        await runtime().health.require_ready("enhancer")
    await runtime().health.require_ready("image")

    job = registry.submit(prompt, payload.model_dump())
    base_url = settings.public_url or str(request.base_url).rstrip("/")

    if payload.wait and await registry.wait(job, settings.sync_timeout_seconds):
        if job.status == DONE:
            return JSONResponse(content=_sync_body(job, base_url, payload.include_image))
        # The job failed inside the window; the caller gets the real status
        # rather than a 202 for something that is already over.
        raise ImageIntError(
            _STATUS_BY_CODE.get(job.error_code, 500),
            job.error_code or "internal_error",
            job.error_message or "Die Bilderzeugung ist fehlgeschlagen.",
        )

    return JSONResponse(
        status_code=202,
        content=_pending_body(job, base_url),
        headers={"Retry-After": str(POLL_AFTER)},
    )


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
@app.get("/v1/jobs", tags=["jobs"], dependencies=[Depends(require_token)])
async def list_jobs(limit: int = 50) -> dict:
    """Recent jobs, newest first."""

    registry = runtime().registry
    base_url = runtime().settings.public_url
    jobs = registry.list(max(1, min(limit, 500)))
    return {
        "ok": True,
        "count": len(jobs),
        "stats": registry.stats(),
        "jobs": [_job_body(job, base_url) for job in jobs],
    }


@app.get("/v1/jobs/{job_id}", tags=["jobs"], dependencies=[Depends(require_token)])
async def job_status(job_id: str) -> dict:
    """State of one job. This is what a client polls after a 202."""

    registry = runtime().registry
    job = registry.get(job_id)
    body = _job_body(job, runtime().settings.public_url)
    if job.status not in ("done", "error"):
        body["poll_after_seconds"] = POLL_AFTER
    return body


def _image_response(job_id: str) -> Response:
    registry = runtime().registry
    job = registry.get(job_id)

    if job.status == ERROR:
        raise ImageIntError(
            _STATUS_BY_CODE.get(job.error_code, 500),
            job.error_code or "internal_error",
            job.error_message or "Die Bilderzeugung ist fehlgeschlagen.",
        )
    if job.status != DONE:
        raise ImageIntError(
            409,
            "job_not_finished",
            f"Der Bildauftrag {job_id} ist noch nicht fertig (Status {job.status}).",
            headers={"Retry-After": str(POLL_AFTER)},
        )

    data = registry.image(job_id)
    if not data:
        raise not_found(
            f"Für den Bildauftrag {job_id} liegt keine Bilddatei mehr vor "
            f"(Aufbewahrung {runtime().settings.job_retention_seconds // 3600} h)."
        )

    # Job ids are unique and an image never changes, so this is safe to cache
    # for as long as the job is retained.
    return Response(
        content=data,
        media_type=registry.image_content_type(job_id),
        headers={
            "Cache-Control": f"private, max-age={runtime().settings.job_retention_seconds}",
            "Content-Disposition": f'inline; filename="imageint-{job_id}.png"',
            "ETag": f'"{job_id}"',
            "X-ImageInt-Job": job_id,
            "X-ImageInt-Width": str(job.width),
            "X-ImageInt-Height": str(job.height),
            "X-ImageInt-Model": job.model,
        },
    )


@app.get("/v1/jobs/{job_id}/image", tags=["jobs"], dependencies=[Depends(require_token)])
async def job_image(job_id: str) -> Response:
    """The finished image of one job."""

    return _image_response(job_id)


@app.get("/v1/images/{job_id}", tags=["jobs"], dependencies=[Depends(require_token)])
async def image_by_id(job_id: str) -> Response:
    """The finished image, under the URL the client was told to fetch.

    Deliberately the same handler as ``/v1/jobs/{id}/image``: two spellings, one
    behaviour. Without a token this URL is only usable from inside the compose
    network, which is why ``IMAGEINT_PUBLIC_URL`` and the token belong together.
    """

    return _image_response(job_id)
