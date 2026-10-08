"""HTTP surface of the image server.

The server speaks the OpenAI Images API on purpose: the gateway already talks
that wire format, so replacing the model backend needs no change on the gateway
side. Three routes matter:

``GET /health``
    ``200`` when the pipeline is ready, ``503`` while it is still loading, and
    ``500`` when loading failed. That is the distinction the gateway's loading
    contract is built on.

``GET /v1/models``
    The served model id, so a deployment can be inspected without guessing.

``POST /v1/warmup``
    Start loading the weights without rendering anything, and answer at once.
    It exists for the one configuration in which ``/health`` alone cannot make
    progress: a container started with ``IMAGEINT_IMAGE_PRELOAD=false`` loads
    lazily, so it answers ``503`` until a render arrives -- and a client that
    insists on ``200`` first would wait forever.

``POST /v1/images/generations``
    One prompt in, one base64 PNG out. Every field is optional except the
    prompt, and every default matches the Qwen-Image-2.1 model card.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import DOCUMENTED_SIZES, Settings, load_settings, snap
from .engine import BaseEngine, EngineError, RenderRequest, build_engine

log = logging.getLogger("imageint.image")

#: HTTP status the gateway reads as "still loading".
LOADING_STATUS = 503


class ServerError(Exception):
    """An error the client should see verbatim, in the OpenAI error shape."""

    def __init__(self, status_code: int, message: str, code: str = "error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code


class ImageRequest(BaseModel):
    """Body of ``POST /v1/images/generations``.

    ``size`` is the OpenAI spelling and wins over ``width``/``height``; the
    gateway sends ``size`` because that is what the Images API documents.
    """

    model: str = ""
    prompt: str = Field(min_length=1)
    n: int = 1
    size: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    num_inference_steps: Optional[int] = None
    true_cfg_scale: Optional[float] = None
    negative_prompt: str = ""
    seed: Optional[int] = None
    response_format: str = "b64_json"
    stream: bool = False


def _parse_size(value: str) -> Optional[tuple[int, int]]:
    raw = (value or "").strip().lower().replace(" ", "")
    if not raw:
        return None
    for separator in ("x", "*", "×"):
        if separator in raw:
            left, _, right = raw.partition(separator)
            try:
                return int(left), int(right)
            except ValueError:
                return None
    return None


def resolve_size(payload: ImageRequest, settings: Settings) -> tuple[int, int]:
    """Turn the request into the canvas that is actually rendered.

    Raises :class:`ValueError` naming the offending value, because a silently
    different canvas is worse than a rejected request.
    """

    if payload.size:
        parsed = _parse_size(payload.size)
        if parsed is None:
            raise ValueError(f"size '{payload.size}' ist nicht im Format BREITExHÖHE.")
        width, height = parsed
    elif payload.width and payload.height:
        width, height = int(payload.width), int(payload.height)
    else:
        width, height = DOCUMENTED_SIZES[0]

    if width <= 0 or height <= 0:
        raise ValueError("Breite und Höhe müssen positiv sein.")

    width, height = snap(width), snap(height)
    if width * height > settings.max_pixels:
        raise ValueError(
            f"{width}x{height} überschreitet das Limit von {settings.max_pixels} Pixeln."
        )
    return width, height


def error_body(status_code: int, message: str, code: str) -> JSONResponse:
    """OpenAI-shaped error body, which is what the gateway parses."""

    return JSONResponse(
        {
            "ok": False,
            "error": {"message": message, "type": code, "code": code},
            "message": message,
        },
        status_code=status_code,
    )


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Build the application. Split out so tests can pass their own settings."""

    resolved = settings or load_settings()
    engine = build_engine(resolved)
    started_at = time.time()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Read through app.state so a test (or an operator swap) can replace the
        # engine after the app has been created.
        app.state.engine.start()
        try:
            yield
        finally:
            app.state.engine.close()

    app = FastAPI(
        title="ImageInt image server",
        version=__version__,
        summary="Qwen-Image-2.1 on CPU through diffusers and transformers",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.engine = engine

    @app.exception_handler(ServerError)
    async def server_error_handler(_request: Request, exc: ServerError) -> JSONResponse:
        return error_body(exc.status_code, exc.message, exc.code)

    async def require_token(request: Request) -> None:
        """Enforce the shared secret, if one is configured."""

        expected = app.state.settings.token
        if not expected:
            return
        supplied = request.headers.get("X-Auth-Token") or ""
        if not supplied:
            header = request.headers.get("Authorization") or ""
            scheme, _, value = header.partition(" ")
            if scheme.lower() == "bearer":
                supplied = value.strip()
        if supplied != expected:
            raise ServerError(
                401,
                "Nicht autorisiert: X-Auth-Token fehlt oder ist falsch.",
                "unauthorized",
            )

    @app.get("/", tags=["service"])
    async def root() -> dict:
        return {
            "service": "ImageInt image server",
            "version": __version__,
            "model": app.state.settings.model,
            "engine": app.state.engine.name,
            "uptime_seconds": round(time.time() - started_at, 1),
            "docs": "/docs",
        }

    @app.get("/health", tags=["service"])
    async def health() -> JSONResponse:
        status = app.state.engine.status()
        state = status["state"]
        if state == "ready":
            return JSONResponse({"ok": True, "status": "ready", **status}, status_code=200)
        if state == "error":
            return JSONResponse(
                {"ok": False, "status": "error", "message": status["error"], **status},
                status_code=500,
            )
        return JSONResponse(
            {
                "ok": False,
                "status": "loading",
                "message": "Das Bildmodell wird noch geladen.",
                **status,
            },
            status_code=LOADING_STATUS,
        )

    @app.get("/v1/models", tags=["service"], dependencies=[Depends(require_token)])
    async def models() -> dict:
        status = app.state.engine.status()
        return {
            "object": "list",
            "data": [
                {
                    "id": app.state.settings.model,
                    "object": "model",
                    "created": int(started_at),
                    "owned_by": "imageint",
                    "device": status["device"],
                    "dtype": status["dtype"],
                    "quant": status["quant"],
                    "engine": status["engine"],
                }
            ],
        }

    @app.post("/v1/warmup", tags=["service"], dependencies=[Depends(require_token)])
    async def warmup() -> JSONResponse:
        """Begin loading the weights, without rendering anything.

        A container with ``IMAGEINT_IMAGE_PRELOAD=false`` stays in ``idle``
        until a render request arrives. That is fine for a client that simply
        posts a render, but a client that waits for ``/health`` to turn green
        before it posts anything would wait forever. This route breaks that
        circle: it starts the load in the background and answers immediately.

        Idempotent and cheap: while a load is already running (or finished) it
        does nothing at all, so it is safe to call repeatedly.
        """

        engine = app.state.engine
        state = engine.status()["state"]
        if state == "idle":
            threading.Thread(
                target=engine.ensure_loaded,
                name="imageint-image-warmup",
                daemon=True,
            ).start()
            state = "loading"
            log.info("[image] warm-up requested, loading in the background")

        ready = state == "ready"
        return JSONResponse(
            {
                "ok": True,
                "state": state,
                "loading": not ready,
                "message": (
                    "Das Bildmodell ist geladen."
                    if ready
                    else "Das Bildmodell wird geladen."
                ),
            },
            status_code=200 if ready else 202,
        )

    @app.post(
        "/v1/images/generations",
        tags=["generation"],
        dependencies=[Depends(require_token)],
    )
    async def generations(payload: ImageRequest) -> JSONResponse:
        if payload.n != 1:
            raise ServerError(
                400,
                "Dieser Server rendert genau ein Bild pro Anfrage (n=1).",
                "unsupported_n",
            )
        prompt = payload.prompt.strip()
        if not prompt:
            raise ServerError(400, "Der Prompt darf nicht leer sein.", "bad_request")
        if len(prompt) > resolved.max_prompt_chars:
            raise ServerError(
                400,
                f"Der Prompt ist länger als {resolved.max_prompt_chars} Zeichen.",
                "bad_request",
            )

        try:
            width, height = resolve_size(payload, resolved)
        except ValueError as exc:
            raise ServerError(400, str(exc), "bad_request") from exc

        request = RenderRequest(
            prompt=prompt,
            negative_prompt=payload.negative_prompt,
            width=width,
            height=height,
            steps=payload.num_inference_steps or resolved.steps,
            true_cfg_scale=(
                resolved.true_cfg_scale
                if payload.true_cfg_scale is None
                else payload.true_cfg_scale
            ),
            seed=payload.seed,
        )

        try:
            image, meta = app.state.engine.generate(request)
        except EngineError as exc:
            status = app.state.engine.status()
            code = LOADING_STATUS if status["loading"] else 500
            raise ServerError(code, str(exc), "engine_unavailable") from exc

        log.info(
            "[image] rendered %sx%s in %s ms (seed %s)",
            width,
            height,
            meta.get("latency_ms"),
            meta.get("seed"),
        )

        return JSONResponse(
            {
                "created": int(time.time()),
                "model": resolved.model,
                "engine": app.state.engine.name,
                "size": f"{width}x{height}",
                "data": [
                    {
                        "b64_json": base64.b64encode(image).decode("ascii"),
                        "width": width,
                        "height": height,
                        "model": resolved.model,
                        "seed": meta.get("seed"),
                    }
                ],
                "latency_ms": meta.get("latency_ms"),
            }
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("[image] unhandled error on %s", request.url.path)
        return error_body(500, f"Unerwarteter Fehler: {exc}", "internal_error")

    return app


__all__ = ["BaseEngine", "ServerError", "create_app", "resolve_size"]
