"""HTTP surface of the prompt-enhancer server.

The server speaks the OpenAI chat-completions API on purpose: the gateway
already talks that wire format to vLLM, so replacing the backend needs no
change on the gateway side. Four routes matter:

``GET /health``
    ``200`` when the model is ready, ``503`` while it is still loading, and
    ``500`` when loading failed. That is the distinction the gateway's loading
    contract is built on.

``GET /v1/models``
    The served model id, so a deployment can be inspected without guessing.

``POST /v1/warmup``
    Start loading the weights without generating anything, and answer at once.
    Same reason as on the image server: a container started with
    ``IMAGEINT_PE_PRELOAD=false`` loads lazily, so ``/health`` stays ``503``
    until a request arrives -- and a client that insists on ``200`` first would
    wait forever.

``POST /v1/chat/completions``
    One chat request in, one enhanced prompt out. ``stream: true`` -- what the
    gateway sends -- answers with server-sent events in the OpenAI chunk shape,
    including a separate ``reasoning_content`` field for the thinking block,
    which is what vLLM's ``--reasoning-parser qwen3`` used to produce.

The thinking block is deliberately *not* part of ``content``. That keeps the
gateway's ``split_thinking`` a no-op instead of making it depend on the model
finishing its reasoning, and it means the answer the gateway parses is exactly
the JSON object the model was asked for.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Iterator, Optional

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, load_settings
from .engine import (
    BaseEngine,
    ChatRequest,
    EngineError,
    build_engine,
    last_user_text,
)

log = logging.getLogger("imageint.enhancer")

#: HTTP status the gateway reads as "still loading".
LOADING_STATUS = 503

#: Seconds of silence after which the stream emits an SSE comment. A CPU
#: prefill of a few thousand tokens can take minutes before the first token
#: arrives, and the gateway's read timeout covers the gap between two chunks.
KEEPALIVE_SECONDS = 15.0

#: Upper bound on the messages of one request. The gateway sends two.
MAX_MESSAGES = 64


class ServerError(Exception):
    """An error the client should see verbatim, in the OpenAI error shape."""

    def __init__(self, status_code: int, message: str, code: str = "error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code


class ChatCompletionRequest(BaseModel):
    """Body of ``POST /v1/chat/completions``.

    ``messages`` stays a list of plain dicts: the checkpoint's chat template
    consumes the part-list content shape the gateway sends, and coercing it
    into a model here would flatten exactly that shape away. Unknown fields are
    ignored on purpose, so a newer client is not rejected outright.
    """

    model: str = ""
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    seed: Optional[int] = None
    stream: bool = False
    stream_options: Optional[dict[str, Any]] = None
    chat_template_kwargs: Optional[dict[str, Any]] = None


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


def wants_usage(payload: ChatCompletionRequest) -> bool:
    """Whether the caller asked for a usage report.

    The gateway always sends ``stream_options.include_usage``, so this is what
    keeps the final chunk -- and with it the token counts in the gateway's
    metrics -- unchanged from the vLLM deployment.
    """

    options = payload.stream_options or {}
    return bool(options.get("include_usage", False))


def wants_thinking(payload: ChatCompletionRequest, settings: Settings) -> bool:
    """The thinking switch, with the operator's setting as the fallback."""

    kwargs = payload.chat_template_kwargs or {}
    value = kwargs.get("enable_thinking")
    if value is None:
        return settings.enable_thinking
    return bool(value)


def normalise(payload: ChatCompletionRequest, settings: Settings) -> ChatRequest:
    """Validate the request and turn it into an engine-level job."""

    messages = [message for message in payload.messages if isinstance(message, dict)]
    if not messages:
        raise ServerError(400, "Der Request enthält keine Nachrichten.", "bad_request")
    if len(messages) > MAX_MESSAGES:
        raise ServerError(
            400,
            f"Der Request enthält mehr als {MAX_MESSAGES} Nachrichten.",
            "bad_request",
        )
    for message in messages:
        if not isinstance(message.get("role"), str) or not message["role"]:
            raise ServerError(400, "Jede Nachricht braucht eine Rolle.", "bad_request")

    prompt = last_user_text(messages)
    if not prompt.strip():
        raise ServerError(400, "Der zu verbessernde Prompt ist leer.", "bad_request")
    if len(prompt) > settings.max_prompt_chars:
        raise ServerError(
            400,
            f"Der Prompt ist länger als {settings.max_prompt_chars} Zeichen.",
            "bad_request",
        )

    # ``IMAGEINT_PE_MAX_NEW_TOKENS`` is a ceiling, not just a default: a client
    # may ask for a shorter answer, never for a longer one than the operator
    # budgeted for. An explicit 0 is a malformed request rather than a missing
    # field, so the two are told apart instead of both falling back.
    requested = (
        payload.max_tokens
        if payload.max_tokens is not None
        else payload.max_completion_tokens
    )
    max_tokens = int(requested) if requested is not None else settings.max_new_tokens
    if max_tokens > settings.max_new_tokens:
        log.debug(
            "[enhancer] max_tokens=%d auf das Server-Budget %d begrenzt",
            max_tokens,
            settings.max_new_tokens,
        )
        max_tokens = settings.max_new_tokens
    if max_tokens < 1:
        raise ServerError(400, "max_tokens muss positiv sein.", "bad_request")

    return ChatRequest(
        messages=tuple(messages),
        temperature=settings.temperature if payload.temperature is None else payload.temperature,
        top_p=settings.top_p if payload.top_p is None else payload.top_p,
        top_k=settings.top_k if payload.top_k is None else payload.top_k,
        min_p=settings.min_p if payload.min_p is None else payload.min_p,
        presence_penalty=(
            settings.presence_penalty
            if payload.presence_penalty is None
            else payload.presence_penalty
        ),
        max_tokens=max_tokens,
        seed=settings.seed if payload.seed is None else payload.seed,
        enable_thinking=wants_thinking(payload, settings),
    )


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: dict,
    *,
    finish_reason: Optional[str] = None,
    usage: Optional[dict] = None,
) -> str:
    body: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}
        ],
    }
    if usage is not None:
        body["usage"] = usage
    return _sse(body)


def _stream(
    engine: BaseEngine,
    request: ChatRequest,
    completion_id: str,
    created: int,
    model: str,
    include_usage: bool,
) -> Iterator[str]:
    """Bridge the engine's synchronous generator to an SSE response.

    The engine runs in its own thread rather than being iterated directly for
    three reasons: an SSE comment can be sent while nothing is generated, so a
    slow prefill does not trip the client's read timeout; the request can be
    abandoned when the client hangs up; and the single-flight lock is released
    at that point instead of after the full answer.
    """

    usage: dict[str, Any] = {}
    failure: list[BaseException] = []
    stop = threading.Event()
    items: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=64)

    def offer(item: tuple[str, Any]) -> bool:
        """Hand one item over, giving up if the consumer has gone away."""

        while not stop.is_set():
            try:
                items.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        stream = engine.stream(request, usage)
        try:
            for reasoning, content in stream:
                if not offer(("delta", (reasoning, content))):
                    break
        except BaseException as exc:  # noqa: BLE001 - forwarded to the client
            failure.append(exc)
        finally:
            # Closing the generator unwinds the engine's lock and, in the
            # transformers engine, stops the decoding thread.
            stream.close()
            offer(("end", None))

    worker = threading.Thread(target=produce, name="imageint-enhancer-stream", daemon=True)
    worker.start()

    try:
        yield _chunk(completion_id, created, model, {"role": "assistant", "content": ""})
        while True:
            try:
                kind, payload = items.get(timeout=KEEPALIVE_SECONDS)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if kind == "end":
                break
            reasoning, content = payload
            if reasoning:
                yield _chunk(completion_id, created, model, {"reasoning_content": reasoning})
            if content:
                yield _chunk(completion_id, created, model, {"content": content})

        if failure:
            log.error("[enhancer] Generierung abgebrochen: %s", failure[0])
            yield _chunk(
                completion_id, created, model, {}, finish_reason="error"
            )
        else:
            yield _chunk(completion_id, created, model, {}, finish_reason="stop")

        if include_usage:
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                        "completion_tokens": int(usage.get("completion_tokens") or 0),
                        "total_tokens": int(usage.get("total_tokens") or 0),
                    },
                }
            )
        yield "data: [DONE]\n\n"
    finally:
        stop.set()
        worker.join(timeout=5.0)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Build the application. Split out so tests can pass their own settings."""

    resolved = settings or load_settings()
    engine = build_engine(resolved)
    started_at = time.time()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine.start()
        try:
            yield
        finally:
            app.state.engine.close()

    app = FastAPI(
        title="ImageInt enhancer server",
        version=__version__,
        summary="Qwen-Image-2.1-PE-T2I on CPU through transformers",
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
            "service": "ImageInt enhancer server",
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
                "message": "Das Enhancer-Modell wird noch geladen.",
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
        """Begin loading the weights, without generating anything.

        A container with ``IMAGEINT_PE_PRELOAD=false`` stays in ``idle`` until a
        chat request arrives. This route breaks the resulting circle between
        the gateway waiting for ``/health`` and the server waiting for a
        request. Idempotent and cheap: while a load is running, or after it has
        finished, it does nothing at all.
        """

        engine = app.state.engine
        state = engine.status()["state"]
        if state == "idle":
            threading.Thread(
                target=engine.ensure_loaded,
                name="imageint-enhancer-warmup",
                daemon=True,
            ).start()
            state = "loading"
            log.info("[enhancer] warm-up requested, loading in the background")

        ready = state == "ready"
        return JSONResponse(
            {
                "ok": True,
                "state": state,
                "loading": not ready,
                "message": (
                    "Das Enhancer-Modell ist geladen."
                    if ready
                    else "Das Enhancer-Modell wird geladen."
                ),
            },
            status_code=200 if ready else 202,
        )

    @app.post(
        "/v1/chat/completions",
        tags=["generation"],
        dependencies=[Depends(require_token)],
    )
    def chat_completions(payload: ChatCompletionRequest):
        """Rewrite one prompt.

        Declared synchronous so FastAPI runs it in a worker thread: the engine
        may have to load 19 GB of weights before it can answer, and blocking
        the event loop for that would stall ``/health`` along with it.
        """

        request = normalise(payload, resolved)

        try:
            app.state.engine.ensure_loaded()
        except EngineError as exc:
            raise ServerError(500, str(exc), "engine_unavailable") from exc

        status = app.state.engine.status()
        if status["state"] != "ready":
            code = 500 if status["state"] == "error" else LOADING_STATUS
            raise ServerError(
                code,
                status["error"] or "Das Enhancer-Modell ist nicht geladen.",
                "engine_unavailable",
            )

        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        model = payload.model or resolved.model

        log.info(
            "[enhancer] %d Nachricht(en), %d Zeichen, max_tokens=%d, thinking=%s",
            len(request.messages),
            len(last_user_text(request.messages)),
            request.max_tokens,
            request.enable_thinking,
        )

        if payload.stream:
            return StreamingResponse(
                _stream(
                    app.state.engine,
                    request,
                    completion_id,
                    created,
                    model,
                    wants_usage(payload),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    # Tells an intermediate proxy not to buffer the stream.
                    "X-Accel-Buffering": "no",
                },
            )

        return JSONResponse(
            _collect(app.state.engine, request, completion_id, created, model)
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("[enhancer] unhandled error on %s", request.url.path)
        return error_body(500, f"Unerwarteter Fehler: {exc}", "internal_error")

    return app


def _collect(
    engine: BaseEngine,
    request: ChatRequest,
    completion_id: str,
    created: int,
    model: str,
) -> dict:
    """Run the generation to completion and return one non-streamed answer."""

    reasoning: list[str] = []
    content: list[str] = []
    usage: dict[str, Any] = {}
    started = time.monotonic()
    try:
        for part_reasoning, part_content in engine.stream(request, usage):
            if part_reasoning:
                reasoning.append(part_reasoning)
            if part_content:
                content.append(part_content)
    except EngineError as exc:
        status = engine.status()
        code = LOADING_STATUS if status["loading"] else 500
        raise ServerError(code, str(exc), "engine_unavailable") from exc

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "engine": engine.name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(content),
                    "reasoning_content": "".join(reasoning),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        },
        "latency_ms": int(round((time.monotonic() - started) * 1000)),
    }


__all__ = [
    "ChatCompletionRequest",
    "ServerError",
    "create_app",
    "normalise",
    "wants_thinking",
    "wants_usage",
]
