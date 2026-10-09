"""HTTP client for the two model servers.

Both model servers speak the OpenAI wire format, so one module covers them:

``chat_completion``
    ``POST /v1/chat/completions`` on the prompt enhancer. Qwen-Image-2.1's
    enhancer is a fine-tuned Qwen3.5-VL 9B that *thinks* before it answers, and
    on a CPU host that thinking block can take minutes. The request is therefore
    streamed: the read timeout then applies between chunks instead of to the
    whole generation, and a stalled server is still detected.

``generate_image``
    ``POST /v1/images/generations`` on the image server. The picture comes back
    as base64 in ``data[0].b64_json``; the chat-completions shape
    (``choices[0].message.content[0].image_url.url``) is accepted as well, so the
    same client works whichever route the deployed server exposes.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Mapping, Optional

import httpx

from .config import IMAGE_ROUTE_PATHS, EnhancerSettings, ImageSettings
from .errors import not_configured, upstream_timeout, upstream_unavailable

log = logging.getLogger("imageint.vllm")


def _headers(token: str, *, json_body: bool = False) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if json_body:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
        # vLLM also honours the api-key header used by the other model servers
        # in this project, so both spellings work.
        headers["X-Auth-Token"] = token
    return headers


def _upstream_message(decoded: Any) -> str:
    """Pull the human-readable text out of an OpenAI-style error body."""

    if not isinstance(decoded, dict):
        return ""
    error = decoded.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")
    if error:
        return str(error)
    return str(decoded.get("message") or "")


async def health(url: str, token: str, model: str, *, timeout: float = 10.0) -> dict:
    """Probe one vLLM server through its ``/health`` endpoint.

    vLLM answers ``200`` once the engine is up and ``503`` while it is still
    building the model, which is exactly the distinction the loading contract
    needs. A closed port stays indistinguishable from a slow first start here;
    :mod:`app.health` turns that into the ``starting`` state.
    """

    if not url:
        return {
            "ok": False,
            "loading": False,
            "configured": False,
            "message": "Keine Modell-URL konfiguriert.",
            "http": 0,
            "url": "",
            "model": model,
        }

    result = {"configured": True, "url": url, "model": model}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url + "/health", headers=_headers(token))
    except httpx.TimeoutException:
        return {
            **result,
            "ok": False,
            "loading": False,
            "http": 0,
            "message": "Modellserver antwortet nicht (Timeout).",
        }
    except httpx.HTTPError as exc:
        return {
            **result,
            "ok": False,
            "loading": False,
            "http": 0,
            "message": f"Modellserver nicht erreichbar: {exc}",
        }

    if response.status_code == 200:
        return {
            **result,
            "ok": True,
            "loading": False,
            "http": 200,
            "message": "Modellserver erreichbar.",
        }

    if response.status_code == 503:
        return {
            **result,
            "ok": False,
            "loading": True,
            "http": 503,
            "message": "Modellserver lädt das Modell noch (HTTP 503).",
        }

    return {
        **result,
        "ok": False,
        "loading": False,
        "http": response.status_code,
        "message": f"Modellserver meldet HTTP {response.status_code}.",
    }


async def warmup(url: str, token: str, *, timeout: float = 10.0) -> bool:
    """Ask a lazy model server to start loading, and do not wait for it.

    Both model servers implement this route. It matters for exactly one
    deployment each: ``IMAGEINT_IMAGE_PRELOAD=false`` or
    ``IMAGEINT_PE_PRELOAD=false`` leaves the container in ``idle`` answering
    ``503`` until a request arrives, so a gateway that refuses to send one
    before it sees ``200`` would deadlock with it. The gateway calls this as a
    side effect of the readiness probe, which breaks the circle
    without changing what any client sees.

    Best effort by design: a model server that does not implement the route
    (vLLM, an older image server) answers 404, and that is not an error worth
    reporting. Returns whether the request reached the server at all.
    """

    if not url:
        return False

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url + "/v1/warmup", headers=_headers(token))
    except httpx.HTTPError as exc:
        log.debug("Warm-up an %s fehlgeschlagen: %s", url, exc)
        return False

    if response.status_code >= 400:
        log.debug("Warm-up an %s: HTTP %s", url, response.status_code)
    return True


async def chat_completion(
    settings: EnhancerSettings,
    payload: Mapping[str, Any],
    *,
    timeout: Optional[int] = None,
) -> dict:
    """Run one streamed chat completion against the prompt enhancer.

    Returns ``{"ok", "text", "reasoning", "model", "latency_ms", "usage"}`` and
    raises :class:`~app.errors.ImageIntError` on a transport or upstream error.
    """

    if not settings.configured:
        raise not_configured("Es ist kein Modellserver für den Prompt-Enhancer konfiguriert.")

    body = dict(payload)
    body["stream"] = True
    effective_timeout = timeout or settings.timeout

    text: list[str] = []
    reasoning: list[str] = []
    usage: dict[str, Any] = {}
    model = str(body.get("model") or settings.model)

    started_at = time.monotonic()
    try:
        # A generous read timeout: on CPU the thinking block alone can run for
        # minutes, and with streaming the timer only covers the gap between two
        # chunks rather than the whole generation.
        async with httpx.AsyncClient(timeout=effective_timeout) as client:
            async with client.stream(
                "POST",
                settings.url + "/v1/chat/completions",
                json=body,
                headers=_headers(settings.token, json_body=True),
            ) as response:
                if response.status_code != 200:
                    raw = await response.aread()
                    raise upstream_unavailable(
                        "enhancer_error",
                        _error_text(raw) or f"Modell-Fehler (HTTP {response.status_code})",
                    )

                async for line in response.aiter_lines():
                    chunk = _sse_data(line)
                    if chunk is None:
                        continue
                    if chunk == "[DONE]":
                        break
                    try:
                        decoded = json.loads(chunk)
                    except ValueError:
                        continue
                    if not isinstance(decoded, dict):
                        continue
                    if isinstance(decoded.get("usage"), dict):
                        usage = decoded["usage"]
                    if decoded.get("model"):
                        model = str(decoded["model"])
                    for choice in decoded.get("choices") or []:
                        delta = choice.get("delta") or choice.get("message") or {}
                        if not isinstance(delta, dict):
                            continue
                        # `--reasoning-parser qwen3` splits the thinking block
                        # into its own field; without it the text arrives inline
                        # and enhancer.split_thinking recovers it.
                        for key in ("reasoning_content", "reasoning"):
                            part = delta.get(key)
                            if isinstance(part, str) and part:
                                reasoning.append(part)
                        part = delta.get("content")
                        if isinstance(part, str) and part:
                            text.append(part)
                        elif isinstance(part, list):
                            for piece in part:
                                if isinstance(piece, Mapping) and piece.get("type") == "text":
                                    text.append(str(piece.get("text") or ""))
    except httpx.TimeoutException:
        raise upstream_timeout(
            "enhancer_timeout",
            f"Der Prompt-Enhancer hat nach {effective_timeout} s nicht geantwortet.",
        )
    except httpx.HTTPError as exc:
        raise upstream_unavailable("enhancer_unavailable", f"Prompt-Enhancer nicht erreichbar: {exc}")

    latency_ms = int(round((time.monotonic() - started_at) * 1000))
    return {
        "ok": True,
        "text": "".join(text),
        "reasoning": "".join(reasoning),
        "model": model,
        "latency_ms": latency_ms,
        "usage": {
            "prompt": int(usage.get("prompt_tokens") or 0),
            "completion": int(usage.get("completion_tokens") or 0),
            "total": int(usage.get("total_tokens") or 0),
        },
    }


def _error_text(raw: bytes) -> str:
    try:
        decoded = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return ""
    return _upstream_message(decoded)


def _sse_data(line: str) -> Optional[str]:
    """Return the payload of one SSE ``data:`` line, or ``None`` for filler."""

    if not line:
        return None
    if line.startswith("data:"):
        return line[5:].strip()
    # Some deployments answer without the SSE framing when streaming is refused.
    stripped = line.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    return None


def build_image_payload(
    settings: ImageSettings,
    prompt: str,
    width: int,
    height: int,
    *,
    negative_prompt: str = "",
    steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    """Assemble the image-server request body for the configured route."""

    effective_steps = settings.steps if steps is None else int(steps)
    effective_negative = (negative_prompt or settings.negative_prompt).strip()

    if settings.route == "chat":
        # The Omni example drives image generation through the chat route, with
        # the image parameters in `extra_body`.
        extra: dict[str, Any] = {
            "height": int(height),
            "width": int(width),
            "num_inference_steps": effective_steps,
            "true_cfg_scale": settings.true_cfg_scale,
        }
        if effective_negative:
            extra["negative_prompt"] = effective_negative
        if seed is not None:
            extra["seed"] = int(seed)
        return {
            "model": settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "extra_body": extra,
        }

    payload: dict[str, Any] = {
        "model": settings.model,
        "prompt": prompt,
        # The dedicated Images API takes the canvas as `size`; `width`/`height`
        # belong to the chat-completions route and are ignored here.
        "size": f"{width}x{height}",
        "n": 1,
        "num_inference_steps": effective_steps,
        "true_cfg_scale": settings.true_cfg_scale,
    }
    if effective_negative:
        payload["negative_prompt"] = effective_negative
    if seed is not None:
        payload["seed"] = int(seed)
    return payload


async def generate_image(
    settings: ImageSettings,
    payload: Mapping[str, Any],
    *,
    timeout: Optional[int] = None,
) -> dict:
    """Render one image and return it as bytes.

    Returns ``{"ok", "image", "content_type", "width", "height", "model",
    "latency_ms", "seed", "raw_meta"}``.
    """

    if not settings.configured:
        raise not_configured("Es ist kein Modellserver für die Bilderzeugung konfiguriert.")

    effective_timeout = timeout or settings.timeout
    body = dict(payload)
    body["stream"] = False
    path = IMAGE_ROUTE_PATHS.get(settings.route, IMAGE_ROUTE_PATHS["images"])

    started_at = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=effective_timeout) as client:
            response = await client.post(
                settings.url + path,
                json=body,
                headers=_headers(settings.token, json_body=True),
            )
    except httpx.TimeoutException:
        raise upstream_timeout(
            "image_timeout",
            f"Die Bilderzeugung hat nach {effective_timeout} s nicht geantwortet.",
        )
    except httpx.HTTPError as exc:
        raise upstream_unavailable("image_unavailable", f"Bildserver nicht erreichbar: {exc}")

    latency_ms = int(round((time.monotonic() - started_at) * 1000))

    try:
        decoded = response.json()
    except ValueError:
        decoded = None

    if response.status_code != 200 or not isinstance(decoded, dict):
        raise upstream_unavailable(
            "image_error",
            _upstream_message(decoded) or f"Bild-Fehler (HTTP {response.status_code})",
        )

    image, content_type = _extract_image(decoded)
    if image is None:
        raise upstream_unavailable(
            "image_empty",
            "Der Bildserver hat geantwortet, aber kein Bild geliefert.",
        )

    meta = _extract_meta(decoded)
    return {
        "ok": True,
        "image": image,
        "content_type": content_type,
        "width": int(meta.get("width") or 0),
        "height": int(meta.get("height") or 0),
        "model": str(meta.get("model") or settings.model),
        "seed": meta.get("seed"),
        "latency_ms": latency_ms,
        "raw_meta": meta,
    }


def _extract_image(decoded: Mapping[str, Any]) -> tuple[Optional[bytes], str]:
    """Find the picture in either response shape and decode it to bytes."""

    for entry in decoded.get("data") or []:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("b64_json"):
            try:
                return base64.b64decode(str(entry["b64_json"])), "image/png"
            except (ValueError, TypeError):
                continue
        url = str(entry.get("url") or "")
        if url.startswith("data:"):
            return _decode_data_uri(url)

    # Chat-completions shape, used by the "chat" compatibility route.
    for choice in decoded.get("choices") or []:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message") or {}
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            image_url = part.get("image_url") or {}
            url = str(image_url.get("url") or "") if isinstance(image_url, Mapping) else ""
            if url.startswith("data:"):
                return _decode_data_uri(url)

    return None, ""


def _decode_data_uri(uri: str) -> tuple[Optional[bytes], str]:
    header, _, payload = uri.partition(",")
    if not payload:
        return None, ""
    content_type = "image/png"
    if header.startswith("data:") and ";" in header:
        content_type = header[5:].split(";", 1)[0] or content_type
    try:
        return base64.b64decode(payload), content_type
    except (ValueError, TypeError):
        return None, ""


def _extract_meta(decoded: Mapping[str, Any]) -> dict:
    """Dimensions, model and seed as far as the response reports them."""

    for entry in decoded.get("data") or []:
        if isinstance(entry, Mapping):
            return {
                "width": entry.get("width"),
                "height": entry.get("height"),
                "model": entry.get("model"),
                "seed": entry.get("seed"),
            }
    return {
        "model": decoded.get("model"),
        "seed": decoded.get("seed"),
    }
