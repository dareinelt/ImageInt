"""Shared fixtures for the ImageInt gateway tests.

The tests exercise the whole chain -- HTTP route, job registry, pipeline,
prompt enhancer, vLLM client -- against a fake vLLM instead of mocking each
module separately. That is deliberate: the parts that are easy to get wrong here
are the seams (the SSE parsing of the enhancer's answer, the 202/job flow, the
data-URI extraction), and they only exist between the modules.

The fake upstream is an ``httpx.MockTransport`` swapped in for
``httpx.AsyncClient``, so the real client code -- timeouts, streaming, header
handling -- runs against canned bytes.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import httpx
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, vllm  # noqa: E402

ENHANCER_HOST = "enhancer"
IMAGE_HOST = "image"

ENHANCER_URL = f"http://{ENHANCER_HOST}:8000"
IMAGE_URL = f"http://{IMAGE_HOST}:8000"

#: The answer contract of the documented prompt enhancer.
ENHANCER_ANSWER = {
    "rewritten_prompt": (
        "A red panda standing in fresh powder snow, thick winter fur, soft "
        "overcast daylight, shallow depth of field, photographic realism."
    ),
    "wh_ratio": "3:2",
}


def png_bytes(size: tuple = (8, 8), color: tuple = (200, 30, 30)) -> bytes:
    """A real PNG, so the tests can prove what was stored is one."""

    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def sse_body(chunks: list) -> bytes:
    """Encode chat-completion deltas as an SSE stream, the way vLLM sends them."""

    lines = []
    for chunk in chunks:
        lines.append(f"data: {json.dumps(chunk)}")
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


class FakeVllm:
    """Mutable state of the two fake model servers."""

    def __init__(self) -> None:
        #: HTTP status of ``GET /health`` per host. 503 means "still loading".
        self.health_status = {ENHANCER_HOST: 200, IMAGE_HOST: 200}
        #: Answer of the enhancer, split into the fields vLLM would report.
        self.enhancer_reasoning = "The user wants a picture of a red panda. Ratio 3:2."
        self.enhancer_answer = json.dumps(ENHANCER_ANSWER, ensure_ascii=False)
        #: Set to a raw string to bypass the answer object entirely.
        self.enhancer_raw = ""
        #: Status of ``POST /v1/images/generations``.
        self.image_status = 200
        self.image_size = (8, 8)
        #: ``images`` answers with ``data[0].b64_json``, ``chat`` with the
        #: Omni chat shape ``choices[0].message.content[0].image_url.url``.
        self.image_shape = "images"
        self.image_error = {"error": {"message": "Modell nicht geladen"}}
        #: Every request the fake saw, as ``(method, path)``.
        self.calls: list = []
        self.image_requests: list = []
        self.enhancer_requests: list = []
        self.image_headers: list = []
        self.enhancer_headers: list = []
        #: Hosts that were asked to start a lazy load through ``/v1/warmup``.
        self.warmups: list = []
        #: Answer of ``POST /v1/warmup``; 202 is what the image server returns
        #: while the load runs, 404 is what any other server returns.
        self.warmup_status = 202
        #: Paths whose requests raise a timeout instead of answering.
        self.timeout_paths: set = set()
        #: Simulate a closed port for every request.
        self.offline = False

    def reset_calls(self) -> None:
        self.calls.clear()
        self.image_requests.clear()
        self.enhancer_requests.clear()
        self.image_headers.clear()
        self.enhancer_headers.clear()
        self.warmups.clear()

    # -- responses --------------------------------------------------------- #

    def enhancer_stream(self) -> bytes:
        if self.enhancer_raw:
            answer = self.enhancer_raw
        else:
            answer = self.enhancer_answer
        chunks = []
        if self.enhancer_reasoning:
            chunks.append({"choices": [{"delta": {"reasoning_content": self.enhancer_reasoning}}]})
        for piece in answer:
            chunks.append({"choices": [{"delta": {"content": piece}}]})
        chunks.append(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 640, "completion_tokens": 128, "total_tokens": 768},
                "model": "Qwen-Image-2.1-PE-T2I",
            }
        )
        return sse_body(chunks)

    def image_body(self) -> dict:
        width, height = self.image_size
        encoded = base64.b64encode(png_bytes(self.image_size)).decode("ascii")
        if self.image_shape == "chat":
            return {
                "created": 0,
                "model": "Qwen/Qwen-Image-2.1",
                "seed": 42,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                                }
                            ],
                        },
                    }
                ],
            }
        return {
            "created": 0,
            "data": [
                {
                    "b64_json": encoded,
                    "width": width,
                    "height": height,
                    "model": "Qwen/Qwen-Image-2.1",
                    "seed": 42,
                }
            ],
        }

    # -- transport --------------------------------------------------------- #

    def handler(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host or ""
        path = request.url.path
        self.calls.append((request.method, path))

        if self.offline:
            raise httpx.ConnectError("Verbindung abgelehnt (Test)")

        if path in self.timeout_paths:
            raise httpx.ReadTimeout("Zeitüberschreitung (Test)")

        if path == "/health":
            status = self.health_status.get(host, 200)
            if status == 200:
                return httpx.Response(200, content=b"")
            return httpx.Response(status, content=b"")

        if path == "/v1/warmup":
            self.warmups.append(host)
            return httpx.Response(self.warmup_status, json={"ok": True, "state": "loading"})

        if path == "/v1/chat/completions" and host == ENHANCER_HOST:
            self.enhancer_requests.append(json.loads(request.content.decode("utf-8")))
            self.enhancer_headers.append(dict(request.headers))
            if self.health_status.get(host, 200) != 200:
                return httpx.Response(self.health_status.get(host, 200), content=b"")
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=self.enhancer_stream(),
            )

        if path == "/v1/images/generations" or (
            path == "/v1/chat/completions" and host == IMAGE_HOST
        ):
            self.image_requests.append(json.loads(request.content.decode("utf-8")))
            self.image_headers.append(dict(request.headers))
            if self.image_status != 200:
                return httpx.Response(self.image_status, json=self.image_error)
            return httpx.Response(200, json=self.image_body())

        return httpx.Response(404, json={"error": {"message": f"unbekannter Pfad {path}"}})


@pytest.fixture
def fake() -> FakeVllm:
    return FakeVllm()


@pytest.fixture(autouse=True)
def fake_transport(monkeypatch, fake):
    """Route every httpx request of the app through the fake upstream."""

    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(fake.handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(vllm.httpx, "AsyncClient", factory)
    return fake


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    """A complete, deterministic configuration, and no leakage between tests."""

    values = {
        "IMAGEINT_TOKEN": "test-token",
        "IMAGEINT_PE_URL": ENHANCER_URL,
        "IMAGEINT_IMAGE_URL": IMAGE_URL,
        "IMAGEINT_STORAGE_DIR": str(tmp_path / "images"),
        "IMAGEINT_SYNC_TIMEOUT": "30",
        # No caching and no grace period: the tests want the state they set up,
        # not the state a warm-up produced.
        "IMAGEINT_HEALTH_CACHE_SECONDS": "0",
        "IMAGEINT_STARTING_GRACE_SECONDS": "0",
        "IMAGEINT_LOADING_RETRY_AFTER": "15",
        "IMAGEINT_LOG_LEVEL": "WARNING",
        "IMAGEINT_HOST_CHECK": "auto",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    config.reset_settings()
    yield monkeypatch
    config.reset_settings()


@pytest.fixture
def client_factory(environment, fake_transport):
    """Build a ``TestClient`` with extra environment overrides.

    Used by the tests that need a different configuration than the shared
    ``environment`` -- a chat-route image server, a published base URL, a
    single job slot. Must be entered as a context manager so the app's lifespan
    runs.
    """

    from fastapi.testclient import TestClient

    from app.main import app

    def build(**overrides):
        for name, value in overrides.items():
            environment.setenv(name, str(value))
        config.reset_settings()
        client = TestClient(app)
        token = str(overrides.get("IMAGEINT_TOKEN", "test-token"))
        if token:
            client.headers.update({"X-Auth-Token": token})
        return client

    return build


@pytest.fixture
def client(environment, fake_transport):
    """A ``TestClient`` with the app's lifespan running."""

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        test_client.headers.update({"X-Auth-Token": "test-token"})
        yield test_client
