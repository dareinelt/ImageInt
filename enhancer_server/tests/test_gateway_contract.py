"""Wire compatibility with the gateway.

The transformers-based enhancer replaced the vLLM container, and the whole
point of the replacement is that the gateway did *not* have to change. These
tests drive the gateway's own request builder and its own SSE reader against a
live enhancer server, so a drift in either direction fails here rather than in
production.

The server really is spoken to over a socket: ``chat_completion`` builds its
own ``httpx.AsyncClient``, so there is no transport to inject.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi.testclient import TestClient

from enhancer_server.app import create_app
from enhancer_server.config import Settings

ROOT = Path(__file__).resolve().parents[2]
GATEWAY = ROOT / "gateway"

# The gateway package is called `app`, the enhancer's is `enhancer_server`, so
# both can be imported side by side.
if str(GATEWAY) not in sys.path:
    sys.path.insert(0, str(GATEWAY))

vllm = pytest.importorskip("app.vllm")
config = pytest.importorskip("app.config")
enhancer = pytest.importorskip("app.enhancer")

MODEL = "Qwen/Qwen-Image-2.1-PE-T2I"
PROMPT = "Eine Katze im Regen, fotorealistisch"


def settings(**overrides) -> Settings:
    base = dict(engine="stub", preload=True, stub_delay=0.0)
    base.update(overrides)
    return Settings(**base)


def serve(settings_obj: Settings) -> tuple[str, uvicorn.Server, threading.Thread]:
    """Run the app on a free port and return its base URL."""

    app = create_app(settings_obj)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, name="imageint-enhancer-test", daemon=True)
    thread.start()

    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    if not server.started:  # pragma: no cover - only on a broken environment
        raise RuntimeError("Der Testserver ist nicht gestartet.")
    port = server.servers[0].sockets[0].getsockname()[1]
    return f"http://127.0.0.1:{port}", server, thread


def stop(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def base_url():
    url, server, thread = serve(settings())
    try:
        yield url
    finally:
        stop(server, thread)


@pytest.fixture(scope="module")
def secured_url():
    url, server, thread = serve(settings(token="geheim"))
    try:
        yield url
    finally:
        stop(server, thread)


def gateway_settings(url: str, **overrides) -> "config.EnhancerSettings":
    base = dict(url=url, model=MODEL, timeout=60)
    base.update(overrides)
    return config.EnhancerSettings(**base)


async def test_the_gateway_enhance_call_survives_the_round_trip(base_url):
    """The unmodified gateway call has to work against the new server."""

    result = await enhancer.enhance(gateway_settings(base_url), PROMPT)

    assert result["parse_ok"] is True
    assert result["prompt"].startswith("Stub-Aufbereitung")
    assert PROMPT in result["prompt"]
    assert result["model"] == MODEL
    assert result["width"] > 0 and result["height"] > 0
    assert result["usage"]["prompt"] > 0
    assert result["usage"]["completion"] > 0
    assert result["latency_ms"] >= 0


async def test_the_thinking_block_arrives_in_its_own_field(base_url):
    """The server takes the place of ``--reasoning-parser qwen3``.

    The gateway recovers an inline thinking block with ``split_thinking``; when
    the server splits it itself the answer must already be free of it, so the
    two mechanisms must not both fire and leave an empty answer behind.
    """

    settings_obj = gateway_settings(base_url)
    payload = enhancer.build_payload(settings_obj, PROMPT)
    result = await vllm.chat_completion(settings_obj, payload)

    assert result["ok"] is True
    assert result["reasoning"].strip()
    assert "<think>" not in result["reasoning"]

    thinking, answer = enhancer.split_thinking(result["text"])
    assert thinking == ""
    assert answer.startswith("{")
    assert enhancer.parse_answer(answer)["positive_prompt"].startswith("Stub-Aufbereitung")


async def test_the_sampling_profile_the_gateway_sends_is_accepted(base_url):
    """A sampling parameter transformers does not know must not be a 400."""

    settings_obj = gateway_settings(base_url)
    payload = enhancer.build_payload(settings_obj, PROMPT)
    assert payload["presence_penalty"] == settings_obj.presence_penalty
    assert payload["min_p"] == settings_obj.min_p

    result = await vllm.chat_completion(settings_obj, payload)
    assert result["ok"] is True


async def test_the_gateway_health_probe_reports_a_ready_server(base_url):
    probe = await vllm.health(base_url, "", MODEL)
    assert probe["ok"] is True
    assert probe["loading"] is False


async def test_the_shared_secret_the_gateway_sends_is_accepted(secured_url):
    """Both header spellings the gateway sends have to open the door."""

    settings_obj = gateway_settings(secured_url, token="geheim")
    result = await vllm.chat_completion(
        settings_obj, enhancer.build_payload(settings_obj, PROMPT)
    )
    assert result["ok"] is True

    wrong = gateway_settings(secured_url, token="falsch")
    with pytest.raises(Exception) as failure:
        await vllm.chat_completion(
            wrong, enhancer.build_payload(wrong, PROMPT)
        )
    assert "Nicht autorisiert" in str(failure.value)


def test_the_stub_engine_is_what_the_contract_tests_ran_against():
    """A guard against the tests silently passing with a different engine."""

    app = create_app(settings())
    with TestClient(app) as client:
        assert client.get("/").json()["engine"] == "stub"
