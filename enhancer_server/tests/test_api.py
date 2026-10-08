"""The HTTP contract the gateway depends on.

Four things matter to the gateway and are asserted here:

1. ``GET /health`` answers 200 when ready, 503 while loading and 500 when the
   checkpoint could not be loaded. The loading window is minutes long on a CPU
   host, so "not ready yet" must never look like "broken".
2. ``POST /v1/chat/completions`` with ``stream: true`` answers the streamed
   OpenAI chat-completions shape the gateway's SSE reader expects: the thinking
   block in ``delta.reasoning_content``, the answer in ``delta.content``, a
   final ``finish_reason`` and ``data: [DONE]``.
3. The usage chunk only appears when ``stream_options.include_usage`` asks for
   it, and carries token counts rather than zeros.
4. Errors carry a message under ``error.message``, which is what the gateway
   forwards to the chat.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from enhancer_server.app import (
    MAX_MESSAGES,
    ChatCompletionRequest,
    ServerError,
    create_app,
    normalise,
)
from enhancer_server.config import Settings
from enhancer_server.engine import BaseEngine, EngineError

PROMPT = "Eine Katze im Regen, fotorealistisch"

MESSAGES = [
    {"role": "system", "content": [{"type": "text", "text": "Du bist der Enhancer."}]},
    {"role": "user", "content": [{"type": "text", "text": PROMPT}]},
]


def settings(**overrides) -> Settings:
    base = dict(engine="stub", preload=True, stub_delay=0.0)
    base.update(overrides)
    return Settings(**base)


def body(**overrides) -> dict:
    base = {"model": "Qwen/Qwen-Image-2.1-PE-T2I", "messages": MESSAGES}
    base.update(overrides)
    return base


class FailingEngine(BaseEngine):
    name = "failing"

    def _load(self) -> None:
        raise EngineError("Kein Speicher für die Gewichte.")


class LoadingEngine(BaseEngine):
    """An engine that never finishes loading, to pin the 503 window."""

    name = "loading"

    def ensure_loaded(self) -> None:
        self._state = "loading"

    def _load(self) -> None:  # pragma: no cover - never reached
        pass


class SlowEngine(BaseEngine):
    """An engine whose load only finishes when the test lets it."""

    name = "slow"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.release = threading.Event()

    def _load(self) -> None:
        assert self.release.wait(10) is True


@contextmanager
def running(settings_obj: Settings, engine_cls=None):
    """A started app, with the engine optionally replaced by a test double."""

    app = create_app(settings_obj)
    if engine_cls is not None:
        app.state.engine = engine_cls(settings_obj)
    with TestClient(app) as client:
        yield app, client


@pytest.fixture
def client():
    app = create_app(settings())
    with TestClient(app) as test_client:
        # The stub loads instantly, but it does so on a worker thread.
        assert app.state.engine.wait_ready(10) is True
        yield test_client


def events(text: str) -> list:
    """Parse an SSE body into its frames, keeping ``[DONE]`` as a marker."""

    parsed = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        parsed.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return parsed


def deltas(frames: list, field: str) -> str:
    parts = []
    for frame in frames:
        if frame == "[DONE]":
            continue
        for choice in frame.get("choices") or []:
            value = (choice.get("delta") or {}).get(field)
            if value:
                parts.append(value)
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Service routes
# --------------------------------------------------------------------------- #
def test_root_describes_the_service(client):
    payload = client.get("/").json()
    assert payload["service"] == "ImageInt enhancer server"
    assert payload["model"] == "Qwen/Qwen-Image-2.1-PE-T2I"
    assert payload["engine"] == "stub"


def test_health_is_ok_when_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["ok"] is True


def test_health_reports_a_loading_engine_as_503():
    with running(settings(preload=False), LoadingEngine) as (_app, client):
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "loading"
    assert response.json()["ok"] is False


def test_health_reports_a_broken_load_as_500():
    with running(settings(), FailingEngine) as (_app, client):
        response = client.get("/health")
    assert response.status_code == 500
    assert response.json()["status"] == "error"
    assert "Gewichte" in response.json()["message"]


def test_the_model_list_names_the_checkpoint(client):
    payload = client.get("/v1/models").json()
    assert payload["object"] == "list"
    entry = payload["data"][0]
    assert entry["id"] == "Qwen/Qwen-Image-2.1-PE-T2I"
    assert entry["device"] == "cpu"
    assert entry["dtype"] == "bfloat16"
    assert entry["engine"] == "stub"


# --------------------------------------------------------------------------- #
# The shared secret
# --------------------------------------------------------------------------- #
def test_a_configured_token_is_required_on_the_model_list():
    with running(settings(token="geheim")) as (_app, client):
        assert client.get("/v1/models").status_code == 401
        assert client.get("/v1/models", headers={"X-Auth-Token": "geheim"}).status_code == 200


def test_the_token_may_arrive_as_a_bearer_header():
    with running(settings(token="geheim")) as (_app, client):
        response = client.get(
            "/v1/models", headers={"Authorization": "Bearer geheim"}
        )
        assert response.status_code == 200


def test_a_wrong_token_is_rejected_with_the_openai_error_shape():
    with running(settings(token="geheim")) as (_app, client):
        response = client.post(
            "/v1/chat/completions", json=body(), headers={"X-Auth-Token": "falsch"}
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_health_and_the_root_stay_open_with_a_token():
    # The container healthcheck cannot send the secret, so these two routes must
    # keep answering without it.
    with running(settings(token="geheim")) as (_app, client):
        assert client.get("/health").status_code == 200
        assert client.get("/").status_code == 200


# --------------------------------------------------------------------------- #
# Warm-up
# --------------------------------------------------------------------------- #
def test_warmup_starts_a_load_that_was_not_preloaded():
    with running(settings(preload=False), SlowEngine) as (app, client):
        assert client.get("/health").status_code == 503

        first = client.post("/v1/warmup")
        assert first.status_code == 202
        assert first.json()["state"] == "loading"

        # Idempotent: a second nudge while the load runs changes nothing.
        assert client.post("/v1/warmup").status_code == 202

        app.state.engine.release.set()
        deadline = time.time() + 10
        while time.time() < deadline and client.get("/health").status_code != 200:
            time.sleep(0.02)
        assert client.get("/health").status_code == 200

        done = client.post("/v1/warmup")
    assert done.status_code == 200
    assert done.json()["state"] == "ready"


def test_warmup_on_a_preloaded_engine_is_a_no_op():
    with running(settings(preload=True), SlowEngine) as (app, client):
        app.state.engine.release.set()
        assert app.state.engine.wait_ready(10) is True
        response = client.post("/v1/warmup")
    assert response.status_code == 200
    assert response.json()["loading"] is False


# --------------------------------------------------------------------------- #
# The streamed contract
# --------------------------------------------------------------------------- #
def test_a_streamed_answer_splits_thinking_from_the_prompt(client):
    response = client.post(
        "/v1/chat/completions",
        json=body(stream=True, stream_options={"include_usage": True}),
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    frames = events(response.text)
    assert frames[-1] == "[DONE]"

    reasoning = deltas(frames, "reasoning_content")
    content = deltas(frames, "content")
    assert reasoning.startswith("Der Stub-Enhancer")
    assert content.startswith("{")
    assert '"rewritten_prompt"' in content
    # The thinking block must not leak into the answer: that is exactly what the
    # gateway's own split_thinking() would then have to undo.
    assert "</think>" not in content
    assert "<think>" not in content


def test_the_first_frame_announces_the_assistant_role(client):
    response = client.post("/v1/chat/completions", json=body(stream=True))
    frames = events(response.text)
    first = frames[0]
    assert first["object"] == "chat.completion.chunk"
    assert first["id"].startswith("chatcmpl-")
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": ""}


def test_the_last_frame_closes_the_stream_with_stop(client):
    response = client.post("/v1/chat/completions", json=body(stream=True))
    frames = [frame for frame in events(response.text) if frame != "[DONE]"]
    assert frames[-1]["choices"][0]["finish_reason"] == "stop"


def test_the_usage_chunk_carries_token_counts(client):
    response = client.post(
        "/v1/chat/completions",
        json=body(stream=True, stream_options={"include_usage": True}),
    )
    frames = events(response.text)
    usage_frames = [
        frame
        for frame in frames
        if frame != "[DONE]" and frame.get("choices") == [] and "usage" in frame
    ]
    assert len(usage_frames) == 1
    usage = usage_frames[0]["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_no_usage_chunk_without_the_option(client):
    response = client.post("/v1/chat/completions", json=body(stream=True))
    frames = events(response.text)
    assert not [frame for frame in frames if frame != "[DONE]" and "usage" in frame]


def test_the_answer_is_the_last_thing_before_done(client):
    response = client.post("/v1/chat/completions", json=body(stream=True))
    frames = [frame for frame in events(response.text) if frame != "[DONE]"]
    content = deltas(frames, "content")
    assert content.endswith("}")


# --------------------------------------------------------------------------- #
# The non-streamed contract
# --------------------------------------------------------------------------- #
def test_a_plain_request_answers_one_completion(client):
    response = client.post("/v1/chat/completions", json=body())
    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["finish_reason"] == "stop"
    message = payload["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert json.loads(message["content"])["rewritten_prompt"].startswith(
        "Stub-Aufbereitung"
    )
    assert message["reasoning_content"].startswith("Der Stub-Enhancer")
    assert payload["usage"]["prompt_tokens"] > 0
    assert payload["latency_ms"] >= 0


def test_the_thinking_block_can_be_switched_off_per_request(client):
    response = client.post(
        "/v1/chat/completions",
        json=body(chat_template_kwargs={"enable_thinking": False}),
    )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == ""
    assert message["content"].startswith("{")


def test_the_operator_setting_is_the_fallback_for_thinking():
    app = create_app(settings(enable_thinking=False))
    with TestClient(app) as client:
        assert app.state.engine.wait_ready(10) is True
        message = client.post("/v1/chat/completions", json=body()).json()["choices"][0][
            "message"
        ]
    assert message["reasoning_content"] == ""


def test_an_unknown_field_does_not_reject_the_request(client):
    response = client.post("/v1/chat/completions", json=body(quatsch=1, top_k=42))
    assert response.status_code == 200


def test_the_prompt_is_carried_into_the_answer(client):
    response = client.post("/v1/chat/completions", json=body())
    content = response.json()["choices"][0]["message"]["content"]
    assert PROMPT in content


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "payload",
    [
        {"messages": []},
        {"messages": [{"role": "user", "content": "   "}]},
        {"messages": [{"role": "", "content": "x"}]},
        {"messages": [{"content": "x"}]},
        {"messages": [{"role": "user", "content": "x"}] * (MAX_MESSAGES + 1)},
    ],
)
def test_a_malformed_request_is_a_400(client, payload):
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


def test_a_prompt_over_the_operator_limit_is_rejected(client):
    app = create_app(settings(max_prompt_chars=20))
    with TestClient(app) as limited:
        assert app.state.engine.wait_ready(10) is True
        response = limited.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "x" * 21}]},
        )
    assert response.status_code == 400
    assert "20 Zeichen" in response.json()["error"]["message"]


def test_a_non_positive_max_tokens_is_rejected(client):
    response = client.post("/v1/chat/completions", json=body(max_tokens=0))
    assert response.status_code == 400
    assert "positiv" in response.json()["error"]["message"]


def test_max_tokens_is_clamped_to_the_operator_budget():
    request = normalise(
        ChatCompletionRequest(**body(max_tokens=100000)), settings(max_new_tokens=64)
    )
    assert request.max_tokens == 64


def test_max_completion_tokens_is_accepted_as_an_alias():
    request = normalise(
        ChatCompletionRequest(**body(max_completion_tokens=128)),
        settings(max_new_tokens=64),
    )
    assert request.max_tokens == 64


def test_the_operator_sampling_defaults_fill_in_missing_fields():
    request = normalise(ChatCompletionRequest(**body()), settings())
    assert request.temperature == 1.0
    assert request.top_p == 0.95
    assert request.top_k == 20
    assert request.min_p == 0.0
    assert request.presence_penalty == 1.5
    assert request.max_tokens == 16256


def test_the_operator_seed_is_used_when_the_request_omits_one():
    assert normalise(ChatCompletionRequest(**body()), settings(seed=7)).seed == 7
    assert normalise(ChatCompletionRequest(**body(seed=99)), settings(seed=7)).seed == 99


def test_normalise_refuses_an_empty_message_list():
    with pytest.raises(ServerError):
        normalise(ChatCompletionRequest(messages=[]), settings())


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #
def test_a_loading_engine_answers_503_not_500():
    with running(settings(preload=False), LoadingEngine) as (_app, client):
        response = client.post("/v1/chat/completions", json=body())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "engine_unavailable"


def test_a_broken_engine_answers_500_with_the_reason():
    with running(settings(), FailingEngine) as (_app, client):
        response = client.post("/v1/chat/completions", json=body())
    assert response.status_code == 500
    assert "Gewichte" in response.json()["error"]["message"]
