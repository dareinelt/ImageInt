"""The HTTP contract the gateway depends on.

Three things matter to the gateway and are therefore asserted here:

1. ``GET /health`` answers 200 when ready, 503 while loading and 500 when the
   model could not be loaded -- the loading window is minutes long on a CPU
   host, so "not ready yet" must never look like "broken".
2. ``POST /v1/images/generations`` answers the OpenAI Images shape, i.e. a
   base64 PNG under ``data[0].b64_json`` plus the size metadata.
3. Errors carry a message under ``error.message``, which is what the gateway
   forwards to the chat.
"""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from image_server.app import ImageRequest, create_app, resolve_size
from image_server.config import DEFAULT_MAX_PIXELS, Settings
from image_server.engine import BaseEngine, EngineError

PROMPT = "Ein Bergsee im Morgennebel, fotorealistisch"


def settings(**overrides) -> Settings:
    base = dict(engine="stub", preload=True, stub_delay=0.0, cpu_threads=0)
    base.update(overrides)
    return Settings(**base)


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


@pytest.fixture
def client():
    app = create_app(settings())
    with TestClient(app) as test_client:
        # The stub loads instantly, but it does so on a worker thread.
        assert app.state.engine.wait_ready(10) is True
        yield test_client


def decode(payload: dict) -> Image.Image:
    raw = base64.b64decode(payload["data"][0]["b64_json"])
    return Image.open(io.BytesIO(raw))


# --------------------------------------------------------------------------- #
# Service routes
# --------------------------------------------------------------------------- #
def test_root_describes_the_service(client):
    body = client.get("/").json()
    assert body["service"] == "ImageInt image server"
    assert body["model"] == "Qwen/Qwen-Image-2.1"
    assert body["engine"] == "stub"


def test_health_is_ok_when_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_health_is_503_while_loading():
    app = create_app(settings(preload=False))
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "loading"
    assert body["ok"] is False
    assert "geladen" in body["message"]


def test_health_is_500_when_the_model_could_not_be_loaded():
    app = create_app(settings())
    app.state.engine = FailingEngine(settings())
    app.state.engine.ensure_loaded()
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 500
    body = response.json()
    assert body["status"] == "error"
    assert "Speicher" in body["message"]


def test_warmup_starts_a_lazy_load_and_answers_at_once():
    app = create_app(settings(preload=False, stub_delay=0.0))
    with TestClient(app) as client:
        # PRELOAD=false leaves the engine idle: /health says 503 and nothing
        # would ever start the load unless somebody asked for a render.
        assert client.get("/health").status_code == 503
        response = client.post("/v1/warmup")
        assert response.status_code == 202
        assert response.json()["state"] in ("idle", "loading")
        assert response.json()["loading"] is True
        # The load runs in the background and finishes on its own.
        assert app.state.engine.wait_ready(10) is True
        done = client.post("/v1/warmup")
    assert done.status_code == 200
    assert done.json()["state"] == "ready"
    assert done.json()["loading"] is False


def test_warmup_is_idempotent_while_loading():
    app = create_app(settings(preload=False))
    app.state.engine = LoadingEngine(settings(preload=False))
    with TestClient(app) as client:
        first = client.post("/v1/warmup")
        second = client.post("/v1/warmup")
    # An engine that is already loading must not be asked to load again; both
    # calls answer the same 202 with the same state.
    assert first.status_code == second.status_code == 202
    assert first.json() == second.json()


def test_warmup_requires_the_token():
    app = create_app(settings(token="geheim"))
    with TestClient(app) as client:
        assert client.post("/v1/warmup").status_code == 401
        assert client.post("/v1/warmup", headers={"X-Auth-Token": "geheim"}).status_code == 200


def test_models_lists_the_served_checkpoint(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    entry = body["data"][0]
    assert entry["id"] == "Qwen/Qwen-Image-2.1"
    assert entry["device"] == "cpu"
    assert entry["dtype"] == "bfloat16"
    assert entry["quant"] == "none"
    assert entry["engine"] == "stub"


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def test_generation_returns_a_base64_png(client):
    response = client.post("/v1/images/generations", json={"prompt": PROMPT})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "Qwen/Qwen-Image-2.1"
    assert body["engine"] == "stub"
    assert body["latency_ms"] >= 0

    entry = body["data"][0]
    assert entry["model"] == "Qwen/Qwen-Image-2.1"
    assert entry["seed"] is not None
    # The default canvas is the first documented aspect ratio.
    assert (entry["width"], entry["height"]) == (2048, 2048)
    assert body["size"] == "2048x2048"
    with decode(body) as image:
        assert image.size == (2048, 2048)
        assert image.format == "PNG"


@pytest.mark.parametrize(
    "size,expected",
    [
        ("1024x1024", (1024, 1024)),
        ("2400x1792", (2400, 1792)),
        ("2752x1536", (2752, 1536)),
        ("1536x2752", (1536, 2752)),
        ("1000x1000", (992, 992)),
    ],
)
def test_generation_honours_the_size(client, size, expected):
    body = client.post(
        "/v1/images/generations", json={"prompt": PROMPT, "size": size}
    ).json()
    entry = body["data"][0]
    assert (entry["width"], entry["height"]) == expected
    assert body["size"] == f"{expected[0]}x{expected[1]}"


def test_width_and_height_are_accepted_as_an_alternative_to_size(client):
    body = client.post(
        "/v1/images/generations",
        json={"prompt": PROMPT, "width": 1024, "height": 768},
    ).json()
    assert body["size"] == "1024x768"


def test_a_seed_makes_the_render_reproducible(client):
    first = client.post(
        "/v1/images/generations", json={"prompt": PROMPT, "seed": 1234, "size": "512x512"}
    ).json()
    second = client.post(
        "/v1/images/generations", json={"prompt": PROMPT, "seed": 1234, "size": "512x512"}
    ).json()
    assert first["data"][0]["seed"] == 1234
    assert first["data"][0]["b64_json"] == second["data"][0]["b64_json"]


def test_steps_and_guidance_are_passed_through(client):
    body = client.post(
        "/v1/images/generations",
        json={
            "prompt": PROMPT,
            "num_inference_steps": 8,
            "true_cfg_scale": 4.0,
            "negative_prompt": "unscharf",
            "size": "512x512",
        },
    ).json()
    assert body["data"][0]["width"] == 512


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #
def test_a_malformed_size_is_rejected(client):
    response = client.post("/v1/images/generations", json={"prompt": PROMPT, "size": "groß"})
    assert response.status_code == 400
    assert "BREITExHÖHE" in response.json()["error"]["message"]


def test_a_canvas_over_the_pixel_budget_is_rejected(client):
    response = client.post(
        "/v1/images/generations", json={"prompt": PROMPT, "size": "4096x4096"}
    )
    assert response.status_code == 400
    assert str(DEFAULT_MAX_PIXELS) in response.json()["error"]["message"]


def test_more_than_one_image_is_rejected(client):
    response = client.post("/v1/images/generations", json={"prompt": PROMPT, "n": 2})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_n"


def test_an_empty_prompt_is_rejected_by_the_schema(client):
    assert client.post("/v1/images/generations", json={"prompt": ""}).status_code == 422


def test_an_overlong_prompt_is_rejected():
    app = create_app(settings(max_prompt_chars=32))
    with TestClient(app) as client:
        response = client.post("/v1/images/generations", json={"prompt": "x" * 64})
    assert response.status_code == 400
    assert "32 Zeichen" in response.json()["error"]["message"]


def test_a_broken_engine_is_reported_as_500_with_a_message():
    app = create_app(settings())
    broken = FailingEngine(settings())
    broken.ensure_loaded()
    app.state.engine = broken
    with TestClient(app) as client:
        response = client.post("/v1/images/generations", json={"prompt": PROMPT})
    assert response.status_code == 500
    assert "Speicher" in response.json()["error"]["message"]


def test_a_loading_engine_is_reported_as_503():
    app = create_app(settings(preload=False))
    app.state.engine = LoadingEngine(settings(preload=False))
    with TestClient(app) as client:
        # The model is not there yet: the gateway must be told to retry rather
        # than to give up, and the render is never attempted.
        response = client.post("/v1/images/generations", json={"prompt": PROMPT})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "engine_unavailable"


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
def test_requests_are_open_without_a_configured_token(client):
    assert client.post("/v1/images/generations", json={"prompt": PROMPT}).status_code == 200


def test_a_configured_token_is_enforced():
    app = create_app(settings(token="geheim"))
    with TestClient(app) as client:
        app.state.engine.ensure_loaded()
        assert client.get("/health").status_code == 200  # health stays open
        assert client.get("/v1/models").status_code == 401
        assert client.post("/v1/images/generations", json={"prompt": PROMPT}).status_code == 401

        headers = {"X-Auth-Token": "geheim"}
        assert client.get("/v1/models", headers=headers).status_code == 200
        assert (
            client.post(
                "/v1/images/generations", json={"prompt": PROMPT}, headers=headers
            ).status_code
            == 200
        )

        bearer = {"Authorization": "Bearer geheim"}
        assert client.get("/v1/models", headers=bearer).status_code == 200
        assert client.get("/v1/models", headers={"X-Auth-Token": "falsch"}).status_code == 401


def test_the_unauthorized_body_carries_a_message():
    app = create_app(settings(token="geheim"))
    with TestClient(app) as client:
        body = client.get("/v1/models").json()
    assert body["error"]["code"] == "unauthorized"
    assert "Token" in body["error"]["message"]


# --------------------------------------------------------------------------- #
# resolve_size()
# --------------------------------------------------------------------------- #
def test_resolve_size_defaults_to_the_first_documented_ratio():
    assert resolve_size(ImageRequest(prompt="x"), settings()) == (2048, 2048)


def test_resolve_size_prefers_size_over_width_and_height():
    payload = ImageRequest(prompt="x", size="1024x768", width=512, height=512)
    assert resolve_size(payload, settings()) == (1024, 768)


def test_resolve_size_accepts_the_multiplication_sign():
    assert resolve_size(ImageRequest(prompt="x", size="1024×768"), settings()) == (1024, 768)


@pytest.mark.parametrize("size", ["0x0", "-64x64", "abc", "1024", "1024x"])
def test_resolve_size_rejects_nonsense(size):
    with pytest.raises(ValueError):
        resolve_size(ImageRequest(prompt="x", size=size), settings())


def test_resolve_size_respects_a_custom_budget():
    tight = settings(max_pixels=1024 * 1024)
    assert resolve_size(ImageRequest(prompt="x", size="1024x1024"), tight) == (1024, 1024)
    with pytest.raises(ValueError):
        resolve_size(ImageRequest(prompt="x", size="2048x2048"), tight)
