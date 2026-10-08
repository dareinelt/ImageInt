"""End-to-end tests of the HTTP surface.

Everything below goes through the real FastAPI app -- routing, authentication,
the job registry, the pipeline, the enhancer client and the vLLM client -- with
only the two upstream model servers replaced by the fake in ``conftest``.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from conftest import ENHANCER_ANSWER, ENHANCER_HOST, IMAGE_HOST, png_bytes
from app import main as main_module

GENERATE = "/v1/images/generations"


def poll_until_done(client, job_id: str, tries: int = 50) -> dict:
    """Poll a job the way a client would, until it reaches a terminal state."""

    body = {}
    for _ in range(tries):
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        if body["status"] in ("done", "error"):
            return body
    pytest.fail(f"Auftrag {job_id} wurde nicht fertig: {body}")


def block_the_runner(client) -> None:
    """Occupy every generation slot, so the next request must be refused."""

    async def never(job):
        job.stage = "rendering"
        await asyncio.Event().wait()

    main_module.runtime().registry.runner = never


# --------------------------------------------------------------------------- #
# Service endpoints
# --------------------------------------------------------------------------- #
def test_root_banner_names_the_tool(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "imageint"
    assert body["tool"] == "generate_image"
    assert GENERATE in body["endpoints"]


def test_health_needs_no_token(client):
    response = client.get("/health", headers={"X-Auth-Token": ""})
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_ready_needs_no_token_and_reports_ready(client):
    response = client.get("/v1/ready", headers={"X-Auth-Token": ""})
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["state"] == "ready"
    assert body["components"]["enhancer"]["ready"] is True
    assert body["components"]["image"]["ready"] is True


def test_ready_reports_a_loading_model_with_retry_after(client, fake):
    fake.health_status[IMAGE_HOST] = 503
    response = client.get("/v1/ready", headers={"X-Auth-Token": ""})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "15"
    body = response.json()
    assert body["ok"] is False
    assert body["state"] == "loading"
    assert body["transient"] is True


def test_models_reports_both_components(client):
    body = client.get("/v1/models").json()
    assert body["models"]["enhancer"]["role"] == "prompt-enhancer"
    assert body["models"]["image"]["role"] == "image-generation"
    assert body["quant"] == "none"


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #
def test_a_missing_token_is_unauthorized(client):
    response = client.get("/v1/config", headers={"X-Auth-Token": ""})
    assert response.status_code == 401
    assert response.json()["ok"] is False
    assert response.json()["error"] == "unauthorized"


def test_a_wrong_token_is_rejected(client):
    response = client.get("/v1/config", headers={"X-Auth-Token": "falsch"})
    assert response.status_code == 401


def test_a_bearer_token_is_accepted(client):
    response = client.get("/v1/config", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200


def test_an_empty_token_disables_authentication(client_factory):
    with client_factory(IMAGEINT_TOKEN="") as client:
        assert client.get("/v1/config").status_code == 200


# --------------------------------------------------------------------------- #
# Error envelope
# --------------------------------------------------------------------------- #
def test_an_unknown_job_is_not_found(client):
    response = client.get("/v1/jobs/gibtesnicht")
    assert response.status_code == 404
    body = response.json()
    assert body["ok"] is False
    assert body["error"] == "not_found"
    assert "gibtesnicht" in body["message"]


def test_a_malformed_body_uses_the_envelope(client):
    response = client.post(GENERATE, json={"kein_prompt": 1})
    assert response.status_code == 400
    body = response.json()
    assert body["ok"] is False
    assert body["error"] == "bad_request"


def test_an_empty_prompt_is_rejected(client):
    response = client.post(GENERATE, json={"prompt": "   "})
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_a_too_long_prompt_is_rejected(client_factory):
    with client_factory(IMAGEINT_MAX_PROMPT_CHARS=16) as client:
        response = client.post(GENERATE, json={"prompt": "x" * 17})
    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_config_masks_the_token(client):
    body = client.get("/v1/config").json()
    assert "test-token" not in json.dumps(body)
    assert body["token_masked"] == "te****en"
    assert body["image"]["quant"] == "none"
    assert body["image"]["route"] == "images"
    assert body["image"]["route_path"] == "/v1/images/generations"


def test_config_reports_the_chat_route_when_selected(client_factory):
    with client_factory(IMAGEINT_IMAGE_ROUTE="chat") as client:
        body = client.get("/v1/config").json()
    assert body["image"]["route"] == "chat"
    assert body["image"]["route_path"] == "/v1/chat/completions"


def test_ratios_are_the_documented_ones(client):
    body = client.get("/v1/ratios").json()
    assert body["documented"]["16:9"] == {"width": 2752, "height": 1536}
    assert body["documented"]["1:1"] == {"width": 2048, "height": 2048}
    assert body["multiple"] == 64
    # The budget must not shrink a documented canvas.
    assert body["max_pixels"] >= 2752 * 1536


def test_the_public_url_overrides_the_request_base_url(client_factory):
    with client_factory(IMAGEINT_PUBLIC_URL="https://bild.example.test/") as client:
        job_id = client.post(GENERATE, json={"prompt": "Ein Berg", "wait": False}).json()[
            "job_id"
        ]
        body = client.get(f"/v1/jobs/{job_id}").json()
    assert body["image_url"] == f"https://bild.example.test/v1/images/{job_id}"


# --------------------------------------------------------------------------- #
# Prompt enhancement
# --------------------------------------------------------------------------- #
def test_enhance_returns_the_rewritten_prompt_and_ratio(client, fake):
    body = client.post("/v1/enhance", json={"prompt": "Ein roter Panda im Schnee"}).json()
    assert body["ok"] is True
    assert body["prompt"] == ENHANCER_ANSWER["rewritten_prompt"]
    assert body["wh_ratio"] == "3:2"
    assert (body["width"], body["height"]) == (2528, 1696)
    assert body["parse_ok"] is True
    assert body["model"] == "Qwen-Image-2.1-PE-T2I"


def test_enhance_rejects_an_empty_prompt(client):
    response = client.post("/v1/enhance", json={"prompt": " "})
    assert response.status_code == 400
    assert response.json()["error"] == "bad_request"


def test_enhance_is_503_while_the_enhancer_loads(client, fake):
    fake.health_status[ENHANCER_HOST] = 503
    response = client.post("/v1/enhance", json={"prompt": "Ein Berg"})
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "service_loading"
    assert body["component"] == "enhancer"
    assert response.headers["Retry-After"] == "15"


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def test_generation_returns_the_finished_image(client, fake):
    response = client.post(GENERATE, json={"prompt": "Ein roter Panda im Schnee"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ok"] is True
    assert body["status"] == "done"
    assert body["wh_ratio"] == "3:2"
    assert base64.b64decode(body["b64_json"]).startswith(b"\x89PNG")
    assert body["status_url"].endswith(f"/v1/jobs/{body['job_id']}")


def test_generation_sends_the_enhanced_prompt_to_the_image_model(client, fake):
    client.post(GENERATE, json={"prompt": "Ein roter Panda im Schnee"})
    payload = fake.image_requests[-1]
    assert payload["prompt"] == ENHANCER_ANSWER["rewritten_prompt"]
    assert payload["size"] == "2528x1696"
    assert payload["num_inference_steps"] == 40
    assert payload["true_cfg_scale"] == 1.0


def test_generation_can_skip_the_enhancer(client, fake):
    client.post(GENERATE, json={"prompt": "Ein roter Panda", "enhance": False})
    assert fake.enhancer_requests == []
    assert fake.image_requests[-1]["prompt"] == "Ein roter Panda"


def test_a_client_that_omits_enhance_follows_the_operator_setting(client_factory, fake):
    # LLMInt only ever sends a prompt, so the default has to be "whatever the
    # operator configured". Otherwise IMAGEINT_PE_ENABLED=false would break
    # every request with an error the client cannot act on.
    with client_factory(IMAGEINT_PE_ENABLED="false") as client:
        response = client.post(GENERATE, json={"prompt": "Ein Berg"})
    assert response.status_code == 200, response.text
    assert fake.enhancer_requests == []
    assert fake.image_requests[-1]["prompt"] == "Ein Berg"


def test_a_disabled_enhancer_still_honours_an_explicit_request(client_factory, fake):
    # Asking for a switched-off enhancer is a real conflict and keeps its own
    # explanation rather than a generic loading error.
    with client_factory(IMAGEINT_PE_ENABLED="false") as client:
        response = client.post(GENERATE, json={"prompt": "Ein Berg", "enhance": True})
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "not_configured"
    assert "IMAGEINT_PE_ENABLED" in body["message"]
    assert fake.image_requests == []


def test_generation_honours_an_explicit_ratio(client, fake):
    client.post(GENERATE, json={"prompt": "Ein Berg", "ratio": "16:9"})
    assert fake.image_requests[-1]["size"] == "2752x1536"


def test_generation_uses_the_chat_route_when_configured(client_factory, fake):
    fake.image_shape = "chat"
    with client_factory(IMAGEINT_IMAGE_ROUTE="chat") as client:
        response = client.post(
            GENERATE, json={"prompt": "Ein Berg", "enhance": False, "ratio": "16:9"}
        )
    assert response.status_code == 200, response.text
    assert base64.b64decode(response.json()["b64_json"]).startswith(b"\x89PNG")
    # No enhancer request was made, so this chat call is the image render.
    assert ("POST", "/v1/chat/completions") in fake.calls
    payload = fake.image_requests[-1]
    assert payload["messages"] == [{"role": "user", "content": "Ein Berg"}]
    assert payload["extra_body"]["width"] == 2752
    assert payload["extra_body"]["height"] == 1536
    assert payload["extra_body"]["num_inference_steps"] == 40


def test_generation_without_wait_returns_202_and_a_pollable_job(client):
    response = client.post(GENERATE, json={"prompt": "Ein Berg", "wait": False})
    assert response.status_code == 202
    body = response.json()
    assert body["ok"] is True
    assert body["poll_after_seconds"] == 15
    assert response.headers["Retry-After"] == "15"

    job = poll_until_done(client, body["job_id"])
    assert job["ok"] is True
    assert job["image_url"].endswith(f"/v1/images/{body['job_id']}")

    image = client.get(job["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    # The canvas the model reports wins over the requested one, so the
    # metadata describes the file that really exists (the fake renders 8x8).
    assert image.headers["X-ImageInt-Width"] == "8"
    assert image.headers["X-ImageInt-Model"] == "Qwen/Qwen-Image-2.1"
    assert image.content.startswith(b"\x89PNG")


def test_the_job_list_contains_the_job(client):
    job_id = client.post(GENERATE, json={"prompt": "Ein Berg"}).json()["job_id"]
    body = client.get("/v1/jobs").json()
    assert body["count"] == 1
    assert body["jobs"][0]["job_id"] == job_id
    assert body["jobs"][0]["prompt"] == "Ein Berg"
    assert body["stats"]["running"] + body["stats"]["done"] >= 1


def test_generation_is_503_while_the_image_model_loads(client, fake):
    fake.health_status[IMAGE_HOST] = 503
    response = client.post(GENERATE, json={"prompt": "Ein Berg"})
    assert response.status_code == 503
    assert response.json()["error"] == "service_loading"
    assert fake.image_requests == []


def test_a_loading_image_model_is_asked_to_start_loading(client, fake):
    # The image container can be configured to load lazily, in which case its
    # /health stays 503 until a render arrives. A gateway that refuses to send
    # one would deadlock with it, so the readiness probe nudges the model
    # server instead. The client still gets the unchanged "not yet" answer.
    fake.health_status[IMAGE_HOST] = 503
    response = client.post(GENERATE, json={"prompt": "Ein Berg", "enhance": False})
    assert response.status_code == 503
    assert response.json()["error"] == "service_loading"
    assert fake.warmups == [IMAGE_HOST]
    assert fake.image_requests == []


def test_a_loading_enhancer_is_asked_to_start_loading(client, fake):
    # The enhancer container can be configured to load lazily too
    # (IMAGEINT_PE_PRELOAD=false), so it gets the same nudge as the image
    # server. The client still gets the unchanged "not yet" answer.
    fake.health_status[ENHANCER_HOST] = 503
    response = client.post(GENERATE, json={"prompt": "Ein Berg"})
    assert response.status_code == 503
    assert response.json()["component"] == "enhancer"
    assert fake.warmups == [ENHANCER_HOST]


def test_a_ready_image_model_is_not_asked_to_warm_up(client, fake):
    assert client.post(GENERATE, json={"prompt": "Ein Berg"}).status_code == 200
    assert fake.warmups == []


def test_a_missing_warmup_route_is_harmless(client, fake):
    # Pointed at something that is not the image server, the nudge answers 404.
    # That must not turn into a visible error: readiness is unchanged.
    fake.health_status[IMAGE_HOST] = 503
    fake.warmup_status = 404
    response = client.post(GENERATE, json={"prompt": "Ein Berg", "enhance": False})
    assert response.status_code == 503
    assert response.json()["error"] == "service_loading"


def test_an_unreachable_image_model_is_not_asked_to_warm_up(client, fake):
    # Nothing is listening, so there is nobody to nudge.
    fake.offline = True
    response = client.post(GENERATE, json={"prompt": "Ein Berg", "enhance": False})
    assert response.status_code == 502
    assert fake.warmups == []


def test_an_upstream_failure_is_reported_with_its_status(client, fake):
    fake.image_status = 500
    response = client.post(GENERATE, json={"prompt": "Ein Berg"})
    assert response.status_code == 502
    body = response.json()
    assert body["ok"] is False
    assert body["error"] == "image_error"


def test_an_unreachable_model_is_reported_as_unavailable(client, fake):
    fake.offline = True
    response = client.post(GENERATE, json={"prompt": "Ein Berg", "enhance": False})
    assert response.status_code == 502
    assert response.json()["error"] == "image_unavailable"


def test_an_unreachable_enhancer_is_reported_as_unavailable(client, fake):
    fake.offline = True
    response = client.post(GENERATE, json={"prompt": "Ein Berg"})
    assert response.status_code == 502
    assert response.json()["error"] == "enhancer_unavailable"


def test_a_full_queue_is_refused_with_busy(client_factory):
    with client_factory(
        IMAGEINT_MAX_CONCURRENT_JOBS=1,
        IMAGEINT_MAX_QUEUED_JOBS=0,
    ) as client:
        block_the_runner(client)
        first = client.post(GENERATE, json={"prompt": "Ein Berg", "wait": False})
        assert first.status_code == 202

        second = client.post(GENERATE, json={"prompt": "Noch ein Berg", "wait": False})
        assert second.status_code == 503
        body = second.json()
        assert body["error"] == "busy"
        assert second.headers["Retry-After"] == "30"


def test_the_image_of_a_running_job_is_409(client_factory):
    with client_factory() as client:
        block_the_runner(client)
        job_id = client.post(GENERATE, json={"prompt": "Ein Berg", "wait": False}).json()[
            "job_id"
        ]
        response = client.get(f"/v1/jobs/{job_id}/image")
    assert response.status_code == 409
    assert response.json()["error"] == "job_not_finished"


def test_an_image_that_was_cleaned_up_is_not_found(client):
    job_id = client.post(GENERATE, json={"prompt": "Ein Berg"}).json()["job_id"]
    assert client.get(f"/v1/images/{job_id}").status_code == 200

    main_module.runtime().registry.store.delete(job_id)
    response = client.get(f"/v1/images/{job_id}")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_a_stored_image_is_a_real_png(client, fake):
    fake.image_size = (16, 16)
    job_id = client.post(GENERATE, json={"prompt": "Ein Berg"}).json()["job_id"]
    stored = main_module.runtime().registry.store.read(job_id)
    assert stored == png_bytes((16, 16))
