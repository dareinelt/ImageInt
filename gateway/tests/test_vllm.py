"""Tests for the vLLM client: health probe, SSE parsing and image extraction."""

import base64
import json

import pytest

from app import vllm
from app.config import ImageSettings
from app.errors import ImageIntError
from conftest import IMAGE_URL, png_bytes


def image_settings(**overrides) -> ImageSettings:
    base = {"url": IMAGE_URL, "model": "Qwen/Qwen-Image-2.1", "token": ""}
    base.update(overrides)
    return ImageSettings(**base)


# -- health probe ----------------------------------------------------------- #


async def test_health_reports_a_ready_server(fake):
    result = await vllm.health(IMAGE_URL, "", "Qwen/Qwen-Image-2.1")
    assert result["ok"] is True
    assert result["loading"] is False
    assert result["http"] == 200


async def test_health_reports_a_loading_server(fake):
    fake.health_status["image"] = 503
    result = await vllm.health(IMAGE_URL, "", "Qwen/Qwen-Image-2.1")
    assert result["ok"] is False
    assert result["loading"] is True


async def test_health_reports_an_unreachable_server(fake):
    fake.offline = True
    result = await vllm.health("http://image:8000", "", "m")
    assert result["ok"] is False
    assert result["loading"] is False
    assert "nicht erreichbar" in result["message"]


async def test_health_reports_a_timeout(fake):
    fake.timeout_paths.add("/health")
    result = await vllm.health("http://image:8000", "", "m")
    assert result["ok"] is False
    assert result["http"] == 0


async def test_health_reports_an_unconfigured_server():
    result = await vllm.health("", "", "m")
    assert result["configured"] is False
    assert result["ok"] is False


async def test_health_reports_the_reason_a_server_failed(fake):
    """A model server that could not load its weights explains why in the body;
    that reason must not be dropped in favour of the bare status code."""

    fake.health_status["image"] = 500
    fake.health_bodies["image"] = {
        "ok": False,
        "state": "error",
        "message": "ValueError: `quantization_config` must be an instance of "
        "`PipelineQuantizationConfig`.",
    }

    result = await vllm.health(IMAGE_URL, "", "Qwen/Qwen-Image-2.1")

    assert result["ok"] is False
    assert result["loading"] is False
    assert result["http"] == 500
    assert "HTTP 500" in result["message"]
    assert "PipelineQuantizationConfig" in result["message"]


async def test_health_reports_a_failed_server_without_a_body(fake):
    fake.health_status["image"] = 500
    result = await vllm.health(IMAGE_URL, "", "Qwen/Qwen-Image-2.1")
    assert result["message"] == "Modellserver meldet HTTP 500."


# -- download progress ------------------------------------------------------ #

PROGRESS = {
    "known": True,
    "percent": 42.5,
    "bytes_downloaded": 123456789,
    "bytes_total": 296352743424,
    "files_downloaded": 3,
    "files_total": 12,
    "current_file": "diffusion_pytorch_model-00002-of-00004.safetensors",
    "eta_seconds": 120,
    "bytes_per_second": 1048576.5,
}


async def test_health_forwards_the_download_progress(fake):
    fake.health_status["image"] = 503
    fake.health_bodies["image"] = {"ok": False, "state": "loading", "progress": PROGRESS}

    result = await vllm.health(IMAGE_URL, "", "Qwen/Qwen-Image-2.1")

    assert result["loading"] is True
    assert result["progress"]["percent"] == 42.5
    assert result["progress"]["bytes_total"] == 296352743424
    assert result["progress"]["current_file"] == PROGRESS["current_file"]
    assert result["progress"]["eta_seconds"] == 120
    assert result["progress"]["bytes_per_second"] == 1048576.5


async def test_health_forwards_the_progress_of_a_ready_server(fake):
    fake.health_bodies["image"] = {"state": "ready", "progress": {"known": True, "percent": 100}}

    result = await vllm.health(IMAGE_URL, "", "Qwen/Qwen-Image-2.1")

    assert result["ok"] is True
    assert result["progress"] == {"known": True, "percent": 100.0}


async def test_health_omits_the_progress_when_the_server_reports_none(fake):
    fake.health_bodies["image"] = {"state": "ready"}
    result = await vllm.health(IMAGE_URL, "", "Qwen/Qwen-Image-2.1")
    assert "progress" not in result


async def test_health_omits_the_progress_of_a_bodyless_server(fake):
    result = await vllm.health(IMAGE_URL, "", "Qwen/Qwen-Image-2.1")
    assert "progress" not in result


async def test_health_ignores_a_malformed_progress(fake):
    fake.health_bodies["image"] = {"progress": {"known": True, "percent": "viel"}}
    result = await vllm.health(IMAGE_URL, "", "Qwen/Qwen-Image-2.1")
    assert "progress" not in result


@pytest.mark.parametrize("percent", [-5, 0, 100, 137.5])
def test_the_progress_percentage_is_clamped(percent):
    cleaned = vllm._clean_progress({"known": True, "percent": percent})
    assert 0.0 <= cleaned["percent"] <= 100.0


def test_the_progress_rate_is_kept_as_a_float():
    cleaned = vllm._clean_progress(
        {"known": False, "bytes_downloaded": 10, "bytes_per_second": 1048576.5}
    )
    assert cleaned["bytes_per_second"] == 1048576.5

    clamped = vllm._clean_progress(
        {"known": False, "bytes_downloaded": 10, "bytes_per_second": -5}
    )
    assert clamped["bytes_per_second"] == 0.0

    ignored = vllm._clean_progress(
        {"known": False, "bytes_downloaded": 10, "bytes_per_second": "5"}
    )
    assert "bytes_per_second" not in ignored


# -- the image request ------------------------------------------------------ #


def test_the_images_payload_uses_size_and_the_documented_sampling():
    payload = vllm.build_image_payload(image_settings(), "ein Haus", 2528, 1696)
    assert payload["size"] == "2528x1696"
    assert payload["n"] == 1
    assert payload["num_inference_steps"] == 40
    # Qwen-Image-2.1 is sampled without guidance, so the default is 1.0 and
    # a negative prompt would have no effect.
    assert payload["true_cfg_scale"] == 1.0
    assert "negative_prompt" not in payload
    assert "seed" not in payload


def test_the_images_payload_carries_the_optional_fields():
    payload = vllm.build_image_payload(
        image_settings(), "ein Haus", 1024, 1024, negative_prompt="unscharf", seed=7
    )
    assert payload["negative_prompt"] == "unscharf"
    assert payload["seed"] == 7


def test_the_configured_negative_prompt_is_the_default():
    settings = image_settings(negative_prompt="verwaschen")
    assert vllm.build_image_payload(settings, "ein Haus", 1024, 1024)["negative_prompt"] == "verwaschen"
    # An explicit one wins over the configured one.
    payload = vllm.build_image_payload(settings, "ein Haus", 1024, 1024, negative_prompt="grell")
    assert payload["negative_prompt"] == "grell"


def test_the_chat_payload_puts_the_image_parameters_in_extra_body():
    payload = vllm.build_image_payload(
        image_settings(route="chat"), "ein Haus", 2528, 1696, negative_prompt="unscharf", seed=3
    )
    assert payload["messages"] == [{"role": "user", "content": "ein Haus"}]
    assert "prompt" not in payload and "size" not in payload
    assert payload["extra_body"] == {
        "width": 2528,
        "height": 1696,
        "num_inference_steps": 40,
        "true_cfg_scale": 1.0,
        "negative_prompt": "unscharf",
        "seed": 3,
    }


async def test_generate_image_decodes_the_images_answer(fake):
    result = await vllm.generate_image(
        image_settings(), vllm.build_image_payload(image_settings(), "ein Haus", 8, 8)
    )
    assert result["ok"] is True
    assert result["image"].startswith(b"\x89PNG")
    assert result["content_type"] == "image/png"
    assert (result["width"], result["height"]) == (8, 8)
    assert result["seed"] == 42
    assert fake.calls[-1] == ("POST", "/v1/images/generations")


async def test_generate_image_decodes_the_chat_answer(fake):
    fake.image_shape = "chat"
    settings = image_settings(route="chat")
    result = await vllm.generate_image(
        settings, vllm.build_image_payload(settings, "ein Haus", 8, 8)
    )
    assert result["image"].startswith(b"\x89PNG")
    assert fake.calls[-1] == ("POST", "/v1/chat/completions")


async def test_generate_image_reads_a_data_uri_in_the_images_answer(fake):
    encoded = base64.b64encode(png_bytes((8, 8))).decode("ascii")
    fake.image_body = lambda: {
        "data": [{"url": f"data:image/webp;base64,{encoded}", "width": 8, "height": 8}]
    }
    result = await vllm.generate_image(
        image_settings(), vllm.build_image_payload(image_settings(), "ein Haus", 8, 8)
    )
    assert result["image"].startswith(b"\x89PNG")
    assert result["content_type"] == "image/webp"


async def test_generate_image_sends_the_token(fake):
    settings = image_settings(token="geheim")
    await vllm.generate_image(
        settings, vllm.build_image_payload(settings, "ein Haus", 8, 8)
    )
    headers = fake.image_headers[-1]
    assert headers["x-auth-token"] == "geheim"
    assert headers["authorization"] == "Bearer geheim"


async def test_generate_image_reports_an_upstream_error(fake):
    fake.image_status = 500
    with pytest.raises(ImageIntError) as excinfo:
        await vllm.generate_image(
            image_settings(), vllm.build_image_payload(image_settings(), "ein Haus", 8, 8)
        )
    assert excinfo.value.code == "image_error"
    assert "Modell nicht geladen" in excinfo.value.message


async def test_generate_image_reports_an_empty_answer(fake):
    fake.image_body = lambda: {"data": []}
    with pytest.raises(ImageIntError) as excinfo:
        await vllm.generate_image(
            image_settings(), vllm.build_image_payload(image_settings(), "ein Haus", 8, 8)
        )
    assert excinfo.value.code == "image_empty"


async def test_generate_image_reports_a_timeout(fake):
    fake.timeout_paths.add("/v1/images/generations")
    with pytest.raises(ImageIntError) as excinfo:
        await vllm.generate_image(
            image_settings(), vllm.build_image_payload(image_settings(), "ein Haus", 8, 8)
        )
    assert excinfo.value.code == "image_timeout"
    assert excinfo.value.status_code == 504


async def test_generate_image_reports_a_missing_url():
    settings = image_settings(url="")
    with pytest.raises(ImageIntError) as excinfo:
        await vllm.generate_image(settings, {"prompt": "ein Haus"})
    assert excinfo.value.code == "not_configured"


# -- the small parsers ------------------------------------------------------ #


@pytest.mark.parametrize(
    "line,expected",
    [
        ('data: {"a": 1}', '{"a": 1}'),
        ("data:[DONE]", "[DONE]"),
        ("", None),
        (": keep-alive", None),
        ("event: message", None),
        ('{"a": 1}', '{"a": 1}'),
    ],
)
def test_sse_data_reads_both_framings(line, expected):
    assert vllm._sse_data(line) == expected


def test_error_text_reads_the_openai_envelope():
    body = json.dumps({"error": {"message": "Modell nicht geladen"}}).encode()
    assert vllm._error_text(body) == "Modell nicht geladen"
    assert vllm._error_text(b"kein json") == ""
