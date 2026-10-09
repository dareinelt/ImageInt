"""Tests for the documented Qwen-Image-2.1 prompt enhancer.

The delicate parts are the answer parser (the enhancer writes prose *and* a JSON
object, and a rewritten prompt may itself contain braces) and the split of the
thinking block from the answer. Both are tested against the shapes the
checkpoint actually produces.
"""

import json

import pytest

from app import enhancer
from app.config import EnhancerSettings, get_settings
from app.errors import ImageIntError


def settings(**overrides) -> EnhancerSettings:
    base = {
        "url": "http://enhancer:8000",
        "model": "Qwen-Image-2.1-PE-T2I",
        "token": "",
    }
    base.update(overrides)
    return EnhancerSettings(**base)


# -- the vendored system prompt -------------------------------------------- #


def test_the_documented_system_prompt_is_vendored_verbatim():
    prompt = get_settings().enhancer.system_prompt()
    assert len(prompt) > 9000
    assert "rewritten_prompt" in prompt
    assert "wh_ratio" in prompt


def test_the_system_prompt_is_sent_as_a_text_part():
    messages = enhancer.build_messages("SYS", "ein rotes Haus")
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == [{"type": "text", "text": "SYS"}]
    assert messages[1]["content"] == [{"type": "text", "text": "ein rotes Haus"}]


def test_the_payload_uses_the_documented_t2i_sampling_profile():
    payload = enhancer.build_payload(settings(), "ein rotes Haus")
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    # 1.5 belongs to the text-to-image task; the edit task uses 0.0 and the two
    # are not interchangeable.
    assert payload["presence_penalty"] == 1.5
    assert payload["max_tokens"] == 16256
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["messages"][1]["content"][0]["text"] == "ein rotes Haus"


# -- splitting the thinking block ------------------------------------------ #


def test_split_thinking_recovers_an_inline_block():
    thinking, answer = enhancer.split_thinking("ich überlege\n</think>\n\nhier die Antwort")
    assert thinking == "ich überlege"
    assert answer == "hier die Antwort"


def test_split_thinking_handles_a_complete_block():
    thinking, answer = enhancer.split_thinking("<think>kurz</think>Antwort")
    assert thinking == "kurz"
    assert answer == "Antwort"


def test_split_thinking_flags_an_unterminated_block():
    thinking, answer = enhancer.split_thinking("<think>noch am denken")
    assert thinking == "noch am denken"
    assert answer == ""


def test_split_thinking_passes_plain_text_through():
    thinking, answer = enhancer.split_thinking("  nur eine Antwort  ")
    assert thinking == ""
    assert answer == "nur eine Antwort"


# -- the answer parser ------------------------------------------------------ #


def test_parse_answer_reads_the_declared_contract():
    parsed = enhancer.parse_answer(
        'Fertig.\n{"rewritten_prompt": "ein rotes Haus", "wh_ratio": "3:2"}'
    )
    assert parsed["parse_ok"] is True
    assert parsed["positive_prompt"] == "ein rotes Haus"
    assert parsed["wh_ratio"] == "3:2"


def test_parse_answer_prefers_the_last_object():
    parsed = enhancer.parse_answer(
        '{"rewritten_prompt": "alt", "wh_ratio": "1:1"}\n'
        'korrigiert:\n{"rewritten_prompt": "neu", "wh_ratio": "16:9"}'
    )
    assert parsed["positive_prompt"] == "neu"
    assert parsed["wh_ratio"] == "16:9"


def test_parse_answer_survives_braces_in_the_prose_after_the_object():
    parsed = enhancer.parse_answer(
        '{"rewritten_prompt": "ein Haus", "wh_ratio": "3:2"}\n'
        "Hinweis: {dieser Text ist kein JSON}"
    )
    assert parsed["parse_ok"] is True
    assert parsed["positive_prompt"] == "ein Haus"


def test_parse_answer_survives_braces_inside_the_prompt():
    parsed = enhancer.parse_answer(
        '{"rewritten_prompt": "ein Schild mit {ACHTUNG} darauf", "wh_ratio": "1:1"}'
    )
    assert parsed["parse_ok"] is True
    assert parsed["positive_prompt"] == "ein Schild mit {ACHTUNG} darauf"


def test_parse_answer_accepts_the_mis_typed_key():
    parsed = enhancer.parse_answer('{"rewrited_prompt": "ein Haus", "wh_ratio": "3:2"}')
    assert parsed["parse_ok"] is True
    assert parsed["positive_prompt"] == "ein Haus"


def test_parse_answer_repairs_nearly_valid_json():
    parsed = enhancer.parse_answer('{"rewritten_prompt": "ein Haus", "wh_ratio": "3:2",}')
    assert parsed["parse_ok"] is True
    assert parsed["positive_prompt"] == "ein Haus"


def test_parse_answer_reads_the_negative_prompt_when_present():
    parsed = enhancer.parse_answer(
        '{"rewritten_prompt": "ein Haus", "wh_ratio": "3:2", "negative_prompt": "unscharf"}'
    )
    assert parsed["negative_prompt"] == "unscharf"


def test_parse_answer_falls_back_to_the_raw_answer():
    parsed = enhancer.parse_answer("Ein rotes Haus auf einer grünen Wiese.")
    assert parsed["parse_ok"] is False
    assert parsed["positive_prompt"] == "Ein rotes Haus auf einer grünen Wiese."
    assert parsed["wh_ratio"] == ""


def test_parse_answer_ignores_objects_without_a_prompt():
    parsed = enhancer.parse_answer('{"wh_ratio": "3:2"}')
    assert parsed["parse_ok"] is False
    assert parsed["positive_prompt"] == '{"wh_ratio": "3:2"}'


def test_parse_answer_ignores_an_empty_prompt():
    parsed = enhancer.parse_answer('{"rewritten_prompt": "   ", "wh_ratio": "3:2"}')
    assert parsed["parse_ok"] is False


def test_parse_answer_tolerates_an_empty_answer():
    parsed = enhancer.parse_answer("")
    assert parsed["parse_ok"] is False
    assert parsed["positive_prompt"] == ""


# -- the full call ---------------------------------------------------------- #


async def test_enhance_uses_the_answer_and_the_ratio(fake):
    result = await enhancer.enhance(settings(), "ein rotes Haus")
    assert result["parse_ok"] is True
    assert result["prompt"].startswith("A red panda")
    assert result["wh_ratio"] == "3:2"
    assert (result["width"], result["height"]) == (2528, 1696)
    assert result["model"] == "Qwen-Image-2.1-PE-T2I"
    assert result["usage"]["total"] == 768
    assert "red panda" in result["thinking"]


async def test_enhance_sends_the_documented_system_prompt(fake):
    await enhancer.enhance(settings(), "ein rotes Haus")
    body = fake.enhancer_requests[-1]
    assert body["stream"] is True
    assert body["presence_penalty"] == 1.5
    sent = body["messages"][0]["content"][0]["text"]
    assert sent == get_settings().enhancer.system_prompt()


async def test_enhance_recovers_inline_thinking(fake):
    fake.enhancer_reasoning = ""
    fake.enhancer_raw = (
        "Der Nutzer möchte ein Haus.</think>\n"
        + json.dumps({"rewritten_prompt": "ein rotes Haus", "wh_ratio": "3:2"})
    )
    result = await enhancer.enhance(settings(), "ein rotes Haus")
    assert result["parse_ok"] is True
    assert result["thinking"] == "Der Nutzer möchte ein Haus."
    assert result["prompt"] == "ein rotes Haus"


async def test_enhance_keeps_the_generation_when_the_answer_is_unparseable(fake):
    fake.enhancer_raw = "Ein rotes Haus auf einer grünen Wiese."
    result = await enhancer.enhance(settings(), "ein rotes Haus")
    assert result["parse_ok"] is False
    assert result["prompt"] == "Ein rotes Haus auf einer grünen Wiese."
    # No ratio in the answer means the native square.
    assert (result["width"], result["height"]) == (2048, 2048)


async def test_enhance_clamps_to_the_pixel_budget(fake):
    result = await enhancer.enhance(settings(), "ein rotes Haus", max_pixels=1024 * 1024)
    assert result["width"] * result["height"] <= 1024 * 1024


async def test_enhance_rejects_an_empty_prompt():
    with pytest.raises(ImageIntError) as excinfo:
        await enhancer.enhance(settings(), "   ")
    assert excinfo.value.code == "empty_prompt"


async def test_enhance_is_off_when_the_enhancer_is_disabled():
    with pytest.raises(ImageIntError) as excinfo:
        await enhancer.enhance(settings(enabled=False), "ein rotes Haus")
    assert excinfo.value.code == "not_configured"


async def test_enhance_reports_a_missing_url():
    with pytest.raises(ImageIntError) as excinfo:
        await enhancer.enhance(settings(url=""), "ein rotes Haus")
    assert excinfo.value.code == "not_configured"


async def test_enhance_reports_a_timeout(fake):
    fake.timeout_paths.add("/v1/chat/completions")
    with pytest.raises(ImageIntError) as excinfo:
        await enhancer.enhance(settings(), "ein rotes Haus")
    assert excinfo.value.code == "enhancer_timeout"
    assert excinfo.value.status_code == 504


async def test_enhance_reports_a_loading_model(fake):
    fake.health_status["enhancer"] = 503
    with pytest.raises(ImageIntError) as excinfo:
        await enhancer.enhance(settings(), "ein rotes Haus")
    assert excinfo.value.code == "enhancer_error"
    assert excinfo.value.status_code == 502


async def test_enhance_passes_umlauts_through(fake):
    await enhancer.enhance(settings(), "ein grünes Blatt auf weißem Schnee")
    sent = fake.enhancer_requests[-1]
    assert sent["messages"][1]["content"][0]["text"] == "ein grünes Blatt auf weißem Schnee"


# -- reporting -------------------------------------------------------------- #


def test_describe_ratio_flags_documented_ratios():
    assert enhancer.describe_ratio("3:2") == {
        "wh_ratio": "3:2",
        "width": 2528,
        "height": 1696,
        "documented": True,
    }
    assert enhancer.describe_ratio("21:9")["documented"] is False
    assert enhancer.describe_ratio("quatsch")["wh_ratio"] == "1:1"


def test_profile_reports_the_effective_sampling():
    reported = enhancer.profile()
    assert reported["model"] == "Qwen/Qwen-Image-2.1-PE-T2I"
    assert reported["sampling"]["presence_penalty"] == 1.5
    assert reported["system_prompt_chars"] > 9000
    assert "3:2" in reported["ratios"]
