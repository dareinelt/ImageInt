"""Configuration of the prompt-enhancer server.

The defaults are part of the contract: bf16 on CPU through transformers, no
quantisation, and the t2i sampling profile of the checkpoint verbatim. Anything
that would silently change the sampled prompt has to fail loudly instead, which
is what most of these tests pin down.
"""

from __future__ import annotations

import os

import pytest

from enhancer_server import config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from an environment without any ImageInt variable."""

    for key in list(os.environ):
        if key.startswith("IMAGEINT_PE_"):
            monkeypatch.delenv(key, raising=False)


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
def test_documented_defaults():
    settings = config.load_settings()
    assert settings.model == "Qwen/Qwen-Image-2.1-PE-T2I"
    # transformers, not vLLM: vLLM's CPU backend cannot initialise this
    # checkpoint, which is why the enhancer is a transformers server.
    assert settings.engine == "transformers"
    assert settings.device == "cpu"
    # bf16 replaces the earlier Q4 requirement: Q4 was a vLLM/GGUF feature and
    # is not available on this stack.
    assert settings.dtype == "bfloat16"
    assert settings.quant == "none"
    assert settings.preload is True
    assert settings.trust_remote_code is True
    assert settings.enable_thinking is True
    assert settings.port == 8000
    assert settings.auth_required is False


def test_the_sampling_profile_is_the_one_the_model_card_documents():
    settings = config.load_settings()
    assert settings.temperature == 1.0
    assert settings.top_p == 0.95
    assert settings.top_k == 20
    assert settings.min_p == 0.0
    # The presence penalty is the number that is easy to get wrong: it does not
    # fail when it is off, it changes what is sampled.
    assert settings.presence_penalty == 1.5
    assert settings.max_new_tokens == 16256


def test_the_answer_budget_fits_into_the_context():
    settings = config.load_settings()
    assert settings.max_new_tokens < settings.context


def test_the_default_prompt_limit_leaves_room_for_the_system_prompt():
    """The gateway sends a ~10 kB system prompt plus the user's request.

    The limit applies to the user's prompt alone; if it were applied to the
    whole conversation the documented default would reject every real call.
    """

    settings = config.load_settings()
    assert settings.max_prompt_chars == 4000


def test_engine_choices_are_exactly_transformers_and_stub():
    assert config.ENGINES == ("transformers", "stub")


def test_quantisation_choices_are_the_quanto_ones():
    assert config.QUANTS == ("none", "int8", "int4")


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def test_model_and_token_come_from_the_same_variables_as_on_the_gateway(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PE_MODEL", "Qwen/Qwen-Image-2.1-PE-Edit")
    monkeypatch.setenv("IMAGEINT_PE_TOKEN", "geheim")
    settings = config.load_settings()
    assert settings.model == "Qwen/Qwen-Image-2.1-PE-Edit"
    assert settings.token == "geheim"
    assert settings.auth_required is True


def test_an_unknown_engine_falls_back_to_transformers(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PE_ENGINE", "vllm")
    assert config.load_settings().engine == "transformers"


def test_an_unknown_dtype_falls_back_to_bfloat16(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PE_DTYPE", "q4_k_m")
    assert config.load_settings().dtype == "bfloat16"


def test_quantisation_can_be_switched_to_int8(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PE_QUANT", "int8")
    assert config.load_settings().quant == "int8"


def test_a_cuda_device_is_selectable(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PE_DEVICE", "cuda")
    assert config.load_settings().device == "cuda"


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "nein", "FALSE"])
def test_false_like_flags_turn_preload_off(monkeypatch, raw):
    monkeypatch.setenv("IMAGEINT_PE_PRELOAD", raw)
    assert config.load_settings().preload is False


def test_preload_defaults_to_on_for_anything_else(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PE_PRELOAD", "vielleicht")
    assert config.load_settings().preload is True


def test_thinking_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PE_ENABLE_THINKING", "false")
    assert config.load_settings().enable_thinking is False


def test_numeric_values_are_clamped_to_their_documented_range(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PE_TEMPERATURE", "99")
    monkeypatch.setenv("IMAGEINT_PE_TOP_P", "-1")
    monkeypatch.setenv("IMAGEINT_PE_PRESENCE_PENALTY", "10")
    monkeypatch.setenv("IMAGEINT_PE_MAX_NEW_TOKENS", "1")
    settings = config.load_settings()
    assert settings.temperature == 2.0
    assert settings.top_p == 0.0
    assert settings.presence_penalty == 2.0
    assert settings.max_new_tokens == 256


def test_a_garbage_number_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PE_MAX_NEW_TOKENS", "viele")
    monkeypatch.setenv("IMAGEINT_PE_PORT", "achttausend")
    settings = config.load_settings()
    assert settings.max_new_tokens == config.DEFAULT_MAX_NEW_TOKENS
    assert settings.port == 8000


def test_an_out_of_range_port_is_clamped(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PE_PORT", "70000")
    assert config.load_settings().port == 65535


def test_a_revision_can_be_pinned(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PE_REVISION", "refs/pr/7")
    assert config.load_settings().revision == "refs/pr/7"


def test_an_empty_value_is_not_taken_as_a_setting(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PE_MODEL", "   ")
    assert config.load_settings().model == config.DEFAULT_MODEL


def test_the_torch_dtype_property_follows_the_dtype(monkeypatch):
    monkeypatch.setenv("IMAGEINT_PE_DTYPE", "float32")
    assert config.load_settings().torch_dtype == "float32"


def test_settings_are_frozen():
    settings = config.load_settings()
    with pytest.raises(Exception):
        settings.model = "etwas anderes"  # type: ignore[misc]
