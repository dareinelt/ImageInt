"""Configuration of the image server.

The defaults are part of the contract: bf16 on CPU through diffusers, no
quantisation, and sampling *without* guidance. Anything that would silently
change a rendered picture has to fail loudly instead, which is what most of
these tests pin down.
"""

from __future__ import annotations

import os

import pytest

from image_server import config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test from an environment without any ImageInt variable."""

    for key in list(os.environ):
        if key.startswith("IMAGEINT_IMAGE_"):
            monkeypatch.delenv(key, raising=False)


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
def test_documented_defaults():
    settings = config.load_settings()
    assert settings.model == "Qwen/Qwen-Image-2.1"
    assert settings.engine == "diffusers"
    assert settings.device == "cpu"
    # bf16 replaces the earlier Q4_K_M requirement: Q4 was a vLLM-omni/GGUF
    # feature, and vLLM-omni cannot serve this model on a CPU.
    assert settings.dtype == "bfloat16"
    assert settings.quant == "none"
    assert settings.steps == 40
    # The model card states Qwen-Image-2.1 is meant to be sampled without
    # guidance, so the default is 1.0 and not 4.0.
    assert settings.true_cfg_scale == 1.0
    assert settings.use_kv_cache is True
    assert settings.preload is True
    assert settings.port == 8000
    assert settings.auth_required is False


def test_documented_sizes_are_on_the_patch_grid_and_within_budget():
    for width, height in config.DOCUMENTED_SIZES:
        assert width % config.MULTIPLE == 0
        assert height % config.MULTIPLE == 0
        assert width * height <= config.DEFAULT_MAX_PIXELS


# --------------------------------------------------------------------------- #
# Environment overrides
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,value,attribute,expected",
    [
        ("IMAGEINT_IMAGE_MODEL", "Qwen/Qwen-Image-2.1-PE-T2I", "model", "Qwen/Qwen-Image-2.1-PE-T2I"),
        ("IMAGEINT_IMAGE_DTYPE", "float32", "dtype", "float32"),
        ("IMAGEINT_IMAGE_DEVICE", "cuda", "device", "cuda"),
        ("IMAGEINT_IMAGE_QUANT", "int4", "quant", "int4"),
        ("IMAGEINT_IMAGE_ENGINE", "stub", "engine", "stub"),
        ("IMAGEINT_IMAGE_TOKEN", "s3cret", "token", "s3cret"),
        ("IMAGEINT_IMAGE_NEGATIVE_PROMPT", "blurry", "negative_prompt", "blurry"),
        ("IMAGEINT_IMAGE_CACHE_DIR", "/models/hf", "cache_dir", "/models/hf"),
    ],
)
def test_text_overrides(monkeypatch, name, value, attribute, expected):
    monkeypatch.setenv(name, value)
    assert getattr(config.load_settings(), attribute) == expected


def test_numeric_overrides(monkeypatch):
    monkeypatch.setenv("IMAGEINT_IMAGE_STEPS", "8")
    monkeypatch.setenv("IMAGEINT_IMAGE_TRUE_CFG_SCALE", "4.5")
    monkeypatch.setenv("IMAGEINT_IMAGE_CPU_THREADS", "8")
    monkeypatch.setenv("IMAGEINT_IMAGE_MAX_PIXELS", "1048576")
    settings = config.load_settings()
    assert settings.steps == 8
    assert settings.true_cfg_scale == 4.5
    assert settings.cpu_threads == 8
    assert settings.max_pixels == 1048576


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", "ja"])
def test_truthy_flags(monkeypatch, raw):
    monkeypatch.setenv("IMAGEINT_IMAGE_KV_CACHE", raw)
    assert config.load_settings().use_kv_cache is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", "nein"])
def test_falsy_flags(monkeypatch, raw):
    monkeypatch.setenv("IMAGEINT_IMAGE_KV_CACHE", raw)
    assert config.load_settings().use_kv_cache is False


def test_preload_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("IMAGEINT_IMAGE_PRELOAD", "false")
    assert config.load_settings().preload is False


# --------------------------------------------------------------------------- #
# Invalid input never changes the rendering silently
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,value,attribute,fallback",
    [
        # An unsupported dtype must not become float16 by accident.
        ("IMAGEINT_IMAGE_DTYPE", "int4", "dtype", "bfloat16"),
        ("IMAGEINT_IMAGE_DEVICE", "tpu", "device", "cpu"),
        ("IMAGEINT_IMAGE_QUANT", "Q4_K_M", "quant", "none"),
        ("IMAGEINT_IMAGE_ENGINE", "vllm-omni", "engine", "diffusers"),
        ("IMAGEINT_IMAGE_STEPS", "viele", "steps", 40),
        ("IMAGEINT_IMAGE_PORT", "achttausend", "port", 8000),
    ],
)
def test_invalid_values_fall_back_to_the_default(monkeypatch, name, value, attribute, fallback):
    monkeypatch.setenv(name, value)
    assert getattr(config.load_settings(), attribute) == fallback


def test_numbers_are_clamped_to_their_range(monkeypatch):
    monkeypatch.setenv("IMAGEINT_IMAGE_STEPS", "0")
    monkeypatch.setenv("IMAGEINT_IMAGE_PORT", "70000")
    monkeypatch.setenv("IMAGEINT_IMAGE_TRUE_CFG_SCALE", "99")
    settings = config.load_settings()
    assert settings.steps == 1
    assert settings.port == 65535
    assert settings.true_cfg_scale == 20.0


# --------------------------------------------------------------------------- #
# snap()
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value,expected",
    [
        # Already on the grid: unchanged.
        (2048, 2048),
        (2400, 2400),
        (1792, 1792),
        (2752, 2752),
        (1536, 1536),
        # Off the grid: rounded to the nearest multiple of 32.
        (2050, 2048),
        (1000, 992),
        (1990, 1984),
        # Below or above the accepted range: clamped.
        (1, config.MIN_SIDE),
        (9000, config.MAX_SIDE),
    ],
)
def test_snap(value, expected):
    assert config.snap(value) == expected


def test_snap_always_returns_a_multiple_of_the_grid():
    for value in range(1, 5000, 37):
        assert config.snap(value) % config.MULTIPLE == 0
