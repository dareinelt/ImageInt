"""Engine behaviour: the loading state machine, the stub renderer and the
diffusers version guard.

None of these tests needs torch or the 33 GB checkpoint. What they pin down is
the part the gateway depends on -- that ``status()`` distinguishes "still
loading" from "broken", and that a missing or too old diffusers fails with a
message a human can act on.
"""

from __future__ import annotations

import io
import sys
import threading
import time
import types

import pytest
from PIL import Image

from image_server import engine as engine_module
from image_server.config import Settings, load_settings
from image_server.engine import (
    BaseEngine,
    DiffusersEngine,
    EngineError,
    RenderRequest,
    StubEngine,
    build_engine,
)


def stub_settings(**overrides) -> Settings:
    base = dict(engine="stub", preload=False, stub_delay=0.0)
    base.update(overrides)
    return Settings(**base)


def request(**overrides) -> RenderRequest:
    base = dict(
        prompt="Eine Katze im Regen",
        width=1024,
        height=1024,
        steps=40,
        true_cfg_scale=1.0,
    )
    base.update(overrides)
    return RenderRequest(**base)


class FailingEngine(BaseEngine):
    """An engine whose weights never arrive, to exercise the error state."""

    name = "failing"

    def _load(self) -> None:
        raise EngineError("Gewichte konnten nicht geladen werden.")


# --------------------------------------------------------------------------- #
# build_engine
# --------------------------------------------------------------------------- #
def test_build_engine_selects_the_stub():
    assert isinstance(build_engine(stub_settings()), StubEngine)


def test_build_engine_defaults_to_diffusers():
    assert isinstance(build_engine(Settings(engine="diffusers")), DiffusersEngine)


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #
def test_a_fresh_engine_is_loading_not_broken():
    status = StubEngine(stub_settings()).status()
    assert status["state"] == "idle"
    # "idle" counts as loading, which is what makes /health answer 503 instead
    # of 500 during the first seconds of a cold start.
    assert status["loading"] is True
    assert status["loaded"] is False
    assert status["error"] == ""


def test_preload_disabled_defers_the_load():
    stub = StubEngine(stub_settings(preload=False))
    stub.start()
    assert stub.status()["state"] == "idle"


def test_start_loads_in_the_background():
    stub = StubEngine(stub_settings(preload=True))
    stub.start()
    assert stub.wait_ready(10) is True
    status = stub.status()
    assert status["state"] == "ready"
    assert status["loaded"] is True
    assert status["loading"] is False


def test_a_failed_load_reports_error_and_stops_loading():
    failing = FailingEngine(stub_settings())
    failing.ensure_loaded()
    status = failing.status()
    assert status["state"] == "error"
    assert status["loading"] is False
    assert "Gewichte" in status["error"]


def test_generate_on_a_broken_engine_raises():
    failing = FailingEngine(stub_settings())
    with pytest.raises(EngineError):
        failing.generate(request())


def test_generate_loads_lazily_when_preload_is_off():
    stub = StubEngine(stub_settings(preload=False))
    image, _meta = stub.generate(request())
    assert image.startswith(b"\x89PNG")
    assert stub.status()["state"] == "ready"


def test_status_reports_the_configured_engine_knobs():
    status = StubEngine(
        stub_settings(dtype="float32", quant="int8", device="cuda")
    ).status()
    assert status["engine"] == "stub"
    assert status["dtype"] == "float32"
    assert status["quant"] == "int8"
    assert status["device"] == "cuda"


def test_close_resets_the_engine():
    stub = StubEngine(stub_settings())
    stub.ensure_loaded()
    stub.close()
    assert stub.status()["state"] == "idle"


# --------------------------------------------------------------------------- #
# Stub rendering
# --------------------------------------------------------------------------- #
def test_stub_renders_a_png_of_the_requested_size():
    stub = StubEngine(stub_settings())
    image, meta = stub.generate(request(width=1024, height=768))
    with Image.open(io.BytesIO(image)) as decoded:
        assert decoded.size == (1024, 768)
        assert decoded.format == "PNG"
    assert meta["seed"] is not None
    assert meta["latency_ms"] >= 0


def test_stub_is_deterministic_for_a_given_seed():
    stub = StubEngine(stub_settings())
    first, _ = stub.generate(request(seed=42))
    second, _ = stub.generate(request(seed=42))
    assert first == second


def test_stub_differs_between_seeds():
    stub = StubEngine(stub_settings())
    first, _ = stub.generate(request(seed=1))
    second, _ = stub.generate(request(seed=2))
    assert first != second


def test_stub_honours_the_configured_delay():
    stub = StubEngine(stub_settings(stub_delay=0.2))
    started = time.monotonic()
    stub.generate(request(width=512, height=512))
    assert time.monotonic() - started >= 0.2


def test_renders_are_serialised_by_the_render_lock():
    """Two concurrent requests must not interleave inside the pipeline."""

    active = 0
    peak = 0
    guard = threading.Lock()

    class CountingStub(StubEngine):
        def _render(self, req):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.05)
                return super()._render(req)
            finally:
                with guard:
                    active -= 1

    stub = CountingStub(stub_settings())
    threads = [
        threading.Thread(target=stub.generate, args=(request(width=512, height=512),))
        for _ in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert peak == 1


# --------------------------------------------------------------------------- #
# diffusers version guard
# --------------------------------------------------------------------------- #
def test_a_diffusers_without_the_pipeline_fails_with_an_actionable_message(monkeypatch):
    """QwenImage21Pipeline needs diffusers>=0.41.0; older versions must not
    silently fall back to a different pipeline."""

    fake_torch = types.SimpleNamespace(
        bfloat16=object(),
        set_num_threads=lambda _n: None,
    )
    monkeypatch.setattr(DiffusersEngine, "_torch", lambda self: fake_torch)
    # A diffusers module that exists but predates the Qwen-Image-2.1 pipeline.
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace())

    engine = DiffusersEngine(stub_settings(engine="diffusers"))
    engine.ensure_loaded()

    status = engine.status()
    assert status["state"] == "error"
    assert "diffusers>=0.41.0" in status["error"]
    with pytest.raises(EngineError):
        engine.generate(request())


def test_the_pipeline_is_built_with_dtype_device_and_kv_cache(monkeypatch):
    """The wiring between settings and the pipeline call, without torch."""

    calls: dict = {}

    class FakePipeline:
        def __init__(self):
            self.images = []

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls["model"] = model
            calls["kwargs"] = kwargs
            return cls()

        def to(self, device):
            calls["device"] = device

        def set_progress_bar_config(self, **kwargs):
            calls["progress"] = kwargs

        def __call__(self, **kwargs):
            calls["render"] = kwargs
            return types.SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    fake_torch = types.SimpleNamespace(
        bfloat16=object(),
        set_num_threads=lambda n: calls.setdefault("threads", n),
        Generator=lambda device: types.SimpleNamespace(manual_seed=lambda s: calls.setdefault("seed", s)),
        inference_mode=lambda: _nullcontext(),
    )
    monkeypatch.setattr(DiffusersEngine, "_torch", lambda self: fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(QwenImage21Pipeline=FakePipeline),
    )

    engine = DiffusersEngine(
        stub_settings(
            engine="diffusers",
            cpu_threads=8,
            cache_dir="/models/hf",
            revision="main",
            negative_prompt="unscharf",
        )
    )
    image, meta = engine.generate(
        request(
            prompt="Ein Bergsee",
            width=1024,
            height=1024,
            steps=12,
            true_cfg_scale=1.0,
            seed=7,
        )
    )

    assert image.startswith(b"\x89PNG")
    assert meta["seed"] == 7
    assert calls["model"] == "Qwen/Qwen-Image-2.1"
    assert calls["kwargs"]["cache_dir"] == "/models/hf"
    assert calls["kwargs"]["revision"] == "main"
    assert calls["device"] == "cpu"
    assert calls["threads"] == 8
    assert calls["progress"] == {"disable": True}
    assert calls["render"]["num_inference_steps"] == 12
    assert calls["render"]["use_kv_cache"] is True
    # Guidance is off (1.0), so a negative prompt would have no effect and must
    # not be forwarded.
    assert "negative_prompt" not in calls["render"]


def test_a_negative_prompt_is_forwarded_when_guidance_is_on(monkeypatch):
    calls: dict = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            return cls()

        def to(self, device):
            return None

        def set_progress_bar_config(self, **kwargs):
            return None

        def __call__(self, **kwargs):
            calls["render"] = kwargs
            return types.SimpleNamespace(images=[Image.new("RGB", (64, 64))])

    fake_torch = types.SimpleNamespace(
        bfloat16=object(),
        Generator=lambda device: types.SimpleNamespace(manual_seed=lambda s: None),
        inference_mode=lambda: _nullcontext(),
    )
    monkeypatch.setattr(DiffusersEngine, "_torch", lambda self: fake_torch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(QwenImage21Pipeline=FakePipeline),
    )

    engine = DiffusersEngine(stub_settings(engine="diffusers", negative_prompt="default"))
    engine.generate(request(true_cfg_scale=4.0, negative_prompt="hässlich"))
    assert calls["render"]["negative_prompt"] == "hässlich"


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def test_quantisation_config_is_only_built_when_requested():
    engine = DiffusersEngine(stub_settings(engine="diffusers", quant="none"))
    assert engine._quantization_config(types.SimpleNamespace()) is None


def test_load_settings_round_trip_is_used_by_the_engine(monkeypatch):
    monkeypatch.setenv("IMAGEINT_IMAGE_ENGINE", "stub")
    settings = load_settings()
    assert isinstance(build_engine(settings), StubEngine)
