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


# --------------------------------------------------------------------------- #
# Download progress
# --------------------------------------------------------------------------- #
class RecordingProgress:
    """Stand-in for the tracker, so the wiring needs no download."""

    def __init__(self, payload: dict | None = None) -> None:
        self.payload = payload or {}
        self.started = 0
        self.stopped = 0
        self.finished = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def finish(self) -> None:
        self.finished += 1

    def snapshot(self) -> dict:
        return self.payload


def test_a_stub_engine_tracks_no_download():
    assert StubEngine(stub_settings())._progress is None


def test_a_diffusers_engine_tracks_its_checkpoint_download():
    engine = DiffusersEngine(Settings(engine="diffusers", model="Qwen/Qwen-Image-2.1"))
    assert engine._progress is not None
    assert engine._progress.repo_id == "Qwen/Qwen-Image-2.1"


def test_status_omits_the_progress_while_nothing_downloads():
    assert "progress" not in StubEngine(stub_settings()).status()


def test_status_carries_the_download_progress():
    engine = StubEngine(stub_settings())
    engine._progress = RecordingProgress({"known": True, "percent": 42.5})
    engine._state = "loading"

    assert engine.status()["progress"]["percent"] == 42.5


def test_a_successful_load_starts_and_finishes_the_tracker():
    engine = StubEngine(stub_settings())
    progress = RecordingProgress()
    engine._progress = progress

    engine.ensure_loaded()

    assert progress.started == 1
    assert progress.finished == 1
    assert progress.stopped == 0


def test_a_failed_load_stops_the_tracker():
    engine = FailingEngine(stub_settings())
    progress = RecordingProgress()
    engine._progress = progress

    engine.ensure_loaded()

    assert progress.started == 1
    assert progress.stopped == 1
    assert progress.finished == 0


def test_close_stops_the_tracker():
    engine = StubEngine(stub_settings())
    progress = RecordingProgress()
    engine._progress = progress

    engine.close()

    assert progress.stopped == 1


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


def _fake_quant_modules(monkeypatch):
    """A diffusers/transformers pair that exposes only the quantisation configs.

    diffusers>=0.41.0 wants a ``PipelineQuantizationConfig`` that maps a
    component name to a config, and each library has its own ``QuantoConfig``.
    """

    class FakeQuantoConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakePipelineQuantizationConfig:
        def __init__(self, quant_mapping=None):
            self.quant_mapping = quant_mapping or {}

    diffusers = types.ModuleType("diffusers")
    quantizers = types.ModuleType("diffusers.quantizers")
    quantizers.PipelineQuantizationConfig = FakePipelineQuantizationConfig
    diffusers_quant = types.ModuleType("diffusers.quantizers.quantization_config")
    diffusers_quant.QuantoConfig = FakeQuantoConfig

    transformers = types.ModuleType("transformers")
    transformers_utils = types.ModuleType("transformers.utils")
    transformers_quant = types.ModuleType("transformers.utils.quantization_config")
    transformers_quant.QuantoConfig = FakeQuantoConfig

    diffusers.quantizers = quantizers
    transformers.utils = transformers_utils
    transformers_utils.quantization_config = transformers_quant

    for name, module in (
        ("diffusers", diffusers),
        ("diffusers.quantizers", quantizers),
        ("diffusers.quantizers.quantization_config", diffusers_quant),
        ("transformers", transformers),
        ("transformers.utils", transformers_utils),
        ("transformers.utils.quantization_config", transformers_quant),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    return FakePipelineQuantizationConfig


@pytest.mark.parametrize("quant", ["int8", "int4"])
def test_quantisation_maps_the_components_diffusers_expects(monkeypatch, quant):
    """A bare ``QuantoConfig`` is rejected by diffusers>=0.41.0 and used to fail
    the whole load after the download had finished."""

    fake_config = _fake_quant_modules(monkeypatch)

    engine = DiffusersEngine(stub_settings(engine="diffusers", quant=quant))
    config = engine._quantization_config(types.SimpleNamespace())

    assert isinstance(config, fake_config)
    # Only the two components that dominate the footprint are quantised.
    assert set(config.quant_mapping) == {"transformer", "text_encoder"}
    assert config.quant_mapping["transformer"].kwargs == {"weights_dtype": quant}
    assert config.quant_mapping["text_encoder"].kwargs == {"weights": quant}


def test_quantisation_without_the_stack_fails_with_an_actionable_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "diffusers.quantizers", None)

    engine = DiffusersEngine(stub_settings(engine="diffusers", quant="int8"))

    with pytest.raises(EngineError) as excinfo:
        engine._quantization_config(types.SimpleNamespace())

    assert "optimum-quanto" in str(excinfo.value)


def test_load_settings_round_trip_is_used_by_the_engine(monkeypatch):
    monkeypatch.setenv("IMAGEINT_IMAGE_ENGINE", "stub")
    settings = load_settings()
    assert isinstance(build_engine(settings), StubEngine)

# --------------------------------------------------------------------------- #
# IMAGEINT_IMAGE_GPU_UTILIZATION -- the CUDA memory cap
# --------------------------------------------------------------------------- #
class FakeCuda:
    """Just enough of ``torch.cuda`` to observe the cap being applied."""

    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.calls: list = []

    def is_available(self) -> bool:
        return self.available

    def current_device(self) -> int:
        return 0

    def set_per_process_memory_fraction(self, fraction, device) -> None:
        self.calls.append((fraction, device))


class FakeTorch:
    def __init__(self, available: bool = True) -> None:
        self.cuda = FakeCuda(available)


def test_the_vram_cap_is_applied_before_the_weights_are_loaded():
    engine = DiffusersEngine(stub_settings(device="cuda", gpu_utilization=0.75))
    torch = FakeTorch()
    engine._limit_device_memory(torch)
    assert torch.cuda.calls == [(0.75, 0)]


def test_a_gpu_utilization_of_zero_leaves_torch_alone():
    # 0 is the default and means "no cap", so a CPU host is unaffected.
    engine = DiffusersEngine(stub_settings(device="cpu", gpu_utilization=0.0))
    torch = FakeTorch()
    engine._limit_device_memory(torch)
    assert torch.cuda.calls == []


def test_a_cap_without_cuda_is_reported_and_harmless(caplog):
    engine = DiffusersEngine(stub_settings(device="cuda", gpu_utilization=0.9))
    torch = FakeTorch(available=False)
    engine._limit_device_memory(torch)
    assert torch.cuda.calls == []
    assert any("wirkungslos" in record.message for record in caplog.records)


def test_status_reports_the_gpu_utilization():
    engine = DiffusersEngine(stub_settings(device="cuda", gpu_utilization=0.5))
    assert engine.status()["gpu_utilization"] == 0.5
