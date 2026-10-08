"""Engine behaviour: the thinking splitter, the stub generator and the
transformers wiring.

None of these tests needs torch, transformers or the 19 GB checkpoint. What
they pin down is the part the gateway depends on -- that ``status()``
distinguishes "still loading" from "broken", that the thinking block arrives in
its own field instead of inside the answer, and that the checkpoint is loaded
with the dtype, the device and the chat template the model card documents.

The transformers version is replaced by a fake module for the tests that need
it. That is the same trick ``image_server/tests/test_engine.py`` uses for
diffusers: the wiring is what is being verified, not the arithmetic.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from contextlib import nullcontext
from typing import Optional

import pytest

from enhancer_server import engine as engine_module
from enhancer_server.config import Settings
from enhancer_server.engine import (
    BaseEngine,
    ChatRequest,
    EngineError,
    StubEngine,
    ThinkingSplitter,
    TransformersEngine,
    build_engine,
    flatten_content,
    last_user_text,
    message_text,
    starts_in_thinking,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def stub_settings(**overrides) -> Settings:
    base = dict(engine="stub", preload=False, stub_delay=0.0)
    base.update(overrides)
    return Settings(**base)


def request(**overrides) -> ChatRequest:
    base = dict(
        messages=({"role": "user", "content": "Eine Katze im Regen"},),
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        max_tokens=512,
        seed=7,
        enable_thinking=True,
    )
    base.update(overrides)
    return ChatRequest(**base)


def run_splitter(chunks, *, start_in_thinking=True):
    """Feed ``chunks`` through a splitter and return ``(reasoning, answer)``."""

    splitter = ThinkingSplitter(start_in_thinking=start_in_thinking)
    pairs = []
    for chunk in chunks:
        pairs.extend(splitter.feed(chunk))
    pairs.extend(splitter.flush())
    reasoning = "".join(part for part, _ in pairs)
    answer = "".join(part for _, part in pairs)
    return reasoning, answer


class FailingEngine(BaseEngine):
    """An engine whose weights never arrive, to exercise the error state."""

    name = "failing"

    def _load(self) -> None:
        raise EngineError("Gewichte konnten nicht geladen werden.")


class CountingEngine(BaseEngine):
    """An engine that counts how often it was asked to load."""

    name = "counting"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.loads = 0

    def _load(self) -> None:
        self.loads += 1
        time.sleep(0.01)

    def _generate(self, request: ChatRequest, usage: Optional[dict] = None):
        return iter(())


# --------------------------------------------------------------------------- #
# ThinkingSplitter
# --------------------------------------------------------------------------- #
def test_the_thinking_block_becomes_reasoning():
    reasoning, answer = run_splitter(
        ["Ich überlege.\n", "Noch mehr.\n", "</think>\n", '{"rewritten_prompt": "x"}']
    )
    assert reasoning == "Ich überlege.\nNoch mehr.\n"
    assert answer == '{"rewritten_prompt": "x"}'


def test_a_close_marker_split_across_chunks_does_not_leak_into_the_answer():
    """The token stream decides where a chunk ends, not the marker."""

    reasoning, answer = run_splitter(["Gedanke", "</thi", "nk>", "Antwort"])
    assert reasoning == "Gedanke"
    assert answer == "Antwort"


def test_a_partial_marker_at_the_end_is_held_back_until_it_is_decided():
    splitter = ThinkingSplitter(start_in_thinking=True)
    assert splitter.feed("Gedanke<") == [("Gedanke", "")]
    assert splitter.feed("x") == [("<x", "")]
    assert splitter.flush() == []


def test_an_unterminated_thinking_block_is_reported_instead_of_dropped():
    """A truncated answer must not look like an empty one."""

    reasoning, answer = run_splitter(["nur gedacht, nie fertig"])
    assert reasoning == "nur gedacht, nie fertig"
    assert answer == ""


def test_the_opening_marker_is_stripped_when_the_model_emits_it():
    reasoning, answer = run_splitter(["<think>", "Gedanke", "</think>", "Antwort"])
    assert reasoning == "Gedanke"
    assert answer == "Antwort"


def test_a_template_without_the_prefill_delivers_content_from_the_first_token():
    reasoning, answer = run_splitter(['{"rewritten_prompt": "x"}'], start_in_thinking=False)
    assert reasoning == ""
    assert answer == '{"rewritten_prompt": "x"}'


def test_a_close_marker_without_a_prefill_is_still_honoured():
    reasoning, answer = run_splitter(
        ["<think>", "Gedanke", "</think>", "Antwort"], start_in_thinking=False
    )
    assert reasoning == "Gedanke"
    assert answer == "Antwort"


def test_empty_chunks_are_ignored():
    splitter = ThinkingSplitter(start_in_thinking=True)
    assert splitter.feed("") == []
    assert splitter.flush() == []


def test_flush_after_a_complete_answer_emits_nothing_twice():
    reasoning, answer = run_splitter(["Gedanke", "</think>", "Antwort"])
    assert (reasoning, answer) == ("Gedanke", "Antwort")


def test_marker_like_text_inside_the_answer_is_kept():
    reasoning, answer = run_splitter(
        ["Gedanke", "</think>", "Er sagte <think> und meinte es."]
    )
    assert reasoning == "Gedanke"
    assert answer == "Er sagte <think> und meinte es."


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("<|im_start|>assistant\n<think>\n", True),
        ("<|im_start|>assistant\n<think>\nGedanke\n</think>\n", False),
        ("<|im_start|>assistant\n", False),
        ("", False),
    ],
)
def test_starts_in_thinking_follows_the_rendered_prompt(prompt, expected):
    assert starts_in_thinking(prompt) is expected


# --------------------------------------------------------------------------- #
# Message helpers
# --------------------------------------------------------------------------- #
def test_message_text_accepts_a_bare_string():
    assert message_text({"role": "user", "content": "Hallo"}) == "Hallo"


def test_message_text_joins_the_part_list_the_gateway_sends():
    message = {
        "role": "user",
        "content": [{"type": "text", "text": "Hallo "}, {"type": "text", "text": "Welt"}],
    }
    assert message_text(message) == "Hallo Welt"


def test_message_text_of_a_missing_content_is_empty():
    assert message_text({"role": "user"}) == ""


def test_last_user_text_ignores_the_system_prompt():
    messages = (
        {"role": "system", "content": "Du bist ein Umschreiber."},
        {"role": "user", "content": "Ein Berg"},
    )
    assert last_user_text(messages) == "Ein Berg"


def test_last_user_text_takes_the_final_user_turn():
    messages = (
        {"role": "user", "content": "Ein Berg"},
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "Doch lieber ein See"},
    )
    assert last_user_text(messages) == "Doch lieber ein See"


def test_last_user_text_falls_back_to_a_non_user_turn():
    assert last_user_text(({"role": "system", "content": "Nur System"},)) == "Nur System"


def test_flatten_content_turns_part_lists_into_strings():
    messages = ({"role": "user", "content": [{"type": "text", "text": "Hallo"}]},)
    assert flatten_content(messages) == [{"role": "user", "content": "Hallo"}]


# --------------------------------------------------------------------------- #
# BaseEngine
# --------------------------------------------------------------------------- #
def test_build_engine_picks_the_stub():
    assert isinstance(build_engine(stub_settings()), StubEngine)


def test_build_engine_picks_transformers_without_importing_torch():
    """Constructing the engine must stay cheap; torch is imported on load."""

    built = build_engine(Settings(engine="transformers"))
    assert isinstance(built, TransformersEngine)


def test_a_fresh_engine_reports_idle_and_not_loaded():
    status = CountingEngine(stub_settings()).status()
    assert status["state"] == "idle"
    assert status["loading"] is True
    assert status["loaded"] is False


def test_ensure_loaded_moves_to_ready():
    engine = CountingEngine(stub_settings())
    engine.ensure_loaded()
    assert engine.status()["state"] == "ready"
    assert engine.status()["loaded"] is True
    assert engine.status()["load_seconds"] >= 0


def test_ensure_loaded_only_loads_once_under_concurrency():
    engine = CountingEngine(stub_settings())
    threads = [threading.Thread(target=engine.ensure_loaded) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert engine.loads == 1


def test_a_failing_load_is_reported_as_error_not_as_loading():
    engine = FailingEngine(stub_settings())
    engine.ensure_loaded()
    status = engine.status()
    assert status["state"] == "error"
    assert "Gewichte" in status["error"]
    assert status["loading"] is False
    assert status["loaded"] is False


def test_generating_from_a_broken_engine_raises_instead_of_hanging():
    engine = FailingEngine(stub_settings())
    with pytest.raises(EngineError):
        list(engine.stream(request()))


def test_preload_off_leaves_the_engine_idle_until_a_request_arrives():
    engine = CountingEngine(stub_settings(preload=False))
    engine.start()
    assert engine.status()["state"] == "idle"
    list(engine.stream(request()))
    assert engine.status()["state"] == "ready"


def test_preload_on_loads_in_the_background():
    engine = CountingEngine(stub_settings(preload=True))
    engine.start()
    assert engine.wait_ready(10) is True
    assert engine.loads == 1


def test_close_releases_the_engine():
    engine = CountingEngine(stub_settings(preload=True))
    engine.ensure_loaded()
    engine.close()
    assert engine.status()["state"] == "idle"


def test_the_seed_is_taken_from_the_request_when_it_is_given():
    engine = CountingEngine(stub_settings())
    assert engine._pick_seed(1234) == 1234


def test_a_random_seed_stays_inside_the_supported_range():
    engine = CountingEngine(stub_settings())
    for _ in range(20):
        assert 0 <= engine._pick_seed(None) < 2**31


# --------------------------------------------------------------------------- #
# StubEngine
# --------------------------------------------------------------------------- #
def test_the_stub_streams_reasoning_and_then_the_answer():
    engine = StubEngine(stub_settings())
    usage: dict = {}
    pairs = list(engine.stream(request(), usage))
    reasoning = "".join(part for part, _ in pairs)
    answer = "".join(part for _, part in pairs)
    assert reasoning.startswith("Der Stub-Enhancer")
    assert answer.startswith("{")
    assert '"rewritten_prompt"' in answer
    assert usage["completion_tokens"] > 0


def test_the_stub_answer_carries_the_prompt_through():
    engine = StubEngine(stub_settings())
    answer = "".join(part for _, part in engine.stream(request()))
    assert "Eine Katze im Regen" in answer


def test_the_stub_can_skip_the_thinking_block():
    engine = StubEngine(stub_settings(enable_thinking=False))
    pairs = list(engine.stream(request(enable_thinking=False)))
    assert "".join(part for part, _ in pairs) == ""
    assert "".join(part for _, part in pairs).startswith("{")


def test_the_stub_is_deterministic_for_a_given_seed():
    engine = StubEngine(stub_settings())
    first = "".join(part for _, part in engine.stream(request(seed=99)))
    second = "".join(part for _, part in engine.stream(request(seed=99)))
    assert first == second


# --------------------------------------------------------------------------- #
# Fake transformers, for the wiring of TransformersEngine
# --------------------------------------------------------------------------- #
class FakeTensor:
    """Just enough tensor to survive ``.to()`` and ``.shape``."""

    def __init__(self, rows) -> None:
        self.rows = rows

    def to(self, device):
        self.device = device
        return self

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)


class FakeTokenizer:
    def __init__(self, template=None) -> None:
        self.template = template
        self.template_calls: list = []
        self.tokenize_calls: list = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        if self.template is not None:
            return self.template(messages, kwargs)
        body = "".join(message_text(message) for message in messages)
        return body + "<|im_start|>assistant\n<think>\n"

    def __call__(self, text=None, add_special_tokens=True, return_tensors=None):
        self.tokenize_calls.append((text, add_special_tokens))
        single = isinstance(text, str)
        items = [text] if single else list(text or [])
        rows = [[1] * (len(item) // 4 + 1) for item in items]
        if return_tensors is None:
            # A real tokenizer returns a flat list of ids for one sequence and a
            # list of lists for a batch, which is what count_tokens() relies on.
            return {"input_ids": rows[0] if single else rows}
        return {"input_ids": FakeTensor(rows), "attention_mask": FakeTensor(rows)}


class FakeProcessor:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, **kwargs):
        return self.tokenizer(**kwargs)


class FakeModel:
    def __init__(self, max_position_embeddings=32768) -> None:
        self.config = types.SimpleNamespace(
            max_position_embeddings=max_position_embeddings
        )
        self.device = None
        self.evaluated = False
        self.generate_kwargs: dict = {}

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs


class FakeTransformers:
    """A stand-in for the ``transformers`` module.

    Records every ``from_pretrained`` call and every streamer construction, so
    a test can assert what the engine asked for rather than what came back.
    """

    def __init__(
        self,
        *,
        stream_script=(),
        generation_fields=(),
        quanto: bool = True,
        processor_error: Optional[Exception] = None,
        tokenizer_error: Optional[Exception] = None,
        model_classes=("AutoModelForCausalLM",),
        template=None,
        max_position_embeddings: int = 32768,
    ) -> None:
        self.stream_script = list(stream_script)
        self.processor_error = processor_error
        self.tokenizer_error = tokenizer_error
        self.model = FakeModel(max_position_embeddings)
        self.processor = FakeProcessor(FakeTokenizer(template))
        self.tokenizer = self.processor.tokenizer
        self.processor_calls: list = []
        self.tokenizer_calls: list = []
        self.model_calls: list = []
        self.quanto_calls: list = []
        self.streamer_kwargs: dict = {}
        self.streamer_ended = False
        fake = self

        class FakeGenerationConfig:
            __dataclass_fields__ = {
                name: None
                for name in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "do_sample",
                    "max_new_tokens",
                    *generation_fields,
                )
            }

        class FakeStoppingCriteria:
            pass

        class FakeStoppingCriteriaList(list):
            pass

        class FakeQuantoConfig:
            def __init__(self, weights=None) -> None:
                fake.quanto_calls.append(weights)
                self.weights = weights

        class FakeTextIteratorStreamer:
            def __init__(self, tokenizer, **kwargs) -> None:
                fake.streamer_kwargs = kwargs

            def __iter__(self):
                yield from fake.stream_script

            def end(self) -> None:
                fake.streamer_ended = True

        module = types.ModuleType("transformers")
        module.GenerationConfig = FakeGenerationConfig
        module.StoppingCriteria = FakeStoppingCriteria
        module.StoppingCriteriaList = FakeStoppingCriteriaList
        module.TextIteratorStreamer = FakeTextIteratorStreamer
        if quanto:
            module.QuantoConfig = FakeQuantoConfig

        class AutoProcessor:
            @staticmethod
            def from_pretrained(model, **kwargs):
                fake.processor_calls.append((model, kwargs))
                if fake.processor_error is not None:
                    raise fake.processor_error
                return fake.processor

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(model, **kwargs):
                fake.tokenizer_calls.append((model, kwargs))
                if fake.tokenizer_error is not None:
                    raise fake.tokenizer_error
                return fake.tokenizer

        module.AutoProcessor = AutoProcessor
        module.AutoTokenizer = AutoTokenizer

        for name in model_classes:
            setattr(module, name, _model_class(name, fake))
        self.module = module

    def install(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "transformers", self.module)


def _model_class(name, fake):
    class _Model:
        @staticmethod
        def from_pretrained(model, **kwargs):
            fake.model_calls.append((model, kwargs))
            return fake.model

    _Model.__name__ = name
    return _Model


def fake_torch(monkeypatch, *, threads_log=None) -> types.ModuleType:
    module = types.ModuleType("torch")
    module.Tensor = FakeTensor
    module.bfloat16 = "bfloat16"
    module.float32 = "float32"
    module.float16 = "float16"
    module.manual_seed = lambda seed: None
    module.set_num_threads = lambda count: (
        threads_log.append(count) if threads_log is not None else None
    )
    module.inference_mode = lambda: nullcontext()
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def ready_engine(fake: FakeTransformers, monkeypatch, **overrides) -> TransformersEngine:
    """A TransformersEngine with the fake module installed and weights loaded."""

    fake_torch(monkeypatch, threads_log=overrides.pop("threads_log", None))
    fake.install(monkeypatch)
    engine = TransformersEngine(stub_settings(**overrides))
    engine.ensure_loaded()
    return engine


# --------------------------------------------------------------------------- #
# TransformersEngine: loading
# --------------------------------------------------------------------------- #
def test_the_weights_are_loaded_with_dtype_device_and_remote_code(monkeypatch):
    fake = FakeTransformers()
    engine = ready_engine(fake, monkeypatch, quant="none", cpu_threads=4)

    assert engine.status()["state"] == "ready"
    model, kwargs = fake.model_calls[0]
    assert model == "Qwen/Qwen-Image-2.1-PE-T2I"
    assert kwargs["torch_dtype"] == "bfloat16"
    assert kwargs["trust_remote_code"] is True
    assert "quantization_config" not in kwargs
    assert fake.model.device == "cpu"
    assert fake.model.evaluated is True


def test_int8_quantisation_is_requested_from_quanto(monkeypatch):
    fake = FakeTransformers()
    ready_engine(fake, monkeypatch, quant="int8")
    assert fake.quanto_calls == ["int8"]
    assert fake.model_calls[0][1]["quantization_config"].weights == "int8"


def test_quantisation_without_quanto_fails_with_an_actionable_message(monkeypatch):
    fake = FakeTransformers(quanto=False)
    fake_torch(monkeypatch)
    fake.install(monkeypatch)
    engine = TransformersEngine(stub_settings(quant="int4"))
    engine.ensure_loaded()
    assert engine.status()["state"] == "error"
    assert "QuantoConfig" in engine.status()["error"]


def test_the_cpu_thread_count_is_applied_to_torch(monkeypatch):
    fake = FakeTransformers()
    threads: list = []
    ready_engine(fake, monkeypatch, cpu_threads=8, threads_log=threads)
    assert threads == [8]


def test_torchs_default_thread_count_is_left_alone(monkeypatch):
    fake = FakeTransformers()
    threads: list = []
    ready_engine(fake, monkeypatch, cpu_threads=0, threads_log=threads)
    assert threads == []


def test_the_processor_is_preferred_over_the_tokenizer(monkeypatch):
    fake = FakeTransformers()
    ready_engine(fake, monkeypatch)
    assert len(fake.processor_calls) == 1
    assert fake.tokenizer_calls == []


def test_the_tokenizer_is_the_fallback_when_there_is_no_processor(monkeypatch):
    fake = FakeTransformers(processor_error=ValueError("kein Prozessor"))
    ready_engine(fake, monkeypatch)
    assert len(fake.tokenizer_calls) == 1


def test_a_model_without_a_chat_template_fails_loudly(monkeypatch):
    fake = FakeTransformers(processor_error=ValueError("nein"), tokenizer_error=ValueError("nein"))
    fake_torch(monkeypatch)
    fake.install(monkeypatch)
    engine = TransformersEngine(stub_settings())
    engine.ensure_loaded()
    assert engine.status()["state"] == "error"
    assert "AutoProcessor" in engine.status()["error"]


def test_the_first_available_model_class_is_used(monkeypatch):
    fake = FakeTransformers(model_classes=("Qwen3VLForConditionalGeneration",))
    ready_engine(fake, monkeypatch)
    assert fake.model_calls[0][0] == "Qwen/Qwen-Image-2.1-PE-T2I"


def test_a_transformers_without_any_known_model_class_is_rejected(monkeypatch):
    fake = FakeTransformers(model_classes=())
    fake_torch(monkeypatch)
    fake.install(monkeypatch)
    engine = TransformersEngine(stub_settings())
    engine.ensure_loaded()
    assert engine.status()["state"] == "error"
    assert "AutoModelForCausalLM" in engine.status()["error"]


def test_an_unknown_dtype_is_rejected(monkeypatch):
    fake = FakeTransformers()
    fake_torch(monkeypatch)
    fake.install(monkeypatch)
    engine = TransformersEngine(stub_settings(dtype="int4"))
    engine.ensure_loaded()
    assert engine.status()["state"] == "error"


def test_a_tight_answer_budget_is_warned_about(monkeypatch, caplog):
    fake = FakeTransformers(max_position_embeddings=4096)
    with caplog.at_level("WARNING"):
        ready_engine(fake, monkeypatch, max_new_tokens=4096)
    assert "IMAGEINT_PE_MAX_NEW_TOKENS" in caplog.text


# --------------------------------------------------------------------------- #
# TransformersEngine: the chat template
# --------------------------------------------------------------------------- #
def test_the_part_list_shape_the_gateway_sends_is_used_first(monkeypatch):
    fake = FakeTransformers()
    engine = ready_engine(fake, monkeypatch)
    parts = [{"type": "text", "text": "Eine Katze im Regen"}]
    engine.render_chat(request(messages=({"role": "user", "content": parts},)))
    messages, kwargs = fake.tokenizer.template_calls[0]
    assert messages[0]["content"] == parts
    assert kwargs["tokenize"] is False
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["enable_thinking"] is True
    assert len(fake.tokenizer.template_calls) == 1


def test_a_template_that_rejects_lists_gets_the_flattened_form(monkeypatch):
    def template(messages, kwargs):
        if isinstance(messages[0]["content"], list):
            raise TypeError("nur Strings")
        return "<|im_start|>assistant\n"

    fake = FakeTransformers(template=template)
    engine = ready_engine(fake, monkeypatch)
    engine.render_chat(request())
    assert fake.tokenizer.template_calls[-1][0][0]["content"] == "Eine Katze im Regen"


def test_a_template_without_the_thinking_switch_is_retried_without_it(monkeypatch):
    def template(messages, kwargs):
        if "enable_thinking" in kwargs:
            raise TypeError("unerwartetes Argument")
        return "<|im_start|>assistant\n"

    fake = FakeTransformers(template=template)
    engine = ready_engine(fake, monkeypatch)
    engine.render_chat(request())
    assert "enable_thinking" not in fake.tokenizer.template_calls[-1][1]


def test_thinking_off_is_forwarded_as_a_template_argument(monkeypatch):
    fake = FakeTransformers()
    engine = ready_engine(fake, monkeypatch)
    engine.render_chat(request(enable_thinking=False))
    assert fake.tokenizer.template_calls[0][1]["enable_thinking"] is False


def test_a_template_that_never_works_raises_an_actionable_error(monkeypatch):
    def template(messages, kwargs):
        raise TypeError("kaputt")

    fake = FakeTransformers(template=template)
    engine = ready_engine(fake, monkeypatch)
    with pytest.raises(EngineError, match="Chat-Vorlage"):
        engine.render_chat(request())


def test_the_splitter_follows_the_rendered_template(monkeypatch):
    fake = FakeTransformers(template=lambda m, k: "<|im_start|>assistant\n")
    engine = ready_engine(fake, monkeypatch)
    assert engine._splitter(request()).in_thinking is False


# --------------------------------------------------------------------------- #
# TransformersEngine: sampling and generation
# --------------------------------------------------------------------------- #
def test_the_documented_sampling_parameters_are_passed_through(monkeypatch):
    fake = FakeTransformers(
        generation_fields=("min_p", "presence_penalty"),
        stream_script=["<think>\n", "Gedanke", "</think>", "Antwort"],
    )
    engine = ready_engine(fake, monkeypatch)
    list(engine.stream(request()))
    kwargs = fake.model.generate_kwargs
    assert kwargs["temperature"] == 1.0
    assert kwargs["top_p"] == 0.95
    assert kwargs["top_k"] == 20
    assert kwargs["min_p"] == 0.0
    assert kwargs["presence_penalty"] == 1.5
    assert kwargs["max_new_tokens"] == 512
    assert kwargs["do_sample"] is True


def test_a_missing_presence_penalty_is_reported_once(monkeypatch, caplog):
    fake = FakeTransformers(stream_script=["<think>\n", "</think>", "Antwort"])
    engine = ready_engine(fake, monkeypatch)
    with caplog.at_level("WARNING"):
        list(engine.stream(request()))
        list(engine.stream(request()))
    assert caplog.text.count("presence_penalty") == 1
    assert "presence_penalty" not in fake.model.generate_kwargs


def test_temperature_zero_turns_sampling_off(monkeypatch):
    fake = FakeTransformers(stream_script=["<think>\n", "</think>", "Antwort"])
    engine = ready_engine(fake, monkeypatch)
    list(engine.stream(request(temperature=0.0)))
    assert fake.model.generate_kwargs["do_sample"] is False


def test_the_streamer_skips_the_prompt_and_special_tokens(monkeypatch):
    fake = FakeTransformers(stream_script=["<think>\n", "</think>", "Antwort"])
    engine = ready_engine(fake, monkeypatch)
    list(engine.stream(request()))
    assert fake.streamer_kwargs == {
        "skip_prompt": True,
        "skip_special_tokens": True,
        "timeout": None,
    }


def test_the_streaming_answer_is_split_into_reasoning_and_content(monkeypatch):
    fake = FakeTransformers(
        stream_script=["<think>\n", "Ich überlege.\n", "</think>\n", '{"a": 1}']
    )
    engine = ready_engine(fake, monkeypatch)
    pairs = list(engine.stream(request()))
    assert "".join(part for part, _ in pairs) == "Ich überlege.\n"
    assert "".join(part for _, part in pairs) == '{"a": 1}'


def test_the_usage_report_counts_prompt_and_completion(monkeypatch):
    fake = FakeTransformers(stream_script=["<think>\n", "</think>", "Antwort"])
    engine = ready_engine(fake, monkeypatch)
    usage: dict = {}
    list(engine.stream(request(), usage))
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_a_generation_failure_reaches_the_caller(monkeypatch):
    fake = FakeTransformers(stream_script=["<think>\n", "</think>", "Antwort"])
    engine = ready_engine(fake, monkeypatch)

    def explode(**kwargs):
        raise RuntimeError("kein Speicher")

    fake.model.generate = explode
    with pytest.raises(EngineError, match="kein Speicher"):
        list(engine.stream(request()))


def test_aborting_the_stream_stops_the_decoder(monkeypatch):
    """A client that hangs up must not keep the single-flight lock for minutes."""

    fake = FakeTransformers(stream_script=["<think>\n", "viel Text", "</think>", "x"] * 50)
    engine = ready_engine(fake, monkeypatch)
    stream = engine.stream(request())
    next(stream)
    stream.close()
    # The lock is free again, so a second generation can start immediately.
    assert engine._generate_lock.acquire(timeout=2) is True
    engine._generate_lock.release()


def test_the_abort_criterion_is_absent_without_transformers(monkeypatch):
    fake = FakeTransformers()
    engine = ready_engine(fake, monkeypatch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace())
    assert engine._abort_criteria(threading.Event()) is None


def test_non_tensor_processor_output_is_dropped(monkeypatch):
    fake = FakeTransformers()
    engine = ready_engine(fake, monkeypatch)

    def noisy(text=None, add_special_tokens=True, return_tensors=None):
        return {
            "input_ids": FakeTensor([[1, 2, 3]]),
            "attention_mask": FakeTensor([[1, 1, 1]]),
            "pixel_values": None,
            "image_grid_thw": "kein Tensor",
        }

    engine._processor = types.SimpleNamespace(__call__=noisy)
    inputs = engine._tokenize("<|im_start|>assistant\n")
    assert set(inputs) == {"input_ids", "attention_mask"}


def test_a_processor_that_returns_no_input_ids_falls_back(monkeypatch):
    fake = FakeTransformers()
    engine = ready_engine(fake, monkeypatch)
    engine._processor = types.SimpleNamespace(
        __call__=lambda **kwargs: {"pixel_values": FakeTensor([[1]])}
    )
    inputs = engine._tokenize("<|im_start|>assistant\n")
    assert "input_ids" in inputs


def test_counting_tokens_of_an_empty_answer_is_zero(monkeypatch):
    fake = FakeTransformers()
    engine = ready_engine(fake, monkeypatch)
    assert engine.count_tokens("") == 0


def test_close_drops_the_weights(monkeypatch):
    fake = FakeTransformers()
    engine = ready_engine(fake, monkeypatch)
    engine.close()
    assert engine._model is None
    assert engine._tokenizer is None


def test_a_prompt_that_overflows_the_context_is_warned_about(monkeypatch, caplog):
    fake = FakeTransformers(
        max_position_embeddings=2048, stream_script=["<think>\n", "</think>", "x"]
    )
    engine = ready_engine(fake, monkeypatch)
    long_request = request(
        messages=({"role": "user", "content": "sehr " * 2000},), max_tokens=1024
    )
    with caplog.at_level("WARNING"):
        list(engine.stream(long_request))
    assert "überschreitet den Kontext" in caplog.text


def test_the_module_helpers_stay_exported():
    for name in ("EngineError", "StubEngine", "TransformersEngine", "build_engine"):
        assert hasattr(engine_module, name)
