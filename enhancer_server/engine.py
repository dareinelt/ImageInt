"""The generation backends behind ``/v1/chat/completions``.

Two engines implement one small interface:

``TransformersEngine``
    Runs ``Qwen-Image-2.1-PE-T2I`` through transformers -- the chat template of
    the checkpoint, its own modelling code and a ``TextIteratorStreamer`` for
    the token-by-token answer. This is the CPU path the service is built
    around, and the same code runs on a CUDA device.

``StubEngine``
    Answers with a canned enhanced prompt. It exists so the HTTP contract, the
    gateway integration and the documentation can be exercised on a machine
    that cannot hold 19 GB of weights; it is never a production backend.

Both are single-flight on purpose: one 9B decoder saturates an 8-core CPU, and
a model object is not safe to call from two threads at once.
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

from .config import Settings

log = logging.getLogger("imageint.enhancer.engine")

#: Model classes that can host the checkpoint, in the order they are tried. The
#: enhancer is a Qwen3.5-VL derivative, so the plain causal-LM entry point may
#: not be the right one -- but which one is right depends on the transformers
#: version, so the class is looked up by name at load time instead of being
#: imported directly.
MODEL_CLASSES = (
    "AutoModelForCausalLM",
    "Qwen3VLForConditionalGeneration",
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
)


class EngineError(RuntimeError):
    """Raised when the engine cannot load or cannot generate."""


@dataclass(frozen=True)
class ChatRequest:
    """One normalised chat-completion call."""

    messages: tuple[dict, ...]
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    max_tokens: int
    seed: Optional[int] = None
    enable_thinking: bool = True


def message_text(message: dict) -> str:
    """The text of one message, whether it is a string or a part list."""

    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for piece in content:
            if isinstance(piece, dict) and piece.get("type") == "text":
                parts.append(str(piece.get("text") or ""))
            elif isinstance(piece, str):
                parts.append(piece)
        return "".join(parts)
    return ""


def last_user_text(messages: Sequence[dict]) -> str:
    """The text of the last user turn, or the last turn of any role."""

    fallback = ""
    user_text = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        text = message_text(message)
        if not text:
            continue
        fallback = text
        if message.get("role") == "user":
            user_text = text
    return user_text or fallback


def flatten_content(messages: Sequence[dict]) -> list[dict]:
    """Turn part-list content back into plain strings.

    The checkpoint's chat template expects the part-list shape and that is what
    is sent, but a template that only understands bare strings would render the
    list as its Python ``repr``. This is the second attempt, not the first.
    """

    flattened = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        copy = dict(message)
        copy["content"] = message_text(message)
        flattened.append(copy)
    return flattened


def starts_in_thinking(rendered_prompt: str) -> bool:
    """Whether generation begins *inside* a thinking block.

    The chat template of the checkpoint pre-fills ``<think>\\n`` before the
    model is asked to continue, so the first generated token is already inside
    the block and the model closes it with ``</think>``. Counting the markers in
    the rendered prompt is what tells the two cases apart -- assuming the
    pre-fill instead would swallow the whole answer whenever thinking is off.
    """

    return rendered_prompt.count("<think>") > rendered_prompt.count("</think>")


class ThinkingSplitter:
    """Route a decoded stream into its ``(reasoning, answer)`` halves.

    vLLM does this with ``--reasoning-parser qwen3``; this server has to do it
    itself, because transformers returns one flat token stream. Every yielded
    pair is ``(reasoning_delta, content_delta)`` and one of the two is empty.

    A chunk boundary can fall inside ``</think>``, so a possible partial marker
    at the end of the buffer is held back until the next chunk decides it.
    Without that, ``</thi`` would leak into the answer.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, *, start_in_thinking: bool = True) -> None:
        self._buffer = ""
        self._in_thinking = start_in_thinking
        self._emitted = False
        self._reasoning_started = False
        self._content_started = False

    @property
    def in_thinking(self) -> bool:
        return self._in_thinking

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Absorb one delta and return the pieces that are safe to emit."""

        if not text:
            return []
        self._buffer += text

        # A template that does not pre-fill the block may still see the model
        # open one, and the marker can arrive in pieces. Nothing is emitted
        # until that is decided either way.
        if not self._in_thinking and not self._emitted:
            if self._buffer.startswith(self.OPEN):
                self._in_thinking = True
            elif self.OPEN.startswith(self._buffer):
                return []

        out: list[tuple[str, str]] = []

        if self._in_thinking:
            if self.CLOSE in self._buffer:
                head, _, tail = self._buffer.partition(self.CLOSE)
                reasoning = self._strip_open(head)
                self._buffer = tail
                self._in_thinking = False
                self._emitted = True
                self._emit(out, reasoning, "")
            else:
                keep = self._hold_back()
                cut = len(self._buffer) - keep
                if cut <= 0:
                    return out
                reasoning = self._strip_open(self._buffer[:cut])
                self._buffer = self._buffer[cut:]
                self._emit(out, reasoning, "")
                return out

        if self._buffer:
            piece = self._buffer
            self._buffer = ""
            self._emitted = True
            self._emit(out, "", piece)
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Emit what is still buffered.

        An unterminated thinking block means the token budget ran out before
        the answer. The text is handed over as reasoning rather than dropped,
        so the caller can see that the generation was cut off instead of
        receiving a silently empty prompt.
        """

        if not self._buffer:
            return []
        text = self._strip_open(self._buffer)
        self._buffer = ""
        out: list[tuple[str, str]] = []
        if self._in_thinking:
            self._emit(out, text, "")
        else:
            self._emit(out, "", text)
        return out

    def _emit(self, out: list[tuple[str, str]], reasoning: str, content: str) -> None:
        """Append the non-empty parts of one pair, trimming the lead-in.

        The template pre-fills ``<think>\\n`` and the model answers with
        ``</think>\\n\\n``, so both streams would otherwise start with the
        newlines those markers left behind.
        """

        if reasoning:
            if not self._reasoning_started:
                reasoning = reasoning.lstrip()
                self._reasoning_started = bool(reasoning)
            if reasoning:
                out.append((reasoning, ""))
        if content:
            if not self._content_started:
                content = content.lstrip()
                self._content_started = bool(content)
            if content:
                out.append(("", content))

    def _strip_open(self, text: str) -> str:
        if not self._emitted and text.startswith(self.OPEN):
            return text[len(self.OPEN) :]
        return text

    def _hold_back(self) -> int:
        """Length of the trailing slice that could still become a marker."""

        keep = 0
        for marker in (self.CLOSE, self.OPEN):
            limit = min(len(self._buffer), len(marker) - 1)
            for size in range(limit, 0, -1):
                if self._buffer.endswith(marker[:size]):
                    keep = max(keep, size)
                    break
        return keep


class BaseEngine:
    """Loading state, locking and status reporting shared by both engines."""

    #: Value of the ``engine`` field in every response.
    name = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        # idle -> loading -> ready | error. Read without the lock from the
        # health endpoint; a plain attribute read is atomic enough here.
        self._state = "idle"
        self._error = ""
        self._load_seconds = 0.0
        self._loaded_at = 0.0
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        """Begin loading in the background so ``/health`` can report progress."""

        if not self.settings.preload:
            log.info("[enhancer] preload disabled, loading on first request")
            return
        self._thread = threading.Thread(
            target=self.ensure_loaded, name="imageint-enhancer-load", daemon=True
        )
        self._thread.start()

    def wait_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until the engine is ready or the timeout expires."""

        if self._thread is None:
            return self._state == "ready"
        self._thread.join(timeout)
        return self._state == "ready"

    def ensure_loaded(self) -> None:
        """Load the weights once, no matter how many callers ask for it."""

        if self._state == "ready":
            return
        with self._load_lock:
            if self._state == "ready":
                return
            self._state = "loading"
            self._error = ""
            started = time.monotonic()
            log.info(
                "[enhancer] loading %s on %s (%s, %s)",
                self.settings.model,
                self.settings.device,
                self.settings.dtype,
                self.settings.quant,
            )
            try:
                self._load()
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                self._state = "error"
                self._error = f"{type(exc).__name__}: {exc}"
                log.exception("[enhancer] loading %s failed", self.settings.model)
                return
            self._load_seconds = time.monotonic() - started
            self._loaded_at = time.time()
            self._state = "ready"
            log.info(
                "[enhancer] %s ready after %.1f s",
                self.settings.model,
                self._load_seconds,
            )

    # -- generation -------------------------------------------------------- #

    def stream(
        self,
        request: ChatRequest,
        usage: Optional[dict] = None,
    ) -> Iterator[tuple[str, str]]:
        """Generate one answer, yielding ``(reasoning_delta, content_delta)``.

        ``usage`` is filled in by the engine, which is the only side that can
        count tokens: it is passed in rather than returned because the deltas
        are streamed while the counts are only known at the end.
        """

        self.ensure_loaded()
        if self._state != "ready":
            raise EngineError(self._error or "Das Enhancer-Modell ist nicht geladen.")
        with self._generate_lock:
            splitter = self._splitter(request)
            for chunk in self._generate(request, usage):
                for piece in splitter.feed(chunk):
                    yield piece
            for piece in splitter.flush():
                yield piece

    def _load(self) -> None:
        raise NotImplementedError

    def _generate(
        self, request: ChatRequest, usage: Optional[dict] = None
    ) -> Iterator[str]:
        raise NotImplementedError

    def _splitter(self, request: ChatRequest) -> ThinkingSplitter:
        """The splitter for this request.

        ``enable_thinking`` has already been resolved against the operator's
        setting by ``normalise()``, so following the request keeps the splitter
        consistent with what the engine actually generates. An engine whose
        template decides for itself -- the transformers one -- overrides this
        with the rendered prompt.
        """

        return ThinkingSplitter(start_in_thinking=bool(request.enable_thinking))

    def close(self) -> None:
        """Release whatever the engine holds. Called on application shutdown."""

        self._state = "idle"
        self._error = ""

    # -- reporting --------------------------------------------------------- #

    def status(self) -> dict:
        return {
            "engine": self.name,
            "model": self.settings.model,
            "device": self.settings.device,
            "dtype": self.settings.dtype,
            "quant": self.settings.quant,
            "state": self._state,
            "loaded": self._state == "ready",
            "loading": self._state in ("idle", "loading"),
            "error": self._error,
            "load_seconds": round(self._load_seconds, 1),
            "loaded_at": self._loaded_at,
        }

    # -- helpers ----------------------------------------------------------- #

    def _pick_seed(self, requested: Optional[int]) -> int:
        if requested is not None:
            return int(requested) % (2**31)
        return random.SystemRandom().randrange(0, 2**31)


class TransformersEngine(BaseEngine):
    """Qwen-Image-2.1-PE-T2I through transformers."""

    name = "transformers"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._warned_presence_penalty = False

    # -- loading ----------------------------------------------------------- #

    def _torch(self):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - container always has it
            raise EngineError(
                "torch ist nicht installiert; dieser Server braucht torch und "
                "transformers."
            ) from exc
        return torch

    def _load(self) -> None:
        torch = self._torch()
        try:
            import transformers
        except ImportError as exc:  # pragma: no cover - container always has it
            raise EngineError("transformers ist nicht installiert.") from exc

        if self.settings.cpu_threads:
            torch.set_num_threads(self.settings.cpu_threads)

        dtype = getattr(torch, self.settings.dtype, None)
        if dtype is None:
            raise EngineError(f"Unbekannter dtype: {self.settings.dtype}")

        common: dict[str, Any] = {"trust_remote_code": self.settings.trust_remote_code}
        if self.settings.revision:
            common["revision"] = self.settings.revision
        if self.settings.cache_dir:
            common["cache_dir"] = self.settings.cache_dir

        # The processor owns the chat template, including the multimodal part
        # shape the checkpoint is trained on; the tokenizer is the fallback for
        # a transformers build that does not offer one.
        self._processor = self._load_processor(transformers, common)
        self._tokenizer = getattr(self._processor, "tokenizer", None) or self._processor
        if not hasattr(self._tokenizer, "apply_chat_template"):
            raise EngineError(
                f"Für {self.settings.model} ist keine Chat-Vorlage verfügbar; "
                "der Enhancer braucht sie, um den Prompt aufzubereiten."
            )

        model_cls, model_name = self._resolve_model_class(transformers)
        kwargs = dict(common, torch_dtype=dtype)
        quant_config = self._quantization_config(transformers)
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config

        log.info("[enhancer] loading weights as %s", model_name)
        model = model_cls.from_pretrained(self.settings.model, **kwargs)
        model.to(self.settings.device)
        model.eval()
        self._model = model

        self._warn_if_context_is_tight()

    def _load_processor(self, transformers, common: dict):
        """The processor, falling back to the plain tokenizer."""

        last_error: Optional[Exception] = None
        for name in ("AutoProcessor", "AutoTokenizer"):
            factory = getattr(transformers, name, None)
            if factory is None:
                continue
            try:
                return factory.from_pretrained(self.settings.model, **common)
            except Exception as exc:  # noqa: BLE001 - the fallback is the point
                last_error = exc
                log.debug("[enhancer] %s did not work: %s", name, exc)
        raise EngineError(
            f"Weder AutoProcessor noch AutoTokenizer konnten "
            f"{self.settings.model} laden: {last_error}"
        )

    def _resolve_model_class(self, transformers) -> tuple[Any, str]:
        """The first model class of :data:`MODEL_CLASSES` that exists.

        Which class can host the checkpoint depends on the transformers
        version, so the candidates are resolved by name instead of imported.
        """

        available = [
            name for name in MODEL_CLASSES if getattr(transformers, name, None) is not None
        ]
        if not available:
            raise EngineError(
                "Diese transformers-Version kennt keine der Klassen "
                f"{', '.join(MODEL_CLASSES)}."
            )
        return getattr(transformers, available[0]), available[0]

    def _quantization_config(self, transformers):
        """Translate ``IMAGEINT_PE_QUANT`` into a quanto configuration."""

        quant = self.settings.quant
        if quant == "none":
            return None
        config_cls = getattr(transformers, "QuantoConfig", None)
        if config_cls is None:
            raise EngineError(
                "Für die Quantisierung wird QuantoConfig aus transformers "
                "gebraucht; diese Version hat es nicht."
            )
        try:
            return config_cls(weights=quant)
        except Exception as exc:  # noqa: BLE001 - reported with its own message
            raise EngineError(
                f"QuantoConfig(weights={quant!r}) ist nicht möglich: {exc}"
            ) from exc

    def close(self) -> None:
        # bf16 weights are tens of gigabytes; dropping the references is the
        # fastest way to hand the memory back on shutdown.
        self._model = None
        self._tokenizer = None
        self._processor = None
        super().close()

    # -- chat template ----------------------------------------------------- #

    def render_chat(self, request: ChatRequest) -> str:
        """The prompt text the checkpoint is asked to continue.

        The part-list content the gateway sends is tried first, because that is
        the shape the checkpoint's template expects; the flattened form and the
        template without the thinking switch are the fallbacks, so an unusual
        checkpoint still produces a prompt instead of a 500.
        """

        template_kwargs: dict[str, Any] = {}
        if request.enable_thinking is not None:
            template_kwargs["enable_thinking"] = bool(request.enable_thinking)

        attempts = (
            (list(request.messages), template_kwargs),
            (flatten_content(request.messages), template_kwargs),
            (list(request.messages), {}),
            (flatten_content(request.messages), {}),
        )
        last_error: Optional[Exception] = None
        for messages, kwargs in attempts:
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - the fallbacks are the point
                last_error = exc
        raise EngineError(
            f"Die Chat-Vorlage von {self.settings.model} konnte nicht angewandt "
            f"werden: {last_error}"
        )

    def _splitter(self, request: ChatRequest) -> ThinkingSplitter:
        # Rendering is a Jinja pass over a few hundred tokens; doing it a second
        # time is cheaper than threading the result through the stream.
        return ThinkingSplitter(
            start_in_thinking=starts_in_thinking(self.render_chat(request))
        )

    # -- generation -------------------------------------------------------- #

    def _generate(
        self, request: ChatRequest, usage: Optional[dict] = None
    ) -> Iterator[str]:
        torch = self._torch()
        from transformers import TextIteratorStreamer

        model = self._model
        if model is None:  # pragma: no cover - guarded by ensure_loaded()
            raise EngineError("Das Enhancer-Modell ist nicht geladen.")

        prompt_text = self.render_chat(request)
        inputs = self._tokenize(prompt_text)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        self._warn_if_too_long(prompt_tokens, request.max_tokens)

        seed = self._pick_seed(request.seed)
        torch.manual_seed(seed)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None
        )
        abort = threading.Event()
        kwargs = dict(inputs, streamer=streamer)
        kwargs.update(self._sampling_kwargs(request))
        criteria = self._abort_criteria(abort)
        if criteria is not None:
            kwargs["stopping_criteria"] = criteria

        failure: list[BaseException] = []

        def run() -> None:
            try:
                with torch.inference_mode():
                    model.generate(**kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                failure.append(exc)
                # The streamer is what unblocks the consumer; without an end
                # marker it would wait for the timeout that never comes.
                streamer.end()

        thread = threading.Thread(target=run, name="imageint-enhancer-generate", daemon=True)
        thread.start()

        pieces: list[str] = []
        try:
            for chunk in streamer:
                if chunk:
                    pieces.append(chunk)
                    yield chunk
        finally:
            # A client that hangs up closes this generator, and the decoding
            # thread has to stop with it: otherwise it would keep all eight
            # cores busy -- and hold the single-flight lock -- for minutes
            # after nobody is listening any more.
            abort.set()
            thread.join()

        if failure:
            raise EngineError(f"Die Generierung ist fehlgeschlagen: {failure[0]}")

        if usage is not None:
            answer = "".join(pieces)
            usage["prompt_tokens"] = prompt_tokens
            usage["completion_tokens"] = self.count_tokens(answer)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    def _tokenize(self, prompt_text: str) -> dict:
        """Tokenise the rendered prompt, as tensors on the target device."""

        torch = self._torch()
        last_error: Optional[Exception] = None
        for source in (self._processor, self._tokenizer):
            if source is None:
                continue
            try:
                encoded = source(text=[prompt_text], return_tensors="pt")
            except Exception as exc:  # noqa: BLE001 - the fallback is the point
                last_error = exc
                continue
            if not isinstance(encoded, dict):
                encoded = dict(encoded)
            # A multimodal processor also returns pixel tensors; there is no
            # image in this request, so anything that is not a tensor or is
            # None is dropped rather than passed to generate().
            clean = {
                key: value
                for key, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
            if "input_ids" not in clean:
                last_error = EngineError("Die Tokenisierung ergab keine input_ids.")
                continue
            return {key: value.to(self.settings.device) for key, value in clean.items()}
        raise EngineError(f"Der Prompt konnte nicht tokenisiert werden: {last_error}")

    def _abort_criteria(self, abort: threading.Event):
        """A stopping criterion that ends the generation when asked to.

        Without it a closed response generator would leave ``model.generate``
        running to completion in the background.
        """

        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
        except ImportError:  # pragma: no cover - transformers is required
            return None

        class _Abort(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ARG002
                return abort.is_set()

        return StoppingCriteriaList([_Abort()])

    def _sampling_kwargs(self, request: ChatRequest) -> dict:
        """Sampling parameters, filtered by what this transformers supports."""

        from transformers import GenerationConfig

        fields = getattr(GenerationConfig, "__dataclass_fields__", {})
        wanted = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "min_p": request.min_p,
            "presence_penalty": request.presence_penalty,
        }
        kwargs: dict[str, Any] = {"max_new_tokens": int(request.max_tokens)}
        for key, value in wanted.items():
            if key in fields:
                kwargs[key] = value
            elif key == "presence_penalty" and value:
                # Older transformers has no presence penalty. Dropping it
                # silently would change the distribution the checkpoint was
                # tuned for, so it is reported once instead.
                if not self._warned_presence_penalty:
                    self._warned_presence_penalty = True
                    log.warning(
                        "[enhancer] Diese transformers-Version kennt keinen "
                        "presence_penalty (%.2f); die Stichproben weichen vom "
                        "dokumentierten Profil ab.",
                        value,
                    )
        if "temperature" not in fields or request.temperature == 0:
            kwargs["do_sample"] = False
        else:
            kwargs["do_sample"] = True
        return kwargs

    def count_tokens(self, text: str) -> int:
        """Token count of a finished answer, for the usage report."""

        tokenizer = self._tokenizer
        if tokenizer is None or not text:
            return 0
        try:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception:  # noqa: BLE001 - usage is cosmetic
            return 0

    # -- diagnostics ------------------------------------------------------- #

    def _warn_if_context_is_tight(self) -> None:
        model_config = getattr(self._model, "config", None)
        limit = int(
            getattr(model_config, "max_position_embeddings", 0)
            or self.settings.context
        )
        budget = self.settings.max_new_tokens
        if limit and budget > limit // 2:
            log.warning(
                "[enhancer] IMAGEINT_PE_MAX_NEW_TOKENS=%d ist mehr als die Hälfte "
                "des Kontexts (%d). Kürzere Budgets antworten deutlich schneller "
                "und laufen nicht in ein abgeschnittenes Denk-Block.",
                budget,
                limit,
            )

    def _warn_if_too_long(self, prompt_tokens: int, max_new_tokens: int) -> None:
        limit = int(
            getattr(getattr(self._model, "config", None), "max_position_embeddings", 0)
            or self.settings.context
        )
        if limit and prompt_tokens + max_new_tokens > limit:
            log.warning(
                "[enhancer] Prompt (%d Token) plus Budget (%d) überschreitet den "
                "Kontext (%d); die Antwort wird abgeschnitten.",
                prompt_tokens,
                max_new_tokens,
                limit,
            )


class StubEngine(BaseEngine):
    """A canned enhancer answer, for tests and documentation."""

    name = "stub"

    def _load(self) -> None:
        """Nothing to load: the answer is canned."""

    def _generate(
        self, request: ChatRequest, usage: Optional[dict] = None
    ) -> Iterator[str]:
        prompt = last_user_text(request.messages)
        seed = self._pick_seed(request.seed)
        delay = max(0.0, self.settings.stub_delay)

        # The checkpoint's template pre-fills the opening marker, so a real
        # answer arrives *inside* the thinking block and closes it. Emitting
        # the same shape keeps the splitter on the path production uses.
        if request.enable_thinking:
            yield "<think>\n"
            if delay:
                time.sleep(delay)
            yield (
                "Der Stub-Enhancer erfindet kein Denken; er gibt nur die Form "
                "der echten Antwort wieder.\n"
            )
            if delay:
                time.sleep(delay)
            yield "</think>\n"

        yield json.dumps(
            {
                "rewritten_prompt": (
                    f"Stub-Aufbereitung für {self.settings.model}: {prompt}"
                ),
                "negative_prompt": "",
                "wh_ratio": "1:1",
            },
            ensure_ascii=False,
        )
        if delay:
            time.sleep(delay)

        if usage is not None:
            usage["prompt_tokens"] = max(1, len(prompt) // 4)
            usage["completion_tokens"] = 24
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        log.info("[enhancer] stub answer for %d characters of prompt (seed %d)", len(prompt), seed)


def build_engine(settings: Settings) -> BaseEngine:
    """Instantiate the engine named by ``IMAGEINT_PE_ENGINE``."""

    if settings.engine == "stub":
        return StubEngine(settings)
    return TransformersEngine(settings)
