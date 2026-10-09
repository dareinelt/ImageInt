"""Configuration of the prompt-enhancer server.

Like the gateway and the image server, this process is configured entirely
through the environment, so one ``.env`` file describes a whole deployment.
The names are the same ones the gateway reads: where a value has to be
identical on both sides (``IMAGEINT_PE_MODEL``, ``IMAGEINT_PE_TOKEN``) it is
read from the same variable, and the sampling defaults carry the same names
and the same numbers as the ones in ``gateway/app/config.py``. The gateway
sends every sampling parameter with each request anyway; they exist here so the
server behaves identically when it is driven directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

#: The checkpoint. ``Qwen-Image-2.1-PE-T2I`` is the text-to-image prompt
#: enhancer the Qwen-Image-2.1 model card documents: a fine-tuned Qwen3.5-VL 9B
#: that rewrites a short request into the long structured prompt the diffusion
#: model was trained on, and that also picks the canvas ratio.
DEFAULT_MODEL = "Qwen/Qwen-Image-2.1-PE-T2I"

#: Weight formats. bfloat16 is what the model card documents and therefore the
#: default. float32 is the escape hatch for a CPU without bf16 support; it
#: doubles the resident size to roughly 38 GB and is not the reference
#: configuration. float16 is accepted because transformers offers it, but is a
#: poor fit for a CPU decoder.
DTYPES = ("bfloat16", "float32", "float16")

#: Devices the model can be placed on. ``cpu`` is the requirement of this
#: service, ``cuda`` is the prepared NVIDIA path, ``mps`` exists so the server
#: can be run natively on an Apple Silicon development machine.
DEVICES = ("cpu", "cuda", "mps")

#: Quantisation of the weights. ``none`` keeps the dtype above and is the
#: documented production setting; ``int8`` and ``int4`` are the small-footprint
#: modes for a test machine, applied through optimum-quanto, which is the
#: quantisation backend that works on a CPU.
QUANTS = ("none", "int8", "int4")

#: Generation backends. ``transformers`` runs the real checkpoint. ``stub``
#: answers with a canned enhanced prompt and exists so the API, the gateway and
#: the documentation can be exercised without a 19 GB download.
ENGINES = ("transformers", "stub")

#: The t2i sampling profile of the checkpoint, verbatim from the reference
#: implementation that ships with the weights. These numbers are *not*
#: interchangeable with the edit task's: a wrong presence penalty does not
#: fail, it silently changes the distribution that is sampled from.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 20
DEFAULT_MIN_P = 0.0
DEFAULT_PRESENCE_PENALTY = 1.5

#: The enhancer emits a long structured prompt, so the answer budget is large.
DEFAULT_MAX_NEW_TOKENS = 16256

#: Context the checkpoint was trained with. transformers takes the real limit
#: from the checkpoint's own config; this value is only used to warn when
#: prompt plus answer budget cannot fit, which on a CPU host is worth knowing
#: before a request runs for an hour.
DEFAULT_CONTEXT = 32768

#: Upper length of an accepted prompt, in characters.
DEFAULT_MAX_PROMPT_CHARS = 4000


def _text(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


def _number(name: str, default: int, low: int, high: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(float(raw))
    except ValueError:
        return default
    return max(low, min(high, value))


def _decimal(name: str, default: float, low: float, high: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(low, min(high, value))


def _flag(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off", "nein")


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = (os.environ.get(name) or "").strip().lower()
    return value if value in allowed else default


@dataclass(frozen=True)
class Settings:
    """Everything the server reads from the environment."""

    model: str = DEFAULT_MODEL
    #: Optional commit, tag or branch of the checkpoint repository.
    revision: str = ""
    dtype: str = "bfloat16"
    device: str = "cpu"
    quant: str = "none"
    engine: str = "transformers"
    #: Shared secret. When set, every request except ``/health`` must carry it as
    #: ``X-Auth-Token`` or ``Authorization: Bearer``.
    token: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    #: Load the weights during startup instead of on the first request. Keeping
    #: this true is what makes the gateway's loading contract meaningful: the
    #: process is up, ``/health`` answers 503, and the model becomes available
    #: minutes later without a request ever timing out on a cold cache.
    preload: bool = True
    #: Torch intra-op threads. 0 leaves torch's own default, which is one thread
    #: per core -- the right value on the 8-core reference machine.
    cpu_threads: int = 0
    #: Sampling defaults, used only when a request leaves the field out. The
    #: gateway sends all of them, so these mirror the gateway's own defaults.
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    min_p: float = DEFAULT_MIN_P
    presence_penalty: float = DEFAULT_PRESENCE_PENALTY
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    seed: int = 42
    #: Drive the checkpoint with its thinking block enabled, which is how the
    #: reference implementation calls it. The gateway asks for the same thing
    #: per request through ``chat_template_kwargs``.
    enable_thinking: bool = True
    #: Context the checkpoint was trained with. Informational: used for a
    #: startup warning, not enforced.
    context: int = DEFAULT_CONTEXT
    #: Upper length of an accepted prompt.
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS
    #: The checkpoint needs ``trust_remote_code`` for its own modelling code.
    trust_remote_code: bool = True
    #: Artificial delay of the stub engine, in seconds, so a slow enhancement can
    #: be demonstrated without a model.
    stub_delay: float = 0.0
    #: Path of the model cache. ``HF_HOME`` is the usual way to set this; the
    #: variable exists so the container can be pointed at a mounted volume.
    cache_dir: str = ""

    @property
    def auth_required(self) -> bool:
        return self.token != ""

    @property
    def torch_dtype(self) -> str:
        """The dtype actually handed to ``from_pretrained``."""

        return self.dtype


def load_settings() -> Settings:
    """Build a :class:`Settings` from the current environment."""

    return Settings(
        model=_text("IMAGEINT_PE_MODEL", DEFAULT_MODEL),
        revision=_text("IMAGEINT_PE_REVISION", ""),
        dtype=_choice("IMAGEINT_PE_DTYPE", "bfloat16", DTYPES),
        device=_choice("IMAGEINT_PE_DEVICE", "cpu", DEVICES),
        quant=_choice("IMAGEINT_PE_QUANT", "none", QUANTS),
        engine=_choice("IMAGEINT_PE_ENGINE", "transformers", ENGINES),
        token=_text("IMAGEINT_PE_TOKEN", ""),
        host=_text("IMAGEINT_PE_HOST", "0.0.0.0"),
        port=_number("IMAGEINT_PE_PORT", 8000, 1, 65535),
        preload=_flag("IMAGEINT_PE_PRELOAD", True),
        cpu_threads=_number("IMAGEINT_PE_CPU_THREADS", 0, 0, 1024),
        temperature=_decimal("IMAGEINT_PE_TEMPERATURE", DEFAULT_TEMPERATURE, 0.0, 2.0),
        top_p=_decimal("IMAGEINT_PE_TOP_P", DEFAULT_TOP_P, 0.0, 1.0),
        top_k=_number("IMAGEINT_PE_TOP_K", DEFAULT_TOP_K, 0, 1000),
        min_p=_decimal("IMAGEINT_PE_MIN_P", DEFAULT_MIN_P, 0.0, 1.0),
        presence_penalty=_decimal(
            "IMAGEINT_PE_PRESENCE_PENALTY", DEFAULT_PRESENCE_PENALTY, -2.0, 2.0
        ),
        max_new_tokens=_number(
            "IMAGEINT_PE_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS, 256, 131072
        ),
        seed=_number("IMAGEINT_PE_SEED", 42, 0, 2**31 - 1),
        enable_thinking=_flag("IMAGEINT_PE_ENABLE_THINKING", True),
        context=_number("IMAGEINT_PE_CONTEXT", DEFAULT_CONTEXT, 1024, 10**7),
        max_prompt_chars=_number(
            "IMAGEINT_PE_MAX_PROMPT_CHARS", DEFAULT_MAX_PROMPT_CHARS, 1, 100000
        ),
        trust_remote_code=_flag("IMAGEINT_PE_TRUST_REMOTE_CODE", True),
        stub_delay=_decimal("IMAGEINT_PE_STUB_DELAY", 0.0, 0.0, 3600.0),
        cache_dir=_text("IMAGEINT_PE_CACHE_DIR", ""),
    )
