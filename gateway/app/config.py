"""Environment-driven configuration of the ImageInt gateway.

Every value has a working default so the container starts without an .env file.
See ``.env.example`` in the repository root for the documented variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import ratios

#: Where the two vLLM servers live on the compose network. Both are internal
#: services; only the gateway publishes a port.
DEFAULT_ENHANCER_URL = "http://enhancer:8000"
DEFAULT_IMAGE_URL = "http://image:8000"

#: The prompt enhancer of Qwen-Image-2.1 is a fine-tuned Qwen3.5-VL 9B. Both
#: sides read this from the same variable: the enhancer container passes it to
#: ``from_pretrained``, the gateway sends it as the ``model`` field of the
#: request. It is therefore the full repository id, not an alias.
DEFAULT_ENHANCER_MODEL = "Qwen/Qwen-Image-2.1-PE-T2I"

#: The image model of Qwen-Image-2.1, served by the ImageInt image server
#: (diffusers + transformers on CPU).
DEFAULT_IMAGE_MODEL = "Qwen/Qwen-Image-2.1"

LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")

#: The two routes an image server can expose, and their URL paths. ``images`` is
#: the OpenAI Images API and the only route the ImageInt image server implements;
#: ``chat`` is kept so a deployment that fronts a chat-completions image model
#: still works.
IMAGE_ROUTES = ("images", "chat")
IMAGE_ROUTE_PATHS = {
    "images": "/v1/images/generations",
    "chat": "/v1/chat/completions",
}

#: Directory the bundled prompt-enhancer system prompt is read from. It is
#: copied from the checkpoint (``system_prompt.txt`` of Qwen-Image-2.1-PE-T2I),
#: because the answer contract is part of what the weights were trained on.
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

#: The production sampling settings of the t2i prompt enhancer. They are not
#: interchangeable with the edit task: a wrong presence penalty does not fail,
#: it quietly changes the distribution that is sampled from.
ENHANCER_TEMPERATURE = 1.0
ENHANCER_TOP_P = 0.95
ENHANCER_TOP_K = 20
ENHANCER_MIN_P = 0.0
ENHANCER_PRESENCE_PENALTY = 1.5
ENHANCER_MAX_NEW_TOKENS = 16256

#: Denoising steps and guidance of the image model. 40 steps is what the
#: Qwen-Image-2.1 model card documents. The guidance default is 1.0 because the
#: model card and the pipeline docstring both state that Qwen-Image-2.1 is meant
#: to be sampled *without* guidance; a negative prompt only has an effect above
#: 1.0, which is why ``IMAGEINT_IMAGE_NEGATIVE_PROMPT`` is normally empty.
IMAGE_STEPS = 40
IMAGE_TRUE_CFG_SCALE = 1.0

#: The model card's native canvas is 2048x2048; the ratio table below maps the
#: enhancer's ``wh_ratio`` onto the documented sizes.
DEFAULT_IMAGE_SIZE = (2048, 2048)


def _text(name: str, default: str = "") -> str:
    value = (os.getenv(name) or "").strip()
    return value or default


def _number(name: str, default: int, low: int, high: int) -> int:
    raw = _text(name)
    if raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(low, min(high, value))


def _decimal(name: str, default: float, low: float, high: float) -> float:
    raw = _text(name)
    if raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(low, min(high, value))


def _flag(name: str, default: bool) -> bool:
    raw = _text(name).lower()
    if raw == "":
        return default
    return raw in ("1", "true", "yes", "on", "ja")


def _image_route(raw: str) -> str:
    value = (raw or "").strip().lower()
    return value if value in IMAGE_ROUTES else "images"


@dataclass(frozen=True)
class EnhancerSettings:
    """Connection to the vLLM server that hosts the prompt enhancer."""

    url: str = ""
    token: str = ""
    timeout: int = 900
    model: str = DEFAULT_ENHANCER_MODEL
    enabled: bool = True
    system_prompt_file: str = ""
    temperature: float = ENHANCER_TEMPERATURE
    top_p: float = ENHANCER_TOP_P
    top_k: int = ENHANCER_TOP_K
    min_p: float = ENHANCER_MIN_P
    presence_penalty: float = ENHANCER_PRESENCE_PENALTY
    max_new_tokens: int = ENHANCER_MAX_NEW_TOKENS
    seed: int = 42

    @property
    def configured(self) -> bool:
        return self.url != ""

    def system_prompt(self) -> str:
        """The enhancer's system prompt, from the configured file or the bundle.

        Preferring the file that ships with the checkpoint is deliberate: the
        answer contract is what the weights were trained on, so a prompt that
        travels with the weights cannot drift out of sync with them.
        """

        if self.system_prompt_file:
            path = Path(self.system_prompt_file)
            if not path.is_file():
                raise FileNotFoundError(
                    f"IMAGEINT_PE_SYSTEM_PROMPT_FILE {self.system_prompt_file!r} is not a file"
                )
            return path.read_text(encoding="utf-8").strip()
        bundled = PROMPT_DIR / "pe_t2i_system_prompt.txt"
        return bundled.read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class ImageSettings:
    """Connection to the image server that hosts Qwen-Image-2.1."""

    url: str = ""
    token: str = ""
    timeout: int = 1800
    model: str = DEFAULT_IMAGE_MODEL
    #: Reported in ``GET /v1/config`` only -- the weight format is a property of
    #: the image server, which defaults to bf16 (``none``).
    quant: str = "none"
    #: Which route renders the picture. ``images`` is the dedicated Images API
    #: (``/v1/images/generations``), which is what the ImageInt image server
    #: implements and therefore the default. ``chat`` is
    #: ``/v1/chat/completions`` with the image parameters in ``extra_body``, for
    #: a deployment that fronts a chat-completions image model instead. Both
    #: answer in a shape this client understands, but only one of them exists on
    #: a given server, so the route is configuration rather than a guess.
    route: str = "images"
    steps: int = IMAGE_STEPS
    true_cfg_scale: float = IMAGE_TRUE_CFG_SCALE
    negative_prompt: str = ""
    max_pixels: int = 2048 * 2048

    @property
    def configured(self) -> bool:
        return self.url != ""


@dataclass(frozen=True)
class Settings:
    """Complete gateway configuration."""

    token: str = ""
    public_url: str = ""
    max_prompt_chars: int = 4000
    log_level: str = "INFO"
    health_cache_seconds: int = 5
    loading_retry_after: int = 15
    starting_grace_seconds: int = 900
    #: How long ``POST /v1/images/generations`` waits for a finished image
    #: before it answers 202 with a job id instead. Long generations on CPU are
    #: the normal case, so the async path is the documented default.
    sync_timeout_seconds: int = 120
    #: Concurrent generations. One keeps the CPU model from thrashing; raise it
    #: together with the vLLM ``--max-num-seqs`` when the host has headroom.
    max_concurrent_jobs: int = 1
    #: How many further requests may wait for a free slot before the endpoint
    #: answers 503 ``busy``. Waiting is the friendlier answer on a CPU host --
    #: the queue is what makes a second chat request succeed instead of fail --
    #: but it has to be bounded, so a flood still gets a clear refusal.
    max_queued_jobs: int = 2
    #: How many finished jobs stay retrievable, and for how long.
    max_jobs: int = 200
    job_retention_seconds: int = 24 * 3600
    storage_dir: str = "/data/images"
    enhancer: EnhancerSettings = field(default_factory=EnhancerSettings)
    image: ImageSettings = field(default_factory=ImageSettings)

    @property
    def auth_required(self) -> bool:
        return self.token != ""


def load_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment."""

    log_level = _text("IMAGEINT_LOG_LEVEL", "INFO").upper()
    if log_level not in LOG_LEVELS:
        log_level = "INFO"

    return Settings(
        token=_text("IMAGEINT_TOKEN"),
        public_url=_text("IMAGEINT_PUBLIC_URL").rstrip("/"),
        max_prompt_chars=_number("IMAGEINT_MAX_PROMPT_CHARS", 4000, 16, 200000),
        log_level=log_level,
        health_cache_seconds=_number("IMAGEINT_HEALTH_CACHE_SECONDS", 5, 0, 300),
        loading_retry_after=_number("IMAGEINT_LOADING_RETRY_AFTER", 15, 1, 3600),
        starting_grace_seconds=_number("IMAGEINT_STARTING_GRACE_SECONDS", 900, 0, 86400),
        sync_timeout_seconds=_number("IMAGEINT_SYNC_TIMEOUT", 120, 0, 7200),
        max_concurrent_jobs=_number("IMAGEINT_MAX_CONCURRENT_JOBS", 1, 1, 64),
        max_queued_jobs=_number("IMAGEINT_MAX_QUEUED_JOBS", 2, 0, 1000),
        max_jobs=_number("IMAGEINT_MAX_JOBS", 200, 1, 100000),
        job_retention_seconds=_number(
            "IMAGEINT_JOB_RETENTION_SECONDS", 24 * 3600, 60, 30 * 24 * 3600
        ),
        storage_dir=_text("IMAGEINT_STORAGE_DIR", "/data/images"),
        enhancer=EnhancerSettings(
            url=_text("IMAGEINT_PE_URL", DEFAULT_ENHANCER_URL).rstrip("/"),
            token=_text("IMAGEINT_PE_TOKEN"),
            timeout=_number("IMAGEINT_PE_TIMEOUT", 900, 10, 7200),
            model=_text("IMAGEINT_PE_MODEL", DEFAULT_ENHANCER_MODEL),
            enabled=_flag("IMAGEINT_PE_ENABLED", True),
            system_prompt_file=_text("IMAGEINT_PE_SYSTEM_PROMPT_FILE"),
            temperature=_decimal(
                "IMAGEINT_PE_TEMPERATURE", ENHANCER_TEMPERATURE, 0.0, 2.0
            ),
            top_p=_decimal("IMAGEINT_PE_TOP_P", ENHANCER_TOP_P, 0.0, 1.0),
            top_k=_number("IMAGEINT_PE_TOP_K", ENHANCER_TOP_K, 0, 1000),
            min_p=_decimal("IMAGEINT_PE_MIN_P", ENHANCER_MIN_P, 0.0, 1.0),
            presence_penalty=_decimal(
                "IMAGEINT_PE_PRESENCE_PENALTY", ENHANCER_PRESENCE_PENALTY, -2.0, 2.0
            ),
            max_new_tokens=_number(
                "IMAGEINT_PE_MAX_NEW_TOKENS", ENHANCER_MAX_NEW_TOKENS, 256, 131072
            ),
            seed=_number("IMAGEINT_PE_SEED", 42, 0, 2**31 - 1),
        ),
        image=ImageSettings(
            url=_text("IMAGEINT_IMAGE_URL", DEFAULT_IMAGE_URL).rstrip("/"),
            token=_text("IMAGEINT_IMAGE_TOKEN"),
            timeout=_number("IMAGEINT_IMAGE_TIMEOUT", 1800, 10, 14400),
            model=_text("IMAGEINT_IMAGE_MODEL", DEFAULT_IMAGE_MODEL),
            quant=_text("IMAGEINT_IMAGE_QUANT", "none"),
            route=_image_route(_text("IMAGEINT_IMAGE_ROUTE", "images")),
            steps=_number("IMAGEINT_IMAGE_STEPS", IMAGE_STEPS, 1, 200),
            true_cfg_scale=_decimal(
                "IMAGEINT_IMAGE_TRUE_CFG_SCALE", IMAGE_TRUE_CFG_SCALE, 0.0, 20.0
            ),
            negative_prompt=_text("IMAGEINT_IMAGE_NEGATIVE_PROMPT"),
            max_pixels=_number(
                "IMAGEINT_IMAGE_MAX_PIXELS",
                ratios.MAX_DOCUMENTED_PIXELS,
                65536,
                16 * 1024 * 1024,
            ),
        ),
    )


_cache: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the process-wide settings, reading the environment once."""

    global _cache
    if _cache is None:
        _cache = load_settings()
    return _cache


def reset_settings() -> None:
    """Drop the cached settings; used by tests and after environment changes."""

    global _cache
    _cache = None
