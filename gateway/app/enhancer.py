"""The prompt enhancer of Qwen-Image-2.1, as documented by the model.

Qwen-Image-2.1 ships a dedicated prompt enhancer, ``Qwen-Image-2.1-PE-T2I``: a
fine-tuned Qwen3.5-VL 9B that rewrites a short request into the long, structured
prompt the diffusion model was trained on and picks a canvas ratio for it. The
model card calls it a required part of the pipeline, and this module is a
faithful port of the reference implementation that ships with the weights
(``prompt_rewrite/`` in ``QwenLM/Qwen-Image-2.1``):

* the system prompt is the checkpoint's own ``system_prompt.txt``, vendored into
  :data:`app.config.PROMPT_DIR` and used verbatim;
* the sampling profile is the t2i profile -- temperature 1.0, top_p 0.95,
  top_k 20, min_p 0.0, presence_penalty 1.5, 16256 new tokens. These numbers are
  *not* interchangeable with the edit task's; a wrong presence penalty does not
  fail, it silently changes the distribution that is sampled from;
* the answer is a JSON object at the end of the answer section,
  ``{"rewritten_prompt": ..., "wh_ratio": ...}``, recovered with a balanced-brace
  scan because a greedy ``\\{.*\\}`` breaks on any brace in the prose after it.

Two deliberate additions on top of the reference:

* the model is asked to think (``enable_thinking``), which is how the checkpoint
  is meant to be driven. With ``--reasoning-parser qwen3`` vLLM returns that
  block separately; without it the text still arrives inline and
  :func:`split_thinking` recovers it from ``</think>``.
* a parse failure is reported instead of hidden. ``parse_ok`` is False when the
  enhancer produced no usable JSON, in which case the raw answer is used as the
  prompt so a generation is never lost -- but the caller can see that it
  happened. On a CPU host a silent fallback is expensive to notice.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from . import ratios, vllm
from .config import EnhancerSettings
from .errors import bad_request, not_configured

log = logging.getLogger("imageint.enhancer")

# json_repair fixes answers that are *nearly* valid JSON -- a trailing comma, an
# unescaped quote. Optional: without it the balanced-brace scan still handles
# every well-formed answer, it just gives up a little sooner. Worth having,
# because on a CPU host a retry costs minutes.
try:  # pragma: no cover - import guard
    import json_repair  # type: ignore
except ImportError:  # pragma: no cover - import guard
    json_repair = None


def build_messages(system_prompt: str, user_prompt: str) -> list:
    """The two-turn conversation the enhancer is trained on.

    The text-to-image task takes no source image, so the user turn is plain
    text. The system prompt is passed as a text part rather than a bare string
    because that is the shape the checkpoint's chat template expects.
    """

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]


def build_payload(settings: EnhancerSettings, user_prompt: str) -> dict:
    """Assemble the chat-completion body for one enhancement."""

    return {
        "model": settings.model,
        "messages": build_messages(settings.system_prompt(), user_prompt),
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "min_p": settings.min_p,
        "presence_penalty": settings.presence_penalty,
        "max_tokens": settings.max_new_tokens,
        "seed": settings.seed,
        # The checkpoint is driven with its thinking block enabled; the server
        # splits it off when it was started with `--reasoning-parser qwen3`.
        "chat_template_kwargs": {"enable_thinking": True},
        "stream_options": {"include_usage": True},
    }


def split_thinking(text: str) -> tuple[str, str]:
    """Split a decoded generation into ``(thinking, answer)``.

    The chat template pre-fills ``<think>\\n`` before generation, so the decoded
    text normally starts *inside* the thinking block and closes it with
    ``</think>``. An unterminated block means the token budget ran out before the
    answer, which is a failure the caller should see rather than a silent empty
    prompt.
    """

    if "</think>" in text:
        thinking, _, answer = text.partition("</think>")
        if "<think>" in thinking:
            thinking = thinking.partition("<think>")[2]
        return thinking.strip(), answer.strip()
    if "<think>" in text:
        return text.partition("<think>")[2].strip(), ""
    return "", text.strip()


def _balanced_spans(answer: str) -> list:
    """Every balanced top-level ``{...}`` span in ``answer``, in order.

    Braces inside JSON string literals are skipped, so a rewritten prompt that
    itself contains ``{`` does not break the scan.
    """

    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, char in enumerate(answer):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(answer[start : index + 1])
    return spans


def _as_obj(candidate: str) -> Optional[dict]:
    """Strict JSON first, then json_repair if it is installed."""

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        if json_repair is None:
            return None
        try:
            obj = json_repair.loads(candidate)
        except Exception:  # noqa: BLE001 - the library raises its own types
            return None
    return obj if isinstance(obj, dict) else None


def parse_answer(answer: str) -> dict:
    """Parse the answer section into the task's declared fields.

    Returns ``{"positive_prompt", "negative_prompt", "wh_ratio", "parse_ok"}``.
    On failure ``positive_prompt`` falls back to the raw answer so the model's
    output is never lost, and ``parse_ok`` is False -- the only way to tell a
    fallback from a clean parse.
    """

    answer = (answer or "").strip()
    # The answer object is emitted at the end, so scan candidates last-first.
    for candidate in reversed(_balanced_spans(answer)):
        obj = _as_obj(candidate)
        if obj is None:
            continue
        # Some training runs mis-typed the key as `rewrited_prompt`; accept both.
        rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
        if not isinstance(rewritten, str) or not rewritten.strip():
            continue
        return {
            "positive_prompt": rewritten.strip(),
            "negative_prompt": str(obj.get("negative_prompt") or "").strip(),
            "wh_ratio": str(obj.get("wh_ratio") or "").strip(),
            "parse_ok": True,
        }

    return {
        "positive_prompt": answer,
        "negative_prompt": "",
        "wh_ratio": "",
        "parse_ok": False,
    }


async def enhance(
    settings: EnhancerSettings,
    prompt: str,
    *,
    max_pixels: int = 0,
    timeout: Optional[int] = None,
) -> dict:
    """Rewrite one prompt and pick the canvas it should be rendered on.

    Returns ``{"prompt", "wh_ratio", "width", "height", "thinking", "raw",
    "parse_ok", "model", "latency_ms", "usage"}``, where ``prompt`` is the
    enhanced prompt and ``raw`` the untouched answer of the enhancer.
    """

    prompt = (prompt or "").strip()
    if not prompt:
        raise bad_request("Es wurde kein Prompt übergeben.", "empty_prompt")
    if not settings.enabled:
        raise not_configured(
            "Der Prompt-Enhancer ist per IMAGEINT_PE_ENABLED=0 abgeschaltet. "
            "Ohne ihn ist der Prompt nicht dokumentationskonform aufzubereiten."
        )

    result = await vllm.chat_completion(settings, build_payload(settings, prompt), timeout=timeout)

    thinking, answer = split_thinking(result["text"])
    if not thinking and result["reasoning"]:
        # Served with `--reasoning-parser qwen3`, which removes the block from
        # `content` and hands it over in its own field.
        thinking = result["reasoning"].strip()

    parsed = parse_answer(answer)
    if not parsed["parse_ok"]:
        log.warning(
            "Der Prompt-Enhancer hat kein auswertbares JSON geliefert; die Antwort "
            "wird unverändert als Prompt verwendet."
        )

    width, height = ratios.size_for(parsed["wh_ratio"], max_pixels)

    return {
        "prompt": parsed["positive_prompt"],
        "negative_prompt": parsed["negative_prompt"],
        "wh_ratio": ratios.normalize(parsed["wh_ratio"]),
        "width": width,
        "height": height,
        "thinking": thinking,
        "raw": answer,
        "parse_ok": parsed["parse_ok"],
        "model": result["model"],
        "latency_ms": result["latency_ms"],
        "usage": result["usage"],
    }


def describe_ratio(ratio: str) -> dict:
    """Report how one ratio is interpreted, for the config endpoint."""

    width, height = ratios.size_for(ratio)
    return {
        "wh_ratio": ratios.normalize(ratio) or "1:1",
        "width": width,
        "height": height,
        "documented": ratios.normalize(ratio) in ratios.DOCUMENTED,
    }


def profile() -> dict[str, Any]:
    """The effective enhancer profile, reported by ``GET /v1/config``."""

    from .config import get_settings

    settings = get_settings().enhancer
    return {
        "url": settings.url,
        "model": settings.model,
        "enabled": settings.enabled,
        "system_prompt_chars": len(settings.system_prompt()),
        "sampling": {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            "min_p": settings.min_p,
            "presence_penalty": settings.presence_penalty,
            "max_tokens": settings.max_new_tokens,
            "seed": settings.seed,
        },
        "ratios": {name: list(size) for name, size in ratios.DOCUMENTED.items()},
    }
