"""Canvas sizes for the aspect ratios the Qwen-Image-2.1 model card documents.

The prompt enhancer answers with a ``wh_ratio`` string rather than with pixels
(``"3:2"``), so the gateway has to turn that back into a canvas. The seven ratios
below are the ones the model card recommends and are used verbatim.

The enhancer's own system prompt names a few more (``1:2``, ``21:9``, ``4:5`` …).
For those, and for any ratio a client sends directly, the size is derived from
the native 2K area with the same aspect ratio, snapped to a multiple of 64 and
clamped to the configured pixel budget, so an unusual ratio degrades into a
valid canvas instead of a rejected request.
"""
from __future__ import annotations

import math
import re
from typing import Optional, Tuple

#: Recommended sizes from the Qwen-Image-2.1 model card, used unchanged.
DOCUMENTED: dict[str, Tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

#: Native canvas of the model, used as the target area of a derived size.
NATIVE_PIXELS = 2048 * 2048

#: The largest area the model card documents (2400x1792). The configured pixel
#: budget defaults to this, so none of the documented canvases is silently
#: shrunk -- an explicit lower ``IMAGEINT_IMAGE_MAX_PIXELS`` still applies.
MAX_DOCUMENTED_PIXELS = max(width * height for width, height in DOCUMENTED.values())

#: The square every unknown ratio degrades into.
DEFAULT_RATIO = "1:1"
DEFAULT_SIZE: Tuple[int, int] = DOCUMENTED[DEFAULT_RATIO]

#: Diffusion models need dimensions divisible by the VAE stride; 64 is a safe
#: and conventional multiple, and 256 the smallest canvas worth rendering.
MULTIPLE = 64
MIN_SIDE = 256

_RATIO_RE = re.compile(r"^\s*(\d{1,5})\s*[:/x×*]\s*(\d{1,5})\s*$", re.IGNORECASE)
#: ``1024x768``, ``1024X768``, ``1024×768``, ``1024*768``, ``1024 768``.
_SIZE_RE = re.compile(r"^\s*(\d{1,5})\s*(?:[x×*]|\s)\s*(\d{1,5})\s*$", re.IGNORECASE)


def normalize(raw: str) -> str:
    """Reduce a ratio string to its canonical ``"a:b"`` form.

    Accepts ``"3:2"``, ``"3/2"``, ``"3x2"`` and ``" 3 : 2 "``; returns ``""``
    when the value is not a ratio at all.
    """

    match = _RATIO_RE.match(raw or "")
    if not match:
        return ""
    left, right = int(match.group(1)), int(match.group(2))
    if left <= 0 or right <= 0:
        return ""
    divisor = math.gcd(left, right)
    return f"{left // divisor}:{right // divisor}"


def _snap(value: float) -> int:
    return max(MIN_SIDE, int(round(value / MULTIPLE)) * MULTIPLE)


def size_for(ratio: str, max_pixels: int = 0) -> Tuple[int, int]:
    """Canvas for one ratio, falling back to the native square when unknown."""

    canonical = normalize(ratio)
    if canonical in DOCUMENTED:
        width, height = DOCUMENTED[canonical]
        return _clamp(width, height, max_pixels)

    if not canonical:
        return _clamp(*DEFAULT_SIZE, max_pixels)

    left, right = (int(part) for part in canonical.split(":"))
    # Keep the native area and solve for the sides of this aspect ratio.
    width = math.sqrt(NATIVE_PIXELS * left / right)
    height = math.sqrt(NATIVE_PIXELS * right / left)
    return _clamp(_snap(width), _snap(height), max_pixels)


def _clamp(width: int, height: int, max_pixels: int) -> Tuple[int, int]:
    """Scale a canvas down until it fits the pixel budget, keeping the ratio."""

    if max_pixels <= 0 or width * height <= max_pixels:
        return width, height
    factor = math.sqrt(max_pixels / float(width * height))
    # Rounding up to the next stride can overshoot the budget again, so shrink
    # the scale until the snapped canvas fits.
    for _ in range(64):
        scaled = (_snap(width * factor), _snap(height * factor))
        if scaled[0] * scaled[1] <= max_pixels:
            return scaled
        factor *= 0.97
    return _snap(width * factor), _snap(height * factor)


def parse_size(raw: str, max_pixels: int = 0) -> Optional[Tuple[int, int]]:
    """Read an explicit ``"1024x1024"`` canvas, or ``None`` when unparseable.

    A ratio (``"3:2"``) is deliberately *not* a size and yields ``None`` so the
    caller can fall back to :func:`size_for`; the two spellings share the ``x``
    separator and would otherwise be ambiguous.
    """

    text = (raw or "").strip()
    if not text or ":" in text or "/" in text:
        return None
    match = _SIZE_RE.match(text)
    if not match:
        return None
    left, right = int(match.group(1)), int(match.group(2))
    if left <= 0 or right <= 0:
        return None
    return _clamp(_snap(left), _snap(right), max_pixels)
