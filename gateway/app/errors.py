"""Uniform error model of the ImageInt API.

Every failing request answers with the same JSON envelope::

    {"ok": false, "error": "<code>", "message": "<German text>"}

The codes are stable identifiers the client may branch on; the message is meant
for the administrator and is shown verbatim in the LLMInt admin area.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from fastapi import HTTPException


class ImageIntError(HTTPException):
    """HTTPException carrying a machine-readable ``error`` code."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        headers: Optional[Mapping[str, str]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        detail: dict[str, Any] = {"ok": False, "error": code, "message": message}
        if extra:
            detail.update(extra)
        super().__init__(status_code=status_code, detail=detail, headers=dict(headers or {}))
        self.code = code
        self.message = message


def unauthorized(
    message: str = "Ungültiger oder fehlender Token.", code: str = "unauthorized"
) -> ImageIntError:
    return ImageIntError(401, code, message)


def bad_request(message: str, code: str = "bad_request") -> ImageIntError:
    return ImageIntError(400, code, message)


def payload_too_large(message: str, code: str = "payload_too_large") -> ImageIntError:
    return ImageIntError(413, code, message)


def not_found(message: str, code: str = "not_found") -> ImageIntError:
    return ImageIntError(404, code, message)


def not_configured(message: str, code: str = "not_configured") -> ImageIntError:
    return ImageIntError(503, code, message)


def host_unsupported(message: str) -> ImageIntError:
    """The endpoint cannot run the models on this machine.

    Unlike a missing GPU this is not a tuning hint but a hard stop: the AVX2
    kernels the CPU build uses do not exist here, so no request can succeed.
    """

    return ImageIntError(503, "host_unsupported", message)


def upstream_unavailable(code: str, message: str) -> ImageIntError:
    return ImageIntError(502, code, message)


def upstream_timeout(code: str, message: str) -> ImageIntError:
    return ImageIntError(504, code, message)


def busy(message: str, retry_after: int) -> ImageIntError:
    """All generation slots are taken; the client should retry later."""

    seconds = max(1, int(retry_after))
    return ImageIntError(
        503,
        "busy",
        message,
        headers={"Retry-After": str(seconds)},
        extra={"retry_after": seconds},
    )


def service_loading(component: str, message: str, retry_after: int) -> ImageIntError:
    """The request is valid but a model server is still downloading or loading.

    This is a *transient* state, not a failure: the client should keep its input
    and try again after ``Retry-After`` seconds instead of showing an error.
    """

    seconds = max(1, int(retry_after))
    return ImageIntError(
        503,
        "service_loading",
        message,
        headers={"Retry-After": str(seconds)},
        extra={"component": component, "retry_after": seconds},
    )
