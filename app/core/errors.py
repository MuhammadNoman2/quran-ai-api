"""Structured API errors.

Clients get a stable `code` they can branch on and a human-readable `message`.
Internal detail - model names, stack traces, file paths - never crosses the
boundary.
"""

from __future__ import annotations

from enum import Enum

from fastapi import Request, status
from fastapi.responses import JSONResponse


class ErrorCode(str, Enum):
    INVALID_AUDIO = "INVALID_AUDIO"
    AUDIO_TOO_LARGE = "AUDIO_TOO_LARGE"
    AUDIO_TOO_LONG = "AUDIO_TOO_LONG"
    VERSE_NOT_FOUND = "VERSE_NOT_FOUND"
    VERSE_NOT_IDENTIFIED = "VERSE_NOT_IDENTIFIED"
    UNAUTHORIZED = "UNAUTHORIZED"
    RATE_LIMITED = "RATE_LIMITED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class APIError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        http_status: int = status.HTTP_400_BAD_REQUEST,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


def error_body(code: ErrorCode, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code.value, "message": message}}


async def api_error_handler(_: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.http_status, content=error_body(exc.code, exc.message)
    )


async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Never leak internals. The details are logged, not returned."""
    import logging

    logging.getLogger(__name__).exception("unhandled error", exc_info=exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_body(ErrorCode.INTERNAL_ERROR, "internal server error"),
    )
