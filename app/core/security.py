"""Authentication and rate limiting.

Both are real seams we will need, kept deliberately thin. Authentication is
**disabled** when no API keys are configured, which is right for local
development and wrong for production - `/health` reports which mode is active so
the mistake is visible rather than silent.

The rate limiter is an in-process fixed-window counter. That is honest for a
single-process deployment (which is what the target VPS runs) and would need
replacing with a shared store behind multiple workers. Said here rather than
discovered later.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import status

from app.core.errors import APIError, ErrorCode


class AuthProvider:
    def __init__(self, keys: set[str]) -> None:
        self._keys = keys

    @property
    def enabled(self) -> bool:
        return bool(self._keys)

    def authenticate(self, api_key: str | None) -> str:
        """Return the caller's identity, or 'anonymous' when auth is off."""
        if not self.enabled:
            return "anonymous"
        if not api_key or api_key not in self._keys:
            raise APIError(
                ErrorCode.UNAUTHORIZED,
                "missing or invalid API key",
                status.HTTP_401_UNAUTHORIZED,
            )
        return api_key


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self._limit = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    @property
    def enabled(self) -> bool:
        return self._limit > 0

    def check(self, identity: str, now: float | None = None) -> None:
        if not self.enabled:
            return
        now = now if now is not None else time.monotonic()
        window = self._hits[identity]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= self._limit:
            raise APIError(
                ErrorCode.RATE_LIMITED,
                f"rate limit of {self._limit} requests/minute exceeded",
                status.HTTP_429_TOO_MANY_REQUESTS,
            )
        window.append(now)
