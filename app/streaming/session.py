"""Recitation sessions.

A session is created over REST and then, in Phase 6, streamed over WebSocket.
Storage is in-process: sessions are ephemeral, expire, and hold no audio. That
is a deliberate choice over SQLite - there is nothing here worth persisting, and
persisting session records tied to voice input is a privacy liability rather
than a feature.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from app.asr.base import Tier


@dataclass
class Session:
    id: str
    surah: int | None
    ayah: int | None
    tier: Tier
    created_at: float
    expires_at: float
    sample_rate: int = 16_000
    audio_format: str = "pcm_s16le"

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def seconds_remaining(self) -> float:
        return max(0.0, self.expires_at - time.time())


class SessionStore:
    """In-process session registry with a hard capacity limit.

    The capacity limit is the mechanism behind MAX_CONCURRENT_SESSIONS: on the
    target VPS three concurrent streaming sessions is what the CPU supports, and
    a fourth must be refused with a clear error rather than degrading everyone.
    """

    def __init__(self, capacity: int, ttl_seconds: float) -> None:
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._sessions: dict[str, Session] = {}

    def _evict_expired(self) -> None:
        for key in [k for k, s in self._sessions.items() if s.is_expired]:
            del self._sessions[key]

    @property
    def active_count(self) -> int:
        self._evict_expired()
        return len(self._sessions)

    @property
    def capacity(self) -> int:
        return self._capacity

    def has_room(self) -> bool:
        return self.active_count < self._capacity

    def create(
        self,
        *,
        surah: int | None = None,
        ayah: int | None = None,
        tier: Tier = Tier.COMMITTED,
        sample_rate: int = 16_000,
        audio_format: str = "pcm_s16le",
    ) -> Session:
        self._evict_expired()
        if len(self._sessions) >= self._capacity:
            raise CapacityExceeded(
                f"all {self._capacity} concurrent sessions are in use"
            )
        now = time.time()
        session = Session(
            id=str(uuid.uuid4()),
            surah=surah,
            ayah=ayah,
            tier=tier,
            created_at=now,
            expires_at=now + self._ttl,
            sample_rate=sample_rate,
            audio_format=audio_format,
        )
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session and session.is_expired:
            del self._sessions[session_id]
            return None
        return session

    def close(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def clear(self) -> None:
        self._sessions.clear()


class CapacityExceeded(RuntimeError):
    pass
