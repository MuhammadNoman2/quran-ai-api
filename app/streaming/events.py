"""WebSocket event protocol.

Every message is JSON with a protocol version, so a client can detect a breaking
change instead of silently misreading a new shape:

    {"v": 1, "event": "word_confirmed", ...}

Some event names below are defined but **not emitted yet**. They belong to
speech segmentation and provisional/confirmed handling, which arrive in Phase 7.
They are declared here so the protocol can be documented as a whole, and a test
asserts the unimplemented ones never appear - a client must not have to guess
whether silence means "not supported" or "nothing happened".
"""

from __future__ import annotations

from enum import Enum
from typing import Any

PROTOCOL_VERSION = 1


class EventType(str, Enum):
    # ── lifecycle ────────────────────────────────────────────────────────────
    SESSION_STARTED = "session_started"
    READY = "ready"
    SESSION_COMPLETED = "session_completed"

    # ── recognition ──────────────────────────────────────────────────────────
    PARTIAL_TRANSCRIPT = "partial_transcript"
    WORD_CONFIRMED = "word_confirmed"
    MISTAKE_DETECTED = "mistake_detected"
    AYAH_PROGRESS = "ayah_progress"
    AYAH_COMPLETED = "ayah_completed"

    # ── diagnostics ──────────────────────────────────────────────────────────
    WARNING = "warning"
    ERROR = "error"

    # ── Phase 7: declared, not yet emitted ───────────────────────────────────
    SPEECH_STARTED = "speech_started"
    SPEECH_STOPPED = "speech_stopped"
    WORD_DETECTED = "word_detected"
    CORRECTION = "correction"


#: Events the server does not produce yet. Tested, so this cannot drift.
NOT_YET_EMITTED: frozenset[EventType] = frozenset(
    {
        EventType.SPEECH_STARTED,
        EventType.SPEECH_STOPPED,
        EventType.WORD_DETECTED,
        EventType.CORRECTION,
    }
)


class ErrorCode(str, Enum):
    INVALID_MESSAGE = "INVALID_MESSAGE"
    INVALID_AUDIO = "INVALID_AUDIO"
    NOT_STARTED = "NOT_STARTED"
    ALREADY_STARTED = "ALREADY_STARTED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    VERSE_NOT_FOUND = "VERSE_NOT_FOUND"
    VERSE_NOT_IDENTIFIED = "VERSE_NOT_IDENTIFIED"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def event(kind: EventType, **fields: Any) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "event": kind.value, **fields}


# ── builders ─────────────────────────────────────────────────────────────────


def session_started(session_id: str, surah: int | None, ayah: int | None) -> dict[str, Any]:
    return event(
        EventType.SESSION_STARTED, session_id=session_id, surah=surah, ayah=ayah
    )


def ready(
    *, sample_rate: int, audio_format: str, expected_words: int | None, engine: dict[str, str]
) -> dict[str, Any]:
    return event(
        EventType.READY,
        sample_rate=sample_rate,
        format=audio_format,
        expected_words=expected_words,
        engine=engine,
    )


def partial_transcript(text: str, *, seconds: float) -> dict[str, Any]:
    """Provisional. The text may change as more audio arrives."""
    return event(
        EventType.PARTIAL_TRANSCRIPT, text=text, audio_seconds=round(seconds, 2), final=False
    )


def word_confirmed(*, index: int, expected: str, spoken: str | None, confidence: float) -> dict[str, Any]:
    return event(
        EventType.WORD_CONFIRMED,
        word_index=index,
        expected=expected,
        spoken=spoken,
        status="correct",
        confidence=round(confidence, 4),
    )


def mistake_detected(
    *,
    index: int | None,
    expected: str | None,
    spoken: str | None,
    category: str,
    confidence: float,
    surah: int,
    ayah: int,
) -> dict[str, Any]:
    """Only ever built from a confirmed, high-confidence finding."""
    return event(
        EventType.MISTAKE_DETECTED,
        surah=surah,
        ayah=ayah,
        word_index=index,
        expected=expected,
        spoken=spoken,
        error_type=category,
        confidence=round(confidence, 4),
    )


def ayah_progress(*, surah: int, ayah: int, words_confirmed: int, words_total: int) -> dict[str, Any]:
    return event(
        EventType.AYAH_PROGRESS,
        surah=surah,
        ayah=ayah,
        words_confirmed=words_confirmed,
        words_total=words_total,
        fraction=round(words_confirmed / words_total, 3) if words_total else 0.0,
    )


def ayah_completed(*, surah: int, ayah: int, score: float, errors: int, observations: int) -> dict[str, Any]:
    return event(
        EventType.AYAH_COMPLETED,
        surah=surah,
        ayah=ayah,
        word_accuracy_score=score,
        errors=errors,
        observations=observations,
    )


def session_completed(*, session_id: str, seconds: float, segments: int) -> dict[str, Any]:
    return event(
        EventType.SESSION_COMPLETED,
        session_id=session_id,
        audio_seconds=round(seconds, 2),
        segments_analysed=segments,
    )


def warning(message: str, **fields: Any) -> dict[str, Any]:
    return event(EventType.WARNING, message=message, **fields)


def error(code: ErrorCode, message: str, *, fatal: bool = False) -> dict[str, Any]:
    """A non-fatal error must not end the session (brief 40)."""
    return event(EventType.ERROR, code=code.value, message=message, fatal=fatal)
