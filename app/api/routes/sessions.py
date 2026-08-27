"""Session lifecycle. The WebSocket stream attaches to these in Phase 6."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from app.api.deps import get_session_store, get_settings, require_caller
from app.asr.base import Tier
from app.core.config import Settings
from app.core.errors import APIError, ErrorCode
from app.streaming.session import CapacityExceeded, SessionStore

router = APIRouter(
    prefix="/sessions", tags=["sessions"], dependencies=[Depends(require_caller)]
)

SUPPORTED_SAMPLE_RATES = (8_000, 16_000, 22_050, 24_000, 44_100, 48_000)
SUPPORTED_FORMATS = ("pcm_s16le",)


class CreateSession(BaseModel):
    surah: int | None = Field(None, ge=1, le=114)
    ayah: int | None = Field(None, ge=1)
    tier: Tier = Tier.COMMITTED
    sample_rate: int = 16_000
    format: str = "pcm_s16le"


class SessionResponse(BaseModel):
    session_id: str
    surah: int | None
    ayah: int | None
    tier: Tier
    sample_rate: int
    format: str
    expires_in_seconds: float
    websocket_url: str = Field(..., description="Where to stream audio (Phase 6)")
    active_sessions: int
    capacity: int


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Open a recitation session",
)
def create_session(
    body: CreateSession,
    store: SessionStore = Depends(get_session_store),
    config: Settings = Depends(get_settings),
) -> SessionResponse:
    """Reserve one of the server's concurrent session slots.

    Capacity is finite and deliberately small: on a CPU deployment each live
    session costs real inference time, so the server refuses a session it cannot
    serve properly rather than degrading every session already running.
    """
    if body.sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise APIError(
            ErrorCode.VALIDATION_ERROR,
            f"unsupported sample rate {body.sample_rate}; "
            f"supported: {list(SUPPORTED_SAMPLE_RATES)}",
        )
    if body.format not in SUPPORTED_FORMATS:
        raise APIError(
            ErrorCode.VALIDATION_ERROR,
            f"unsupported format {body.format!r}; supported: {list(SUPPORTED_FORMATS)}",
        )
    if (body.surah is None) != (body.ayah is None):
        raise APIError(
            ErrorCode.VALIDATION_ERROR, "supply both surah and ayah, or neither"
        )

    try:
        session = store.create(
            surah=body.surah,
            ayah=body.ayah,
            tier=body.tier,
            sample_rate=body.sample_rate,
            audio_format=body.format,
        )
    except CapacityExceeded as exc:
        raise APIError(
            ErrorCode.RATE_LIMITED, str(exc), status.HTTP_503_SERVICE_UNAVAILABLE
        ) from exc

    return SessionResponse(
        session_id=session.id,
        surah=session.surah,
        ayah=session.ayah,
        tier=session.tier,
        sample_rate=session.sample_rate,
        format=session.audio_format,
        expires_in_seconds=round(session.seconds_remaining, 1),
        websocket_url=f"{config.api_prefix}/sessions/{session.id}/stream",
        active_sessions=store.active_count,
        capacity=store.capacity,
    )


@router.get("/{session_id}", response_model=SessionResponse, summary="Session status")
def get_session(
    session_id: str,
    store: SessionStore = Depends(get_session_store),
    config: Settings = Depends(get_settings),
) -> SessionResponse:
    session = store.get(session_id)
    if session is None:
        raise APIError(
            ErrorCode.VALIDATION_ERROR,
            "unknown or expired session",
            status.HTTP_404_NOT_FOUND,
        )
    return SessionResponse(
        session_id=session.id,
        surah=session.surah,
        ayah=session.ayah,
        tier=session.tier,
        sample_rate=session.sample_rate,
        format=session.audio_format,
        expires_in_seconds=round(session.seconds_remaining, 1),
        websocket_url=f"{config.api_prefix}/sessions/{session.id}/stream",
        active_sessions=store.active_count,
        capacity=store.capacity,
    )


@router.delete(
    "/{session_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Close a session"
)
def close_session(
    session_id: str, store: SessionStore = Depends(get_session_store)
) -> None:
    if not store.close(session_id):
        raise APIError(
            ErrorCode.VALIDATION_ERROR,
            "unknown or expired session",
            status.HTTP_404_NOT_FOUND,
        )
