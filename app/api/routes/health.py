"""Liveness and configuration disclosure."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import (
    get_auth, get_pronunciation, get_rate_limiter, get_registry, get_settings,
)
from app.asr.registry import ASRRegistry
from app.core.config import Settings
from app.core.security import AuthProvider, RateLimiter
from app.pronunciation.analyzer import PronunciationAnalyzer

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str = "ok"
    engines: list[dict[str, str]] = Field(
        ..., description="Configured ASR tiers, by opaque label - never model identity"
    )
    device: str = Field(
        ..., description="Device actually resolved and in use, not the configured value"
    )
    compute_type: str
    auth_enabled: bool = Field(
        ..., description="False means the API is unauthenticated - fine locally, not in production"
    )
    rate_limiting_enabled: bool
    store_audio: bool = Field(..., description="True means recordings are persisted")
    phoneme_analysis: dict[str, str] = Field(
        ...,
        description=(
            "Whether phoneme and Tajweed analysis can run here. When unavailable, "
            "the API will not report phoneme or Tajweed findings at all."
        ),
    )


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health(
    registry: ASRRegistry = Depends(get_registry),
    config: Settings = Depends(get_settings),
    auth: AuthProvider = Depends(get_auth),
    limiter: RateLimiter = Depends(get_rate_limiter),
    pronunciation: PronunciationAnalyzer = Depends(get_pronunciation),
) -> HealthResponse:
    """Reports which device and configuration are actually in use.

    Deliberately surfaces `auth_enabled` and `store_audio`: running unauthenticated
    or unexpectedly persisting user audio should be visible, not silent.
    """
    engines = registry.describe()
    return HealthResponse(
        engines=engines,
        # `config.device` may be "auto"; report what it actually resolved to.
        device=engines[0]["device"] if engines else config.device,
        compute_type=config.compute_type,
        auth_enabled=auth.enabled,
        rate_limiting_enabled=limiter.enabled,
        store_audio=config.store_audio,
        phoneme_analysis={
            "available": str(pronunciation.available).lower(),
            "reason": pronunciation.unavailable_reason or "",
        },
    )
