"""Shared dependencies.

Everything expensive is built once and reused: the Quran asset, the verse index
and the ASR engines are process-wide, never per request. Building an engine per
request would be the single easiest way to make this server unusable.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, Header

from app.asr.base import ASREngine, Tier
from app.asr.registry import ASRRegistry
from app.core.config import Settings, settings
from app.core.security import AuthProvider, RateLimiter
from app.quran.repository import QuranRepository
from app.recitations.repository import RecitationRepository
from app.quran.verse_locator import VerseLocator
from app.pronunciation.analyzer import PronunciationAnalyzer
from app.pronunciation.muaalem import MuaalemRecognizer
from app.recitation.analyzer import RecitationAnalyzer
from app.streaming.session import SessionStore


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return settings


@lru_cache(maxsize=1)
def get_repository() -> QuranRepository:
    return QuranRepository()


@lru_cache(maxsize=1)
def get_locator() -> VerseLocator:
    return VerseLocator(get_repository())


@lru_cache(maxsize=1)
def get_registry() -> ASRRegistry:
    return ASRRegistry(get_settings())


@lru_cache(maxsize=1)
def get_auth() -> AuthProvider:
    return AuthProvider(get_settings().allowed_api_keys)


@lru_cache(maxsize=1)
def get_rate_limiter() -> RateLimiter:
    return RateLimiter(get_settings().rate_limit_per_minute)


@lru_cache(maxsize=1)
def get_session_store() -> SessionStore:
    config = get_settings()
    return SessionStore(config.max_concurrent_sessions, config.max_session_seconds)


@lru_cache(maxsize=1)
def get_pronunciation() -> PronunciationAnalyzer:
    """Phoneme analysis. Reports itself unavailable where torch cannot run."""
    return PronunciationAnalyzer(MuaalemRecognizer(device=get_settings().device))


@lru_cache(maxsize=1)
def get_recitations() -> RecitationRepository:
    return RecitationRepository(get_settings().recitations_dir)


def get_engine(tier: Tier = Tier.COMMITTED) -> ASREngine:
    return get_registry().get(tier)


def get_analyzer(tier: Tier = Tier.COMMITTED) -> RecitationAnalyzer:
    return RecitationAnalyzer(
        get_engine(tier), repository=get_repository(), locator=get_locator()
    )


def require_caller(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    auth: AuthProvider = Depends(get_auth),
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> str:
    """Authenticate, then rate limit. Returns the caller identity for logging."""
    identity = auth.authenticate(x_api_key)
    limiter.check(identity)
    return identity


def reset_caches() -> None:
    """Drop cached singletons - used by tests that change settings.

    Tolerates a member having been replaced (by monkeypatch, say) rather than
    exploding on a missing `cache_clear`.
    """
    for fn in (
        get_settings, get_repository, get_locator, get_registry,
        get_auth, get_rate_limiter, get_session_store, get_pronunciation,
        get_recitations,
    ):
        clear = getattr(fn, "cache_clear", None)
        if clear is not None:
            clear()
