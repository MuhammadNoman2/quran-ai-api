"""FastAPI application.

    uvicorn app.main:app --reload

Clients need only the base URL, an optional API key, and audio. No model, no ML
dependency, no knowledge of what runs inside.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import get_locator, get_registry, get_settings
from app.api.routes import health, quran, recitation, sessions
from app.core.errors import APIError, api_error_handler, unhandled_error_handler
from app.core.logging import configure

logger = logging.getLogger(__name__)

DESCRIPTION = """
Local-first Quran recitation recognition and word-level correction.

### What it does
Send a recitation, get back a word-by-word comparison against the canonical
Quran text, with a deterministic accuracy score.

### What it deliberately does not do
Report a mistake it is not confident about. Speech recognition hallucinates -
measured on this project's own models, a professional reciter's clean recording
produces spurious trailing words on nearly every short clip. Those are filtered
by confidence and by speech-boundary checks, and anything that survives as
doubtful is returned under `observations` rather than `errors`. Telling someone
their correct recitation was wrong is the worst failure this API can have.

`word_accuracy_score` covers words only. It is **not** a Tajweed score, and no
Tajweed or phoneme-level judgement is available yet.

### Audio
wav, mp3, m4a, ogg or flac. Decoded server-side; any sample rate is accepted and
resampled to 16 kHz mono. Recordings are held in memory and dropped after
analysis - they are never written to disk or logged.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_settings()
    configure(config.log_level)

    # The verse index takes ~2s to build. Doing it now means the first request
    # does not pay for it.
    get_locator().warm()

    registry = get_registry()
    if config.preload_models:
        registry.preload()

    logger.info(
        "service ready",
        extra={
            "device": config.device,
            "compute_type": config.compute_type,
            "preloaded": config.preload_models,
            "auth_enabled": config.auth_enabled,
            "store_audio": config.store_audio,
            "max_concurrent_sessions": config.max_concurrent_sessions,
        },
    )
    if not config.auth_enabled:
        logger.warning("authentication is disabled - do not run this in production")
    if config.store_audio:
        logger.warning("STORE_AUDIO is enabled - user recordings will be persisted")

    yield
    registry.unload_all()


def create_app() -> FastAPI:
    config = get_settings()
    app = FastAPI(
        title="Quran AI Recitation API",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Liveness and effective configuration"},
            {"name": "quran", "description": "Canonical Quran reference data"},
            {"name": "recitation", "description": "Analysis and verse detection"},
            {"name": "sessions", "description": "Session lifecycle for streaming"},
        ],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)

    for module in (health, quran, recitation, sessions):
        app.include_router(module.router, prefix=config.api_prefix)
    return app


app = create_app()
