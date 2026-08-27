"""Serving verified recitation audio.

Local files are preferred: they are faster, work offline, and avoid depending on
a third party staying up. When a file is not cached the upstream URL is returned
instead, so a client can still play something - but the licence status travels
with every response either way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings
from app.recitations.registry import DEFAULT_RECITER, LicenceStatus, Reciter, get

logger = logging.getLogger(__name__)


class RecitationUnavailable(RuntimeError):
    pass


class RecitationForbidden(RuntimeError):
    """Raised when licence status forbids serving the audio from this server."""


@dataclass(frozen=True)
class Recitation:
    surah: int
    ayah: int
    reciter: Reciter
    audio_url: str
    local: bool
    size_bytes: int | None = None


class RecitationRepository:
    def __init__(self, root: Path | None = None) -> None:
        self._root = root or settings.recitations_dir

    def path_for(self, reciter_id: str, surah: int, ayah: int) -> Path:
        return self._root / reciter_id / f"{surah:03d}{ayah:03d}.mp3"

    def get(
        self, surah: int, ayah: int, reciter_id: str | None = None, *, base_url: str = ""
    ) -> Recitation:
        reciter = get(reciter_id or DEFAULT_RECITER)
        if reciter is None:
            raise RecitationUnavailable(f"unknown reciter {reciter_id!r}")

        if reciter.licence is LicenceStatus.RESTRICTED:
            raise RecitationForbidden(
                f"{reciter.name_en}: {reciter.licence_note}"
            )

        path = self.path_for(reciter.id, surah, ayah)
        if path.exists():
            if not reciter.redistributable and not settings.serve_unverified_audio:
                # Cached locally, but nobody has confirmed we may serve it. Point
                # at the source instead of quietly redistributing a performance.
                logger.debug(
                    "not serving cached audio of unverified licence",
                    extra={"reciter": reciter.id},
                )
                return Recitation(
                    surah=surah, ayah=ayah, reciter=reciter,
                    audio_url=reciter.url_for(surah, ayah), local=False,
                )
            return Recitation(
                surah=surah, ayah=ayah, reciter=reciter,
                audio_url=f"{base_url}/{reciter.id}/{surah:03d}{ayah:03d}.mp3",
                local=True, size_bytes=path.stat().st_size,
            )

        return Recitation(
            surah=surah, ayah=ayah, reciter=reciter,
            audio_url=reciter.url_for(surah, ayah), local=False,
        )

    def cached_count(self, reciter_id: str) -> int:
        directory = self._root / reciter_id
        return len(list(directory.glob("*.mp3"))) if directory.is_dir() else 0
