"""Verified recitation audio - Mode B of the brief.

Recitations are never synthesised. They are performances by named reciters, and
this API's job is to point at the right one and be honest about whether it may be
redistributed.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query, status

from app.api.deps import get_recitations, get_repository, get_settings, require_caller
from app.core.config import Settings
from app.core.errors import APIError, ErrorCode
from app.models.schemas import RecitationOut, ReciterListOut, ReciterOut
from app.quran.repository import QuranRepository
from app.recitations.registry import DEFAULT_RECITER, all_reciters
from app.recitations.repository import (
    RecitationForbidden,
    RecitationRepository,
    RecitationUnavailable,
)

router = APIRouter(tags=["recitations"], dependencies=[Depends(require_caller)])


@router.get("/reciters", response_model=ReciterListOut, summary="Available reciters")
def list_reciters(
    repo: RecitationRepository = Depends(get_recitations),
    config: Settings = Depends(get_settings),
) -> ReciterListOut:
    """Reciters, with the licence status of their recordings.

    **Read `licence` before building on any of these.** No source publishes
    per-recitation licences for the recordings themselves, so every entry here is
    `unknown` until someone obtains written permission. That is fine for local
    development and evaluation, and not fine for shipping.

    Note in particular that quranicaudio.com's MIT licence covers its *website
    software*, not the audio.
    """
    return ReciterListOut(
        reciters=[
            ReciterOut(
                id=r.id, name_en=r.name_en, name_ar=r.name_ar,
                style=r.style.value, riwaya=r.riwaya, source=r.source,
                licence=r.licence.value, licence_note=r.licence_note,
                cached_ayat=repo.cached_count(r.id),
            )
            for r in all_reciters()
        ],
        default=DEFAULT_RECITER,
        serve_unverified_audio=config.serve_unverified_audio,
    )


@router.get(
    "/recitations/{surah}/{ayah}",
    response_model=RecitationOut,
    summary="Verified recitation audio for an ayah",
)
def get_recitation(
    surah: int = Path(..., ge=1, le=114),
    ayah: int = Path(..., ge=1),
    reciter: str = Query(DEFAULT_RECITER),
    repo: RecitationRepository = Depends(get_recitations),
    quran: QuranRepository = Depends(get_repository),
) -> RecitationOut:
    """A recitation of this ayah by a named reciter.

    `served_locally` tells you where the audio comes from. When it is false the
    URL points at the original source, either because the file is not cached or
    because its licence has not been verified and this server will not
    redistribute it. `licence` and `licence_note` always travel with the response.
    """
    try:
        quran.ayah(surah, ayah)
    except KeyError as exc:
        raise APIError(
            ErrorCode.VERSE_NOT_FOUND, str(exc), status.HTTP_404_NOT_FOUND
        ) from exc

    try:
        recitation = repo.get(surah, ayah, reciter, base_url="/audio")
    except RecitationForbidden as exc:
        raise APIError(
            ErrorCode.VALIDATION_ERROR, str(exc), status.HTTP_403_FORBIDDEN
        ) from exc
    except RecitationUnavailable as exc:
        raise APIError(
            ErrorCode.VERSE_NOT_FOUND, str(exc), status.HTTP_404_NOT_FOUND
        ) from exc

    return RecitationOut(
        surah=recitation.surah,
        ayah=recitation.ayah,
        reciter=recitation.reciter.id,
        reciter_name=recitation.reciter.name_en,
        audio_url=recitation.audio_url,
        served_locally=recitation.local,
        size_bytes=recitation.size_bytes,
        licence=recitation.reciter.licence.value,
        licence_note=recitation.reciter.licence_note,
    )
