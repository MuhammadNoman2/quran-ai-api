"""Canonical Quran reference data. Read-only by construction."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, status

from app.api.deps import get_repository, require_caller
from app.core.errors import APIError, ErrorCode
from app.models.schemas import Ayah, Surah
from app.quran.repository import QuranRepository

router = APIRouter(tags=["quran"], dependencies=[Depends(require_caller)])


@router.get("/surahs", response_model=list[Surah], summary="List all surahs")
def list_surahs(repo: QuranRepository = Depends(get_repository)) -> list[Surah]:
    return repo.surahs()


@router.get("/surahs/{surah_id}", response_model=Surah, summary="One surah")
def get_surah(
    surah_id: int = Path(..., ge=1, le=114),
    repo: QuranRepository = Depends(get_repository),
) -> Surah:
    try:
        return repo.surah(surah_id)
    except KeyError as exc:
        raise APIError(
            ErrorCode.VERSE_NOT_FOUND, str(exc), status.HTTP_404_NOT_FOUND
        ) from exc


@router.get(
    "/surahs/{surah_id}/ayahs/{ayah_id}",
    response_model=Ayah,
    summary="One ayah, with word-by-word breakdown",
)
def get_ayah(
    surah_id: int = Path(..., ge=1, le=114),
    ayah_id: int = Path(..., ge=1),
    repo: QuranRepository = Depends(get_repository),
) -> Ayah:
    """Returns the canonical Uthmani text, the Imlaey form, and per-word data.

    `words[].index` is the Imlaey word position used everywhere in analysis
    results. `words[].uthmani_index` groups them onto Uthmani words for display,
    since one Uthmani word can be two Imlaey words.
    """
    try:
        return repo.ayah(surah_id, ayah_id)
    except KeyError as exc:
        raise APIError(
            ErrorCode.VERSE_NOT_FOUND, str(exc), status.HTTP_404_NOT_FOUND
        ) from exc
