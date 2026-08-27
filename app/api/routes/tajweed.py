"""Tajweed rules: the catalogue, and where they apply in a verse."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, status

from app.api.deps import get_repository, require_caller
from app.core.errors import APIError, ErrorCode
from app.models.schemas import (
    TajweedOccurrenceOut,
    TajweedResponse,
    TajweedRuleOut,
    TajweedRulesResponse,
)
from app.quran.repository import QuranRepository
from app.tajweed.analyzer import TajweedAnalyzer
from app.tajweed.rules import RULES, summary

router = APIRouter(tags=["tajweed"], dependencies=[Depends(require_caller)])


@router.get("/tajweed/rules", response_model=TajweedRulesResponse, summary="The rule catalogue")
def list_rules() -> TajweedRulesResponse:
    """Every rule this system knows about, and how far it can actually check each one.

    `verification` is the honest part. Only the madd family and the doubled
    ghunnah are **measured** - the engine names the rule and compares counts.
    Most rules are **positional**: a phoneme difference in the right word is
    suggestive but not conclusive. Qalqalah is **occurrence_only** - removing it
    from a phoneme string produces no finding, so its performance cannot be
    checked. Ishmam is **not_detectable** by definition: it makes no sound.
    """
    return TajweedRulesResponse(
        rules=[
            TajweedRuleOut(
                key=r.key, name_en=r.name_en, name_ar=r.name_ar,
                category=r.category.value, verification=r.verification.value,
                definition=r.definition, note=r.note,
            )
            for r in RULES.values()
        ],
        verification_summary=summary(),
    )


@router.get(
    "/surahs/{surah_id}/ayahs/{ayah_id}/tajweed",
    response_model=TajweedResponse,
    summary="Tajweed rules applying in an ayah",
)
def ayah_tajweed(
    surah_id: int = Path(..., ge=1, le=114),
    ayah_id: int = Path(..., ge=1),
    repo: QuranRepository = Depends(get_repository),
) -> TajweedResponse:
    """Where each rule applies in this verse, derived from the canonical text.

    Needs no model and no audio, so it is always available - useful on its own for
    teaching, and it is the map that pronunciation findings are correlated against.

    `verified` is false here: this endpoint does not listen to anything. An empty
    `checks` list means nothing was checked, not that every rule was kept.
    """
    try:
        repo.ayah(surah_id, ayah_id)
    except KeyError as exc:
        raise APIError(
            ErrorCode.VERSE_NOT_FOUND, str(exc), status.HTTP_404_NOT_FOUND
        ) from exc

    report = TajweedAnalyzer(repo).report(surah_id, ayah_id)
    return TajweedResponse(
        surah=surah_id,
        ayah=ayah_id,
        verified=report.verified,
        occurrences=[
            TajweedOccurrenceOut(
                rule=o.rule.key, name_en=o.rule.name_en, name_ar=o.rule.name_ar,
                verification=o.rule.verification.value,
                uthmani_word_index=o.uthmani_word_index,
                word_indices=list(o.word_indices), word=o.word_text,
            )
            for o in report.occurrences
        ],
        rule_counts=report.rule_counts,
        unavailable_reason=report.unavailable_reason,
    )
