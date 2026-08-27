"""Pronunciation checking against phoneme-level evidence.

This endpoint exists so a client can ask for phoneme and Tajweed feedback and get
a straight answer about whether it was actually possible - rather than an empty
findings list that reads like a clean pass.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, UploadFile, status

from app.api.deps import get_pronunciation, get_repository, get_settings, require_caller
from app.core.config import Settings
from app.core.errors import APIError, ErrorCode
from app.models.schemas import (
    PronunciationFindingOut, PronunciationResponse, TajweedCheckOut,
)
from app.pronunciation.analyzer import PronunciationAnalyzer
from app.quran.repository import QuranRepository
from app.tajweed.analyzer import TajweedAnalyzer

router = APIRouter(
    prefix="/recitation", tags=["recitation"], dependencies=[Depends(require_caller)]
)
logger = logging.getLogger(__name__)


@router.post(
    "/pronunciation",
    response_model=PronunciationResponse,
    summary="Check pronunciation at the phoneme level",
)
async def check_pronunciation(
    audio: UploadFile = File(...),
    surah: int = Form(..., ge=1, le=114),
    ayah: int = Form(..., ge=1),
    analyzer: PronunciationAnalyzer = Depends(get_pronunciation),
    repo: QuranRepository = Depends(get_repository),
    config: Settings = Depends(get_settings),
) -> PronunciationResponse:
    """Compare the phonemes heard against the phonemes expected.

    **Check `available` before reading `findings`.** When it is false, no phoneme
    analysis ran and an empty `findings` list means "not checked", not "nothing
    wrong". `unavailable_reason` says why. The expected phonemes are returned
    either way, since knowing what should be recited is useful on its own.

    Findings are classified as `articulation` (a different letter), `tashkeel`
    (a short vowel, shadda or sukun), or `tajweed` (a rule with a defined count,
    such as a madd held too briefly). Tajweed findings name the rule and the
    count expected.

    `tajweed_checks` gives the per-rule outcome for every rule applying in this
    verse. A rule is `missed` only when the phonetic engine named it and compared
    counts; `possibly_missed` means a phoneme difference merely landed in the
    right word, and `not_checked` means the rule's performance cannot be verified
    at all - qalqalah, for instance. See `docs/phonetics.md`.
    """
    from app.api.routes.recitation import read_audio

    try:
        repo.ayah(surah, ayah)
    except KeyError as exc:
        raise APIError(
            ErrorCode.VERSE_NOT_FOUND, str(exc), status.HTTP_404_NOT_FOUND
        ) from exc

    samples = await read_audio(audio, config)
    try:
        report = analyzer.analyze(samples, surah, ayah)
    finally:
        del samples

    logger.info(
        "pronunciation checked",
        extra={
            "surah": surah, "ayah": ayah,
            "available": report.available,
            "findings": len(report.findings),
            "engine": report.engine,
        },
    )

    tajweed = TajweedAnalyzer(repo).verify(surah, ayah, report)

    return PronunciationResponse(
        available=report.available,
        surah=report.surah,
        ayah=report.ayah,
        reference_phonemes=report.reference_phonemes,
        recognized_phonemes=report.recognized_phonemes,
        findings=[
            PronunciationFindingOut(
                kind=f.kind.value,
                operation=f.operation.value,
                word_index=f.word_index,
                expected_phonemes=f.expected_phonemes,
                spoken_phonemes=f.spoken_phonemes,
                expected_length=f.expected_length,
                spoken_length=f.spoken_length,
                tajweed_rule=f.tajweed_rule,
                tajweed_rule_ar=f.tajweed_rule_ar,
                detail=f.detail,
            )
            for f in report.findings
        ],
        unavailable_reason=report.unavailable_reason,
        engine=report.engine,
        tajweed_checks=[
            TajweedCheckOut(
                rule=c.occurrence.rule.key,
                name_en=c.occurrence.rule.name_en,
                word_indices=list(c.occurrence.word_indices),
                status=c.status,
                confidence=c.confidence,
                detail=c.detail,
            )
            for c in tajweed.checks
        ],
    )
