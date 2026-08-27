"""Pydantic models for Quran reference data.

Versioned from the first commit: `schema_version` is part of the prepared data
asset so a future change is detectable rather than silently incompatible.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1


class Word(BaseModel):
    """One word of an ayah, in every representation we need."""

    index: int = Field(..., description="1-based Imlaey word position - the alignment index")
    uthmani: str = Field(..., description="Canonical display form (Tanzil, immutable)")
    imlaey: str = Field(..., description="Simple-script form; the ASR's orthography")
    normalized: str = Field(..., description="normalize_for_asr_matching(imlaey)")
    uthmani_index: int = Field(
        ...,
        description=(
            "1-based Uthmani word this belongs to. Several Imlaey words can share one "
            "Uthmani word (يَـٰٓأَيُّهَا -> يَا أَيُّهَا), so clients that display Uthmani "
            "should group by this rather than by `index`."
        ),
    )


class Ayah(BaseModel):
    surah: int
    ayah: int
    surah_name: str
    uthmani: str = Field(..., description="Canonical display text - never modified")
    imlaey: str
    normalized: str = Field(..., description="Whole-ayah ASR matching form")
    words: list[Word]

    @property
    def word_count(self) -> int:
        return len(self.words)


class Surah(BaseModel):
    surah: int
    name: str
    ayah_count: int


class QuranAsset(BaseModel):
    """The prepared, immutable Quran data file."""

    schema_version: int = SCHEMA_VERSION
    source: str = "Tanzil Quran Text (Uthmani + Imlaey), CC-BY 3.0, via quran-transcript"
    attribution: str = (
        "Tanzil Quran Text Copyright (C) 2007-2024 Tanzil Project - https://tanzil.net - "
        "Licensed under Creative Commons Attribution 3.0. Verbatim copies only."
    )
    surahs: list[Surah]
    ayat: list[Ayah]


# ── Recitation analysis (Phase 4) ─────────────────────────────────────────────


class WordAnalysis(BaseModel):
    """One position in the comparison, as the API reports it."""

    index: int | None = Field(None, description="1-based expected word index; null for extras")
    expected: str | None = None
    spoken: str | None = None
    status: str
    band: str
    category: str | None = Field(None, description="Error category, if we are reporting one")
    confidence: float
    similarity: float = 0.0
    note: str | None = Field(
        None, description="Why a mechanical finding was not reported as a mistake"
    )


class RecitationAnalysis(BaseModel):
    schema_version: int = SCHEMA_VERSION
    session_id: str
    surah: int
    ayah: int
    expected_text: str
    recognized_text: str
    word_accuracy_score: float = Field(
        ..., description="Word-level accuracy only. NOT a Tajweed score."
    )
    score_formula: str
    words: list[WordAnalysis]
    errors: list[WordAnalysis] = Field(
        default_factory=list, description="Confirmed mistakes only"
    )
    observations: list[WordAnalysis] = Field(
        default_factory=list,
        description="Possible issues we are not confident enough to call mistakes",
    )
    engine: dict[str, str]
    audio_seconds: float
    processing_seconds: float
    verse_detected: bool = Field(
        False, description="True when the verse was inferred rather than supplied"
    )
    verse_confidence: float | None = None
