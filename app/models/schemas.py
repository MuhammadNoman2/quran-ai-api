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


# ── Phonetics and pronunciation (Phase 8) ─────────────────────────────────────


class PhoneticWord(BaseModel):
    index: int = Field(..., description="1-based word position within the ayah")
    uthmani: str
    phonemes: str = Field(..., description="Quran Phonetic Script for this word")


class PhoneticsResponse(BaseModel):
    """What *should* be recited, phonetically. Needs no model, always available."""

    surah: int
    ayah: int
    uthmani: str
    phonemes: str
    words: list[PhoneticWord]
    tajweed_rules: list[str] = Field(
        ..., description="Rules that apply somewhere in this ayah"
    )
    sifat: list[dict[str, object]] = Field(
        default_factory=list,
        description="Ten articulation attributes per phoneme group",
    )
    rewaya: str
    madd_lengths: dict[str, int] = Field(
        ..., description="The madd counts these expectations were built from"
    )


class PronunciationFindingOut(BaseModel):
    kind: str = Field(..., description="articulation | tashkeel | tajweed | unknown")
    operation: str
    word_index: int | None = None
    expected_phonemes: str
    spoken_phonemes: str
    expected_length: int | None = None
    spoken_length: int | None = None
    tajweed_rule: str | None = None
    tajweed_rule_ar: str | None = None
    detail: str | None = None


class PronunciationResponse(BaseModel):
    available: bool = Field(
        ...,
        description=(
            "False means no phoneme analysis was performed. `findings` being empty "
            "then means 'not checked', NOT 'nothing wrong'."
        ),
    )
    surah: int
    ayah: int
    reference_phonemes: str
    recognized_phonemes: str | None = None
    findings: list[PronunciationFindingOut] = Field(default_factory=list)
    unavailable_reason: str | None = None
    engine: str | None = None
    tajweed_checks: list["TajweedCheckOut"] = Field(
        default_factory=list,
        description=(
            "Per-rule outcome, present only when phoneme analysis ran. A rule is "
            "'missed' only when the engine named it and compared counts; "
            "'possibly_missed' means a phoneme difference merely landed where the "
            "rule applies."
        ),
    )


# ── Tajweed (Phase 9) ─────────────────────────────────────────────────────────


class TajweedRuleOut(BaseModel):
    key: str
    name_en: str
    name_ar: str
    category: str
    verification: str = Field(
        ...,
        description=(
            "measured = the rule is named and its count compared; "
            "positional = a phoneme difference at the right place, suggestive only; "
            "occurrence_only = we can locate it but cannot verify performance; "
            "not_detectable = it produces no audible evidence at all"
        ),
    )
    definition: str
    note: str | None = None


class TajweedOccurrenceOut(BaseModel):
    rule: str
    name_en: str
    name_ar: str
    verification: str
    uthmani_word_index: int
    word_indices: list[int] = Field(
        ..., description="Imlaey word indices, matching analysis results"
    )
    word: str


class TajweedCheckOut(BaseModel):
    rule: str
    name_en: str
    word_indices: list[int]
    status: str = Field(..., description="kept | possibly_missed | missed | not_checked")
    confidence: str = Field(..., description="measured | positional | none")
    detail: str | None = None


class TajweedResponse(BaseModel):
    surah: int
    ayah: int
    verified: bool = Field(
        ...,
        description=(
            "False means only occurrences were computed. `checks` being empty "
            "then means 'not checked', NOT 'all rules kept'."
        ),
    )
    occurrences: list[TajweedOccurrenceOut]
    rule_counts: dict[str, int]
    checks: list[TajweedCheckOut] = Field(default_factory=list)
    unavailable_reason: str | None = None


class TajweedRulesResponse(BaseModel):
    rules: list[TajweedRuleOut]
    verification_summary: dict[str, int]


# ── Verified recitation audio (Phase 10) ──────────────────────────────────────


class ReciterOut(BaseModel):
    id: str
    name_en: str
    name_ar: str
    style: str
    riwaya: str
    source: str
    licence: str = Field(..., description="verified | unknown | restricted")
    licence_note: str
    cached_ayat: int = Field(0, description="How many ayat are cached locally")


class RecitationOut(BaseModel):
    surah: int
    ayah: int
    reciter: str
    reciter_name: str
    audio_url: str
    served_locally: bool = Field(
        ..., description="False means the URL points at the original source"
    )
    size_bytes: int | None = None
    licence: str
    licence_note: str


class ReciterListOut(BaseModel):
    reciters: list[ReciterOut]
    default: str
    serve_unverified_audio: bool = Field(
        ...,
        description=(
            "When false, audio whose licence is unverified is never served from "
            "this server even if cached locally."
        ),
    )
