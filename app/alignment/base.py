"""Types shared by the alignment engine."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.enums import ConfidenceBand, ErrorCategory, WordStatus


@dataclass(frozen=True)
class ExpectedWord:
    """A word of the canonical Quran text."""

    text: str
    """Canonical Uthmani form - what the user sees."""

    normalized: str
    index: int
    """1-based Imlaey word position within the ayah."""

    surah: int
    ayah: int

    @property
    def position(self) -> str:
        return f"{self.surah}:{self.ayah}:{self.index}"


@dataclass(frozen=True)
class SpokenWord:
    """A word the recognizer produced."""

    text: str
    normalized: str
    start_time: float
    end_time: float
    confidence: float


@dataclass(frozen=True)
class AlignmentResult:
    """One position in the comparison.

    Exactly one of `expected_word` / `spoken_word` may be None:
    None expected -> the reciter said something extra;
    None spoken   -> the reciter omitted a word.
    """

    status: WordStatus
    expected_word: ExpectedWord | None = None
    spoken_word: SpokenWord | None = None
    confidence: float = 0.0
    band: ConfidenceBand = ConfidenceBand.UNCERTAIN
    category: ErrorCategory | None = None
    similarity: float = 0.0
    """Normalized-form similarity, 0..1. Distinguishes a near-miss from a different word."""

    suppressed_reason: str | None = None
    """Set when a mechanical finding was withheld from the user, and why."""

    @property
    def is_error(self) -> bool:
        """True only for findings we are willing to report as a mistake."""
        return self.band is ConfidenceBand.HIGH_CONFIDENCE_ERROR and self.category is not None

    @property
    def index(self) -> int | None:
        return self.expected_word.index if self.expected_word else None
