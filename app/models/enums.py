"""Statuses and error categories.

Two vocabularies are kept deliberately separate:

* `WordStatus` is what the **aligner** concluded about a position - a mechanical
  fact about two sequences of tokens.
* `ErrorCategory` is what we are willing to **tell the user**, which is a much
  stronger claim and depends on confidence and on which evidence exists.

Conflating them is how a Quran app ends up telling a child they made a Tajweed
mistake because a speech model was unsure about a word.
"""

from __future__ import annotations

from enum import Enum


class WordStatus(str, Enum):
    """Outcome of aligning one expected word against the recognized sequence."""

    CORRECT = "correct"
    SUBSTITUTED = "substituted"
    MISSING = "missing"
    EXTRA = "extra"
    REPEATED = "repeated"
    UNKNOWN = "unknown"


class ErrorCategory(str, Enum):
    """What we report. Only categories with actual evidence behind them."""

    WORD_SUBSTITUTION = "word_substitution"
    WORD_MISSING = "word_missing"
    WORD_EXTRA = "word_extra"
    WORD_REPETITION = "word_repetition"

    #: Words differ but are phonetically close (قل vs كل). Could be the reciter's
    #: articulation or the recognizer's mistake - we cannot yet tell which, and
    #: saying otherwise would be a false correction.
    POSSIBLE_PRONUNCIATION_ERROR = "possible_pronunciation_error"

    #: Requires phoneme-level evidence. Phase 8. Emitting this before then is a
    #: lie, and a test enforces that it never happens.
    PHONEME_ERROR = "phoneme_error"

    #: Requires a Tajweed rule engine. Phase 9. Same enforcement.
    TAJWEED_ERROR = "tajweed_error"

    UNKNOWN = "unknown"


#: Categories no code path may emit until the corresponding analysis exists.
NOT_YET_DETECTABLE: frozenset[ErrorCategory] = frozenset(
    {ErrorCategory.PHONEME_ERROR, ErrorCategory.TAJWEED_ERROR}
)


class ConfidenceBand(str, Enum):
    """How much we trust the judgement at a position."""

    HIGH_CONFIDENCE_CORRECT = "high_confidence_correct"
    HIGH_CONFIDENCE_ERROR = "high_confidence_error"
    UNCERTAIN = "uncertain"


class ReviewState(str, Enum):
    """Streaming lifecycle (Phase 7). Only CONFIRMED may assert a mistake."""

    PENDING = "pending"
    PROVISIONAL = "provisional"
    CONFIRMED = "confirmed"
