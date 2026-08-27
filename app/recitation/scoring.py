"""Deterministic word-accuracy scoring.

    score = 100
            - 5.0 x substitutions
            - 5.0 x missing words
            - 2.0 x extra words
            - 1.0 x repetitions
    clamped to [0, 100]

Design rules, all of which matter:

* **Only confirmed errors count.** Anything the confidence policy marked
  UNCERTAIN scores zero penalty. A reciter is never docked points because the
  recognizer was unsure.
* **Omissions and substitutions weigh most** - they change the text. Repetitions
  weigh least: repeating a word is a stumble, not a misreading.
* This is `word_accuracy_score`, **not** a Tajweed score, and nothing here may be
  labelled as one until a Tajweed engine exists.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.alignment.base import AlignmentResult
from app.models.enums import ErrorCategory

PENALTIES: dict[ErrorCategory, float] = {
    ErrorCategory.WORD_SUBSTITUTION: 5.0,
    ErrorCategory.WORD_MISSING: 5.0,
    ErrorCategory.WORD_EXTRA: 2.0,
    ErrorCategory.WORD_REPETITION: 1.0,
    # Explicitly zero: an unconfirmed pronunciation difference is not a mistake.
    ErrorCategory.POSSIBLE_PRONUNCIATION_ERROR: 0.0,
}


@dataclass(frozen=True)
class ScoreBreakdown:
    score: float
    correct: int
    substituted: int
    missing: int
    extra: int
    repeated: int
    uncertain: int
    total_expected: int
    penalty: float

    @property
    def formula(self) -> str:
        return (
            "100 - 5.0*substitutions - 5.0*missing - 2.0*extra - 1.0*repetitions; "
            "uncertain findings carry no penalty"
        )


def score(results: list[AlignmentResult], total_expected: int) -> ScoreBreakdown:
    counts = dict.fromkeys(PENALTIES, 0)
    correct = uncertain = 0
    penalty = 0.0

    for r in results:
        if r.is_error and r.category in PENALTIES:
            counts[r.category] += 1
            penalty += PENALTIES[r.category]
        elif r.band.value == "high_confidence_correct":
            correct += 1
        else:
            uncertain += 1

    return ScoreBreakdown(
        score=round(max(0.0, min(100.0, 100.0 - penalty)), 2),
        correct=correct,
        substituted=counts[ErrorCategory.WORD_SUBSTITUTION],
        missing=counts[ErrorCategory.WORD_MISSING],
        extra=counts[ErrorCategory.WORD_EXTRA],
        repeated=counts[ErrorCategory.WORD_REPETITION],
        uncertain=uncertain,
        total_expected=total_expected,
        penalty=round(penalty, 2),
    )
