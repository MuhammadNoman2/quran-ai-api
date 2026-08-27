"""Decide what we are willing to tell the reciter.

The aligner reports mechanical facts. This module turns those into claims, and
it is deliberately conservative: a Quran learning tool that wrongly corrects a
child who recited properly does real harm, so anything short of solid evidence
becomes UNCERTAIN rather than an accusation.

Two independent gates catch hallucinated words, because neither is sufficient
alone (docs/model-selection.md 3.2):

1. **Confidence gate.** Genuine words were measured at probability 1.00 while
   hallucinated tails scored 0.17-0.87.
2. **Speech-boundary gate.** Hallucinations are emitted *after* the reciter has
   stopped, so a word starting past the end of detected speech is suspect no
   matter how confident the model claims to be. This catches the 0.86-0.87 cases
   that a threshold alone would let through.
"""

from __future__ import annotations

from dataclasses import replace

from app.alignment.base import AlignmentResult
from app.asr.base import Tier
from app.models.enums import ConfidenceBand, ErrorCategory, WordStatus

#: Letter pairs that are genuinely easy to confuse - either by a learner
#: articulating from the wrong makhraj, or by a recognizer. Taken from the
#: confusions named in the brief plus the classical emphatic/plain pairs.
#: A substitution differing by exactly one of these is reported as a *possible*
#: pronunciation difference, never as a confirmed wrong word.
CONFUSABLE_LETTERS: frozenset[frozenset[str]] = frozenset(
    frozenset(pair)
    for pair in [
        ("ق", "ك"),
        ("ص", "س"),
        ("ط", "ت"),
        ("ح", "ه"),
        ("ع", "ء"),
        ("ع", "أ"),
        ("ذ", "ز"),
        ("ذ", "ظ"),
        ("ظ", "ز"),
        ("ض", "د"),
        ("ض", "ظ"),
        ("ث", "س"),
        ("غ", "خ"),
        ("ه", "ء"),
    ]
)


def is_confusable_pair(a: str, b: str) -> bool:
    """True when `a` and `b` differ in exactly one phonetically close letter.

    Length must match: an inserted or dropped letter makes a different word
    (رب vs ربك), not a mispronounced one.
    """
    if a == b or len(a) != len(b):
        return False
    diffs = [(x, y) for x, y in zip(a, b) if x != y]
    if len(diffs) != 1:
        return False
    return frozenset(diffs[0]) in CONFUSABLE_LETTERS


class ConfidencePolicy:
    def __init__(
        self,
        word_conf_min: float = 0.90,
        *,
        speech_end_tolerance: float = 0.25,
    ) -> None:
        self._min = word_conf_min
        self._tolerance = speech_end_tolerance

    # ── gates ────────────────────────────────────────────────────────────────

    def _suspect_reason(
        self, result: AlignmentResult, speech_end: float | None
    ) -> str | None:
        word = result.spoken_word
        if word is None:
            return None
        if word.confidence < self._min:
            return f"confidence {word.confidence:.2f} below gate {self._min:.2f}"
        if speech_end is not None and word.start_time > speech_end + self._tolerance:
            return (
                f"word starts at {word.start_time:.2f}s, after speech ended "
                f"at {speech_end:.2f}s"
            )
        return None

    # ── judgement ────────────────────────────────────────────────────────────

    def judge(
        self,
        results: list[AlignmentResult],
        *,
        tier: Tier = Tier.COMMITTED,
        speech_end: float | None = None,
    ) -> list[AlignmentResult]:
        return [self._judge_one(r, tier, speech_end) for r in results]

    def _judge_one(
        self, result: AlignmentResult, tier: Tier, speech_end: float | None
    ) -> AlignmentResult:
        suspect = self._suspect_reason(result, speech_end)

        # The provisional tier drives the live display. It may confirm that a word
        # was recognized - that is what lets the UI highlight progress as the
        # reciter speaks - but it may never assert a mistake. The asymmetry is
        # deliberate: failing to flag a real error merely delays feedback until
        # the committed tier catches it, whereas a false accusation is the harm
        # this system exists to prevent. Measured justification: the tiny model
        # emitted a wrong word (خلفهمون) at 0.90 confidence.
        if tier is Tier.PROVISIONAL:
            if result.status is WordStatus.CORRECT and not suspect:
                return replace(result, band=ConfidenceBand.HIGH_CONFIDENCE_CORRECT)
            return replace(
                result,
                band=ConfidenceBand.UNCERTAIN,
                category=None,
                suppressed_reason="provisional tier may not report mistakes",
            )

        if result.status is WordStatus.CORRECT:
            if suspect:
                return replace(
                    result, band=ConfidenceBand.UNCERTAIN, suppressed_reason=suspect
                )
            return replace(result, band=ConfidenceBand.HIGH_CONFIDENCE_CORRECT)

        if result.status is WordStatus.MISSING:
            # No spoken word to doubt. A word absent from a confident
            # transcription is the strongest evidence we get at this layer.
            return replace(
                result,
                band=ConfidenceBand.HIGH_CONFIDENCE_ERROR,
                category=ErrorCategory.WORD_MISSING,
            )

        if suspect:
            # The whole point: a low-confidence extra word is a hallucination
            # candidate, not the reciter's mistake.
            return replace(
                result, band=ConfidenceBand.UNCERTAIN, suppressed_reason=suspect
            )

        if result.status is WordStatus.EXTRA:
            return replace(
                result,
                band=ConfidenceBand.HIGH_CONFIDENCE_ERROR,
                category=ErrorCategory.WORD_EXTRA,
            )

        if result.status is WordStatus.REPEATED:
            return replace(
                result,
                band=ConfidenceBand.HIGH_CONFIDENCE_ERROR,
                category=ErrorCategory.WORD_REPETITION,
            )

        if result.status is WordStatus.SUBSTITUTED:
            expected = result.expected_word.normalized if result.expected_word else ""
            spoken = result.spoken_word.normalized if result.spoken_word else ""
            if is_confusable_pair(expected, spoken):
                # Could be articulation, could be the recognizer. We say so
                # rather than pretending to know which.
                return replace(
                    result,
                    band=ConfidenceBand.UNCERTAIN,
                    category=ErrorCategory.POSSIBLE_PRONUNCIATION_ERROR,
                    suppressed_reason=(
                        "single confusable letter - needs phoneme evidence to confirm"
                    ),
                )
            return replace(
                result,
                band=ConfidenceBand.HIGH_CONFIDENCE_ERROR,
                category=ErrorCategory.WORD_SUBSTITUTION,
            )

        return replace(result, band=ConfidenceBand.UNCERTAIN)
