"""What we are willing to claim, and what we refuse to claim."""

import pytest

from app.alignment.base import AlignmentResult, ExpectedWord, SpokenWord
from app.asr.base import Tier
from app.models.enums import ConfidenceBand, ErrorCategory, WordStatus
from app.recitation.confidence import ConfidencePolicy, is_confusable_pair


def result(status, expected=None, spoken=None, conf=1.0, start=0.0, similarity=0.0):
    return AlignmentResult(
        status=status,
        expected_word=ExpectedWord(expected, expected, 1, 1, 2) if expected else None,
        spoken_word=SpokenWord(spoken, spoken, start, start + 0.4, conf) if spoken else None,
        confidence=conf,
        similarity=similarity,
    )


@pytest.fixture
def policy():
    return ConfidencePolicy(0.90)


class TestConfusablePairs:
    @pytest.mark.parametrize(
        "a,b", [("قل", "كل"), ("صراط", "سراط"), ("طه", "ته"), ("حمد", "همد")]
    )
    def test_single_close_letter_is_confusable(self, a, b):
        assert is_confusable_pair(a, b)

    @pytest.mark.parametrize(
        "a,b",
        [
            ("رب", "ربك"),      # insertion - a different word, not a slip
            ("رب", "مالك"),     # unrelated
            ("رب", "رب"),       # identical
            ("قل", "قم"),       # لام/ميم are not phonetically close
        ],
    )
    def test_others_are_not(self, a, b):
        assert not is_confusable_pair(a, b)

    def test_two_differing_letters_is_not_a_slip(self):
        assert not is_confusable_pair("قصد", "كسد")


class TestConfidenceGate:
    def test_high_confidence_correct_word(self, policy):
        judged = policy.judge([result(WordStatus.CORRECT, "رب", "رب", conf=1.0)])[0]
        assert judged.band is ConfidenceBand.HIGH_CONFIDENCE_CORRECT
        assert judged.category is None

    def test_low_confidence_correct_word_is_uncertain(self, policy):
        judged = policy.judge([result(WordStatus.CORRECT, "رب", "رب", conf=0.4)])[0]
        assert judged.band is ConfidenceBand.UNCERTAIN

    def test_low_confidence_extra_is_never_reported(self, policy):
        """The central defence: a hallucinated tail must not become an error."""
        judged = policy.judge([result(WordStatus.EXTRA, None, "قط", conf=0.17)])[0]
        assert judged.band is ConfidenceBand.UNCERTAIN
        assert judged.is_error is False
        assert "below gate" in judged.suppressed_reason

    def test_high_confidence_extra_is_reported(self, policy):
        judged = policy.judge([result(WordStatus.EXTRA, None, "زيادة", conf=0.99)])[0]
        assert judged.is_error
        assert judged.category is ErrorCategory.WORD_EXTRA


class TestSpeechBoundaryGate:
    def test_word_after_speech_ends_is_suppressed_despite_high_confidence(self, policy):
        """Catches the 0.86-0.87 hallucinations a threshold alone lets through."""
        judged = policy.judge(
            [result(WordStatus.EXTRA, None, "والمؤمنين", conf=0.99, start=9.0)],
            speech_end=5.0,
        )[0]
        assert judged.band is ConfidenceBand.UNCERTAIN
        assert "after speech ended" in judged.suppressed_reason

    def test_word_inside_speech_is_unaffected(self, policy):
        judged = policy.judge(
            [result(WordStatus.EXTRA, None, "زيادة", conf=0.99, start=2.0)],
            speech_end=5.0,
        )[0]
        assert judged.is_error

    def test_small_overhang_is_tolerated(self, policy):
        judged = policy.judge(
            [result(WordStatus.EXTRA, None, "زيادة", conf=0.99, start=5.1)],
            speech_end=5.0,
        )[0]
        assert judged.is_error


class TestCategories:
    def test_missing_word_is_reported_without_a_spoken_word_to_doubt(self, policy):
        judged = policy.judge([result(WordStatus.MISSING, "رب", None)])[0]
        assert judged.is_error
        assert judged.category is ErrorCategory.WORD_MISSING

    def test_unrelated_substitution_is_a_word_substitution(self, policy):
        judged = policy.judge([result(WordStatus.SUBSTITUTED, "رب", "مالك", conf=0.99)])[0]
        assert judged.category is ErrorCategory.WORD_SUBSTITUTION
        assert judged.is_error

    def test_confusable_substitution_is_only_a_possibility(self, policy):
        """قل -> كل could be the reciter or the recognizer. We say so."""
        judged = policy.judge([result(WordStatus.SUBSTITUTED, "قل", "كل", conf=0.99)])[0]
        assert judged.category is ErrorCategory.POSSIBLE_PRONUNCIATION_ERROR
        assert judged.band is ConfidenceBand.UNCERTAIN
        assert judged.is_error is False, "must not be asserted as a confirmed mistake"

    def test_repetition_category(self, policy):
        judged = policy.judge([result(WordStatus.REPEATED, None, "رب", conf=0.99)])[0]
        assert judged.category is ErrorCategory.WORD_REPETITION


class TestTierRules:
    def test_provisional_never_reports_a_mistake(self, policy):
        for status in (
            WordStatus.EXTRA, WordStatus.MISSING,
            WordStatus.SUBSTITUTED, WordStatus.REPEATED,
        ):
            judged = policy.judge(
                [result(status, "رب", "مالك", conf=0.99)], tier=Tier.PROVISIONAL
            )[0]
            assert judged.is_error is False
            assert judged.category is None

    def test_provisional_may_still_confirm_a_correct_word(self, policy):
        """Otherwise the live view could never highlight progress.

        The asymmetry is deliberate: a missed error is caught later by the
        committed tier, a false accusation is not recoverable.
        """
        judged = policy.judge(
            [result(WordStatus.CORRECT, "رب", "رب", conf=1.0)], tier=Tier.PROVISIONAL
        )[0]
        assert judged.band is ConfidenceBand.HIGH_CONFIDENCE_CORRECT

    def test_provisional_low_confidence_correct_stays_uncertain(self, policy):
        judged = policy.judge(
            [result(WordStatus.CORRECT, "رب", "رب", conf=0.3)], tier=Tier.PROVISIONAL
        )[0]
        assert judged.band is ConfidenceBand.UNCERTAIN


class TestConfigurableThreshold:
    def test_threshold_is_honoured(self):
        strict = ConfidencePolicy(0.99)
        lenient = ConfidencePolicy(0.50)
        r = [result(WordStatus.EXTRA, None, "زيادة", conf=0.95)]
        assert strict.judge(r)[0].is_error is False
        assert lenient.judge(r)[0].is_error is True
