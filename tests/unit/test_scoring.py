"""The scoring formula. Deterministic and documented, per the brief."""

from app.alignment.base import AlignmentResult, ExpectedWord, SpokenWord
from app.models.enums import ConfidenceBand, ErrorCategory, WordStatus
from app.recitation.scoring import PENALTIES, score


def confirmed(category):
    return AlignmentResult(
        status=WordStatus.SUBSTITUTED,
        expected_word=ExpectedWord("رب", "رب", 1, 1, 2),
        spoken_word=SpokenWord("مالك", "مالك", 0, 1, 0.99),
        band=ConfidenceBand.HIGH_CONFIDENCE_ERROR,
        category=category,
    )


def correct():
    return AlignmentResult(
        status=WordStatus.CORRECT,
        expected_word=ExpectedWord("رب", "رب", 1, 1, 2),
        spoken_word=SpokenWord("رب", "رب", 0, 1, 1.0),
        band=ConfidenceBand.HIGH_CONFIDENCE_CORRECT,
    )


def uncertain(category=None):
    return AlignmentResult(
        status=WordStatus.EXTRA,
        spoken_word=SpokenWord("قط", "قط", 0, 1, 0.17),
        band=ConfidenceBand.UNCERTAIN,
        category=category,
    )


class TestFormula:
    def test_perfect_recitation_scores_100(self):
        assert score([correct()] * 4, 4).score == 100.0

    def test_no_results_scores_100(self):
        assert score([], 4).score == 100.0

    def test_penalties_match_the_documented_weights(self):
        assert PENALTIES[ErrorCategory.WORD_SUBSTITUTION] == 5.0
        assert PENALTIES[ErrorCategory.WORD_MISSING] == 5.0
        assert PENALTIES[ErrorCategory.WORD_EXTRA] == 2.0
        assert PENALTIES[ErrorCategory.WORD_REPETITION] == 1.0

    def test_single_substitution(self):
        assert score([correct()] * 3 + [confirmed(ErrorCategory.WORD_SUBSTITUTION)], 4).score == 95.0

    def test_missing_and_extra_combine(self):
        results = [
            confirmed(ErrorCategory.WORD_MISSING),
            confirmed(ErrorCategory.WORD_EXTRA),
        ]
        assert score(results, 4).score == 93.0

    def test_repetition_is_the_lightest_penalty(self):
        """Repeating a word is a stumble, not a misreading."""
        assert score([confirmed(ErrorCategory.WORD_REPETITION)], 4).score == 99.0

    def test_score_never_goes_below_zero(self):
        assert score([confirmed(ErrorCategory.WORD_MISSING)] * 50, 50).score == 0.0


class TestUncertaintyIsFree:
    def test_uncertain_findings_cost_nothing(self):
        """A reciter is never docked points because the recognizer was unsure."""
        assert score([correct()] * 4 + [uncertain()], 4).score == 100.0

    def test_possible_pronunciation_error_costs_nothing(self):
        results = [correct()] * 3 + [
            uncertain(ErrorCategory.POSSIBLE_PRONUNCIATION_ERROR)
        ]
        assert score(results, 4).score == 100.0
        assert PENALTIES[ErrorCategory.POSSIBLE_PRONUNCIATION_ERROR] == 0.0

    def test_uncertain_findings_are_counted_separately(self):
        breakdown = score([correct()] * 4 + [uncertain(), uncertain()], 4)
        assert breakdown.correct == 4
        assert breakdown.uncertain == 2


class TestBreakdown:
    def test_counts_each_category(self):
        results = [
            correct(),
            confirmed(ErrorCategory.WORD_SUBSTITUTION),
            confirmed(ErrorCategory.WORD_MISSING),
            confirmed(ErrorCategory.WORD_EXTRA),
            confirmed(ErrorCategory.WORD_REPETITION),
        ]
        b = score(results, 5)
        assert (b.correct, b.substituted, b.missing, b.extra, b.repeated) == (1, 1, 1, 1, 1)
        assert b.penalty == 13.0

    def test_formula_is_documented_in_the_output(self):
        assert "100" in score([], 1).formula
        assert "uncertain" in score([], 1).formula.lower()
