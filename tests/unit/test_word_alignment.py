"""Mechanical alignment. No judgement about the reciter happens here."""

import pytest

from app.alignment.base import ExpectedWord, SpokenWord
from app.alignment.word_alignment import (
    GAP_COST,
    WordAligner,
    similarity,
    substitution_cost,
)
from app.models.enums import WordStatus


def E(words):
    return [ExpectedWord(w, w, i + 1, 1, 2) for i, w in enumerate(words)]


def S(words, conf=1.0):
    return [SpokenWord(w, w, i * 0.5, (i + 1) * 0.5, conf) for i, w in enumerate(words)]


@pytest.fixture
def aligner():
    return WordAligner()


def statuses(results):
    return [r.status for r in results]


class TestSimilarity:
    def test_identical_is_one(self):
        assert similarity("رب", "رب") == 1.0

    def test_unrelated_is_low(self):
        assert similarity("رب", "مالك") < 0.3

    def test_near_miss_is_high(self):
        assert similarity("رب", "ربك") > 0.7

    def test_substitution_cost_is_inverse_similarity(self):
        assert substitution_cost("رب", "رب") == 0.0
        assert substitution_cost("رب", "مالك") == pytest.approx(1.0)


class TestBasicAlignment:
    def test_perfect_recitation(self, aligner):
        words = ["الحمد", "لله", "رب", "العالمين"]
        results = aligner.align(E(words), S(words))
        assert statuses(results) == [WordStatus.CORRECT] * 4

    def test_substitution(self, aligner):
        results = aligner.align(
            E(["الحمد", "لله", "رب", "العالمين"]),
            S(["الحمد", "لله", "مالك", "العالمين"]),
        )
        assert statuses(results) == [
            WordStatus.CORRECT, WordStatus.CORRECT,
            WordStatus.SUBSTITUTED, WordStatus.CORRECT,
        ]
        assert results[2].expected_word.normalized == "رب"
        assert results[2].spoken_word.normalized == "مالك"

    def test_missing_word(self, aligner):
        results = aligner.align(
            E(["الحمد", "لله", "رب", "العالمين"]), S(["الحمد", "لله", "العالمين"])
        )
        assert WordStatus.MISSING in statuses(results)
        missing = next(r for r in results if r.status is WordStatus.MISSING)
        assert missing.expected_word.normalized == "رب"
        assert missing.spoken_word is None

    def test_extra_word(self, aligner):
        results = aligner.align(
            E(["قل", "هو", "الله", "أحد"]), S(["قل", "هو", "الله", "أحد", "قط"])
        )
        extra = next(r for r in results if r.status is WordStatus.EXTRA)
        assert extra.expected_word is None
        assert extra.spoken_word.normalized == "قط"

    def test_empty_spoken_makes_everything_missing(self, aligner):
        results = aligner.align(E(["قل", "هو"]), [])
        assert statuses(results) == [WordStatus.MISSING, WordStatus.MISSING]

    def test_empty_expected_makes_everything_extra(self, aligner):
        results = aligner.align([], S(["قل", "هو"]))
        assert all(r.status in (WordStatus.EXTRA, WordStatus.REPEATED) for r in results)

    def test_both_empty(self, aligner):
        assert aligner.align([], []) == []


class TestRepetition:
    @pytest.mark.parametrize(
        "expected,spoken",
        [
            (["الحمد", "لله", "رب"], ["الحمد", "الحمد", "لله", "رب"]),   # first
            (["الحمد", "لله", "رب"], ["الحمد", "لله", "لله", "رب"]),     # middle
            (["الحمد", "لله", "رب"], ["الحمد", "لله", "رب", "رب"]),      # last
        ],
    )
    def test_detected_at_any_position(self, aligner, expected, spoken):
        """The aligner may place a duplicate on either side of the match, so
        looking only backwards missed about half of these."""
        results = aligner.align(E(expected), S(spoken))
        assert WordStatus.REPEATED in statuses(results)

    def test_dissimilar_extra_is_not_a_repetition(self, aligner):
        results = aligner.align(
            E(["قل", "هو", "الله", "أحد"]), S(["قل", "هو", "الله", "أحد", "قط"])
        )
        assert WordStatus.REPEATED not in statuses(results)


class TestGapCost:
    def test_gap_is_cheaper_than_a_full_substitution(self):
        """But two gaps must cost more, so a wholly different word aligns as a
        substitution rather than 'missing + extra'."""
        assert GAP_COST < 1.0
        assert 2 * GAP_COST > 1.0

    def test_unrelated_word_becomes_a_substitution_not_two_gaps(self, aligner):
        results = aligner.align(E(["رب"]), S(["مالك"]))
        assert statuses(results) == [WordStatus.SUBSTITUTED]


class TestOrdering:
    def test_swapped_words_surface_as_missing_plus_extra(self, aligner):
        """Documents the representation: there is no dedicated 'reordered'
        status, a transposition appears as one omission and one insertion."""
        results = aligner.align(
            E(["الحمد", "لله", "رب"]), S(["لله", "الحمد", "رب"])
        )
        assert WordStatus.MISSING in statuses(results)
        assert {WordStatus.EXTRA, WordStatus.REPEATED} & set(statuses(results))


class TestHarakatAreInvisibleHere:
    def test_shadda_difference_aligns_as_correct(self, aligner):
        """رَبِّ and رَبَ both normalize to رب.

        Deliberate. Text ASR gives no reliable evidence about short vowels, so
        reporting a harakat mistake here would be a false correction. Harakat
        belongs to Phase 8 where phoneme evidence exists.
        """
        from app.quran.text_normalizer import normalize_for_asr_matching as N

        expected = [ExpectedWord("رَبِّ", N("رَبِّ"), 1, 1, 2)]
        spoken = [SpokenWord("رَبَ", N("رَبَ"), 0.0, 0.5, 1.0)]
        assert aligner.align(expected, spoken)[0].status is WordStatus.CORRECT
