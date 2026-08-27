"""Scenario B: locating a verse from recognized text alone."""

import pytest

from app.quran.repository import QuranRepository
from app.quran.verse_locator import SINGLE_WORD_MAX_OCCURRENCES, VerseLocator

repo = QuranRepository()


@pytest.fixture(scope="module")
def locator():
    loc = VerseLocator()
    loc.warm()
    return loc


class TestExactMatches:
    @pytest.mark.parametrize(
        "surah,ayah",
        [(1, 5), (2, 255), (36, 1), (112, 1), (114, 1)],
    )
    def test_finds_an_ayah_from_its_own_text(self, locator, surah, ayah):
        match = locator.locate_best(repo.ayah(surah, ayah).normalized)
        assert match is not None
        assert (match.surah, match.ayah) == (surah, ayah)
        assert match.confidence == 1.0
        assert match.word_index == 1

    def test_matched_text_is_canonical_uthmani(self, locator):
        match = locator.locate_best(repo.ayah(112, 1).normalized)
        assert match.matched_text == repo.ayah(112, 1).uthmani
        assert "ٱ" in match.matched_text, "must return Uthmani, not the normalized form"

    def test_reference_string(self, locator):
        assert locator.locate_best(repo.ayah(2, 255).normalized).reference == "2:255"


class TestNoMatch:
    def test_non_quran_text_returns_nothing(self, locator):
        assert locator.locate("this is not quran at all") == []
        assert locator.locate_best("hello world testing") is None

    def test_common_single_word_is_refused(self, locator):
        """الله occurs 2155 times - answering would be a guess dressed as an answer."""
        assert locator.locate("الله") == []
        assert locator.locate("من") == []

    def test_empty_query(self, locator):
        assert locator.locate("") == []
        assert locator.locate("   ") == []


class TestHallucinationTolerance:
    """Real ASR output ends in hallucinated words; the locator must cope."""

    @pytest.mark.parametrize(
        "text,surah,ayah",
        [
            ("قُلْ هُوَ اللَّهُ أَحَدٌ قُطْ قُطْ", 112, 1),
            ("قُلْ هُوَ اللَّهُ أَحَدٌ مُعْلَمُونَ وَالْمُسْلِمُ", 112, 1),
            ("الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ وَالْمُؤْمِنِينَ", 1, 2),
        ],
    )
    def test_still_finds_the_right_verse(self, locator, text, surah, ayah):
        match = locator.locate_best(text)
        assert match is not None
        assert (match.surah, match.ayah) == (surah, ayah)

    def test_tail_does_not_stretch_the_match_into_the_next_ayah(self, locator):
        """Regression: the span was computed from query length, so two
        hallucinated words made a 4-word ayah look like a cross-ayah run."""
        match = locator.locate_best("قُلْ هُوَ اللَّهُ أَحَدٌ قُطْ قُطْ")
        assert match.spans_multiple_ayat is False
        assert match.matched_word_count == 4
        assert match.query_word_count == 6


class TestLearnerErrors:
    """The users are learners. One wrong word must not mean 'verse not found'."""

    @pytest.mark.parametrize(
        "text,surah,ayah",
        [
            ("الحمد لله مالك العالمين", 1, 2),   # مالك for رب
            ("كل هو الله أحد", 112, 1),           # كل for قل
            ("الله لا إله إلا رب الحي القيوم", 2, 255),  # رب for هو
        ],
    )
    def test_substitution_still_locates_the_verse(self, locator, text, surah, ayah):
        """A substitution in a short verse destroys every trigram, so this only
        works because of the bigram fallback."""
        match = locator.locate_best(text)
        assert match is not None, "bigram fallback should have caught this"
        assert (match.surah, match.ayah) == (surah, ayah)
        assert match.confidence < 1.0
        assert match.matched_word_count < match.query_word_count


class TestSingleWordAyat:
    """28 ayat are one word - the muqatta'at. They must still be findable."""

    @pytest.mark.parametrize(
        "text,surah,ayah",
        [("يس", 36, 1), ("طه", 20, 1), ("كهيعص", 19, 1), ("ص", 38, 1)],
    )
    def test_rare_single_word_ayah_is_found(self, locator, text, surah, ayah):
        match = locator.locate_best(text)
        assert match is not None
        assert (match.surah, match.ayah) == (surah, ayah)
        assert match.is_ambiguous is False

    def test_repeated_single_word_ayah_reports_alternatives(self, locator):
        """الم opens six surahs; naming one without saying so would be wrong."""
        match = locator.locate_best("الم")
        assert match.alternatives == 5
        assert match.is_ambiguous

    def test_threshold_is_between_the_two_populations(self, locator):
        assert 6 <= SINGLE_WORD_MAX_OCCURRENCES < 2155


class TestPartialRecitation:
    """Streaming delivers the opening words before the verse is finished."""

    def test_three_words_is_enough_for_a_distinctive_opening(self, locator):
        match = locator.locate_best("الله لا إله")
        assert (match.surah, match.ayah) == (2, 255)

    def test_short_common_opening_is_reported_as_ambiguous(self, locator):
        """'الله لا' occurs all over the Quran - saying so is the correct answer."""
        match = locator.locate_best("الله لا")
        assert match is not None
        assert match.is_ambiguous
        assert match.alternatives > 10


class TestRepeatedVerses:
    def test_ar_rahman_refrain_reports_its_alternatives(self, locator):
        """فبأي آلاء ربكما تكذبان appears 31 times. Picking one silently would lie."""
        match = locator.locate_best("فبأي آلاء ربكما تكذبان")
        assert match.surah == 55
        assert match.alternatives == 30
        assert match.is_ambiguous

    def test_unique_verse_is_not_ambiguous(self, locator):
        assert locator.locate_best(repo.ayah(112, 1).normalized).is_ambiguous is False

    def test_top_k_returns_distinct_locations(self, locator):
        matches = locator.locate("الحمد لله رب العالمين", top_k=3)
        assert len(matches) == 3
        assert len({(m.surah, m.ayah) for m in matches}) == 3


class TestSurahHint:
    def test_hint_disambiguates_a_repeated_phrase(self, locator):
        match = locator.locate_best("الحمد لله رب العالمين", surah_hint=1)
        assert (match.surah, match.ayah) == (1, 2)
        assert match.is_ambiguous is False

    def test_hint_selects_a_different_valid_occurrence(self, locator):
        match = locator.locate_best("الحمد لله رب العالمين", surah_hint=10)
        assert match.surah == 10

    def test_impossible_hint_falls_back_rather_than_failing(self, locator):
        """A wrong hint must not make a findable verse unfindable."""
        match = locator.locate_best(repo.ayah(112, 1).normalized, surah_hint=2)
        assert match is not None
        assert (match.surah, match.ayah) == (112, 1)


class TestCrossAyah:
    def test_genuine_continuation_is_flagged(self, locator):
        text = repo.ayah(1, 6).normalized + " " + repo.ayah(1, 7).normalized
        match = locator.locate_best(text)
        assert (match.surah, match.ayah) == (1, 6)
        assert match.spans_multiple_ayat is True
        assert match.confidence == 1.0


class TestMidVerseStart:
    def test_reports_the_word_offset_within_the_ayah(self, locator):
        ayah = repo.ayah(2, 255)
        tail = " ".join(w.normalized for w in ayah.words[6:14])
        match = locator.locate_best(tail)
        assert (match.surah, match.ayah) == (2, 255)
        assert match.word_index == 7


class TestIndex:
    def test_stream_covers_every_word(self, locator):
        assert len(locator._stream) == 77800
        assert len(locator._positions) == len(locator._stream)

    def test_positions_align_with_the_repository(self, locator):
        pos = locator._positions[0]
        assert (pos.surah, pos.ayah, pos.word_index) == (1, 1, 1)
