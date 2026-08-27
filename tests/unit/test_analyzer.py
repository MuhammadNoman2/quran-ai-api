"""End-to-end analysis with a fake engine, so behaviour is exact and repeatable."""

import numpy as np
import pytest

from app.asr.base import FakeASREngine, Tier
from app.models.enums import ErrorCategory
from app.quran.repository import QuranRepository
from app.recitation.analyzer import RecitationAnalyzer, VerseNotIdentified

repo = QuranRepository()
AUDIO = np.zeros(16_000 * 3, dtype=np.float32)

FATIHA_2 = "الحمد لله رب العالمين"
IKHLAS_1 = "قل هو الله أحد"


def analyzer(text, probabilities=None, tier=Tier.COMMITTED):
    engine = FakeASREngine(
        text=text, word_probabilities=probabilities or {}, tier=tier
    )
    engine.load()
    return RecitationAnalyzer(engine)


class TestPerfectRecitation:
    def test_scores_100_with_no_errors(self):
        result = analyzer(FATIHA_2).analyze(AUDIO, surah=1, ayah=2)
        assert result.word_accuracy_score == 100.0
        assert result.errors == []
        assert len(result.words) == 4

    def test_reports_canonical_uthmani_as_expected_text(self):
        result = analyzer(FATIHA_2).analyze(AUDIO, surah=1, ayah=2)
        assert result.expected_text == repo.ayah(1, 2).uthmani
        assert "ٱ" in result.expected_text


class TestFalseCorrectionDefence:
    """The failure mode the brief calls extremely harmful. These are the guards."""

    @pytest.mark.parametrize(
        "text,probs,surah,ayah",
        [
            (f"{IKHLAS_1} قط قط", {"قط": 0.17}, 112, 1),
            (f"{FATIHA_2} الحمد", {"الحمد": 0.46}, 1, 2),
            (f"{IKHLAS_1} معلمون والمسلم", {"معلمون": 0.67, "والمسلم": 0.75}, 112, 1),
        ],
    )
    def test_hallucinated_tail_never_becomes_an_error(self, text, probs, surah, ayah):
        result = analyzer(text, probs).analyze(AUDIO, surah=surah, ayah=ayah)
        assert result.errors == [], f"false correction: {[e.spoken for e in result.errors]}"
        assert result.word_accuracy_score == 100.0

    def test_suppression_reason_is_recorded(self):
        result = analyzer(f"{IKHLAS_1} قط", {"قط": 0.17}).analyze(AUDIO, surah=112, ayah=1)
        suppressed = [w for w in result.words if w.note]
        assert suppressed
        assert "below gate" in suppressed[0].note

    def test_low_confidence_never_penalises_the_score(self):
        result = analyzer(FATIHA_2, {"رب": 0.3}).analyze(AUDIO, surah=1, ayah=2)
        assert result.word_accuracy_score == 100.0
        assert result.errors == []


class TestRealErrorsAreStillCaught:
    """Suppressing hallucinations must not suppress genuine mistakes."""

    def test_missing_word(self):
        result = analyzer("الحمد لله العالمين").analyze(AUDIO, surah=1, ayah=2)
        assert len(result.errors) == 1
        assert result.errors[0].category == ErrorCategory.WORD_MISSING.value
        assert result.word_accuracy_score == 95.0

    def test_substituted_word(self):
        result = analyzer("الحمد لله مالك العالمين").analyze(AUDIO, surah=1, ayah=2)
        assert len(result.errors) == 1
        assert result.errors[0].category == ErrorCategory.WORD_SUBSTITUTION.value
        assert result.errors[0].expected == repo.ayah(1, 2).words[2].uthmani

    def test_confident_extra_word(self):
        result = analyzer(f"{IKHLAS_1} زيادة").analyze(AUDIO, surah=112, ayah=1)
        assert len(result.errors) == 1
        assert result.errors[0].category == ErrorCategory.WORD_EXTRA.value

    def test_repeated_word(self):
        result = analyzer("قل هو هو الله أحد").analyze(AUDIO, surah=112, ayah=1)
        assert len(result.errors) == 1
        assert result.errors[0].category == ErrorCategory.WORD_REPETITION.value
        assert result.word_accuracy_score == 99.0


class TestObservations:
    def test_confusable_substitution_is_an_observation_not_an_error(self):
        """كل for قل: reported, but not asserted as a confirmed mistake."""
        result = analyzer("كل هو الله أحد").analyze(AUDIO, surah=112, ayah=1)
        assert result.errors == []
        assert len(result.observations) == 1
        assert (
            result.observations[0].category
            == ErrorCategory.POSSIBLE_PRONUNCIATION_ERROR.value
        )
        assert result.word_accuracy_score == 100.0


class TestVerseDetection:
    def test_infers_the_verse_when_not_supplied(self):
        result = analyzer(repo.ayah(2, 255).normalized).analyze(AUDIO)
        assert (result.surah, result.ayah) == (2, 255)
        assert result.verse_detected is True
        assert result.verse_confidence is not None

    def test_supplied_verse_is_not_marked_as_detected(self):
        result = analyzer(FATIHA_2).analyze(AUDIO, surah=1, ayah=2)
        assert result.verse_detected is False
        assert result.verse_confidence is None

    def test_unidentifiable_audio_raises_rather_than_guessing(self):
        with pytest.raises(VerseNotIdentified):
            analyzer("hello world this is not quran").analyze(AUDIO)


class TestTierEnforcement:
    def test_provisional_tier_reports_no_errors_even_for_real_mistakes(self):
        result = analyzer(
            "الحمد لله مالك العالمين", tier=Tier.PROVISIONAL
        ).analyze(AUDIO, surah=1, ayah=2)
        assert result.errors == []

    def test_committed_tier_reports_the_same_mistake(self):
        result = analyzer("الحمد لله مالك العالمين").analyze(AUDIO, surah=1, ayah=2)
        assert len(result.errors) == 1


class TestUndetectableCategoriesAreImpossible:
    def test_phoneme_and_tajweed_errors_are_never_emitted(self):
        """Phases 8 and 9 do not exist. Claiming them would be a lie."""
        for text in (FATIHA_2, "الحمد لله مالك العالمين", f"{FATIHA_2} زيادة"):
            result = analyzer(text).analyze(AUDIO, surah=1, ayah=2)
            categories = {w.category for w in result.words}
            assert ErrorCategory.PHONEME_ERROR.value not in categories
            assert ErrorCategory.TAJWEED_ERROR.value not in categories


class TestResponseShape:
    def test_matches_the_documented_contract(self):
        result = analyzer(FATIHA_2).analyze(AUDIO, surah=1, ayah=2)
        assert result.schema_version == 1
        assert result.session_id
        assert result.score_formula
        assert result.engine == {"engine": "fake", "tier": "committed"}

    def test_score_is_labelled_word_accuracy_not_tajweed(self):
        result = analyzer(FATIHA_2).analyze(AUDIO, surah=1, ayah=2)
        assert hasattr(result, "word_accuracy_score")
        assert not hasattr(result, "tajweed_score")

    def test_session_id_is_unique_per_analysis(self):
        a = analyzer(FATIHA_2)
        assert a.analyze(AUDIO, surah=1, ayah=2).session_id != a.analyze(
            AUDIO, surah=1, ayah=2
        ).session_id
