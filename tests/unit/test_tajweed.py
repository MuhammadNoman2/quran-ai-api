"""The Tajweed registry and analyzer.

The statuses in `rules.py` are claims about what this system can detect, so the
tests that matter most here are the ones proving those claims - including the
negative ones.
"""

import numpy as np
import pytest

from app.pronunciation.analyzer import PronunciationAnalyzer
from app.pronunciation.base import ScriptedPhonemeRecognizer
from app.pronunciation.comparator import compare
from app.pronunciation.reference import phonetics_for
from app.tajweed.analyzer import TajweedAnalyzer
from app.tajweed.rules import RULES, Verification, by_verification, get, summary

AUDIO = np.zeros(16_000, dtype=np.float32)


@pytest.fixture(scope="module")
def analyzer():
    return TajweedAnalyzer()


class TestRegistry:
    def test_covers_the_classical_rules_the_brief_names(self):
        for key in (
            "madd_tabii", "ghunnah_mushaddadah", "qalqala_sughra",
            "idgham_bi_ghunnah", "ikhfaa", "iqlab", "izhar",
            "izhar_shafawi", "waqf_diacritic_drop",
        ):
            assert get(key) is not None, f"{key} missing from the registry"

    def test_every_rule_is_fully_described(self):
        for rule in RULES.values():
            assert rule.name_en and rule.name_ar and rule.definition
            assert rule.category and rule.verification

    def test_verification_statuses_are_all_used(self):
        counts = summary()
        for status in Verification:
            assert status.value in counts

    def test_rules_that_cannot_be_verified_explain_themselves(self):
        for status in (Verification.OCCURRENCE_ONLY, Verification.NOT_DETECTABLE):
            for rule in by_verification(status):
                assert rule.note, f"{rule.key} claims {status.value} without saying why"

    def test_ishmam_is_marked_undetectable(self):
        """It is a rounding of the lips with no sound. No model can hear it."""
        assert get("ishmam").verification is Verification.NOT_DETECTABLE
        assert get("ishmam").is_verifiable is False

    def test_qalqalah_is_occurrence_only(self):
        """Measured: removing the qalqalah marker produces no finding at all."""
        for key in ("qalqala_sughra", "qalqala_kubra", "qalqala_akbar"):
            assert get(key).verification is Verification.OCCURRENCE_ONLY

    def test_madd_rules_are_measured(self):
        for key in ("madd_tabii", "madd_munfasil", "madd_muttasil", "madd_arid_lissukun"):
            assert get(key).verification is Verification.MEASURED


class TestClaimsAreTrue:
    """These prove the statuses rather than trusting them."""

    def test_a_shortened_madd_really_is_named_and_measured(self):
        reference = phonetics_for(1, 2)
        findings = compare(reference, reference.phonemes.replace("اا", "ا", 1))
        assert findings and findings[0].tajweed_rule == "Normal Madd"
        assert findings[0].expected_length and findings[0].spoken_length

    def test_removing_qalqalah_really_produces_nothing(self):
        """The evidence for marking qalqalah occurrence-only. If this ever starts
        producing a finding, the status in rules.py should be revisited."""
        reference = phonetics_for(112, 1)
        assert "ڇ" in reference.phonemes, "expected a qalqalah marker in this verse"
        assert compare(reference, reference.phonemes.replace("ڇ", "")) == ()

    def test_an_assimilation_failure_is_visible_but_unnamed(self):
        """Which is exactly why those rules are positional, not measured."""
        reference = phonetics_for(2, 3)
        findings = compare(reference, reference.phonemes.replace("ں", "ن", 1))
        assert findings
        assert not any(f.tajweed_rule == "Ikhfa" for f in findings)


class TestOccurrences:
    def test_finds_rules_in_a_short_verse(self, analyzer):
        occurrences = analyzer.occurrences(112, 1)
        keys = {o.rule.key for o in occurrences}
        assert "qalqala_kubra" in keys, "أَحَدٌ ends in a qalqalah letter at a stop"
        assert "lam_shamsiyyah" in keys, "ٱللَّهُ has a solar lam"

    def test_occurrences_are_attributed_to_words(self, analyzer):
        for occurrence in analyzer.occurrences(1, 2):
            assert occurrence.word_indices
            assert occurrence.word_text

    def test_word_indices_are_imlaey_based(self, analyzer):
        """Analysis results index by Imlaey word, so occurrences must too."""
        from app.quran.repository import QuranRepository

        ayah = QuranRepository().ayah(2, 21)   # one Uthmani word, two Imlaey words
        valid = {w.index for w in ayah.words}
        for occurrence in analyzer.occurrences(2, 21):
            assert set(occurrence.word_indices) <= valid

    def test_counts_are_reported(self, analyzer):
        counts = analyzer.report(1, 2).rule_counts
        assert counts.get("madd_tabii", 0) >= 1

    def test_report_without_audio_is_not_marked_verified(self, analyzer):
        report = analyzer.report(1, 2)
        assert report.verified is False
        assert report.checks == ()
        assert report.unavailable_reason


class TestVerification:
    def _pronunciation(self, surah, ayah, mutate):
        reference = phonetics_for(surah, ayah)
        recognizer = ScriptedPhonemeRecognizer(phonemes=mutate(reference.phonemes))
        return PronunciationAnalyzer(recognizer).analyze(AUDIO, surah, ayah)

    def test_clean_recitation_marks_verifiable_rules_kept(self, analyzer):
        report = analyzer.verify(1, 2, self._pronunciation(1, 2, lambda p: p))
        assert report.verified
        statuses = {c.status for c in report.checks}
        assert "kept" in statuses
        assert "missed" not in statuses

    def test_a_shortened_madd_is_reported_as_missed(self, analyzer):
        report = analyzer.verify(
            1, 2, self._pronunciation(1, 2, lambda p: p.replace("اا", "ا", 1))
        )
        missed = [c for c in report.checks if c.status == "missed"]
        assert missed
        assert missed[0].confidence == "measured"
        assert "count" in missed[0].detail

    def test_qalqalah_is_never_checked_even_with_audio(self, analyzer):
        report = analyzer.verify(112, 1, self._pronunciation(112, 1, lambda p: p))
        qalqalah = [c for c in report.checks if c.occurrence.rule.category.value == "qalqalah"]
        assert qalqalah
        for check in qalqalah:
            assert check.status == "not_checked"
            assert check.confidence == "none"

    def test_positional_rules_are_only_possibly_missed(self, analyzer):
        """Never 'missed'. A phoneme difference in the right word is suggestive."""
        report = analyzer.verify(
            2, 3, self._pronunciation(2, 3, lambda p: p.replace("ں", "ن", 1))
        )
        for check in report.checks:
            if check.confidence == "positional":
                assert check.status in ("kept", "possibly_missed")

    def test_without_a_phoneme_model_nothing_is_verified(self, analyzer):
        report = analyzer.verify(1, 2, PronunciationAnalyzer(None).analyze(AUDIO, 1, 2))
        assert report.verified is False
        assert report.checks == ()
        assert report.unavailable_reason
        assert report.occurrences, "occurrences are still available without a model"
