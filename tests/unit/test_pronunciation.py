"""Phoneme-level analysis.

The reference side needs no model and is tested directly. The recognized side is
driven by a scripted recognizer, so the comparison and reporting logic is
exercised exactly - without a 600M-parameter model, and without pretending one is
present when it is not.
"""

import numpy as np
import pytest

from app.pronunciation.analyzer import PronunciationAnalyzer
from app.pronunciation.base import (
    FindingKind,
    Operation,
    PhonemeRecognizer,
    ScriptedPhonemeRecognizer,
)
from app.pronunciation.comparator import compare
from app.pronunciation.muaalem import MuaalemRecognizer
from app.pronunciation.reference import phonetics_for

AUDIO = np.zeros(16_000, dtype=np.float32)


class TestReference:
    def test_produces_phonemes_for_an_ayah(self):
        reference = phonetics_for(112, 1)
        assert reference.phonemes
        assert reference.uthmani.startswith("قُلْ")

    def test_one_phoneme_word_per_uthmani_word(self):
        reference = phonetics_for(1, 2)
        assert len(reference.words) == 4
        assert [w.index for w in reference.words] == [1, 2, 3, 4]
        assert all(w.phonemes for w in reference.words)

    def test_attributes_a_character_position_to_a_word(self):
        reference = phonetics_for(1, 2)
        assert reference.word_at(reference.words[2].span[0]) == 3
        assert reference.word_at(0) == 1

    def test_position_outside_any_word_returns_none(self):
        assert phonetics_for(1, 2).word_at(10_000) is None

    def test_carries_ten_articulation_attributes(self):
        entry = phonetics_for(112, 1).sifat[0]
        assert len(entry.attributes) == 10
        for key in ("hams_or_jahr", "qalqla", "ghonna", "tafkheem_or_taqeeq"):
            assert key in entry.attributes

    def test_reports_the_tajweed_rules_present(self):
        rules = phonetics_for(1, 2).tajweed_rules
        assert "Normal Madd" in rules

    def test_is_cached(self):
        assert phonetics_for(1, 1) is phonetics_for(1, 1)


class TestComparator:
    def test_identical_phonemes_produce_no_findings(self):
        reference = phonetics_for(112, 1)
        assert compare(reference, reference.phonemes) == ()

    def test_empty_recognition_produces_no_findings(self):
        """Nothing heard is not the same as everything wrong."""
        assert compare(phonetics_for(112, 1), "") == ()

    def test_a_swapped_letter_is_an_articulation_finding(self):
        reference = phonetics_for(112, 1)
        findings = compare(reference, reference.phonemes.replace("ق", "ك", 1))
        assert len(findings) == 1
        assert findings[0].kind is FindingKind.ARTICULATION
        assert findings[0].operation is Operation.REPLACE
        assert findings[0].word_index == 1
        assert "ك" in findings[0].spoken_phonemes

    def test_a_dropped_shadda_is_a_tashkeel_finding(self):
        reference = phonetics_for(112, 1)
        findings = compare(reference, reference.phonemes.replace("لل", "ل", 1))
        assert findings[0].kind is FindingKind.TASHKEEL

    def test_a_shortened_madd_is_a_tajweed_finding_with_the_rule_named(self):
        reference = phonetics_for(1, 2)
        findings = compare(reference, reference.phonemes.replace("اا", "ا", 1))
        finding = findings[0]
        assert finding.kind is FindingKind.TAJWEED
        assert finding.is_tajweed
        assert finding.tajweed_rule == "Normal Madd"
        assert finding.tajweed_rule_ar
        assert finding.expected_length == 2
        assert finding.spoken_length == 1
        assert "count of 2" in finding.detail

    def test_a_lengthened_madd_is_also_caught(self):
        reference = phonetics_for(1, 2)
        findings = compare(reference, reference.phonemes.replace("اا", "ااا", 1))
        assert findings[0].kind is FindingKind.TAJWEED
        assert findings[0].spoken_length == 3

    def test_findings_are_attributed_to_words(self):
        reference = phonetics_for(1, 2)
        findings = compare(reference, reference.phonemes.replace("اا", "ا", 1))
        assert findings[0].word_index == 2

    def test_garbage_input_does_not_raise(self):
        """A pronunciation pass must never take down an analysis that worked."""
        assert isinstance(compare(phonetics_for(1, 2), "!!! not phonemes !!!"), tuple)


class TestAnalyzerWithARecognizer:
    def _analyzer(self, phonemes):
        return PronunciationAnalyzer(ScriptedPhonemeRecognizer(phonemes=phonemes))

    def test_clean_recitation_reports_no_findings(self):
        reference = phonetics_for(112, 1)
        report = self._analyzer(reference.phonemes).analyze(AUDIO, 112, 1)
        assert report.available
        assert report.findings == ()
        assert report.recognized_phonemes == reference.phonemes

    def test_a_mispronunciation_is_reported(self):
        reference = phonetics_for(112, 1)
        report = self._analyzer(reference.phonemes.replace("ق", "ك", 1)).analyze(AUDIO, 112, 1)
        assert report.available
        assert len(report.findings) == 1
        assert report.engine == "scripted"

    def test_tajweed_findings_are_separable(self):
        reference = phonetics_for(1, 2)
        report = self._analyzer(reference.phonemes.replace("اا", "ا", 1)).analyze(AUDIO, 1, 2)
        assert len(report.tajweed_findings) == 1

    def test_recognizer_failure_degrades_instead_of_raising(self):
        class Broken(ScriptedPhonemeRecognizer):
            def recognize(self, audio, sample_rate=16_000, *, reference=None):
                raise RuntimeError("model exploded")

        report = PronunciationAnalyzer(Broken()).analyze(AUDIO, 112, 1)
        assert report.available is False
        assert "model exploded" in report.unavailable_reason
        assert report.reference_phonemes, "the expected phonetics are still useful"


class TestAnalyzerWithoutARecognizer:
    def test_no_recognizer_configured(self):
        analyzer = PronunciationAnalyzer(None)
        assert analyzer.available is False
        assert "no phoneme recognizer" in analyzer.unavailable_reason

    def test_reference_is_still_returned(self):
        """`findings == []` with `available == False` means 'not checked'."""
        report = PronunciationAnalyzer(None).analyze(AUDIO, 1, 2)
        assert report.available is False
        assert report.findings == ()
        assert report.reference_phonemes
        assert report.unavailable_reason

    def test_reference_works_without_any_model(self):
        assert PronunciationAnalyzer(None).reference(2, 255).phonemes


class TestMuaalemAvailability:
    def test_reports_availability_without_raising(self):
        recognizer = MuaalemRecognizer()
        assert isinstance(recognizer.available, bool)

    def test_gives_a_reason_when_unavailable(self):
        recognizer = MuaalemRecognizer()
        if not recognizer.available:
            assert recognizer.unavailable_reason
            assert "torch" in recognizer.unavailable_reason.lower()

    def test_recognizing_while_unavailable_raises_clearly(self):
        recognizer = MuaalemRecognizer()
        if not recognizer.available:
            with pytest.raises(RuntimeError, match="unavailable"):
                recognizer.recognize(AUDIO)

    def test_is_a_real_recognizer(self):
        assert isinstance(MuaalemRecognizer(), PhonemeRecognizer)

    def test_load_is_safe_when_unavailable(self):
        MuaalemRecognizer().load()  # must not raise
