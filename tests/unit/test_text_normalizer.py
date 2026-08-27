"""Normalization behaviour, including the cases measured from real ASR output."""

import pytest
from quran_transcript import Aya

from app.quran.text_normalizer import (
    canonical_display_text,
    normalize_for_asr_matching,
    normalize_for_phoneme_comparison,
    normalize_for_search,
    words_for_asr_matching,
)

FATIHA_2_UTHMANI = "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ"
FATIHA_2_IMLAEY = "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"


class TestCanonical:
    def test_is_identity(self):
        assert canonical_display_text(FATIHA_2_UTHMANI) == FATIHA_2_UTHMANI

    def test_does_not_apply_unicode_normalization(self):
        """NFC/NFD would silently alter the Tanzil text, which its licence forbids."""
        for aya in [Aya(1, 1), Aya(2, 255), Aya(112, 1)]:
            t = aya.get().uthmani
            assert canonical_display_text(t) == t


class TestSearch:
    def test_strips_all_diacritics_and_unifies_alef(self):
        assert normalize_for_search(FATIHA_2_UTHMANI) == "الحمد لله رب العلمين"

    def test_unifies_alef_variants(self):
        for variant in "آأإٱ":
            assert normalize_for_search(variant + "بد") == "ابد"

    def test_maps_teh_marbuta_and_alef_maksura(self):
        assert normalize_for_search("رحمة") == "رحمه"
        assert normalize_for_search("علىٰ") == "علي"

    def test_drops_standalone_hamza(self):
        assert normalize_for_search("شيء") == "شي"

    def test_collapses_whitespace(self):
        assert normalize_for_search("  الحمد   لله  ") == "الحمد لله"


class TestAsrMatching:
    def test_imlaey_is_the_matching_representation_not_uthmani(self):
        """Documents a real, intentional limitation.

        Uthmani writes ٱلْعَـٰلَمِينَ with a dagger alef where Imlaey writes الْعَالَمِينَ
        with a full alef. Since dagger alef is dropped (the ASR drops it too),
        the two scripts do NOT reduce to the same string:

            uthmani -> العلمين
            imlaey  -> العالمين   <- what the ASR actually emits

        So alignment must always be fed the Imlaey text. QuranRepository builds
        `Word.normalized` from Imlaey for exactly this reason. This test exists to
        make the constraint fail loudly if anyone "fixes" the normalizer to make
        both scripts agree - doing so would break matching on 1:3 instead.
        """
        assert normalize_for_asr_matching(FATIHA_2_UTHMANI) == "الحمد لله رب العلمين"
        assert normalize_for_asr_matching(FATIHA_2_IMLAEY) == "الحمد لله رب العالمين"
        assert normalize_for_asr_matching(FATIHA_2_UTHMANI) != normalize_for_asr_matching(
            FATIHA_2_IMLAEY
        )

    def test_imlaey_form_matches_what_the_asr_emits(self):
        """The contract that actually matters."""
        measured_asr_output = "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"
        assert normalize_for_asr_matching(measured_asr_output) == normalize_for_asr_matching(
            FATIHA_2_IMLAEY
        )

    def test_preserves_hamza_distinctions(self):
        """Unlike search: ء/أ/إ carry contrasts pronunciation analysis needs."""
        assert normalize_for_asr_matching("أحد") == "أحد"
        assert normalize_for_asr_matching("إله") == "إله"
        assert normalize_for_asr_matching("شيء") == "شيء"

    def test_preserves_teh_marbuta(self):
        assert normalize_for_asr_matching("رحمة") == "رحمة"

    def test_alef_wasla_becomes_alef(self):
        assert normalize_for_asr_matching("ٱلْحَمْدُ") == "الحمد"

    def test_alef_maksura_becomes_yeh(self):
        """Uthmani ٱلَّذِى vs ASR الَّذِي - measured."""
        assert normalize_for_asr_matching("ٱلَّذِى") == normalize_for_asr_matching("الَّذِي")

    def test_dagger_alef_is_dropped_not_expanded(self):
        """Regression: mapping dagger alef to alef produced الرحمان and broke 1:3.

        The ASR emits الرحمن - the dagger alef simply vanishes.
        """
        assert normalize_for_asr_matching("الرَّحْمَٰنِ") == "الرحمن"
        assert normalize_for_asr_matching("إِلَٰهَ") == "إله"

    @pytest.mark.parametrize(
        "surah,ayah,hypothesis",
        [
            # Verbatim outputs measured from the CT2 Quran Whisper models.
            (1, 2, "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"),
            (1, 3, "الرَّحْمَنِ الرَّحِيمِ"),
            (112, 1, "قُلْ هُوَ اللَّهُ أَحَدٌ"),
        ],
    )
    def test_real_asr_output_matches_reference(self, surah, ayah, hypothesis):
        reference = normalize_for_asr_matching(Aya(surah, ayah).get().imlaey)
        assert normalize_for_asr_matching(hypothesis) == reference

    def test_hallucinated_tail_appears_only_as_extra_words(self):
        """Base model appends garbage; the real words must still align as a prefix."""
        hyp = words_for_asr_matching("قُلْ هُوَ اللَّهُ أَحَدٌ قُطْ قُطْ")
        ref = words_for_asr_matching(Aya(112, 1).get().imlaey)
        assert hyp[: len(ref)] == ref
        assert hyp[len(ref) :] == ["قط", "قط"]


class TestPhonemeComparison:
    def test_preserves_shadda(self):
        """رَبِّ vs رَبِ is exactly the contrast Phase 8 must detect."""
        assert normalize_for_phoneme_comparison("رَبِّ") != normalize_for_phoneme_comparison("رَبِ")

    def test_preserves_harakat_and_sukun(self):
        out = normalize_for_phoneme_comparison(FATIHA_2_UTHMANI)
        for mark in ("َ", "ِ", "ُ", "ْ"):
            assert mark in out

    def test_preserves_dagger_alef(self):
        assert "ٰ" in normalize_for_phoneme_comparison("الرَّحْمَٰنِ")

    def test_strips_only_tatweel_and_waqf_marks(self):
        assert "ـ" not in normalize_for_phoneme_comparison("الرحـــمن")


class TestOrdering:
    def test_aggressiveness_is_monotonic(self):
        """search <= asr_matching <= phoneme, in information retained."""
        t = Aya(2, 255).get().uthmani
        assert len(normalize_for_search(t)) <= len(normalize_for_asr_matching(t))
        assert len(normalize_for_asr_matching(t)) <= len(normalize_for_phoneme_comparison(t))
