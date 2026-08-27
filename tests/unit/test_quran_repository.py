"""Integrity of the prepared Quran data asset.

The Phase 1 exit criterion is here: the canonical text must round-trip
byte-identical to the Tanzil source. The licence permits verbatim copies only,
so this is a legal requirement as much as a correctness one.
"""

import pytest
from quran_transcript import Aya

from app.models.schemas import SCHEMA_VERSION
from app.quran.repository import QuranRepository
from app.quran.text_normalizer import normalize_for_asr_matching

repo = QuranRepository()


class TestStructure:
    def test_114_surahs(self):
        assert len(repo.surahs()) == 114

    def test_6236_ayat(self):
        assert len(repo.all_ayat()) == 6236

    def test_surah_ayah_counts_are_consistent(self):
        for s in repo.surahs():
            assert len(repo.ayat_of(s.surah)) == s.ayah_count

    def test_known_surah_lengths(self):
        assert repo.surah(1).ayah_count == 7
        assert repo.surah(2).ayah_count == 286
        assert repo.surah(114).ayah_count == 6

    def test_schema_version_recorded(self):
        assert repo._asset.schema_version == SCHEMA_VERSION

    def test_attribution_present(self):
        """Tanzil CC-BY 3.0 requires attribution to travel with the text."""
        a = repo.attribution
        assert "Tanzil" in a and "tanzil.net" in a


class TestCanonicalTextIsUnmodified:
    """Phase 1 exit criterion."""

    def test_every_ayah_matches_tanzil_byte_for_byte(self):
        mismatches = []
        for aya in Aya(1, 1).get_ayat_after():
            f = aya.get()
            stored = repo.ayah(f.sura_idx, f.aya_idx)
            if stored.uthmani != f.uthmani:
                mismatches.append((f.sura_idx, f.aya_idx))
            if stored.imlaey != f.imlaey:
                mismatches.append((f.sura_idx, f.aya_idx, "imlaey"))
        assert mismatches == []

    def test_uthmani_orthography_is_not_flattened_to_imlaey(self):
        """The specific corruption to guard against.

        Uthmani is distinguished by alef wasla (U+0671) and dagger alef (U+0670).
        If a preparation bug ever wrote Imlaey into the uthmani field, those code
        points would vanish. Asserted by code point rather than by literal, since
        hand-typed Arabic literals differ in combining-mark order.
        """
        bismillah = repo.ayah(1, 1).uthmani
        assert "\u0671" in bismillah, "alef wasla missing - Uthmani was flattened"
        assert "\u0670" in bismillah, "dagger alef missing - Uthmani was flattened"
        assert repo.ayah(1, 1).imlaey.count("\u0671") == 0, "Imlaey must not contain alef wasla"

    def test_uthmani_and_imlaey_are_different_representations(self):
        differing = sum(1 for a in repo.all_ayat() if a.uthmani != a.imlaey)
        assert differing > 6000, "the two scripts should differ for nearly every ayah"


class TestWords:
    def test_total_word_count(self):
        assert sum(a.word_count for a in repo.all_ayat()) == 77800

    def test_indices_are_contiguous_and_1_based(self):
        for a in repo.all_ayat():
            assert [w.index for w in a.words] == list(range(1, len(a.words) + 1))

    def test_words_join_back_to_the_imlaey_ayah(self):
        for a in repo.all_ayat():
            assert " ".join(w.imlaey for w in a.words) == a.imlaey

    def test_normalized_field_matches_normalizer(self):
        a = repo.ayah(2, 255)
        for w in a.words:
            assert w.normalized == normalize_for_asr_matching(w.imlaey)

    def test_normalized_ayah_is_the_join_of_normalized_words(self):
        for a in repo.all_ayat():
            assert a.normalized == " ".join(w.normalized for w in a.words)

    def test_no_empty_normalized_words(self):
        """An empty token would silently corrupt alignment indices."""
        empty = [
            (a.surah, a.ayah, w.index)
            for a in repo.all_ayat()
            for w in a.words
            if not w.normalized
        ]
        assert empty == []


class TestUthmaniGrouping:
    def test_uthmani_index_is_non_decreasing_and_starts_at_1(self):
        for a in repo.all_ayat():
            idx = [w.uthmani_index for w in a.words]
            assert idx[0] == 1
            assert all(b - x in (0, 1) for x, b in zip(idx, idx[1:]))

    def test_group_count_equals_uthmani_word_count(self):
        for aya in Aya(1, 1).get_ayat_after():
            f = aya.get()
            stored = repo.ayah(f.sura_idx, f.aya_idx)
            assert max(w.uthmani_index for w in stored.words) == len(f.uthmani_words)

    def test_split_word_shares_one_uthmani_word(self):
        """One Uthmani word (yaa-ayyuha) maps to two Imlaey words."""
        a = repo.ayah(2, 21)
        assert a.words[0].uthmani_index == a.words[1].uthmani_index == 1
        assert a.words[0].uthmani == a.words[1].uthmani
        # the two Imlaey words are distinct and concatenate to the Uthmani word's rasm
        assert a.words[0].normalized != a.words[1].normalized
        assert a.words[0].normalized + a.words[1].normalized == "ياأيها"
        assert a.words[2].uthmani_index == 2

    def test_number_of_split_ayat_matches_the_corpus(self):
        """363 of 6236 ayat have more Imlaey words than Uthmani words."""
        split = [
            a for a in repo.all_ayat()
            if max(w.uthmani_index for w in a.words) != len(a.words)
        ]
        assert len(split) == 363

    def test_words_with_equal_counts_map_one_to_one(self):
        a = repo.ayah(1, 2)
        assert [w.uthmani_index for w in a.words] == [1, 2, 3, 4]


class TestLookups:
    def test_unknown_ayah_raises(self):
        with pytest.raises(KeyError):
            repo.ayah(1, 99)

    def test_unknown_surah_raises(self):
        with pytest.raises(KeyError):
            repo.surah(115)

    def test_surah_name(self):
        assert repo.surah(1).name == "الفاتحة"
