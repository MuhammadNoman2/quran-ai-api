"""Expected phonetics for a verse: what *should* be recited.

Derived from the canonical Uthmani text with `quran-transcript`, which produces
the Quran Phonetic Script plus, for every phoneme group, ten articulation
attributes (sifat) and any Tajweed rules that apply.

This runs with no model, no torch and no GPU, so the reference side of
pronunciation analysis is available everywhere - including on machines where the
acoustic model cannot run at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from quran_transcript import Aya, MoshafAttributes, quran_phonetizer

from app.core.config import settings


@lru_cache(maxsize=1)
def moshaf_attributes() -> MoshafAttributes:
    """Recitation style. Madd lengths are a legitimate choice, not a constant.

    Hafs permits a range for several madd types, so a school teaching four counts
    and one teaching two are both correct. Getting this wrong would make the
    system flag correct recitation as wrong, which is why it is configuration
    rather than a hardcoded number.
    """
    return MoshafAttributes(
        rewaya=settings.rewaya,
        madd_monfasel_len=settings.madd_monfasel_len,
        madd_mottasel_len=settings.madd_mottasel_len,
        madd_mottasel_waqf=settings.madd_mottasel_waqf,
        madd_aared_len=settings.madd_aared_len,
    )


@dataclass(frozen=True)
class PhonemeWord:
    index: int
    """1-based word position, matching the Uthmani word order."""

    uthmani: str
    phonemes: str
    span: tuple[int, int]
    """Character span within the Uthmani ayah text."""


@dataclass(frozen=True)
class SifaEntry:
    phonemes: str
    attributes: dict[str, str]


@dataclass(frozen=True)
class PhoneticReference:
    surah: int
    ayah: int
    uthmani: str
    phonemes: str
    words: tuple[PhonemeWord, ...]
    sifat: tuple[SifaEntry, ...]
    mappings: tuple
    tajweed_rules: tuple[str, ...]
    raw: object = None
    """The phonetizer's own output object, with word spaces.

    The Muaalem model takes an output object rather than a phoneme string - it
    reads `.phonemes` off it - so the richer form has to survive as far as the
    recognizer instead of being flattened to text here.
    """

    model_phonemes: str = ""
    """The same phonemes without word spaces.

    The acoustic model requires this form: given a spaced reference its tokenizer
    fails outright ("Unable to create tensor…"), because the space-separated
    phoneme string no longer lines up with the per-group sifat list. Word spaces
    are kept in `phonemes` because that is what makes word attribution possible,
    so both representations are carried.
    """

    model_raw: object = None
    model_mappings: tuple = ()

    def word_at(self, uthmani_index: int) -> int | None:
        """Which word a character position in the Uthmani text belongs to."""
        for word in self.words:
            if word.span[0] <= uthmani_index < word.span[1]:
                return word.index
        return None


@lru_cache(maxsize=512)
def phonetics_for(surah: int, ayah: int) -> PhoneticReference:
    """Expected phonetics for one ayah. Cached - the text never changes."""
    uthmani = Aya(surah, ayah).get().uthmani
    attributes = moshaf_attributes()
    output = quran_phonetizer(uthmani, attributes, remove_spaces=False)
    unspaced = quran_phonetizer(uthmani, attributes, remove_spaces=True)

    # Uthmani words and phoneme words come out in the same order, so they pair up
    # positionally. Character spans come from scanning the original text.
    uthmani_words = uthmani.split()
    phoneme_words = output.phonemes.split(" ")
    spans: list[tuple[int, int]] = []
    cursor = 0
    for word in uthmani_words:
        start = uthmani.index(word, cursor)
        spans.append((start, start + len(word)))
        cursor = start + len(word)

    words = tuple(
        PhonemeWord(
            index=i + 1,
            uthmani=uthmani_words[i],
            phonemes=phoneme_words[i] if i < len(phoneme_words) else "",
            span=spans[i],
        )
        for i in range(len(uthmani_words))
    )

    sifat = tuple(
        SifaEntry(
            phonemes=entry.phonemes,
            attributes={
                field: getattr(entry, field)
                for field in (
                    "hams_or_jahr", "shidda_or_rakhawa", "tafkheem_or_taqeeq",
                    "itbaq", "safeer", "qalqla", "tikraar", "tafashie",
                    "istitala", "ghonna",
                )
            },
        )
        for entry in output.sifat
    )

    rules: list[str] = []
    for mapping in output.mappings:
        for rule in getattr(mapping, "tajweed_rules", None) or []:
            name = _rule_name(rule)
            if name and name not in rules:
                rules.append(name)

    return PhoneticReference(
        surah=surah,
        ayah=ayah,
        uthmani=uthmani,
        phonemes=output.phonemes,
        words=words,
        sifat=sifat,
        mappings=tuple(output.mappings),
        tajweed_rules=tuple(rules),
        raw=output,
        model_phonemes=unspaced.phonemes,
        model_raw=unspaced,
        model_mappings=tuple(unspaced.mappings),
    )


def _rule_name(rule: object) -> str | None:
    """English name of a Tajweed rule object, whatever shape it arrives in."""
    name = getattr(rule, "name", None)
    if name is None:
        return str(rule) if rule else None
    return getattr(name, "en", None) or getattr(name, "ar", None) or str(name)


def _rule_name_ar(rule: object) -> str | None:
    name = getattr(rule, "name", None)
    return getattr(name, "ar", None) if name is not None else None
