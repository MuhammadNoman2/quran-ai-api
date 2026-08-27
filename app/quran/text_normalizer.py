"""Arabic/Quranic text normalization.

The canonical Quran text is immutable reference data (Tanzil, CC-BY 3.0, which
explicitly forbids modification). Nothing in this module ever mutates stored text -
every function returns a *derived* representation for a specific downstream purpose.

Four representations, four purposes:

    canonical_display_text()          identity - what the user sees
    normalize_for_search()            aggressive - human search, verse lookup
    normalize_for_asr_matching()      moderate - aligning ASR output to Quran words
    normalize_for_phoneme_comparison() minimal - keeps everything phonetics needs

The character classes below were derived by scanning all 6236 ayat of the bundled
Tanzil text (see scripts/inspect_charset.py), not from assumption: the Uthmani text
uses 62 distinct code points and the Imlaey text 46.
"""

from __future__ import annotations

import re
import unicodedata

# ── Character classes (measured from the corpus) ──────────────────────────────

#: Short vowels, tanween, shadda, sukun. U+064B..U+0652 contiguous.
HARAKAT = "".join(chr(c) for c in range(0x064B, 0x0653))

#: Combining marks that sit on a letter but are not short vowels.
#: U+0653 maddah, U+0654 hamza above, U+0655 hamza below.
HAMZA_MARKS = "ٕٓٔ"

#: Dagger / superscript alef. Represents a long /aa/ that is not written as a
#: full alef. Handled separately because it is a *vowel*, not an annotation.
SUPERSCRIPT_ALEF = "ٰ"

#: Tatweel / kashida - purely typographic elongation, never phonetic.
TATWEEL = "ـ"

#: Quranic annotation symbols: waqf marks, silent-letter markers (U+06DF small
#: high rounded zero), madd markers (U+06E5/U+06E6 small waw/yeh), sajda marks.
#: These carry recitation metadata but are not letters.
QURANIC_MARKS = "".join(chr(c) for c in range(0x06D6, 0x06EE))

_WS = re.compile(r"\s+")


def _collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _strip(text: str, chars: str) -> str:
    return text.translate({ord(c): None for c in chars})


# ── 1. Canonical ──────────────────────────────────────────────────────────────

def canonical_display_text(text: str) -> str:
    """Return the canonical text unchanged.

    Removes:   nothing.
    Preserves: everything, byte for byte.

    Exists so that "which representation is this?" is always answerable at the
    call site, and so the immutability requirement is visible in code rather
    than implied. Never apply Unicode normalization here - NFC/NFD would alter
    the Tanzil text.
    """
    return text


# ── 2. Search ─────────────────────────────────────────────────────────────────

def normalize_for_search(text: str) -> str:
    """Most aggressive form: a bare consonant skeleton.

    Removes:   all harakat, tanween, shadda, sukun, dagger alef, maddah/hamza
               combining marks, tatweel, all Quranic annotation symbols,
               all non-Arabic-letter characters.
    Unifies:   alef variants (آ أ إ ٱ) -> ا;  alef maksura ى -> ي;
               teh marbuta ة -> ه;  hamza carriers (ؤ ئ) -> و / ي;
               standalone hamza ء is dropped.
    Preserves: consonant identity and word boundaries only.

    Use for: human search boxes and VerseLocator candidate generation, where
    recall matters more than precision. Do NOT use for error detection - it
    deliberately destroys the hamza and vowel distinctions that a learner can
    get wrong.
    """
    text = _strip(text, HARAKAT + HAMZA_MARKS + SUPERSCRIPT_ALEF + TATWEEL + QURANIC_MARKS)
    text = text.translate(str.maketrans({
        "آ": "ا", "أ": "ا", "إ": "ا", "ٱ": "ا",
        "ى": "ي", "ة": "ه",
        "ؤ": "و", "ئ": "ي", "ء": None,
    }))
    return _collapse_ws(text)


# ── 3. ASR matching ───────────────────────────────────────────────────────────

def normalize_for_asr_matching(text: str) -> str:
    """Alignment form: what we compare ASR output against.

    Removes:   all harakat, tanween, shadda, sukun, tatweel, Quranic annotation
               symbols, maddah/hamza combining marks, and **dagger alef**.
    Unifies:   alef wasla ٱ -> ا  (ASR never emits ٱ - measured);
               alef maksura ى -> ي (Uthmani ٱلَّذِى vs ASR الَّذِي - measured);
               alef with madda آ -> ا.

    Dagger alef is *dropped*, not mapped to a full alef. This was determined
    empirically, not by inspection: the Imlaey text has already rewritten every
    dagger alef that is conventionally spelled with a full alef (Uthmani
    ٱلْعَـٰلَمِينَ -> Imlaey الْعَالَمِينَ). The 3218 that remain in Imlaey are exactly
    the ones written *without* an alef - ٱلرَّحْمَـٰنِ, إِلَـٰهَ - and the ASR omits them
    too, emitting الرحمن and إله. Mapping them to alef produced الرحمان and إلاه
    and broke matching on Surah al-Fatiha verse 3.
    Preserves: **hamza distinctions** (ء أ إ ؤ ئ) and teh marbuta ة.

    Deliberately less aggressive than normalize_for_search: collapsing hamza
    would erase the ء/ع contrast that pronunciation analysis needs to detect,
    and teh marbuta vs heh is a real recitation distinction.

    Feed this the **Imlaey** text, not the Uthmani text. Imlaey is already the
    orthography the ASR emits, so matching against it avoids a whole class of
    script-mismatch failures. Uthmani stays the display representation.
    """
    text = _strip(text, HARAKAT + HAMZA_MARKS + TATWEEL + QURANIC_MARKS)
    text = _strip(text, SUPERSCRIPT_ALEF)     # dropped, not mapped - see docstring
    text = text.translate(str.maketrans({
        "ٱ": "ا",           # alef wasla   -> alef
        "آ": "ا",           # alef madda   -> alef
        "ى": "ي",           # alef maksura -> yeh
    }))
    return _collapse_ws(text)


# ── 4. Phoneme comparison ─────────────────────────────────────────────────────

def normalize_for_phoneme_comparison(text: str) -> str:
    """Minimal form: everything phonetically meaningful is preserved.

    Removes:   tatweel (typographic only) and Quranic annotation symbols that
               are not vowels - waqf and sajda marks. Collapses whitespace.
    Preserves: **all harakat, tanween, shadda, sukun, dagger alef, maddah and
               hamza marks, and every letter form including alef variants.**

    Shadda distinguishes رَبِّ from رَبِ. Sukun and dagger alef determine madd
    length. Removing any of them would destroy the information Phase 8/9 exists
    to analyse, so this function removes almost nothing.

    This is a pass-through guard, not a phonetizer: actual grapheme-to-phoneme
    conversion is delegated to quran-transcript's QPS and quranic-phonemizer.
    """
    keep_vowel_marks = {SUPERSCRIPT_ALEF, "ۥ", "ۦ"}
    drop = "".join(c for c in QURANIC_MARKS if c not in keep_vowel_marks) + TATWEEL
    return _collapse_ws(_strip(text, drop))


# ── Word helpers ──────────────────────────────────────────────────────────────

def words_for_asr_matching(text: str) -> list[str]:
    """Split into normalized word tokens, dropping empties."""
    return [w for w in normalize_for_asr_matching(text).split(" ") if w]


def describe(text: str) -> dict[str, str]:
    """All four representations at once - for debugging and API introspection."""
    return {
        "canonical": canonical_display_text(text),
        "search": normalize_for_search(text),
        "asr_matching": normalize_for_asr_matching(text),
        "phoneme": normalize_for_phoneme_comparison(text),
    }


def unicode_report(text: str) -> list[tuple[str, str, str]]:
    """(char, 'U+XXXX', unicode name) for every distinct char. Diagnostics."""
    return [(c, f"U+{ord(c):04X}", unicodedata.name(c, "?")) for c in dict.fromkeys(text)]
