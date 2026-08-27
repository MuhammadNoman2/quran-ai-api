"""The Tajweed rule registry.

Every rule carries an explicit **verification status**, because the brief is
emphatic that a rule which cannot be checked reliably must say so rather than
being quietly implemented as something weaker. Each status below was assigned by
testing what the phonetic comparator actually does when the rule is violated -
not by assuming.

The tests behind the statuses are in `tests/unit/test_tajweed.py`, and the
findings that set them are:

* Shortening a madd produces a `tajweed` finding that **names the rule** and
  gives both the expected and the heard count. -> MEASURED
* Failing an assimilation (saying a clear noon where ikhfa applies) produces a
  phoneme difference, but the engine does **not** attribute it to a rule. We
  correlate it by position instead, which is weaker evidence. -> POSITIONAL
* Removing the qalqalah marker produces **no finding at all**. Its performance
  cannot be verified here. -> OCCURRENCE_ONLY
* Ishmam is a rounding of the lips with no audible output, so no acoustic model
  can detect it. -> NOT_DETECTABLE
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verification(str, Enum):
    MEASURED = "measured"
    """The engine names the rule and compares counts. Strongest evidence."""

    POSITIONAL = "positional"
    """A phoneme difference falls where the rule applies. Reported as *may not
    have been performed*, never as a confirmed violation."""

    OCCURRENCE_ONLY = "occurrence_only"
    """We can say where the rule applies, but cannot check whether it was done."""

    NOT_DETECTABLE = "not_detectable"
    """No audible evidence exists. It will never be verifiable from audio."""


class Category(str, Enum):
    NOON_SAKIN = "noon_sakin"
    MEEM_SAKIN = "meem_sakin"
    MADD = "madd"
    QALQALAH = "qalqalah"
    GHUNNAH = "ghunnah"
    LAM = "lam"
    HEAVY_LIGHT = "heavy_light"
    IDGHAM = "idgham"
    HAMZA = "hamza"
    WAQF = "waqf"
    SPECIAL = "special"


@dataclass(frozen=True)
class TajweedRule:
    key: str
    name_en: str
    name_ar: str
    category: Category
    verification: Verification
    definition: str
    note: str | None = None

    @property
    def is_verifiable(self) -> bool:
        return self.verification in (Verification.MEASURED, Verification.POSITIONAL)


def _rule(key, en, ar, category, verification, definition, note=None) -> TajweedRule:
    return TajweedRule(key, en, ar, category, verification, definition, note)


M, P, O, N = (
    Verification.MEASURED,
    Verification.POSITIONAL,
    Verification.OCCURRENCE_ONLY,
    Verification.NOT_DETECTABLE,
)

RULES: dict[str, TajweedRule] = {
    r.key: r
    for r in [
        # ── Madd: the family with a defined count, so the only one we can measure
        _rule("madd_tabii", "Normal Madd", "المد الطبيعي", Category.MADD, M,
              "A natural lengthening of two counts on a madd letter with no hamza or sukun after it."),
        _rule("madd_munfasil", "Separate Madd", "المد المنفصل", Category.MADD, M,
              "A madd letter at the end of a word followed by a hamza beginning the next word."),
        _rule("madd_muttasil", "Connected Madd", "المد المتصل", Category.MADD, M,
              "A madd letter followed by a hamza within the same word. Obligatory lengthening."),
        _rule("madd_arid_lissukun", "Aared Madd", "المد العارض للسكون", Category.MADD, M,
              "A madd letter before a letter made silent by stopping at the end of a phrase."),
        _rule("madd_lazim", "Necessary Madd", "المد اللازم", Category.MADD, M,
              "A madd letter followed by an inherent sukun. Six counts, without exception."),
        _rule("madd_badal", "Substitute Madd", "مد البدل", Category.MADD, M,
              "A hamza followed by a madd letter, where the madd replaces an original hamza."),
        _rule("madd_iwad", "Compensatory Madd", "مد العوض", Category.MADD, M,
              "Lengthening that replaces tanween fath when stopping at the end of a word."),
        _rule("madd_leen", "Soft Madd", "مد اللين", Category.MADD, M,
              "Waw or yaa with sukun preceded by fatha, lengthened when stopping."),
        _rule("madd_silah", "Silah Madd", "مد الصلة", Category.MADD, M,
              "Lengthening of the attached pronoun haa between two voweled letters."),

        # ── Ghunnah: a nasal hold with a duration, so also measurable
        _rule("ghunnah_mushaddadah", "Doubled Ghunnah", "الغنة المشددة", Category.GHUNNAH, M,
              "A two-count nasal hold on noon or meem carrying a shadda.",
              "Detected by duration. The engine sometimes labels the hold as a madd, "
              "so the count is trustworthy while the rule name may not be."),

        # ── Noon sakin and tanween: assimilation, correlated by position
        _rule("izhar", "Izhar", "الإظهار", Category.NOON_SAKIN, P,
              "Noon sakin or tanween pronounced clearly before a throat letter."),
        _rule("idgham_bi_ghunnah", "Idgham with Ghunnah", "الإدغام بغنة", Category.NOON_SAKIN, P,
              "Noon sakin or tanween merged into the following letter, with nasalisation."),
        _rule("idgham_bila_ghunnah", "Idgham without Ghunnah", "الإدغام بغير غنة", Category.NOON_SAKIN, P,
              "Noon sakin or tanween merged into a following lam or raa, without nasalisation."),
        _rule("iqlab", "Iqlab", "الإقلاب", Category.NOON_SAKIN, P,
              "Noon sakin or tanween turned into a hidden meem before baa."),
        _rule("ikhfaa", "Ikhfa", "الإخفاء", Category.NOON_SAKIN, P,
              "Noon sakin or tanween pronounced between clarity and merging, with nasalisation."),

        # ── Meem sakin
        _rule("ikhfaa_shafawi", "Labial Ikhfa", "الإخفاء الشفوي", Category.MEEM_SAKIN, P,
              "Meem sakin hidden with nasalisation before baa."),
        _rule("idgham_shafawi", "Labial Idgham", "الإدغام الشفوي", Category.MEEM_SAKIN, P,
              "Meem sakin merged into a following meem, with nasalisation."),
        _rule("izhar_shafawi", "Labial Izhar", "الإظهار الشفوي", Category.MEEM_SAKIN, P,
              "Meem sakin pronounced clearly before any letter other than baa or meem."),

        # ── Other idgham
        _rule("idgham_mutamathilayn", "Idgham of Identicals", "إدغام المتماثلين", Category.IDGHAM, P,
              "Two identical letters merged, the first silent."),
        _rule("idgham_mutajanisayn_kamil", "Complete Idgham of Kindred", "إدغام المتجانسين الكامل", Category.IDGHAM, P,
              "Letters sharing an articulation point merged completely."),
        _rule("idgham_mutajanisayn_naqis", "Partial Idgham of Kindred", "إدغام المتجانسين الناقص", Category.IDGHAM, P,
              "Letters sharing an articulation point merged while one attribute remains."),
        _rule("idgham_mutaqaribayn", "Idgham of Near Letters", "إدغام المتقاربين", Category.IDGHAM, P,
              "Letters close in articulation merged together."),

        # ── Lam
        _rule("lam_shamsiyyah", "Solar Lam", "اللام الشمسية", Category.LAM, P,
              "The lam of the definite article is silent and merges into the following letter."),
        _rule("lam_qamariyyah", "Lunar Lam", "اللام القمرية", Category.LAM, P,
              "The lam of the definite article is pronounced clearly."),

        # ── Heaviness
        _rule("tafkheem", "Tafkheem", "التفخيم", Category.HEAVY_LIGHT, P,
              "A letter pronounced heavily, filling the mouth.",
              "The phonetic script records heaviness as an attribute, so a difference "
              "is visible, but it is not attributed to this rule by name."),
        _rule("tarqeeq", "Tarqeeq", "الترقيق", Category.HEAVY_LIGHT, P,
              "A letter pronounced lightly."),

        # ── Qalqalah: located, but a violation produced no finding when tested
        _rule("qalqala_sughra", "Minor Qalqalah", "القلقلة الصغرى", Category.QALQALAH, O,
              "A slight echo on a qalqalah letter carrying sukun in the middle of a word.",
              "Occurrence is reliable. Performance is NOT verified: removing the "
              "qalqalah marker from the phoneme string produced no finding at all."),
        _rule("qalqala_kubra", "Major Qalqalah", "القلقلة الكبرى", Category.QALQALAH, O,
              "A stronger echo on a qalqalah letter made silent by stopping.",
              "Occurrence only, for the same measured reason as minor qalqalah."),
        _rule("qalqala_akbar", "Greatest Qalqalah", "القلقلة الأكبر", Category.QALQALAH, O,
              "The strongest echo, on a qalqalah letter with shadda at a stop.",
              "Occurrence only."),

        # ── Hamzat wasl
        _rule("hamza_wasl_silent", "Silent Hamzat Wasl", "همزة الوصل الساقطة", Category.HAMZA, P,
              "The connecting hamza is dropped when the word is joined to what precedes it."),
        _rule("hamza_wasl_fatha", "Hamzat Wasl with Fatha", "همزة الوصل بالفتح", Category.HAMZA, P,
              "The connecting hamza takes a fatha when beginning with it."),
        _rule("hamza_wasl_kasra", "Hamzat Wasl with Kasra", "همزة الوصل بالكسر", Category.HAMZA, P,
              "The connecting hamza takes a kasra when beginning with it."),
        _rule("hamza_wasl_damma", "Hamzat Wasl with Damma", "همزة الوصل بالضم", Category.HAMZA, P,
              "The connecting hamza takes a damma when beginning with it."),
        _rule("ibdal_hamza", "Hamza Substitution", "إبدال الهمزة", Category.HAMZA, P,
              "A hamza replaced by a madd letter."),

        # ── Waqf
        _rule("waqf_diacritic_drop", "Stopping Silence", "السكون عند الوقف", Category.WAQF, P,
              "The final diacritic is dropped when stopping on a word."),
        _rule("waqf_taa_marbuta", "Taa Marbuta at a Stop", "التاء المربوطة عند الوقف", Category.WAQF, P,
              "Taa marbuta is pronounced as haa when stopping."),
        _rule("waqf_silah_drop", "Silah Dropped at a Stop", "حذف الصلة عند الوقف", Category.WAQF, P,
              "The lengthening of the attached pronoun is dropped when stopping."),
        _rule("pausal_alif", "Pausal Alif", "ألف الوقف", Category.WAQF, P,
              "An alif pronounced only when stopping."),

        # ── Special
        _rule("imala", "Imala", "الإمالة", Category.SPECIAL, O,
              "A fatha inclined towards a kasra. Occurs once in Hafs, in 11:41.",
              "Occurrence only: a single instance, and no violation test exists for it."),
        _rule("ishmam", "Ishmam", "الإشمام", Category.SPECIAL, N,
              "The lips are rounded without producing sound, in 12:11.",
              "NOT DETECTABLE by any acoustic model, by definition - it makes no sound. "
              "The Muaalem paper excludes it for the same reason."),
        _rule("tashil", "Tashil", "التسهيل", Category.SPECIAL, O,
              "A hamza eased between hamza and alif. Occurs once in Hafs, in 41:44.",
              "Occurrence only."),
        _rule("iltiqa_haraka", "Vowel Meeting", "التقاء الساكنين", Category.SPECIAL, P,
              "A vowel inserted where two silent letters would otherwise meet."),
        _rule("iltiqa_shortening", "Shortening at a Meeting", "حذف حرف المد", Category.SPECIAL, P,
              "A madd letter shortened where two silent letters meet."),
        _rule("orthographic_silence", "Silent Letter", "الحرف غير المنطوق", Category.SPECIAL, P,
              "A letter written but not pronounced."),
    ]
}


def get(key: str) -> TajweedRule | None:
    return RULES.get(key)


def by_verification(status: Verification) -> list[TajweedRule]:
    return [r for r in RULES.values() if r.verification is status]


def summary() -> dict[str, int]:
    counts: dict[str, int] = {}
    for rule in RULES.values():
        counts[rule.verification.value] = counts.get(rule.verification.value, 0) + 1
    return counts
