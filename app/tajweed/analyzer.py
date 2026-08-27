"""Tajweed analysis: where rules apply, and - where possible - whether they were kept.

Two things are deliberately separate, because they have very different evidence
behind them.

**Occurrence** is derived from the canonical text by `quranic-phonemizer`. It is
reliable, needs no model, and is useful on its own: "this verse contains three
ikhfa and two connected madd" is a teaching aid.

**Performance** requires phoneme evidence, and even then the confidence varies by
rule (see `rules.py`). A madd is measured against its expected count. An
assimilation is only correlated by position - we say it *may* not have been
performed, never that it was wrong. Qalqalah cannot be checked at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from app.pronunciation.base import PronunciationFinding, PronunciationReport
from app.quran.repository import QuranRepository
from app.tajweed.rules import RULES, TajweedRule, Verification

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleOccurrence:
    rule: TajweedRule
    uthmani_word_index: int
    """1-based Uthmani word, as the text engine counts words."""

    word_indices: tuple[int, ...]
    """The Imlaey word indices this covers - what analysis results use."""

    word_text: str


@dataclass(frozen=True)
class RuleCheck:
    """Whether one occurrence appears to have been performed."""

    occurrence: RuleOccurrence
    status: str
    """kept | possibly_missed | missed | not_checked"""

    confidence: str
    """measured | positional | none"""

    detail: str | None = None


@dataclass(frozen=True)
class TajweedReport:
    surah: int
    ayah: int
    occurrences: tuple[RuleOccurrence, ...]
    checks: tuple[RuleCheck, ...] = ()
    verified: bool = False
    unavailable_reason: str | None = None

    @property
    def rule_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for occurrence in self.occurrences:
            counts[occurrence.rule.key] = counts.get(occurrence.rule.key, 0) + 1
        return counts


class TajweedAnalyzer:
    def __init__(self, repository: QuranRepository | None = None) -> None:
        self._repo = repository or QuranRepository()

    # ── occurrence ───────────────────────────────────────────────────────────

    def occurrences(self, surah: int, ayah: int) -> tuple[RuleOccurrence, ...]:
        """Where every rule applies in this verse. Always available."""
        return _occurrences(surah, ayah, self._repo)

    def report(self, surah: int, ayah: int) -> TajweedReport:
        return TajweedReport(
            surah=surah,
            ayah=ayah,
            occurrences=self.occurrences(surah, ayah),
            verified=False,
            unavailable_reason="no audio was analysed - occurrences only",
        )

    # ── performance ──────────────────────────────────────────────────────────

    def verify(self, surah: int, ayah: int, pronunciation: PronunciationReport) -> TajweedReport:
        """Correlate phoneme findings with rule occurrences.

        A rule is only ever reported as *missed* when the phonetic engine named it
        and compared counts. Everything else is `possibly_missed` at best - a
        phoneme difference landed where the rule applies, which is suggestive, not
        conclusive.
        """
        occurrences = self.occurrences(surah, ayah)

        if not pronunciation.available:
            return TajweedReport(
                surah=surah,
                ayah=ayah,
                occurrences=occurrences,
                verified=False,
                unavailable_reason=pronunciation.unavailable_reason,
            )

        by_word: dict[int, list[PronunciationFinding]] = {}
        for finding in pronunciation.findings:
            if finding.word_index is not None:
                by_word.setdefault(finding.word_index, []).append(finding)

        checks = tuple(
            self._check(occurrence, by_word) for occurrence in occurrences
        )
        return TajweedReport(
            surah=surah, ayah=ayah, occurrences=occurrences, checks=checks, verified=True
        )

    @staticmethod
    def _check(
        occurrence: RuleOccurrence, by_word: dict[int, list[PronunciationFinding]]
    ) -> RuleCheck:
        rule = occurrence.rule

        if rule.verification in (Verification.OCCURRENCE_ONLY, Verification.NOT_DETECTABLE):
            return RuleCheck(
                occurrence=occurrence,
                status="not_checked",
                confidence="none",
                detail=rule.note or "this rule's performance cannot be verified",
            )

        findings = [f for i in occurrence.word_indices for f in by_word.get(i, [])]
        if not findings:
            return RuleCheck(occurrence=occurrence, status="kept", confidence=rule.verification.value)

        # The engine named this rule and compared counts - the strongest evidence.
        named = [f for f in findings if f.tajweed_rule and f.tajweed_rule == rule.name_en]
        if named:
            return RuleCheck(
                occurrence=occurrence,
                status="missed",
                confidence="measured",
                detail=named[0].detail,
            )

        if rule.verification is Verification.POSITIONAL:
            return RuleCheck(
                occurrence=occurrence,
                status="possibly_missed",
                confidence="positional",
                detail=(
                    f"a pronunciation difference was heard in this word "
                    f"({findings[0].detail}); it may or may not relate to this rule"
                ),
            )

        return RuleCheck(occurrence=occurrence, status="kept", confidence=rule.verification.value)


@lru_cache(maxsize=512)
def _occurrences(surah: int, ayah: int, repo: QuranRepository) -> tuple[RuleOccurrence, ...]:
    # Degrades to "no occurrences" rather than failing the request. Tajweed
    # annotation is one feature among many, and an optional analysis should never
    # be able to take down an endpoint - including when the package that provides
    # it is missing, which is exactly how the container build caught it being
    # absent from requirements.txt.
    try:
        result = _phonemizer().phonemize(f"{surah}:{ayah}")
    except ImportError:
        logger.error("quranic-phonemizer is not installed; tajweed annotation disabled")
        return ()
    except Exception:
        logger.exception("tajweed annotation failed", extra={"surah": surah, "ayah": ayah})
        return ()

    ayah_data = repo.ayah(surah, ayah)
    # quranic-phonemizer counts Uthmani words; analysis results are indexed by
    # Imlaey word, and one Uthmani word can be two Imlaey words.
    imlaey_by_uthmani: dict[int, list[int]] = {}
    for word in ayah_data.words:
        imlaey_by_uthmani.setdefault(word.uthmani_index, []).append(word.index)

    uthmani_words = ayah_data.uthmani.split()
    occurrences: list[RuleOccurrence] = []

    for instance in result.rules:
        key = getattr(instance.rule, "value", None)
        rule = RULES.get(key)
        if rule is None:
            continue
        # `source` is an index into the unit sequence, but some rule instances
        # carry none (they attach to a host letter instead, or to nothing at all).
        anchor = instance.source
        if anchor is None:
            anchor = getattr(instance, "host", None)
        if not isinstance(anchor, int):
            continue
        try:
            uthmani_index = result.units[anchor].word + 1
        except (IndexError, AttributeError, TypeError):
            continue
        occurrences.append(
            RuleOccurrence(
                rule=rule,
                uthmani_word_index=uthmani_index,
                word_indices=tuple(imlaey_by_uthmani.get(uthmani_index, ())),
                word_text=(
                    uthmani_words[uthmani_index - 1]
                    if 0 < uthmani_index <= len(uthmani_words)
                    else ""
                ),
            )
        )
    return tuple(occurrences)


@lru_cache(maxsize=1)
def _phonemizer():
    from quranic_phonemizer import Phonemizer

    return Phonemizer()
