"""Global sequence alignment of recognized words against the expected ayah.

Needleman-Wunsch over word tokens. The substitution cost is derived from
character similarity of the *normalized* forms, so near-misses are cheap and
unrelated words are expensive:

    رب  vs ربك    similarity 0.80 -> cost 0.20   (probably one word, misheard)
    رب  vs مالك   similarity 0.00 -> cost 1.00   (probably a real substitution)

That distinction is what later lets us separate a probable pronunciation or
recognition difference from a genuine wrong word, per the brief's insistence
that those are not the same claim.

A note on what this layer *cannot* see. Normalization strips harakat, so
رَبِّ and رَبَ both reduce to رب and align as CORRECT. That is deliberate: the
text ASR gives us no reliable evidence about short vowels or shadda, and
reporting a harakat mistake from a diacritic the recognizer guessed would be
exactly the false correction this system exists to avoid. Harakat and Tajweed
belong to Phase 8/9, where phoneme evidence exists.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from app.alignment.base import AlignmentResult, ExpectedWord, SpokenWord
from app.models.enums import WordStatus

GAP_COST = 0.6
"""Cost of an insertion or deletion.

Set below 1.0 deliberately. Two gaps (a deletion plus an insertion) cost 1.2,
which is more than the 1.0 maximum substitution cost, so the aligner prefers to
call a wholly different word a substitution rather than "missing + extra". That
matches how people actually misread: they say the wrong word far more often than
they simultaneously drop one and invent another.
"""

REPETITION_SIMILARITY = 0.9
"""How close an extra word must be to its predecessor to count as a repetition."""


def similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def substitution_cost(a: str, b: str) -> float:
    return 1.0 - similarity(a, b)


class WordAligner:
    """Aligns spoken words to expected words. Purely mechanical - no judgement.

    Confidence gating and error categorisation happen downstream in
    app/recitation/, so this class stays testable in isolation and has no opinion
    about whether the reciter made a mistake.
    """

    def __init__(self, gap_cost: float = GAP_COST) -> None:
        self._gap = gap_cost

    def align(
        self, expected: list[ExpectedWord], spoken: list[SpokenWord]
    ) -> list[AlignmentResult]:
        n, m = len(expected), len(spoken)

        # cost[i][j] = best cost aligning expected[:i] with spoken[:j]
        cost = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            cost[i][0] = i * self._gap
        for j in range(1, m + 1):
            cost[0][j] = j * self._gap

        for i in range(1, n + 1):
            e = expected[i - 1].normalized
            for j in range(1, m + 1):
                s = spoken[j - 1].normalized
                cost[i][j] = min(
                    cost[i - 1][j - 1] + substitution_cost(e, s),  # align
                    cost[i - 1][j] + self._gap,                    # expected unmatched
                    cost[i][j - 1] + self._gap,                    # spoken unmatched
                )

        results = self._traceback(expected, spoken, cost)
        return self._mark_repetitions(results)

    def _traceback(
        self,
        expected: list[ExpectedWord],
        spoken: list[SpokenWord],
        cost: list[list[float]],
    ) -> list[AlignmentResult]:
        i, j = len(expected), len(spoken)
        out: list[AlignmentResult] = []

        while i > 0 or j > 0:
            if i > 0 and j > 0:
                e, s = expected[i - 1], spoken[j - 1]
                sim = similarity(e.normalized, s.normalized)
                if abs(cost[i][j] - (cost[i - 1][j - 1] + (1.0 - sim))) < 1e-9:
                    out.append(
                        AlignmentResult(
                            status=WordStatus.CORRECT if sim == 1.0 else WordStatus.SUBSTITUTED,
                            expected_word=e,
                            spoken_word=s,
                            similarity=sim,
                            confidence=s.confidence,
                        )
                    )
                    i, j = i - 1, j - 1
                    continue
            if i > 0 and abs(cost[i][j] - (cost[i - 1][j] + self._gap)) < 1e-9:
                out.append(
                    AlignmentResult(status=WordStatus.MISSING, expected_word=expected[i - 1])
                )
                i -= 1
                continue
            s = spoken[j - 1]
            out.append(
                AlignmentResult(
                    status=WordStatus.EXTRA, spoken_word=s, confidence=s.confidence
                )
            )
            j -= 1

        out.reverse()
        return out

    def _mark_repetitions(self, results: list[AlignmentResult]) -> list[AlignmentResult]:
        """Reclassify EXTRA words that merely repeat what was just said.

        A reciter who stumbles and repeats a word has not added a wrong word, and
        conflating the two would penalise a very ordinary kind of hesitation.

        Neighbours on **both** sides are checked. When a word is duplicated, the
        aligner may legitimately place the surplus token either before or after
        the matched one - the cost is identical - so looking only backwards
        misses roughly half of all repetitions.
        """
        out: list[AlignmentResult] = []
        for idx, r in enumerate(results):
            if r.status is not WordStatus.EXTRA or r.spoken_word is None:
                out.append(r)
                continue
            best = 0.0
            for neighbour in self._adjacent_spoken(results, idx):
                best = max(best, similarity(neighbour.normalized, r.spoken_word.normalized))
            if best >= REPETITION_SIMILARITY:
                out.append(
                    AlignmentResult(
                        status=WordStatus.REPEATED,
                        expected_word=None,
                        spoken_word=r.spoken_word,
                        confidence=r.confidence,
                        similarity=best,
                    )
                )
            else:
                out.append(r)
        return out

    @staticmethod
    def _adjacent_spoken(results: list[AlignmentResult], idx: int) -> list[SpokenWord]:
        """The nearest spoken word on each side of `idx`."""
        found: list[SpokenWord] = []
        for step in (-1, 1):
            k = idx + step
            while 0 <= k < len(results):
                if results[k].spoken_word is not None:
                    found.append(results[k].spoken_word)
                    break
                k += step
        return found
