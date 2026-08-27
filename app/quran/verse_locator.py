"""Scenario B: work out which verse is being recited when the client didn't say.

Kept entirely separate from the ASR engine, per the brief: discovery logic must
not leak into transcription.

Why a custom index rather than quran-transcript's built-ins - both were evaluated:

- `quran_transcript.search()` needs a pivot ayah and a window. That is *local
  position tracking* around a known point, useful in Phase 7, not cold discovery.
- `PhoneticSearch` does search the whole Quran and is fast (2 ms), but it queries
  the *phonetic* script, and phonetizing ASR output fails outright:
  `ValueError: For Letter alif: ا can not start a phoneme script`. The phonetizer
  requires Uthmani orthography; the ASR emits Imlaey. It becomes usable in Phase 8
  when a phoneme model produces phonemes directly - not here.

So the index is built over the normalized Imlaey text, which is exactly what the
ASR gives us.

The Quran is indexed as **one continuous stream of 77800 words**, not per ayah.
That way a reciter who runs from one ayah into the next matches contiguously
instead of falling off the end of an ayah-shaped index.
"""

from __future__ import annotations

import logging
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import cached_property

from app.quran.repository import QuranRepository
from app.quran.text_normalizer import normalize_for_asr_matching

logger = logging.getLogger(__name__)

NGRAM = 3
MIN_QUERY_WORDS = 1
DEFAULT_MIN_CONFIDENCE = 0.55

#: A one-word query is answered only if that word is nearly unique in the Quran.
#: 28 ayat are a single word - يس, طه, كهيعص and the other muqatta'at - and they
#: would otherwise be unfindable. The rarest occur once; الم occurs 6 times. By
#: contrast الله occurs 2155 times and من 2763, so this threshold cleanly admits
#: the distinctive openings while refusing to guess from a common word.
SINGLE_WORD_MAX_OCCURRENCES = 8


@dataclass(frozen=True)
class VerseMatch:
    """Where the locator thinks the reciter is."""

    surah: int
    ayah: int
    word_index: int
    """1-based Imlaey word within `ayah` where the match starts."""

    confidence: float
    matched_text: str
    """Canonical Uthmani text of the matched span."""

    query_word_count: int
    matched_word_count: int
    spans_multiple_ayat: bool = False
    alternatives: int = 0
    """Other locations matching equally well. The Quran repeats verses verbatim."""

    @property
    def is_ambiguous(self) -> bool:
        return self.alternatives > 0

    @property
    def reference(self) -> str:
        return f"{self.surah}:{self.ayah}"


@dataclass
class _Position:
    surah: int
    ayah: int
    word_index: int


class VerseLocator:
    def __init__(
        self,
        repository: QuranRepository | None = None,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self._repo = repository or QuranRepository()
        self._min_confidence = min_confidence

    # ── index ─────────────────────────────────────────────────────────────────

    @cached_property
    def _stream(self) -> list[str]:
        """Every normalized word in the Quran, in order."""
        return [w.normalized for a in self._repo.all_ayat() for w in a.words]

    @cached_property
    def _positions(self) -> list[_Position]:
        return [
            _Position(a.surah, a.ayah, w.index)
            for a in self._repo.all_ayat()
            for w in a.words
        ]

    @cached_property
    def _ayah_starts(self) -> list[int]:
        """Global index at which each ayah begins - for span checks."""
        starts, running = [], 0
        for a in self._repo.all_ayat():
            starts.append(running)
            running += a.word_count
        return starts

    @cached_property
    def _index(self) -> dict[tuple[str, ...], list[int]]:
        index: dict[tuple[str, ...], list[int]] = defaultdict(list)
        stream = self._stream
        for i in range(len(stream) - NGRAM + 1):
            index[tuple(stream[i : i + NGRAM])].append(i)
        return dict(index)

    @cached_property
    def _unigram_index(self) -> dict[tuple[str, ...], list[int]]:
        index: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i, word in enumerate(self._stream):
            index[(word,)].append(i)
        return dict(index)

    @cached_property
    def _bigram_index(self) -> dict[tuple[str, ...], list[int]]:
        """Fallback for two-word queries, which trigrams cannot serve."""
        index: dict[tuple[str, ...], list[int]] = defaultdict(list)
        stream = self._stream
        for i in range(len(stream) - 1):
            index[tuple(stream[i : i + 2])].append(i)
        return dict(index)

    def warm(self) -> None:
        """Build the index eagerly - call at startup, not on the first request."""
        _ = self._index, self._bigram_index, self._unigram_index
        _ = self._positions, self._ayah_starts

    # ── query ─────────────────────────────────────────────────────────────────

    def _vote(self, query: list[str], n: int) -> dict[int, int]:
        """Offset voting. An n-gram at query position i found at stream position p
        implies the recitation starts at p - i, so contiguous matches pile onto
        one offset while coincidental word hits scatter."""
        index = self._index_for(n)
        votes: dict[int, int] = defaultdict(int)
        for i in range(len(query) - n + 1):
            for p in index.get(tuple(query[i : i + n]), ()):
                start = p - i
                if start >= 0:
                    votes[start] += 1
        return votes

    def _index_for(self, n: int) -> dict[tuple[str, ...], list[int]]:
        return {3: self._index, 2: self._bigram_index, 1: self._unigram_index}[n]

    def locate(self, text: str, *, top_k: int = 3, surah_hint: int | None = None) -> list[VerseMatch]:
        """Rank likely locations for `text`. Empty list means "I don't know".

        Trigrams first, bigrams as a fallback. The fallback is not a nicety: on a
        four-word verse, a single substituted word destroys *every* trigram, so a
        learner who misreads one word would get no match at all - and learners
        misreading words is the entire point of this system.

        A single-word query is answered only when the word is rare enough to be
        meaningful (see SINGLE_WORD_MAX_OCCURRENCES).
        """
        query = [w for w in normalize_for_asr_matching(text).split(" ") if w]
        if len(query) < MIN_QUERY_WORDS:
            return []

        if len(query) == 1:
            postings = self._unigram_index.get((query[0],), ())
            if not postings or len(postings) > SINGLE_WORD_MAX_OCCURRENCES:
                return []
            return self._locate_with(query, 1, top_k, surah_hint)

        results = self._locate_with(query, NGRAM, top_k, surah_hint) if len(query) >= NGRAM else []
        if not results:
            results = self._locate_with(query, 2, top_k, surah_hint)
        return results

    def _locate_with(
        self, query: list[str], n: int, top_k: int, surah_hint: int | None
    ) -> list[VerseMatch]:
        if len(query) < n:
            return []
        votes = self._vote(query, n)
        if not votes:
            return []

        top_votes = max(votes.values())
        # Keep near-best offsets so repeated verses surface as alternatives.
        candidates = [s for s, v in votes.items() if v >= top_votes * 0.8]
        if surah_hint is not None:
            filtered = [s for s in candidates if self._positions[s].surah == surah_hint]
            candidates = filtered or candidates

        scored: list[tuple[float, int, int, int]] = []
        for start in candidates[:200]:  # cap work on very repetitive phrases
            window = self._stream[start : start + len(query)]
            matcher = SequenceMatcher(None, query, window, autojunk=False)
            blocks = [b for b in matcher.get_matching_blocks() if b.size]
            if not blocks:
                continue
            matched = sum(b.size for b in blocks)
            # End of the genuinely matched region - NOT start + len(query).
            # A hallucinated tail must not stretch the match into the next ayah.
            last = blocks[-1].b + blocks[-1].size - 1
            scored.append((matcher.ratio(), start, matched, last))
        if not scored:
            return []
        scored.sort(key=lambda t: (-t[0], t[1]))

        best_ratio = scored[0][0]
        if best_ratio < self._min_confidence:
            return []
        ties = sum(1 for r, *_ in scored if abs(r - best_ratio) < 1e-9)

        return [
            self._to_match(start, len(query), ratio, ties - 1, matched, last)
            for ratio, start, matched, last in scored[:top_k]
        ]

    def locate_best(self, text: str, *, surah_hint: int | None = None) -> VerseMatch | None:
        matches = self.locate(text, top_k=1, surah_hint=surah_hint)
        return matches[0] if matches else None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_match(
        self,
        start: int,
        query_len: int,
        ratio: float,
        alternatives: int,
        matched_count: int,
        last_offset: int,
    ) -> VerseMatch:
        pos = self._positions[start]
        end = min(start + last_offset, len(self._stream) - 1)
        end_pos = self._positions[end]
        ayah = self._repo.ayah(pos.surah, pos.ayah)

        matched_words = ayah.words[pos.word_index - 1 : pos.word_index - 1 + last_offset + 1]
        uthmani_indices = sorted({w.uthmani_index for w in matched_words})
        uthmani_all = ayah.uthmani.split()
        matched_text = " ".join(
            uthmani_all[i - 1] for i in uthmani_indices if i - 1 < len(uthmani_all)
        )

        return VerseMatch(
            surah=pos.surah,
            ayah=pos.ayah,
            word_index=pos.word_index,
            confidence=round(ratio, 4),
            matched_text=matched_text,
            query_word_count=query_len,
            matched_word_count=matched_count,
            spans_multiple_ayat=(end_pos.surah, end_pos.ayah) != (pos.surah, pos.ayah),
            alternatives=alternatives,
        )

    def _ayah_of_global(self, index: int) -> int:  # pragma: no cover - helper
        return bisect_right(self._ayah_starts, index) - 1
