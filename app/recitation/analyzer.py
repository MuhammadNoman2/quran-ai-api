"""Orchestrates one recitation analysis.

Depends on the `ASREngine` **interface**, never on a concrete model - the brief
is explicit about this, and it is what lets the engine be swapped later.

The pipeline:

    audio -> ASREngine -> SpokenWord[]
                            |
    Quran repository -> ExpectedWord[]
                            |
                        WordAligner        (mechanical facts)
                            |
                     ConfidencePolicy      (what we are willing to claim)
                            |
                         scoring           (deterministic, documented)
"""

from __future__ import annotations

import time
import uuid

import numpy as np

from app.alignment.base import AlignmentResult, ExpectedWord, SpokenWord
from app.alignment.word_alignment import WordAligner
from app.asr.base import ASREngine, ASRResult, Tier
from app.core.config import settings
from app.models.enums import ConfidenceBand, ErrorCategory, NOT_YET_DETECTABLE
from app.models.schemas import RecitationAnalysis, WordAnalysis
from app.quran.repository import QuranRepository
from app.quran.text_normalizer import normalize_for_asr_matching
from app.quran.verse_locator import VerseLocator
from app.recitation.confidence import ConfidencePolicy
from app.recitation.scoring import ScoreBreakdown, score


class VerseNotIdentified(RuntimeError):
    """Raised when no surah/ayah was supplied and none could be inferred."""


class RecitationAnalyzer:
    def __init__(
        self,
        engine: ASREngine,
        *,
        repository: QuranRepository | None = None,
        aligner: WordAligner | None = None,
        policy: ConfidencePolicy | None = None,
        locator: VerseLocator | None = None,
    ) -> None:
        self._engine = engine
        self._repo = repository or QuranRepository()
        self._aligner = aligner or WordAligner()
        self._policy = policy or ConfidencePolicy(settings.word_conf_min)
        self._locator = locator

    # ── public API ───────────────────────────────────────────────────────────

    def analyze(
        self,
        audio: np.ndarray,
        *,
        surah: int | None = None,
        ayah: int | None = None,
        sample_rate: int = 16_000,
        session_id: str | None = None,
    ) -> RecitationAnalysis:
        started = time.perf_counter()
        asr = self._engine.transcribe(audio, sample_rate)

        detected = False
        verse_confidence: float | None = None
        if surah is None or ayah is None:
            surah, ayah, verse_confidence = self._detect_verse(asr.text)
            detected = True

        expected_ayah = self._repo.ayah(surah, ayah)
        results = self.compare(asr, surah, ayah)
        breakdown = score(results, expected_ayah.word_count)

        words = [self._to_word_analysis(r) for r in results]
        return RecitationAnalysis(
            session_id=session_id or str(uuid.uuid4()),
            surah=surah,
            ayah=ayah,
            expected_text=expected_ayah.uthmani,
            recognized_text=asr.text,
            word_accuracy_score=breakdown.score,
            score_formula=breakdown.formula,
            words=words,
            errors=[w for w, r in zip(words, results) if r.is_error],
            observations=[
                w
                for w, r in zip(words, results)
                if not r.is_error and r.category is not None
            ],
            engine=asr.engine.public(),
            audio_seconds=round(asr.audio_duration, 3),
            processing_seconds=round(time.perf_counter() - started, 3),
            verse_detected=detected,
            verse_confidence=verse_confidence,
        )

    def compare(self, asr: ASRResult, surah: int, ayah: int) -> list[AlignmentResult]:
        """Align and judge, without building the API response.

        Exposed separately so the streaming layer can reuse it per segment.
        """
        expected = self.expected_words(surah, ayah)
        spoken = self.spoken_words(asr)
        aligned = self._aligner.align(expected, spoken)
        judged = self._policy.judge(
            aligned, tier=asr.engine.tier, speech_end=asr.speech_end()
        )
        self._assert_no_undetectable_claims(judged)
        return judged

    # ── helpers ──────────────────────────────────────────────────────────────

    def expected_words(self, surah: int, ayah: int) -> list[ExpectedWord]:
        a = self._repo.ayah(surah, ayah)
        return [
            ExpectedWord(
                text=w.uthmani,
                normalized=w.normalized,
                index=w.index,
                surah=surah,
                ayah=ayah,
            )
            for w in a.words
        ]

    @staticmethod
    def spoken_words(asr: ASRResult) -> list[SpokenWord]:
        spoken: list[SpokenWord] = []
        for w in asr.words:
            normalized = normalize_for_asr_matching(w.text)
            if not normalized:
                continue
            spoken.append(
                SpokenWord(
                    text=w.text,
                    normalized=normalized,
                    start_time=w.start,
                    end_time=w.end,
                    confidence=w.probability,
                )
            )
        return spoken

    def _detect_verse(self, text: str) -> tuple[int, int, float]:
        locator = self._locator or VerseLocator(self._repo)
        match = locator.locate_best(text)
        if match is None:
            raise VerseNotIdentified(
                "could not identify a verse from the recognized text"
            )
        return match.surah, match.ayah, match.confidence

    @staticmethod
    def _assert_no_undetectable_claims(results: list[AlignmentResult]) -> None:
        """Phases 8 and 9 do not exist yet, so their categories must never appear.

        A guard rather than a comment, because the failure mode - telling a user
        they made a Tajweed error we cannot actually detect - is exactly what the
        brief forbids.
        """
        for r in results:
            if r.category in NOT_YET_DETECTABLE:
                raise AssertionError(
                    f"{r.category} requires analysis that does not exist yet"
                )

    @staticmethod
    def _to_word_analysis(r: AlignmentResult) -> WordAnalysis:
        return WordAnalysis(
            index=r.index,
            expected=r.expected_word.text if r.expected_word else None,
            spoken=r.spoken_word.text if r.spoken_word else None,
            status=r.status.value,
            band=r.band.value,
            category=r.category.value if r.category else None,
            confidence=round(r.confidence, 4),
            similarity=round(r.similarity, 4),
            note=r.suppressed_reason,
        )
