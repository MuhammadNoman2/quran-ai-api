"""Pronunciation analysis, or an honest account of why there is none."""

from __future__ import annotations

import logging

import numpy as np

from app.pronunciation.base import PhonemeRecognizer, PronunciationReport
from app.pronunciation.comparator import compare
from app.pronunciation.reference import PhoneticReference, phonetics_for

logger = logging.getLogger(__name__)


class PronunciationAnalyzer:
    """Compares a recitation's phonemes against the expected ones.

    Depends on the `PhonemeRecognizer` interface, never on a concrete model - the
    same rule the ASR layer follows, and for the same reason: the acoustic model
    here is the piece most likely to be replaced.
    """

    def __init__(self, recognizer: PhonemeRecognizer | None = None) -> None:
        self._recognizer = recognizer

    @property
    def available(self) -> bool:
        return self._recognizer is not None and self._recognizer.available

    @property
    def unavailable_reason(self) -> str | None:
        if self._recognizer is None:
            return "no phoneme recognizer is configured"
        return self._recognizer.unavailable_reason

    def reference(self, surah: int, ayah: int) -> PhoneticReference:
        """Expected phonetics. Always available - it needs no model."""
        return phonetics_for(surah, ayah)

    def analyze(
        self, audio: np.ndarray, surah: int, ayah: int, sample_rate: int = 16_000
    ) -> PronunciationReport:
        reference = self.reference(surah, ayah)

        if not self.available:
            # The reference is still returned: knowing what *should* be recited is
            # useful on its own, and saying "unavailable" is more honest than
            # returning an empty findings list that looks like a clean pass.
            return PronunciationReport(
                available=False,
                surah=surah,
                ayah=ayah,
                reference_phonemes=reference.phonemes,
                unavailable_reason=self.unavailable_reason,
            )

        assert self._recognizer is not None
        try:
            recognized = self._recognizer.recognize(
                audio, sample_rate, reference=reference.phonemes
            )
        except Exception as exc:
            logger.exception("phoneme recognition failed", extra={"surah": surah, "ayah": ayah})
            return PronunciationReport(
                available=False,
                surah=surah,
                ayah=ayah,
                reference_phonemes=reference.phonemes,
                unavailable_reason=f"phoneme recognition failed: {exc}",
                engine=self._recognizer.name,
            )

        return PronunciationReport(
            available=True,
            surah=surah,
            ayah=ayah,
            reference_phonemes=reference.phonemes,
            recognized_phonemes=recognized.phonemes,
            findings=compare(reference, recognized.phonemes),
            engine=self._recognizer.name,
        )
