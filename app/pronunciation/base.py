"""Pronunciation analysis: types and the recognizer interface.

The split that matters here is between what we can compute from the text and
what needs a model listening to audio.

**Reference phonetics is available today.** `quran-transcript` converts Uthmani
text into the Quran Phonetic Script, with articulation attributes (sifat) and
Tajweed rules attached, and it does that with no torch and no GPU. That tells us
what *should* be said.

**Recognized phonetics needs an acoustic model** that transcribes audio to
phonemes. The only credible open one is the Muaalem multi-level CTC model, which
requires PyTorch - uninstallable on this project's development machine
(docs/model-selection.md 1.1). So the recognizer is an interface with an adapter
that works where torch does, and the analyzer reports honestly when it is absent
rather than inventing findings.

Nothing here may be reported as a confirmed phoneme or Tajweed error while no
recognizer is present. `PronunciationReport.available` is how the API says so.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class FindingKind(str, Enum):
    """What sort of difference was found, using the phonetic engine's own taxonomy."""

    ARTICULATION = "articulation"
    """A different letter was produced - قُ heard as كُ."""

    TASHKEEL = "tashkeel"
    """Short vowel, shadda or sukun differs."""

    TAJWEED = "tajweed"
    """A rule with a defined expectation was not met - a madd held for the wrong count."""

    UNKNOWN = "unknown"


class Operation(str, Enum):
    REPLACE = "replace"
    DELETE = "delete"
    INSERT = "insert"


@dataclass(frozen=True)
class RecognizedPhonemes:
    """What an acoustic model heard, in Quran Phonetic Script."""

    phonemes: str
    confidence: float = 1.0
    sifat: tuple = ()


@dataclass(frozen=True)
class PronunciationFinding:
    """One phonetic difference between what was expected and what was heard."""

    kind: FindingKind
    operation: Operation
    expected_phonemes: str
    spoken_phonemes: str
    word_index: int | None
    """1-based Imlaey word, when the difference could be attributed to one."""

    uthmani_span: tuple[int, int] | None = None
    expected_length: int | None = None
    spoken_length: int | None = None
    tajweed_rule: str | None = None
    tajweed_rule_ar: str | None = None
    detail: str | None = None

    @property
    def is_tajweed(self) -> bool:
        return self.kind is FindingKind.TAJWEED


@dataclass(frozen=True)
class PronunciationReport:
    """The result of a pronunciation pass - including 'we could not do one'."""

    available: bool
    surah: int
    ayah: int
    reference_phonemes: str
    recognized_phonemes: str | None = None
    findings: tuple[PronunciationFinding, ...] = ()
    unavailable_reason: str | None = None
    engine: str | None = None

    @property
    def tajweed_findings(self) -> tuple[PronunciationFinding, ...]:
        return tuple(f for f in self.findings if f.is_tajweed)


class PhonemeRecognizer(ABC):
    """Transcribes audio into Quran Phonetic Script.

    Deliberately narrow. It does not align, judge, or know about verses - it only
    reports what it heard, so the comparison logic stays testable without a model.
    """

    @property
    @abstractmethod
    def available(self) -> bool:
        """False when this recognizer cannot run here. Must not raise."""

    @property
    @abstractmethod
    def unavailable_reason(self) -> str | None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def recognize(
        self, audio: np.ndarray, sample_rate: int = 16_000, *, reference: str | None = None
    ) -> RecognizedPhonemes:
        """Transcribe audio to phonemes.

        `reference` is the expected phoneme string. The Muaalem model accepts it
        as context; recognizers that do not need it may ignore it.
        """


@dataclass
class ScriptedPhonemeRecognizer(PhonemeRecognizer):
    """Returns a phoneme string given to it. For tests, not for production.

    Lets the comparison and reporting layers be exercised exactly, without a
    600M-parameter model and without pretending a model is present.
    """

    phonemes: str = ""
    confidence: float = 1.0
    _loaded: bool = field(default=False, init=False)

    @property
    def available(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> str | None:
        return None

    @property
    def name(self) -> str:
        return "scripted"

    def load(self) -> None:
        self._loaded = True

    def recognize(
        self, audio: np.ndarray, sample_rate: int = 16_000, *, reference: str | None = None
    ) -> RecognizedPhonemes:
        return RecognizedPhonemes(phonemes=self.phonemes, confidence=self.confidence)
