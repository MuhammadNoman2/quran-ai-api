"""ASR abstraction.

Nothing outside this package may import faster_whisper, ctranslate2 or any other
engine library. The rest of the application depends on `ASREngine` only, so the
model can be replaced without touching alignment, scoring or the API layer.

Two design decisions worth stating explicitly:

1. `ASRWord.probability` is **required**, not optional. It is the primary signal
   for separating genuine words from hallucinations - measured on real Quran
   audio, genuine words score 1.00 while hallucinated trailing words score
   0.17-0.87 (docs/model-selection.md 3.2). An engine that cannot report
   per-word confidence is not usable in this system.

2. There is no `transcribe_stream()`. Whisper has no incremental decode, so such
   a method could only re-transcribe a buffer while pretending to be streaming.
   Streaming state belongs in app/streaming/, which calls `transcribe()` on
   VAD-delimited segments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Tier(str, Enum):
    """Which role an engine plays. Enforced, not advisory.

    PROVISIONAL drives the live view and may never report a mistake: the tiny
    model was measured emitting a wrong word at 0.90 confidence.
    COMMITTED is the only tier permitted to assert an error.
    """

    PROVISIONAL = "provisional"
    COMMITTED = "committed"


@dataclass(frozen=True)
class EngineInfo:
    """What the API is allowed to reveal about an engine.

    `name` is an opaque label. Model identifiers, library names and versions stay
    server-side - clients must not be able to depend on them.
    """

    name: str
    tier: Tier
    device: str
    compute_type: str

    def public(self) -> dict[str, str]:
        return {"engine": self.name, "tier": self.tier.value}


@dataclass(frozen=True)
class ASRWord:
    text: str
    start: float
    end: float
    probability: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability out of range: {self.probability}")
        if self.end < self.start:
            raise ValueError(f"word ends before it starts: {self.start} > {self.end}")


@dataclass(frozen=True)
class ASRResult:
    """One transcription pass over one audio segment."""

    text: str
    words: tuple[ASRWord, ...]
    language: str
    audio_duration: float
    inference_seconds: float
    engine: EngineInfo
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

    @property
    def real_time_factor(self) -> float:
        """<1.0 means faster than real time."""
        return self.inference_seconds / self.audio_duration if self.audio_duration else 0.0

    def words_below(self, threshold: float) -> tuple[ASRWord, ...]:
        """Words whose confidence falls under `threshold`. Hallucination candidates."""
        return tuple(w for w in self.words if w.probability < threshold)

    def speech_end(self) -> float:
        """End time of the last word, or 0.0. Used with VAD to catch hallucinations."""
        return self.words[-1].end if self.words else 0.0


class ASREngine(ABC):
    """Interface every speech recognition backend must satisfy."""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory. Idempotent. Must not download if cached."""

    @abstractmethod
    def unload(self) -> None:
        """Release the model. Idempotent."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """Transcribe mono float32 audio in [-1, 1].

        Raises ValueError on wrong dtype, shape or sample rate rather than
        silently resampling - a caller passing 44.1 kHz has a bug worth surfacing.
        """

    @abstractmethod
    def describe(self) -> EngineInfo: ...

    def __enter__(self) -> ASREngine:
        self.load()
        return self

    def __exit__(self, *exc: object) -> None:
        self.unload()


@dataclass
class FakeASREngine(ASREngine):
    """Deterministic engine for tests. Returns whatever it was constructed with.

    Lets alignment, scoring and streaming be tested without loading a model or
    depending on model behaviour that may change.
    """

    text: str = ""
    word_probabilities: dict[str, float] = field(default_factory=dict)
    default_probability: float = 1.0
    tier: Tier = Tier.COMMITTED
    word_duration: float = 0.5
    _loaded: bool = field(default=False, init=False)

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def describe(self) -> EngineInfo:
        return EngineInfo(name="fake", tier=self.tier, device="cpu", compute_type="none")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        if not self._loaded:
            raise RuntimeError("engine not loaded")
        tokens = self.text.split()
        words = tuple(
            ASRWord(
                text=t,
                start=i * self.word_duration,
                end=(i + 1) * self.word_duration,
                probability=self.word_probabilities.get(t, self.default_probability),
            )
            for i, t in enumerate(tokens)
        )
        return ASRResult(
            text=self.text,
            words=words,
            language="ar",
            audio_duration=len(audio) / sample_rate if sample_rate else 0.0,
            inference_seconds=0.0,
            engine=self.describe(),
        )
