"""Voice activity detection.

Uses the Silero VAD that ships inside faster-whisper and runs on ONNX Runtime,
so no PyTorch is involved - which matters, because torch cannot be installed on
this project's development machine at all (docs/model-selection.md 1.1).

Measured cost on real recitation: 11-25 ms for a few seconds of audio, against
~2500 ms for one recognition pass. VAD is effectively free, and it pays for
itself twice over:

1. **Scheduling.** Recognition runs at natural pauses (waqf) instead of on a
   fixed timer, so a segment ends where the reciter ended a phrase rather than
   in the middle of a word.
2. **Hallucination evidence.** Whisper invents words *after* speech stops. On the
   Husary clip, speech ends at 5.48 s and the model produced a spurious
   الْحَمْدُ at 5.60-6.24 s. Knowing where speech actually ended lets the
   confidence policy reject such words on timing alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

#: Fine-grained segmentation. The processor decides what counts as a pause worth
#: acting on; merging pauses here would hide that decision inside the detector.
_MIN_SILENCE_MS = 100
_MIN_SPEECH_MS = 120


@dataclass(frozen=True)
class SpeechState:
    is_speaking: bool
    speech_started_at: float | None
    speech_ended_at: float | None
    """End of the most recent speech, in seconds from the start of the buffer."""

    silence_seconds: float
    """Silence between the end of speech and the end of the buffer."""

    total_speech_seconds: float
    segments: list[tuple[float, float]] = field(default_factory=list)

    @property
    def has_speech(self) -> bool:
        return bool(self.segments)


@lru_cache(maxsize=1)
def _options(threshold: float):
    from faster_whisper.vad import VadOptions

    return VadOptions(
        threshold=threshold,
        min_silence_duration_ms=_MIN_SILENCE_MS,
        min_speech_duration_ms=_MIN_SPEECH_MS,
        speech_pad_ms=0,
    )


class SpeechDetector:
    """Stateless per call: the caller owns the audio, this reports what is in it."""

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold

    def analyse(self, audio: np.ndarray, sample_rate: int = 16_000) -> SpeechState:
        from faster_whisper.vad import get_speech_timestamps

        if audio.size == 0:
            return SpeechState(False, None, None, 0.0, 0.0, [])

        stamps = get_speech_timestamps(audio, _options(self._threshold), sample_rate)
        duration = audio.size / sample_rate
        if not stamps:
            return SpeechState(False, None, None, duration, 0.0, [])

        segments = [(s["start"] / sample_rate, s["end"] / sample_rate) for s in stamps]
        speech_end = segments[-1][1]
        silence = max(0.0, duration - speech_end)

        return SpeechState(
            # Speech that runs to the very end of the buffer means the reciter is
            # still going; the pause threshold is the caller's judgement to make.
            is_speaking=silence < _MIN_SILENCE_MS / 1000,
            speech_started_at=segments[0][0],
            speech_ended_at=speech_end,
            silence_seconds=silence,
            total_speech_seconds=sum(end - start for start, end in segments),
            segments=segments,
        )

    def warm(self) -> None:
        """Load the ONNX model now rather than on the first frame of a session."""
        self.analyse(np.zeros(1600, dtype=np.float32))
