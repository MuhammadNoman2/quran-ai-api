"""Server-side rolling audio buffer.

Microphone packets arrive far faster than we can transcribe. Measured on this
project's CPU, one inference costs roughly `2.5s + 0.13 x duration` *regardless
of how short the clip is*, because Whisper pads every window to 30 seconds. So
audio must accumulate here and be handed to the recognizer in worthwhile
segments - feeding each 20 ms packet to the model is not slow, it is impossible.

The buffer is bounded. If a client outruns the server the oldest audio is
dropped and the caller is told, which degrades gracefully instead of growing
until the process dies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BYTES_PER_SAMPLE = 2  # pcm_s16le
INT16_FULL_SCALE = 32768.0


class AudioBufferError(ValueError):
    """Raised for malformed input. Callers turn this into an error event, not a crash."""


@dataclass
class BufferStats:
    seconds_buffered: float
    seconds_dropped: float
    total_seconds_received: float


class RollingBuffer:
    """Accumulates PCM audio and hands out segments.

    Append is cheap; the concatenation cost is paid once per segment rather than
    once per packet.
    """

    def __init__(self, sample_rate: int = 16_000, max_seconds: float = 60.0) -> None:
        if sample_rate <= 0:
            raise AudioBufferError("sample_rate must be positive")
        self._sample_rate = sample_rate
        self._max_samples = int(max_seconds * sample_rate)
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self._dropped = 0
        self._received = 0

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def seconds(self) -> float:
        return self._samples / self._sample_rate

    @property
    def is_empty(self) -> bool:
        return self._samples == 0

    def stats(self) -> BufferStats:
        return BufferStats(
            seconds_buffered=round(self.seconds, 3),
            seconds_dropped=round(self._dropped / self._sample_rate, 3),
            total_seconds_received=round(self._received / self._sample_rate, 3),
        )

    # ── writing ──────────────────────────────────────────────────────────────

    def append_pcm(self, data: bytes) -> bool:
        """Append little-endian signed 16-bit PCM. Returns False if audio was dropped.

        A frame with an odd byte count cannot be s16le. Rather than silently
        truncating - which would shift every subsequent sample and corrupt the
        stream - it is rejected.
        """
        if not data:
            return True
        if len(data) % BYTES_PER_SAMPLE:
            raise AudioBufferError(
                f"pcm_s16le needs an even byte count, got {len(data)}"
            )
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE
        return self.append_samples(samples)

    def append_samples(self, samples: np.ndarray) -> bool:
        if samples.ndim != 1:
            raise AudioBufferError(f"expected mono audio, got shape {samples.shape}")
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        self._chunks.append(samples)
        self._samples += samples.size
        self._received += samples.size
        return not self._trim()

    def _trim(self) -> bool:
        """Drop the oldest audio past capacity. Returns True if anything was dropped."""
        if self._samples <= self._max_samples:
            return False
        dropped = 0
        while self._samples - self._chunks[0].size >= self._max_samples:
            head = self._chunks.pop(0)
            self._samples -= head.size
            dropped += head.size
        if self._samples > self._max_samples:
            overflow = self._samples - self._max_samples
            self._chunks[0] = self._chunks[0][overflow:]
            self._samples -= overflow
            dropped += overflow
        self._dropped += dropped
        return dropped > 0

    # ── reading ──────────────────────────────────────────────────────────────

    def snapshot(self) -> np.ndarray:
        """Everything buffered, without consuming it."""
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self._chunks) > 1:
            self._chunks = [np.concatenate(self._chunks)]
        return self._chunks[0]

    def take(self, keep_tail_seconds: float = 0.0) -> np.ndarray:
        """Consume the buffer, optionally leaving a tail behind.

        The tail is why this is not just a queue: overlapping successive segments
        keeps a word that straddles a boundary from being cut in half and
        misrecognised - which would surface to the reciter as a mistake they did
        not make.
        """
        segment = self.snapshot()
        if segment.size == 0:
            return segment
        keep = min(int(keep_tail_seconds * self._sample_rate), segment.size)
        tail = segment[segment.size - keep :] if keep else np.zeros(0, dtype=np.float32)
        self._chunks = [tail] if tail.size else []
        self._samples = tail.size
        return segment

    def drop_before(self, seconds: float) -> float:
        """Discard audio before `seconds`. Returns how much was actually dropped.

        Used to advance the analysis window once words have been confirmed, so
        the cost of a cumulative pass stays bounded on a long recitation.
        """
        if seconds <= 0:
            return 0.0
        buffered = self.snapshot()
        cut = min(int(seconds * self._sample_rate), buffered.size)
        if cut <= 0:
            return 0.0
        self._chunks = [buffered[cut:]] if cut < buffered.size else []
        self._samples = max(0, buffered.size - cut)
        return cut / self._sample_rate

    def clear(self) -> None:
        self._chunks.clear()
        self._samples = 0
