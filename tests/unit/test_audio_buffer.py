"""Rolling buffer behaviour, including how it degrades under pressure."""

import numpy as np
import pytest

from app.audio.buffer import AudioBufferError, RollingBuffer


def pcm(seconds: float, sample_rate: int = 16_000) -> bytes:
    n = int(seconds * sample_rate)
    return (np.sin(np.linspace(0, 50, n)) * 10_000).astype("<i2").tobytes()


class TestAppend:
    def test_accumulates_duration(self):
        b = RollingBuffer(16_000, max_seconds=10)
        b.append_pcm(pcm(1.0))
        b.append_pcm(pcm(0.5))
        assert b.seconds == pytest.approx(1.5, abs=0.01)

    def test_empty_frame_is_a_no_op(self):
        b = RollingBuffer(16_000)
        assert b.append_pcm(b"") is True
        assert b.is_empty

    def test_odd_byte_count_is_rejected(self):
        """Truncating would shift every later sample and corrupt the stream."""
        b = RollingBuffer(16_000)
        with pytest.raises(AudioBufferError, match="even byte count"):
            b.append_pcm(b"\x00\x00\x00")

    def test_converts_to_float32_in_range(self):
        b = RollingBuffer(16_000)
        b.append_pcm(pcm(0.1))
        snapshot = b.snapshot()
        assert snapshot.dtype == np.float32
        assert np.abs(snapshot).max() <= 1.0


class TestBounding:
    def test_drops_oldest_audio_past_capacity(self):
        b = RollingBuffer(16_000, max_seconds=2.0)
        assert b.append_pcm(pcm(1.0)) is True
        assert b.append_pcm(pcm(1.0)) is True
        assert b.append_pcm(pcm(1.0)) is False, "should report the drop"
        assert b.seconds == pytest.approx(2.0, abs=0.01)

    def test_reports_what_was_dropped(self):
        b = RollingBuffer(16_000, max_seconds=2.0)
        for _ in range(3):
            b.append_pcm(pcm(1.0))
        stats = b.stats()
        assert stats.seconds_dropped == pytest.approx(1.0, abs=0.01)
        assert stats.total_seconds_received == pytest.approx(3.0, abs=0.01)

    def test_a_single_oversized_frame_is_bounded(self):
        b = RollingBuffer(16_000, max_seconds=1.0)
        b.append_pcm(pcm(5.0))
        assert b.seconds == pytest.approx(1.0, abs=0.01)


class TestTake:
    def test_consumes_the_buffer(self):
        b = RollingBuffer(16_000, max_seconds=10)
        b.append_pcm(pcm(2.0))
        segment = b.take()
        assert segment.size == pytest.approx(32_000, abs=100)
        assert b.is_empty

    def test_keeps_an_overlap_tail(self):
        """Overlap stops a word straddling a boundary being cut in half."""
        b = RollingBuffer(16_000, max_seconds=10)
        b.append_pcm(pcm(2.0))
        b.take(keep_tail_seconds=0.5)
        assert b.seconds == pytest.approx(0.5, abs=0.01)

    def test_take_on_empty_buffer(self):
        assert RollingBuffer(16_000).take().size == 0

    def test_snapshot_does_not_consume(self):
        b = RollingBuffer(16_000, max_seconds=10)
        b.append_pcm(pcm(1.0))
        b.snapshot()
        assert b.seconds == pytest.approx(1.0, abs=0.01)


class TestValidation:
    def test_rejects_non_positive_sample_rate(self):
        with pytest.raises(AudioBufferError):
            RollingBuffer(0)

    def test_rejects_stereo_samples(self):
        with pytest.raises(AudioBufferError, match="mono"):
            RollingBuffer(16_000).append_samples(np.zeros((10, 2), dtype=np.float32))
