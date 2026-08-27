"""Audio decoding and the ASR input contract."""

import numpy as np
import pytest
import soundfile as sf

from app.audio.decoder import (
    AudioDecodeError,
    TARGET_SAMPLE_RATE,
    decode,
    duration_seconds,
    validate_pcm,
)


@pytest.fixture
def tone_wav(tmp_path):
    """1 second of 440 Hz at 44.1 kHz stereo - deliberately not the target format."""
    sr = 44_100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    sf.write(path, np.stack([mono, mono], axis=1), sr)
    return path


class TestDecode:
    def test_resamples_and_downmixes(self, tone_wav):
        audio = decode(tone_wav)
        assert audio.ndim == 1
        assert audio.dtype == np.float32
        assert abs(duration_seconds(audio) - 1.0) < 0.05

    def test_output_is_in_range(self, tone_wav):
        assert np.abs(decode(tone_wav)).max() <= 1.0

    def test_accepts_bytes_and_path_identically(self, tone_wav):
        assert np.array_equal(decode(tone_wav), decode(tone_wav.read_bytes()))

    def test_honours_requested_sample_rate(self, tone_wav):
        assert len(decode(tone_wav, sample_rate=8000)) == pytest.approx(8000, abs=200)

    def test_rejects_garbage(self):
        with pytest.raises(AudioDecodeError):
            decode(b"this is definitely not audio")

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(AudioDecodeError):
            decode(tmp_path / "nope.wav")

    def test_rejects_empty_input(self):
        with pytest.raises(AudioDecodeError):
            decode(b"")


class TestValidatePcm:
    def test_accepts_valid_input(self):
        validate_pcm(np.zeros(1600, dtype=np.float32), TARGET_SAMPLE_RATE)

    def test_rejects_stereo(self):
        with pytest.raises(AudioDecodeError, match="mono"):
            validate_pcm(np.zeros((100, 2), dtype=np.float32), TARGET_SAMPLE_RATE)

    def test_rejects_wrong_dtype(self):
        with pytest.raises(AudioDecodeError, match="float32"):
            validate_pcm(np.zeros(100, dtype=np.int16), TARGET_SAMPLE_RATE)

    def test_refuses_to_silently_resample(self):
        """A client sending 44.1 kHz has a bug; hiding it would hide theirs."""
        with pytest.raises(AudioDecodeError, match="resample first"):
            validate_pcm(np.zeros(100, dtype=np.float32), 44_100)

    def test_rejects_empty(self):
        with pytest.raises(AudioDecodeError, match="empty"):
            validate_pcm(np.zeros(0, dtype=np.float32), TARGET_SAMPLE_RATE)
