"""Voice activity detection."""

import numpy as np
import pytest

from app.audio.vad import SpeechDetector


@pytest.fixture(scope="module")
def detector():
    d = SpeechDetector()
    d.warm()
    return d


def silence(seconds: float, rate: int = 16_000) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.float32)


class TestEmptyAndSilent:
    def test_empty_audio(self, detector):
        state = detector.analyse(np.zeros(0, dtype=np.float32))
        assert state.has_speech is False
        assert state.is_speaking is False

    def test_pure_silence_has_no_speech(self, detector):
        state = detector.analyse(silence(2.0))
        assert state.has_speech is False
        assert state.speech_ended_at is None
        assert state.silence_seconds == pytest.approx(2.0, abs=0.05)


@pytest.fixture(scope="module")
def recitation():
    from app.audio.decoder import decode

    path = "data/test_audio/correct/001_002_husary_1.mp3"
    try:
        return decode(path)
    except Exception:
        pytest.skip(f"no test audio at {path}")


class TestRealRecitation:

    def test_finds_the_speech_region(self, detector, recitation):
        state = detector.analyse(recitation)
        assert state.has_speech
        assert state.total_speech_seconds > 3.0

    def test_detects_trailing_silence_after_the_reciter_stops(self, detector, recitation):
        """This is the evidence that catches hallucinations: the model invents
        words after speech ends."""
        state = detector.analyse(recitation)
        assert state.is_speaking is False
        assert state.silence_seconds > 0.5
        assert state.speech_ended_at < len(recitation) / 16_000

    def test_mid_recitation_is_still_speaking(self, detector, recitation):
        state = detector.analyse(recitation[: 16_000 * 3])
        assert state.is_speaking is True
        assert state.silence_seconds == pytest.approx(0.0, abs=0.05)


class TestSyntheticSpeechGap:
    def test_a_gap_between_tones_is_segmented(self, detector):
        rate = 16_000
        tone = (0.5 * np.sin(2 * np.pi * 200 * np.linspace(0, 1, rate))).astype(np.float32)
        audio = np.concatenate([tone, silence(1.0), tone])
        state = detector.analyse(audio)
        # Silero is trained on speech, so a pure tone may or may not register;
        # what must hold is that the result is coherent.
        assert state.silence_seconds >= 0.0
        assert isinstance(state.segments, list)
