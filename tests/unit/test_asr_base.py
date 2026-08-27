"""The ASR contract, tested without loading any model."""

import numpy as np
import pytest

from app.asr.base import ASREngine, ASRResult, ASRWord, EngineInfo, FakeASREngine, Tier


class TestASRWord:
    def test_rejects_probability_out_of_range(self):
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError, match="probability"):
                ASRWord(text="قل", start=0.0, end=0.5, probability=bad)

    def test_rejects_negative_duration(self):
        with pytest.raises(ValueError, match="ends before"):
            ASRWord(text="قل", start=1.0, end=0.5, probability=1.0)

    def test_accepts_boundaries(self):
        assert ASRWord("قل", 0.0, 0.0, 0.0).probability == 0.0
        assert ASRWord("قل", 0.0, 1.0, 1.0).probability == 1.0


def _result(words, duration=4.0, inference=2.0):
    return ASRResult(
        text=" ".join(w.text for w in words),
        words=tuple(words),
        language="ar",
        audio_duration=duration,
        inference_seconds=inference,
        engine=EngineInfo("test", Tier.COMMITTED, "cpu", "int8"),
    )


class TestASRResult:
    def test_words_below_finds_hallucination_candidates(self):
        words = [
            ASRWord("قل", 0.0, 0.4, 1.0),
            ASRWord("هو", 0.4, 0.8, 1.0),
            ASRWord("قط", 0.8, 1.0, 0.17),
        ]
        low = _result(words).words_below(0.9)
        assert [w.text for w in low] == ["قط"]

    def test_words_below_is_empty_for_a_clean_result(self):
        words = [ASRWord("قل", 0.0, 0.4, 1.0)]
        assert _result(words).words_below(0.9) == ()

    def test_real_time_factor(self):
        assert _result([], duration=4.0, inference=2.0).real_time_factor == 0.5

    def test_real_time_factor_with_zero_duration_does_not_divide_by_zero(self):
        assert _result([], duration=0.0).real_time_factor == 0.0

    def test_speech_end_is_last_word_end(self):
        words = [ASRWord("قل", 0.0, 0.4, 1.0), ASRWord("هو", 0.4, 1.25, 1.0)]
        assert _result(words).speech_end() == 1.25

    def test_speech_end_of_empty_result(self):
        assert _result([]).speech_end() == 0.0


class TestEngineInfo:
    def test_public_view_hides_the_model_identity(self):
        info = EngineInfo("quran-asr-committed", Tier.COMMITTED, "cpu", "int8")
        public = info.public()
        assert public == {"engine": "quran-asr-committed", "tier": "committed"}
        assert "whisper" not in str(public).lower()
        assert "ctranslate" not in str(public).lower()


class TestFakeEngine:
    def test_requires_load_before_transcribe(self):
        with pytest.raises(RuntimeError, match="not loaded"):
            FakeASREngine(text="قل").transcribe(np.zeros(16, dtype=np.float32))

    def test_context_manager_loads_and_unloads(self):
        engine = FakeASREngine(text="قل هو")
        assert not engine.is_loaded
        with engine as e:
            assert e.is_loaded
        assert not engine.is_loaded

    def test_returns_configured_probabilities(self):
        engine = FakeASREngine(text="قل هو قط", word_probabilities={"قط": 0.2})
        with engine:
            result = engine.transcribe(np.zeros(16000, dtype=np.float32))
        assert [round(w.probability, 2) for w in result.words] == [1.0, 1.0, 0.2]
        assert result.audio_duration == 1.0

    def test_is_a_real_asrengine(self):
        assert isinstance(FakeASREngine(), ASREngine)


class TestTier:
    def test_values_are_stable_strings(self):
        """Tier names appear in config and API output; changing them is breaking."""
        assert Tier.PROVISIONAL.value == "provisional"
        assert Tier.COMMITTED.value == "committed"


class TestNoStreamingMethod:
    def test_interface_has_no_transcribe_stream(self):
        """Deliberate: Whisper cannot decode incrementally (architecture 4)."""
        assert not hasattr(ASREngine, "transcribe_stream")
