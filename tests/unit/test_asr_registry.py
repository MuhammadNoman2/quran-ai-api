"""Registry behaviour and the device-resolution rules."""

import pytest

from app.asr.base import FakeASREngine, Tier
from app.asr.registry import ASRRegistry
from app.asr.whisper_engine import FasterWhisperEngine, resolve_device
from app.core.config import Settings


class TestResolveDevice:
    def test_explicit_device_passes_through(self):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda") == "cuda"

    def test_auto_resolves_to_something_ctranslate2_supports(self):
        assert resolve_device("auto") in {"cpu", "cuda"}

    def test_mps_is_rejected_loudly(self):
        """CTranslate2 has no MPS backend; silently using CPU would mislead."""
        with pytest.raises(ValueError, match="MPS"):
            resolve_device("mps")


class TestRegistry:
    def test_returns_the_same_engine_instance_per_tier(self):
        """Engines are shared across sessions; one model per session wastes CPU."""
        registry = ASRRegistry(Settings())
        assert registry.get(Tier.COMMITTED) is registry.get(Tier.COMMITTED)

    def test_tiers_are_separate_engines(self):
        registry = ASRRegistry(Settings())
        assert registry.get(Tier.PROVISIONAL) is not registry.get(Tier.COMMITTED)

    def test_tiers_use_their_configured_models(self):
        settings = Settings(
            asr_provisional_model="model-p", asr_committed_model="model-c"
        )
        registry = ASRRegistry(settings)
        assert registry.get(Tier.PROVISIONAL)._model_id == "model-p"
        assert registry.get(Tier.COMMITTED)._model_id == "model-c"

    def test_engines_are_not_loaded_until_asked(self):
        """Constructing must not download or load a model."""
        registry = ASRRegistry(Settings())
        assert not registry.get(Tier.COMMITTED).is_loaded

    def test_register_substitutes_a_fake(self):
        registry = ASRRegistry(Settings())
        fake = FakeASREngine(text="قل هو الله أحد")
        registry.register(Tier.COMMITTED, fake)
        assert registry.get(Tier.COMMITTED) is fake

    def test_describe_exposes_no_model_identity(self):
        registry = ASRRegistry(Settings())
        registry.get(Tier.COMMITTED)
        blob = str(registry.describe()).lower()
        assert "whisper" not in blob and "tarteel" not in blob


class TestEngineConstruction:
    def test_defaults_to_the_measured_decoder_settings(self):
        """condition_on_previous_text=True was chosen by A/B measurement."""
        engine = FasterWhisperEngine("m", name="n", tier=Tier.COMMITTED, device="cpu")
        assert engine._condition_on_previous_text is True
        assert engine._beam_size == 5

    def test_engine_name_is_opaque(self):
        engine = FasterWhisperEngine(
            "tarteel-ai/whisper-base", name="quran-asr-committed",
            tier=Tier.COMMITTED, device="cpu",
        )
        assert engine.describe().name == "quran-asr-committed"
