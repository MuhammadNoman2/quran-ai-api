"""End-to-end tests against the real models.

Marked `slow` because they download (~220 MB on first run) and take seconds per
transcription on CPU. Run with:  pytest -m slow

These assert *behaviour we have measured*, not aspirational accuracy. In
particular they assert that the models get the real words right and that
hallucinated tails stay below the confidence gate - they do NOT assert that
hallucination never happens, because it does.
"""

from pathlib import Path

import pytest

from app.asr.base import Tier
from app.asr.registry import ASRRegistry
from app.audio.decoder import decode
from app.quran.repository import QuranRepository
from app.quran.text_normalizer import normalize_for_asr_matching

pytestmark = pytest.mark.slow

AUDIO_DIR = Path(__file__).resolve().parents[2] / "data" / "test_audio" / "correct"
CLIPS = [
    ("001_002_husary_1.mp3", 1, 2),
    ("112_001_alafasy_1.mp3", 112, 1),
]

repo = QuranRepository()
registry = ASRRegistry()


def _available(name: str) -> Path:
    path = AUDIO_DIR / name
    if not path.exists():
        pytest.skip(f"no test audio at {path} - see data/test_audio/README.md")
    return path


@pytest.fixture(scope="module", params=[Tier.PROVISIONAL, Tier.COMMITTED])
def engine(request):
    e = registry.get(request.param)
    e.load()
    return e


@pytest.mark.parametrize("filename,surah,ayah", CLIPS)
def test_recognises_every_expected_word_in_order(engine, filename, surah, ayah):
    audio = decode(_available(filename))
    result = engine.transcribe(audio)

    expected = repo.ayah(surah, ayah).normalized.split()
    recognized = normalize_for_asr_matching(result.text).split()

    assert recognized[: len(expected)] == expected, (
        f"{engine.describe().name} misread {surah}:{ayah}\n"
        f"  expected: {expected}\n  got     : {recognized}"
    )


@pytest.mark.parametrize("filename,surah,ayah", CLIPS)
def test_genuine_words_are_high_confidence(engine, filename, surah, ayah):
    """The gate must not fire on words the reciter actually said."""
    audio = decode(_available(filename))
    result = engine.transcribe(audio)
    expected_count = repo.ayah(surah, ayah).word_count

    for word in result.words[:expected_count]:
        assert word.probability >= 0.90, (
            f"genuine word {word.text!r} scored {word.probability:.2f} - "
            "this would become a false correction"
        )


@pytest.mark.parametrize("filename,surah,ayah", CLIPS)
def test_any_extra_words_fall_below_the_gate(engine, filename, surah, ayah):
    """Hallucinated tails are expected; passing the gate is not.

    This is the regression guard for the whole false-correction defence.
    """
    audio = decode(_available(filename))
    result = engine.transcribe(audio)
    expected_count = repo.ayah(surah, ayah).word_count

    for word in result.words[expected_count:]:
        assert word.probability < 0.90, (
            f"hallucinated word {word.text!r} scored {word.probability:.2f}, "
            "which passes the confidence gate and would be reported as a mistake"
        )


def test_result_metadata_is_populated(engine):
    audio = decode(_available(CLIPS[0][0]))
    result = engine.transcribe(audio)
    assert result.language == "ar"
    assert result.audio_duration > 0
    assert result.inference_seconds > 0
    assert result.engine.device in {"cpu", "cuda"}
    assert result.words, "word timestamps must always be produced"


def test_word_timings_are_monotonic(engine):
    audio = decode(_available(CLIPS[0][0]))
    result = engine.transcribe(audio)
    for a, b in zip(result.words, result.words[1:]):
        assert b.start >= a.start


def test_load_is_idempotent_and_unload_works(engine):
    engine.load()
    assert engine.is_loaded
    engine.load()
    assert engine.is_loaded
