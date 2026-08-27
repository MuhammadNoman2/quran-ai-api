"""The Phase 4 exit criterion: correct recitations must produce zero corrections.

This is the regression guard for the whole project. The base model was measured
hallucinating trailing words on *every* short clip, so without the confidence and
speech-boundary gates each of these clips would report an extra-word error against
a professional reciter.

Run with: pytest -m slow
"""

from pathlib import Path

import pytest

from app.asr.base import Tier
from app.asr.registry import ASRRegistry
from app.audio.decoder import decode
from app.recitation.analyzer import RecitationAnalyzer

pytestmark = pytest.mark.slow

CORRECT_DIR = Path(__file__).resolve().parents[2] / "data" / "test_audio" / "correct"
CLIPS = [("001_002_husary_1.mp3", 1, 2), ("112_001_alafasy_1.mp3", 112, 1)]

registry = ASRRegistry()


def _clip(name: str) -> Path:
    path = CORRECT_DIR / name
    if not path.exists():
        pytest.skip(f"no test audio at {path} - see data/test_audio/README.md")
    return path


@pytest.fixture(scope="module", params=[Tier.PROVISIONAL, Tier.COMMITTED])
def analyzer(request):
    engine = registry.get(request.param)
    engine.load()
    return RecitationAnalyzer(engine)


@pytest.mark.parametrize("filename,surah,ayah", CLIPS)
def test_correct_recitation_produces_no_errors(analyzer, filename, surah, ayah):
    result = analyzer.analyze(decode(_clip(filename)), surah=surah, ayah=ayah)
    assert result.errors == [], (
        f"FALSE CORRECTION on a correct recitation of {surah}:{ayah}\n"
        f"  recognized: {result.recognized_text}\n"
        f"  reported  : {[(e.expected, e.spoken, e.category) for e in result.errors]}"
    )


@pytest.mark.parametrize("filename,surah,ayah", CLIPS)
def test_correct_recitation_scores_full_marks(analyzer, filename, surah, ayah):
    result = analyzer.analyze(decode(_clip(filename)), surah=surah, ayah=ayah)
    assert result.word_accuracy_score == 100.0


@pytest.mark.parametrize("filename,surah,ayah", CLIPS)
def test_every_expected_word_is_matched(analyzer, filename, surah, ayah):
    """Zero errors would be trivially achievable by reporting nothing."""
    result = analyzer.analyze(decode(_clip(filename)), surah=surah, ayah=ayah)
    matched = [w for w in result.words if w.status == "correct"]
    assert len(matched) == len(
        [w for w in result.words if w.index is not None]
    ), "some expected words were not recognized at all"


@pytest.mark.parametrize("filename,surah,ayah", CLIPS)
def test_hallucinations_are_present_but_suppressed(analyzer, filename, surah, ayah):
    """Documents reality: the models do hallucinate; the gates are what save us."""
    result = analyzer.analyze(decode(_clip(filename)), surah=surah, ayah=ayah)
    extras = [w for w in result.words if w.index is None]
    for extra in extras:
        assert extra.note, f"extra word {extra.spoken!r} was not given a suppression reason"


def test_verse_detection_works_on_real_audio(analyzer):
    """Scenario B end to end: audio in, correct verse out, nothing supplied."""
    result = analyzer.analyze(decode(_clip("112_001_alafasy_1.mp3")))
    assert (result.surah, result.ayah) == (112, 1)
    assert result.verse_detected
