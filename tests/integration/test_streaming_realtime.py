"""Phase 7 behaviour against the real models and real recitation audio.

VAD-driven scheduling cannot be tested with synthetic tones - Silero is trained
on speech - so these use actual recordings and are marked slow.

    pytest -m slow tests/integration/test_streaming_realtime.py
"""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.audio.decoder import decode
from app.main import create_app

pytestmark = pytest.mark.slow

CORRECT = Path(__file__).resolve().parents[2] / "data" / "test_audio" / "correct"


def pcm_for(name: str) -> bytes:
    path = CORRECT / name
    if not path.exists():
        pytest.skip(f"no test audio at {path}")
    return (np.clip(decode(str(path)), -1, 1) * 32767).astype("<i2").tobytes()


@pytest.fixture(scope="module")
def client():
    deps.reset_caches()
    with TestClient(create_app()) as c:
        yield c
    deps.reset_caches()


def stream(client, name: str, surah: int, ayah: int) -> list[dict]:
    pcm = pcm_for(name)
    session = client.post(
        "/api/v1/sessions", json={"surah": surah, "ayah": ayah}
    ).json()["session_id"]
    frame = 16_000 * 2 // 10
    messages: list[dict] = []
    with client.websocket_connect(f"/api/v1/sessions/{session}/stream") as ws:
        ws.send_text(json.dumps({"type": "start", "surah": surah, "ayah": ayah}))
        for offset in range(0, len(pcm), frame):
            ws.send_bytes(pcm[offset : offset + frame])
        ws.send_text(json.dumps({"type": "stop"}))
        for _ in range(400):
            message = ws.receive_json()
            messages.append(message)
            if message["event"] == "session_completed":
                break
    return messages


def kinds(messages: list[dict]) -> list[str]:
    return [m["event"] for m in messages]


class TestShortAyah:
    @pytest.fixture(scope="class")
    def messages(self, client):
        return stream(client, "112_001_alafasy_1.mp3", 112, 1)

    def test_no_false_corrections(self, messages):
        mistakes = [m for m in messages if m["event"] == "mistake_detected"]
        assert mistakes == [], f"false correction: {mistakes}"

    def test_full_score(self, messages):
        completed = next(m for m in messages if m["event"] == "ayah_completed")
        assert completed["word_accuracy_score"] == 100.0

    def test_all_words_confirmed(self, messages):
        confirmed = [m["word_index"] for m in messages if m["event"] == "word_confirmed"]
        assert sorted(confirmed) == [1, 2, 3, 4]

    def test_speech_activity_is_reported(self, messages):
        assert "speech_started" in kinds(messages)


class TestLongAyah:
    """Ayat al-Kursi: 60.7 s, 50 words.

    This verse produced 46 false 'missing word' reports before the window
    advancement and contiguous-settled fixes, so it is the regression guard for
    both.
    """

    @pytest.fixture(scope="class")
    def messages(self, client):
        return stream(client, "002_255_husary_1.mp3", 2, 255)

    def test_no_false_corrections_on_a_long_verse(self, messages):
        mistakes = [m for m in messages if m["event"] == "mistake_detected"]
        assert mistakes == [], (
            f"{len(mistakes)} false corrections on a correct recitation: "
            f"{[(m['word_index'], m['error_type']) for m in mistakes[:5]]}"
        )

    def test_full_score(self, messages):
        completed = next(m for m in messages if m["event"] == "ayah_completed")
        assert completed["word_accuracy_score"] == 100.0

    def test_most_words_are_confirmed(self, messages):
        confirmed = {m["word_index"] for m in messages if m["event"] == "word_confirmed"}
        assert len(confirmed) >= 40, "the verse has 50 words; most should be confirmed"

    def test_confirmation_starts_early_not_at_the_end(self, messages):
        """Before window advancement, nothing was confirmed until the final pass."""
        order = kinds(messages)
        first_confirm = order.index("word_confirmed")
        assert first_confirm < order.index("session_completed") - 5

    def test_any_provisional_words_are_marked_as_such(self, messages):
        """The fast tier is a latency optimisation, not a correctness one.

        Whether it runs at all depends on pacing: a client uploading a recording
        faster than real time never gives it a gap to run in, and that is fine.
        What must hold is that anything it emits is labelled provisional.
        """
        detected = [m for m in messages if m["event"] == "word_detected"]
        assert all(m["state"] == "provisional" for m in detected)


class TestGenuineErrorsStillCaught:
    def test_wrong_verse_is_reported(self, client):
        """Suppressing hallucinations must not suppress real mistakes."""
        messages = stream(client, "001_002_husary_1.mp3", 1, 3)
        mistakes = [m for m in messages if m["event"] == "mistake_detected"]
        assert mistakes, "reciting the wrong verse must be reported"
        completed = next(m for m in messages if m["event"] == "ayah_completed")
        assert completed["word_accuracy_score"] < 100.0
