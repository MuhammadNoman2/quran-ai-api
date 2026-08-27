"""WebSocket streaming contract, with a deterministic fake recognizer."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.asr.base import FakeASREngine, Tier
from app.core.config import Settings
from app.main import create_app
from app.streaming.events import NOT_YET_EMITTED, PROTOCOL_VERSION

FATIHA_2 = "الحمد لله رب العالمين"


def pcm(seconds: float = 5.0, sample_rate: int = 16_000) -> bytes:
    n = int(seconds * sample_rate)
    return (np.sin(np.linspace(0, 400, n)) * 8_000).astype("<i2").tobytes()


def install(text: str, probs: dict | None = None, tier: Tier = Tier.COMMITTED) -> None:
    engine = FakeASREngine(text=text, word_probabilities=probs or {}, tier=tier)
    engine.load()
    deps.get_registry().register(tier, engine)


@pytest.fixture
def client():
    deps.reset_caches()
    with TestClient(create_app()) as c:
        install(FATIHA_2)
        yield c
    deps.reset_caches()


def open_session(client, **body) -> str:
    return client.post("/api/v1/sessions", json=body or {"surah": 1, "ayah": 2}).json()[
        "session_id"
    ]


def drain(ws, stop_at="session_completed", limit=60) -> list[dict]:
    out = []
    for _ in range(limit):
        message = ws.receive_json()
        out.append(message)
        if message["event"] == stop_at:
            break
    return out


def kinds(messages) -> list[str]:
    return [m["event"] for m in messages]


class TestHandshake:
    def test_start_produces_session_started_then_ready(self, client):
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start", "surah": 1, "ayah": 2}))
            assert ws.receive_json()["event"] == "session_started"
            ready = ws.receive_json()
            assert ready["event"] == "ready"
            assert ready["expected_words"] == 4
            assert ready["sample_rate"] == 16_000

    def test_every_message_carries_the_protocol_version(self, client):
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            assert ws.receive_json()["v"] == PROTOCOL_VERSION

    def test_ready_reports_the_engine_by_opaque_label(self, client):
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            ws.receive_json()
            engine = ws.receive_json()["engine"]
            assert "tier" in engine
            assert "whisper" not in json.dumps(engine).lower()

    def test_unknown_session_is_refused(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect("/api/v1/sessions/nope/stream"):
                pass

    def test_starting_twice_is_reported(self, client):
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            drain(ws, stop_at="ready")
            ws.send_text(json.dumps({"type": "start"}))
            message = ws.receive_json()
            assert message["event"] == "error"
            assert message["code"] == "ALREADY_STARTED"


class TestRecognitionFlow:
    def test_words_are_confirmed_and_the_ayah_completes(self, client):
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            ws.send_bytes(pcm())
            ws.send_text(json.dumps({"type": "stop"}))
            messages = drain(ws)

        confirmed = [m for m in messages if m["event"] == "word_confirmed"]
        assert [m["word_index"] for m in confirmed] == [1, 2, 3, 4]
        completed = next(m for m in messages if m["event"] == "ayah_completed")
        assert completed["word_accuracy_score"] == 100.0
        assert kinds(messages)[-1] == "session_completed"

    def test_a_word_is_confirmed_only_once(self, client):
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            for _ in range(3):
                ws.send_bytes(pcm(5.0))
                ws.send_text(json.dumps({"type": "flush"}))
            ws.send_text(json.dumps({"type": "stop"}))
            messages = drain(ws, limit=120)
        indices = [m["word_index"] for m in messages if m["event"] == "word_confirmed"]
        assert len(indices) == len(set(indices))

    def test_partial_transcript_is_emitted(self, client):
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            ws.send_bytes(pcm())
            ws.send_text(json.dumps({"type": "stop"}))
            messages = drain(ws)
        partial = next(m for m in messages if m["event"] == "partial_transcript")
        assert partial["final"] is False
        assert partial["text"] == FATIHA_2


class TestMistakes:
    def test_a_real_substitution_is_reported(self, client):
        install("الحمد لله مالك العالمين")
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            ws.send_bytes(pcm())
            ws.send_text(json.dumps({"type": "stop"}))
            messages = drain(ws)
        mistakes = [m for m in messages if m["event"] == "mistake_detected"]
        assert mistakes
        assert mistakes[0]["error_type"] == "word_substitution"
        assert mistakes[0]["word_index"] == 3

    def test_a_hallucinated_tail_is_never_reported(self, client):
        install(f"{FATIHA_2} الحمد", {"الحمد": 0.46})
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            ws.send_bytes(pcm())
            ws.send_text(json.dumps({"type": "stop"}))
            messages = drain(ws)
        assert not [m for m in messages if m["event"] == "mistake_detected"]
        assert next(m for m in messages if m["event"] == "ayah_completed")[
            "word_accuracy_score"
        ] == 100.0

    def test_words_not_yet_reached_are_not_called_missing(self, client):
        """A student mid-verse has not omitted the rest of it.

        Only `stop` settles the whole ayah; until then, unspoken words are
        pending, not mistakes.
        """
        install("الحمد لله")           # only the first two of four words
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            ws.send_bytes(pcm())
            ws.send_text(json.dumps({"type": "flush"}))
            messages = []
            for _ in range(10):
                messages.append(ws.receive_json())
                if messages[-1]["event"] == "ayah_progress":
                    break
            assert not [m for m in messages if m["event"] == "mistake_detected"]

            # Finishing there does settle them as genuinely missing.
            ws.send_text(json.dumps({"type": "stop"}))
            final = drain(ws)
        assert [m for m in final if m["event"] == "mistake_detected"]

    def test_the_same_mistake_is_not_repeated(self, client):
        install("الحمد لله مالك العالمين")
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            for _ in range(3):
                ws.send_bytes(pcm(5.0))
                ws.send_text(json.dumps({"type": "flush"}))
            ws.send_text(json.dumps({"type": "stop"}))
            messages = drain(ws, limit=120)
        keys = [(m["word_index"], m["error_type"]) for m in messages if m["event"] == "mistake_detected"]
        assert len(keys) == len(set(keys))


class TestResilience:
    """A bad frame must never end a session (brief 40)."""

    def test_audio_before_start(self, client):
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_bytes(pcm(0.1))
            message = ws.receive_json()
            assert message["code"] == "NOT_STARTED"
            ws.send_text(json.dumps({"type": "start"}))
            assert ws.receive_json()["event"] == "session_started"

    def test_malformed_json(self, client):
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text("{not json")
            assert ws.receive_json()["code"] == "INVALID_MESSAGE"
            ws.send_text(json.dumps({"type": "start"}))
            assert ws.receive_json()["event"] == "session_started"

    def test_odd_length_pcm_frame(self, client):
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            drain(ws, stop_at="ready")
            ws.send_bytes(b"\x00\x00\x00")
            message = ws.receive_json()
            assert message["event"] == "error"
            assert message["code"] == "INVALID_AUDIO"
            assert message["fatal"] is False
            ws.send_bytes(pcm())
            ws.send_text(json.dumps({"type": "stop"}))
            assert "session_completed" in kinds(drain(ws))

    def test_unknown_message_type(self, client):
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "nonsense"}))
            message = ws.receive_json()
            assert message["code"] == "INVALID_MESSAGE"
            assert "nonsense" in message["message"]

    def test_nonexistent_ayah_at_start(self, client):
        sid = open_session(client, surah=None, ayah=None)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start", "surah": 1, "ayah": 999}))
            assert ws.receive_json()["code"] == "VERSE_NOT_FOUND"

    def test_half_a_reference(self, client):
        sid = open_session(client, surah=None, ayah=None)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start", "surah": 1}))
            assert ws.receive_json()["code"] == "INVALID_MESSAGE"


class TestUnimplementedEvents:
    def test_phase_7_events_never_appear(self, client):
        sid = open_session(client)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            ws.send_bytes(pcm())
            ws.send_text(json.dumps({"type": "stop"}))
            seen = set(kinds(drain(ws)))
        assert seen.isdisjoint({e.value for e in NOT_YET_EMITTED})


class TestVerseDetection:
    def test_verse_is_inferred_when_the_session_did_not_name_one(self, client):
        sid = open_session(client, surah=None, ayah=None)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            ws.send_bytes(pcm())
            ws.send_text(json.dumps({"type": "stop"}))
            messages = drain(ws)
        started = [m for m in messages if m["event"] == "session_started"]
        assert started[-1]["surah"] == 1 and started[-1]["ayah"] == 2

    def test_unrecognisable_audio_warns_rather_than_failing(self, client):
        install("hello world not quran at all")
        sid = open_session(client, surah=None, ayah=None)
        with client.websocket_connect(f"/api/v1/sessions/{sid}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            ws.send_bytes(pcm())
            ws.send_text(json.dumps({"type": "stop"}))
            messages = drain(ws)
        assert [m for m in messages if m["event"] == "warning"]
        assert kinds(messages)[-1] == "session_completed"


class TestSessionLifecycle:
    def test_closing_the_stream_frees_the_slot(self, client):
        ids = [open_session(client) for _ in range(3)]
        assert client.post("/api/v1/sessions", json={}).status_code == 503
        with client.websocket_connect(f"/api/v1/sessions/{ids[0]}/stream") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            drain(ws, stop_at="ready")
        assert client.post("/api/v1/sessions", json={}).status_code == 201


class TestAuth:
    def test_websocket_requires_a_key_when_configured(self):
        deps.reset_caches()
        config = Settings(api_keys="secret")
        from app.core.security import AuthProvider, RateLimiter

        app = create_app()
        app.dependency_overrides[deps.get_settings] = lambda: config
        app.dependency_overrides[deps.get_auth] = lambda: AuthProvider({"secret"})
        app.dependency_overrides[deps.get_rate_limiter] = lambda: RateLimiter(0)
        with TestClient(app) as c:
            install(FATIHA_2)
            sid = c.post(
                "/api/v1/sessions", json={"surah": 1, "ayah": 2},
                headers={"X-API-Key": "secret"},
            ).json()["session_id"]
            with pytest.raises(Exception):
                with c.websocket_connect(f"/api/v1/sessions/{sid}/stream"):
                    pass
            with c.websocket_connect(
                f"/api/v1/sessions/{sid}/stream?api_key=secret"
            ) as ws:
                ws.send_text(json.dumps({"type": "start"}))
                assert ws.receive_json()["event"] == "session_started"
        deps.reset_caches()
