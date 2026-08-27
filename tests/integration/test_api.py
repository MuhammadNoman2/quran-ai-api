"""HTTP contract tests.

A fake ASR engine is injected so these run fast and assert exact behaviour.
Real-model behaviour is covered in test_false_correction.py.
"""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.api import deps
from app.asr.base import FakeASREngine, Tier
from app.core.config import Settings
from app.main import create_app

FATIHA_2 = "الحمد لله رب العالمين"
IKHLAS_1 = "قل هو الله أحد"


def wav_bytes(seconds: float = 2.0, sample_rate: int = 16_000) -> bytes:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    buffer = io.BytesIO()
    sf.write(buffer, 0.3 * np.sin(2 * np.pi * 220 * t), sample_rate, format="WAV")
    return buffer.getvalue()


def upload(data: bytes = b"", name: str = "clip.wav"):
    return {"audio": (name, io.BytesIO(data or wav_bytes()), "audio/wav")}


@pytest.fixture
def client(monkeypatch):
    """A client whose ASR is deterministic and whose caches are isolated."""
    deps.reset_caches()
    with TestClient(create_app()) as c:
        c.set_asr = lambda text, probs=None, tier=Tier.COMMITTED: _install(text, probs, tier)
        _install(FATIHA_2, None, Tier.COMMITTED)
        _install(FATIHA_2, None, Tier.PROVISIONAL)
        yield c
    deps.reset_caches()


def _install(text, probs, tier):
    engine = FakeASREngine(text=text, word_probabilities=probs or {}, tier=tier)
    engine.load()
    deps.get_registry().register(tier, engine)


def app_with(**kwargs):
    """An app configured differently, via FastAPI's dependency overrides.

    Monkeypatching `deps.get_settings` does not work: FastAPI captures the
    callable inside `Depends(...)` at import time, so the routes keep the
    original. `dependency_overrides` is the supported mechanism.
    """
    from app.core.security import AuthProvider, RateLimiter

    deps.reset_caches()
    config = Settings(**kwargs)
    auth = AuthProvider(config.allowed_api_keys)
    limiter = RateLimiter(config.rate_limit_per_minute)

    application = create_app()
    application.dependency_overrides[deps.get_settings] = lambda: config
    application.dependency_overrides[deps.get_auth] = lambda: auth
    application.dependency_overrides[deps.get_rate_limiter] = lambda: limiter
    return application


class TestHealth:
    def test_reports_ok(self, client):
        body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"

    def test_reports_resolved_device_not_the_literal_auto(self, client):
        assert client.get("/api/v1/health").json()["device"] in {"cpu", "cuda"}

    def test_surfaces_security_posture(self, client):
        body = client.get("/api/v1/health").json()
        assert body["auth_enabled"] is False
        assert body["store_audio"] is False

    def test_does_not_leak_model_identity(self, client):
        body = str(client.get("/api/v1/health").json()).lower()
        assert "whisper" not in body and "tarteel" not in body


class TestQuranEndpoints:
    def test_lists_114_surahs(self, client):
        assert len(client.get("/api/v1/surahs").json()) == 114

    def test_single_surah(self, client):
        body = client.get("/api/v1/surahs/1").json()
        assert body["ayah_count"] == 7

    def test_single_ayah_includes_words(self, client):
        body = client.get("/api/v1/surahs/1/ayahs/2").json()
        assert body["uthmani"].startswith("ٱ")
        assert len(body["words"]) == 4
        assert body["words"][0]["uthmani_index"] == 1

    def test_unknown_ayah_is_404(self, client):
        r = client.get("/api/v1/surahs/1/ayahs/99")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "VERSE_NOT_FOUND"

    def test_surah_out_of_range_is_422(self, client):
        assert client.get("/api/v1/surahs/115").status_code == 422


class TestAnalyze:
    def test_perfect_recitation(self, client):
        r = client.post(
            "/api/v1/recitation/analyze", files=upload(), data={"surah": 1, "ayah": 2}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["word_accuracy_score"] == 100.0
        assert body["errors"] == []
        assert body["expected_text"].startswith("ٱ")

    def test_detects_the_verse_when_not_supplied(self, client):
        r = client.post("/api/v1/recitation/analyze", files=upload())
        body = r.json()
        assert body["verse_detected"] is True
        assert (body["surah"], body["ayah"]) == (1, 2)

    def test_reports_a_real_mistake(self, client):
        client.set_asr("الحمد لله مالك العالمين")
        body = client.post(
            "/api/v1/recitation/analyze", files=upload(), data={"surah": 1, "ayah": 2}
        ).json()
        assert len(body["errors"]) == 1
        assert body["errors"][0]["category"] == "word_substitution"

    def test_hallucination_is_not_reported_as_an_error(self, client):
        client.set_asr(f"{FATIHA_2} الحمد", {"الحمد": 0.46})
        body = client.post(
            "/api/v1/recitation/analyze", files=upload(), data={"surah": 1, "ayah": 2}
        ).json()
        assert body["errors"] == []
        assert body["word_accuracy_score"] == 100.0

    def test_half_a_reference_is_rejected(self, client):
        r = client.post("/api/v1/recitation/analyze", files=upload(), data={"surah": 1})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_unknown_verse_is_404(self, client):
        r = client.post(
            "/api/v1/recitation/analyze", files=upload(), data={"surah": 1, "ayah": 99}
        )
        assert r.status_code == 404

    def test_unidentifiable_recitation(self, client):
        client.set_asr("hello world not quran")
        r = client.post("/api/v1/recitation/analyze", files=upload())
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "VERSE_NOT_IDENTIFIED"

    def test_score_is_not_called_a_tajweed_score(self, client):
        body = client.post(
            "/api/v1/recitation/analyze", files=upload(), data={"surah": 1, "ayah": 2}
        ).json()
        assert "word_accuracy_score" in body
        assert "tajweed" not in str(body).lower()


class TestAudioValidation:
    def test_garbage_upload_is_rejected(self, client):
        r = client.post(
            "/api/v1/recitation/analyze",
            files=upload(b"definitely not audio"),
            data={"surah": 1, "ayah": 2},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "INVALID_AUDIO"

    def test_empty_upload_is_rejected(self, client):
        r = client.post(
            "/api/v1/recitation/analyze",
            files={"audio": ("x.wav", io.BytesIO(b""), "audio/wav")},
            data={"surah": 1, "ayah": 2},
        )
        assert r.status_code == 400

    def test_oversized_upload_is_rejected(self):
        with TestClient(app_with(max_audio_bytes=1024)) as c:
            _install(FATIHA_2, None, Tier.COMMITTED)
            r = c.post(
                "/api/v1/recitation/analyze",
                files=upload(wav_bytes(2.0)),
                data={"surah": 1, "ayah": 2},
            )
            assert r.status_code == 413
            assert r.json()["error"]["code"] == "AUDIO_TOO_LARGE"
        deps.reset_caches()

    def test_overlong_audio_is_rejected(self):
        with TestClient(app_with(max_audio_seconds=0.5)) as c:
            _install(FATIHA_2, None, Tier.COMMITTED)
            r = c.post(
                "/api/v1/recitation/analyze",
                files=upload(wav_bytes(3.0)),
                data={"surah": 1, "ayah": 2},
            )
            assert r.status_code == 413
            assert r.json()["error"]["code"] == "AUDIO_TOO_LONG"
        deps.reset_caches()


class TestPrivacy:
    def test_analysis_writes_no_files(self, client, tmp_path, monkeypatch):
        """STORE_AUDIO is false, so no code path may persist a recording."""
        monkeypatch.chdir(tmp_path)
        before = set(tmp_path.rglob("*"))
        client.post(
            "/api/v1/recitation/analyze", files=upload(), data={"surah": 1, "ayah": 2}
        )
        assert set(tmp_path.rglob("*")) == before

    def test_response_contains_no_audio(self, client):
        body = client.post(
            "/api/v1/recitation/analyze", files=upload(), data={"surah": 1, "ayah": 2}
        ).json()
        assert "audio" not in body
        assert "filename" not in str(body)


class TestDetectVerse:
    def test_from_text(self, client):
        r = client.post("/api/v1/recitation/detect-verse", data={"text": FATIHA_2})
        assert r.status_code == 200
        assert (r.json()["surah"], r.json()["ayah"]) == (1, 2)

    def test_from_audio(self, client):
        client.set_asr(IKHLAS_1, tier=Tier.PROVISIONAL)
        body = client.post("/api/v1/recitation/detect-verse", files=upload()).json()
        assert (body["surah"], body["ayah"]) == (112, 1)

    def test_repeated_verse_reports_alternatives(self, client):
        body = client.post(
            "/api/v1/recitation/detect-verse", data={"text": "فبأي آلاء ربكما تكذبان"}
        ).json()
        assert body["alternatives"] == 30

    def test_requires_exactly_one_input(self, client):
        r = client.post("/api/v1/recitation/detect-verse")
        assert r.status_code == 400
        r = client.post(
            "/api/v1/recitation/detect-verse", files=upload(), data={"text": FATIHA_2}
        )
        assert r.status_code == 400

    def test_unmatched_text_is_404(self, client):
        r = client.post("/api/v1/recitation/detect-verse", data={"text": "hello world"})
        assert r.status_code == 404


class TestSessions:
    def test_create_and_fetch(self, client):
        created = client.post("/api/v1/sessions", json={"surah": 1, "ayah": 2}).json()
        assert created["capacity"] == 3
        fetched = client.get(f"/api/v1/sessions/{created['session_id']}").json()
        assert fetched["session_id"] == created["session_id"]

    def test_websocket_url_is_advertised(self, client):
        body = client.post("/api/v1/sessions", json={}).json()
        assert body["websocket_url"].endswith("/stream")

    def test_capacity_is_enforced(self, client):
        for _ in range(3):
            assert client.post("/api/v1/sessions", json={}).status_code == 201
        r = client.post("/api/v1/sessions", json={})
        assert r.status_code == 503
        assert "concurrent sessions" in r.json()["error"]["message"]

    def test_closing_frees_a_slot(self, client):
        ids = [client.post("/api/v1/sessions", json={}).json()["session_id"] for _ in range(3)]
        assert client.post("/api/v1/sessions", json={}).status_code == 503
        assert client.delete(f"/api/v1/sessions/{ids[0]}").status_code == 204
        assert client.post("/api/v1/sessions", json={}).status_code == 201

    def test_unsupported_sample_rate_is_rejected(self, client):
        r = client.post("/api/v1/sessions", json={"sample_rate": 12345})
        assert r.status_code == 400
        assert "unsupported sample rate" in r.json()["error"]["message"]

    def test_unsupported_format_is_rejected(self, client):
        r = client.post("/api/v1/sessions", json={"format": "mp3"})
        assert r.status_code == 400

    def test_unknown_session_is_404(self, client):
        assert client.get("/api/v1/sessions/nope").status_code == 404


class TestAuth:
    def test_disabled_by_default(self, client):
        assert client.get("/api/v1/surahs").status_code == 200

    def test_enabled_when_keys_configured(self):
        with TestClient(app_with(api_keys="secret-key")) as c:
            assert c.get("/api/v1/surahs").status_code == 401
            assert c.get("/api/v1/surahs", headers={"X-API-Key": "wrong"}).status_code == 401
            assert c.get("/api/v1/surahs", headers={"X-API-Key": "secret-key"}).status_code == 200
            assert c.get("/api/v1/health").status_code == 200, "health must stay open"
        deps.reset_caches()


class TestRateLimit:
    def test_disabled_by_default(self, client):
        for _ in range(20):
            assert client.get("/api/v1/surahs").status_code == 200

    def test_enforced_when_configured(self):
        with TestClient(app_with(rate_limit_per_minute=3)) as c:
            for _ in range(3):
                assert c.get("/api/v1/surahs").status_code == 200
            r = c.get("/api/v1/surahs")
            assert r.status_code == 429
            assert r.json()["error"]["code"] == "RATE_LIMITED"
        deps.reset_caches()


class TestOpenAPI:
    def test_spec_is_served(self, client):
        spec = client.get("/openapi.json").json()
        assert spec["info"]["title"] == "Quran AI Recitation API"

    def test_all_documented_endpoints_exist(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        for expected in (
            "/api/v1/health",
            "/api/v1/surahs",
            "/api/v1/surahs/{surah_id}",
            "/api/v1/surahs/{surah_id}/ayahs/{ayah_id}",
            "/api/v1/recitation/analyze",
            "/api/v1/recitation/detect-verse",
            "/api/v1/sessions",
        ):
            assert expected in paths

    def test_docs_page_renders(self, client):
        assert client.get("/docs").status_code == 200


class TestPhonetics:
    def test_returns_expected_phonemes(self, client):
        body = client.get("/api/v1/surahs/1/ayahs/2/phonetics").json()
        assert body["phonemes"]
        assert len(body["words"]) == 4
        assert all(w["phonemes"] for w in body["words"])

    def test_names_the_tajweed_rules_present(self, client):
        body = client.get("/api/v1/surahs/1/ayahs/2/phonetics").json()
        assert "Normal Madd" in body["tajweed_rules"]

    def test_discloses_the_recitation_style_it_assumed(self, client):
        """Hafs permits a range of madd lengths, so the expectations are only
        correct for a stated style. Returning them silently would invite a client
        to treat one school's timing as universal."""
        body = client.get("/api/v1/surahs/1/ayahs/2/phonetics").json()
        assert body["rewaya"] == "hafs"
        assert set(body["madd_lengths"]) == {"monfasel", "mottasel", "mottasel_waqf", "aared"}

    def test_carries_articulation_attributes(self, client):
        body = client.get("/api/v1/surahs/112/ayahs/1/phonetics").json()
        assert body["sifat"]
        assert "ghonna" in body["sifat"][0]

    def test_unknown_ayah_is_404(self, client):
        assert client.get("/api/v1/surahs/1/ayahs/99/phonetics").status_code == 404


class TestPronunciationEndpoint:
    def test_reports_unavailability_honestly(self, client):
        """On a machine without torch there is no phoneme model, and the API must
        say so rather than returning an empty findings list that reads as a pass."""
        r = client.post(
            "/api/v1/recitation/pronunciation",
            files=upload(),
            data={"surah": 1, "ayah": 2},
        )
        assert r.status_code == 200
        body = r.json()
        if not body["available"]:
            assert body["findings"] == []
            assert body["unavailable_reason"]
            assert body["reference_phonemes"], "expected phonetics are still useful"

    def test_unknown_ayah_is_404(self, client):
        r = client.post(
            "/api/v1/recitation/pronunciation",
            files=upload(),
            data={"surah": 1, "ayah": 99},
        )
        assert r.status_code == 404

    def test_rejects_bad_audio(self, client):
        r = client.post(
            "/api/v1/recitation/pronunciation",
            files=upload(b"not audio"),
            data={"surah": 1, "ayah": 2},
        )
        assert r.status_code == 400


class TestHealthPhonemeCapability:
    def test_reports_whether_phoneme_analysis_can_run(self, client):
        body = client.get("/api/v1/health").json()
        assert "phoneme_analysis" in body
        assert body["phoneme_analysis"]["available"] in {"true", "false"}

    def test_gives_a_reason_when_it_cannot(self, client):
        capability = client.get("/api/v1/health").json()["phoneme_analysis"]
        if capability["available"] == "false":
            assert capability["reason"]
