"""Recitation audio, and the licensing discipline around it."""

import pytest

from app.core.config import Settings
from app.recitations.registry import (
    DEFAULT_RECITER, LicenceStatus, all_reciters, get,
)
from app.recitations.repository import (
    RecitationForbidden, RecitationRepository, RecitationUnavailable,
)


class TestRegistry:
    def test_reciters_are_registered(self):
        assert len(all_reciters()) >= 5

    def test_default_reciter_exists(self):
        assert get(DEFAULT_RECITER) is not None

    def test_every_reciter_declares_a_licence_and_explains_it(self):
        """A recitation is a performance. Silence about rights is not acceptable."""
        for reciter in all_reciters():
            assert reciter.licence in LicenceStatus
            assert reciter.licence_note, f"{reciter.id} states no licence note"

    def test_unverified_reciters_are_not_redistributable(self):
        for reciter in all_reciters():
            if reciter.licence is not LicenceStatus.VERIFIED:
                assert reciter.redistributable is False

    def test_url_is_built_with_zero_padded_keys(self):
        assert get("husary").url_for(1, 2).endswith("001002.mp3")
        assert get("husary").url_for(114, 6).endswith("114006.mp3")

    def test_unknown_reciter_returns_none(self):
        assert get("not-a-reciter") is None


class TestRepository:
    def test_uncached_ayah_points_at_the_source(self, tmp_path):
        recitation = RecitationRepository(tmp_path).get(1, 2, "husary")
        assert recitation.local is False
        assert recitation.audio_url.startswith("https://")

    def test_unknown_reciter_raises(self, tmp_path):
        with pytest.raises(RecitationUnavailable):
            RecitationRepository(tmp_path).get(1, 2, "nobody")

    def test_cached_but_unverified_audio_is_not_served_locally(self, tmp_path, monkeypatch):
        """The central rule: caching a file is not permission to redistribute it."""
        from app.core import config

        path = tmp_path / "husary" / "001002.mp3"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fake mp3")

        monkeypatch.setattr(config.settings, "serve_unverified_audio", False)
        recitation = RecitationRepository(tmp_path).get(1, 2, "husary", base_url="/audio")
        assert recitation.local is False
        assert recitation.audio_url.startswith("https://")

    def test_it_is_served_when_explicitly_enabled(self, tmp_path, monkeypatch):
        from app.core import config

        path = tmp_path / "husary" / "001002.mp3"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"fake mp3")

        monkeypatch.setattr(config.settings, "serve_unverified_audio", True)
        recitation = RecitationRepository(tmp_path).get(1, 2, "husary", base_url="/audio")
        assert recitation.local is True
        assert recitation.audio_url == "/audio/husary/001002.mp3"
        assert recitation.size_bytes == len(b"fake mp3")

    def test_counts_cached_files(self, tmp_path):
        directory = tmp_path / "husary"
        directory.mkdir(parents=True)
        for name in ("001001.mp3", "001002.mp3"):
            (directory / name).write_bytes(b"x")
        assert RecitationRepository(tmp_path).cached_count("husary") == 2
        assert RecitationRepository(tmp_path).cached_count("alafasy") == 0


class TestDefaults:
    def test_serving_unverified_audio_is_off_by_default(self):
        """Shipping unlicensed recordings should require a deliberate decision."""
        assert Settings().serve_unverified_audio is False
