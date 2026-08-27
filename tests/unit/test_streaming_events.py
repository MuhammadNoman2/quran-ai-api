"""The wire protocol."""

from app.streaming import events
from app.streaming.events import NOT_YET_EMITTED, PROTOCOL_VERSION, EventType


class TestEnvelope:
    def test_every_event_carries_a_protocol_version(self):
        built = [
            events.session_started("s", 1, 2),
            events.ready(sample_rate=16_000, audio_format="pcm_s16le", expected_words=4, engine={}),
            events.partial_transcript("x", seconds=1.0),
            events.word_confirmed(index=1, expected="a", spoken="a", confidence=1.0),
            events.ayah_progress(surah=1, ayah=2, words_confirmed=1, words_total=4),
            events.warning("careful"),
            events.error(events.ErrorCode.INVALID_AUDIO, "bad"),
        ]
        for message in built:
            assert message["v"] == PROTOCOL_VERSION
            assert "event" in message

    def test_event_names_are_stable(self):
        """Clients switch on these strings; renaming one is a breaking change."""
        for name in (
            "session_started", "ready", "partial_transcript", "word_confirmed",
            "mistake_detected", "ayah_progress", "ayah_completed",
            "session_completed", "warning", "error",
        ):
            assert EventType(name)


class TestSemantics:
    def test_partial_transcript_is_marked_not_final(self):
        assert events.partial_transcript("x", seconds=1.0)["final"] is False

    def test_errors_are_non_fatal_by_default(self):
        """A bad frame must not end a session."""
        assert events.error(events.ErrorCode.INVALID_AUDIO, "bad")["fatal"] is False

    def test_progress_fraction_handles_zero_total(self):
        assert events.ayah_progress(surah=1, ayah=2, words_confirmed=0, words_total=0)["fraction"] == 0.0

    def test_mistake_carries_the_location_and_category(self):
        message = events.mistake_detected(
            index=3, expected="رَبِّ", spoken="مالك",
            category="word_substitution", confidence=0.97, surah=1, ayah=2,
        )
        assert (message["surah"], message["ayah"], message["word_index"]) == (1, 2, 3)
        assert message["error_type"] == "word_substitution"


class TestUnimplemented:
    def test_phase_7_events_are_declared_but_not_built(self):
        """Declared so the protocol can be documented whole; a client must not
        have to guess whether silence means unsupported or nothing-happened."""
        assert NOT_YET_EMITTED
        for kind in NOT_YET_EMITTED:
            assert not hasattr(events, kind.value), (
                f"{kind.value} has a builder - move it out of NOT_YET_EMITTED"
            )
