"""The two rules that decide when a streaming finding may be reported.

Both exist because of bugs found by running real audio through the socket, not
by reasoning about it, so both have regression tests naming what went wrong.
"""

import pytest

from app.alignment.base import AlignmentResult, ExpectedWord, SpokenWord
from app.models.enums import WordStatus
from app.streaming.manager import StreamProcessor


def result(index: int, status: WordStatus) -> AlignmentResult:
    word = ExpectedWord(f"w{index}", f"w{index}", index, 2, 255)
    spoken = (
        SpokenWord(f"w{index}", f"w{index}", index * 0.5, index * 0.5 + 0.4, 1.0)
        if status in (WordStatus.CORRECT, WordStatus.SUBSTITUTED)
        else None
    )
    return AlignmentResult(status=status, expected_word=word, spoken_word=spoken)


def sequence(*statuses: WordStatus) -> list[AlignmentResult]:
    return [result(i + 1, s) for i, s in enumerate(statuses)]


C, M = WordStatus.CORRECT, WordStatus.MISSING


class TestSettledIndex:
    def test_nothing_matched_settles_nothing(self):
        assert StreamProcessor._settled_index(sequence(M, M, M)) == 0

    def test_the_word_being_spoken_is_never_judged(self):
        """Settled is one behind the last match, so the current word is safe."""
        assert StreamProcessor._settled_index(sequence(C, C, C)) == 2

    def test_stops_at_a_gap(self):
        """A single unmatched word mid-stream is pending, not omitted - the
        reciter may simply not have reached it yet."""
        assert StreamProcessor._settled_index(sequence(C, C, M, M, M)) == 1

    def test_a_gap_followed_by_enough_matches_is_a_real_skip(self):
        """Three consecutive matches after a gap is evidence the reciter moved
        on, so the skipped word becomes reportable."""
        settled = StreamProcessor._settled_index(sequence(C, C, M, C, C, C))
        assert settled >= 3

    def test_scattered_far_ahead_match_does_not_settle_everything(self):
        """The regression that mattered.

        Quran text repeats short words, so aligning a partial recitation against
        a whole verse produces accidental matches far ahead of the reciter. Using
        max(matched) treated everything before such a match as settled and
        reported 46 words of a correct Ayat al-Kursi as omitted.
        """
        results = sequence(C, C, M, M, M, M, M, M, M, M, M, M, C)
        settled = StreamProcessor._settled_index(results)
        assert settled == 1, "an isolated far-ahead match must not settle the gap"

    def test_a_long_correct_run_settles_progressively(self):
        assert StreamProcessor._settled_index(sequence(*[C] * 20)) == 19


class TestBufferWindow:
    def test_drop_before_advances_the_window(self):
        from app.audio.buffer import RollingBuffer
        import numpy as np

        buffer = RollingBuffer(16_000, max_seconds=60)
        buffer.append_samples(np.zeros(16_000 * 10, dtype=np.float32))
        dropped = buffer.drop_before(4.0)
        assert dropped == pytest.approx(4.0, abs=0.01)
        assert buffer.seconds == pytest.approx(6.0, abs=0.01)

    def test_dropping_more_than_held_keeps_nothing(self):
        from app.audio.buffer import RollingBuffer
        import numpy as np

        buffer = RollingBuffer(16_000, max_seconds=60)
        buffer.append_samples(np.zeros(16_000 * 2, dtype=np.float32))
        buffer.drop_before(10.0)
        assert buffer.is_empty

    def test_dropping_zero_is_a_no_op(self):
        from app.audio.buffer import RollingBuffer
        import numpy as np

        buffer = RollingBuffer(16_000, max_seconds=60)
        buffer.append_samples(np.zeros(16_000, dtype=np.float32))
        assert buffer.drop_before(0) == 0.0
        assert buffer.seconds == pytest.approx(1.0, abs=0.01)


class TestAnalyzerWindowOffset:
    def test_from_word_skips_already_committed_words(self):
        """Without this, advancing the window would make every earlier word look
        omitted."""
        from app.asr.base import FakeASREngine
        from app.recitation.analyzer import RecitationAnalyzer
        import numpy as np

        engine = FakeASREngine(text="رب العالمين")
        engine.load()
        analyzer = RecitationAnalyzer(engine)
        asr = engine.transcribe(np.zeros(16_000, dtype=np.float32))

        full = analyzer.compare(asr, 1, 2)
        windowed = analyzer.compare(asr, 1, 2, from_word=2)

        assert len(full) > len(windowed)
        assert all(
            r.expected_word.index > 2 for r in windowed if r.expected_word is not None
        )
