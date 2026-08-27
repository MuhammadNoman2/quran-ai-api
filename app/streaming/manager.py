"""Per-session streaming state and inference scheduling.

Two decisions shape this file, both forced by measurement.

**Cumulative analysis, not stitched segments.** Each recognition pass runs over
all the audio buffered so far for the current ayah, not just the newest slice.
Whisper costs ~2.5s of fixed overhead per call whatever the clip length, so a
slightly longer clip is nearly free while stitching independent slices is not:
words straddling a boundary get cut in half and misheard, which surfaces to the
reciter as a mistake they did not make. Re-aligning the whole ayah every pass
also means a word can be *revised* as more audio arrives rather than being
locked in early.

**Nothing is reported until the reciter has moved past it.** A partial
recitation aligned against a full ayah marks every not-yet-spoken word as
MISSING. Reporting those would accuse a student of skipping words they are still
about to say. So findings are only emitted for positions the reciter has
demonstrably passed - see `_settled_index`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from app.alignment.base import AlignmentResult
from app.audio.buffer import AudioBufferError, RollingBuffer
from app.core.config import Settings
from app.models.enums import WordStatus
from app.recitation.analyzer import RecitationAnalyzer, VerseNotIdentified
from app.recitation.scoring import score
from app.streaming import events
from app.streaming.events import ErrorCode, EventType
from app.streaming.session import Session

logger = logging.getLogger(__name__)

Send = Callable[[dict], Awaitable[None]]


@dataclass
class StreamState:
    surah: int | None = None
    ayah: int | None = None
    started: bool = False
    finished: bool = False
    segments_analysed: int = 0
    confirmed_words: set[int] = field(default_factory=set)
    reported_mistakes: set[tuple[int | None, str]] = field(default_factory=set)
    last_results: list[AlignmentResult] = field(default_factory=list)
    expected_word_count: int = 0
    last_progress_reported: int = -1


class StreamProcessor:
    """Drives one WebSocket session."""

    def __init__(
        self,
        session: Session,
        analyzer: RecitationAnalyzer,
        config: Settings,
        send: Send,
    ) -> None:
        self._session = session
        self._analyzer = analyzer
        self._config = config
        self._send = send
        self._buffer = RollingBuffer(
            session.sample_rate, max_seconds=config.stream_max_buffer_seconds
        )
        self.state = StreamState(surah=session.surah, ayah=session.ayah)
        self._busy = False
        self._pending = False
        self._analysed_samples = 0

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self, message: dict) -> None:
        if self.state.started:
            await self._send(
                events.error(ErrorCode.ALREADY_STARTED, "session already started")
            )
            return

        surah = message.get("surah", self.state.surah)
        ayah = message.get("ayah", self.state.ayah)
        if (surah is None) != (ayah is None):
            await self._send(
                events.error(
                    ErrorCode.INVALID_MESSAGE, "supply both surah and ayah, or neither"
                )
            )
            return

        if surah is not None:
            try:
                expected = self._analyzer.expected_words(int(surah), int(ayah))
            except KeyError:
                await self._send(
                    events.error(ErrorCode.VERSE_NOT_FOUND, f"ayah {surah}:{ayah} does not exist")
                )
                return
            self.state.expected_word_count = len(expected)

        self.state.surah, self.state.ayah = surah, ayah
        self.state.started = True

        await self._send(events.session_started(self._session.id, surah, ayah))
        await self._send(
            events.ready(
                sample_rate=self._session.sample_rate,
                audio_format=self._session.audio_format,
                expected_words=self.state.expected_word_count or None,
                engine=self._analyzer._engine.describe().public(),
            )
        )

    async def on_audio(self, data: bytes) -> None:
        """Buffer a binary frame. A malformed frame never ends the session."""
        if not self.state.started:
            await self._send(
                events.error(ErrorCode.NOT_STARTED, 'send {"type":"start"} before audio')
            )
            return
        try:
            kept = self._buffer.append_pcm(data)
        except AudioBufferError as exc:
            await self._send(events.error(ErrorCode.INVALID_AUDIO, str(exc)))
            return
        if not kept:
            await self._send(
                events.warning(
                    "audio dropped: the server is behind and the buffer is full",
                    **self._buffer.stats().__dict__,
                )
            )
        if self._unanalysed_seconds() >= self._config.stream_segment_seconds:
            await self.maybe_analyse()

    def _unanalysed_seconds(self) -> float:
        """Audio arrived since the last recognition pass.

        The trigger must be on *new* audio, not on total buffered length. Because
        analysis is cumulative the buffer never shrinks, so triggering on total
        length fires again on every single frame once the threshold is first
        crossed - measured at 24 inference passes for one 6-second clip, roughly
        40 seconds of CPU for 6 seconds of audio.
        """
        return (self._buffer.snapshot().size - self._analysed_samples) / self._buffer.sample_rate

    async def flush(self) -> None:
        await self.maybe_analyse(force=True)

    async def stop(self) -> None:
        if self.state.finished:
            return
        await self.maybe_analyse(force=True, final=True)
        self.state.finished = True
        await self._send(
            events.session_completed(
                session_id=self._session.id,
                seconds=self._buffer.stats().total_seconds_received,
                segments=self.state.segments_analysed,
            )
        )

    # ── inference ────────────────────────────────────────────────────────────

    async def maybe_analyse(self, *, force: bool = False, final: bool = False) -> None:
        """Run a pass unless one is already running.

        One inference at a time per session. Audio keeps buffering while a pass
        runs, so nothing is lost - it is simply analysed by the next pass, which
        is exactly what cumulative analysis makes safe.
        """
        if self._buffer.is_empty or not self.state.started:
            return
        if self._busy:
            self._pending = True
            return

        self._busy = True
        try:
            await self._analyse(final=final)
        finally:
            self._busy = False

        if self._pending and not final:
            self._pending = False
            await self.maybe_analyse(force=force)

    async def _analyse(self, *, final: bool) -> None:
        audio = self._buffer.snapshot()
        if audio.size == 0:
            return

        # Nothing new since the last pass - typically `stop` arriving right after
        # an analysis. Re-running would cost a full inference (~2.5s measured) to
        # produce identical output, so reuse the previous alignment.
        if audio.size == self._analysed_samples and self.state.last_results:
            await self._emit_findings(self.state.last_results, final=final)
            return
        self._analysed_samples = audio.size

        # Off the event loop: a pass takes seconds, and the socket must keep
        # accepting audio throughout.
        try:
            asr = await asyncio.to_thread(
                self._analyzer._engine.transcribe, audio, self._buffer.sample_rate
            )
        except Exception:
            logger.exception("transcription failed", extra={"session_id": self._session.id})
            await self._send(
                events.error(ErrorCode.INTERNAL_ERROR, "recognition failed for this segment")
            )
            return

        self.state.segments_analysed += 1
        await self._send(
            events.partial_transcript(asr.text, seconds=len(audio) / self._buffer.sample_rate)
        )

        if self.state.surah is None:
            if not await self._resolve_verse(asr.text):
                return

        assert self.state.surah is not None and self.state.ayah is not None
        results = self._analyzer.compare(asr, self.state.surah, self.state.ayah)
        self.state.last_results = results
        await self._emit_findings(results, final=final)

    async def _resolve_verse(self, text: str) -> bool:
        try:
            surah, ayah, _ = self._analyzer._detect_verse(text)
        except VerseNotIdentified:
            await self._send(
                events.warning("still listening - no verse identified yet")
            )
            return False
        self.state.surah, self.state.ayah = surah, ayah
        self.state.expected_word_count = len(self._analyzer.expected_words(surah, ayah))
        await self._send(events.session_started(self._session.id, surah, ayah))
        return True

    # ── reporting ────────────────────────────────────────────────────────────

    @staticmethod
    def _settled_index(results: list[AlignmentResult]) -> int:
        """Highest expected-word index the reciter has demonstrably passed.

        A word only counts as passed once a *later* word has been matched, so the
        word currently being spoken is never judged. Without this, a student
        pausing mid-verse would be told they omitted everything after the pause.
        """
        matched = [
            r.expected_word.index
            for r in results
            if r.expected_word is not None
            and r.status in (WordStatus.CORRECT, WordStatus.SUBSTITUTED)
        ]
        return max(matched) - 1 if matched else 0

    async def _emit_findings(self, results: list[AlignmentResult], *, final: bool) -> None:
        settled = len(results) if final else self._settled_index(results)

        for result in results:
            index = result.expected_word.index if result.expected_word else None

            if result.status is WordStatus.CORRECT and result.band.value.endswith("correct"):
                if index is not None and index not in self.state.confirmed_words:
                    self.state.confirmed_words.add(index)
                    await self._send(
                        events.word_confirmed(
                            index=index,
                            expected=result.expected_word.text,
                            spoken=result.spoken_word.text if result.spoken_word else None,
                            confidence=result.confidence,
                        )
                    )
                continue

            if not result.is_error:
                continue
            # Not yet reached: the reciter may still be about to say it.
            if index is not None and index > settled:
                continue
            key = (index, result.category.value)
            if key in self.state.reported_mistakes:
                continue
            self.state.reported_mistakes.add(key)
            await self._send(
                events.mistake_detected(
                    index=index,
                    expected=result.expected_word.text if result.expected_word else None,
                    spoken=result.spoken_word.text if result.spoken_word else None,
                    category=result.category.value,
                    confidence=result.confidence,
                    surah=self.state.surah,
                    ayah=self.state.ayah,
                )
            )

        confirmed = len(self.state.confirmed_words)
        if self.state.expected_word_count and confirmed != self.state.last_progress_reported:
            # Only when it actually moved - repeating identical progress is noise
            # for a UI that redraws on every event.
            self.state.last_progress_reported = confirmed
            await self._send(
                events.ayah_progress(
                    surah=self.state.surah,
                    ayah=self.state.ayah,
                    words_confirmed=confirmed,
                    words_total=self.state.expected_word_count,
                )
            )

        if final:
            breakdown = score(results, self.state.expected_word_count)
            await self._send(
                events.ayah_completed(
                    surah=self.state.surah,
                    ayah=self.state.ayah,
                    score=breakdown.score,
                    errors=sum(1 for r in results if r.is_error),
                    observations=sum(
                        1 for r in results if not r.is_error and r.category is not None
                    ),
                )
            )
