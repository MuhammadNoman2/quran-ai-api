"""Per-session streaming state, VAD-driven scheduling, two-tier recognition.

Three ideas, each forced by measurement rather than taste.

**Recognition is scheduled by speech, not by a timer.** Silero VAD costs 11-25 ms
against ~2500 ms for a recognition pass, so it is effectively free to know
whether the reciter is still speaking. Segments therefore end at natural pauses
(waqf) instead of mid-word.

**Two tiers, with different authority.** The fast tier (~1.7 s) drives the live
display and may confirm that a word was *recognized*; it may never assert a
mistake, because it was measured emitting a wrong word at 0.90 confidence. The
slower tier (~5 s) runs at each pause and is the only thing allowed to say the
reciter got something wrong. When it revises a provisional judgement, a
`correction` event says so.

**Analysis is cumulative.** Whisper's ~2.5 s fixed overhead per call makes a
longer clip nearly free, while stitching independent slices cuts words at
boundaries and mishears them - which would reach the reciter as a mistake they
did not make.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace

import numpy as np

from app.alignment.base import AlignmentResult
from app.audio.buffer import AudioBufferError, RollingBuffer
from app.audio.vad import SpeechDetector, SpeechState
from app.core.config import Settings
from app.models.enums import ConfidenceBand, ErrorCategory, WordStatus
from app.recitation.analyzer import RecitationAnalyzer, VerseNotIdentified
from app.recitation.scoring import score_reported
from app.streaming import events
from app.streaming.events import ErrorCode
from app.streaming.session import Session

logger = logging.getLogger(__name__)

Send = Callable[[dict], Awaitable[None]]

#: VAD runs on a trailing window, not the whole buffer, so its cost stays flat
#: as a recitation gets longer.
VAD_WINDOW_SECONDS = 3.0

#: And no more often than this much new audio, so a 100 ms frame rate does not
#: turn into ten VAD calls a second per session.
VAD_INTERVAL_SECONDS = 0.2


@dataclass
class StreamState:
    surah: int | None = None
    ayah: int | None = None
    started: bool = False
    finished: bool = False
    speaking: bool = False
    segments_analysed: int = 0
    provisional_passes: int = 0
    confirmed_words: set[int] = field(default_factory=set)
    detected_words: dict[int, str] = field(default_factory=dict)
    reported_mistakes: set[tuple[int | None, str]] = field(default_factory=set)
    last_results: list[AlignmentResult] = field(default_factory=list)
    expected_word_count: int = 0
    last_progress_reported: int = -1
    mistake_evidence: dict[tuple[int | None, str], int] = field(default_factory=dict)
    """How many independent committed passes have seen each candidate mistake."""

    committed_prefix: int = 0
    """Expected words already confirmed and rolled out of the analysis window."""

    audio_dropped: bool = False


class StreamProcessor:
    def __init__(
        self,
        session: Session,
        committed: RecitationAnalyzer,
        provisional: RecitationAnalyzer | None,
        config: Settings,
        send: Send,
    ) -> None:
        self._session = session
        self._committed = committed
        self._provisional = provisional
        self._config = config
        self._send = send
        self._buffer = RollingBuffer(
            session.sample_rate, max_seconds=config.stream_max_buffer_seconds
        )
        self._detector = SpeechDetector(config.vad_threshold)
        self.state = StreamState(surah=session.surah, ayah=session.ayah)
        self._busy = False
        self._worker: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._want_committed = False
        self._want_provisional = False
        self._committed_samples = 0
        self._provisional_samples = 0
        self._vad_samples = 0
        self._committed_speech_end: float | None = None
        self._speech: SpeechState | None = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self, message: dict) -> None:
        if self.state.started:
            await self._send(events.error(ErrorCode.ALREADY_STARTED, "session already started"))
            return

        surah = message.get("surah", self.state.surah)
        ayah = message.get("ayah", self.state.ayah)
        if (surah is None) != (ayah is None):
            await self._send(
                events.error(ErrorCode.INVALID_MESSAGE, "supply both surah and ayah, or neither")
            )
            return

        if surah is not None:
            try:
                self.state.expected_word_count = len(
                    self._committed.expected_words(int(surah), int(ayah))
                )
            except KeyError:
                await self._send(
                    events.error(ErrorCode.VERSE_NOT_FOUND, f"ayah {surah}:{ayah} does not exist")
                )
                return

        self.state.surah, self.state.ayah = surah, ayah
        self.state.started = True
        await self._send(events.session_started(self._session.id, surah, ayah))
        await self._send(
            events.ready(
                sample_rate=self._session.sample_rate,
                audio_format=self._session.audio_format,
                expected_words=self.state.expected_word_count or None,
                engine=self._committed._engine.describe().public(),
            )
        )

    async def on_audio(self, data: bytes) -> None:
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
            # Losing audio invalidates conclusions about anything before what
            # remains, so record it - the reporting path must not call a word
            # omitted when we simply threw its audio away.
            self.state.audio_dropped = True
            await self._send(
                events.warning(
                    "audio dropped: the server is behind and the buffer is full",
                    **self._buffer.stats().__dict__,
                )
            )
        await self._on_new_audio()

    # ── scheduling ───────────────────────────────────────────────────────────

    def _ensure_worker(self) -> None:
        """One worker per session, started lazily.

        Recognition must not be awaited inside the WebSocket receive loop: a pass
        takes seconds, during which the server would stop reading the socket and
        stop answering keepalive pings. Measured on a 60s verse, that killed the
        connection outright.
        """
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._worker_loop())

    def _request(self, *, committed: bool) -> None:
        """Ask for a pass. Committed requests always win.

        An earlier version simply dropped a request while another pass was
        running, which meant a pause-triggered committed pass could be discarded
        because a provisional pass happened to be in flight. On a long verse that
        left every word unconfirmed until the very end, and the analysis window
        never advanced.
        """
        if committed:
            self._want_committed = True
            self._want_provisional = False
        elif not self._want_committed:
            self._want_provisional = True
        self._ensure_worker()
        self._wake.set()

    async def _worker_loop(self) -> None:
        while not self.state.finished:
            await self._wake.wait()
            self._wake.clear()
            try:
                if self._want_committed:
                    self._want_committed = False
                    self._want_provisional = False
                    await self._run_committed()
                elif self._want_provisional:
                    self._want_provisional = False
                    await self._run_provisional()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "recognition pass failed", extra={"session_id": self._session.id}
                )

    async def _drain(self) -> None:
        """Finish any in-flight pass and stop the worker."""
        self._want_committed = False
        self._want_provisional = False
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
        self._worker = None

    async def flush(self) -> None:
        await self._drain()
        await self._run_committed()

    async def stop(self) -> None:
        if self.state.finished:
            return
        await self._drain()
        await self._run_committed(final=True)
        self.state.finished = True
        await self._send(
            events.session_completed(
                session_id=self._session.id,
                seconds=self._buffer.stats().total_seconds_received,
                segments=self.state.segments_analysed,
            )
        )

    # ── speech tracking ──────────────────────────────────────────────────────

    def _update_speech(self) -> SpeechState | None:
        """Run VAD on a trailing window, throttled. Returns None when skipped."""
        audio = self._buffer.snapshot()
        rate = self._buffer.sample_rate
        if audio.size - self._vad_samples < VAD_INTERVAL_SECONDS * rate:
            return None
        self._vad_samples = audio.size

        window_samples = int(VAD_WINDOW_SECONDS * rate)
        offset_samples = max(0, audio.size - window_samples)
        local = self._detector.analyse(audio[offset_samples:], rate)

        offset = offset_samples / rate
        self._speech = SpeechState(
            is_speaking=local.is_speaking,
            speech_started_at=(local.speech_started_at + offset)
            if local.speech_started_at is not None
            else None,
            speech_ended_at=(local.speech_ended_at + offset)
            if local.speech_ended_at is not None
            else None,
            silence_seconds=local.silence_seconds,
            total_speech_seconds=local.total_speech_seconds,
            segments=[(a + offset, b + offset) for a, b in local.segments],
        )
        return self._speech

    async def _on_new_audio(self) -> None:
        speech = self._update_speech()
        if speech is None:
            return

        rate = self._buffer.sample_rate

        if speech.is_speaking and not self.state.speaking:
            self.state.speaking = True
            await self._send(events.speech_started(at=speech.speech_started_at or 0.0))

        pause_reached = (
            not speech.is_speaking
            and speech.has_speech
            and speech.silence_seconds >= self._config.stream_pause_seconds
        )

        if pause_reached and self.state.speaking:
            self.state.speaking = False
            await self._send(
                events.speech_stopped(
                    at=speech.speech_ended_at or 0.0, silence_seconds=speech.silence_seconds
                )
            )
            self._request(committed=True)
            return

        unheard = (self._buffer.snapshot().size - self._provisional_samples) / rate
        if (
            self.state.speaking
            and self._provisional is not None
            and unheard >= self._config.stream_provisional_seconds
        ):
            self._request(committed=False)

    # ── recognition passes ───────────────────────────────────────────────────

    async def _transcribe(self, analyzer: RecitationAnalyzer, audio: np.ndarray):
        # Off the event loop: a pass takes seconds and the socket must keep
        # accepting audio throughout. If it does not, the connection dies on a
        # keepalive ping timeout.
        return await asyncio.to_thread(
            analyzer._engine.transcribe, audio, self._buffer.sample_rate
        )

    async def _compare(self, analyzer: RecitationAnalyzer, asr):
        """Alignment also runs off the loop - measured at 780 ms on a 50-word
        ayah, which is far too long to block the socket."""
        return await asyncio.to_thread(
            analyzer.compare,
            asr,
            self.state.surah,
            self.state.ayah,
            speech_end=self._speech.speech_ended_at if self._speech else None,
            from_word=self.state.committed_prefix,
        )

    async def _run_provisional(self) -> None:
        """Fast, display-only pass. Never reports a mistake."""
        if self._busy or self._provisional is None:
            return
        audio = self._buffer.snapshot()
        if audio.size == 0:
            return
        self._busy = True
        try:
            self._provisional_samples = audio.size
            asr = await self._transcribe(self._provisional, audio)
            self.state.provisional_passes += 1
            await self._send(
                events.partial_transcript(
                    asr.text, seconds=audio.size / self._buffer.sample_rate
                )
            )
            if self.state.surah is None:
                return
            results = await self._compare(self._provisional, asr)
            await self._emit_provisional(results)
        except Exception:
            logger.exception("provisional pass failed", extra={"session_id": self._session.id})
        finally:
            self._busy = False

    async def _run_committed(self, *, final: bool = False) -> None:
        if self._busy:
            return
        audio = self._buffer.snapshot()
        if audio.size == 0 or not self.state.started:
            return

        self._busy = True
        try:
            # Re-run only when there is new *speech*, not merely new samples.
            # `stop` typically arrives a second or two after the pause that
            # already triggered a pass, and those extra seconds are silence.
            # Comparing raw sample counts made that look like new material and
            # cost a full ~5s inference to reproduce the same alignment.
            speech_end = self._speech.speech_ended_at if self._speech else None
            no_new_speech = (
                speech_end is not None
                and self._committed_speech_end is not None
                and speech_end <= self._committed_speech_end + 0.05
            )
            if self.state.last_results and (
                no_new_speech or audio.size == self._committed_samples
            ):
                await self._emit_committed(self.state.last_results, final=final)
                return
            self._committed_samples = audio.size
            self._committed_speech_end = speech_end
            self._provisional_samples = audio.size

            try:
                asr = await self._transcribe(self._committed, audio)
            except Exception:
                logger.exception("transcription failed", extra={"session_id": self._session.id})
                await self._send(
                    events.error(ErrorCode.INTERNAL_ERROR, "recognition failed for this segment")
                )
                return

            self.state.segments_analysed += 1
            await self._send(
                events.partial_transcript(
                    asr.text, seconds=audio.size / self._buffer.sample_rate
                )
            )

            if self.state.surah is None and not await self._resolve_verse(asr.text):
                return

            results = await self._compare(self._committed, asr)
            self.state.last_results = results
            await self._emit_committed(results, final=final)
            if not final:
                await self._advance_window(results)
        finally:
            self._busy = False

    async def _corroborate(
        self, result: AlignmentResult, primary: list[AlignmentResult]
    ) -> bool:
        """Ask the other engine whether it agrees, before reporting a mistake.

        Only reached for a mistake seen in a single pass at the end of a session,
        where there will be no second pass to agree with it. Two different models
        agreeing is stronger evidence than one model twice, and the cost is one
        cheap pass over the current window - paid only when a mistake is actually
        suspected.

        Disagreement suppresses the report. That is the deliberate direction: a
        missed mistake is a lesser harm than telling someone their correct
        recitation was wrong.
        """
        if self._provisional is None or result.expected_word is None:
            return True  # nothing to ask; fall back to the single pass
        audio = self._buffer.snapshot()
        if audio.size == 0:
            return True
        try:
            asr = await self._transcribe(self._provisional, audio)
            second = await self._compare(self._provisional, asr)
        except Exception:
            logger.exception("corroboration failed", extra={"session_id": self._session.id})
            return True

        index = result.expected_word.index
        for other in second:
            if other.expected_word is None or other.expected_word.index != index:
                continue
            if other.status is WordStatus.CORRECT:
                return False  # the second engine heard it fine - we are not sure
            if other.status is WordStatus.SUBSTITUTED:
                return True  # both heard something, both disagree with the text
            # Both engines say MISSING. That is agreement only if they also
            # agree about how far the recitation got.
            #
            # If the primary engine matched words *after* this one but the second
            # did not, the second simply stopped transcribing - the fast model
            # truncates long audio - so it has no opinion here, and treating its
            # silence as agreement reported a correct reciter as wrong.
            #
            # If neither matched anything later, they agree the recitation ended
            # before this word, which is exactly what a genuine omission at the
            # end of a verse looks like.
            def reached_past(results: list[AlignmentResult]) -> bool:
                return any(
                    o.expected_word is not None
                    and o.expected_word.index > index
                    and o.status in (WordStatus.CORRECT, WordStatus.SUBSTITUTED)
                    for o in results
                )

            return reached_past(second) or not reached_past(primary)
        return True

    async def _advance_window(self, results: list[AlignmentResult]) -> None:
        """Commit confirmed words and drop their audio.

        Cumulative analysis is accurate but its cost grows with duration, and
        Whisper attends only 30s regardless. On a long verse the window must
        advance or every pass gets slower than the last - measured at 21s for
        one pass over 60s of audio, which is slower than real time and killed
        the connection on a keepalive timeout.

        Only audio up to the last *confirmed* word is dropped, so nothing still
        under judgement is discarded.
        """
        if self._buffer.seconds < self._config.stream_window_seconds:
            return
        settled = self._settled_index(results)
        cut: float | None = None
        prefix = self.state.committed_prefix
        for result in results:
            if (
                result.expected_word is not None
                and result.spoken_word is not None
                and result.status is WordStatus.CORRECT
                and result.expected_word.index <= settled
                and result.expected_word.index in self.state.confirmed_words
            ):
                cut = result.spoken_word.end_time
                prefix = result.expected_word.index
        if cut is None or prefix <= self.state.committed_prefix:
            return

        dropped = self._buffer.drop_before(cut)
        if dropped <= 0:
            return
        self.state.committed_prefix = prefix
        # Every offset was relative to the old window start.
        self._committed_samples = 0
        self._provisional_samples = 0
        self._vad_samples = 0
        self._committed_speech_end = None
        self._speech = None
        logger.info(
            "window advanced",
            extra={
                "session_id": self._session.id,
                "committed_prefix": prefix,
                "dropped_seconds": round(dropped, 2),
                "buffered_seconds": round(self._buffer.seconds, 2),
            },
        )

    async def _resolve_verse(self, text: str) -> bool:
        try:
            surah, ayah, _ = self._committed._detect_verse(text)
        except VerseNotIdentified:
            await self._send(events.warning("still listening - no verse identified yet"))
            return False
        self.state.surah, self.state.ayah = surah, ayah
        self.state.expected_word_count = len(self._committed.expected_words(surah, ayah))
        await self._send(events.session_started(self._session.id, surah, ayah))
        return True

    # ── reporting ────────────────────────────────────────────────────────────

    async def _emit_provisional(self, results: list[AlignmentResult]) -> None:
        for result in results:
            if result.expected_word is None or result.status is not WordStatus.CORRECT:
                continue
            index = result.expected_word.index
            if index in self.state.confirmed_words or index in self.state.detected_words:
                continue
            self.state.detected_words[index] = result.status.value
            await self._send(
                events.word_detected(
                    index=index,
                    expected=result.expected_word.text,
                    spoken=result.spoken_word.text if result.spoken_word else None,
                    confidence=result.confidence,
                )
            )

    @staticmethod
    def _settled_index(results: list[AlignmentResult], lookahead: int = 3) -> int:
        """Highest expected-word index the reciter has demonstrably passed.

        This must be a *contiguous* notion, not `max(matched)`. Quran text
        repeats short words heavily - Ayat al-Kursi alone contains ما, من, ولا
        and إلا several times each - so aligning a partial recitation against a
        whole verse produces scattered accidental matches far ahead of where the
        reciter actually is. Taking the maximum treated everything before such a
        match as settled and reported 46 words of a correct recitation as
        omitted.

        So: walk forward from the start of the window and stop at the first gap,
        unless the gap is followed by `lookahead` consecutive matches - which is
        real evidence the reciter moved on rather than an alignment artefact.
        """
        matched = [
            (r.expected_word.index, r.status in (WordStatus.CORRECT, WordStatus.SUBSTITUTED))
            for r in results
            if r.expected_word is not None
        ]
        if not matched:
            return 0

        settled = 0
        for position, (index, is_match) in enumerate(matched):
            if is_match:
                settled = index
                continue
            following = [ok for _, ok in matched[position + 1 : position + 1 + lookahead]]
            if len(following) == lookahead and all(following):
                continue  # genuinely skipped; the reciter has moved past it
            break
        return max(0, settled - 1)

    async def _emit_committed(self, results: list[AlignmentResult], *, final: bool) -> None:
        settled = self.state.expected_word_count if final else self._settled_index(results)

        # If audio was thrown away, words before the first match may simply have
        # had their audio discarded. Calling those omitted would be a false
        # correction caused by our own back-pressure, not by the reciter.
        floor = 0
        if self.state.audio_dropped:
            matched = [
                r.expected_word.index
                for r in results
                if r.expected_word is not None
                and r.status in (WordStatus.CORRECT, WordStatus.SUBSTITUTED)
            ]
            floor = min(matched) if matched else 0

        for result in results:
            index = result.expected_word.index if result.expected_word else None

            if result.status is WordStatus.CORRECT and result.band.value.endswith("correct"):
                if index is not None and index not in self.state.confirmed_words:
                    self.state.confirmed_words.add(index)
                    self.state.detected_words.pop(index, None)
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
            if index is not None and (index > settled or index < floor):
                continue
            key = (index, result.category.value)
            if key in self.state.reported_mistakes:
                continue

            # Agreement before accusation. A single pass is not enough evidence:
            # the recognizer makes its own substitution errors on professional
            # recitation (measured - it heard حِفْرُهُمَا for حِفْظُهُمَا), and
            # reporting those tells a correct reciter they were wrong.
            self.state.mistake_evidence[key] = self.state.mistake_evidence.get(key, 0) + 1
            if self.state.mistake_evidence[key] < 2:
                if not final:
                    continue
                if not await self._corroborate(result, results):
                    logger.info(
                        "mistake withheld - not corroborated",
                        extra={
                            "session_id": self._session.id,
                            "word_index": index,
                            "category": result.category.value,
                        },
                    )
                    continue

            self.state.reported_mistakes.add(key)

            # The fast tier had lit this word up as recognized. Say plainly that
            # the judgement changed rather than letting the UI hold both.
            if index is not None and index in self.state.detected_words:
                await self._send(
                    events.correction(
                        index=index,
                        previous=self.state.detected_words.pop(index),
                        current=result.status.value,
                        reason="revised by the committed tier",
                    )
                )

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
            # Score from the session's own record of what it reported, not from
            # the last pass. Window advancement means the final results cover
            # only the tail of a long verse, so scoring them forgave every
            # mistake reported in an earlier window.
            breakdown = score_reported(
                [ErrorCategory(category) for _, category in self.state.reported_mistakes],
                self.state.expected_word_count,
                correct=len(self.state.confirmed_words),
            )
            await self._send(
                events.ayah_completed(
                    surah=self.state.surah,
                    ayah=self.state.ayah,
                    score=breakdown.score,
                    errors=len(self.state.reported_mistakes),
                    observations=sum(
                        1 for r in results if not r.is_error and r.category is not None
                    ),
                )
            )
