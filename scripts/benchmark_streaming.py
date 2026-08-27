#!/usr/bin/env python3
"""Measure end-to-end streaming latency against a running server.

    uvicorn app.main:app --port 8000
    python scripts/benchmark_streaming.py data/test_audio/correct/001_002_husary_1.mp3 1 2

Reports, per event, the wall time from the start of the stream, and the two
numbers that decide whether streaming is viable on a given machine:

    real-time factor  - total wall time / audio duration. Above 1.0 the server
                        falls behind a continuously reciting user.
    correction lag    - seconds between the reciter finishing a phrase and the
                        confirmed feedback arriving.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
import websockets  # noqa: E402

BASE_HTTP = os.environ.get("QURAN_API_URL", "http://127.0.0.1:8000/api/v1")
BASE_WS = BASE_HTTP.replace("http://", "ws://").replace("https://", "wss://")
FRAME_MS = 100
SAMPLE_RATE = 16_000


def to_pcm(path: str) -> bytes:
    import av

    with av.open(path) as container:
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=SAMPLE_RATE
        )
        out = []
        for frame in container.decode(container.streams.audio[0]):
            for r in resampler.resample(frame):
                out.append(r.to_ndarray().tobytes())
        for r in resampler.resample(None):
            out.append(r.to_ndarray().tobytes())
    return b"".join(out)


async def run(path: str, surah: int | None, ayah: int | None) -> int:
    body = {"surah": surah, "ayah": ayah} if surah else {}
    session = httpx.post(f"{BASE_HTTP}/sessions", json=body, timeout=30).json()
    if "session_id" not in session:
        print(f"could not open a session: {session}", file=sys.stderr)
        return 1

    pcm = to_pcm(path)
    audio_seconds = len(pcm) / (SAMPLE_RATE * 2)
    frame = SAMPLE_RATE * 2 * FRAME_MS // 1000
    timeline: list[tuple[float, str]] = []
    speech_stopped_at: float | None = None
    confirm_lags: list[float] = []
    started = time.perf_counter()

    async with websockets.connect(
        f"{BASE_WS}/sessions/{session['session_id']}/stream"
    ) as ws:
        await ws.send(json.dumps({"type": "start", "surah": surah, "ayah": ayah}))

        async def receive() -> None:
            nonlocal speech_stopped_at
            async for raw in ws:
                message = json.loads(raw)
                elapsed = time.perf_counter() - started
                timeline.append((elapsed, message["event"]))
                if message["event"] == "speech_stopped":
                    speech_stopped_at = elapsed
                if message["event"] in ("word_confirmed", "mistake_detected") and speech_stopped_at:
                    confirm_lags.append(elapsed - speech_stopped_at)
                if message["event"] == "session_completed":
                    return

        reader = asyncio.create_task(receive())
        for offset in range(0, len(pcm), frame):
            await ws.send(pcm[offset : offset + frame])
            await asyncio.sleep(FRAME_MS / 1000)
        await ws.send(json.dumps({"type": "stop"}))
        await reader

    total = time.perf_counter() - started
    print(f"\n{'wall':>8}  event")
    for elapsed, name in timeline:
        print(f"{elapsed:8.2f}  {name}")

    firsts: dict[str, float] = {}
    for elapsed, name in timeline:
        firsts.setdefault(name, elapsed)

    print(f"\naudio duration        {audio_seconds:6.2f}s")
    print(f"total wall time       {total:6.2f}s")
    print(f"real-time factor      {total / audio_seconds:6.2f}   "
          f"{'OK - keeps up' if total / audio_seconds < 1 else 'FALLS BEHIND a continuous reciter'}")
    if "word_detected" in firsts:
        print(f"first live word at    {firsts['word_detected']:6.2f}s  (provisional, display only)")
    if "word_confirmed" in firsts:
        print(f"first confirmed at    {firsts['word_confirmed']:6.2f}s")
    if confirm_lags:
        print(f"correction lag        {statistics.median(confirm_lags):6.2f}s  "
              "(pause -> confirmed feedback)")
    print("\nThese numbers are for THIS machine. Re-run on the target host before "
          "quoting them to anyone.")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    surah = int(sys.argv[2]) if len(sys.argv) > 2 else None
    ayah = int(sys.argv[3]) if len(sys.argv) > 3 else None
    return asyncio.run(run(sys.argv[1], surah, ayah))


if __name__ == "__main__":
    raise SystemExit(main())
