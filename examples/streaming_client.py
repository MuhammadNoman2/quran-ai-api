#!/usr/bin/env python3
"""Stream a recitation over WebSocket and print the events.

    python examples/streaming_client.py data/test_audio/correct/001_002_husary_1.mp3 1 2

Streams the file in 100 ms frames, the way a microphone would. Requires only
`websockets` and `httpx` - no ML dependency of any kind.
"""

from __future__ import annotations

import asyncio
import json
import sys

import os

import httpx
import websockets

# Override with QURAN_API_URL / QURAN_API_KEY.
BASE_HTTP = os.environ.get("QURAN_API_URL", "http://127.0.0.1:8000/api/v1")
BASE_WS = BASE_HTTP.replace("http://", "ws://").replace("https://", "wss://")
API_KEY: str | None = os.environ.get("QURAN_API_KEY") or None
FRAME_MS = 100
SAMPLE_RATE = 16_000


def to_pcm(path: str) -> bytes:
    """Decode any audio file to pcm_s16le, mono, 16 kHz.

    Deliberately self-contained. A real client captures this straight from the
    microphone and needs none of this; the decode here exists only because we are
    replaying a file. Note what is *not* imported: nothing from the server.
    """
    import av

    with av.open(path) as container:
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="s16", layout="mono", rate=SAMPLE_RATE
        )
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().tobytes())
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray().tobytes())
    return b"".join(chunks)


def describe(message: dict) -> str:
    kind = message["event"]
    if kind == "ready":
        return f"ready - {message['expected_words']} expected words, engine {message['engine']['engine']}"
    if kind == "partial_transcript":
        return f"partial [{message['audio_seconds']:>5.2f}s]  {message['text']}"
    if kind == "word_confirmed":
        return f"  OK  word {message['word_index']}: {message['expected']}"
    if kind == "mistake_detected":
        return (
            f"  !!  word {message['word_index']}: expected {message['expected']!r}, "
            f"heard {message['spoken']!r} ({message['error_type']}, "
            f"confidence {message['confidence']})"
        )
    if kind == "ayah_progress":
        return f"progress {message['words_confirmed']}/{message['words_total']}"
    if kind == "ayah_completed":
        return f"ayah complete - score {message['word_accuracy_score']}, {message['errors']} mistake(s)"
    if kind == "session_completed":
        return f"session done - {message['audio_seconds']}s, {message['segments_analysed']} pass(es)"
    if kind in ("warning", "error"):
        return f"{kind.upper()}: {message.get('message')}"
    return json.dumps(message, ensure_ascii=False)


async def run(path: str, surah: int | None, ayah: int | None) -> None:
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    body = {"surah": surah, "ayah": ayah} if surah else {}
    session = httpx.post(f"{BASE_HTTP}/sessions", json=body, headers=headers, timeout=30).json()
    if "session_id" not in session:
        raise SystemExit(f"could not open a session: {session}")

    url = f"{BASE_WS}/sessions/{session['session_id']}/stream"
    if API_KEY:
        url += f"?api_key={API_KEY}"

    pcm = to_pcm(path)
    frame = SAMPLE_RATE * 2 * FRAME_MS // 1000

    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "start", "surah": surah, "ayah": ayah}))

        async def receive() -> None:
            async for raw in ws:
                message = json.loads(raw)
                print(describe(message))
                if message["event"] == "session_completed":
                    return

        reader = asyncio.create_task(receive())
        for offset in range(0, len(pcm), frame):
            await ws.send(pcm[offset : offset + frame])
            await asyncio.sleep(FRAME_MS / 1000)   # pace it like real capture
        await ws.send(json.dumps({"type": "stop"}))
        await reader


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    surah = int(sys.argv[2]) if len(sys.argv) > 2 else None
    ayah = int(sys.argv[3]) if len(sys.argv) > 3 else None
    asyncio.run(run(sys.argv[1], surah, ayah))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
