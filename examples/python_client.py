#!/usr/bin/env python3
"""Minimal Python client. The only dependency is an HTTP library.

    python examples/python_client.py data/test_audio/correct/001_002_husary_1.mp3 1 2

Note what is absent: no torch, no faster-whisper, no model download. A client
integrating this API needs the base URL, an optional key, and audio.
"""

from __future__ import annotations

import sys

import os

import httpx

# Override with QURAN_API_URL / QURAN_API_KEY.
BASE_URL = os.environ.get("QURAN_API_URL", "http://127.0.0.1:8000/api/v1")
API_KEY: str | None = os.environ.get("QURAN_API_KEY") or None


def headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def analyze(path: str, surah: int | None = None, ayah: int | None = None) -> dict:
    data: dict[str, str] = {}
    if surah and ayah:
        data = {"surah": str(surah), "ayah": str(ayah)}
    with open(path, "rb") as fh:
        response = httpx.post(
            f"{BASE_URL}/recitation/analyze",
            files={"audio": (path.rsplit("/", 1)[-1], fh, "application/octet-stream")},
            data=data,
            headers=headers(),
            timeout=120.0,
        )
    if response.status_code != 200:
        error = response.json().get("error", {})
        raise SystemExit(f"error {error.get('code')}: {error.get('message')}")
    return response.json()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    surah = int(sys.argv[2]) if len(sys.argv) > 2 else None
    ayah = int(sys.argv[3]) if len(sys.argv) > 3 else None

    result = analyze(path, surah, ayah)
    print(f"\n{result['surah']}:{result['ayah']}  score {result['word_accuracy_score']}")
    print(f"expected   : {result['expected_text']}")
    print(f"recognized : {result['recognized_text']}\n")

    for word in result["words"]:
        mark = {"correct": "OK ", "substituted": "SUB", "missing": "DEL",
                "extra": "INS", "repeated": "REP"}.get(word["status"], "?  ")
        print(f"  {mark} {str(word['index'] or '-'):>3}  {word['expected'] or '(not expected)'}")

    if result["errors"]:
        print("\nmistakes:")
        for e in result["errors"]:
            print(f"  word {e['index']}: expected {e['expected']}, heard {e['spoken']} ({e['category']})")
    else:
        print("\nno mistakes reported")

    if result["observations"]:
        print("\npossible issues (not confirmed - do not present these as mistakes):")
        for o in result["observations"]:
            print(f"  word {o['index']}: {o['category']} - {o['note'] or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
