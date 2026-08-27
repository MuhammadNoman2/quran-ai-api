#!/usr/bin/env python3
"""Exercise every endpoint against a running server.

    python scripts/smoke_test.py                      # localhost:8000
    python scripts/smoke_test.py http://127.0.0.1:8100

Exists because unit tests run inside a virtual environment that already has every
package installed, so they cannot catch a dependency missing from
requirements.txt. A container build can, and did: `quranic-phonemizer` was absent
and the Tajweed endpoint returned 500 in the image while every test passed
locally. Run this against any deployment before trusting it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

CLIP = Path("data/test_audio/correct/001_002_husary_1.mp3")


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
    api = f"{base}/api/v1"
    failures: list[str] = []
    passed = 0

    def check(label: str, fn) -> None:
        nonlocal passed
        try:
            detail = fn()
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
            print(f"  FAIL  {label:38s} {type(exc).__name__}: {exc}")
            return
        passed += 1
        print(f"  ok    {label:38s} {detail or ''}")

    def get(path: str) -> dict:
        response = httpx.get(f"{api}{path}", timeout=60)
        response.raise_for_status()
        return response.json()

    print(f"\nsmoke test against {base}\n")

    check("GET /health", lambda: f"device={get('/health')['device']}")
    check("GET /surahs", lambda: f"{len(get('/surahs'))} surahs")
    check("GET /surahs/1", lambda: f"{get('/surahs/1')['ayah_count']} ayat")
    check("GET ayah 1:2", lambda: f"{len(get('/surahs/1/ayahs/2')['words'])} words")
    check("GET phonetics 1:2", lambda: get("/surahs/1/ayahs/2/phonetics")["phonemes"][:24])
    check("GET tajweed 112:1",
          lambda: f"{len(get('/surahs/112/ayahs/1/tajweed')['occurrences'])} occurrences")
    check("GET tajweed/rules", lambda: f"{len(get('/tajweed/rules')['rules'])} rules")
    check("GET /reciters", lambda: f"{len(get('/reciters')['reciters'])} reciters")
    check("GET /recitations/1/2", lambda: get("/recitations/1/2")["licence"])

    def detect() -> str:
        r = httpx.post(f"{api}/recitation/detect-verse",
                       data={"text": "قل هو الله أحد"}, timeout=60)
        r.raise_for_status()
        body = r.json()
        return f"{body['surah']}:{body['ayah']}"

    check("POST /recitation/detect-verse", detect)

    def analyze() -> str:
        if not CLIP.exists():
            return "skipped - no test audio"
        with CLIP.open("rb") as fh:
            r = httpx.post(
                f"{api}/recitation/analyze",
                files={"audio": (CLIP.name, fh, "audio/mpeg")},
                data={"surah": 1, "ayah": 2}, timeout=300,
            )
        r.raise_for_status()
        body = r.json()
        assert body["errors"] == [], f"false correction: {body['errors']}"
        return f"score {body['word_accuracy_score']}, {len(body['errors'])} mistakes"

    check("POST /recitation/analyze", analyze)

    def pronunciation() -> str:
        if not CLIP.exists():
            return "skipped - no test audio"
        with CLIP.open("rb") as fh:
            r = httpx.post(
                f"{api}/recitation/pronunciation",
                files={"audio": (CLIP.name, fh, "audio/mpeg")},
                data={"surah": 1, "ayah": 2}, timeout=300,
            )
        r.raise_for_status()
        body = r.json()
        return f"available={body['available']}" + (
            f", {len(body['findings'])} findings" if body["available"] else ""
        )

    check("POST /recitation/pronunciation", pronunciation)

    def session() -> str:
        r = httpx.post(f"{api}/sessions", json={"surah": 1, "ayah": 2}, timeout=30)
        r.raise_for_status()
        sid = r.json()["session_id"]
        httpx.delete(f"{api}/sessions/{sid}", timeout=30)
        return "created and closed"

    check("POST /sessions", session)
    check("GET /openapi.json",
          lambda: f"{len(httpx.get(f'{base}/openapi.json', timeout=30).json()['paths'])} paths")
    check("GET / (demo page)",
          lambda: f"HTTP {httpx.get(base, timeout=30).status_code}")

    print(f"\n{passed} passed, {len(failures)} failed")
    for failure in failures:
        print(f"  {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
