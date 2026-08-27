#!/usr/bin/env python3
"""Offline transcription check - the Phase 2 milestone.

    python scripts/test_recitation.py --audio data/test_audio/correct/001_002.mp3 \
        --surah 1 --ayah 2

Transcribes one audio file and shows it beside the expected ayah. It deliberately
does **not** align or judge correctness - that is Phase 4. The point here is to
confirm the engine works end to end and to eyeball what the model actually emits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.asr.base import Tier  # noqa: E402
from app.asr.registry import ASRRegistry  # noqa: E402
from app.audio.decoder import decode  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.quran.repository import QuranRepository  # noqa: E402
from app.quran.text_normalizer import normalize_for_asr_matching  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Transcribe a recitation and show the reference")
    p.add_argument("--audio", required=True, type=Path)
    p.add_argument("--surah", type=int)
    p.add_argument("--ayah", type=int)
    p.add_argument(
        "--tier",
        choices=[t.value for t in Tier],
        default=Tier.COMMITTED.value,
        help="provisional = tiny (fast, display only); committed = base (corrections)",
    )
    p.add_argument("--json", action="store_true", help="machine-readable output only")
    args = p.parse_args()

    if not args.audio.exists():
        print(f"error: no such file: {args.audio}", file=sys.stderr)
        return 2

    audio = decode(args.audio)
    engine = ASRRegistry().get(Tier(args.tier))
    result = engine.transcribe(audio)

    payload: dict[str, object] = {
        "transcription": result.text,
        "confidence": round(
            sum(w.probability for w in result.words) / len(result.words), 4
        )
        if result.words
        else 0.0,
    }

    reference = None
    if args.surah and args.ayah:
        reference = QuranRepository().ayah(args.surah, args.ayah)
        payload |= {
            "surah": args.surah,
            "ayah": args.ayah,
            "expected_text": reference.uthmani,
            "expected_words": reference.word_count,
        }

    payload |= {
        "recognized_words": len(result.words),
        "audio_seconds": round(result.audio_duration, 2),
        "inference_seconds": round(result.inference_seconds, 2),
        "real_time_factor": round(result.real_time_factor, 3),
        "engine": result.engine.public(),
        "device": result.engine.device,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    low = result.words_below(settings.word_conf_min)
    print(f"\n  engine        {result.engine.name}  ({result.engine.device}, "
          f"{result.engine.compute_type})")
    print(f"  audio         {result.audio_duration:.2f}s")
    print(f"  inference     {result.inference_seconds:.2f}s  (RTF {result.real_time_factor:.2f})")
    print(f"  mean word conf{float(payload['confidence']):>7.3f}")
    if reference:
        print(f"\n  expected  ({reference.word_count} words)  {reference.uthmani}")
        print(f"  normalized                   {reference.normalized}")
    print(f"\n  recognized ({len(result.words)} words)  {result.text}")
    print(f"  normalized                   {normalize_for_asr_matching(result.text)}")

    if low:
        print(f"\n  {len(low)} word(s) below the {settings.word_conf_min} confidence gate:")
        for w in low:
            print(f"     {w.probability:5.2f}  [{w.start:6.2f}-{w.end:6.2f}s]  {w.text}")
        print("\n  These are hallucination candidates. Phase 4 decides what to do with")
        print("  them - this script only reports, it never judges the reciter.")
    else:
        print(f"\n  all words at or above the {settings.word_conf_min} confidence gate")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
