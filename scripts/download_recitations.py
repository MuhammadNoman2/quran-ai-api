#!/usr/bin/env python3
"""Cache recitation audio locally.

    python scripts/download_recitations.py --reciter husary --surah 1
    python scripts/download_recitations.py --reciter husary --surah 2 --ayah 255

Downloading is for local development and evaluation. It does **not** grant any
right to redistribute the recordings: a recitation is a performance carrying the
reciter's and publisher's rights, and no source publishes per-recitation licences.
The server will not serve unverified audio unless SERVE_UNVERIFIED_AUDIO is
explicitly enabled.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402
from app.quran.repository import QuranRepository  # noqa: E402
from app.recitations.registry import LicenceStatus, all_reciters, get  # noqa: E402
from app.recitations.repository import RecitationRepository  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reciter", default="husary")
    parser.add_argument("--surah", type=int, help="omit to download the whole Quran")
    parser.add_argument("--ayah", type=int, help="a single ayah")
    parser.add_argument("--list", action="store_true", help="list reciters and exit")
    parser.add_argument("--delay", type=float, default=0.25, help="seconds between requests")
    args = parser.parse_args()

    if args.list:
        for reciter in all_reciters():
            print(f"  {reciter.id:18s} {reciter.name_en:38s} licence={reciter.licence.value}")
        return 0

    reciter = get(args.reciter)
    if reciter is None:
        print(f"unknown reciter {args.reciter!r}; try --list", file=sys.stderr)
        return 2
    if reciter.licence is LicenceStatus.RESTRICTED:
        print(f"{reciter.name_en} is marked restricted: {reciter.licence_note}", file=sys.stderr)
        return 2

    quran = QuranRepository()
    store = RecitationRepository(settings.recitations_dir)

    if args.surah and args.ayah:
        targets = [(args.surah, args.ayah)]
    elif args.surah:
        targets = [(args.surah, a.ayah) for a in quran.ayat_of(args.surah)]
    else:
        targets = [(a.surah, a.ayah) for a in quran.all_ayat()]

    print(f"{reciter.name_en} - {len(targets)} ayah(s) from {reciter.source}")
    if reciter.licence is not LicenceStatus.VERIFIED:
        print(f"  NOTE: licence is '{reciter.licence.value}'. {reciter.licence_note}")

    downloaded = skipped = failed = 0
    for surah, ayah in targets:
        path = store.path_for(reciter.id, surah, ayah)
        if path.exists():
            skipped += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(reciter.url_for(surah, ayah), timeout=30) as response:
                path.write_bytes(response.read())
            downloaded += 1
        except Exception as exc:
            print(f"  ! {surah}:{ayah} failed: {exc}", file=sys.stderr)
            failed += 1
            continue
        if downloaded % 25 == 0:
            print(f"  {downloaded} downloaded…")
        time.sleep(args.delay)

    print(f"\ndownloaded {downloaded}, already present {skipped}, failed {failed}")
    print(f"cached under {settings.recitations_dir / reciter.id}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
