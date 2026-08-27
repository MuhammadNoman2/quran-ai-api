#!/usr/bin/env python3
"""Measure accuracy on the local test corpus - above all, false corrections.

    python scripts/evaluate_recitation.py
    python scripts/evaluate_recitation.py --tier provisional --dir data/test_audio

Reads `manifest.csv` from each subdirectory of data/test_audio (see its README)
and reports, per category and overall:

    word accuracy, substitution / deletion / insertion rates,
    FALSE CORRECTION RATE, missed error rate, and latency.

The headline number is the **false correction rate**: how often the system tells
a reciter they made a mistake when they did not. For a Quran learning tool that
is the metric that decides whether the system is safe to put in front of someone.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.asr.base import Tier  # noqa: E402
from app.asr.registry import ASRRegistry  # noqa: E402
from app.audio.decoder import decode  # noqa: E402
from app.recitation.analyzer import RecitationAnalyzer  # noqa: E402

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


@dataclass
class Case:
    path: Path
    surah: int
    ayah: int
    expected_error: str
    expected_index: int | None
    category: str


@dataclass
class Tally:
    total: int = 0
    clean_total: int = 0
    false_corrections: int = 0
    clean_clips_with_a_false_correction: int = 0
    error_total: int = 0
    errors_found: int = 0
    words_expected: int = 0
    words_correct: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    latencies: list[float] = field(default_factory=list)


def load_cases(root: Path) -> list[Case]:
    cases: list[Case] = []
    for manifest in sorted(root.glob("*/manifest.csv")):
        category = manifest.parent.name
        with manifest.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                path = manifest.parent / row["file"]
                if not path.exists():
                    print(f"  ! missing audio: {path}", file=sys.stderr)
                    continue
                cases.append(
                    Case(
                        path=path,
                        surah=int(row["surah"]),
                        ayah=int(row["ayah"]),
                        expected_error=(row.get("expected_error_type") or "none").strip(),
                        expected_index=int(row["expected_word_index"])
                        if row.get("expected_word_index", "").strip()
                        else None,
                        category=category,
                    )
                )
    return cases


def evaluate(cases: list[Case], analyzer: RecitationAnalyzer) -> dict[str, Tally]:
    tallies: dict[str, Tally] = defaultdict(Tally)
    for case in cases:
        for key in (case.category, "OVERALL"):
            tallies[key].total += 1
        started = time.perf_counter()
        result = analyzer.analyze(decode(case.path), surah=case.surah, ayah=case.ayah)
        elapsed = time.perf_counter() - started

        reported = len(result.errors)
        expects_no_error = case.expected_error in ("none", "")

        for key in (case.category, "OVERALL"):
            t = tallies[key]
            t.latencies.append(elapsed)
            t.words_expected += len([w for w in result.words if w.index is not None])
            t.words_correct += len([w for w in result.words if w.status == "correct"])
            t.substitutions += len([w for w in result.words if w.status == "substituted"])
            t.deletions += len([w for w in result.words if w.status == "missing"])
            t.insertions += len(
                [w for w in result.words if w.status in ("extra", "repeated")]
            )
            if expects_no_error:
                t.clean_total += 1
                t.false_corrections += reported
                t.clean_clips_with_a_false_correction += 1 if reported else 0
            else:
                t.error_total += 1
                t.errors_found += 1 if reported else 0
    return tallies


def rate(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:6.2f}%" if denominator else "     -"


def report(tallies: dict[str, Tally]) -> None:
    header = (
        f"{'category':<20}{'clips':>6}{'wordAcc':>9}{'sub':>8}{'del':>8}{'ins':>8}"
        f"{'FALSE CORR':>12}{'missed':>9}{'p50 s':>8}"
    )
    print("\n" + header)
    print("-" * len(header))
    for key in sorted(tallies, key=lambda k: (k == "OVERALL", k)):
        t = tallies[key]
        p50 = statistics.median(t.latencies) if t.latencies else 0.0
        print(
            f"{key:<20}{t.total:>6}"
            f"{rate(t.words_correct, t.words_expected):>9}"
            f"{rate(t.substitutions, t.words_expected):>8}"
            f"{rate(t.deletions, t.words_expected):>8}"
            f"{rate(t.insertions, t.words_expected):>8}"
            f"{rate(t.clean_clips_with_a_false_correction, t.clean_total):>12}"
            f"{rate(t.error_total - t.errors_found, t.error_total):>9}"
            f"{p50:>8.2f}"
        )
    overall = tallies.get("OVERALL")
    if overall and overall.clean_total:
        print(
            f"\nFALSE CORRECTION RATE: "
            f"{overall.clean_clips_with_a_false_correction}/{overall.clean_total} clean clips "
            f"({rate(overall.clean_clips_with_a_false_correction, overall.clean_total).strip()}) "
            f"were wrongly corrected."
        )
        print("This is the number that decides whether the system is safe to ship.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("data/test_audio"))
    p.add_argument("--tier", choices=[t.value for t in Tier], default=Tier.COMMITTED.value)
    args = p.parse_args()

    cases = load_cases(args.dir)
    if not cases:
        print(
            f"No manifest.csv found under {args.dir}.\n"
            "Add recordings and a manifest as described in data/test_audio/README.md.\n"
            "Until then there is no evidence about accuracy - do not quote any.",
            file=sys.stderr,
        )
        return 1

    engine = ASRRegistry().get(Tier(args.tier))
    engine.load()
    print(f"engine: {engine.describe().name} ({engine.describe().device}) - {len(cases)} clips")
    report(evaluate(cases, RecitationAnalyzer(engine)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
