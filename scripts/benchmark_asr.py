#!/usr/bin/env python3
"""Measure ASR performance on this machine.

    python scripts/benchmark_asr.py
    python scripts/benchmark_asr.py --tier committed --runs 5

Reports the metrics the brief asks for: model load time, first inference, average
inference, audio duration, processing duration, real-time factor, and CPU, RAM
and (where present) GPU memory.

Numbers are only ever true of the machine they were measured on. Run this on the
target host before quoting anything to anyone.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.asr.base import Tier  # noqa: E402
from app.asr.registry import ASRRegistry  # noqa: E402
from app.audio.decoder import decode  # noqa: E402
from app.core.config import settings  # noqa: E402

AUDIO_DIR = Path("data/test_audio/correct")


@dataclass
class Result:
    label: str
    tier: str
    audio_seconds: float
    load_seconds: float
    first_seconds: float
    mean_seconds: float
    median_seconds: float
    rtf: float
    words: int
    peak_rss_mb: float
    cpu_percent: float


def hardware() -> dict[str, str]:
    import platform

    info = {
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    try:
        import psutil

        info["cores_physical"] = str(psutil.cpu_count(logical=False))
        info["cores_logical"] = str(psutil.cpu_count(logical=True))
        info["ram_gb"] = f"{psutil.virtual_memory().total / 1024**3:.1f}"
    except ImportError:
        pass
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        info["cuda_devices"] = str(count)
    except Exception:
        info["cuda_devices"] = "0"
    # AVX-512 VNNI is worth 2-3x on int8 and is the single biggest per-core factor.
    try:
        flags = Path("/proc/cpuinfo").read_text(errors="ignore")
        info["avx512_vnni"] = "yes" if "avx512_vnni" in flags else "no"
    except OSError:
        info["avx512_vnni"] = "unknown (not Linux)"
    return info


def gpu_memory_mb() -> float | None:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        return 0.0


def benchmark(path: Path, tier: Tier, runs: int) -> Result:
    audio = decode(path)
    duration = len(audio) / 16_000

    gc.collect()
    baseline = rss_mb()

    registry = ASRRegistry()
    engine = registry.get(tier)
    started = time.perf_counter()
    engine.load()
    load_seconds = time.perf_counter() - started

    try:
        import psutil

        psutil.cpu_percent(interval=None)
    except ImportError:
        pass

    started = time.perf_counter()
    result = engine.transcribe(audio)
    first = time.perf_counter() - started

    timings = []
    for _ in range(max(0, runs - 1)):
        started = time.perf_counter()
        engine.transcribe(audio)
        timings.append(time.perf_counter() - started)
    timings = timings or [first]

    try:
        import psutil

        cpu = psutil.cpu_percent(interval=None)
    except ImportError:
        cpu = 0.0

    return Result(
        label=path.name,
        tier=tier.value,
        audio_seconds=duration,
        load_seconds=load_seconds,
        first_seconds=first,
        mean_seconds=statistics.mean(timings),
        median_seconds=statistics.median(timings),
        rtf=statistics.median(timings) / duration,
        words=len(result.words),
        peak_rss_mb=max(0.0, rss_mb() - baseline),
        cpu_percent=cpu,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=[t.value for t in Tier] + ["all"], default="all")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dir", type=Path, default=AUDIO_DIR)
    args = parser.parse_args()

    clips = sorted(p for p in args.dir.glob("*") if p.suffix in {".mp3", ".wav", ".m4a"})
    if not clips:
        print(f"no audio in {args.dir} - see data/test_audio/README.md", file=sys.stderr)
        return 1

    print("\nhardware")
    for key, value in hardware().items():
        print(f"  {key:18s} {value}")
    print(f"\nconfig             device={settings.device} compute={settings.compute_type} "
          f"threads={settings.cpu_threads} workers={settings.num_workers}")

    tiers = [Tier(args.tier)] if args.tier != "all" else list(Tier)
    results = [benchmark(clip, tier, args.runs) for tier in tiers for clip in clips]

    header = (
        f"\n{'clip':<26}{'tier':<13}{'audio':>7}{'load':>7}{'first':>8}"
        f"{'mean':>8}{'median':>8}{'RTF':>7}{'words':>7}{'RSS MB':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.label[:25]:<26}{r.tier:<13}{r.audio_seconds:>6.1f}s{r.load_seconds:>6.2f}s"
            f"{r.first_seconds:>7.2f}s{r.mean_seconds:>7.2f}s{r.median_seconds:>7.2f}s"
            f"{r.rtf:>7.2f}{r.words:>7}{r.peak_rss_mb:>9.0f}"
        )

    for tier in tiers:
        subset = [r for r in results if r.tier == tier.value]
        if subset:
            print(f"\n{tier.value:12s} median RTF {statistics.median(r.rtf for r in subset):.2f}"
                  f"   {'keeps up with a live reciter' if statistics.median(r.rtf for r in subset) < 1 else 'FALLS BEHIND a live reciter'}")

    vram = gpu_memory_mb()
    print(f"\nGPU memory in use  {f'{vram:.0f} MB' if vram is not None else 'no GPU detected'}")
    print("\nThese numbers describe THIS machine only. Re-run on the target host.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
