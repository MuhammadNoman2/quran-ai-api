# Performance

All numbers below were measured on the development machine and are true of
nothing else. Re-run the benchmarks on the target host.

```
Intel Core i7-9750H · 6 cores / 12 threads · 16 GB · macOS x86_64
no CUDA, no AVX-512
```

## Offline recognition

`python scripts/benchmark_asr.py`

| clip | tier | audio | load | first | median | RTF |
|---|---|---|---|---|---|---|
| al-Fatiha 1:2 | provisional | 6.3 s | 2.57 s | 2.72 s | 2.59 s | 0.41 |
| Ayat al-Kursi | provisional | 60.7 s | 1.47 s | 7.49 s | 7.48 s | **0.12** |
| al-Ikhlas 112:1 | provisional | 2.9 s | 1.49 s | 1.76 s | 1.70 s | 0.58 |
| al-Fatiha 1:2 | committed | 6.3 s | 2.17 s | 5.94 s | 5.66 s | 0.90 |
| Ayat al-Kursi | committed | 60.7 s | 2.10 s | 21.47 s | 21.42 s | 0.35 |
| al-Ikhlas 112:1 | committed | 2.9 s | 2.06 s | 6.59 s | 6.52 s | **2.23** |

Resident memory is roughly 100–170 MB per engine.

**The shape to notice: short clips have the worst RTF.** Whisper pads every
window to 30 s, so cost is about `2.5 s + 0.13 × duration` and the fixed part
dominates. A 2.9 s clip costs nearly as much as a 6.3 s one. Two consequences:

* Batching short clips into one request is far cheaper than sending them
  separately.
* Streaming cannot run a pass per audio chunk. This is why recognition is
  scheduled by speech pauses, not by a timer.

## Streaming

`python scripts/benchmark_streaming.py <file> <surah> <ayah>`

| verse | audio | RTF | confirmed lag after a pause |
|---|---|---|---|
| al-Fatiha 1:2 | 6.3 s | 1.65 | 4.3 s |
| Ayat al-Kursi | 60.7 s | **1.10** | 4.9 s |

Longer verses score *better*, because the fixed per-call overhead amortises and
the advancing window keeps each pass's cost flat rather than letting it grow with
the recitation.

RTF above 1.0 means this laptop cannot keep pace with a continuously reciting
user. The target VPS (EPYC 9354P, Zen 4, AVX-512 VNNI) should be 2.5–3× faster
per core, putting streaming comfortably under 1.0.

## Phoneme analysis

Only available where PyTorch runs — the `phonemes` container or a GPU host. The
Muaalem model is 0.6 B parameters and downloads about 2.4 GB on first use. On CPU
inside Docker it took minutes per clip, which makes it suitable for
**asynchronous** analysis rather than live feedback. On a GPU it is practical
interactively; that has not been measured here and should not be quoted until it
is.

## What actually moves the numbers

1. **AVX-512 VNNI** — 2–3× on int8. The single biggest per-core factor. Check
   with `lscpu | grep avx512_vnni` before renting a host.
2. **`CPU_THREADS`** — 2 per worker. Six threads cost about twice the CPU of two
   for 1.4× the speed, so concurrency should come from more workers.
3. **`int8`** — 2–4× over float32 on CPU, already the default.
4. **VAD scheduling** — roughly 30 % of a session is silence never transcribed.
5. **The advancing window** — without it, one pass over a 60 s verse took 21 s
   and got worse with every pass.
6. **`PRELOAD_MODELS`** — moves 1.5–2.5 s off the first request.
