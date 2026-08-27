# Deployment

Three images, because the three deployments have genuinely different needs.

| Profile | Image | Size | What it adds |
|---|---|---|---|
| `api` (default) | `quran-api:cpu` | **958 MB** | Recognition, alignment, streaming, Tajweed occurrence |
| `phonemes` | `quran-api:phonemes` | **2.82 GB** | PyTorch + the Muaalem model: phoneme and Tajweed *verification* |
| `gpu` | `quran-api:gpu` | 2.82 GB + CUDA | The same, on a GPU |

```bash
docker compose -f docker/docker-compose.yml up api                 # CPU, port 8000
docker compose -f docker/docker-compose.yml --profile phonemes up   # port 8001
docker compose -f docker/docker-compose.yml --profile gpu up        # port 8002
```

## Why the split

The default image is **torch-free**. The inference path is CTranslate2 + ONNX
Runtime, which is faster on CPU and roughly 1.9 GB smaller than a torch install.
Torch is needed by exactly one optional feature — phoneme verification — and it
cannot be installed on macOS x86_64 at all, so keeping it in a separate image is
what lets the main service stay small *and* lets development happen on a Mac.

## Verified, not assumed

Both images were built and exercised end to end. Three real problems surfaced
that no local test could have caught:

* **`quranic-phonemizer` was missing from `requirements.txt`.** It was installed
  in the development venv, so every test passed while the container returned 500
  from the Tajweed endpoint. `scripts/smoke_test.py` exists because of this: unit
  tests run inside an environment that already has everything installed and
  therefore cannot catch dependency drift.
* **transformers 5.x broke `quran-muaalem`.** It imports
  `_HIDDEN_STATES_START_POSITION`, which the v5 major release removed, so the
  model failed at import. Verified that 4.55, 4.56 and 4.57 still export it;
  the image pins `transformers>=4.55,<5`.
* **The acoustic model rejects a spaced reference.** Its tokenizer fails with
  "Unable to create tensor…" when the phoneme string contains word spaces,
  because they no longer line up with the per-group sifat list. The reference is
  now carried in both forms — spaced for word attribution, unspaced for the model.

## Smoke test

```bash
python scripts/smoke_test.py http://127.0.0.1:8000
```

Exercises all fifteen endpoints against a running deployment. Run it after any
image build; it is the check that catches packaging drift.

## Configuration that matters

```
DEVICE=cpu                  # cpu | cuda | auto
COMPUTE_TYPE=int8           # int8 on CPU, float16 on GPU
CPU_THREADS=2               # per worker, NOT total
NUM_WORKERS=3
MAX_CONCURRENT_SESSIONS=3
PRELOAD_MODELS=true         # pay model load at boot, not on the first request
STORE_AUDIO=false
API_KEYS=                   # empty disables authentication
SERVE_UNVERIFIED_AUDIO=false
```

`CPU_THREADS=2` is deliberate. Thread scaling on this workload is poor — six
threads cost roughly twice the CPU of two for 1.4× the speed — so concurrency
comes from more workers, not fatter ones.

**Mount a volume at `/models`.** Without it every recreated container
re-downloads about 220 MB of weights (2.4 GB more for the phoneme model). The
compose file does this with a named volume.

## Sizing

Measured on the development machine (Intel i7-9750H, no AVX-512):

| | RTF | Verdict |
|---|---|---|
| provisional tier | 0.41 | keeps up |
| committed tier | 0.90 | keeps up, barely |
| streaming, 60 s verse | 1.10 | falls behind a continuous reciter |

The target VPS (EPYC 9354P, Zen 4, **AVX-512 VNNI**) should be 2.5–3× faster per
core, which is the difference between falling behind and comfortable headroom.
VNNI is the single biggest per-core factor for int8 inference — check for it with
`lscpu | grep avx512_vnni` before renting any host, and re-run
`scripts/benchmark_asr.py` there rather than trusting these numbers.

## Security before exposing this publicly

* Set `API_KEYS`. Authentication is off when it is empty, and startup logs a
  warning saying so.
* Put it behind TLS. The demo page needs a secure context for microphone access.
* Keep `STORE_AUDIO=false` unless you have a reason and a retention policy —
  particularly with children's voices (see `cost-and-deployment.md` §10.6).
* The rate limiter is in-process. Behind multiple workers it counts per worker,
  so use a gateway limiter if that matters.
