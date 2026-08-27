# Quran AI Recitation & Correction API

A local-first API for Quran recitation recognition and word-level error detection.
Client applications integrate recitation features over REST and WebSocket without
knowing anything about the underlying models.

**Status: Phase 4 complete** (Quran data, normalization, ASR, verse detection,
word alignment and error detection).
See `docs/` for the model selection, architecture and cost analysis.

## Documentation

| Document | Contents |
|---|---|
| [`docs/model-selection.md`](docs/model-selection.md) | Models evaluated, benchmarks actually run, licences, risks |
| [`docs/architecture.md`](docs/architecture.md) | Layering, ASR abstraction, alignment, streaming design |
| [`docs/cost-and-deployment.md`](docs/cost-and-deployment.md) | GPU/CPU costs, hosting, scaling ladder |

## Requirements

- Python 3.10+ (developed on 3.12)
- No GPU required. No PyTorch required — the inference path is CTranslate2 + ONNX Runtime.
- FFmpeg is **not** required (audio is decoded with PyAV).

## Setup

Everything installs into a project-local virtual environment. Nothing is installed
system-wide.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
python scripts/prepare_quran_data.py
scripts/test_recitation.py
scripts/evaluate_recitation.py
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
copy .env.example .env
python scripts\prepare_quran_data.py
```

> **Intel Mac note:** `onnxruntime` must stay at `1.23.2` — it is the last release with
> a macOS x86_64 wheel. PyTorch has shipped no macOS x86_64 wheel since 2.2.2, which is
> why the inference path avoids torch entirely. See `docs/model-selection.md` §1.1.

## Tests

```bash
pytest            # 196 fast tests, no model download
pytest -m slow    # 36 integration tests against the real models (~220 MB first run)
```

## Transcribing a recitation

```bash
python scripts/test_recitation.py
scripts/evaluate_recitation.py \
  --audio data/test_audio/correct/001_002_husary_1.mp3 --surah 1 --ayah 2
```

Add `--tier provisional` for the fast tiny model, or `--json` for machine-readable
output. The script reports what the model heard and which words fall below the
confidence gate; it does **not** judge the reciter - use the evaluation script for that.

## Measuring accuracy

```bash
python scripts/evaluate_recitation.py
```

Reports word accuracy, substitution/deletion/insertion rates and - the number that
matters - the **false correction rate**: how often the system tells a reciter they
made a mistake when they did not. Add your own recordings first; see
`data/test_audio/README.md`.

## Project layout

```
app/
  asr/
    base.py               ASREngine interface, ASRResult, tiers
    whisper_engine.py     the only module that imports faster_whisper
    registry.py           one shared engine per tier
  audio/decoder.py        PyAV decoding to 16 kHz mono float32
  core/config.py          environment-driven settings
  models/schemas.py       versioned Pydantic models
  quran/
    repository.py         read-only access to the prepared Quran asset
    verse_locator.py      Scenario B: which verse is this? (n-gram + fuzzy)
  alignment/
    base.py               ExpectedWord, SpokenWord, AlignmentResult
    word_alignment.py     Needleman-Wunsch over word tokens
  recitation/
    confidence.py         the gates that prevent false corrections
    scoring.py            deterministic word-accuracy score
    analyzer.py           orchestrates the pipeline
    text_normalizer.py    the four normalization representations
data/quran/quran.json     prepared asset (git-ignored, built by script)
data/test_audio/          evaluation corpus (see its README)
scripts/prepare_quran_data.py
scripts/test_recitation.py
scripts/evaluate_recitation.py
tests/unit/  tests/integration/
docs/
```

## Quran text attribution

This project uses the **Tanzil Quran Text** (Uthmani and Imlaey), via the
[`quran-transcript`](https://github.com/obadx/quran-transcript) package.

> Tanzil Quran Text
> Copyright © 2007–2024 Tanzil Project — <https://tanzil.net>
> Licensed under Creative Commons Attribution 3.0.
> Permission is granted to copy and distribute verbatim copies of this text,
> but **changing it is not allowed**.

The canonical text is stored unmodified. Every normalized or derived form is kept in a
separate field, and a test asserts the canonical text matches the source byte for byte.

## Licence

Application code: TBD. Third-party model and data licences are catalogued in
`docs/model-selection.md` §6.
