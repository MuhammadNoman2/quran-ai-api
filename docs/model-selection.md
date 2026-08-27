# Model Selection Report

**Status:** proposal — awaiting approval
**Date:** 2026-08-27
**Scope:** STEP 1–2 of the development plan. No application code has been written.

Every claim in this document that is marked **[measured]** was produced by actually
installing and running the component on the target development machine on 2026-08-27.
Claims marked **[reported]** come from model cards, papers or repositories and have
**not** been independently verified. Nothing here is assumed from prior knowledge.

---

## 1. Target environment (measured)

| Property | Value |
|---|---|
| Machine | MacBook Pro, Intel Core i7-9750H @ 2.60 GHz (6 cores / 12 threads) |
| Architecture | `x86_64` (**not** Apple Silicon) |
| RAM | 16 GB |
| OS | macOS (Darwin 25.5.0) |
| Python | 3.12.10 |
| GPU | No CUDA. **No MPS** — MPS requires Apple Silicon |
| FFmpeg binary | Not installed |

This environment turned out to be the single most important constraint in the whole
selection, for the reason in the next section.

### 1.1 Blocking finding: PyTorch no longer ships macOS x86_64 wheels

I checked PyPI directly rather than trusting documentation:

| torch version | macOS wheels published |
|---|---|
| 2.2.2 | `macosx_10_9_x86_64` **and** `macosx_11_0_arm64` |
| 2.7.0 | arm64 only |
| 2.9.0 | arm64 only |
| 2.13.0 (latest) | arm64 only |

**PyTorch has shipped no macOS x86_64 wheel since 2.2.2.** [measured]

The same is true of ONNX Runtime: the last version with a macOS x86_64 wheel is
**1.23.2**; 1.29.0 (latest) is arm64-only. [measured]

Consequences, which drive the entire recommendation:

* Any component requiring `torch>=2.3` **cannot be pip-installed on this machine.**
  That includes `quran-muaalem` (requires `torch>=2.7.0`), `transformers`-based
  Whisper inference, `recitations-segmenter`, and the `silero-vad` PyPI package
  (which depends on `torch`).
* A **torch-free inference path is not a preference here, it is a requirement**
  for local development. CTranslate2 + ONNX Runtime provide exactly that.
* Torch-dependent components remain fully available, but must run in a
  `linux/amd64` Docker container (torch publishes manylinux x86_64 wheels) or on a
  GPU server. This is a normal, acceptable arrangement — it just has to be a
  deliberate architectural boundary rather than a surprise in month three.

---

## 2. Candidates investigated

### 2.1 ASR models

| Model | Base | Size | License | CPU-local? | Verdict |
|---|---|---|---|---|---|
| `tarteel-ai/whisper-base-ar-quran` | whisper-base | 74 M | **Apache-2.0** | via CT2 | **Selected (default)** |
| `OdyAsh/faster-whisper-base-ar-quran` | ↑ converted to CTranslate2 | ~145 MB int8 | Apache-2.0 | **yes** [measured] | **Selected (bootstrap artifact)** |
| `tarteel-ai/whisper-tiny-ar-quran` | whisper-tiny | 39 M | Apache-2.0 | yes | Fallback / low-power |
| `MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix` (+ `-ct2`) | large-v3-turbo | 809 M | Apache-2.0 (base repo; the `-ct2` repo declares **no license**) | too slow [measured] | **Selected (GPU / offline high-accuracy tier)** |
| `obadx/muaalem-model-v3_2` | w2v-BERT 2.0 | 0.6 B | MIT | no (needs torch) | **Selected (phoneme/Tajweed tier, Docker/GPU)** |
| `Quran-Lab/zipformer_p-arabic-v3` | Zipformer | — | `license: other` | untested | Deferred — licence unclear, flag before use |
| `Nuwaisir/Quran_speech_recognizer`, `IbrahimSalah/*`, various `wav2vec2-large-xlsr-quran-demo` | wav2vec2 | — | mixed | needs torch | Rejected — demo-grade, low adoption, unmaintained |

Tarteel reports **WER 5.75%** for `whisper-base-ar-quran` [reported]. I did not
reproduce that number; my own testing below is qualitative on 8 clips, not a WER benchmark.

### 2.2 Quran text, phonetics and Tajweed

| Component | Purpose | License | Deps | Installed & run? |
|---|---|---|---|---|
| **`quran-transcript` 0.6.0** (obadx) | Uthmani + Imlaey Quran text, word indexing, Quran Phonetic Script (QPS) + 10 Sifat attributes per phoneme, phonetic fuzzy search, `explain_error()` | Code **MIT**; bundled text **Tanzil CC-BY 3.0** | numpy, pydantic, Levenshtein, xmltodict, fuzzysearch — **no torch** | **yes** [measured] |
| **`quranic-phonemizer` 2.15.3** (Hetchy) | Tajweed-aware IPA G2P; per-rule annotations with source/host attribution; waqf handling | **MIT** | **PyYAML only** | **yes** [measured] |
| `quran-muaalem` 0.2.1 (obadx) | Inference wrapper for the multi-level CTC pronunciation model | **MIT** | torch ≥2.7, transformers | **no — blocked on this machine** |
| `recitations-segmenter` 1.0.0 (obadx) | Waqf (pause) segmentation, 99.58% acc [reported] | **MIT** | torch | no — blocked; and 0.6 B params is heavy for VAD |
| `Quran-Lab/mfa-quran-hafs` | Montreal Forced Aligner acoustic model + 20,967-entry contextual Hafs lexicon | **Apache-2.0** | MFA 3.4 (conda/Kaldi) | not attempted — heavy |

### 2.3 Runtime / infrastructure

| Component | Version | License | macOS x86_64 wheel | Verified |
|---|---|---|---|---|
| CTranslate2 | 4.8.1 | MIT | yes | **imported OK** [measured] |
| faster-whisper | 1.2.1 | MIT | pure python | **ran inference** [measured] |
| ONNX Runtime | **1.23.2 (pin required)** | MIT | yes (last one) | **imported OK**, providers: CoreML, CPU [measured] |
| PyAV (`av`) | 18.1.0 | BSD-3 / LGPL-linked FFmpeg | yes | decoded MP3 without a system FFmpeg binary [measured] |
| Silero VAD | 6.2.1 (ONNX weights) | MIT | via onnxruntime | to be wired in Phase 6 |

`faster-whisper` bundles a Silero VAD ONNX model and runs it through ONNX Runtime,
so we get VAD **without** the torch-dependent `silero-vad` package.

---

## 3. Benchmarks I actually ran [measured]

Setup: `faster-whisper` 1.2.1 / CTranslate2 4.8.1, `device=cpu`, `compute_type=int8`,
`OMP_NUM_THREADS=6`, `beam_size=5`, `word_timestamps=True`. Audio: verse-level MP3s
from everyayah.com (Husary and Alafasy, 128 kbps).

### 3.1 Speed

| Model | Warm load | 2.9 s clip | 5.5 s clip | 51.9 s clip | 60.7 s clip |
|---|---|---|---|---|---|
| Tarteel base (int8) | **1.62 s** | 2.7 s (RTF 0.91) | 2.8 s (RTF 0.51) | 9.1 s (RTF 0.17) | 8.7 s (RTF 0.14) |
| large-v3-turbo Quran (int8) | 7.65 s | **15.8 s (RTF 5.43)** | 15.4 s (RTF 2.78) | — | 66.4 s (RTF 1.09) |

**The most important number in this table is the fixed cost.** The base model costs
roughly **2.2–3.2 s per `transcribe()` call regardless of clip length**, because Whisper
pads every window to 30 s. Cost ≈ `2.5 s + 0.13 × duration`.

This determines what real-time can mean on this hardware: **we cannot run an inference
pass more often than roughly every 2.5–3 s on this CPU.** Any streaming design based on
"transcribe every 500 ms" is not physically possible here. Per §32 of the brief I am not
claiming a latency target — this is the measured floor to design around.

`large-v3-turbo` is **not viable for real-time on this CPU** (15.8 s to process 2.9 s of
audio). It is viable on GPU, and viable on CPU for non-interactive offline analysis.

One counter-intuitive measured result: **greedy decoding (`beam_size=1`) was *slower***
(4.0 s vs 2.7 s), because it triggers temperature-fallback retries. Do not assume
beam=1 is a speed win — benchmark it.

### 3.2 Accuracy, and the failure mode that matters most

On the core words, the base model is strong. On Ayat al-Kursi (60.7 s, 50 words) it
reproduced essentially the whole verse correctly.

But **every short clip ended in hallucinated trailing words**:

| Reference | Base model output |
|---|---|
| `قُلْ هُوَ ٱللَّهُ أَحَدٌ` | `قُلْ هُوَ اللَّهُ أَحَدٌ` **`قُطْ قُطْ`** |
| `قُلْ هُوَ ٱللَّهُ أَحَدٌ` | `قُلْ هُوَ اللَّهُ أَحَدٌ` **`قُطْتُمْ قُلْ`** |
| `ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ` | `الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ` **`الْحَمْدُ`** |
| `ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ` | `الرَّحْمَنِ الرَّحِيمِ` **`الرَّحِيمِ`** |

**A naive implementation would report every one of these as an EXTRA or REPEATED word —
i.e. it would tell a user who recited Surah al-Ikhlas perfectly that they added words.**
That is precisely the false-correction failure the brief calls extremely harmful (§14, §34).

I then tested whether decoder settings fix it. **They do not:**

| Config | Result |
|---|---|
| baseline | `... أَحَدٌ قُطْ قُطْ` |
| `condition_on_previous_text=False` | `... أَحَدٌ مُعْلَمُونَ وَالْمُسْلِمُ` (different garbage) |
| `+ vad_filter=True` | unchanged |
| `+ beam_size=1` | `... أَحَدٌ مَا يَغْفَى مَا يَغْفَى وَالْمُؤْمِنِينَ` (worse, and slower) |
| `+ no_repeat_ngram_size=3, compression_ratio_threshold=2.4` | unchanged |

**What *does* separate them is the per-word probability** (available via
`word_timestamps=True`):

```
قُلْ 1.00   هُوَ 1.00   اللَّهُ 1.00   أَحَدٌ 1.00   |   قُطْ 0.17   قُطْ 0.18
الْحَمْدُ 1.00  لِلَّهِ 1.00  رَبِّ 1.00  الْعَالَمِينَ 1.00  |  الْحَمْدُ 0.46
```

Genuine words scored **1.00**; hallucinated tails scored **0.17–0.87**. This is a clean,
usable signal — and it is the empirical justification for the confidence-gating design
in §14 of the brief. It is not a complete solution on its own (two hallucinations
scored 0.86–0.87), so the architecture pairs it with a second independent signal:
hallucinated words are emitted *after speech has ended*, so their word timestamps fall
outside the VAD-detected speech region.

By contrast, **`large-v3-turbo` produced no trailing hallucinations at all** on the same
clips — clean output, all low-probability words were genuine words it was merely unsure
about. Model size buys robustness here; it just costs more than this CPU can pay in real time.

### 3.3 Script mismatch (design-relevant)

The ASR emits **Imlaey-style** orthography, not Uthmani:

```
reference (Uthmani): ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ
ASR output (Imlaey): الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ
```

`ٱ` → `ا`, dagger alif `ـٰ` → `ا`. Diacritics are also emitted inconsistently
(`turbo` produced bare `الحمد` at probability 0.69). Matching must therefore run on a
normalized form, and `quran-transcript` ships exactly the needed asset — a bundled
`quran-uthmani-imlaey-map.json` (13 MB) giving the Uthmani↔Imlaey correspondence.

### 3.4 Both models make genuine substitution errors

Base produced `خَلْفَهُونَ` for `خَلْفَهُمْ`; both models produced `حِفْرُهُمَا` for
`حِفْظُهُمَا` — on a professional reciter's audio. **The ASR's own error rate is a
direct source of false corrections.** This is a measured argument for never reporting a
single-pass disagreement as a confirmed user mistake.

---

## 4. Recommendation

### 4.1 Three ASR tiers behind one interface

| Tier | Model | Runs where | Used for |
|---|---|---|---|
| **A — default / streaming** | Tarteel base → CTranslate2 int8 | CPU, this laptop | Live recitation, Phases 2–7 |
| **B — high accuracy** | large-v3-turbo Quran CT2 | GPU (or CPU offline) | Offline analysis; second-opinion pass to confirm a suspected mistake before reporting it |
| **C — phoneme / Tajweed** | `obadx/muaalem-model-v3_2` (multi-level CTC) | Docker `linux/amd64` or GPU | Phases 8–9 only |

Tier B is not decoration. Given §3.4, the honest way to avoid false corrections is:
**Tier A flags a candidate mistake; Tier B confirms it before we tell the user they were
wrong.** Tier A alone stays "uncertain".

### 4.2 Supporting stack

* **Quran text:** `quran-transcript` (bundles Tanzil Uthmani + Imlaey offline, 36 MB, no network at runtime).
* **Reference phonetics/Tajweed:** `quranic-phonemizer` for IPA + rule annotation (zero deps), `quran-transcript`'s QPS + Sifat for the Muaalem-compatible representation. These are *reference-side* — they describe what *should* be recited; they are not detection.
* **VAD:** Silero ONNX via `faster-whisper`'s bundled copy (torch-free).
* **Audio decode:** PyAV — no system FFmpeg needed. (FFmpeg still recommended for the data-prep scripts.)
* **Forced alignment:** deferred. `Quran-Lab/mfa-quran-hafs` (Apache-2.0) is the strongest option but needs the Kaldi-based MFA toolchain; evaluate in Phase 8, not before.

### 4.3 Model artifact policy

Convert `tarteel-ai/whisper-base-ar-quran` to CTranslate2 **ourselves**, once, inside a
`linux/amd64` container (conversion needs torch, which the container has), and cache the
result under `models/`. The third-party `OdyAsh/faster-whisper-base-ar-quran` conversion
works today and is fine as a bootstrap so development is not blocked — but a
community-uploaded binary should not be a permanent production dependency.

---

## 5. What I rejected, and why

* **Plain `openai/whisper-*`** — the brief explicitly warns against picking the popular Whisper model by default (§48). The Quran fine-tune is both smaller and better suited.
* **`transformers` + PyTorch for the main ASR path** — uninstallable on this machine (§1.1), and slower than CTranslate2 even where it does install.
* **`silero-vad` PyPI package** — drags in torch for a model we can run in ONNX.
* **`Quran-Lab/zipformer_p-arabic-v3`** — `license: other`, unexamined. Per §3 of the brief, flagged rather than adopted.
* **Any LLM in the inference path** — excluded by §30/§46.
* **TTS for Mode B** — excluded by §28.

---

## 6. Licence register

| Asset | Licence | Obligation |
|---|---|---|
| Tanzil Quran text (via `quran-transcript`) | **CC-BY 3.0** | Attribute Tanzil Project, link to tanzil.net, ship the copyright block, **verbatim only — modification prohibited** |
| `quran-transcript` code | MIT | notice |
| `quranic-phonemizer` | MIT | notice |
| `quran-muaalem`, Muaalem models, `recitations-segmenter` | MIT | notice |
| `tarteel-ai/whisper-*-ar-quran` | Apache-2.0 | notice + NOTICE file |
| large-v3-turbo Quran LoRA mix | Apache-2.0 on the source repo; **the `-ct2` repo states no licence** | ⚠️ convert from the Apache-2.0 source ourselves rather than depending on the unlicensed derivative |
| CTranslate2 / faster-whisper / ONNX Runtime / Silero VAD | MIT | notice |
| PyAV | BSD-3, links LGPL FFmpeg | keep FFmpeg dynamically linked; do not statically link GPL builds |
| `Quran-Lab/mfa-quran-hafs` | Apache-2.0 | notice |
| **everyayah.com audio** | **⚠️ no licence statement found on the site** | **Unresolved — see below** |

### 6.1 Flagged commercial risk: recitation audio (Mode B)

everyayah.com publishes no terms of use that I could find. Recitation recordings carry
the reciter's and publisher's rights; "widely mirrored on the Internet Archive" is not a
licence. **I would not ship everyayah audio in a commercial product without written
permission.** Recommended path: source Mode B audio from the Quranic Universal Library
(`qul.tarteel.ai`), which publishes per-recitation licence metadata, and record the
licence per reciter in the `RecitationRepository`. For local development the files I
downloaded are fine as test fixtures; they must not be redistributed.

The Tanzil "changing it is not allowed" clause is worth noting because it *reinforces*
§7 of the brief: the canonical text must stay byte-for-byte immutable, with all
normalization living in a separate derived representation. The licence requires the
architecture we already wanted.

---

## 7. Open risks

1. **Hallucinated trailing words** (§3.2) — highest-priority correctness risk. Mitigation designed in `architecture.md`; must be measured as a false-correction rate in Phase 4, not assumed solved.
2. **~2.5 s minimum inference cost on this CPU** (§3.1) — caps real-time responsiveness. Phase 6/7 must be designed around it; a GPU changes the picture entirely.
3. **ASR's own substitution errors** (§3.4) — motivates the two-tier confirm strategy.
4. **Muaalem tier unavailable natively on this laptop** — Phases 8–9 require Docker. Not a blocker for Phases 1–7.
5. **No WER benchmark yet** — my testing was 8 clips of professional reciters. Learner audio (the actual target population) will be harder. Real evaluation needs the Phase 33 test corpus.
6. **`obadx/muaalem-model-v3-mini` and `muaalem-streaming-rnn-v0` are undocumented** — potentially much better fits for real-time than the 0.6 B model. Worth investigating at Phase 8; not relied upon now.

---

## 8. Sources

- https://huggingface.co/tarteel-ai/whisper-base-ar-quran
- https://huggingface.co/OdyAsh/faster-whisper-base-ar-quran
- https://huggingface.co/MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix
- https://huggingface.co/obadx/muaalem-model-v3_2 · https://huggingface.co/papers/2509.00094 · https://arxiv.org/abs/2509.00094
- https://github.com/obadx/quran-muaalem · https://github.com/obadx/quran-transcript · https://github.com/obadx/recitations-segmenter
- https://github.com/Hetchy/Quranic-Phonemizer
- https://huggingface.co/Quran-Lab/mfa-quran-hafs
- https://github.com/SYSTRAN/faster-whisper · https://github.com/snakers4/silero-vad
- https://tanzil.net/docs/text_license · https://qul.tarteel.ai/ · https://everyayah.com/
- https://github.com/AbdirahmanNomad/IqraAI · https://github.com/yayaiu6/Real-Time-Quran-recitation-tracker-System
