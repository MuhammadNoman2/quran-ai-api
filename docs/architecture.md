# Architecture

**Status:** proposal — awaiting approval
**Date:** 2026-08-27
**Companion document:** [`model-selection.md`](./model-selection.md)

This describes the minimum viable architecture. It is deliberately shaped by the
measurements in `model-selection.md` §3 rather than by a generic template — three of the
decisions below exist specifically because of numbers I observed on this machine.

---

## 1. The three measurements that shaped this design

| Measurement | Architectural consequence |
|---|---|
| Whisper base costs **~2.5 s fixed per call** on this CPU, independent of clip length | Streaming cannot be chunk-driven. Inference is **VAD-segment-driven**, and a "provisional" tier exists to give the UI something to show during the 2.5 s gap. |
| **Hallucinated trailing words** appear on every short clip, and decoder flags don't fix them | Confidence gating is not a nice-to-have layer bolted on at the end — it sits **between** the ASR and the alignment engine, as a mandatory filter stage. |
| Both ASR tiers make **genuine substitution errors** on professional recitation | A single pass can never *confirm* a user mistake. Confirmation requires either a second-tier model or repeated agreement. |

---

## 2. Layering

The rule is one-directional dependency: upper layers depend on interfaces, never on
concrete models.

```
                    ┌────────────────────────────────────────┐
   FastAPI          │  api/routes: health quran recitation   │
   (REST + WS)      │              sessions  websocket       │
                    └────────────────────┬───────────────────┘
                                         │  Pydantic schemas only
                    ┌────────────────────▼───────────────────┐
   Services         │ quran_service  recitation_service      │
   (orchestration)  │ analysis_service                       │
                    └────────────────────┬───────────────────┘
                                         │
     ┌───────────────┬───────────────────┼──────────────┬──────────────────┐
     ▼               ▼                   ▼              ▼                  ▼
┌──────────┐  ┌─────────────┐   ┌────────────────┐  ┌──────────┐  ┌───────────────┐
│  quran/  │  │   asr/      │   │  alignment/    │  │recitation│  │  streaming/   │
│repository│  │ ASREngine   │   │ WordAligner    │  │ analyzer │  │ session mgr   │
│normalizer│  │ (interface) │   │                │  │ scoring  │  │ buffer  VAD   │
│ matcher  │  └──────┬──────┘   └────────────────┘  │correction│  │ events        │
│ locator  │         │                              └──────────┘  └───────────────┘
└──────────┘   ┌─────┴─────┬──────────────┐
               ▼           ▼              ▼
        FasterWhisper  MuaalemEngine   (future)
          Engine        (Docker/GPU)
```

`RecitationAnalyzer` receives an `ASREngine`. It never imports `faster_whisper`.
This is what makes Tier A → Tier B swapping (and the eventual model replacement the
brief requires) a configuration change rather than a rewrite.

---

## 3. Quran data layer

Source: `quran-transcript`, which bundles Tanzil Uthmani + Imlaey offline (36 MB, no
network at runtime).

Each ayah is materialised **once** at prepare time into an immutable JSON asset, and
loaded into SQLite for indexed lookup:

```
Ayah
  surah, ayah, surah_name
  uthmani           # canonical display text — byte-for-byte Tanzil, NEVER modified
  imlaey            # canonical simple script
  words[]           # word_index, uthmani, imlaey, normalized forms
  phonetic          # QPS phonemes + Sifat (quran-transcript)   [optional, Phase 8]
  ipa               # IPA + tajweed rule spans (quranic-phonemizer) [optional, Phase 9]
```

The Tanzil licence forbids modification, which aligns with §7 of the brief. The
canonical text is therefore stored once and only ever *read*; every transformation
produces a **separate derived field**.

### 3.1 Normalization functions

The measured script mismatch (`ٱ`→`ا`, `ـٰ`→`ا`, inconsistent diacritics — see
model-selection §3.3) defines what these must do.

| Function | Removes | Preserves | Used by |
|---|---|---|---|
| `canonical_display_text()` | nothing — identity | everything | API responses, UI |
| `normalize_for_search()` | tashkeel, tatweel, Quranic annotation marks, punctuation; unifies alif/hamza/ya/ta-marbuta variants | letter skeleton | `VerseLocator`, user search |
| `normalize_for_asr_matching()` | tashkeel, tatweel, annotation marks; **normalizes Uthmani→Imlaey orthography** (`ٱ`→`ا`, dagger alif→`ا`); keeps hamza distinctions | consonant skeleton + hamza | `WordAligner` |
| `normalize_for_phoneme_comparison()` | **nothing orthographic** — delegates to QPS/IPA | full harakat, shadda, madd length, sukun | Phases 8–9 |

`normalize_for_asr_matching` is deliberately *less* aggressive than
`normalize_for_search`: collapsing hamza forms would destroy the `ء`/`ع` distinction that
Phase 8 needs to detect. Each function's exact behaviour gets a docstring table and
unit tests with the transformations enumerated.

---

## 4. ASR abstraction

```python
class ASREngine(Protocol):
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> ASRResult: ...
    def describe(self) -> EngineInfo:   # model name, device, compute type
        ...

@dataclass
class ASRResult:
    text: str
    words: list[ASRWord]      # text, start, end, probability
    language: str
    avg_logprob: float
    duration: float
```

`ASRWord.probability` is a **required** field of the interface, not an optional extra —
it is the primary hallucination signal (model-selection §3.2). An engine that cannot
supply per-word confidence is not usable in this system, and the interface says so.

Streaming is **not** a method on `ASREngine`. Whisper has no incremental decode, and
inventing a `transcribe_stream()` that internally re-transcribes a buffer would be the
"fake interface for a component we have not validated" that §43 warns against. Streaming
state lives in `streaming/session.py`, which calls plain `transcribe()` on VAD-delimited
segments.

### 4.1 Model management

```
MODEL_NAME=OdyAsh/faster-whisper-base-ar-quran
MODEL_PATH=./models
DEVICE=auto            # auto → cuda | cpu   (MPS: unavailable on Intel Mac)
COMPUTE_TYPE=int8
ASR_TIER_B_MODEL=      # optional; empty disables second-opinion confirmation
CONFIRM_STRATEGY=agreement   # agreement (CPU) | tier_b (GPU) | none
```

Loaded once at startup into a process-level registry (`lazy_load=true` optional).
Warm load measured at 1.62 s. Downloads go to `MODEL_PATH` and are never re-fetched.
`GET /health` reports the resolved model, device and compute type.

---

## 5. Alignment and error detection

Needleman–Wunsch global alignment over normalized word tokens, with a substitution cost
derived from Levenshtein distance on the normalized forms — so `ربّ` vs `رب` costs far
less than `ربّ` vs `مالك`, which is what lets us distinguish a probable pronunciation
difference from a real word substitution.

```
ExpectedWord  : text, normalized_text, index, ayah, position
SpokenWord    : text, normalized_text, start_time, end_time, confidence
AlignmentResult: expected_word, spoken_word, status, confidence, timing
```

Statuses: `CORRECT · SUBSTITUTED · MISSING · EXTRA · REPEATED · UNKNOWN`.

Error categories are kept **separate** from alignment statuses, per §24:

```
WORD_SUBSTITUTION  WORD_MISSING  WORD_EXTRA  WORD_REPETITION
POSSIBLE_PRONUNCIATION_ERROR
PHONEME_ERROR      ← Phase 8 only, requires phoneme evidence
TAJWEED_ERROR      ← Phase 9 only, requires rule detection
UNKNOWN
```

A normalized-form near-match (`رَبِّ`→`رَبَ`) is reported as
`POSSIBLE_PRONUNCIATION_ERROR`, never as `TAJWEED_ERROR`. Until Phase 8 exists, nothing
in the codebase is permitted to emit the last two — enforced by a unit test.

### 5.1 The hallucination filter (mandatory stage)

Sits between ASR and alignment. Two independent signals, from measurement:

1. **Word probability gate** — genuine words measured at 1.00, hallucinations at 0.17–0.87.
2. **Speech-boundary gate** — hallucinated words are emitted after speech ends, so
   `word.start > vad_speech_end` marks them regardless of probability. This catches the
   0.86–0.87 cases that a probability threshold alone would miss.

A word failing either gate is **dropped to `UNCERTAIN`, never reported as EXTRA**.
Both thresholds are configurable (`WORD_CONF_MIN`, default 0.90) and must be tuned
against the Phase 4 false-correction metric, not guessed.

### 5.2 Confidence policy

| Band | Condition | Reported as |
|---|---|---|
| `HIGH_CONFIDENCE_CORRECT` | aligned, prob ≥ 0.90 | `correct` |
| `HIGH_CONFIDENCE_ERROR` | mismatch, prob ≥ 0.90, **and** Tier-B confirms (if enabled) | the specific error category |
| `UNCERTAIN` | anything else | `uncertain` — explicitly *not* a mistake |

Because both ASR tiers were measured making real substitution errors on professional
recitation, **a Tier-A-only disagreement is `UNCERTAIN`, not an error.** Given the
brief's position that false corrections are the worst failure mode, the system defaults
to silence over accusation.

**CPU-only deployments (the current target — see cost-and-deployment §10.4):** Tier B
cannot run without a GPU, so confirmation instead comes from **agreement**, which
exploits a measured property — hallucinated words are *unstable* across decodes
(`قُطْ قُطْ` / `مُعْلَمُونَ` / `مَا يَغْفَى` from the same clip) while genuine words are
identical every time. A mismatch is confirmed only if it survives (a) two overlapping
inference windows, or (b) a re-decode of the suspect segment at a different temperature.
`CONFIRM_STRATEGY = agreement | tier_b | none` selects between them, so enabling a GPU
later is a config change.

### 5.3 Scoring

```
score = 100
        − 5.0 × substitutions
        − 5.0 × missing
        − 2.0 × extra
        − 1.0 × repetitions
        − 0.0 × uncertain          # uncertain never costs the user points
clamped to [0, 100]
```

Deterministic, documented, unit-tested, and computed **only** from confirmed word-level
statuses. It is labelled `word_accuracy_score` in the API — **not** a Tajweed score —
until a Tajweed engine exists.

---

## 6. Verse location

`VerseLocator` is separate from the ASR engine, per §12.

* **Scenario A** — client supplies surah/ayah. Highest accuracy; the default.
* **Scenario B** — inferred: normalize the transcript with `normalize_for_search()`,
  look up an n-gram index built over all 6,236 ayat, then rank candidates by
  Levenshtein ratio. `quran-transcript` ships a `PhoneticSearch` with prebuilt indices
  that is the first thing to evaluate here before writing our own.

Returns `{surah, ayah, confidence, matched_text}` — and is allowed to return no match.

---

## 7. Streaming design

The 2.5 s fixed inference cost makes this the most constrained part of the system.

```
mic → WS binary frames (PCM s16le 16 kHz)
        │
        ▼
   RollingBuffer (ring, bounded, memory-only)
        │
        ▼
   Silero VAD (ONNX, <1 ms per 30 ms frame)  ──► speech_started / speech_stopped events
        │
        ├─ speech continues ──► emit provisional position updates (no ASR)
        │
        ▼  on pause ≥ SILENCE_MS, or segment ≥ MAX_SEGMENT_S
   ASR segment inference  (~2.5 s + 0.13 × segment_len)
        │
        ▼
   hallucination filter → aligner → confidence policy → scoring
        │
        ▼
   WS events
```

VAD is what makes this affordable: it stops us paying 2.5 s to transcribe silence, and
it supplies the `vad_speech_end` boundary the hallucination filter needs. The two roles
justify it independently.

Inference runs in a worker thread (`asyncio.to_thread`) so the event loop keeps
accepting audio during the 2.5 s pass. One in-flight inference per session; if a segment
closes while one is running it queues, and the buffer is bounded so a slow consumer
degrades rather than exhausting memory.

### 7.1 Provisional vs committed

* **PROVISIONAL** — from the newest segment; may change. Never emits `mistake_detected`.
* **COMMITTED** — the segment is closed, confidence gates passed, and (if Tier B is
  enabled) confirmed. Only committed results can assert a mistake.

Word state machine: `PENDING → CONFIRMED → {CORRECT | ERROR | UNCERTAIN}`.

This is what gives the client responsive UI without premature accusation: position
updates flow continuously, corrections arrive only when earned.

### 7.2 Event protocol

`session_started · ready · speech_started · speech_stopped · partial_transcript ·
word_detected · word_confirmed · mistake_detected · correction · ayah_progress ·
ayah_completed · session_completed · warning · error`

Versioned (`"v": 1`) and documented in `docs/realtime.md`. A malformed or undecodable
audio frame emits `{"event":"error","code":"INVALID_AUDIO"}` and the session
**continues** — a bad chunk never terminates a session (§40).

---

## 8. API surface

```
GET  /api/v1/health
GET  /api/v1/surahs
GET  /api/v1/surahs/{surah_id}
GET  /api/v1/surahs/{surah_id}/ayahs/{ayah_id}
POST /api/v1/recitation/analyze        multipart: audio, surah, ayah
POST /api/v1/recitation/detect-verse
POST /api/v1/sessions
GET  /api/v1/recitations/{surah}/{ayah}
WS   /api/v1/sessions/{session_id}/stream
```

Responses expose `engine` as an opaque label (e.g. `"quran-asr-base"`), never
`faster_whisper` internals. Schemas are versioned from the first commit.

---

## 9. Cross-cutting

**Privacy** — `STORE_AUDIO=false` default; buffers are memory-only and dropped on
session end. Audio bytes are never logged. Enforced by a test asserting no filesystem
write occurs during analysis when disabled.

**Security** — max upload size, max session duration, sample-rate and codec allow-lists,
per-frame validation. `AuthProvider` and `RateLimiter` are interfaces with a no-op local
implementation — thin, but they are real seams we will need, not speculative abstraction.

**Logging** — structured JSON: `session_id`, timestamps, model, device, latency
breakdown, ASR/alignment/error summaries. No raw audio.

**Deployment** — CPU profile runs everything except Tier C. GPU profile adds Tier B and
Tier C. Tier C is reachable from the Intel Mac only via the `linux/amd64` container,
because of the torch constraint in model-selection §1.1.

---

## 10. Phase mapping

| Phase | Delivers | Exit test |
|---|---|---|
| 1 | Quran data + SQLite + JSON assets | canonical text round-trips byte-identical |
| 2 | `FasterWhisperEngine`, `scripts/test_recitation.py` | transcribes a real ayah with word probabilities |
| 3 | `VerseLocator` | finds correct ayah for held-out clips |
| 4 | Aligner + hallucination filter + error detection | **false-correction rate on clean recitations** |
| 5 | `POST /recitation/analyze` | schema-validated end-to-end |
| 6 | WebSocket + rolling buffer | session survives malformed frames |
| 7 | VAD + provisional/committed | measured event latency |
| 8–9 | Phoneme + Tajweed (Docker) | per-rule detectability, `NOT_IMPLEMENTED` where unproven |
| 10 | Verified recitation audio | licence recorded per reciter |
| 11–12 | Benchmarks + Docker | published latency table |

Phase 4's exit criterion is deliberately the false-correction rate rather than accuracy.
Given the measured hallucination behaviour, that is the number that decides whether this
system is safe to put in front of someone memorising Quran.

---

## 11. Deliberately not built yet

`transcribe_stream()` on the ASR interface · a Tajweed rule engine before Phase 9 ·
forced alignment before it is evaluated · a plugin system for models · caching layers ·
anything Docker-related before the local pipeline runs.

Per §43: smallest working version, tested, benchmarked, documented — then improved.
