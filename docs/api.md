# API Reference

Base path `/api/v1`. Interactive docs at `/docs`, machine-readable at `/openapi.json`.

## The one thing to understand before integrating

**`errors` and `observations` are different claims, and your UI must treat them differently.**

| Field | Meaning | How to present it |
|---|---|---|
| `errors` | Mistakes we are confident about | Safe to show as "you got this wrong" |
| `observations` | Something looked off, but we are not confident | Show neutrally, or not at all. **Never** as a mistake. |

Speech recognition hallucinates. Measured on this project's own models, a clean
recording by a professional reciter produces spurious trailing words on nearly
every short clip. Those are filtered out server-side, but anything that survives
as doubtful lands in `observations`. A client that renders `observations` in red
re-introduces exactly the failure this API is built to avoid.

`word_accuracy_score` counts words only. **It is not a Tajweed score.** No Tajweed
or phoneme-level judgement exists yet; the API will never return one until it does.

## Authentication

Send `X-API-Key` when the server has keys configured. When it does not, the API is
open — `GET /health` reports `auth_enabled` so you can tell which mode you are in.
`/health` itself is always reachable.

## Endpoints

### `GET /health`
Liveness plus the effective configuration: resolved device, engine tiers, whether
auth is on, whether audio is being stored. Engines are named by opaque label
(`quran-asr-committed`); model identities are never exposed.

### `GET /surahs` · `GET /surahs/{id}` · `GET /surahs/{id}/ayahs/{id}`
Canonical Quran reference data. The ayah response includes a word-by-word breakdown.

`words[].index` is the **Imlaey** word position, and it is the index used in every
analysis result. `words[].uthmani_index` groups those onto Uthmani words for
display, because one Uthmani word can be two Imlaey words (`يَـٰٓأَيُّهَا` →
`يَا أَيُّهَا`). 363 of the 6,236 ayat are affected — group by `uthmani_index`
when rendering Uthmani text.

### `POST /recitation/analyze`
`multipart/form-data`: `audio` (required), `surah` + `ayah` (both or neither),
`tier` (`committed` default, or `provisional`).

Supplying surah and ayah is the highest-accuracy mode. Omit both and the verse is
inferred; the response sets `verse_detected: true` and `verse_confidence`.

```jsonc
{
  "session_id": "…", "surah": 1, "ayah": 2,
  "expected_text": "ٱلْحَمْدُ لِلَّهِ رَبِّ ٱلْعَـٰلَمِينَ",
  "recognized_text": "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ الْحَمْدُ",
  "word_accuracy_score": 100.0,
  "score_formula": "100 - 5.0*substitutions - 5.0*missing - 2.0*extra - 1.0*repetitions; …",
  "words": [
    { "index": 1, "expected": "ٱلْحَمْدُ", "spoken": "الْحَمْدُ",
      "status": "correct", "band": "high_confidence_correct", "confidence": 1.0 },
    { "index": null, "expected": null, "spoken": "الْحَمْدُ",
      "status": "extra", "band": "uncertain", "confidence": 0.46,
      "note": "confidence 0.46 below gate 0.90" }
  ],
  "errors": [], "observations": [],
  "engine": { "engine": "quran-asr-committed", "tier": "committed" }
}
```

That example is real output. The reciter was correct; the model invented a fifth
word; the gate suppressed it; the score stayed 100.

**Statuses:** `correct` `substituted` `missing` `extra` `repeated` `unknown`
**Categories:** `word_substitution` `word_missing` `word_extra` `word_repetition`
`possible_pronunciation_error`
**Bands:** `high_confidence_correct` `high_confidence_error` `uncertain`

`index` is `null` for words the reciter added — there is no expected position for them.

### `POST /recitation/detect-verse`
Send `audio` **or** `text`, not both. Returns the location plus `alternatives`,
the count of other equally good matches. The Ar-Rahman refrain returns 30, because
it genuinely occurs 31 times — repeated verses are reported, never silently resolved.

### `POST /sessions` · `GET /sessions/{id}` · `DELETE /sessions/{id}`
Reserve one of the server's concurrent streaming slots. Capacity is small and
deliberate: each live session costs real CPU inference time, so the server refuses
a session it cannot serve well rather than degrading every session already running.
A refusal is `503` with code `RATE_LIMITED`. Close sessions you finish with.

The WebSocket stream at the advertised `websocket_url` arrives in Phase 6.

## Errors

```json
{ "error": { "code": "INVALID_AUDIO", "message": "could not decode audio: …" } }
```

| Code | HTTP | |
|---|---|---|
| `VALIDATION_ERROR` | 400 | Bad combination of fields |
| `INVALID_AUDIO` | 400 | Undecodable or empty upload |
| `VERSE_NOT_IDENTIFIED` | 400/404 | No verse matched |
| `UNAUTHORIZED` | 401 | Missing or wrong `X-API-Key` |
| `VERSE_NOT_FOUND` | 404 | That surah/ayah does not exist |
| `AUDIO_TOO_LARGE` / `AUDIO_TOO_LONG` | 413 | Over the configured limit |
| `RATE_LIMITED` | 429/503 | Too many requests, or no session capacity |
| `INTERNAL_ERROR` | 500 | Logged server-side; no detail is returned |

## Audio

wav, mp3, m4a, ogg, flac, webm — anything PyAV decodes. Any sample rate; the server
downmixes to 16 kHz mono. Limits are `MAX_AUDIO_BYTES` (25 MB) and
`MAX_AUDIO_SECONDS` (300 s).

## Latency

Measured on the development machine (Intel i7-9750H, CPU, int8), per request:

| Tier | ~6 s clip |
|---|---|
| `provisional` | ~1.7 s |
| `committed` | ~5.4 s |

Whisper pads every window to 30 s, so cost is roughly `2.5 s + 0.13 × duration` —
a two-second clip costs nearly as much as a six-second one. Batching short clips
into one request is therefore much cheaper than sending them separately. Add ~2 s
for the first request if `PRELOAD_MODELS` is off.

## Clients

`examples/python_client.py` and `examples/browser_client.html`. Neither needs any
ML dependency — base URL, optional key, audio.
