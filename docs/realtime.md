# Realtime protocol

    WS /api/v1/sessions/{session_id}/stream

Create the session over REST first (`POST /api/v1/sessions`), then connect. The
split is deliberate: when the server is at capacity it can refuse with a normal
`503` and a readable message, instead of accepting a socket and dropping it.

## Message flow

```
client                                   server
  |-- POST /sessions  ------------------->|
  |<-- 201 {session_id, websocket_url} ---|
  |-- WS connect ------------------------>|
  |-- {"type":"start","surah":1,"ayah":2} |
  |<-- session_started                    |
  |<-- ready                              |
  |-- <binary pcm_s16le frames> --------->|
  |<-- partial_transcript                 |
  |<-- word_confirmed / mistake_detected  |
  |<-- ayah_progress                      |
  |-- {"type":"stop"} ------------------->|
  |<-- ayah_completed                     |
  |<-- session_completed                  |
```

`{"type":"flush"}` forces a recognition pass without ending the session.

## Audio format

`pcm_s16le`, mono, at the session's `sample_rate` (16 kHz default). Send raw
binary frames — no container, no header. 100 ms frames work well; frame size is
not otherwise significant, because the server decides when to run recognition.

An odd byte count cannot be `s16le` and is rejected: truncating would shift every
later sample and corrupt the stream.

## Events

Every message is `{"v": 1, "event": "...", ...}`. Version bumps signal a breaking change.

| Event | Meaning |
|---|---|
| `session_started` | Session opened. Re-sent if the verse is later inferred. |
| `ready` | Server is accepting audio. Carries `expected_words` and the engine label. |
| `partial_transcript` | Recognition so far. `final: false` — **it may change.** |
| `word_confirmed` | An expected word was recognized at high confidence. |
| `mistake_detected` | A confirmed mistake. See below for what "confirmed" means. |
| `ayah_progress` | Words confirmed out of the total. Sent only when it moves. |
| `ayah_completed` | Final score for the ayah. |
| `session_completed` | Last message. |
| `warning` | Non-fatal: audio dropped, verse not yet identified. |
| `error` | `{code, message, fatal}`. `fatal: false` means the session continues. |

Declared but **not emitted yet** (Phase 7): `speech_started`, `speech_stopped`,
`word_detected`, `correction`. They are listed so the protocol is documented as a
whole; a test asserts the server never sends them, so a client can treat their
absence as "not implemented" rather than "nothing happened".

## When a mistake is reported

Two rules, and both exist to avoid accusing a reciter who did nothing wrong.

**Nothing is judged until the reciter has moved past it.** A partial recitation
aligned against the full ayah marks every not-yet-spoken word as missing. A word
only becomes eligible for judgement once a *later* word has been matched, so the
word currently being spoken is never judged. A student pausing mid-verse is not
told they skipped the rest of it. `stop` settles the whole ayah — that is when
genuinely omitted words are finally reported.

**Confidence gates apply as they do offline.** Low-confidence words and words
starting after speech ended are hallucination candidates and never become
mistakes. See `docs/api.md`.

A given mistake is reported at most once per session.

## How the server schedules recognition

Whisper costs roughly `2.5s + 0.13 × duration` per call on CPU — a fixed cost
that dominates for short clips. So:

* Audio accumulates in a bounded server-side buffer.
* A pass runs once `STREAM_SEGMENT_SECONDS` (4 s) of **new** audio has arrived —
  new, not total, or the trigger refires on every frame once the threshold is
  first crossed.
* Each pass re-analyses **all** the audio for the current ayah, not just the
  newest slice. Slightly longer clips are nearly free given the fixed overhead,
  while stitching independent slices cuts words at boundaries and mishears them.
* One pass at a time per session. Audio keeps buffering during a pass and is
  picked up by the next one.
* The buffer is bounded (`STREAM_MAX_BUFFER_SECONDS`, 60 s). Past that the oldest
  audio is dropped and a `warning` is sent — the server degrades rather than grows.

Practical consequence for your UI: **feedback lags speech by roughly one
inference.** Use `partial_transcript` and `ayah_progress` to show liveness, and
treat `mistake_detected` as arriving a beat later.

## Capacity

`MAX_CONCURRENT_SESSIONS` (3 by default) is a hard limit, because each live
session costs real CPU. A fourth `POST /sessions` returns `503`. Closing the
socket frees the slot; so does `DELETE /sessions/{id}`.

## Errors that do not end the session

Malformed JSON, an unknown message type, an odd-length PCM frame, a failed
recognition pass — each produces an `error` event with `fatal: false` and the
session continues. Only authentication failure (close code `4401`) and an
unknown or expired session (`4404`) close the socket.

## Example

`examples/streaming_client.py` streams a file in 100 ms frames and prints every event.
