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

| `speech_started` | The reciter began speaking. `at` is in stream seconds. |
| `speech_stopped` | A pause long enough to count as the end of a phrase. |
| `word_detected` | **Provisional.** The fast tier recognized a word. Display only. |
| `correction` | A provisional judgement was revised. |

### Provisional versus confirmed

Two tiers run, with different authority:

* **`word_detected`** comes from the fast model (~1.7 s). Use it to light a word
  up as the reciter passes it. It is **never** a mistake claim - that model was
  measured emitting a wrong word at 0.90 confidence.
* **`word_confirmed`** and **`mistake_detected`** come from the slower model at
  each pause. Only these carry authority.
* **`correction`** fires when the slower tier disagrees with what the fast tier
  showed. Replace the provisional styling; do not show both.

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
mistakes. VAD supplies the speech-end boundary, which is much stronger evidence
than the last recognized word's timestamp — a hallucinated word carries its own
plausible timestamp and so moves the very boundary meant to catch it.

**A mistake needs corroboration before it is reported.** One pass is not enough:
the recognizer makes its own substitution errors on professional recitation
(measured — it heard `حِفْرُهُمَا` for `حِفْظُهُمَا`). A finding must either be seen
in two independent passes, or be confirmed by the other tier. When the two
engines disagree, the finding is withheld and **not counted against the score** —
a suppressed mistake that still cost points would be the same false correction
expressed as a number.

A given mistake is reported at most once per session.

## How the server schedules recognition

Whisper costs roughly `2.5s + 0.13 × duration` per call on CPU — a fixed cost
that dominates for short clips. Silero VAD costs 11–25 ms, so knowing whether the
reciter is still speaking is effectively free. Therefore:

* Recognition is scheduled by **speech**, not by a timer. A confirmed pass runs
  when VAD sees `STREAM_PAUSE_SECONDS` (0.6 s) of silence — a natural waqf, not an
  arbitrary boundary mid-word.
* While speech continues, the fast tier runs every `STREAM_PROVISIONAL_SECONDS`
  (2.5 s) of new audio to keep the display alive.
* A confirmed pass always outranks a provisional one. An earlier version simply
  dropped whichever request arrived while another pass was running, so a
  pause-triggered pass could be discarded because a provisional happened to be in
  flight — on a 60 s verse that left every word unconfirmed until the very end.
* Passes run **off** the WebSocket receive loop. Awaiting one inline stopped the
  server reading the socket for seconds, so it stopped answering keepalive pings
  and the connection was dropped.
* Each pass re-analyses all audio **in the current window**, not just the newest
  slice — stitching independent slices cuts words at boundaries and mishears them.
* Once `STREAM_WINDOW_SECONDS` (15 s) has accumulated, confirmed words are
  committed and their audio is dropped. Cumulative analysis is accurate but its
  cost grows with duration, and Whisper attends only 30 s regardless; without
  advancing, one pass over a 60 s verse took 21 s — slower than real time.
* The buffer is bounded (`STREAM_MAX_BUFFER_SECONDS`, 120 s). Past that the oldest
  audio is dropped and a `warning` is sent — the server degrades rather than grows.

Practical consequence for your UI: **confirmed feedback lags a phrase by roughly
one inference.** Use `word_detected` and `ayah_progress` for liveness, and treat
`mistake_detected` as arriving a beat later.

## Measured latency

Streamed at 100 ms frames on the development machine (Intel i7-9750H, CPU, int8):

| Verse | Audio | RTF | Confirmed lag after a pause |
|---|---|---|---|
| al-Ikhlas 112:1 | 2.9 s | 2.77 | — |
| al-Fatiha 1:2 | 6.3 s | 1.65 | 4.3 s |
| Ayat al-Kursi 2:255 | 60.7 s | **1.10** | 4.9 s |

RTF above 1.0 means this laptop cannot keep up with a continuously reciting user.
It has no AVX-512; the target VPS (EPYC 9354P, Zen 4, AVX-512 VNNI) should be
2.5–3× faster, which puts RTF around 0.4. **Re-run `scripts/benchmark_streaming.py`
on the target host before quoting any number.**

Note the direction: the *longer* the verse, the better the RTF, because the fixed
per-call overhead is amortised and the window keeps the per-pass cost flat.

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
