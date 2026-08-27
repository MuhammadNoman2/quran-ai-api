# Test audio

Recordings for evaluating accuracy and, most importantly, the **false-correction
rate** — how often the system tells a correct reciter they were wrong.

**No audio is committed to git.** These directories are placeholders.

## Layout

| Directory | What to put in it |
|---|---|
| `correct/` | Correct recitations. The most important set — any error reported here is a false correction. |
| `missing_word/` | Reciter skips a word |
| `extra_word/` | Reciter adds a word |
| `repeated_word/` | Reciter repeats a word |
| `substituted_word/` | Reciter says a different word |
| `pronunciation/` | Correct words, wrong articulation (ق→ك, ص→س, ط→ت, ح→ه, ع→ء) |
| `noise/` | Background noise, room echo, poor microphone |
| `different_speakers/` | Children, adults, non-Arabic speakers, varying fluency |

## Adding a recording

1. Save as 16 kHz mono WAV (or MP3 — PyAV decodes both, no system FFmpeg needed).
2. Name it `{surah}_{ayah}_{speaker}_{n}.wav`, e.g. `001_002_child01_1.wav`.
3. Add a line to `manifest.csv` in the same directory:

   ```csv
   file,surah,ayah,expected_error_type,expected_word_index,notes
   001_002_child01_1.wav,1,2,none,,clean read by a 9-year-old
   001_002_child02_1.wav,1,2,word_missing,3,skipped rabbi
   ```

`expected_error_type` uses the categories in `docs/architecture.md` §5.

## Why the `correct/` set matters most

A Quran learning tool that wrongly corrects a child who recited properly does real
harm. The base model was measured hallucinating trailing words on **every** short
clip (`docs/model-selection.md` §3.2), so this set is the primary regression guard.
Aim for at least 50 clips across several speakers before trusting any accuracy claim.
