# Models

This directory holds downloaded/converted model artifacts. It is git-ignored
except for this file — models are never committed.

| Role | Model | Licence | Size |
|---|---|---|---|
| Provisional (live view) | `hafizku/faster-whisper-tiny-ar-quran` | Apache-2.0 | ~75 MB int8 |
| Committed (corrections) | `OdyAsh/faster-whisper-base-ar-quran` | Apache-2.0 | ~145 MB int8 |

**The provisional tier may never report a mistake.** Measured: tiny produced a wrong
word at 0.90 confidence on Ayat al-Kursi. See `docs/model-selection.md` §3.

Models download on first use into this directory and are **not** re-fetched.
See `docs/model-selection.md` §4.3 for why we should eventually convert
`tarteel-ai/whisper-base-ar-quran` to CTranslate2 ourselves rather than rely on a
community-uploaded binary.
