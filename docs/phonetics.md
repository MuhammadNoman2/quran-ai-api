# Phonetics, pronunciation and Tajweed

## What is possible, and what is not

Two halves, with very different availability.

**The reference half needs no model.** `quran-transcript` converts the canonical
Uthmani text into the Quran Phonetic Script (QPS), with ten articulation
attributes (sifat) per phoneme group and the Tajweed rules that apply. Pure
Python, no torch, no GPU — so `GET /surahs/{s}/ayahs/{a}/phonetics` works
everywhere and always.

**The recognition half needs an acoustic phoneme model.** Only one credible open
model exists: `obadx/muaalem-model-v3_2` (MIT, 0.6 B parameters, wav2vec2-BERT,
multi-level CTC). It requires PyTorch, which **cannot be installed on macOS
x86_64** — no wheel has been published since 2.2.2, and `quran-muaalem` requires
`torch>=2.7`. See `model-selection.md` §1.1.

So on the development Mac, phoneme analysis reports itself **unavailable**, with
a reason. In the `linux/amd64` container, or on a GPU host, it runs.

## The rule that matters

`available: false` means **not checked**. It does not mean nothing was wrong.

```jsonc
{ "available": false,
  "findings": [],                     // "not checked" — NOT a clean pass
  "unavailable_reason": "PyTorch is not installed. On macOS x86_64 it cannot be…",
  "reference_phonemes": "قُل هُوَ للَااهُ ءَحَدڇ" }
```

A client that renders an empty `findings` list as "pronunciation perfect" would be
inventing a result. Check `available` first. `GET /health` reports the same
capability up front under `phoneme_analysis`, so a client can decide whether to
show the feature at all.

## What the comparison detects

Differences between expected and heard phonemes are classified by the phonetic
engine itself, not by us:

| `kind` | Meaning | Example |
|---|---|---|
| `articulation` | A different letter was produced | `قُ` heard as `كُ` |
| `tashkeel` | Short vowel, shadda or sukun differs | shadda dropped from `للَ` |
| `tajweed` | A rule with a defined count was not met | madd held for 1 instead of 2 |

Tajweed findings name the rule in English and Arabic and give both counts:

```json
{ "kind": "tajweed", "word_index": 2,
  "tajweed_rule": "Normal Madd", "tajweed_rule_ar": "المد الطبيعي",
  "expected_length": 2, "spoken_length": 1,
  "detail": "Normal Madd: expected a count of 2, heard 1" }
```

## Recitation style is a choice, not a constant

Hafs permits a **range** for several madd types. A school teaching four counts and
one teaching two are both correct. These are configuration:

```
REWAYA=hafs
MADD_MONFASEL_LEN=4
MADD_MOTTASEL_LEN=4
MADD_MOTTASEL_WAQF=4
MADD_AARED_LEN=4
```

`GET .../phonetics` returns the values it assumed, so a client can never mistake
one school's timing for a universal rule. Setting these wrongly would make the
system flag correct recitation as wrong — the exact failure this project treats as
most serious.

## Why the text pipeline still refuses to judge pronunciation

The word-level analyzer (`docs/api.md`) works from ASR *text*, which carries no
reliable evidence about short vowels or madd length. It therefore never emits
`phoneme_error` or `tajweed_error`, and a test enforces that. Those claims come
only from this pipeline, only when a phoneme model actually ran.

A near-miss in the text pipeline — `قل` heard as `كل` — is reported as
`possible_pronunciation_error` with no score penalty. That is a *hypothesis*
awaiting phoneme evidence, not a verdict.
