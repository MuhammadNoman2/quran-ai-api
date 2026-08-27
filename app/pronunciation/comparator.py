"""Compare expected phonemes against what was heard.

Wraps `quran_transcript.explain_error`, which aligns two Quran Phonetic Script
strings and classifies each difference as an articulation, tashkeel or Tajweed
problem - and, for Tajweed, names the rule and the count it expected.

The wrapping is not ceremony. It keeps the library's shape out of the API (the
upstream field is spelled `preditected_ph`), attributes every finding to a word
index, and normalises rule objects into plain names.
"""

from __future__ import annotations

import logging

from quran_transcript import explain_error

from app.pronunciation.base import FindingKind, Operation, PronunciationFinding
from app.pronunciation.reference import PhoneticReference, _rule_name, _rule_name_ar

logger = logging.getLogger(__name__)

_KIND = {
    "normal": FindingKind.ARTICULATION,
    "tashkeel": FindingKind.TASHKEEL,
    "tajweed": FindingKind.TAJWEED,
}

_OPERATION = {
    "replace": Operation.REPLACE,
    "delete": Operation.DELETE,
    "insert": Operation.INSERT,
}


def compare(reference: PhoneticReference, recognized: str) -> tuple[PronunciationFinding, ...]:
    """Differences between the expected phonemes and `recognized`.

    Returns an empty tuple when they agree. Any failure inside the phonetic
    engine yields no findings rather than an exception: a pronunciation pass is
    an enhancement, and it must never be able to take down an analysis that
    otherwise succeeded.
    """
    if not recognized:
        return ()

    # An acoustic model returns phonemes with no word spaces; the reference is
    # carried in both forms, so compare like with like. Mixing them would align a
    # spaced string against an unspaced one and invent differences at every word
    # boundary.
    if " " in recognized or " " not in reference.phonemes:
        expected, mappings = reference.phonemes, reference.mappings
    else:
        expected, mappings = reference.model_phonemes, reference.model_mappings

    if recognized == expected:
        return ()

    try:
        raw = explain_error(reference.uthmani, expected, recognized, list(mappings))
    except Exception:
        logger.exception(
            "phonetic comparison failed",
            extra={"surah": reference.surah, "ayah": reference.ayah},
        )
        return ()

    return tuple(_convert(reference, error) for error in raw)


def _convert(reference: PhoneticReference, error: object) -> PronunciationFinding:
    span = getattr(error, "uthmani_pos", None)
    word_index = reference.word_at(span[0]) if span else None

    rules = (
        getattr(error, "ref_tajweed_rules", None)
        or getattr(error, "replaced_tajweed_rules", None)
        or getattr(error, "missing_tajweed_rules", None)
        or []
    )
    rule = rules[0] if rules else None

    return PronunciationFinding(
        kind=_KIND.get(getattr(error, "error_type", ""), FindingKind.UNKNOWN),
        operation=_OPERATION.get(
            getattr(error, "speech_error_type", ""), Operation.REPLACE
        ),
        expected_phonemes=getattr(error, "expected_ph", "") or "",
        # Upstream spelling; corrected at the boundary rather than propagated.
        spoken_phonemes=getattr(error, "preditected_ph", "") or "",
        word_index=word_index,
        uthmani_span=tuple(span) if span else None,
        expected_length=getattr(error, "expected_len", None),
        spoken_length=getattr(error, "predicted_len", None),
        tajweed_rule=_rule_name(rule) if rule is not None else None,
        tajweed_rule_ar=_rule_name_ar(rule) if rule is not None else None,
        detail=_describe(error, rule),
    )


def _describe(error: object, rule: object | None) -> str | None:
    """A short, honest sentence. No claim beyond what the engine actually found."""
    expected_len = getattr(error, "expected_len", None)
    spoken_len = getattr(error, "predicted_len", None)
    if rule is not None and expected_len is not None and spoken_len is not None:
        name = _rule_name(rule) or "a Tajweed rule"
        return f"{name}: expected a count of {expected_len}, heard {spoken_len}"
    operation = getattr(error, "speech_error_type", "")
    expected = getattr(error, "expected_ph", "")
    spoken = getattr(error, "preditected_ph", "")
    if operation == "delete":
        return f"expected {expected!r} was not heard"
    if operation == "insert":
        return f"heard {spoken!r} where nothing was expected"
    if expected or spoken:
        return f"expected {expected!r}, heard {spoken!r}"
    return None
