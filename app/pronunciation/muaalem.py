"""Adapter for the Muaalem multi-level CTC phoneme model.

`obadx/muaalem-model-v3_2` (MIT, 0.6B parameters, wav2vec2-BERT) transcribes
Quranic audio into the Quran Phonetic Script along with articulation attributes.
It is the only credible open model for this job, and it is what makes
phoneme-level findings possible at all.

It requires PyTorch, which **cannot be installed on this project's development
machine**: PyTorch has published no macOS x86_64 wheel since 2.2.2, and
`quran-muaalem` requires torch>=2.7 (docs/model-selection.md 1.1). On Linux or a
GPU host it installs normally.

So this adapter reports itself unavailable rather than failing at call time, and
the analyzer degrades to reference-only output. The alternative - guessing at
phoneme errors without a phoneme model - is exactly the kind of invented finding
the brief forbids.
"""

from __future__ import annotations

import logging

import numpy as np

from app.pronunciation.base import PhonemeRecognizer, RecognizedPhonemes

logger = logging.getLogger(__name__)


class MuaalemRecognizer(PhonemeRecognizer):
    def __init__(self, model_id: str = "obadx/muaalem-model-v3_2", device: str = "auto") -> None:
        self._model_id = model_id
        self._device = device
        self._model = None
        self._reason: str | None = None
        self._checked = False

    # ── availability ─────────────────────────────────────────────────────────

    def _check(self) -> None:
        """Decide once whether this can run here. Never raises."""
        if self._checked:
            return
        self._checked = True
        try:
            import torch  # noqa: F401
        except ImportError:
            self._reason = (
                "PyTorch is not installed. On macOS x86_64 it cannot be - no wheel "
                "has been published since 2.2.2. Run the phoneme model in the "
                "linux/amd64 container or on a GPU host."
            )
            return
        try:
            import quran_muaalem  # noqa: F401
        except ImportError:
            self._reason = (
                "the quran-muaalem package is not installed "
                "(pip install quran-muaalem librosa)"
            )
            return
        self._reason = None

    @property
    def available(self) -> bool:
        self._check()
        return self._reason is None

    @property
    def unavailable_reason(self) -> str | None:
        self._check()
        return self._reason

    @property
    def name(self) -> str:
        return "muaalem"

    # ── inference ────────────────────────────────────────────────────────────

    def load(self) -> None:
        if not self.available or self._model is not None:
            return
        import torch
        from quran_muaalem import Muaalem

        device = self._device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = Muaalem(device=device)
        logger.info("loaded phoneme model", extra={"model": self._model_id, "device": device})

    def recognize(
        self, audio: np.ndarray, sample_rate: int = 16_000, *, reference: object = None
    ) -> RecognizedPhonemes:
        if not self.available:
            raise RuntimeError(f"phoneme recognition unavailable: {self.unavailable_reason}")
        if self._model is None:
            self.load()
        assert self._model is not None

        # Muaalem reads `.phonemes` off the phonetizer's own output object, so a
        # plain string is not accepted - it raised AttributeError when given one.
        # The unspaced form: the tokenizer rejects a spaced reference outright.
        raw = getattr(reference, "model_raw", None) or getattr(reference, "raw", None)
        context = [raw] if raw is not None else None
        results = self._model([audio], context, sampling_rate=sample_rate)
        first = results[0]
        return RecognizedPhonemes(
            phonemes=first.phonemes.text,
            sifat=tuple(getattr(first, "sifat", ()) or ()),
        )


def default_recognizer() -> PhonemeRecognizer:
    return MuaalemRecognizer()
