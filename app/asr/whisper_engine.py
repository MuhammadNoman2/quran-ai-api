"""CTranslate2 / faster-whisper implementation of ASREngine.

This is the only module that imports faster_whisper. Replacing the backend means
writing a sibling of this file, not touching anything downstream.

Decoder settings are set from measurement, not defaults - see
docs/model-selection.md 3.2:

- `beam_size=5`: greedy (beam_size=1) was measured *slower* (4.0s vs 2.7s),
  because it triggers temperature-fallback retries, and it hallucinated more.
- `condition_on_previous_text=True`: chosen by A/B measurement over 8 clips, not
  by reasoning. Setting it to False - which seems right, since segments are
  independent - made the tiny model dramatically worse: 5/8 clips came back
  completely clean with True versus 0/8 with False, and False produced two
  hallucinated words *above* the 0.90 confidence gate, which is precisely the
  dangerous case. For the base model the two settings were equivalent (12 vs 11
  extra words, none above the gate). Overridable per engine.
- `vad_filter=False`: measured to make no difference to hallucination while
  costing time. VAD lives in the streaming layer, where it also decides *when*
  to run inference.
- `word_timestamps=True`: mandatory. Per-word probability is the hallucination
  signal, and word timings feed the speech-boundary check.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from app.asr.base import ASREngine, ASRResult, ASRWord, EngineInfo, Tier
from app.audio.decoder import TARGET_SAMPLE_RATE, validate_pcm

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Resolve 'auto' to a device CTranslate2 actually supports.

    CTranslate2 supports CPU and CUDA. It has **no MPS backend**, and this
    project's dev machine is an Intel Mac where MPS does not exist either, so
    'mps' is rejected rather than silently downgraded.
    """
    requested = requested.lower()
    if requested == "mps":
        raise ValueError(
            "CTranslate2 has no MPS backend; use DEVICE=cpu or DEVICE=cuda"
        )
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:  # pragma: no cover - depends on host
        logger.debug("CUDA probe failed; falling back to CPU", exc_info=True)
    return "cpu"


class FasterWhisperEngine(ASREngine):
    """A Whisper-family model served through CTranslate2."""

    def __init__(
        self,
        model_id: str,
        *,
        name: str,
        tier: Tier,
        device: str = "auto",
        compute_type: str = "int8",
        cpu_threads: int = 2,
        num_workers: int = 1,
        download_root: Path | str | None = None,
        beam_size: int = 5,
        language: str = "ar",
        condition_on_previous_text: bool = True,
    ) -> None:
        self._model_id = model_id
        self._name = name
        self._tier = tier
        self._device = resolve_device(device)
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        self._num_workers = num_workers
        self._download_root = str(download_root) if download_root else None
        self._beam_size = beam_size
        self._language = language
        self._condition_on_previous_text = condition_on_previous_text
        self._model = None
        self._lock = threading.Lock()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel

            started = time.perf_counter()
            self._model = WhisperModel(
                self._model_id,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=self._cpu_threads,
                num_workers=self._num_workers,
                download_root=self._download_root,
            )
            logger.info(
                "loaded engine=%s tier=%s device=%s compute=%s in %.2fs",
                self._name,
                self._tier.value,
                self._device,
                self._compute_type,
                time.perf_counter() - started,
            )

    def unload(self) -> None:
        with self._lock:
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def describe(self) -> EngineInfo:
        return EngineInfo(
            name=self._name,
            tier=self._tier,
            device=self._device,
            compute_type=self._compute_type,
        )

    # ── inference ─────────────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> ASRResult:
        validate_pcm(audio, sample_rate)
        if self._model is None:
            self.load()
        assert self._model is not None

        started = time.perf_counter()
        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            word_timestamps=True,
            condition_on_previous_text=self._condition_on_previous_text,
            vad_filter=False,
        )
        segments = list(segments)  # the generator is lazy; this is where work happens
        elapsed = time.perf_counter() - started

        words = tuple(
            ASRWord(
                text=w.word.strip(),
                start=float(w.start),
                end=float(w.end),
                # CTranslate2 can return values a hair above 1.0
                probability=min(1.0, max(0.0, float(w.probability))),
            )
            for s in segments
            for w in (s.words or [])
            if w.word.strip()
        )

        return ASRResult(
            text=" ".join(s.text.strip() for s in segments).strip(),
            words=words,
            language=info.language,
            audio_duration=float(info.duration),
            inference_seconds=elapsed,
            engine=self.describe(),
            avg_logprob=float(segments[0].avg_logprob) if segments else 0.0,
            no_speech_prob=float(segments[0].no_speech_prob) if segments else 1.0,
        )
