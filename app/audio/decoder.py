"""Audio decoding to the canonical ASR input format: 16 kHz mono float32 in [-1, 1].

Uses PyAV directly rather than shelling out to FFmpeg, so no system FFmpeg binary
is required (verified on macOS with no ffmpeg installed). PyAV is a general audio
library - the ASR engine libraries stay confined to app/asr/.
"""

from __future__ import annotations

import io
from pathlib import Path

import av
import numpy as np

TARGET_SAMPLE_RATE = 16_000

SUPPORTED_SAMPLE_RATES = (8_000, 16_000, 22_050, 24_000, 44_100, 48_000)


class AudioDecodeError(ValueError):
    """Raised when input cannot be decoded. Callers turn this into INVALID_AUDIO."""


def decode(source: str | Path | bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Decode a file path or raw bytes to mono float32 at `sample_rate`.

    Handles any container PyAV supports (wav, mp3, ogg, m4a, webm...), downmixes
    to mono and resamples. Returns a 1-D float32 array in [-1, 1].
    """
    handle: object = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    try:
        with av.open(handle, metadata_errors="ignore") as container:
            if not container.streams.audio:
                raise AudioDecodeError("no audio stream in input")
            stream = container.streams.audio[0]
            stream.thread_type = "AUTO"
            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=sample_rate
            )
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
            # flush
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1))
    except AudioDecodeError:
        raise
    except Exception as exc:  # PyAV raises a wide variety of errors
        raise AudioDecodeError(f"could not decode audio: {exc}") from exc

    if not chunks:
        raise AudioDecodeError("input decoded to zero samples")

    pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
    return np.clip(pcm, -1.0, 1.0)


def validate_pcm(audio: np.ndarray, sample_rate: int) -> None:
    """Guard the ASR input contract. Raises AudioDecodeError on violation.

    Deliberately refuses to resample silently: a caller sending 44.1 kHz has a
    bug, and quietly fixing it would hide a real problem in the client.
    """
    if audio.ndim != 1:
        raise AudioDecodeError(f"expected mono 1-D audio, got shape {audio.shape}")
    if audio.dtype != np.float32:
        raise AudioDecodeError(f"expected float32, got {audio.dtype}")
    if sample_rate != TARGET_SAMPLE_RATE:
        raise AudioDecodeError(
            f"expected {TARGET_SAMPLE_RATE} Hz, got {sample_rate} Hz - resample first"
        )
    if audio.size == 0:
        raise AudioDecodeError("empty audio")


def duration_seconds(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
    return len(audio) / sample_rate
