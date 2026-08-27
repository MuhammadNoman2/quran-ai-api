"""Offline recitation analysis.

Privacy: uploaded audio is held in memory, decoded, analysed and dropped. It is
never written to disk and never logged. `STORE_AUDIO` is honoured here - and
when it is false, as it is by default, there is simply no code path that
persists a recording.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from pydantic import BaseModel, Field

from app.api.deps import get_locator, get_settings, require_caller
from app.asr.base import Tier
from app.audio.decoder import AudioDecodeError, decode, duration_seconds
from app.core.config import Settings
from app.core.errors import APIError, ErrorCode
from app.models.schemas import RecitationAnalysis
from app.quran.verse_locator import VerseLocator
from app.recitation.analyzer import RecitationAnalyzer, VerseNotIdentified

router = APIRouter(
    prefix="/recitation", tags=["recitation"], dependencies=[Depends(require_caller)]
)
logger = logging.getLogger(__name__)


async def read_audio(upload: UploadFile, config: Settings) -> np.ndarray:
    """Read, size-check, decode and duration-check an upload. Memory only.

    The size limit is enforced while reading rather than after, so an oversized
    upload cannot exhaust memory before being rejected.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1 << 20):
        total += len(chunk)
        if total > config.max_audio_bytes:
            raise APIError(
                ErrorCode.AUDIO_TOO_LARGE,
                f"audio exceeds {config.max_audio_bytes} bytes",
                status.HTTP_413_CONTENT_TOO_LARGE,
            )
        chunks.append(chunk)

    if not total:
        raise APIError(ErrorCode.INVALID_AUDIO, "empty upload")

    try:
        audio = decode(b"".join(chunks))
    except AudioDecodeError as exc:
        raise APIError(ErrorCode.INVALID_AUDIO, str(exc)) from exc

    seconds = duration_seconds(audio)
    if seconds > config.max_audio_seconds:
        raise APIError(
            ErrorCode.AUDIO_TOO_LONG,
            f"audio is {seconds:.1f}s, limit is {config.max_audio_seconds:.0f}s",
            status.HTTP_413_CONTENT_TOO_LARGE,
        )
    return audio


class VerseDetection(BaseModel):
    surah: int
    ayah: int
    word_index: int = Field(..., description="Where in the ayah the recitation starts")
    confidence: float
    matched_text: str
    recognized_text: str
    spans_multiple_ayat: bool
    alternatives: int = Field(
        ..., description="Other equally good locations. The Quran repeats verses verbatim."
    )


@router.post(
    "/analyze",
    response_model=RecitationAnalysis,
    summary="Analyse a recitation against the Quran",
)
async def analyze(
    audio: UploadFile = File(..., description="wav, mp3, m4a, ogg or flac"),
    surah: int | None = Form(None, ge=1, le=114),
    ayah: int | None = Form(None, ge=1),
    tier: Tier = Form(
        Tier.COMMITTED,
        description="`committed` may report mistakes; `provisional` is faster and never does",
    ),
    caller: str = Depends(require_caller),
    config: Settings = Depends(get_settings),
    locator: VerseLocator = Depends(get_locator),
) -> RecitationAnalysis:
    """Word-level comparison of a recitation against the canonical text.

    Supply `surah` and `ayah` for the highest accuracy. Omit both and the verse
    is inferred from the audio, which is reported via `verse_detected`.

    `errors` contains only mistakes we are confident about. Findings we are not
    confident enough to assert - a low-confidence extra word, a single confusable
    letter - appear in `observations` instead and carry no score penalty. That
    asymmetry is deliberate: wrongly correcting a correct recitation is the worst
    thing this API can do.

    `word_accuracy_score` reflects words only. It is **not** a Tajweed score.
    """
    from app.api.deps import get_analyzer  # local import keeps the module import-light

    if (surah is None) != (ayah is None):
        raise APIError(
            ErrorCode.VALIDATION_ERROR,
            "supply both surah and ayah, or neither",
        )

    started = time.perf_counter()
    samples = await read_audio(audio, config)
    analyzer: RecitationAnalyzer = get_analyzer(tier)

    try:
        result = analyzer.analyze(samples, surah=surah, ayah=ayah)
    except VerseNotIdentified as exc:
        raise APIError(ErrorCode.VERSE_NOT_IDENTIFIED, str(exc)) from exc
    except KeyError as exc:
        raise APIError(
            ErrorCode.VERSE_NOT_FOUND, str(exc), status.HTTP_404_NOT_FOUND
        ) from exc
    finally:
        del samples  # audio is memory-only; drop it as soon as we are done

    logger.info(
        "recitation analysed",
        extra={
            "session_id": result.session_id,
            "caller": caller,
            "surah": result.surah,
            "ayah": result.ayah,
            "engine": result.engine.get("engine"),
            "tier": result.engine.get("tier"),
            "score": result.word_accuracy_score,
            "errors": len(result.errors),
            "observations": len(result.observations),
            "audio_seconds": result.audio_seconds,
            "latency_seconds": round(time.perf_counter() - started, 3),
        },
    )
    return result


@router.post(
    "/detect-verse",
    response_model=VerseDetection,
    summary="Identify which verse is being recited",
)
async def detect_verse(
    audio: UploadFile | None = File(None),
    text: str | None = Form(None, description="Skip ASR and match this text directly"),
    tier: Tier = Form(Tier.PROVISIONAL),
    config: Settings = Depends(get_settings),
    locator: VerseLocator = Depends(get_locator),
) -> VerseDetection:
    """Locate a recitation in the Quran without being told where it is.

    Repeated verses are reported rather than silently resolved: `alternatives`
    counts other equally good locations, and is 30 for the refrain of Surah
    Ar-Rahman because it genuinely occurs 31 times.
    """
    from app.api.deps import get_engine

    if (audio is None) == (text is None):
        raise APIError(ErrorCode.VALIDATION_ERROR, "supply either audio or text")

    if text is not None:
        recognized = text
    else:
        assert audio is not None
        samples = await read_audio(audio, config)
        recognized = get_engine(tier).transcribe(samples).text
        del samples

    match = locator.locate_best(recognized)
    if match is None:
        raise APIError(
            ErrorCode.VERSE_NOT_IDENTIFIED,
            "no verse matched the recognized text",
            status.HTTP_404_NOT_FOUND,
        )

    return VerseDetection(
        surah=match.surah,
        ayah=match.ayah,
        word_index=match.word_index,
        confidence=match.confidence,
        matched_text=match.matched_text,
        recognized_text=recognized,
        spans_multiple_ayat=match.spans_multiple_ayat,
        alternatives=match.alternatives,
    )
