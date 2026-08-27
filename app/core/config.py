"""Application configuration. Environment-driven, no hardcoded model paths."""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    quran_data_dir: Path = Path("./data/quran")

    # ASR - see docs/model-selection.md. Two tiers by design:
    # provisional drives the live view and may never report a mistake;
    # committed is the only tier permitted to emit mistake_detected.
    asr_provisional_model: str = "hafizku/faster-whisper-tiny-ar-quran"
    asr_committed_model: str = "OdyAsh/faster-whisper-base-ar-quran"
    model_path: Path = Path("./models")
    device: str = "auto"
    compute_type: str = "int8"
    cpu_threads: int = 2
    num_workers: int = 3

    max_concurrent_sessions: int = 3

    # ─── Streaming (Phase 6) ──────────────────────────────────────────────
    #: How much audio to accumulate before running a recognition pass.
    #: Inference costs ~2.5s regardless of clip length, so very short segments
    #: waste almost the whole budget on fixed overhead. Benchmark before changing.
    stream_segment_seconds: float = 4.0

    #: Audio replayed at the start of the next segment so a word straddling a
    #: boundary is not cut in half and misheard as a mistake.
    stream_overlap_seconds: float = 0.75

    #: Hard cap on buffered audio. Past this the oldest is dropped and the
    #: client is warned, rather than the process growing without limit.
    stream_max_buffer_seconds: float = 120.0

    # ─── VAD-driven scheduling (Phase 7) ──────────────────────────────────
    vad_threshold: float = 0.5

    #: Silence that counts as the end of a phrase and triggers a committed pass.
    #: Quran recitation pauses at waqf, so this is roughly "the reciter stopped
    #: for breath or at a stopping point" rather than an arbitrary timer.
    stream_pause_seconds: float = 0.6

    #: New audio before a provisional (fast, display-only) pass while speech
    #: continues. Lower feels more live but costs an inference each time.
    stream_provisional_seconds: float = 2.5

    #: Once this much audio has accumulated, confirmed words are committed and
    #: their audio is dropped from the analysis window. Cumulative analysis is
    #: accurate but its cost grows with duration (~2.5s + 0.13 x seconds), and
    #: Whisper only attends 30s anyway - so on a long verse the window has to
    #: advance or the server falls further behind with every pass.
    stream_window_seconds: float = 15.0
    confirm_strategy: str = "agreement"
    word_conf_min: float = 0.90

    # ─── API ──────────────────────────────────────────────────────────────
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]

    #: Load models during startup instead of on the first request. Off by
    #: default so tests and `--reload` stay fast; turn on in production.
    preload_models: bool = False

    # ─── Limits (see docs/architecture.md 9) ──────────────────────────────
    max_audio_bytes: int = 25 * 1024 * 1024
    max_audio_seconds: float = 300.0
    max_session_seconds: float = 1800.0

    #: Comma-separated keys. Empty disables authentication, which is the
    #: correct default for local development and wrong for production.
    api_keys: str = ""

    #: Requests per minute per key. 0 disables limiting.
    rate_limit_per_minute: int = 0

    # ─── Recitation style (Phase 8) ───────────────────────────────────────
    #: Hafs permits a range for several madd types, so these are a legitimate
    #: teaching choice, not constants. Getting them wrong would flag correct
    #: recitation as wrong.
    rewaya: str = "hafs"
    madd_monfasel_len: int = 4
    madd_mottasel_len: int = 4
    madd_mottasel_waqf: int = 4
    madd_aared_len: int = 4

    store_audio: bool = False
    log_level: str = "INFO"

    @property
    def allowed_api_keys(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def auth_enabled(self) -> bool:
        return bool(self.allowed_api_keys)

    @property
    def quran_asset_path(self) -> Path:
        return self.quran_data_dir / "quran.json"


settings = Settings()
