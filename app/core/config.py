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
