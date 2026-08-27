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

    store_audio: bool = False
    log_level: str = "INFO"

    @property
    def quran_asset_path(self) -> Path:
        return self.quran_data_dir / "quran.json"


settings = Settings()
