"""Holds the loaded engines and enforces the tier rule.

The two tiers exist because of measurement, not architecture taste: the tiny
model is ~3x faster and feels live, but was measured emitting a wrong word at
0.90 confidence. It therefore drives the display and never the corrections.
"""

from __future__ import annotations

import logging

from app.asr.base import ASREngine, Tier
from app.asr.whisper_engine import FasterWhisperEngine
from app.core.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)


class ASRRegistry:
    """Owns one engine per tier. Engines are shared across sessions.

    Never construct an engine per session: a shared model with `num_workers`
    gives real parallelism, while per-session models waste memory and CPU.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or default_settings
        self._engines: dict[Tier, ASREngine] = {}

    def _build(self, tier: Tier) -> ASREngine:
        s = self._settings
        model_id = (
            s.asr_provisional_model if tier is Tier.PROVISIONAL else s.asr_committed_model
        )
        return FasterWhisperEngine(
            model_id,
            name=f"quran-asr-{tier.value}",
            tier=tier,
            device=s.device,
            compute_type=s.compute_type,
            cpu_threads=s.cpu_threads,
            num_workers=s.num_workers,
            download_root=s.model_path,
        )

    def get(self, tier: Tier) -> ASREngine:
        if tier not in self._engines:
            self._engines[tier] = self._build(tier)
        return self._engines[tier]

    def register(self, tier: Tier, engine: ASREngine) -> None:
        """Inject an engine - used by tests to substitute FakeASREngine."""
        self._engines[tier] = engine

    def preload(self, *tiers: Tier) -> None:
        """Load at startup so the first request does not pay the load cost."""
        for tier in tiers or (Tier.PROVISIONAL, Tier.COMMITTED):
            self.get(tier).load()

    def unload_all(self) -> None:
        for engine in self._engines.values():
            engine.unload()

    def describe(self) -> list[dict[str, str]]:
        """For GET /health - which tiers exist, and whether they are loaded.

        Constructs any tier not yet built. That is cheap - constructing an engine
        does not load or download a model - and it means /health reports the real
        configuration before the first request rather than an empty list.
        """
        for tier in (Tier.PROVISIONAL, Tier.COMMITTED):
            self.get(tier)
        return [
            {
                **engine.describe().public(),
                "device": engine.describe().device,
                "compute_type": engine.describe().compute_type,
                "loaded": str(engine.is_loaded).lower(),
            }
            for engine in self._engines.values()
        ]
