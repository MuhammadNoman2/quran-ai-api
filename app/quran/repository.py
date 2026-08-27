"""Read-only access to the prepared Quran data asset.

Loads once into memory (the whole Quran is a few MB) and serves lookups from
dicts. The asset is treated as immutable: nothing here mutates or rewrites it.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

from app.core.config import settings
from app.models.schemas import Ayah, QuranAsset, Surah


class QuranRepository:
    def __init__(self, asset_path: Path | None = None) -> None:
        self._path = asset_path or settings.quran_asset_path

    @cached_property
    def _asset(self) -> QuranAsset:
        if not self._path.exists():
            raise FileNotFoundError(
                f"Quran data asset not found at {self._path}. "
                "Run: python scripts/prepare_quran_data.py"
            )
        return QuranAsset.model_validate_json(self._path.read_text(encoding="utf-8"))

    @cached_property
    def _by_ref(self) -> dict[tuple[int, int], Ayah]:
        return {(a.surah, a.ayah): a for a in self._asset.ayat}

    @cached_property
    def _by_surah(self) -> dict[int, Surah]:
        return {s.surah: s for s in self._asset.surahs}

    # ── queries ───────────────────────────────────────────────────────────────

    def surahs(self) -> list[Surah]:
        return self._asset.surahs

    def surah(self, surah: int) -> Surah:
        if surah not in self._by_surah:
            raise KeyError(f"surah {surah} out of range (1-114)")
        return self._by_surah[surah]

    def ayah(self, surah: int, ayah: int) -> Ayah:
        key = (surah, ayah)
        if key not in self._by_ref:
            raise KeyError(f"ayah {surah}:{ayah} does not exist")
        return self._by_ref[key]

    def ayat_of(self, surah: int) -> list[Ayah]:
        return [a for a in self._asset.ayat if a.surah == surah]

    def all_ayat(self) -> list[Ayah]:
        return self._asset.ayat

    @property
    def attribution(self) -> str:
        """Tanzil CC-BY 3.0 requires this be surfaced. Do not strip it."""
        return self._asset.attribution
