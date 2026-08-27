#!/usr/bin/env python3
"""Build the immutable Quran data asset from quran-transcript (Tanzil text).

    python scripts/prepare_quran_data.py

Writes data/quran/quran.json. Run once; the output is reference data and is not
regenerated at runtime. The Tanzil licence (CC-BY 3.0) permits verbatim copies
only, so the canonical Uthmani text is copied through untouched - every derived
form is stored in a separate field.

Word indexing is Imlaey-based, because Imlaey is the orthography the ASR emits
and therefore the one we align against. 363 of the 6236 ayat have more Imlaey
words than Uthmani words (يَـٰٓأَيُّهَا -> يَا أَيُّهَا); `uthmani_index` records
which Uthmani word each Imlaey word belongs to.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quran_transcript import Aya  # noqa: E402
from quran_transcript.utils import PartOfUthmaniWord  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.models.schemas import Ayah, QuranAsset, Surah, Word  # noqa: E402
from app.quran.text_normalizer import normalize_for_asr_matching  # noqa: E402


def group_imlaey_to_uthmani(aya: Aya, n_imlaey: int) -> list[int]:
    """Return, for each 1-based Imlaey word, the 1-based Uthmani word it belongs to.

    Uses the library as the authority: `get_by_imlaey_words(start, window)` - where
    `start` is 0-based and `window` is a word count - raises PartOfUthmaniWord when
    the span cuts an Uthmani word in half, so we widen the window until it resolves.
    That is exactly the grouping we want.
    """
    groups: list[int] = [0] * n_imlaey
    uth_idx = 0
    start = 0  # 0-based, as the library expects
    while start < n_imlaey:
        window = 1
        while True:
            try:
                aya.get_by_imlaey_words(start, window)
                break
            except PartOfUthmaniWord:
                window += 1
                if start + window > n_imlaey:  # pragma: no cover - corrupt data
                    raise
        uth_idx += 1
        for w in range(start, start + window):
            groups[w] = uth_idx
        start += window
    return groups


def build() -> QuranAsset:
    surahs: list[Surah] = []
    ayat: list[Ayah] = []
    seen_surah: set[int] = set()

    for aya in Aya(1, 1).get_ayat_after():
        f = aya.get()
        uth_words = f.uthmani_words
        iml_words = f.imlaey_words

        if len(uth_words) == len(iml_words):
            groups = list(range(1, len(iml_words) + 1))
        else:
            groups = group_imlaey_to_uthmani(aya, len(iml_words))

        words = [
            Word(
                index=i + 1,
                imlaey=iml,
                normalized=normalize_for_asr_matching(iml),
                uthmani=uth_words[groups[i] - 1],
                uthmani_index=groups[i],
            )
            for i, iml in enumerate(iml_words)
        ]

        ayat.append(
            Ayah(
                surah=f.sura_idx,
                ayah=f.aya_idx,
                surah_name=f.sura_name,
                uthmani=f.uthmani,
                imlaey=f.imlaey,
                normalized=normalize_for_asr_matching(f.imlaey),
                words=words,
            )
        )

        if f.sura_idx not in seen_surah:
            seen_surah.add(f.sura_idx)
            surahs.append(
                Surah(surah=f.sura_idx, name=f.sura_name, ayah_count=f.num_ayat_in_sura)
            )

    return QuranAsset(surahs=surahs, ayat=ayat)


def main() -> None:
    asset = build()
    out = settings.quran_asset_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(asset.model_dump(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    words = sum(a.word_count for a in asset.ayat)
    print(f"surahs : {len(asset.surahs)}")
    print(f"ayat   : {len(asset.ayat)}")
    print(f"words  : {words}")
    print(f"written: {out}  ({out.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    main()
