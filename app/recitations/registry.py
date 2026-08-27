"""Reciters, and - the part that actually matters - their licensing status.

Mode B of the brief: return a verified recitation, never synthesise one.

Licensing here is genuinely unresolved, and this module refuses to paper over it.
Research found no source publishing per-recitation licences for the *recordings*:

* **everyayah.com** publishes no terms of use at all.
* **QUL (qul.tarteel.ai)** lists 133 recitations but shows no per-recitation
  licence metadata.
* **quranicaudio.com's** repository is MIT licensed - but that covers the
  **website software**, not the recordings. Reading it as a licence for the audio
  would be a serious and easy mistake.

A recitation is a performance. It carries the reciter's and the publisher's
rights, and "widely mirrored on the internet" is not a licence. So every reciter
below declares a `licence` status, unknown by default, and the API always reports
it. Serving audio whose licence is unknown is **off by default** - a developer
should have to decide that deliberately rather than discover it after shipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LicenceStatus(str, Enum):
    VERIFIED = "verified"
    """Written permission or an explicit licence is on file for redistribution."""

    UNKNOWN = "unknown"
    """No licence statement was found. Fine for local development, not for shipping."""

    RESTRICTED = "restricted"
    """Known to prohibit redistribution."""


class Style(str, Enum):
    MURATTAL = "murattal"
    MUJAWWAD = "mujawwad"


@dataclass(frozen=True)
class Reciter:
    id: str
    name_en: str
    name_ar: str
    style: Style
    riwaya: str
    source: str
    licence: LicenceStatus
    licence_note: str
    url_template: str
    """Upstream URL, formatted with `surah` and `ayah` as three-digit numbers."""

    def url_for(self, surah: int, ayah: int) -> str:
        return self.url_template.format(key=f"{surah:03d}{ayah:03d}")

    @property
    def redistributable(self) -> bool:
        return self.licence is LicenceStatus.VERIFIED


_UNKNOWN_NOTE = (
    "No licence statement was found for these recordings. Usable for local "
    "development and evaluation. Obtain written permission before redistributing "
    "them or serving them in a commercial product."
)

RECITERS: dict[str, Reciter] = {
    r.id: r
    for r in [
        Reciter(
            id="husary",
            name_en="Mahmoud Khalil Al-Husary",
            name_ar="محمود خليل الحصري",
            style=Style.MURATTAL,
            riwaya="hafs",
            source="everyayah.com",
            licence=LicenceStatus.UNKNOWN,
            licence_note=_UNKNOWN_NOTE,
            url_template="https://everyayah.com/data/Husary_128kbps/{key}.mp3",
        ),
        Reciter(
            id="husary_mujawwad",
            name_en="Mahmoud Khalil Al-Husary (Mujawwad)",
            name_ar="محمود خليل الحصري - مجود",
            style=Style.MUJAWWAD,
            riwaya="hafs",
            source="everyayah.com",
            licence=LicenceStatus.UNKNOWN,
            licence_note=_UNKNOWN_NOTE,
            url_template="https://everyayah.com/data/Husary_128kbps_Mujawwad/{key}.mp3",
        ),
        Reciter(
            id="alafasy",
            name_en="Mishari Rashid al-Afasy",
            name_ar="مشاري راشد العفاسي",
            style=Style.MURATTAL,
            riwaya="hafs",
            source="everyayah.com",
            licence=LicenceStatus.UNKNOWN,
            licence_note=_UNKNOWN_NOTE,
            url_template="https://everyayah.com/data/Alafasy_128kbps/{key}.mp3",
        ),
        Reciter(
            id="abdulbasit",
            name_en="Abdul Basit Abdul Samad",
            name_ar="عبد الباسط عبد الصمد",
            style=Style.MURATTAL,
            riwaya="hafs",
            source="everyayah.com",
            licence=LicenceStatus.UNKNOWN,
            licence_note=_UNKNOWN_NOTE,
            url_template="https://everyayah.com/data/Abdul_Basit_Murattal_192kbps/{key}.mp3",
        ),
        Reciter(
            id="minshawi",
            name_en="Mohamed Siddiq El-Minshawi",
            name_ar="محمد صديق المنشاوي",
            style=Style.MURATTAL,
            riwaya="hafs",
            source="everyayah.com",
            licence=LicenceStatus.UNKNOWN,
            licence_note=_UNKNOWN_NOTE,
            url_template="https://everyayah.com/data/Minshawy_Murattal_128kbps/{key}.mp3",
        ),
    ]
}

DEFAULT_RECITER = "husary"


def get(reciter_id: str) -> Reciter | None:
    return RECITERS.get(reciter_id)


def all_reciters() -> list[Reciter]:
    return list(RECITERS.values())
