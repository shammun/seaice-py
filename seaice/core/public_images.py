"""Registry of public-domain substitutes for the book's (copyrighted, non-redistributable) images.

One entry per ``(chapter, book file name)``.  :func:`seaice.core.io.load_image` falls back to the registered
substitute when the reader has no private copy of the book image, downloading it once into
``data/online/<chapter>/<filename>``.  Provenance (URL, credit, licence) is also recorded in
``data/online/SOURCES.md`` — the only file under ``data/online/`` that is committed.

Rules for new entries (CLAUDE.md rule 12, data-sources skill):

* public-domain or equivalently free imagery only (NASA imagery is not copyrighted; credit it anyway);
* verify the URL with a real request before registering it and note the date in ``SOURCES.md``;
* keep the substitute's size and colour type compatible with what the chapter's notebook does with the book
  image (ch02 crops rows 201–211 / 700–731 and columns 901–931 and reads pixel (1076, 675), so the substitute is
  requested at the book image's own 2048×1536 RGB size).
"""
from __future__ import annotations

_NASA_WORLDVIEW = (
    "https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot"
    "&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&CRS=EPSG:4326&FORMAT=image/jpeg"
    "&WIDTH=2048&HEIGHT=1536&WRAP=DAY"
)

#: ``{(chapter, book file name lower-cased): {"url", "filename", "credit", "licence", "description"}}``
REGISTRY: dict[tuple[str, str], dict[str, str]] = {
    ("ch02", "rgb.jpg"): {
        "url": _NASA_WORLDVIEW + "&TIME=2019-07-25&BBOX=71.5,-142.0,74.0,-132.0",
        "filename": "nasa_modis_terra_beaufort_miz_2019-07-25.jpg",
        "credit": "NASA Worldview Snapshots, MODIS/Terra corrected reflectance (true colour), Beaufort Sea marginal "
                  "ice zone, 25 July 2019, 71.5–74°N 142–132°W",
        "licence": "NASA imagery is in the public domain (https://www.earthdata.nasa.gov/engage/open-data-services-and-software/data-and-information-policy)",
        "description": "Ice floes and open water in the Beaufort Sea marginal ice zone; 2048×1536 RGB like the book's rgb.JPG",
    },
}


def lookup(chapter: str, name: str) -> dict[str, str] | None:
    """Registry entry for a book image (file name matched case-insensitively), or ``None`` if none is registered."""
    return REGISTRY.get((chapter, name.lower()))


def registered() -> list[tuple[str, str]]:
    """All ``(chapter, name)`` keys with a public-domain substitute."""
    return sorted(REGISTRY)
