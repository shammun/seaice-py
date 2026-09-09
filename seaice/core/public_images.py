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

_NASA_WORLDVIEW_4290 = (
    "https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot"
    "&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&CRS=EPSG:4326&FORMAT=image/jpeg"
    "&WIDTH=4290&HEIGHT=2856&WRAP=DAY"
)
_NASA_PD = "NASA imagery is in the public domain (https://www.earthdata.nasa.gov/engage/open-data-services-and-software/data-and-information-policy)"

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
    # --- ch03: the book's three 4290×2856 RGB images (Figs 3.9(a), 3.10(a), 3.11(a)); substitutes requested at the
    # same size so block partitions (2 rows × 3 columns, local Otsu) and every pixel index in the notebook stay valid.
    # Ice cover chosen to mirror the book's ~15 % / ~32 % / ~73 % ice concentrations (verified 2026-09-09).
    ("ch03", "1.jpg"): {
        "url": _NASA_WORLDVIEW_4290 + "&TIME=2019-07-25&BBOX=71.0,-140.0,73.0,-137.0",
        "filename": "nasa_modis_terra_beaufort_sparse_2019-07-25.jpg",
        "credit": "NASA Worldview Snapshots, MODIS/Terra corrected reflectance (true colour), Beaufort Sea marginal "
                  "ice zone, 25 July 2019, 71–73°N 140–137°W",
        "licence": _NASA_PD,
        "description": "Sparse brash ice and small floes in open water (~16 % bright pixels), like the book's low-concentration image 1",
    },
    ("ch03", "2.jpg"): {
        "url": _NASA_WORLDVIEW_4290 + "&TIME=2019-07-25&BBOX=73.5,-148.0,75.5,-145.0",
        "filename": "nasa_modis_terra_beaufort_iceedge_2019-07-25.jpg",
        "credit": "NASA Worldview Snapshots, MODIS/Terra corrected reflectance (true colour), Beaufort Sea ice edge, "
                  "25 July 2019, 73.5–75.5°N 148–145°W",
        "licence": _NASA_PD,
        "description": "Pack-ice edge: floe field above, open water below (~30 % bright pixels), like the book's image 2 and the "
                       "unshipped ch3ice.jpg / t.jpg (Figs 3.2–3.5) that image 2 stands in for",
    },
    ("ch03", "test.jpg"): {
        "url": _NASA_WORLDVIEW_4290 + "&TIME=2019-07-25&BBOX=75.0,-150.0,77.0,-147.0",
        "filename": "nasa_modis_terra_beaufort_dense_2019-07-25.jpg",
        "credit": "NASA Worldview Snapshots, MODIS/Terra corrected reflectance (true colour), Beaufort Sea pack ice, "
                  "25 July 2019, 75–77°N 150–147°W",
        "licence": _NASA_PD,
        "description": "Dense floe field with leads (~70 % bright pixels), like the book's high-concentration image 3 (test.jpg)",
    },
}


def lookup(chapter: str, name: str) -> dict[str, str] | None:
    """Registry entry for a book image (file name matched case-insensitively), or ``None`` if none is registered."""
    return REGISTRY.get((chapter, name.lower()))


def registered() -> list[tuple[str, str]]:
    """All ``(chapter, name)`` keys with a public-domain substitute."""
    return sorted(REGISTRY)
