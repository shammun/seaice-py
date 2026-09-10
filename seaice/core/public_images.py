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
_NASA_WORLDVIEW_81 = (
    "https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot"
    "&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&CRS=EPSG:4326&FORMAT=image/jpeg"
    "&WIDTH=81&HEIGHT=96&WRAP=DAY"
)
_NASA_WORLDVIEW_394 = (
    "https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot"
    "&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&CRS=EPSG:4326&FORMAT=image/jpeg"
    "&WIDTH=394&HEIGHT=1038&WRAP=DAY"
)
_NASA_WORLDVIEW_148 = (
    "https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot"
    "&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&CRS=EPSG:4326&FORMAT=image/jpeg"
    "&WIDTH=148&HEIGHT=108&WRAP=DAY"
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
    # --- ch04: the book's 4290×2856 test.jpg (a different image from ch03's test.jpg; Fig 4.3(a) is its crop
    # rows 1600–2151 / cols 1979–2552 in MATLAB terms, so the substitute must have floe structure in that window).
    # Same scene as the ch03 dense substitute: the crop window holds several floes separated by leads (verified
    # 2026-09-09: 72 % bright pixels, std 55 in the window).
    ("ch04", "test.jpg"): {
        "url": _NASA_WORLDVIEW_4290 + "&TIME=2019-07-25&BBOX=75.0,-150.0,77.0,-147.0",
        "filename": "nasa_modis_terra_beaufort_dense_2019-07-25.jpg",
        "credit": "NASA Worldview Snapshots, MODIS/Terra corrected reflectance (true colour), Beaufort Sea pack ice, "
                  "25 July 2019, 75–77°N 150–147°W",
        "licence": _NASA_PD,
        "description": "Dense floe field with leads; the Fig 4.3(a) crop window (rows 1600–2151, cols 1979–2552) holds "
                       "several floes separated by leads, standing in for the book's two-floe crop",
    },
    # --- ch05: the book's 81×96 RGB toy image q.jpg (two touching floes on dark water, Figs 5.4–5.14).  The substitute
    # is a MODIS scene requested at the same 81×96 size over a ~3.7 km × 15 km window (upsampled from the 250 m native
    # resolution) so that a few large floes, touching at the bottom of the frame, fill the frame like the book's toy
    # (verified 2026-09-09: 46 % bright pixels, gray std 71; 8 other windows/zoom levels rejected as too fragmented).
    ("ch05", "q.jpg"): {
        "url": _NASA_WORLDVIEW_81 + "&TIME=2019-07-25&BBOX=76.6898,-148.9137,76.8302,-148.4262",
        "filename": "nasa_modis_terra_beaufort_floes_81x96_2019-07-25.jpg",
        "credit": "NASA Worldview Snapshots, MODIS/Terra corrected reflectance (true colour), Beaufort Sea pack ice, "
                  "25 July 2019, 76.69–76.83°N 148.91–148.43°W",
        "licence": _NASA_PD,
        "description": "A few large floes on dark water, touching at the bottom of the frame; 81×96 RGB like the book's q.jpg",
    },
    # --- ch06: the book ships two images in matlab/ch6 and both are read by its scripts.
    # `sea_ice_test.jpg` (394×1038 RGB, the tall shipborne view used by sea_ice_demo.m and for test/dist.m) and
    # `test8.jpg` (148×108 RGB, the small GVF-snake demo image of for test/for_test.m, whose initial contour is a
    # circle of radius 20 centred on (x0, y0) = (80, 40), so the substitute must carry a floe boundary there).
    # `alg_seg_gray.jpg` also ships in `for test/` but no .m file reads it, so it gets no substitute.
    ("ch06", "sea_ice_test.jpg"): {
        "url": _NASA_WORLDVIEW_394 + "&TIME=2019-07-25&BBOX=73.60,-151.20,75.60,-150.4409",
        "filename": "nasa_modis_terra_beaufort_floes_394x1038_2019-07-25.jpg",
        "credit": "NASA Worldview Snapshots, MODIS/Terra corrected reflectance (true colour), Beaufort Sea pack ice, "
                  "25 July 2019, 73.60–75.60°N 151.20–150.44°W",
        "licence": _NASA_PD,
        "description": "Distinct ice floes separated by leads over open water, in a tall strip; "
                       "394×1038 RGB like the book's sea_ice_test.jpg",
    },
    ("ch06", "test8.jpg"): {
        "url": _NASA_WORLDVIEW_148 + "&TIME=2019-07-25&BBOX=74.60,-151.00,74.95,-150.5204",
        "filename": "nasa_modis_terra_beaufort_floe_148x108_2019-07-25.jpg",
        "credit": "NASA Worldview Snapshots, MODIS/Terra corrected reflectance (true colour), Beaufort Sea pack ice, "
                  "25 July 2019, 74.60–74.95°N 151.00–150.52°W",
        "licence": _NASA_PD,
        "description": "Individual floes on dark water with a floe boundary crossing the snake's initial circle "
                       "at (80, 40); 148×108 RGB like the book's test8.jpg",
    },
}


def lookup(chapter: str, name: str) -> dict[str, str] | None:
    """Registry entry for a book image (file name matched case-insensitively), or ``None`` if none is registered."""
    return REGISTRY.get((chapter, name.lower()))


def registered() -> list[tuple[str, str]]:
    """All ``(chapter, name)`` keys with a public-domain substitute."""
    return sorted(REGISTRY)
