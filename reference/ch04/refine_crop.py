"""Refine the Fig. 4.3(a) crop rectangle of ``test.jpg`` against the bitmap embedded in ``chapters/ch04.pdf``
(analysis/ch04.md risk 4: the template match that gave MATLAB ``im(1600:2151, 1979:2552)`` may be off by 1–2 px).

The PDF stores Fig. 4.3(a) as a 131×126 JPEG thumbnail (xref 142 on printed page 63), i.e. downsampled by ≈ 4.38.
For every candidate offset ``(dr, dc)`` in ``[-R, R]²`` the crop of the MATLAB-gray ``test.jpg`` is area-averaged to
131×126 and its normalised cross-correlation with the (inverted) thumbnail is measured; the best offset, its NCC and
the NCC at the analyst's rectangle are written to ``outputs/ch04/verify/crop_refine.json``.  A scale search
(crop size ±3 %) is included so that a wrong size would show up as a better NCC at a different size.

Usage: .venv/Scripts/python.exe reference/ch04/refine_crop.py [--radius 8]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from seaice.ch04_ice_edge_detection import FIG_4_3A_CROP  # noqa: E402
from seaice.core.io import read_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402

PDF = ROOT / "chapters/ch04.pdf"
FIRST_PRINTED = 59
OUT = ROOT / "outputs/ch04/verify"


def thumbnail_fig_4_3a() -> np.ndarray:
    """The embedded Fig. 4.3(a) bitmap (left panel of printed p. 63), as float gray, un-inverted."""
    doc = pymupdf.open(PDF)
    page = doc[63 - FIRST_PRINTED]
    best = None
    for im in page.get_images(full=True):
        xref = im[0]
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        r = rects[0]
        if best is None or r.x0 < best[1].x0:
            best = (xref, r)
    info = doc.extract_image(best[0])
    img = np.asarray(Image.open(io.BytesIO(info["image"])).convert("L"), dtype=np.float64)
    return img


def area_average(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize((shape[1], shape[0]), Image.BOX), dtype=np.float64)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=int, default=8)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    gray = rgb2gray_matlab(read_image(ROOT / "data/book/ch04/test.jpg")).astype(np.float64)
    thumb = thumbnail_fig_4_3a()
    th, tw = thumb.shape
    r0, r1 = FIG_4_3A_CROP[0].start, FIG_4_3A_CROP[0].stop
    c0, c1 = FIG_4_3A_CROP[1].start, FIG_4_3A_CROP[1].stop
    H, W = r1 - r0, c1 - c0

    # polarity: the analysis says the PDF bitmap is stored inverted; decide by NCC sign at the nominal rectangle
    nominal = area_average(gray[r0:r1, c0:c1], (th, tw))
    s_plain, s_inv = ncc(nominal, thumb), ncc(nominal, 255 - thumb)
    ref = thumb if s_plain >= s_inv else 255 - thumb
    inverted = s_inv > s_plain

    R = args.radius
    grid = np.zeros((2 * R + 1, 2 * R + 1))
    for i, dr in enumerate(range(-R, R + 1)):
        for j, dc in enumerate(range(-R, R + 1)):
            cand = gray[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            grid[i, j] = ncc(area_average(cand, (th, tw)), ref)
    bi, bj = np.unravel_index(int(np.argmax(grid)), grid.shape)
    best = {"dr": int(bi - R), "dc": int(bj - R), "ncc": float(grid[bi, bj])}
    within2 = grid[R - 2:R + 3, R - 2:R + 3]
    wi, wj = np.unravel_index(int(np.argmax(within2)), within2.shape)
    best2 = {"dr": int(wi - 2), "dc": int(wj - 2), "ncc": float(within2[wi, wj])}

    # scale search: same top-left, size scaled by s
    scales = {}
    for s in (0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03):
        h, w = int(round(H * s)), int(round(W * s))
        cand = gray[r0:r0 + h, c0:c0 + w]
        scales[f"{s:.2f}"] = ncc(area_average(cand, (th, tw)), ref)

    res = {
        "thumbnail_size": [th, tw], "thumbnail_inverted_in_pdf": bool(inverted),
        "nominal_crop_0based": {"rows": [r0, r1], "cols": [c0, c1]},
        "nominal_crop_matlab": {"rows": [r0 + 1, r1], "cols": [c0 + 1, c1]},
        "ncc_nominal": float(grid[R, R]),
        "best_within_pm2": best2, "best_within_radius": best, "search_radius": R,
        "ncc_grid_pm2": within2.round(5).tolist(),
        "scale_search_ncc": scales,
    }
    (OUT / "crop_refine.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
