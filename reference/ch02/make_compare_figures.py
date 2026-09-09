"""L3 evidence for chapter 2: side-by-side figure comparisons (Python | MATLAB | book page).

For every book figure that the MATLAB code produces (and the text-only figures the Python scripts reproduce) this
writes ``outputs/ch02/verify/fig_X_Y_compare.png`` (copied to ``reports/ch02/figures/``) with

* top row   — the Python output PNGs from ``outputs/ch02/`` (produced by ``scripts/ch02_*.py --no-show``),
* bottom row — the images MATLAB wrote with ``imwrite``/``print`` while ``reference/ch02/make_refs.py`` ran the
  original ``.m`` files (``outputs/ch02/verify/matlab/``), when they exist,
* right column — the book page rendered from ``chapters/ch02.pdf`` with pymupdf at 110 dpi.

Where both sides are *raw* images of the same size (not matplotlib/MATLAB figure canvases) the pixel difference
between the Python PNG and the MATLAB PNG is measured and written to ``outputs/ch02/verify/image_diffs.json``.

Usage: .venv/Scripts/python.exe reference/ch02/make_compare_figures.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pymupdf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/ch02"
ML = OUT / "verify/matlab"
VERIFY = OUT / "verify"
REPORT_FIG = ROOT / "reports/ch02/figures"
PDF = ROOT / "chapters/ch02.pdf"
FIRST_PRINTED_PAGE = 11  # chapters/ch02.pdf page index 0 = printed page 11


def page_image(printed_page: int, dpi: int = 110) -> np.ndarray:
    doc = pymupdf.open(PDF)
    pix = doc[printed_page - FIRST_PRINTED_PAGE].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img[..., :3]


def _read(p: Path) -> np.ndarray | None:
    """Read a PNG; 1-bit PNGs written by MATLAB ``imwrite(logical)`` come back as 0/1 and are scaled to 0/255."""
    if not p.exists():
        return None
    img = iio.imread(p)
    if img.dtype == np.bool_ or (img.dtype == np.uint8 and img.max() <= 1):
        img = img.astype(np.uint8) * 255
    return img


def compose(name: str, py: list[tuple[str, Path]], ml: list[tuple[str, Path]], page: int, caption: str) -> Path:
    ncol = max(len(py), len(ml), 1)
    nrow = 2 if ml else 1
    fig = plt.figure(figsize=(3.2 * ncol + 4.5, 3.2 * nrow + 0.8))
    gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[1] * ncol + [1.4])
    for r, (row_label, items) in enumerate([("Python (seaice)", py)] + ([("MATLAB R2025a (original .m)", ml)] if ml else [])):
        for c in range(ncol):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")
            if c < len(items):
                title, path = items[c]
                img = _read(path)
                if img is None:
                    ax.text(0.5, 0.5, f"missing:\n{path.name}", ha="center", va="center", fontsize=8)
                else:
                    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0 if img.ndim == 2 else None,
                              vmax=255 if img.ndim == 2 else None, interpolation="nearest")
                ax.set_title(f"{row_label}\n{title}" if c == 0 else title, fontsize=8)
    ax = fig.add_subplot(gs[:, ncol])
    ax.imshow(page_image(page))
    ax.axis("off")
    ax.set_title(f"book p. {page}", fontsize=9)
    fig.suptitle(caption, fontsize=10)
    fig.tight_layout()
    dest = VERIFY / f"{name}_compare.png"
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy(dest, REPORT_FIG / dest.name)
    return dest


def image_diff(py_path: Path, ml_path: Path) -> dict | None:
    a, b = _read(py_path), _read(ml_path)
    if a is None or b is None:
        return None
    if a.ndim == 3 and b.ndim == 2:
        a = a[..., 0] if np.array_equal(a[..., 0], a[..., 1]) else a
    if a.shape != b.shape:
        return {"python": py_path.name, "matlab": ml_path.name, "shape_python": list(a.shape), "shape_matlab": list(b.shape),
                "comparable": False}
    d = np.abs(a.astype(int) - b.astype(int))
    return {"python": py_path.name, "matlab": ml_path.name, "shape": list(a.shape), "comparable": True,
            "max_abs_diff": int(d.max()), "n_diff": int((d > 0).sum()), "frac_diff": float((d > 0).mean())}


def main() -> int:
    VERIFY.mkdir(parents=True, exist_ok=True)
    figs = [
        ("fig_2_03", [("red.png", OUT / "red.png"), ("green.png", OUT / "green.png"), ("blue.png", OUT / "blue.png")],
         [("Ir colormap", ML / "fig_2_03_red.png"), ("Ig colormap", ML / "fig_2_03_green.png"),
          ("Ib colormap", ML / "fig_2_03_blue.png")], 14,
         "Fig. 2.3 - R/G/B channel images of rgb.JPG (histogram.m saveas red/green/blue.png)"),
        ("fig_2_04", [("Ic = imcomplement(Ir)", OUT / "color_image_Ic.png"), ("Im", OUT / "color_image_Im.png"),
                      ("Iy", OUT / "color_image_Iy.png"), ("I_cmy", OUT / "color_image_I_cmy.png")],
         [("Ic (imshow [])", ML / "fig_2_04_Ic.png"), ("Im", ML / "fig_2_04_Im.png"), ("Iy", ML / "fig_2_04_Iy.png"),
          ("I_cmy", ML / "fig_2_04_I_cmy.png")], 14, "Fig. 2.4 - CMY components (color_image.m, Eq. 2.3)"),
        ("fig_2_05", [("Ih (script, 2*Ig bug)", OUT / "color_image_H_matlab.png"), ("Is", OUT / "color_image_S_matlab.png"),
                      ("Ii", OUT / "color_image_I_matlab.png")],
         [("Ih (imshow [])", ML / "fig_2_05_Ih.png"), ("Is", ML / "fig_2_05_Is.png"), ("Ii", ML / "fig_2_05_Ii.png")], 15,
         "Fig. 2.5 - HSI components exactly as color_image.m computes them (buggy V1 -> flat hue)"),
        ("fig_2_05_corrected", [("H, S, I (book Eq. 2.6)", OUT / "fig_2_05_hsi.png")],
         [("Ih corrected (2*Ib)", ML / "fig_2_05_Ih_corrected.png"), ("Is corrected", ML / "fig_2_05_Is_corrected.png"),
          ("Ii", ML / "fig_2_05_Ii.png")], 15, "Fig. 2.5 - HSI with the book's Eq. (2.6a) (V1 uses 2*Ib)"),
        ("fig_2_07", [("rgb2gray(rgb.JPG)", OUT / "gray.png"), ("imhist", OUT / "fig_2_07_gray_hist.png")],
         [("rgb2gray", ML / "fig_2_07_gray.png"), ("imhist", ML / "fig_2_07_imhist.png")], 17,
         "Fig. 2.7 - grayscale image + imhist (SUBSTITUTE image: the book's Fig. 2.7 source image is not shipped)"),
        ("fig_2_08", [("R/G/B histograms", OUT / "fig_2_08_rgb_hist.png")], [("plot(x, y_r, ...)", ML / "fig_2_08_rgb_hist.png")],
         17, "Fig. 2.8 - rgb.JPG channel histograms (histogram.m)"),
        ("fig_2_14", [("point", OUT / "fig_2_14_point_image.png"), ("-bwdist euclidean", OUT / "fig_2_14_point_dt_euclidean.png"),
                      ("cityblock", OUT / "fig_2_14_point_dt_cityblock.png"), ("chessboard", OUT / "fig_2_14_point_dt_chessboard.png")],
         [("img", ML / "fig_2_14_point.png"), ("euclidean", ML / "fig_2_14_euclidean.png"),
          ("cityblock", ML / "fig_2_14_cityblock.png"), ("chessboard", ML / "fig_2_14_chessboard.png")], 23,
         "Fig. 2.14 - distance maps from one pixel (distance_transform.m, imshow(imgDist, []))"),
        ("fig_2_19", [("bim = bound2im(b)", OUT / "chain_diff_bim.png"), ("starting point", OUT / "chain_diff_starting_point.png"),
                      ("Fig 2.19 panels", OUT / "fig_2_19_boundary.png")],
         [("bim", ML / "fig_2_19_bim.png"), ("imshow + text", ML / "fig_2_19_chain_diff_figure.png")], 30,
         "Fig. 2.19/2.20 - boundary of the 8x9 object and the starting point (chain_diff.m)"),
        ("fig_2_22", [("interp2 grid 1:0.4:32", OUT / "sec_2_8_interp2_grid.png")],
         [("nearest", ML / "fig_2_22_nearest.png"), ("linear", ML / "fig_2_22_bilinear.png"), ("cubic", ML / "fig_2_22_bicubic.png")],
         33, "Sec. 2.8 - interp2 nearest / bilinear / bicubic on a 32x32 crop (text-only section; MATLAB interp2 reference)"),
        ("fig_2_23", [("resize x4 nearest", OUT / "interp_resize4_nearest.png"), ("bilinear", OUT / "interp_resize4_bilinear.png"),
                      ("bicubic", OUT / "interp_resize4_bicubic.png")],
         [("imresize nearest", ML / "fig_2_23_resize4_nearest.png"), ("bilinear", ML / "fig_2_23_resize4_bilinear.png"),
          ("bicubic", ML / "fig_2_23_resize4_bicubic.png")], 34,
         "Sec. 2.8 - resize() vs imresize (x4) - labelled approx (border handling)"),
        # text-only figures: Python vs the book page only
        ("fig_2_01_02", [("Fig 2.1 gray values", OUT / "fig_2_01_gray_values.png"), ("Fig 2.2 binary", OUT / "fig_2_02_binary_pattern.png")],
         [], 12, "Figs. 2.1-2.2 - image types (text only)"),
        ("fig_2_06", [("indexed image", OUT / "fig_2_06_indexed_image.png")], [], 16, "Fig. 2.6 - indexed image structure (text only)"),
        ("fig_2_09_10", [("Fig 2.9", OUT / "fig_2_09_neighborhoods.png"), ("Fig 2.10", OUT / "fig_2_10_paths.png")], [], 18,
         "Figs. 2.9-2.10 - neighbourhoods and paths (text only)"),
        ("fig_2_11", [("components", OUT / "fig_2_11_components.png")], [], 20, "Fig. 2.11 - 5 (4-adj) vs 2 (8-adj) components"),
        ("fig_2_12", [("7x7 DT", OUT / "fig_2_12_dt7x7.png")], [], 21, "Fig. 2.12 - Euclidean DT of the 7x7 matrix"),
        ("fig_2_13", [("centre distances", OUT / "fig_2_13_center_distances.png")], [], 22, "Fig. 2.13 - city-block / chessboard"),
        ("fig_2_15", [("convolution", OUT / "fig_2_15_convolution.png")], [], 24, "Fig. 2.15 - 3x3 convolution (Eq. 2.14 vs Eq. 2.15 as printed)"),
        ("fig_2_16", [("set operations", OUT / "fig_2_16_set_operations.png")], [], 26, "Fig. 2.16 - set operations"),
        ("fig_2_17", [("reflection / translation", OUT / "fig_2_17_reflection_translation.png")], [], 27,
         "Fig. 2.17 - reflection and translation"),
        ("fig_2_18", [("direction numbering", OUT / "fig_2_18_direction_numbering.png")], [], 29, "Fig. 2.18 - chain-code directions"),
        ("fig_2_25", [("Keys kernel", OUT / "fig_2_25_keys_kernel.png")], [], 35, "Fig. 2.25 / Eq. 2.41 - bicubic kernel"),
    ]
    written = []
    for name, py, ml, page, caption in figs:
        written.append(compose(name, py, ml, page, caption))
        print("wrote", written[-1].name)

    pairs = [
        (OUT / "color_image_Ic.png", ML / "fig_2_04_Ic.png"), (OUT / "color_image_Im.png", ML / "fig_2_04_Im.png"),
        (OUT / "color_image_Iy.png", ML / "fig_2_04_Iy.png"), (OUT / "color_image_I_cmy.png", ML / "fig_2_04_I_cmy.png"),
        (OUT / "color_image_H_matlab.png", ML / "fig_2_05_Ih.png"), (OUT / "color_image_S_matlab.png", ML / "fig_2_05_Is.png"),
        (OUT / "color_image_I_matlab.png", ML / "fig_2_05_Ii.png"), (OUT / "gray.png", ML / "fig_2_07_gray.png"),
        (OUT / "fig_2_14_point_image.png", ML / "fig_2_14_point.png"),
        (OUT / "fig_2_14_point_dt_euclidean.png", ML / "fig_2_14_euclidean.png"),
        (OUT / "fig_2_14_point_dt_cityblock.png", ML / "fig_2_14_cityblock.png"),
        (OUT / "fig_2_14_point_dt_chessboard.png", ML / "fig_2_14_chessboard.png"),
        (OUT / "chain_diff_bim.png", ML / "fig_2_19_bim.png"),
        (OUT / "interp_resize4_nearest.png", ML / "fig_2_23_resize4_nearest.png"),
        (OUT / "interp_resize4_bilinear.png", ML / "fig_2_23_resize4_bilinear.png"),
        (OUT / "interp_resize4_bicubic.png", ML / "fig_2_23_resize4_bicubic.png"),
    ]
    diffs = [r for r in (image_diff(a, b) for a, b in pairs) if r is not None]
    (VERIFY / "image_diffs.json").write_text(json.dumps(diffs, indent=2))
    for r in diffs:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
