"""L3 evidence for chapter 4: side-by-side figure comparisons (Python | MATLAB | book page).

For every book figure of chapter 4 this writes ``outputs/ch04/verify/fig_X_Y_compare.png`` (copied to
``reports/ch04/figures/``, git-ignored) with

* top row    — the Python output PNGs from ``outputs/ch04/`` (produced by ``scripts/ch04_*.py --no-show``),
* bottom row — the images MATLAB wrote with ``imwrite`` while ``reference/ch04/make_refs.py`` ran the original
  ``derivative.m`` / ``morphology.m`` (``outputs/ch04/verify/matlab/``), when the ``.m`` code produces the figure,
* right column — the book page rendered from ``chapters/ch04.pdf`` with pymupdf at 110 dpi.

Where both sides are *raw* images of the same size the pixel difference between the Python PNG and the MATLAB PNG is
measured and written to ``outputs/ch04/verify/image_diffs.json``.

Usage: .venv/Scripts/python.exe reference/ch04/make_compare_figures.py
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
OUT = ROOT / "outputs/ch04"
ML = OUT / "verify/matlab"
VERIFY = OUT / "verify"
REPORT_FIG = ROOT / "reports/ch04/figures"
PDF = ROOT / "chapters/ch04.pdf"
FIRST_PRINTED_PAGE = 59  # chapters/ch04.pdf page index 0 = printed page 59


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


def _thumb(img: np.ndarray, max_side: int = 900) -> np.ndarray:
    step = int(np.ceil(max(img.shape[:2]) / max_side))
    return img[::step, ::step] if step > 1 else img


def compose(name: str, py: list[tuple[str, Path]], ml: list[tuple[str, Path]], page: int, caption: str) -> Path:
    ncol = max(len(py), len(ml), 1)
    nrow = 2 if ml else 1
    fig = plt.figure(figsize=(3.4 * ncol + 4.5, 3.0 * nrow + 0.9))
    gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[1] * ncol + [1.4])
    rows = [("Python (seaice)", py)] + ([("MATLAB R2025a (original .m)", ml)] if ml else [])
    for r, (row_label, items) in enumerate(rows):
        for c in range(ncol):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")
            if c < len(items):
                title, path = items[c]
                img = _read(path)
                if img is None:
                    ax.text(0.5, 0.5, f"missing:\n{path.name}", ha="center", va="center", fontsize=8)
                else:
                    img = _thumb(img)
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
        return {"python": py_path.name, "matlab": ml_path.name, "shape_python": list(a.shape),
                "shape_matlab": list(b.shape), "comparable": False}
    d = np.abs(a.astype(int) - b.astype(int))
    return {"python": py_path.name, "matlab": ml_path.name, "shape": list(a.shape), "comparable": True,
            "max_abs_diff": int(d.max()), "n_diff": int((d > 0).sum()), "frac_diff": float((d > 0).mean())}


def main() -> int:
    VERIFY.mkdir(parents=True, exist_ok=True)
    figs = [
        ("fig_4_03", [("(a) crop of test.jpg", OUT / "fig_4_03a_crop.png"),
                      ("(b) Sobel T=0.05", OUT / "fig_4_03b_sobel_T0.05.png"),
                      ("(c) Prewitt T=0.05", OUT / "fig_4_03c_prewitt_T0.05.png")],
         [("rgb2gray(im(1600:2151,1979:2552,:))", ML / "crop_gray.png"),
          ("derivative.m on the crop", ML / "crop_bw_sobel.png"),
          ("edge(im,'prewitt',0.05)", ML / "crop_bw_prewitt.png")], 63,
         "Fig. 4.3 - Sobel (derivative.m) and Prewitt edges, T = 0.05, of the Fig. 4.3(a) crop"),
        ("fig_4_06", [("LoG 13x13, sigma 2, T 0.005", OUT / "fig_4_06_log_s2_T0.005.png")],
         [("edge(im,'log',0.005,2)", ML / "crop_bw_log.png")], 66,
         "Fig. 4.6 - LoG zero-crossing edges of Fig. 4.3(a) (text only: MATLAB edge('log'))"),
        ("fig_4_07", [("strel shapes (a)-(d)", OUT / "sec_4_2_fig4_7_strels.png")], [], 67,
         "Fig. 4.7 - structuring elements (text only; (c) = strel('disk',5) 9x9/69 px, (d) = strel('diamond',5))"),
        ("fig_4_08", [("erosion / dilation walk-through", OUT / "sec_4_2_1_fig4_8.png")], [], 70,
         "Fig. 4.8 - erosion and dilation of the printed 3x7 rectangle by the cross SE (text only; matrices asserted)"),
        ("fig_4_09", [("(a) im2bw", OUT / "fig_4_09a_bw.png"), ("(b) erosion r=15", OUT / "fig_4_09b_erosion.png"),
                      ("(c) dilation r=15", OUT / "fig_4_09c_dilation.png")],
         [("I", ML / "crop_I.png"), ("J = imerode(I,SE)", ML / "crop_J.png"), ("K = imdilate(I,SE)", ML / "crop_K.png")], 71,
         "Fig. 4.9 - binary erosion / dilation of Fig. 4.3(a), disk r = 15 (morphology.m on the crop with r = 15)"),
        ("fig_4_10", [("(a) gray erosion", OUT / "fig_4_10a_gray_erosion.png"), ("(b) gray dilation", OUT / "fig_4_10b_gray_dilation.png")],
         [("X = imerode(im,SE)", ML / "crop_X.png"), ("Y = imdilate(im,SE)", ML / "crop_Y.png")], 71,
         "Fig. 4.10 - grayscale erosion / dilation of Fig. 4.3(a), disk r = 15"),
        ("fig_4_15", [("(a) basic", OUT / "fig_4_15a_basic.png"), ("(b) internal", OUT / "fig_4_15b_internal.png"),
                      ("(c) external", OUT / "fig_4_15c_external.png")],
         [("BW = K - J", ML / "crop_BW.png"), ("BW1 = I - J", ML / "crop_BW1.png"), ("BW2 = K - I", ML / "crop_BW2.png")], 79,
         "Fig. 4.15 - binary morphological gradients of Fig. 4.3(a), disk r = 15 (caption '157' = 15)"),
        ("fig_4_16", [("(a) basic", OUT / "fig_4_16a_gray_basic.png"), ("(b) internal", OUT / "fig_4_16b_gray_internal.png"),
                      ("(c) external", OUT / "fig_4_16c_gray_external.png")],
         [("basic = Y - X", ML / "crop_basic.png"), ("internal = im - X", ML / "crop_internal.png"),
          ("external = Y - im", ML / "crop_external.png")], 79,
         "Fig. 4.16 - grayscale morphological gradients of Fig. 4.3(a), disk r = 15"),
        ("sec_4_1_derivative_full", [("derivative.m: BW on test.jpg", OUT / "sec_4_1_1_derivative_sobel_T0.05_full.png")],
         [("figure, imshow(BW)", ML / "derivative_bw.png")], 63,
         "derivative.m as shipped: edge(double(rgb2gray(test.jpg))/256, 'sobel', 0.05) on the full image"),
        ("sec_4_2_morphology_full", [("I", OUT / "sec_4_2_bw_otsu_r7.png"), ("J", OUT / "sec_4_2_erosion_r7.png"),
                                     ("K", OUT / "sec_4_2_dilation_r7.png"), ("BW1", OUT / "sec_4_2_internal_r7.png"),
                                     ("BW2", OUT / "sec_4_2_external_r7.png"), ("BW", OUT / "sec_4_2_basic_r7.png")],
         [("I", ML / "morphology_I.png"), ("J", ML / "morphology_J.png"), ("K", ML / "morphology_K.png"),
          ("BW1", ML / "morphology_BW1.png"), ("BW2", ML / "morphology_BW2.png"), ("BW", ML / "morphology_BW.png")], 79,
         "morphology.m as shipped (binary block): strel('dis',7) on the full test.jpg"),
        ("sec_4_2_morphology_full_gray", [("X", OUT / "sec_4_2_gray_erosion_r7.png"), ("Y", OUT / "sec_4_2_gray_dilation_r7.png"),
                                          ("internal", OUT / "sec_4_2_gray_internal_r7.png"),
                                          ("external", OUT / "sec_4_2_gray_external_r7.png"),
                                          ("basic", OUT / "sec_4_2_gray_basic_r7.png")],
         [("X", ML / "morphology_X.png"), ("Y", ML / "morphology_Y.png"), ("internal", ML / "morphology_internal.png"),
          ("external", ML / "morphology_external.png"), ("basic", ML / "morphology_basic.png")], 79,
         "morphology.m as shipped (grayscale block): strel('dis',7) on the full test.jpg"),
        # text-only 1-D sketches and the unshipped-image experiments
        ("fig_4_11_4_12", [("closing / opening 1-D", OUT / "sec_4_2_2_open_close_1d.png")], [], 72,
         "Figs. 4.11-4.12 - 1-D closing / opening sketches (text only; synthetic profile)"),
        ("fig_4_13_4_14", [("reconstruction 1-D", OUT / "sec_4_2_3_reconstruction_1d.png")], [], 75,
         "Figs. 4.13-4.14 - 1-D reconstruction by dilation / erosion (text only; synthetic profile)"),
        ("fig_4_17", [("Fig. 4.17-style on test.jpg", OUT / "sec_4_3_fig4_17_full_panels.png")], [], 80,
         "Fig. 4.17 - UNSHIPPED image: Sobel T = 0.05 vs internal gradient r = 5 reproduced on test.jpg (unverified)"),
        ("fig_4_18", [("Fig. 4.18-style crop of test.jpg", OUT / "sec_4_3_fig4_18_crop_panels.png")], [], 81,
         "Fig. 4.18 - UNSHIPPED image: connected floes, Sobel vs internal gradient on a test.jpg crop (unverified)"),
        ("fig_4_19", [("T = 0.05 vs 0.03", OUT / "sec_4_3_fig4_19_sobel_T0.05_vs_T0.03.png")], [], 81,
         "Fig. 4.19 - UNSHIPPED image: Sobel threshold 0.05 vs 0.03 on the test.jpg crop (unverified)"),
        ("fig_4_20", [("r = 5 vs 16", OUT / "sec_4_3_fig4_20_internal_r5_vs_r16.png")], [], 82,
         "Fig. 4.20 - UNSHIPPED image: internal gradient r = 5 vs r = 16 on the test.jpg crop (unverified)"),
    ]
    diffs = []
    for name, py, ml, page, caption in figs:
        dest = compose(name, py, ml, page, caption)
        print("wrote", dest)
        for (t1, p1), (t2, p2) in zip(py, ml):
            d = image_diff(p1, p2)
            if d is not None:
                d["figure"] = name
                diffs.append(d)
    (VERIFY / "image_diffs.json").write_text(json.dumps(diffs, indent=2))
    for d in diffs:
        print(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
