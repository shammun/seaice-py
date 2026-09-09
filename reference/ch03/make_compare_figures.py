"""L3 evidence for chapter 3: side-by-side figure comparisons (Python | MATLAB | book page).

For every book figure that the MATLAB code produces (and the text-only figures the Python scripts reproduce) this
writes ``outputs/ch03/verify/fig_X_Y_compare.png`` (copied to ``reports/ch03/figures/``, git-ignored) with

* top row    — the Python output PNGs from ``outputs/ch03/`` (produced by ``scripts/ch03_*.py --no-show``),
* bottom row — the images MATLAB wrote with ``imwrite``/``print`` while ``reference/ch03/make_refs.py`` ran the
  original ``.m`` files (``outputs/ch03/verify/matlab/``), when they exist,
* right column — the book page rendered from ``chapters/ch03.pdf`` with pymupdf at 110 dpi.

Where both sides are *raw* images of the same size the pixel difference between the Python PNG and the MATLAB PNG is
measured and written to ``outputs/ch03/verify/image_diffs.json``.

Usage: .venv/Scripts/python.exe reference/ch03/make_compare_figures.py
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
OUT = ROOT / "outputs/ch03"
ML = OUT / "verify/matlab"
VERIFY = OUT / "verify"
REPORT_FIG = ROOT / "reports/ch03/figures"
PDF = ROOT / "chapters/ch03.pdf"
FIRST_PRINTED_PAGE = 37  # chapters/ch03.pdf page index 0 = printed page 37


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
    """Subsample very large rasters (4290x2856) for the composite so the PNG stays small."""
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
    figs = []
    for key, n3, fig, page in (("1", "1", "3_09", 53), ("2", "2", "3_10", 54), ("test", "3", "3_11", 55)):
        figs.append((f"fig_{fig}", [(f"Otsu bw ({key}.jpg)", OUT / f"otsu_bw_{key}.png"),
                                    ("k-means k=2 (kmeans.m mask1)", OUT / f"kmeans2_mask_{key}_book.png")],
                     [("im2bw(I, graythresh(I))", ML / f"otsu_bw_{key}.png"),
                      ("imshow(mask1) (gray)", ML / f"kmeans2_mask_{key}.png"),
                      ("colormap('default') as displayed", ML / f"kmeans2_mask_{key}_parula.png")], page,
                     f"Fig. {fig.replace('_', '.')} - sea ice image {n3}: (b) Otsu (Otsu.m), (c) k-means k=2 (kmeans.m)"))
    figs += [
        ("fig_3_12", [("multi-Otsu N=2 (imshow(seg,[]))", OUT / "multi_otsu_seg_test.png"),
                      ("k-means k=3 (mask1)", OUT / "kmeans3_mask_test_book.png")],
         [("imquantize seg", ML / "multi_otsu_seg_test.png"), ("mask1 (gray)", ML / "kmeans3_mask_test.png"),
          ("colormap('default')", ML / "kmeans3_mask_test_parula.png")], 57,
         "Fig. 3.12 - image 3: (a) multi Otsu with 2 thresholds (Otsu.m), (b) k-means with 3 clusters (kmeans.m)"),
        ("fig_3_03", [("I > 108 (substitute 2.jpg)", OUT / "separability_bw_k108.png")],
         [("separability.m bw", ML / "separability_bw_k108.png")], 43,
         "Fig. 3.3 - separability.m with k = 108 on the SUBSTITUTE 2.jpg (ch3ice.jpg is not shipped)"),
        ("fig_3_04", [("(a) uneven input (substitute)", OUT / "local_otsu_input_gray.png"),
                      ("(b) global Otsu", OUT / "fig_3_04ab_uneven_global_otsu.png"),
                      ("(c) block bw", OUT / "local_otsu_bw.png"),
                      ("(c) 2x3 subplot", OUT / "fig_3_04c_local_otsu_blocks.png")],
         [("rgb2gray(imread('t.jpg'))", ML / "local_otsu_input_gray.png"),
          ("im2bw(I, graythresh(I))", ML / "local_otsu_global_bw.png"),
          ("block-wise im2bw", ML / "local_otsu_bw.png"),
          ("local_Otsu.m figure", ML / "fig_3_04c_local_otsu_blocks.png")], 45,
         "Fig. 3.4 - local_Otsu.m on the SUBSTITUTE (2.jpg + synthetic illumination ramp; t.jpg is not shipped)"),
        ("fig_3_05", [("multi-Otsu N=2 on 2.jpg", OUT / "multi_otsu_seg_2.png")],
         [("imquantize seg", ML / "multi_otsu_seg_2.png")], 46,
         "Fig. 3.5 - 3-level Otsu (Otsu.m) on the SUBSTITUTE 2.jpg (thresholds 61/142 coincide with the book)"),
        ("fig_3_07", [("k=2 on 2.jpg", OUT / "kmeans2_mask_2_book.png"), ("k=3 on 2.jpg", OUT / "kmeans3_mask_2_book.png")],
         [("kmeans.m k=2", ML / "kmeans2_mask_2.png"), ("kmeans.m k=3", ML / "kmeans3_mask_2.png")], 51,
         "Fig. 3.7 - kmeans.m with k = 2 / 3 on the SUBSTITUTE 2.jpg (Fig. 3.2(a) is not shipped)"),
        # text-only figures: Python vs the book page only
        ("fig_3_01", [("bimodal sketch", OUT / "fig_3_01_bimodal_sketch.png")], [], 38, "Fig. 3.1 - bimodal histogram (text only)"),
        ("fig_3_02", [("T = 125 on 2.jpg", OUT / "fig_3_02_global_threshold.png")], [], 39,
         "Fig. 3.2 - hand-picked global threshold (text only; SUBSTITUTE image)"),
        ("fig_3_06", [("k-means process", OUT / "fig_3_06_kmeans_2d_process.png")], [], 50,
         "Fig. 3.6 - 2-D k-means process (text only, synthetic points)"),
        ("fig_3_08", [("outlier", OUT / "fig_3_08_kmeans_outlier.png")], [], 52,
         "Fig. 3.8 - effect of an outlier (text only, synthetic points)"),
    ]
    for name, py, ml, page, caption in figs:
        print("wrote", compose(name, py, ml, page, caption).name)

    pairs = [(OUT / f"otsu_bw_{k}.png", ML / f"otsu_bw_{k}.png") for k in ("1", "2", "test")]
    pairs += [(OUT / f"multi_otsu_seg_{k}.png", ML / f"multi_otsu_seg_{k}.png") for k in ("1", "2", "test")]
    pairs += [(OUT / f"kmeans{n}_mask_{k}_book.png", ML / f"kmeans{n}_mask_{k}.png")
              for k in ("1", "2", "test") for n in (2, 3)]
    pairs += [(OUT / "local_otsu_bw.png", ML / "local_otsu_bw.png"),
              (OUT / "local_otsu_input_gray.png", ML / "local_otsu_input_gray.png"),
              (OUT / "separability_bw_k108.png", ML / "separability_bw_k108.png")]
    diffs = [r for r in (image_diff(a, b) for a, b in pairs) if r is not None]
    (VERIFY / "image_diffs.json").write_text(json.dumps(diffs, indent=2))
    for r in diffs:
        print(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
