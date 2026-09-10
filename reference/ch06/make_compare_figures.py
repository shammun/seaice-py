"""L3 evidence for chapter 6: side-by-side figure comparisons (Python | MATLAB | book page).

Writes ``outputs/ch06/verify/<name>_compare.png`` (copied to ``reports/ch06/figures/``, git-ignored) with

* top row     -- the Python output PNGs from ``outputs/ch06/`` (``scripts/ch06_*.py --no-show``),
* bottom row  -- the images MATLAB wrote with ``imwrite`` while ``reference/ch06/make_refs.py`` ran the original
  ``.m`` code (``outputs/ch06/verify/matlab/``),
* right column -- the book page rendered from ``chapters/ch06.pdf`` with pymupdf.

Raw image pairs of the same size are differenced pixel-wise into ``outputs/ch06/verify/image_diffs.json``.

Usage: .venv/Scripts/python.exe reference/ch06/make_compare_figures.py
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
OUT = ROOT / "outputs/ch06"
VERIFY = OUT / "verify"
ML = VERIFY / "matlab"
REPORT_FIG = ROOT / "reports/ch06/figures"
PDF = ROOT / "chapters/ch06.pdf"
FIRST_PRINTED_PAGE = 109  # chapters/ch06.pdf page index 0 = printed page 109


def page_image(printed_page: int, dpi: int = 110) -> np.ndarray:
    doc = pymupdf.open(PDF)
    pix = doc[printed_page - FIRST_PRINTED_PAGE].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img[..., :3]


def _read(p: Path) -> np.ndarray | None:
    if not p.exists():
        return None
    img = iio.imread(p)
    if img.dtype == np.bool_ or (img.dtype == np.uint8 and img.max() <= 1):
        img = img.astype(np.uint8) * 255
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    return img


def _thumb(img: np.ndarray, max_side: int = 900) -> np.ndarray:
    step = int(np.ceil(max(img.shape[:2]) / max_side))
    return img[::step, ::step] if step > 1 else img


def compose(name: str, py, ml, page: int | None, caption: str) -> Path:
    ncol = max(len(py), len(ml), 1)
    nrow = 2 if ml else 1
    extra = 1 if page else 0
    fig = plt.figure(figsize=(3.2 * ncol + 4.5 * extra, 3.2 * nrow + 0.9))
    gs = fig.add_gridspec(nrow, ncol + extra, width_ratios=[1] * ncol + ([1.4] if extra else []))
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
                    ax.imshow(img, cmap="gray" if img.ndim == 2 else None,
                              vmin=0 if img.ndim == 2 else None, vmax=255 if img.ndim == 2 else None,
                              interpolation="nearest")
                ax.set_title(f"{row_label}\n{title}" if c == 0 else title, fontsize=8)
    if page:
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


def image_diff(py_path: Path, ml_path: Path) -> dict:
    a, b = _read(py_path), _read(ml_path)
    if a is None or b is None:
        return {"python": py_path.name, "matlab": ml_path.name, "comparable": False, "missing": True}
    if a.ndim == 3 and b.ndim == 2:
        a = a[..., 0]
    if b.ndim == 3 and a.ndim == 2:
        b = b[..., 0]
    if a.shape != b.shape:
        return {"python": py_path.name, "matlab": ml_path.name, "shape_python": list(a.shape),
                "shape_matlab": list(b.shape), "comparable": False}
    d = np.abs(a.astype(int) - b.astype(int))
    return {"python": py_path.name, "matlab": ml_path.name, "shape": list(a.shape), "comparable": True,
            "max_abs_diff": int(d.max()), "n_diff": int((d > 0).sum()), "frac_diff": float((d > 0).mean())}


FIGS = [
    ("fig_6_14", [("(a)/(b) 8x8 blob, city-block DT, seed '+', initial circle",
                   OUT / "fig_6_14_contour_initialization.png")], [], 134,
     "Fig. 6.14 - contour initialization from the distance transform (printed 8x8 matrices)"),
    ("fig_6_15", [("(a) grayscale", OUT / "fig_6_15_a_input.png"),
                  ("(b) binary", OUT / "fig_6_15_b_binary.png"),
                  ("(c) distance transform", OUT / "fig_6_15_c_distance_transform.png"),
                  ("(d) regional maxima", OUT / "fig_6_15_d_regional_maxima.png"),
                  ("(e) seeds + circles", OUT / "fig_6_15_e_seeds_and_circles.png"),
                  ("(f) segmentation", OUT / "fig_6_15_f_segmentation.png")],
     [("rgb2gray(I)", ML / "alg_gray.png"), ("bw", ML / "alg_bw.png"),
      ("imshow(img_Dist, [])", ML / "alg_dist.png"), ("Dis_img", ML / "alg_maxima.png"),
      ("dis (dilated maxima)", ML / "alg_seeds.png"), ("bw1", ML / "alg_bw1.png")], 136,
     "Fig. 6.15 - Algorithm 1 on alg_seg_gray.jpg (GVF_distance.m, sea_ice_demo parameters)"),
    ("fig_6_16", [("(a)-(e) 110x186 circles, GVF via 5/30/100/250 iterations",
                   OUT / "fig_6_16_panels.png")], [], 139,
     "Fig. 6.16 - GVF capture range vs the number of iterations (synthetic, printed specification)"),
    ("fig_6_17", [("snake on the 61-px circle: 30 vs 250 GVF iterations",
                   OUT / "fig_6_17_snake_capture_range.png")], [], 140,
     "Fig. 6.17 - snake convergence under two capture ranges"),
    ("sec_6_3_3_dist", [("local maxima", OUT / "sec_6_3_3_dist_a_local_maxima.png"),
                        ("dilated maxima (seeds)", OUT / "sec_6_3_3_dist_b_dilated_maxima.png"),
                        ("imregionalmin(-D)", OUT / "sec_6_3_3_dist_c_regional_minima.png"),
                        ("city-block DT", OUT / "sec_6_3_3_dist_d_distance_transform.png"),
                        ("seeds + r = D/sqrt(2)", OUT / "sec_6_3_3_dist_g_dt_radius.png")],
     [("dis", ML / "dist_dis.png"), ("dis1", ML / "dist_dis1.png"),
      ("Dis_img", ML / "dist_Dis_img.png"), ("imshow(img_Dist, [])", ML / "dist_img_Dist.png"),
      ("bw", ML / "dist_bw.png")], None,
     "for test/dist.m on sea_ice_test.jpg (transposed frame; no book figure)"),
    ("sec_6_2_for_test", [("GVF quiver (64x64)", OUT / "sec_6_2_for_test_a_gvf_quiver.png"),
                          ("snake evolution", OUT / "sec_6_1_2_for_test_b_snake_evolution.png"),
                          ("binary", OUT / "sec_6_1_2_for_test_c_binary.png"),
                          ("binary + boundary", OUT / "sec_6_1_2_for_test_d_binary_with_boundary.png")],
     [("quiver figure", ML / "ft_fig1.png"), ("snake figure", ML / "ft_fig2.png"),
      ("bw", ML / "ft_bw.png"), ("bw after the burn", ML / "ft_bw_burn.png")], None,
     "for test/for_test.m on test8.jpg: GVF field and one manual contour (no book figure)"),
    ("sec_6_4_sea_ice_demo", [("Otsu mask", OUT / "sec_6_4_demo_b_otsu_mask.png"),
                              ("k-means mask bk", OUT / "sec_6_4_demo_c_kmeans_mask.png"),
                              ("bw0 residual", OUT / "sec_6_4_demo_d_kmeans_residual.png"),
                              ("three-level out (pass1 + 0.5*pass2)", OUT / "sec_6_4_demo_g_three_level.png")],
     [("bw", ML / "sit_bw.png"), ("bk", ML / "sit_bk.png"), ("bw0", ML / "sit_bw0.png"),
      ("bw2 (failing components)", ML / "sit_bw2.png")], None,
     "sea_ice_demo.m stages on sea_ice_test.jpg (MATLAB row = the stage reference, no snake loop)"),
]

RAW_PAIRS = [
    ("fig_6_15_a_input.png", "alg_gray.png"),
    ("fig_6_15_b_binary.png", "alg_bw.png"),
    ("fig_6_15_c_distance_transform.png", "alg_dist.png"),
    ("fig_6_15_d_regional_maxima.png", "alg_maxima.png"),
    ("fig_6_15_f_segmentation.png", "alg_bw1.png"),
    ("sec_6_3_3_dist_a_local_maxima.png", "dist_dis.png"),
    ("sec_6_3_3_dist_b_dilated_maxima.png", "dist_dis1.png"),
    ("sec_6_3_3_dist_c_regional_minima.png", "dist_Dis_img.png"),
    ("sec_6_3_3_dist_d_distance_transform.png", "dist_img_Dist.png"),
    ("sec_6_1_2_for_test_c_binary.png", "ft_bw.png"),
    ("sec_6_1_2_for_test_d_binary_with_boundary.png", "ft_bw_burn.png"),
    ("sec_6_2_for_test_e_edge_map.png", "ft_f2.png"),
    ("sec_6_4_demo_b_otsu_mask.png", "sit_bw.png"),
    ("sec_6_4_demo_c_kmeans_mask.png", "sit_bk.png"),
    ("sec_6_4_demo_d_kmeans_residual.png", "sit_bw0.png"),
]


def main() -> int:
    VERIFY.mkdir(parents=True, exist_ok=True)
    made = []
    for name, py, ml, page, caption in FIGS:
        try:
            made.append(str(compose(name, py, ml, page, caption)))
        except Exception as exc:  # keep going
            print(f"  ! {name}: {exc}")
    diffs = [image_diff(OUT / a, ML / b) for a, b in RAW_PAIRS]
    (VERIFY / "image_diffs.json").write_text(json.dumps(diffs, indent=2))
    ok = [d for d in diffs if d.get("comparable")]
    ident = [d for d in ok if d["n_diff"] == 0]
    print(f"{len(made)} compare figures -> {REPORT_FIG}")
    print(f"raw image pairs: {len(ok)} comparable, {len(ident)} identical")
    for d in ok:
        if d["n_diff"]:
            print(f"  {d['python']} vs {d['matlab']}: {d['n_diff']} px differ, max {d['max_abs_diff']}")
    for d in diffs:
        if not d.get("comparable"):
            print(f"  (not comparable) {d['python']} vs {d['matlab']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
