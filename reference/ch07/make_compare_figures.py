"""L3 evidence for chapter 7: side-by-side figure comparisons (Python | MATLAB | book page).

Chapter 7's own figures are mostly **printed binary matrices** (Figs. 7.2-7.8), so the comparison is drawn
directly from the arrays: the top row is the Python port's block, the middle row is the array MATLAB produced
while ``reference/ch07/make_refs.py`` ran the original ``.m`` file, and the right-hand column is the book page
rendered from ``chapters/ch07.pdf``.  Differing cells are ringed in red.

The §7.2 identification figures (7.13-7.16) belong to an image the book does **not** ship, so the last figure
compares the Python and MATLAB identification layers of ch7's own ``sea_ice_test.jpg`` instead and says so.

Usage: .venv/Scripts/python.exe reference/ch07/make_compare_figures.py
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
from scipy.io import loadmat  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from seaice import ch07_ice_type as ch7  # noqa: E402
from seaice.core import synth  # noqa: E402
from seaice.core.plotting import label2rgb  # noqa: E402

REF = ROOT / "reference/ch07"
OUT = ROOT / "outputs/ch07"
VERIFY = OUT / "verify"
ML = VERIFY / "matlab"
REPORT_FIG = ROOT / "reports/ch07/figures"
PDF = ROOT / "chapters/ch07.pdf"
FIRST_PRINTED_PAGE = 145


def page_image(printed_page: int, dpi: int = 110) -> np.ndarray:
    doc = pymupdf.open(PDF)
    pix = doc[printed_page - FIRST_PRINTED_PAGE].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img[..., :3]


def _show_block(ax, a, title, ref=None):
    a = np.asarray(a, float)
    ax.imshow(1.0 - a, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(f"{title}\nsum {int(a.sum())}", fontsize=7)
    if ref is not None:
        diff = np.argwhere(a != np.asarray(ref, float))
        for r, c in diff:
            ax.add_patch(plt.Circle((c, r), 0.45, fill=False, color="red", lw=1.4))
        if len(diff):
            ax.set_title(f"{title}\nsum {int(a.sum())} — {len(diff)} px differ", fontsize=7, color="red")


def compose_blocks(name: str, rows: list[tuple[str, list[tuple[str, np.ndarray]]]],
                   page: int | None, caption: str, diff_rows: tuple[int, int] | None = None) -> Path:
    ncol = max(len(items) for _, items in rows)
    nrow = len(rows)
    extra = 1 if page else 0
    fig = plt.figure(figsize=(1.55 * ncol + 5.0 * extra, 2.0 * nrow + 1.0))
    gs = fig.add_gridspec(nrow, ncol + extra, width_ratios=[1] * ncol + ([3.0] if extra else []))
    for r, (label, items) in enumerate(rows):
        for c in range(ncol):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")
            if c < len(items):
                title, arr = items[c]
                ref = None
                if diff_rows and r == diff_rows[0]:
                    other = rows[diff_rows[1]][1]
                    if c < len(other):
                        ref = other[c][1]
                _show_block(ax, arr, title if c else f"{label}\n{title}", ref)
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


def main() -> int:
    VERIFY.mkdir(parents=True, exist_ok=True)
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    made, diffs = [], {}
    d = loadmat(str(REF / "clf.mat"))
    d2 = loadmat(str(REF / "fig767.mat")) if (REF / "fig767.mat").exists() else None

    # ---- Fig. 7.2 --------------------------------------------------------------------------------------------
    r = ch7.morphological_cleaning(synth.FIG_7_2_IMAGE.astype(float), np.ones((2, 2), bool))
    names = [("(a) I", "I"), ("(b) closing f1", "f1"), ("(c) opening f2", "f2"), ("(d) cleaning f0", "f0")]
    made.append(compose_blocks(
        "fig_7_2",
        [("Python", [(t, getattr(r, k) if k != "I" else synth.FIG_7_2_IMAGE) for t, k in names]),
         ("MATLAB R2025a", [(t, d[f"cl_{k}"]) for t, k in names]),
         ("book (printed matrix)", [("(a)", synth.FIG_7_2_IMAGE), ("(b)", synth.FIG_7_2_CLOSED),
                                    ("(c)", synth.FIG_7_2_OPENED), ("(d)", synth.FIG_7_2_CLEANED)])],
        146, "Fig. 7.2 — morphological cleaning of the printed 13x23 matrix, SE = strel('square', 2)",
        diff_rows=(0, 1)))
    diffs["fig_7_2"] = {t: int(np.count_nonzero(np.asarray(getattr(r, k) if k != "I" else synth.FIG_7_2_IMAGE,
                                                           float) != np.asarray(d[f"cl_{k}"], float)))
                        for t, k in names}

    # ---- Figs. 7.3 / 7.4 -------------------------------------------------------------------------------------
    for fig_no, page, se, pre, steps in (("7_3", 148, "square3", "lb3", synth.FIG_7_3_STEPS),
                                         ("7_4", 149, "diamond1", "lb1", synth.FIG_7_4_STEPS)):
        res = ch7.connected_component_extract(synth.FIG_7_3_IMAGE, synth.FIG_7_3_SEED, se, max_iter=20)
        nb = len(steps)
        ml = ([d[f"{pre}_xx"]] + [d[f"{pre}_x{i}"] for i in range(1, 10)])[:nb]
        py = list(res.blocks)[:nb]
        labels = ["X0 (+) B"] + [f"X{i}" for i in range(1, nb)]
        made.append(compose_blocks(
            f"fig_{fig_no}",
            [("Python", list(zip(labels, py))), ("MATLAB R2025a", list(zip(labels, ml))),
             ("book (printed)", list(zip(labels, steps)))],
            page,
            f"Fig. {fig_no.replace('_', '.')} — Eq. (7.1) connected-component extraction, SE = {se}",
            diff_rows=(0, 1)))
        diffs[f"fig_{fig_no}"] = [int(np.count_nonzero(np.asarray(p, float) != np.asarray(m, float)))
                                  for p, m in zip(py, ml)]

    # ---- Fig. 7.5 --------------------------------------------------------------------------------------------
    res = ch7.hole_fill_dilation(synth.FIG_7_3_IMAGE, synth.FIG_7_5_SEED, "diamond1", max_iter=20)
    nb = len(synth.FIG_7_5_STEPS)
    ml = [d["fi_xx"]] + [d[f"fi_x{i}"] for i in range(1, 10)]
    py = list(res.blocks) + [res.filled]
    labels = ["X0 (+) B"] + [f"X{i}" for i in range(1, nb - 1)] + ["X (u) A"]
    ml_row = ml[:nb - 1] + [d["fi_I1"]]
    made.append(compose_blocks(
        "fig_7_5",
        [("Python", list(zip(labels, py[:nb - 1] + [res.filled]))),
         ("MATLAB R2025a", list(zip(labels, ml_row))),
         ("book (printed)", list(zip(labels, synth.FIG_7_5_STEPS)))],
        150, "Fig. 7.5 — Eq. (7.2) hole filling by constrained dilation (cross SE, seed inside the hole)",
        diff_rows=(0, 1)))

    # ---- Figs. 7.6 / 7.7 -------------------------------------------------------------------------------------
    if d2 is not None:
        for fig_no, page, se, pre, steps in (("7_6", 151, "diamond1", "f76", synth.FIG_7_6_STEPS),
                                             ("7_7", 152, "square3", "f77", synth.FIG_7_7_STEPS)):
            res = ch7.hole_fill_dilation(synth.FIG_7_6_IMAGE, synth.FIG_7_5_SEED, se, max_iter=20)
            nb = len(steps)
            mlb = [d2[f"{pre}_xx"]] + [d2[f"{pre}_x{i}"] for i in range(1, 10)]
            if fig_no == "7_6":
                py = list(res.blocks)[:nb - 1] + [res.filled]
                mlrow = mlb[:nb - 1] + [d2["f76_I1"]]
                labels = ["X0 (+) B", "X1", "X2 = fixed point", "X (u) A"]
            else:
                py = list(res.blocks)[:nb]
                mlrow = mlb[:nb]
                labels = ["X0 (+) B"] + [f"X{i}" for i in range(1, nb)]
            book_row = list(steps)
            if fig_no == "7_7":
                book_row = list(synth.FIG_7_7_STEPS_BOOK)
            made.append(compose_blocks(
                f"fig_{fig_no}",
                [("Python", list(zip(labels, py))), ("MATLAB R2025a", list(zip(labels, mlrow))),
                 ("book (printed)", list(zip(labels, book_row)))],
                page,
                f"Fig. {fig_no.replace('_', '.')} — Eq. (7.2) failure case ({se}); "
                + ("the printed X8 has 53 px, MATLAB's has 55 — a book typo at 1-based (8,9) and (9,9)"
                   if fig_no == "7_7" else "the cross SE fills only the hole containing the seed"),
                diff_rows=(2, 1)))
            diffs[f"fig_{fig_no}"] = [int(np.count_nonzero(np.asarray(b, float) != np.asarray(m, float)))
                                      for b, m in zip(book_row, mlrow)]

    # ---- Fig. 7.8 --------------------------------------------------------------------------------------------
    I0 = np.asarray(d["fr1_I0"], bool)
    marker = ch7.border_marker(I0)
    H = ch7.hole_fill_reconstruct(I0, conn=4)
    py = [~I0, marker, np.asarray(d["fr1_x1"], bool), np.asarray(d["fr1_x2"], bool),
          np.asarray(d["fr1_x4"], bool), H, H & ~I0]
    py = [~I0, marker, ch7.hole_fill_reconstruct(I0, conn=4), H | I0, H & ~I0]
    labels = ["F^c", "F_m (Eq. 7.3)", "H = [R(F_m)]^c", "H (u) F", "H (n) F^c"]
    mlrow = [d["fr1_I"], d["fr1_x0"], d["fr1_x10"], d["fr1_I1"], d["fr1_I2"]]
    made.append(compose_blocks(
        "fig_7_8",
        [("Python", list(zip(labels, py))), ("MATLAB R2025a", list(zip(labels, mlrow)))],
        153, "Fig. 7.8 — Eqs. (7.3)/(7.4) automatic hole filling by reconstruction (9x11, cross SE)",
        diff_rows=(0, 1)))

    # ---- Fig. 7.22 / the real image --------------------------------------------------------------------------
    jpg = ROOT / "data/book/ch07/Sea_Ice_Floe_Identification/sea_ice_test.jpg"
    if jpg.exists():
        rgb = iio.imread(jpg)
        fig = plt.figure(figsize=(13, 5))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(np.swapaxes(rgb, 0, 1))
        ax.axis("off")
        ax.set_title("ch7's sea_ice_test.jpg, transposed (394 x 1038)", fontsize=9)
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(page_image(168))
        ax.axis("off")
        ax.set_title("book p. 168 (Fig. 7.22)", fontsize=9)
        fig.suptitle("Fig. 7.22 — the MIZ image of the §7.3.2 sensitivity study", fontsize=10)
        fig.tight_layout()
        dest = VERIFY / "fig_7_22_compare.png"
        fig.savefig(dest, dpi=110)
        plt.close(fig)
        shutil.copy(dest, REPORT_FIG / dest.name)
        made.append(str(dest))

    # ---- §7.2 identification layers: Python vs MATLAB on ch7's own image --------------------------------------
    demo = REF / "demo.mat"
    if demo.exists():
        dd = loadmat(str(demo))
        r = ch7.ice_shape_enhancement(dd["bk"], dd["seg"], nbins=50)
        layers = [("index (floe + brash, Eq. 7.6 colours)", label2rgb(r.index, cmap="jet",
                                                                     background=(1, 1, 1), shuffle=False)),
                  ("index_floe", r.index_floe > 0), ("index_brash", r.index_brash > 0),
                  ("index_slush", r.index_slush), ("index_water", r.index_water),
                  ("index_residue", r.index_residue)]
        ml_files = ["demo_index_rgb.png", "demo_index_floe.png", "demo_index_brash.png",
                    "demo_index_slush.png", "demo_index_water.png", "demo_index_residue.png"]
        fig = plt.figure(figsize=(4.2, 12))
        n = len(layers)
        px = {}
        for i, ((title, arr), mf) in enumerate(zip(layers, ml_files)):
            ax = fig.add_subplot(n, 2, 2 * i + 1)
            a = np.asarray(arr)
            ax.imshow(np.swapaxes(a, 0, 1) if a.ndim == 2 else np.swapaxes(a, 0, 1),
                      cmap="gray" if a.ndim == 2 else None, interpolation="nearest")
            ax.axis("off")
            ax.set_title(f"Python {title}", fontsize=6)
            ax = fig.add_subplot(n, 2, 2 * i + 2)
            p = ML / mf
            if p.exists():
                m = iio.imread(p)
                ax.imshow(np.swapaxes(m, 0, 1) if m.ndim == 2 else np.swapaxes(m, 0, 1),
                          cmap="gray" if m.ndim == 2 else None, interpolation="nearest")
                if a.ndim == 2:
                    mm = (m > 0) if m.ndim == 2 else (m[..., 0] > 0)
                    px[title] = int(np.count_nonzero((np.asarray(a) != 0) != mm))
            ax.axis("off")
            ax.set_title(f"MATLAB {mf}", fontsize=6)
        fig.suptitle("§7.2 identification layers on ch7's sea_ice_test.jpg\n"
                     "(NOT a book figure — Figs. 7.13-7.16 use a 205x263 image the book does not ship)",
                     fontsize=8)
        fig.tight_layout()
        dest = VERIFY / "sec_7_2_identification_compare.png"
        fig.savefig(dest, dpi=140)
        plt.close(fig)
        shutil.copy(dest, REPORT_FIG / dest.name)
        made.append(str(dest))
        diffs["sec_7_2_layers_px"] = px

    (VERIFY / "figure_diffs.json").write_text(json.dumps(diffs, indent=1), encoding="utf-8")
    print(f"{len(made)} compare figures -> {REPORT_FIG}")
    print(json.dumps(diffs, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
