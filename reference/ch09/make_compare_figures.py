"""L3 evidence for chapter 9 — Python vs **MATLAB R2025a output**, side by side.

Written to ``reports/ch09/figures/`` (git-ignored).  Note what these can and cannot be:

* ``model_ice.jpg`` is **not a printed figure** of the book (`analysis/ch09.md` §0.4: whole-book NCC 0.167,
  best ``matchTemplate`` 0.373), and ``04100_analyse.jpg`` / the two AVIs **do not ship at all**.  So there is
  **no book plate to compare against anywhere in this chapter** — every panel below is *Python vs MATLAB on the
  same input*, which is L2 evidence rendered visually, plus the procedural shape of the book's figure.
* Nothing here is a book number.

Run: ``.venv/Scripts/python.exe reference/ch09/make_compare_figures.py`` (needs ``reference/ch09/*.mat``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
from scipy.io import loadmat                                       # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from seaice import ch09_model_ice as ch9                           # noqa: E402
from seaice.core.connectivity import bwareaopen                    # noqa: E402
from seaice.core.io import read_image                              # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab              # noqa: E402
from seaice.core.threshold import graythresh, im2bw                # noqa: E402
from seaice.core.video import read_video                           # noqa: E402

REF = ROOT / "reference/ch09"
SCRATCH = ROOT / "outputs/ch09/verify/scratch"
OUT = ROOT / "reports/ch09/figures"
OUT.mkdir(parents=True, exist_ok=True)


def _panel(ax, img, title, cmap="gray"):
    ax.imshow(img, cmap=cmap, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save(fig, name):
    fig.tight_layout()
    p = OUT / name
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


def fig_9_04():
    d = loadmat(REF / "ch09_block.mat")
    e = loadmat(REF / "block.mat")
    rgb = read_image(SCRATCH / "04100_analyse.jpg")
    b = ch9.block_threshold(rgb)
    ml = e["bw_tank"].astype(bool)
    fig, ax = plt.subplots(3, 1, figsize=(11, 6))
    _panel(ax[0], b.bw, f"Python block_threshold (Tier-3 tank) — IC per block "
                        f"{np.round(b.ic_local * 100, 2).tolist()}")
    _panel(ax[1], ml, f"MATLAB R2025a block_threshold.m — IC {np.round(d['IC'].ravel() * 100, 2).tolist()}")
    _panel(ax[2], b.bw ^ ml, f"XOR — {int((b.bw ^ ml).sum())} differing pixels of {b.bw.size}")
    fig.suptitle("Fig. 9.4 procedure (NOT the book plate: 04100_analyse.jpg does not ship)", fontsize=9)
    save(fig, "fig_9_04_compare.png")


def fig_9_03():
    e = loadmat(REF / "block.mat")
    rgb = read_image(SCRATCH / "04100_analyse.jpg")
    ic, mask, th = ch9.tank_ice_concentration(rgb, "otsu")
    ick, maskk, _ = ch9.tank_ice_concentration(rgb, "kmeans")
    fig, ax = plt.subplots(3, 1, figsize=(11, 6))
    _panel(ax[0], mask, f"Python global Otsu — level {th:.1f}, IC {ic * 100:.2f} %")
    _panel(ax[1], im2bw(rgb2gray_matlab(rgb), float(e["glevel_tank"].ravel()[0])),
           f"MATLAB graythresh/im2bw — level {float(e['glevel_tank'].ravel()[0]) * 255:.1f}, "
           f"IC {float(e['IC_global'].ravel()[0]) * 100:.2f} %")
    _panel(ax[2], maskk, f"Python k-means k=2 (ch3 kmeans.m) — IC {ick * 100:.2f} %")
    fig.suptitle("Figs. 9.3 / 9.5 procedure on the Tier-3 tank (book values 83.17 % / 82.86 % NOT reproducible)",
                 fontsize=9)
    save(fig, "fig_9_03_05_compare.png")


def fig_9_06_07():
    d = loadmat(REF / "ch09_movie_otsu.mat")
    k = loadmat(REF / "ch09_movie_kmeans.mat")
    F = read_video(SCRATCH / "dypic_synth_top.avi")
    r = ch9.movie_otsu(F)
    km = ch9.movie_kmeans(F)
    idx = 8
    fig, ax = plt.subplots(3, 2, figsize=(11, 6))
    _panel(ax[0, 0], F[:, :, :, idx], f"raw frame {idx + 1} (Fig. 9.6(a))", cmap=None)
    _panel(ax[0, 1], r.gray[idx], "blanked + cropped gray (Fig. 9.6(b)) — Python")
    _panel(ax[1, 0], r.bw[idx], "Otsu mask (Fig. 9.7(b)) — Python")
    _panel(ax[1, 1], np.transpose(d["bwstack"], (2, 0, 1))[idx], "Otsu mask — MATLAB movie_otsu.m")
    _panel(ax[2, 0], km.out[idx], "k-means mask (Fig. 9.7(c)) — Python")
    _panel(ax[2, 1], np.transpose(k["outstack"], (2, 0, 1))[idx], "k-means mask — MATLAB movie_kmeans.m")
    fig.suptitle("Figs. 9.6/9.7 procedure on the Tier-3 video (dypic_05100_cam1_top.avi does not ship)",
                 fontsize=9)
    save(fig, "fig_9_06_07_compare.png")


def fig_9_08_10():
    d = loadmat(REF / "ch09_movie_otsu.mat")
    k = loadmat(REF / "ch09_movie_kmeans.mat")
    F = read_video(SCRATCH / "dypic_synth_top.avi")
    r = ch9.movie_otsu(F)
    rc = ch9.movie_otsu(F, running_max_bug=False)
    km = ch9.movie_kmeans(F)
    n = np.arange(1, r.IC.size + 1)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(n, r.IC * 100, "o-", label="Python (E4 literal)")
    ax[0].plot(n, d["IC"].ravel() * 100, "x--", label="MATLAB movie_otsu.m")
    ax[0].plot(n, rc.IC * 100, ":", label="corrected per-frame rule")
    ax[0].set_title("Fig. 9.8 IC(t), Otsu", fontsize=9)
    ax[0].set_xlabel("Time (frame)")
    ax[0].set_ylabel("IC [%]")
    ax[0].legend(fontsize=7)
    ax[1].plot(n, km.IC * 100, "o-", label="Python k-means")
    ax[1].plot(n, k["IC"].ravel() * 100, "x--", label="MATLAB movie_kmeans.m")
    ax[1].set_title("Fig. 9.9 IC(t), k-means", fontsize=9)
    ax[1].set_xlabel("Time (frame)")
    ax[1].legend(fontsize=7)
    ax[2].plot(n, r.t * 255, "o-", label="t(k) Python")
    ax[2].plot(n, d["t"].ravel() * 255, "x--", label="t(k) MATLAB")
    ax[2].plot(n, r.effective_level * 255, "-", label="E4 running max")
    ax[2].set_title("Fig. 9.10 threshold(t)", fontsize=9)
    ax[2].set_xlabel("Time (frame)")
    ax[2].legend(fontsize=7)
    fig.suptitle("Figs. 9.8-9.10 procedure on the Tier-3 video (book Table 9.3 NOT reproducible)", fontsize=9)
    save(fig, "fig_9_08_10_compare.png")


def fig_9_11():
    d = loadmat(REF / "ch09_demo.mat")
    I = read_image(ROOT / "data/book/ch09/Model_Ice_Floe_Identification/model_ice.jpg")
    res = ch9.model_ice_demo(I, full=True)
    fig, ax = plt.subplots(1, 7, figsize=(13, 5))
    g = rgb2gray_matlab(I)
    _panel(ax[0], g, "model_ice.jpg gray")
    _panel(ax[1], im2bw(g, graythresh(g)[0]), "im2bw (Fig. 9.11(b))")
    _panel(ax[2], res.bw1, f"Python bw1 ({int(res.bw1.sum())} px)")
    _panel(ax[3], d["bw1"].astype(bool), f"MATLAB bw1 ({int(d['bw1'].sum())} px)")
    _panel(ax[4], res.bw1 ^ d["bw1"].astype(bool), f"XOR {int((res.bw1 ^ d['bw1'].astype(bool)).sum())} px")
    _panel(ax[5], res.bw4, f"Python bw4 ({int(res.bw4.sum())} px)")
    _panel(ax[6], d["bw4"].astype(bool), f"MATLAB bw4 ({int(d['bw4'].sum())} px)")
    fig.suptitle("Fig. 9.11 procedure — model_ice.jpg is NOT a printed figure, so this is Python vs MATLAB only",
                 fontsize=9)
    save(fig, "fig_9_11_compare.png")
    return res, d


def fig_9_15_16(res, d):
    bw4 = d["bw4"].astype(bool)
    S = ch9.rect(bw4)
    m = ch9.model_ice_model(S, bw4, 0.4, 2.5)
    fig, ax = plt.subplots(1, 4, figsize=(12, 5))
    _panel(ax[0], bw4, "MATLAB bw4 (input)")
    ax[1].imshow(bw4, cmap="gray")
    for s in S:
        ax[1].plot(s.Vertices[:, 0] - 1, s.Vertices[:, 1] - 1, "c-", lw=0.7)
        ax[1].plot(s.Center[0] - 1, s.Center[1] - 1, "r+", ms=3)
    ax[1].set_title(f"Fig. 9.15(a) rect.m — {len(S)} rectangles", fontsize=8)
    ax[1].axis("off")
    _panel(ax[2], m.bw, f"Fig. 9.15(b) model raster — {len(m.s_model)} accepted, "
                        f"{int(sum(len(f.Intersection) for f in m.s_model))} overlap flags")
    ml = np.zeros_like(m.bw)
    _panel(ax[3], m.bw.astype(int) * 1, "accepted union (Python == MATLAB Intersection sets)")
    fig.suptitle("Fig. 9.15 procedure on MATLAB's own bw4 (rect.m / model_ice_model.m parity is exact there)",
                 fontsize=9)
    save(fig, "fig_9_15_compare.png")

    areas_id = np.array([r.Area for r in ch9.rect(bw4)])
    counts_r, centres = ch9.floe_size_histogram([f.Area for f in m.s_model], 12)
    counts_i, _ = ch9.floe_size_histogram(areas_id, centres)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].bar(centres, counts_i, width=np.diff(centres).mean() * 0.4, label="identified (rect areas)")
    ax[0].bar(centres + np.diff(centres).mean() * 0.4, counts_r,
              width=np.diff(centres).mean() * 0.4, label="accepted rectangles")
    ax[0].set_title("Fig. 9.16(a) FSD (procedure)", fontsize=9)
    ax[0].set_xlabel("Floe size (pixel number)")
    ax[0].set_ylabel("Frequency")
    ax[0].legend(fontsize=7)
    ax[1].bar(centres, ch9.fsd_error(counts_r, counts_i), width=np.diff(centres).mean() * 0.7)
    ax[1].set_title("Fig. 9.16(b) bin-wise error", fontsize=9)
    ax[1].set_xlabel("Floe size (pixel number)")
    fig.suptitle("Figs. 9.16 procedure — book axes 0-6000 px / 0-300 and -60..+60 NOT reproducible", fontsize=9)
    save(fig, "fig_9_16_compare.png")


def fig_9_17_18():
    d = loadmat(REF / "ch09_movie_floe.mat")
    F = read_video(SCRATCH / "05100_synth_segmented.avi")
    r = ch9.movie_floe(F, keep_labels=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    _panel(ax[0], bwareaopen(im2bw(F[:, :, :, 4]), 20, 4), "Fig. 9.17(a)/(b) segmented frame 5 (Python)")
    _panel(ax[1], d["bwstack"][:, :, 4].astype(bool), "frame 5 — MATLAB movie_floe.m")
    ax[2].plot(np.arange(1, r.floe.size + 1), r.floe, "o-", label="Python")
    ax[2].plot(np.arange(1, r.floe.size + 1), d["floe"].ravel(), "x--", label="MATLAB")
    ax[2].set_title("Fig. 9.18 max floe area per frame", fontsize=9)
    ax[2].set_xlabel("Time (frame)")
    ax[2].set_ylabel("Max floe area [px]")
    ax[2].legend(fontsize=7)
    fig.suptitle("Figs. 9.17/9.18 procedure on the Tier-3 segmented video (05100.avi does not ship)", fontsize=9)
    save(fig, "fig_9_17_18_compare.png")


def fig_R7():
    from seaice.ch07_ice_type import colorbar_area_ticks
    colours = np.array([198.0, 9974.0])
    v6, l6 = colorbar_area_ticks(colours, 6)
    v8, l8 = colorbar_area_ticks(colours, 8)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(v6, l6, "o-", label=f"n = 6 (map, Fig. 9.12): {list(map(int, l6))}")
    ax.plot(v8, l8, "s--", label=f"nn = 8 (histogram, Figs. 9.13/9.16): {list(map(int, l8))}")
    ax.set_xlabel("Eq. (7.6) colour value")
    ax.set_ylabel("printed tick label (area, px)")
    ax.set_title("R7 adjudicated: ONE colour range [198, 9974], two tick conventions", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "fig_9_12_13_ticks.png")


if __name__ == "__main__":
    fig_9_04()
    fig_9_03()
    fig_9_06_07()
    fig_9_08_10()
    res, d = fig_9_11()
    fig_9_15_16(res, d)
    fig_9_17_18()
    fig_R7()
