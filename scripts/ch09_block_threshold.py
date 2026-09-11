"""Port of ``MATLAB_ROOT/ch9/block_threshold.m`` — §9.2.1 ice concentration from the overall tank image.

The M-file reads ``04100_analyse.jpg``, converts to gray, splits it into a ``2 x 3`` grid and thresholds each
block with its own Otsu level, printing ``IC=..%`` / ``Threshold=..`` under each tile (**Fig. 9.4**).  It differs
from ch3's ``local_Otsu.m`` in one numeric respect: line 26 counts ice with ``>=`` where ch3 uses ``>``
(``core.threshold.block_otsu(..., compare='ge')``), and it drops ch3's pixel-weighted overall ``IC``, so the
caption's "average IC" is the **unweighted mean of the six blocks** (book number N4: 498.88/6 = 83.1467 %).

This script also covers the two §9.2.1 results the book prints with **no MATLAB code**: the grayscale histogram
(Fig. 9.2), global Otsu (Fig. 9.3) and k-means with k = 2 (Fig. 9.5) — all Chapter-3 routines applied unchanged.

DATA: ``04100_analyse.jpg`` is an HSVA/DYPIC asset that was **never published** (risk R1).  If it is not in
``--data``, the script falls back to the seeded Tier-3 tank of ``core.synth.model_ice_tank`` (written once to
``data/synthetic/ch09/04100_analyse.jpg`` as JPEG q95 4:4:4 and read back, the ch03 ``t.jpg`` precedent) and says
so on every line it prints.  **The book's numbers N2/N3/N5/N6 are never reproduced and never invented.**

Usage: ``python scripts/ch09_block_threshold.py [--image 04100_analyse.jpg] [--rows 2] [--cols 3]
[--data data/book/ch09] [--out outputs/ch09] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch09_model_ice import (FIG_9_4_IC, FIG_9_4_THRESH, TABLE_9_2, block_threshold,  # noqa: E402
                                   ensure_synthetic_tank, tank_ice_concentration)
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.histogram import imhist  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import num2str as _num2str, rgb2gray_matlab  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402
from seaice.core.threshold import block_otsu  # noqa: E402

CH = "ch09"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="04100_analyse.jpg", help="block_threshold.m line 2")
    p.add_argument("--rows", type=int, default=2, help="n_r (script: 2)")
    p.add_argument("--cols", type=int, default=3, help="n_c (script: 3)")
    p.add_argument("--regenerate", action="store_true", help="rewrite the synthetic tank JPEG")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    try:
        rgb, source = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
        synthetic = False
    except FileNotFoundError:
        path, rgb = ensure_synthetic_tank(regenerate=args.regenerate)
        source = f"SYNTHETIC (Tier 3) {path.name}"
        synthetic = True
        print(f"NOTE: {args.image} is not present — it is an HSVA/DYPIC asset that was never published "
              f"(analysis/ch09.md risk R1).  Using the seeded synthetic tank {path}.")
        print("      Every number below is computed on that synthetic image.  The book's Fig. 9.3/9.4 and "
              "Table 9.2 values are NOT reproducible and are printed only for reference.")

    gray = rgb2gray_matlab(rgb) if rgb.ndim == 3 else rgb
    r, c = gray.shape
    print(f"\n=== block_threshold.m on {source} ({r} x {c}) ===")
    if r % args.rows or c % args.cols:
        print(f"SKIP: {r} x {c} does not divide into {args.rows} x {args.cols} equal blocks; MATLAB errors on "
              "the non-integer block indices (block_threshold.m lines 9-11).")
        return 0

    res = block_threshold(rgb, args.rows, args.cols)
    ch3 = block_otsu(gray, args.rows, args.cols, compare="gt")   # ch3's `>` rule, for the one-line comparison
    print(f"n_r = {args.rows}, n_c = {args.cols}; block size {r // args.rows} x {c // args.cols}")
    print("block  threshold (th = 255*graythresh)   IC (>=, ch9)   IC (>, ch3)   pixels at exactly th")
    for b in range(args.rows * args.cols):
        at_th = int(res.counts[b] - ch3.counts[b])
        print(f"  {b + 1}    {res.thresholds[b]:22.4f}   {res.ic_local[b] * 100:9.4f} %   "
              f"{ch3.ic_local[b] * 100:8.4f} %   {at_th:6d}")
    print(f"average IC (unweighted mean of the {args.rows * args.cols} blocks, the Fig. 9.4 caption's number, "
          f"book N4) = {res.ic_mean * 100:.4f} %  ->  title '{_num2str(round(res.ic_mean * 100, 4))}'")
    print(f"average threshold = {res.thresh_mean:.4f}")
    print(f"ch3's pixel-weighted IC (the line ch9 DELETED) = {res.ic * 100:.4f} %")

    ic_g, mask_g, th_g = tank_ice_concentration(rgb, "otsu")
    ic_k, mask_k, _ = tank_ice_concentration(rgb, "kmeans")             # the authors' ch3/kmeans.m
    print(f"\nFig. 9.3 global Otsu : IC = {ic_g * 100:.2f} %, threshold = {th_g:.1f}")
    print(f"Fig. 9.5 k-means k=2 : IC = {ic_k * 100:.2f} %  (core.clustering.kmeans_gray = the authors' "
          "ch3/kmeans.m, deterministic)")
    print(f"Fig. 9.4 local Otsu  : IC = {res.ic_mean * 100:.2f} %")
    print("p. 199's explanation — a bimodal histogram, near-uniform illumination and a single ice type make the "
          "two Otsu assumptions hold, so global ~ local ~ k-means:")
    print(f"   spread of the three = {max(ic_g, ic_k, res.ic_mean) * 100 - min(ic_g, ic_k, res.ic_mean) * 100:.2f} "
          "percentage points")
    if synthetic:
        print("\nBOOK NUMBERS (NOT reproduced — the image is missing; these are printed for orientation only):")
        print(f"   Fig. 9.3  IC = {TABLE_9_2[5100]['global_otsu']} %, threshold = 84")
        print(f"   Fig. 9.4  block ICs {FIG_9_4_IC}, thresholds {FIG_9_4_THRESH}")
        print(f"   Fig. 9.4  caption average IC = {sum(FIG_9_4_IC) / 6:.4f} % -> '83.14 %', "
              f"average threshold = {sum(FIG_9_4_THRESH) / 6:.4f} -> '84'   [N4: arithmetic, self-consistent]")
        print(f"   Table 9.2 run 5100: target {TABLE_9_2[5100]['target']} %, "
              f"Otsu {TABLE_9_2[5100]['global_otsu']} %, local {TABLE_9_2[5100]['local_otsu']} %, "
              f"k-means {TABLE_9_2[5100]['kmeans']} %")

    written = [save_image(out / "fig_9_01_tank_image.png", gray)]

    counts, x = imhist(gray)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, counts, width=1.0, color="0.25")
    ax.set_xlabel("gray level")
    ax.set_ylabel("frequency")
    ax.set_title("Fig. 9.2 — histogram of the overall tank image"
                 + (" (synthetic stand-in)" if synthetic else ""))
    written.append(finish_figure(fig, out / "fig_9_02_tank_histogram.png", args.show))

    written.append(save_image(out / "fig_9_03_global_otsu.png", mask_g))
    written.append(save_image(out / "fig_9_05_kmeans.png", mask_k))

    fig, axes = plt.subplots(args.rows, args.cols, figsize=(4.2 * args.cols, 3.0 * args.rows))
    axes = np.atleast_2d(axes)
    for i in range(args.rows):
        for j in range(args.cols):
            b = i * args.cols + j
            sl = res.slices[b]
            ic0 = res.ic_local[b] * 100.0
            imshow_matlab(axes[i, j], res.bw[sl],
                          title=f"$IC$={_num2str(ic0)}%\nThreshold={_num2str(res.thresholds[b])}")
            axes[i, j].axis("off")
    fig.suptitle("Fig. 9.4 — local (2 x 3 block) Otsu" + (" on the synthetic tank" if synthetic else ""))
    written.append(finish_figure(fig, out / "fig_9_04_block_threshold.png", args.show))
    written.append(save_image(out / "fig_9_04_block_mask.png", res.bw))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
