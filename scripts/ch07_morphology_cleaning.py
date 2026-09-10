"""Port of ``MATLAB_ROOT/ch7/cleaning & labeling & filling/morphology_cleaning.m`` — Book §7.1.1, Fig. 7.2.

Morphological cleaning = **closing first, then opening**, with a 2x2 square structuring element, on the 13x23
binary matrix the M-file hard-codes (which is also the matrix printed as Fig. 7.2(a)).  The script reproduces
both MATLAB figures: ``[I, f1, f2, f0]`` (Fig. 7.2(a)-(d)) and the explicit ``[c1, c2, c3, c4]`` dilate/erode
chain that spells the same result out step by step (in the code, not in the book text).

Usage: ``python scripts/ch07_morphology_cleaning.py [--data data/book/ch07] [--out outputs/ch07] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch07_ice_type import morphological_cleaning  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.morphology import strel  # noqa: E402
from seaice.core.plotting import finish_figure, save_image, show_matrix  # noqa: E402
from seaice.core.synth import FIG_7_2_CLEANED, FIG_7_2_CLOSED, FIG_7_2_IMAGE, FIG_7_2_OPENED  # noqa: E402

CH = "ch07"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--se-size", type=int, default=2, help="side of the square SE (morphology_cleaning.m line 17: 2)")
    args = p.parse_args(argv)
    _, out = resolve_dirs(args)

    se = strel("square", args.se_size)
    # `I` is a **double** 0/1 matrix in the M-file, so the port keeps that class.
    I = FIG_7_2_IMAGE.astype(np.float64)
    r = morphological_cleaning(I, se)

    print(f"=== morphology_cleaning.m (Book section 7.1.1, Fig. 7.2) — SE = strel('square', {args.se_size}) "
          f"({se.shape[0]}x{se.shape[1]}, {int(se.sum())} px) ===")
    print(f"image {I.shape[0]}x{I.shape[1]}, {int(I.sum())} object pixels")
    print(f"  f1 = imclose(I, se)      : {int(r.f1.sum()):4d} px   (Fig. 7.2(b))")
    print(f"  f2 = imopen(I, se)       : {int(r.f2.sum()):4d} px   (Fig. 7.2(c))")
    print(f"  f0 = imopen(f1, se)      : {int(r.f0.sum()):4d} px   (Fig. 7.2(d), the cleaning)")
    print(f"  c1 = imdilate(I, se)     : {int(r.c1.sum()):4d} px")
    print(f"  c2 = imerode(c1, se)     : {int(r.c2.sum()):4d} px   (== f1: "
          f"{np.array_equal(r.c2 != 0, r.f1 != 0)})")
    print(f"  c3 = imerode(c2, se)     : {int(r.c3.sum()):4d} px")
    print(f"  c4 = imdilate(c3, se)    : {int(r.c4.sum()):4d} px   (== f0: "
          f"{np.array_equal(r.c4 != 0, r.f0 != 0)})")

    if args.se_size == 2:  # only the book's SE reproduces the printed blocks
        for name, got, want in (("Fig. 7.2(b)", r.f1, FIG_7_2_CLOSED), ("Fig. 7.2(c)", r.f2, FIG_7_2_OPENED),
                                ("Fig. 7.2(d)", r.f0, FIG_7_2_CLEANED)):
            print(f"  {name} vs the printed matrix: {int(np.abs((got != 0).astype(int) - want.astype(int)).sum())} px differ")

    written = []
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    for ax, mat, title in ((axes[0, 0], r.I, "(a) I — binary image matrix"),
                           (axes[0, 1], r.f1, "(b) f1 = imclose(I, se)"),
                           (axes[1, 0], r.f2, "(c) f2 = imopen(I, se)"),
                           (axes[1, 1], r.f0, "(d) f0 = imopen(f1, se) — morphological cleaning")):
        show_matrix(ax, np.asarray(mat, dtype=int), fmt="{:d}", title=title, fontsize=6)
    fig.suptitle("Fig. 7.2 — morphological cleaning (closing then opening), strel('square', 2)")
    written.append(finish_figure(fig, out / "fig_7_02_cleaning.png", args.show))

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    for ax, mat, title in ((axes[0, 0], r.c1, "c1 = imdilate(I, se)"),
                           (axes[0, 1], r.c2, "c2 = imerode(c1, se)  = closing"),
                           (axes[1, 0], r.c3, "c3 = imerode(c2, se)"),
                           (axes[1, 1], r.c4, "c4 = imdilate(c3, se) = cleaning")):
        show_matrix(ax, np.asarray(mat, dtype=int), fmt="{:d}", title=title, fontsize=6)
    fig.suptitle("morphology_cleaning.m second figure — the explicit dilate/erode chain (not in the book text)")
    written.append(finish_figure(fig, out / "sec_7_1_1_dilate_erode_chain.png", args.show))

    for name, mat in (("a_image", r.I), ("b_closed", r.f1), ("c_opened", r.f2), ("d_cleaned", r.f0)):
        written.append(save_image(out / f"sec_7_1_1_{name}.png", np.asarray(mat) != 0))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
