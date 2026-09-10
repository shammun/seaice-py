"""Book **Figure 6.14** — contour initialization from the distance transform (§6.3.3, pp. 133–136).

No MATLAB file draws this figure; it is the printed 8x8 worked example that explains the seed rule the shipped
code implements (``GVF_distance.m`` lines 113–130, ported as ``ch06_gvf_snake.initialize_contours``):

* (a) an 8x8 binary ice blob (``core.synth.FIG_6_14_IMAGE``),
* (b) its **city-block** distance transform (``core.synth.FIG_6_14_DISTANCE``) — "a regional maximum consisting
  of three local maxima, a seed, and an initial contour": the three pixels of value 3 form **one** regional
  maximum, the '+' marks the seed (the centre of the dilated maxima region) and the circle has radius
  ``D(seed)/sqrt(2) = 3/sqrt(2) = 2.1213`` (footnote 4, p. 135 — the divisor compensates the city-block metric).

The script also demonstrates Eqs. (6.57)/(6.58) (``M_max = I - R^D_I(I-1)``, the non-negative twin) on the same
matrix, and prints both matrices so the printed truth can be checked digit by digit.

Usage: ``python scripts/ch06_contour_init.py [--se-radius 3] [--out outputs/ch06] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch06_gvf_snake import CIRCLE_T, CITYBLOCK_RADIUS_DIVISOR, initialize_contours  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.distance import bwdist  # noqa: E402
from seaice.core.morphology import imregionalmin, regional_maxima_by_reconstruction  # noqa: E402
from seaice.core.plotting import finish_figure, show_matrix  # noqa: E402
from seaice.core.synth import FIG_6_14_DISTANCE, FIG_6_14_IMAGE  # noqa: E402

CH = "ch06"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--se-radius", type=int, default=3, help="strel('disk', r) used to merge maxima (script: 3)")
    args = p.parse_args(argv)
    _, out = resolve_dirs(args)

    A = FIG_6_14_IMAGE
    D = bwdist(~A, "cityblock")
    print("=== Fig. 6.14 (printed 8x8 truth) ===")
    print("(a) binary image matrix:")
    print(A.astype(int))
    print("(b) city-block distance transform:")
    print(D.astype(int))
    print(f"bwdist(~A, 'cityblock') == printed Fig. 6.14(b): {np.array_equal(D, FIG_6_14_DISTANCE)}")

    # The script's recipe: regional maxima of D = regional minima of -D with a -Inf background.
    imgd = -D.astype(np.float64)
    imgd[~A] = -np.inf
    minima = imregionalmin(imgd)
    dis = minima & A
    pts = np.argwhere(dis)
    print(f"regional maximum: {int(dis.sum())} local maxima at (row, col) 1-based "
          f"{[(int(r) + 1, int(c) + 1) for r, c in pts]}, all of value {sorted(set(D[dis].tolist()))}")

    # Book Eqs. (6.57)/(6.58) on the same matrix (the text's own wording).
    d_book = D.astype(np.float64).copy()
    d_book[~A] = -np.inf
    m657 = regional_maxima_by_reconstruction(d_book, form="6.57") & A
    m658 = regional_maxima_by_reconstruction(d_book, form="6.58") & A
    print(f"Eq. (6.57) M_max = I - R^D_I(I-1) gives the same {int(m657.sum())} pixels: {np.array_equal(m657, dis)}; "
          f"Eq. (6.58) too: {np.array_equal(m658, dis)}")

    init = initialize_contours(A, se_radius=args.se_radius)
    print(f"after imdilate(dis, strel('disk', {args.se_radius})) and bwlabel(.., 8): {init.num} seed(s)")
    for n in range(init.num):
        cx, cy = init.centroids[n]
        print(f"  seed {n + 1}: centroid (x, y) = ({cx:.5f}, {cy:.5f}) -> round = ({round(cx)}, {round(cy)}), "
              f"D(seed) = {D[int(round(cy)) - 1, int(round(cx)) - 1]:.0f}, "
              f"r = D/sqrt(2) = {init.radii[n]:.4f}")
    print(f"footnote 4 check: 3/sqrt(2) = {3.0 / CITYBLOCK_RADIUS_DIVISOR:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    show_matrix(axes[0], A.astype(int), fmt="{:d}", cmap="Blues", title="(a) binary image matrix")
    show_matrix(axes[1], D.astype(int), fmt="{:d}", cmap="Blues",
                title="(b) city-block distance transform, seed and initial contour")
    axes[1].plot(pts[:, 1], pts[:, 0], "rs", ms=14, mfc="none", mew=1.5, label="regional maximum (3 local maxima)")
    for n in range(init.num):
        cx, cy = init.centroids[n]
        r = init.radii[n]
        axes[1].plot(cx - 1, cy - 1, "g+", ms=14, mew=2, label="seed" if n == 0 else None)
        axes[1].plot(cx - 1 + r * np.cos(CIRCLE_T), cy - 1 + r * np.sin(CIRCLE_T), "g-", lw=1.5,
                     label=f"initial contour r = {r:.4f}" if n == 0 else None)
    axes[1].legend(loc="upper right", fontsize=7)
    fig.suptitle("Figure 6.14 — contour initialization algorithm based on the distance transform")
    written = [finish_figure(fig, out / "fig_6_14_contour_initialization.png", args.show)]
    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
