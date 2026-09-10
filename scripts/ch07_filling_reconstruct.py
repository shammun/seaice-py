"""Port of ``ch7/cleaning & labeling & filling/filling_reconstruct.m`` — Book §7.1.3.2, Eqs. (7.3)-(7.4), Fig. 7.8.

Automatic hole filling by morphological reconstruction.  The marker is Eq. (7.3)::

    F_m(x, y) = 1 - F(x, y)   on the border of F,   0 otherwise

and the filled image is Eq. (7.4)::

    H = [ R^D_{F^c}(F_m) ]^c

"this algorithm is fully automatic since it does not need any information of the holes" (p. 152).  The M-file
unrolls the reconstruction as nine constrained dilations (``x1``..``x9``) and then complements (``x10 = ~x9``);
the library form is ``imfill(F, 'holes')`` = :func:`seaice.core.morphology.imfill`.

``--i0 alternative`` runs the commented image of ``filling_reconstruct.m`` lines 13-21, and ``--se square3``
the commented SE of line 34.

Usage: ``python scripts/ch07_filling_reconstruct.py [--se diamond1|square3] [--i0 shipped|alternative]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.ndimage as ndi  # noqa: E402

from seaice.ch07_ice_type import border_marker, hole_fill_reconstruct  # noqa: E402
from seaice.ch07_ice_type import _constrained_dilation, _resolve_se  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.morphology import imfill  # noqa: E402
from seaice.core.plotting import finish_figure, save_image, show_matrix  # noqa: E402
from seaice.core.synth import FIG_7_8_IMAGE, FIG_7_8_MARKER, FIG_7_8_STEPS, _m  # noqa: E402

CH = "ch07"

#: ``filling_reconstruct.m`` lines 13-21 — the commented-out alternative ``I0`` (5 extra pixels).
I0_ALTERNATIVE = _m(
    """
    0 0 0 0 0 0 0 0 1 1 1
    0 1 1 1 1 1 0 0 1 0 1
    0 1 1 0 0 1 0 0 1 0 1
    0 1 1 0 0 1 1 0 1 1 1
    0 1 1 1 1 0 1 1 0 1 0
    0 0 1 1 1 0 0 1 0 0 0
    0 0 0 0 1 1 1 1 0 0 0
    1 1 0 0 0 0 1 0 0 0 0
    1 0 0 1 0 0 0 0 0 0 0
    """
).astype(bool)


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--se", default="diamond1", choices=["diamond1", "square3"],
                   help="filling_reconstruct.m line 33 (diamond1) or the commented line 34 (square3)")
    p.add_argument("--i0", default="shipped", choices=["shipped", "alternative"],
                   help="the file's active I0 (lines 3-11) or the commented one (lines 13-21)")
    p.add_argument("--max-iter", type=int, default=9)
    args = p.parse_args(argv)
    _, out = resolve_dirs(args)

    F = FIG_7_8_IMAGE if args.i0 == "shipped" else I0_ALTERNATIVE
    conn = 4 if args.se == "diamond1" else 8
    Fm = border_marker(F)                        # Eq. (7.3)
    r = _constrained_dilation(~F, Fm, _resolve_se(args.se), args.max_iter)
    H = hole_fill_reconstruct(F, conn)           # Eq. (7.4)
    I1 = H | F                                   # the M-file's `I1 = x10 | I0`
    I2 = H & ~F                                  # the M-file's `I2 = x10 & I` — the filled holes

    print(f"=== filling_reconstruct.m (Book section 7.1.3.2, Eqs. (7.3)-(7.4), Fig. 7.8) — "
          f"I0 = {args.i0}, SE = {args.se} ({conn}-connectivity) ===")
    print(f"F: {F.shape[0]}x{F.shape[1]}, {int(F.sum())} object px; F^c {int((~F).sum())} px")
    print(f"Eq. (7.3) marker F_m: {int(Fm.sum())} px "
          f"(== the M-file's hard-coded x0: {np.array_equal(Fm, FIG_7_8_MARKER) if args.i0 == 'shipped' else 'n/a'})")
    print("reconstruction block sums [F_m (+) B, D^(1), D^(2), ...]:", [int(b.sum()) for b in r.blocks])
    print(f"fixed point D^({r.n_iter - 1}) (D^({r.n_iter}) == D^({r.n_iter - 1})); "
          f"H = [R^D_(F^c)(F_m)]^c has {int(H.sum())} px, "
          f"of which {int(I2.sum())} are filled holes; I1 = H | F has {int(I1.sum())} px")
    print(f"H == imfill(F, 'holes'): {np.array_equal(H, imfill(F, 'hole', conn=conn))}")
    print(f"H == scipy.ndimage.binary_fill_holes(F): "
          f"{np.array_equal(H, ndi.binary_fill_holes(F, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) if conn == 4 else np.ones((3, 3))))}")
    print(f"imfill dtype round-trip — logical in -> {imfill(F).dtype}, double in -> "
          f"{imfill(F.astype(np.float64), 'hole').dtype} (the class ice_shape_enhancement.m depends on)")

    if args.i0 == "shipped" and args.se == "diamond1":
        blocks = list(r.blocks[:5]) + [H, I2]
        diffs = [int((a != b).sum()) for a, b in zip(FIG_7_8_STEPS, blocks)]
        print(f"vs the 7 printed blocks of Fig. 7.8(e): {diffs} pixels differ")

    n = len(r.blocks)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(10, 2.4 * ((n + 1) // 2)))
    for j, ax in enumerate(np.asarray(axes).ravel()):
        if j >= n:
            ax.axis("off")
            continue
        title = "F$_m$ $\\oplus$ B" if j == 0 else f"D$^{{({j})}}$"
        show_matrix(ax, r.blocks[j].astype(int), fmt="{:d}", title=title, fontsize=7)
    book_case = args.i0 == "shipped" and args.se == "diamond1"   # only this one is the printed Fig. 7.8
    tag = f"{args.i0}_{args.se}"
    fig.suptitle(f"Fig. 7.8(e) — Eq. (7.4) reconstruction-based hole filling ({tag})")
    name = "fig_7_08_filling_reconstruct.png" if book_case else f"sec_7_1_3_2_reconstruct_{tag}.png"
    written = [finish_figure(fig, out / name, args.show)]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    show_matrix(axes[0], F.astype(int), fmt="{:d}", title="(b) F — binary image")
    show_matrix(axes[1], (~F).astype(int), fmt="{:d}", title="(c) F$^c$ — complement")
    show_matrix(axes[2], Fm.astype(int), fmt="{:d}", title="(d) F$_m$ — Eq. (7.3) marker")
    show_matrix(axes[3], H.astype(int), fmt="{:d}", title=f"H — Eq. (7.4) ({int(H.sum())} px)")
    show_matrix(axes[4], I2.astype(int), fmt="{:d}", title=f"H $\\cap$ F$^c$ — filled holes ({int(I2.sum())} px)")
    fig.suptitle("Fig. 7.8(b)-(d) and the final result")
    written.append(finish_figure(fig, out / f"sec_7_1_3_2_panels_{tag}.png", args.show))
    written.append(save_image(out / f"sec_7_1_3_2_filled_{tag}.png", H))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
