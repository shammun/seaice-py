"""Port of ``MATLAB_ROOT/ch7/cleaning & labeling & filling/labeling.m`` — Book §7.1.2, Eq. (7.1), Figs. 7.3/7.4.

Connected-component extraction by **constrained (geodesic) dilation**::

    X_k = (X_{k-1} (+) B) n A,   k = 1, 2, 3, ...,   X_0 = {p}

stopped when ``X_k == X_{k-1}``.  ``--se square3`` is the 3x3 square (8-connectivity, Fig. 7.3, the M-file's
line 23) and ``--se diamond1`` the cross (4-connectivity, Fig. 7.4, the commented line 24).

Usage: ``python scripts/ch07_labeling.py [--se square3|diamond1] [--out outputs/ch07] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch07_ice_type import connected_component_extract  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.connectivity import label_components  # noqa: E402
from seaice.core.plotting import finish_figure, save_image, show_matrix  # noqa: E402
from seaice.core.synth import FIG_7_3_IMAGE, FIG_7_3_SEED, FIG_7_3_STEPS, FIG_7_4_STEPS  # noqa: E402

CH = "ch07"
#: Which book figure each SE produces, and the printed blocks to compare against.
CASES = {"square3": ("7.3", FIG_7_3_STEPS, 8), "diamond1": ("7.4", FIG_7_4_STEPS, 4)}


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--se", default="square3", choices=sorted(CASES),
                   help="labeling.m line 23 (square3 = Fig. 7.3) or the commented line 24 (diamond1 = Fig. 7.4)")
    p.add_argument("--max-iter", type=int, default=9, help="the M-file unrolls the recursion to x9")
    args = p.parse_args(argv)
    _, out = resolve_dirs(args)

    fig_no, printed, conn = CASES[args.se]
    I, x0 = FIG_7_3_IMAGE, FIG_7_3_SEED
    r = connected_component_extract(I, x0, args.se, max_iter=args.max_iter)

    seed_rc = tuple(int(v) for v in np.argwhere(x0)[0])
    print(f"=== labeling.m (Book section 7.1.2, Eq. (7.1), Fig. {fig_no}) — SE = {args.se} "
          f"({conn}-connectivity) ===")
    print(f"image {I.shape[0]}x{I.shape[1]}, {int(I.sum())} object pixels; "
          f"seed p at 0-based {seed_rc} (1-based {(seed_rc[0] + 1, seed_rc[1] + 1)})")
    print("block sums  [X_0 (+) B, X_1, X_2, ...]:", [int(b.sum()) for b in r.blocks])
    print(f"fixed point X_{r.n_iter - 1} (X_{r.n_iter} == X_{r.n_iter - 1}, converged = {r.converged}) — "
          f"the {r.n_iter}th printed block; the component has {int(r.component.sum())} pixels")

    L = label_components(I, conn)
    same = np.array_equal(r.component, L == L[seed_rc])
    print(f"cross-check: equals the bwlabel(I, {conn}) component containing p -> {same}")
    diffs = [int((a != b).sum()) for a, b in zip(printed, r.blocks)]
    print(f"vs the {len(printed)} printed blocks of Fig. {fig_no}(d): {diffs} pixels differ")

    n = len(printed)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(9, 2.2 * ((n + 1) // 2)))
    for j, ax in enumerate(np.asarray(axes).ravel()):
        if j >= n:
            ax.axis("off")
            continue
        title = "X_0 $\\oplus$ B" if j == 0 else f"X_{j}"
        show_matrix(ax, r.blocks[j].astype(int), fmt="{:d}", title=title, fontsize=7)
    fig.suptitle(f"Fig. {fig_no}(d) — Eq. (7.1) constrained dilation, SE = {args.se} ({conn}-connectivity)")
    written = [finish_figure(fig, out / f"fig_7_{fig_no.split('.')[1]:0>2}_labeling_{args.se}.png", args.show)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    show_matrix(axes[0], I.astype(int), fmt="{:d}", title="(b) A — binary image matrix")
    show_matrix(axes[1], x0.astype(int), fmt="{:d}", title="(c) X_0 = {p} — initial step")
    show_matrix(axes[2], r.component.astype(int), fmt="{:d}",
                title=f"extracted component ({int(r.component.sum())} px)")
    fig.suptitle(f"Fig. {fig_no}(a)-(c) — input, seed and the extracted connected component")
    written.append(finish_figure(fig, out / f"sec_7_1_2_input_{args.se}.png", args.show))
    written.append(save_image(out / f"sec_7_1_2_component_{args.se}.png", r.component))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
