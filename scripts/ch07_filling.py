"""Port of ``MATLAB_ROOT/ch7/cleaning & labeling & filling/filling.m`` — Book §7.1.3.1, Eq. (7.2), Figs. 7.5-7.7.

Hole filling by constrained dilation of a seed **inside the hole**::

    X_k = (X_{k-1} (+) B) n A^c,    filled image = X_n u A

Three cases, all from the same 9x9 image family and the same seed at 1-based (3, 4):

* ``--image fig75 --se diamond1``  Fig. 7.5 — one 4-connected hole, filled (this is what ``filling.m`` runs);
* ``--image fig76 --se diamond1``  Fig. 7.6 — the same image **plus one pixel at 1-based (5, 5)**, which makes
  two 4-connected holes; the cross SE fills only the one containing the seed;
* ``--image fig76 --se square3``   Fig. 7.7 — the hole is 8-connected to the background, so the 3x3 square SE
  floods the whole background and nothing is filled.

The Fig. 7.6/7.7 image is printed in the book but is **not** in any shipped ``.m`` file; it lives in
``seaice.core.synth.FIG_7_6_IMAGE``.

Usage: ``python scripts/ch07_filling.py [--image fig75|fig76] [--se diamond1|square3] [--out outputs/ch07]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch07_ice_type import hole_fill_dilation, hole_fill_reconstruct  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.morphology import imfill  # noqa: E402
from seaice.core.plotting import finish_figure, save_image, show_matrix  # noqa: E402
from seaice.core.synth import (FIG_7_3_IMAGE, FIG_7_5_SEED, FIG_7_5_STEPS, FIG_7_6_IMAGE,  # noqa: E402
                               FIG_7_6_STEPS, FIG_7_7_STEPS, FIG_7_7_STEPS_BOOK, FIG_7_7_X8_TYPO)

CH = "ch07"
IMAGES = {"fig75": FIG_7_3_IMAGE, "fig76": FIG_7_6_IMAGE}
#: (book figure, printed blocks, whether the last printed block is the `X_n u A` union)
CASES = {("fig75", "diamond1"): ("7.5", FIG_7_5_STEPS, True),
         ("fig76", "diamond1"): ("7.6", FIG_7_6_STEPS, True),
         ("fig76", "square3"): ("7.7", FIG_7_7_STEPS, False)}


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="fig75", choices=sorted(IMAGES),
                   help="fig75 = filling.m's I0 (= labeling.m's image); fig76 = the same plus (5, 5) 1-based")
    p.add_argument("--se", default="diamond1", choices=["diamond1", "square3"],
                   help="filling.m line 23 uses strel('diamond', 1)")
    p.add_argument("--max-iter", type=int, default=10, help="the M-file unrolls the recursion to x9")
    args = p.parse_args(argv)
    _, out = resolve_dirs(args)

    I0 = IMAGES[args.image]
    r = hole_fill_dilation(I0, FIG_7_5_SEED, args.se, max_iter=args.max_iter)
    case = CASES.get((args.image, args.se))
    fig_no = case[0] if case else f"{args.image}/{args.se}"

    seed_rc = tuple(int(v) for v in np.argwhere(FIG_7_5_SEED)[0])
    print(f"=== filling.m (Book section 7.1.3.1, Eq. (7.2), Fig. {fig_no}) — image {args.image}, SE {args.se} ===")
    print(f"image {I0.shape[0]}x{I0.shape[1]}, {int(I0.sum())} object pixels, complement {int((~I0).sum())} px; "
          f"seed p at 1-based {(seed_rc[0] + 1, seed_rc[1] + 1)}")
    print("block sums  [X_0 (+) B, X_1, X_2, ...]:", [int(b.sum()) for b in r.blocks])
    print(f"fixed point X_{r.n_iter - 1} (X_{r.n_iter} == X_{r.n_iter - 1}, converged = {r.converged}); "
          f"X_n has {int(r.component.sum())} px, filled image X_n u A has {int(r.filled.sum())} px")

    auto = hole_fill_reconstruct(I0, 4)
    print(f"automatic filling (Eq. (7.4), section 7.1.3.2) would give {int(auto.sum())} px "
          f"(= imfill(A, 'holes'): {np.array_equal(auto, imfill(I0))})")
    if r.filled.sum() != auto.sum():
        print("  -> Eq. (7.2) does NOT reach the same result here: this is the failure case the book "
              f"illustrates with Fig. {fig_no}")

    if case:
        _, printed, has_union = case
        blocks = list(r.blocks[:len(printed) - 1]) + [r.filled] if has_union else list(r.blocks[:len(printed)])
        diffs = [int((a != b).sum()) for a, b in zip(printed, blocks)]
        print(f"vs the {len(printed)} printed blocks of Fig. {fig_no}(e): {diffs} pixels differ")
        if fig_no == "7.7":
            book = [int((a != b).sum()) for a, b in zip(FIG_7_7_STEPS_BOOK, blocks)]
            print(f"vs the blocks **exactly as printed** (with the X_8 typo): {book} pixels differ — the book "
                  f"prints 0 at 1-based {[(a + 1, b + 1) for a, b in FIG_7_7_X8_TYPO]} of X_8 (sum 53) where the "
                  f"recursion gives 1 (sum 55); X_7 already has those neighbours set and the printed X_9 = A^c "
                  f"agrees with the recursion, so this is a book typo")

    n = len(r.blocks)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(9, 2.2 * ((n + 1) // 2)))
    for j, ax in enumerate(np.asarray(axes).ravel()):
        if j >= n:
            ax.axis("off")
            continue
        title = "X_0 $\\oplus$ B" if j == 0 else f"X_{j}"
        show_matrix(ax, r.blocks[j].astype(int), fmt="{:d}", title=title, fontsize=7)
    fig.suptitle(f"Fig. {fig_no}(e) — Eq. (7.2) hole filling by constrained dilation, SE = {args.se}")
    tag = fig_no.replace(".", "_").replace("/", "_")
    written = [finish_figure(fig, out / f"fig_7_{fig_no.split('.')[1]:0>2}_filling_{args.image}_{args.se}.png"
                             if case else out / f"sec_7_1_3_filling_{tag}.png", args.show)]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    show_matrix(axes[0], I0.astype(int), fmt="{:d}", title="(b) A — binary image")
    show_matrix(axes[1], r.complement.astype(int), fmt="{:d}", title="(c) A$^c$ — complement image")
    show_matrix(axes[2], FIG_7_5_SEED.astype(int), fmt="{:d}", title="(d) X_0 = {p} — initial step")
    show_matrix(axes[3], r.filled.astype(int), fmt="{:d}", title=f"X_n $\\cup$ A ({int(r.filled.sum())} px)")
    fig.suptitle(f"Fig. {fig_no}(b)-(d) — input, complement, seed and the filled image")
    written.append(finish_figure(fig, out / f"sec_7_1_3_input_{tag}.png", args.show))
    written.append(save_image(out / f"sec_7_1_3_filled_{tag}.png", r.filled))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
