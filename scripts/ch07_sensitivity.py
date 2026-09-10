"""Book §7.3.2 "A preliminary sensitivity study" (pp. 168-174) — Figs. 7.24 and 7.27.

Two one-at-a-time sweeps over the 394x1038 marginal-ice-zone image of **Fig. 7.22** (= ``sea_ice_test.jpg``),
counting the ice floes and brash ice pieces that Algorithms 3 -> 4 -> 5 identify:

* **§7.3.2.1** the upper limit on the **snake's** evolution iterations is varied **1 -> 122** while the GVF field
  is fixed at **500** iterations (Fig. 7.24, y axis 0...1600);
* **§7.3.2.2** the number of **GVF** field iterations is varied **1 -> 1500 in steps of 20** while the snake
  limit is fixed at **100** (Fig. 7.27, y axis 0...2000).

**This script is `unverified` by construction.**  The book prints **no counts at all** for these two figures —
only the axis ranges and the tick positions (Fig. 7.24 x ticks 1, 6, 11, ..., 121; Fig. 7.27 x ticks 1, 61, 121,
..., 1441) — so there is nothing to compare a number against.  What *can* be reproduced is the **shape** of the
curves the text describes: rising while more and more snakes reach the floe boundaries, a peak, then a fall to a
steady value once every snake converges (§7.3.2.1), and a high, noisy, over-segmented count at very few GVF
iterations settling as the field diffuses (§7.3.2.2), with the brash curves oscillating more than the floe ones
because "the boundaries of brash ice are weaker".  No number printed by this script is a book value.

By default the script runs only the **7 settings the book actually prints a figure for** (snake 1 / 10 / 70 from
Figs. 7.25 and 7.26(a)-(c), GVF 1 / 61 / 181 from Figs. 7.28(a)-(c), plus the drivers' own 500/100, which is
shared by both lists).  ``--full-sweep`` runs the book's full ranges (25 + 75 pipeline runs) and is slow.

Usage: ``python scripts/ch07_sensitivity.py [--param both|snake|gvf] [--values 1 10 70 100] [--full-sweep]
[--downscale 2] [--save-panels] [--data data/book/ch07] [--out outputs/ch07] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch06_gvf_snake import BOOK_PARAMS  # noqa: E402
from seaice.ch07_ice_type import colorbar_area_ticks, ice_shape_enhancement, sea_ice_edge_detection  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure, label2rgb, save_image  # noqa: E402

CH = "ch07"
P = BOOK_PARAMS["sea_ice_demo"]

#: The settings for which the book prints a panel: Figs. 7.25/7.26(a)-(c) and 7.28(a)-(c), plus the drivers'
#: own 500/100.  Seven distinct pipeline runs in total (the (Num 500, iter 100) point is in both lists).
BOOK_PANELS = {"snake": (1, 10, 70, 100), "gvf": (1, 61, 181, 500)}
#: The book's own sweep ranges (§7.3.2.1 "from 1 to 122", §7.3.2.2 "from 1 to 1500 at intervals of 20").  The
#: snake step is **not** stated in the text; 5 is this port's reading of Fig. 7.24's tick labels 1, 6, ..., 121.
FULL_SWEEP = {"snake": tuple(range(1, 123, 5)), "gvf": tuple(range(1, 1501, 20))}
#: The colour-bar tick integers printed under the panels (pp. 171, 173).  Printed for reference only — the port
#: reproduces the *procedure*, and the chapter's pipeline is `near`, so these are NOT a parity target.
BOOK_TICKS = {("snake", 1): (2, 184, 407, 695, 1100, 1792, 8112),
              ("snake", 10): (2, 165, 359, 601, 921, 1394, 2322),
              ("snake", 70): (2, 173, 378, 638, 990, 1537, 2839),
              ("gvf", 1): (2, 178, 391, 663, 1038, 1645, 3439),
              ("gvf", 61): (2, 165, 361, 604, 926, 1404, 2353),
              ("gvf", 181): (2, 165, 361, 605, 927, 1406, 2359)}
#: The y-axis ranges the book prints (the only quantitative information in Figs. 7.24 and 7.27).
BOOK_YLIM = {"snake": (0, 1600), "gvf": (0, 2000)}


def one_run(rgb, num: int, iters: int):
    """Algorithms 3 -> 4 -> 5 once, returning ``(n_floe, n_brash, seconds, IceShapeEnhancement)``."""
    t0 = time.time()
    g = sea_ice_edge_detection(rgb, kms0=P["kms0"], sigma=P["sigma"], GradientOn=P["GradientOn"],
                               GVFOn=P["GVFOn"], Num=num, mu=P["mu"], iter=iters, alpha=P["alpha"],
                               beta=P["beta"], gamma=P["gamma"], kappa=P["kappa"], Dmin=P["Dmin"],
                               Dmax=P["Dmax"], Ra_min=P["Ra_min"], Ra=P["Ra"], Rc=P["Rc"], Rl=P["Rl"],
                               se_radius=P["se_radius"], timer=P["timer"], keep_history=False)
    e = ice_shape_enhancement(g.bk, g.out, min_floe=P["min_floe"], min_brash=P["min_brash"], se_th=P["se_th"],
                              nbins=0)
    return len(e.ice_floe), len(e.brash_ice), time.time() - t0, e


def sweep(rgb, param: str, values, out: Path, save_panels: bool, show: bool):
    """Run one sweep and return the per-value records."""
    fixed = P["Num"] if param == "snake" else P["iter"]
    fixed_name = "GVF iterations" if param == "snake" else "snake iterations"
    print(f"\n--- section 7.3.2.{1 if param == 'snake' else 2}: sweeping the "
          f"{'snake' if param == 'snake' else 'GVF'} iterations over {list(values)} "
          f"with {fixed_name} fixed at {fixed} ---")
    rows = []
    for v in values:
        num, iters = (fixed, v) if param == "snake" else (v, fixed)
        n_floe, n_brash, dt, e = one_run(rgb, num, iters)
        colours = np.concatenate([e.color_floe, e.color_brash]) if (e.color_floe.size or e.color_brash.size) \
            else np.zeros(0)
        _, ticks = colorbar_area_ticks(colours, 6)
        rows.append((int(v), n_floe, n_brash, n_floe + n_brash, dt, ticks.tolist()))
        book = BOOK_TICKS.get((param, int(v)))
        # Indicative only: the pipeline is `near` (ch6) and the default run is downscaled, so agreement or
        # disagreement with the printed panel ticks is NOT a parity measurement.
        extra = f"   [book panel ticks {list(book)} — indicative only]" if book else ""
        print(f"  {param} = {v:>5d} (Num {num}, iter {iters}): floe {n_floe:5d}, brash {n_brash:5d}, "
              f"total {n_floe + n_brash:5d}  ({dt:5.1f} s)  ticks {ticks.tolist()}{extra}")
        if save_panels and book is not None:
            save_image(out / f"sec_7_3_2_{param}_{v}_identification.png",
                       label2rgb(e.index, cmap="jet", background=(1, 1, 1), shuffle=False))
    return rows


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="sea_ice_test.jpg", help="Fig. 7.22, the 394x1038 MIZ image")
    p.add_argument("--param", default="both", choices=["both", "snake", "gvf"])
    p.add_argument("--values", type=int, nargs="+", default=None,
                   help="explicit sweep values (overrides the defaults; requires --param snake or gvf)")
    p.add_argument("--full-sweep", action="store_true",
                   help="the book's own ranges: snake 1..121 step 5 (25 runs), GVF 1..1481 step 20 (75 runs)")
    p.add_argument("--downscale", type=int, default=2,
                   help="process every K-th row/column; 1 = the book's own resolution (default: %(default)s)")
    p.add_argument("--save-panels", action="store_true",
                   help="also save the identification image of each setting the book prints a panel for")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    if args.values is not None and args.param == "both":
        p.error("--values needs --param snake or --param gvf")

    try:
        rgb, label = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: private image absent ({exc})")
        return 0
    if args.downscale > 1:
        rgb = rgb[::args.downscale, ::args.downscale]

    params = ["snake", "gvf"] if args.param == "both" else [args.param]
    table = FULL_SWEEP if args.full_sweep else BOOK_PANELS
    plans = {q: (args.values if args.values is not None else table[q]) for q in params}

    print(f"=== section 7.3.2 sensitivity study on {args.image} [{label}] "
          f"{rgb.shape[0]}x{rgb.shape[1]} (downscale {args.downscale}) ===")
    print(f"mode: {'FULL SWEEP (the book ranges)' if args.full_sweep else 'the 7 settings the book prints a panel for'}"
          f"; {sum(len(v) for v in plans.values())} pipeline run(s)")
    print("UNVERIFIED BY CONSTRUCTION: the book prints no counts for Figs. 7.24/7.27, only the axis ranges "
          f"({', '.join(f'{q}: y {BOOK_YLIM[q]}' for q in params)}) and tick positions. Nothing below is a book "
          "number; only the SHAPE of the curves is comparable.")

    t0 = time.time()
    results = {q: sweep(rgb, q, plans[q], out, args.save_panels, args.show) for q in params}
    print(f"\ntotal {time.time() - t0:.1f} s")

    written = []
    for q, rows in results.items():
        x = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(x, [r[1] for r in rows], "o-", label="Floe")
        ax.plot(x, [r[2] for r in rows], "s-", label="Brash")
        ax.plot(x, [r[3] for r in rows], "^-", label="Floe+Brash")
        ax.set_xlabel("Snake iterations" if q == "snake" else "GVF iterations")
        ax.set_ylabel("Number of ice floes/brash ice")
        ax.set_xticks(x)
        ax.legend()
        ax.grid(alpha=0.3)
        fig_no = "7.24" if q == "snake" else "7.27"
        fixed = P["Num"] if q == "snake" else P["iter"]
        ax.set_title(f"procedure of Fig. {fig_no} — "
                     f"{'GVF' if q == 'snake' else 'snake'} iterations fixed at {fixed}, "
                     f"downscale {args.downscale} (book y axis {BOOK_YLIM[q]}; counts are NOT book values)")
        written.append(finish_figure(fig, out / f"sec_7_3_2_{q}_sensitivity.png", args.show))

    if len(results) == 2:
        print("\nshape of the curves (what §7.3.2 actually claims):")
        for q, rows in results.items():
            tot = [r[3] for r in rows]
            print(f"  {q}: totals {tot} -> "
                  f"{'rises then settles' if len(tot) > 2 and tot[0] < max(tot) and tot[-1] <= max(tot) else 'see the plot'}"
                  f"; extreme setting {rows[0][0]} gives {tot[0]}, the drivers' setting {rows[-1][0]} gives {tot[-1]}")
        print("  (§7.3.2.1: few snake iterations leave the initial contours in place -> over-segmentation that "
              "Algorithm 4 then merges into under-segmentation; §7.3.2.2: few GVF iterations give a small "
              "capture range so noise dominates -> over-segmentation, settling as the field diffuses.)")

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
