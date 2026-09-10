"""Port of ``MATLAB_ROOT/ch{6,7}/Sea_Ice_Floe_Identification/color_hist_comparison.m`` — Figure 8.15.

"Floe size distribution error ... due to the shape simplification" (p. 188): the polygonized floe-size histogram
minus the identified (pixel-area) one, **on the same bin centres**::

    [z , n ] = hist(floe_area , min_x : inter : max_x);   % floe      = sea_ice_model polygons
    [z0, n0] = hist(floe_area0, min_x : inter : max_x);   % ice_floe  = identified pieces
    z_d = z - z0;

Two things to keep straight:

* this is **not** ``main_WL_new.m``'s ``count_error``, which shifts the raw histogram by one bin
  (analysis/ch08.md C1 / risk R20); that one is saved as ``sec_8_3_delta_n_mcd.png`` by
  ``scripts/ch08_main_WL_new.py``;
* the shipped file sets ``max_x = 6000`` (line 10) where ``color_hist.m`` sets 3500, but the colour bar printed
  under Figure 8.15 is the ``max_x = 3500`` one (20 ... 3487).  The figure was therefore made with the other
  file's constant - a shipped-code / printed-figure discrepancy (risk R9).  The default here is the **shipped**
  6000; ``--max-x 3500`` reproduces the printed colour bar.

Usage: ``python scripts/ch08_color_hist_comparison.py [--max-x 6000|3500] [--input demo|iceimage]
[--data data/book/ch08] [--out outputs/ch08] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ch08_color_hist import add_colorbar, load_areas  # noqa: E402
from seaice.ch08_applications import color_hist_comparison  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.plotting import finish_figure  # noqa: E402

CH = "ch08"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--input", default="demo", choices=["demo", "iceimage"])
    p.add_argument("--mat", default="IceImage_290915_2_jpg.0000179.mat")
    p.add_argument("--image", default="sea_ice_test.jpg")
    p.add_argument("--min-x", type=float, default=20, help="color_hist_comparison.m line 8")
    p.add_argument("--inter", type=float, default=70, help="line 9")
    p.add_argument("--max-x", type=float, default=6000, help="line 10 (the SHIPPED value; the figure used 3500)")
    p.add_argument("--nn", type=int, default=8, help="line 45")
    p.add_argument("--source", default="both")   # unused; keeps load_areas' signature happy
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    raw, poly, label = load_areas(args, data, out)
    if raw is None:
        print(f"SKIP {Path(__file__).name}: private data absent (need {args.mat} under {data.as_posix()} "
              "or sea_ice_test.jpg under data/book/ch07)")
        return 0

    c = color_hist_comparison(poly, raw, min_x=args.min_x, inter=args.inter, max_x=args.max_x, nn=args.nn)
    print(f"=== color_hist_comparison.m on {label} ===")
    print(f"  min_x = {args.min_x:g}, inter = {args.inter:g}, max_x = {args.max_x:g} "
          f"({'the SHIPPED value' if args.max_x == 6000 else 'color_hist.m value = the printed colour bar'}), "
          f"nn = {args.nn}")
    print(f"  {c.nbins} bins, centres {c.centers[0]:g} .. {c.centers[-1]:g}; bars at n + inter/2 = "
          f"{c.n[0]:g} .. {c.n[-1]:g}")
    print(f"  z  (polygon areas) sums to {int(c.z.sum())}, z0 (pixel areas) sums to {int(c.z0.sum())}")
    print(f"  z_d = z - z0: range {int(c.z_d.min())} .. {int(c.z_d.max())}, sum {int(c.z_d.sum())} "
          "(zero by construction - both histograms count every floe)")
    print(f"  z_d[:10] = {c.z_d[:10].astype(int).tolist()}")
    print(f"  colour bar: ysh = {c.tick_values.astype(int).tolist()}")
    print(f"  tick labels YT = {c.tick_labels.tolist()}")
    book = [20, 149, 297, 471, 682, 950, 1317, 1902, 3487]
    print(f"  book Fig. 8.15 prints {book} -> "
          f"{'MATCH' if c.tick_labels.tolist() == book else 'DIFFERS (this is the max_x = 6000 list; the printed one needs --max-x 3500)'}")

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("jet")
    norm = np.clip((c.color.astype(float) - c.color_min) / max(c.color_max - c.color_min, 1), 0, 1)
    ax.bar(c.n, c.z_d, width=args.inter, color=cmap(norm), edgecolor="k", linewidth=0.2)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("floe size [pixels]", fontsize=12)
    ax.set_ylabel("difference in number of ice floes", fontsize=12)
    ax.set_title(f"Figure 8.15 - floe size distribution error ($z - z_0$), max_x = {args.max_x:g}")
    ax.margins(x=0.01)
    add_colorbar(fig, ax, c)
    suffix = "" if args.max_x == 6000 else f"_maxx{int(args.max_x)}"
    written = [finish_figure(fig, out / f"fig_8_15_error_histogram{suffix}.png", args.show)]

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
