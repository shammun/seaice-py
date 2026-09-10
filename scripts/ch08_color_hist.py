"""Port of ``MATLAB_ROOT/ch{6,7}/Sea_Ice_Floe_Identification/color_hist.m`` — Figures 8.11 and 8.14.

The **same** M-file draws both histograms: run it on ``ice_floe`` (the identified pieces, areas in pixels) for
Figure 8.11 and on ``floe`` (the ``sea_ice_model`` polygons) for Figure 8.14::

    [z, n] = hist(floe_area, min_x : inter : max_x);      % min_x = 20, inter = 70, max_x = 3500
    nbins  = length(z);   n = n + inter/2;                % the bars sit 35 to the right of the hist centres
    color(i) = fix( (1 - exp(-n(i)/1000)) * 10000 );      % Eq. (7.6) on the SHIFTED centre
    nn = 8;  d = fix((color_max-color_min)/nn);  ysh = color_min : d : color_max;
    YT{1,i} = -round(1000 * log(1 - ysh(i)/10000));       % Eq. (7.6) inverted = the tick labels

Lines 23-25 and 31-33 colour the individual bars through **HG1** handle graphics
(``get(h,'Children')`` -> ``'Faces'`` -> ``'FaceVertexCData'``), which errors in R2025a; the numbers they consume
are all computed by ``seaice.ch08_applications.color_hist`` and the drawing is done here with matplotlib
(the same ``jet`` colour at the same ``color(i)``).

The printed tick list of Figures 8.11/8.14/8.15 (20, 149, 297, 471, 682, 950, 1317, 1902, 3487) follows from the
three literals ``min_x = 20``, ``max_x = 3500``, ``nn = 8`` **alone** and says nothing about the data
(analysis/ch08.md risk R10) - it is printed here as arithmetic evidence only.

Usage: ``python scripts/ch08_color_hist.py [--source raw|polygon|both] [--input iceimage|demo]
[--min-x 20] [--inter 70] [--max-x 3500] [--nn 8] [--data data/book/ch08] [--out outputs/ch08] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch08_applications import color_hist, iceimage_to_pieces, sea_ice_field, sea_ice_model  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.icestruct import load_iceimage_mat  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure  # noqa: E402

CH = "ch08"


def load_areas(args, data: Path, out: Path):
    """Return ``(raw_areas, polygon_areas, label)`` — the two inputs ``color_hist.m`` can be given."""
    if args.input == "iceimage":
        hits = sorted(data.rglob(args.mat))
        if not hits:
            return None, None, None
        ice = load_iceimage_mat(hits[0])
        raw = np.array([f.Area for f in ice.Floe], dtype=np.float64)
        poly = np.array([f.Polygon.Area for f in ice.Floe], dtype=np.float64)
        return raw, poly, f"the shipped IceImage ({hits[0].name}, section 8.3 field)"

    for chapter, folder in (("ch08", data), ("ch07", None)):
        try:
            rgb, _ = load_image(chapter, args.image, allow_fallback=False, data_dir=folder, verbose=False)
            break
        except FileNotFoundError:
            rgb = None
    if rgb is None:
        return None, None, None
    enh = sea_ice_field(rgb, cache=out / f"stage3_{Path(args.image).stem}_{rgb.shape[0]}x{rgb.shape[1]}.npz")
    m = sea_ice_model(enh.ice_floe, enh.brash_ice, enh.index_floe, raster=False)
    raw = np.array([p.Area for p in enh.ice_floe], dtype=np.float64)
    poly = np.array([f.Area for f in m.floe], dtype=np.float64)
    return raw, poly, f"the section 8.2 pipeline on {args.image} ({len(enh.ice_floe)} floes)"


def draw(ax, ch, title: str) -> None:
    """``bar(n(1:nbins), z(1:nbins))`` with each bar coloured ``jet`` at ``color(i)`` (the HG1 block, redrawn)."""
    cmap = plt.get_cmap("jet")
    c = ch.color.astype(float)
    lo, hi = float(ch.color_min), float(ch.color_max)
    norm = np.clip((c - lo) / max(hi - lo, 1.0), 0, 1)
    ax.bar(ch.n, ch.z, width=ch.params["inter"], color=cmap(norm), edgecolor="k", linewidth=0.2)
    ax.set_xlabel("floe size [pixels]", fontsize=12)
    ax.set_ylabel("number of ice floes", fontsize=12)
    ax.set_title(title)
    ax.margins(x=0.01)


def add_colorbar(fig, ax, ch) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    sm = ScalarMappable(norm=Normalize(ch.color_min, ch.color_max), cmap=plt.get_cmap("jet"))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    cb.set_ticks(np.linspace(ch.color_min, ch.color_max, ch.tick_values.size))
    cb.set_ticklabels([str(int(v)) for v in ch.tick_labels])
    cb.set_label("floe size [pixels]")


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--source", default="both", choices=["raw", "polygon", "both"],
                   help="raw = Figure 8.11 (ice_floe.Area), polygon = Figure 8.14 (floe.Area)")
    p.add_argument("--input", default="demo", choices=["demo", "iceimage"],
                   help="demo = the section 8.2 field of Figs. 8.11/8.14; iceimage = the section 8.3 field")
    p.add_argument("--mat", default="IceImage_290915_2_jpg.0000179.mat")
    p.add_argument("--image", default="sea_ice_test.jpg")
    p.add_argument("--min-x", type=float, default=20, help="color_hist.m line 5")
    p.add_argument("--inter", type=float, default=70, help="line 6")
    p.add_argument("--max-x", type=float, default=3500, help="line 7")
    p.add_argument("--nn", type=int, default=8, help="line 36")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    raw, poly, label = load_areas(args, data, out)
    if raw is None:
        print(f"SKIP {Path(__file__).name}: private data absent (need {args.mat} under {data.as_posix()} "
              "or sea_ice_test.jpg under data/book/ch07)")
        return 0

    print(f"=== color_hist.m on {label} ===")
    print(f"  min_x = {args.min_x:g}, inter = {args.inter:g}, max_x = {args.max_x:g}, nn = {args.nn}")

    written = []
    jobs = []
    if args.source in ("raw", "both"):
        jobs.append(("raw", raw, "Figure 8.11" if args.input == "demo" else "Section 8.3 field",
                     "fig_8_11_floe_size_histogram" if args.input == "demo" else "sec_8_3_floe_size_histogram",
                     "identified ice floes (pixel areas)"))
    if args.source in ("polygon", "both"):
        jobs.append(("polygon", poly, "Figure 8.14" if args.input == "demo" else "Section 8.3 field",
                     "fig_8_14_polygon_size_histogram" if args.input == "demo"
                     else "sec_8_3_polygon_size_histogram",
                     "polygonized ice floes (sea_ice_model areas)"))

    for tag, areas, figno, fname, what in jobs:
        ch = color_hist(areas, min_x=args.min_x, inter=args.inter, max_x=args.max_x, nn=args.nn)
        print(f"  [{tag}] {areas.size} floes, areas {areas.min():.1f} .. {areas.max():.1f} px")
        print(f"        hist centres {ch.centers[0]:g} .. {ch.centers[-1]:g} ({ch.nbins} bins); "
              f"shifted bar positions n + inter/2 = {ch.n[0]:g} .. {ch.n[-1]:g}")
        print(f"        z sums to {int(ch.z.sum())} (the two outer bins of the centres form are UNBOUNDED, so "
              "nothing is dropped)")
        print(f"        z[:8] = {ch.z[:8].astype(int).tolist()}, argmax bin {int(ch.z.argmax())} "
              f"({int(ch.z.max())} floes at {ch.n[int(ch.z.argmax())]:g} px)")
        print(f"        Eq. (7.6) colours of the shifted centres: {ch.color[:5].tolist()} ... {ch.color[-1]}")
        print(f"        colour bar (constants-only, risk R10): ysh = {ch.tick_values.astype(int).tolist()}")
        print(f"        tick labels YT = {ch.tick_labels.tolist()}")
        if abs(args.min_x - 20) < 1e-9 and abs(args.max_x - 3500) < 1e-9 and args.nn == 8:
            book = [20, 149, 297, 471, 682, 950, 1317, 1902, 3487]
            print(f"        book Figs. 8.11/8.14/8.15: {book} -> "
                  f"{'MATCH' if ch.tick_labels.tolist() == book else 'DIFFERS'}")

        fig, ax = plt.subplots(figsize=(10, 5))
        draw(ax, ch, f"{figno} - floe size distribution of the {what}")
        add_colorbar(fig, ax, ch)
        written.append(finish_figure(fig, out / f"{fname}.png", args.show))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
