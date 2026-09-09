"""Port of ``MATLAB_ROOT/ch5/watershed_based/main.m`` — watershed segmentation followed by the authors'
neighbouring-region merging (§5.2, Fig. 5.13 flow chart, **Fig. 5.14(b)–(i)**, Fig. 5.15 kernel).

Steps (literal): Otsu mask of the RGB ``q.jpg`` (b); ``D = bwdist(~bw, 'cityblock')`` (c); ``L = watershed(-D);
w = L == 0`` (d); ``seg = bw & ~w`` = over-segmented floes (e); junction lines ``f = bitand(bw, w)`` with their
ending points from the kernel ``[0 -1 0; -1 4 -1; 0 -1 0]`` (f); for every 4-connected line: reconstruct the union
of the line and its neighbouring regions, find its concave boundary points (differential chain code), and delete
the line — merge the regions — when neither ending point is concave (g)(h); final result (i).

Usage: ``python scripts/ch05_main.py [--metric cityblock] [--endpoint-rule max|ge3] [--no-sequential]
[--data data/book/ch05] [--out outputs/ch05] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch05_watershed import BOOK_PARAMS, neighboring_region_merging, otsu_mask  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, label2rgb, save_image  # noqa: E402
from seaice.core.connectivity import label_components  # noqa: E402

CH = "ch05"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--metric", default=BOOK_PARAMS["metric_merging"],
                   choices=["cityblock", "chessboard", "euclidean", "quasi-euclidean"], help="bwdist metric (script: cityblock)")
    p.add_argument("--endpoint-rule", default="max", choices=["max", "ge3"],
                   help="'max' = the script (abs(imfilter) >= max), 'ge3' = the text (>= 3)")
    p.add_argument("--sequential", action="store_true", default=True,
                   help="update seg inside the loop (the script; default)")
    p.add_argument("--no-sequential", dest="sequential", action="store_false",
                   help="evaluate every line against the original over-segmentation (the text's description)")
    p.add_argument("--image", default="q.jpg")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        rgb, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: skipped: private image absent ({exc})")
        return 0
    bw = otsu_mask(rgb)  # Step 1: img = im2bw(img, graythresh(img)) on the RGB
    res = neighboring_region_merging(bw, args.metric, args.endpoint_rule, args.sequential)
    M, N = bw.shape
    literal = args.metric == "cityblock" and args.endpoint_rule == "max" and args.sequential
    pre = "fig_5_14" if literal else f"sec_5_2_{args.metric}_{args.endpoint_rule}" + ("" if args.sequential else "_nonseq") + "_"
    print(f"main.m on {args.image} ({M}x{N}): {int(bw.sum())} ice px (Otsu on RGB); metric '{args.metric}', "
          f"endpoint rule '{args.endpoint_rule}', sequential = {args.sequential}")
    print(f"D = bwdist(~bw): max {float(res.D.max()):g}; watershed(-D): {int(res.L.max())} basins, "
          f"ridge w {int(res.w.sum())} px; junction lines f = bw & w: {int(res.f.sum())} px in {res.num} 4-connected lines")
    print(f"over-segmented seg0: {res.n_floes_before} regions")
    for ln in res.lines:
        ep = "; ".join(f"({r + 1}, {c + 1})" for r, c in ln.endpoints.tolist())
        cep = "; ".join(f"({r + 1}, {c + 1})" for r, c in ln.concave_endpoints.tolist()) or "none"
        n_reg = int(np.unique(ln.region[ln.region > 0]).size)
        print(f"line {ln.label}: {ln.n_pixels} px; ending points {ep}; merged region label {int(ln.region.max())} "
              f"({int((ln.region > 0).sum())} px, {n_reg} label value) with {ln.concave.shape[0]} concave boundary points; "
              f"concave ending points: {cep} -> {'REMOVED (regions merged)' if ln.removed else 'kept'}")
    print(f"result: {res.n_removed} of {res.num} lines removed; floes {res.n_floes_before} -> {res.n_floes_after}  "
          f"(book Fig. 5.14: 3 lines, 2 removed, 4 -> 2)")

    written = [save_image(out / f"{pre}b_binary.png", bw),
               save_image(out / f"{pre}c_inverse_{args.metric}_dt.png", -res.D, autoscale=True),
               save_image(out / f"{pre}d_watershed_lines.png", res.w),
               save_image(out / f"{pre}e_oversegmented.png", res.seg0),
               save_image(out / f"{pre}f_junction_lines.png", res.f),
               save_image(out / f"{pre}i_final.png", res.seg)]
    # (f) junction lines with their ending points (red), as MATLAB's plot(y, x, 'r.')
    fig, ax = plt.subplots(figsize=(6, 7))
    imshow_matlab(ax, res.f, title=f"(f) f = bitand(bw, w): {res.num} junction lines + ending points")
    for ln in res.lines:
        ax.plot(ln.endpoints[:, 1], ln.endpoints[:, 0], "r.", ms=9)
        ax.text(ln.pixels[:, 1].mean() + 2, ln.pixels[:, 0].mean(), str(ln.label), color="y", fontsize=9)
    written.append(finish_figure(fig, out / f"{pre}f_junction_lines_endpoints.png", args.show))
    # (g)/(h) per line: region with concave points and ending points, and the segmentation after the decision
    seg_running = res.seg0.copy()
    for ln in res.lines:
        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        imshow_matlab(axes[0], ln.region > 0, title=f"(g) line {ln.label}: reconstructed neighbouring region")
        axes[0].plot(ln.concave[:, 1], ln.concave[:, 0], "c.", ms=7, label="concave boundary points")
        axes[0].plot(ln.endpoints[:, 1], ln.endpoints[:, 0], "r+", ms=10, mew=1.5, label="ending points")
        axes[0].plot(ln.pixels[:, 1], ln.pixels[:, 0], "y.", ms=3, label="junction line")
        axes[0].legend(loc="lower right", fontsize=7)
        if ln.removed:
            seg_running[ln.pixels[:, 0], ln.pixels[:, 1]] = True
        imshow_matlab(axes[1], label2rgb(label_components(seg_running)),
                      title=f"(h) after line {ln.label}: {'removed (merged)' if ln.removed else 'kept'} → "
                            f"{int(label_components(seg_running).max())} regions")
        written.append(finish_figure(fig, out / f"{pre}g_h_line_{ln.label}_{'removed' if ln.removed else 'kept'}.png", args.show))
    fig, axes = plt.subplots(2, 4, figsize=(17, 10))
    imshow_matlab(axes[0, 0], bw, title="(b) Otsu mask")
    imshow_matlab(axes[0, 1], -res.D, autoscale=True, title=f"(c) -bwdist(~bw, '{args.metric}')")
    imshow_matlab(axes[0, 2], res.w, title="(d) watershed lines")
    imshow_matlab(axes[0, 3], label2rgb(label_components(res.seg0)), title=f"(e) over-segmented: {res.n_floes_before} regions")
    imshow_matlab(axes[1, 0], res.f, title="(f) junction lines + ending points")
    for ln in res.lines:
        axes[1, 0].plot(ln.endpoints[:, 1], ln.endpoints[:, 0], "r.", ms=8)
    imshow_matlab(axes[1, 1], rgb, title="(a) q.jpg")
    imshow_matlab(axes[1, 2], res.seg, title=f"(i) final: {res.n_floes_after} floes")
    imshow_matlab(axes[1, 3], label2rgb(label_components(res.seg)), title="(i) label2rgb(bwlabel(seg))")
    fig.suptitle("Fig. 5.14  Neighbouring-region merging (watershed_based/main.m)")
    written.append(finish_figure(fig, out / f"{pre.rstrip('_')}_panels.png", args.show))
    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
