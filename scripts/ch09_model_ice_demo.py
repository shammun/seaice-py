"""Port of ``MATLAB_ROOT/ch9/Model_Ice_Floe_Identification/model_ice_demo.m`` — the chapter's only driver.

The M-file is a parameter block (lines 9–38, :data:`seaice.ch09_model_ice.BOOK_PARAMS_CH9`) followed by::

    bw1 = GVF_distance(I, sigma, GradientOn, GVFOn, Num, mu, iter, alpha, beta, gamma, kappa,
                       Dmin, Dmax, Ra_min, Ra, Rc, Rl, se, timer);   % Algorithm 7, timer = 2
    bw2 = bwareaopen(bw1, Ra_min);          bw3 = bwareaopen(bw2, Ra_min, 4);
    bw4 = imfill(bw3, 8, 'holes');
    % S = rect(bw4);  k1 = 0.4; k2 = 2.5;  s_model = model_ice_model(S, bw4, k1, k2);   <- lines 53-66, COMMENTED

``--full`` runs the commented block as well (the ch06 ``--full`` precedent).  ``GVF_distance.m`` here is
**byte-identical** (md5 ``39cca98a5e06ac8e4700caf586444fc9``) to ch6's and ch7's, so no new code is written for
it: :func:`seaice.ch06_gvf_snake.gvf_distance` is reused (rule 9).  Only the ch9 *parameters* are new —
``Ra_min`` 20 (ch6: 10), ``Ra`` 1000 (2500), ``Num`` 150 (500), ``iter`` 150 (100), ``kappa`` 0.6 (0.5),
``timer`` **2** (1), and **no k-means stage**.  ``timer = 2`` is the §9.3.1 remedy: when aligned square floes
touch, binarization leaves no hole between them, the distance-transform seeds merge and one pass cannot split
them (Fig. 9.11(a)–(c)), so contour initialization + segmentation is run **again** (Fig. 9.11(d)(e)).

``--stop count`` opts into Algorithm 7's own convergence test (lines 2/6/7/19 of p. 206), which the shipped
``GVF_distance.m`` computes (``num``) but never applies — **gap G1**.

DATA: ``model_ice.jpg`` **is** shipped (181 x 76 x 3).  Note (analysis §0.4) that it is **not a printed figure**
of the book: a whole-book NCC search over ~1800 embedded bitmaps peaked at 0.167 and a multi-scale template match
inside the chapter's pages at 0.373 (ch06's Fig. 6.15(a) match was 0.9882).  So every §9.3 figure this script
writes is a *procedural* reproduction, never a side-by-side figure match (risk R12).

Usage: ``python scripts/ch09_model_ice_demo.py [--image model_ice.jpg] [--full] [--num 150] [--iter 150]
[--timer 2] [--stop criteria|count] [--ratio ellipse|minrect] [--max-seeds N] [--downscale K]
[--data data/book/ch09] [--out outputs/ch09] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch06_gvf_snake import component_criteria  # noqa: E402
from seaice.ch09_model_ice import (BOOK_PARAMS_CH9, model_ice_demo, rect_ice_concentration)  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.connectivity import label_components  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, label2rgb, save_image  # noqa: E402
from seaice.core.threshold import ice_concentration  # noqa: E402

CH = "ch09"
P = BOOK_PARAMS_CH9


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="model_ice.jpg", help="model_ice_demo.m line 9")
    p.add_argument("--num", type=int, default=P["Num"], help="GVF iterations (script: 150)")
    p.add_argument("--iter", type=int, default=P["iter"], help="snake iterations (script: 150)")
    p.add_argument("--timer", type=int, default=P["timer"], help="outer re-segmentation passes (script: 2)")
    p.add_argument("--stop", default="criteria", choices=["criteria", "count"],
                   help="'criteria' = the shipped code; 'count' = Algorithm 7's own N0 != N1 test (gap G1)")
    p.add_argument("--ratio", default="ellipse", choices=["ellipse", "minrect"],
                   help="criterion 3: 'ellipse' = the code (default), 'minrect' = the book text (p. 205)")
    p.add_argument("--max-seeds", type=int, default=None, help="cap the contours per pass (runtime guard)")
    p.add_argument("--downscale", type=int, default=1, help="process every K-th row/column (runtime guard)")
    p.add_argument("--solver", default="auto", choices=["auto", "dense", "circulant"])
    p.add_argument("--full", action="store_true", help="also run lines 53-66 (rect + model_ice_model)")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    try:
        rgb, source = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: skipped: private image absent ({exc})")
        return 0
    if args.downscale > 1:
        rgb = rgb[::args.downscale, ::args.downscale]
        print(f"NOTE: --downscale {args.downscale} -> {rgb.shape[0]}x{rgb.shape[1]}")

    print(f"=== model_ice_demo.m on {args.image} ({rgb.shape[0]} x {rgb.shape[1]}), source: {source} ===")
    print("parameters (model_ice_demo.m lines 11-38): "
          + ", ".join(f"{k} = {v}" for k, v in P.items() if k not in ("k1", "k2")))
    print("NOTE: model_ice.jpg is NOT a printed figure of the book (analysis/ch09.md sec. 0.4) — the figures "
          "below are procedural reproductions, not side-by-side figure matches (risk R12).")

    t0 = time.time()
    d = model_ice_demo(rgb, full=args.full, max_seeds=args.max_seeds, solver=args.solver, stop=args.stop,
                       params=dict(Num=args.num, iter=args.iter, timer=args.timer))
    elapsed = time.time() - t0

    g = d.gvf
    print(f"\ngraythresh(rgb2gray(I)) = {g.level:.15f}  ->  255*level = {g.level * 255:.1f}")
    print(f"bw = im2bw(I, level): {int(g.bw.sum())} of {g.bw.size} px = {100 * g.bw.mean():.2f} % "
          f"(raw ice concentration before segmentation)")
    print(f"bwlabel(bw, 4) = {label_components(g.bw, 4).max()} components  <- exactly the sec. 9.3.1 problem: "
          "aligned squares that touch leave no hole, so the seeds merge")
    for rec in g.passes:
        tag = "stopped (no component fails the criteria)" if rec.stopped and not rec.stopped_on_count else \
            ("stopped (Algorithm 7 count convergence, gap G1)" if rec.stopped_on_count else "")
        print(f"  pass {rec.time}: {rec.num} components, {rec.k.size} failing the criteria"
              + (f", {len(rec.seeds)} contours run" if rec.init is not None else "") + (f"  [{tag}]" if tag else ""))
    print(f"seeds total {g.n_seeds} (run {g.n_seeds_run}); guards: {g.n_skipped} contour(s) skipped, "
          f"{g.n_below_range} burn point(s) below the image, {g.n_above_range} above it")
    print(f"bw1 = {int(d.bw1.sum())} px  ->  bw2 = bwareaopen(bw1, {P['Ra_min']}) = {int(d.bw2.sum())} px "
          f"({label_components(d.bw2, 8).max()} 8-comp)")
    print(f"bw3 = bwareaopen(bw2, {P['Ra_min']}, 4) = {int(d.bw3.sum())} px "
          f"({label_components(d.bw3, 4).max()} 4-comp)")
    print(f"bw4 = imfill(bw3, 8, 'holes') = {int(d.bw4.sum())} px "
          f"({label_components(d.bw4, 4).max()} 4-comp)")
    print(f"IC of the segmented image (the sec. 9.3.2 'boundary pixels turned to water' figure, book 76.96 %) "
          f"= {ice_concentration(d.bw4) * 100:.2f} %")
    print(f"wall clock {elapsed:.1f} s (Num = {args.num}, iter = {args.iter}, timer = {args.timer}, "
          f"solver = {args.solver}, stop = {args.stop})")

    if args.ratio == "minrect":
        _l, n, a, sol, maj, mino, rl, k = component_criteria(d.bw4, P["Ra"], P["Rc"], P["Rl"], 4,
                                                             ratio="minrect")
        _l2, _n2, _a2, _s2, _m2, _mi2, rl_e, k_e = component_criteria(d.bw4, P["Ra"], P["Rc"], P["Rl"], 4)
        print(f"\ncriterion 3 on bw4 — the book's minimum-bounding-RECTANGLE ratio (p. 205) vs the code's "
              f"ELLIPSE ratio (GVF_distance.m:90-95):")
        print(f"  minrect: {k.size} of {n} components fail; ratio range {rl.min():.4f} .. {rl.max():.4f}")
        print(f"  ellipse: {k_e.size} of {n} components fail; ratio range {rl_e.min():.4f} .. {rl_e.max():.4f}")
        print("  [DEVIATION: the 'minrect' branch is the book's TEXT; no shipped .m computes it, so it can "
              "never be used for a parity claim.]")

    written = [
        save_image(out / "fig_9_11_a_model_ice_gray.png", g.gray),
        save_image(out / "fig_9_11_b_binary.png", g.bw),
        save_image(out / "fig_9_11_c_pass1_labels.png",
                   label2rgb(label_components(g.passes[0].label > 0, 4))),
        save_image(out / "fig_9_11_d_segmented_bw1.png", d.bw1),
        save_image(out / "fig_9_11_e_cleaned_bw4.png", d.bw4),
        save_image(out / "fig_9_12_identification.png", label2rgb(label_components(d.bw4, 4))),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 11))
    imshow_matlab(axes[0, 0], g.gray, title="rgb2gray(I)")
    imshow_matlab(axes[0, 1], g.bw, title=f"bw = im2bw(I, {g.level * 255:.0f}/255)")
    imshow_matlab(axes[0, 2], d.bw1, title=f"bw1 = GVF_distance(..., timer = {args.timer})")
    imshow_matlab(axes[1, 0], d.bw2, title=f"bw2 = bwareaopen(bw1, {P['Ra_min']})")
    imshow_matlab(axes[1, 1], d.bw4, title="bw4 = imfill(bw3, 8, 'holes')")
    imshow_matlab(axes[1, 2], label2rgb(label_components(d.bw4, 4)),
                  title=f"{label_components(d.bw4, 4).max()} identified floes")
    fig.suptitle("model_ice_demo.m — Algorithm 7 on the crowded rectangular model-ice crop (Fig. 9.11)")
    written.append(finish_figure(fig, out / "fig_9_11_model_ice_demo.png", args.show))

    if args.full and d.S is not None and d.model is not None:
        S, mdl = d.S, d.model
        shape = d.bw4.shape
        print(f"\n--- lines 53-66 (commented out in the shipped file), run by --full ---")
        print(f"rect(bw4): {len(S)} minimum-area bounding rectangles; areas "
              f"{min(s.Area for s in S):.2f} .. {max(s.Area for s in S):.2f}, "
              f"sum {sum(s.Area for s in S):.2f}")
        print(f"  NOTE: RectFloe.Area/.Perimeter are the RECTANGLE's, not the pixel count "
              f"(bw4 has {int(d.bw4.sum())} ice px)")
        print(f"L/W ratios k = |v1-v2|/|v3-v2| (NOT normalised to >= 1, erratum E7): "
              f"{mdl.ratios.min():.4f} .. {mdl.ratios.max():.4f}")
        print(f"model_ice_model(S, bw4, k1 = {P['k1']}, k2 = {P['k2']}): {mdl.accepted.size} of {len(S)} "
              f"rectangles accepted ({(mdl.ratios <= P['k1']).sum()} below k1, "
              f"{(mdl.ratios >= P['k2']).sum()} above k2)")
        n_inter = sum(f.Intersection.size for f in mdl.s_model)
        print(f"overlap flags (polybool intersection, `if xx ~= NaN` = `~isempty`): {n_inter} entries over "
              f"{len(mdl.s_model)} floes")
        print(f"bw (raster union of the accepted rectangles) = {int(mdl.bw.sum())} px")
        print(f"\nthe three ice concentrations the book compares on p. 208 (book: 76.96 / 83.17 / 87.75 %):")
        print(f"  segmentation (bw4)                  = {ice_concentration(d.bw4) * 100:.2f} %")
        print(f"  global Otsu (bw)                    = {ice_concentration(g.bw) * 100:.2f} %")
        print(f"  sum of ALL rectangle areas / M*N    = "
              f"{sum(s.Area for s in S) / (shape[0] * shape[1]) * 100:.2f} %")
        print(f"  sum of ACCEPTED rectangle areas/M*N = {rect_ice_concentration(mdl, shape) * 100:.2f} % "
              "(overlaps counted twice — that is the book's point)")
        print("  [these are NOT the book's numbers: model_ice.jpg is a different image (risk R12)]")

        written.append(save_image(out / "fig_9_15_b_model_raster.png", mdl.bw))
        fig, axes = plt.subplots(1, 2, figsize=(9, 11))
        imshow_matlab(axes[0], d.bw4, title="bw4 with the minimum-area bounding rectangles")
        for s in S:
            v = s.Vertices
            axes[0].plot(v[:, 0] - 1, v[:, 1] - 1, "b-", lw=0.8)
            axes[0].plot(s.Center[0] - 1, s.Center[1] - 1, "r+", ms=5)
        imshow_matlab(axes[1], mdl.bw, title=f"model floes ({mdl.accepted.size} accepted, "
                                             f"{P['k1']} < k < {P['k2']})")
        for f in mdl.s_model:
            axes[1].plot(f.Vertices[:, 0] - 1, f.Vertices[:, 1] - 1, "b-", lw=0.8)
            axes[1].plot(f.Center[0] - 1, f.Center[1] - 1, "r+", ms=5)
        fig.suptitle("Fig. 9.15 — rectangularization (a) and the model-ice floe model (b)")
        written.append(finish_figure(fig, out / "fig_9_15_rect_and_model.png", args.show))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
