"""Port of ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/GVF_distance.m`` — **Algorithm 1** (§6.4, p. 136),
run on ``alg_seg_gray.jpg`` = **Figure 6.15(a)** with the ``sea_ice_demo.m`` parameter set.

``GVF_distance.m`` is a function with no caller inside ch6 (ch9's ``model_ice_demo.m`` calls it), so this script
is its driver.  ``alg_seg_gray.jpg`` is the image the book's Figure 6.15 is made from (NCC 0.9882 against the
bitmap on page 27 of the chapter PDF), and the six panels below are exactly its sub-figures:

* (a) the grayscale ice image,
* (b) the binary image ``im2bw(I, graythresh(I))``,
* (c) its city-block distance transform,
* (d) the binary image with the regional maxima marked '+',

  Note on (c)/(d): the shipped ``GVF_distance.m`` computes the distance map and its regional maxima on **bw2**,
  i.e. only the components that failed the ``Ra``/``Rc``/``Rl`` re-segmentation criteria of book Ch. 9 p. 205,
  whereas book Fig. 6.15(c)/(d) show them for the whole binary image (Algorithm 1 step 3: "D <- distance map of
  SEGMENTATION").  This script is faithful to the shipped code, so its panels (c)/(d) cover fewer floes than the
  printed figure; panels (a), (b), (e) and (f) are directly comparable with p. 136.

* (e) the binary image with the merged seeds '+' and the initial circles,
* (f) the segmentation result: the GVF-snake boundaries superimposed, so the connected floes are separated.
  "Note that the edge pixels are specifically labeled as *residue ice* for special handling in subsequent use."

Usage: ``python scripts/ch06_gvf_distance.py [--image alg_seg_gray.jpg] [--num 500] [--iter 100]
[--max-seeds N] [--solver auto|dense|circulant] [--data data/book/ch06] [--out outputs/ch06] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch06_gvf_snake import BOOK_PARAMS, CIRCLE_T, gvf_distance  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.connectivity import label_components  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, label2rgb, save_image, snake_plot  # noqa: E402

CH = "ch06"
P = BOOK_PARAMS["sea_ice_demo"]
#: Only ``alg_seg_gray.jpg`` corresponds to a printed book figure (6.15); other images get ``sec_`` names.
FIG_IMAGE = "alg_seg_gray.jpg"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default=FIG_IMAGE, help="book image (default: the Fig. 6.15 source)")
    p.add_argument("--num", type=int, default=P["Num"], help="GVF iterations (sea_ice_demo.m: 500)")
    p.add_argument("--iter", type=int, default=P["iter"], help="snake iterations (sea_ice_demo.m: 100)")
    p.add_argument("--ra", type=float, default=P["Ra"], help="maximum ice piece area Ra (2500)")
    p.add_argument("--ra-min", type=float, default=P["Ra_min"], help="minimum ice piece area Ra_min (10)")
    p.add_argument("--rc", type=float, default=P["Rc"], help="convexity threshold Rc (0.9)")
    p.add_argument("--rl", type=float, default=P["Rl"], help="length/width threshold Rl (2)")
    p.add_argument("--max-seeds", type=int, default=None, help="cap the number of contours (runtime guard)")
    p.add_argument("--solver", default="auto", choices=["auto", "dense", "circulant"],
                   help="snakedeform linear solver ('dense' = MATLAB's literal inv)")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        rgb, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: skipped: private image absent ({exc})")
        return 0

    pre = "fig_6_15" if args.image.lower() == FIG_IMAGE else f"sec_6_4_{Path(args.image).stem}"
    t0 = time.time()
    res = gvf_distance(rgb, sigma=P["sigma"], GradientOn=P["GradientOn"], GVFOn=P["GVFOn"], Num=args.num,
                       mu=P["mu"], iter=args.iter, alpha=P["alpha"], beta=P["beta"], gamma=P["gamma"],
                       kappa=P["kappa"], Dmin=P["Dmin"], Dmax=P["Dmax"], Ra_min=args.ra_min, Ra=args.ra,
                       Rc=args.rc, Rl=args.rl, se_radius=P["se_radius"], timer=P["timer"],
                       max_seeds=args.max_seeds, solver=args.solver)
    elapsed = time.time() - t0
    rec = res.passes[0]
    init = rec.init

    print(f"=== GVF_distance.m on {args.image} ({rgb.shape[0]}x{rgb.shape[1]}) ===")
    print(f"graythresh(rgb2gray(I)) = {res.level:.6f} (255*level = {255 * res.level:g}); "
          f"ice pixels {int(res.bw.sum())} ({100.0 * res.bw.mean():.2f} %)")
    print(f"edge map |grad I|: max {res.f2.max():.3f}; GVF({args.num}, mu = {P['mu']}): "
          f"|v| max {np.hypot(res.u, res.v).max():.4f}")
    print(f"bwlabel(bw, 4): {rec.num} components; failing the ch9 p. 205 criteria: {rec.k.size} "
          f"(Area > {args.ra:g}: {int((rec.area > args.ra).sum())}, "
          f"Solidity < {args.rc:g}: {int((rec.solidity < args.rc).sum())}, "
          f"Major/Minor > {args.rl:g}: {int((rec.rl > args.rl).sum())})")
    if rec.num:
        print(f"  Area: min {rec.area.min():.0f}, median {np.median(rec.area):.0f}, max {rec.area.max():.0f}; "
              f"Solidity: min {np.nanmin(rec.solidity):.4f}, median {np.nanmedian(rec.solidity):.4f}")
    if init is not None:
        print(f"contour initialization: {int(init.dis.sum())} local maxima -> "
              f"imdilate(strel('disk', {P['se_radius']})) -> {init.num} seeds; "
              f"radii min {init.radii.min():.3f}, median {np.median(init.radii):.3f}, max {init.radii.max():.3f}")
    print(f"snakes run: {res.n_seeds_run} of {res.n_seeds}; contour lengths "
          f"{[len(s.x_final) for s in rec.seeds[:8]]}{' ...' if len(rec.seeds) > 8 else ''}")
    burnt = int(res.bw.sum()) - int(res.bw1.sum())
    n_before = int(label_components(res.bw, 4).max())
    n_after = int(label_components(res.bw1, 4).max())
    print(f"boundaries burnt into the mask: {burnt} pixels (the 'residue ice' of p. 135); "
          f"connected floes {n_before} -> {n_after}")
    # Review S9: the two guards the M-code does not have must be visible, not silent.
    print(f"guards (review S9): {res.n_skipped} contour(s) skipped for < 3 vertices after the polybool clip; "
          f"{res.n_below_range} burn point(s) dropped below the image (MATLAB has no lower guard and would "
          f"error), {res.n_above_range} dropped by the M-code's own 'xx <= s2 & yy <= s1'")
    print(f"wall clock {elapsed:.1f} s (solver = {args.solver})")

    written = [
        save_image(out / f"{pre}_a_input.png", res.gray),
        save_image(out / f"{pre}_b_binary.png", res.bw),
    ]
    if init is not None:
        written.append(save_image(out / f"{pre}_c_distance_transform.png", init.img_dist, autoscale=True))
        pts = np.argwhere(init.dis != 0)
        fig, ax = plt.subplots(figsize=(6, 6))
        imshow_matlab(ax, rec.bw2, title=f"(d) regional maxima of the distance map ({pts.shape[0]} px)")
        ax.plot(pts[:, 1], pts[:, 0], "r+", ms=4, mew=0.7)
        written.append(finish_figure(fig, out / f"{pre}_d_regional_maxima.png", args.show))

        fig, ax = plt.subplots(figsize=(6, 6))
        imshow_matlab(ax, res.bw, title=f"(e) seeds and initial contours ({init.num} seeds)")
        for n in range(init.num):
            cx, cy = init.centroids[n]
            r = init.radii[n]
            ax.plot(cx - 1, cy - 1, "r+", ms=5, mew=0.9)
            ax.plot(cx - 1 + r * np.cos(CIRCLE_T), cy - 1 + r * np.sin(CIRCLE_T), "b-", lw=0.7)
        written.append(finish_figure(fig, out / f"{pre}_e_seeds_and_circles.png", args.show))

    written.append(save_image(out / f"{pre}_f_segmentation.png", res.bw1))
    written.append(save_image(out / f"{pre}_f_segmentation_labels.png",
                              label2rgb(label_components(res.bw1, 4))))
    written.append(save_image(out / f"{pre}_residue_ice.png", res.bw & ~res.bw1))

    fig, ax = plt.subplots(figsize=(6, 6))
    imshow_matlab(ax, res.bw, title=f"(f) final GVF-snake boundaries ({len(rec.seeds)} contours)")
    for s in rec.seeds:
        snake_plot(ax, s.x_final, s.y_final, "r-", lw=0.7)
    written.append(finish_figure(fig, out / f"{pre}_f_boundaries_overlay.png", args.show))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    imshow_matlab(axes[0, 0], res.gray, title="(a) grayscale ice image")
    imshow_matlab(axes[0, 1], res.bw, title="(b) im2bw(I, graythresh(I))")
    if init is not None:
        imshow_matlab(axes[0, 2], init.img_dist, autoscale=True, title="(c) city-block distance transform")
        pts = np.argwhere(init.dis != 0)
        imshow_matlab(axes[1, 0], rec.bw2, title="(d) regional maxima")
        axes[1, 0].plot(pts[:, 1], pts[:, 0], "r+", ms=3, mew=0.5)
        imshow_matlab(axes[1, 1], res.bw, title=f"(e) seeds and initial circles ({init.num})")
        for n in range(init.num):
            cx, cy = init.centroids[n]
            r = init.radii[n]
            axes[1, 1].plot(cx - 1, cy - 1, "r+", ms=4, mew=0.7)
            axes[1, 1].plot(cx - 1 + r * np.cos(CIRCLE_T), cy - 1 + r * np.sin(CIRCLE_T), "b-", lw=0.5)
    imshow_matlab(axes[1, 2], label2rgb(label_components(res.bw1, 4)),
                  title=f"(f) segmentation: {n_before} -> {n_after} floes")
    fig.suptitle("Figure 6.15 — GVF snake-based ice image segmentation (Algorithm 1) on alg_seg_gray.jpg")
    written.append(finish_figure(fig, out / f"{pre}_panels.png", args.show))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
