"""Port of ``MATLAB_ROOT/ch6/for test/for_test.m`` — one manual GVF snake on ``test8.jpg`` (§6.1.2 / §6.2).

The closest thing the shipped code has to the book's own snake-evolution figures (Figs. 6.5, 6.8, 6.10):
a single circular contour at ``(x0, y0) = (80, 40)`` with ``r = 20`` is deformed in the GVF field of the
gradient edge map, redrawn after every 5-iteration block, and finally burnt into the Otsu mask.

Literal steps (``for_test.m`` lines 4–121)::

    sigma = 0; GradientOn = 1; GVFOn = 1; Num = 150; mu = 0.1;
    iter = 50; alpha = 0.05; beta = 0; gamma = 1; kappa = 0.5; Dmin = 0; Dmax = 1;
    se = strel('disk', 3);  r = 20; x0 = 80; y0 = 40; timer = 1;      % Ra_min/Ra/Rc/Rl/se/d/order are UNUSED
    I  = rgb2gray(imread('test8.jpg'));   bw = im2bw(I, graythresh(I));
    f  = I;  f2 = abs(gradient2(double(f)));  [u, v] = GVF(f2, mu, Num);
    mag = sqrt(u.*u + v.*v);  px = u ./ (mag + 1e-10);  py = v ./ (mag + 1e-10);
    xSpace = (1:size(bw,1)/64:size(bw,1));  ySpace = (1:size(bw,2)/64:size(bw,2));
    qx = interp2(px, xSpace, ySpace');  qy = interp2(py, xSpace, ySpace');  quiver(xSpace, ySpace, qx, qy);
    t = 0:0.05:6.28;  x = x0 + r*cos(t);  y = y0 + r*sin(t);
    [x, y] = polybool('intersection', s_2, s_1, x, y);        % clip to the image rectangle
    [x, y] = snakeinterp(x, y, Dmax, Dmin);   snakedisp(x, y, 'r')
    for i = 1:ceil(iter/5), [x,y] = snakedeform(...,5); [x,y] = snakeinterp(...); snakedisp(x,y,'y'); end
    bw = im2bw(I, graythresh(I));  xx = ceil(x); yy = ceil(y);
    for i = 1:len, if xx(i) <= h && yy(i) <= v, bw(yy(i), xx(i)) = 0; end, end

Note the quiver grid: ``xSpace`` spans ``size(bw, 1)`` = the number of **rows** but is passed to ``interp2`` as
the **column** coordinate (and ``ySpace`` the other way round).  That is what the script does, so it is what is
reproduced here — on the 108x148 ``test8.jpg`` both ranges stay inside the image, so the field is merely sampled
on a skewed grid.  Note also that ``for_test.m`` clips **before** interpolating, while ``GVF_distance.m``
interpolates first — reproduced in each place.

Usage: ``python scripts/ch06_for_test.py [--num 150] [--iter 50] [--r 20] [--x0 80] [--y0 40]
[--data data/book/ch06] [--out outputs/ch06] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch06_gvf_snake import BOOK_PARAMS, CIRCLE_T, gvf_force_field  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.interp import interp2  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, quiver_field, save_image, snake_plot  # noqa: E402
from seaice.core.polygon import clip_polygon_rect  # noqa: E402
from seaice.core.snake import snakedeform, snakeinterp  # noqa: E402
from seaice.core.threshold import graythresh, im2bw  # noqa: E402

CH = "ch06"
P = BOOK_PARAMS["for_test"]


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--num", type=int, default=P["Num"], help="GVF iterations (script: 150)")
    p.add_argument("--iter", type=int, default=P["iter"], help="snake iterations (script: 50)")
    p.add_argument("--r", type=float, default=P["r"], help="initial circle radius (script: 20)")
    p.add_argument("--x0", type=float, default=P["x0"], help="initial contour centre, column (script: 80)")
    p.add_argument("--y0", type=float, default=P["y0"], help="initial contour centre, row (script: 40)")
    p.add_argument("--image", default="test8.jpg")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        II, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: skipped: private image absent ({exc})")
        return 0

    t0 = time.time()
    I = rgb2gray_matlab(II)
    level, _ = graythresh(I)
    bw = im2bw(I, level)
    s1, s2 = bw.shape
    f2, u, v, px, py = gvf_force_field(I, sigma=P["sigma"], gradient_on=bool(P["GradientOn"]),
                                       gvf_on=bool(P["GVFOn"]), num=args.num, mu=P["mu"])

    print(f"=== for_test.m on {args.image} ({s1}x{s2}) ===")
    print(f"graythresh = {level:.6f} (255*level = {255 * level:g}); ice pixels {int(bw.sum())} "
          f"({100.0 * bw.mean():.2f} %)")
    print(f"edge map f2 = |grad(double(I))|: min {f2.min():.4f}, max {f2.max():.4f}")
    print(f"GVF({args.num} iterations, mu = {P['mu']}): |v| min {np.hypot(u, v).min():.4g}, "
          f"max {np.hypot(u, v).max():.4g}")

    # --- the script's quiver grid (literal, including the swapped roles of xSpace/ySpace) -------------------
    x_space = np.arange(1.0, s1 + 1e-9, s1 / 64.0)
    y_space = np.arange(1.0, s2 + 1e-9, s2 / 64.0)
    XX, YY = np.meshgrid(x_space, y_space)            # XX = column coordinate, YY = row coordinate
    qx = interp2(px, YY - 1.0, XX - 1.0, method="bilinear")
    qy = interp2(py, YY - 1.0, XX - 1.0, method="bilinear")
    print(f"quiver grid: xSpace {len(x_space)} pts in [1, {x_space[-1]:.3f}], "
          f"ySpace {len(y_space)} pts in [1, {y_space[-1]:.3f}]")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.quiver(XX, YY, qx, qy, color="b", scale=60)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"GVF field via {args.num} iterations (for_test.m quiver, 64x64 grid)")
    written = [finish_figure(fig, out / "sec_6_2_for_test_a_gvf_quiver.png", args.show)]

    # --- the single manual contour -------------------------------------------------------------------------
    x = args.x0 + args.r * np.cos(CIRCLE_T)
    y = args.y0 + args.r * np.sin(CIRCLE_T)
    xc, yc = clip_polygon_rect(x, y, (0.0, float(s2)), (0.0, float(s1)))  # polybool('intersection', ...)
    xi, yi = snakeinterp(xc, yc, P["Dmax"], P["Dmin"])
    history = [(xi.copy(), yi.copy())]
    xs, ys = xi, yi
    n_blocks = int(np.ceil(args.iter / 5))
    floor5 = int(np.floor(args.iter / 5))
    for i in range(1, n_blocks + 1):
        steps = 5 if i <= floor5 else args.iter - floor5 * 5
        xs, ys = snakedeform(xs, ys, P["alpha"], P["beta"], P["gamma"], P["kappa"], px, py, steps)
        xs, ys = snakeinterp(xs, ys, P["Dmax"], P["Dmin"])
        history.append((xs.copy(), ys.copy()))
    print(f"initial circle: {len(CIRCLE_T)} points (t = 0:0.05:6.28), after polybool {xc.size}, "
          f"after snakeinterp {xi.size}")
    print(f"snake: {n_blocks} blocks of 5 iterations; N per block "
          f"{[len(h[0]) for h in history]}")
    print(f"final contour: {len(xs)} points, x in [{xs.min():.2f}, {xs.max():.2f}], "
          f"y in [{ys.min():.2f}, {ys.max():.2f}]")

    fig, ax = plt.subplots(figsize=(8, 6))
    imshow_matlab(ax, II, title="snake evolution (red = initial, yellow = every 5 iterations, green = final)")
    snake_plot(ax, *history[0], "r-", lw=1.2)
    for h in history[1:-1]:
        snake_plot(ax, *h, "y-", lw=0.6)
    snake_plot(ax, *history[-1], "g-", lw=1.4)
    written.append(finish_figure(fig, out / "sec_6_1_2_for_test_b_snake_evolution.png", args.show))

    # --- burn the boundary into the Otsu mask (lines 113-121) ----------------------------------------------
    bw_out = im2bw(I, graythresh(I)[0]).copy()
    xx = np.ceil(xs).astype(np.int64)
    yy = np.ceil(ys).astype(np.int64)
    ok = (xx <= s2) & (yy <= s1) & (xx >= 1) & (yy >= 1)
    bw_out[yy[ok] - 1, xx[ok] - 1] = False
    print(f"burnt {int(bw.sum()) - int(bw_out.sum())} pixels into bw ({int(ok.sum())} contour points in range)")
    written.append(save_image(out / "sec_6_1_2_for_test_c_binary.png", bw))
    written.append(save_image(out / "sec_6_1_2_for_test_d_binary_with_boundary.png", bw_out))
    written.append(save_image(out / "sec_6_2_for_test_e_edge_map.png", f2, autoscale=True))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    quiver_field(axes[0], px, py, step=3, scale=60)
    axes[0].set_title("normalised GVF force field px, py (every 3rd pixel)")
    imshow_matlab(axes[1], f2, autoscale=True, title="edge map f2 = |grad I|")
    written.append(finish_figure(fig, out / "sec_6_2_for_test_f_field_and_edge_map.png", args.show))

    print(f"\nwall clock {time.time() - t0:.1f} s")
    print("figures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
