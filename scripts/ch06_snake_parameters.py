"""Book **Figures 6.2, 6.3, 6.5 and 6.6** — what alpha and beta do, how a snake converges, and when it fails.

No MATLAB file draws these and the source images are not shipped, so the *mechanism* is reproduced on synthetic
fixtures (``sec_`` file names).  The book's own values are the defaults: footnote 2, p. 121 fixes
**alpha = 0.05, beta = 0.0** for every snake in the book, and ``sea_ice_demo.m`` adds gamma = 1, kappa = 0.5.

* **Fig. 6.2(a)-(c)** — the elasticity ``alpha`` of Eq. (6.3) ``E_int = 1/2 alpha |dc/ds|^2 + 1/2 beta |d2c/ds2|^2``
  controls point spacing and tension: "the larger alpha is, the more evenly spaced the snake points are, the
  fewer loops and ripples the snake has, and the shorter the snake becomes"; ``alpha = 0`` allows uneven spacing.
* **Fig. 6.3(a)-(c)** — the rigidity ``beta`` controls smoothness: large ``beta`` gives smooth contours,
  ``beta = 0`` allows corners.
* **Fig. 6.5** — the iterative procedure of Eqs. (6.39)/(6.40): the snake is redrawn after every block.
* **Fig. 6.6** — a snake started too far from the object: the external force is ~0 there (the tiny capture range
  of section 6.1.3), so the contour "will evolve under its own energy ... and form a circle that keeps
  shrinking" (p. 115) instead of finding the boundary.

Figures 6.2/6.3 deliberately run ``snakedeform`` **without** ``snakeinterp``: the resampling step would force a
uniform spacing of at most ``Dmax`` and hide precisely the effect alpha has.

Usage: ``python scripts/ch06_snake_parameters.py [--alphas 0 0.05 0.5] [--betas 0 0.05 0.5] [--iters 200]
[--out outputs/ch06] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch06_gvf_snake import BOOK_PARAMS, CIRCLE_T  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, snake_plot  # noqa: E402
from seaice.core.snake import gaussian_blur, gradient2, gradient2_magnitude, gvf, snakedeform, \
    snakeinterp  # noqa: E402
from seaice.core.synth import synthetic_floe_field  # noqa: E402

CH = "ch06"
S = BOOK_PARAMS["sea_ice_demo"]


def spacing_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """``(total length, mean spacing, std of the spacing)`` of a closed contour (Euclidean, cyclic)."""
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    d = np.hypot(dx, dy)
    return float(d.sum()), float(d.mean()), float(d.std())


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.05, 0.5],
                   help="Fig. 6.2 elasticity values (the book uses 0.05)")
    p.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.05, 0.5],
                   help="Fig. 6.3 rigidity values (the book uses 0)")
    p.add_argument("--iters", type=int, default=200, help="snake iterations for Figs. 6.2/6.3")
    p.add_argument("--blocks", type=int, default=30, help="5-iteration blocks for Figs. 6.5/6.6")
    p.add_argument("--gvf-iters", type=int, default=200, help="GVF iterations for the force field")
    args = p.parse_args(argv)
    _, out = resolve_dirs(args)
    t0 = time.time()

    img = synthetic_floe_field(shape=(128, 128), n_floes=1, seed=4, radius=(38, 39))
    f2 = gradient2_magnitude(img.astype(np.float64))          # edge map, Eq. (6.7)
    u, v = gvf(f2, S["mu"], args.gvf_iters)                   # Eqs. (6.41)/(6.53)
    mag = np.sqrt(u * u + v * v)
    px, py = u / (mag + 1e-10), v / (mag + 1e-10)             # the drivers' unit-normalised force
    M, N = img.shape
    rr, cc = np.nonzero(img > 0)
    cy, cx = rr.mean() + 1.0, cc.mean() + 1.0
    written = []

    # ---- Figs. 6.2 and 6.3: alpha and beta ---------------------------------------------------------------
    x0 = cx + 12.0 * np.cos(CIRCLE_T)
    y0 = cy + 12.0 * np.sin(CIRCLE_T)
    for tag, values, fixed, name, figno in (("alpha", args.alphas, S["beta"], "beta", "6_02"),
                                            ("beta", args.betas, S["alpha"], "alpha", "6_03")):
        print(f"\n=== Fig. {figno.replace('_', '.')}: influence of {tag} "
              f"({name} = {fixed}, gamma = {S['gamma']}, kappa = {S['kappa']}, {args.iters} iterations, "
              f"no snakeinterp) ===")
        print(f"{tag:>8} | {'points':>6} | {'length':>8} | {'mean spacing':>12} | {'std spacing':>11}")
        curves = []
        for val in values:
            a, b = (val, fixed) if tag == "alpha" else (fixed, val)
            xs, ys = snakedeform(x0, y0, a, b, S["gamma"], S["kappa"], px, py, args.iters)
            L, m, sd = spacing_stats(xs, ys)
            print(f"{val:>8g} | {len(xs):>6d} | {L:8.2f} | {m:12.4f} | {sd:11.4f}")
            curves.append((val, xs, ys))
        fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 5))
        axes = np.atleast_1d(axes)
        for ax, (val, xs, ys) in zip(axes, curves):
            imshow_matlab(ax, img, title=f"{tag} = {val:g} ({name} = {fixed:g})")
            snake_plot(ax, x0, y0, "r--", lw=0.8)
            snake_plot(ax, xs, ys, "g-", lw=1.2)
            ax.plot(xs - 1, ys - 1, "y.", ms=2)
        fig.suptitle(f"Figure {figno.replace('_', '.')} — influence of {tag} on the snake "
                     f"(yellow dots = snake points)")
        written.append(finish_figure(fig, out / f"sec_6_1_1_1_fig_{figno}_{tag}_influence.png", args.show))

    # ---- Fig. 6.5: the iterative procedure ---------------------------------------------------------------
    print(f"\n=== Fig. 6.5: iterative convergence (alpha = {S['alpha']}, beta = {S['beta']}, "
          f"kappa = {S['kappa']}, {args.blocks} blocks of 5 iterations) ===")
    xs, ys = snakeinterp(x0, y0, S["Dmax"], S["Dmin"])
    history = [(xs.copy(), ys.copy())]
    for i in range(args.blocks):
        xs, ys = snakedeform(xs, ys, S["alpha"], S["beta"], S["gamma"], S["kappa"], px, py, 5)
        xs, ys = snakeinterp(xs, ys, S["Dmax"], S["Dmin"])
        history.append((xs.copy(), ys.copy()))
    for i in (0, args.blocks // 4, args.blocks // 2, args.blocks):
        L, m, sd = spacing_stats(*history[i])
        print(f"  after {5 * i:>3d} iterations: {len(history[i][0]):>5d} points, length {L:8.2f}, "
              f"mean spacing {m:.4f}")
    fig, ax = plt.subplots(figsize=(6, 6))
    imshow_matlab(ax, img, title=f"snake evolution, {5 * args.blocks} iterations")
    for h in history[1:-1]:
        snake_plot(ax, *h, "y-", lw=0.5)
    snake_plot(ax, *history[0], "r-", lw=1.2)
    snake_plot(ax, *history[-1], "g-", lw=1.5)
    fig.suptitle("Figure 6.5 — the iterative snake procedure (Eqs. 6.39/6.40)")
    written.append(finish_figure(fig, out / "sec_6_1_2_fig_6_05_convergence.png", args.show))

    # ---- Fig. 6.6: started too far away ------------------------------------------------------------------
    # The traditional external force of Eq. (6.11) is ~0 far from the object -- the small capture range of
    # section 6.1.3 -- so the contour evolves under its internal energy alone and keeps shrinking (p. 115).
    E = -gradient2_magnitude(gaussian_blur(img.astype(np.float64), 2.0)) ** 2   # Eq. (6.11), gamma = 1
    ex, ey = gradient2(E)
    tx, ty = -ex, -ey
    tscale = float(np.hypot(tx, ty).max()) or 1.0
    tpx, tpy = tx / tscale, ty / tscale
    print("\n=== Fig. 6.6: a snake started too far from the object (traditional force field) ===")
    results = {}
    for tag, (ppx, ppy, r_start) in (("far (traditional)", (tpx, tpy, 58.0)),
                                     ("near (traditional)", (tpx, tpy, 44.0)),
                                     ("far (GVF)", (px, py, 58.0))):
        xa = cx + r_start * np.cos(CIRCLE_T)
        ya = cy + r_start * np.sin(CIRCLE_T)
        xs2, ys2 = snakeinterp(xa, ya, S["Dmax"], S["Dmin"])
        hist2 = [(xs2.copy(), ys2.copy())]
        for _ in range(args.blocks):
            xs2, ys2 = snakedeform(xs2, ys2, S["alpha"], S["beta"], S["gamma"], S["kappa"], ppx, ppy, 5)
            xs2, ys2 = snakeinterp(xs2, ys2, S["Dmax"], S["Dmin"])
            hist2.append((xs2.copy(), ys2.copy()))
        radius = 0.5 * (xs2.max() - xs2.min())
        rows = np.clip(np.round(ys2).astype(int) - 1, 0, M - 1)
        cols2 = np.clip(np.round(xs2).astype(int) - 1, 0, N - 1)
        on_object = (img[rows, cols2] > 0).mean()
        print(f"  start r = {r_start:>4.0f}, {tag:>18}: final half-width {radius:6.2f} px "
              f"(object radius ~ 38.5), {100 * on_object:5.1f} % of the contour points sit on the floe")
        results[tag] = (hist2, xs2, ys2)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (tag, (hist2, xs2, ys2)) in zip(axes, results.items()):
        imshow_matlab(ax, img, title=tag)
        for h in hist2[1:-1]:
            snake_plot(ax, *h, "y-", lw=0.4)
        snake_plot(ax, *hist2[0], "r-", lw=1.0)
        snake_plot(ax, xs2, ys2, "g-", lw=1.4)
    fig.suptitle("Figure 6.6 — a traditional snake started outside the capture range fails; GVF does not")
    written.append(finish_figure(fig, out / "sec_6_1_3_fig_6_06_capture_range_failure.png", args.show))

    print(f"\nwall clock {time.time() - t0:.1f} s")
    print("figures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
