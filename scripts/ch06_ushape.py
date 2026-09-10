"""Book **Figures 6.4, 6.7 and 6.9** — external energy, its force field, and why GVF beats the traditional snake.

No MATLAB file draws these (the shipped code only ever runs the GVF variant on real images), and the source
images are not shipped, so the *mechanism* is reproduced on synthetic fixtures and labelled ``sec_`` accordingly:

* **Fig. 6.4** (a) a binary ice floe, (b) its external energy ``E_ext = -gamma |grad(G_sigma * I)|^2``
  (Eq. 6.11, caption sigma = **5**), (c) the force field ``F_ext = -grad E_ext`` (Eq. 6.29);
* **Fig. 6.7** the classic U shape (:func:`seaice.core.synth.u_shape`): (a) the object, (b) its external energy
  at sigma = **4**, (c) a traditional snake that cannot be pulled into the boundary concavity (§6.1.3);
* **Fig. 6.9** the same U with the **GVF** field: the force has a component that points *into* the channel
  (§6.2), so the snake reaches the bottom of the concavity.

The script prints the quantitative version of the book's argument: the mean force component along the channel
axis inside the concavity, for the traditional field and for the GVF field.

Usage: ``python scripts/ch06_ushape.py [--sigma-floe 5] [--sigma-u 4] [--gvf-iters 200] [--mu 0.1]
[--blocks 40] [--out outputs/ch06] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch06_gvf_snake import BOOK_PARAMS, CIRCLE_T, external_energy  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, quiver_field, save_image, snake_plot  # noqa: E402
from seaice.core.snake import gaussian_blur, gradient2, gvf, snakedeform, snakeinterp  # noqa: E402
from seaice.core.synth import synthetic_floe_field, u_shape  # noqa: E402

CH = "ch06"
S = BOOK_PARAMS["sea_ice_demo"]
F = BOOK_PARAMS["figures"]


def evolve(px, py, x, y, blocks: int, alpha: float, beta: float, gamma: float, kappa: float):
    """``ceil(iter/5)`` blocks of ``snakedeform(..., 5)`` + ``snakeinterp`` — the drivers' inner loop."""
    xs, ys = snakeinterp(x, y, S["Dmax"], S["Dmin"])
    history = [(xs.copy(), ys.copy())]
    for _ in range(blocks):
        xs, ys = snakedeform(xs, ys, alpha, beta, gamma, kappa, px, py, 5)
        xs, ys = snakeinterp(xs, ys, S["Dmax"], S["Dmin"])
        history.append((xs.copy(), ys.copy()))
    return xs, ys, history


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--sigma-floe", type=float, default=F["sigma_fig_6_4"], help="Fig. 6.4(b) sigma (5)")
    p.add_argument("--sigma-u", type=float, default=F["sigma_fig_6_7"], help="Fig. 6.7(b) sigma (4)")
    p.add_argument("--gvf-iters", type=int, default=200, help="GVF iterations for Fig. 6.9")
    p.add_argument("--mu", type=float, default=S["mu"], help="GVF regularization (footnote 3: 0.1)")
    p.add_argument("--blocks", type=int, default=40, help="5-iteration snake blocks")
    p.add_argument("--thickness", type=int, default=3, help="stroke width of the U (line drawing)")
    p.add_argument("--gap", type=int, default=28, help="width of the concavity")
    p.add_argument("--sigma-sweep", type=float, nargs="+", default=[1.0, 2.0, 4.0],
                   help="Gaussian sigmas for the capture-range table (Eq. 6.10)")
    p.add_argument("--kappa", type=float, default=1.0,
                   help="external force weight for the unit-maximum-scaled fields (the drivers use 0.5 on the "
                        "per-pixel-normalised field)")
    args = p.parse_args(argv)
    _, out = resolve_dirs(args)
    t0 = time.time()
    written = []

    # ---------------- Fig. 6.4: external energy of a binary ice floe -------------------------------------
    floe = synthetic_floe_field(shape=(128, 128), n_floes=1, seed=4, radius=(34, 35))
    E = external_energy(floe.astype(np.float64), kind="edge", gamma=1.0, sigma=args.sigma_floe)
    ex, ey = gradient2(E)
    fx, fy = -ex, -ey  # Eq. (6.29)
    print(f"=== Fig. 6.4 (synthetic binary floe, {floe.shape[0]}x{floe.shape[1]}) ===")
    print(f"E_ext = -|grad(G_sigma * I)|^2 with sigma = {args.sigma_floe:g}: "
          f"min {E.min():.4g}, max {E.max():.4g}")
    print(f"|F_ext| = |grad E_ext|: max {np.hypot(fx, fy).max():.4g}; "
          f"{100.0 * (np.hypot(fx, fy) > 0.01 * np.hypot(fx, fy).max()).mean():.1f} % of pixels carry > 1 % of "
          "the peak force (this is the small 'capture range' of section 6.1.3)")
    written += [save_image(out / "sec_6_1_1_2_fig_6_04_a_binary_floe.png", floe),
                save_image(out / "sec_6_1_1_2_fig_6_04_b_external_energy.png", E, autoscale=True)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    imshow_matlab(axes[0], floe, title="(a) binary ice floe")
    imshow_matlab(axes[1], E, autoscale=True, title=f"(b) E_ext = -|grad(G_s * I)|^2, sigma = {args.sigma_floe:g}")
    quiver_field(axes[2], fx, fy, step=3)
    axes[2].set_title("(c) F_ext = -grad E_ext  (Eq. 6.29)")
    fig.suptitle("Figure 6.4 — external energy and force field of a traditional snake (synthetic floe)")
    written.append(finish_figure(fig, out / "sec_6_1_1_2_fig_6_04_panels.png", args.show))

    # ---------------- Figs. 6.7 / 6.9: the U shape ------------------------------------------------------
    # Xu & Prince's classic test object is a thin-stroke U (a line drawing), for which the book's Eq. (6.13)
    # binary external energy E_ext = -gamma (G_sigma * I) applies directly: the image itself is already a ridge.
    U = u_shape(shape=(96, 96), thickness=args.thickness, gap=args.gap, margin=14)
    M, N = U.shape
    # The concavity: background pixels with object pixels to their left, right and below.
    has_left = np.cumsum(U, axis=1) > 0
    has_right = np.cumsum(U[:, ::-1], axis=1)[:, ::-1] > 0
    has_below = np.cumsum(U[::-1, :], axis=0)[::-1, :] > 0
    channel = (~U) & has_left & has_right & has_below
    base_row = int(np.nonzero(U.sum(axis=1) == U.sum(axis=1).max())[0].min())
    mouth_row = int(np.nonzero(channel.any(axis=1))[0].min())
    print(f"\n=== Figs. 6.7 / 6.9 (U shape {M}x{N}, stroke {args.thickness} px, {int(U.sum())} object px) ===")
    print(f"concavity: {int(channel.sum())} px between rows {mouth_row + 1} (mouth) and {base_row + 1} (base)")

    def unit_max(a, b):
        """Scale a vector field by its **global** maximum magnitude.

        Not a per-pixel normalisation: `GVF_distance.m` does normalise per pixel (`px = u/(mag + 1e-10)`), but
        that gives every pixel a unit-length force and would erase exactly the property these figures are about
        -- the traditional field's small capture range (sections 6.1.3 and 6.5.2).
        """
        m = float(np.hypot(a, b).max()) or 1.0
        return a / m, b / m

    # Book Eq. (6.13): E_ext = -gamma (G_sigma * I)  =>  F_ext = -grad E_ext = gamma grad(G_sigma * I).
    cy, cx = M / 2.0, N / 2.0
    r0 = 0.42 * min(M, N)
    x0 = cx + r0 * np.cos(CIRCLE_T)
    y0 = cy + r0 * np.sin(CIRCLE_T)

    def penetration(xs, ys):
        rows = np.clip(np.round(ys).astype(int) - 1, 0, M - 1)
        colz = np.clip(np.round(xs).astype(int) - 1, 0, N - 1)
        in_ch = channel[rows, colz]
        if not in_ch.any():
            return in_ch, None, 0.0
        deep = int(rows[in_ch].max()) + 1
        return in_ch, deep, 100.0 * (deep - mouth_row) / max(base_row - mouth_row, 1)

    print("\ntraditional snake, Eq. (6.13) E_ext = -(G_sigma * I): the Gaussian of Eq. (6.10) is what buys the")
    print("capture range, and section 6.1.3 is about what happens when it is small:")
    print(f"{'sigma':>6} | {'channel px with > 1% of peak |F|':>32} | {'mean downward comp.':>19} | "
          f"{'snake penetration':>17}")
    trad_fields = {}
    for sg in args.sigma_sweep:
        blur = gaussian_blur(U.astype(np.float64), sg)
        bx, by = unit_max(*gradient2(blur))
        xs, ys, hist = evolve(bx, by, x0, y0, args.blocks, S["alpha"], S["beta"], S["gamma"], args.kappa)
        _, deep, pen = penetration(xs, ys)
        trad_fields[sg] = (bx, by, xs, ys, hist, deep, pen)
        print(f"{sg:>6g} | {100.0 * (np.hypot(bx, by)[channel] > 0.01).mean():31.1f}% | "
              f"{by[channel].mean():+19.4f} | {pen:16.0f} %")

    f_blur = gaussian_blur(U.astype(np.float64), args.sigma_u)
    Eu = -f_blur                                     # Eq. (6.13) with gamma = 1
    gu, gv = gvf(f_blur / (f_blur.max() or 1.0), args.mu, args.gvf_iters)   # Eqs. (6.41)/(6.53)
    gpx, gpy = unit_max(gu, gv)
    gxs, gys, ghist = evolve(gpx, gpy, x0, y0, args.blocks, S["alpha"], S["beta"], S["gamma"], args.kappa)
    g_in, g_deep, g_pen = penetration(gxs, gys)
    print(f"\nGVF ({args.gvf_iters} iterations, mu = {args.mu:g}) on the sigma = {args.sigma_u:g} edge map: "
          f"{100.0 * (np.hypot(gpx, gpy)[channel] > 0.01).mean():.1f}% of channel pixels carry > 1 % of the peak "
          f"force,\n    mean downward component {gpy[channel].mean():+.4f}, snake penetration {g_pen:.0f} % "
          f"(deepest row {g_deep} of {base_row + 1}), {len(gxs)} contour points")
    print(f"snakes started on a circle of radius {r0:.1f} around the U, {args.blocks} blocks of 5 iterations, "
          f"kappa = {args.kappa:g}, alpha = {S['alpha']}, beta = {S['beta']}")

    sg_show = min(args.sigma_sweep)  # the small-capture-range regime that Fig. 6.7(c) is about
    tpx, tpy, txs, tys, thist, t_deep, t_pen = trad_fields[sg_show]
    results = {"traditional": (txs, tys, thist, None, t_deep), "gvf": (gxs, gys, ghist, g_in, g_deep)}

    written += [save_image(out / "sec_6_1_3_fig_6_07_a_ushape.png", U),
                save_image(out / "sec_6_1_3_fig_6_07_b_external_energy.png", Eu, autoscale=True)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    imshow_matlab(axes[0, 0], U, title="(a) U-shaped object")
    imshow_matlab(axes[0, 1], Eu, autoscale=True,
                  title=f"(b) E_ext = -G_s * I, sigma = {args.sigma_u:g} (Eq. 6.13)")
    quiver_field(axes[0, 2], tpx, tpy, step=2)
    axes[0, 2].set_title(f"Fig. 6.7 traditional force field, sigma = {sg_show:g}")
    imshow_matlab(axes[1, 0], U, title=f"(c) traditional snake, sigma = {sg_show:g}: {t_pen:.0f} % penetration")
    for h in results["traditional"][2][:-1]:
        snake_plot(axes[1, 0], *h, "y-", lw=0.4)
    snake_plot(axes[1, 0], *results["traditional"][2][0], "r-", lw=1.0)
    snake_plot(axes[1, 0], results["traditional"][0], results["traditional"][1], "g-", lw=1.4)
    quiver_field(axes[1, 1], gpx, gpy, step=2)
    axes[1, 1].set_title(f"Fig. 6.9 GVF field ({args.gvf_iters} iterations)")
    imshow_matlab(axes[1, 2], U, title=f"GVF snake: {g_pen:.0f} % penetration")
    for h in results["gvf"][2][:-1]:
        snake_plot(axes[1, 2], *h, "y-", lw=0.4)
    snake_plot(axes[1, 2], *results["gvf"][2][0], "r-", lw=1.0)
    snake_plot(axes[1, 2], results["gvf"][0], results["gvf"][1], "g-", lw=1.4)
    fig.suptitle("Figures 6.7 / 6.9 - the boundary-concavity problem and how GVF solves it (synthetic U)")
    written.append(finish_figure(fig, out / "sec_6_2_fig_6_07_6_09_ushape.png", args.show))

    print(f"\nwall clock {time.time() - t0:.1f} s")
    print("figures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
