"""Book **Figures 6.16 and 6.17** — the GVF capture range as a function of the number of GVF iterations (§6.5.2).

No MATLAB file draws these; §6.5.2 specifies the experiment completely:

* Fig. 6.16(a) "a 110x186 binary image containing a large circle with 61-pixel wide diameter and a small circle
  with 9-pixel wide diameter" (:func:`seaice.core.synth.fig_6_16_circles`; the circle *positions* are not
  printed, so they are chosen to match the printed layout — the image is Tier-3 synthetic);
* Fig. 6.16(b)–(e): the GVF field via **5, 30, 100 and 250** iterations;
* Fig. 6.17(a)/(b): the evolution of the GVF snake to the boundary of the **large** circle under **30** vs
  **250** GVF iterations.

The two claims the text makes are measured and printed:
"Figure 6.16(b) shows that 5 iterations are sufficient for the 9-pixel wide diameter [small] circle" and
"Figure 6.16(c) implies that 30 iterations are still [not enough] for the [61-pixel] circle".  The measure used
is the fraction of *interior* pixels of each circle whose GVF magnitude exceeds a threshold, i.e. how far the
force field has diffused inwards from the boundary.

Usage: ``python scripts/ch06_capture_range.py [--iters 5 30 100 250] [--snake-iters 30 250]
[--mu 0.1] [--out outputs/ch06] [--show]``
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
from seaice.core.distance import bwdist  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, quiver_field, save_image, snake_plot  # noqa: E402
from seaice.core.snake import gradient2_magnitude, gvf, snakedeform, snakeinterp  # noqa: E402
from seaice.core.synth import fig_6_16_circles  # noqa: E402

CH = "ch06"
F = BOOK_PARAMS["figures"]
S = BOOK_PARAMS["sea_ice_demo"]


def filled_fraction(mag: np.ndarray, mask: np.ndarray, thresh: float) -> float:
    """Fraction of the pixels of ``mask`` where the field magnitude exceeds ``thresh`` (how far the GVF has
    diffused into the object)."""
    return float((mag[mask] > thresh).mean()) if mask.any() else float("nan")


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--iters", type=int, nargs="+", default=list(F["gvf_iters_fig_6_16"]),
                   help="GVF iteration counts of Fig. 6.16(b)-(e) (5 30 100 250)")
    p.add_argument("--snake-iters", type=int, nargs="+", default=list(F["gvf_iters_fig_6_17"]),
                   help="GVF iteration counts of Fig. 6.17(a)/(b) (30 250)")
    p.add_argument("--mu", type=float, default=S["mu"], help="GVF regularization (footnote 3, p. 125: 0.1)")
    p.add_argument("--thresh", type=float, default=0.02, help="magnitude threshold for the 'filled' measure")
    p.add_argument("--snake-blocks", type=int, default=40, help="5-iteration snake blocks for Fig. 6.17")
    p.add_argument("--edge-map", default="gradient", choices=["gradient", "binary"],
                   help="f = |grad I| (Eq. 6.7, the chapter's code) or f = I (Eq. 6.12, binary images)")
    args = p.parse_args(argv)
    _, out = resolve_dirs(args)

    t0 = time.time()
    bw = fig_6_16_circles()
    M, N = bw.shape
    # Split the two circles again so their interiors can be measured separately.
    from seaice.core.connectivity import label_components
    lab = label_components(bw, 8)
    sizes = [int((lab == k).sum()) for k in range(1, int(lab.max()) + 1)]
    big_k = int(np.argmax(sizes)) + 1
    small_k = int(np.argmin(sizes)) + 1
    big, small = lab == big_k, lab == small_k
    print(f"=== Fig. 6.16(a): {M}x{N} binary image, {int(bw.sum())} object pixels ===")
    for name, m in (("large circle", big), ("small circle", small)):
        rr, cc = np.nonzero(m)
        d = 2.0 * float(bwdist(~m, "euclidean").max())
        print(f"  {name}: {int(m.sum())} px, centre (row {rr.mean() + 1:.1f}, col {cc.mean() + 1:.1f}), "
              f"diameter ~ {d:.0f} px")

    # Edge map: the chapter's own recipe f = |grad I| (Eq. 6.7; GVF_distance.m line 58), so the GVF vectors
    # converge on the circle boundary from *both* sides.  --edge-map binary uses the Eq. (6.12) binary form
    # instead (a ring of inward-pointing vectors), which cannot attract a snake from inside.
    f = gradient2_magnitude(bw.astype(np.float64)) if args.edge_map == "gradient" else bw.astype(np.float64)
    written = [save_image(out / "fig_6_16_a_binary.png", bw)]
    panels = []
    print(f"\n=== Fig. 6.16(b)-(e): GVF fields (mu = {args.mu}) ===")
    print(f"{'iterations':>10} | {'|v| max':>9} | {'large circle filled':>19} | {'small circle filled':>19}")
    for it in args.iters:
        u, v = gvf(f, args.mu, it)
        mag = np.hypot(u, v)
        fb = filled_fraction(mag, big, args.thresh)
        fs = filled_fraction(mag, small, args.thresh)
        print(f"{it:>10} | {mag.max():9.4f} | {100 * fb:18.1f}% | {100 * fs:18.1f}%")
        panels.append((it, u, v, mag))
        fig, ax = plt.subplots(figsize=(11, 6.5))
        quiver_field(ax, u, v, step=2, scale=25)
        ax.set_title(f"(b-e) GVF field via {it} iterations")
        written.append(finish_figure(fig, out / f"fig_6_16_gvf_{it:03d}_iterations.png", args.show))

    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(11, 4.0 * (len(panels) + 1)))
    imshow_matlab(axes[0], bw, title=f"(a) {M}x{N} binary image: 61-px and 9-px diameter circles")
    for ax, (it, u, v, _) in zip(axes[1:], panels):
        quiver_field(ax, u, v, step=2, scale=25)
        ax.set_title(f"GVF field via {it} iterations")
    fig.suptitle("Figure 6.16 — GVF force fields via different iteration counts")
    written.append(finish_figure(fig, out / "fig_6_16_panels.png", args.show))

    # ---- Fig. 6.17: snake evolution on the large circle -----------------------------------------------------
    rr, cc = np.nonzero(big)
    cy, cx = rr.mean() + 1.0, cc.mean() + 1.0
    r0 = 6.0  # a small contour at the centre, as in the book's figure
    print(f"\n=== Fig. 6.17: snake on the large circle from a r = {r0:g} circle at ({cx:.1f}, {cy:.1f}) ===")
    fig, axes = plt.subplots(1, len(args.snake_iters), figsize=(6.5 * len(args.snake_iters), 5))
    axes = np.atleast_1d(axes)
    for ax, it in zip(axes, args.snake_iters):
        u, v = gvf(f, args.mu, it)
        mag = np.sqrt(u * u + v * v)
        px, py = u / (mag + 1e-10), v / (mag + 1e-10)
        x = cx + r0 * np.cos(CIRCLE_T)
        y = cy + r0 * np.sin(CIRCLE_T)
        xs, ys = snakeinterp(x, y, S["Dmax"], S["Dmin"])
        history = [(xs.copy(), ys.copy())]
        for _ in range(args.snake_blocks):
            xs, ys = snakedeform(xs, ys, S["alpha"], S["beta"], S["gamma"], S["kappa"], px, py, 5)
            xs, ys = snakeinterp(xs, ys, S["Dmax"], S["Dmin"])
            history.append((xs.copy(), ys.copy()))
        # how close the final contour is to the true circle boundary
        d_out = bwdist(~big, "euclidean")
        rows = np.clip(np.round(ys).astype(int) - 1, 0, M - 1)
        cols = np.clip(np.round(xs).astype(int) - 1, 0, N - 1)
        inside = big[rows, cols]
        print(f"  {it:>3} GVF iterations: final contour {len(xs)} points, "
              f"{100 * inside.mean():.1f}% of them inside the circle, "
              f"mean distance to the boundary {d_out[rows, cols].mean():.2f} px, "
              f"enclosed radius ~ {0.5 * (xs.max() - xs.min()):.1f} px (true 30.5)")
        imshow_matlab(ax, bw, title=f"({'ab'[list(args.snake_iters).index(it)]}) evolution under {it} GVF iterations")
        for h in history[:-1]:
            snake_plot(ax, *h, "y-", lw=0.5)
        snake_plot(ax, *history[0], "r-", lw=1.2)
        snake_plot(ax, *history[-1], "g-", lw=1.4)
        ax.set_xlim(cx - 60, cx + 60)
        ax.set_ylim(cy + 55, cy - 55)
    fig.suptitle("Figure 6.17 — snake evolution to the large circle under different GVF capture ranges")
    written.append(finish_figure(fig, out / "fig_6_17_snake_capture_range.png", args.show))

    print(f"\nwall clock {time.time() - t0:.1f} s")
    print("figures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
