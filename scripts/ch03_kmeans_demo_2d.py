"""Book §3.2 (text only) — distance measures (Eqs. 3.29–3.34) and the 2-D k-means demonstration (Figs. 3.6, 3.8).

No MATLAB file: Fig. 3.6 illustrates Steps 1–4 of §3.2.2 (Eqs. 3.35–3.37) on a small 2-D point set with random
initial centroids; Fig. 3.8 shows how one outlier distorts the final clusters.  The points are synthetic and
seeded (:func:`seaice.core.synth.two_clusters_2d`); the algorithm is :func:`seaice.core.clustering.kmeans_lloyd`.
No book data is needed, so this script never skips.

Usage: ``python scripts/ch03_kmeans_demo_2d.py [--seed 0] [--out outputs/ch03] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch03_ice_pixel_detection import kmeans_demo_2d  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.plotting import finish_figure  # noqa: E402

CH = "ch03"
COLORS = ("tab:blue", "tab:red", "tab:green")


def _panel(ax, X, centers=None, labels=None, title=""):
    if labels is None:
        ax.plot(X[:, 0], X[:, 1], "o", color="0.4", markersize=5, label="Data")
    else:
        for i in range(int(labels.max()) + 1):
            sel = labels == i
            ax.plot(X[sel, 0], X[sel, 1], "o", color=COLORS[i % 3], markersize=5, label=f"Cluster {i + 1}")
    if centers is not None:
        ax.plot(centers[:, 0], centers[:, 1], "k*", markersize=14, label="Centroids")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper left", fontsize=7, frameon=False)


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0, help="seed for the synthetic points and the random init")
    args = p.parse_args(argv)
    _, out = resolve_dirs(args)
    demo = kmeans_demo_2d(seed=args.seed)
    X, res = demo["X"], demo["result"]
    hist = res.history

    # --- key numbers ------------------------------------------------------------------------------------------
    print(f"{X.shape[0]} synthetic points, k = 2, random initial centroids (seed {args.seed})")
    for it, (c, lab, J) in enumerate(hist):
        print(f"  iteration {it}: centroids {np.round(c, 3).tolist()}, J = {J:.4f}, sizes "
              f"{[int((lab == i).sum()) for i in range(2)]}")
    print(f"converged: {res.converged} after {res.n_iter} iterations; J non-increasing: "
          f"{all(hist[i + 1][2] <= hist[i][2] + 1e-12 for i in range(len(hist) - 1))}")
    print(f"Eqs. (3.29)-(3.34) between a = {np.round(demo['a'], 3).tolist()} and b = {np.round(demo['b'], 3).tolist()}:")
    for m, v in demo["distances"].items():
        print(f"  {m:12s} = {v:.4f}")
    ro = demo["result_outlier"]
    print(f"Fig. 3.8: without outlier cluster sizes {[int((demo['result_clean'].labels == i).sum()) for i in range(2)]}; "
          f"with outlier at {demo['outlier']}: sizes {[int((ro.labels == i).sum()) for i in range(2)]} "
          f"(both compact clusters merged into one)")

    # --- Fig. 3.6: five panels --------------------------------------------------------------------------------
    c0, l0, _ = hist[0]
    c1, l1, _ = hist[1] if len(hist) > 1 else hist[0]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    _panel(axes[0, 0], X, title="(a) Data points")
    _panel(axes[0, 1], X, centers=c0, title="(b) Select initial centroids at random")
    _panel(axes[0, 2], X, centers=c0, labels=l0, title="(c) Assign each data point to the nearest centroid (Eq. 3.36)")
    _panel(axes[1, 0], X, centers=c1, labels=l0, title="(d) Recalculate the centroids (Eq. 3.37)")
    _panel(axes[1, 1], X, centers=res.centers, labels=res.labels,
           title=f"(e) Repeat until the clustering does not change ({res.n_iter} iterations)")
    axes[1, 2].plot([h[2] for h in hist], "ko-")
    axes[1, 2].set_xlabel("iteration")
    axes[1, 2].set_ylabel("J (Eq. 3.35)")
    axes[1, 2].set_title("objective function J per iteration", fontsize=9)
    fig.suptitle("Fig. 3.6  An example of the k-means clustering process (synthetic data)")
    written = [finish_figure(fig, out / "fig_3_06_kmeans_2d_process.png", args.show)]

    # --- Fig. 3.8: outlier effect -----------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    rc = demo["result_clean"]
    _panel(axes[0], X, centers=rc.centers, labels=rc.labels, title="(a) Two compact and well-separated clusters")
    _panel(axes[1], demo["X_out"], centers=ro.centers, labels=ro.labels,
           title="(b) Effect of an outlier on the cluster configuration")
    fig.suptitle("Fig. 3.8  An outlier forces the two compact clusters into the same cluster (synthetic data)")
    written.append(finish_figure(fig, out / "fig_3_08_kmeans_outlier.png", args.show))

    print("figures written:")
    for pth in written:
        print("  ", pth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
