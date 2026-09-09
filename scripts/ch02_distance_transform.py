"""Port of ``MATLAB_ROOT/ch2/distance_transform.m`` — distance maps from a single pixel (Fig. 2.14) + Figs. 2.12–2.13.

Book §2.4, Eqs. (2.9)–(2.12).  The MATLAB script builds a 201×201 image with one centre pixel and shows
``-bwdist(point, metric)`` with ``imshow(..., [])`` for the Euclidean, city-block and chessboard metrics.
The text-only 7×7 examples of Fig. 2.12 (Euclidean DT of a small matrix) and Fig. 2.13 (city-block / chessboard
distance from the centre) are reproduced as value grids.

Usage: ``python scripts/ch02_distance_transform.py [--out outputs/ch02] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch02_preliminaries import distance_fixture_examples, point_distance_maps  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image, show_matrix  # noqa: E402

CH = "ch02"


def main(argv: list[str] | None = None) -> int:
    args = chapter_argparser(CH, __doc__.splitlines()[0]).parse_args(argv)
    _, out = resolve_dirs(args)
    written: list[Path] = []

    # --- Fig. 2.14: the script -------------------------------------------------------------------------------
    maps = point_distance_maps(201)
    written.append(save_image(out / "fig_2_14_point_image.png", maps["point"]))  # imshow(img)
    for metric in ("euclidean", "cityblock", "chessboard", "quasi-euclidean"):
        written.append(save_image(out / f"fig_2_14_point_dt_{metric.replace('-', '_')}.png", maps[metric],
                                  autoscale=True))  # imshow(imgDist, [])
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    imshow_matlab(axes[0], maps["point"], title="(a) 201x201 image, one pixel at (101,101)")
    for ax, metric, title in zip(axes[1:], ("euclidean", "cityblock", "chessboard"),
                                 ("(b) Euclidean, Eq. (2.10)", "(c) City-block, Eq. (2.11)",
                                  "(d) Chessboard, Eq. (2.12)")):
        imshow_matlab(ax, maps[metric], autoscale=True, title=title)
    fig.suptitle("Fig. 2.14  -bwdist(point, metric) displayed with imshow(..., [])")
    written.append(finish_figure(fig, out / "fig_2_14_point_dt.png", args.show))

    # --- Fig. 2.12: 7x7 matrix and its Euclidean distance transform ----------------------------------------
    fx = distance_fixture_examples()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    show_matrix(axes[0], fx["A12"].astype(int), title="(a) Binary image")
    show_matrix(axes[1], np.round(fx["D12"], 4), fmt="{:.4g}", title="(b) Euclidean distance transform, Eq. (2.9)",
                highlight=fx["A12"])
    fig.suptitle("Fig. 2.12  Distance transform of a small binary matrix")
    written.append(finish_figure(fig, out / "fig_2_12_dt7x7.png", args.show))

    # --- Fig. 2.13: city-block and chessboard distances from the centre pixel ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    show_matrix(axes[0], fx["C4"].astype(int), title="(a) City-block distance d4, Eq. (2.11)",
                highlight=fx["C4"] <= 2)
    show_matrix(axes[1], fx["C8"].astype(int), title="(b) Chessboard distance d8, Eq. (2.12)",
                highlight=fx["C8"] <= 2)
    fig.suptitle("Fig. 2.13  Distances from the centre of a 7x7 grid (shaded: r <= 2)")
    written.append(finish_figure(fig, out / "fig_2_13_center_distances.png", args.show))

    # --- key numbers ----------------------------------------------------------------------------------------
    print("Fig. 2.12(b) reproduced:", np.allclose(fx["D12"], fx["D12_book"], atol=1e-4))
    print("Fig. 2.12(b) distinct values:", np.unique(np.round(fx["D12"], 4)).tolist())
    print(f"Fig. 2.13 corner values: city-block {fx['C4'][0, 0]:.0f} (book 6), chessboard {fx['C8'][0, 0]:.0f} (book 3)")
    for metric in ("euclidean", "cityblock", "chessboard", "quasi-euclidean"):
        d = -maps[metric]
        print(f"Fig. 2.14 {metric:16s}: min {d.min():.4f} at centre, max {d.max():.4f} at the corners")
    print("figures written:")
    for p in written:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
