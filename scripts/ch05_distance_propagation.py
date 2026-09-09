"""Port of ``MATLAB_ROOT/ch5/distance_propagation.m`` — inverse distance map of a single point (propagation shape of
a metric, §5.1.2 pp. 93–94) and, with ``--source q.jpg`` (the script's commented lines), the contour plots of the
three distance transforms of the Fig. 5.8(a) mask that make up **Fig. 5.10** (with the zoom windows
cols 24–40 × rows 44–64 and cols 18–30 × rows 45–90).

Default = the literal script: 201×201 point image, ``imgDist = -bwdist(~img, 'cityblock')``, ``dist0 = imgDist +
|min|``, ``dist = uint8(dist0·255/max)``, ``image(dist, 'CDataMapping', 'scaled'); imcontour(imgDist)`` →
``sec_5_1_2_point_propagation_cityblock.png``.  ``--metric all`` runs the three metrics.

Usage: ``python scripts/ch05_distance_propagation.py [--source point|q.jpg] [--metric cityblock|euclidean|chessboard|all]
[--size 201] [--data data/book/ch05] [--out outputs/ch05] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch05_watershed import inverse_distance_map, otsu_mask  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import contour_overlay, finish_figure, imshow_matlab, imshow_scale, save_image  # noqa: E402
from seaice.core.synth import point_image  # noqa: E402

CH = "ch05"
METRICS = ["cityblock", "euclidean", "chessboard"]
#: Fig. 5.10(b)/(c) zoom windows (1-based MATLAB axes as printed: cols 24–40 × rows 44–64 and cols 18–30 × rows 45–90).
ZOOMS = [("fig_5_10b_zoom_cols24_40_rows44_64", (24, 40), (44, 64)), ("fig_5_10c_zoom_cols18_30_rows45_90", (18, 30), (45, 90))]


def run_point(size: int, metrics: list[str], out: Path, show: bool, written: list[Path]) -> None:
    img = point_image(size)  # img = zeros(201); img(101, 101) = 1
    bw = ~img  # img = ~img  → the script then takes bwdist(~img) = distance to the point
    for metric in metrics:
        res = inverse_distance_map(bw, metric)
        c = size // 2
        print(f"point {size}x{size}, {metric}: imgDist min {float(res.imgDist.min()):g} at corner, 0 at the centre "
              f"({c + 1}, {c + 1}); dist0 max {float(res.dist0.max()):g}; dist uint8 range [{int(res.dist.min())}, "
              f"{int(res.dist.max())}], centre value {int(res.dist[c, c])}")
        written.append(save_image(out / f"sec_5_1_2_point_inverse_distance_{metric}.png", res.imgDist, autoscale=True))
        path = out / f"sec_5_1_2_point_propagation_{metric}.png"
        contour_overlay(res.dist, res.imgDist, path, levels=12, show=show,
                        title=f"image(dist, 'CDataMapping', 'scaled') + imcontour(imgDist) — {metric}")
        written.append(path)


def run_q(rgb: np.ndarray, out: Path, show: bool, written: list[Path]) -> None:
    bw = otsu_mask(rgb)  # img = im2bw(img, graythresh(img))  (RGB → all planes histogrammed)
    maps = {m: inverse_distance_map(bw, m) for m in METRICS}
    for m, res in maps.items():
        print(f"q.jpg {m}: imgDist min {float(res.imgDist.min()):g}, dist uint8 max {int(res.dist.max())}")
        written.append(save_image(out / f"sec_5_1_2_q_inverse_distance_{m}.png", res.imgDist, autoscale=True))
    M, N = bw.shape
    panels = [("fig_5_10a_contours", None, None)] + [(name, xl, yl) for name, xl, yl in ZOOMS]
    for name, xl, yl in panels:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
        for ax, m in zip(axes, METRICS):
            res = maps[m]
            ax.imshow(imshow_scale(res.dist, autoscale=True), cmap="gray", vmin=0, vmax=1,
                      extent=(0.5, N + 0.5, M + 0.5, 0.5), interpolation="nearest")
            X, Y = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))
            ax.contour(X, Y, res.imgDist, levels=14, cmap="viridis", linewidths=0.8)
            if xl is not None:
                ax.set_xlim(xl[0], xl[1])
                ax.set_ylim(yl[1], yl[0])
            ax.set_title(f"-bwdist(~img, '{m}')")
        sub = "full image" if xl is None else f"zoom cols {xl[0]}–{xl[1]}, rows {yl[0]}–{yl[1]}"
        fig.suptitle(f"Fig. 5.10  Contours of the inverse distance maps of Fig. 5.8(a) ({sub})")
        written.append(finish_figure(fig, out / f"{name}.png", show))


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--source", default="point", choices=["point", "q.jpg"],
                   help="'point' = the script's 201×201 single-pixel image; 'q.jpg' = the commented lines (Fig. 5.10)")
    p.add_argument("--metric", default="cityblock", choices=METRICS + ["all"], help="metric for --source point")
    p.add_argument("--size", type=int, default=201, help="side of the point image (script: 201)")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    written: list[Path] = []
    if args.source == "point":
        run_point(args.size, METRICS if args.metric == "all" else [args.metric], out, args.show, written)
    else:
        try:
            rgb, _ = load_image(CH, "q.jpg", allow_fallback=False, data_dir=data, verbose=False)
        except FileNotFoundError as exc:
            print(f"SKIP {Path(__file__).name}: skipped: private image absent ({exc})")
            return 0
        run_q(rgb, out, args.show, written)
    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
