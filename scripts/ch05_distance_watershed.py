"""Port of ``MATLAB_ROOT/ch5/distance_watershed.m`` — watershed of the inverse distance map of the Otsu mask of
``q.jpg`` (§5.1.2; **Fig. 5.8** with the chessboard metric: (a) mask, (b) inverse DT, (d) regional minima punched
out in black, (e) watershed line, (f) segmented floes; Fig. 5.9 rows for the Euclidean / city-block metrics).

Literal steps: ``img = im2bw(img, graythresh(img))`` (on the **RGB**: ``graythresh`` histograms all planes →
130/255), ``imgDist = -bwdist(~img, metric)``, ``Dis_img = imregionalmin(imgDist)``, ``dis = Dis_img .* bw``,
``[p, q] = find(dis == 1)`` (red ``+`` overlay), ``bw0(dis) = 0``, ``imgLabel = watershed(imgDist)``,
``bgm = imgLabel == 0``, ``img(bgm) = 0``, ``img = bwareaopen(img, 5)``, ``label2rgb(bwlabel(img))``.

Usage: ``python scripts/ch05_distance_watershed.py [--metric chessboard|cityblock|euclidean|quasi-euclidean|all]
[--min-area 5] [--data data/book/ch05] [--out outputs/ch05] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

from seaice.ch05_watershed import BOOK_PARAMS, distance_watershed, otsu_mask  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, label2rgb, save_image  # noqa: E402

CH = "ch05"
METRICS = ["chessboard", "cityblock", "euclidean", "quasi-euclidean"]
#: file prefixes: the script's chessboard run is Fig. 5.8, Euclidean / city-block are the rows of Fig. 5.9.
PREFIX = {"chessboard": "fig_5_08", "euclidean": "fig_5_09_euclidean", "cityblock": "fig_5_09_cityblock",
          "quasi-euclidean": "sec_5_1_2_quasi_euclidean"}
PANEL = {"chessboard": {"bw": "a_binary", "imgDist": "b_inverse_chessboard_dt", "minima_overlay": "d_regional_minima",
                        "ridge": "e_watershed_line", "seg_ao": "f_segmented"}}


def run(rgb, metric: str, min_area: int, out: Path, show: bool, written: list[Path]) -> None:
    bw = otsu_mask(rgb)
    res = distance_watershed(bw, metric, min_area)
    M, N = bw.shape
    print(f"\n=== distance_watershed.m, metric = '{metric}' ({M}x{N}, {int(bw.sum())} ice px, Otsu on RGB) ===")
    print(f"imgDist = -bwdist(~img): min {float(res.imgDist.min()):g}; regional minima: {res.n_minima} components, "
          f"{res.n_minimum_px} pixels (inside ice: {int(res.dis.sum())})")
    print(f"watershed: {res.n_basins} basins, ridge {int(res.ridge.sum())} px ({int((res.ridge & bw).sum())} inside ice); "
          f"floes after bwareaopen(…, {min_area}): {res.n_floes}")
    pts = res.minima_points + 1
    print("minimum pixels (1-based row, col), find order: " + " ".join(f"({r},{c})" for r, c in pts[:20].tolist())
          + (" …" if len(pts) > 20 else ""))
    pre = PREFIX[metric]
    names = PANEL.get(metric, {"bw": "_binary", "imgDist": "_inverse_dt", "minima_overlay": "_regional_minima",
                               "ridge": "_watershed_line", "seg_ao": "_segmented"})
    written.append(save_image(out / f"{pre}{names['bw']}.png", res.bw))
    written.append(save_image(out / f"{pre}{names['imgDist']}.png", res.imgDist, autoscale=True))
    written.append(save_image(out / f"{pre}_minima_mask.png", ~res.minima))
    written.append(save_image(out / f"{pre}{names['minima_overlay']}.png", res.minima_overlay))
    written.append(save_image(out / f"{pre}{names['ridge']}.png", res.ridge))
    written.append(save_image(out / f"{pre}_segmented_before_areaopen.png", res.seg))
    written.append(save_image(out / f"{pre}{names['seg_ao']}.png", res.seg_ao))
    written.append(save_image(out / f"{pre}_labels.png", label2rgb(res.labels)))
    fig, axes = plt.subplots(2, 3, figsize=(13, 10))
    imshow_matlab(axes[0, 0], res.bw, title="(a) img = im2bw(img, graythresh(img))")
    imshow_matlab(axes[0, 1], res.imgDist, autoscale=True, title=f"(b) imgDist = -bwdist(~img, '{metric}')")
    imshow_matlab(axes[0, 2], res.bw, title=f"(c) regional minima ({res.n_minima} comps / {res.n_minimum_px} px)")
    axes[0, 2].plot(res.minima_points[:, 1], res.minima_points[:, 0], "r+", ms=6)
    imshow_matlab(axes[1, 0], res.minima_overlay, title="(d) bw0: minima punched out")
    imshow_matlab(axes[1, 1], res.ridge, title=f"(e) bgm = watershed(imgDist) == 0 ({res.n_basins} basins)")
    imshow_matlab(axes[1, 2], label2rgb(res.labels), title=f"(f) bwareaopen(img(~bgm), {min_area}): {res.n_floes} floes")
    fig.suptitle(f"Fig. 5.8-style panels — distance_watershed.m with the '{metric}' metric")
    written.append(finish_figure(fig, out / f"{pre}_panels.png", show))


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--metric", default=BOOK_PARAMS["metric_fig_5_8"], choices=METRICS + ["all"],
                   help="bwdist metric (script: chessboard = Fig. 5.8; 'all' also runs the Fig. 5.9 / 5.11 metrics)")
    p.add_argument("--min-area", type=int, default=BOOK_PARAMS["min_area"], help="bwareaopen threshold (script: 5)")
    p.add_argument("--image", default="q.jpg")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        rgb, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: skipped: private image absent ({exc})")
        return 0
    written: list[Path] = []
    for m in (METRICS if args.metric == "all" else [args.metric]):
        run(rgb, m, args.min_area, out, args.show, written)
    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
