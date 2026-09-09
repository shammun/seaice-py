"""Port of ``MATLAB_ROOT/ch5/marker_watershed.m`` — marker-controlled watershed of the inverse city-block distance
map of the Otsu mask of ``q.jpg`` (§5.1.3, **Fig. 5.12**: (a) regional minima — "four regional minima consisting of
18 local minimum pixels", (b) marker = minima dilated by a 5-radius disk — "two connected regions", (c) imposed
distance map, (d) watershed line, (e) the two floes).

Literal steps: ``img = im2bw(img, graythresh(img))`` (RGB), ``imgDist = -bwdist(~img, 'cityblock')``,
``Dis_img = imregionalmin(imgDist)``, ``marker = imdilate(Dis_img, strel('disk', 5))`` (MATLAB's octagonal disk),
``imgDist = imimposemin(imgDist, marker)`` (single precision, −Inf markers), ``bw0(marker .* bw == 1) = 0``,
``imgLabel = watershed(imgDist)``, ``img(bgm) = 0``, ``bwareaopen(img, 5)``, ``label2rgb(bwlabel(img))``.
``--centroid-markers`` runs the script's commented block (one-pixel markers at ``floor(regionprops Centroid)``).

Usage: ``python scripts/ch05_marker_watershed.py [--metric cityblock] [--radius 5] [--centroid-markers] [--min-area 5]
[--data data/book/ch05] [--out outputs/ch05] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch05_watershed import BOOK_PARAMS, marker_watershed, otsu_mask  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.connectivity import count_components  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.morphology import imregionalmin, strel  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, label2rgb, save_image  # noqa: E402

CH = "ch05"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--metric", default="cityblock", choices=["cityblock", "chessboard", "euclidean", "quasi-euclidean"])
    p.add_argument("--radius", type=int, default=BOOK_PARAMS["marker_disk_radius"], help="strel('disk', r) (script: 5)")
    p.add_argument("--centroid-markers", action="store_true", help="use the commented one-pixel centroid markers")
    p.add_argument("--min-area", type=int, default=BOOK_PARAMS["min_area"])
    p.add_argument("--image", default="q.jpg")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        rgb, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: skipped: private image absent ({exc})")
        return 0
    bw = otsu_mask(rgb)
    res = marker_watershed(bw, args.metric, args.radius, args.centroid_markers, args.min_area)
    se = strel("disk", args.radius)
    literal = args.metric == "cityblock" and args.radius == BOOK_PARAMS["marker_disk_radius"] and not args.centroid_markers
    pre = "fig_5_12" if literal else f"sec_5_1_3_{args.metric}_r{args.radius}" + ("_centroid" if args.centroid_markers else "") + "_"
    M, N = bw.shape
    print(f"marker_watershed.m on {args.image} ({M}x{N}, {int(bw.sum())} ice px, Otsu on RGB), metric '{args.metric}'")
    print(f"regional minima of -bwdist: {res.n_minima} components, {res.n_minimum_px} pixels  (book p. 96: 4 / 18)")
    print(f"marker = imdilate(minima, strel('disk', {args.radius})) [{se.shape[0]}x{se.shape[1]}, {int(se.sum())} px octagon]: "
          f"{int(res.marker.sum())} px, {count_components(res.marker)} components  (book: 2); markers used: {res.n_markers}")
    if res.centroids is not None:
        print("centroid markers (regionprops Centroid, 1-based x, y): " + "; ".join(f"({x:.3f}, {y:.3f})" for x, y in res.centroids))
    imp = res.imposed
    print(f"imimposemin: class {imp.dtype}, {int(np.isneginf(imp).sum())} pixels at -Inf, finite range "
          f"[{float(imp[np.isfinite(imp)].min()):.4f}, {float(imp[np.isfinite(imp)].max()):.4f}]; "
          f"imregionalmin(imposed) == marker: {bool(np.array_equal(imregionalmin(imp), res.marker0 if res.marker0 is not None else res.marker))}")
    print(f"watershed(imposed): {res.n_basins} basins, ridge {int(res.ridge.sum())} px ({int((res.ridge & bw).sum())} inside ice); "
          f"floes after bwareaopen(…, {args.min_area}): {res.n_floes}  (book Fig. 5.12(e): 2)")

    written = [save_image(out / f"sec_5_1_3_inverse_{args.metric}_dt.png", res.imgDist0, autoscale=True),
               save_image(out / f"{pre}a_regional_minima.png", ~res.minima),
               save_image(out / f"{pre}b_marker.png", res.marker),
               save_image(out / f"{pre}c_imposed_distance.png", np.where(np.isfinite(imp), imp, imp[np.isfinite(imp)].min()), autoscale=True),
               save_image(out / "sec_5_1_3_marker_overlay.png" if literal else out / f"{pre.rstrip('_')}_marker_overlay.png", res.marker_overlay),
               save_image(out / f"{pre}d_watershed_line.png", res.ridge),
               save_image(out / f"{pre}e_segmented.png", res.seg_ao),
               save_image(out / (f"{pre.rstrip('_')}_labels.png" if not literal else "sec_5_1_3_labels.png"), label2rgb(res.labels))]
    if res.marker0 is not None:
        written.append(save_image(out / f"{pre.rstrip('_')}_marker0.png", res.marker0))
    fig, axes = plt.subplots(2, 3, figsize=(13, 10))
    imshow_matlab(axes[0, 0], ~res.minima, title=f"(a) ~imregionalmin(imgDist): {res.n_minima} minima / {res.n_minimum_px} px")
    imshow_matlab(axes[0, 1], res.marker if res.marker0 is None else res.marker0,
                  title=f"(b) marker = imdilate(minima, disk {args.radius}): {res.n_markers} regions")
    imshow_matlab(axes[0, 2], np.where(np.isfinite(imp), imp, imp[np.isfinite(imp)].min()), autoscale=True,
                  title="(c) imimposemin(imgDist, marker) (−Inf shown as min)")
    imshow_matlab(axes[1, 0], res.marker_overlay, title="bw0: marker punched out of the mask")
    imshow_matlab(axes[1, 1], res.ridge, title=f"(d) watershed line ({res.n_basins} basins)")
    imshow_matlab(axes[1, 2], label2rgb(res.labels), title=f"(e) segmented: {res.n_floes} floes")
    fig.suptitle(f"Fig. 5.12  Marker-controlled watershed (marker_watershed.m, '{args.metric}', disk r = {args.radius})")
    written.append(finish_figure(fig, out / f"{pre.rstrip('_')}_panels.png", args.show))
    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
