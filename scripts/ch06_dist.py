"""Port of ``MATLAB_ROOT/ch6/for test/dist.m`` — §6.3.3 seeds and initial circles on ``sea_ice_test.jpg``.

Literal steps::

    se = strel('disk', 5);  r = 15;
    [I1, MAP] = imread('C:\\Users\\qinz\\Desktop\\...\\sea_ice_test.jpg');   % rewritten to load_image
    I  = rgb2gray(I1);
    bw = im2bw(I', graythresh(I'));            % NOTE the TRANSPOSE -- everything downstream is transposed
    img_Dist = bwdist(~bw, 'cityblock');
    imgDist  = -bwdist(~bw, 'cityblock');  imgDist(~bw) = -inf;
    Dis_img  = imregionalmin(imgDist);
    dis  = Dis_img .* bw;      dis1 = imdilate(dis, se);
    [p, q] = find(dis == 1);                   % column-major order
    ... six figures: dis, dis1, Dis_img, imshow(img_Dist, []), bw + green '+',
        bw + red '+' seeds and blue circles of the fixed radius r = 15,
        bw + circles of r0 = |img_Dist(round(cy), round(cx))|/sqrt(2)  (footnote 4, p. 135; r0 == 0 -> 2),
        label2rgb(bwlabel(bw, 4), @jet, [1 1 1], 'shuffle') + the same circles.

Two literalisms are preserved and flagged: the **hard-coded absolute path** of line 6 is replaced by
``seaice.core.io.load_image('ch06', 'sea_ice_test.jpg')`` (the only possible port), and the **transpose** of
line 8 is reproduced by default (``--no-transpose`` runs the same code in the normal frame; the two masks are
exact transposes of each other because ``graythresh`` is transpose invariant).

Usage: ``python scripts/ch06_dist.py [--transpose/--no-transpose] [--se-radius 5] [--radius 15]
[--data data/book/ch06] [--out outputs/ch06] [--show]``
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch06_gvf_snake import CIRCLE_T, CITYBLOCK_RADIUS_DIVISOR  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.connectivity import label_components  # noqa: E402
from seaice.core.distance import bwdist  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import matlab_round, rgb2gray_matlab  # noqa: E402
from seaice.core.morphology import imdilate, imregionalmin, strel  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, label2rgb, save_image  # noqa: E402
from seaice.core.regionprops import regionprops  # noqa: E402
from seaice.core.threshold import graythresh, im2bw  # noqa: E402

CH = "ch06"


def circle_overlay(ax, centroids: np.ndarray, radii: np.ndarray) -> None:
    """``plot(cen(1), cen(2), 'r+')`` and ``plot(x, y, 'b')`` of ``dist.m`` lines 33–34 / 47–48 / 63–64."""
    for (cx, cy), r in zip(centroids, radii):
        ax.plot(cx - 1, cy - 1, "r+", ms=5, mew=1)
        ax.plot(cx - 1 + r * np.cos(CIRCLE_T), cy - 1 + r * np.sin(CIRCLE_T), "b-", lw=0.6)


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--transpose", action=argparse.BooleanOptionalAction, default=True,
                   help="binarise the transposed image, as dist.m line 8 does (default: the script's behaviour)")
    p.add_argument("--se-radius", type=int, default=5, help="strel('disk', r) of line 3 (script: 5)")
    p.add_argument("--radius", type=float, default=15.0, help="fixed initial radius of line 4 (script: 15)")
    p.add_argument("--image", default="sea_ice_test.jpg")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        rgb, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: skipped: private image absent ({exc})")
        return 0

    t0 = time.time()
    I = rgb2gray_matlab(rgb)
    src = I.T if args.transpose else I
    level, _ = graythresh(src)
    bw = im2bw(src, level)
    bw1 = bw
    img_dist = bwdist(~bw1, "cityblock")
    imgd = -img_dist.astype(np.float64)
    imgd[~bw1] = -np.inf
    dis_img = imregionalmin(imgd)
    dis = (dis_img & bw1).astype(np.float64)  # Dis_img .* bw1  (logical .* logical -> double)
    se = strel("disk", args.se_radius)
    dis1 = imdilate(dis, se)
    # [p, q] = find(dis == 1): MATLAB's column-major order
    q_idx, p_idx = np.nonzero(dis.T == 1)  # column-major: iterate columns first
    pq = np.column_stack([p_idx, q_idx])   # (row, col), 0-based

    label, num = label_components(dis1 != 0, 8), None
    num = int(label.max())
    stats = regionprops(label, ("Centroid",))
    centroids = np.array([s.Centroid for s in stats]).reshape(-1, 2)
    M, N = bw.shape
    radii_fixed = np.full(num, float(args.radius))
    radii_dt = np.zeros(num)
    for n in range(num):
        cx, cy = centroids[n]
        rr = int(matlab_round(cy)) - 1
        cc = int(matlab_round(cx)) - 1
        r0 = abs(float(img_dist[min(max(rr, 0), M - 1), min(max(cc, 0), N - 1)]) / CITYBLOCK_RADIUS_DIVISOR)
        radii_dt[n] = 2.0 if r0 == 0 else r0

    l4 = label_components(bw, 4)
    print(f"=== dist.m on {args.image} ({'transposed' if args.transpose else 'normal'} frame) ===")
    print(f"image {rgb.shape[0]}x{rgb.shape[1]} RGB -> gray; bw is {M}x{N}")
    print(f"graythresh(I{chr(39) if args.transpose else ''}) = {level:.6f} (255*level = {255 * level:g}); "
          f"ice pixels {int(bw.sum())} ({100.0 * bw.mean():.2f} %)")
    print(f"bwdist(~bw, 'cityblock'): max {float(img_dist.max()):.0f}; "
          f"imregionalmin(-D with -Inf background): {int(dis_img.sum())} px, "
          f"{int(dis.sum())} of them inside the ice")
    print(f"find(dis == 1): {pq.shape[0]} points, first five (1-based row, col) "
          f"{[(int(r) + 1, int(c) + 1) for r, c in pq[:5]]}")
    print(f"imdilate(dis, strel('disk', {args.se_radius})) -> bwlabel(.., 8): {num} seeds")
    print(f"radii from D(seed)/sqrt(2): min {radii_dt.min():.4f}, median {np.median(radii_dt):.4f}, "
          f"max {radii_dt.max():.4f}; {int((radii_dt == 2.0).sum())} seeds fell back to r = 2")
    print(f"bwlabel(bw, 4): {int(l4.max())} connected ice components")

    written = [
        save_image(out / "sec_6_3_3_dist_a_local_maxima.png", dis != 0),
        save_image(out / "sec_6_3_3_dist_b_dilated_maxima.png", dis1 != 0),
        save_image(out / "sec_6_3_3_dist_c_regional_minima.png", dis_img),
        save_image(out / "sec_6_3_3_dist_d_distance_transform.png", img_dist, autoscale=True),
    ]

    fig, ax = plt.subplots(figsize=(5, 11) if M > N else (11, 5))
    imshow_matlab(ax, bw, title=f"bw + local maxima ('g+', {pq.shape[0]} px)")
    ax.plot(pq[:, 1], pq[:, 0], "g+", ms=3, mew=0.6)
    written.append(finish_figure(fig, out / "sec_6_3_3_dist_e_local_maxima_overlay.png", args.show))

    for tag, radii, title in (("f_fixed_radius", radii_fixed, f"seeds and circles of the fixed r = {args.radius:g}"),
                              ("g_dt_radius", radii_dt, "seeds and circles of r = D(seed)/sqrt(2)")):
        fig, ax = plt.subplots(figsize=(5, 11) if M > N else (11, 5))
        imshow_matlab(ax, bw, title=title)
        circle_overlay(ax, centroids, radii)
        written.append(finish_figure(fig, out / f"sec_6_3_3_dist_{tag}.png", args.show))

    fig, ax = plt.subplots(figsize=(5, 11) if M > N else (11, 5))
    imshow_matlab(ax, label2rgb(l4), title=f"label2rgb(bwlabel(bw, 4)) ({int(l4.max())} components) + circles")
    circle_overlay(ax, centroids, radii_dt)
    written.append(finish_figure(fig, out / "sec_6_3_3_dist_h_labels_with_circles.png", args.show))

    print(f"\nwall clock {time.time() - t0:.1f} s")
    print("figures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
