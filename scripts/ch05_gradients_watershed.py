"""Port of ``MATLAB_ROOT/ch5/gradients_watershed.m`` — watershed of the Sobel gradient of ``q.jpg`` (§5.1.1):
**Fig. 5.5** (a) gradient magnitude, (c) watershed labels, (d) ridges superimposed in black; **Fig. 5.6** (a) the
gradient after a 7×7 square close-opening, (b) its watershed labels, (c) overlay.

Literal steps: ``I = rgb2gray(I)``; ``bw = im2bw(I, graythresh(I))`` (unused); ``hy = fspecial('sobel'); hx = hy';
Iy = imfilter(double(I), hy, 'replicate'); Ix = imfilter(double(I), hx, 'replicate'); g = sqrt(Ix.^2 + Iy.^2)``
(0–255 units, unscaled kernels); ``l = watershed(g); wr = l == 0; f = I; f(wr) = 0``; ``g2 = imclose(imopen(g,
ones(7,7)), ones(7,7)); l2 = watershed(g2); ...``.  ``imshow(l, [])`` (autoscaled labels) is reproduced as such.

Usage: ``python scripts/ch05_gradients_watershed.py [--smooth 7] [--data data/book/ch05] [--out outputs/ch05] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

from seaice.ch05_watershed import BOOK_PARAMS, gradient_watershed  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402

CH = "ch05"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--smooth", type=int, default=BOOK_PARAMS["close_open_size"],
                   help="side of the square SE of the close-opening (script: 7; 0 = skip the second half)")
    p.add_argument("--image", default="q.jpg")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        rgb, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: skipped: private image absent ({exc})")
        return 0
    gray = rgb2gray_matlab(rgb)
    res = gradient_watershed(gray, args.smooth or None)
    M, N = gray.shape
    print(f"gradients_watershed.m on {args.image} ({M}x{N}); Otsu mask (gray) {int(res.bw.sum())} ice px (unused by the script)")
    print(f"Sobel magnitude g: max {float(res.g.max()):.4f}, mean {float(res.g.mean()):.4f} (0–255 units, unscaled kernels)")
    print(f"watershed(g): {res.n_basins} basins, {int(res.ridge.sum())} ridge px  (book Fig. 5.5(c): far too many lines)")
    written = [save_image(out / "fig_5_05a_sobel_gradient.png", res.g, autoscale=True),
               save_image(out / "fig_5_05c_watershed_labels.png", res.L, autoscale=True),
               save_image(out / "sec_5_1_1_ridges.png", res.ridge),
               save_image(out / "fig_5_05d_overlay.png", res.overlay)]
    if res.g2 is not None:
        print(f"g2 = imclose(imopen(g, ones({args.smooth}))): max {float(res.g2.max()):.4f}; watershed(g2): "
              f"{res.n_basins2} basins, {int(res.ridge2.sum())} ridge px  (book Fig. 5.6(b): 'fewer, still extraneous')")
        written += [save_image(out / "fig_5_06a_smoothed_gradient.png", res.g2, autoscale=True),
                    save_image(out / "fig_5_06b_watershed_labels.png", res.L2, autoscale=True),
                    save_image(out / "sec_5_1_1_ridges_smoothed.png", res.ridge2),
                    save_image(out / "fig_5_06c_overlay.png", res.overlay2)]
    rows = 2 if res.g2 is not None else 1
    fig, axes = plt.subplots(rows, 3, figsize=(13, 5 * rows), squeeze=False)
    imshow_matlab(axes[0, 0], res.g, autoscale=True, title="Fig. 5.5(a) g = |Sobel| (imshow(g, []))")
    imshow_matlab(axes[0, 1], res.L, autoscale=True, title=f"(c) watershed(g): {res.n_basins} basins (imshow(l, []))")
    imshow_matlab(axes[0, 2], res.overlay, title="(d) f = I; f(wr) = 0")
    if res.g2 is not None:
        imshow_matlab(axes[1, 0], res.g2, autoscale=True, title=f"Fig. 5.6(a) g2 = close-open (ones({args.smooth}))")
        imshow_matlab(axes[1, 1], res.L2, autoscale=True, title=f"(b) watershed(g2): {res.n_basins2} basins")
        imshow_matlab(axes[1, 2], res.overlay2, title="(c) f2 = I; f2(wr2) = 0")
    fig.suptitle("§5.1.1  Watershed segmentation using gradients (gradients_watershed.m)")
    written.append(finish_figure(fig, out / "fig_5_05_5_06_panels.png", args.show))
    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
