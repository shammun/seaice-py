"""Port of ``MATLAB_ROOT/ch5/direct_watershed.m`` — watershed of the gray image itself (§5.1 p. 89: the gray
level is "rarely a good segmentation function"; no book figure).

The MATLAB script converts ``q.jpg`` to gray, computes ``watershed(im)`` on the uint8 image, shows the ridge mask,
superimposes the ridges in black on the gray image and colours ``bwlabel`` of the result with
``label2rgb(..., 'jet', 'k', 'shuffle')``.  Figures: ``sec_5_1_direct_watershed_ridges.png``,
``sec_5_1_direct_watershed_overlay.png``, ``sec_5_1_direct_watershed_labels.png`` (colours display-only) and a
three-panel summary ``sec_5_1_direct_watershed_panels.png``.

Usage: ``python scripts/ch05_direct_watershed.py [--image q.jpg] [--data data/book/ch05] [--out outputs/ch05] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

from seaice.ch05_watershed import direct_watershed  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, label2rgb, save_image  # noqa: E402

CH = "ch05"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="q.jpg", help="book image (default: %(default)s)")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        rgb, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: skipped: private image absent ({exc})")
        return 0
    gray = rgb2gray_matlab(rgb)  # im = rgb2gray(img)
    res = direct_watershed(gray)
    M, N = gray.shape
    print(f"direct_watershed.m on {args.image}: {M}x{N} gray; Otsu level (gray) {float(res.bw.mean()):.4f} ice fraction")
    print(f"watershed(im): {res.n_basins} basins, {int(res.ridge.sum())} ridge pixels; "
          f"bwlabel(im with ridges = 0): {res.n_components} components")

    written = [save_image(out / "sec_5_1_direct_watershed_ridges.png", res.ridge),
               save_image(out / "sec_5_1_direct_watershed_overlay.png", res.overlay),
               save_image(out / "sec_5_1_direct_watershed_labels.png", label2rgb(res.labels))]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    imshow_matlab(axes[0], res.ridge, title=f"bgm = watershed(im) == 0  ({res.n_basins} basins)")
    imshow_matlab(axes[1], res.overlay, title="im(bgm) = 0")
    imshow_matlab(axes[2], label2rgb(res.labels), title=f"label2rgb(bwlabel(im))  ({res.n_components} regions)")
    fig.suptitle("§5.1  Watershed of the gray image itself (direct_watershed.m) — massive over-segmentation")
    written.append(finish_figure(fig, out / "sec_5_1_direct_watershed_panels.png", args.show))
    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
