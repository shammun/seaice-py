"""Text-only demo of Book §2.1.1 / §2.1.3 — grayscale (Fig. 2.1), binary (Fig. 2.2) and indexed (Fig. 2.6) images.

No MATLAB script exists for these figures.  The grayscale image is ``rgb2gray_matlab(rgb.JPG)`` (MATLAB
``rgb2gray`` weights); an 11×11 window of it is printed next to the values printed in Fig. 2.1 (whose source
location is unknown — the values are illustrative, not load-bearing).  Fig. 2.2 is a synthetic 16×16 pattern;
Fig. 2.6 shows a tiny index matrix, its colormap and the resulting RGB image.

Usage: ``python scripts/ch02_image_types.py [--data data/book/ch02] [--out outputs/ch02] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch02_preliminaries import image_type_examples  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_book_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image, show_matrix  # noqa: E402

CH = "ch02"


def main(argv: list[str] | None = None) -> int:
    args = chapter_argparser(CH, __doc__.splitlines()[0]).parse_args(argv)
    data, out = resolve_dirs(args)
    I = load_book_image(CH, "rgb.jpg", data_dir=data)
    ex = image_type_examples(I)
    written: list[Path] = []

    # --- Fig. 2.1: pixel values in a grayscale image ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    imshow_matlab(axes[0], ex["gray"], title="Grayscale image: rgb2gray(rgb.JPG), 8-bit (256 levels)")
    axes[0].add_patch(plt.Rectangle((899.5, 199.5), 11, 11, fill=False, edgecolor="r", lw=1.5))
    show_matrix(axes[1], ex["gray_crop"], fmt="{:d}", cmap="Greys",
                highlight=(ex["gray_crop"].astype(float) - 150) / 105, title="11x11 window at rows 201-211, cols 901-911")
    show_matrix(axes[2], ex["fig_2_1_printed"], fmt="{:d}", cmap="Greys",
                highlight=(ex["fig_2_1_printed"].astype(float) - 150) / 105, title="Values printed in Fig. 2.1")
    fig.suptitle("Fig. 2.1  Pixel values in a grayscale image")
    written.append(finish_figure(fig, out / "fig_2_01_gray_values.png", args.show))

    # --- Fig. 2.2: binary image -----------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    show_matrix(axes[0], ex["pattern"].astype(int), cmap="Greys", title="Synthetic 16x16 binary pattern (B = 1)")
    imshow_matlab(axes[1], ex["binary"], title="rgb2gray(rgb.JPG) > 128 (logical image)")
    fig.suptitle("Fig. 2.2  Binary images: logical arrays of 0s and 1s")
    written.append(finish_figure(fig, out / "fig_2_02_binary_pattern.png", args.show))
    written.append(save_image(out / "binary_threshold_128.png", ex["binary"]))

    # --- Fig. 2.6: indexed image ------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    show_matrix(axes[0], ex["indexed"], fmt="{:d}", cmap="Greys", highlight=np.zeros_like(ex["indexed"]),
                title="Index matrix (0-based)")
    cm = ex["cmap"]
    axes[1].imshow(cm[:, None, :], aspect="auto", interpolation="nearest")
    for k, row in enumerate(cm):
        axes[1].text(0, k, f"{k}: [{row[0]:.2f} {row[1]:.2f} {row[2]:.2f}]", ha="center", va="center",
                     color="w" if row.sum() < 1.2 else "k")
    axes[1].set_axis_off()
    axes[1].set_title("Colormap (m x 3, values in [0,1])")
    axes[2].imshow(ex["indexed_rgb"], interpolation="nearest")
    axes[2].set_axis_off()
    axes[2].set_title("Resulting RGB image")
    fig.suptitle("Fig. 2.6  Structure of an indexed image")
    written.append(finish_figure(fig, out / "fig_2_06_indexed_image.png", args.show))

    print(f"gray image {ex['gray'].shape} dtype {ex['gray'].dtype}, {ex['bits']} bits -> {ex['levels']} levels, "
          f"min {ex['gray'].min()} max {ex['gray'].max()}")
    print("11x11 window (rows 201-211, cols 901-911, 1-based):")
    print(ex["gray_crop"])
    print(f"binary image (>128): {ex['binary'].mean() * 100:.2f}% ones")
    print("figures written:")
    for p in written:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
