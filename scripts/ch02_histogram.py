"""Port of ``MATLAB_ROOT/ch2/histogram.m`` — channel images (Fig. 2.3), gray histogram (Fig. 2.7), RGB histograms (Fig. 2.8).

Book §2.1.2.1 (Fig. 2.3), §2.2 Eqs. (2.7)–(2.8), Figs. 2.7–2.8.  The MATLAB script shows the colour image, the
three channels with pure red/green/blue colormaps (saved by ``saveas`` as ``red.png``/``green.png``/``blue.png``),
``imhist`` of ``rgb2gray(I)`` and the three channel histograms with ``ylim([0 80000])``.

Data note: the printed Fig. 2.7 uses a different (denser floe field) image that is not shipped with chapter 2
and matches none of the images shipped with ch3–ch5 (histogram peaks compared); Fig. 2.7 is therefore
illustrated here with ``rgb2gray(rgb.JPG)`` — a documented **substitute**, not a reproduction.

Usage: ``python scripts/ch02_histogram.py [--data data/book/ch02] [--out outputs/ch02] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from seaice.ch02_preliminaries import channel_histograms, gray_histogram  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402

CH = "ch02"


def _ramp(channel: int) -> ListedColormap:
    """``colormap([[0:1/255:1]', zeros(256,1), zeros(256,1)])`` and its green/blue analogues."""
    cm = np.zeros((256, 3))
    cm[:, channel] = np.arange(256) / 255.0
    return ListedColormap(cm)


def main(argv: list[str] | None = None) -> int:
    args = chapter_argparser(CH, __doc__.splitlines()[0]).parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        I, _ = load_image(CH, "rgb.jpg", allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:  # private book image absent: scripts never use the public substitute
        print(f"SKIP {Path(__file__).name}: {exc}")
        return 0
    ch = channel_histograms(I)
    written: list[Path] = []

    # --- Fig. 2.3: colour image and the three channel images (red.png / green.png / blue.png) ---------------
    for name, key, k in (("red", "R", 0), ("green", "G", 1), ("blue", "B", 2)):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.imshow(ch[key], cmap=_ramp(k), vmin=0, vmax=255)
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)  # iptsetpref('ImshowBorder','tight')
        written.append(finish_figure(fig, out / f"{name}.png", args.show))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    imshow_matlab(axes[0, 0], I, title="Color image")
    for ax, key, k, title in zip(axes.flat[1:], ("R", "G", "B"), (0, 1, 2),
                                 ("Red component image", "Green component image", "Blue component image")):
        ax.imshow(ch[key], cmap=_ramp(k), vmin=0, vmax=255)
        ax.set_axis_off()
        ax.set_title(title)
    px = ch["pixel_1076_675"]
    for ax in axes.flat:
        ax.plot(674, 1075, ".k", markersize=8)
    fig.suptitle(f"Fig. 2.3  Pixel values in an RGB image: R(1076,675)={px[0]}, G={px[1]}, B={px[2]}")
    written.append(finish_figure(fig, out / "fig_2_03_rgb_channels.png", args.show))

    # --- Fig. 2.7: grayscale image + imhist (substitute image, see module docstring) --------------------------
    G = rgb2gray_matlab(I)
    gh = gray_histogram(G)
    written.append(save_image(out / "gray.png", G))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    imshow_matlab(axes[0], G, title="(a) Grayscale image  [substitute: rgb2gray(rgb.JPG)]")
    axes[1].bar(gh["x"], gh["counts"], width=1.0, color="k")
    axes[1].set_xlim(-0.5, 255.5)
    axes[1].set_xlabel("Intensity value")
    axes[1].set_ylabel("Number of pixels")
    axes[1].set_title("(b) imhist(I), Eq. (2.7)")
    fig.suptitle("Fig. 2.7 (illustrated with rgb.JPG; the printed figure uses a different image)")
    written.append(finish_figure(fig, out / "fig_2_07_gray_hist.png", args.show))

    # --- Fig. 2.8: colour image + RGB histograms --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    imshow_matlab(axes[0], I, title="(a) Color image")
    x = ch["x"]
    axes[1].plot(x, ch["y_r"], "r-", x, ch["y_g"], "g--", x, ch["y_b"], "b:", linewidth=2.5)
    axes[1].legend(["Red", "Green", "Blue"], loc="best", frameon=False)
    axes[1].set_xlim(0, 255)
    axes[1].set_ylim(0, 80000)
    axes[1].set_xlabel("Intensity value")
    axes[1].set_ylabel("Number of pixels")
    axes[1].set_title("(b) Channel histograms")
    fig.suptitle("Fig. 2.8  RGB image and its channel histograms")
    written.append(finish_figure(fig, out / "fig_2_08_rgb_hist.png", args.show))

    # --- key numbers ----------------------------------------------------------------------------------------
    print(f"I(1076,675) = {px.tolist()}   (book: 28, 76, 114)")
    print(f"gray: size {G.shape}, num == imhist counts: {np.array_equal(gh['num'], gh['counts'])}, "
          f"sum(GP) = {gh['GP'].sum():.12f}, peak {gh['counts'].max()} @ level {int(gh['counts'].argmax())}")
    for key, name in (("y_r", "R"), ("y_g", "G"), ("y_b", "B")):
        y = ch[key]
        print(f"{name} histogram: peak {int(y.max())} @ level {int(y.argmax())}")
    print("figures written:")
    for p in written:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
