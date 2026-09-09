"""Text-only demo of Book §2.8 — nearest-neighbour, bilinear and bicubic interpolation, Eqs. (2.32)–(2.41).

No MATLAB script exists.  A 32×32 crop of ``rgb2gray(rgb.JPG)`` (MATLAB ``G(700:731, 900:931)``) is upsampled on
the grid ``1:0.4:32`` (as ``interp2`` would) and by ×4 with the ``imresize`` pixel-centre convention, with the
three methods side by side.  These panels are section illustrations (``sec_2_8_*.png``), not reproductions of
Figs. 2.22–2.24 (which are concept diagrams); the Keys kernel of Eq. (2.41) / Fig. 2.25 is plotted.

Usage: ``python scripts/ch02_interpolation.py [--data data/book/ch02] [--out outputs/ch02] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch02_preliminaries import interpolation_demo  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_book_image  # noqa: E402
from seaice.core.plotting import finish_figure, save_image  # noqa: E402

CH = "ch02"


def main(argv: list[str] | None = None) -> int:
    args = chapter_argparser(CH, __doc__.splitlines()[0]).parse_args(argv)
    data, out = resolve_dirs(args)
    I = load_book_image(CH, "rgb.jpg", data_dir=data)
    ex = interpolation_demo(I)
    written: list[Path] = []

    titles = {"nearest": "Nearest neighbour, Eq. (2.33)", "bilinear": "Bilinear, Eqs. (2.35)-(2.38)",
              "bicubic": f"Bicubic (Keys a = {ex['a']}), Eqs. (2.40)-(2.41)"}
    for prefix, label, fname in (("grid_", "interp2 grid 1:0.4:32", "sec_2_8_interp2_grid.png"),
                                 ("resize_", "resize x4 (imresize pixel-centre convention)",
                                  "sec_2_8_resize_x4.png")):
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
        axes[0].imshow(ex["P"], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        axes[0].set_title("Input crop P (32x32)")
        for ax, m in zip(axes[1:], ("nearest", "bilinear", "bicubic")):
            ax.imshow(ex[prefix + m], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(titles[m])
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(f"Section 2.8  Image interpolation — {label}")
        written.append(finish_figure(fig, out / fname, args.show))
    for m in ("nearest", "bilinear", "bicubic"):
        written.append(save_image(out / f"interp_resize4_{m}.png", ex["resize_" + m] / 255.0))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ex["kernel_x"], ex["kernel_keys"], label=f"rc(x), a = {ex['a']} (MATLAB)")
    ax.plot(ex["kernel_x"], ex["kernel_keys_opencv"], "--", label="rc(x), a = -0.75 (OpenCV)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("rc(x)")
    ax.legend()
    ax.set_title("Fig. 2.25 / Eq. (2.41)  Cubic convolution kernel")
    written.append(finish_figure(fig, out / "fig_2_25_keys_kernel.png", args.show))

    P = ex["P"]
    print(f"crop P: shape {P.shape}, min {P.min():.0f}, max {P.max():.0f}; query grid {ex['U'].shape}")
    for m in ("nearest", "bilinear", "bicubic"):
        g, r = ex["grid_" + m], ex["resize_" + m]
        sub = g[::5, ::5]  # step 0.4 → every 5th query is a lattice point (0, 2, 4, ...)
        lattice_ok = bool(np.allclose(sub, P[::2, ::2][:sub.shape[0], :sub.shape[1]], atol=1e-9))
        print(f"{m:9s}: grid {g.shape} range [{np.nanmin(g):.2f}, {np.nanmax(g):.2f}]; resize {r.shape} "
              f"range [{r.min():.2f}, {r.max():.2f}]; lattice values reproduced: {lattice_ok}")
    print("keys_kernel(0, +-1, +-2) =", ex["kernel_keys"][[250, 150, 350, 50, 450]].round(6).tolist())
    print("figures written:")
    for p in written:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
