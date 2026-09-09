"""Text-only demo of Book §2.5 — discrete convolution, Eqs. (2.13)–(2.15), Fig. 2.15.

No MATLAB script exists.  A 6×6 integer image is convolved with a 3×3 kernel (``conv2`` semantics, Eq. 2.14);
the nine terms of Eq. (2.15) are written out at one pixel; MATLAB ``imfilter`` (correlation) is shown next to
true convolution to make the flip explicit.

Usage: ``python scripts/ch02_convolution.py [--out outputs/ch02] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch02_preliminaries import convolution_example  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.plotting import finish_figure, show_matrix  # noqa: E402

CH = "ch02"


def main(argv: list[str] | None = None) -> int:
    args = chapter_argparser(CH, __doc__.splitlines()[0]).parse_args(argv)
    _, out = resolve_dirs(args)
    ex = convolution_example(seed=0)
    x, y = ex["x"], ex["y"]
    written: list[Path] = []

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    hl = np.zeros_like(ex["f"])
    hl[x - 1:x + 2, y - 1:y + 2] = 0.6
    hl[x, y] = 1.0
    show_matrix(axes[0], ex["f"].astype(int), fmt="{:d}", highlight=hl, title=f"Image f (3x3 neighbourhood of ({x + 1},{y + 1}))")
    show_matrix(axes[1], ex["w"], fmt="{:g}", highlight=np.abs(ex["w"]) / 2, title="Kernel w(s,t), centre = w(0,0)")
    show_matrix(axes[2], ex["h_conv2"], fmt="{:g}", highlight=(ex["h_conv2"] - ex["h_conv2"].min()) /
                (np.ptp(ex["h_conv2"]) + 1e-9), title="h = conv2(f, w, 'same')  (Eq. 2.14)")
    show_matrix(axes[3], ex["h_imfilter_corr"], fmt="{:g}", highlight=(ex["h_imfilter_corr"] - ex["h_imfilter_corr"].min())
                / (np.ptp(ex["h_imfilter_corr"]) + 1e-9), title="imfilter(f, w) = correlation (kernel not flipped)")
    fig.suptitle(f"Fig. 2.15  Convolution with a 3x3 kernel; Eq. (2.15) at ({x + 1},{y + 1}) gives h = {ex['h_xy']:g}")
    written.append(finish_figure(fig, out / "fig_2_15_convolution.png", args.show))

    print("f =\n", ex["f"].astype(int))
    print("w =\n", ex["w"])
    print(f"Eq. (2.15) at (x, y) = ({x}, {y}) [0-based]:")
    total = 0.0
    for s, t, ws, fv in ex["terms"]:
        print(f"  w({s:+d},{t:+d}) * f(x{-s:+d}, y{-t:+d}) = {ws:g} * {fv:g} = {ws * fv:g}")
        total += ws * fv
    print(f"  sum = {total:g}  (conv2 gives {ex['h_conv2'][x, y]:g}, conv_at gives {ex['h_xy']:g})")
    print("h = conv2(f, w, 'same') =\n", ex["h_conv2"])
    print("imfilter(f, w) (correlation) =\n", ex["h_imfilter_corr"])
    print("figures written:")
    for p in written:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
