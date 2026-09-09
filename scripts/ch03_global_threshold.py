"""Book §3.1.1 (text only) — global thresholding with a hand-picked T, Eq. (3.1), and the histogram with the threshold
marker (Figs. 3.1–3.2).

No MATLAB file: the book chooses ``T = 125`` in the valley of the bimodal histogram of Fig. 3.2(a) (IC = 41.47 %).
That image is not shipped; ``2.jpg`` (its histogram twin, see ``analysis/ch03.md``) is used as the labelled
substitute, so the printed IC is for the substitute.  Otsu's automatic threshold (§3.1.1.1) is printed for
comparison.

Usage: ``python scripts/ch03_global_threshold.py [--T 125] [--data data/book/ch03] [--out outputs/ch03] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch03_ice_pixel_detection import fixed_threshold, otsu_segmentation, \
    plot_histogram_with_threshold  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402
from seaice.core.synth import bimodal_image  # noqa: E402

CH = "ch03"
SUBSTITUTE = "2.jpg"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--T", type=float, default=125.0, help="hand-picked global threshold (book Fig. 3.2: 125)")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    written: list[Path] = []

    # --- Fig. 3.1: sketch of a bimodal histogram (synthetic image) --------------------------------------------
    syn = bimodal_image(seed=0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    imshow_matlab(axes[0], syn, title="synthetic two-mode image")
    plot_histogram_with_threshold(axes[1], syn, T=125.0)
    axes[1].set_title("bimodal histogram with a threshold T in the valley")
    fig.suptitle("Fig. 3.1 (sketch)  Bimodal histogram — synthetic data")
    written.append(finish_figure(fig, out / "fig_3_01_bimodal_sketch.png", args.show))

    try:
        rgb, _ = load_image(CH, SUBSTITUTE, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:  # private book image absent: scripts never use the public substitute
        print(f"SKIP {Path(__file__).name} (book image part): {exc}")
        print("figures written:")
        for pth in written:
            print("  ", pth)
        return 0
    print(f"NOTE: Fig. 3.2(a) is not shipped; using {SUBSTITUTE} as the labelled SUBSTITUTE.")
    I = rgb2gray_matlab(rgb)
    T = args.T
    bw, ic = fixed_threshold(I, T)
    ot = otsu_segmentation(I)
    bw108, ic108 = fixed_threshold(I, 108.0)
    print(f"Eq. (3.1) with T = {T:g}: IC = {100 * ic:.2f} %   (book Fig. 3.2(c), unshipped image: 41.47 %)")
    print(f"Eq. (3.1) with T = 108:   IC = {100 * ic108:.2f} %   (book Fig. 3.3: 42.14 %)")
    print(f"Otsu (Sec. 3.1.1.1): t* = {ot['threshold']:g}, IC = {100 * ot['ic']:.2f} %, eta = {ot['em']:.4f}")
    counts = ot["counts"]
    print(f"histogram: peak {int(counts.max())} @ level {int(counts.argmax())}, gray min/max = {int(I.min())}/{int(I.max())}")

    # --- Fig. 3.2: (a) image, (b) histogram with T, (c) binarised ---------------------------------------------
    written.append(save_image(out / f"global_threshold_bw_T{int(T)}.png", bw))
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
    imshow_matlab(axes[0], I, title=f"(a) {SUBSTITUTE} (substitute for Fig. 3.2(a))")
    plot_histogram_with_threshold(axes[1], I, T)
    axes[1].set_title(f"(b) histogram, T = {T:g}")
    imshow_matlab(axes[2], bw, title=f"(c) g = f > T, IC = {100 * ic:.2f} %")
    fig.suptitle("Fig. 3.2  Global thresholding by a hand-picked T (Eq. 3.1)  [substitute image]")
    written.append(finish_figure(fig, out / "fig_3_02_global_threshold.png", args.show))

    print("figures written:")
    for pth in written:
        print("  ", pth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
