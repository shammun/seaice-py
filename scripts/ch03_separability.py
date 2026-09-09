"""Port of ``MATLAB_ROOT/ch3/separability.m`` — fixed threshold ``k = 108`` and the Otsu separability η (Fig. 3.3).

Book §3.1.1 Eq. (3.1) and §3.1.1.1 Eqs. (3.3)–(3.22): the script binarises with ``I > k``, computes ``IC`` and
η = σ_B²/σ_G² from the normalised histogram.  It reads ``ch3ice.jpg`` (Fig. 3.2(a), the OMAE-2012 image), which is
**not shipped**; per ``analysis/ch03.md`` §5 the labelled substitute is ``2.jpg`` (histogram twin: Otsu t* 107
vs 108, η 0.967 vs 0.964).  The book's η(108) = 0.9643 / η(125) = 0.9620 / IC 42.14 % are therefore not
reproducible; the printed values are for the substitute.

The script's off-by-one (its η is evaluated for classes ``0..k−1`` / ``k..255``, i.e. at ``t = k − 1``, while the
mask uses ``I > k``) is reproduced in ``eta`` and reported next to the book-convention ``η(k)``.

Usage: ``python scripts/ch03_separability.py [--k 108] [--data data/book/ch03] [--out outputs/ch03] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch03_ice_pixel_detection import separability_script  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402
from seaice.core.threshold import graythresh, separability  # noqa: E402

CH = "ch03"
SUBSTITUTE = "2.jpg"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--k", type=int, default=108, help="fixed threshold of the script (default 108)")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        rgb, _ = load_image(CH, SUBSTITUTE, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:  # private book image absent: scripts never use the public substitute
        print(f"SKIP {Path(__file__).name}: {exc}")
        return 0
    print(f"NOTE: 'ch3ice.jpg' (Fig. 3.2(a)) is not shipped; using {SUBSTITUTE} as the labelled SUBSTITUTE.")
    I = rgb2gray_matlab(rgb)
    k = args.k
    res = separability_script(I, k)
    level, em = graythresh(I)

    # --- key numbers ------------------------------------------------------------------------------------------
    print(f"k = {k}: IC = {100 * res['IC']:.2f} %   (book Fig. 3.3, unshipped image: 42.14 %)")
    print(f"script variables: mg = {res['mg']:.4f} (1-based, = mG + 1), sigma2_g = {res['sigma2_g']:.4f}, "
          f"sigma2_b = {res['sigma2_b']:.4f}, eta = {res['eta']:.4f}")
    print(f"  -> the script's eta is eta(t = {res['t_eval']}) in the book convention; eta(t = {k}) = {res['eta_book']:.4f}"
          f"   (book p. 43: eta(108) = 0.9643, eta(125) = 0.9620 for the unshipped image)")
    print(f"  eta(125) = {separability(I, 125):.4f}")
    print(f"graythresh: t* = {255 * level:g}, em = eta(t*) = {em:.4f}; otsu_criterion: t* = {res['t_star']:g}, "
          f"eta* = {res['eta_star']:.4f}")
    assert abs(em - res["eta_star"]) < 1e-9 and abs(255 * level - res["t_star"]) < 1e-9

    # --- Fig. 3.3: binarised image at k; extra: eta(t) curve --------------------------------------------------
    written = [save_image(out / f"separability_bw_k{k}.png", res["bw"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    imshow_matlab(axes[0], I, title=f"(a) {SUBSTITUTE} (substitute for Fig. 3.2(a))")
    imshow_matlab(axes[1], res["bw"], autoscale=True, title=f"(b) I > {k}, IC = {100 * res['IC']:.2f} %")
    fig.suptitle(f"Fig. 3.3  Global threshold t = {k}, separability eta = {res['eta']:.4f}  [substitute image]")
    written.append(finish_figure(fig, out / "fig_3_03_separability.png", args.show))

    t = np.arange(256)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t, res["eta_curve"], "k-", linewidth=1.5, label=r"$\eta(t)=\sigma_B^2(t)/\sigma_G^2$  (Eq. 3.20)")
    ax.axvline(res["t_star"], color="r", linestyle="--", label=f"Otsu t* = {res['t_star']:g}, eta = {res['eta_star']:.4f}")
    ax.axvline(k, color="b", linestyle=":", label=f"k = {k}, eta = {res['eta_book']:.4f}")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 1)
    ax.set_xlabel("threshold t")
    ax.set_ylabel(r"$\eta$")
    ax.legend(loc="lower center", frameon=False)
    ax.set_title("§3.1.1.1  Separability measure vs threshold (Eqs. 3.20–3.22)")
    written.append(finish_figure(fig, out / "sec_3_1_1_eta_curve.png", args.show))

    print("figures written:")
    for pth in written:
        print("  ", pth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
