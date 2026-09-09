"""Port of ``MATLAB_ROOT/ch4/derivative.m`` — Sobel edge detection with T = 0.05 (Fig. 4.3(b)), plus the text-only
Prewitt (Fig. 4.3(c)) and LoG (Fig. 4.6) variants on the Fig. 4.3(a) crop.

Book §4.1.1 (Eqs. 4.2, 4.4, 4.7–4.8) and §4.1.2 (Eqs. 4.14–4.15).  The MATLAB script reads ``test.jpg``, converts
with ``double(im)/256`` (literally /256, not ``im2double``), calls ``edge(im, 'sobel', 0.05)`` and shows ``BW``.  Its
three commented-out post-processing lines (``medfilt2``, ``bwareaopen``, ``conv2`` smoothing) are available as
``--median``, ``--min-area 20`` and ``--smooth`` (off by default).

Default run = the literal script on the full image (``sec_4_1_1_derivative_<method>_T<T>.png``) **and** the book
figures on the Fig. 4.3(a) crop (``fig_4_03b_sobel_T0.05.png``, ``fig_4_03c_prewitt_T0.05.png``,
``fig_4_06_log_s2_T0.005.png`` + panel figures); ``--no-book-figures`` skips the latter, ``--crop fig4_3a`` runs the
literal pipeline on the crop instead of the full image.

Usage: ``python scripts/ch04_derivative.py [--thresh 0.05] [--method sobel] [--crop none] [--median] [--min-area 20]
[--smooth] [--data data/book/ch04] [--out outputs/ch04] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch04_ice_edge_detection import BOOK_PARAMS, FIG_4_3A_CROP_MATLAB, fig_4_3a, \
    sobel_edges_script  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402

CH = "ch04"


def _fmt(x: float) -> str:
    return f"{x:g}"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="test.jpg", help="book image (default: %(default)s)")
    p.add_argument("--thresh", type=float, default=None,
                   help="edge threshold T (default: 0.05 = script; pass a negative value for MATLAB's automatic T)")
    p.add_argument("--method", default="sobel", choices=["sobel", "prewitt", "log", "roberts"],
                   help="edge method for the literal pipeline (default: %(default)s)")
    p.add_argument("--sigma", type=float, default=BOOK_PARAMS["log_sigma"], help="LoG sigma (default: %(default)s)")
    p.add_argument("--crop", default="none", choices=["none", "fig4_3a"],
                   help="run the literal pipeline on the full image (script) or on the Fig. 4.3(a) crop")
    p.add_argument("--book-figures", action="store_true", default=True, help="also reproduce Figs. 4.3(b)(c)/4.6 on the crop (default)")
    p.add_argument("--no-book-figures", dest="book_figures", action="store_false")
    p.add_argument("--median", action="store_true", help="enable the commented 'medfilt2(im,[3 3])' pre-filter")
    p.add_argument("--min-area", type=int, default=None, help="enable the commented 'bwareaopen(BW, N)' (script: 20)")
    p.add_argument("--smooth", action="store_true", help="enable the commented 'conv2(double(BW), double(msk))' smoothing")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    written: list[Path] = []

    try:
        rgb, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name} ({args.image}): {exc}")
        return 0
    gray = rgb2gray_matlab(rgb)  # line 3: im = rgb2gray(im)
    print(f"=== derivative.m on {args.image}: {gray.shape[0]}x{gray.shape[1]} gray (rgb2gray), im = double(im)/256 ===")

    # --- literal pipeline (lines 5–17) -----------------------------------------------------------------------
    src = gray
    tag = "full"
    if args.crop == "fig4_3a":
        src = fig_4_3a(gray)
        tag = "crop"
        print(f"crop = Fig. 4.3(a): MATLAB im({FIG_4_3A_CROP_MATLAB[0][0]}:{FIG_4_3A_CROP_MATLAB[0][1]}, "
              f"{FIG_4_3A_CROP_MATLAB[1][0]}:{FIG_4_3A_CROP_MATLAB[1][1]}) -> {src.shape}")
    T = 0.05 if args.thresh is None else (None if args.thresh < 0 else args.thresh)
    if args.method == "log" and args.thresh is None:
        T = BOOK_PARAMS["log_T"]
    t0 = time.perf_counter()
    res = sobel_edges_script(src, T, method=args.method, sigma=args.sigma, median=args.median,
                             min_area=args.min_area, smooth=args.smooth)
    dt = time.perf_counter() - t0
    flags = "".join(f" {f}" for f, on in (("medfilt2", args.median), (f"bwareaopen({args.min_area})", args.min_area is not None),
                                          ("conv2-smooth", args.smooth)) if on)
    print(f"edge(im, '{args.method}', {'auto' if T is None else _fmt(T)}){' with' + flags if flags else ''}: "
          f"threshold used = {res['thresh']:.7g}, edge pixels = {res['n_edge']} "
          f"({100 * res['n_edge'] / res['bw'].size:.3f} % of {res['r']}x{res['c']}), {dt:.2f} s")
    if T is None:
        print("  (automatic MATLAB threshold: T = sqrt(4 * mean(bx^2 + by^2)))")
    name = f"sec_4_1_1_derivative_{args.method}_T{_fmt(res['thresh']) if T is not None else 'auto'}_{tag}.png"
    written.append(save_image(out / name, res["bw"]))  # line 17: figure, imshow(BW)
    if args.smooth:
        sm = res["bw_smooth"]
        print(f"  conv2(double(BW), msk) 'full' output: {sm.shape}, max {sm.max():g} (no longer binary)")
        written.append(save_image(out / name.replace(".png", "_conv2_full.png"), sm, autoscale=True))

    # --- book figures on the Fig. 4.3(a) crop ----------------------------------------------------------------
    if args.book_figures:
        crop = fig_4_3a(gray)
        print(f"\n--- Book figures on the Fig. 4.3(a) crop {crop.shape} (MATLAB im({FIG_4_3A_CROP_MATLAB[0][0]}:"
              f"{FIG_4_3A_CROP_MATLAB[0][1]}, {FIG_4_3A_CROP_MATLAB[1][0]}:{FIG_4_3A_CROP_MATLAB[1][1]})) ---")
        Tb = BOOK_PARAMS["sobel_T"]
        sob = sobel_edges_script(crop, Tb, "sobel")
        pre = sobel_edges_script(crop, Tb, "prewitt")
        log = sobel_edges_script(crop, BOOK_PARAMS["log_T"], "log", sigma=BOOK_PARAMS["log_sigma"])
        auto = sobel_edges_script(crop, None, "sobel")
        print(f"Fig. 4.3(b) Sobel   T = {Tb}: {sob['n_edge']} edge px ({100 * sob['n_edge'] / sob['bw'].size:.2f} %)")
        print(f"Fig. 4.3(c) Prewitt T = {Tb}: {pre['n_edge']} edge px ({100 * pre['n_edge'] / pre['bw'].size:.2f} %)")
        print(f"Fig. 4.6    LoG 13x13, sigma = {BOOK_PARAMS['log_sigma']}, T = {BOOK_PARAMS['log_T']}: {log['n_edge']} edge px "
              f"({100 * log['n_edge'] / log['bw'].size:.2f} %)")
        print(f"(automatic Sobel threshold on the crop would be {auto['thresh']:.5f} -> {auto['n_edge']} px)")
        agree = int((sob["bw"] & pre["bw"]).sum())
        print(f"Sobel/Prewitt overlap: {agree} px common ({100 * agree / max(sob['n_edge'], 1):.1f} % of the Sobel edges) — §4.3 (a): the two are alike")
        written.append(save_image(out / "fig_4_03a_crop.png", crop))
        written.append(save_image(out / "fig_4_03b_sobel_T0.05.png", sob["bw"]))
        written.append(save_image(out / "fig_4_03c_prewitt_T0.05.png", pre["bw"]))
        written.append(save_image(out / "fig_4_06_log_s2_T0.005.png", log["bw"]))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
        imshow_matlab(axes[0], crop, title="(a) Fig. 4.3(a): crop of test.jpg")
        imshow_matlab(axes[1], sob["bw"], title=f"(b) Sobel, T = {Tb}: {sob['n_edge']} px")
        imshow_matlab(axes[2], pre["bw"], title=f"(c) Prewitt, T = {Tb}: {pre['n_edge']} px")
        fig.suptitle("Fig. 4.3  Derivative edge detection (edge.m port: /8 and /6 kernels, replicate padding, thinning)")
        written.append(finish_figure(fig, out / "fig_4_03_sobel_prewitt_panels.png", args.show))
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
        imshow_matlab(axes[0], crop, title="Fig. 4.3(a)")
        imshow_matlab(axes[1], log["bw"], title=f"LoG 13×13, σ = 2, T = 0.005: {log['n_edge']} px")
        fig.suptitle("Fig. 4.6  LoG edge detection (zero-crossings; T applies to the jump across the crossing)")
        written.append(finish_figure(fig, out / "fig_4_06_log_panels.png", args.show))

    print("\nfigures written:")
    for pth in written:
        print("  ", pth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
