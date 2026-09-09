"""Port of ``MATLAB_ROOT/ch2/color_image.m`` — CMY (Fig. 2.4) and HSI (Fig. 2.5) components of ``rgb.JPG``.

Book §2.1.2.2–2.1.2.3, Eqs. (2.3)–(2.6).  The MATLAB script shows 8 figures (I, I_cmy, Ic, Im, Iy, Ih, Is, Ii);
each is saved here as a PNG with ``imshow`` semantics (``[]`` autoscale where the script uses it).  Two HSI hue
variants are saved: the book's Eq. (2.6a) and the script's buggy ``2*Ig`` line (which is what the printed
Fig. 2.5(a) shows).  The CMYK equations (2.4)–(2.5), text-only, are demonstrated as well.

Usage: ``python scripts/ch02_color_image.py [--data data/book/ch02] [--out outputs/ch02] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch02_preliminaries import color_components  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_book_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402

CH = "ch02"


def main(argv: list[str] | None = None) -> int:
    args = chapter_argparser(CH, __doc__.splitlines()[0]).parse_args(argv)
    data, out = resolve_dirs(args)
    I = load_book_image(CH, "rgb.jpg", data_dir=data)  # I = imread('rgb.jpg')
    res = color_components(I)
    written: list[Path] = []

    # --- the eight imshow figures of the script -------------------------------------------------------------
    written.append(save_image(out / "color_image_I.png", I))
    written.append(save_image(out / "color_image_I_cmy.png", res["I_cmy"]))
    for key in ("Ic", "Im", "Iy", "H_matlab", "S_matlab", "I_matlab"):
        written.append(save_image(out / f"color_image_{key}.png", res[key], autoscale=True))  # imshow(x, [])

    # --- Fig. 2.4: C, M, Y component images -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, key, title in zip(axes, ("Ic", "Im", "Iy"), ("(a) Cyan", "(b) Magenta", "(c) Yellow")):
        imshow_matlab(ax, res[key], autoscale=True, title=f"{title} component image")
    fig.suptitle("Fig. 2.4  CMY components, Eq. (2.3):  [C; M; Y] = [1; 1; 1] - [R; G; B]")
    written.append(finish_figure(fig, out / "fig_2_04_cmy.png", args.show))

    # --- Fig. 2.5: H, S, I  (book equation) and the MATLAB-script variant -----------------------------------
    for suffix, keys, note in (
        ("", ("H", "S", "I"), "Eq. (2.6a-c) as printed (V1 uses 2B/sqrt(6))"),
        ("_matlab_bug", ("H_matlab", "S_matlab", "I_matlab"),
         "color_image.m line 21 (2*Ig instead of 2*Ib): H = atan(-1) = -pi/4 everywhere"),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, key, title in zip(axes, keys, ("(a) Hue", "(b) Saturation", "(c) Intensity")):
            imshow_matlab(ax, res[key], autoscale=True, title=f"{title} component image")
        fig.suptitle(f"Fig. 2.5  HSI components — {note}")
        written.append(finish_figure(fig, out / f"fig_2_05_hsi{suffix}.png", args.show))

    # --- CMYK, Eqs. (2.4)-(2.5) (text only) ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for k, (ax, title) in enumerate(zip(axes, ("C", "M", "Y", "K"))):
        imshow_matlab(ax, res["CMYK"][..., k], autoscale=False, title=f"{title} (u = 1, b = 1)")
    fig.suptitle("CMYK components, Eqs. (2.4)-(2.5): Kb = min(1-R, 1-G, 1-B)")
    written.append(finish_figure(fig, out / "cmyk_components.png", args.show))

    # --- key numbers ----------------------------------------------------------------------------------------
    r, c = 1075, 674  # book pixel (1076, 675), Fig. 2.3
    print(f"image {I.shape} dtype={I.dtype}; pixel (1076,675) RGB = {I[r, c].tolist()}  CMY = {res['I_cmy'][r, c].tolist()}")
    print(f"HSI (book, RGB in [0,1]) at (1076,675): H={res['H'][r, c]:.6f} rad  S={res['S'][r, c]:.6f}  I={res['I'][r, c]:.6f}")
    Hb = res["H_matlab"]
    finite = np.isfinite(Hb)
    print(f"HSI (color_image.m): H unique finite values = {np.unique(np.round(Hb[finite], 6)).tolist()}, "
          f"NaN fraction = {(~finite).mean():.6f};  I_matlab max = {res['I_matlab'].max():.3f} (0-255 units)")
    print(f"CMYK at (1076,675): {np.round(res['CMYK'][r, c], 4).tolist()}")
    print("figures written:")
    for p in written:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
