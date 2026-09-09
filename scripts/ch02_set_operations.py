"""Text-only demo of Book §2.6 — set and logical operations, Eqs. (2.16)–(2.30), Table 2.1, Figs. 2.16–2.17.

No MATLAB script exists.  Binary sets A, B (16×16, synthetic) → complement, union, intersection, difference
(Fig. 2.16); an L-shape → reflection and translation (Fig. 2.17); grayscale complement / union / intersection on
``rgb2gray(rgb.JPG)`` (Eqs. 2.28–2.30); truth tables and the bitwise example ``57 AND 207 = 9`` (p. 29).

Usage: ``python scripts/ch02_set_operations.py [--data data/book/ch02] [--out outputs/ch02] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch02_preliminaries import set_operation_examples  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, show_matrix  # noqa: E402
from seaice.core.setops import truth_tables  # noqa: E402

CH = "ch02"


def main(argv: list[str] | None = None) -> int:
    args = chapter_argparser(CH, __doc__.splitlines()[0]).parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        I, _ = load_image(CH, "rgb.jpg", allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:  # private book image absent: scripts never use the public substitute
        print(f"SKIP {Path(__file__).name}: {exc}")
        return 0
    gray = rgb2gray_matlab(I)
    ex = set_operation_examples(gray)
    written: list[Path] = []

    # --- Fig. 2.16 -------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    panels = (("A", "A"), ("B", "B"), ("A_c", "Complement  A^c  (Eq. 2.21)"), ("A_or_B", "Union  A ∪ B  (Eq. 2.22)"),
              ("A_and_B", "Intersection  A ∩ B  (Eq. 2.23)"), ("A_minus_B", "Difference  A − B  (Eq. 2.24)"))
    for ax, (key, title) in zip(axes.flat, panels):
        show_matrix(ax, ex[key].astype(int), fmt="{:d}", cmap="Greys", fontsize=6, title=title)
    fig.suptitle("Fig. 2.16  Basic set operations on binary images")
    written.append(finish_figure(fig, out / "fig_2_16_set_operations.png", args.show))

    # --- Fig. 2.17 -------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, key, origin, title in ((axes[0], "L", ex["L_origin"], "A (origin = black dot)"),
                                   (axes[1], "L_hat", ex["L_hat_origin"], "Reflection  Â  (Eq. 2.25)"),
                                   (axes[2], "L_z", tuple(np.add(ex["L_origin"], ex["z"])),
                                    f"Translation  (A)_z, z = {ex['z']}  (Eq. 2.26)")):
        show_matrix(ax, ex[key].astype(int), fmt="", cmap="Greys", title=title)
        ax.plot(origin[1], origin[0], "ko", ms=7)
    fig.suptitle("Fig. 2.17  Reflection and translation of a set A")
    written.append(finish_figure(fig, out / "fig_2_17_reflection_translation.png", args.show))

    # --- grayscale set operations, Eqs. (2.28)-(2.30) ----------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    imshow_matlab(axes[0], ex["gray"], title="A = rgb2gray(rgb.JPG)")
    imshow_matlab(axes[1], ex["gray_c"], title="A^c = L − A, L = 255  (Eq. 2.28)")
    imshow_matlab(axes[2], ex["gray_union"], title="A ∪ A^c = max  (Eq. 2.29)")
    imshow_matlab(axes[3], ex["gray_intersection"], title="A ∩ A^c = min  (Eq. 2.30)")
    fig.suptitle("Section 2.6.2  Set operations on grayscale images")
    written.append(finish_figure(fig, out / "gray_set_operations.png", args.show))

    print("Table 2.1 truth tables:", truth_tables())
    print(f"57 AND 207 = {ex['bitwise_57_and_207']}  (book: 9;  00111001 & 11001111 = 00001001)")
    print(f"|A| = {ex['A'].sum()}, |B| = {ex['B'].sum()}, |A or B| = {ex['A_or_B'].sum()}, |A and B| = {ex['A_and_B'].sum()}, "
          f"|A - B| = {ex['A_minus_B'].sum()}, |A^c| = {ex['A_c'].sum()}")
    print(f"L origin {ex['L_origin']} -> reflected origin {ex['L_hat_origin']}; translation z = {ex['z']}")
    print("figures written:")
    for p in written:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
