"""Port of ``MATLAB_ROOT/ch4/morphology.m`` — binary and grayscale erosion / dilation and the three morphological
gradients with a disk structuring element (Figs. 4.9, 4.10, 4.15, 4.16), plus the text-only demonstrations of
§4.2 (Fig. 4.7 structuring elements, Fig. 4.8 erosion/dilation matrices, Figs. 4.11–4.14 1-D profiles).

Book §4.2.1 (Eqs. 4.16–4.21) and §4.2.4 (Eqs. 4.39–4.42).  The MATLAB script uses ``strel('dis', 7)`` (MATLAB's
prefix matching for ``'disk'``: a 13×13, 157-pixel *approximate* disk) on the full ``test.jpg`` and shows 12 images;
the book figures use a 15-pixel disk on the Fig. 4.3(a) crop (Fig. 4.15's caption "157-pixel-radius" is a typo).

Default run = the literal script (``--radius 7``, full image → ``sec_4_2_*_r7.png``) **and** the book configuration
(crop + r = 15 → ``fig_4_09*``, ``fig_4_10*``, ``fig_4_15*``, ``fig_4_16*``) **and** the demos
(``--demo all``: ``sec_4_2_fig4_7_strels.png``, ``sec_4_2_1_fig4_8.png``, ``sec_4_2_2_open_close_1d.png``,
``sec_4_2_3_reconstruction_1d.png``).

Usage: ``python scripts/ch04_morphology.py [--radius 7] [--crop none|fig4_3a] [--no-book-figures]
[--demo all|fig4_8|profiles|strels|none] [--data data/book/ch04] [--out outputs/ch04] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch04_ice_edge_detection import BOOK_PARAMS, FIG_4_3A_CROP_MATLAB, fig_4_3a, fig_4_8_demo, \
    gradient_identity_check, morphological_edges, profile_open_close_demo, profile_reconstruction_demo  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402
from seaice.core.morphology import strel  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image, show_matrix  # noqa: E402

CH = "ch04"

#: MATLAB variable → (file slug, panel title); order = the script's figure order (lines 4–43).
PANELS = [
    ("im", "gray", "rgb2gray(im)"), ("I", "bw_otsu", "I = im2bw(im, graythresh(im))"),
    ("J", "erosion", "J = imerode(I, SE)"), ("K", "dilation", "K = imdilate(I, SE)"),
    ("BW1", "internal", "BW1 = I − J (internal)"), ("BW2", "external", "BW2 = K − I (external)"),
    ("BW", "basic", "BW = K − J (basic)"),
    ("X", "gray_erosion", "X = imerode(im, SE)"), ("Y", "gray_dilation", "Y = imdilate(im, SE)"),
    ("internal", "gray_internal", "internal = im − X"), ("external", "gray_external", "external = Y − im"),
    ("basic", "gray_basic", "basic = Y − X"),
]
#: Book figure file names for the crop + r = 15 configuration (Figs. 4.9, 4.10, 4.15, 4.16 panel order from the captions).
BOOK_NAMES = {
    "I": "fig_4_09a_bw", "J": "fig_4_09b_erosion", "K": "fig_4_09c_dilation",
    "X": "fig_4_10a_gray_erosion", "Y": "fig_4_10b_gray_dilation",
    "BW": "fig_4_15a_basic", "BW1": "fig_4_15b_internal", "BW2": "fig_4_15c_external",
    "basic": "fig_4_16a_gray_basic", "internal": "fig_4_16b_gray_internal", "external": "fig_4_16c_gray_external",
}


def run_pipeline(gray: np.ndarray, radius: int, label: str) -> dict:
    t0 = time.perf_counter()
    res = morphological_edges(gray, radius=radius)
    dt = time.perf_counter() - t0
    SE = res["SE"]
    M, N = gray.shape
    print(f"\n=== morphology.m on {label}: {M}x{N}, SE = strel('dis', {radius}) -> {SE.shape[0]}x{SE.shape[1]}, "
          f"{int(SE.sum())} px (approximate disk, n = 4); {dt:.2f} s ===")
    print(f"graythresh: level = {res['level']:.6f} -> t* = {res['threshold']:g}; ice pixels I: {int(res['I'].sum())} "
          f"(IC = {100 * res['ic']:.2f} %)")
    print(f"binary: erosion J = {int(res['J'].sum())} px, dilation K = {int(res['K'].sum())} px; "
          f"internal BW1 = {int(res['BW1'].sum())}, external BW2 = {int(res['BW2'].sum())}, basic BW = {int(res['BW'].sum())} px")
    print(f"gray  : internal max {int(res['internal'].max())} / mean {res['internal'].mean():.3f}; "
          f"external max {int(res['external'].max())} / mean {res['external'].mean():.3f}; basic mean {res['basic'].mean():.3f}")
    chk = gradient_identity_check(res)
    print(f"Eq. 4.42 (internal + external = basic): binary {chk['binary']}, gray {chk['gray']}")
    return res


def save_panels(res: dict, out: Path, prefix: str, radius: int, book: bool, show: bool, written: list[Path]) -> None:
    for var, slug, _ in PANELS:
        if book and var in BOOK_NAMES:
            name = f"{BOOK_NAMES[var]}.png"
        else:
            name = f"{prefix}_{slug}_r{radius}.png" if var != "im" else f"{prefix}_{slug}.png"
        written.append(save_image(out / name, res[var]))
    groups = [("binary", ["I", "J", "K"], "Fig. 4.9" if book else "§4.2.1 binary"),
              ("gray_erode_dilate", ["im", "X", "Y"], "Fig. 4.10" if book else "§4.2.1 grayscale"),
              ("binary_gradients", ["BW", "BW1", "BW2"], "Fig. 4.15" if book else "§4.2.4 binary gradients"),
              ("gray_gradients", ["basic", "internal", "external"], "Fig. 4.16" if book else "§4.2.4 grayscale gradients")]
    titles = {v: t for v, _, t in PANELS}
    fignum = {"binary": "fig_4_09_panels", "gray_erode_dilate": "fig_4_10_panels", "binary_gradients": "fig_4_15_panels",
              "gray_gradients": "fig_4_16_panels"}
    for key, vars_, head in groups:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
        for ax, v in zip(axes, vars_):
            imshow_matlab(ax, res[v], title=titles[v])
        fig.suptitle(f"{head}  (disk r = {radius}, {int(res['SE'].sum())}-px SE)")
        name = f"{fignum[key]}.png" if book else f"{prefix}_{key}_panels_r{radius}.png"
        written.append(finish_figure(fig, out / name, show))


def demo_strels(out: Path, show: bool, written: list[Path]) -> None:
    ses = [("(a) square 3×3", strel("square", 3)), ("(b) rectangle 3×5", strel("rectangle", [3, 5])),
           ("(c) disk r = 5 (MATLAB n = 4): 9×9, 69 px", strel("disk", 5)), ("(d) diamond 5: 11×11", strel("diamond", 5)),
           ("strel('dis', 7) of morphology.m: 13×13, 157 px", strel("dis", 7)), ("exact Euclidean disk r = 5 (n = 0): 11×11, 81 px", strel("disk", 5, 0))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (title, se) in zip(axes.ravel(), ses):
        show_matrix(ax, se.astype(int), title=title, fontsize=6)
    fig.suptitle("Fig. 4.7  Structuring elements (MATLAB strel; the default disk is an octagonal approximation)")
    written.append(finish_figure(fig, out / "sec_4_2_fig4_7_strels.png", show))
    print("\nFig. 4.7 structuring elements: " + "; ".join(f"{t.split(':')[0]} -> {se.shape[0]}x{se.shape[1]}/{int(se.sum())} px" for t, se in ses))


def demo_fig4_8(out: Path, show: bool, written: list[Path]) -> None:
    d = fig_4_8_demo()
    print(f"\nFig. 4.8 (p. 70): erosion of the 3x7 rectangle by the cross SE -> {int(d['eroded'].sum())} px "
          f"(book (d): {int(d['eroded_book'].sum())}, match = {d['erosion_matches']}); dilation -> {int(d['dilated'].sum())} px "
          f"(book (f): {int(d['dilated_book'].sum())}, match = {d['dilation_matches']})")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    show_matrix(axes[0, 0], d["image"].astype(int), title="(a) binary image with a rectangular object", fontsize=6)
    show_matrix(axes[0, 1], d["se"].astype(int), title="(b) cross-shaped SE, origin at centre", fontsize=9)
    show_matrix(axes[1, 0], d["eroded"].astype(int), title=f"(d) erosion A ⊖ B (matches book: {d['erosion_matches']})", fontsize=6)
    show_matrix(axes[1, 1], d["dilated"].astype(int), title=f"(f) dilation A ⊕ B (matches book: {d['dilation_matches']})", fontsize=6)
    fig.suptitle("Fig. 4.8  Erosion (Eq. 4.16) and dilation (Eq. 4.18) — imerode / imdilate on the printed matrices")
    written.append(finish_figure(fig, out / "sec_4_2_1_fig4_8.png", show))


def demo_profiles(out: Path, show: bool, written: list[Path]) -> None:
    oc = profile_open_close_demo()
    x = np.arange(oc["f"].size)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(x, oc["f"], "k", lw=2, label="f (synthetic profile)")
    axes[0].plot(x, oc["dilation"], "C1", lw=1, alpha=0.7, label="f ⊕ b")
    axes[0].plot(x, oc["closing"], "C3", lw=2, label=f"closing (f ⊕ b) ⊖ b, Eq. 4.22 (b = {int(oc['se_length'])}-px line)")
    axes[0].set_title("Fig. 4.11 (sketch)  Closing fills dark details narrower than b (the crack, the pit)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(x, oc["f"], "k", lw=2, label="f")
    axes[1].plot(x, oc["erosion"], "C1", lw=1, alpha=0.7, label="f ⊖ b")
    axes[1].plot(x, oc["opening"], "C0", lw=2, label="opening (f ⊖ b) ⊕ b, Eq. 4.23")
    axes[1].set_title("Fig. 4.12 (sketch)  Opening removes bright details narrower than b (the speck)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("pixel")
    written.append(finish_figure(fig, out / "sec_4_2_2_open_close_1d.png", show))
    rc = profile_reconstruction_demo()
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(x, rc["f"], "k", lw=2, label="mask g = f")
    axes[0].plot(x, rc["marker_dilation"], "C1", lw=1, label=f"marker f − h (h = {rc['h']})")
    axes[0].plot(x, rc["rec_dilation"], "C3", lw=2, label=f"R_g^D(f − h), Eq. 4.34 (k = {rc['k_dilation']} steps)")
    axes[0].set_title(f"Fig. 4.13 (sketch)  Reconstruction by dilation — peaks lower than h are flattened (iterative == imreconstruct: {rc['iterative_equals_dilation']})")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(x, rc["f"], "k", lw=2, label="mask g = f")
    axes[1].plot(x, rc["marker_erosion"], "C1", lw=1, label="marker f + h")
    axes[1].plot(x, rc["rec_erosion"], "C0", lw=2, label=f"R_g^E(f + h), Eq. 4.38 (k = {rc['k_erosion']} steps)")
    axes[1].set_title(f"Fig. 4.14 (sketch)  Reconstruction by erosion — valleys shallower than h are filled (iterative == skimage: {rc['iterative_equals_erosion']})")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("pixel")
    written.append(finish_figure(fig, out / "sec_4_2_3_reconstruction_1d.png", show))
    print(f"\n1-D demos: reconstruction by dilation converged in k = {rc['k_dilation']} steps, by erosion in k = {rc['k_erosion']} "
          f"(literal Eq. 4.34/4.38 loops equal imreconstruct: {rc['iterative_equals_dilation']}, {rc['iterative_equals_erosion']})")


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="test.jpg", help="book image (default: %(default)s)")
    p.add_argument("--radius", type=int, default=BOOK_PARAMS["disk_r_script"],
                   help="disk radius for the literal pipeline (script: 7; book figures: 15)")
    p.add_argument("--crop", default="none", choices=["none", "fig4_3a"],
                   help="run the literal pipeline on the full image (script) or on the Fig. 4.3(a) crop")
    p.add_argument("--book-figures", action="store_true", default=True,
                   help="also run crop + r = 15 for Figs. 4.9/4.10/4.15/4.16 (default)")
    p.add_argument("--no-book-figures", dest="book_figures", action="store_false")
    p.add_argument("--demo", default="all", choices=["all", "fig4_8", "profiles", "strels", "none"],
                   help="text-only demonstrations to render (default: all)")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    written: list[Path] = []

    if args.demo in ("all", "strels"):
        demo_strels(out, args.show, written)
    if args.demo in ("all", "fig4_8"):
        demo_fig4_8(out, args.show, written)
    if args.demo in ("all", "profiles"):
        demo_profiles(out, args.show, written)

    try:
        rgb, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"\nSKIP {Path(__file__).name} ({args.image}): {exc}")
        _list(written)
        return 0
    gray = rgb2gray_matlab(rgb)  # line 3

    src, label = gray, f"{args.image} (full image, as the script)"
    if args.crop == "fig4_3a":
        src = fig_4_3a(gray)
        label = f"Fig. 4.3(a) crop of {args.image}"
    res = run_pipeline(src, args.radius, label)
    is_book = args.crop == "fig4_3a" and args.radius == BOOK_PARAMS["disk_r_figs"]
    save_panels(res, out, "sec_4_2" + ("_crop" if args.crop == "fig4_3a" else ""), args.radius, is_book, args.show, written)

    if args.book_figures and not is_book:
        crop = fig_4_3a(gray)
        print(f"\n--- Book configuration: Fig. 4.3(a) crop (MATLAB im({FIG_4_3A_CROP_MATLAB[0][0]}:{FIG_4_3A_CROP_MATLAB[0][1]}, "
              f"{FIG_4_3A_CROP_MATLAB[1][0]}:{FIG_4_3A_CROP_MATLAB[1][1]})), disk r = {BOOK_PARAMS['disk_r_figs']} ---")
        res_b = run_pipeline(crop, BOOK_PARAMS["disk_r_figs"], "Fig. 4.3(a) crop")
        save_panels(res_b, out, "sec_4_2_crop", BOOK_PARAMS["disk_r_figs"], True, args.show, written)

    _list(written)
    return 0


def _list(written: list[Path]) -> None:
    print("\nfigures written:")
    for pth in written:
        print("  ", pth)


if __name__ == "__main__":
    sys.exit(main())
