"""§4.3 "Experimental results and discussion" (Figs. 4.17–4.20) — Sobel T = 0.05 vs 0.03 and the binary internal
gradient with 5- vs 16-pixel disks, on ``test.jpg`` and on a crop of touching floes.

The book's §4.3 image (Fig. 4.17(a), a pack-ice edge with dense floes on the left) and its connected-floe crop
Fig. 4.18(a) are **not shipped** with the MATLAB code (analysis/ch04.md §5); this driver reproduces the *experiments*
qualitatively on ``test.jpg`` (Fig. 4.17-style full frame) and on a user-selectable crop of touching floes
(``--crop r0:r1,c0:c1``, 0-based half-open; default rows 300:1100, cols 50:950 of ``test.jpg`` — the dense left part).
Numbers printed here are therefore not book values; the algorithms themselves are the verified ports.

Outputs: ``sec_4_3_fig4_17_full_panels.png`` (image / Sobel 0.05 / internal r = 5), ``sec_4_3_fig4_18_crop_panels.png``,
``sec_4_3_fig4_19_sobel_T0.05_vs_T0.03.png``, ``sec_4_3_fig4_20_internal_r5_vs_r16.png`` and the single maps.

Usage: ``python scripts/ch04_experiments.py [--crop 300:1100,50:950] [--thresholds 0.05 0.03] [--radii 5 16]
[--data data/book/ch04] [--out outputs/ch04] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch04_ice_edge_detection import BOOK_PARAMS, experiment_internal_gradient, \
    experiment_sobel_thresholds  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.connectivity import count_components  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402

CH = "ch04"
DEFAULT_CROP = "300:1100,50:950"


def parse_crop(spec: str) -> tuple[slice, slice]:
    rows, cols = spec.split(",")
    r0, r1 = (int(v) for v in rows.split(":"))
    c0, c1 = (int(v) for v in cols.split(":"))
    return slice(r0, r1), slice(c0, c1)


def _fmt(x: float) -> str:
    return f"{x:g}"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="test.jpg", help="book image (default: %(default)s)")
    p.add_argument("--crop", default=DEFAULT_CROP, help="touching-floe crop r0:r1,c0:c1 (0-based; default: %(default)s)")
    p.add_argument("--thresholds", type=float, nargs="+", default=[BOOK_PARAMS["sobel_T"], BOOK_PARAMS["sobel_T_low"]],
                   help="Sobel thresholds (Fig. 4.19: 0.05 vs 0.03)")
    p.add_argument("--radii", type=int, nargs="+", default=[BOOK_PARAMS["disk_r_thin"], BOOK_PARAMS["disk_r_thick"]],
                   help="disk radii for the internal gradient (Fig. 4.20: 5 vs 16)")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    written: list[Path] = []

    print("NOTE: the sea-ice image of Book §4.3 (Fig. 4.17(a)) and its crop (Fig. 4.18(a)) are NOT shipped with the MATLAB "
          "code; the experiments below run on test.jpg and on a crop of touching floes chosen in the port phase.\n"
          "      They reproduce the book's *procedure* (Sobel T = 0.05 / 0.03; internal gradient r = 5 / 16), not its pixels.")
    try:
        rgb, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name} ({args.image}): {exc}")
        return 0
    gray = rgb2gray_matlab(rgb)
    rs, cs = parse_crop(args.crop)
    crop = gray[rs, cs]
    T0, T1 = args.thresholds[0], args.thresholds[-1]
    r0, r1 = args.radii[0], args.radii[-1]

    # --- Fig. 4.17-style: full frame, Sobel T0 and internal gradient r0 ------------------------------------------
    print(f"\n=== Fig. 4.17-style on the full {args.image} ({gray.shape[0]}x{gray.shape[1]}) ===")
    sob = experiment_sobel_thresholds(gray, tuple(args.thresholds))
    grad = experiment_internal_gradient(gray, tuple(args.radii))
    for T, r in sob.items():
        print(f"Sobel T = {_fmt(T)}: {r['n_edge']} edge px ({100 * r['n_edge'] / r['bw'].size:.3f} %), "
              f"{count_components(r['bw'], 8)} 8-connected edge components")
    print(f"Otsu level = {grad[r0]['level']:.6f} (t* = {255 * grad[r0]['level']:g})")
    for r, g in grad.items():
        print(f"internal gradient, disk r = {r} ({g['SE'].shape[0]}x{g['SE'].shape[1]}, {int(g['SE'].sum())} px): "
              f"{g['n_edge']} edge px, floes (ice components after removing the edge band) = "
              f"{count_components(g['I'] & (g['BW1'] == 0), 8)}")
    written.append(save_image(out / f"sec_4_3_fig4_17b_sobel_T{_fmt(T0)}_full.png", sob[T0]["bw"]))
    written.append(save_image(out / f"sec_4_3_fig4_17c_internal_r{r0}_full.png", grad[r0]["BW1"]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    imshow_matlab(axes[0], gray, title=f"(a) {args.image} (book: an unshipped pack-ice image)")
    imshow_matlab(axes[1], sob[T0]["bw"], title=f"(b) Sobel, T = {_fmt(T0)}: {sob[T0]['n_edge']} px")
    imshow_matlab(axes[2], grad[r0]["BW1"], title=f"(c) internal gradient, disk r = {r0}: {grad[r0]['n_edge']} px")
    fig.suptitle("Fig. 4.17-style comparison on test.jpg (procedure of §4.3; the book's image is not shipped)")
    written.append(finish_figure(fig, out / "sec_4_3_fig4_17_full_panels.png", args.show))

    # --- Figs. 4.18–4.20-style: touching-floe crop -----------------------------------------------------------------
    print(f"\n=== Figs. 4.18–4.20-style on the crop rows {rs.start}:{rs.stop}, cols {cs.start}:{cs.stop} ({crop.shape[0]}x{crop.shape[1]}) ===")
    sobc = experiment_sobel_thresholds(crop, tuple(args.thresholds))
    gradc = experiment_internal_gradient(crop, tuple(args.radii))
    for T, r in sobc.items():
        print(f"Sobel T = {_fmt(T)}: {r['n_edge']} edge px ({100 * r['n_edge'] / r['bw'].size:.3f} %), "
              f"{count_components(r['bw'], 8)} edge components")
    n_ice = count_components(gradc[r0]["I"], 8)
    for r, g in gradc.items():
        interior = g["I"] & (g["BW1"] == 0)
        print(f"internal gradient r = {r}: {g['n_edge']} edge px; ice components: mask {n_ice} -> interior after the edge band "
              f"{count_components(interior, 8)} (larger r separates weakly connected floes but shrinks them)")
        written.append(save_image(out / f"sec_4_3_fig4_20_internal_r{r}_crop.png", g["BW1"]))
    for T, r in sobc.items():
        written.append(save_image(out / f"sec_4_3_fig4_19_sobel_T{_fmt(T)}_crop.png", r["bw"]))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    imshow_matlab(axes[0], crop, title="(a) touching-floe crop of test.jpg")
    imshow_matlab(axes[1], sobc[T0]["bw"], title=f"(b) Sobel, T = {_fmt(T0)}: weak edges between floes are open")
    imshow_matlab(axes[2], gradc[r0]["BW1"], title=f"(c) internal gradient r = {r0}: closed edges, weak ones lost")
    fig.suptitle("Fig. 4.18-style: connected floes — derivative vs morphological edges")
    written.append(finish_figure(fig, out / "sec_4_3_fig4_18_crop_panels.png", args.show))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    imshow_matlab(axes[0], sobc[T0]["bw"], title=f"(a) Sobel T = {_fmt(T0)}: {sobc[T0]['n_edge']} px")
    imshow_matlab(axes[1], sobc[T1]["bw"], title=f"(b) Sobel T = {_fmt(T1)}: {sobc[T1]['n_edge']} px (more weak edges, more noise)")
    fig.suptitle("Fig. 4.19-style: Sobel threshold sensitivity")
    written.append(finish_figure(fig, out / f"sec_4_3_fig4_19_sobel_T{_fmt(T0)}_vs_T{_fmt(T1)}.png", args.show))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    imshow_matlab(axes[0], gradc[r0]["BW1"], title=f"(a) internal gradient, r = {r0}: thin edges, floes stay connected")
    imshow_matlab(axes[1], gradc[r1]["BW1"], title=f"(b) internal gradient, r = {r1}: thick edges, weak connections break")
    fig.suptitle("Fig. 4.20-style: structuring-element size (Eq. 4.40 on the Otsu mask)")
    written.append(finish_figure(fig, out / f"sec_4_3_fig4_20_internal_r{r0}_vs_r{r1}.png", args.show))

    print("\nConclusion of §4.3 (book): neither method separates tightly connected floes -> watershed (Ch5) / GVF snake (Ch6).")
    print("\nfigures written:")
    for pth in written:
        print("  ", pth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
