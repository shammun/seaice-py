"""Port of ``MATLAB_ROOT/ch8/MCD/main_WL_new.m`` — the section 8.3 driver (Figures 8.19 and 8.20).

The M-file loads the shipped Appendix-B structure of the OATRC'15 helicopter frame (2888 ice floes, 3452 brash
pieces), sizes every floe twice with **Eq. (8.1)** ``L = sqrt(4A/pi)`` — once from its **polygon** area and once
from its **pixel count** — and calls ``plot_color_bar_and_floe`` for each::

    load('IceImage_290915_2_jpg.0000179.mat')
    length_over_Pixel = 1.1794;   color_limit_N = 30;
    Poly_MCD(i) = sqrt(IceImage.Floe(i).Polygon.Area*length_over_Pixel^2*4/pi);
    [Poly_counts,Poly_centers] = plot_color_bar_and_floe(30, Poly_MCD, Y_limi, N, [IceImage.Floe.Polygon], lop)
    Raw_MCD(i)  = sqrt(IceImage.Floe(i).Area        *length_over_Pixel^2*4/pi);
    [Raw_counts ,Raw_centers ] = plot_color_bar_and_floe(30, Raw_MCD , Y_limi, N, [IceImage.Floe]        , lop)
    count_error = Poly_counts-[Raw_counts(2:end) 0];

The **second** call is Figure 8.19 (the ``Pixels`` branch) and its histogram is Figure 8.20; the polygon versions
and the ``count_error`` bar chart are produced by the code but are **not printed in the book**, so they are saved
as ``sec_8_3_*``.  ``cd('E:\\NTNU\\CRC\\latex\\matlab\\ch8\\MCD')`` (line 5) is dropped — the data is resolved
through ``seaice.core.io``.

Usage: ``python scripts/ch08_main_WL_new.py [--mat IceImage_290915_2_jpg.0000179.mat]
[--length-over-pixel 1.1794] [--color-limit-n 30] [--data data/book/ch08] [--out outputs/ch08] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch08_applications import COLOR_LIMIT_N, LENGTH_OVER_PIXEL, mcd_analysis  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.icestruct import load_iceimage_mat  # noqa: E402
from seaice.core.plotting import finish_figure, mcd_colorbar, save_image  # noqa: E402

CH = "ch08"


def _mcd_bar(ax, centers: np.ndarray, counts: np.ndarray, color_m: np.ndarray, color_limit_n: int,
             ylabel: str) -> None:
    """``plot_color_bar_and_floe.m`` lines 21-43 — one bar per centre, coloured ``color_M(min(i, N), :)``."""
    idx = np.minimum(np.arange(1, centers.size + 1), color_limit_n) - 1
    ax.bar(centers, counts, width=1.0, color=color_m[idx], edgecolor="none")
    ax.set_xlabel("MCD $L_{i}$ [m]", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--mat", default="IceImage_290915_2_jpg.0000179.mat", help="main_WL_new.m line 7")
    p.add_argument("--length-over-pixel", type=float, default=LENGTH_OVER_PIXEL, help="line 13 (m/px)")
    p.add_argument("--color-limit-n", type=int, default=COLOR_LIMIT_N, help="line 21")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    mat = data / "MCD" / args.mat
    if not mat.exists():
        alt = list(data.rglob(args.mat))
        if not alt:
            print(f"SKIP {Path(__file__).name}: private data absent ({mat.as_posix()} not found; the two "
                  "ch8 .mat files are shipped with the book's MATLAB archive and are not in the public repo)")
            return 0
        mat = alt[0]

    ice = load_iceimage_mat(mat)
    print(f"=== main_WL_new.m on {mat.name} ===")
    print(f"  {ice.summary()}")
    print(f"  length_over_Pixel = {args.length_over_pixel} m/px (line 13; the book never prints it - p. 190 "
          "says the scale comes from IB Oden's known length)")
    print(f"  color_limit_N = {args.color_limit_n}, Y_limi = IceImage.Param.NumPix_y = {ice.Param.NumPix_y}")

    r = mcd_analysis(ice, length_over_pixel=args.length_over_pixel, color_limit_n=args.color_limit_n)

    for tag, mcd, res in (("Raw (pixel areas, Fig. 8.19/8.20)", r.raw_mcd, r.raw),
                          ("Poly (polygon areas, unprinted)", r.poly_mcd, r.poly)):
        k = int(res.counts.argmax())
        print(f"  {tag}: MCD {mcd.min():.4f} .. {mcd.max():.4f} m, mean {mcd.mean():.4f}; "
              f"hist(MCD, 1:100) sums to {int(res.counts.sum())}, peak {int(res.counts.max())} at "
              f"{res.centers[k]:g} m; branch = {res.kind}")
    print(f"  colour indices (nearest centre): raw {r.raw.color_index.min()}..{r.raw.color_index.max()}, "
          f"{int((r.raw.color_index > args.color_limit_n).sum())} floes clamped to the {args.color_limit_n}th "
          "colour (the book's '>=30 [m]')")
    print(f"  rgbImage grown by MATLAB to {r.raw.rgb_image.shape[:2]} "
          f"(= (max y, max x) over all painted pixels; the image is "
          f"{ice.Param.NumPix_y}x{ice.Param.NumPix_x})")
    nz = np.flatnonzero(r.count_error)
    print(f"  count_error = Poly_counts - [Raw_counts(2:end) 0]: {nz.size} non-zero bins, range "
          f"{r.count_error.min():g} .. {r.count_error.max():g}, sum {r.count_error.sum():g} "
          "(a ONE-BIN-SHIFTED difference; Fig. 8.15 is the unshifted one, see ch08_color_hist_comparison.py)")

    written = []

    # ---- Figure 8.19: the floe map (the `Pixels` branch) ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(r.raw.rgb_image)
    ax.plot(r.raw.centres_xy[:, 0] - 1, r.raw.centres_xy[:, 1] - 1, "w.", markersize=1.5)
    ax.set_axis_off()
    ax.set_title("Figure 8.19 — ice floes coloured by MCD (white dots = floe centres)")
    mcd_colorbar(fig, ax, r.raw.color_m, (r.raw.centers[0], r.raw.centers[args.color_limit_n - 1]))
    written.append(finish_figure(fig, out / "fig_8_19_floe_mcd_map.png", args.show))
    written.append(save_image(out / "fig_8_19_floe_mcd_map_raw.png",
                              np.clip(r.raw.rgb_image * 255.0, 0, 255).astype(np.uint8)))

    # ---- Figure 8.20: the MCD histogram ------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    _mcd_bar(ax, r.raw.centers, r.raw.counts, r.raw.color_m, args.color_limit_n,
             "Number of ice floes $N$")
    ax.set_title("Figure 8.20 — MCD histogram of the floes of Figure 8.19")
    mcd_colorbar(fig, ax, r.raw.color_m, (r.raw.centers[0], r.raw.centers[args.color_limit_n - 1]))
    written.append(finish_figure(fig, out / "fig_8_20_mcd_histogram.png", args.show))

    # ---- the two unprinted polygon figures + the count_error bar chart -----------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))
    for i, poly in enumerate(r.poly.polygons):
        ax.fill(poly[:, 0], poly[:, 1], color=r.poly.color_m[r.poly.color_clamped[i] - 1], linewidth=0)
    # `plot_color_bar_and_floe.m` line 106: Y_limi*length_over_Pixel - centre(i,2)*length_over_Pixel
    ax.plot(r.poly.centres_xy[:, 0] * args.length_over_pixel,
            ice.Param.NumPix_y * args.length_over_pixel - r.poly.centres_xy[:, 1] * args.length_over_pixel,
            "w.", markersize=1.5)
    ax.set_aspect("equal")
    ax.margins(0)
    ax.set_facecolor("k")
    ax.set_xlabel("$X$ [m]")
    ax.set_ylabel("$Y$ [m]")
    ax.set_title("main_WL_new.m line 25 — polygonized floes coloured by MCD (not printed in the book)")
    mcd_colorbar(fig, ax, r.poly.color_m, (r.poly.centers[0], r.poly.centers[args.color_limit_n - 1]))
    written.append(finish_figure(fig, out / "sec_8_3_polygon_mcd_map.png", args.show))

    fig, ax = plt.subplots(figsize=(10, 5))
    _mcd_bar(ax, r.poly.centers, r.poly.counts, r.poly.color_m, args.color_limit_n,
             "Number of ice floes $N$")
    ax.set_title("main_WL_new.m line 25 — MCD histogram of the polygonized floes (not printed)")
    mcd_colorbar(fig, ax, r.poly.color_m, (r.poly.centers[0], r.poly.centers[args.color_limit_n - 1]))
    written.append(finish_figure(fig, out / "sec_8_3_polygon_mcd_histogram.png", args.show))

    fig, ax = plt.subplots(figsize=(10, 5))
    _mcd_bar(ax, r.poly.centers, r.count_error, r.raw.color_m, args.color_limit_n,
             "Change in ice floes Numbers $\\Delta N$")
    ax.set_title("main_WL_new.m lines 41-66 — $\\Delta N_k = P_k - R_{k+1}$ (one-bin shift; not printed)")
    mcd_colorbar(fig, ax, r.raw.color_m, (r.raw.centers[0], r.raw.centers[args.color_limit_n - 1]))
    written.append(finish_figure(fig, out / "sec_8_3_delta_n_mcd.png", args.show))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
