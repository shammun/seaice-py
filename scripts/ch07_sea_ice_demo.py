"""Port of ``MATLAB_ROOT/ch7/Sea_Ice_Floe_Identification/sea_ice_demo.m`` — the chapter's driver.

The M-file is a parameter block (lines 9-46) followed by three calls::

    [seg, bk] = seaice_kmean_GVF_forenhancement(I, kms0, ...);              % Algorithm 3  (ch6)
    [out, index_floe, ice_floe, index_brash, brash_ice, index_slush, ...
     index_water, index_residue, coverage] = ice_shape_enhancement(bk, seg, ...
                                             min_floe, min_brash, se_th);   % Algorithms 4+5  (this chapter)
    [floe, brash] = sea_ice_model(ice_floe, brash_ice, index_floe);         % section 8.2  (ch8, deferred)

Compared with ch6's copy of the same file, the ch7 parameter block **adds** ``se_th = 50``, ``min_floe = 40`` and
``min_brash = 1`` (lines 27-29).  The image is ch7's **own** ``sea_ice_test.jpg``: same photograph as ch6's, but
a different JPEG encoding (84 % of the samples differ, mean |delta| 2.93), so every count measured on it has to be
re-measured here.  It is also the book's **Fig. 7.22** (printed transposed).

The §7.2 source image (Figs. 7.10-7.16, 205x263) is *not* shipped with the book, so the book's "154 ice floes and
189 brash ice pieces" and its four coverage percentages **cannot** be reproduced; the panels this script writes
are the same procedure applied to ``sea_ice_test.jpg`` and are named ``sec_7_2_*``, not ``fig_7_1x_*``.

Usage: ``python scripts/ch07_sea_ice_demo.py [--image sea_ice_test.jpg] [--num 500] [--iter 100]
[--min-floe 40] [--min-brash 1] [--se-th 50] [--book-threshold] [--no-cache] [--max-seeds N] [--downscale K]
[--data data/book/ch07] [--out outputs/ch07] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch06_gvf_snake import BOOK_PARAMS  # noqa: E402
from seaice.ch07_ice_type import colorbar_area_ticks, ice_shape_enhancement, sea_ice_edge_detection  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, label2rgb, save_image  # noqa: E402

CH = "ch07"
P = BOOK_PARAMS["sea_ice_demo"]


def algorithm3(rgb, args, out_dir: Path):
    """Run (or reload) Algorithm 3 — ``seaice_kmean_GVF_forenhancement.m``, already ported in ch6.

    The result is cached in ``outputs/ch07/`` because the two ch7 drivers both need it and it costs ~45 s.
    Only ``seg`` and ``bk`` are cached; they are the only two arguments ``ice_shape_enhancement`` takes.
    """
    cache = out_dir / f"stage3_{Path(args.image).stem}_{rgb.shape[0]}x{rgb.shape[1]}_{args.num}_{args.iter}.npz"
    if cache.exists() and not args.no_cache:
        z = np.load(cache)
        print(f"Algorithm 3: loaded the cached seg/bk from {cache.name} (delete it or pass --no-cache to recompute)")
        return z["seg"], z["bk"], None
    t0 = time.time()
    res = sea_ice_edge_detection(rgb, kms0=args.kms0, sigma=P["sigma"], GradientOn=P["GradientOn"],
                                 GVFOn=P["GVFOn"], Num=args.num, mu=P["mu"], iter=args.iter, alpha=P["alpha"],
                                 beta=P["beta"], gamma=P["gamma"], kappa=P["kappa"], Dmin=P["Dmin"],
                                 Dmax=P["Dmax"], Ra_min=P["Ra_min"], Ra=P["Ra"], Rc=P["Rc"], Rl=P["Rl"],
                                 se_radius=P["se_radius"], timer=P["timer"], max_seeds=args.max_seeds,
                                 keep_history=False)
    print(f"Algorithm 3 (ch6 seaice_kmean_gvf): {time.time() - t0:.1f} s")
    np.savez_compressed(cache, seg=res.out, bk=res.bk)
    return res.out, res.bk, res


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="sea_ice_test.jpg", help="sea_ice_demo.m line 9")
    p.add_argument("--num", type=int, default=P["Num"], help="GVF iterations (script: 500)")
    p.add_argument("--iter", type=int, default=P["iter"], help="snake iterations (script: 100)")
    p.add_argument("--kms0", type=int, default=P["kms0"], help="k-means clusters (script: 3)")
    p.add_argument("--se-th", type=float, default=P["se_th"], help="Eq. (7.5) size_th (script: 50)")
    p.add_argument("--min-floe", type=float, default=P["min_floe"], help="Algorithm 5 T_floe (script: 40)")
    p.add_argument("--min-brash", type=float, default=P["min_brash"], help="smallest brash area (script: 1)")
    p.add_argument("--nbins", type=int, default=50, help="FSD bins (ice_shape_enhancement.m line 211)")
    p.add_argument("--book-threshold", action="store_true",
                   help="Algorithm 5's printed '>=' instead of the code's '>' (analysis/ch07.md risk R3)")
    p.add_argument("--no-crop", action="store_true",
                   help="run the per-piece morphology on full-image scratch arrays, like the M-file (slow)")
    p.add_argument("--max-seeds", type=int, default=None, help="cap the contours per pass (runtime guard)")
    p.add_argument("--downscale", type=int, default=1, help="process every K-th row/column (runtime guard)")
    p.add_argument("--no-cache", action="store_true", help="recompute Algorithm 3 instead of reusing the cache")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    try:
        rgb, label = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: private image absent ({exc})")
        return 0
    if args.downscale > 1:
        rgb = rgb[::args.downscale, ::args.downscale]
        print(f"NOTE: --downscale {args.downscale} -> {rgb.shape[0]}x{rgb.shape[1]}; the book's numbers only "
              "hold at the original resolution")

    print(f"=== sea_ice_demo.m (ch7) on {args.image} [{label}] {rgb.shape[0]}x{rgb.shape[1]} ===")
    print("parameter block (lines 9-46): " + ", ".join(f"{k} = {v}" for k, v in P.items()))
    print(f"in use: Num = {args.num}, iter = {args.iter}, kms0 = {args.kms0}, se_th = {args.se_th}, "
          f"min_floe = {args.min_floe}, min_brash = {args.min_brash}")

    seg, bk, stage = algorithm3(rgb, args, out)
    print(f"SEGMENTATION_seaice: levels {np.unique(seg).tolist()} — light ice {int((seg == 1).sum())} px, "
          f"dark ice {int((seg == 0.5).sum())} px, water {int((seg == 0).sum())} px; "
          f"ICE (k-means) {int((bk != 0).sum())} px")

    t0 = time.time()
    r = ice_shape_enhancement(bk, seg, min_floe=args.min_floe, min_brash=args.min_brash, se_th=args.se_th,
                              nbins=args.nbins, crop=not args.no_crop, book_threshold=args.book_threshold)
    print(f"Algorithms 4+5 (ice_shape_enhancement.m): {time.time() - t0:.1f} s")
    print(f"  labelled pieces: {r.nn_bw} light + {r.nn_k} dark = {r.ice_area.size}; "
          f"areas {r.ice_area.min() if r.ice_area.size else 0}..{r.ice_area.max() if r.ice_area.size else 0} px, "
          f"sorted small -> large (stable)")
    present = np.bincount(r.out.astype(np.int64).ravel(), minlength=r.t + 1)[1:]
    print(f"  superimposed pieces t = max(out) = {r.t}; {int((present == 0).sum())} label(s) completely "
          "overwritten by a larger piece (the `if area0 ~= 0` guard of line 125)")
    print(f"  ice floes  : {len(r.ice_floe)}  (areas > {args.min_floe} px)")
    print(f"  brash ice  : {len(r.brash_ice)}  (areas > {args.min_brash} px)")
    cov = r.coverage.as_percent()
    print(f"  coverage   : {cov['IceFloe']:.2f} % ice floe, {cov['BrashIce']:.2f} % brash ice, "
          f"{cov['Slush']:.2f} % slush, {cov['Water']:.2f} % water  (sum {sum(cov.values()):.2f} %)")
    print(f"  residue    : {int((r.index_residue != 0).sum())} px (Fig. 7.16 — slush no identified piece covers)")
    area_ice = np.concatenate([r.color_floe, r.color_brash]) if (r.color_floe.size or r.color_brash.size) \
        else np.zeros(0)
    values, ticks = colorbar_area_ticks(area_ice, 6)
    print(f"  Eq. (7.6) colour bar (n = 6): ticks {ticks.tolist()}  [book Fig. 7.13: 3, 131, 277, 448, 656, "
          "917, 1273 — a different image, not comparable]")
    if r.fsd is not None:
        print(f"  FSD ({r.fsd.nbins} bins): {int(r.fsd.counts.sum())} floes, centres "
              f"{r.fsd.centers[0]:.1f}..{r.fsd.centers[-1]:.1f} px, modal bin {int(r.fsd.counts.argmax())} "
              f"({int(r.fsd.counts.max())} floes)")
    print("NOTE: the book's 154/189 pieces and 60.52/3.34/16.03/20.11 % belong to Fig. 7.10(a), a 205x263 image "
          "that is NOT shipped with the code — those numbers cannot be reproduced here.")
    print("NOTE: sea_ice_demo.m line 57 `sea_ice_model` is book section 8.2 and is ported in ch08 - run "
          "scripts/ch08_sea_ice_model.py --source demo for it (it reuses this script's Algorithm-3 cache).")

    written = []
    # Fig. 7.22 — the book prints this image transposed.
    written.append(save_image(out / "fig_7_22_miz_image.png", np.swapaxes(rgb, 0, 1)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 7))
    imshow_matlab(axes[0, 0], rgb, title="(a) input image")
    imshow_matlab(axes[0, 1], seg == 1, title="'light ice' segmentation (SEG_L)")
    imshow_matlab(axes[1, 0], seg == 0.5, title="'dark ice' segmentation (SEG_D)")
    imshow_matlab(axes[1, 1], seg, autoscale=True, title="SEGMENTATION (Algorithm 3, levels 0 / 0.5 / 1)")
    fig.suptitle("Section 7.2.2 Algorithm 3 — sea ice edge detection (the procedure of Figs. 7.10/7.11)")
    written.append(finish_figure(fig, out / "sec_7_2_2_segmentation.png", args.show))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    imshow_matlab(axes[0], seg, autoscale=True, title="(a) segmentation (Algorithm 3)")
    imshow_matlab(axes[1], r.fill != 0, title="(b) after hole filling only")
    imshow_matlab(axes[2], r.out != 0, title=f"(c) after shape enhancement ({r.t} pieces)")
    fig.suptitle("Section 7.2.3 Algorithm 4 — sea ice shape enhancement (the procedure of Fig. 7.12)")
    written.append(finish_figure(fig, out / "sec_7_2_3_enhancement.png", args.show))

    fig, ax = plt.subplots(figsize=(14, 6))
    rgb_index = label2rgb(r.index, cmap="jet", background=(1, 1, 1), shuffle=False)
    ax.imshow(rgb_index)
    if r.floe_cen.size:
        ax.plot(r.floe_cen[:, 0] - 1, r.floe_cen[:, 1] - 1, "k*", markersize=4)
    if r.brash_cen.size:
        ax.plot(r.brash_cen[:, 0] - 1, r.brash_cen[:, 1] - 1, "k.", markersize=2)
    ax.axis("off")
    ax.set_title("Eq. (7.6) size-coded ice pieces; black dots = piece positions "
                 f"(colour-bar ticks {ticks.tolist()})")
    written.append(finish_figure(fig, out / "sec_7_2_4_colorized.png", args.show))

    fig, axes = plt.subplots(2, 2, figsize=(16, 7))
    for ax, layer, title in ((axes[0, 0], r.index_floe != 0, "(a) ice floes"),
                             (axes[0, 1], r.index_brash != 0, "(b) brash ice"),
                             (axes[1, 0], r.index_slush != 0, "(c) slush"),
                             (axes[1, 1], r.index_water != 0, "(d) water")):
        imshow_matlab(ax, layer, title=title)
    fig.suptitle("Section 7.2.4 Algorithm 5 — the four layers (the procedure of Fig. 7.14)")
    written.append(finish_figure(fig, out / "sec_7_2_4_layers.png", args.show))

    if r.fsd is not None and r.fsd.counts.sum():
        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = plt.get_cmap("jet")
        cols = r.fsd.colors.astype(float)
        norm = (cols - cols.min()) / max(cols.max() - cols.min(), 1.0)
        width = (r.fsd.centers[1] - r.fsd.centers[0]) if r.fsd.centers.size > 1 else 1.0
        ax.bar(r.fsd.centers, r.fsd.counts, width=width, color=cmap(norm), edgecolor="k", linewidth=0.2)
        ax.set_xlabel("floe area (pixels)")
        ax.set_ylabel("number of floes")
        ax.set_title(f"Floe size distribution, hist(floe_area, {r.fsd.nbins}) "
                     "(the procedure of Fig. 7.15; bars coloured by Eq. (7.6))")
        written.append(finish_figure(fig, out / "sec_7_2_4_fsd.png", args.show))

    written.append(save_image(out / "sec_7_2_4_residue.png", r.index_residue != 0))
    written.append(save_image(out / "sec_7_2_3_out_labels.png",
                              label2rgb(r.out, cmap="jet", background=(1, 1, 1), shuffle=True)))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
