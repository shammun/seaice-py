"""Book §7.3.1.1 "Local processing" + **Algorithm 6** steps 1-5 (pp. 163-164, 168), Fig. 7.18.

The overall sea ice image is divided into **overlapping** sub-images, Algorithm 3 (ice edge detection) is run on
each one, the overlapping parts are removed and the sub-segmentations are stitched into one overall
segmentation; Algorithms 4 and 5 then run on the stitched image exactly as in ``ch07_sea_ice_demo.py``.

**No MATLAB file exists for this section** and the book fixes **none** of the parameters: not the tile size, not
the overlap size, not the stitching rule.  They are all options here; the defaults are this port's choice.
Step 6 of Algorithm 6, the geometric calibration of §7.3.1.2, is deliberately **not** performed — it needs
Appendix A.1.1's camera model (shooting angle 20 deg, FOV 46 deg) and lands in ch10
(``seaice.ch07_ice_type.resample_categorical`` is the only piece of it ch07 owns).

The book's own §7.3.1 image (Fig. 7.17, an oblique aerial scene) is **not shipped**, so this runs the procedure
on ``sea_ice_test.jpg``; the figures are named ``sec_7_3_1_*``, never ``fig_7_1x_*``, and none of the book's
2511/2624 pieces or 65.98/5.03/17.52/11.47 % can be reproduced without the source image.

Usage: ``python scripts/ch07_local_processing.py [--tile 160] [--overlap 32] [--merge crop|max|first]
[--downscale 2] [--num 500] [--iter 100] [--no-global] [--data data/book/ch07] [--out outputs/ch07] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch06_gvf_snake import BOOK_PARAMS  # noqa: E402
from seaice.ch07_ice_type import ice_shape_enhancement, local_segmentation, sea_ice_edge_detection  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, label2rgb, save_image  # noqa: E402

CH = "ch07"
P = BOOK_PARAMS["sea_ice_demo"]


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="sea_ice_test.jpg")
    p.add_argument("--tile", type=int, default=160,
                   help="sub-image side in pixels (the book states none; default: %(default)s)")
    p.add_argument("--overlap", type=int, default=32,
                   help="overlap between neighbouring sub-images (the book states none; default: %(default)s)")
    p.add_argument("--merge", default="crop", choices=["crop", "max", "first"],
                   help="stitching rule: 'crop' = the prose's 'the overlapping parts are removed'; "
                        "'max' = Fig. 7.18's 'Superimpose'; 'first' = raster-order (default: %(default)s)")
    p.add_argument("--downscale", type=int, default=2,
                   help="process every K-th row/column; 1 = the image's own resolution (default: %(default)s, "
                        "which keeps the demo near a minute; --downscale 1 --tile 256 is the full-resolution run)")
    p.add_argument("--num", type=int, default=P["Num"], help="GVF iterations per sub-image (script: 500)")
    p.add_argument("--iter", type=int, default=P["iter"], help="snake iterations per sub-image (script: 100)")
    p.add_argument("--kms0", type=int, default=P["kms0"])
    p.add_argument("--no-global", action="store_true",
                   help="skip the whole-image Algorithm 3 run that the local one is compared against")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    try:
        rgb, label = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: private image absent ({exc})")
        return 0
    if args.downscale > 1:
        rgb = rgb[::args.downscale, ::args.downscale]

    M, N = rgb.shape[:2]
    print(f"=== Algorithm 6 steps 1-5 / section 7.3.1.1 local processing on {args.image} [{label}] "
          f"{M}x{N} (downscale {args.downscale}) ===")
    print(f"tile {args.tile}, overlap {args.overlap}, merge '{args.merge}', Num {args.num}, iter {args.iter}")
    print("NOTE: the book specifies NO tile size, NO overlap size and NO stitching rule for section 7.3.1.1 — "
          "all three are this port's choice (parity: reimplemented, analysis/ch07.md risk R13).")

    t0 = time.time()

    def progress(k, n, rec):
        r0, r1, c0, c1 = rec.bounds
        msg = f"  tile {k:2d}/{n} [{r0}:{r1}, {c0}:{c1}] -> {rec.n_light} light + {rec.n_dark} dark pieces"
        print(msg + (f"   !! {rec.error}" if rec.error else ""))

    ls = local_segmentation(rgb, tile=args.tile, overlap=args.overlap, merge=args.merge, progress=progress,
                            kms0=args.kms0, sigma=P["sigma"], GradientOn=P["GradientOn"], GVFOn=P["GVFOn"],
                            Num=args.num, mu=P["mu"], iter=args.iter, alpha=P["alpha"], beta=P["beta"],
                            gamma=P["gamma"], kappa=P["kappa"], Dmin=P["Dmin"], Dmax=P["Dmax"],
                            Ra_min=P["Ra_min"], Ra=P["Ra"], Rc=P["Rc"], Rl=P["Rl"], se_radius=P["se_radius"],
                            timer=P["timer"], keep_history=False)
    t_local = time.time() - t0
    bad = [r for r in ls.tiles if r.error]
    print(f"steps 1-5: {ls.rows}x{ls.cols} = {len(ls.tiles)} sub-images of {ls.tile[0]}x{ls.tile[1]} px, "
          f"overlap {ls.overlap}, stitched with '{ls.merge}' in {t_local:.1f} s"
          + (f"; {len(bad)} degenerate sub-image(s): {[b.index for b in bad]}" if bad else ""))
    print(f"stitched SEG: light {int((ls.seg == 1).sum())} px, dark {int((ls.seg == 0.5).sum())} px, "
          f"water {int((ls.seg == 0).sum())} px; ICE {int((ls.bk != 0).sum())} px")
    print("step 6 (geometric calibration, section 7.3.1.2) is NOT performed here: it needs Appendix A.1.1's "
          "camera model (shooting angle 20 deg, FOV 46 deg) and is ported in ch10 (risk R14).")

    # ---- Algorithm 6 steps 7-10 on the stitched segmentation ---------------------------------------------
    e = ice_shape_enhancement(ls.bk, ls.seg, min_floe=P["min_floe"], min_brash=P["min_brash"], se_th=P["se_th"])
    cov = e.coverage.as_percent()
    print(f"steps 7-10 (Algorithms 4+5 on SEG): {e.t} identified pieces -> {len(e.ice_floe)} ice floes, "
          f"{len(e.brash_ice)} brash pieces")
    print(f"  coverage: {cov['IceFloe']:.2f} % floe, {cov['BrashIce']:.2f} % brash, {cov['Slush']:.2f} % slush, "
          f"{cov['Water']:.2f} % water")

    # ---- the whole-image run this is supposed to improve on ----------------------------------------------
    g_seg = g_e = None
    if not args.no_global:
        t0 = time.time()
        g = sea_ice_edge_detection(rgb, kms0=args.kms0, sigma=P["sigma"], GradientOn=P["GradientOn"],
                                   GVFOn=P["GVFOn"], Num=args.num, mu=P["mu"], iter=args.iter,
                                   alpha=P["alpha"], beta=P["beta"], gamma=P["gamma"], kappa=P["kappa"],
                                   Dmin=P["Dmin"], Dmax=P["Dmax"], Ra_min=P["Ra_min"], Ra=P["Ra"],
                                   Rc=P["Rc"], Rl=P["Rl"], se_radius=P["se_radius"], timer=P["timer"],
                                   keep_history=False)
        t_global = time.time() - t0
        g_seg = g.out
        g_e = ice_shape_enhancement(g.bk, g_seg, min_floe=P["min_floe"], min_brash=P["min_brash"],
                                    se_th=P["se_th"])
        print(f"whole-image Algorithm 3 for comparison ({t_global:.1f} s): {g_e.t} pieces -> "
              f"{len(g_e.ice_floe)} floes, {len(g_e.brash_ice)} brash; the two segmentations differ on "
              f"{int((ls.seg != g_seg).sum())} of {M * N} px ({100.0 * (ls.seg != g_seg).mean():.2f} %)")
        print("  (p. 163: local processing costs 'more processing time and possibly manual intervention' — "
              f"here {t_local:.1f} s vs {t_global:.1f} s; which of the two is *better* cannot be decided "
              "without the ground truth the book does not ship)")
    print("NOTE: the book's Fig. 7.17 scene is not shipped, so its 2511/2624 pieces and "
          "65.98/5.03/17.52/11.47 % are not reproducible.")

    # ---- figures ------------------------------------------------------------------------------------------
    written = []
    fig, ax = plt.subplots(figsize=(7, 14 * M / max(N, 1) / 2 + 2))
    imshow_matlab(ax, rgb)
    for rec in ls.tiles:
        r0, r1, c0, c1 = rec.bounds
        ax.add_patch(mpatches.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, ec="r", lw=1.0))
        k0, k1, k2, k3 = rec.core
        ax.add_patch(mpatches.Rectangle((k2 - 0.5, k0 - 0.5), k3 - k2, k1 - k0, fill=False, ec="y", lw=1.2,
                                        ls="--"))
    ax.set_title(f"section 7.3.1.1 division: {ls.rows}x{ls.cols} sub-images of {ls.tile[0]}x{ls.tile[1]} px, "
                 f"overlap {ls.overlap[0]}x{ls.overlap[1]}\nred = sub-image, yellow = the core kept after the "
                 "overlap is removed")
    written.append(finish_figure(fig, out / "sec_7_3_1_1_tiles.png", args.show))

    demo = ls.tiles[len(ls.tiles) // 2]
    dr0, dr1, dc0, dc1 = demo.bounds
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    imshow_matlab(axes[0], rgb, title="overall sea-ice image")
    imshow_matlab(axes[1], rgb[dr0:dr1, dc0:dc1], title=f"one smaller, overlapping region {demo.index}")
    imshow_matlab(axes[2], ls.seg[dr0:dr1, dc0:dc1], autoscale=True, title="its sub-segmentation (Algorithm 3)")
    imshow_matlab(axes[3], ls.seg, autoscale=True, title="overall segmentation (stitched)")
    fig.suptitle("Fig. 7.18 — local segmentation procedure (divide -> ice edge detection -> superimpose)")
    written.append(finish_figure(fig, out / "sec_7_3_1_1_local_segmentation.png", args.show))

    n_panels = 3 if g_seg is None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 6))
    imshow_matlab(axes[0], ls.seg, autoscale=True, title=f"local SEG (merge '{ls.merge}')")
    axes[1].imshow(label2rgb(e.out, cmap="jet", background=(1, 1, 1), shuffle=True))
    axes[1].set_title(f"Algorithm 4 on it — {e.t} pieces")
    axes[1].axis("off")
    imshow_matlab(axes[2], e.index_floe != 0, title=f"floes ({len(e.ice_floe)}) — Algorithm 5")
    if g_seg is not None:
        imshow_matlab(axes[3], ls.seg != g_seg, title=f"local vs whole-image SEG ({int((ls.seg != g_seg).sum())} px)")
    fig.suptitle("Algorithm 6 steps 5, 7 and 8-10 (step 6, the geometric calibration, is ch10's)")
    written.append(finish_figure(fig, out / "sec_7_3_1_3_algorithm6.png", args.show))

    written.append(save_image(out / "sec_7_3_1_1_stitched_seg.png", ls.seg / max(ls.seg.max(), 1.0)))
    written.append(save_image(out / "sec_7_3_1_3_identification.png",
                              label2rgb(e.index, cmap="jet", background=(1, 1, 1), shuffle=False)))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
