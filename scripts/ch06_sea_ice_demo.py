"""Port of ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/sea_ice_demo.m`` — the chapter's only driver.

The MATLAB script is a parameter block followed by three calls::

    [seg, bk] = seaice_kmean_GVF_forenhancement(I, kms0, sigma, GradientOn, GVFOn, Num, mu, iter, alpha, beta,
                                                gamma, kappa, Dmin, Dmax, Ra_min, Ra, Rc, Rl, se, timer);
    [out, index_floe, ice_floe, index_brash, brash_ice, index_slush, index_water, index_residue, coverage] = ...
        ice_shape_enhancement(bk, seg, min_floe, min_brash, se_th);
    [floe, brash] = sea_ice_model(ice_floe, brash_ice, index_floe);

Only the **first** call belongs to chapter 6.  ``ice_shape_enhancement.m`` implements book **§7.1 "Ice shape
enhancement"** (the file is byte-identical in ``ch7/Sea_Ice_Floe_Identification/`` and produces §7.2's four
layers plus the coverage percentages quoted in §8.3), and ``sea_ice_model.m`` implements **§8.2 / Appendix B**
(``.Polygon``, ``.Circle``, ``.Intersect``).

``--full`` now also runs the second call through :func:`seaice.ch07_ice_type.ice_shape_enhancement` (ported in
ch07) and reports its layers; the third call is still ch08's and is reported as deferred.  Note that this script
runs on **ch6's** ``sea_ice_test.jpg``; ch7 ships a *different JPEG encoding* of the same photograph, so the
piece counts differ from ``scripts/ch07_sea_ice_demo.py``'s — use that script for the chapter-7 numbers.

Output of the ch6 part: ``seg`` (= ``out`` of the M-file) has **three levels** — 1 = bright ice (Otsu pass),
0.5 = dark/slush ice (k-means residual pass), 0 = water — and ``bk`` is the k-means ice mask.

Usage: ``python scripts/ch06_sea_ice_demo.py [--image sea_ice_test.jpg] [--num 500] [--iter 100]
[--max-seeds N] [--downscale K] [--kmeans-impl lloyd++|sklearn] [--strict-bwareaopen] [--full]
[--data data/book/ch06] [--out outputs/ch06] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch06_gvf_snake import BOOK_PARAMS, seaice_kmean_gvf  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.connectivity import label_components  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, label2rgb, save_image  # noqa: E402

CH = "ch06"
P = BOOK_PARAMS["sea_ice_demo"]


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="sea_ice_test.jpg", help="sea_ice_demo.m line 9")
    p.add_argument("--num", type=int, default=P["Num"], help="GVF iterations (script: 500)")
    p.add_argument("--iter", type=int, default=P["iter"], help="snake iterations (script: 100)")
    p.add_argument("--kms0", type=int, default=P["kms0"], help="k-means cluster count (script: 3)")
    p.add_argument("--max-seeds", type=int, default=None,
                   help="cap the contours per pass (runtime guard; default: all, ~45 s on the book image)")
    p.add_argument("--downscale", type=int, default=1,
                   help="process every K-th row/column (runtime guard; 1 = the book's own resolution)")
    p.add_argument("--kmeans-impl", default="lloyd++", choices=["lloyd++", "sklearn"],
                   help="stand-in for the Statistics Toolbox kmeans (see the DEVIATION note in the module)")
    p.add_argument("--kmeans-seed", type=int, default=0)
    p.add_argument("--strict-bwareaopen", action="store_true",
                   help="keep only the +1 pixels of bw_kmeans - bw (the script keeps the -1 pixels too)")
    p.add_argument("--solver", default="auto", choices=["auto", "dense", "circulant"])
    p.add_argument("--full", action="store_true",
                   help="also run ice_shape_enhancement (ch07); sea_ice_model stays deferred to ch08")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        rgb, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: skipped: private image absent ({exc})")
        return 0
    if args.downscale > 1:
        rgb = rgb[::args.downscale, ::args.downscale]
        print(f"NOTE: --downscale {args.downscale} -> {rgb.shape[0]}x{rgb.shape[1]}; "
              "the book's numbers only hold at the original resolution")

    t0 = time.time()
    res = seaice_kmean_gvf(rgb, kms0=args.kms0, sigma=P["sigma"], GradientOn=P["GradientOn"],
                           GVFOn=P["GVFOn"], Num=args.num, mu=P["mu"], iter=args.iter, alpha=P["alpha"],
                           beta=P["beta"], gamma=P["gamma"], kappa=P["kappa"], Dmin=P["Dmin"],
                           Dmax=P["Dmax"], Ra_min=P["Ra_min"], Ra=P["Ra"], Rc=P["Rc"], Rl=P["Rl"],
                           se_radius=P["se_radius"], timer=P["timer"], max_seeds=args.max_seeds,
                           keep_history=False, kmeans_impl=args.kmeans_impl, kmeans_seed=args.kmeans_seed,
                           strict_bwareaopen=args.strict_bwareaopen, solver=args.solver)
    elapsed = time.time() - t0
    seg, bk = res.out, res.bk

    print(f"=== sea_ice_demo.m on {args.image} ({rgb.shape[0]}x{rgb.shape[1]}) ===")
    print("parameters: " + ", ".join(f"{k} = {v}" for k, v in P.items()))
    print(f"Otsu mask: {int(res.bw.sum())} ice px ({100.0 * res.bw.mean():.2f} %)")
    print(f"k-means ({args.kmeans_impl}, k = {args.kms0}): cluster means "
          f"{np.array2string(np.sort(res.s0), precision=3)}; darkest cluster = water -> "
          f"bk has {int(bk.sum())} ice px ({100.0 * bk.mean():.2f} %)")
    print(f"bw0 = bk - bw: {int((res.bw0_raw > 0).sum())} pixels at +1 and {res.n_negative} at -1 "
          f"(the script's bwareaopen keeps both); after bwareaopen(.., {P['Ra_min']}, 4): {int(res.bw0.sum())}")
    for tag, pr in (("pass 1 (Otsu)", res.pass1), ("pass 2 (k-means residual)", res.pass2)):
        rec = pr.passes[0]
        print(f"{tag}: {rec.num} components, {rec.k.size} failing the criteria, {pr.n_seeds} seeds "
              f"({pr.n_seeds_run} run); {int(pr.bw.sum()) - int(pr.bw1.sum())} boundary pixels burnt; "
              f"floes {int(label_components(pr.bw, 4).max())} -> {int(label_components(pr.bw1, 4).max())}")
        # Review S9: the guards the M-code does not have, counted rather than silent.
        print(f"  guards: {pr.n_skipped} contour(s) skipped (< 3 vertices after the polybool clip), "
              f"{pr.n_below_range} burn point(s) below the image (MATLAB would error), "
              f"{pr.n_above_range} above it (the M-code's own guard)")
    print(f"out = bw1 + 0.5*bw0: levels {np.unique(seg).tolist()}; "
          f"bright ice {int((seg == 1).sum())} px, slush {int((seg == 0.5).sum())} px, "
          f"water {int((seg == 0).sum())} px")
    print(f"wall clock {elapsed:.1f} s (GVF {args.num} iterations, solver = {args.solver})")

    written = [
        save_image(out / "sec_6_4_demo_a_gray.png", res.gray),
        save_image(out / "sec_6_4_demo_b_otsu_mask.png", res.bw),
        save_image(out / "sec_6_4_demo_c_kmeans_mask.png", bk != 0),
        save_image(out / "sec_6_4_demo_d_kmeans_residual.png", res.bw0),
        save_image(out / "sec_6_4_demo_e_pass1_segmentation.png", res.pass1.bw1),
        save_image(out / "sec_6_4_demo_f_pass2_segmentation.png", res.pass2.bw1),
        save_image(out / "sec_6_4_demo_g_three_level.png", seg),
        save_image(out / "sec_6_4_demo_h_labels.png", label2rgb(label_components(res.pass1.bw1, 4))),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 12))
    imshow_matlab(axes[0, 0], res.gray, title="rgb2gray(I)")
    imshow_matlab(axes[0, 1], res.bw, title="bw = im2bw(I, graythresh(I))")
    imshow_matlab(axes[0, 2], bk != 0, title=f"bk = k-means ice mask (k = {args.kms0})")
    imshow_matlab(axes[1, 0], res.bw0, title="bw0 = bwareaopen(bk - bw, Ra_min, 4)")
    imshow_matlab(axes[1, 1], res.pass1.bw1, title="pass 1: bright ice with snake boundaries")
    imshow_matlab(axes[1, 2], seg, autoscale=True, title="out = bw1 + 0.5*bw0 (3 levels)")
    fig.suptitle("sea_ice_demo.m — seaice_kmean_GVF_forenhancement (the ch6 part of the pipeline)")
    written.append(finish_figure(fig, out / "sec_6_4_demo_panels.png", args.show))

    print("\nfigures written:")
    for w in written:
        print("  ", w)

    if args.full:
        # sea_ice_demo.m line 53 — ported in ch07 (book section 7.1.4 / 7.2.3 / 7.2.4, Algorithms 2/4/5).
        from seaice.ch07_ice_type import ice_shape_enhancement

        t0 = time.time()
        e = ice_shape_enhancement(bk, seg, min_floe=P["min_floe"], min_brash=P["min_brash"], se_th=P["se_th"])
        cov = e.coverage.as_percent()
        print(f"\nice_shape_enhancement.m (ch07, {time.time() - t0:.1f} s): "
              f"{e.nn_bw} light + {e.nn_k} dark pieces -> {e.t} identified; "
              f"{len(e.ice_floe)} ice floes, {len(e.brash_ice)} brash pieces")
        print(f"  coverage: {cov['IceFloe']:.2f} % floe, {cov['BrashIce']:.2f} % brash, "
              f"{cov['Slush']:.2f} % slush, {cov['Water']:.2f} % water")
        print("  NOTE: this is ch6's copy of sea_ice_test.jpg; scripts/ch07_sea_ice_demo.py runs ch7's copy, "
              "which is a different JPEG encoding of the same photograph and gives different counts.")
        written.append(save_image(out / "sec_6_4_demo_i_identification.png",
                                  label2rgb(e.out, cmap="jet", background=(1, 1, 1), shuffle=True)))
        print("  wrote", written[-1])
        print("sea_ice_demo.m line 57 `sea_ice_model` implements book section 8.2 / Appendix B and is ported "
              "in ch08 — not run here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
