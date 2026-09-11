"""Port of ``MATLAB_ROOT/ch9/movie_kmeans.m`` — §9.2.2 per-frame k-means (k = 2) ice concentration.

Identical pre-processing to ``movie_otsu.m`` (blank the vessel, crop, ``rgb2gray``), then ``im2double``, a
column-major flatten, ``kmeans(ima, 2, 'EmptyAction', 'singleton')``, the brighter cluster as ice,
``IC(k) = length(p)/(si(1)*si(2) - r2*c2)`` (**the same 77 115 denominator as Otsu**) and a 0/255 mask written to
``05400_kmeans.avi`` — **erratum E6**: the input is run *5100*, the output is named run *5400*.

# DEVIATION: `approx` — ch9 ships no ``kmeans.m``, so line 44 resolves to the **Statistics Toolbox** routine
# (k-means++ start, RNG-dependent).  The stand-in is ``core.clustering.kmeans_lloyd(X, 2, init='kmeans++',
# seed=...)``; compare sorted cluster means and pixel agreement, never label numbers.  Everything around the
# clustering is exact.  (Unlike the *text-only* k-means of §9.2.1, which uses the authors' own deterministic
# ``ch3/kmeans.m`` — see ``scripts/ch09_block_threshold.py``.)

DATA: same as ``ch09_movie_otsu.py`` — the AVI was never published (risk R1); the seeded Tier-3 Uncompressed AVI
stands in and every book number stays `unverified`.

Usage: ``python scripts/ch09_movie_kmeans.py [--video PATH] [--frames 24] [--k 2] [--seed 0] [--no-avi]
[--data data/book/ch09] [--out outputs/ch09] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch09_model_ice import (CROP_5100, IC_DENOMINATOR_5100,  # noqa: E402
                                   ensure_synthetic_tank_video, movie_kmeans, movie_otsu)
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402
from seaice.core.video import read_video, write_video  # noqa: E402

CH = "ch09"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--video", default="dypic_05100_cam1_top.avi", help="movie_kmeans.m line 7")
    p.add_argument("--frames", type=int, default=24, help="frames in the synthetic stand-in (default 24)")
    p.add_argument("--k", type=int, default=2, help="kms (script: 2)")
    p.add_argument("--seed", type=int, default=0, help="seed of the k-means stand-in (the toolbox has an RNG)")
    p.add_argument("--impl", default="lloyd++", choices=["lloyd++", "sklearn"])
    p.add_argument("--no-avi", action="store_true", help="skip writing 05400_kmeans.avi")
    p.add_argument("--regenerate", action="store_true", help="rewrite the synthetic AVI")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    candidate = Path(args.video)
    if not candidate.is_absolute():
        candidate = data / args.video
    if candidate.exists():
        frames = read_video(candidate)
        source, synthetic = str(candidate), False
    else:
        path, frames = ensure_synthetic_tank_video(n_frames=args.frames, regenerate=args.regenerate)
        source, synthetic = f"SYNTHETIC (Tier 3) {path.name}", True
        print(f"NOTE: {args.video} is not present (HSVA/DYPIC, never published — risk R1).  Using {path}.")
        print("      Book numbers N9-N12 are NOT reproducible here and are NOT invented.")

    H, W, _, N = frames.shape
    y1, y2, x1, x2 = CROP_5100
    print(f"\n=== movie_kmeans.m on {source} ===")
    print(f"{N} frames of {H} x {W}; si = [{y2 - y1 + 1} {x2 - x1 + 1}]; "
          f"IC denominator si(1)*si(2) - r2*c2 = {IC_DENOMINATOR_5100} (the same as movie_otsu.m)")
    if H < y2 or W < x2:
        print(f"SKIP: the video is {H} x {W}; the script's hard-coded crop needs at least {y2} x {x2}.")
        return 0

    res = movie_kmeans(frames, k=args.k, impl=args.impl, seed=args.seed)
    otsu = movie_otsu(frames)          # for the side-by-side of Fig. 9.8

    print(f"\nk-means stand-in: {args.impl}, seed {args.seed}  [DEVIATION: approx — the toolbox routine's RNG "
          "cannot be reproduced; compare sorted cluster means and pixel agreement, never labels]")
    print(" k   cluster means s (sorted)          a(k)=ice   #ice px    IC (%)   Otsu IC (%)")
    for k in range(N):
        sm = np.sort(res.s[k])
        print(f"{k + 1:3d}   {np.array2string(sm, precision=6, floatmode='fixed')}   {res.a[k]:6d}   "
              f"{int(round(res.IC[k] * res.denominator)):8d}  {res.IC[k] * 100:7.3f}  {otsu.IC[k] * 100:10.3f}")
    print(f"\nIC range {res.IC.min() * 100:.2f} .. {res.IC.max() * 100:.2f} %, mean {res.IC.mean() * 100:.2f} %")
    print(f"mean |IC(kmeans) - IC(Otsu, as shipped)| = {np.abs(res.IC - otsu.IC).mean() * 100:.3f} pp")
    print(f"out(k).cdata: {res.out.shape} {res.out.dtype}, levels {np.unique(res.out).tolist()} "
          "(uint8(mask*255) then a COLUMN-MAJOR reshape, line 58)")
    print("erratum E6: the input is run 5100 and the output AVI is named 05400_kmeans.avi — reproduced.")

    mid = N // 2
    written = [save_image(out / "fig_9_07_c_kmeans_mask.png", res.out[mid])]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, N + 1), res.IC * 100, "-o", ms=3, label=f"k-means, k = {args.k}")
    ax.plot(np.arange(1, N + 1), otsu.IC * 100, "--", lw=1, label="global Otsu (movie_otsu.m as shipped)")
    ax.set_xlabel("Time")
    ax.set_ylabel("IC")
    ax.legend(fontsize=8)
    ax.set_title("Figs. 9.8/9.9 — ice concentration per frame" + (" [synthetic]" if synthetic else ""))
    written.append(finish_figure(fig, out / "fig_9_09_ic_time_kmeans.png", args.show))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    imshow_matlab(axes[0], otsu.gray[mid], title="gray frame (blanked + cropped)")
    imshow_matlab(axes[1], otsu.bw[mid], title="Fig. 9.7(b) global Otsu")
    imshow_matlab(axes[2], res.out[mid], title=f"Fig. 9.7(c) k-means, k = {args.k}")
    fig.suptitle("movie_kmeans.m vs movie_otsu.m on the same frame")
    written.append(finish_figure(fig, out / "fig_9_07_frame_comparison.png", args.show))

    if not args.no_avi:
        # movie2avi(M, '05400_kmeans.avi', 'FPS', 12) -- line 79, erratum E6 (the input is run 5100)
        written.append(write_video(out / "05400_kmeans.avi", res.out, 12.0))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
