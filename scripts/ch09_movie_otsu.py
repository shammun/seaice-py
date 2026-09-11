"""Port of ``MATLAB_ROOT/ch9/movie_otsu.m`` — §9.2.2 per-frame global-Otsu ice concentration of a tank video.

The M-file reads ``dypic_05100_cam1_top.avi``, blanks the vessel with a black rectangle, crops away the tank
impediments (**Fig. 9.6**), thresholds every frame with its own Otsu level (**Fig. 9.7(a)(b)**), plots ``IC``
against the frame index (**Figs. 9.8/9.10**, "Time" because the video is decimated to 1 fps) and writes
``otsu.avi`` at 12 fps.

Two errata are reproduced literally (see :func:`seaice.ch09_model_ice.movie_otsu`):

* **E4** line 39 ``if I(k).cdata(i,j) >= t*255`` compares against the *whole growing vector* ``t``, so ``if``
  means ``all(...)`` and the threshold that actually counts is the **running maximum** ``max_{j<=k} t(j)``.
  ``--corrected`` switches to the per-frame level, which is what Figs. 9.8/9.10 claim to show.
* **E5** line 40 ``n = n+1`` increments **every** element of ``n``; the printed ``n`` vector is therefore
  ``n[j] = Σ_{m>=j} count_m``.  ``IC`` escapes E5 (``n(k)`` is read inside the same iteration) but not E4.

``im2bw`` on line 45 uses the **per-frame** level with a strict ``>``, so the written AVI and the plotted IC
legitimately disagree — both are produced here.

DATA: ``dypic_05100_cam1_top.avi`` is an HSVA/DYPIC asset that was **never published** (risk R1); with no
``--video`` the script uses the seeded Tier-3 ``core.synth.model_ice_tank_video`` written as an **Uncompressed
AVI** (risk R2: a lossy container would make every number a decoder comparison).  Book numbers N9–N12 are
**not** reproducible and are never invented.

Usage: ``python scripts/ch09_movie_otsu.py [--video PATH] [--frames 24] [--corrected] [--no-avi]
[--data data/book/ch09] [--out outputs/ch09] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch09_model_ice import (BOX_5100, CROP_5100, IC_DENOMINATOR_5100, TABLE_9_3,  # noqa: E402
                                   ensure_synthetic_tank_video, movie_otsu, preprocess_frame)
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402
from seaice.core.video import read_video, write_video  # noqa: E402

CH = "ch09"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--video", default="dypic_05100_cam1_top.avi", help="movie_otsu.m line 7")
    p.add_argument("--frames", type=int, default=24, help="frames in the synthetic stand-in (default 24)")
    p.add_argument("--corrected", action="store_true",
                   help="disable erratum E4 (use the per-frame Otsu level instead of the running maximum)")
    p.add_argument("--count-rule", default="ge", choices=["ge", "gt"], help="line 39's `>=` (default) or `>`")
    p.add_argument("--no-avi", action="store_true", help="skip writing otsu.avi")
    p.add_argument("--regenerate", action="store_true", help="rewrite the synthetic AVI")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    candidate = Path(args.video)
    if not candidate.is_absolute():
        candidate = data / args.video
    if candidate.exists():
        frames = read_video(candidate)
        source, synthetic = str(candidate), False
        print(f"NOTE: reading a user-supplied video ({candidate.name}).  If it is a LOSSY container, MATLAB's "
              "codec and ffmpeg can disagree by several gray levels, which moves every Otsu threshold — the "
              "parity label for this run drops to `near`/`approx` (analysis/ch09.md risk R2).")
    else:
        path, frames = ensure_synthetic_tank_video(n_frames=args.frames, regenerate=args.regenerate)
        source, synthetic = f"SYNTHETIC (Tier 3) {path.name}", True
        print(f"NOTE: {args.video} is not present — it is an HSVA/DYPIC asset that was never published "
              f"(analysis/ch09.md risk R1).  Using the seeded synthetic Uncompressed AVI {path}.")
        print("      Book numbers N9-N12 (Fig. 9.7's IC = 87.26 % / threshold = 100 at t = 816 s, the ~89 % "
              "saturation, Table 9.3) are NOT reproducible here and are NOT invented.")

    H, W, _, N = frames.shape
    y1, y2, x1, x2 = CROP_5100
    y3, _, x3, x4 = BOX_5100
    print(f"\n=== movie_otsu.m on {source} ===")
    print(f"{N} frames of {H} x {W}; crop y {y1}:{y2}, x {x1}:{x2} -> {y2 - y1 + 1} x {x2 - x1 + 1}; "
          f"vessel box y {y3}:{y2}, x {x3}:{x4} -> {y2 - y3 + 1} x {x4 - x3 + 1} = "
          f"{(y2 - y3 + 1) * (x4 - x3 + 1)} px")
    print(f"IC denominator r*c - r2*c2 = {IC_DENOMINATOR_5100}  [book number N8, pure arithmetic]")
    if H < y2 or W < x2:
        print(f"SKIP: the video is {H} x {W}; the script's hard-coded crop needs at least {y2} x {x2}.")
        return 0

    res = movie_otsu(frames, running_max_bug=not args.corrected, count_rule=args.count_rule)
    lit = movie_otsu(frames, running_max_bug=True, count_rule=args.count_rule) if args.corrected else res
    cor = res if args.corrected else movie_otsu(frames, running_max_bug=False, count_rule=args.count_rule)

    print(f"\nrule in force: {'per-frame level (E4 CORRECTED, --corrected)' if args.corrected else 'running maximum (E4, the shipped code)'}"
          f"; count rule `{args.count_rule}`")
    print(" k   t(k)      255*t(k)   effective 255*t   count      IC (%)   IC corrected (%)")
    for k in range(N):
        print(f"{k + 1:3d}  {res.t[k]:.6f}  {res.t[k] * 255:8.2f}  {res.effective_level[k] * 255:14.2f}  "
              f"{res.counts[k]:8d}  {res.IC[k] * 100:7.3f}  {cor.IC[k] * 100:15.3f}")
    n_dec = int((np.diff(res.t) < 0).sum())
    print(f"\nframes where the Otsu threshold DECREASED: {n_dec} "
          "(without at least one, erratum E4 would be invisible — the fixture could not discriminate)")
    print(f"max |IC(E4) - IC(corrected)| = {np.abs(lit.IC - cor.IC).max() * 100:.3f} percentage points")
    print(f"n after erratum E5 (n[j] = sum_{{m>=j}} count_m) = {lit.n.tolist()}")
    print(f"the E5-free per-frame counts           = {lit.counts.tolist()}")
    print(f"IC range {res.IC.min() * 100:.2f} .. {res.IC.max() * 100:.2f} %, mean {res.IC.mean() * 100:.2f} %")
    if synthetic:
        print("\nBOOK NUMBERS (NOT reproduced; printed for orientation only): Table 9.3 saturation start / "
              "average IC per run: " + ", ".join(f"{r}: {t} s / {ic} %" for r, (t, ic) in TABLE_9_3.items()))

    # Fig. 9.6 — the pre-processing, on the middle frame
    mid = N // 2
    raw = frames[:, :, :, mid]
    cropped, _I1, _I2, _r2, _c2 = preprocess_frame(raw, CROP_5100, BOX_5100)
    written = [save_image(out / "fig_9_06_a_raw_frame.png", raw),
               save_image(out / "fig_9_06_b_blanked_cropped.png", cropped),
               save_image(out / "fig_9_07_a_gray_frame.png", res.gray[mid]),
               save_image(out / "fig_9_07_b_otsu_mask.png", res.bw[mid])]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, N + 1), res.IC * 100, "-o", ms=3,
            label="movie_otsu.m as shipped (E4: running max)" if not args.corrected else "per-frame level")
    if not args.corrected:
        ax.plot(np.arange(1, N + 1), cor.IC * 100, "--", lw=1,
                label="corrected: per-frame Otsu level")
    ax.set_xlabel("Time")
    ax.set_ylabel("IC")
    ax.legend(fontsize=8)
    ax.set_title("Fig. 9.8 — ice concentration per frame (global Otsu)"
                 + (" [synthetic]" if synthetic else ""))
    written.append(finish_figure(fig, out / "fig_9_08_ic_time_otsu.png", args.show))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, N + 1), res.t * 255, "-o", ms=3, label="graythresh per frame, 255*t(k)")
    ax.plot(np.arange(1, N + 1), res.effective_level * 255, "--", lw=1,
            label="effective threshold actually counted (E4)")
    ax.set_xlabel("Time")
    ax.set_ylabel("threshold")
    ax.legend(fontsize=8)
    ax.set_title("Fig. 9.10 — Otsu threshold per frame" + (" [synthetic]" if synthetic else ""))
    written.append(finish_figure(fig, out / "fig_9_10_threshold_time.png", args.show))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    imshow_matlab(axes[0], raw, title="mov(k).cdata (raw frame)")
    imshow_matlab(axes[1], cropped, title="blanked + cropped (Fig. 9.6(b))")
    imshow_matlab(axes[2], res.bw[mid], title=f"im2bw(I, t({mid + 1})) = {res.t[mid] * 255:.1f}")
    fig.suptitle("movie_otsu.m — pre-processing and per-frame segmentation")
    written.append(finish_figure(fig, out / "fig_9_06_preprocessing.png", args.show))

    if not args.no_avi:
        avi = write_video(out / "otsu.avi", res.bw, 12.0)   # movie2avi(M, 'otsu.avi', 'FPS', 12) — line 65
        written.append(avi)
        print(f"\notsu.avi: {N} frames at 12 fps, Uncompressed AVI "
              "(`movie2avi` was removed in R2025a; the reference run patches it to VideoWriter + writeVideo)")

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
