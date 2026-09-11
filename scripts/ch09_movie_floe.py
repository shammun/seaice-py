"""Port of ``MATLAB_ROOT/ch9/Model_Ice_Floe_Identification/movie_floe.m`` — §9.3.3 maximum floe size in video.

The M-file reads an AVI, binarises each frame with a **level-free** ``im2bw`` (⇒ 0.5 ⇒ ``rgb2gray`` then
``> 127.5``), cleans it with ``bwareaopen(·, 20, 4)``, labels with ``bwlabel(·, 4)``, takes
``regionprops(cc, 'basic')`` and records ``floe(k) = max([icedata.Area])`` — the y value of **Fig. 9.18**, in
pixels.  There is **no plotting code** in the file: Fig. 9.18 was made interactively, so this script plots it.

Two facts about this file decide how it may be used:

* Because line 18 has no level, the AVI must **already hold the segmented (binary) frames**.  The per-frame
  Algorithm-7 segmentation that §9.3.3 describes has **no shipped code at all** — gap **G2**.  ``--segment``
  wires :func:`seaice.ch09_model_ice.segment_video` in front of it; that wiring is **ours, not the book's**, and
  its output may never enter a parity claim.
* ``mmreader`` (line 5) was **removed** from R2025a; ``VideoReader`` replaces it.  That is the single IO-only
  patch the reference run needs (``reference/ch09/patches.json``).

Risk **R13, corrected**: on a frame with no surviving component ``max(ice_areas)`` is ``[]`` and
``floe(k) = []`` is a null assignment *past the end* of the vector — R2025a **errors**
(``MATLAB:matrix:singleSubscriptNumelMismatch``), it does **not** silently delete, because deletion needs
``k <= numel(floe)`` and this loop appends one element per frame.  ``--empty`` selects ``raise`` (default, the
literal behaviour), ``delete`` (the deletion semantics, for the reachable case) or ``nan``.

DATA: ``05100.avi`` is an HSVA/DYPIC asset that was **never published** (risk R1); the seeded Tier-3
``core.synth.segmented_floe_video`` stands in, written as an **Uncompressed AVI**.  Book number N19 (Fig. 9.18's
0–1000 s × 0–3e4 px axes) is **not** reproducible and is not invented.

Usage: ``python scripts/ch09_movie_floe.py [--video PATH] [--frames 40] [--min-area 20] [--conn 4]
[--empty raise|delete|nan] [--blank-frame K] [--segment] [--data data/book/ch09] [--out outputs/ch09] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch09_model_ice import ensure_synthetic_segmented_video, movie_floe  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.connectivity import bwareaopen, label_components  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, label2rgb, save_image  # noqa: E402
from seaice.core.threshold import im2bw  # noqa: E402
from seaice.core.video import read_video  # noqa: E402

CH = "ch09"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--video", default="05100.avi", help="movie_floe.m line 5 (mmreader -> VideoReader)")
    p.add_argument("--frames", type=int, default=40, help="frames in the synthetic stand-in")
    p.add_argument("--min-area", type=int, default=20, help="bwareaopen(im, 20, 4) — line 20")
    p.add_argument("--conn", type=int, default=4, choices=[4, 8], help="bwareaopen/bwlabel connectivity")
    p.add_argument("--empty", default="raise", choices=["raise", "delete", "nan"],
                   help="what to do on a blank frame (risk R13; 'raise' = MATLAB R2025a)")
    p.add_argument("--blank-frame", type=int, default=None,
                   help="plant an empty frame at this index in the synthetic video (exercises R13)")
    p.add_argument("--segment", action="store_true",
                   help="run OUR per-frame Algorithm-7 segmentation first (gap G2 — not the book's code, SLOW)")
    p.add_argument("--downscale", type=int, default=2, help="--segment runtime guard")
    p.add_argument("--max-seeds", type=int, default=20, help="--segment runtime guard")
    p.add_argument("--regenerate", action="store_true")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    candidate = Path(args.video)
    if not candidate.is_absolute():
        candidate = data / args.video
    if candidate.exists():
        frames = read_video(candidate)
        source, synthetic = str(candidate), False
    else:
        path, frames = ensure_synthetic_segmented_video(n_frames=args.frames, regenerate=args.regenerate,
                                                        blank_frame=args.blank_frame)
        source, synthetic = f"SYNTHETIC (Tier 3) {path.name}", True
        print(f"NOTE: {args.video} is not present (HSVA/DYPIC, never published — risk R1).  Using {path}.")
        print("      Book number N19 (Fig. 9.18's axes, 0-1000 s x 0-3e4 px) is NOT reproducible and NOT "
              "invented.")

    H, W, _, N = frames.shape
    print(f"\n=== movie_floe.m on {source} ===")
    print(f"{N} frames of {H} x {W}; im2bw with NO level -> 0.5 -> rgb2gray then > 127.5 "
          "(so the AVI must already hold segmented frames — gap G2)")

    if args.segment:
        from seaice.ch09_model_ice import segment_video

        print("\n*** --segment: running OUR per-frame Algorithm-7 wiring (gap G2).  This is NOT the book's "
              "code and its output may not be used in any parity claim. ***")
        masks = segment_video(frames, downscale=args.downscale, max_seeds=args.max_seeds,
                              progress=lambda k, n: print(f"    frame {k + 1}/{n}", end="\r"))
        frames = np.repeat((masks.astype(np.uint8) * 255)[..., None], 3, axis=3)
        frames = np.transpose(frames, (1, 2, 3, 0))
        H, W, _, N = frames.shape
        print(f"\n    segmented to {N} frames of {H} x {W}")

    # Show what the fixture pins before the statistic is taken (test-design notes 2 and 3).
    bw0 = im2bw(frames[:, :, :, 0])
    lab4_before = label_components(bw0, 4)
    areas0 = np.bincount(lab4_before.ravel())[1:]
    kept = bwareaopen(bw0, args.min_area, args.conn)
    print(f"frame 1: {lab4_before.max()} 4-connected components (8-connected: "
          f"{label_components(bw0, 8).max()} — the difference is the diagonal-touch pair that pins "
          "`bwlabel(.., 4)`)")
    print(f"         component areas include {int((areas0 == 19).sum())} of exactly 19 px and "
          f"{int((areas0 == 20).sum())} of exactly 20 px; after bwareaopen(.., {args.min_area}, {args.conn}) "
          f"-> {label_components(kept, args.conn).max()} components "
          "(the 19-px one is dropped, the 20-px one kept: the rule is `>= P`)")

    try:
        res = movie_floe(frames, min_area=args.min_area, conn=args.conn, empty=args.empty, keep_labels=True)
    except ValueError as exc:
        print(f"\nmovie_floe raised, reproducing MATLAB R2025a: {exc}")
        print("Re-run with --empty delete or --empty nan to continue past the blank frame.")
        return 0

    aligned = len(res.floe) == len(res.areas)
    if not aligned:
        print(f"\nNOTE: --empty {args.empty} shortened `floe` to {len(res.floe)} for {len(res.areas)} frames — "
              "every index after a blank frame is SHIFTED, exactly what MATLAB's `x(k) = []` does in the "
              "reachable case.  The `components` column is therefore dropped below.")
    print(f"\n k   components   max floe area [px]   (Fig. 9.18's y value)")
    for k in range(len(res.floe)):
        comp = f"{res.areas[k].size:10d}" if aligned else f"{'--':>10s}"
        print(f"{k + 1:3d}   {comp}   {res.floe[k]:18.0f}")
    print(f"\nfloe: {len(res.floe)} values, range {np.nanmin(res.floe):.0f} .. {np.nanmax(res.floe):.0f} px, "
          f"mean {np.nanmean(res.floe):.1f}")
    print(f"the maximum is NOT monotone ({int((np.diff(res.floe) < 0).sum())} decreases) and the component that "
          "attains it changes — a monotone series could not distinguish `max` from `last`")
    if res.empty_frames.size:
        print(f"blank frames: {res.empty_frames.tolist()} (handled as --empty {args.empty}; "
              "MATLAB R2025a would error — risk R13 corrected)")

    written = [save_image(out / "fig_9_17_a_segmented_frame.png", im2bw(frames[:, :, :, 0])),
               save_image(out / "fig_9_17_b_labels.png", label2rgb(res.labels[0]))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(res.floe) + 1), res.floe, "-o", ms=3)
    ax.set_xlabel("Time")
    ax.set_ylabel("maximum floe size [px]")
    ax.set_title("Fig. 9.18 — maximum floe area per frame" + (" [synthetic]" if synthetic else ""))
    written.append(finish_figure(fig, out / "fig_9_18_max_floe_size.png", args.show))

    mid = min(len(res.labels) - 1, N // 2)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    imshow_matlab(axes[0], frames[:, :, :, 0], title="mov(1).cdata (already segmented)")
    imshow_matlab(axes[1], bwareaopen(im2bw(frames[:, :, :, 0]), args.min_area, args.conn),
                  title=f"bwareaopen(im, {args.min_area}, {args.conn})")
    imshow_matlab(axes[2], label2rgb(res.labels[mid]), title=f"bwlabel(.., {args.conn}) of frame {mid + 1}")
    fig.suptitle("movie_floe.m — Fig. 9.17: the per-frame chain behind Fig. 9.18")
    written.append(finish_figure(fig, out / "fig_9_17_movie_floe_chain.png", args.show))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
