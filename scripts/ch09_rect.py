"""Port of ``MATLAB_ROOT/ch9/Model_Ice_Floe_Identification/rect.m`` — §9.3.2.1 floe rectangularization.

``rect.m`` is a **function**, not a script: the book never prints a driver for it (``model_ice_demo.m`` line 54
is commented out).  This driver exercises it on the segmented shipped image *and* on the constructed fixtures the
analysis's test-design notes call for, so every branch of ``minboundrect`` and every trap of the port is visible:

* a **non-square** component — a square fixture cannot catch the ``(c, r)`` vs ``(r, c)`` argument order of
  line 52 (a transposition there silently swaps every rectangle's length and width; ch08 review lesson);
* a **single pixel** and a **straight 1-px line** — ``minboundrect``'s ``nedges in {0, 1, 2}`` special cases;
* a **rotated** rectangle — where the minimum-area rectangle is not the axis-aligned bounding box;
* components with an **L/W ratio below 1** — without them the ``k > k1`` half of ``model_ice_model``'s symmetric
  band is never exercised.

Erratum **E8**: ``rect.m`` line 32 tests ``nargin < 3`` inside a **two**-argument function, so ``metric`` is dead
code — only ``'a'`` is reachable and line 52 hard-codes it anyway.  The port keeps and validates the argument,
then ignores it exactly as MATLAB does.

Usage: ``python scripts/ch09_rect.py [--image model_ice.jpg] [--metric a] [--data data/book/ch09]
[--out outputs/ch09] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch09_model_ice import BOOK_PARAMS_CH9, model_ice_demo, rect  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.connectivity import label_components  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402

CH = "ch09"


def fixtures() -> dict[str, np.ndarray]:
    """The constructed masks of the test-design notes (all in ONE non-square image, 40 x 64)."""
    out: dict[str, np.ndarray] = {}
    a = np.zeros((40, 64), dtype=bool)
    a[4:9, 6:26] = True                       # non-square axis-aligned: 5 rows x 20 cols -> L/W ~ 4.75
    out["non_square_20x5"] = a
    b = np.zeros((40, 64), dtype=bool)
    b[20, 30] = True                          # a single pixel (nedges == 0/1)
    out["single_pixel"] = b
    c = np.zeros((40, 64), dtype=bool)
    c[12, 8:30] = True                        # a straight 1-px line (nedges == 2)
    out["line_1px"] = c
    d = np.zeros((40, 64), dtype=bool)
    rr, cc = np.mgrid[0:40, 0:64]
    ang = np.deg2rad(30.0)
    u = (cc - 32) * np.cos(ang) + (rr - 20) * np.sin(ang)
    v = -(cc - 32) * np.sin(ang) + (rr - 20) * np.cos(ang)
    d[(np.abs(u) <= 12) & (np.abs(v) <= 5)] = True
    out["rotated_30deg_24x10"] = d
    e = np.zeros((40, 64), dtype=bool)
    e[6:26, 40:45] = True                     # tall: the FIRST minboundrect side is the SHORT one -> k < 1
    out["tall_5x20"] = e
    return out


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="model_ice.jpg", help="the image to segment before rectangularizing")
    p.add_argument("--metric", default="a", help="rect.m's dead `metric` argument (erratum E8)")
    p.add_argument("--max-seeds", type=int, default=None)
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    print("=== rect.m on constructed fixtures (a 40 x 64 NON-SQUARE canvas; a square one could not catch the "
          "(c, r) argument order of line 52) ===")
    print(f"{'fixture':24s} {'px':>5s} {'rect area':>10s} {'perimeter':>10s} {'side1':>7s} {'side2':>7s} "
          f"{'k = s1/s2':>10s}  center")
    for name, mask in fixtures().items():
        S = rect(mask, args.metric)
        for s in S:
            v = s.Vertices
            s1 = float(np.hypot(v[0, 0] - v[1, 0], v[0, 1] - v[1, 1]))
            s2 = float(np.hypot(v[2, 0] - v[1, 0], v[2, 1] - v[1, 1]))
            k = s1 / s2 if s2 else float("inf")
            print(f"{name:24s} {int(mask.sum()):5d} {s.Area:10.4f} {s.Perimeter:10.4f} {s1:7.3f} {s2:7.3f} "
                  f"{k:10.4f}  ({s.Center[0]:.2f}, {s.Center[1]:.2f})")
    print("note: `tall_5x20` has k = s1/s2 < 1 — model_ice_model's band (0.4, 2.5) is SYMMETRIC because `k` is "
          "not normalised to >= 1 (erratum E7 / risk R9), so both halves need a fixture.")
    print("note: Area/Perimeter are the RECTANGLE's (minboundrect), NOT the pixel count — a different meaning "
          "from ch7's IcePiece.Area and ch8's Floe.Area.")
    print(f"metric = {args.metric!r} was validated and then IGNORED (erratum E8: `if (nargin<3)` in a "
          "2-argument function is always true, so only 'a' is reachable).")

    written = []
    try:
        rgb, source = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"\nSKIP the real-image part: private image absent ({exc})")
        rgb = None
    if rgb is not None:
        d = model_ice_demo(rgb, max_seeds=args.max_seeds)
        S = rect(d.bw4, args.metric)
        lab = label_components(d.bw4, 4)
        print(f"\n=== rect.m on bw4 of {args.image} ({source}) — {lab.max()} 4-connected components ===")
        areas = np.array([s.Area for s in S])
        perims = np.array([s.Perimeter for s in S])
        print(f"{len(S)} rectangles; area {areas.min():.4f} .. {areas.max():.4f}, sum {areas.sum():.4f}; "
              f"perimeter {perims.min():.4f} .. {perims.max():.4f}")
        print(f"sum(rect area) / (M*N) = {areas.sum() / d.bw4.size * 100:.2f} % "
              f"(vs {100 * d.bw4.mean():.2f} % of segmented pixels — rectangles over-cover, which is the "
              "sec. 9.3.2.1 point)")
        print(" i   area      perimeter   center (x, y)")
        for i, s in enumerate(S[:10], 1):
            print(f"{i:3d}  {s.Area:9.3f}  {s.Perimeter:9.3f}   ({s.Center[0]:7.2f}, {s.Center[1]:7.2f})")
        if len(S) > 10:
            print(f"  ... {len(S) - 10} more")

        fig, ax = plt.subplots(figsize=(5, 11))
        imshow_matlab(ax, d.bw4, title=f"Fig. 9.15(a) — {len(S)} minimum-area bounding rectangles")
        for s in S:
            ax.plot(s.Vertices[:, 0] - 1, s.Vertices[:, 1] - 1, "b-", lw=0.9)
            ax.plot(s.Center[0] - 1, s.Center[1] - 1, "r+", ms=6)
        written.append(finish_figure(fig, out / "fig_9_15_a_rectangles.png", args.show))
        written.append(save_image(out / "fig_9_15_a_bw4.png", d.bw4))

    fig, axes = plt.subplots(1, 5, figsize=(18, 3))
    for ax, (name, mask) in zip(axes, fixtures().items()):
        imshow_matlab(ax, mask, title=name)
        for s in rect(mask, args.metric):
            ax.plot(s.Vertices[:, 0] - 1, s.Vertices[:, 1] - 1, "b-", lw=1.2)
            ax.plot(s.Center[0] - 1, s.Center[1] - 1, "r+", ms=7)
    fig.suptitle("rect.m — minimum-area bounding rectangles of the constructed fixtures")
    written.append(finish_figure(fig, out / "sec_9_3_2_1_rect_fixtures.png", args.show))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    _ = BOOK_PARAMS_CH9
    return 0


if __name__ == "__main__":
    sys.exit(main())
