"""Port of ``MATLAB_ROOT/ch{6,7}/Sea_Ice_Floe_Identification/sea_ice_model.m`` — section 8.2.1, Figures 8.12/8.13.

Every ice floe is replaced by its **convex hull** ("bounding minimum-area polygon", p. 184) and every brash piece
by an **area-equivalent disk** ``r = sqrt(A/pi)``; each modelled piece then stores an *overlap flag* listing the
serial numbers of the floes and brash pieces whose boundary it crosses.

Two input sources:

* ``--source demo`` (default) — the section 8.2 case study: Figure 8.8 is ch6/ch7's ``sea_ice_test.jpg``
  (analysis finding 5), run through Algorithms 3/4/5 ("carried out directly", p. 182) and then through
  ``sea_ice_model``.  This is what Figures 8.12(a)(b) and 8.13 show.
* ``--source iceimage`` — re-run ``sea_ice_model`` on the authors' **own** stored pieces (the section 8.3 field,
  2888 floes + 3452 brash) and compare every output against the shipped ``Polygon``/``Circle``/``Intersect``
  sub-structures.  That is the strongest available reference for this file.

The double loop is ``O(N^2)`` with ``polyxpoly`` inside (2888^2 + 2*2888*3452 = 28.3 M pairs for the section 8.3
field).  ``--brute-force`` disables the bounding-box prefilter and runs the M-file's literal loop; the prefilter
is provably equivalent (see :func:`seaice.ch08_applications.sea_ice_model`).

Usage: ``python scripts/ch08_sea_ice_model.py [--source demo|iceimage] [--brute-force] [--no-raster]
[--strict-containment] [--intersect-impl fast|polyxpoly] [--limit N] [--data data/book/ch08]
[--out outputs/ch08] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch08_applications import iceimage_to_pieces, sea_ice_field, sea_ice_model  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.icestruct import load_iceimage_mat  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure, save_image  # noqa: E402

CH = "ch08"


def _load_fig88(data: Path, name: str):
    """Figure 8.8 = ch7's ``sea_ice_test.jpg`` (NCC +0.982 vs the inverted transpose; analysis finding 5).

    ch8 ships no image of its own, so the private copy is looked for under ``data/book/ch08`` first (in case the
    reader put one there) and then under ``data/book/ch07`` — never a third copy, and never a fallback
    substitute (scripts use ``allow_fallback=False``).
    """
    for chapter, folder in (("ch08", data), ("ch07", None)):
        try:
            return load_image(chapter, name, allow_fallback=False, data_dir=folder, verbose=False), chapter
        except FileNotFoundError:
            continue
    return None, None


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--source", default="demo", choices=["demo", "iceimage"])
    p.add_argument("--image", default="sea_ice_test.jpg", help="Figure 8.8 (= ch6/ch7's sea_ice_test.jpg)")
    p.add_argument("--mat", default="IceImage_290915_2_jpg.0000179.mat")
    p.add_argument("--brute-force", action="store_true", help="disable the AABB prefilter (the literal loop)")
    p.add_argument("--no-raster", action="store_true", help="skip bw_floe / bw_brash")
    p.add_argument("--raster-full", action="store_true", help="rasterise on the whole image (literal roipoly)")
    p.add_argument("--strict-containment", action="store_true",
                   help="also flag containment without a boundary crossing (NOT what the M-file does)")
    p.add_argument("--intersect-impl", default="fast", choices=["fast", "polyxpoly"])
    p.add_argument("--limit", type=int, default=0, help="use only the first N floes and N brash pieces")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    if args.source == "iceimage":
        hits = sorted(data.rglob(args.mat))
        if not hits:
            print(f"SKIP {Path(__file__).name}: private data absent ({args.mat} not found under "
                  f"{data.as_posix()})")
            return 0
        ice = load_iceimage_mat(hits[0])
        ice_floe, brash_ice = iceimage_to_pieces(ice)
        img = np.zeros((ice.Param.NumPix_y, ice.Param.NumPix_x))
        label = f"the shipped IceImage ({hits[0].name})"
        enh = None
    else:
        loaded, chapter = _load_fig88(data, args.image)
        if loaded is None:
            print(f"SKIP {Path(__file__).name}: private image absent ({args.image}; Figure 8.8 is ch6/ch7's "
                  "sea_ice_test.jpg, which ships with the book's MATLAB archive)")
            return 0
        (rgb, _), chapter = loaded, chapter
        print(f"Figure 8.8 = {args.image} from data/book/{chapter} ({rgb.shape[0]}x{rgb.shape[1]}); "
              "the book prints it transposed and inverted (= Figure 7.22)")
        t0 = time.time()
        enh = sea_ice_field(rgb, cache=out / f"stage3_{Path(args.image).stem}_{rgb.shape[0]}x{rgb.shape[1]}.npz")
        print(f"Algorithms 3+4+5 (section 8.2, p. 182): {time.time() - t0:.1f} s -> "
              f"{len(enh.ice_floe)} ice floes, {len(enh.brash_ice)} brash pieces")
        cov = enh.coverage.as_percent()
        print(f"  coverage {cov['IceFloe']:.2f} / {cov['BrashIce']:.2f} / {cov['Slush']:.2f} / "
              f"{cov['Water']:.2f} %  [book section 8.2: 498 floes, 201 brash, "
              "76.73 / 0.46 / 9.05 / 13.76 % - the authors' parameter set is NOT printed, so these are "
              "expected to differ (risk R7); ch07 measured 433 / 274 on the same image]")
        ice_floe, brash_ice = enh.ice_floe, enh.brash_ice
        img = enh.index_floe
        label = f"the section 8.2 pipeline on {args.image}"

    if args.limit:
        ice_floe = ice_floe[:args.limit]
        brash_ice = brash_ice[:args.limit]
        print(f"NOTE: --limit {args.limit} -> {len(ice_floe)} floes / {len(brash_ice)} brash pieces")

    print(f"=== sea_ice_model.m on {label}: {len(ice_floe)} floes, {len(brash_ice)} brash pieces ===")
    t0 = time.time()
    m = sea_ice_model(ice_floe, brash_ice, img,
                      prefilter=not args.brute_force, raster=not args.no_raster,
                      raster_crop=not args.raster_full, strict_containment=args.strict_containment,
                      intersect_impl=args.intersect_impl)
    dt = time.time() - t0
    print(f"  {dt:.1f} s; crossing tests actually run: {m.n_pairs_tested} of {m.n_pairs_total} "
          f"({'brute force' if args.brute_force else 'AABB prefilter'}, impl = {args.intersect_impl})")
    areas = np.array([f.Area for f in m.floe])
    raw = np.array([int(f.Area) for f in ice_floe], dtype=np.float64)
    print(f"  polygon areas {areas.min():.1f} .. {areas.max():.1f} px^2 (raw pixel counts "
          f"{raw.min():.0f} .. {raw.max():.0f}); {int((areas < raw).sum())} of {areas.size} polygons have a "
          "SMALLER continuous area than the pixel count they enclose (the raster footprint is still a superset)")
    print(f"  brash radii {min(b.Radius for b in m.brash):.3f} .. {max(b.Radius for b in m.brash):.3f} px; "
          f"t = 0:0.05:6.28 has 126 angles, last = 6.25 rad (the circle does NOT close)")
    n_ff = sum(b.Intersect.floe.size for b in m.floe)
    n_fb = sum(b.Intersect.brash.size for b in m.floe)
    n_bb = sum(b.Intersect.brash.size for b in m.brash)
    n_bf = sum(b.Intersect.floe.size for b in m.brash)
    print(f"  overlap flags: floe-floe {n_ff}, floe-brash {n_fb}, brash-brash {n_bb}, brash-floe {n_bf}")
    if not args.no_raster:
        print(f"  bw_floe {int(m.bw_floe.sum())} px ({100 * m.bw_floe.mean():.2f} %), "
              f"bw_brash {int(m.bw_brash.sum())} px ({100 * m.bw_brash.mean():.2f} %)")

    if args.source == "iceimage" and not args.limit:
        ok_a = max(abs(a.Area - b.Polygon.Area) for a, b in zip(m.floe, ice.Floe))
        ok_c = max(float(np.abs(a.Center - b.Polygon.Center).max()) for a, b in zip(m.floe, ice.Floe))
        ok_p = max(abs(a.Perimeter - b.Polygon.Perimeter) for a, b in zip(m.floe, ice.Floe))
        vsets = sum(set(map(tuple, a.Vertices[:-1])) == set(map(tuple, np.asarray(b.Polygon.Vertices, float)))
                    for a, b in zip(m.floe, ice.Floe))
        dr = max(abs(a.Radius - b.Circle.Radius) for a, b in zip(m.brash, ice.Brash))
        dp = max(abs(a.Perimeter - b.Circle.Perimeter) for a, b in zip(m.brash, ice.Brash))
        eq = lambda a, b: set(np.asarray(a).tolist()) == set(np.asarray(b).tolist())  # noqa: E731
        s_ff = sum(eq(a.Intersect.floe, b.Polygon.Intersect.floe) for a, b in zip(m.floe, ice.Floe))
        s_fb = sum(eq(a.Intersect.brash, b.Polygon.Intersect.brash) for a, b in zip(m.floe, ice.Floe))
        s_bb = sum(eq(a.Intersect.brash, b.Circle.Intersect.brash) for a, b in zip(m.brash, ice.Brash))
        s_bf = sum(eq(a.Intersect.floe, b.Circle.Intersect.floe) for a, b in zip(m.brash, ice.Brash))
        print("  --- against the authors' own stored structure (L3) ---")
        print(f"    Polygon.Area / Center / Perimeter: max |delta| {ok_a:.3e} / {ok_c:.3e} / {ok_p:.3e}")
        print(f"    Polygon.Vertices as SETS: {vsets} / {len(m.floe)} identical "
              "(MATLAB's convhull keeps collinear points and starts elsewhere, so order is not compared)")
        print(f"    Circle.Radius / Perimeter: max |delta| {dr:.3e} / {dp:.3e}")
        print(f"    Intersect sets: floe.floe {s_ff}/{len(m.floe)}, floe.brash {s_fb}/{len(m.floe)}, "
              f"brash.brash {s_bb}/{len(m.brash)}, brash.floe {s_bf}/{len(m.brash)}")

    # Figures 8.12/8.13 belong to section 8.2 (the `demo` source); the `iceimage` source is the section 8.3
    # field, whose model figures the book never prints -> `sec_8_3_*`.
    pre = "fig_8_12" if args.source == "demo" else "sec_8_3_model"
    pre13 = "fig_8_13_model_closeup" if args.source == "demo" else "sec_8_3_model_closeup"
    # The book prints the section 8.2 image (Figure 8.8 = Figure 7.22) TRANSPOSED, so Figures 8.12/8.13 are shown
    # in that orientation: the raster is transposed and the plotted (x, y) are swapped with it.
    tp = (args.source == "demo")
    XY = (lambda x, y: (y, x)) if tp else (lambda x, y: (x, y))

    written = []
    if not args.no_raster:
        # ---- Figure 8.12(a)(b) ---------------------------------------------------------------------------------
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        axes[0].imshow(m.bw_floe.T if tp else m.bw_floe, cmap="gray", vmin=0, vmax=1)
        for f in m.floe:
            axes[0].plot(*XY(f.Vertices[:, 0] - 1, f.Vertices[:, 1] - 1), linewidth=0.4)
            axes[0].plot(*XY(f.Center[0] - 1, f.Center[1] - 1), "r+", markersize=2)
        axes[0].set_axis_off()
        axes[0].set_title("(a) polygonized ice floes (convex hulls) with their centres")
        axes[1].imshow(m.bw_brash.T if tp else m.bw_brash, cmap="gray", vmin=0, vmax=1)
        for b in m.brash:
            axes[1].plot(*XY(b.Center[0] - 1, b.Center[1] - 1), "r.", markersize=1)
        axes[1].set_axis_off()
        axes[1].set_title("(b) circularized brash ice (area-equivalent disks)")
        fig.suptitle(("Figure 8.12 - " if args.source == "demo" else "Section 8.3 field - ")
                     + "the sea ice numerical model of section 8.2.1")
        written.append(finish_figure(fig, out / f"{pre}_sea_ice_model.png", args.show))
        written.append(save_image(out / f"{pre}a_polygonized_floes.png", m.bw_floe.T if tp else m.bw_floe))
        written.append(save_image(out / f"{pre}b_circularized_brash.png", m.bw_brash.T if tp else m.bw_brash))

    # ---- Figure 8.13: close-up (the book does not say which region; the densest 200x200 box is used) -----------
    H, W = np.asarray(img).shape[:2]
    cx = np.array([f.Center[0] for f in m.floe])
    cy = np.array([f.Center[1] for f in m.floe])
    if cx.size:
        w = min(200, W)
        h = min(200, H)
        x0 = int(np.clip(np.median(cx) - w / 2, 0, max(W - w, 0)))
        y0 = int(np.clip(np.median(cy) - h / 2, 0, max(H - h, 0)))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(x0, x0 + w)
        ax.set_ylim(y0 + h, y0)
        ax.set_aspect("equal")
        for i, f in enumerate(m.floe):
            if x0 - 50 <= f.Center[0] <= x0 + w + 50 and y0 - 50 <= f.Center[1] <= y0 + h + 50:
                ax.plot(f.Vertices[:, 0] - 1, f.Vertices[:, 1] - 1, "b-", linewidth=1)
                ax.plot(ice_floe[i].Center.ravel()[0] - 1, ice_floe[i].Center.ravel()[1] - 1, "k*", markersize=6)
                ax.plot(f.Center[0] - 1, f.Center[1] - 1, "r+", markersize=8)
        t = np.linspace(0, 2 * np.pi, 60)
        for i, b in enumerate(m.brash):
            if x0 - 50 <= b.Center[0] <= x0 + w + 50 and y0 - 50 <= b.Center[1] <= y0 + h + 50:
                ax.plot(b.Center[0] - 1 + b.Radius * np.cos(t), b.Center[1] - 1 + b.Radius * np.sin(t),
                        "g-", linewidth=0.8)
                ax.plot(b.Center[0] - 1, b.Center[1] - 1, "r.", markersize=4)
        ax.set_title(f"{'Figure 8.13' if args.source == 'demo' else 'Model close-up'} - close-up at rows {y0}..{y0 + h}, cols {x0}..{x0 + w} "
                     "(black * / red + = identified vs modelled centres)")
        written.append(finish_figure(fig, out / f"{pre13}.png", args.show))

    print("\nfigures written:")
    for w_ in written:
        print("  ", w_)
    return 0


if __name__ == "__main__":
    sys.exit(main())
