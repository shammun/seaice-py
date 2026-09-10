"""Port of ``ch7/Sea_Ice_Floe_Identification/ice_shape_enhancement.m`` — Book §7.1.4/§7.2.3/§7.2.4.

**Algorithm 2** (ice shape enhancement, p. 153), **Algorithm 4** (the same on the merged light+dark labelling,
p. 158) and **Algorithm 5** (sea ice types classification, p. 160), plus **Eq. (7.5)** (the size-adapted disk
radius) and **Eq. (7.6)** (the size-coded colour).

The script demonstrates the three things the M-file's own figures cannot show:

1. **why the pieces must be sorted small -> large** (p. 151) — the same input processed in both orders;
2. the **Eq. (7.5)** radius rule and the **Eq. (7.6)** colour map together with the colour-bar tick arithmetic
   that produces the book's printed integers (Fig. 7.13: 3, 131, 277, 448, 656, 917, 1273);
3. the enhancement of the real segmentation of ``sea_ice_test.jpg`` (``--source book``; needs the private image,
   and re-uses the Algorithm-3 cache written by ``scripts/ch07_sea_ice_demo.py``).

``--source synthetic`` (the default) needs no book data at all.

Usage: ``python scripts/ch07_ice_shape_enhancement.py [--source synthetic|book] [--min-floe 40]
[--min-brash 1] [--se-th 50] [--book-threshold] [--data data/book/ch07] [--out outputs/ch07] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch06_gvf_snake import BOOK_PARAMS, seaice_kmean_gvf  # noqa: E402
from seaice.ch07_ice_type import (adaptive_se_radius, color_to_area, colorbar_area_ticks,  # noqa: E402
                                  ice_shape_enhancement, size_color)
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, label2rgb, save_image  # noqa: E402

CH = "ch07"
P = BOOK_PARAMS["sea_ice_demo"]

#: The seven colour-bar tick integers printed under Fig. 7.13 (p. 159) — a pure-arithmetic book truth.
FIG_7_13_TICKS = (3, 131, 277, 448, 656, 917, 1273)


def synthetic_segmentation(shape: tuple[int, int] = (120, 200), seed: int = 0):
    """A three-level ``seg`` (1 = light ice, 0.5 = dark ice, 0 = water) plus the k-means ``bk``.

    Built so that every branch of ``ice_shape_enhancement.m`` is exercised: a large floe with a hole that
    contains a smaller piece (the reason Algorithm 2 sorts small -> large), pieces touching the image border,
    a dark piece overlapping a light one (line 54 ``k = k - k.*bw``), a 1-pixel piece (dropped by the code's
    strict ``> min_brash``) and pieces straddling the Eq. (7.5) ``se_th`` boundary.
    """
    rng = np.random.default_rng(seed)
    seg = np.zeros(shape)
    seg[10:60, 10:70] = 1.0          # a big floe ...
    seg[25:40, 25:45] = 0.0          # ... with a hole ...
    seg[30:34, 30:36] = 1.0          # ... containing a smaller piece
    seg[0:12, 150:180] = 1.0         # touches the top border
    seg[100:120, 0:15] = 1.0         # touches the left/bottom border
    seg[60:90, 90:140] = 0.5         # dark ice ...
    seg[70:76, 100:110] = 1.0        # ... overlapping a light piece
    seg[95, 60] = 1.0                # a 1-pixel piece
    seg[95:100, 70:78] = 1.0         # 40 px exactly (T_floe boundary)
    seg[105:110, 90:100] = 1.0       # 50 px exactly (Eq. (7.5) se_th boundary)
    seg[20, 80] = 1.0
    seg[22, 82] = 1.0
    bk = (seg > 0).astype(np.float64)
    bk[rng.random(shape) > 0.94] = 1.0   # k-means finds some ice the snakes never segmented -> slush
    return seg, bk


def book_segmentation(args, data: Path, out: Path):
    """``seg``/``bk`` of the real image: the Algorithm-3 cache of ``ch07_sea_ice_demo.py``, or a fresh run."""
    rgb, label = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    cache = out / f"stage3_{Path(args.image).stem}_{rgb.shape[0]}x{rgb.shape[1]}_{args.num}_{args.iter}.npz"
    if cache.exists():
        z = np.load(cache)
        print(f"Algorithm 3: reusing {cache.name}")
        return z["seg"], z["bk"], rgb, label
    t0 = time.time()
    res = seaice_kmean_gvf(rgb, kms0=P["kms0"], sigma=P["sigma"], GradientOn=P["GradientOn"], GVFOn=P["GVFOn"],
                           Num=args.num, mu=P["mu"], iter=args.iter, alpha=P["alpha"], beta=P["beta"],
                           gamma=P["gamma"], kappa=P["kappa"], Dmin=P["Dmin"], Dmax=P["Dmax"],
                           Ra_min=P["Ra_min"], Ra=P["Ra"], Rc=P["Rc"], Rl=P["Rl"], se_radius=P["se_radius"],
                           timer=P["timer"], keep_history=False)
    print(f"Algorithm 3 (ch6 seaice_kmean_gvf): {time.time() - t0:.1f} s")
    np.savez_compressed(cache, seg=res.out, bk=res.bk)
    return res.out, res.bk, rgb, label


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--source", default="synthetic", choices=["synthetic", "book"])
    p.add_argument("--image", default="sea_ice_test.jpg")
    p.add_argument("--num", type=int, default=P["Num"])
    p.add_argument("--iter", type=int, default=P["iter"])
    p.add_argument("--se-th", type=float, default=P["se_th"], help="Eq. (7.5) size_th (script: 50)")
    p.add_argument("--min-floe", type=float, default=P["min_floe"], help="Algorithm 5 T_floe (script: 40)")
    p.add_argument("--min-brash", type=float, default=P["min_brash"], help="smallest brash area (script: 1)")
    p.add_argument("--nbins", type=int, default=50)
    p.add_argument("--book-threshold", action="store_true",
                   help="Algorithm 5's printed '>=' instead of the code's '>'")
    p.add_argument("--no-crop", action="store_true", help="full-image scratch arrays, like the M-file")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    rgb = None
    if args.source == "book":
        try:
            seg, bk, rgb, label = book_segmentation(args, data, out)
        except FileNotFoundError as exc:
            print(f"SKIP {Path(__file__).name}: private image absent ({exc})")
            return 0
        title = f"{args.image} [{label}]"
    else:
        seg, bk = synthetic_segmentation()
        title = "synthetic three-level segmentation (Tier-3 fixture)"

    print(f"=== ice_shape_enhancement.m (Algorithms 2/4/5, Eqs. (7.5)/(7.6)) on {title} ===")
    print(f"seg {seg.shape[0]}x{seg.shape[1]}, levels {np.unique(seg).tolist()}: "
          f"light {int((seg == 1).sum())} px, dark {int((seg == 0.5).sum())} px, water {int((seg == 0).sum())} px")

    # ---- Eq. (7.5) and Eq. (7.6), before any image processing --------------------------------------------
    print(f"\nEq. (7.5) adaptive disk radius (size_th = {args.se_th:g}):")
    probes = [1, int(args.se_th) - 1, int(args.se_th), int(args.se_th) + 1, 500]
    print("   area   ", "  ".join(f"{a:>6d}" for a in probes))
    print("   radius ", "  ".join(f"{adaptive_se_radius(a, args.se_th):>6d}" for a in probes))
    print("Eq. (7.6) size colour, C1 = 10000, C2 = 1000:")
    areas = np.array([1, 10, 100, 500, 1000, 3000, 10000])
    print("   area   ", "  ".join(f"{a:>6d}" for a in areas))
    print("   colour ", "  ".join(f"{c:>6d}" for c in size_color(areas)))
    print("   back   ", "  ".join(f"{a:>6d}" for a in color_to_area(size_color(areas))))
    got = colorbar_area_ticks(np.array([29.0, 7200.0]), 6)[1]
    print(f"colour-bar ticks for colour values 29..7200 with n = 6: {got.tolist()}")
    print(f"   book Fig. 7.13 prints                              : {list(FIG_7_13_TICKS)}  "
          f"-> {'identical' if tuple(got.tolist()) == FIG_7_13_TICKS else 'DIFFERENT'}")

    # ---- Algorithms 2/4/5 -------------------------------------------------------------------------------
    t0 = time.time()
    r = ice_shape_enhancement(bk, seg, min_floe=args.min_floe, min_brash=args.min_brash, se_th=args.se_th,
                              nbins=args.nbins, crop=not args.no_crop, book_threshold=args.book_threshold)
    dt = time.time() - t0
    print(f"\nAlgorithm 2 loop ({dt:.2f} s, crop = {not args.no_crop}):")
    print(f"  {r.nn_bw} light + {r.nn_k} dark pieces; ice_area = {r.ice_area.tolist() if r.ice_area.size <= 20 else str(r.ice_area.size) + ' values'}")
    if r.ice_area.size <= 20:
        print(f"  sort order (0-based, stable) = {r.order.tolist()} -> areas {r.ice_area[r.order].tolist()}")
    print(f"  t = max(out) = {r.t}; fill (holes only) {int((r.fill != 0).sum())} px, "
          f"out (cleaned) {int((r.out != 0).sum())} px")
    print(f"  ice floes {len(r.ice_floe)} (areas {[q.Area for q in r.ice_floe][:12]}{' ...' if len(r.ice_floe) > 12 else ''})")
    print(f"  brash ice {len(r.brash_ice)} (areas {[q.Area for q in r.brash_ice][:12]}{' ...' if len(r.brash_ice) > 12 else ''})")
    cov = r.coverage.as_percent()
    print(f"  coverage: {cov['IceFloe']:.2f} % floe, {cov['BrashIce']:.2f} % brash, {cov['Slush']:.2f} % slush, "
          f"{cov['Water']:.2f} % water (sum {sum(cov.values()):.2f} %)")
    if r.color_floe.size or r.color_brash.size:
        vals, ticks = colorbar_area_ticks(np.concatenate([r.color_floe, r.color_brash]), 6)
        print(f"  Eq. (7.6) colour-bar ticks (n = 6): {ticks.tolist()}")

    # ---- why the sort matters (p. 151) -------------------------------------------------------------------
    big_first = _large_to_small(seg, args)
    n_asc, a_asc = _effective(r.out)
    n_desc, a_desc = _effective(big_first)
    bnd_asc, bnd_desc = _internal_boundary(r.out), _internal_boundary(big_first)
    diff = int((bnd_asc != bnd_desc).sum())
    print("\n'the arrangement of ice pieces in order of increasing size is required' (p. 151):")
    print(f"  small -> large : t = {r.t}, {n_asc} non-empty labels ({r.t - n_asc} overwritten by a larger "
          f"piece), {int((r.out != 0).sum())} ice px, largest piece {int(a_asc.max()) if a_asc.size else 0} px")
    print(f"  large -> small : t = {int(big_first.max())}, {n_desc} non-empty labels, "
          f"{int((big_first != 0).sum())} ice px, largest piece {int(a_desc.max()) if a_desc.size else 0} px")
    print(f"  the two partitions disagree on {diff} internal-boundary pixels — with the wrong order a small "
          "piece inside a larger floe survives as a separate identification and cuts the floe into a ring")

    if args.source == "book":
        print("\nNOTE: the book's 154/189 pieces and 60.52/3.34/16.03/20.11 % belong to Fig. 7.10(a), which is "
              "not shipped — they are not reproducible here.")

    # ---- figures ------------------------------------------------------------------------------------------
    tag = "book" if args.source == "book" else "synthetic"
    written = []
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    imshow_matlab(axes[0, 0], seg, autoscale=True, title="SEGMENTATION (1 = light, 0.5 = dark, 0 = water)")
    imshow_matlab(axes[0, 1], r.fill != 0, title="after hole filling (Algorithm 2 step 4, first imfill)")
    axes[1, 0].imshow(label2rgb(r.out, cmap="jet", background=(1, 1, 1), shuffle=True))
    axes[1, 0].set_title(f"IDENTIFICATION — out, {r.t} labelled pieces")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(label2rgb(r.index, cmap="jet", background=(1, 1, 1), shuffle=False))
    if r.floe_cen.size:
        axes[1, 1].plot(r.floe_cen[:, 0] - 1, r.floe_cen[:, 1] - 1, "k*", markersize=5)
    if r.brash_cen.size:
        axes[1, 1].plot(r.brash_cen[:, 0] - 1, r.brash_cen[:, 1] - 1, "k.", markersize=3)
    axes[1, 1].set_title("Eq. (7.6) size-coded colours + piece positions")
    axes[1, 1].axis("off")
    fig.suptitle("Algorithms 2/4 — ice shape enhancement")
    written.append(finish_figure(fig, out / f"sec_7_1_4_enhancement_{tag}.png", args.show))

    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    for ax, layer, t in ((axes[0, 0], r.index_floe != 0, "(a) ice floes"),
                         (axes[0, 1], r.index_brash != 0, "(b) brash ice"),
                         (axes[1, 0], r.index_slush != 0, "(c) slush"),
                         (axes[1, 1], r.index_water != 0, "(d) water")):
        imshow_matlab(ax, layer, title=t)
    fig.suptitle("Algorithm 5 — the four sea ice type layers")
    written.append(finish_figure(fig, out / f"sec_7_2_4_types_{tag}.png", args.show))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    a = np.arange(0, 8001)
    axes[0].plot(a, size_color(a))
    for th in (args.min_brash, args.min_floe, args.se_th):
        axes[0].axvline(th, color="r", ls=":", lw=0.8)
    axes[0].set_xlabel("piece area (pixels)")
    axes[0].set_ylabel("Color(p)")
    axes[0].set_title("Eq. (7.6)  Color = 10000 (1 - exp(-area/1000)); red = min_brash / min_floe / se_th")
    vals, ticks = colorbar_area_ticks(np.array([29.0, 7200.0]), 6)
    axes[1].plot(vals, ticks, "o-")
    for v, tk in zip(vals, ticks):
        axes[1].annotate(str(tk), (v, tk), textcoords="offset points", xytext=(4, -10), fontsize=8)
    axes[1].set_xlabel("colour value (ysh)")
    axes[1].set_ylabel("-round(1000 log(1 - ysh/10000))")
    axes[1].set_title(f"colour-bar ticks, n = 6 — book Fig. 7.13: {list(FIG_7_13_TICKS)}")
    fig.suptitle("Eq. (7.6) and its inverse (the colour-bar labels)")
    written.append(finish_figure(fig, out / "sec_7_2_3_eq_7_6.png", args.show))

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].imshow(label2rgb(r.out, cmap="jet", background=(1, 1, 1), shuffle=True))
    axes[0].set_title(f"small -> large ({n_asc} pieces) — the book's order")
    axes[0].axis("off")
    axes[1].imshow(label2rgb(big_first, cmap="jet", background=(1, 1, 1), shuffle=True))
    axes[1].set_title(f"large -> small ({n_desc} pieces)")
    axes[1].axis("off")
    imshow_matlab(axes[2], bnd_asc != bnd_desc, title=f"internal boundaries that differ ({diff} px)")
    fig.suptitle("Section 7.1.4, p. 151 — 'the smaller ice piece contained in a larger ice floe may not be removed'")
    written.append(finish_figure(fig, out / f"sec_7_1_4_sort_order_{tag}.png", args.show))

    written.append(save_image(out / f"sec_7_1_4_out_{tag}.png",
                              label2rgb(r.out, cmap="jet", background=(1, 1, 1), shuffle=True)))
    written.append(save_image(out / f"sec_7_2_4_residue_{tag}.png", r.index_residue != 0))
    if rgb is not None:
        written.append(save_image(out / "sec_7_2_1_input.png", rgb))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


def _effective(L: np.ndarray) -> tuple[int, np.ndarray]:
    """Number of labels of ``L`` that still have pixels, and their areas.

    A label can end up empty because a later, larger piece overwrote it completely — exactly the situation the
    ``if area0 ~= 0`` guard of ``ice_shape_enhancement.m`` line 125 exists for.
    """
    n = int(L.max()) if L.size else 0
    counts = np.bincount(L.astype(np.int64).ravel(), minlength=n + 1)[1:]
    return int((counts > 0).sum()), counts[counts > 0]


def _internal_boundary(L: np.ndarray) -> np.ndarray:
    """Pixels where two 4-adjacent **ice** pixels carry different labels — the partition's internal boundaries.

    Label *numbers* depend on the processing order, so the two orders can only be compared through an
    order-independent quantity; this is one.
    """
    b = np.zeros(L.shape, dtype=bool)
    ice = L != 0
    b[:, :-1] |= ice[:, :-1] & ice[:, 1:] & (L[:, :-1] != L[:, 1:])
    b[:-1, :] |= ice[:-1, :] & ice[1:, :] & (L[:-1, :] != L[1:, :])
    return b


def _large_to_small(seg: np.ndarray, args) -> np.ndarray:
    """The same Algorithm-2 loop with the pieces taken **large -> small** — the order p. 151 warns against.

    ``ice_shape_enhancement`` always sorts ascending (that is the algorithm), so the counter-example is written
    out here; it calls exactly the same primitives, only the ``argsort`` is reversed.  This is a demonstration,
    not a port: no MATLAB file computes it.
    """
    from seaice.ch07_ice_type import _index_lists, _piece_morphology, adaptive_se_radius as _r
    from seaice.core.connectivity import label_components

    M, N = seg.shape
    bw = (seg == 1).astype(np.float64)
    k = (seg == 0.5).astype(np.float64)
    k = k - k * bw
    lists, areas = [], []
    for src in (bw, k):
        L = label_components(src != 0, 4)
        n = int(L.max())
        for idx in _index_lists(L, n):
            lists.append(idx)
            areas.append(idx.size)
    order = np.argsort(np.asarray(areas), kind="stable")[::-1]      # large -> small
    out = np.zeros((M, N))
    t = 0
    for i in order:
        idx = lists[i]
        rows, cols = idx % M, idx // M
        r0, c0, _, b = _piece_morphology((M, N), rows, cols, _r(idx.size, args.se_th), True)
        L2 = label_components(b != 0, 4)
        for pp in _index_lists(L2, int(L2.max())):
            if pp.size:
                t += 1
                out[pp % b.shape[0] + r0, pp // b.shape[0] + c0] = t
    return out


if __name__ == "__main__":
    sys.exit(main())
