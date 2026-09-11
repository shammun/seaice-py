"""Port of ``MATLAB_ROOT/ch9/Model_Ice_Floe_Identification/model_ice_model.m`` — §9.3.2.1 model-ice floe model.

``model_ice_model.m`` is a **function**, called from the commented block of ``model_ice_demo.m`` (line 66).  For
an ice–structure interaction simulator every floe is modelled as a rectangle = its minimum-area bounding
rectangle, rectangles whose length/width ratio falls outside ``(k1, k2)`` are dropped, the survivors are
rasterised (**Fig. 9.15(b)**) and each one carries a list of the floes it **overlaps**.

This driver:

1. runs the whole §9.3.2/§9.3.2.1 chain on the shipped ``model_ice.jpg`` and prints the record of every model
   floe plus the three ice concentrations the book compares on p. 208 (76.96 / 83.17 / 87.75 %);
2. reproduces the **R2025a ``polybool`` probe** on four constructed rectangle pairs — contained, disjoint,
   partially overlapping and **edge-touching** — because only *emptiness* of ``polybool``'s output is consumed
   and only emptiness is reproducible (``core.polygon.clip_polygon_convex(drop_degenerate=True)``);
3. plots the floe-size distribution of the rectangles against that of the identified floes and their bin-wise
   difference (**Fig. 9.16(a)(b)**).

Errata reproduced: **E7/R9** — ``k = |v1-v2| / |v3-v2|`` is *not* normalised to ``>= 1``, so the shipped
``(0.4, 2.5)`` is a **symmetric band**, not the one-sided filter p. 207 describes.  **E9** — ``if xx ~= NaN`` is
``if ~isempty(xx)``; unlike ch08's ``polyxpoly`` version this *does* detect containment.

Usage: ``python scripts/ch09_model_ice_model.py [--image model_ice.jpg] [--k1 0.4] [--k2 2.5]
[--data data/book/ch09] [--out outputs/ch09] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch09_model_ice import (BOOK_PARAMS_CH9, floe_size_histogram, fsd_error,  # noqa: E402
                                   model_ice_demo, model_ice_model, rect, rect_ice_concentration)
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.connectivity import label_components  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402
from seaice.core.polygon import clip_polygon_convex  # noqa: E402
from seaice.core.regionprops import regionprops  # noqa: E402
from seaice.core.threshold import ice_concentration  # noqa: E402

CH = "ch09"
P = BOOK_PARAMS_CH9


def _rect_ring(x0: float, y0: float, x1: float, y1: float) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([x0, x1, x1, x0, x0], float), np.array([y0, y0, y1, y1, y0], float))


def polybool_probe() -> None:
    """The four cases the R2025a probe pins (analysis/ch09.md §5.2) — reproduced by the Python clip."""
    base = _rect_ring(0, 0, 10, 10)
    cases = {
        "contained  (strictly inside)": (_rect_ring(2, 2, 4, 4), 5, True),
        "disjoint": (_rect_ring(20, 20, 30, 30), 0, False),
        "partially overlapping": (_rect_ring(5, 5, 15, 15), 5, True),
        "edge-touching (shared edge)": (_rect_ring(10, 0, 20, 10), 0, False),
        "vertex-touching (one corner)": (_rect_ring(10, 10, 20, 20), 0, False),
    }
    print("polybool('intersection', ...) — emptiness is the ONLY thing model_ice_model.m consumes "
          "(`if xx ~= NaN` == `~isempty(xx)`):")
    print(f"{'case':32s} {'numel(xx) here':>15s} {'R2025a polybool':>17s} {'overlap flagged':>17s}")
    for name, (other, expected, flagged) in cases.items():
        xx, _yy = clip_polygon_convex(base[0], base[1], other[0], other[1])
        note = str(expected) if name != "vertex-touching (one corner)" else "not probed"
        print(f"{name:32s} {xx.size:15d} {note:>17s} {str(xx.size > 0):>17s}"
              + ("" if (xx.size > 0) == flagged else "   <-- MISMATCH"))
    print("the decisive one is EDGE-TOUCHING: naive Sutherland-Hodgman returns a degenerate zero-area polygon, "
          "which `drop_degenerate=True` discards so the emptiness matches GPC exactly.")


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="model_ice.jpg")
    p.add_argument("--k1", type=float, default=P["k1"], help="model_ice_demo.m line 65 (0.4)")
    p.add_argument("--k2", type=float, default=P["k2"], help="model_ice_demo.m line 65 (2.5)")
    p.add_argument("--bins", type=int, default=10, help="floe-size-distribution bins (Figs. 9.13/9.16)")
    p.add_argument("--max-seeds", type=int, default=None)
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    polybool_probe()

    try:
        rgb, source = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"\nSKIP {Path(__file__).name}: private image absent ({exc}); the polybool probe above needs no data")
        return 0

    d = model_ice_demo(rgb, max_seeds=args.max_seeds)
    S = rect(d.bw4)
    mdl = model_ice_model(S, d.bw4, args.k1, args.k2)
    shape = d.bw4.shape

    print(f"\n=== model_ice_model.m on bw4 of {args.image} ({source}), k1 = {args.k1}, k2 = {args.k2} ===")
    print(f"input: {len(S)} rectangularized floes from rect.m")
    print(f"k = |v1-v2|/|v3-v2| (NOT normalised to >= 1 — erratum E7): "
          f"{mdl.ratios.min():.4f} .. {mdl.ratios.max():.4f}")
    print(f"  {(mdl.ratios <= args.k1).sum()} rejected by k <= k1 (the half a 'ratios all > 1' fixture could "
          f"never exercise), {(mdl.ratios >= args.k2).sum()} rejected by k >= k2")
    print(f"accepted: {mdl.accepted.size} of {len(S)}")
    print("  i   area      perimeter   center (x, y)        k        Intersection (1-based into the accepted list)")
    for i, f in enumerate(mdl.s_model, 1):
        k = mdl.ratios[mdl.accepted[i - 1]]
        print(f"{i:3d}  {f.Area:9.3f}  {f.Perimeter:9.3f}  ({f.Center[0]:7.2f}, {f.Center[1]:7.2f})  "
              f"{k:7.4f}   {f.Intersection.tolist()}")
    n_inter = sum(f.Intersection.size for f in mdl.s_model)
    print(f"{n_inter} overlap entries in total ({n_inter // 2} unordered pairs if the relation is symmetric: "
          f"{'symmetric' if _is_symmetric(mdl) else 'NOT symmetric'})")
    print(f"bw (raster union of the accepted rectangles) = {int(mdl.bw.sum())} px "
          f"({100 * mdl.bw.mean():.2f} % of the image)")

    print(f"\nthe three ice concentrations of p. 208 (book: 76.96 / 83.17 / 87.75 %; these are OUR image, "
          "risk R12):")
    print(f"  segmentation (bw4)                    = {ice_concentration(d.bw4) * 100:6.2f} %")
    print(f"  global Otsu (bw)                      = {ice_concentration(d.gvf.bw) * 100:6.2f} %")
    print(f"  sum of ALL rectangle areas / (M*N)    = {sum(s.Area for s in S) / (shape[0] * shape[1]) * 100:6.2f} %")
    print(f"  sum of ACCEPTED rect. areas / (M*N)   = {rect_ice_concentration(mdl, shape) * 100:6.2f} %")
    print("  the last one exceeds the raster union because OVERLAPS ARE COUNTED TWICE — that is exactly the "
          "book's explanation of why 87.75 % > 83.17 % > 76.96 %.")

    # Figs. 9.13 / 9.16 — floe size distributions on the SAME bin centres
    ident_areas = np.array([s.Area for s in regionprops(label_components(d.bw4, 4), "Area")])
    rect_areas = np.array([f.Area for f in mdl.s_model])
    lo = float(min(ident_areas.min(), rect_areas.min()))
    hi = float(max(ident_areas.max(), rect_areas.max()))
    centres = np.linspace(lo, hi, args.bins)
    z_ident, x_c = floe_size_histogram(ident_areas, centres)
    z_rect, _ = floe_size_histogram(rect_areas, centres)
    err = fsd_error(z_rect, z_ident)
    print(f"\nFig. 9.13 / 9.16 floe size distribution on {args.bins} shared bin centres "
          f"({lo:.1f} .. {hi:.1f} px):")
    print(f"  identified floes (pixel areas) counts = {z_ident.astype(int).tolist()}")
    print(f"  model rectangles (rect areas)  counts = {z_rect.astype(int).tolist()}")
    print(f"  Fig. 9.16(b) bin-wise error           = {err.astype(int).tolist()}")

    written = [save_image(out / "fig_9_15_b_model_floes.png", mdl.bw)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 11))
    imshow_matlab(axes[0], d.bw4, title="Fig. 9.15(a) identified floes + rectangles")
    for s in S:
        axes[0].plot(s.Vertices[:, 0] - 1, s.Vertices[:, 1] - 1, "b-", lw=0.8)
    imshow_matlab(axes[1], mdl.bw, title=f"Fig. 9.15(b) model floes ({mdl.accepted.size} accepted)")
    for f in mdl.s_model:
        axes[1].plot(f.Vertices[:, 0] - 1, f.Vertices[:, 1] - 1, "b-", lw=0.8)
        axes[1].plot(f.Center[0] - 1, f.Center[1] - 1, "r+", ms=5)
    fig.suptitle("model_ice_model.m — rectangularized floes and their raster union")
    written.append(finish_figure(fig, out / "fig_9_15_model_ice_model.png", args.show))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    w = (x_c[1] - x_c[0]) * 0.4 if x_c.size > 1 else 1.0
    axes[0].bar(x_c - w / 2, z_ident, width=w, label="identified floes (Fig. 9.13)")
    axes[0].bar(x_c + w / 2, z_rect, width=w, label="model rectangles (Fig. 9.16(a))")
    axes[0].set_xlabel("floe size [px]")
    axes[0].set_ylabel("frequency")
    axes[0].legend(fontsize=8)
    axes[1].bar(x_c, err, width=w * 2, color="firebrick")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("floe size [px]")
    axes[1].set_ylabel("error")
    axes[1].set_title("Fig. 9.16(b) — bin-wise error")
    fig.suptitle("Floe size distribution: identified floes vs model rectangles")
    written.append(finish_figure(fig, out / "fig_9_16_fsd_and_error.png", args.show))

    print("\nfigures written:")
    for w_ in written:
        print("  ", w_)
    return 0


def _is_symmetric(mdl) -> bool:
    s = {(i + 1, j) for i, f in enumerate(mdl.s_model) for j in f.Intersection.tolist()}
    return all((j, i) in s for (i, j) in s)


if __name__ == "__main__":
    sys.exit(main())
