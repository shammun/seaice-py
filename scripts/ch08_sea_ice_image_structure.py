"""Port of ``MATLAB_ROOT/ch{6,7}/Sea_Ice_Floe_Identification/SeaIce_Image_Structure.m`` — **Appendix B**.

Packs ch7's identified pieces and ch8's ``sea_ice_model`` output into the ``IceImage`` structure of Appendix B
(pp. 221-225; Figures B.1-B.3 are MATLAB-IDE screenshots of exactly this structure, derived from Figure 8.8)::

    Floe(i)  = {Center, Area, Perimeter, Polygon{Vertices, Center, Area, Perimeter, Intersect}, Pixels}
    Brash(i) = {Center, Area, Circle{Radius, Perimeter, Intersect}, Pixels}
    Param    = 17 fields;   Field = 15 fields including the FSD cell array
    FSD{i}   = [int_min(i), int_max(i), num(i)]     % from hist(area, min:inter:max) -- CENTRES, not intervals

Two modes:

* ``--source demo`` (default) — build the structure from the section 8.2 field (Figure 8.8), which is what
  Figures B.1-B.3 show, and dump it.
* ``--source iceimage`` — **verify**: load the shipped section 8.3 structure, rebuild ``Field.FSD`` and the
  coverages from its own contents and compare field by field.

``Field.FSD``'s printed interval labels do **not** describe the counting rule (analysis/ch08.md C5 / risk R11):
``num`` comes from ``hist`` with ``int_min`` as bin **centres**, so the real edges are the midpoints and the two
outer bins are unbounded.  The script prints the label-based recount beside the real one to show they differ.

Usage: ``python scripts/ch08_sea_ice_image_structure.py [--source demo|iceimage] [--save out.mat]
[--data data/book/ch08] [--out outputs/ch08] [--show]``
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch08_applications import (matlab_colon, sea_ice_field, sea_ice_image_structure,  # noqa: E402
                                      sea_ice_model)
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.histogram import hist as matlab_hist  # noqa: E402
from seaice.core.icestruct import load_iceimage_mat, overlap_graph, save_iceimage_mat  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure  # noqa: E402

CH = "ch08"


def dump(ice, n_show: int = 2) -> None:
    """Print the structure the way Figures B.1-B.3 show it in the MATLAB IDE."""
    print("IceImage")
    print("  .Param")
    for k, v in ice.Param.__dict__.items():
        print(f"      .{k:<12} {'[]' if v is None else v}")
    print("  .Field")
    for k, v in ice.Field.__dict__.items():
        if k == "FSD":
            print(f"      .{k:<12} 1x{len(v)} cell   e.g. {[t.tolist() for t in v[:3]]}")
        else:
            print(f"      .{k:<12} {'[]' if v is None else (f'{v:.10f}' if isinstance(v, float) else v)}")
    print(f"  .Floe   {len(ice.Floe)}x1 struct")
    for i, f in enumerate(ice.Floe[:n_show]):
        print(f"      ({i + 1}).Center {np.round(np.asarray(f.Center), 6).tolist()}  .Area {f.Area}  "
              f".Perimeter {f.Perimeter}")
        p = f.Polygon
        print(f"           .Polygon.Vertices {np.asarray(p.Vertices).shape} (open ring)  .Center "
              f"{np.round(p.Center, 6).tolist()}  .Area {p.Area}  .Perimeter {p.Perimeter:.6f}")
        print(f"           .Polygon.Intersect.floe {p.Intersect.floe.tolist()}  .brash "
              f"{p.Intersect.brash.tolist()}")
        print(f"           .Pixels {np.asarray(f.Pixels).shape}")
    print(f"  .Brash  {len(ice.Brash)}x1 struct")
    for i, b in enumerate(ice.Brash[:n_show]):
        print(f"      ({i + 1}).Center {np.round(np.asarray(b.Center), 6).tolist()}  .Area {b.Area}")
        print(f"           .Circle.Radius {b.Circle.Radius:.6f}  .Perimeter {b.Circle.Perimeter:.6f}  "
              f".Intersect.floe {b.Circle.Intersect.floe.tolist()}  .brash {b.Circle.Intersect.brash.tolist()}")
        print(f"           .Pixels {np.asarray(b.Pixels).shape}")


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--source", default="demo", choices=["demo", "iceimage"])
    p.add_argument("--image", default="sea_ice_test.jpg")
    p.add_argument("--mat", default="IceImage_290915_2_jpg.0000179.mat")
    p.add_argument("--save", default="", help="also write the structure back as a MATLAB v5 .mat")
    p.add_argument("--nbins", type=int, default=50, help="SeaIce_Image_Structure.m line 96")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    written = []
    if args.source == "iceimage":
        hits = sorted(data.rglob(args.mat))
        if not hits:
            print(f"SKIP {Path(__file__).name}: private data absent ({args.mat} not under {data.as_posix()})")
            return 0
        ice = load_iceimage_mat(hits[0])
        print(f"=== Appendix B: the shipped structure {hits[0].name} ===")
        dump(ice)

        # ---- verify the four printed coverages and the FSD block ------------------------------------------------
        print("\n--- checks against the book (p. 190) and against the FSD counting rule ---")
        print(f"  coverages: {100 * ice.Field.CovFloes:.2f} / {100 * ice.Field.CovBrash:.2f} / "
              f"{100 * ice.Field.CovSlush:.2f} / {100 * ice.Field.CovWater:.2f} %  "
              f"(book: 58.00 / 4.85 / 21.21 / 15.94 %; sum "
              f"{100 * (ice.Field.CovFloes + ice.Field.CovBrash + ice.Field.CovSlush + ice.Field.CovWater):.4f} % "
              "- the book TRUNCATES the water value, 15.9460 -> 15.94)")
        npx = ice.Param.NumPix_x * ice.Param.NumPix_y
        af = sum(f.Area for f in ice.Floe)
        ab = sum(b.Area for b in ice.Brash)
        print(f"  CovFloes recomputed from the floe pixel counts: {af}/{npx} = {af / npx:.10f} "
              f"(stored {ice.Field.CovFloes:.10f}, |delta| = {abs(af / npx - ice.Field.CovFloes):.3e})")
        print(f"  CovBrash recomputed: {ab}/{npx} = {ab / npx:.10f} "
              f"(|delta| = {abs(ab / npx - ice.Field.CovBrash):.3e})")
        print(f"  NumFloes {ice.Field.NumFloes} / NumBrash {ice.Field.NumBrash} (book p. 190: 2888 / 3452)")

        areas = np.array([f.Area for f in ice.Floe], dtype=np.float64)
        mn, mx = areas.min(), areas.max()
        inter = float(np.trunc((mx - mn) / args.nbins))
        centres = matlab_colon(mn, inter, mx)
        z, n_c = matlab_hist(areas, centres)
        int_max = np.append(n_c[1:] - 1, mx)
        mine = np.stack([n_c, int_max, z], 1).astype(np.int64)
        ref = np.array([np.asarray(t, dtype=np.int64) for t in ice.Field.FSD])
        print(f"  FSD: inter = fix(({mx:.0f} - {mn:.0f})/{args.nbins}) = {inter:.0f}, "
              f"{len(centres)} centres {centres[0]:.0f}:{inter:.0f}:{centres[-1]:.0f}")
        print(f"       recomputed with core.histogram.hist: {mine.shape[0]} triplets, "
              f"max |delta| vs the shipped FSD = {np.abs(mine - ref).max()}")
        naive = np.array([int(((areas >= t[0]) & (areas <= t[1])).sum()) for t in ref])
        print(f"       a naive 'count areas in [int_min, int_max]' gives {naive[:4].tolist()}... against the "
              f"stored {ref[:4, 2].tolist()}... -> {int((naive == ref[:, 2]).sum())} of {len(ref)} triplets "
              "(the labels are NOT the bins - risk R11)")

        g = overlap_graph(ice)
        print(f"  overlap graph: floe-floe {g['n_floe_floe']}, floe-brash {g['n_floe_brash']}, "
              f"brash-floe {g['n_brash_floe']}, brash-brash {g['n_brash_brash']}; "
              f"symmetric ff={g['floe_floe_symmetric']} bb={g['brash_brash_symmetric']} "
              f"fb-consistent={g['floe_brash_consistent']}")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(ref[:, 0], ref[:, 2], width=inter, color="steelblue", edgecolor="k", linewidth=0.2)
        ax.set_xlabel("floe area [pixels] (interval minimum)")
        ax.set_ylabel("number of floes")
        ax.set_title("Appendix B Field.FSD of the shipped structure (51 triplets, section 8.3 field)")
        written.append(finish_figure(fig, out / "sec_b_1_shipped_fsd.png", args.show))
        name = "sec_b_1_iceimage_roundtrip.mat"
    else:
        rgb = None
        for chapter, folder in (("ch08", data), ("ch07", None)):
            try:
                rgb, _ = load_image(chapter, args.image, allow_fallback=False, data_dir=folder, verbose=False)
                break
            except FileNotFoundError:
                rgb = None
        if rgb is None:
            print(f"SKIP {Path(__file__).name}: private image absent ({args.image})")
            return 0
        t0 = time.time()
        enh = sea_ice_field(rgb, cache=out / f"stage3_{Path(args.image).stem}_{rgb.shape[0]}x{rgb.shape[1]}.npz")
        m = sea_ice_model(enh.ice_floe, enh.brash_ice, enh.index_floe, raster=False)
        ice = sea_ice_image_structure(enh.ice_floe, enh.brash_ice, m.floe, m.brash, enh.coverage,
                                      enh.index_floe, enh.index_residue, nbins=args.nbins)
        print(f"=== Appendix B built from the section 8.2 field (Figure 8.8), {time.time() - t0:.1f} s ===")
        print("    (Figures B.1-B.3 are screenshots of exactly this structure)")
        dump(ice)
        print("\n  NOTE: the script hard-codes Location = 'Ny-Alesund', Creator = 'UAV', LengthSI_x = 50, "
              "LengthSI_y = 18 (lines 54-55, 62-63, 78-79); the SHIPPED .mat instead holds Creator = "
              "'Helicopter', PrjName = 'OATRC 2015' and empty lengths, i.e. it came from a section 8.3 variant "
              "of this section 8.2 script (risk R12).")
        print(f"  CovOther = {ice.Field.CovOther:.10f} "
              f"({int(np.count_nonzero(enh.index_residue))} residue px / "
              f"{ice.Param.NumPix_x * ice.Param.NumPix_y}) - never printed in the book")

        fig, ax = plt.subplots(figsize=(10, 5))
        fsd = np.array([t for t in ice.Field.FSD])
        ax.bar(fsd[:, 0], fsd[:, 2], width=max(fsd[1, 0] - fsd[0, 0], 1), color="steelblue",
               edgecolor="k", linewidth=0.2)
        ax.set_xlabel("floe area [pixels] (interval minimum)")
        ax.set_ylabel("number of floes")
        ax.set_title(f"Appendix B Field.FSD of the section 8.2 field ({len(ice.Field.FSD)} triplets)")
        written.append(finish_figure(fig, out / "sec_b_1_fsd.png", args.show))
        name = "sec_b_1_sea_ice_image_structure.mat"

    target = Path(args.save) if args.save else (out / name)
    if not target.is_absolute():
        target = out / target
    save_iceimage_mat(target, ice)
    back = load_iceimage_mat(target)
    ok = (len(back.Floe) == len(ice.Floe) and len(back.Brash) == len(ice.Brash)
          and len(back.Field.FSD) == len(ice.Field.FSD)
          and all(np.array_equal(np.asarray(a.Pixels), np.asarray(b.Pixels))
                  for a, b in zip(ice.Floe, back.Floe)))
    print(f"\n  saved as {target} and re-loaded: round-trip {'OK' if ok else 'FAILED'}")
    written.append(target)

    print("\nfiles written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
