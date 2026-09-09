"""§5.3 "Experimental results and discussion" (Figs. 5.18–5.20, Table 5.1) — procedure only.

The Ny-Ålesund May-2011 images of §5.3 are not shipped (brash ice was removed *manually* before segmentation, so
Table 5.1 cannot be reproduced).  This driver applies the §5.2 pipeline (:func:`neighboring_region_merging`:
Otsu → city-block inverse distance map → 8-connected watershed → junction lines → concave ending points → merging)
to (1) a touching-floe crop of a same-expedition image shipped with ch03/ch04 (default: the Fig. 4.3(a) two-floe
crop of ``data/book/ch04/test.jpg``; skipped when the private image is absent) and (2) the synthetic two-floe
image ``synth.two_touching_floes`` (always).  Prints floes before/after and the lines removed; saves
``sec_5_3_<image>_<crop>_{mask,oversegmented,junction_lines,final}.png`` and a panel per input.

Usage: ``python scripts/ch05_experiments.py [--chapter ch04] [--image test.jpg] [--crop fig4_3a|r0:r1,c0:c1|none]
[--metric cityblock] [--endpoint-rule max|ge3] [--data data/book/ch05] [--out outputs/ch05] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch04_ice_edge_detection import FIG_4_3A_CROP  # noqa: E402
from seaice.ch05_watershed import neighboring_region_merging, otsu_mask  # noqa: E402
from seaice.core import synth  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.connectivity import label_components  # noqa: E402
from seaice.core.io import REPO_ROOT, load_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, label2rgb, save_image  # noqa: E402

CH = "ch05"


def parse_crop(spec: str) -> tuple[slice, slice] | None:
    if spec == "none":
        return None
    if spec == "fig4_3a":
        return FIG_4_3A_CROP
    rows, cols = spec.split(",")
    r0, r1 = (int(v) for v in rows.split(":"))
    c0, c1 = (int(v) for v in cols.split(":"))
    return slice(r0, r1), slice(c0, c1)


def run_one(rgb: np.ndarray, tag: str, metric: str, rule: str, out: Path, show: bool, written: list[Path]) -> None:
    bw = otsu_mask(rgb)
    res = neighboring_region_merging(bw, metric, rule)
    M, N = bw.shape
    print(f"\n=== {tag}: {M}x{N}, {int(bw.sum())} ice px ({100 * bw.mean():.1f} %), metric '{metric}', rule '{rule}' ===")
    print(f"connected ice regions before watershed: {int(label_components(bw).max())}; watershed basins {int(res.L.max())}; "
          f"junction lines {res.num} ({int(res.f.sum())} px)")
    n_kept = res.num - res.n_removed
    print(f"over-segmented regions {res.n_floes_before} -> after merging {res.n_floes_after}; "
          f"lines removed {res.n_removed}, kept {n_kept} (Table 5.1 counts are manual and not reproducible here)")
    for ln in res.lines[:12]:
        print(f"   line {ln.label:3d}: {ln.n_pixels:4d} px, {ln.endpoints.shape[0]} ending pts, "
              f"{ln.concave_endpoints.shape[0]} concave -> {'removed' if ln.removed else 'kept'}")
    if res.num > 12:
        print(f"   … {res.num - 12} more lines")
    written += [save_image(out / f"sec_5_3_{tag}_mask.png", bw),
                save_image(out / f"sec_5_3_{tag}_oversegmented.png", res.seg0),
                save_image(out / f"sec_5_3_{tag}_junction_lines.png", res.f),
                save_image(out / f"sec_5_3_{tag}_final.png", res.seg)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
    imshow_matlab(axes[0], rgb, title=f"{tag}")
    imshow_matlab(axes[1], label2rgb(label_components(res.seg0)), title=f"watershed: {res.n_floes_before} regions")
    imshow_matlab(axes[2], res.f, title=f"{res.num} junction lines (red = removed)")
    for ln in res.lines:
        if ln.removed:
            axes[2].plot(ln.pixels[:, 1], ln.pixels[:, 0], "r.", ms=2)
    imshow_matlab(axes[3], label2rgb(label_components(res.seg)), title=f"after merging: {res.n_floes_after} floes")
    fig.suptitle("§5.3 (Fig. 5.18-style) watershed + neighbouring-region merging — procedure on a substitute image")
    written.append(finish_figure(fig, out / f"sec_5_3_{tag}_panels.png", show))


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--chapter", default="ch04", help="chapter folder of the substitute image (ch03 or ch04)")
    p.add_argument("--image", default="test.jpg")
    p.add_argument("--crop", default="fig4_3a", help="'fig4_3a', 'none' or r0:r1,c0:c1 (0-based, exclusive)")
    p.add_argument("--metric", default="cityblock", choices=["cityblock", "chessboard", "euclidean", "quasi-euclidean"])
    p.add_argument("--endpoint-rule", default="max", choices=["max", "ge3"])
    p.add_argument("--no-synthetic", dest="synthetic", action="store_false", default=True)
    args = p.parse_args(argv)
    _, out = resolve_dirs(args)
    written: list[Path] = []
    data_dir = REPO_ROOT / "data" / "book" / args.chapter
    try:
        rgb, _ = load_image(args.chapter, args.image, allow_fallback=False, data_dir=data_dir, verbose=False)
        crop = parse_crop(args.crop)
        if crop is not None:
            rgb = rgb[crop[0], crop[1]]
        tag = f"{args.chapter}_{Path(args.image).stem}_{args.crop.replace(':', '-').replace(',', '_')}"
        run_one(rgb, tag, args.metric, args.endpoint_rule, out, args.show, written)
    except FileNotFoundError as exc:
        print(f"SKIP {args.chapter}/{args.image}: skipped: private image absent ({exc})")
    if args.synthetic:
        run_one(synth.two_touching_floes(), "synthetic_two_floes", args.metric, args.endpoint_rule, out, args.show, written)
    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
