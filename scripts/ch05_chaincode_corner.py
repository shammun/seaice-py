"""Port of ``MATLAB_ROOT/ch5/chaincode_corner.m`` — concave boundary points of the Otsu mask of ``q.jpg`` by the
differential chain code (§5.2.1.2, Eqs. 5.9–5.19; **Fig. 5.17**: the two gray dot clusters at the junction
notches).  The script body (lines 6–74) is ``watershed_based/freeman_concave.m``; both are :func:`freeman_concave`.

Literal steps: ``B = im2bw(I, graythresh(I))`` (RGB); ``b = boundaries(B, 8, 'cww')`` (typo → clockwise);
``b = b{1}`` (first object); ``bim = bound2im(...)``; ``c = fchcode(b)``; relative / absolute / summed /
differential codes; ``p = find(Diff >= 3 & Diff <= 10); concave = b(p, :)``; ``imshow(bim); plot(concave, 'r.')``.

Usage: ``python scripts/ch05_chaincode_corner.py [--object first|longest] [--data data/book/ch05] [--out outputs/ch05] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch05_watershed import BOOK_PARAMS, freeman_concave  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402

CH = "ch05"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--object", default="first", choices=["first", "longest"],
                   help="b{1} (the script) or the longest boundary (the [max_d, k] the script computes and ignores)")
    p.add_argument("--image", default="q.jpg")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        rgb, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: skipped: private image absent ({exc})")
        return 0
    r = freeman_concave(rgb, object=args.object)
    M, N = r.bw.shape
    print(f"chaincode_corner.m on {args.image} ({M}x{N}): {r.n_boundaries} object(s), using b{{{1 if args.object == 'first' else r.longest + 1}}}")
    print(f"boundary: {r.boundary.shape[0]} points (closed), {r.code.fcc.size} chain codes, start x0y0 = {r.code.x0y0_matlab} (1-based)")
    print(f"Eq. 5.11 A(N) = {int(r.A[-1])} (expected -8 for a clockwise trace); R in [{int(r.R.min())}, {int(r.R.max())}]; "
          f"Diff in [{int(r.Diff.min())}, {int(r.Diff.max())}]")
    print(f"concave points (Diff in [{BOOK_PARAMS['concave_min']}, {BOOK_PARAMS['concave_max']}], i.e. "
          f"{BOOK_PARAMS['concave_min'] * 15}°–{BOOK_PARAMS['concave_max'] * 15}°): {r.points.shape[0]}  (1-based row, col):")
    for (rr, cc), i, d in zip(r.points_matlab.tolist(), (r.index + 1).tolist(), r.Diff[r.index].tolist()):
        print(f"   p = {i:4d}  ({rr}, {cc})  Diff = {d}")
    written = [save_image(out / "sec_5_2_1_boundary_image.png", r.bim), save_image(out / "sec_5_2_1_otsu_mask.png", r.bw)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    imshow_matlab(axes[0], r.bim, title="bim = bound2im(b, M, N, min(b(:,1)), min(b(:,2)))")
    axes[0].plot(r.points[:, 1], r.points[:, 0], "r.", ms=8)
    axes[0].plot(r.code.x0y0[1], r.code.x0y0[0], "g+", ms=10, label="starting point")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].plot(np.arange(1, r.Diff.size + 1), r.Diff, "k-", lw=1)
    axes[1].axhspan(BOOK_PARAMS["concave_min"], BOOK_PARAMS["concave_max"], color="r", alpha=0.15, label="concave band 3..10")
    axes[1].plot(r.index + 1, r.Diff[r.index], "r.", ms=8)
    axes[1].set_xlabel("boundary point i")
    axes[1].set_ylabel("differential chain code D(i)  (θ = D × 15°, Eq. 5.19)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Fig. 5.17  Concave points of the floe boundary by the differential chain code (chaincode_corner.m)")
    written.append(finish_figure(fig, out / "fig_5_17_concave_points.png", args.show))
    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
