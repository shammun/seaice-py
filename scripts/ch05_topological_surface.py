"""Port of ``MATLAB_ROOT/ch5/topological_surface.m`` — the three topographic surfaces of §5.1: **Fig. 5.1(b)(c)**
(complement of the gray ``q.jpg`` and its surface), **Fig. 5.5(b)** (Sobel gradient surface) and **Fig. 5.8(c)**
(surface of the inverse chessboard distance map of the Otsu mask of the *gray* image, 128/255).

MATLAB: ``x = imcomplement(I)``; ``surf(double(x))`` with ``'FaceColor', 'texturemap', 'EdgeColor', 'none'``,
``colormap(copper)``, ``xlim([0 85])`` (three times).  Surface rendering is display-only; the arrays are exact.

Usage: ``python scripts/ch05_topological_surface.py [--data data/book/ch05] [--out outputs/ch05] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seaice.ch05_watershed import topographic_surfaces  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.plotting import save_image, surface_plot  # noqa: E402

CH = "ch05"
XLIM = (0, 85)  # xlim([0 85]) in the script


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="q.jpg")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        rgb, _ = load_image(CH, args.image, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:
        print(f"SKIP {Path(__file__).name}: skipped: private image absent ({exc})")
        return 0
    s = topographic_surfaces(rgb)
    M, N = s["gray"].shape
    print(f"topological_surface.m on {args.image} ({M}x{N}); gray range [{int(s['gray'].min())}, {int(s['gray'].max())}]")
    print(f"x = imcomplement(I): range [{int(s['complement'].min())}, {int(s['complement'].max())}] (= 255 - I)")
    print(f"g = |Sobel|: max {float(s['gradient'].max()):.4f}")
    print(f"img = im2bw(I, graythresh(I)) on the GRAY image: {int(s['bw'].sum())} ice px; "
          f"d = -bwdist(~img, 'chessboard'): min {float(s['neg_chessboard_dt'].min()):g}")
    written = [save_image(out / "fig_5_01a_gray.png", s["gray"]),
               save_image(out / "fig_5_01b_complement.png", s["complement"]),
               save_image(out / "sec_5_1_gray_otsu_mask.png", s["bw"])]
    for name, key, title in [("fig_5_01c_surface_complement", "complement", "Fig. 5.1(c)  surf(imcomplement(I))"),
                             ("fig_5_05b_surface_gradient", "gradient", "Fig. 5.5(b)  surf(|Sobel gradient|)"),
                             ("fig_5_08c_surface_chessboard_dt", "neg_chessboard_dt", "Fig. 5.8(c)  surf(-bwdist(~img, 'chessboard'))")]:
        path = out / f"{name}.png"
        surface_plot(s[key], path, cmap="copper", xlim=XLIM, title=title, show=args.show)
        written.append(path)
    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
