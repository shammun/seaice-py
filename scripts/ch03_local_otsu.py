"""Port of ``MATLAB_ROOT/ch3/local_Otsu.m`` — local (2×3 block) Otsu thresholding under uneven illumination (Fig. 3.4).

Book §3.1.2.  The script reads ``t.jpg`` (Fig. 3.4(a) = Fig. 3.2(a) with a "factitious uneven illumination"),
which is **not shipped** with the book code.  Following ``analysis/ch03.md`` §5 the labelled substitute is
``2.jpg`` (the histogram twin of Fig. 3.2(a)) with a synthetic illumination ramp
(:func:`seaice.core.synth.uneven_illumination`: x1.5 + 40 levels at the left edge to x0.5 - 40 at the right).
The ramped RGB image is written once, deterministically, to ``data/synthetic/ch03/t_uneven_2_g0.5_b40.jpg``
(git-ignored) and **read back**, so that MATLAB
(``imread('t.jpg')`` on a copy of that file) and Python process identical bytes.  The Fig. 3.4 numbers
(global t = 176; block thresholds 92/132/177/98/126/176; block ICs 73.8472/20.9869/25.1055/86.496/21.2756/
20.2267 %; overall IC 41.32 %) therefore cannot be reproduced — the printed values are for the substitute.

Usage: ``python scripts/ch03_local_otsu.py [--no-ramp] [--gain 0.5] [--bias 40] [--data data/book/ch03] [--out outputs/ch03]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from seaice.ch03_ice_pixel_detection import local_otsu  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import REPO_ROOT, load_image, read_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402
from seaice.core.synth import uneven_illumination  # noqa: E402

CH = "ch03"
SUBSTITUTE = "2.jpg"  # analysis/ch03.md §5: labelled substitute for the unshipped t.jpg / ch3ice.jpg


def ramped_substitute(rgb: np.ndarray, synthetic_dir: Path, gain: float, bias: float) -> tuple[np.ndarray, Path]:
    """Build (once) and re-read the ramped stand-in for ``t.jpg`` so MATLAB and Python read the same JPEG."""
    path = synthetic_dir / f"t_uneven_{SUBSTITUTE.split('.')[0]}_g{gain:g}_b{bias:g}.jpg"
    if not path.is_file():
        synthetic_dir.mkdir(parents=True, exist_ok=True)
        ramped = uneven_illumination(rgb, gain=gain, axis=1, bias=bias, kind="ramp")
        Image.fromarray(ramped).save(path, quality=95, subsampling=0)
        print(f"wrote synthetic uneven-illumination image: {path.as_posix()}")
    return read_image(path), path


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--ramp", action="store_true", default=True, dest="ramp",
                   help="apply the synthetic illumination ramp (default)")
    p.add_argument("--no-ramp", action="store_false", dest="ramp", help="use the plain substitute image")
    p.add_argument("--gain", type=float, default=0.5, help="multiplicative ramp: x(1+gain) left ... x(1-gain) right")
    p.add_argument("--bias", type=float, default=40.0, help="additive ramp: +bias left ... -bias right (gray levels)")
    p.add_argument("--synthetic-dir", default="data/synthetic/ch03", help="where the ramped JPEG is stored")
    p.add_argument("--n_r", type=int, default=2, help="blocks along rows (script: 2)")
    p.add_argument("--n_c", type=int, default=3, help="blocks along columns (script: 3)")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    try:
        rgb, _ = load_image(CH, SUBSTITUTE, allow_fallback=False, data_dir=data, verbose=False)
    except FileNotFoundError as exc:  # private book image absent: scripts never use the public substitute
        print(f"SKIP {Path(__file__).name}: {exc}")
        return 0
    print(f"NOTE: 't.jpg' (Fig. 3.4(a)) is not shipped; using {SUBSTITUTE} as the labelled SUBSTITUTE"
          + (" with a synthetic illumination ramp." if args.ramp else " (no ramp)."))
    src_path = None
    if args.ramp:
        sdir = Path(args.synthetic_dir)
        if not sdir.is_absolute():
            sdir = REPO_ROOT / sdir
        rgb, src_path = ramped_substitute(rgb, sdir, args.gain, args.bias)
        print(f"input read back from: {src_path.as_posix()}")
    I = rgb2gray_matlab(rgb)
    r, c = I.shape
    res = local_otsu(I, args.n_r, args.n_c)

    # --- key numbers ------------------------------------------------------------------------------------------
    print(f"image {r}x{c}, blocks {args.n_r}x{args.n_c} of {r // args.n_r}x{c // args.n_c}")
    print(f"global Otsu on the uneven image: t = {res['global_threshold']:g}, IC = {100 * res['global_ic']:.2f} %"
          f"   (book Fig. 3.4(b): t = 176 on the unshipped image)")
    for b, (th, n, icl) in enumerate(zip(res["thresholds"], res["counts"], res["ic_local"]), start=1):
        print(f"  block {b}: Threshold = {th:g}, num = {int(n)}, IC_local = {100 * icl:.4f} %")
    print(f"local Otsu overall IC = {100 * res['ic']:.2f} %   (book Fig. 3.4(c): 41.32 %; block thresholds "
          "92/132/177/98/126/176 -- unshipped image, not comparable)")

    # --- Fig. 3.4: (a) uneven image, (b) global Otsu, (c) block Otsu with the script's subplot titles --------
    written = [save_image(out / "local_otsu_bw.png", res["bw"]), save_image(out / "local_otsu_input_gray.png", I)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    imshow_matlab(axes[0], I, title="(a) substitute image with synthetic uneven illumination" if args.ramp
                  else "(a) substitute image (no ramp)")
    imshow_matlab(axes[1], res["global_bw"],
                  title=f"(b) global Otsu, t = {res['global_threshold']:g}, IC = {100 * res['global_ic']:.2f} %")
    fig.suptitle("Fig. 3.4(a, b)  Global Otsu under uneven illumination  [substitute image]")
    written.append(finish_figure(fig, out / "fig_3_04ab_uneven_global_otsu.png", args.show))

    fig, axes = plt.subplots(args.n_r, args.n_c, figsize=(4.2 * args.n_c, 3.3 * args.n_r))
    for ax, sl, title in zip(np.atleast_1d(axes).ravel(), res["slices"], res["titles"]):
        imshow_matlab(ax, res["bw"][sl], title=title)  # imshow(im2bw(temp, t)); title({IC=..%; Threshold=..})
    fig.suptitle(f"Fig. 3.4(c)  Local Otsu, {args.n_r}x{args.n_c} blocks, overall IC = {100 * res['ic']:.2f} %  "
                 "[substitute image]")
    written.append(finish_figure(fig, out / "fig_3_04c_local_otsu_blocks.png", args.show))

    print("figures written:")
    for pth in written:
        print("  ", pth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
