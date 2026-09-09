"""Port of ``MATLAB_ROOT/ch3/Otsu.m`` — global Otsu ice detection (Figs. 3.9(b)–3.11(b), Table 3.1) and multi-Otsu
with two thresholds (Fig. 3.12(a), Tables 3.2–3.3).

Book §3.1.1.1 (Eqs. 3.1, 3.20–3.22) and §3.1.3 (Eqs. 3.23–3.28), §3.3.  The MATLAB script hard-codes
``imread('test.jpg')`` (= "sea ice image 3", Fig. 3.11(a)); the authors re-ran it on ``1.jpg`` / ``2.jpg`` for
Figs. 3.9–3.10, so this driver takes ``--image {1,2,test,all}`` (default ``all``).

Book values reproduced (Table 3.1 / 3.2 / 3.3): Otsu t* = 123 / 107 / 182 → IC 15.36 / 32.05 / 72.63 %;
multi-Otsu on image 3: thresholds 120 / 197, coverages 3.50 / 42.11 / 54.39 %, means 63.8908 / 177.0690 / 218.1751.

Usage: ``python scripts/ch03_otsu.py [--image all] [--data data/book/ch03] [--out outputs/ch03] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch03_ice_pixel_detection import BOOK_IMAGES, BOOK_VALUES, multi_otsu_segmentation, \
    otsu_segmentation  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402

CH = "ch03"
FIG_NUM = {"1": "3_09", "2": "3_10", "test": "3_11"}


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="all", choices=["1", "2", "test", "all"],
                   help="which book image to process (default: all)")
    p.add_argument("--N", type=int, default=2, help="number of multi-Otsu thresholds (script: N = 2)")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)
    ids = ["1", "2", "test"] if args.image == "all" else [args.image]
    written: list[Path] = []

    for img_id in ids:
        fname, label, fig = BOOK_IMAGES[img_id]
        try:
            rgb, _ = load_image(CH, fname, allow_fallback=False, data_dir=data, verbose=False)
        except FileNotFoundError as exc:  # private book image absent: scripts never use the public substitute
            print(f"SKIP {Path(__file__).name} ({fname}): {exc}")
            continue
        I = rgb2gray_matlab(rgb)  # I = rgb2gray(imread(...))
        r, c = I.shape
        print(f"\n=== {fname} = {label} ({fig}(a)), {r}x{c} ===")

        # --- Otsu (lines 7–14) --------------------------------------------------------------------------------
        ot = otsu_segmentation(I)
        book_ic = BOOK_VALUES["otsu_ic"][img_id]
        print(f"graythresh: level = {ot['level']:.6f}  ->  t* = {ot['threshold']:g},  em = eta(t*) = {ot['em']:.4f}")
        print(f"Otsu IC = {100 * ot['ic']:.2f} %   (book {fig}(b) / Table 3.1: {book_ic:.2f} %)")
        written.append(save_image(out / f"otsu_bw_{img_id}.png", ot["bw"]))
        fig_, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        imshow_matlab(axes[0], I, title=f"(a) {label} (gray)")
        imshow_matlab(axes[1], ot["bw"], title=f"(b) Otsu, t* = {ot['threshold']:g}, IC = {100 * ot['ic']:.2f} %")
        fig_.suptitle(f"{fig}  Otsu thresholding of {label}  (book: IC = {book_ic:.2f} %)")
        written.append(finish_figure(fig_, out / f"fig_{FIG_NUM[img_id]}_otsu_image{'3' if img_id == 'test' else img_id}.png",
                                     args.show))

        # --- multi-Otsu (lines 28–43) --------------------------------------------------------------------------
        mo = multi_otsu_segmentation(I, args.N)
        th = ", ".join(str(int(v)) for v in mo["thresh"])
        print(f"multithresh(I, {args.N}): thresholds = [{th}],  metric = {mo['metric']:.4f}")
        cov = " / ".join(f"{100 * v:.2f}" for v in mo["coverage"])
        avg = " / ".join(f"{v:.4f}" for v in mo["average_intensity"])
        avg_s = " / ".join(f"{v:.4f}" for v in mo["average_intensity_script"])
        print(f"  coverage (class 1..{args.N + 1}, 1 = darkest) = {cov} %   IC (classes 2..{args.N + 1}) = {100 * mo['ic']:.2f} %")
        print(f"  average intensity (correct)              = {avg}")
        if not np.allclose(mo["average_intensity"], mo["average_intensity_script"], equal_nan=True):
            print(f"  average intensity as Otsu.m prints it     = {avg_s}   (stale 's' buffer, lines 39-42)")
        if img_id == "test" and args.N == 2:
            bc = BOOK_VALUES["multi_otsu_coverage_test"]
            bm = BOOK_VALUES["multi_otsu_means_test"]
            print(f"  book Table 3.2: water {bc[0]:.2f} / ice 2 {bc[1]:.2f} / ice 1 {bc[2]:.2f} %, IC 96.50 %;"
                  f"  Table 3.3 means: {bm[0]:.4f} / {bm[1]:.4f} / {bm[2]:.4f}")
        written.append(save_image(out / f"multi_otsu_seg_{img_id}.png", mo["seg"], autoscale=True))
        fig_, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        imshow_matlab(axes[0], I, title=f"(a) {label} (gray)")
        imshow_matlab(axes[1], mo["seg"], autoscale=True,
                      title=f"multi-Otsu, thresholds [{th}], coverage {cov} %")
        if img_id == "test":
            fig_.suptitle("Fig. 3.12(a)  Multi Otsu with 2 thresholds on sea ice image 3 (Table 3.2: 54.39 / 42.11 / 3.50 %)")
            name = "fig_3_12a_multi_otsu_image3.png"
        else:
            fig_.suptitle(f"§3.1.3  Multi Otsu (N = {args.N}) on {label} (not a book figure)")
            name = f"sec_3_1_3_multi_otsu_image{img_id}.png"
        written.append(finish_figure(fig_, out / name, args.show))

    print("\nfigures written:")
    for pth in written:
        print("  ", pth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
