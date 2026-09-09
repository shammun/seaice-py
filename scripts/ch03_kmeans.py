"""Port of ``MATLAB_ROOT/ch3/kmeans.m`` — the authors' k-means on the gray histogram (Figs. 3.9(c)–3.12(b), Tables 3.1–3.3).

Book §3.2.2 (Eqs. 3.35–3.37) and §3.3.  The MATLAB script hard-codes ``test.jpg`` and ``k = 3`` (Fig. 3.12(b));
with ``k = 2`` it gives Figs. 3.9(c)–3.11(c) / Table 3.1.  This driver runs ``k = 2`` on every image and ``k = 3``
on image 3 by default (``--image``, ``--k`` to change).

Every k-means number in the book comes from the script's **units bug** (the final mask compares the unshifted
image with centroids in ``ima − min + 1`` units, lines 64–70): those are printed as "book / script" (``shift_bug=
True``) next to the consistent Eqs. (3.36)–(3.37) result ("correct").  Book values: k = 2 IC 15.65 / 32.49 /
96.50 %; k = 3 on image 3: 77.91 / 19.20 / 2.89 %, IC 97.11 %, means 209.0405 / 161.6657 / 53.8234.

Usage: ``python scripts/ch03_kmeans.py [--image all] [--k 2,3] [--data data/book/ch03] [--out outputs/ch03]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch03_ice_pixel_detection import BOOK_IMAGES, BOOK_VALUES, kmeans_segmentation  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402
from seaice.core.plotting import finish_figure, imshow_matlab, save_image  # noqa: E402

CH = "ch03"
FIG_NUM = {"1": "3_09", "2": "3_10", "test": "3_11"}


def _fmt(v, f="{:.2f}"):
    return " / ".join(f.format(x) for x in v)


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--image", default="all", choices=["1", "2", "test", "all"], help="book image (default all)")
    p.add_argument("--k", default=None, help="comma-separated cluster counts (default: 2 for all images, 2,3 for test)")
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
        ima = rgb2gray_matlab(rgb)
        ks = [int(v) for v in args.k.split(",")] if args.k else ([2, 3] if img_id == "test" else [2])
        n3 = "3" if img_id == "test" else img_id
        print(f"\n=== {fname} = {label} ({fig}(a)), {ima.shape[0]}x{ima.shape[1]}, min = {int(ima.min())} ===")
        for k in ks:
            book = kmeans_segmentation(ima, k, shift_bug=True)  # what kmeans.m computes (book numbers)
            corr = kmeans_segmentation(ima, k, shift_bug=False)  # Eqs. (3.36)–(3.37) applied consistently
            assert np.allclose(book["centroids"], corr["centroids"])  # same centroids, only the mask differs
            print(f"--- k = {k}: converged in {book['n_iter']} iterations; centroids (gray units) = "
                  f"{_fmt(book['centroids'], '{:.4f}')}; script mu (shifted) = {_fmt(book['centroids_shifted'], '{:.4f}')}")
            print(f"  book / script (shift bug): boundaries {_fmt(book['boundaries'], '{:.1f}')}, coverage "
                  f"{_fmt(100 * book['coverage'])} %, IC = {100 * book['ic']:.2f} %, means {_fmt(book['average_intensity'], '{:.4f}')}")
            if not np.allclose(book["average_intensity"], book["average_intensity_script"], equal_nan=True):
                print(f"    (means as kmeans.m prints them, stale 'ss' buffer: {_fmt(book['average_intensity_script'], '{:.4f}')})")
            print(f"  correct (Eqs. 3.36-3.37):  boundaries {_fmt(corr['boundaries'], '{:.1f}')}, coverage "
                  f"{_fmt(100 * corr['coverage'])} %, IC = {100 * corr['ic']:.2f} %, means {_fmt(corr['average_intensity'], '{:.4f}')}")
            if k == 2:
                print(f"  book {fig}(c) / Table 3.1: IC = {BOOK_VALUES['kmeans2_ic'][img_id]:.2f} %")
            if k == 3 and img_id == "test":
                bc, bm = BOOK_VALUES["kmeans3_coverage_test"], BOOK_VALUES["kmeans3_means_test"]
                print(f"  book Fig. 3.12(b) / Table 3.2: {_fmt(bc)} %, IC 97.11 %;  Table 3.3 means: {_fmt(bm, '{:.4f}')}")

            written.append(save_image(out / f"kmeans{k}_mask_{img_id}_book.png", book["mask1"]))
            written.append(save_image(out / f"kmeans{k}_mask_{img_id}_correct.png", corr["mask1"]))
            fig_, axes = plt.subplots(1, 3, figsize=(18, 4.4))
            imshow_matlab(axes[0], ima, title=f"(a) {label}")
            imshow_matlab(axes[1], book["mask1"], title=f"(b) kmeans.m mask (shift bug), IC = {100 * book['ic']:.2f} %")
            imshow_matlab(axes[2], corr["mask1"], title=f"(c) Eqs. 3.36-3.37 consistently, IC = {100 * corr['ic']:.2f} %")
            if k == 2:
                fig_.suptitle(f"{fig}(c)  K-means with 2 clusters on {label}  (book: IC = "
                              f"{BOOK_VALUES['kmeans2_ic'][img_id]:.2f} %)")
                name = f"fig_{FIG_NUM[img_id]}_kmeans2_image{n3}.png"
            elif k == 3 and img_id == "test":
                fig_.suptitle("Fig. 3.12(b)  K-means with 3 clusters on sea ice image 3 (Table 3.2: 77.91 / 19.20 / 2.89 %)")
                name = "fig_3_12b_kmeans3_image3.png"
            else:
                fig_.suptitle(f"§3.2.2  K-means with {k} clusters on {label} (not a book figure)")
                name = f"sec_3_2_2_kmeans{k}_image{n3}.png"
            written.append(finish_figure(fig_, out / name, args.show))

    print("\nfigures written:")
    for pth in written:
        print("  ", pth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
