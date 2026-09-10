"""Section 8.1 — shipborne ice concentration by Otsu and by k-means (k = 3).  **Text only: no MATLAB code.**

Section 8.1 of the book (pp. 176-181) describes a 4-lens "Ice Concentration" camera on IB *Frej* during OATRC'15,
one frame every 10 s, linearly rectified with the 4-corner method of Appendix A.1.2 (-> ch10) and then thresholded
two ways:

* **global Otsu** (section 3.1) - white = ice;
* **k-means with k = 3** - water (black) / **wet ice** (grey: rubble, young ice, melt ponds) / dry ice (white),
  with **both the white and the grey cluster counted as ice**, which is why the k-means series of Figure 8.5 sits
  systematically above the Otsu one.  "If we choose two clusters for the k-means method (k = 2), then this method
  is approximately reduced to the Otsu thresholding method" (p. 179).

**None of section 8.1's data ships with the book** (Figures 8.1-8.6 are third-party photographs credited to Lu,
Zhang, Lubbad, Loset & Skjetne, OTC Arctic Technology Conference 2016), so the 6-hour time series of Figure 8.5
and its two discrepancy events **cannot be reproduced** - every number the book prints for section 8.1 stays
``unverified``.  What this script does is run the *procedure*: on a folder of frames if you have one
(``--frames``), otherwise on a labelled synthetic ice field, and it also measures the "k = 2 is approximately
Otsu" claim.

The linear rectification that precedes all of this is Appendix A.1.2 and belongs to **ch10** - no second
rectifier is written here.

Usage: ``python scripts/ch08_ice_concentration_series.py [--frames DIR] [--k 3] [--seed 0]
[--out outputs/ch08] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch08_applications import shipborne_ice_concentration  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.io import read_image  # noqa: E402
from seaice.core.plotting import finish_figure, save_image  # noqa: E402
from seaice.core.synth import synthetic_floe_field  # noqa: E402

CH = "ch08"
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--frames", default="", help="folder of rectified frames (a Figure 8.5 style time series)")
    p.add_argument("--k", type=int, default=3, help="k-means clusters (the book uses 3)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    _, out = resolve_dirs(args)

    print("=== Section 8.1 - shipborne ice concentration (procedure only; no book data ships) ===")

    if args.frames:
        folder = Path(args.frames)
        frames = sorted(f for f in folder.rglob("*") if f.suffix.lower() in SUFFIXES)
        if not frames:
            print(f"SKIP {Path(__file__).name}: no frames found under {folder.as_posix()} "
                  "(section 8.1's own imagery is not shipped with the book and is not redistributable)")
            return 0
        otsu, kmeans, names = [], [], []
        for f in frames:
            img = read_image(f)
            otsu.append(shipborne_ice_concentration(img, "otsu").concentration)
            kmeans.append(shipborne_ice_concentration(img, "kmeans", k=args.k, seed=args.seed).concentration)
            names.append(f.name)
        otsu = np.array(otsu)
        kmeans = np.array(kmeans)
        print(f"  {len(frames)} frames from {folder.as_posix()}")
        print(f"  Otsu     : mean {100 * otsu.mean():.2f} %, range {100 * otsu.min():.2f} .. "
              f"{100 * otsu.max():.2f} %")
        print(f"  k-means(k={args.k}, ice = the {args.k - 1} brightest clusters): mean {100 * kmeans.mean():.2f} %, "
              f"range {100 * kmeans.min():.2f} .. {100 * kmeans.max():.2f} %")
        print(f"  k-means - Otsu: mean {100 * (kmeans - otsu).mean():+.2f} pp "
              "(the book reports k-means systematically HIGHER, p. 180)")
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(100 * otsu, "b.-", label="Otsu")
        ax.plot(100 * kmeans, f"r.-", label=f"k-means (k = {args.k})")
        ax.set_xlabel("frame")
        ax.set_ylabel("ice concentration [%]")
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title("Section 8.1.3 procedure - ice concentration time series (NOT the book's data)")
        w = finish_figure(fig, out / "sec_8_1_concentration_series.png", args.show)
        print("\nfigures written:\n  ", w)
        return 0

    # ---- no frames: run the procedure on a labelled synthetic field ---------------------------------------------
    # Three levels, the way section 8.1.2 reads the k-means clusters: water (dark), "wet ice" (grey rubble /
    # young ice / melt ponds) and dry ice (white).  The grey level is deliberately put BELOW the Otsu threshold
    # the bimodal water+ice mass produces, which is exactly the Event-#1 situation of p. 181 (dark young ice:
    # k-means -> ice, Otsu -> water).
    rng = np.random.default_rng(args.seed)
    dry = synthetic_floe_field((300, 400), n_floes=14, seed=args.seed, radius=(18, 45)) > 0
    wet = (synthetic_floe_field((300, 400), n_floes=10, seed=args.seed + 1, radius=(10, 26)) > 0) & ~dry
    gray = np.full((300, 400), 12.0)
    gray[wet] = 75.0
    gray[dry] = 210.0
    gray = np.clip(np.floor(gray + rng.normal(0.0, 6.0, gray.shape) + 0.5), 0, 255).astype(np.uint8)
    print(f"  synthetic layers: dry ice {100 * dry.mean():.2f} %, wet ice {100 * wet.mean():.2f} %, "
          f"water {100 * (~dry & ~wet).mean():.2f} %")

    print("  NO frames given -> demonstrating the two methods on a SYNTHETIC three-level ice field "
          "(Tier 3; this is not the book's imagery and its numbers are not book numbers)")
    o = shipborne_ice_concentration(gray, "otsu")
    k3 = shipborne_ice_concentration(gray, "kmeans", k=args.k, seed=args.seed)
    k2 = shipborne_ice_concentration(gray, "kmeans", k=2, ice_clusters=1, seed=args.seed)
    print(f"  Otsu           : level = {o.level:.6f} ({255 * o.level:.1f}/255), IC = {100 * o.concentration:.2f} %")
    print(f"  k-means (k = {args.k}): centres {np.round(k3.centers, 2).tolist()}, "
          f"ice = the 2 brightest clusters, IC = {100 * k3.concentration:.2f} %")
    print(f"  k-means (k = 2)  : centres {np.round(k2.centers, 2).tolist()}, IC = {100 * k2.concentration:.2f} %")
    agree = float((k2.mask == o.mask).mean())
    print(f"  p. 179 'k = 2 is approximately reduced to Otsu': the two masks agree on {100 * agree:.2f} % of "
          f"pixels ({100 * abs(k2.concentration - o.concentration):.2f} pp apart) - measured, not the book's claim")
    print(f"  k-means(k={args.k}) - Otsu = {100 * (k3.concentration - o.concentration):+.2f} pp "
          "(the book's Figure 8.5 shows the same sign: the grey 'wet ice' cluster counts as ice)")

    written = [save_image(out / "sec_8_1_synthetic_frame.png", gray),
               save_image(out / "sec_8_1_mask_otsu.png", o.mask),
               save_image(out / "sec_8_1_mask_kmeans3.png", k3.mask)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, im, title in ((axes[0], gray, "synthetic frame (Tier 3)"),
                          (axes[1], o.mask, f"Otsu: {100 * o.concentration:.1f} %"),
                          (axes[2], k3.labels, f"k-means k = {args.k} (3 classes)"),
                          (axes[3], k3.mask, f"ice = grey + white: {100 * k3.concentration:.1f} %")):
        ax.imshow(im, cmap="gray")
        ax.set_axis_off()
        ax.set_title(title)
    fig.suptitle("Section 8.1.2 procedure - Otsu vs k-means(3) ice concentration (NOT the book's data)")
    written.append(finish_figure(fig, out / "sec_8_1_methods.png", args.show))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
