"""L3 evidence for chapter 8: side-by-side comparisons of our figures with the printed ones.

Two kinds of comparison:

* **Fig. 8.19** is a *bitmap* in the PDF (``chapters/ch08.pdf`` page index 16, xref 429).  The Python
  reconstruction is the ``rgbImage`` that ``plot_color_bar_and_floe.m``'s ``Pixels`` branch paints, which
  ``tests/test_ch08.py`` already shows is byte-identical to MATLAB's.  The best crop of the printed bitmap is
  found by a coarse-to-fine search over crop rectangles maximising the RGB normalised cross-correlation, and
  both are drawn side by side with the NCC in the title.
* every other figure the data allows is compared against the **rendered PDF page** at 150 dpi (the MATLAB
  bar/loglog figures are vector art in the PDF, so there is no bitmap to correlate against).

Nothing written here is committed: ``reports/**/figures/`` is git-ignored (CLAUDE.md rule 12).

Usage: ``.venv/Scripts/python.exe reference/ch08/make_compare_figures.py``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pymupdf  # noqa: E402
from scipy.io import loadmat  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from seaice import ch08_applications as ch8  # noqa: E402
from seaice.core.histogram import hist as mhist  # noqa: E402
from seaice.core.icestruct import load_iceimage_mat  # noqa: E402
from seaice.core.stats import cumulative_size_distribution  # noqa: E402

PDF = ROOT / "chapters/ch08.pdf"
FIRST_PRINTED_PAGE = 175
DATA = ROOT / "data/book/ch08/MCD"
OUT = ROOT / "reports/ch08_verification/figures"
OUTPUTS = ROOT / "outputs/ch08"


def page_image(printed_page: int, dpi: int = 150) -> np.ndarray:
    doc = pymupdf.open(PDF)
    pix = doc[printed_page - FIRST_PRINTED_PAGE].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img[..., :3]


def embedded_image(page_index: int, xref: int) -> np.ndarray:
    doc = pymupdf.open(PDF)
    pix = pymupdf.Pixmap(doc, xref)
    if pix.n > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return a[..., :3]


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize (no SciPy dependency; the comparison is a correlation, not a metric)."""
    r = (np.arange(h) * img.shape[0] / h).astype(int).clip(0, img.shape[0] - 1)
    c = (np.arange(w) * img.shape[1] / w).astype(int).clip(0, img.shape[1] - 1)
    return img[np.ix_(r, c)]


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d else 0.0


def best_crop(book: np.ndarray, ours: np.ndarray, thumb: int = 64) -> tuple[float, tuple[int, int, int, int]]:
    """Coarse-to-fine search for the crop of ``book`` best matching ``ours`` (same aspect ratio)."""
    H, W = book.shape[:2]
    aspect = ours.shape[0] / ours.shape[1]
    tw = thumb
    th = max(int(round(thumb * aspect)), 8)
    small_ours = _resize(ours, th, tw).astype(np.float64)
    best = (-2.0, (0, 0, H, W))
    widths = np.unique(np.round(np.linspace(0.55 * W, W, 12)).astype(int))
    for w in widths:
        h = int(round(w * aspect))
        if h > H:
            continue
        for y0 in range(0, H - h + 1, max((H - h) // 12, 1)):
            for x0 in range(0, W - w + 1, max((W - w) // 12, 1)):
                v = ncc(_resize(book[y0:y0 + h, x0:x0 + w], th, tw), small_ours)
                if v > best[0]:
                    best = (v, (y0, x0, h, w))
    # refine around the coarse optimum
    v0, (y0, x0, h, w) = best
    for dy in range(-6, 7, 2):
        for dx in range(-6, 7, 2):
            for dw in range(-12, 13, 4):
                ww = w + dw
                hh = int(round(ww * aspect))
                yy, xx = y0 + dy, x0 + dx
                if yy < 0 or xx < 0 or yy + hh > H or xx + ww > W or ww < 16:
                    continue
                v = ncc(_resize(book[yy:yy + hh, xx:xx + ww], th, tw), small_ours)
                if v > best[0]:
                    best = (v, (yy, xx, hh, ww))
    return best


def side_by_side(name: str, ours: np.ndarray, book: np.ndarray, title_l: str, title_r: str,
                 subtitle: str = "") -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].imshow(ours)
    ax[0].set_title(title_l, fontsize=9)
    ax[1].imshow(book)
    ax[1].set_title(title_r, fontsize=9)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    if subtitle:
        fig.suptitle(subtitle, fontsize=10)
    fig.tight_layout()
    p = OUT / f"{name}.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def main() -> int:
    if not (DATA / "IceImage_290915_2_jpg.0000179.mat").exists():
        print("SKIP: private book data absent")
        return 0
    results: dict[str, dict] = {}
    ice = load_iceimage_mat(DATA / "IceImage_290915_2_jpg.0000179.mat")
    a = ch8.mcd_analysis(ice)

    # ---- Fig. 8.19 (bitmap): the painted MCD map -----------------------------------------------------------
    import cv2
    ours = np.ascontiguousarray(np.clip(np.round(a.raw.rgb_image * 255), 0, 255).astype(np.uint8))
    book = np.ascontiguousarray(embedded_image(16, 429))     # the PDF stores it in CMYK - convert first
    _, (y0, x0, h, w) = best_crop(book, ours)
    # refine at full resolution with an area-averaged resize of OUR map onto the candidate crop
    aspect = ours.shape[0] / ours.shape[1]
    best = (-2.0, (y0, x0, h, w))
    H, W = book.shape[:2]
    for ww in range(max(w - 40, 16), min(w + 41, W) + 1, 4):
        hh = int(round(ww * aspect))
        if hh > H:
            continue
        sm = cv2.resize(ours, (ww, hh), interpolation=cv2.INTER_AREA)
        for yy in range(max(y0 - 24, 0), min(y0 + 25, H - hh) + 1, 2):
            for xx in range(max(x0 - 32, 0), min(x0 + 33, W - ww) + 1, 2):
                val = ncc(np.ascontiguousarray(book[yy:yy + hh, xx:xx + ww]), sm)
                if val > best[0]:
                    best = (val, (yy, xx, hh, ww))
    v, (y0, x0, h, w) = best
    crop = np.ascontiguousarray(book[y0:y0 + h, x0:x0 + w]).astype(np.float64)
    sm = cv2.resize(ours, (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
    # the printed bitmap is a screened, JPEG-compressed, ~3x downsampled print: a sub-pixel blur is the fair
    # allowance for the registration error, so both the raw and the mildly smoothed NCC are recorded
    blurred = {f"ncc_blur{s}": round(ncc(cv2.GaussianBlur(crop, (0, 0), s), cv2.GaussianBlur(sm, (0, 0), s)), 4)
               for s in (0.8, 1.2, 1.8)}
    p = side_by_side("fig_8_19_compare", ours, book[y0:y0 + h, x0:x0 + w],
                     "Python: plot_color_bar_and_floe 'Pixels' branch (627x1114, jet(30))",
                     f"book Fig. 8.19 bitmap, best crop {w}x{h} at ({x0},{y0})",
                     f"Fig. 8.19 - RGB NCC = {v:.4f} raw, {blurred['ncc_blur1.2']:.4f} after a 1.2 px Gaussian")
    results["fig_8_19"] = {"ncc": round(v, 4), **blurred,
                           "crop": [int(x0), int(y0), int(w), int(h)], "png": p.name}
    print(f"Fig. 8.19: NCC {v:.4f} raw / {blurred}  crop {w}x{h} at ({x0},{y0})  -> {p}")

    # ---- Fig. 8.20 (vector): the MCD histogram --------------------------------------------------------------
    counts, centers = mhist(a.raw_mcd, ch8.MCD_X_BIN)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    idx = np.minimum(np.arange(1, centers.size + 1), a.color_limit_n) - 1
    ax.bar(centers, counts, width=1.0, color=a.raw.color_m[idx])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 450)
    ax.set_xlabel("MCD $L_i$ [m]")
    ax.set_ylabel("Number of ice floes $N$")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    tmp = OUT / "_tmp_820.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    import imageio.v3 as iio
    p = side_by_side("fig_8_20_compare", iio.imread(tmp)[..., :3], page_image(192),
                     "Python: hist(Raw_MCD, 1:100), jet(30) bars, y-limit 450",
                     "book p. 192 (Fig. 8.20 top)",
                     f"Fig. 8.20 - peak {int(counts.max())} at {centers[int(counts.argmax())]:.0f} m, "
                     f"sum {int(counts.sum())}")
    tmp.unlink(missing_ok=True)
    results["fig_8_20"] = {"peak": int(counts.max()), "peak_at": float(centers[int(counts.argmax())]),
                           "png": p.name}
    print(f"Fig. 8.20 -> {p}")

    # ---- Fig. 8.21 (vector): the cumulative FSD and the power-law fit ---------------------------------------
    raw = np.asarray(loadmat(str(DATA / "MCD_results.mat"))["Raw_MCD"]).ravel()
    f = ch8.cumulative_fsd_powerlaw(raw)
    L, Nc = cumulative_size_distribution(raw)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(L, Nc, "ro", ms=4, mfc="none")
    ax.loglog(f.x_plot, f.y_plot, "k", lw=2)
    ax.set_xlabel("MCD [m]")
    ax.set_ylabel("Cumulative Frequency")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(["Observed data", "Power law fitting'"], loc="lower left")
    fig.tight_layout()
    tmp = OUT / "_tmp_821.png"
    fig.savefig(tmp, dpi=110)
    plt.close(fig)
    p = side_by_side("fig_8_21_compare", iio.imread(tmp)[..., :3], page_image(193),
                     f"Python: Eqs. (8.2)/(8.3), alpha = {f.alpha:.4f}",
                     "book p. 193 (Fig. 8.21)",
                     f"Fig. 8.21 - alpha = {f.alpha:.6f} (book prints 1.3704)")
    tmp.unlink(missing_ok=True)
    results["fig_8_21"] = {"alpha": round(f.alpha, 6), "png": p.name}
    print(f"Fig. 8.21 -> {p}")

    # ---- Figs. 8.11-8.15: the section 8.2 pipeline figures the porter already produced ----------------------
    for name, produced, printed_page, note in (
            ("fig_8_11", "fig_8_11_floe_size_histogram.png", 186, "floe-size histogram of Fig. 8.10"),
            ("fig_8_12", "fig_8_12_sea_ice_model.png", 187, "polygonized floes / circularized brash"),
            ("fig_8_13", "fig_8_13_model_closeup.png", 187, "close-up with both centre marks"),
            ("fig_8_14", "fig_8_14_polygon_size_histogram.png", 188, "polygonized floe-size histogram"),
            ("fig_8_15", "fig_8_15_error_histogram_maxx3500.png", 188, "the histogram difference"),
    ):
        src = OUTPUTS / produced
        if not src.exists():
            print(f"  skip {name}: {src.name} not produced yet")
            continue
        p = side_by_side(f"{name}_compare", iio.imread(src)[..., :3], page_image(printed_page),
                         f"Python: outputs/ch08/{produced}", f"book p. {printed_page}",
                         f"{name.replace('_', '.')} - {note} (section 8.2 counts do NOT reproduce: see report)")
        results[name] = {"png": p.name, "source": produced}
        print(f"{name} -> {p}")

    (OUT / "figure_metrics.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
