"""Chapter 3 — Ice Pixel Detection: demo-level functions.

Each function reproduces one MATLAB script of ``MATLAB_ROOT/ch3/`` or one text-only example of the book, using the
primitives in :mod:`seaice.core.threshold`, :mod:`seaice.core.clustering` and :mod:`seaice.core.synth`.  The
``scripts/ch03_*.py`` drivers only load data, call these functions and write figures to ``outputs/ch03/``.

Book: Zhang & Skjetne (2018), Chapter 3, pp. 37–58.  The output of the chapter is a binary (or k-level) ice mask
and the ice concentration ``IC = #ice pixels / (M N)``.

MATLAB ↔ Python map (analysis/ch03.md §3):

* ``Otsu.m`` lines 2–14 → :func:`otsu_segmentation`; lines 28–43 → :func:`multi_otsu_segmentation`
* ``local_Otsu.m`` → :func:`local_otsu`
* ``separability.m`` → :func:`separability_script`
* ``kmeans.m`` → :func:`kmeans_segmentation`
* text-only Eq. (3.1) / Fig. 3.2 → :func:`fixed_threshold`, :func:`plot_histogram_with_threshold`
* text-only Figs. 3.6 / 3.8 → :func:`kmeans_demo_2d`
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .core import synth
from .core.clustering import KMeansResult, kmeans_gray, kmeans_lloyd, pairwise_distance
from .core.histogram import imhist
from .core.matlab_compat import num2str as _mc_num2str, rgb2gray_matlab
from .core.threshold import block_otsu, class_coverage, class_mean_intensity, graythresh, ice_concentration, \
    im2bw, imquantize, multithresh, otsu_criterion

#: Book image ids → shipped file names and the figure that shows them (§3.3, Figs. 3.9–3.11).
BOOK_IMAGES = {
    "1": ("1.jpg", "sea ice image 1", "Fig. 3.9"),
    "2": ("2.jpg", "sea ice image 2", "Fig. 3.10"),
    "test": ("test.jpg", "sea ice image 3", "Fig. 3.11"),
}

#: Values quoted in the book for the shipped images (Table 3.1–3.3, figure captions), keyed by image id.
BOOK_VALUES = {
    "otsu_ic": {"1": 15.36, "2": 32.05, "test": 72.63},  # Table 3.1, Otsu row (%)
    "kmeans2_ic": {"1": 15.65, "2": 32.49, "test": 96.50},  # Table 3.1, K-means row (%)
    "multi_otsu_coverage_test": (3.50, 42.11, 54.39),  # Table 3.2 (water, ice group 2, ice group 1) (%)
    "multi_otsu_means_test": (63.8908, 177.0690, 218.1751),  # Table 3.3
    "kmeans3_coverage_test": (2.89, 19.20, 77.91),  # Table 3.2
    "kmeans3_means_test": (53.8234, 161.6657, 209.0405),  # Table 3.3 ("209,0405" as printed)
}


def _as_gray(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    return rgb2gray_matlab(img) if img.ndim == 3 else img


# ---------------------------------------------------------------------------------------------------------------
# §3.1.1.1 / §3.1.3  Otsu.m
# ---------------------------------------------------------------------------------------------------------------


def otsu_segmentation(gray: np.ndarray) -> dict[str, Any]:
    """Global Otsu ice-pixel detection — port of ``MATLAB_ROOT/ch3/Otsu.m`` lines 2–14.

    Book: §3.1.1.1, Eqs. (3.1), (3.20)–(3.22); Fig. 3.3 (t = 108, IC 42.14 % on the unshipped Fig. 3.2(a)),
    Figs. 3.9(b)–3.11(b) and Table 3.1 (IC 15.36 / 32.05 / 72.63 % for images 1 / 2 / 3).

    Steps: ``t = graythresh(I)``, ``bw = im2bw(I, t)``, ``ic = length(find(bw == 1)) / (r c)``.

    Returns
    -------
    dict with ``level`` (``t`` ∈ [0, 1]), ``threshold`` (``255 t``, float; half-integer on ties), ``em`` (η(t*),
    Eq. 3.22), ``bw`` (bool mask, Eq. 3.1 with strict ``>``), ``ic`` (fraction), ``counts`` (256-bin histogram).
    """
    gray = _as_gray(gray)
    t, em = graythresh(gray)
    bw = im2bw(gray, t)
    counts, _ = imhist(gray)
    return {"level": t, "threshold": 255.0 * t, "em": em, "bw": bw, "ic": ice_concentration(bw), "counts": counts}


def stale_mean_intensity(gray: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Emulate the **stale-buffer bug** of ``Otsu.m`` lines 39–42 / ``kmeans.m`` lines 76–79.

    The scripts fill ``s(j) = I(p(j))`` for ``j = 1..length(p)`` without clearing ``s`` between classes, then take
    ``sum(s(:)) / length(p)``.  When class ``i`` has fewer pixels than an earlier class, ``s`` still holds the tail
    of the previous, longer fill, so the "average" includes stale values.  MATLAB indexes ``p = find(seg == i)`` in
    column-major order — reproduced here with ``order='F'``.  Harmless on ``test.jpg`` (class sizes increase),
    wrong on ``1.jpg`` / ``2.jpg``.  Provided so the verifier can compare against the MATLAB reference values; the
    library functions return the correct means.
    """
    g = np.asarray(gray).astype(np.float64).ravel(order="F")
    lab = np.asarray(labels).ravel(order="F")
    buf = np.zeros(0)
    out = np.full(k, np.nan)
    for i in range(1, k + 1):
        vals = g[lab == i]
        n = vals.size
        if n == 0:
            continue
        if n >= buf.size:
            buf = vals.copy()
        else:
            buf = buf.copy()
            buf[:n] = vals
        out[i - 1] = buf.sum() / n
    return out


def multi_otsu_segmentation(gray: np.ndarray, N: int = 2) -> dict[str, Any]:
    """Multilevel Otsu (``N`` thresholds, ``N + 1`` classes) — port of ``MATLAB_ROOT/ch3/Otsu.m`` lines 28–43.

    Book: §3.1.3, Eqs. (3.23)–(3.28); Fig. 3.5 (61 / 142 on the unshipped Fig. 3.2(a)); Fig. 3.12(a), Table 3.2
    (image 3: water 3.50 %, ice group 2 42.11 %, ice group 1 54.39 %, IC 96.50 %) and Table 3.3 (means
    63.8908 / 177.0690 / 218.1751).

    Steps: ``thresh = multithresh(I, N)``, ``seg = imquantize(I, thresh)`` (classes ``1..N+1``, 1 = darkest),
    ``coverage(i) = #(seg == i) / (r c)``, ``average_intensity(i)`` = mean gray level of class ``i``.

    Returns
    -------
    dict with ``thresh`` (uint8, (N,)), ``metric`` (Eq. 3.28), ``seg`` (int, ``1..N+1``), ``coverage`` ((N+1,)
    fractions), ``average_intensity`` (correct class means), ``average_intensity_script`` (the stale-buffer values
    the MATLAB script prints — equal to the correct ones on ``test.jpg``), ``ic`` = coverage of all classes but the
    darkest (Table 3.2 "IC").
    """
    gray = _as_gray(gray)
    thresh, metric = multithresh(gray, N)
    seg = imquantize(gray, thresh)
    coverage = class_coverage(seg, N + 1)
    avg = class_mean_intensity(gray, seg, N + 1)
    stale = stale_mean_intensity(gray, seg, N + 1)
    return {"thresh": thresh, "metric": metric, "seg": seg, "coverage": coverage, "average_intensity": avg,
            "average_intensity_script": stale, "ic": float(coverage[1:].sum())}


# ---------------------------------------------------------------------------------------------------------------
# §3.1.2  local_Otsu.m
# ---------------------------------------------------------------------------------------------------------------


def local_otsu(gray: np.ndarray, n_r: int = 2, n_c: int = 3) -> dict[str, Any]:
    """Local (block) Otsu thresholding — port of ``MATLAB_ROOT/ch3/local_Otsu.m``.

    Book: §3.1.2, Fig. 3.4 (unshipped ``t.jpg`` = Fig. 3.2(a) with an artificial illumination ramp; global
    t = 176 in (b); 2×3 blocks with thresholds 92/132/177/98/126/176 and block ICs 73.8472/20.9869/25.1055/
    86.496/21.2756/20.2267 %, overall IC 41.32 % in (c)).

    Returns
    -------
    dict with the :class:`~seaice.core.threshold.BlockOtsu` fields (``thresholds``, ``levels``, ``counts``,
    ``ic_local``, ``ic``, ``bw``, ``slices``) plus the *global* Otsu of the same image for the Fig. 3.4(b)
    comparison (``global_threshold``, ``global_bw``, ``global_ic``) and ``titles`` — the subplot titles of the
    script (``IC=<num2str(ic0)>%`` / ``Threshold=<th>``, ``num2str`` = 4 significant digits).
    """
    gray = _as_gray(gray)
    blk = block_otsu(gray, n_r, n_c)
    glob = otsu_segmentation(gray)
    titles = [f"IC={_num2str(100 * ic)}%\nThreshold={_num2str(th)}" for ic, th in zip(blk.ic_local, blk.thresholds)]
    return {"thresholds": blk.thresholds, "levels": blk.levels, "counts": blk.counts, "ic_local": blk.ic_local,
            "ic": blk.ic, "bw": blk.bw, "slices": blk.slices, "n_r": n_r, "n_c": n_c,
            "global_threshold": glob["threshold"], "global_bw": glob["bw"], "global_ic": glob["ic"],
            "titles": titles}


#: MATLAB ``num2str`` for a scalar double — **promoted to** :func:`seaice.core.matlab_compat.num2str` when
#: chapter 9's ``block_threshold.m`` became the second caller (ch03 open item 3).  This name is kept as an alias
#: so every chapter-3 call site and test keeps working; the implementation lives in ``core`` only.
_num2str = _mc_num2str


# ---------------------------------------------------------------------------------------------------------------
# §3.1.1 / §3.1.1.1  separability.m
# ---------------------------------------------------------------------------------------------------------------


def separability_script(gray: np.ndarray, k: int = 108) -> dict[str, Any]:
    """Fixed threshold ``k`` and the Otsu separability — literal port of ``MATLAB_ROOT/ch3/separability.m``.

    Book: §3.1.1 Eq. (3.1) (``bw = I > k``, IC) → Fig. 3.3; §3.1.1.1 Eqs. (3.3), (3.4), (3.8), (3.9), (3.16) last
    form, (3.17), (3.20) → η(108) = 0.9643 and η(125) = 0.9620 quoted on p. 43 (for the unshipped Fig. 3.2(a)).

    The script's quirks are reproduced exactly (they do not change η's value but shift where it is evaluated):

    * it uses the **1-based bin index** ``i = 1..256`` as the intensity, so ``mg = mG + 1`` and ``m = m(t) + P0``
      (σ_B² and σ_G² are shift-invariant, so ``eta`` is unaffected);
    * its class C0 is ``bins 1..k`` = **levels 0..k−1**, i.e. ``eta`` is η evaluated at ``t = k − 1`` (book
      convention, C0 = ``0..t``) while the mask uses ``I > k`` (t = k).  ``eta_book`` gives η at ``t = k`` itself.

    Returns
    -------
    dict with the script variables ``bw`` (float 0/1 like ``zeros(size(I))``), ``IC``, ``p``, ``mg``, ``sigma2_g``,
    ``sigma2_b``, ``eta``, plus ``t_eval`` (= k − 1), ``eta_book`` (η(k), book convention), ``eta_curve``
    (η(t) for all t, :func:`~seaice.core.threshold.otsu_criterion`), ``t_star``, ``eta_star``.
    """
    gray = _as_gray(gray)
    m_rows, n_cols = gray.shape
    pp = gray > k  # find(I > k)
    bw = pp.astype(np.float64)
    IC = float(np.count_nonzero(pp) / (m_rows * n_cols))
    fxy, _ = imhist(gray, 256)
    p = fxy.astype(np.float64) / (m_rows * n_cols)
    i = np.arange(1, 257, dtype=np.float64)  # 1-based bin index used as intensity
    mg = float(np.sum(i * p))
    sigma2_g = float(np.sum((i - mg) ** 2 * p))
    m = float(np.sum(i[:k] * p[:k]))  # for i = 1 : k
    p0 = float(np.sum(p[:k]))
    sigma2_b = (mg * p0 - m) ** 2 / (p0 * (1.0 - p0))
    eta = sigma2_b / sigma2_g
    curves = otsu_criterion(fxy)
    return {"bw": bw, "IC": IC, "p": p, "mg": mg, "sigma2_g": sigma2_g, "sigma2_b": sigma2_b, "eta": float(eta),
            "k": k, "t_eval": k - 1, "eta_book": float(curves.eta[k]) if k < 256 else float("nan"),
            "eta_curve": curves.eta, "t_star": curves.t_star, "eta_star": curves.eta_star, "curves": curves}


# ---------------------------------------------------------------------------------------------------------------
# §3.2.2  kmeans.m
# ---------------------------------------------------------------------------------------------------------------


def kmeans_segmentation(gray: np.ndarray, k: int = 3, shift_bug: bool = False) -> dict[str, Any]:
    """k-means ice-pixel detection on the gray histogram — port of ``MATLAB_ROOT/ch3/kmeans.m``.

    Book: §3.2.2 Eqs. (3.35)–(3.37), §3.3; Figs. 3.9(c)–3.11(c) (k = 2, IC 15.65 / 32.49 / 96.50 %), Fig. 3.12(b)
    (k = 3: 77.91 / 19.20 / 2.89 %, IC 97.11 %), Table 3.3 (means 209.0405 / 161.6657 / 53.8234).

    ``shift_bug=True`` reproduces the script's mask (unshifted image vs shifted centroids, lines 64–70) and
    therefore the book's numbers; ``False`` applies Eqs. (3.36)–(3.37) consistently (see
    :func:`~seaice.core.clustering.kmeans_gray`).  With ``False`` on image 3 (k = 3) the class boundaries
    ``(c_i + c_{i+1})/2`` = 120.5 / 197.6 coincide with multi-Otsu's 120 / 197 and the centroids equal the
    Table 3.3 multi-Otsu means — the book's remark that both methods minimise the within-class variance.

    Returns
    -------
    dict with ``centroids`` (gray units), ``centroids_shifted`` (script ``mu``), ``offset``, ``mask`` (``1..k``),
    ``mask1`` (``i/k`` display image), ``coverage`` (script ``ic`` vector), ``counts`` (``n``), ``average_intensity``
    (correct), ``average_intensity_script`` (stale-buffer emulation), ``boundaries`` (midpoints between sorted
    centroids, in the units the mask was computed in), ``n_iter``, ``ic`` (1 − coverage of the darkest cluster),
    ``shift_bug``.
    """
    gray = _as_gray(gray)
    res = kmeans_gray(gray, k, shift_bug=shift_bug)
    mu_sorted = np.sort(res.centroids_shifted)
    # boundaries in image units: shifted mu vs unshifted image if the bug is on, else in gray units
    mids = (mu_sorted[:-1] + mu_sorted[1:]) / 2.0
    boundaries = mids if shift_bug else mids + res.offset - 1.0
    return {"centroids": res.centroids, "centroids_shifted": res.centroids_shifted, "offset": res.offset,
            "mask": res.mask, "mask1": res.mask1, "coverage": res.coverage, "counts": res.counts,
            "average_intensity": res.average_intensity,
            "average_intensity_script": stale_mean_intensity(gray, res.mask, k), "boundaries": boundaries,
            "n_iter": res.n_iter, "ic": res.ic, "shift_bug": shift_bug}


# ---------------------------------------------------------------------------------------------------------------
# §3.1.1 text-only: Eq. (3.1) with a hand-picked T (Fig. 3.2)
# ---------------------------------------------------------------------------------------------------------------


def fixed_threshold(gray: np.ndarray, T: float) -> tuple[np.ndarray, float]:
    """Binarise with a fixed global threshold — Book Eq. (3.1) ``g = 1 if f > T else 0``, Fig. 3.2(c) (T = 125
    → IC 41.47 % on the unshipped Fig. 3.2(a)).  No ``.m`` file (``separability.m`` lines 11–14 are the same
    code).  Returns ``(bw, ic)``."""
    gray = _as_gray(gray)
    bw = gray.astype(np.float64) > T
    return bw, ice_concentration(bw)


def plot_histogram_with_threshold(ax, gray: np.ndarray, T: float, labels: bool = True, color: str = "k") -> None:
    """Draw ``imhist(gray)`` on ``ax`` with a vertical line at ``T`` and the "Background / Objects" annotation of
    Book Fig. 3.1 / Fig. 3.2(b) (text only)."""
    gray = _as_gray(gray)
    counts, x = imhist(gray)
    ax.bar(x, counts, width=1.0, color=color)
    ax.axvline(T, color="r", linestyle="--", linewidth=1.5)
    ax.set_xlim(-0.5, 255.5)
    ax.set_xlabel("Intensity value")
    ax.set_ylabel("Number of pixels")
    if labels:
        ymax = counts.max()
        ax.text(T / 2, ymax * 0.95, "Background\n(water)", ha="center", va="top")
        ax.text((T + 255) / 2, ymax * 0.95, "Objects\n(ice)", ha="center", va="top")
        ax.text(T, ymax * 0.55, f" T = {T:g}", color="r", ha="left")


# ---------------------------------------------------------------------------------------------------------------
# §3.2.2 text-only: 2-D k-means demonstration (Figs. 3.6, 3.8) and distance measures (Eqs. 3.29–3.34)
# ---------------------------------------------------------------------------------------------------------------


def kmeans_demo_2d(seed: int = 0, k: int = 2, n_per: int = 15) -> dict[str, Any]:
    """Synthetic re-creation of Book Fig. 3.6 (k-means process with random initial centroids) and Fig. 3.8
    (effect of an outlier), plus the six distances of Eqs. (3.29)–(3.34) between two sample points.

    Returns ``X`` (points), ``result`` (:class:`~seaice.core.clustering.KMeansResult` with ``history`` for the
    five panels), ``X_out`` / ``result_clean`` / ``result_outlier`` (Fig. 3.8: same clusters without and with the
    outlier; the outlier run starts from initial centroids that include the outlier, i.e. one of the "random"
    picks landed on it), ``distances`` (dict metric → value for ``a = X[0]``, ``b = X[-1]``).
    """
    X = synth.two_clusters_2d(seed=seed, n_per=n_per)
    result: KMeansResult = kmeans_lloyd(X, k, init="random", seed=seed)
    outlier = (9.8, 0.4)
    X_out = synth.two_clusters_2d(seed=seed, n_per=n_per, outlier=outlier)
    result_clean = kmeans_lloyd(X, k, init="random", seed=seed)
    init = np.vstack([X_out[0], X_out[-1]])  # one blob point and the outlier as the "random" initial centroids
    result_outlier = kmeans_lloyd(X_out, k, init=init)
    a, b = X[0], X[-1]
    distances = {m: float(pairwise_distance(a[None, :], b[None, :], m, cov=np.cov(X, rowvar=False))[0, 0])
                 for m in ("euclidean", "sqeuclidean", "chebyshev", "cityblock", "cosine", "mahalanobis")}
    return {"X": X, "result": result, "X_out": X_out, "result_clean": result_clean,
            "result_outlier": result_outlier, "outlier": outlier, "a": a, "b": b, "distances": distances}
