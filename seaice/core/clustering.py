"""Clustering — distance measures, Lloyd's k-means and the authors' 1-D histogram k-means for ice pixels.

Book: Chapter 3 §3.2, Eqs. (3.29)–(3.37), Figs. 3.6–3.8, 3.9(c)–3.12(b).  MATLAB source: ``MATLAB_ROOT/ch3/kmeans.m``
— the authors' own script (it *shadows* the Statistics Toolbox ``kmeans``; the algorithm is deterministic:
equal-division initialisation, 1-D absolute-difference distances on the gray-level histogram).  The 2-D
demonstrations of Figs. 3.6 and 3.8 have no ``.m`` file and are re-implemented from Eqs. (3.35)–(3.37).

Later chapters (ch6/ch7/ch9) call the *Statistics Toolbox* ``kmeans(..., 'EmptyAction', 'singleton')`` (k-means++,
random) — a different algorithm; :func:`kmeans_lloyd` leaves room for it (``init='kmeans++'``) but that mapping is
decided in ch6.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import distance as _sd

from .matlab_compat import rgb2gray_matlab

# ---------------------------------------------------------------------------------------------------------------
# Distance measures, Eqs. (3.29)–(3.34)
# ---------------------------------------------------------------------------------------------------------------

_METRICS = {
    "euclidean": "euclidean",  # Eq. (3.29)
    "sqeuclidean": "sqeuclidean",  # Eq. (3.30)
    "chebyshev": "chebyshev",  # Eq. (3.31) maximum norm
    "cityblock": "cityblock",  # Eq. (3.32)
    "cosine": "cosine",  # Eq. (3.33) — see note
    "mahalanobis": "mahalanobis",  # Eq. (3.34)
}


def pairwise_distance(X: np.ndarray, Y: np.ndarray, metric: str = "euclidean",
                      cov: np.ndarray | None = None) -> np.ndarray:
    """Distance between every row of ``X`` and every row of ``Y`` — Book §3.2, Eqs. (3.29)–(3.34) (text only).

    Parameters
    ----------
    X, Y : (n, d), (m, d) arrays
        Points (a 1-D array is treated as ``n`` scalars).
    metric : {'euclidean', 'sqeuclidean', 'chebyshev', 'cityblock', 'cosine', 'mahalanobis'}
        ``'chebyshev'`` is the book's *maximum norm* (Eq. 3.31).  ``'cosine'``: the book's Eq. (3.33) prints the
        cosine **similarity** ``aᵀb / (|a||b|)`` and calls it a distance; this function returns the usual cosine
        *distance* ``1 − aᵀb/(|a||b|)`` (0 for parallel vectors) so that smaller means closer, like the other five.
        ``'mahalanobis'`` uses ``S = cov`` (Eq. 3.34) or, if ``None``, the covariance of the pooled rows of ``X``
        and ``Y``.

    Returns
    -------
    (n, m) float64 array.  Parity: exact (closed forms via ``scipy.spatial.distance.cdist``).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]  # n scalars
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"pairwise_distance: dimension mismatch {X.shape} vs {Y.shape}")
    if metric not in _METRICS:
        raise ValueError(f"pairwise_distance: unknown metric {metric!r}; choose one of {sorted(_METRICS)}")
    if metric == "mahalanobis":
        S = np.cov(np.vstack([X, Y]), rowvar=False) if cov is None else np.asarray(cov, dtype=np.float64)
        S = np.atleast_2d(S)
        return _sd.cdist(X, Y, metric="mahalanobis", VI=np.linalg.inv(S))
    return _sd.cdist(X, Y, metric=_METRICS[metric])


# ---------------------------------------------------------------------------------------------------------------
# Lloyd's k-means from Eqs. (3.35)–(3.37) (Figs. 3.6, 3.8)
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class KMeansResult:
    """Output of :func:`kmeans_lloyd`.

    Attributes
    ----------
    centers : (k, d) final centroids ``c_i`` (Eq. 3.37)
    labels : (n,) 0-based cluster index per point (Eq. 3.36)
    J : within-cluster sum of squared distances at convergence (Eq. 3.35)
    n_iter : number of assignment/update iterations performed
    converged : True if the centroids stopped changing before ``max_iter``
    history : list of ``(centers, labels, J)`` after every iteration (``history[0]`` = initial centroids with the
        first assignment), for step-by-step figures like Fig. 3.6
    """

    centers: np.ndarray
    labels: np.ndarray
    J: float
    n_iter: int
    converged: bool
    history: list[tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)


def _assign(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (3.36): nearest centroid by squared Euclidean distance; ties → lowest cluster index."""
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1)
    return labels, d2[np.arange(X.shape[0]), labels]


def objective_J(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """Eq. (3.35): ``J = Σ_i Σ_{x ∈ S_i} ‖x − c_i‖²``."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return float(((X - centers[labels]) ** 2).sum())


def kmeans_lloyd(X: np.ndarray, k: int, init: str | np.ndarray = "random", seed: int | None = 0,
                 max_iter: int = 100, tol: float = 0.0) -> KMeansResult:
    """Lloyd's k-means, Book §3.2.2 Steps 1–4, Eqs. (3.35)–(3.37) — reference implementation for Figs. 3.6/3.8.

    No MATLAB file: the book's 2-D examples are text only (the authors' ``kmeans.m`` is the 1-D histogram version,
    :func:`kmeans_gray`; the Statistics Toolbox ``kmeans`` used in ch6+ is k-means++/random and *not* this).

    Parameters
    ----------
    X : (n, d) array (or (n,) for 1-D data)
    k : number of clusters
    init : {'random', 'equal', 'kmeans++'} or (k, d) array
        ``'random'`` = ``k`` distinct data points chosen with ``seed`` (Fig. 3.6(b) "select initial centroids at
        random"); ``'equal'`` = equal-division points ``min + i (max − min)/(k+1)`` per coordinate (the
        initialisation of ``kmeans.m`` / §3.3); ``'kmeans++'`` = D² sampling (for later chapters); an array is used
        as given (Step 1 "specified by the user").
    seed : int or None
        Seed for ``'random'`` / ``'kmeans++'``.
    max_iter : int
        Safety bound on iterations.
    tol : float
        Stop when ``max |c_new − c_old| ≤ tol``; the default ``0.0`` is the book's Step 4 (centroids *unchanged*),
        which Lloyd's iteration reaches exactly in finitely many steps because the assignment is deterministic.

    Returns
    -------
    KMeansResult

    Notes
    -----
    Empty clusters keep their previous centroid (documented choice; the book does not say, MATLAB's Statistics
    ``kmeans`` would error or apply ``'EmptyAction'``).  Ties in Eq. (3.36) go to the lowest cluster index (as
    ``find(c == min(c))`` → ``cc(1)`` in ``kmeans.m``).  Parity: reimplemented (text only); property tests:
    ``J`` non-increasing, agreement with :func:`kmeans_gray` on 1-D data with ``init='equal'``.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    if k < 1 or k > n:
        raise ValueError(f"kmeans_lloyd: need 1 <= k <= n, got k={k}, n={n}")
    rng = np.random.default_rng(seed)
    if isinstance(init, str):
        if init == "random":
            centers = X[rng.choice(n, size=k, replace=False)].copy()
        elif init == "equal":
            lo, hi = X.min(axis=0), X.max(axis=0)
            centers = np.array([lo + (i + 1) * (hi - lo) / (k + 1) for i in range(k)])
        elif init == "kmeans++":
            centers = np.empty((k, d))
            centers[0] = X[rng.integers(n)]
            for i in range(1, k):
                d2 = ((X[:, None, :] - centers[None, :i, :]) ** 2).sum(axis=2).min(axis=1)
                probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
                centers[i] = X[rng.choice(n, p=probs)]
        else:
            raise ValueError(f"kmeans_lloyd: unknown init {init!r}")
    else:
        centers = np.array(init, dtype=np.float64).reshape(k, d).copy()

    history: list[tuple[np.ndarray, np.ndarray, float]] = []
    labels, _ = _assign(X, centers)
    history.append((centers.copy(), labels.copy(), objective_J(X, centers, labels)))
    converged = False
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        new_centers = centers.copy()
        for i in range(k):  # Eq. (3.37)
            sel = labels == i
            if sel.any():
                new_centers[i] = X[sel].mean(axis=0)
        labels, _ = _assign(X, new_centers)  # Eq. (3.36)
        history.append((new_centers.copy(), labels.copy(), objective_J(X, new_centers, labels)))
        change = float(np.max(np.abs(new_centers - centers)))
        centers = new_centers
        if change <= tol:
            converged = True
            break
    return KMeansResult(centers=centers, labels=labels, J=history[-1][2], n_iter=n_iter, converged=converged,
                        history=history)


# ---------------------------------------------------------------------------------------------------------------
# The authors' 1-D histogram k-means (kmeans.m)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KMeansGray:
    """Output of :func:`kmeans_gray` (names follow ``kmeans.m``).

    Attributes
    ----------
    centroids : (k,) final centroids in **gray-level units** (``mu + min − 1``)
    centroids_shifted : (k,) the script's ``mu`` (in ``ima − min + 1`` units)
    offset : the script's ``mi`` (minimum gray level); shifted value = gray − offset + 1
    mask : (M, N) int64 cluster index ``1..k`` per pixel (script ``mask``)
    mask1 : (M, N) float display image ``i/k`` (script ``mask1``)
    coverage : (k,) fraction of pixels per cluster (script ``ic = n / (M N)``)
    counts : (k,) pixel counts per cluster (script ``n``)
    average_intensity : (k,) mean gray level per cluster (correct means; the script's stale-buffer values are
        reproduced by :func:`seaice.ch03_ice_pixel_detection.stale_mean_intensity`)
    n_iter : number of ``while`` iterations until ``mu == oldmu``
    converged : True (the function raises instead of looping forever)
    shift_bug : whether the mask was built with the script's units bug
    ic : ice concentration = 1 − coverage of the darkest cluster (Table 3.1/3.2 convention)
    """

    centroids: np.ndarray
    centroids_shifted: np.ndarray
    offset: float
    mask: np.ndarray
    mask1: np.ndarray
    coverage: np.ndarray
    counts: np.ndarray
    average_intensity: np.ndarray
    n_iter: int
    converged: bool
    shift_bug: bool
    ic: float


def kmeans_gray(gray: np.ndarray, k: int = 3, shift_bug: bool = False, max_iter: int = 1000) -> KMeansGray:
    """The authors' k-means on the gray-level histogram — line-by-line port of ``MATLAB_ROOT/ch3/kmeans.m``.

    Book: §3.2.2 Eqs. (3.35)–(3.37) applied to 1-D gray levels, §3.3 ("the equal division points are chosen as
    the initialization"), Figs. 3.7, 3.9(c)–3.11(c) (k = 2), Fig. 3.12(b) (k = 3), Tables 3.1–3.3.

    Algorithm (script line numbers)
    -------------------------------
    9–13   ``ima = double(gray)(:) − min + 1`` (shift so the minimum level becomes 1);
    19–28  histogram ``h(v)`` over shifted values ``v = 1..m``, ``m = max + 1``; ``ind = find(h)`` = present values;
    32     ``mu = (1:k) * m / (k+1)`` — equal-division initial centroids;
    36–58  repeat: assign every present value to ``argmin |v − mu|`` (first index on ties, line 44), update
           ``mu(i) = Σ_a a h(a) / Σ_a h(a)`` (histogram-weighted mean, Eq. 3.37), stop when ``mu == oldmu`` exactly;
    61–70  ``mask(i,j) = argmin |copy(i,j) − mu|`` — **units bug**: ``copy`` is the *unshifted* image while ``mu``
           lives in shifted units, so every class boundary is ``min − 1`` levels too low (20 on ``test.jpg``);
    72–88  per-cluster counts ``n``, coverages ``ic = n/(M N)``, ``mask1 = i/k`` and ``average_intensity``.

    Parameters
    ----------
    gray : (M, N) uint8 (or RGB, converted with ``rgb2gray_matlab``)
    k : number of clusters
    shift_bug : bool
        ``False`` (default): the mask compares the *shifted* image with ``mu`` — Eqs. (3.36)–(3.37) applied
        consistently (the text is the authority).  ``True``: reproduce lines 64–70 literally; **every k-means number
        printed in the book (15.65 %, 32.49 %, 96.50 %, 77.91/19.20/2.89 %, Table 3.3) requires this**.
    max_iter : int
        Guard against non-termination.

    Returns
    -------
    KMeansGray

    Deviation from MATLAB
    ---------------------
    * An empty cluster makes the script's ``mu(i) = 0/0 = NaN`` and, since ``NaN == NaN`` is false, loops forever;
      this port raises ``ValueError`` instead.  ``max_iter`` exhaustion raises ``RuntimeError``.
    * ``average_intensity`` is the true per-cluster mean; the script's ``ss`` buffer (line 77) is never cleared,
      so its value is wrong whenever a cluster is smaller than an earlier one (harmless on ``test.jpg`` with k = 3).
    * ``mask1`` is returned as the gray display image ``i/k``; the script's ``colormap('default')`` (parula) is a
      display choice not reproduced (book Fig. 3.12(b) is gray).

    Parity: exact with ``shift_bug=True`` (``mu`` to 1e-9, ``mask`` identical, coverages / means to 4 decimals).
    """
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = rgb2gray_matlab(gray)
    if k < 1:
        raise ValueError("kmeans_gray: k must be >= 1")
    copy = gray.astype(np.float64)  # copy = ima (double)
    ima = copy.ravel(order="F")  # ima(:) (order irrelevant for the histogram)
    mi = float(ima.min())  # mi = min(ima)
    ima = ima - mi + 1.0  # shift: min → 1
    m = int(ima.max()) + 1  # m = max(ima) + 1  (values are integers for uint8 input)
    if not np.all(ima == np.floor(ima)):
        raise ValueError("kmeans_gray: gray levels must be integers (MATLAB indexes h(ima(i)))")
    h = np.bincount(ima.astype(np.int64), minlength=m + 1)[1:m + 1].astype(np.float64)  # h(1..m)
    ind = np.flatnonzero(h) + 1  # find(h): present shifted values (1-based values)
    mu = np.arange(1, k + 1, dtype=np.float64) * m / (k + 1)  # equal division points
    hc = np.zeros(m + 1, dtype=np.int64)  # hc(1..m), index 0 unused
    n_iter = 0
    converged = False
    for n_iter in range(1, max_iter + 1):
        oldmu = mu.copy()
        c = np.abs(ind[:, None].astype(np.float64) - mu[None, :])  # |value − mu| for all present values
        hc[ind] = np.argmin(c, axis=1) + 1  # cc(1): first index attaining the minimum
        for i in range(1, k + 1):
            a = np.flatnonzero(hc == i)  # a = find(hc == i) (1-based values because hc index = value)
            denom = h[a - 1].sum()
            if denom == 0:
                raise ValueError(f"kmeans_gray: cluster {i} became empty (MATLAB's kmeans.m would loop forever "
                                 "on the resulting NaN centroid); reduce k")
            mu[i - 1] = np.sum(a * h[a - 1]) / denom
        if np.array_equal(mu, oldmu):  # if (mu == oldmu) — all equal, exact
            converged = True
            break
    if not converged:
        raise RuntimeError(f"kmeans_gray: no convergence after {max_iter} iterations")

    # mask (lines 61–70): nearest centroid per pixel, first index on ties.  The image is integer-valued, so the
    # per-pixel argmin is a lookup over the gray levels (identical result, no (M, N, k) temporary).
    lo, hi = int(copy.min()), int(copy.max())
    levels = np.arange(lo, hi + 1, dtype=np.float64)
    values = levels if shift_bug else levels - mi + 1.0  # script compares the UNSHIFTED copy with shifted mu
    lut = np.argmin(np.abs(values[:, None] - mu[None, :]), axis=1) + 1  # a(1)
    mask = lut[copy.astype(np.int64) - lo]
    counts = np.array([np.count_nonzero(mask == i) for i in range(1, k + 1)], dtype=np.int64)
    coverage = counts / mask.size  # ic = n / (s(1) * s(2))
    mask1 = mask.astype(np.float64) / k  # mask1(p) = 1/k * i
    avg = np.array([copy[mask == i].mean() if counts[i - 1] else np.nan for i in range(1, k + 1)])
    centroids = mu + mi - 1.0  # back to gray-level units
    darkest = int(np.argmin(mu))
    ic = float(1.0 - coverage[darkest]) if k > 1 else 1.0
    return KMeansGray(centroids=centroids, centroids_shifted=mu.copy(), offset=mi, mask=mask, mask1=mask1,
                      coverage=coverage, counts=counts, average_intensity=avg, n_iter=n_iter, converged=converged,
                      shift_bug=shift_bug, ic=ic)
