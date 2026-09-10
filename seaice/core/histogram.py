"""Image histograms, Book §2.2, Eqs. (2.7)–(2.8).

MATLAB source: ``MATLAB_ROOT/ch2/histogram.m`` (manual loop ``num(k+1) = length(find(I == k))``,
``GP(k+1) = num/(m*n)`` and the toolbox ``imhist``).  Reused by ch3 (Otsu / separability) and ch6/ch7 colour
statistics.
"""
from __future__ import annotations

import numpy as np

from .matlab_compat import matlab_round


def _default_nbins(img: np.ndarray, nbins: int | None) -> int:
    """MATLAB ``imhist`` default: ``n = 2`` for logical (binary) images, ``n = 256`` for all other classes."""
    if nbins is None:
        return 2 if img.dtype == np.bool_ else 256
    return int(nbins)


def imhist(img: np.ndarray, nbins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Histogram ``h(r_k) = n_k`` of an intensity image — MATLAB ``[counts, x] = imhist(I, n)``.

    Book: §2.2, Eq. (2.7) ``h(r_k) = n_k`` (number of pixels with level ``r_k``, ``k = 0..L-1``), Figs. 2.7–2.8.
    MATLAB source: ``histogram.m`` lines 25–28 (manual version) and 44, 50–52 (``imhist``).

    Parameters
    ----------
    img : ndarray
        uint8 / uint16 / bool / float image.  Floats are assumed in ``[0, 1]`` (MATLAB convention).
    nbins : int or None, default None
        Number of equally spaced bins.  ``None`` applies MATLAB's ``imhist`` rule: **2 bins for a logical
        (binary) image**, 256 bins for every other class (``imhist(BW)`` returns ``[n_false; n_true]`` with
        centres ``[0; 1]``, checked against R2025a).  For uint8 with 256 bins every bin is one gray level (exact).

    Returns
    -------
    counts : int64 array (nbins,)
    centers : float64 array (nbins,)
        Bin centres ``x``; for uint8/256 bins these are ``0, 1, ..., 255``.

    Notes
    -----
    MATLAB assigns an integer value ``v`` to bin ``round(v * (n - 1) / top)`` (bin centres ``k * top/(n-1)``);
    floats use ``round(v * (n - 1))``.  Parity: exact for uint8 / 256 bins; other bin counts follow MATLAB's
    rule but are not verified against MATLAB in ch2.
    """
    img = np.asarray(img)
    nbins = _default_nbins(img, nbins)
    if img.dtype == np.bool_:
        top, vals = 1.0, img.astype(np.float64)
    elif img.dtype == np.uint8:
        top, vals = 255.0, img.astype(np.float64)
    elif img.dtype == np.uint16:
        top, vals = 65535.0, img.astype(np.float64)
    else:
        top, vals = 1.0, np.clip(img.astype(np.float64), 0.0, 1.0)
    if img.dtype == np.uint8 and nbins == 256:
        counts = np.bincount(img.ravel(), minlength=256).astype(np.int64)
        return counts, np.arange(256, dtype=np.float64)
    idx = matlab_round(vals * (nbins - 1) / top).astype(np.int64)
    idx = np.clip(idx, 0, nbins - 1)
    counts = np.bincount(idx.ravel(), minlength=nbins).astype(np.int64)
    centers = np.arange(nbins, dtype=np.float64) * top / (nbins - 1)
    return counts, centers


def normalized_histogram(img: np.ndarray, nbins: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Normalised histogram ``p(r_k) = n_k / (M N)`` — the probability of gray level ``r_k``.

    Book: §2.2, Eq. (2.8).  MATLAB source: ``histogram.m`` line 27 ``GP(k+1) = length(find(I == k)) / (m * n)``.
    Returns ``(p, centers)`` with ``p.sum() == 1``.  ``nbins=None`` follows the same MATLAB default as
    :func:`imhist` (2 bins for logical images, 256 otherwise).
    """
    counts, centers = imhist(img, nbins)
    return counts.astype(np.float64) / float(np.asarray(img).size), centers


def hist(y: np.ndarray, bins: int | np.ndarray = 10) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB ``[counts, centers] = hist(y, bins)`` — histogram with **bin centres**, not edges.

    Book: §7.2.4 floe size distribution (Fig. 7.15) and §8.3.  MATLAB source:
    ``ch7/Sea_Ice_Floe_Identification/ice_shape_enhancement.m`` line 212 ``[z, n] = hist(floe_area, nbins)``
    (``nbins = 50``) and ``color_hist.m`` lines 18/21 ``hist(x, min_x:inter:max_x)`` (the explicit-centres form).
    Ported line by line from R2025a ``toolbox/matlab/graphics/math/hist.m`` lines 36–101.

    Parameters
    ----------
    y : array_like
        Data (flattened; non-finite values are excluded from the min/max, MATLAB lines 50–59).
    bins : int or array_like
        Scalar ``n`` → ``n`` equal-width bins spanning ``[min(y), max(y)]``, whose **centres** are returned
        (``edges = linspace(miny, maxy, n+1)``, ``x = edges(1:end-1) + binwidth/2``).  When ``min(y) == max(y)``
        MATLAB widens the range to ``[miny - floor(n/2) - 0.5, maxy + ceil(n/2) - 0.5]`` (line 71–74).
        A vector → those values are the bin **centres**; the internal edges are their midpoints and the two outer
        bins are unbounded (lines 85–87), so values outside the centre range are *counted*, not dropped.

    Returns
    -------
    counts : int64 array
    centers : float64 array

    Notes
    -----
    The comparison edges are ``edges + eps(edges)`` (line 93), i.e. the next representable float above each edge,
    so a value that sits exactly on an edge falls in the **lower** bin; ``histc``'s overflow bin is then folded
    into the last real bin (lines 98–101).  ``np.histogram`` does neither and uses edges rather than centres —
    getting this wrong shifts every FSD bar by half a bin (analysis/ch07.md risk R7).
    Parity: exact (line-by-line port of ``hist.m``).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    scalar_bins = np.isscalar(bins) or (np.ndim(bins) == 0)

    if y.size == 0:  # hist.m lines 38-45
        centers = np.arange(1.0, float(int(bins)) + 1.0) if scalar_bins else np.asarray(bins, dtype=np.float64)
        return np.zeros(centers.shape, dtype=np.int64), centers

    finite = y[np.isfinite(y)]
    miny, maxy = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)

    if scalar_bins:
        n = int(bins)
        if n < 1:
            raise ValueError("hist: the number of bins must be >= 1")
        if miny == maxy:  # hist.m lines 71-74
            miny = miny - np.floor(n / 2) - 0.5
            maxy = maxy + np.ceil(n / 2) - 0.5
        edges = np.linspace(miny, maxy, n + 1)
        binwidth = edges[1] - edges[0]
        centers = edges[:-1] + binwidth / 2.0
        edges[0], edges[-1] = -np.inf, np.inf
    else:
        centers = np.asarray(bins, dtype=np.float64).ravel()
        mid = centers[:-1] + np.diff(centers) / 2.0
        edges = np.concatenate(([-np.inf], mid, [np.inf]))

    edgesc = np.nextafter(edges, np.inf)  # = edges + eps(edges) (hist.m line 93)
    edgesc[0], edgesc[-1] = -np.inf, np.inf
    idx = np.searchsorted(edgesc, y[np.isfinite(y) | (y == np.inf)], side="right") - 1
    counts = np.bincount(np.clip(idx, 0, edgesc.size - 1), minlength=edgesc.size).astype(np.int64)
    if counts.size > 1:  # histc's overflow bin folded into the last real bin (hist.m lines 98-101)
        counts[-2] += counts[-1]
    return counts[:-1], centers
