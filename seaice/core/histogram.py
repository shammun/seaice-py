"""Image histograms, Book §2.2, Eqs. (2.7)–(2.8).

MATLAB source: ``MATLAB_ROOT/ch2/histogram.m`` (manual loop ``num(k+1) = length(find(I == k))``,
``GP(k+1) = num/(m*n)`` and the toolbox ``imhist``).  Reused by ch3 (Otsu / separability) and ch6/ch7 colour
statistics.
"""
from __future__ import annotations

import numpy as np

from .matlab_compat import matlab_round


def imhist(img: np.ndarray, nbins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Histogram ``h(r_k) = n_k`` of an intensity image — MATLAB ``[counts, x] = imhist(I, n)``.

    Book: §2.2, Eq. (2.7) ``h(r_k) = n_k`` (number of pixels with level ``r_k``, ``k = 0..L-1``), Figs. 2.7–2.8.
    MATLAB source: ``histogram.m`` lines 25–28 (manual version) and 44, 50–52 (``imhist``).

    Parameters
    ----------
    img : ndarray
        uint8 / uint16 / bool / float image.  Floats are assumed in ``[0, 1]`` (MATLAB convention).
    nbins : int, default 256
        Number of equally spaced bins.  For uint8 with ``nbins=256`` every bin is one gray level (exact).

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


def normalized_histogram(img: np.ndarray, nbins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Normalised histogram ``p(r_k) = n_k / (M N)`` — the probability of gray level ``r_k``.

    Book: §2.2, Eq. (2.8).  MATLAB source: ``histogram.m`` line 27 ``GP(k+1) = length(find(I == k)) / (m * n)``.
    Returns ``(p, centers)`` with ``p.sum() == 1``.
    """
    counts, centers = imhist(img, nbins)
    return counts.astype(np.float64) / float(np.asarray(img).size), centers
