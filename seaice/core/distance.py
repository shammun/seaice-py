"""Distance metrics and distance transforms — Book §2.4, Eqs. (2.9)–(2.12), Figs. 2.12–2.14.

MATLAB source: ``MATLAB_ROOT/ch2/distance_transform.m`` (``bwdist`` with ``'euclidean'``, ``'cityblock'``,
``'chessboard'``).  Reused by ch5 (watershed markers), ch6 (``GVF_distance.m``, ``dist.m``).

Two deliberately distinct functions:

* :func:`distance_transform` — the *book's* Eq. (2.9): distance of every **object** pixel to the nearest
  **background** pixel (= MATLAB ``bwdist(~f)``; = ``scipy.ndimage.distance_transform_edt(f)``).
* :func:`bwdist` — MATLAB semantics: distance of every pixel to the nearest **nonzero** pixel
  (= ``distance_transform(~bw)``).  ``distance_transform.m`` calls ``bwdist`` on a single-point image.

MATLAB returns ``single``; we keep float64 (differences ~1e-7 → compare with ``atol=1e-4``).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

METRICS = ("euclidean", "cityblock", "chessboard", "quasi-euclidean")
SQRT2 = np.sqrt(2.0)


def pixel_distance(p, q, metric: str = "euclidean") -> np.ndarray:
    """Distance between pixel coordinates ``p = (x, y)`` and ``q = (u, v)`` (arrays broadcast).

    Book: Eq. (2.10) Euclidean ``sqrt((x-u)² + (y-v)²)``; Eq. (2.11) city-block ``|x-u| + |y-v|``;
    Eq. (2.12) chessboard ``max(|x-u|, |y-v|)``.  ``'quasi-euclidean'`` follows MATLAB's definition
    ``max(|dx|,|dy|) + (√2 - 1) min(|dx|,|dy|)`` (length of the shortest 8-connected path with steps 1 / √2).
    Satisfies the metric axioms listed under Eq. (2.9).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    dx = np.abs(p[..., 0] - q[..., 0])
    dy = np.abs(p[..., 1] - q[..., 1])
    if metric == "euclidean":
        return np.sqrt(dx ** 2 + dy ** 2)  # Eq. (2.10)
    if metric == "cityblock":
        return dx + dy  # Eq. (2.11)
    if metric == "chessboard":
        return np.maximum(dx, dy)  # Eq. (2.12)
    if metric == "quasi-euclidean":
        return np.maximum(dx, dy) + (SQRT2 - 1.0) * np.minimum(dx, dy)
    raise ValueError(f"metric must be one of {METRICS}")


def quasi_euclidean_dt(obj: np.ndarray) -> np.ndarray:
    """Quasi-Euclidean distance of every nonzero pixel of ``obj`` to the nearest zero pixel (chamfer 1, √2).

    Two-pass (forward/backward raster) chamfer transform with a 3×3 mask, which is *exact* for the weighted
    8-connected path metric (MATLAB ``bwdist(~obj, 'quasi-euclidean')``).  Each row is processed with a
    vectorised prefix/suffix minimum (``d[c] = min_k d[k] + |c - k|``).  Pixels with no background anywhere get
    ``inf`` (MATLAB returns ``Inf`` too).

    Book: §2.4 (metric family of Eqs. 2.10–2.12; not used in ch2 figures — provided for ch5).
    Parity: reimplemented.
    """
    obj = np.asarray(obj) != 0
    M, N = obj.shape
    d = np.where(obj, np.inf, 0.0)
    if not obj.any() or obj.all():
        return d
    cols = np.arange(N, dtype=np.float64)

    def row_relax(row: np.ndarray) -> np.ndarray:
        # horizontal step cost 1 in both directions: min_k row[k] + |c - k|
        fwd = np.minimum.accumulate(row - cols) + cols
        bwd = (np.minimum.accumulate((row + cols)[::-1]))[::-1] - cols
        return np.minimum(fwd, bwd)

    # forward pass: rows top→bottom
    d[0] = row_relax(d[0])
    for r in range(1, M):
        up = d[r - 1]
        cand = np.minimum(d[r], up + 1.0)
        cand[1:] = np.minimum(cand[1:], up[:-1] + SQRT2)
        cand[:-1] = np.minimum(cand[:-1], up[1:] + SQRT2)
        d[r] = row_relax(cand)
    # backward pass: rows bottom→top
    for r in range(M - 2, -1, -1):
        dn = d[r + 1]
        cand = np.minimum(d[r], dn + 1.0)
        cand[1:] = np.minimum(cand[1:], dn[:-1] + SQRT2)
        cand[:-1] = np.minimum(cand[:-1], dn[1:] + SQRT2)
        d[r] = row_relax(cand)
    return d


def distance_transform(bw: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Distance transform of a binary image, Book Eq. (2.9): ``D(p) = 0`` for background, ``min_{q∈B} d(p, q)`` for object.

    Book: §2.4, Eqs. (2.9)–(2.12), Fig. 2.12 (7×7 example, Euclidean), Fig. 2.13.  MATLAB equivalent:
    ``bwdist(~bw, metric)``.

    Parameters
    ----------
    bw : ndarray (M, N)
        Nonzero = object ``O``, zero = background ``B``.
    metric : {'euclidean', 'cityblock', 'chessboard', 'quasi-euclidean'}

    Returns
    -------
    float64 array (M, N).  ``inf`` where the image has no background pixel (MATLAB gives ``Inf``).

    Parity: euclidean → ``scipy.ndimage.distance_transform_edt`` (exact EDT, same as MATLAB since R2009a);
    cityblock/chessboard → ``distance_transform_cdt`` (exact integers); quasi-euclidean → reimplemented.
    """
    obj = np.asarray(bw) != 0
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    if not obj.any():
        return np.zeros(obj.shape, dtype=np.float64)
    if obj.all():
        return np.full(obj.shape, np.inf, dtype=np.float64)
    if metric == "euclidean":
        return ndimage.distance_transform_edt(obj).astype(np.float64)
    if metric == "cityblock":
        return ndimage.distance_transform_cdt(obj, metric="taxicab").astype(np.float64)
    if metric == "chessboard":
        return ndimage.distance_transform_cdt(obj, metric="chessboard").astype(np.float64)
    return quasi_euclidean_dt(obj)


def bwdist(bw: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """MATLAB ``bwdist(BW, metric)``: distance from every pixel to the nearest **nonzero** pixel of ``BW``.

    Book: §2.4.4, Fig. 2.14 (distance maps from a single centre pixel).  MATLAB source:
    ``MATLAB_ROOT/ch2/distance_transform.m`` (``imgDist = -bwdist(~img, metric)`` with ``img = ~point``).
    Note the complement relative to Eq. (2.9): ``bwdist(BW) == distance_transform(~BW)``.
    """
    return distance_transform(~(np.asarray(bw) != 0), metric)


def center_distance_map(size: int = 7, metric: str = "cityblock") -> np.ndarray:
    """Distance of every pixel of a ``size × size`` grid from its centre pixel (Book Fig. 2.13; ``size`` odd).

    Equals ``bwdist(point)``; written with :func:`pixel_distance` so the equations (2.10)–(2.12) are exercised
    directly.
    """
    if size % 2 == 0:
        raise ValueError("size must be odd so the centre is a pixel")
    c = size // 2
    rr, cc = np.mgrid[0:size, 0:size]
    return pixel_distance(np.stack([rr, cc], axis=-1), np.array([c, c]), metric)
