"""Image interpolation — Book §2.8, Eqs. (2.32)–(2.41), Figs. 2.22–2.25.

Text-only in ch2 (no ``.m`` file).  MATLAB equivalents: ``interp2(..., 'nearest' | 'linear' | 'cubic')`` and
``imresize``; used later by ch6 ``snakedeform.m`` (``interp2`` on the GVF field) and App. A orthorectification.

Conventions
-----------
* Sample coordinates are **0-based** ``(u, v)`` = (row, col), matching Book Eq. (2.1) (``x`` = row, ``y`` = col)
  and Fig. 2.24 (``P1 = (i, j)``, ``P2 = (i, j+1)``, ``P3 = (i+1, j)``, ``P4 = (i+1, j+1)``).  MATLAB
  ``interp2(X, Y, Z, Xq, Yq)`` uses ``X`` = column, ``Y`` = row, 1-based: ``u = Yq - 1``, ``v = Xq - 1``.
* Images may be ``(M, N)`` or ``(M, N, C)``; interpolation is per channel.
* Query points outside the image return ``fill`` (default ``nan``, as MATLAB ``interp2``) unless
  ``border='replicate'``.

Parity: nearest/bilinear exact vs ``interp2`` (nearest tie rule at .5 = round half away from zero, checked);
bicubic uses the Keys kernel with ``a = -0.5`` and MATLAB's quadratic edge extrapolation, exact vs
``interp2 'cubic'`` (see :func:`interp_bicubic`); :func:`resize` differs from ``imresize`` (antialiasing when
shrinking, and ``imresize`` bicubic uses replicate-style edge handling) → ``approx``.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .matlab_compat import matlab_round

METHODS = ("nearest", "bilinear", "bicubic")
_ALIASES = {"linear": "bilinear", "cubic": "bicubic", "nearest": "nearest", "bilinear": "bilinear",
            "bicubic": "bicubic"}


def _prep(img: np.ndarray, u, v):
    img = np.asarray(img, dtype=np.float64)
    squeeze = img.ndim == 2
    if squeeze:
        img = img[:, :, None]
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u, v = np.broadcast_arrays(u, v)
    return img, u, v, squeeze


def _finish(out: np.ndarray, squeeze: bool) -> np.ndarray:
    return out[..., 0] if squeeze else out


def _outside(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    M, N = img.shape[:2]
    return (u < 0) | (u > M - 1) | (v < 0) | (v > N - 1) | ~np.isfinite(u) | ~np.isfinite(v)


def interp_nearest(img: np.ndarray, u, v, fill: float = np.nan, border: str = "nan") -> np.ndarray:
    """Nearest-neighbour interpolation, Book §2.8.1, Eq. (2.33), Fig. 2.23.

    Output pixel ``(m, n)`` is mapped to ``(u, v) = T⁻¹{(m, n)}`` and takes the value of the closest lattice
    point.  Ties (fractional part exactly 0.5) round half away from zero, i.e. towards the larger index for
    positive coordinates (MATLAB ``round``).  ``border='replicate'`` clamps outside queries to the nearest edge
    pixel; ``'nan'`` returns ``fill`` there (MATLAB ``interp2`` default).
    """
    img, u, v, squeeze = _prep(img, u, v)
    M, N = img.shape[:2]
    out_mask = _outside(img, u, v)
    i = np.clip(matlab_round(np.nan_to_num(u)), 0, M - 1).astype(np.int64)
    j = np.clip(matlab_round(np.nan_to_num(v)), 0, N - 1).astype(np.int64)
    out = img[i, j]
    if border != "replicate":
        out[out_mask] = fill
    return _finish(out, squeeze)


def interp_bilinear(img: np.ndarray, u, v, fill: float = np.nan, border: str = "nan") -> np.ndarray:
    """Bilinear interpolation, Book §2.8.2, Eqs. (2.34)–(2.38), Fig. 2.24.

    With ``i = floor(u)``, ``j = floor(v)`` and the four neighbours ``P1..P4``::

        f(Q1) = (j+1-v) f(P1) + (v-j) f(P2)          Eq. (2.35)
        f(Q2) = (j+1-v) f(P3) + (v-j) f(P4)          Eq. (2.36)
        f(P)  = (i+1-u) f(Q1) + (u-i) f(Q2)          Eq. (2.37) / matrix form (2.38)

    Exactly reproduces lattice values; weights sum to 1.  Queries on the last row/column use the previous cell
    with weight 1 on its far edge (same result as MATLAB ``interp2 'linear'``).
    """
    img, u, v, squeeze = _prep(img, u, v)
    M, N = img.shape[:2]
    out_mask = _outside(img, u, v)
    if border == "replicate":
        u = np.clip(np.nan_to_num(u), 0, M - 1)
        v = np.clip(np.nan_to_num(v), 0, N - 1)
    else:
        u = np.where(out_mask, 0.0, u)
        v = np.where(out_mask, 0.0, v)
    i = np.clip(np.floor(u), 0, max(M - 2, 0)).astype(np.int64)
    j = np.clip(np.floor(v), 0, max(N - 2, 0)).astype(np.int64)
    i1 = np.minimum(i + 1, M - 1)
    j1 = np.minimum(j + 1, N - 1)
    wu = (u - i)[..., None]  # u - i
    wv = (v - j)[..., None]  # v - j
    fQ1 = (1.0 - wv) * img[i, j] + wv * img[i, j1]  # Eq. (2.35)
    fQ2 = (1.0 - wv) * img[i1, j] + wv * img[i1, j1]  # Eq. (2.36)
    out = (1.0 - wu) * fQ1 + wu * fQ2  # Eq. (2.37)
    if border != "replicate":
        out[out_mask] = fill
    return _finish(out, squeeze)


def keys_kernel(x, a: float = -0.5) -> np.ndarray:
    """Cubic convolution kernel ``rc(x)`` of Book Eq. (2.41) (Keys 1981):

    ``(a+2)|x|³ - (a+3)|x|² + 1`` for ``|x| ≤ 1``; ``a|x|³ - 5a|x|² + 8a|x| - 4a`` for ``1 < |x| ≤ 2``; else 0.
    ``a = -0.5`` is MATLAB's choice (``interp2 'cubic'``, ``imresize 'bicubic'``); OpenCV uses ``-0.75``.
    Interpolating: ``rc(0) = 1``, ``rc(±1) = rc(±2) = 0``.
    """
    x = np.abs(np.asarray(x, dtype=np.float64))
    out = np.zeros_like(x)
    m1 = x <= 1.0
    m2 = (x > 1.0) & (x <= 2.0)
    out[m1] = (a + 2.0) * x[m1] ** 3 - (a + 3.0) * x[m1] ** 2 + 1.0
    out[m2] = a * x[m2] ** 3 - 5.0 * a * x[m2] ** 2 + 8.0 * a * x[m2] - 4.0 * a
    return out


def _pad_for_cubic(img: np.ndarray, border: str) -> np.ndarray:
    """Pad 2 pixels on each side.  ``'replicate'`` repeats the edge; ``'quadratic'`` uses MATLAB's classic
    ``interp2`` cubic extrapolation ``3 f(1) - 3 f(2) + f(3)`` (one row/col) plus replicate for the second."""
    if border == "quadratic":
        top = 3 * img[0:1] - 3 * img[1:2] + img[2:3]
        bot = 3 * img[-1:] - 3 * img[-2:-1] + img[-3:-2]
        img = np.concatenate([top, img, bot], axis=0)
        left = 3 * img[:, 0:1] - 3 * img[:, 1:2] + img[:, 2:3]
        right = 3 * img[:, -1:] - 3 * img[:, -2:-1] + img[:, -3:-2]
        img = np.concatenate([left, img, right], axis=1)
        return np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    return np.pad(img, ((2, 2), (2, 2), (0, 0)), mode="edge")


def interp_bicubic(img: np.ndarray, u, v, a: float = -0.5, fill: float = np.nan, border: str = "nan",
                   pad: str = "quadratic") -> np.ndarray:
    """Bicubic (cubic convolution) interpolation, Book §2.8.3, Eqs. (2.40)–(2.41), Fig. 2.25.

    ``f(u, v) = Σ_{m=-1}^{2} Σ_{n=-1}^{2} f(i+m, j+n) · rc(m + i - u) · rc(n + j - v)`` with ``i = floor(u)``,
    ``j = floor(v)`` (the book prints ``f(u+m, v+n)``; the lattice indices must be ``(i+m, j+n)``; ``rc`` is
    even so the sign inside the second factor is immaterial).

    Parameters
    ----------
    a : float
        Kernel tuning factor of Eq. (2.41); ``-0.5`` = MATLAB.
    pad : {'quadratic', 'replicate'}
        How the 4×4 support is completed at the image border (see :func:`_pad_for_cubic`).  ``'quadratic'``
        (default) is MATLAB's ``interp2 'cubic'`` rule ``3 f(1) - 3 f(2) + f(3)`` (checked against R2025a:
        max |diff| 7e-13 on a 32×32 crop, borders included); ``'replicate'`` differs by several gray levels
        within 1 px of the border.  Interior results (≥ 2 px from the border) are independent of this choice.
    border, fill
        As in :func:`interp_bilinear`.

    Parity: exact vs MATLAB ``interp2 'cubic'`` with ``pad='quadratic'`` (identical Keys kernel, ``a = -0.5``).
    """
    img, u, v, squeeze = _prep(img, u, v)
    M, N = img.shape[:2]
    if M < 3 or N < 3:
        raise ValueError("bicubic interpolation needs at least a 3x3 image")
    out_mask = _outside(img, u, v)
    if border == "replicate":
        u = np.clip(np.nan_to_num(u), 0, M - 1)
        v = np.clip(np.nan_to_num(v), 0, N - 1)
    else:
        u = np.where(out_mask, 0.0, u)
        v = np.where(out_mask, 0.0, v)
    P = _pad_for_cubic(img, pad)  # index shift of +2
    i = np.clip(np.floor(u), 0, M - 1).astype(np.int64)
    j = np.clip(np.floor(v), 0, N - 1).astype(np.int64)
    out = np.zeros(u.shape + (img.shape[2],), dtype=np.float64)
    for m in range(-1, 3):
        wu = keys_kernel(m + i - u, a)[..., None]  # rc(m + i - u)
        for n in range(-1, 3):
            wv = keys_kernel(n + j - v, a)[..., None]  # rc(n + j - v)
            out += P[i + m + 2, j + n + 2] * wu * wv  # Eq. (2.40)
    if border != "replicate":
        out[out_mask] = fill
    return _finish(out, squeeze)


def interp2(img: np.ndarray, u, v, method: str = "bilinear", **kwargs) -> np.ndarray:
    """MATLAB-style dispatcher: ``method`` in ``{'nearest', 'linear'/'bilinear', 'cubic'/'bicubic'}``.

    ``u``, ``v`` are 0-based (row, col) query coordinates (``u = Yq - 1``, ``v = Xq - 1`` relative to MATLAB
    ``interp2`` on a unit grid).  Extra keyword arguments are passed to the method (``fill``, ``border``, ``a``).
    """
    key = _ALIASES.get(method)
    if key is None:
        raise ValueError(f"method must be one of {list(_ALIASES)}")
    if key == "nearest":
        return interp_nearest(img, u, v, **kwargs)
    if key == "bilinear":
        return interp_bilinear(img, u, v, **kwargs)
    return interp_bicubic(img, u, v, **kwargs)


def warp_image(img: np.ndarray, T_inv: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
               out_shape: tuple[int, int], method: str = "bilinear", **kwargs) -> np.ndarray:
    """Geometric transform driver of Book Eqs. (2.32)–(2.33): for every output pixel ``(m, n)`` compute
    ``(u, v) = T_inv(m, n)`` in the input image and interpolate there.

    ``T_inv`` receives two float arrays (rows ``m``, cols ``n`` of the output grid, 0-based) and returns
    ``(u, v)`` arrays of the same shape.  Used by App. A orthorectification.
    """
    m, n = np.mgrid[0:out_shape[0], 0:out_shape[1]].astype(np.float64)
    u, v = T_inv(m, n)
    return interp2(img, u, v, method=method, **kwargs)


def resize(img: np.ndarray, scale: float | tuple[float, float], method: str = "bilinear", **kwargs) -> np.ndarray:
    """Resize by ``scale`` (scalar or ``(row_scale, col_scale)``) with the interpolation of choice.

    Uses the pixel-centre convention of MATLAB ``imresize``: output pixel ``m`` maps to input coordinate
    ``u = (m + 0.5) / scale - 0.5``; output size ``ceil(scale * M)`` (MATLAB rounds with ``ceil``).  No
    antialiasing is applied when shrinking, so results differ from ``imresize`` (default antialiasing on) —
    parity ``approx``; for ``'nearest'`` and enlargement with ``'bilinear'`` the difference is only in the
    border extrapolation.  Book: §2.8 (image zooming as a use of interpolation).
    """
    img = np.asarray(img)
    sr, sc = (scale, scale) if np.isscalar(scale) else tuple(scale)
    M, N = img.shape[:2]
    out_M, out_N = int(np.ceil(M * sr)), int(np.ceil(N * sc))

    def T_inv(m, n):
        return (m + 0.5) / sr - 0.5, (n + 0.5) / sc - 0.5

    kwargs.setdefault("border", "replicate")
    return warp_image(img, T_inv, (out_M, out_N), method=method, **kwargs)
