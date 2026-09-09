"""MATLAB ``edge`` re-implemented — Sobel / Prewitt / Roberts gradient thresholding with thinning, and LoG /
zero-crossing detection.

Book: Chapter 4 §4.1 (Eqs. 4.1–4.15, Figs. 4.3 and 4.6).  MATLAB source: ``MATLAB_ROOT/ch4/derivative.m`` line 7
(``BW = edge(im, 'sobel', 0.05)``); the algorithm is ported from R2025a ``toolbox/images/images/edge.m`` (readable
branches) plus the compiled ``images.internal.builtins.edgesobelprewitt`` / ``computeEdges`` builtins, whose
behaviour was reverse-engineered and confirmed pixel-exact on the book image (analysis/ch04.md §2).

What MATLAB's ``edge`` adds on top of the book's Eqs. (4.2)/(4.4) ``|∇f| > T``:

* gradient kernels are ``fspecial('sobel')/8`` (Prewitt ``/6``) — true derivative estimates on a ``[0, 1]`` image;
* ``imfilter(..., 'replicate')`` padding;
* the squared magnitude ``b = bx² + by²`` is compared with ``cutoff = T²`` (Eq. 4.4), or, when ``T`` is omitted,
  with ``cutoff = 4·mean(b)`` and the returned threshold is ``sqrt(cutoff)`` (Pratt's RMS noise estimate);
* **thinning**: a pixel is an edge only if it is a local maximum of ``b`` along the dominant gradient direction
  (see :func:`thin_gradient`).

``skimage.filters.sobel`` returns a magnitude only and ``skimage.feature.canny`` is a different algorithm — neither
is used here.  Canny is not part of the book (raise ``NotImplementedError``).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .filters import fspecial, imfilter

#: Tolerance used by MATLAB's ``computeEdges`` builtin in the dominant-direction test (``100*eps``).
THIN_EPS = 100.0 * np.finfo(np.float64).eps

_METHODS = ("sobel", "prewitt", "roberts", "log", "zerocross", "canny", "approxcanny", "canny_old")
_DIRECTIONS = ("both", "horizontal", "vertical")


class EdgeResult(NamedTuple):
    """Return value of :func:`edge` — MATLAB's ``[BW, thresh, gv, gh] = edge(...)``.

    ``bw`` bool edge map; ``thresh`` the threshold actually used (``T`` as given, or the automatic one);
    ``gv``/``gh`` the vertical-edge / horizontal-edge gradient images (``bx``/``by``; ``None`` for ``'log'``).
    """

    bw: np.ndarray
    thresh: float
    gv: np.ndarray | None
    gh: np.ndarray | None


def gradient_sobel_prewitt(a: np.ndarray, kind: str = "sobel", kx: float = 1.0, ky: float = 1.0
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``[bx, by, b] = images.internal.builtins.edgesobelprewitt(a, isSobel, kx, ky)`` (R2025a ``edge.m`` line 401).

    ``op = fspecial(kind) / 8`` (Sobel) or ``/ 6`` (Prewitt); ``bx = imfilter(a, op', 'replicate')`` (derivative
    along columns → vertical edges = MATLAB's 4th output ``gv_45``), ``by = imfilter(a, op, 'replicate')``
    (derivative along rows → horizontal edges = ``gh_135``); ``b = kx·bx² + ky·by²`` (Book Eq. 4.4 on scaled
    kernels).  ``kx, ky`` = (1, 1) for ``'both'``, (0, 1) for ``'horizontal'``, (1, 0) for ``'vertical'``.

    Relation to the book's Eq. (4.7) operators (Fig. 4.2(b) kernels, unscaled): ``Gx_book = −8·by``,
    ``Gy_book = −8·bx`` (see :func:`seaice.ch04_ice_edge_detection.gradient_operator`).
    """
    a = np.asarray(a, dtype=np.float64)
    if kind == "sobel":
        op, scale = fspecial("sobel"), 8.0
    elif kind == "prewitt":
        op, scale = fspecial("prewitt"), 6.0
    else:
        raise ValueError("kind must be 'sobel' or 'prewitt'")
    # The builtin filters with the integer kernel and divides afterwards: ``imfilter(a, op)/6`` is bit-identical
    # to MATLAB while ``imfilter(a, op/6)`` differs by 1 ulp on ~1e-16 of the pixels (1/6 is inexact; confirmed
    # against R2025a gv/gh on test.jpg).  For Sobel (/8) both orders are exact.
    bx = imfilter(a, op.T, "replicate") / scale
    by = imfilter(a, op, "replicate") / scale
    b = kx * bx * bx + ky * by * by
    return bx, by, b


def gradient_roberts(a: np.ndarray, kx: float = 1.0, ky: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roberts branch of ``edge.m`` (lines 403–409): masks ``[1 0; 0 -1]/2`` and ``[0 1; -1 0]/2``, replicate padding.

    Not in the book (optional); the 2×2 masks are centred at element (1,1) like MATLAB ``imfilter``.
    """
    a = np.asarray(a, dtype=np.float64)
    x_mask = np.array([[1.0, 0.0], [0.0, -1.0]]) / 2.0
    y_mask = np.array([[0.0, 1.0], [-1.0, 0.0]]) / 2.0
    bx = imfilter(a, x_mask, "replicate")
    by = imfilter(a, y_mask, "replicate")
    b = kx * bx * bx + ky * by * by
    return bx, by, b


def thin_gradient(b: np.ndarray, bx: np.ndarray, by: np.ndarray, kx: float, ky: float,
                  offset: tuple[int, int, int, int] = (0, 0, 0, 0), cutoff: float = 0.0,
                  eps: float = THIN_EPS) -> np.ndarray:
    """MATLAB's compiled ``computeEdges(b, bx, by, kx, ky, offset, 100*eps, cutoff)`` — gradient thinning.

    Rule (reverse-engineered and confirmed with 0 differing pixels against R2025a on the 12.25-Mpx book image and
    on ramp / step / diagonal-tie / random fixtures for every method and direction): with ``bx = |bx|``,
    ``by = |by|`` and ``b`` **zero-padded** by one pixel (an out-of-image neighbour never blocks a maximum),

    .. code-block:: text

        e = b > cutoff & ( (bx >= kx*by - eps & b(r, c-1) <= b(r, c) & b(r, c) > b(r, c+1))
                         | (by >= ky*bx - eps & b(r-1, c) <= b(r, c) & b(r, c) > b(r+1, c)) )

    i.e. a pixel survives if it is a (non-strict on the left/up side, strict on the right/down side) local maximum
    of the squared magnitude along its dominant gradient direction; both branches may fire.  Note the multipliers:
    for ``'horizontal'`` (``kx = 0``) the left/right test is always allowed and the up/down test needs
    ``|by| >= |bx|``; for ``'vertical'`` the roles swap; for ``'both'`` the plain ``|bx| >= |by|`` / ``|by| >= |bx|``
    test results.  ``offset`` shifts the compared neighbours for the Roberts masks (``[-1 1 1 -1]``: row offsets
    of the left/right pair, column offsets of the up/down pair → comparisons along the two diagonals).
    The old M-code that zeroed the 1-px border or replicate-padded ``b`` differs from the builtin at the border.
    """
    b = np.asarray(b, dtype=np.float64)
    bx = np.abs(np.asarray(bx, dtype=np.float64))
    by = np.abs(np.asarray(by, dtype=np.float64))
    bp = np.pad(b, 1, mode="constant", constant_values=0.0)
    m, n = b.shape
    o1, o2, o3, o4 = (int(v) for v in offset)
    left = bp[1 + o1:1 + o1 + m, 0:n]  # b(r + o1, c - 1)
    right = bp[1 + o2:1 + o2 + m, 2:n + 2]  # b(r + o2, c + 1)
    up = bp[0:m, 1 + o3:1 + o3 + n]  # b(r - 1, c + o3)
    down = bp[2:m + 2, 1 + o4:1 + o4 + n]  # b(r + 1, c + o4)
    horiz = (bx >= kx * by - eps) & (left <= b) & (b > right)
    vert = (by >= ky * bx - eps) & (up <= b) & (b > down)
    return (b > cutoff) & (horiz | vert)


def log_zero_crossings(b: np.ndarray, thresh: float) -> np.ndarray:
    """Zero-crossing rule of ``edge.m`` (``'log'`` / ``'zerocross'`` branch, lines 359–393).

    On the interior (MATLAB ``rr = 2:m-1, cc = 2:n-1``) a pixel is an edge if it is the **negative** side of a sign
    change with its right or lower neighbour, or the negative side of a sign change with its left or upper
    neighbour, and the jump across the crossing exceeds ``thresh``:

    ``b(r,c) < 0 & b(r,c+1) > 0 & |b(r,c) − b(r,c+1)| > T`` (and the ``[+ −]``, ``[− +]ᵀ``, ``[+ −]ᵀ`` variants).
    Pixels with ``b == 0`` are edges when their vertical (or horizontal) neighbours have opposite non-zero signs and
    differ by more than ``2T``.  Note the threshold applies to the *jump*, not to ``|b|`` (Book §4.1.2 only says
    "threshold T = 0.005").
    """
    b = np.asarray(b, dtype=np.float64)
    m, n = b.shape
    e = np.zeros((m, n), dtype=bool)
    if m < 3 or n < 3:
        return e
    c = b[1:m - 1, 1:n - 1]
    rgt = b[1:m - 1, 2:n]
    lft = b[1:m - 1, 0:n - 2]
    dwn = b[2:m, 1:n - 1]
    up = b[0:m - 2, 1:n - 1]
    inner = ((c < 0) & (rgt > 0) & (np.abs(c - rgt) > thresh))  # [- +]
    inner |= ((lft > 0) & (c < 0) & (np.abs(lft - c) > thresh))  # [+ -]
    inner |= ((c < 0) & (dwn > 0) & (np.abs(c - dwn) > thresh))  # [- +]'
    inner |= ((up > 0) & (c < 0) & (np.abs(up - c) > thresh))  # [+ -]'
    zero = c == 0
    if zero.any():
        # linear-index neighbours in edge.m: zero∓1 = above/below (column-major), zero∓m = left/right
        inner |= zero & (((up < 0) & (dwn > 0)) | ((up > 0) & (dwn < 0))) & (np.abs(up - dwn) > 2 * thresh)
        inner |= zero & (((lft < 0) & (rgt > 0)) | ((lft > 0) & (rgt < 0))) & (np.abs(lft - rgt) > 2 * thresh)
    e[1:m - 1, 1:n - 1] = inner
    return e


def edge(a: np.ndarray, method: str = "sobel", thresh: float | None = None, direction: str = "both",
         thinning: bool = True, sigma: float = 2.0, H: np.ndarray | None = None) -> EdgeResult:
    """MATLAB ``[BW, thresh, gv, gh] = edge(I, method, thresh, direction, 'thinning'|'nothinning')`` /
    ``edge(I, 'log', thresh, sigma)`` / ``edge(I, 'zerocross', thresh, H)``.

    Book: §4.1.1 Eqs. (4.2), (4.4), (4.7)–(4.8), Fig. 4.3 (Sobel/Prewitt, T = 0.05); §4.1.2 Eqs. (4.14)–(4.15),
    Fig. 4.6 (LoG, 13×13, σ = 2, T = 0.005).  MATLAB source: ``MATLAB_ROOT/ch4/derivative.m`` line 7.

    Parameters
    ----------
    a : ndarray (M, N), float
        Gray image, normally in ``[0, 1]`` (``derivative.m`` uses ``double(im)/256``; the thresholds quoted in the
        book refer to that scaling).  Integer/bool input raises ``TypeError``: MATLAB would convert with
        ``im2single`` (LoG) or scale differently (Sobel builtin) — convert explicitly first.
    method : {'sobel', 'prewitt', 'roberts', 'log', 'zerocross'}
        ``'canny'`` (not in the book) raises ``NotImplementedError`` — use ``skimage.feature.canny(a, sigma=sqrt(2))``
        as an *approx* substitute.
    thresh : float, optional
        Sobel/Prewitt/Roberts: edges where ``bx² + by² > thresh²``; ``None`` → automatic ``sqrt(scale·mean(b))``
        with ``scale = 4`` (6 for Roberts).  LoG/zerocross: jump threshold; ``None`` → ``0.75·mean(|b|)``.
    direction : {'both', 'horizontal', 'vertical'}
        Gradient methods only: ``kx, ky = (1,1) / (0,1) / (1,0)``.
    thinning : bool
        Gradient methods only (``'nothinning'`` → ``b > cutoff``).
    sigma : float
        LoG standard deviation (MATLAB default 2); kernel size ``2*ceil(3*sigma)+1`` (13 for σ = 2).
    H : ndarray, optional
        ``'zerocross'`` kernel (the LoG branch with a user filter).

    Returns
    -------
    EdgeResult
        ``(bw, thresh, gv, gh)``; ``thresh`` mirrors MATLAB's second output (the automatic value when ``thresh``
        was ``None``).  ``gv``/``gh`` are ``bx``/``by`` for the gradient methods, ``None`` otherwise.

    Parity: exact vs MATLAB R2025a — 0 differing pixels for sobel / prewitt / roberts (with and without thinning,
    automatic and given ``T``, all three directions), log (auto and given ``T``, σ = 1.5 / 2) and zerocross, on
    ``test.jpg`` and on ramp / step / diagonal-tie / random fixtures; thresholds and ``gv``/``gh`` bit-identical.
    """
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("edge: I must be 2-D")
    if not np.issubdtype(a.dtype, np.floating):
        raise TypeError("edge: pass a float image (e.g. double(im)/256 as in derivative.m, or im2double)")
    a = a.astype(np.float64, copy=False)
    method = str(method).lower()
    if method not in _METHODS:
        raise ValueError(f"edge: unknown method {method!r}")
    if method in ("canny", "approxcanny", "canny_old"):
        # PARITY: approx (not provided) — MATLAB's Canny (derivative-of-Gaussian, hysteresis with automatic 70th-
        # percentile thresholds, bwselect) differs from skimage.feature.canny; the book never uses it.
        raise NotImplementedError("edge('canny') is not part of the book; use skimage.feature.canny(a, sigma=np.sqrt(2)) "
                                  "(approx parity) if needed")
    direction = str(direction).lower()
    if direction not in _DIRECTIONS:
        raise ValueError(f"edge: direction must be one of {_DIRECTIONS}")
    kx, ky = {"both": (1.0, 1.0), "horizontal": (0.0, 1.0), "vertical": (1.0, 0.0)}[direction]
    m, n = a.shape

    if method in ("log", "zerocross"):
        if H is None:
            fsize = int(np.ceil(sigma * 3)) * 2 + 1  # odd size > 6*sigma
            op = fspecial("log", fsize, sigma)
        else:
            op = np.asarray(H, dtype=np.float64)
        op = op - op.sum() / op.size  # make the op sum to zero
        b = imfilter(a, op, "replicate")
        if thresh is None:
            thresh = 0.75 * float(np.abs(b).sum()) / b.size
        e = log_zero_crossings(b, float(thresh))
        return EdgeResult(e, float(thresh), None, None)

    if method in ("sobel", "prewitt"):
        scale, offset = 4.0, (0, 0, 0, 0)
        bx, by, b = gradient_sobel_prewitt(a, method, kx, ky)
    else:  # roberts
        scale, offset = 6.0, (-1, 1, 1, -1)
        bx, by, b = gradient_roberts(a, kx, ky)
    if thresh is None:  # cutoff from the RMS estimate of noise
        cutoff = scale * float(b.sum()) / b.size
        thresh = float(np.sqrt(cutoff))
    else:
        thresh = float(thresh)
        cutoff = thresh * thresh
    if thinning:
        e = thin_gradient(b, bx, by, kx, ky, offset, cutoff)
    else:
        e = b > cutoff
    return EdgeResult(e, thresh, bx, by)
