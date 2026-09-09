"""Mathematical morphology with MATLAB semantics — ``strel``, ``imerode``/``imdilate``, opening/closing,
reconstruction and morphological gradients.

Book: Chapter 4 §4.2, Eqs. (4.16)–(4.42), Figs. 4.7–4.16.  MATLAB source: ``MATLAB_ROOT/ch4/morphology.m``
(``strel('dis', 7)``, ``imerode``, ``imdilate`` and the three gradients); the structuring-element construction is
ported from R2025a ``+images/+internal/+coder/+strel/StructuringElementHelper.m`` (disk = Adams' periodic-line
decomposition, *not* a Euclidean disk) and ``iptui.intline``.

Conventions reproduced from MATLAB (none of them is stated in the book text):

* the SE origin is ``floor((size + 1) / 2)`` (1-based) = ``((M - 1) // 2, (N - 1) // 2)`` (0-based);
* erosion takes the minimum over the SE placed at ``z`` (Eq. 4.20, ``f(x+s, y+t)``), pixels outside the image
  count as ``1`` / ``+intmax`` / ``+Inf``; dilation takes the maximum over the **reflected** SE (Eqs. 4.18–4.19,
  4.21, ``f(x-s, y-t)``), outside pixels count as ``0`` / ``intmin`` / ``-Inf``;
* a decomposed SE (list of neighbourhoods) is applied sequentially on an image padded by the total reach, which
  equals the single erosion/dilation by the Minkowski sum (what MATLAB does for ``strel('disk', r)``);
* ``imreconstruct`` uses 8-connectivity by default and requires ``marker <= mask``.

Backend: OpenCV ``cv2.erode``/``cv2.dilate`` with ``BORDER_CONSTANT`` and the explicit MATLAB border value
(fast on the 12-Mpx book images); ``scipy.ndimage`` for dtypes OpenCV does not support.  The two backends and a
literal Eq. (4.20)/(4.21) loop were cross-checked on asymmetric and even-sized SEs (see the chapter report).
"""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from scipy import ndimage

from .matlab_compat import matlab_round

# ---------------------------------------------------------------------------------------------------------------
# Structuring elements (§4.2 intro, Fig. 4.7; MATLAB strel)
# ---------------------------------------------------------------------------------------------------------------

_STREL_SHAPES = ("arbitrary", "square", "diamond", "rectangle", "octagon", "line", "disk", "sphere", "ball", "cube",
                 "cuboid", "pair", "periodicline")
_DISK_V = {
    4: np.array([[1, 0], [1, 1], [0, 1], [-1, 1]]),
    6: np.array([[1, 0], [1, 2], [2, 1], [0, 1], [-1, 2], [-2, 1]]),
    8: np.array([[1, 0], [2, 1], [1, 1], [1, 2], [0, 1], [-1, 2], [-1, 1], [-2, 1]]),
}


def _match_shape(shape: str) -> str:
    """MATLAB ``validatestring`` semantics: exact (case-insensitive) match, else a unique prefix (``'dis'`` → ``'disk'``)."""
    key = str(shape).lower()
    if key in _STREL_SHAPES:
        return key
    cands = [s for s in _STREL_SHAPES if s.startswith(key)]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise ValueError(f"strel: unknown shape {shape!r}; expected one of {_STREL_SHAPES}")
    raise ValueError(f"strel: ambiguous shape {shape!r} (matches {cands})")


def se_origin(se: np.ndarray) -> tuple[int, int]:
    """0-based origin of a neighbourhood, MATLAB's ``floor((size + 1) / 2)`` (1-based) → ``((M-1)//2, (N-1)//2)``."""
    M, N = np.asarray(se).shape[:2]
    return (M - 1) // 2, (N - 1) // 2


def _offsets_to_nhood(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Neighbourhood whose 1-pixels sit at the (row, col) offsets ``(r, c)``, sized symmetrically about the origin."""
    r = np.asarray(r, dtype=np.int64).ravel()
    c = np.asarray(c, dtype=np.int64).ravel()
    mr, mc = (int(np.abs(r).max()) if r.size else 0), (int(np.abs(c).max()) if c.size else 0)
    nhood = np.zeros((2 * mr + 1, 2 * mc + 1), dtype=bool)
    nhood[r + mr, c + mc] = True
    return nhood


def intline(x1: int, x2: int, y1: int, y2: int) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB ``iptui.intline`` — integer points of the line from ``(x1, y1)`` to ``(x2, y2)`` (Bresenham-like)."""
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    if dx == 0 and dy == 0:
        return np.array([x1]), np.array([y1])
    flip = False
    if dx >= dy:
        if x1 > x2:
            x1, x2, y1, y2, flip = x2, x1, y2, y1, True
        m = (y2 - y1) / (x2 - x1)
        x = np.arange(x1, x2 + 1)
        y = matlab_round(y1 + m * (x - x1)).astype(np.int64)
    else:
        if y1 > y2:
            x1, x2, y1, y2, flip = x2, x1, y2, y1, True
        m = (x2 - x1) / (y2 - y1)
        y = np.arange(y1, y2 + 1)
        x = matlab_round(x1 + m * (y - y1)).astype(np.int64)
    if flip:
        x, y = x[::-1], y[::-1]
    return np.asarray(x, dtype=np.int64), np.asarray(y, dtype=np.int64)


def periodic_line(p: int, v: Sequence[int]) -> np.ndarray:
    """``strel('periodicline', p, v)``: the ``2p+1`` pixels ``{j·v : j = -p..p}``, ``v = (row, col)`` offset."""
    v = np.asarray(v, dtype=np.int64).ravel()
    j = np.arange(-int(p), int(p) + 1)
    return _offsets_to_nhood(j * v[0], j * v[1])


def line_strel(length: float, angle_deg: float) -> np.ndarray:
    """``strel('line', len, deg)``: symmetric digital line of ``len`` pixels at ``deg`` degrees (counter-clockwise from
    the horizontal), built with ``intline`` exactly as MATLAB does (``x = round((len-1)/2 cos θ)``, ``y = -round(... sin θ)``)."""
    if length < 1:
        return np.zeros((0, 0), dtype=bool)
    theta = np.mod(angle_deg, 180) * np.pi / 180.0
    x = int(matlab_round((length - 1) / 2 * np.cos(theta)))
    y = -int(matlab_round((length - 1) / 2 * np.sin(theta)))
    c, r = intline(-x, x, -y, y)
    return _offsets_to_nhood(r, c)


def minkowski_sum(nhoods: Sequence[np.ndarray]) -> np.ndarray:
    """Dilate a single pixel by each neighbourhood in turn (MATLAB ``imdilate(1, seq, 'full')``): the combined SE."""
    nh = np.ones((1, 1), dtype=bool)
    for s in nhoods:
        s = np.asarray(s) != 0
        M1, N1 = nh.shape
        M2, N2 = s.shape
        out = np.zeros((M1 + M2 - 1, N1 + N2 - 1), dtype=bool)
        for (i, j) in zip(*np.nonzero(s)):
            out[i:i + M1, j:j + N1] |= nh
        nh = out
    return nh


def disk_decomposition(r: int, n: int = 4) -> list[np.ndarray]:
    """The periodic-line (+ line) sequence MATLAB uses for ``strel('disk', r, n)`` (``decompose(se)``), or ``[]`` for the
    exact-disk cases ``n == 0`` / ``r < 3``.  Eroding/dilating by the sequence equals using :func:`strel` ('disk')."""
    r, n = int(r), int(n)
    if r < 3 or n == 0:
        return []
    if n not in _DISK_V:
        raise ValueError("strel('disk'): N must be 0, 4, 6 or 8")
    v = _DISK_V[n]
    theta = np.pi / (2 * n)
    k = 2 * r / (1.0 / np.tan(theta) + 1.0 / np.sin(theta))
    seq = [periodic_line(int(np.floor(k / np.linalg.norm(v[q].astype(np.float64)))), v[q]) for q in range(n)]
    nhood = minkowski_sum(seq)
    M = nhood.shape[0]
    rd = np.nonzero(nhood)[0] - ((M + 1) // 2 - 1)  # row offsets from the centre (MATLAB: rd - floor((M+1)/2))
    max_horiz_radius = int(rd.max())
    length = 2 * (r - max_horiz_radius - 1) + 1
    if length >= 3:
        seq.append(line_strel(length, 0))
        seq.append(line_strel(length, 90))
    return seq


def strel(shape, *params, n: int | None = None) -> np.ndarray:
    """MATLAB ``strel(SHAPE, PARAMS...)`` as a flat boolean neighbourhood (``getnhood(strel(...))``).

    Book: §4.2 intro, Fig. 4.7 ((a) ``strel('square', 3)``, (b) ``strel('rectangle', [3 5])``, (c) ``strel('disk', 5)``
    = the printed 9×9 / 69-pixel matrix, (d) ``strel('diamond', 5)``).  MATLAB source: ``morphology.m`` line 6,
    ``SE = strel('dis', 7)`` — ``'dis'`` is MATLAB's unambiguous-prefix matching for ``'disk'``.

    Shapes: ``'disk'`` (r, n=4 → Adams' periodic-line approximation: r = 7 gives a 13×13 octagon of 157 pixels,
    r < 3 or n = 0 → exact ``x²+y² ≤ r²``), ``'diamond'`` (M: ``|x|+|y| ≤ M``), ``'square'`` (w), ``'rectangle'``
    ([m n]), ``'line'`` (len, deg), ``'periodicline'`` (p, v), ``'octagon'`` (M multiple of 3), ``'pair'`` (offset),
    ``'arbitrary'`` / a bare array (``strel(nhood)``).  3-D shapes raise ``NotImplementedError``.

    Returns a bool array whose origin is ``floor((size + 1) / 2)`` (:func:`se_origin`).  Use
    :func:`disk_decomposition` for MATLAB's decomposed sequence (identical results, faster on large radii).

    Parity: exact (neighbourhoods identical to ``getnhood`` — verified in the chapter-4 report).
    """
    if not isinstance(shape, str):
        return np.asarray(shape) != 0
    kind = _match_shape(shape)
    if kind == "arbitrary":
        if len(params) != 1:
            raise ValueError("strel('arbitrary', NHOOD) needs exactly one neighbourhood")
        return np.asarray(params[0]) != 0
    if kind in ("sphere", "ball", "cube", "cuboid"):
        raise NotImplementedError(f"strel('{kind}') is a 3-D structuring element (not used by the book)")
    if kind == "disk":
        if not 1 <= len(params) <= 2:
            raise ValueError("strel('disk', R [, N])")
        r = int(params[0])
        if n is None:
            n = int(params[1]) if len(params) == 2 else 4
        if r < 0:
            raise ValueError("strel('disk'): R must be non-negative")
        seq = disk_decomposition(r, n)
        if not seq:
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            return xx * xx + yy * yy <= r * r
        return minkowski_sum(seq)
    if kind == "diamond":
        (M,) = params
        M = int(M)
        rr, cc = np.mgrid[-M:M + 1, -M:M + 1]
        return np.abs(rr) + np.abs(cc) <= M
    if kind == "square":
        (w,) = params
        return np.ones((int(w), int(w)), dtype=bool)
    if kind == "rectangle":
        (mn,) = params
        m, nn = (int(v) for v in np.asarray(mn).ravel())
        return np.ones((m, nn), dtype=bool)
    if kind == "line":
        length, deg = params
        return line_strel(float(length), float(deg))
    if kind == "periodicline":
        p, v = params
        return periodic_line(int(p), v)
    if kind == "octagon":
        (M,) = params
        M = int(M)
        if M < 0 or M % 3 != 0:
            raise ValueError("strel('octagon', M): M must be a non-negative multiple of 3")
        k = M // 3
        rr, cc = np.mgrid[-M:M + 1, -M:M + 1]
        return np.abs(rr) + np.abs(cc) <= M + k
    # pair
    (mn,) = params
    off = np.asarray(mn, dtype=np.int64).ravel()
    size = np.abs(off) * 2 + 1
    nhood = np.zeros(tuple(int(s) for s in size), dtype=bool)
    ctr = (size + 1) // 2 - 1
    nhood[ctr[0], ctr[1]] = True
    nhood[ctr[0] + off[0], ctr[1] + off[1]] = True
    return nhood


# ---------------------------------------------------------------------------------------------------------------
# Erosion / dilation (§4.2.1, Eqs. 4.16–4.21; MATLAB imerode / imdilate)
# ---------------------------------------------------------------------------------------------------------------

_CV_DTYPES = (np.uint8, np.uint16, np.int16, np.float32, np.float64)


def _border_values(dtype: np.dtype) -> tuple[float, float]:
    """(erosion pad, dilation pad) = (max, min) representable value: 1/0 for bool, intmax/intmin, +Inf/-Inf."""
    if dtype == np.bool_:
        return 1, 0
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.max, info.min
    return np.inf, -np.inf


def _min_max_filter(x: np.ndarray, se: np.ndarray, origin: tuple[int, int], cval: float, op: str) -> np.ndarray:
    """Correlation-style min (``op='min'``) / max over the SE placed at each pixel: ``out(z) = op_{b∈se} x(z + b - origin)``.

    Constant padding ``cval``.  OpenCV for its supported dtypes (``cv2.erode``/``cv2.dilate`` both use this
    non-reflected form), scipy otherwise.
    """
    M, N = se.shape
    r0, c0 = origin
    if x.dtype.type in _CV_DTYPES and x.ndim == 2:
        kernel = np.ascontiguousarray(se, dtype=np.uint8)
        fn = cv2.erode if op == "min" else cv2.dilate
        return fn(np.ascontiguousarray(x), kernel, anchor=(int(c0), int(r0)), iterations=1,
                  borderType=cv2.BORDER_CONSTANT, borderValue=float(cval))
    # scipy minimum_filter / maximum_filter are correlation-style too (no reflection); their centre is
    # ``size // 2 + origin`` → shift to MATLAB's ``(M-1)//2`` (same trick as filters.imfilter)
    org = (r0 - M // 2, c0 - N // 2)
    fn = ndimage.minimum_filter if op == "min" else ndimage.maximum_filter
    return fn(x, footprint=se, mode="constant", cval=cval, origin=org)


def _as_sequence(se) -> list[np.ndarray]:
    if isinstance(se, np.ndarray):
        return [se != 0]
    return [np.asarray(s) != 0 for s in se]


def _pad_reach(seq: list[np.ndarray], reflected: bool) -> tuple[int, int, int, int]:
    """Total (top, bottom, left, right) padding needed so a sequential decomposition equals the Minkowski sum."""
    top = bottom = left = right = 0
    for s in seq:
        M, N = s.shape
        r0, c0 = se_origin(s)
        up, dn, lf, rt = r0, M - 1 - r0, c0, N - 1 - c0
        if reflected:
            up, dn, lf, rt = dn, up, rt, lf
        top, bottom, left, right = top + up, bottom + dn, left + lf, right + rt
    return top, bottom, left, right


def _morph(I: np.ndarray, se, op: str) -> np.ndarray:
    I = np.asarray(I)
    if I.ndim != 2:
        raise ValueError("imerode/imdilate: 2-D images only")
    seq = _as_sequence(se)
    if not seq or any(s.size == 0 for s in seq):
        raise ValueError("empty structuring element")
    in_dtype = I.dtype
    x = I.astype(np.uint8) if in_dtype == np.bool_ else I
    pad_e, pad_d = _border_values(in_dtype)
    cval = pad_e if op == "min" else pad_d
    reflected = op == "max"
    if len(seq) == 1:
        s = seq[0]
        if reflected:  # Eq. 4.18 / 4.21: dilation uses the reflected SE  →  correlation with the flipped kernel
            M, N = s.shape
            r0, c0 = se_origin(s)
            out = _min_max_filter(x, s[::-1, ::-1], (M - 1 - r0, N - 1 - c0), cval, "max")
        else:
            out = _min_max_filter(x, s, se_origin(s), cval, "min")
    else:  # decomposed SE: pad by the total reach, apply sequentially, crop (MATLAB's getpadsize strategy)
        top, bottom, left, right = _pad_reach(seq, reflected)
        xp = np.pad(x, ((top, bottom), (left, right)), mode="constant", constant_values=cval)
        for s in seq:
            M, N = s.shape
            r0, c0 = se_origin(s)
            if reflected:
                xp = _min_max_filter(xp, s[::-1, ::-1], (M - 1 - r0, N - 1 - c0), cval, "max")
            else:
                xp = _min_max_filter(xp, s, (r0, c0), cval, "min")
        out = xp[top:top + I.shape[0], left:left + I.shape[1]]
    if in_dtype == np.bool_:
        return out.astype(bool)
    return out.astype(in_dtype, copy=False)


def imerode(I: np.ndarray, se) -> np.ndarray:
    """MATLAB ``imerode(I, SE)`` — binary erosion Eq. (4.16) ``A ⊖ B = {z | (B)_z ⊆ A}`` / grayscale erosion
    Eq. (4.20) ``[f ⊖ b](x, y) = min_{(s,t)∈b} f(x+s, y+t)``.

    Book: §4.2.1, Figs. 4.8(c)(d), 4.9(b), 4.10(a).  MATLAB source: ``morphology.m`` lines 13 and 31.
    ``se`` is a bool neighbourhood from :func:`strel` (origin :func:`se_origin`) or a list of neighbourhoods
    (decomposed SE, e.g. :func:`disk_decomposition`).  Out-of-image pixels count as 1 / intmax / +Inf, so objects
    touching the border are *not* eroded from the outside.  dtype is preserved (bool → bool).

    Parity: exact vs MATLAB (0 px on ``morphology.m``'s ``J`` and ``X``).
    """
    return _morph(I, se, "min")


def imdilate(I: np.ndarray, se) -> np.ndarray:
    """MATLAB ``imdilate(I, SE)`` — binary dilation Eq. (4.18) ``A ⊕ B = {z | (B̂)_z ∩ A ≠ ∅}`` with the reflected SE
    Eq. (4.19) / grayscale dilation Eq. (4.21) ``[f ⊕ b](x, y) = max_{(s,t)∈b} f(x−s, y−t)``.

    Book: §4.2.1, Figs. 4.8(e)(f), 4.9(c), 4.10(b).  MATLAB source: ``morphology.m`` lines 16 and 32.
    Out-of-image pixels count as 0 / intmin / −Inf.  See :func:`imerode` for the ``se`` argument.

    Parity: exact vs MATLAB (0 px on ``K`` and ``Y``); the reflection is visible only for asymmetric SEs.
    """
    return _morph(I, se, "max")


def imopen(I: np.ndarray, se) -> np.ndarray:
    """MATLAB ``imopen`` — opening Eq. (4.23) ``A ∘ B = (A ⊖ B) ⊕ B`` (removes bright details smaller than B; Fig. 4.12).

    Book: §4.2.2.2 (text only).  Parity: exact by construction from :func:`imerode`/:func:`imdilate`.
    """
    return imdilate(imerode(I, se), se)


def imclose(I: np.ndarray, se) -> np.ndarray:
    """MATLAB ``imclose`` — closing Eq. (4.22) ``A • B = (A ⊕ B) ⊖ B`` (fills dark details smaller than B; Fig. 4.11).

    Book: §4.2.2.1 (text only).  Parity: exact by construction.
    """
    return imerode(imdilate(I, se), se)


# ---------------------------------------------------------------------------------------------------------------
# Reconstruction (§4.2.3, Eqs. 4.24–4.38; MATLAB imreconstruct)
# ---------------------------------------------------------------------------------------------------------------


def _conn_footprint(conn) -> np.ndarray:
    if isinstance(conn, np.ndarray):
        return conn != 0
    if conn == 8:
        return np.ones((3, 3), dtype=bool)
    if conn == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    raise ValueError("conn must be 4, 8 or a neighbourhood array")


def _check_shapes(marker: np.ndarray, mask: np.ndarray) -> None:
    if marker.shape != mask.shape:
        raise ValueError("marker and mask must have the same size")


def imreconstruct(marker: np.ndarray, mask: np.ndarray, conn: int | np.ndarray = 8) -> np.ndarray:
    """MATLAB ``imreconstruct(marker, mask, conn)`` — morphological reconstruction by dilation.

    Book: §4.2.3, Eqs. (4.24)–(4.27) (binary: ``X_k = (X_{k-1} ⊕ B) ∩ G`` until stable) and (4.31)–(4.34)
    (grayscale: ``X_k = (X_{k-1} ⊕ b) ∧ g``), Fig. 4.13.  Text only in ch4; used by ch5 (marker construction,
    ``imimposemin``, ``imfill``) and ch7.  ``B`` is the connectivity: 8 (3×3, MATLAB default), 4 (cross) or an
    arbitrary neighbourhood.  Requires ``marker <= mask`` everywhere (MATLAB's precondition; ``ValueError``
    otherwise).  dtype of ``marker`` is preserved (bool → bool).

    Implementation: ``skimage.morphology.reconstruction(seed, mask, 'dilation', footprint)`` (Robinson's
    algorithm; same fixed point as the iterative definition — cross-checked by :func:`reconstruct_iterative`).
    Parity: exact (same result as the definition; verified vs MATLAB in the chapter report).
    """
    from skimage.morphology import reconstruction

    marker, mask = np.asarray(marker), np.asarray(mask)
    _check_shapes(marker, mask)
    if np.any(marker.astype(np.float64) > mask.astype(np.float64)):
        raise ValueError("imreconstruct: marker must be <= mask everywhere")
    # PARITY: exact — skimage's reconstruction computes the same unique fixed point as the Eq. (4.27)/(4.34)
    # iteration (a different algorithm, not a different result); reconstruct_iterative is the literal cross-check.
    out = reconstruction(marker.astype(np.float64), mask.astype(np.float64), method="dilation",
                         footprint=_conn_footprint(conn))
    return out.astype(marker.dtype) if marker.dtype != np.bool_ else out > 0


def reconstruct_by_erosion(marker: np.ndarray, mask: np.ndarray, conn: int | np.ndarray = 8) -> np.ndarray:
    """Reconstruction by erosion — Book Eqs. (4.28)–(4.30) (binary, ``X_k = (X_{k-1} ⊖ B) ∪ G``, ``G ⊆ F``) and
    (4.35)–(4.38) (grayscale, ``X_k = (X_{k-1} ⊖ b) ∨ g``, ``f ≥ g``), Fig. 4.14.  No MATLAB builtin (MATLAB users
    write ``imcomplement(imreconstruct(imcomplement(marker), imcomplement(mask)))``); here
    ``skimage.morphology.reconstruction(..., method='erosion')``.  Requires ``marker >= mask``.
    """
    from skimage.morphology import reconstruction

    marker, mask = np.asarray(marker), np.asarray(mask)
    _check_shapes(marker, mask)
    if np.any(marker.astype(np.float64) < mask.astype(np.float64)):
        raise ValueError("reconstruct_by_erosion: marker must be >= mask everywhere")
    # PARITY: reimplemented — no MATLAB builtin; same fixed point as Eq. (4.30)/(4.38) (cross-checked by
    # reconstruct_iterative); equals imcomplement(imreconstruct(imcomplement(marker), imcomplement(mask))).
    out = reconstruction(marker.astype(np.float64), mask.astype(np.float64), method="erosion",
                         footprint=_conn_footprint(conn))
    return out.astype(marker.dtype) if marker.dtype != np.bool_ else out > 0


def geodesic_dilation(F: np.ndarray, G: np.ndarray, se: np.ndarray | None = None, n: int = 1) -> np.ndarray:
    """Geodesic dilation of size ``n`` — Book Eqs. (4.25)–(4.26) ``D_G^(1)(F) = (F ⊕ B) ∩ G``, ``D^(n) = D^(1)[D^(n-1)]``
    (binary) and Eqs. (4.32)–(4.33) ``(f ⊕ b) ∧ g`` (grayscale, pointwise minimum).  ``se`` defaults to the 3×3
    square (8-connectivity)."""
    F, G = np.asarray(F), np.asarray(G)
    _check_shapes(F, G)
    B = np.ones((3, 3), dtype=bool) if se is None else np.asarray(se) != 0
    X = F
    for _ in range(int(n)):
        D = imdilate(X, B)
        X = (D & (G != 0)) if F.dtype == np.bool_ else np.minimum(D, G.astype(D.dtype))
    return X


def geodesic_erosion(F: np.ndarray, G: np.ndarray, se: np.ndarray | None = None, n: int = 1) -> np.ndarray:
    """Geodesic erosion of size ``n`` — Book Eqs. (4.28)–(4.29) ``E_G^(1)(F) = (F ⊖ B) ∪ G`` (binary) and
    Eqs. (4.36)–(4.37) ``(f ⊖ b) ∨ g`` (grayscale, pointwise maximum)."""
    F, G = np.asarray(F), np.asarray(G)
    _check_shapes(F, G)
    B = np.ones((3, 3), dtype=bool) if se is None else np.asarray(se) != 0
    X = F
    for _ in range(int(n)):
        E = imerode(X, B)
        X = (E | (G != 0)) if F.dtype == np.bool_ else np.maximum(E, G.astype(E.dtype))
    return X


def reconstruct_iterative(F: np.ndarray, G: np.ndarray, se: np.ndarray | None = None, method: str = "dilation",
                          max_iter: int = 100000) -> tuple[np.ndarray, int]:
    """Literal iteration of the reconstruction definitions — Book Eq. (4.27) / (4.34) (``method='dilation'``) or
    Eq. (4.30) / (4.38) (``method='erosion'``): ``X_0 = F``, ``X_k = geodesic(X_{k-1})`` until ``X_k == X_{k-1}``.

    Returns ``(X_k, k)`` with ``k`` the number of geodesic steps taken (the first ``k`` with ``X_k = X_{k-1}``).
    Teaching / cross-check function; :func:`imreconstruct` is the production version.
    """
    F, G = np.asarray(F), np.asarray(G)
    step = geodesic_dilation if method == "dilation" else geodesic_erosion
    if method not in ("dilation", "erosion"):
        raise ValueError("method must be 'dilation' or 'erosion'")
    X = F
    for k in range(1, int(max_iter) + 1):
        Xn = step(X, G, se, 1)
        if np.array_equal(Xn, X):
            return Xn, k
        X = Xn
    raise RuntimeError("reconstruct_iterative did not converge")


# ---------------------------------------------------------------------------------------------------------------
# Morphological gradients (§4.2.4, Eqs. 4.39–4.42)
# ---------------------------------------------------------------------------------------------------------------

_GRADIENT_KINDS = {"basic": "basic", "standard": "basic", "beucher": "basic", "internal": "internal",
                   "half-internal": "internal", "external": "external", "half-external": "external"}


def _matlab_minus(A: np.ndarray, B: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """``A - B`` with MATLAB class semantics: logical − logical → double; integers saturate; floats plain."""
    if dtype == np.bool_:
        return A.astype(np.float64) - B.astype(np.float64)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        d = A.astype(np.int64) - B.astype(np.int64)
        return np.clip(d, info.min, info.max).astype(dtype)
    return A.astype(np.float64) - B.astype(np.float64)


def morphological_gradient(I: np.ndarray, se, kind: str = "basic") -> np.ndarray:
    """Morphological gradients — Book Eq. (4.39) basic ``ρ = (A ⊕ B) − (A ⊖ B)``, Eq. (4.40) internal
    ``ρ_int = A − (A ⊖ B)``, Eq. (4.41) external ``ρ_ext = (A ⊕ B) − A``; identity Eq. (4.42) ``ρ_int + ρ_ext = ρ``.

    Book: §4.2.4, Figs. 4.15 (binary, r = 15), 4.16 (grayscale), 4.17(c)/4.18(c)/4.20 (internal, r = 5 / 16).
    MATLAB source: ``morphology.m`` lines 14, 17, 19 (``BW1 = I - J``, ``BW2 = K - I``, ``BW = K - J`` on logicals →
    **double** 0/1) and lines 34–36 (``internal = im - X`` etc. on uint8 → saturating uint8 subtraction, which is
    exact here because every difference is ≥ 0).

    ``kind`` ∈ {'basic' (alias 'standard'), 'internal', 'external'}.  dtype rule (MATLAB's): bool in → float64
    0/1 out; integer in → same integer class, computed in int64 and saturated; float in → float64.
    Parity: exact (0 px / 0 levels vs MATLAB's six arrays).
    """
    I = np.asarray(I)
    k = _GRADIENT_KINDS.get(str(kind).lower())
    if k is None:
        raise ValueError(f"kind must be one of {sorted(set(_GRADIENT_KINDS))}")
    if k == "basic":
        return _matlab_minus(imdilate(I, se), imerode(I, se), I.dtype)  # Eq. (4.39)
    if k == "internal":
        return _matlab_minus(I, imerode(I, se), I.dtype)  # Eq. (4.40)
    return _matlab_minus(imdilate(I, se), I, I.dtype)  # Eq. (4.41)
