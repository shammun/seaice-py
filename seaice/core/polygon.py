"""Planar-polygon primitives: rectangle clipping (``polybool``), ``poly2mask``/``roipoly``, ``polygeom``,
``minboundrect``, ``polyxpoly``, ``convhull`` and ``polyarea``.

Book: §6.5.3 (border effects — the snake's initial circle is clipped to the image rectangle),
§6.3.3/§6.4 (Algorithm 1), Ch. 8 §8.2 (polygon fit of an ice floe, Figs. 8.12–8.13, Appendix B ``.Polygon``),
Ch. 9 §9.3 (rectangularization of model-ice floes).

MATLAB sources: ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/{polygeom.m, minboundrect.m}``; the
``polybool('intersection', …)`` (Mapping Toolbox), ``roipoly``, ``polyxpoly`` and ``convhull`` calls in
``GVF_distance.m``, ``seaice_kmean_GVF_forenhancement.m``, ``for_test.m`` and ``sea_ice_model.m``;
``poly2mask`` is a line-by-line port of R2025a ``toolbox/images/images/eml/poly2mask.m`` (the readable codegen
twin of the compiled builtin).

Attribution
-----------
* ``polygeom.m`` — **H. J. Sommer III**, Ph.D., Professor of Mechanical Engineering, The Pennsylvania State
  University ("02.05.14, tested under MATLAB v5.2"), MATLAB File Exchange.  Green's-theorem boundary integrals.
* ``minboundrect.m`` — **John D'Errico** (woodchips@rochester.rr.com), MATLAB File Exchange, "Release 3.0,
  3/7/07".  Rotating calipers over the convex-hull edge angles.

Both are re-implementations in Python from the published algorithms; the original M-code is not redistributed.
"""
from __future__ import annotations

import numpy as np

from .matlab_compat import matlab_round

__all__ = ["clip_polygon_rect", "poly2mask", "roipoly", "polygeom", "minboundrect", "polyxpoly",
           "convhull", "polyarea"]


# --------------------------------------------------------------------------------------------------------------
# Sutherland-Hodgman clipping -- the stand-in for polybool('intersection', rect, poly)
# --------------------------------------------------------------------------------------------------------------

def clip_polygon_rect(x, y, x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Clip the polygon ``(x, y)`` to an axis-aligned rectangle (Sutherland–Hodgman, 1974).

    Book: §6.5.3 "border effects".  MATLAB source: ``GVF_distance.m`` line 132 /
    ``seaice_kmean_GVF_forenhancement.m`` lines 153, 265 / ``for_test.m`` line 84 —
    ``[x, y] = polybool('intersection', s_2, s_1, x, y)`` with ``s_2 = [0 0 s2 s2]`` (columns) and
    ``s_1 = [0 s1 s1 0]`` (rows), i.e. the image rectangle.  Its purpose is to keep the initial circle inside
    the image so the ``bw1(yy(i), xx(i)) = 0`` write at the end of the loop cannot index ``<= 0``.

    # DEVIATION: `reimplemented` — MATLAB's `polybool` is the Mapping Toolbox front end for the compiled GPC
    # library (`gpcmex`); there is no readable source.  A probe (analysis/ch06.md §0.5) shows it returns the same
    # vertex *set* but **rotates the start vertex and reverses the traversal**, and inserts rectangle corners
    # where the polygon is cut.  Sutherland-Hodgman against the four half-planes is exact for a convex clip
    # region and produces the same set; compare vertex sets / rasterised masks, never vertex order.  This is
    # safe here because everything downstream is invariant under rotation/reversal of a closed contour
    # (`snake_matrix` is circulant, `snakeinterp`'s spacing metric is cyclic, and the mask write is a set).

    Parameters
    ----------
    x, y : array_like
        Polygon vertices (open or closed; the polygon is closed implicitly).
    x_range, y_range : (float, float)
        ``(xmin, xmax)`` and ``(ymin, ymax)`` of the clip rectangle.

    Returns
    -------
    (x, y) : float64 arrays.  Empty arrays when the polygon lies entirely outside the rectangle.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size and x[0] == x[-1] and y[0] == y[-1] and x.size > 1:
        x, y = x[:-1], y[:-1]  # work on the open ring
    poly = list(zip(x.tolist(), y.tolist()))
    xmin, xmax = float(min(x_range)), float(max(x_range))
    ymin, ymax = float(min(y_range)), float(max(y_range))

    # Each edge is (inside test, intersection parameter along the segment).
    edges = [
        (lambda p: p[0] >= xmin, 0, xmin),
        (lambda p: p[0] <= xmax, 0, xmax),
        (lambda p: p[1] >= ymin, 1, ymin),
        (lambda p: p[1] <= ymax, 1, ymax),
    ]
    for inside, axis, bound in edges:
        if not poly:
            break
        out: list[tuple[float, float]] = []
        n = len(poly)
        for i in range(n):
            cur = poly[i]
            prev = poly[i - 1]
            cur_in = inside(cur)
            prev_in = inside(prev)
            if cur_in:
                if not prev_in:
                    out.append(_intersect_axis(prev, cur, axis, bound))
                out.append(cur)
            elif prev_in:
                out.append(_intersect_axis(prev, cur, axis, bound))
        poly = out
    if not poly:
        return np.zeros(0), np.zeros(0)
    # NOTE (ch06 verification, open item 2): MATLAB's `polybool` returns a **closed** ring -- it appends a copy of
    # the first vertex.  With the chapter's `Dmin = 0` that duplicate survives `snakeinterp` (a zero-length segment
    # is never removed, `IDX = (d < dmin)` being false for it), so it is carried into `snakedeform`; the MATLAB
    # reference records `XI0` at 252 points against an open ring's 251.  Closing it here is therefore the correct
    # contract, and what it bought is *structural*, not a lower pixel count: contours matching MATLAB's exact
    # point count went from **2 of 46 to 14 of 46** (16/46 with the circulant solver), and all **46/46** clipped
    # polygons are now closed at MATLAB's exact count.  Closing the ring did **not** move the pixel counts on its
    # own (`GVF_distance.m`'s `bw1` went 61 -> 63 of 31 730); the drop to **16 px (0.050 %)** dense / 14 px
    # default came from reproducing MATLAB's *single-precision* radii and circles (review S4), not from here.
    # The residual is `polybool`'s **start-vertex rotation**, which the verifier proved is not reproducible:
    # 0 of 46 clips follow max-y, min-y, max-x, min-x or lexicographic order.  A rotated start vertex shifts `snakeindex`'s insertion parity by one, the +-1 instability
    # quantified in `reports/ch06_verification.md` Deviation 3.
    if poly[0] != poly[-1]:
        poly = poly + [poly[0]]
    arr = np.asarray(poly, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def _intersect_axis(p, q, axis: int, bound: float) -> tuple[float, float]:
    """Point where the segment ``p → q`` crosses the line ``coordinate[axis] == bound``."""
    d = q[axis] - p[axis]
    t = 0.0 if d == 0 else (bound - p[axis]) / d
    other = 1 - axis
    val = p[other] + t * (q[other] - p[other])
    return (bound, val) if axis == 0 else (val, bound)


# --------------------------------------------------------------------------------------------------------------
# poly2mask / roipoly -- line-by-line port of R2025a eml/poly2mask.m
# --------------------------------------------------------------------------------------------------------------

def _int_line(x1: float, y1: float, x2: float, y2: float) -> tuple[np.ndarray, np.ndarray]:
    """``intLine`` of ``eml/poly2mask.m``: the integer points of the line ``(x1,y1) → (x2,y2)``, stepping along
    whichever coordinate changes most (MATLAB ``round`` = half away from zero)."""
    dx = abs(int(x2 - x1))
    dy = abs(int(y2 - y1))
    if dx == 0 and dy == 0:
        return np.array([x1], dtype=np.float64), np.array([y1], dtype=np.float64)
    if dx >= dy:
        m = (y2 - y1) / (x2 - x1)
        step = 1.0 if x2 > x1 else -1.0
        xv = np.arange(x1, x2 + 0.5 * step, step, dtype=np.float64)
        yv = matlab_round(y1 + m * (xv - x1))
    else:
        m = (x2 - x1) / (y2 - y1)
        step = 1.0 if y2 > y1 else -1.0
        yv = np.arange(y1, y2 + 0.5 * step, step, dtype=np.float64)
        xv = matlab_round(x1 + m * (yv - y1))
    return xv, yv


def poly2mask(x, y, M: int, N: int) -> np.ndarray:
    """MATLAB ``poly2mask(x, y, M, N)`` — rasterise a polygon into an ``M×N`` logical mask.

    Book: used indirectly through :func:`roipoly` (``regionprops`` ``ConvexImage`` → ``ConvexArea`` →
    ``Solidity``, the ``Rc`` criterion of ch9 p. 205 that ``GVF_distance.m`` tests) and directly by ch8
    ``sea_ice_model.m``.  Ported line by line from R2025a ``toolbox/images/images/eml/poly2mask.m``: the polygon
    is closed if necessary, every vertex is snapped to a 5× sub-pixel grid
    (``floor(5(v − 0.5) + 0.5) + 1``), the edges are walked with ``intLine``, column crossings toggle a bit in
    the output and a per-column parity scan fills the interior.

    ``skimage.draw.polygon2mask`` uses a different (pixel-centre) inclusion rule and disagrees on boundary
    pixels — hence the port.  Parity target: exact.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    M = int(M)
    N = int(N)
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size == 0 or M <= 0 or N <= 0:
        return np.zeros((max(M, 0), max(N, 0)), dtype=bool)
    if x[-1] != x[0] or y[-1] != y[0]:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    scale = 5.0
    xs = np.floor(scale * (x - 0.5) + 0.5) + 1.0
    ys = np.floor(scale * (y - 0.5) + 0.5) + 1.0

    xl: list[np.ndarray] = []
    yl: list[np.ndarray] = []
    for i in range(1, xs.size):
        a, b = _int_line(xs[i - 1], ys[i - 1], xs[i], ys[i])
        xl.append(a)
        yl.append(b)
    x_line = np.concatenate(xl)
    y_line = np.concatenate(yl)

    out = np.zeros((M, N), dtype=bool)
    if x_line.size < 2:
        return out
    # Vectorised form of the `for pt = 1:borderSize-1` loop.  Each iteration reads entries pt and pt+1 and
    # writes only entry pt, so the in-place updates of the M-code never feed back into a later iteration.
    dxs = x_line[1:] - x_line[:-1]
    changes = np.abs(dxs) >= 1
    y_cur = np.minimum(y_line[:-1], y_line[1:])
    x_cur = x_line[:-1] - (dxs < 0).astype(np.float64)
    scaled_down = (x_cur + (scale - 1.0) / 2.0) / scale
    on_edge = np.abs(scaled_down - np.floor(scaled_down)) < 1.0 / (scale * 50.0)
    sel = changes & on_edge
    if not sel.any():
        return out
    x_use = matlab_round(scaled_down[sel]).astype(np.int64) + 1
    y_use = np.ceil((y_cur[sel] + (scale - 1.0) / 2.0) / scale).astype(np.int64) + 1
    ok = (x_use >= 2) & (x_use <= N + 1)
    x_use = x_use[ok]
    y_use = y_use[ok]
    col = x_use - 2                                   # 0-based column index (MATLAB xUse-1)
    overflow = y_use > M + 1
    y_use = np.where(overflow, M + 1, np.maximum(2, y_use))
    row = np.where(overflow, M, y_use - 2)            # 0-based row of `out(yUse-1, xUse-1)`; overflow: no toggle
    toggles = ~overflow
    if toggles.any():
        flat = row[toggles] * N + col[toggles]
        counts = np.bincount(flat, minlength=M * N)
        out = (counts.reshape(M, N) % 2).astype(bool)  # XOR is commutative -> parity of the toggle count
    # minY / maxY per column.  Every candidate is >= 1, so MATLAB's "0 means unset" bookkeeping is simply a
    # min / max over the candidates with 0 for columns that were never touched.
    cand_min = np.where(overflow, M + 1, y_use - 1)
    cand_max = np.where(overflow, M + 1, y_use)
    min_y = np.full(N, np.iinfo(np.int64).max, dtype=np.int64)
    max_y = np.zeros(N, dtype=np.int64)
    np.minimum.at(min_y, col, cand_min)
    np.maximum.at(max_y, col, cand_max)
    min_y[min_y == np.iinfo(np.int64).max] = 0
    return _parity_scan(out, min_y, max_y)


def _parity_scan(out: np.ndarray, min_y: np.ndarray, max_y: np.ndarray) -> np.ndarray:
    """``parityScan`` of ``eml/poly2mask.m``: cumulative XOR down each column between ``minY`` and ``maxY-1``."""
    M, N = out.shape
    rows = np.arange(M)[:, None]
    in_range = (min_y[None, :] > 0) & (rows >= (min_y[None, :] - 1)) & (rows <= (max_y[None, :] - 2))
    parity = (np.cumsum(out.astype(np.int64), axis=0) % 2).astype(bool)
    return np.where(in_range, parity, out)


def roipoly(m, n=None, xi=None, yi=None) -> np.ndarray:
    """MATLAB's **non-interactive** ``roipoly(M, N, xi, yi)`` / ``roipoly(I, xi, yi)`` = :func:`poly2mask`.

    Book: Ch. 8 §8.2 (``sea_ice_model.m`` line ``bw = roipoly(img, xp, yp)`` — the convex polygon fitted to a
    floe) and ``regionprops``'s ``ConvexImage`` (``roipoly(M, N, c, r)``).  There is no interactivity problem:
    with three or four arguments MATLAB's ``roipoly`` never opens a figure.  Parity: exact (see
    :func:`poly2mask`).
    """
    if yi is None:  # roipoly(I, xi, yi): m is the image, n = xi, xi = yi
        img = np.asarray(m)
        if img.ndim < 2:
            raise ValueError("roipoly(I, xi, yi) needs a 2-D image as the first argument")
        return poly2mask(n, xi, img.shape[0], img.shape[1])
    return poly2mask(xi, yi, int(m), int(n))  # roipoly(M, N, xi, yi)


# --------------------------------------------------------------------------------------------------------------
# convhull / polyarea
# --------------------------------------------------------------------------------------------------------------

def convhull(x, y, simplify: bool = True) -> np.ndarray:
    """MATLAB ``convhull(x, y)``: 0-based indices of the convex hull, counter-clockwise, **closed**
    (``k[0] == k[-1]``).

    Book: used by ``minboundrect.m`` (``convhull(x, y, {'Qt'})``), by ``sea_ice_model.m``
    (``convhull(x, y, 'simplify', true)`` — Ch. 8 §8.2) and by ``regionprops``'s ``ConvexHull``.
    Built on ``scipy.spatial.ConvexHull``; degenerate inputs (fewer than three points, or all points collinear)
    are handled explicitly because Qhull raises for them while the callers here need the two extreme points.

    Parity: **near** (`reports/ch06_verification.md`).  Two documented differences from MATLAB, neither of which
    changes any value this project derives from the hull: the starting vertex may differ (filling, rasterising
    and the rotating-caliper loop are all rotation invariant), and MATLAB's `convhull` **keeps collinear hull
    points** while Qhull's ``'Qt'`` merges them, so this returns a subset of MATLAB's index list
    (`tests/test_ch06.py` asserts ``mine <= theirs`` for exactly that reason).  The hull *polygon* is identical.
    """
    from scipy.spatial import ConvexHull, QhullError

    pts = np.column_stack([np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()])
    n = pts.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n <= 2:
        return np.r_[np.arange(n), 0]
    try:
        hull = ConvexHull(pts, qhull_options="Qt" if simplify else "Qt Qc")
        v = hull.vertices
    except QhullError:  # collinear / duplicate points: fall back to the two extreme points
        order = np.lexsort((pts[:, 1], pts[:, 0]))
        v = np.array([order[0], order[-1]])
        if v[0] == v[1]:
            v = v[:1]
    return np.r_[v, v[0]].astype(np.int64)


def polyarea(x, y) -> float:
    """MATLAB ``polyarea``: the (unsigned) shoelace area of a simple polygon.

    Parity: **reimplemented** -- the shoelace formula is exact for a simple polygon and agrees with MATLAB on the
    L1 fixtures, but there is no MATLAB reference run for it (review finding S6, 2026-09-10).  Results for a
    self-intersecting polygon are the signed-area cancellation both implementations happen to produce, not a
    guaranteed match.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


# --------------------------------------------------------------------------------------------------------------
# polygeom -- H. J. Sommer III
# --------------------------------------------------------------------------------------------------------------

def polygeom(x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``[geom, iner, cpmo] = polygeom(x, y)`` — area, centroid, perimeter and area moments of inertia of a
    planar polygon by Green's-theorem boundary integrals.

    Book: Ch. 8 §8.2 / Appendix B (``sea_ice_model.m`` uses ``polygeom`` on the convex polygon fitted to each
    ice floe to obtain its area, centroid and perimeter).  MATLAB source:
    ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/polygeom.m`` (H. J. Sommer III, Penn State, File Exchange).

    Returns
    -------
    geom : ``[area, x_cen, y_cen, perimeter]``
    iner : ``[Ixx, Iyy, Ixy, Iuu, Ivv, Iuv]`` (``u, v`` are centroidal axes parallel to ``x, y``)
    cpmo : ``[I1, ang1, I2, ang2, J]`` (principal moments, **angles in radians**, ``J = I1 + I2``)

    The M-file's own self-test (3×5 rectangle with its long axis at 30°) gives ``area 15``,
    ``centroid (3.415, 6.549)``, ``perimeter 16``, ``I1 11.249``, ``ang1 30°`` — reproduced by this port.
    Parity: **exact** for ``geom``, ``iner``, ``I1``, ``I2`` and ``J`` (same closed-form integrals);
    **near** for ``ang1``/``ang2``, whose sign follows the eigenvector convention -- LAPACK's sign is arbitrary,
    so the port fixes it so the largest component is positive, which reproduces the M-file's own ``ang1 = 30°``.
    The reference test only asserts the angles agree modulo π (corrected 2026-09-10, review finding S6).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("X and Y must be the same size")
    n = x.size
    xm = x.mean()
    ym = y.mean()
    x = x - xm  # temporarily shift to the mean of the vertices for accuracy
    y = y - ym
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y

    A = np.sum(y * dx - x * dy) / 2.0
    Axc = np.sum(6 * x * y * dx - 3 * x * x * dy + 3 * y * dx * dx + dx * dx * dy) / 12.0
    Ayc = np.sum(3 * y * y * dx - 6 * x * y * dy - 3 * x * dy * dy - dx * dy * dy) / 12.0
    Ixx = np.sum(2 * y ** 3 * dx - 6 * x * y * y * dy - 6 * x * y * dy * dy
                 - 2 * x * dy ** 3 - 2 * y * dx * dy * dy - dx * dy ** 3) / 12.0
    Iyy = np.sum(6 * x * x * y * dx - 2 * x ** 3 * dy + 6 * x * y * dx * dx
                 + 2 * y * dx ** 3 + 2 * x * dx * dx * dy + dx ** 3 * dy) / 12.0
    Ixy = np.sum(6 * x * y * y * dx - 6 * x * x * y * dy + 3 * y * y * dx * dx
                 - 3 * x * x * dy * dy + 2 * y * dx * dx * dy - 2 * x * dx * dy * dy) / 24.0
    P = float(np.sum(np.sqrt(dx * dx + dy * dy)))

    if A < 0:  # clockwise boundary
        A, Axc, Ayc, Ixx, Iyy, Ixy = -A, -Axc, -Ayc, -Ixx, -Iyy, -Ixy

    xc = Axc / A
    yc = Ayc / A
    Iuu = Ixx - A * yc * yc
    Ivv = Iyy - A * xc * xc
    Iuv = Ixy - A * xc * yc
    J = Iuu + Ivv

    x_cen = xc + xm
    y_cen = yc + ym
    Ixx = Iuu + A * y_cen * y_cen
    Iyy = Ivv + A * x_cen * x_cen
    Ixy = Iuv + A * x_cen * y_cen

    I = np.array([[Iuu, -Iuv], [-Iuv, Ivv]], dtype=np.float64)
    eig_val, eig_vec = np.linalg.eigh(I)  # ascending eigenvalues, like MATLAB's eig for symmetric input
    # DEVIATION: `near` — an eigenvector's sign is arbitrary, and LAPACK's choice differs between MATLAB's
    # `eig` and numpy's `eigh` (the principal-axis angle then differs by exactly pi).  The sign is fixed here so
    # that the component of largest magnitude is positive, which reproduces the M-file's own self-test
    # (ang1 = 30 deg, ang2 = 120 deg); the principal *axes* are identical either way.
    for c in range(2):
        if eig_vec[np.argmax(np.abs(eig_vec[:, c])), c] < 0:
            eig_vec[:, c] = -eig_vec[:, c]
    I1, I2 = float(eig_val[0]), float(eig_val[1])
    ang1 = float(np.arctan2(eig_vec[1, 0], eig_vec[0, 0]))
    ang2 = float(np.arctan2(eig_vec[1, 1], eig_vec[0, 1]))

    geom = np.array([A, x_cen, y_cen, P], dtype=np.float64)
    iner = np.array([Ixx, Iyy, Ixy, Iuu, Ivv, Iuv], dtype=np.float64)
    cpmo = np.array([I1, ang1, I2, ang2, J], dtype=np.float64)
    return geom, iner, cpmo


# --------------------------------------------------------------------------------------------------------------
# minboundrect -- John D'Errico
# --------------------------------------------------------------------------------------------------------------

def minboundrect(x, y, metric: str = "a") -> tuple[np.ndarray, np.ndarray, float, float]:
    """``[rectx, recty, area, perimeter] = minboundrect(x, y, metric)`` — the minimal bounding rectangle of a
    point set (rotating calipers over the convex-hull edge angles).

    Book: Ch. 9 §9.3 "rectangularization" (p. 209) and the third re-segmentation criterion of ch9 p. 205
    ("the length-to-width ratio of the minimum area-bounding rectangle").  Note that ``GVF_distance.m``
    implements that criterion with the **ellipse** axis ratio ``MajorAxisLength/MinorAxisLength`` instead — a
    documented deviation of the shipped code from the text (analysis §2 note).  MATLAB source:
    ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/minboundrect.m`` (John D'Errico, File Exchange, Release 3.0).

    ``metric`` is ``'a'`` (minimum **area**, default) or ``'p'`` (minimum **perimeter**); any unambiguous
    contraction of ``'area'``/``'perimeter'`` is accepted, as in the original.  The ``nedges ∈ {0, 1, 2}``
    special cases are reproduced.  The author's own example (50 000 uniform points in the unit square) gives
    ``area ≈ 0.99994``.  Parity: exact (same algorithm, ``convhull`` via Qhull).
    """
    m = str(metric).lower()
    if m and ("area".startswith(m) or "perimeter".startswith(m)):
        m = m[0]
    else:
        raise ValueError("metric does not match either 'area' or 'perimeter'")

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.size
    if n != y.size:
        raise ValueError("x and y must be the same sizes")

    if n > 3:
        edges = convhull(x, y)
        x = x[edges]
        y = y[edges]
        nedges = x.size - 1
    elif n > 1:
        nedges = n
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    else:
        nedges = n

    if nedges == 0:
        return np.zeros(0), np.zeros(0), float("nan"), float("nan")
    if nedges == 1:
        return np.repeat(x[:1], 5), np.repeat(y[:1], 5), 0.0, 0.0
    if nedges == 2:
        idx = [0, 1, 1, 0, 0]
        per = float(2 * np.sqrt(np.diff(x[:2])[0] ** 2 + np.diff(y[:2])[0] ** 2))
        return x[idx], y[idx], 0.0, per

    ind = np.arange(x.size - 1)
    edgeangles = np.arctan2(y[ind + 1] - y[ind], x[ind + 1] - x[ind])
    edgeangles = np.unique(np.mod(edgeangles, np.pi / 2.0))  # move every angle into the first quadrant

    area = np.inf
    perimeter = np.inf
    met = np.inf
    rectx = recty = np.zeros(5)
    xy = np.column_stack([x, y])
    for theta in edgeangles:
        rot = np.array([[np.cos(-theta), np.sin(-theta)], [-np.sin(-theta), np.cos(-theta)]])
        xyr = xy @ rot
        xymin = xyr.min(axis=0)
        xymax = xyr.max(axis=0)
        A_i = float(np.prod(xymax - xymin))
        P_i = float(2.0 * np.sum(xymax - xymin))
        M_i = A_i if m == "a" else P_i
        if M_i < met:
            met = M_i
            area = A_i
            perimeter = P_i
            rect = np.array([xymin, [xymax[0], xymin[1]], xymax, [xymin[0], xymax[1]], xymin])
            rect = rect @ rot.T
            rectx = rect[:, 0]
            recty = rect[:, 1]
    return rectx, recty, area, perimeter


# --------------------------------------------------------------------------------------------------------------
# polyxpoly
# --------------------------------------------------------------------------------------------------------------

def polyxpoly(x1, y1, x2, y2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mapping Toolbox ``[xi, yi, ii] = polyxpoly(x1, y1, x2, y2)`` — intersection points of two polylines.

    Book: Ch. 8 §8.2 / Appendix B ``.Intersect`` (``sea_ice_model.m`` tests whether two floe polygons overlap).
    Brute-force segment/segment intersection; returns the intersection points and the 0-based indices
    ``[i1, i2]`` of the segments that produced them, sorted by ``(i1, i2)`` as MATLAB does.

    # DEVIATION: `reimplemented` — the Mapping Toolbox routine is not available as readable source; collinear
    # overlaps are reported by their two endpoints here, and duplicate points (a vertex shared by two segments)
    # are removed, which is what MATLAB documents.  Parity target: `reimplemented`/`near` (positions exact to
    # floating point; the ordering of degenerate collinear cases may differ).
    """
    x1 = np.asarray(x1, dtype=np.float64).ravel()
    y1 = np.asarray(y1, dtype=np.float64).ravel()
    x2 = np.asarray(x2, dtype=np.float64).ravel()
    y2 = np.asarray(y2, dtype=np.float64).ravel()
    pts: list[tuple[float, float, int, int]] = []
    for i in range(x1.size - 1):
        p = np.array([x1[i], y1[i]])
        r = np.array([x1[i + 1], y1[i + 1]]) - p
        for j in range(x2.size - 1):
            q = np.array([x2[j], y2[j]])
            s = np.array([x2[j + 1], y2[j + 1]]) - q
            denom = r[0] * s[1] - r[1] * s[0]
            qp = q - p
            if denom == 0:
                if abs(qp[0] * r[1] - qp[1] * r[0]) > 1e-12:
                    continue  # parallel, non-collinear
                rr = float(r @ r)
                if rr == 0:
                    continue
                t0 = float(qp @ r) / rr
                t1 = t0 + float(s @ r) / rr
                lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
                lo, hi = max(lo, 0.0), min(hi, 1.0)
                if lo <= hi:
                    for t in ({lo, hi}):
                        pt = p + t * r
                        pts.append((float(pt[0]), float(pt[1]), i, j))
                continue
            t = (qp[0] * s[1] - qp[1] * s[0]) / denom
            u = (qp[0] * r[1] - qp[1] * r[0]) / denom
            if -1e-12 <= t <= 1 + 1e-12 and -1e-12 <= u <= 1 + 1e-12:
                pt = p + t * r
                pts.append((float(pt[0]), float(pt[1]), i, j))
    if not pts:
        return np.zeros(0), np.zeros(0), np.zeros((0, 2), dtype=np.int64)
    pts.sort(key=lambda t: (t[2], t[3]))
    seen: list[tuple[float, float, int, int]] = []
    for p in pts:
        if not any(abs(p[0] - q[0]) < 1e-12 and abs(p[1] - q[1]) < 1e-12 for q in seen):
            seen.append(p)
    arr = np.asarray(seen, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2:4].astype(np.int64)
