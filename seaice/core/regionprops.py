"""MATLAB ``regionprops`` — Area, Centroid, BoundingBox, ConvexHull/ConvexImage/ConvexArea, Solidity,
Major/MinorAxisLength, Eccentricity, Orientation and Perimeter, computed with **MATLAB's own algorithms**.

Book: the three re-segmentation criteria that ``GVF_distance.m`` / ``seaice_kmean_GVF_forenhancement.m`` apply
to every connected component (book Ch. 9 p. 205, used by the ch6 code): ``Area > Ra``, ``Solidity < Rc``
(criterion 2 — "the ratio between the floe area and its minimum area-bounding polygon area"), and
``MajorAxisLength/MinorAxisLength > Rl`` (criterion 3).  Also Ch. 7 §7.1 (shape enhancement) and Ch. 8 §8.2/§8.3.

MATLAB sources: ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/GVF_distance.m`` lines 85–94 and
``seaice_kmean_GVF_forenhancement.m`` lines 106–115 (``regionprops(label == n, 'Area' | 'Solidity' |
'MajorAxisLength' | 'MinorAxisLength')``), ``for test/dist.m`` lines 29–30 (``'centroid'``); the algorithms are
ported from R2025a ``toolbox/images/images/regionprops.m`` (``ComputeEllipseParams``,
``ComputePerimeterCornerPixelList``, ``ComputeConvexHull``, ``ComputeConvexImage``, ``ComputeSolidity``,
``computePerimeterFromBoundary``).

Why not skimage
---------------
``skimage.measure.regionprops`` 0.26 disagrees with MATLAB on exactly the quantities this chapter thresholds.
On the 12×14 fixture of ``analysis/ch06.md`` §0.5 MATLAB gives ``MajorAxisLength 10.35814``,
``MinorAxisLength 6.76639``, ``Perimeter 25.42200``, ``Orientation −26.04716``, while skimage gives
``10.2935``, ``6.6672``, ``26.8284`` (+5.5 %) and ``+63.95°``.  The chapter's ``Rc = 0.9`` and ``Rl = 2`` sit on
those decision boundaries, so a 1 % error changes which components get re-segmented — and therefore the whole
output.  The differences come from MATLAB's ``+1/12`` second-moment correction (the normalised second central
moment of a unit-length pixel), the ``y``-negation that measures orientation counter-clockwise, the mid-edge
(``bwperim(·, 8)``) convex hull, and the Vossepoel–Smeulders perimeter estimator.

Performance
-----------
``regionprops`` is called here **once per label image**, not once per component: the label image is scanned a
constant number of times and the per-component work is limited to that component's perimeter pixels.  MATLAB's
own ``regionprops(label == n, …)`` loop is ``O(num · M · N)`` and would take minutes on the chapter's images
(Risk R4 of ``analysis/ch06.md``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .connectivity import label_components
from .polygon import convhull, poly2mask

__all__ = ["RegionProps", "regionprops", "region_table", "PROPERTIES"]

#: Properties this module computes (superset of what ch6–ch9 request).
PROPERTIES = ("Area", "Centroid", "BoundingBox", "ConvexHull", "ConvexImage", "ConvexArea", "Solidity",
              "MajorAxisLength", "MinorAxisLength", "Eccentricity", "Orientation", "Perimeter", "PixelIdxList",
              "PixelList", "Image")


@dataclass
class RegionProps:
    """One MATLAB ``regionprops`` struct element.  All coordinates are **1-based MATLAB** conventions.

    Attributes
    ----------
    Area : float
        Number of pixels.
    Centroid : ndarray (2,)
        ``[x, y]`` = ``[mean(col) + 1, mean(row) + 1]``.
    BoundingBox : ndarray (4,)
        ``[minC-0.5, minR-0.5, width, height]``.
    ConvexHull : ndarray (P, 2)
        ``[x, y]`` of the hull through **pixel corners**, closed.
    ConvexImage : ndarray of bool
        The filled hull inside the bounding box (``roipoly``).
    ConvexArea, Solidity, MajorAxisLength, MinorAxisLength, Eccentricity, Orientation, Perimeter : float
    PixelIdxList : ndarray
        0-based **linear (column-major)** indices, as MATLAB stores them.
    PixelList : ndarray (n, 2)
        ``[x, y]`` 1-based coordinates.
    Image : ndarray of bool
        The component cropped to its bounding box.
    """

    label: int = 0
    Area: float = 0.0
    Centroid: np.ndarray = field(default_factory=lambda: np.zeros(2))
    BoundingBox: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.0, 0.0]))
    ConvexHull: np.ndarray | None = None
    ConvexImage: np.ndarray | None = None
    ConvexArea: float = 0.0
    Solidity: float = float("nan")
    MajorAxisLength: float = 0.0
    MinorAxisLength: float = 0.0
    Eccentricity: float = 0.0
    Orientation: float = 0.0
    Perimeter: float = 0.0
    PixelIdxList: np.ndarray | None = None
    PixelList: np.ndarray | None = None
    Image: np.ndarray | None = None


def _as_label_image(L, conn: int) -> np.ndarray:
    """Accept a label matrix or a binary image (which is labelled with ``conn``, MATLAB's ``bwlabel`` numbering).

    A **signed integer** array (what :func:`seaice.core.connectivity.label_components` returns: int32) is always
    taken as a label matrix, even when it holds a single region — re-labelling it would silently change the
    connectivity the caller chose.  ``bool`` and unsigned/float arrays whose maximum is <= 1 are treated as
    binary images.
    """
    a = np.asarray(L)
    if a.dtype == np.bool_:
        return label_components(a, conn)
    if np.issubdtype(a.dtype, np.signedinteger):
        if a.min() < 0:
            raise ValueError("a label matrix must be non-negative")
        return a.astype(np.int64)
    if np.issubdtype(a.dtype, np.floating) and not np.all(np.isfinite(a)):
        raise ValueError("label matrix must be finite")
    if a.max() <= 1:
        return label_components(a != 0, conn)
    return a.astype(np.int64)


def _perimeter_pixels(L: np.ndarray) -> np.ndarray:
    """``bwmorph(component_image, 'perim8')`` for **every** component at once.

    A pixel of component ``k`` is on its 8-connected perimeter when at least one of its eight neighbours is not
    part of component ``k`` (outside the image counts as background — MATLAB's ``bwperim`` pads with 0).  Working
    on the label image rather than on a binary mask is essential: with 4-connected labelling two *different*
    components may touch diagonally, and then a pixel that has eight nonzero neighbours in the binary mask is
    still a perimeter pixel of its own component's ``Image``.
    """
    P = np.pad(L, 1, mode="constant", constant_values=0)
    M, N = L.shape
    out = np.zeros((M, N), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out |= P[1 + dr:1 + dr + M, 1 + dc:1 + dc + N] != L
    return out & (L > 0)


def _trace_boundary(img: np.ndarray) -> np.ndarray:
    """8-connected Moore boundary trace of the single object in ``img`` (0-based ``(row, col)``, closed).

    Equivalent to MATLAB's ``images.internal.builtins.regionboundaries(double(Image), 8)`` used by
    ``computePerimeterFromBoundary``: start at the first object pixel in **column-major** order, walk the Moore
    neighbourhood clockwise, stop when the start pixel is reached again in the same direction (Jacob's
    criterion), and close the list.
    """
    rows, cols = np.nonzero(img)
    if rows.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    order = np.lexsort((rows, cols))  # column-major first occurrence
    r0, c0 = int(rows[order[0]]), int(cols[order[0]])
    if rows.size == 1:
        return np.array([[r0, c0], [r0, c0]], dtype=np.int64)
    # Moore neighbourhood in clockwise order starting from "west"
    nb = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
    M, N = img.shape

    def val(r: int, c: int) -> bool:
        return 0 <= r < M and 0 <= c < N and bool(img[r, c])

    boundary = [(r0, c0)]
    start_dir = 0
    cur = (r0, c0)
    d = start_dir
    first_step = None
    while True:
        found = False
        for k in range(8):
            dd = (d + k) % 8
            rr, cc = cur[0] + nb[dd][0], cur[1] + nb[dd][1]
            if val(rr, cc):
                nxt = (rr, cc)
                d = (dd + 6) % 8  # back up two positions (counter-clockwise) for the next search
                found = True
                break
        if not found:
            break
        if first_step is None:
            first_step = nxt
        elif cur == (r0, c0) and nxt == first_step:
            break
        boundary.append(nxt)
        cur = nxt
        if len(boundary) > 8 * img.size + 8:
            break
    if boundary[-1] != boundary[0]:
        boundary.append(boundary[0])
    return np.asarray(boundary, dtype=np.int64)


def _perimeter_from_boundary(B: np.ndarray) -> float:
    """``computePerimeterFromBoundary`` of R2025a ``regionprops.m`` — the Vossepoel–Smeulders estimator
    ``0.980·(even steps) + 1.406·(odd steps) − 0.091·(corners)``; 0 for a boundary of one pixel or less.

    Reference: Vossepoel & Smeulders, "Vector Code Probability and Metrication Error in the Representation of
    Straight Lines of Finite Length", *Computer Graphics and Image Processing* **20**:347–364, 1982.
    """
    if B.shape[0] < 2:
        return 0.0
    delta = np.diff(B, axis=0) ** 2
    if delta.shape[0] <= 1:
        return 0.0
    is_corner = np.any(np.diff(np.vstack([delta, delta[0:1]]), axis=0) != 0, axis=1)
    is_even = np.any(delta == 0, axis=1)  # a horizontal/vertical step has one zero component
    return float(np.sum(is_even) * 0.980 + np.sum(~is_even) * 1.406 - np.sum(is_corner) * 0.091)


def regionprops(L, properties: str | tuple[str, ...] | list[str] | None = None, conn: int = 8) -> list[RegionProps]:
    """MATLAB ``regionprops(L, properties)`` on a label matrix or a binary image.

    Book: the ``Area`` / ``Solidity`` / ``MajorAxisLength`` / ``MinorAxisLength`` criteria of ch9 p. 205 applied
    in ch6's ``GVF_distance.m`` (lines 85–94) and ``Centroid`` for the contour seeds (§6.3.3).
    MATLAB source: R2025a ``toolbox/images/images/regionprops.m``.

    Parameters
    ----------
    L : ndarray
        Label matrix (integer with more than one region) or binary image, which is labelled with ``conn``
        (default 8, MATLAB's ``bwlabel`` default) in MATLAB's column-major numbering.
    properties : str or sequence of str, optional
        Which properties to compute (``'all'`` / ``None`` = every entry of :data:`PROPERTIES`).  Requesting only
        what is needed matters: ``ConvexArea``/``Solidity`` cost a convex hull and a ``poly2mask`` per component.
    conn : int
        Connectivity used when ``L`` is binary.

    Returns
    -------
    list of :class:`RegionProps`, one per label ``1 … max(L)``, in label order (MATLAB's struct array order).

    Parity: **exact** — measured ≤ 1.07e-14 against MATLAB R2025a across 40 shapes × 10 properties and all 344
    components of the three real masks, in both MATLAB call forms (`reports/ch06_verification.md`).  ``Perimeter``
    is included in that and is exact; the earlier ``near`` caveat about the boundary tracer was written before the
    measurement and is withdrawn (corrected 2026-09-10, review finding S6).
    """
    lab = _as_label_image(L, conn)
    if properties is None or (isinstance(properties, str) and properties.lower() == "all"):
        want = set(PROPERTIES)
    elif isinstance(properties, str):
        want = {_canonical(properties)}
    else:
        want = {_canonical(p) for p in properties}
    # dependencies
    if want & {"Solidity", "ConvexArea"}:
        want |= {"ConvexImage", "Area"}
    if "ConvexImage" in want:
        want |= {"ConvexHull", "BoundingBox"}
    if want & {"MajorAxisLength", "MinorAxisLength", "Eccentricity", "Orientation"}:
        want |= {"Centroid"}

    n = int(lab.max()) if lab.size else 0
    stats = [RegionProps(label=k + 1) for k in range(n)]
    if n == 0:
        return stats

    rows, cols = np.nonzero(lab)
    labels = lab[rows, cols]
    order = np.argsort(labels, kind="stable")
    rows, cols, labels = rows[order], cols[order], labels[order]
    starts = np.searchsorted(labels, np.arange(1, n + 1), side="left")
    ends = np.searchsorted(labels, np.arange(1, n + 1), side="right")
    M, N = lab.shape

    counts = (ends - starts).astype(np.float64)
    sum_x = np.add.reduceat(cols.astype(np.float64) + 1.0, starts) if n else np.zeros(0)
    sum_y = np.add.reduceat(rows.astype(np.float64) + 1.0, starts) if n else np.zeros(0)
    # np.add.reduceat needs non-empty groups; components of a label matrix are never empty by construction
    cx = sum_x / counts
    cy = sum_y / counts

    need_perim_pixels = bool(want & {"ConvexHull", "ConvexImage", "ConvexArea", "Solidity"})
    if need_perim_pixels:
        perim = _perimeter_pixels(lab)
        prow, pcol = np.nonzero(perim)
        plab = lab[prow, pcol]
        porder = np.argsort(plab, kind="stable")
        prow, pcol, plab = prow[porder], pcol[porder], plab[porder]
        pstarts = np.searchsorted(plab, np.arange(1, n + 1), side="left")
        pends = np.searchsorted(plab, np.arange(1, n + 1), side="right")

    for k in range(n):
        s, e = starts[k], ends[k]
        st = stats[k]
        r = rows[s:e]
        c = cols[s:e]
        if "Area" in want:
            st.Area = float(e - s)
        if "Centroid" in want:
            st.Centroid = np.array([cx[k], cy[k]])
        min_r, max_r = int(r.min()), int(r.max())
        min_c, max_c = int(c.min()), int(c.max())
        bbox = np.array([min_c + 1 - 0.5, min_r + 1 - 0.5, max_c - min_c + 1.0, max_r - min_r + 1.0])
        if "BoundingBox" in want:
            st.BoundingBox = bbox
        if "PixelIdxList" in want:
            st.PixelIdxList = (c.astype(np.int64) * M + r.astype(np.int64))  # column-major linear index, 0-based
        if "PixelList" in want:
            st.PixelList = np.column_stack([c + 1, r + 1])
        if want & {"Image", "Perimeter"}:
            sub = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
            sub[r - min_r, c - min_c] = True
            if "Image" in want:
                st.Image = sub
            if "Perimeter" in want:
                st.Perimeter = _perimeter_from_boundary(_trace_boundary(sub))
        if want & {"MajorAxisLength", "MinorAxisLength", "Eccentricity", "Orientation"}:
            _ellipse_params(st, c.astype(np.float64) + 1.0, r.astype(np.float64) + 1.0, cx[k], cy[k])
        if need_perim_pixels and (want & {"ConvexHull", "ConvexImage", "ConvexArea", "Solidity"}):
            ps, pe = pstarts[k], pends[k]
            hull = _convex_hull(prow[ps:pe], pcol[ps:pe])
            if "ConvexHull" in want:
                st.ConvexHull = hull
            if want & {"ConvexImage", "ConvexArea", "Solidity"}:
                h_img = _convex_image(hull, bbox)
                if "ConvexImage" in want:
                    st.ConvexImage = h_img
                st.ConvexArea = float(h_img.sum())
                if "Solidity" in want:
                    st.Solidity = float(st.Area / st.ConvexArea) if st.ConvexArea else float("nan")
    return stats


def _canonical(name: str) -> str:
    """Case-insensitive property name (MATLAB accepts ``'centroid'`` for ``'Centroid'``)."""
    low = {p.lower(): p for p in PROPERTIES}
    key = str(name).lower()
    if key not in low:
        raise ValueError(f"unknown property {name!r}; known: {', '.join(PROPERTIES)}")
    return low[key]


def _ellipse_params(st: RegionProps, x: np.ndarray, y: np.ndarray, xbar: float, ybar: float) -> None:
    """``ComputeEllipseParams`` of R2025a ``regionprops.m`` (equivalent ellipse of the second moments).

    ``uxx = Σx²/N + 1/12``, ``uyy = Σy²/N + 1/12`` (``1/12`` is the normalised second central moment of a
    unit-length pixel — the term skimage omits), ``uxy = Σxy/N`` with ``y`` **negated** so the orientation is
    measured counter-clockwise from the horizontal axis; then
    ``Major/Minor = 2√2 √(uxx + uyy ± common)`` with ``common = √((uxx − uyy)² + 4uxy²)`` and
    ``Orientation = (180/π)·atan(num/den)`` in degrees.
    """
    xs = x - xbar
    ys = -(y - ybar)
    N = xs.size
    uxx = float(np.sum(xs * xs) / N + 1.0 / 12.0)
    uyy = float(np.sum(ys * ys) / N + 1.0 / 12.0)
    uxy = float(np.sum(xs * ys) / N)
    common = np.sqrt((uxx - uyy) ** 2 + 4.0 * uxy ** 2)
    st.MajorAxisLength = float(2.0 * np.sqrt(2.0) * np.sqrt(uxx + uyy + common))
    st.MinorAxisLength = float(2.0 * np.sqrt(2.0) * np.sqrt(max(uxx + uyy - common, 0.0)))
    st.Eccentricity = float(2.0 * np.sqrt((st.MajorAxisLength / 2.0) ** 2 - (st.MinorAxisLength / 2.0) ** 2)
                            / st.MajorAxisLength) if st.MajorAxisLength else 0.0
    if uyy > uxx:
        num = uyy - uxx + np.sqrt((uyy - uxx) ** 2 + 4.0 * uxy ** 2)
        den = 2.0 * uxy
    else:
        num = 2.0 * uxy
        den = uxx - uyy + np.sqrt((uxx - uyy) ** 2 + 4.0 * uxy ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        # den == 0 with num != 0 gives atan(Inf) = pi/2 = 90 degrees in MATLAB too (an axis-aligned region).
        st.Orientation = 0.0 if (num == 0 and den == 0) else float(np.degrees(np.arctan(num / den)))


def _convex_hull(prow: np.ndarray, pcol: np.ndarray) -> np.ndarray:
    """``ComputeConvexHull``: the convex hull of the **corner mid-points** of the perimeter pixels.

    Every perimeter pixel ``(r, c)`` contributes the four points ``(c, r±0.5)`` and ``(c±0.5, r)``, so the hull
    polygon goes through pixel *edges*, not pixel centres.  Returns the closed hull as ``[x, y]`` rows,
    1-based image coordinates.
    """
    if prow.size == 0:
        return np.zeros((0, 2))
    r = prow.astype(np.float64) + 1.0
    c = pcol.astype(np.float64) + 1.0
    rr = np.concatenate([r - 0.5, r, r + 0.5, r])
    cc = np.concatenate([c, c + 0.5, c, c - 0.5])
    idx = convhull(rr, cc)  # MATLAB: convhull(rr, cc) then list(hullIdx, :) with list = [cc rr]
    return np.column_stack([cc[idx], rr[idx]])


def _convex_image(hull: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """``ComputeConvexImage``: fill the hull inside the bounding box with ``roipoly(M, N, c, r)``."""
    if hull.size == 0:
        return np.zeros((int(bbox[3]), int(bbox[2])), dtype=bool)
    M = int(bbox[3])
    N = int(bbox[2])
    first_row = bbox[1] + 0.5
    first_col = bbox[0] + 0.5
    c = hull[:, 0] - first_col + 1.0
    r = hull[:, 1] - first_row + 1.0
    return poly2mask(c, r, M, N)


def region_table(L, properties: tuple[str, ...] = ("Area", "Solidity", "MajorAxisLength", "MinorAxisLength"),
                 conn: int = 8) -> dict[str, np.ndarray]:
    """Column-wise view of :func:`regionprops` — ``{'Area': array, 'Solidity': array, …}`` in label order.

    This is the shape the ch6 code wants: ``a = cat(1, aa.Area)`` etc. followed by
    ``k = unique([find(a > Ra); find(rc < Rc); find(rl > Rl)])``.
    """
    stats = regionprops(L, properties, conn)
    return {p: np.array([getattr(s, _canonical(p)) for s in stats], dtype=np.float64) for p in properties}
