"""Chapter 5 — Watershed-Based Ice Floe Segmentation: the script pipelines and the text-only algorithms.

Book: Zhang & Skjetne (2018), Chapter 5, pp. 83–108.  The chapter's output is a binary floe mask in which touching
floes are separated by watershed lines of a *segmentation function* — the gray image itself (§5.1, useless), its
Sobel gradient (§5.1.1, Figs. 5.5–5.7), the inverted distance transform of the ice mask (§5.1.2, Figs. 5.8–5.11) or
the latter with imposed markers (§5.1.3, Fig. 5.12) — followed by the authors' **neighbouring-region merging**
(§5.2, Figs. 5.13–5.17): watershed junction lines whose two ending points are both convex (differential chain code,
Eqs. 5.9–5.19) are deleted.  The reusable primitives live in :mod:`seaice.core.watershed` (MATLAB ``watershed``)
and :mod:`seaice.core.morphology` (``imregionalmin``, ``imregionalmax``, ``imimposemin``).

MATLAB ↔ Python map (analysis/ch05.md §3):

* ``direct_watershed.m`` → :func:`direct_watershed` (driver ``scripts/ch05_direct_watershed.py``)
* ``distance_propagation.m`` → :func:`inverse_distance_map` (``scripts/ch05_distance_propagation.py``)
* ``distance_watershed.m`` → :func:`distance_watershed` (``scripts/ch05_distance_watershed.py``)
* ``gradients_watershed.m`` → :func:`sobel_magnitude`, :func:`gradient_watershed` (``scripts/ch05_gradients_watershed.py``)
* ``marker_watershed.m`` → :func:`marker_watershed`, :func:`component_centroids` (``scripts/ch05_marker_watershed.py``)
* ``topological_surface.m`` → :func:`topographic_surfaces` (``scripts/ch05_topological_surface.py``)
* ``chaincode_corner.m`` / ``watershed_based/freeman_concave.m`` → :func:`freeman_concave`,
  :func:`differential_chain_code` (``scripts/ch05_chaincode_corner.py``)
* ``watershed_based/main.m`` → :func:`neighboring_region_merging`, :func:`junction_endpoints`,
  :data:`ENDPOINT_KERNEL` (``scripts/ch05_main.py``)
* ``bound2im.m`` / ``boundaries.m`` / ``fchcode.m`` (both copies) → :mod:`seaice.core.chaincode` (ported in ch02)
* text-only Eq. (5.2) → :func:`regional_minima_by_reconstruction`; §5.1.3 Steps 1–2 → :func:`impose_minima_book`;
  Eqs. (5.1), (5.3)–(5.8) → :func:`watershed_immersion`
* §5.3 experiments (unshipped images) → ``scripts/ch05_experiments.py`` (drives :func:`neighboring_region_merging`)

Conventions: masks are bool, distance maps are **float32** (MATLAB ``bwdist`` returns single — analysis risk 14),
label images int32, coordinates 0-based ``(row, col)``; the scripts add 1 when quoting MATLAB positions.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .core import synth
from .core.chaincode import ChainCode, bound2im, boundaries, fchcode
from .core.connectivity import bwareaopen, label_components
from .core.distance import bwdist
from .core.filters import fspecial, imfilter
from .core.matlab_compat import imcomplement, rgb2gray_matlab, to_uint8_saturating
from .core.morphology import imclose, imdilate, imimposemin, imopen, imreconstruct, imregionalmin, \
    reconstruct_by_erosion, strel
from .core.threshold import graythresh, im2bw
from .core.watershed import watershed

#: Parameters quoted in the chapter text / used by the scripts (page numbers in analysis/ch05.md §6).
BOOK_PARAMS = {
    "close_open_size": 7,  # §5.1.1 p. 89: 7×7 square close-opening of the gradient (Fig. 5.6)
    "marker_disk_radius": 5,  # §5.1.3 pp. 96–97: minima dilated by a 5-radius disk (Fig. 5.12(b))
    "min_area": 5,  # distance_watershed.m / marker_watershed.m: bwareaopen(img, 5) (scripts only)
    "concave_min": 3,  # §5.2.1.2 p. 104: concave if 3 <= D(i) <= 10 (45°–150°)
    "concave_max": 10,
    "angle_per_code": 15.0,  # Eq. (5.19): theta = D(i) × 15°
    "metric_merging": "cityblock",  # §5.2 Step 2 p. 99 and main.m
    "metric_fig_5_8": "chessboard",  # distance_watershed.m / Fig. 5.8
}

#: Fig. 5.15(b) / ``main.m`` ``wr = [0 -1 0; -1 4 -1; 0 -1 0]`` — ending-point detection kernel (float64).
ENDPOINT_KERNEL = synth.FIG_5_15_KERNEL.copy()


# ---------------------------------------------------------------------------------------------------------------
# Shared literal steps of the scripts
# ---------------------------------------------------------------------------------------------------------------


def otsu_mask(img: np.ndarray) -> np.ndarray:
    """``im2bw(img, graythresh(img))`` exactly as the scripts write it.

    On an **RGB** array ``graythresh`` histograms all three planes (level 130/255 on ``q.jpg``) while ``im2bw``
    converts with ``rgb2gray`` — the mask differs from ``im2bw(rgb2gray(img), graythresh(rgb2gray(img)))``
    (128/255).  ``distance_watershed.m``, ``marker_watershed.m``, ``main.m`` and ``chaincode_corner.m`` binarise
    the RGB; ``direct_watershed.m``, ``gradients_watershed.m`` and ``topological_surface.m`` binarise the gray image
    — call this function with whatever the script passes (analysis/ch05.md risk 4).
    """
    img = np.asarray(img)
    level, _ = graythresh(img)
    return im2bw(img, level)


def inverse_distance(bw: np.ndarray, metric: str = "cityblock") -> np.ndarray:
    """``-bwdist(~bw, metric)`` in **single** precision (the scripts' ``imgDist``; §5.1.2 "inverse distance map").

    Book: §5.1.2 p. 90 ("the regional minima of the inverted distance transform are the regional maxima of the
    distance transform").  MATLAB source: ``distance_watershed.m`` line 5, ``marker_watershed.m`` line 5,
    ``main.m`` lines 14–15 (``D = bwdist(gc, 'cityblock'); L = watershed(-D)``).  float32 keeps the flooding
    priorities identical to MATLAB's (analysis/ch05.md risk 14).
    """
    bw = np.asarray(bw) != 0
    return (-bwdist(~bw, metric)).astype(np.float32)


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of the 0–255 gray image with **unscaled** kernels and replicate padding.

    Book: §5.1.1 p. 89 (3×3 Sobel, Fig. 5.5(a)).  MATLAB source: ``gradients_watershed.m`` lines 9–13 and
    ``topological_surface.m`` lines 13–17: ``hy = fspecial('sobel'); hx = hy'; Iy = imfilter(double(I), hy,
    'replicate'); Ix = imfilter(double(I), hx, 'replicate'); g = sqrt(Ix.^2 + Iy.^2)`` — no ``/8`` and no ``/255``
    (unlike MATLAB's ``edge``), so ``g`` is in 0–255·8 units.  Parity: exact (0.0 vs MATLAB).
    """
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = rgb2gray_matlab(gray)
    hy = fspecial("sobel")
    hx = hy.T
    I = gray.astype(np.float64)
    Iy = imfilter(I, hy, "replicate")
    Ix = imfilter(I, hx, "replicate")
    return np.sqrt(Ix ** 2 + Iy ** 2)


# ---------------------------------------------------------------------------------------------------------------
# direct_watershed.m (§5.1 p. 89 — the gray image as segmentation function; no book figure)
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class DirectWatershed:
    """Arrays of ``direct_watershed.m``: ``im`` (gray), ``img`` (Otsu mask, unused by the script), ``imgLabel``,
    ``bgm`` (ridges), the overlay ``im(bgm) = 0`` and ``colorimg = bwlabel(im)``."""

    gray: np.ndarray
    bw: np.ndarray
    L: np.ndarray
    ridge: np.ndarray
    overlay: np.ndarray
    labels: np.ndarray
    n_basins: int
    n_components: int


def direct_watershed(gray: np.ndarray) -> DirectWatershed:
    """Watershed of the gray image itself — port of ``MATLAB_ROOT/ch5/direct_watershed.m``.

    Book: §5.1 p. 89 ("the grayscale image itself is rarely a good segmentation function").  Steps: ``im =
    rgb2gray(img)``; ``img = im2bw(im, graythresh(im))`` (computed, unused); ``imgLabel = watershed(im)`` on the
    **uint8** image; ``bgm = imgLabel == 0``; ``im(bgm) = 0`` (ridges superimposed in black);
    ``colorimg = bwlabel(im)`` (nonzero gray = object, 8-connected).  Parity: exact.
    """
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = rgb2gray_matlab(gray)
    bw = otsu_mask(gray)
    L = watershed(gray)
    ridge = L == 0
    overlay = gray.copy()
    overlay[ridge] = 0
    labels = label_components(overlay != 0, 8)
    return DirectWatershed(gray, bw, L, ridge, overlay, labels, int(L.max()), int(labels.max()))


# ---------------------------------------------------------------------------------------------------------------
# distance_propagation.m (§5.1.2 pp. 93–94, Fig. 5.10)
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class DistancePropagation:
    """``imgDist`` (negated distance, single), ``dist0 = imgDist + |min|`` and ``dist = uint8(dist0·255/max)``."""

    imgDist: np.ndarray
    dist0: np.ndarray
    dist: np.ndarray


def inverse_distance_map(bw: np.ndarray, metric: str = "cityblock") -> DistancePropagation:
    """Inverse distance map and its uint8 rescaling — port of ``MATLAB_ROOT/ch5/distance_propagation.m``.

    Book: §5.1.2 pp. 93–94 (propagation shapes of the three metrics: diamond / circle / square, Fig. 5.10).
    MATLAB: ``img = zeros(201); img(101, 101) = 1; img = ~img; imgDist = -bwdist(~img, 'cityblock'); dist0 =
    imgDist + abs(min(min(imgDist))); dist = uint8(dist0 .* 255 / max(max(dist0)))`` (the commented lines use
    ``q.jpg`` instead).  ``bw`` is the *object* mask the script complements (``~img`` = the point, or the ice mask);
    arithmetic in single like MATLAB, ``uint8()`` = round half away from zero + saturate.  Parity: exact.
    """
    imgDist = inverse_distance(bw, metric)
    dist0 = (imgDist + np.float32(abs(float(imgDist.min())))).astype(np.float32)
    mx = np.float32(dist0.max())
    scaled = (dist0 * np.float32(255) / mx).astype(np.float32) if mx > 0 else np.zeros_like(dist0)
    dist = to_uint8_saturating(scaled)
    return DistancePropagation(imgDist, dist0, dist)


# ---------------------------------------------------------------------------------------------------------------
# distance_watershed.m (§5.1.2, Figs. 5.8, 5.9, 5.11)
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class DistanceWatershed:
    """Arrays of ``distance_watershed.m`` (names in parentheses): the Otsu mask (``img``), ``imgDist``, the regional
    minima ``Dis_img``, ``dis = Dis_img .* bw``, their ``find`` coordinates ``(p, q)``, ``bw0`` (mask with the minima
    punched out, Fig. 5.8(d)), ``imgLabel``, ``bgm`` (ridge), ``img(bgm) = 0`` (``seg``), ``bwareaopen(…, 5)``
    (``seg_ao``) and ``colorimg = bwlabel(…)``."""

    bw: np.ndarray
    imgDist: np.ndarray
    minima: np.ndarray
    dis: np.ndarray
    minima_points: np.ndarray
    minima_overlay: np.ndarray
    L: np.ndarray
    ridge: np.ndarray
    seg: np.ndarray
    seg_ao: np.ndarray
    labels: np.ndarray
    n_minima: int
    n_minimum_px: int
    n_basins: int
    n_floes: int
    metric: str


def _find_rows_cols(mask: np.ndarray) -> np.ndarray:
    """``[p, q] = find(mask)`` — 0-based (row, col) pairs in MATLAB's column-major order, shape (n, 2)."""
    cols, rows = np.nonzero(np.asarray(mask).T)
    return np.column_stack([rows, cols]).astype(np.int64)


def distance_watershed(bw: np.ndarray, metric: str = "chessboard", min_area: int = 5) -> DistanceWatershed:
    """Watershed of the inverse distance map — port of ``MATLAB_ROOT/ch5/distance_watershed.m``.

    Book: §5.1.2, Fig. 5.8 (chessboard: (a) mask, (b) ``-D``, (d) regional minima in black, (e) watershed line,
    (f) segmented floes), Fig. 5.9 (Euclidean / city-block rows), Fig. 5.11; the book's remarks p. 93–95: Euclidean
    → most over-segmentation ("islands"), city-block moderate, chessboard least but under-segments.
    MATLAB (line by line): ``img = im2bw(img, graythresh(img))`` (on the **RGB**, done by the caller — pass the
    mask); ``imgDist = -bwdist(~img, metric)``; ``Dis_img = imregionalmin(imgDist)``; ``dis = Dis_img .* bw``;
    ``[p, q] = find(dis == 1)``; ``bw0 = img; bw0(find(dis == 1)) = 0``; ``imgLabel = watershed(imgDist)``;
    ``bgm = imgLabel == 0``; ``img(bgm) = 0``; ``img = bwareaopen(img, 5)``; ``colorimg = bwlabel(img)``.
    Parity: exact (label values included).
    """
    bw = np.asarray(bw) != 0
    imgDist = inverse_distance(bw, metric)
    minima = imregionalmin(imgDist)
    dis = minima & bw  # logical .* logical (double 0/1 in MATLAB)
    pts = _find_rows_cols(dis)
    minima_overlay = bw.copy()
    minima_overlay[dis] = False
    L = watershed(imgDist)
    ridge = L == 0
    seg = bw.copy()
    seg[ridge] = False
    seg_ao = bwareaopen(seg, min_area)
    labels = label_components(seg_ao, 8)
    return DistanceWatershed(bw, imgDist, minima, dis, pts, minima_overlay, L, ridge, seg, seg_ao, labels,
                             int(label_components(minima, 8).max()), int(minima.sum()), int(L.max()),
                             int(labels.max()), metric)


# ---------------------------------------------------------------------------------------------------------------
# gradients_watershed.m (§5.1.1, Figs. 5.5, 5.6)
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class GradientWatershed:
    """Arrays of ``gradients_watershed.m``: ``g`` (Sobel magnitude), ``l``/``wr``/``f`` (labels, ridge, overlay),
    then ``g2 = imclose(imopen(g, ones(7)), ones(7))`` and its ``l2``/``wr2``/``f2`` (``None`` if ``smooth`` is off)."""

    gray: np.ndarray
    bw: np.ndarray
    g: np.ndarray
    L: np.ndarray
    ridge: np.ndarray
    overlay: np.ndarray
    n_basins: int
    g2: np.ndarray | None = None
    L2: np.ndarray | None = None
    ridge2: np.ndarray | None = None
    overlay2: np.ndarray | None = None
    n_basins2: int | None = None


def gradient_watershed(gray: np.ndarray, smooth: int | None = 7) -> GradientWatershed:
    """Watershed of the Sobel gradient, raw and after a square close-opening — port of ``gradients_watershed.m``.

    Book: §5.1.1 pp. 89–90: Fig. 5.5 (gradient, its watershed → far too many lines, overlay in black) and Fig. 5.6
    (gradient smoothed by a 7×7 square close-opening → fewer, still extraneous lines).  MATLAB: ``I = rgb2gray(I)``;
    ``bw = im2bw(I, graythresh(I))`` (gray threshold, unused); ``g = sqrt(Ix.^2 + Iy.^2)`` (:func:`sobel_magnitude`);
    ``l = watershed(g); wr = l == 0; f = I; f(wr) = 0``; ``g2 = imclose(imopen(g, ones(7,7)), ones(7,7)); l2 =
    watershed(g2); wr2 = l2 == 0; f2 = I; f2(wr2) = 0``.  ``smooth`` = side of the square SE (``None`` skips the
    second half).  Parity: exact (``g``/``g2`` 0.0, labels 0 px).
    """
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = rgb2gray_matlab(gray)
    bw = otsu_mask(gray)
    g = sobel_magnitude(gray)
    L = watershed(g)
    ridge = L == 0
    overlay = gray.copy()
    overlay[ridge] = 0
    res = GradientWatershed(gray, bw, g, L, ridge, overlay, int(L.max()))
    if smooth:
        se = np.ones((int(smooth), int(smooth)), dtype=bool)  # ones(7, 7) — an all-ones SE (imclose.m zero pre-pad)
        g2 = imclose(imopen(g, se), se)
        L2 = watershed(g2)
        ridge2 = L2 == 0
        overlay2 = gray.copy()
        overlay2[ridge2] = 0
        res.g2, res.L2, res.ridge2, res.overlay2, res.n_basins2 = g2, L2, ridge2, overlay2, int(L2.max())
    return res


# ---------------------------------------------------------------------------------------------------------------
# marker_watershed.m (§5.1.3, Fig. 5.12)
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class MarkerWatershed:
    """Arrays of ``marker_watershed.m``: ``imgDist`` before imposition (``imgDist0``), ``Dis_img`` (``minima``),
    ``marker = imdilate(Dis_img, strel('disk', 5))``, the optional one-pixel ``marker0`` (commented block),
    ``imgDist = imimposemin(...)`` (``imposed``), ``bw0`` (mask with the marker punched out), ``imgLabel``, ``bgm``,
    ``img(bgm) = 0`` (``seg``), ``bwareaopen`` (``seg_ao``), ``colorimg``."""

    bw: np.ndarray
    imgDist0: np.ndarray
    minima: np.ndarray
    marker: np.ndarray
    imposed: np.ndarray
    marker_overlay: np.ndarray
    L: np.ndarray
    ridge: np.ndarray
    seg: np.ndarray
    seg_ao: np.ndarray
    labels: np.ndarray
    n_minima: int
    n_minimum_px: int
    n_markers: int
    n_basins: int
    n_floes: int
    metric: str
    radius: int
    marker0: np.ndarray | None = None
    centroids: np.ndarray | None = None


def component_centroids(mask: np.ndarray, conn: int = 8) -> np.ndarray:
    """MATLAB ``regionprops(bwlabel(mask), 'Centroid')`` stacked with ``cat(1, cen.Centroid)``: one ``(x, y)`` row per
    component in ``bwlabel`` order, **1-based** means of the column (x) and row (y) indices.

    MATLAB source: the commented block of ``marker_watershed.m`` lines 18–25 (one-pixel markers at
    ``floor(Centroid)``).  Parity: exact by definition (mean of indices).
    """
    lab = label_components(mask, conn)
    n = int(lab.max())
    out = np.zeros((n, 2), dtype=np.float64)
    rows, cols = np.nonzero(lab)
    for k in range(1, n + 1):
        sel = lab[rows, cols] == k
        out[k - 1] = (cols[sel].mean() + 1.0, rows[sel].mean() + 1.0)
    return out


def marker_watershed(bw: np.ndarray, metric: str = "cityblock", radius: int = 5, point_markers: bool = False,
                     min_area: int = 5) -> MarkerWatershed:
    """Marker-controlled watershed of the inverse distance map — port of ``MATLAB_ROOT/ch5/marker_watershed.m``.

    Book: §5.1.3 pp. 95–97, Fig. 5.12: city-block inverse distance map → "four regional minima consisting of 18
    local minimum pixels" (a) → dilated by a 5-radius disk into "a marker image containing two connected regions"
    (b) → minima imposition (c) → watershed line (d) → the two floes (e).  MATLAB: ``imgDist = -bwdist(~img,
    'cityblock'); Dis_img = imregionalmin(imgDist); se = strel('disk', 5); marker = imdilate(Dis_img, se);
    imgDist = imimposemin(imgDist, marker); bw0 = img; bw0(find(marker .* bw == 1)) = 0; imgLabel =
    watershed(imgDist); bgm = imgLabel == 0; img(bgm) = 0; img = bwareaopen(img, 5); colorimg = bwlabel(img)``.
    ``point_markers=True`` runs the commented block (lines 18–25): one pixel per marker component at
    ``floor(regionprops Centroid)`` becomes the marker (``marker0``).  The disk is MATLAB's octagonal
    approximation (:func:`seaice.core.morphology.strel`).  Parity: exact (imposed map bit-identical in single).
    """
    bw = np.asarray(bw) != 0
    imgDist0 = inverse_distance(bw, metric)
    minima = imregionalmin(imgDist0)
    se = strel("disk", int(radius))
    marker = imdilate(minima, se)
    marker0 = centroids = None
    if point_markers:
        centroids = component_centroids(marker, 8)
        marker0 = np.zeros(bw.shape, dtype=bool)
        cen = np.floor(centroids).astype(np.int64)
        marker0[cen[:, 1] - 1, cen[:, 0] - 1] = True  # marker0(cen(2), cen(1)) = 1  (1-based → 0-based)
        marker_used = marker0
    else:
        marker_used = marker
    imposed = imimposemin(imgDist0, marker_used)
    dis = marker_used & bw
    marker_overlay = bw.copy()
    marker_overlay[dis] = False
    L = watershed(imposed)
    ridge = L == 0
    seg = bw.copy()
    seg[ridge] = False
    seg_ao = bwareaopen(seg, min_area)
    labels = label_components(seg_ao, 8)
    return MarkerWatershed(bw, imgDist0, minima, marker, imposed, marker_overlay, L, ridge, seg, seg_ao, labels,
                           int(label_components(minima, 8).max()), int(minima.sum()),
                           int(label_components(marker_used, 8).max()), int(L.max()), int(labels.max()), metric,
                           int(radius), marker0, centroids)


# ---------------------------------------------------------------------------------------------------------------
# topological_surface.m (§5.1 Fig. 5.1(b)(c), Fig. 5.5(b), Fig. 5.8(c))
# ---------------------------------------------------------------------------------------------------------------


def topographic_surfaces(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """The three surfaces of ``MATLAB_ROOT/ch5/topological_surface.m``.

    Book: §5.1 p. 84 (topographic reading of a gray image: Fig. 5.1(b) complement, (c) surface), §5.1.1 Fig. 5.5(b)
    (Sobel gradient surface), §5.1.2 Fig. 5.8(c) (inverse chessboard distance surface).  MATLAB: ``I = rgb2gray(I);
    x = imcomplement(I)`` (uint8 → ``255 - I``); ``g = sqrt(Ix.^2 + Iy.^2)`` (:func:`sobel_magnitude`); ``img =
    im2bw(I, graythresh(I))`` (**gray** threshold, 128/255 on ``q.jpg``); ``d = -bwdist(~img, 'chessboard')``.
    Returns ``{'gray', 'complement', 'gradient', 'bw', 'neg_chessboard_dt'}``.  Parity: exact.
    """
    rgb = np.asarray(rgb)
    gray = rgb2gray_matlab(rgb) if rgb.ndim == 3 else rgb
    x = imcomplement(gray)
    g = sobel_magnitude(gray)
    bw = otsu_mask(gray)
    d = inverse_distance(bw, "chessboard")
    return {"gray": gray, "complement": x, "gradient": g, "bw": bw, "neg_chessboard_dt": d}


# ---------------------------------------------------------------------------------------------------------------
# chaincode_corner.m / freeman_concave.m (§5.2.1.2, Eqs. 5.9–5.19, Fig. 5.17)
# ---------------------------------------------------------------------------------------------------------------


def differential_chain_code(fcc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Relative / absolute / summed / differential chain codes of a closed 8-code — Eqs. (5.9)–(5.18).

    Book: §5.2.1.2 pp. 103–104.  MATLAB source: ``chaincode_corner.m`` lines 29–73 = ``freeman_concave.m``
    lines 17–58, ported **literally** (same wrap-around constants):

    * Eq. (5.9) ``R(i) = mod(C(i) − C(i−1) + 8, 8)`` with ``C(0) = C(N)``; ``R > 4`` → ``R − 8`` (rotation in units
      of 45°, counter-clockwise positive; ``R = 4`` stays +4);
    * Eq. (5.10) ``A(0) = 0``, ``A(i) = A(i−1) + R(i)``  (Eq. 5.11: ``A(N) = −8`` for a clockwise trace);
    * Eqs. (5.12)–(5.14) ``S(i) = A(i) + A(i−1) + A(i−2)`` with ``S(0) = A(0) + A(N−1) + A(N−2) + 16``,
      ``S(1) = A(1) + A(0) + A(N−1) + 8``, then ``S := S − S(0)`` so that ``S(0) = 0`` (the script's
      ``Sum = Sum − Sum0; Sum0 = 0``);
    * Eqs. (5.16)–(5.18) ``D(i) = S(i+3) − S(i)``, ``D(N−j) = S(3−j) − 24 − S(N−j)`` for ``j = 1, 2, 3``, and
      ``D(0) = S(3) − S(0)``; the returned ``Diff`` is ``[D(0), D(1), …, D(N−1)]`` (``Diff(1) = D0; Diff(i) = D(i−1)``).

    Parameters
    ----------
    fcc : array of int, length ``N >= 3``
        Freeman code of a closed boundary traced **clockwise** (:func:`seaice.core.chaincode.fchcode` of
        ``boundaries(B, 8, 'cw')``); the ``'cww'`` typo in the scripts is clockwise.

    Returns
    -------
    (R, A, S, D, Diff) : int64 arrays of length ``N``
        ``D`` is the script's 1-based ``D(1..N)`` (``D(i) = S(i+3) − S(i)``); ``Diff`` is aligned with the boundary
        points so that ``Diff[k]`` is the turning measure at point ``k`` (Eq. 5.19: angle ``= Diff × 15°``).

    Parity: exact (integer arithmetic; pre-check identical on the 262-code boundary of ``q.jpg``).
    """
    c = np.asarray(fcc, dtype=np.int64).ravel()
    l = c.size
    if l < 3:
        raise ValueError("differential_chain_code: the boundary needs at least 3 chain codes (A(l-2) is indexed)")
    R = np.zeros(l, dtype=np.int64)
    R[0] = (c[0] - c[l - 1] + 8) % 8  # R(1) = mod(c(1) - c(l) + 8, 8)
    for i in range(1, l):
        R[i] = (c[i] - c[i - 1] + 8) % 8  # Eq. (5.9a)
    R[R > 4] -= 8  # Eq. (5.9b)
    A0 = 0
    A = np.zeros(l, dtype=np.int64)
    A[0] = R[0]
    for i in range(1, l):
        A[i] = A[i - 1] + R[i]  # Eq. (5.10)
    Sum0 = A0 + A[l - 2] + A[l - 3] + 16  # Eq. (5.13): A(l-1), A(l-2) are 1-based
    S = np.zeros(l, dtype=np.int64)
    S[0] = A[0] + A0 + A[l - 2] + 8  # Eq. (5.14)
    S[1] = A[1] + A[0] + A0
    for i in range(2, l):
        S[i] = A[i] + A[i - 1] + A[i - 2]  # Eq. (5.12)
    S = S - Sum0
    Sum0 = 0
    D0 = S[2] - Sum0
    D = np.zeros(l, dtype=np.int64)
    for i in range(0, l - 3):
        D[i] = S[i + 3] - S[i]  # Eq. (5.16)
    D[l - 3] = S[0] - 24 - S[l - 3]  # Eq. (5.17)/(5.18): D(l-2) = Sum(1) - 24 - Sum(l-2)
    D[l - 2] = S[1] - 24 - S[l - 2]
    D[l - 1] = S[2] - 24 - S[l - 1]
    Diff = np.zeros(l, dtype=np.int64)
    Diff[0] = D0
    Diff[1:] = D[:-1]
    return R, A, S, D, Diff


@dataclass
class ConcaveResult:
    """Output of :func:`freeman_concave` (the script's variables): ``B`` (mask), ``b`` (closed boundary, 0-based),
    ``bim``, ``c`` (chain code), ``R``/``A``/``Sum``/``D``/``Diff``, ``p`` (0-based indices into ``b``) and
    ``concave = b(p, :)`` (0-based (row, col))."""

    bw: np.ndarray
    boundary: np.ndarray
    bim: np.ndarray
    code: ChainCode
    R: np.ndarray
    A: np.ndarray
    S: np.ndarray
    D: np.ndarray
    Diff: np.ndarray
    index: np.ndarray
    points: np.ndarray
    n_boundaries: int
    longest: int

    @property
    def points_matlab(self) -> np.ndarray:
        """1-based ``(row, col)`` concave points as MATLAB prints them."""
        return self.points + 1


def freeman_concave(I: np.ndarray, object: str = "first", lo: int | None = None, hi: int | None = None) -> ConcaveResult:
    """Concave boundary points of an object by the differential chain code — port of
    ``MATLAB_ROOT/ch5/watershed_based/freeman_concave.m`` (= ``chaincode_corner.m`` lines 6–74).

    Book: §5.2.1 (boundary tracing, Fig. 5.16; differential chain code Eqs. 5.9–5.19), concave iff
    ``3 <= D(i) <= 10`` (p. 104), Fig. 5.17.  MATLAB (literal): ``B = im2bw(I, graythresh(I))``; ``b =
    boundaries(B, 8, 'cww')`` (typo = clockwise); ``d = cellfun('length', b); [max_d, k] = max(d)`` (computed,
    **unused**); ``b = b{1}`` (the **first** ``bwlabel`` object, not the longest); ``bim = bound2im(b, M, N,
    min(b(:,1)), min(b(:,2)))``; ``c = fchcode(b)``; the code loops (:func:`differential_chain_code`);
    ``p = find(Diff >= 3 & Diff <= 10); concave = b(p, :)``.

    Parameters
    ----------
    I : ndarray
        What the scripts pass: the RGB ``q.jpg`` (``chaincode_corner.m``; ``graythresh`` histograms all planes,
        ``im2bw`` converts to gray) or the **double** ``{0, 1}``-valued region ``im = imreconstruct(g, connect)``
        of ``main.m`` (the marker ``g`` is ``{0, 1}`` and reconstruction by dilation never exceeds the marker's
        maximum, so ``im`` is 1 on the reconstructed component regardless of its ``bwlabel`` number ``k``;
        ``im2uint8`` maps ``{0, 1}`` to ``{0, 255}`` → the mask is ``I > 0``).  A bool mask is promoted to double
        so that this path applies.
    object : {'first', 'longest'}
        ``'first'`` = the script (``b{1}``); ``'longest'`` uses the ``[max_d, k]`` the script computes and ignores.
    lo, hi : int, optional
        Concavity range (defaults ``BOOK_PARAMS['concave_min'/'concave_max']`` = 3..10).

    Raises ``ValueError`` when the image has no object or the boundary has fewer than 3 codes (MATLAB errors on
    ``A(l-2)``).  Parity: exact (``R``, ``A``, ``Sum``, ``D``, ``Diff``, ``concave`` identical on ``q.jpg``).
    """
    I = np.asarray(I)
    if I.dtype == np.bool_:
        I = I.astype(np.float64)
    B = otsu_mask(I)
    M, N = B.shape
    b_all = boundaries(B, 8, "cw")
    if not b_all:
        raise ValueError("freeman_concave: no object in the image")
    d = [len(x) for x in b_all]
    k = int(np.argmax(d))  # [max_d, k] = max(d)  (unused by the script)
    if object not in ("first", "longest"):
        raise ValueError("object must be 'first' or 'longest'")
    b = b_all[0] if object == "first" else b_all[k]
    bim = bound2im(b, M, N, int(b[:, 0].min()), int(b[:, 1].min()))
    c = fchcode(b)
    R, A, S, D, Diff = differential_chain_code(c.fcc)
    lo = BOOK_PARAMS["concave_min"] if lo is None else int(lo)
    hi = BOOK_PARAMS["concave_max"] if hi is None else int(hi)
    p = np.flatnonzero((Diff >= lo) & (Diff <= hi))
    concave = b[p, :]
    return ConcaveResult(B, b, bim, c, R, A, S, D, Diff, p, concave, len(b_all), k)


# ---------------------------------------------------------------------------------------------------------------
# watershed_based/main.m (§5.2 Steps 1–5, Figs. 5.13–5.15)
# ---------------------------------------------------------------------------------------------------------------


def junction_endpoints(mask: np.ndarray, rule: str = "max") -> np.ndarray:
    """Ending points of a 4-connected, 1-px-thick junction line — §5.2 Step 4 (p. 99), Fig. 5.15.

    ``g1 = imfilter(double(g), wr)`` with the kernel :data:`ENDPOINT_KERNEL` ``[0 -1 0; -1 4 -1; 0 -1 0]``
    (zero-padded correlation, as ``main.m`` lines 33–37): a line pixel responds ``4 − (# line 4-neighbours)``, a
    background pixel ``−(# line 4-neighbours)``.  Then

    * ``rule='max'`` (the script, literal): ``g2 = abs(g1); g2 >= max(g2(:))`` — on a proper line the maximum is 3
      (the 12 patterns of Fig. 5.15(a)); on a closed loop it is 2 and every pixel qualifies; an isolated pixel
      gives 4;
    * ``rule='ge3'`` (the text, p. 101, literal): ``g1 >= 3`` on the **signed** response — exactly the ending
      points (one line 4-neighbour) plus isolated pixels (4).

    The two rules differ in two ways: ``'max'`` is relative (a loop's 2-response pixels qualify, a proper line's
    do not), and its ``abs`` also admits *background* pixels with ≥ 3 line 4-neighbours (response −3), which the
    signed ``'ge3'`` test never does.  Returns 0-based ``(row, col)`` pairs in ``find`` (column-major) order,
    shape ``(n, 2)``.  Parity: exact (``'max'`` vs ``main.m``); ``'ge3'`` is the text's rule, not the script's.
    """
    g = (np.asarray(mask) != 0).astype(np.float64)
    g1 = imfilter(g, ENDPOINT_KERNEL)
    if rule == "max":
        g2 = np.abs(g1)
        T = g2.max()
        hit = g2 >= T
    elif rule == "ge3":
        hit = g1 >= 3
    else:
        raise ValueError("rule must be 'max' or 'ge3'")
    return _find_rows_cols(hit)


@dataclass
class JunctionLine:
    """One junction line of :func:`neighboring_region_merging` (``main.m`` loop variables for label ``i``)."""

    label: int
    pixels: np.ndarray  # (n, 2) 0-based (row, col) in find order (p = find(label == i))
    endpoints: np.ndarray  # ep = [x, y] (0-based)
    region: np.ndarray  # im = imreconstruct(g, connect): the line ∪ its neighbouring regions, {0, 1}-valued double
    concave: np.ndarray  # concave points of that region (0-based)
    concave_endpoints: np.ndarray  # c = intersect(ep, concave, 'rows') (sorted rows, 0-based)
    removed: bool  # numel(c) == 0 → seg(p) = 1

    @property
    def n_pixels(self) -> int:
        return int(self.pixels.shape[0])


@dataclass
class MergeResult:
    """Arrays of ``main.m``: ``bw``, ``D``, ``L``, ``w`` (ridges), ``f`` (junction lines), ``seg0`` (over-segmented,
    Fig. 5.14(e)), ``seg`` (final, Fig. 5.14(i)), ``label``/``num`` (``bwlabel(f, 4)``) and the per-line records."""

    bw: np.ndarray
    D: np.ndarray
    L: np.ndarray
    w: np.ndarray
    f: np.ndarray
    seg0: np.ndarray
    seg: np.ndarray
    label: np.ndarray
    num: int
    lines: list[JunctionLine] = field(default_factory=list)
    metric: str = "cityblock"
    endpoint_rule: str = "max"
    sequential: bool = True

    @property
    def n_floes_before(self) -> int:
        return int(label_components(self.seg0, 8).max())

    @property
    def n_floes_after(self) -> int:
        return int(label_components(self.seg, 8).max())

    @property
    def n_removed(self) -> int:
        return sum(1 for ln in self.lines if ln.removed)


def neighboring_region_merging(bw: np.ndarray, metric: str = "cityblock", endpoint_rule: str = "max",
                               sequential: bool = True) -> MergeResult:
    """Watershed + neighbouring-region merging — port of ``MATLAB_ROOT/ch5/watershed_based/main.m``.

    Book: §5.2 pp. 96–99, Fig. 5.13 (flow chart), Fig. 5.14(b)–(i): **Step 1** binarise (done by the caller — the
    script uses ``im2bw(img, graythresh(img))`` on the RGB); **Step 2** ``D = bwdist(~bw, 'cityblock'); L =
    watershed(-D); w = L == 0`` (8-connected watershed, 4-connected lines); **Step 3** junction lines
    ``f = bitand(bw, w)`` and the over-segmented ``seg = bw & ~w``; ``[label, num] = bwlabel(f, 4)``; **Step 4**
    per line ``i`` (in ``bwlabel`` order): ``p = find(label == i); g(p) = 1``, ending points
    :func:`junction_endpoints`; **Step 5** ``neighbor = seg; neighbor(p) = 1; connect = bwlabel(neighbor); im =
    imreconstruct(g, connect)`` — the union of the line and its neighbouring regions as a ``{0, 1}``-valued double
    (the marker ``g`` is ``{0, 1}``, so the reconstruction is 1 on the whole ``bwlabel`` component that contains
    the line and 0 elsewhere; the component's label number ``k`` never appears in ``im``);
    ``concave = freeman_concave(im); c = intersect(ep, concave, 'rows'); if numel(c) == 0, seg(p) = 1`` — the line
    is deleted (regions merged) when **neither** ending point is concave.

    ``sequential=True`` (the script) updates ``seg`` inside the loop, so later lines see regions already merged by
    earlier ones (their reconstructed region can hold three floes); ``sequential=False`` evaluates every line
    against the original over-segmentation (the text's description, not the code).  ``endpoint_rule`` see
    :func:`junction_endpoints`.  Parity: exact (``f``, ``seg0``, ``seg``, per-line ``ep``/``concave`` vs MATLAB).
    """
    bw = np.asarray(bw) != 0
    m, n = bw.shape
    gc = ~bw
    D = bwdist(gc, metric).astype(np.float32)  # bwdist returns single
    L = watershed(-D)
    w = L == 0
    f = bw & w  # bitand(logical, logical) → logical
    seg = bw & ~w
    seg0 = seg.copy()
    label = label_components(f, 4)
    num = int(label.max())
    lines: list[JunctionLine] = []
    for i in range(1, num + 1):
        p = label == i
        g = np.zeros((m, n), dtype=np.float64)
        g[p] = 1.0
        ep = junction_endpoints(p, endpoint_rule)
        neighbor = (seg if sequential else seg0).copy()
        neighbor[p] = True
        connect = label_components(neighbor, 8).astype(np.float64)
        im = imreconstruct(g, connect)
        conc = freeman_concave(im).points
        c = _intersect_rows(ep, conc)
        removed = c.shape[0] == 0
        if removed:
            seg[p] = True
        lines.append(JunctionLine(i, _find_rows_cols(p), ep, im, conc, c, removed))
    return MergeResult(bw, D, L, w, f, seg0, seg, label, num, lines, metric, endpoint_rule, sequential)


def _intersect_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATLAB ``intersect(A, B, 'rows')``: the common rows, sorted, unique."""
    a = np.asarray(a).reshape(-1, 2)
    b = np.asarray(b).reshape(-1, 2)
    sa = {tuple(int(v) for v in r) for r in a}
    sb = {tuple(int(v) for v in r) for r in b}
    common = sorted(sa & sb)
    return np.array(common, dtype=np.int64).reshape(-1, 2)


# ---------------------------------------------------------------------------------------------------------------
# Text-only algorithms of §5.1 / §5.1.3 (teaching forms)
# ---------------------------------------------------------------------------------------------------------------


def regional_minima_by_reconstruction(I: np.ndarray, conn: int = 8) -> np.ndarray:
    """Regional minima by grayscale reconstruction **by erosion** — Book Eq. (5.2) ``M = R^E_I(I + 1) − I``
    (§5.1 p. 86; Eqs. 4.35–4.38 for ``R^E``): ``M`` is nonzero exactly on the regional minima.

    Returns the bool mask ``M > 0``; must equal :func:`seaice.core.morphology.imregionalmin` (asserted in the tests),
    including the degenerate cases: a constant image (also all ``+Inf`` / all ``-Inf``) is one regional minimum
    (all-True, like MATLAB ``imregionalmin``), ``-Inf`` pixels are always minima, ``+Inf`` pixels next to finite
    ones never.  Parity: reimplemented (no MATLAB call; identity checked against ``imregionalmin``).
    """
    I = np.asarray(I, dtype=np.float64)
    if np.isnan(I).any():
        raise ValueError("NaN values are not allowed")
    if I.size and np.all(I == I.flat[0]):
        # A constant image (finite, all +Inf or all -Inf) is a single plateau with no external boundary, hence one
        # regional minimum: MATLAB imregionalmin returns all-true.  Eq. 5.2 with finite values already gives this
        # (rec = I + 1 > I); with +Inf everywhere `Inf + 1 > Inf` is false, so the plateau case is made explicit.
        return np.ones(I.shape, dtype=bool)
    rec = reconstruct_by_erosion(I + 1.0, I, conn)  # R^E_I(I + 1)
    # M = rec - I > 0  ⇔  rec > I for finite values; a -Inf pixel (where the difference is NaN) is always a
    # regional minimum (nothing is lower), a +Inf pixel next to anything finite never is (rec > I false).
    with np.errstate(invalid="ignore"):
        M = rec > I
    return M | (I == -np.inf)


def impose_minima_book(g: np.ndarray, markers: np.ndarray, conn: int = 8,
                       marker_value: float | None = None) -> np.ndarray:
    """Minima imposition as the book writes it — §5.1.3 p. 96, Steps 1–2:
    **Step 1** ``(g + 1) ∧ f`` (pointwise minimum of ``g + 1`` and the marker image ``f``), **Step 2**
    ``g' = R^E_{(g + 1) ∧ f}(f)`` (reconstruction by erosion of ``f`` under that mask, Eqs. 4.35–4.38).

    ``f`` is built from the bool ``markers``: ``marker_value`` on the markers (the book's 0; for segmentation
    functions with values below 0 — every inverse distance map — the default drops it to ``floor(min g) − 1`` so
    the markers remain minima, which is what MATLAB's ``−Inf`` achieves) and ``+Inf`` elsewhere.  The "+1" is the
    literal book offset (MATLAB's ``imimposemin`` uses ``h = 0.001·range`` for floats, ``1`` for integers); the
    resulting watershed is the same because the flooding order only depends on the non-marker values, which are
    shifted uniformly.  Parity: reimplemented (checked against :func:`seaice.core.morphology.imimposemin` through
    ``watershed`` equality).
    """
    g = np.asarray(g, dtype=np.float64)
    markers = np.asarray(markers) != 0
    if marker_value is None:
        marker_value = 0.0 if g.min() >= 0 else float(np.floor(g.min()) - 1.0)
    f = np.where(markers, float(marker_value), np.inf)
    mask = np.minimum(g + 1.0, f)  # Step 1
    return reconstruct_by_erosion(f, mask, conn)  # Step 2


def _conn_shifts(conn: int) -> list[tuple[int, int]]:
    s4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if conn == 4:
        return s4
    if conn == 8:
        return s4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    raise ValueError("conn must be 4 or 8")


def threshold_set(I: np.ndarray, h: float) -> np.ndarray:
    """Eq. (5.1) ``T_h(I) = {(u, v) | I(u, v) < h}`` — the cross-section of the topographic surface below level ``h``."""
    return np.asarray(I, dtype=np.float64) < h


def watershed_immersion(I: np.ndarray, conn: int = 8) -> np.ndarray:
    """Vincent–Soille-style immersion watershed written from the book's recursion — §5.1 pp. 86–88, Eqs. (5.1),
    (5.3)–(5.8), Fig. 5.4 (teaching form; :func:`seaice.core.watershed.watershed` is the MATLAB-exact one).

    Levels ``h`` run over the distinct values of ``I`` (only histogram levels are visited).  At level ``h`` the
    threshold set ``T_{h+1}(I) = {I <= h}`` (Eq. 5.1) is split into connected components ``S``; for each ``S`` the
    three cases of p. 87 apply with ``C[h−1]`` = the basins so far (Eqs. 5.3–5.8):

    1. ``S ∩ C[h−1] = ∅`` → ``S`` is a new regional minimum: a new basin;
    2. ``S`` meets exactly one basin → the basin grows to ``S`` (``C_h(M_i) = C(M_i) ∩ T_h(I)``, Eq. 5.3);
    3. ``S`` meets several basins → the basins are grown inside ``S`` by constrained 3×3 (``conn``) dilation and a
       one-pixel **dam** is built where two different basins would meet; pixels of ``S`` walled off from every basin
       become new minima.

    The dam construction is done pixel-sequentially (FIFO from the basin fronts, a pixel adjacent to two basins
    becomes a dam) so that dams always separate basins, as the book's Fig. 5.4 illustrates.  Returns int32 labels
    with ``0`` on the dams.  Where plateaus are shared by several minima the dam placement can differ from Meyer's
    flooding (MATLAB); on 1-D profiles and images without ambiguous plateaus both agree.
    Parity: reimplemented (synthetic truths; compared with ``watershed`` in the tests).
    """
    I = np.asarray(I, dtype=np.float64)
    if I.ndim != 2:
        raise ValueError("2-D images only")
    shifts = _conn_shifts(conn)
    M, N = I.shape
    C = np.zeros((M, N), dtype=np.int32)  # C[h-1]
    dam = np.zeros((M, N), dtype=bool)
    n_basins = 0
    for h in np.unique(I):
        T = (I <= h) & ~dam  # T_{h+1}(I) = {I < h + 1} = {I <= h} on the histogram levels (Eq. 5.1)
        comps = label_components(T, conn)
        newC = np.zeros_like(C)
        for k in range(1, int(comps.max()) + 1):
            S = comps == k
            labs = np.unique(C[S])
            labs = labs[labs > 0]
            if labs.size == 0:  # case 1: new minimum
                n_basins += 1
                newC[S] = n_basins
            elif labs.size == 1:  # case 2: the basin grows into S (Eq. 5.3)
                newC[S] = labs[0]
            else:  # case 3: several basins meet in S → constrained growth + dams
                cur = np.where(S, C, 0).astype(np.int32)
                q: deque[tuple[int, int]] = deque()
                rows, cols = np.nonzero(cur)
                seen = cur > 0
                for r, c in zip(rows.tolist(), cols.tolist()):
                    for dr, dc in shifts:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < M and 0 <= cc < N and S[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            q.append((rr, cc))
                while q:
                    r, c = q.popleft()
                    labels_here = set()
                    for dr, dc in shifts:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < M and 0 <= cc < N and cur[rr, cc] > 0:
                            labels_here.add(int(cur[rr, cc]))
                    if len(labels_here) >= 2:
                        dam[r, c] = True  # dam: adjacent to two different basins
                        continue
                    if len(labels_here) == 1:
                        cur[r, c] = labels_here.pop()
                    for dr, dc in shifts:
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < M and 0 <= cc < N and S[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            q.append((rr, cc))
                left = S & (cur == 0) & ~dam  # walled-off parts of S: new minima (Vincent–Soille)
                if left.any():
                    sub = label_components(left, conn)
                    for j in range(1, int(sub.max()) + 1):
                        n_basins += 1
                        cur[sub == j] = n_basins
                newC[S & ~dam] = cur[S & ~dam]
        C = newC
    C[dam] = 0
    return C
