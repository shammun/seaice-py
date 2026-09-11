"""Synthetic inputs and book fixtures (Tier-3 data) for chapter 2 (extended by later chapters).

Book fixtures are transcribed from the printed matrices; the ``distance_transform.m`` and ``chain_diff.m`` inputs
are built exactly as the ``.m`` files define them.
"""
from __future__ import annotations

import numpy as np


def point_image(size: int = 201, center: tuple[int, int] | None = None) -> np.ndarray:
    """Binary image with a single nonzero pixel — Book Fig. 2.14(a); ``distance_transform.m`` lines 3–4
    (``img = zeros(201, 201); img(101, 101) = 1`` → 0-based ``(100, 100)``)."""
    img = np.zeros((size, size), dtype=bool)
    r, c = center if center is not None else (size // 2, size // 2)
    img[r, c] = True
    return img


def _m(rows: str) -> np.ndarray:
    return np.array([[int(ch) for ch in row.split()] for row in rows.strip().splitlines()], dtype=np.int64)


#: Fig. 2.19(a) / ``chain_diff.m``: 8×9 binary object (exactly the hard-coded matrix ``B``).
FIG_2_19_OBJECT = _m(
    """
    0 0 0 0 0 0 0 0 0
    0 0 0 1 1 1 0 0 0
    0 0 0 1 1 1 1 1 0
    0 1 1 1 1 1 1 1 0
    0 1 1 1 1 1 1 1 0
    0 1 1 1 1 1 0 0 0
    0 0 0 1 1 1 0 0 0
    0 0 0 0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 2.11: 10×13 binary image with 5 four-connected / 2 eight-connected components.
FIG_2_11_COMPONENTS = _m(
    """
    0 0 0 0 0 0 0 0 0 0 0 0 0
    0 1 1 1 1 1 1 0 0 1 1 0 0
    0 1 1 1 1 1 1 0 0 1 1 0 0
    0 1 1 1 1 1 1 0 0 1 1 0 0
    0 1 1 1 1 1 1 0 1 0 0 0 0
    0 1 1 1 1 1 1 0 1 1 0 0 0
    0 0 1 0 0 0 0 0 1 1 0 1 1
    0 0 0 1 1 1 0 0 1 1 1 0 0
    0 0 0 1 1 1 0 0 0 1 1 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 2.12(a): 7×7 binary matrix.
FIG_2_12_A = _m(
    """
    0 0 0 0 1 1 0
    0 1 1 1 1 0 0
    1 1 1 1 1 1 1
    1 1 1 1 1 1 0
    0 0 1 1 1 1 0
    0 0 1 1 1 0 0
    0 0 0 0 1 0 0
    """
).astype(bool)

_S2, _S5 = np.sqrt(2.0), np.sqrt(5.0)
#: Fig. 2.12(b): Euclidean distance transform of FIG_2_12_A (values 0, 1, 1.4142, 2, 2.2361).
FIG_2_12_B = np.array(
    [
        [0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [1, _S2, 2, 2, _S2, 1, 1],
        [1, 1, _S2, _S5, 2, 1, 0],
        [0, 0, 1, 2, _S2, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
    ],
    dtype=np.float64,
)

#: Fig. 2.10: 3×3 example for 4-, 8- and m-paths ((a) has no bottom-right pixel).
FIG_2_10_A = _m("0 1 1\n0 1 0\n0 0 0").astype(bool)
FIG_2_10_BC = _m("0 1 1\n0 1 0\n0 0 1").astype(bool)

#: Fig. 2.1: 11×11 gray values of an ice region (as printed, row-major).
FIG_2_1_GRAY = np.array(
    [
        [203, 202, 200, 201, 202, 204, 204, 205, 205, 205, 202],
        [202, 206, 198, 198, 198, 199, 201, 203, 205, 207, 203],
        [197, 197, 199, 198, 198, 198, 199, 200, 201, 202, 205],
        [196, 195, 199, 199, 198, 197, 197, 197, 197, 198, 203],
        [200, 202, 199, 199, 198, 198, 197, 197, 197, 197, 197],
        [201, 202, 198, 199, 199, 199, 199, 199, 199, 199, 199],
        [197, 194, 198, 199, 200, 200, 201, 201, 201, 200, 205],
        [197, 194, 198, 199, 200, 201, 201, 201, 200, 199, 204],
        [202, 203, 199, 199, 200, 201, 201, 199, 198, 197, 198],
        [202, 200, 198, 198, 199, 199, 200, 200, 200, 200, 201],
        [199, 199, 202, 202, 201, 201, 200, 200, 199, 199, 200],
    ],
    dtype=np.uint8,
)

#: Book Fig. 2.21 sequences (p. 31) for the Fig. 2.19 object.
FIG_2_21 = {
    "original": np.array([0, 1, 2, 0, 0, 7, 0, 6, 6, 4, 5, 6, 4, 4, 3, 4, 2, 2]),
    "first_difference": np.array([1, 1, 6, 0, 7, 1, 6, 0, 6, 1, 1, 6, 0, 7, 1, 6, 0, 6]),
    "normalized": np.array([0, 0, 7, 0, 6, 6, 4, 5, 6, 4, 4, 3, 4, 2, 2, 0, 1, 2]),
    "normalized_first_difference_of_normalized": np.array([0, 7, 1, 6, 0, 6, 1, 1, 6, 0, 7, 1, 6, 0, 6, 1, 1, 6]),
    "normalized_first_difference": np.array([0, 6, 1, 1, 6, 0, 7, 1, 6, 0, 6, 1, 1, 6, 0, 7, 1, 6]),
}
#: Fig. 2.20 starting point, 1-based (row, col) as printed.
FIG_2_20_START_MATLAB = (4, 2)


def binary_pattern_16(seed: int = 0) -> np.ndarray:
    """A 16×16 binary pattern in the spirit of Fig. 2.2 (synthetic; the printed values are not load-bearing)."""
    bw = np.zeros((16, 16), dtype=bool)
    bw[:2, :] = True
    rr, cc = np.mgrid[0:16, 0:16]
    bw |= np.abs(rr - 8.5) + np.abs(cc - 7.5) <= 4.5  # diamond
    bw[13:, :] = True
    return bw


def spur_shape() -> np.ndarray:
    """7×9 rectangle with a one-pixel-wide spur (tests the double traversal of ``boundaries.m``)."""
    s = np.zeros((7, 9), dtype=bool)
    s[2:5, 1:6] = True
    s[3, 6:8] = True
    return s


def set_operation_fixtures() -> dict[str, np.ndarray]:
    """Two overlapping 16×16 rectangles ``A``, ``B`` for Fig. 2.16 and an L-shape for Fig. 2.17 (synthetic)."""
    A = np.zeros((16, 16), dtype=bool)
    B = np.zeros((16, 16), dtype=bool)
    A[2:10, 2:10] = True
    B[6:14, 6:14] = True
    L = np.zeros((9, 9), dtype=bool)
    L[2:7, 3] = True  # vertical bar
    L[6, 3:7] = True  # foot
    return {"A": A, "B": B, "L": L, "L_origin": (6, 3)}


def book_fixtures() -> dict[str, np.ndarray]:
    """All chapter-2 fixtures keyed by figure."""
    return {
        "fig_2_1_gray": FIG_2_1_GRAY,
        "fig_2_10_a": FIG_2_10_A,
        "fig_2_10_bc": FIG_2_10_BC,
        "fig_2_11": FIG_2_11_COMPONENTS,
        "fig_2_12_a": FIG_2_12_A,
        "fig_2_12_b": FIG_2_12_B,
        "fig_2_19_a": FIG_2_19_OBJECT,
        "point_201": point_image(201),
    }


# ---------------------------------------------------------------------------------------------------------------
# Chapter 3 additions (§3.1.2 Fig. 3.4(a), §3.2.2 Figs. 3.6 / 3.8, unit tests)
# ---------------------------------------------------------------------------------------------------------------


def uneven_illumination(img: np.ndarray, gain: float = 0.5, axis: int = 1, bias: float = 40.0,
                        offset: float = 0.0, kind: str = "ramp") -> np.ndarray:
    """Add a "factitious uneven illumination" to an image — Book §3.1.2, Fig. 3.4(a) (text only; the ``t.jpg``
    read by ``local_Otsu.m`` is not shipped, so this reproduces the *kind* of input, not the printed image).

    ``kind='ramp'``: multiplicative linear ramp along ``axis`` from ``1 + gain`` (first row/column) to
    ``1 − gain`` (last), plus an additive ramp from ``+bias`` to ``−bias`` gray levels and a constant ``offset``
    (the defaults ``gain=0.5, bias=40`` make global Otsu visibly under-detect ice on the dark side of ``2.jpg``
    while 2×3 block Otsu stays within 1 % of the clean IC, the situation of Fig. 3.4).  ``kind='spot'``: radial vignetting
    ``1 + gain (1 − 2 ρ)`` with ``ρ`` the normalised distance from the centre.  Works on gray or RGB uint8
    input (same factor per channel); the result is rounded and saturated like MATLAB ``uint8()``
    (:func:`~seaice.core.matlab_compat.to_uint8_saturating`).  Deterministic (no noise).
    """
    from .matlab_compat import to_uint8_saturating  # local import: synth stays dependency-free otherwise

    img = np.asarray(img)
    M, N = img.shape[:2]
    if kind == "ramp":
        n = img.shape[axis]
        x = np.linspace(0.0, 1.0, n)
        ramp = 1.0 - 2.0 * x
        factor = 1.0 + gain * ramp
        add = bias * ramp
        factor = factor[:, None] if axis == 0 else factor[None, :]
        add = add[:, None] if axis == 0 else add[None, :]
    elif kind == "spot":
        rr, cc = np.mgrid[0:M, 0:N]
        rho = np.hypot((rr - (M - 1) / 2) / (M / 2), (cc - (N - 1) / 2) / (N / 2))
        ramp = 1.0 - 2.0 * rho / rho.max()
        factor = 1.0 + gain * ramp
        add = bias * ramp
    else:
        raise ValueError(f"uneven_illumination: unknown kind {kind!r}")
    factor = np.broadcast_to(factor, (M, N)).astype(np.float64)
    add = np.broadcast_to(add, (M, N)).astype(np.float64)
    f = img.astype(np.float64)
    if f.ndim == 3:
        factor, add = factor[:, :, None], add[:, :, None]
    return to_uint8_saturating(f * factor + add + offset)


def two_clusters_2d(seed: int = 0, n_per: int = 15, centers: tuple[tuple[float, float], ...] = ((2.5, 2.5), (6.5, 6.5)),
                    spread: float = 0.6, outlier: tuple[float, float] | None = None) -> np.ndarray:
    """Seeded 2-D points in two compact, well-separated Gaussian clusters inside ``[0, 10]²`` — the data of Book
    Figs. 3.6 and 3.8 (text only).  ``outlier=(x, y)`` appends one far point (Fig. 3.8(b)).  Returns ``(n, 2)``."""
    rng = np.random.default_rng(seed)
    pts = [rng.normal(loc=c, scale=spread, size=(n_per, 2)) for c in centers]
    X = np.vstack(pts)
    if outlier is not None:
        X = np.vstack([X, np.asarray(outlier, dtype=np.float64)[None, :]])
    return np.clip(X, 0.0, 10.0)


def bimodal_image(seed: int = 0, shape: tuple[int, int] = (120, 160), dark: float = 60.0, bright: float = 190.0,
                  sigma: float = 12.0, ice_fraction: float = 0.4) -> np.ndarray:
    """Synthetic uint8 image with a bimodal histogram (Fig. 3.1 sketch / unit tests): a rectangular "floe" of
    mean ``bright`` covering ``ice_fraction`` of the area on a ``dark`` background, Gaussian noise ``sigma``."""
    from .matlab_compat import to_uint8_saturating

    rng = np.random.default_rng(seed)
    M, N = shape
    img = np.full((M, N), dark, dtype=np.float64)
    h = int(round(np.sqrt(ice_fraction) * M))
    w = int(round(np.sqrt(ice_fraction) * N))
    r0, c0 = (M - h) // 2, (N - w) // 2
    img[r0:r0 + h, c0:c0 + w] = bright
    img += rng.normal(0.0, sigma, size=(M, N))
    return to_uint8_saturating(img)


# ---------------------------------------------------------------------------------------------------------------
# Chapter 4 additions (§4.2.1 Fig. 4.8 erosion/dilation walk-through, §4.2.2–4.2.3 1-D profiles, unit tests)
# ---------------------------------------------------------------------------------------------------------------

#: Fig. 4.8(a) (p. 70, from DIPUM [49]): 13×17 binary image with a 3×7 rectangular object (rows 6–8, cols 6–12, 1-based).
FIG_4_8_IMAGE = _m(
    """
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 4.8(b): cross-shaped structuring element with the origin at the centre (``B = B̂``).
FIG_4_8_SE = _m("0 1 0\n1 1 1\n0 1 0").astype(bool)

#: Fig. 4.8(d): erosion result — the object shrinks to one row of 5 pixels (row 7, cols 7–11, 1-based).
FIG_4_8_ERODED = _m(
    """
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 1 1 1 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 4.8(f): dilation result — 5 rows / 9 columns with the four corners clipped.
FIG_4_8_DILATED = _m(
    """
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0
    0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0
    0 0 0 0 1 1 1 1 1 1 1 1 1 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    """
).astype(bool)


def two_floes_profile(n: int = 240) -> np.ndarray:
    """1-D intensity profile in the spirit of Book Figs. 4.11–4.14 (the printed curves are sketches, not data).

    Two bright "floes" (plateaus of 170 and 200) separated by a narrow dark crack (width 3, value 60) on a dark
    water background (40), with a 2-pixel bright speck on the water and a 3-pixel dark pit inside the second floe,
    plus gentle slopes at the floe edges.  Returns a ``(1, n)`` uint8 array so the 2-D morphology functions apply
    directly (a ``1×N`` image with a ``1×k`` line SE is exactly the book's 1-D case).  Deterministic.
    """
    from .matlab_compat import to_uint8_saturating

    x = np.arange(n)
    f = np.full(n, 40.0)
    f[(x >= 30) & (x < 100)] = 170.0
    f[(x >= 103) & (x < 190)] = 200.0
    f[(x >= 100) & (x < 103)] = 60.0  # crack between the floes
    f[(x >= 10) & (x < 12)] = 150.0  # bright speck on the water
    f[(x >= 140) & (x < 143)] = 90.0  # dark pit in the second floe
    # sloped edges
    f[25:30] = np.linspace(40, 170, 7)[1:-1]
    f[190:196] = np.linspace(200, 40, 8)[1:-1]
    return to_uint8_saturating(f)[None, :]


def two_blobs_with_marker(shape: tuple[int, int] = (40, 60)) -> tuple[np.ndarray, np.ndarray]:
    """Binary mask with two blobs and a marker inside the first one (Fig. 4.13-style reconstruction test).

    Returns ``(marker, mask)`` bool arrays: reconstruction of ``marker`` under ``mask`` must return exactly the
    first blob.
    """
    M, N = shape
    mask = np.zeros((M, N), dtype=bool)
    rr, cc = np.mgrid[0:M, 0:N]
    mask |= (rr - 18) ** 2 + (cc - 16) ** 2 <= 10 ** 2
    mask |= np.abs(rr - 22) + np.abs(cc - 44) <= 9
    marker = np.zeros_like(mask)
    marker[18, 16] = True
    return marker, mask


# ---------------------------------------------------------------------------------------------------------------
# Chapter 5 fixtures (Figs. 5.15, 5.16; watershed tie fixtures; synthetic touching floes)
# ---------------------------------------------------------------------------------------------------------------

#: Fig. 5.15(a) (p. 101): the 12 "ending point and its 8 surrounding pixels" patterns of a 4-connected, 1-px-thick
#: junction line (centre = ending point, exactly one 4-neighbour).  Transcribed row by row from the printed page;
#: order = reading order (left→right, top→bottom).  Filtering each with :data:`FIG_5_15_KERNEL` gives 3 at the centre.
FIG_5_15_ENDPOINT_PATTERNS = np.array(
    [
        [[0, 1, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [0, 0, 0]],
        [[0, 1, 1], [0, 1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0]],
        [[0, 0, 0], [1, 1, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 1, 0], [0, 0, 0]],
        [[1, 1, 0], [0, 1, 0], [0, 0, 0]],
    ],
    dtype=np.int64,
)

#: Fig. 5.15(b) (p. 101) / ``main.m`` line ``wr = [0 -1 0; -1 4 -1; 0 -1 0]``: ending-point detection kernel.
FIG_5_15_KERNEL = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float64)

#: Fig. 5.16 (p. 102): the 6×6 binary object used to illustrate the boundary tracing algorithm.
FIG_5_16_IMAGE = _m(
    """
    0 0 0 0 0 0
    0 0 1 1 0 0
    0 1 1 1 1 0
    0 0 1 1 1 0
    0 0 0 1 0 0
    0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 5.16: the first traced boundary pixels ``b0 → b1 → b2`` (1-based (row, col) as printed; the text starts at
#: the *uppermost-leftmost* pixel (2, 3)).  DIPUM ``boundaries.m`` starts at the first object pixel in **column-major**
#: order, (3, 2), so its closed list is a rotation of the book's: ``(3,2), (2,3), (2,4), (3,5), …`` — the printed
#: triple is a contiguous sub-sequence, not the head.
FIG_5_16_TRACE_MATLAB = ((2, 3), (2, 4), (3, 5))


def two_touching_floes(shape: tuple[int, int] = (96, 81), seed: int = 0, noise: float = 6.0) -> np.ndarray:
    """Synthetic RGB image in the spirit of ``q.jpg`` (Figs. 5.1(a), 5.8(a)): two bright convex floes on dark water
    that touch along a short junction with concave notches at both ends, plus mild Gaussian noise and a slight
    blue tint of the water.  Deterministic for a given ``seed``; returns uint8 ``(M, N, 3)``.

    Used as a test input / public substitute where the private book image is absent: Otsu binarisation gives one
    connected component, the city-block distance watershed splits it into >= 2 regions, and the notch ending
    points are concave (so the junction line survives neighbouring-region merging).
    """
    from .matlab_compat import to_uint8_saturating

    M, N = shape
    sr, sc = M / 96.0, N / 81.0  # ellipse geometry defined on the 96×81 grid of q.jpg, scaled to ``shape``
    rr, cc = np.mgrid[0:M, 0:N]
    a = ((rr - 30 * sr) / (24 * sr)) ** 2 + ((cc - 30 * sc) / (26 * sc)) ** 2 <= 1.0  # upper-left floe
    b = ((rr - 66 * sr) / (22 * sr)) ** 2 + ((cc - 52 * sc) / (24 * sc)) ** 2 <= 1.0  # lower-right floe (narrow neck)
    ice = a | b
    rng = np.random.default_rng(seed)
    base = np.where(ice, 205.0, 55.0) + rng.normal(0.0, noise, size=(M, N))
    rgb = np.stack([base - 4.0 * (~ice), base, base + 12.0 * (~ice)], axis=-1)
    return to_uint8_saturating(rgb)


def plateau_fixtures() -> dict[str, np.ndarray]:
    """Small constructed images whose watershed depends on the flooding *order* (plateaus, ties, corridors) — the
    kind of case where MATLAB's Meyer flooding and other implementations disagree.  Used to pin
    :func:`seaice.core.watershed.watershed` against MATLAB references.  All values are small integers stored as
    **int16**.  MATLAB's ``watershed`` rejects int16/int32 input (R2025a ``watershed.m`` accepts only
    uint8/uint16/single/double/logical), so the MATLAB references for these fixtures are computed on
    ``double(X)``; the Python port is fed the int16 arrays directly — the flooding only depends on the value
    order, so the two agree.
    """
    fx: dict[str, np.ndarray] = {}
    # even-width plateau (value 2) between two minima (value 1) — the dam position depends on the FIFO order
    t = np.full((5, 10), 5, dtype=np.int16)
    t[2, 1] = 1
    t[2, 8] = 1
    t[1:4, 2:8] = 2
    fx["even_plateau"] = t
    # diagonal / anti-diagonal corners of a plateau with minima at opposite corners
    t = np.full((6, 6), 3, dtype=np.int16)
    t[0, 0] = 1
    t[5, 5] = 1
    t[1:5, 1:5] = 2
    fx["diag_corners"] = t
    fx["anti_diag_corners"] = t[:, ::-1].copy()
    # four minima around a central plateau
    t = np.full((7, 7), 4, dtype=np.int16)
    t[2:5, 2:5] = 2
    for r, c in ((0, 3), (3, 0), (6, 3), (3, 6)):
        t[r, c] = 1
    fx["four_minima_plateau"] = t
    # 2-px-wide corridor joining two basins: the first-pushed side wins the corridor
    t = np.full((9, 12), 6, dtype=np.int16)
    t[1:8, 1:4] = 1
    t[1:8, 8:11] = 1
    t[3:5, 4:8] = 3
    fx["corridor_2px"] = t
    # -Inf planted minima in a float image
    t = np.full((6, 8), 2.0)
    t[1, 1] = -np.inf
    t[4, 6] = -np.inf
    fx["neg_inf_minima"] = t
    # 1-D profiles (row and column) with two basins and a plateau ridge
    prof = np.array([3, 2, 1, 2, 3, 3, 3, 2, 1], dtype=np.int16)
    fx["profile_row"] = prof[None, :].copy()
    fx["profile_col"] = prof[:, None].copy()
    # seeded random fields (ties everywhere)
    rng = np.random.default_rng(5)
    fx["random_u8_20x25"] = rng.integers(2, 6, size=(20, 25)).astype(np.uint8)
    fx["random_u8_40x50"] = rng.integers(2, 6, size=(40, 50)).astype(np.uint8)
    fx["random_binary_60x60"] = rng.integers(1, 3, size=(60, 60)).astype(np.uint8)
    fx["random_single_30x30"] = rng.integers(2, 5, size=(30, 30)).astype(np.float32)
    return fx


# ----------------------------------------------------------------------------------------------------------
# Chapter 6 fixtures (GVF snake)
# ----------------------------------------------------------------------------------------------------------

#: Book **Fig. 6.14(b)**: the city-block distance transform of :data:`FIG_6_14_IMAGE`, transcribed from the
#: printed 8x8 matrix (p. 134).  Its single regional maximum consists of the **three** pixels of value 3
#: ("a regional maximum consisting of three local maxima"); the seed radius is ``3/sqrt(2) = 2.1213``
#: (footnote 4, p. 135).
FIG_6_14_DISTANCE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 2, 2, 2, 2, 1, 0],
    [0, 1, 2, 3, 3, 2, 1, 0],
    [0, 0, 1, 2, 3, 2, 1, 0],
    [0, 0, 1, 1, 2, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.float64)

#: Book **Fig. 6.14(a)**: the 8x8 binary image matrix whose city-block distance transform is
#: :data:`FIG_6_14_DISTANCE` (the printed binary matrix is exactly ``distance > 0``).
FIG_6_14_IMAGE = FIG_6_14_DISTANCE > 0


def fig_6_16_circles(shape: tuple[int, int] = (110, 186), large_diameter: int = 61, small_diameter: int = 9,
                     large_center: tuple[int, int] | None = None,
                     small_center: tuple[int, int] | None = None) -> np.ndarray:
    """Book **Fig. 6.16(a)**: "a 110x186 binary image containing a large circle with 61-pixel wide diameter and a
    small circle with 9-pixel wide diameter" (§6.5.2, p. 139 caption).

    Used for the GVF capture-range experiment (Figs. 6.16(b)–(e): 5 / 30 / 100 / 250 GVF iterations; Fig. 6.17:
    snake evolution on the large circle under 30 vs 250 iterations).

    The two *diameters* and the image size are printed in the book; the circle **positions** are not, so they are
    chosen here to match the printed layout (the large circle left of centre, the small one well to its right,
    both vertically centred).  Tier 3 — synthetic.
    """
    M, N = shape
    if large_center is None:
        large_center = (M // 2, N // 4)
    if small_center is None:
        small_center = (M // 2, (3 * N) // 4)
    rr, cc = np.mgrid[0:M, 0:N]
    big = (rr - large_center[0]) ** 2 + (cc - large_center[1]) ** 2 <= (large_diameter / 2.0) ** 2
    small = (rr - small_center[0]) ** 2 + (cc - small_center[1]) ** 2 <= (small_diameter / 2.0) ** 2
    return big | small


def u_shape(shape: tuple[int, int] = (64, 64), thickness: int = 8, gap: int = 16,
            margin: int = 8) -> np.ndarray:
    """The classic U-shaped test object of Xu & Prince (book **Figs. 6.7 and 6.9**, §6.1.3 / §6.2).

    A traditional snake cannot enter the boundary concavity of a U (Fig. 6.7(c)); a GVF snake can (Fig. 6.9(c)).
    The book gives no pixel dimensions (only sigma = 4 for the smoothed external energy of Fig. 6.7(b)), so the
    shape is parametrised: ``thickness`` is the arm/base width, ``gap`` the width of the concavity, ``margin``
    the distance to the image border.  Tier 3 — synthetic.
    """
    M, N = shape
    bw = np.zeros((M, N), dtype=bool)
    left = (N - gap) // 2 - thickness
    right = (N + gap) // 2
    top = margin
    bottom = M - margin
    bw[top:bottom, left:left + thickness] = True                 # left arm
    bw[top:bottom, right:right + thickness] = True               # right arm
    bw[bottom - thickness:bottom, left:right + thickness] = True  # base
    return bw


def synthetic_floe_field(shape: tuple[int, int] = (200, 200), n_floes: int = 6, seed: int = 0,
                         radius: tuple[int, int] = (14, 30), noise: float = 0.0) -> np.ndarray:
    """A field of overlapping bright discs on a dark background — a stand-in for the aerial floe images the book
    uses in Figs. 6.10–6.13 and 6.18–6.21, which are **not shipped** with the MATLAB code.

    Returns a uint8 grayscale image.  Tier 3 — synthetic, seeded.
    """
    rng = np.random.default_rng(seed)
    M, N = shape
    img = np.zeros((M, N), dtype=np.float64)
    rr, cc = np.mgrid[0:M, 0:N]
    for _ in range(n_floes):
        r = rng.integers(radius[0], radius[1] + 1)
        cy = rng.integers(r, M - r)
        cx = rng.integers(r, N - r)
        img[(rr - cy) ** 2 + (cc - cx) ** 2 <= r * r] = 210.0
    if noise:
        img = img + rng.normal(0.0, noise, img.shape)
    return np.clip(np.floor(img + 0.5), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------
# Chapter 7 fixtures — the printed matrices of Figs. 7.2-7.8 (Book §7.1, pp. 146-153)
#
# Every block below was transcribed from ``chapters/ch07.txt`` (row-major, verified against the ``.m`` literals of
# ``ch7/cleaning & labeling & filling/*.m``) and re-derived with ``core.morphology``: Fig. 7.2(b)/(c)/(d) from
# ``imclose``/``imopen`` with ``strel('square', 2)``, Figs. 7.3-7.7 from the Eq. (7.1)/(7.2) recursion and
# Fig. 7.8 from ``imreconstruct(F_m, F^c, conn=4)``.  **All 48 printed blocks agree 0 px except one**
# (4 + 8 + 7 + 8 + 4 + 10 + 7 step blocks for Figs. 7.2-7.8) — Fig. 7.7(e)
# block ``X_8`` (see :data:`FIG_7_7_STEPS_BOOK` / :data:`FIG_7_7_X8_TYPO`).
# ---------------------------------------------------------------------------------------------------------------

#: Fig. 7.2(a) p. 146 — the 13x23 binary image of ``morphology_cleaning.m`` lines 3-15 (= the printed matrix).
FIG_7_2_IMAGE = _m(
    """
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 1 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 0 0 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 7.2(b) — ``imclose(I, strel('square', 2))`` (111 -> 115 px).
FIG_7_2_CLOSED = _m(
    """
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 1 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 7.2(c) — ``imopen(I, strel('square', 2))`` (111 -> 104 px).
FIG_7_2_OPENED = _m(
    """
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 0 0 0 1 1 1 0 0 0 1 1 0 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 7.2(d) — morphological cleaning = ``imopen(imclose(I, se), se)`` (111 -> 108 px).
FIG_7_2_CLEANED = _m(
    """
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 7.3(b) p. 148 — the 9x9 binary image of ``labeling.m`` lines 3-11 (24 object pixels).
FIG_7_3_IMAGE = _m(
    """
    0 0 0 0 0 0 0 0 0
    0 1 1 1 1 0 0 0 0
    0 1 1 0 0 1 0 0 0
    0 1 1 0 0 1 1 0 0
    0 1 1 1 0 0 1 1 0
    0 0 1 1 1 0 0 1 0
    0 0 0 0 0 1 1 1 0
    0 0 0 0 0 0 1 0 0
    0 0 0 0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 7.3(c) — the initial step X_0 = {p}, the single seed at 1-based (2, 3) (``labeling.m`` lines 13-21).
FIG_7_3_SEED = _m(
    """
    0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 7.3(d) — the 8 printed blocks ``[X_0 (+) B, X_1, ..., X_7]`` of the Eq. (7.1) recursion with the 3x3
#: square SE (8-connectivity).  Sums 9, 5, 8, 12, 17, 20, 23, 24; ``X_7 == A``, i.e. it completes at the 8th
#: printed block (the text's 'completes at the 8th iteration', p. 147).
FIG_7_3_STEPS = (
    _m(
        """
        0 1 1 1 0 0 0 0 0
        0 1 1 1 0 0 0 0 0
        0 1 1 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 0 0 0 0 0
        0 1 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 1 0 0 0 0
        0 1 1 0 0 0 0 0 0
        0 1 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 1 0 0 0 0
        0 1 1 0 0 1 0 0 0
        0 1 1 0 0 0 0 0 0
        0 1 1 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 1 0 0 0 0
        0 1 1 0 0 1 0 0 0
        0 1 1 0 0 1 1 0 0
        0 1 1 1 0 0 0 0 0
        0 0 1 1 1 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 1 0 0 0 0
        0 1 1 0 0 1 0 0 0
        0 1 1 0 0 1 1 0 0
        0 1 1 1 0 0 1 1 0
        0 0 1 1 1 0 0 0 0
        0 0 0 0 0 1 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 1 0 0 0 0
        0 1 1 0 0 1 0 0 0
        0 1 1 0 0 1 1 0 0
        0 1 1 1 0 0 1 1 0
        0 0 1 1 1 0 0 1 0
        0 0 0 0 0 1 1 0 0
        0 0 0 0 0 0 1 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 1 0 0 0 0
        0 1 1 0 0 1 0 0 0
        0 1 1 0 0 1 1 0 0
        0 1 1 1 0 0 1 1 0
        0 0 1 1 1 0 0 1 0
        0 0 0 0 0 1 1 1 0
        0 0 0 0 0 0 1 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
)

#: Fig. 7.4(d) p. 149 — the 7 printed blocks ``[X_0 (+) B, X_1, ..., X_6]`` of Eq. (7.1) with the cross SE
#: (``strel('diamond', 1)``, 4-connectivity), same image and seed.  Sums 5, 4, 7, 9, 12, 13, 14.
FIG_7_4_STEPS = (
    _m(
        """
        0 0 1 0 0 0 0 0 0
        0 1 1 1 0 0 0 0 0
        0 0 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 0 0 0 0 0
        0 0 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 1 0 0 0 0
        0 1 1 0 0 0 0 0 0
        0 0 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 1 0 0 0 0
        0 1 1 0 0 0 0 0 0
        0 1 1 0 0 0 0 0 0
        0 0 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 1 0 0 0 0
        0 1 1 0 0 0 0 0 0
        0 1 1 0 0 0 0 0 0
        0 1 1 1 0 0 0 0 0
        0 0 1 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 1 0 0 0 0
        0 1 1 0 0 0 0 0 0
        0 1 1 0 0 0 0 0 0
        0 1 1 1 0 0 0 0 0
        0 0 1 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 1 0 0 0 0
        0 1 1 0 0 0 0 0 0
        0 1 1 0 0 0 0 0 0
        0 1 1 1 0 0 0 0 0
        0 0 1 1 1 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
)

#: Fig. 7.5(e) p. 150 — the 8 printed blocks of the Eq. (7.2) hole filling of :data:`FIG_7_3_IMAGE` from the
#: seed at 1-based (3, 4) with the cross SE: ``[X_0 (+) B, X_1, ..., X_6, X_6 | A]``.  Sums 5, 3, 4, 5, 6, 7, 8,
#: 32.  (The text says it 'finishes at the 7th iteration'; the printed blocks stop at X_6 before the union, and
#: X_7 == X_6, so both readings give the same final image.)
FIG_7_5_STEPS = (
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 0 0 1 0 0 0 0 0
        0 0 1 1 1 0 0 0 0
        0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 0 1 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 0 1 1 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 0 1 1 0 0 0
        0 0 0 0 0 1 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 0 1 1 0 0 0
        0 0 0 0 0 1 1 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 1 0 0 0 0
        0 1 1 1 1 1 0 0 0
        0 1 1 1 1 1 1 0 0
        0 1 1 1 1 1 1 1 0
        0 0 1 1 1 1 1 1 0
        0 0 0 0 0 1 1 1 0
        0 0 0 0 0 0 1 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
)

#: Fig. 7.5(d) p. 150 — the hole-filling seed ``x0`` of ``filling.m`` lines 13-21: a single pixel at 1-based
#: (3, 4) = 0-based (2, 3), which lies **inside the hole** of :data:`FIG_7_3_IMAGE`.  The same seed is used for
#: Figs. 7.6 and 7.7.
FIG_7_5_SEED = _m(
    """
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 7.6(b) p. 151 — :data:`FIG_7_3_IMAGE` **plus one pixel at 1-based (5, 5)** (25 object pixels), which
#: splits the interior into two 4-connected holes.  This variant is printed in the book but is **not** in any
#: shipped ``.m`` file (``filling.m`` only has the Fig. 7.5 image).
FIG_7_6_IMAGE = _m(
    """
    0 0 0 0 0 0 0 0 0
    0 1 1 1 1 0 0 0 0
    0 1 1 0 0 1 0 0 0
    0 1 1 0 0 1 1 0 0
    0 1 1 1 1 0 1 1 0
    0 0 1 1 1 0 0 1 0
    0 0 0 0 0 1 1 1 0
    0 0 0 0 0 0 1 0 0
    0 0 0 0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 7.6(e) — the 4 printed blocks ``[X_0 (+) B, X_1, X_2, X_2 | A]`` with the cross SE: the recursion
#: converges after 2 iterations and fills **only one** of the two holes (sums 5, 3, 4, 29).
FIG_7_6_STEPS = (
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 0 0 1 0 0 0 0 0
        0 0 1 1 1 0 0 0 0
        0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 1 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 1 1 1 1 0 0 0 0
        0 1 1 1 1 1 0 0 0
        0 1 1 1 1 1 1 0 0
        0 1 1 1 1 0 1 1 0
        0 0 1 1 1 0 0 1 0
        0 0 0 0 0 1 1 1 0
        0 0 0 0 0 0 1 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
)

#: Fig. 7.7(e) p. 152 — the 10 blocks **as printed** ``[X_0 (+) B, X_1, ..., X_9]`` for the same image and seed
#: with the 3x3 square SE, where the hole is 8-connected to the background and the flood escapes.
#: **Block index 8 (``X_8``) contains a book typo**: it prints 0 at 1-based (8, 9) and (9, 9) (sum 53), but
#: ``X_7`` already has (7, 9) = 1 and (8, 8) = 1, so the 3x3 dilation must set them and ``A^c`` allows them —
#: the recursion gives sum 55, and the printed ``X_9 = X_10 = A^c`` agrees with the recursion.
#: :data:`FIG_7_7_STEPS` holds the corrected sequence; this constant keeps the printed variant for the record.
FIG_7_7_STEPS_BOOK = (
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 0 1 1 1 0 0 0 0
        0 0 1 1 1 0 0 0 0
        0 0 1 1 1 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 0 0 1 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 1 1 1 0 0
        0 0 0 0 0 1 1 0 0
        0 0 0 1 1 0 1 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 0 0 1 0 0 0
        0 0 0 0 0 1 1 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 1 1 1 1 1 0
        0 0 0 0 0 1 1 1 0
        0 0 0 1 1 0 1 1 0
        0 0 0 1 1 0 0 1 0
        0 0 0 0 0 1 0 0 0
        0 0 0 0 0 1 1 0 0
        0 0 0 0 1 0 0 0 0
        0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 1 1 1 1 1 1 1
        0 0 0 0 0 1 1 1 1
        0 0 0 1 1 0 1 1 1
        0 0 0 1 1 0 0 1 1
        0 0 0 0 0 1 0 0 1
        0 0 0 0 0 1 1 0 0
        0 0 0 1 1 0 0 0 0
        0 0 0 1 1 1 0 0 0
        0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 1 1 1 1 1 1 1 1
        0 0 0 0 0 1 1 1 1
        0 0 0 1 1 0 1 1 1
        0 0 0 1 1 0 0 1 1
        0 0 0 0 0 1 0 0 1
        0 0 0 0 0 1 1 0 1
        0 0 1 1 1 0 0 0 0
        0 0 1 1 1 1 0 0 0
        0 0 1 1 1 1 1 0 0
        """
    ).astype(bool),
    _m(
        """
        1 1 1 1 1 1 1 1 1
        1 0 0 0 0 1 1 1 1
        0 0 0 1 1 0 1 1 1
        0 0 0 1 1 0 0 1 1
        0 0 0 0 0 1 0 0 1
        0 1 0 0 0 1 1 0 1
        0 1 1 1 1 0 0 0 1
        0 1 1 1 1 1 0 1 0
        0 1 1 1 1 1 1 1 0
        """
    ).astype(bool),
    _m(
        """
        1 1 1 1 1 1 1 1 1
        1 0 0 0 0 1 1 1 1
        1 0 0 1 1 0 1 1 1
        0 0 0 1 1 0 0 1 1
        1 0 0 0 0 1 0 0 1
        1 1 0 0 0 1 1 0 1
        1 1 1 1 1 0 0 0 1
        1 1 1 1 1 1 0 1 0
        1 1 1 1 1 1 1 1 0
        """
    ).astype(bool),
    _m(
        """
        1 1 1 1 1 1 1 1 1
        1 0 0 0 0 1 1 1 1
        1 0 0 1 1 0 1 1 1
        1 0 0 1 1 0 0 1 1
        1 0 0 0 0 1 0 0 1
        1 1 0 0 0 1 1 0 1
        1 1 1 1 1 0 0 0 1
        1 1 1 1 1 1 0 1 1
        1 1 1 1 1 1 1 1 1
        """
    ).astype(bool),
)

#: Fig. 7.7(e) with the ``X_8`` typo corrected (sums 9, 4, 6, 13, 19, 29, 38, 47, **55**, 56).  This is the
#: sequence the Eq. (7.1) recursion actually produces and the one the port reproduces.
def _with_typo_fixed(blocks: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    """``FIG_7_7_STEPS_BOOK`` with the two ``X_8`` typo pixels set (the arrays are frozen read-only below)."""
    out = [b.copy() for b in blocks]
    out[8][7, 8] = True   # 1-based (8, 9)
    out[8][8, 8] = True   # 1-based (9, 9)
    return tuple(out)


FIG_7_7_STEPS = _with_typo_fixed(FIG_7_7_STEPS_BOOK)

#: The two pixels the book prints as 0 in Fig. 7.7(e) ``X_8`` (0-based row, col).
FIG_7_7_X8_TYPO = ((7, 8), (8, 8))

#: Fig. 7.8(b) p. 153 — the 9x11 binary image ``F`` of ``filling_reconstruct.m`` lines 3-11 (39 object pixels).
FIG_7_8_IMAGE = _m(
    """
    0 0 0 0 0 0 0 0 1 1 1
    0 1 1 1 1 0 0 0 1 0 1
    0 1 1 0 0 1 0 0 1 0 1
    0 1 1 0 0 1 1 0 0 1 1
    0 1 1 1 1 0 1 1 0 1 0
    0 0 1 1 1 0 0 1 0 0 0
    0 0 0 0 0 1 1 1 0 0 0
    1 1 0 0 0 0 1 0 0 0 0
    1 0 0 1 0 0 0 0 0 0 0
    """
).astype(bool)

#: Fig. 7.8(d) — the **Eq. (7.3)** border marker ``F_m`` (``filling_reconstruct.m`` lines 23-31): ``1 - F`` on
#: the image border, 0 elsewhere (**27** pixels; the 48 belongs to ``H``, the last-but-one block of
#: :data:`FIG_7_8_STEPS`).  Verified equal to Eq. (7.3) applied to :data:`FIG_7_8_IMAGE`.
FIG_7_8_MARKER = _m(
    """
    1 1 1 1 1 1 1 1 0 0 0
    1 0 0 0 0 0 0 0 0 0 0
    1 0 0 0 0 0 0 0 0 0 0
    1 0 0 0 0 0 0 0 0 0 0
    1 0 0 0 0 0 0 0 0 0 1
    1 0 0 0 0 0 0 0 0 0 1
    1 0 0 0 0 0 0 0 0 0 1
    0 0 0 0 0 0 0 0 0 0 1
    0 1 1 0 1 1 1 1 1 1 1
    """
).astype(bool)

#: Fig. 7.8(e) — the 7 printed blocks ``[F_m (+) B, D^(1), D^(2), D^(3), D^(4), H, H & F^c]`` with the cross SE.
#: Sums 55, 40, 47, 50, 51, 48, 9; the reconstruction converges at ``D^(4)`` and ``H = [R^D_{F^c}(F_m)]^c``
#: (**Eq. 7.4**) has 48 pixels, of which 9 are the filled holes.
FIG_7_8_STEPS = (
    _m(
        """
        1 1 1 1 1 1 1 1 1 0 0
        1 1 1 1 1 1 1 1 0 0 0
        1 1 0 0 0 0 0 0 0 0 0
        1 1 0 0 0 0 0 0 0 0 1
        1 1 0 0 0 0 0 0 0 1 1
        1 1 0 0 0 0 0 0 0 1 1
        1 1 0 0 0 0 0 0 0 1 1
        1 1 1 0 1 1 1 1 1 1 1
        1 1 1 1 1 1 1 1 1 1 1
        """
    ).astype(bool),
    _m(
        """
        1 1 1 1 1 1 1 1 0 0 0
        1 0 0 0 0 1 1 1 0 0 0
        1 0 0 0 0 0 0 0 0 0 0
        1 0 0 0 0 0 0 0 0 0 0
        1 0 0 0 0 0 0 0 0 0 1
        1 1 0 0 0 0 0 0 0 1 1
        1 1 0 0 0 0 0 0 0 1 1
        0 0 1 0 1 1 0 1 1 1 1
        0 1 1 0 1 1 1 1 1 1 1
        """
    ).astype(bool),
    _m(
        """
        1 1 1 1 1 1 1 1 0 0 0
        1 0 0 0 0 1 1 1 0 0 0
        1 0 0 0 0 0 1 1 0 0 0
        1 0 0 0 0 0 0 0 0 0 0
        1 0 0 0 0 0 0 0 0 0 1
        1 1 0 0 0 0 0 0 1 1 1
        1 1 1 0 1 0 0 0 1 1 1
        0 0 1 1 1 1 0 1 1 1 1
        0 1 1 0 1 1 1 1 1 1 1
        """
    ).astype(bool),
    _m(
        """
        1 1 1 1 1 1 1 1 0 0 0
        1 0 0 0 0 1 1 1 0 0 0
        1 0 0 0 0 0 1 1 0 0 0
        1 0 0 0 0 0 0 1 0 0 0
        1 0 0 0 0 0 0 0 1 0 1
        1 1 0 0 0 0 0 0 1 1 1
        1 1 1 1 1 0 0 0 1 1 1
        0 0 1 1 1 1 0 1 1 1 1
        0 1 1 0 1 1 1 1 1 1 1
        """
    ).astype(bool),
    _m(
        """
        1 1 1 1 1 1 1 1 0 0 0
        1 0 0 0 0 1 1 1 0 0 0
        1 0 0 0 0 0 1 1 0 0 0
        1 0 0 0 0 0 0 1 1 0 0
        1 0 0 0 0 0 0 0 1 0 1
        1 1 0 0 0 0 0 0 1 1 1
        1 1 1 1 1 0 0 0 1 1 1
        0 0 1 1 1 1 0 1 1 1 1
        0 1 1 0 1 1 1 1 1 1 1
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 1 1 1
        0 1 1 1 1 0 0 0 1 1 1
        0 1 1 1 1 1 0 0 1 1 1
        0 1 1 1 1 1 1 0 0 1 1
        0 1 1 1 1 1 1 1 0 1 0
        0 0 1 1 1 1 1 1 0 0 0
        0 0 0 0 0 1 1 1 0 0 0
        1 1 0 0 0 0 1 0 0 0 0
        1 0 0 1 0 0 0 0 0 0 0
        """
    ).astype(bool),
    _m(
        """
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 1 0
        0 0 0 1 1 0 0 0 0 1 0
        0 0 0 1 1 0 0 0 0 0 0
        0 0 0 0 0 1 0 0 0 0 0
        0 0 0 0 0 1 1 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0 0 0 0
        """
    ).astype(bool),
)


# ---------------------------------------------------------------------------------------------------------------
# The chapter-7 fixtures are module-level *constants*: freeze them read-only (ch07 review nit N7) so that a
# caller writing into a block cannot corrupt every later use of the same array.  Everything in the package and
# in tests/ only reads them (or copies via ``.astype`` / ``.copy()``), so this is a no-op at run time.
# ---------------------------------------------------------------------------------------------------------------

def _freeze(obj):
    """Set ``write=False`` on an ndarray or on every ndarray of a tuple; returns ``obj``."""
    if isinstance(obj, np.ndarray):
        obj.setflags(write=False)
    elif isinstance(obj, tuple):
        for item in obj:
            _freeze(item)
    return obj


for _name, _value in list(globals().items()):
    if _name.startswith("FIG_7_"):
        _freeze(_value)
del _name, _value


# ---------------------------------------------------------------------------------------------------------------
# Chapter 9 fixtures — model-basin (ice-tank) imagery (Tier 3, seeded)
#
# The three external inputs of chapter 9 (`04100_analyse.jpg`, `dypic_05100_cam1_top.avi`, `05100.avi`) are HSVA /
# DYPIC campaign assets that were **never published**: there is no Tier-2 (free online) and no Tier-4 (login)
# source for a top-view photograph of a *cut, rectangular* model-ice field (`analysis/ch09.md` §8, risk R1).  The
# generators below therefore stand in for them.  They reproduce the *structure* the scripts need, never the book's
# numbers: every quantity computed on them is `unverified` as a book number and only `exact` as a parity number
# against MATLAB running the same `.m` file on the same file.
#
# Each generator's constraints come from `analysis/ch09.md` §8 and are load-bearing for the verifier: they exist so
# that every branch of the ported code is exercised (a fixture that cannot distinguish erratum E4 from the correct
# rule is not a fixture -- pitfall 44).
# ---------------------------------------------------------------------------------------------------------------

def _draw_rect(img: np.ndarray, cy: float, cx: float, h: float, w: float, angle: float, value: float) -> None:
    """Paint a filled, rotated rectangle of extent ``h x w`` centred at ``(cy, cx)`` into ``img``."""
    M, N = img.shape[:2]
    half = 0.5 * (h + w)
    r0 = max(0, int(np.floor(cy - half)))
    r1 = min(M, int(np.ceil(cy + half)) + 1)
    c0 = max(0, int(np.floor(cx - half)))
    c1 = min(N, int(np.ceil(cx + half)) + 1)
    if r1 <= r0 or c1 <= c0:
        return
    rr, cc = np.mgrid[r0:r1, c0:c1]
    ca, sa = np.cos(angle), np.sin(angle)
    u = (cc - cx) * ca + (rr - cy) * sa
    v = -(cc - cx) * sa + (rr - cy) * ca
    sel = (np.abs(u) <= w / 2.0) & (np.abs(v) <= h / 2.0)
    block = img[r0:r1, c0:c1]
    block[sel] = value
    img[r0:r1, c0:c1] = block


def model_ice_tank(shape: tuple[int, int] = (348, 1770), *, floe_sizes: tuple[float, ...] = (0.5, 1.0, 1.5),
                   shares: tuple[float, ...] = (0.45, 0.40, 0.15), target_ic: float = 0.86,
                   px_per_m: float = 26.0, seed: int = 0, ice_level: float = 205.0,
                   water_level: float = 55.0, illumination: float = 0.10, noise: float = 4.0,
                   blur: float = 0.9, corner_triangle: bool = True) -> np.ndarray:
    """A synthetic **overall tank image** — the Tier-3 stand-in for ``04100_analyse.jpg`` (Book Fig. 9.1, p. 197).

    Book: §9.1 (a 54 m level-ice sheet cut into squares and spread over 64 m of tank) and §9.2.1 (the image
    ``block_threshold.m`` thresholds).  The printed Fig. 9.1 bitmap is 442 x 87 px, aspect **5.08 : 1**; the
    default ``shape`` is **348 x 1770** (aspect 5.09 : 1) with ``348 / 2 = 174`` rows and ``1770 / 3 = 590``
    columns, so ``block_threshold.m``'s ``n_r = 2, n_c = 3`` grid slices it into six equal blocks without the
    non-integer indices that make MATLAB error.

    Structure reproduced from the text: bright quasi-square floes of three sizes in the Table 9.1 mix, a
    near-uniform illumination gradient (so the two Otsu assumptions of p. 199 hold and global ~ local ~ k-means),
    and -- when ``corner_triangle`` -- the **out-of-tank bright triangle in the upper-right corner** that p. 199
    blames for the 3-8 % ice-concentration deficit.

    **Tier 3, synthetic, seeded.**  Never present any number computed from it as a book number.

    Parameters
    ----------
    shape : (rows, cols)
    floe_sizes, shares : the Table 9.1 square edge lengths in metres and their percentages
    target_ic : float
        Fraction of the image the floes should cover before overlap (Table 9.1's 86 % for run 5100).
    px_per_m : float
        Model-scale pixels per metre (``1.5 m`` -> 39 px at the default).
    ice_level, water_level, illumination, noise : appearance controls
    blur : float
        Gaussian sigma applied before the noise.  It is **not** cosmetic: hard-edged rectangles on a two-tone
        background leave an *empty band* in the histogram, and every threshold inside that band gives the same
        mask — which would make a ``compare='gt'`` vs ``compare='ge'`` comparison (and a global-vs-local Otsu
        comparison) agree for a reason that has nothing to do with the algorithms (CUMULATIVE pitfall 60, the
        ch08 "degenerate fixture" trap).  The blur populates every gray level between water and ice, so a
        one-level move of the threshold changes the count.
    seed : int

    Returns a uint8 RGB image (``(rows, cols, 3)``) so ``rgb2gray`` in the port has something to convert, exactly
    as ``block_threshold.m`` line 3 expects.
    """
    rng = np.random.default_rng(seed)
    M, N = shape
    img = np.full((M, N), water_level, dtype=np.float64)
    sizes_px = [max(3.0, s * px_per_m) for s in floe_sizes]
    # Place floes size class by size class, largest first, so the big ones are not crowded out, and keep adding
    # until the *union* coverage reaches the class's share of `target_ic` (random placement overlaps, so drawing
    # `target_area / side^2` rectangles would only cover 1 - exp(-target_ic) ~ 58 % of the image).
    ice = np.zeros((M, N), dtype=bool)
    order = np.argsort(sizes_px)[::-1]
    covered = 0.0
    for idx in order:
        side = sizes_px[idx]
        want = covered + target_ic * M * N * shares[idx]
        guard = 0
        while ice.sum() < want and guard < 200000:
            guard += 1
            cy = rng.uniform(side / 2, M - side / 2)
            cx = rng.uniform(side / 2, N - side / 2)
            ang = rng.uniform(-0.12, 0.12)              # the cut squares are nearly, not exactly, aligned
            jitter = rng.uniform(0.9, 1.1)
            _draw_rect(img, cy, cx, side * jitter, side * jitter, ang, ice_level + rng.uniform(-10.0, 10.0))
            _draw_rect(ice, cy, cx, side * jitter, side * jitter, ang, True)
        covered = float(ice.sum())
    if corner_triangle:
        # p. 199: the bright region outside the tank in the upper-right corner of Fig. 9.1.
        rr, cc = np.mgrid[0:M, 0:N]
        tri_h, tri_w = M // 3, N // 12
        img[(rr < tri_h) & (cc > N - 1 - tri_w * (1.0 - rr / max(tri_h, 1)) - 1)] = 240.0
    if illumination:
        ramp = 1.0 + illumination * (np.linspace(-0.5, 0.5, N)[None, :] + np.linspace(-0.3, 0.3, M)[:, None])
        img = img * ramp
    if blur:
        from scipy.ndimage import gaussian_filter

        img = gaussian_filter(img, blur, mode="nearest")
    if noise:
        img = img + rng.normal(0.0, noise, img.shape)
    gray = np.clip(np.floor(img + 0.5), 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def model_ice_tank_video(n_frames: int = 60, shape: tuple[int, int] = (480, 640), *,
                         drift: tuple[float, float] = (0.0, -2.0), seed: int = 0,
                         vessel_box: tuple[tuple[int, int], tuple[int, int]] = ((307, 400), (268, 380)),
                         ic_range: tuple[float, float] = (0.55, 0.88), n_floes: int = 1000,
                         ice_level: float = 200.0, water_level: float = 50.0,
                         highlight: bool = True) -> np.ndarray:
    """A synthetic **top-view tank video** -- the Tier-3 stand-in for ``dypic_05100_cam1_top.avi`` (§9.2.2).

    Book: §9.2.2 (Figs. 9.6-9.10), where a carriage-mounted top-view camera films the managed-ice field around the
    towed vessel and the video is decimated to 1 fps.  MATLAB consumers:
    ``MATLAB_ROOT/ch9/movie_otsu.m`` and ``MATLAB_ROOT/ch9/movie_kmeans.m``.

    The constraints below are **load-bearing** (`analysis/ch09.md` §8, test-design note 1) -- do not relax them:

    1. ``shape`` must be at least ``(400, 521)``: the scripts hard-code the crop ``y 180:400``, ``x 125:521``.
    2. The ice concentration inside the crop **changes over time**, so ``IC(t)`` has structure.
    3. A visually distinct region sits under ``rows 307:400 x cols 268:380`` (the vessel), so the blanking step of
       lines 29-30 is observable.
    4. **At least two consecutive frames have a decreasing Otsu threshold.**  Without that,
       ``max_{j<=k} t(j) == t(k)`` and erratum **E4** of ``movie_otsu.m`` (``if I(i,j) >= t*255`` against the whole
       growing ``t`` vector => the running maximum) is invisible -- the fixture could not distinguish the bug from
       the correct per-frame rule.  It is achieved by ramping the ice brightness up and then down.
    5. A bright **light reflection** on the water (``highlight``), the p.-200 source of the upward IC bias.

    **Tier 3, synthetic, seeded.**  Returns ``(N, H, W, 3)`` uint8 (imageio order -- pass it straight to
    :func:`seaice.core.video.write_video`, which also accepts MATLAB's ``(H, W, 3, N)``).
    """
    M, N = shape
    (y3, y2), (x3, x4) = vessel_box
    if M < y2 or N < 521:
        raise ValueError(f"model_ice_tank_video: shape {shape} is too small for the scripts' hard-coded crop "
                         "(y 180:400, x 125:521) -- need at least (400, 521)")
    rng = np.random.default_rng(seed)
    # Floe field in a coordinate system that drifts; sizes in px roughly matching the 0.5/1.0/1.5 m mix.
    sides = rng.choice([14.0, 22.0, 34.0], size=n_floes, p=[0.45, 0.40, 0.15])
    cy0 = rng.uniform(-40, M + 40, n_floes)
    cx0 = rng.uniform(-60, N + 60, n_floes)
    ang = rng.uniform(-0.15, 0.15, n_floes)
    bright = rng.uniform(-12.0, 12.0, n_floes)
    # Fraction of floes present in frame k: rises, then falls, so IC(t) is non-monotone.
    phase = np.linspace(0.0, 1.0, n_frames)
    present_frac = ic_range[0] + (ic_range[1] - ic_range[0]) * np.sin(np.pi * phase) ** 0.7
    # Constraint 4: a brightness ramp that goes up for the first half and down for the second half moves the
    # per-frame Otsu threshold in both directions.
    gain = 1.0 + 0.22 * np.sin(2.0 * np.pi * phase)
    order = rng.permutation(n_floes)
    out = np.zeros((n_frames, M, N, 3), dtype=np.uint8)
    rr, cc = np.mgrid[0:M, 0:N]
    for k in range(n_frames):
        img = np.full((M, N), water_level, dtype=np.float64)
        keep = order[: int(round(present_frac[k] * n_floes))]
        for i in keep:
            cy = cy0[i] + drift[0] * k
            cx = (cx0[i] + drift[1] * k) % (N + 120) - 60
            _draw_rect(img, cy, cx, sides[i], sides[i], ang[i], (ice_level + bright[i]) * gain[k])
        if highlight:                       # p. 200: tank-bottom lights reflecting off the water
            d2 = (rr - 220.0) ** 2 / 900.0 + (cc - 180.0 - 0.6 * k) ** 2 / 3600.0
            img = img + 120.0 * np.exp(-d2)
        # Constraint 3: the vessel is a distinct dark hull with bright deck marks inside the blanking box.
        img[y3 - 1:y2, x3 - 1:x4] = 20.0
        img[y3 + 9:y3 + 30, x3 + 19:x3 + 60] = 225.0
        img = img + rng.normal(0.0, 3.0, img.shape)
        gray = np.clip(np.floor(img + 0.5), 0, 255).astype(np.uint8)
        out[k] = np.repeat(gray[:, :, None], 3, axis=2)
    return out


def segmented_floe_video(n_frames: int = 60, shape: tuple[int, int] = (240, 320), *, seed: int = 0,
                         blank_frame: int | None = None) -> np.ndarray:
    """A synthetic **already-segmented** floe video -- the Tier-3 stand-in for ``05100.avi`` (§9.3.3, Fig. 9.18).

    Book: §9.3.3 "monitoring maximum floe size".  MATLAB consumer:
    ``MATLAB_ROOT/ch9/Model_Ice_Floe_Identification/movie_floe.m``, whose line 18 is ``im2bw(mov(k).cdata)``
    with **no level** (=> 0.5 => ``rgb2gray`` then ``> 127.5``).  The AVI it reads must therefore already hold the
    **binary** result of the per-frame Algorithm-7 segmentation, which the book never ships (gap **G2**).  Here the
    masks are rendered as 0/255 RGB, so ``im2bw`` at the default level recovers them exactly.

    Constraints (`analysis/ch09.md` §8, load-bearing for the verifier):

    1. The **largest** component changes from frame to frame, so ``floe(k) = max(ice_areas)`` is a real time series.
    2. Every frame contains one component of exactly **19 px** and one of exactly **20 px**, pinning
       ``bwareaopen``'s ``>= P`` rule (19 removed, 20 kept at ``P = 20``).
    3. Every frame contains a pair of blobs that are **8-connected but not 4-connected** (a diagonal touch),
       pinning ``bwlabel(.., 4)`` -- with 8-connectivity they would be one component.
    4. ``blank_frame`` (optional): the index of an **empty** frame, which makes ``max([])`` return ``[]``, so
       ``movie_floe.m`` line 25 executes ``floe(k) = []`` with ``numel(floe) == k-1``.  MATLAB R2025a
       **raises** there — ``MATLAB:matrix:singleSubscriptNumelMismatch``, "Unable to perform assignment because
       the left and right sides have a different number of elements" (the literal ``x(k) = []`` spelling gives
       ``MATLAB:subsdeldimmismatch``); a null assignment *past the end* is not a deletion, and this loop, which
       appends one element per frame, never reaches the ``k <= numel(floe)`` case where deletion happens.  This
       **corrects risk R13** of ``analysis/ch09.md``, which predicted a silent deletion — see
       :func:`seaice.ch09_model_ice.movie_floe` (``empty='raise'`` is the default *because* it is the literal
       behaviour; ``empty='delete'`` keeps the deletion semantics for the reachable case).  ``None`` (default)
       leaves the blank frame out.

    **Tier 3, synthetic, seeded.**  Returns ``(N, H, W, 3)`` uint8.
    """
    rng = np.random.default_rng(seed)
    M, N = shape
    # A 3 x 4 grid inside rows [40, M) keeps the floes **separate** (so "the largest floe" is a floe, not a merged
    # blob) and leaves the top-left corner free for the 19/20-px and diagonal-touch fixtures below.
    n_rows, n_cols = 3, 4
    n_big = n_rows * n_cols
    cell_h = (M - 40) / n_rows
    cell_w = N / n_cols
    cy0 = np.array([40 + (i // n_cols + 0.5) * cell_h for i in range(n_big)])
    cx0 = np.array([(i % n_cols + 0.5) * cell_w for i in range(n_big)])
    sides = rng.permutation(np.linspace(18.0, 40.0, n_big))
    ang = rng.uniform(-0.3, 0.3, n_big)
    # Constraint 1: each floe breathes with its own amplitude, period and phase, so the maximum area is
    # **non-monotone** and the floe that attains it **changes identity** several times over the sequence.  A
    # monotone series (a common first attempt) cannot distinguish `max` from `last` or from a running maximum.
    amp = rng.uniform(4.0, 14.0, n_big)
    period = rng.uniform(11.0, 29.0, n_big)
    phase = rng.uniform(0.0, 2.0 * np.pi, n_big)
    cap = 0.72 * min(cell_h, cell_w)                # never let two neighbours touch
    out = np.zeros((n_frames, M, N, 3), dtype=np.uint8)
    for k in range(n_frames):
        bw = np.zeros((M, N), dtype=np.float64)
        if blank_frame is None or k != blank_frame:
            for i in range(n_big):
                s = float(np.clip(sides[i] + amp[i] * np.sin(2.0 * np.pi * k / period[i] + phase[i]),
                                  8.0, cap))
                _draw_rect(bw, cy0[i], cx0[i], s, s * 0.85, ang[i], 1.0)
            # Constraint 2: exactly 19 px and exactly 20 px, isolated in the top-left corner.
            bw[2:5, 2:8] = 1.0                       # 18 px
            bw[5, 2] = 1.0                           # -> 19 px  (dropped by bwareaopen(.., 20))
            bw[2:5, 12:18] = 1.0                     # 18 px
            bw[5, 12:14] = 1.0                       # -> 20 px  (kept: the rule is `>= P`)
            # Constraint 3: two 5x5 blocks touching only at a corner -> one 8-component, two 4-components.
            bw[M - 24:M - 19, 8:13] = 1.0
            bw[M - 19:M - 14, 13:18] = 1.0
        gray = (bw > 0).astype(np.uint8) * 255
        out[k] = np.repeat(gray[:, :, None], 3, axis=2)
    return out
