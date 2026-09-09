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
