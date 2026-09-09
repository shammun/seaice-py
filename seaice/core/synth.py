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
