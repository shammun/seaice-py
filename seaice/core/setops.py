"""Set and logical operations on binary and grayscale images — Book §2.6, Eqs. (2.16)–(2.30), Table 2.1.

Text-only section (no MATLAB code in ch2).  Binary images are sets ``G = {(x, y) | g(x, y) = 1}`` (Eq. 2.16);
grayscale images are 3-D sets ``{(x, y, z) | z = g(x, y)}`` (Eq. 2.27).  Reflection (2.25) and translation (2.26)
are the building blocks of the morphology in §4.2.
"""
from __future__ import annotations

import numpy as np


def _as_bool(A: np.ndarray) -> np.ndarray:
    return np.asarray(A) != 0


def complement(A: np.ndarray) -> np.ndarray:
    """``Aᶜ = {ω | ω ∉ A} = U - A`` (Eq. 2.21) — logical NOT (Table 2.1a)."""
    return ~_as_bool(A)


def union(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``A ∪ B = {ω | ω ∈ A or ω ∈ B}`` (Eq. 2.22) — logical OR (Table 2.1b)."""
    return _as_bool(A) | _as_bool(B)


def intersection(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``A ∩ B = {ω | ω ∈ A and ω ∈ B}`` (Eq. 2.23) — logical AND (Table 2.1c)."""
    return _as_bool(A) & _as_bool(B)


def difference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``A - B = {ω | ω ∈ A, ω ∉ B} = A ∩ Bᶜ`` (Eq. 2.24)."""
    return _as_bool(A) & ~_as_bool(B)


def reflect(A: np.ndarray, origin: tuple[int, int] | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflection ``Â = {ω | ω = -a, a ∈ A}`` about the set origin (Eq. 2.25) — a 180° rotation.

    Book: §2.6.1, Eq. (2.25), Fig. 2.17.  The image array is flipped in both axes; the origin ``(r0, c0)``
    (0-based array coordinates; default = array centre ``((M-1)//2, (N-1)//2)``) maps to
    ``(M-1-r0, N-1-c0)`` so that, relative to the returned origin, every element ``a`` becomes ``-a``.

    Returns ``(A_reflected, origin_reflected)``.
    """
    A = np.asarray(A)
    M, N = A.shape[:2]
    if origin is None:
        origin = ((M - 1) // 2, (N - 1) // 2)
    r0, c0 = origin
    return A[::-1, ::-1].copy(), (M - 1 - r0, N - 1 - c0)


def translate(A: np.ndarray, z: tuple[int, int], shape: tuple[int, int] | None = None,
              fill=0) -> np.ndarray:
    """Translation ``(A)_z = {c | c = a + z, a ∈ A}`` (Eq. 2.26): shift by ``z = (z1, z2)`` = (rows, cols).

    Book: §2.6.1, Eq. (2.26), Fig. 2.17.  Elements shifted outside the output rectangle (default: the input
    shape) are dropped; vacated cells get ``fill``.  ``shape`` enlarges the canvas (the input is placed at the
    top-left before shifting).
    """
    A = np.asarray(A)
    M, N = A.shape[:2]
    out_shape = (M, N) if shape is None else tuple(shape)
    out = np.full(out_shape[:2] + A.shape[2:], fill, dtype=A.dtype)
    z1, z2 = int(z[0]), int(z[1])
    # source rows r map to r + z1 ; keep those inside [0, out_M)
    r_src0, r_src1 = max(0, -z1), min(M, out_shape[0] - z1)
    c_src0, c_src1 = max(0, -z2), min(N, out_shape[1] - z2)
    if r_src1 > r_src0 and c_src1 > c_src0:
        out[r_src0 + z1:r_src1 + z1, c_src0 + z2:c_src1 + z2] = A[r_src0:r_src1, c_src0:c_src1]
    return out


def gray_complement(I: np.ndarray, L: int | None = None) -> np.ndarray:
    """Grayscale complement ``Aᶜ = {(x, y, L - z)}`` (Eq. 2.28), ``L = 2^k - 1`` (255 for uint8).

    ``L`` defaults to 255 for uint8, 65535 for uint16, 1.0 for float images.  Same result as ``imcomplement``.
    """
    I = np.asarray(I)
    if L is None:
        L = {np.dtype(np.uint8): 255, np.dtype(np.uint16): 65535}.get(I.dtype, 1.0)
    if np.issubdtype(I.dtype, np.integer):
        return (L - I.astype(np.int64)).astype(I.dtype)
    return L - I.astype(np.float64)


def gray_union(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Grayscale union = pixelwise maximum (Eq. 2.29)."""
    return np.maximum(np.asarray(A), np.asarray(B))


def gray_intersection(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Grayscale intersection = pixelwise minimum (Eq. 2.30)."""
    return np.minimum(np.asarray(A), np.asarray(B))


def bitwise_and(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Bitwise AND of integer images (Book §2.6.3, p. 29: ``57 AND 207 = 9``, i.e. ``00111001 & 11001111``)."""
    return np.bitwise_and(np.asarray(A), np.asarray(B))


def bitwise_or(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Bitwise OR of integer images (§2.6.3)."""
    return np.bitwise_or(np.asarray(A), np.asarray(B))


def bitwise_not(A: np.ndarray, nbits: int = 8) -> np.ndarray:
    """Bitwise NOT of an unsigned integer image on ``nbits`` bits (§2.6.3): ``(2^nbits - 1) - A``."""
    A = np.asarray(A)
    return ((1 << nbits) - 1 - A.astype(np.int64)).astype(A.dtype)


def truth_tables() -> dict[str, list[tuple]]:
    """Table 2.1: truth tables of NOT, OR, AND as lists of ``(inputs..., output)`` tuples."""
    return {
        "NOT": [(a, int(not a)) for a in (0, 1)],
        "OR": [(a, b, int(a or b)) for a in (0, 1) for b in (0, 1)],
        "AND": [(a, b, int(a and b)) for a in (0, 1) for b in (0, 1)],
    }
