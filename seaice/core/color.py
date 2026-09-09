"""Colour-space conversions of Book §2.1.2 (RGB, CMY, CMYK, HSI) and indexed images (§2.1.3).

MATLAB source: ``MATLAB_ROOT/ch2/color_image.m`` (CMY via ``imcomplement``, HSI written out explicitly);
CMYK Eqs. (2.4)–(2.5) and the indexed-image structure (Fig. 2.6) are text-only.

All functions accept uint8 ``(M, N, 3)`` or float ``(M, N, 3)`` arrays.  Outputs are float64 in the *normalised*
units of the book text (RGB in ``[0, 1]``) unless stated otherwise.
"""
from __future__ import annotations

import numpy as np

from .matlab_compat import im2double, imcomplement

SQRT6 = np.sqrt(6.0)


def split_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``R, G, B`` planes of an ``(M, N, 3)`` image (Eq. (2.2), ``C(x, y) = [R; G; B]``).

    Book: §2.1.2.1, Eq. (2.2), Fig. 2.3.  MATLAB source: ``color_image.m`` / ``histogram.m`` lines
    ``Ir = I(:, :, 1); Ig = I(:, :, 2); Ib = I(:, :, 3)``.  dtype is preserved (uint8 in, uint8 out).
    """
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (M, N, 3), got {rgb.shape}")
    return rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]


def rgb2cmy(rgb: np.ndarray) -> np.ndarray:
    """CMY components, Eq. (2.3): ``[C; M; Y] = [1; 1; 1] - [R; G; B]`` with RGB normalised to ``[0, 1]``.

    Book: §2.1.2.2, Eq. (2.3), Fig. 2.4.  MATLAB source: ``color_image.m`` line 10 ``I_cmy = imcomplement(I)``.

    Returns the same dtype as the input: uint8 → ``255 - I`` (identical to ``imcomplement``), float → ``1 - I``.
    Parity: exact.
    """
    return imcomplement(np.asarray(rgb))


def rgb2cmyk(rgb: np.ndarray, u: float = 1.0, b: float = 1.0) -> np.ndarray:
    """CMYK components, Eqs. (2.4)–(2.5) (text only — no MATLAB code in ch2).

    ``[C; M; Y; K] = [1; 1; 1; 0] - [R; G; B; 0] - Kb [u; u; u; -b]`` with ``Kb = min(1 - R, 1 - G, 1 - B)``
    (Eq. 2.5); ``u`` = under-colour-removal factor, ``b`` = darkness factor, both in ``[0, 1]``.

    Book: §2.1.2.2, Eqs. (2.4)–(2.5).  Returns float64 ``(M, N, 4)`` in ``[0, 1]`` units (input uint8 is scaled
    by 1/255).  Parity: reimplemented (MATLAB's ``rgb2cmyk``-like ICC conversions are not comparable).
    """
    if not (0.0 <= u <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError("u and b must lie in [0, 1] (Book §2.1.2.2)")
    f = im2double(np.asarray(rgb))
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    Kb = np.minimum.reduce([1.0 - R, 1.0 - G, 1.0 - B])  # Eq. (2.5)
    C = 1.0 - R - u * Kb  # Eq. (2.4), rows 1-3
    M = 1.0 - G - u * Kb
    Y = 1.0 - B - u * Kb
    K = b * Kb  # Eq. (2.4), row 4: 0 - 0 - Kb * (-b)
    return np.stack([C, M, Y, K], axis=-1)


def rgb2hsi(rgb: np.ndarray, use_atan2: bool = False, matlab_bug: bool = False,
            scale: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HSI components ``(H, S, I)`` from Eqs. (2.6a)–(2.6c).

    Eq. (2.6a)::

        [I; V1; V2] = [[1/3, 1/3, 1/3], [-1/√6, -1/√6, 2/√6], [1/√6, -1/√6, 0]] · [R; G; B]

    Eq. (2.6b) ``H = arctan(V2 / V1)``, Eq. (2.6c) ``S = sqrt(V1² + V2²)``.

    Book: §2.1.2.3, Fig. 2.5.  MATLAB source: ``MATLAB_ROOT/ch2/color_image.m`` lines 19–27.

    Parameters
    ----------
    rgb : ndarray (M, N, 3)
        uint8 or float.
    use_atan2 : bool, default False
        The book (and the MATLAB script) use ``atan(V2/V1)`` (range ±π/2, NaN where ``V1 = V2 = 0``).
        ``True`` uses ``atan2(V2, V1)`` for a proper hue in ``(-π, π]`` — a documented deviation.
    matlab_bug : bool, default False
        ``color_image.m`` line 21 reads ``V1 = -Ir/√6 - Ig/√6 + 2*Ig/√6`` (``Ig`` where Eq. 2.6a has ``Ib``), so
        ``V1 = (G - R)/√6 = -V2`` and ``H ≡ atan(-1) = -π/4`` (NaN where ``R == G``) — this is what the printed
        Fig. 2.5(a) shows.  ``True`` reproduces that line for figure comparison only.
    scale : float, optional
        Multiply the (already normalised) RGB by this factor before the transform.  ``None`` → RGB in ``[0, 1]``
        as the text states.  Use ``255`` to match the MATLAB script, which works on ``double(uint8)`` values
        without normalising (``H`` is scale-free; ``S`` and ``I`` scale linearly).

    Returns
    -------
    H, S, I : float64 arrays (M, N).  ``H`` in radians.

    Parity: ``I``, ``S`` exact vs MATLAB (after ×255); ``H`` reimplemented from Eq. (2.6a) (exact vs a corrected
    MATLAB snippet); ``matlab_bug=True`` reproduces the script's ``Ih`` (NaN pattern included).
    """
    f = im2double(np.asarray(rgb))
    if scale is not None:
        f = f * float(scale)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    I = (R + G + B) / 3.0  # Eq. (2.6a) row 1
    if matlab_bug:
        # PARITY: matlab_bug — reproduces color_image.m line 21 (2*Ig instead of 2*Ib) for Fig. 2.5(a) only.
        V1 = -R / SQRT6 - G / SQRT6 + 2.0 * G / SQRT6
    else:
        V1 = -R / SQRT6 - G / SQRT6 + 2.0 * B / SQRT6  # Eq. (2.6a) row 2
    V2 = R / SQRT6 - G / SQRT6  # Eq. (2.6a) row 3
    with np.errstate(divide="ignore", invalid="ignore"):
        if use_atan2:
            H = np.arctan2(V2, V1)
        else:
            H = np.arctan(V2 / V1)  # Eq. (2.6b): NaN where V1 == V2 == 0, ±π/2 where only V1 == 0
    S = np.sqrt(V1 ** 2 + V2 ** 2)  # Eq. (2.6c)
    return H, S, I


def indexed_to_rgb(idx: np.ndarray, cmap: np.ndarray, one_based: bool = False) -> np.ndarray:
    """Expand an indexed image (integer matrix + ``m × 3`` colormap in ``[0, 1]``) to an RGB float image.

    Book: §2.1.3, Fig. 2.6 (text only; MATLAB ``ind2rgb``).  ``one_based=True`` treats the indices as MATLAB
    1-based colormap rows (MATLAB clips out-of-range indices to the colormap range; so do we).
    """
    idx = np.asarray(idx).astype(np.int64)
    cmap = np.asarray(cmap, dtype=np.float64)
    if cmap.ndim != 2 or cmap.shape[1] != 3:
        raise ValueError("cmap must be m x 3")
    if one_based:
        idx = idx - 1
    idx = np.clip(idx, 0, cmap.shape[0] - 1)
    return cmap[idx]
