"""Floe-size statistics — the two size definitions of Ch. 8 §8.3 (pp. 190–192).

* **Eq. (8.1)** ``L_i = sqrt(4 A_i / π)`` — the book calls it the "mean clipper diameter (MCD)".
* **Eq. (8.2)** ``N_c(L) = N(≥L) / N_total`` — the cumulative floe size distribution.

Both are needed again by ch9 (§9.3 monitors the *maximum* floe size over a video), so they live in
:mod:`seaice.core` rather than in the chapter module (analysis/ch08.md §6).
"""
from __future__ import annotations

import numpy as np

__all__ = ["mean_caliper_diameter", "cumulative_size_distribution"]


def mean_caliper_diameter(area, scale: float = 1.0) -> np.ndarray:
    r"""**Eq. (8.1)**, p. 190: :math:`L_i = \sqrt{4 A_i/\pi}` — the size measure of §8.3.

    MATLAB source: ``MATLAB_ROOT/ch8/MCD/main_WL_new.m`` lines 17–18 and 30–31::

        Poly_Area(i) = IceImage.Floe(i).Polygon.Area * length_over_Pixel^2;
        Poly_MCD(i)  = sqrt(Poly_Area(i)*4/pi);

    Parameters
    ----------
    area : array_like
        Piece areas.  In **pixels** if ``scale`` converts them, or already in m² with ``scale = 1``.
    scale : float
        Metres per pixel (``length_over_Pixel``; ``main_WL_new.m`` line 13 sets ``1.1794`` for the §8.3
        helicopter frame — the book never prints it, see p. 190 "IB *Oden*'s known length").  The area is
        multiplied by ``scale**2`` before the square root, exactly as the M-file does.

    Returns
    -------
    ndarray
        ``sqrt(area * scale**2 * 4 / pi)``, float64.

    Notes
    -----
    Naming (analysis/ch08.md finding 7): "mean clipper diameter" is a typo for *mean caliper diameter*
    (Rothrock & Thorndike 1984), and the quantity Eq. (8.1) actually defines is the **area-equivalent circle
    diameter**, not a caliper (support-function) diameter.  The acronym MCD is kept because the MATLAB folder,
    the variable names and the book's figures all use it.  Parity: **exact** (one expression).
    """
    a = np.asarray(area, dtype=np.float64)
    return np.sqrt(a * float(scale) ** 2 * 4.0 / np.pi)


def cumulative_size_distribution(sizes) -> tuple[np.ndarray, np.ndarray]:
    r"""**Eq. (8.2)**, p. 192: :math:`N_c(L)=N(\ge L)/N_{total}`, evaluated at every observed size.

    MATLAB source: ``MATLAB_ROOT/ch8/MCD/fitting_iceFloes_distribution.m`` lines 16–22::

        [sorted_floe_size, index] = sort(Raw_MCD);
        a_sorted_floe_size = Raw_MCD(index);        % identical to sorted_floe_size (dead code)
        MCD = a_sorted_floe_size;
        N_total = size(MCD, 2);
        for i = 1 : size(MCD, 2)
            N_L(i) = size(find(MCD >= MCD(i)), 2)/N_total;
        end

    Returns
    -------
    (L, Nc)
        ``L`` is ``sizes`` **sorted ascending** (MATLAB's ``sort`` is stable, so equal sizes keep their input
        order — irrelevant here because only the values are used) and ``Nc[i]`` is the fraction of pieces whose
        size is ``>= L[i]``.  ``Nc`` starts at 1 and ends at ``1/N``; **ties give repeated values** (all copies
        of a repeated size share the same count), which is why the log-log plot of Fig. 8.21 has vertical
        stacks of markers.

    Notes
    -----
    The text calls this "the number of ice floes per unit area with size no smaller than L", but the formula and
    the code are a plain fraction — there is no area normalisation (analysis/ch08.md §2.1).
    Parity: **exact**.  Implemented with a sorted search rather than the M-file's O(N²) loop; the two agree
    exactly because ``count(x >= L[i])`` is ``N - searchsorted(L, L[i], 'left')`` for sorted ``L``.
    """
    L = np.sort(np.asarray(sizes, dtype=np.float64).ravel())
    n = L.size
    if n == 0:
        return L, np.zeros(0, dtype=np.float64)
    Nc = (n - np.searchsorted(L, L, side="left")) / float(n)
    return L, Nc
