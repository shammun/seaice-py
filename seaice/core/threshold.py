"""Histogram thresholding — Otsu (global, local/block, multilevel) and ice-concentration bookkeeping.

Book: Chapter 3 §3.1, Eqs. (3.1)–(3.28).  MATLAB sources: ``MATLAB_ROOT/ch3/Otsu.m`` (``graythresh``, ``im2bw``,
``multithresh``, ``imquantize``), ``MATLAB_ROOT/ch3/local_Otsu.m`` (block Otsu, identical to ``ch9/block_threshold.m``),
``MATLAB_ROOT/ch3/separability.m`` (η from the equations).  ``graythresh``, ``im2bw`` and ``multithresh`` are ported
line by line from the R2025a toolbox sources (``graythresh.m``, ``otsuthresh.m``, ``im2bw.m``, ``multithresh.m``,
``imquantize.m``) so that later chapters (ch4–ch9 all call ``im2bw(I, graythresh(I))``) get bit-identical masks.

Conventions
-----------
* Gray levels are ``i = 0..L-1`` (book) — the 1-based bin index ``k = i + 1`` of MATLAB's ``otsuthresh`` only shifts
  the means; σ_B² and σ_G² are shift-invariant, so the criterion is identical.
* A threshold ``t`` puts levels ``0..t`` in class C0 (dark, water) and ``t+1..L-1`` in C1 (bright, ice), i.e. the
  binary image is ``I > t`` (Eq. 3.1, strict ``>``; MATLAB ``im2bw`` uses ``I > 255*level``).
* Thresholds stay float64 end-to-end: tie-averaged Otsu thresholds such as ``107.5`` are legitimate (risk 6 of the
  chapter analysis) and ``255 * (i/255) == i`` holds in IEEE double for every ``i``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .histogram import imhist
from .matlab_compat import im2uint8, matlab_round, rgb2gray_matlab, to_uint8_saturating

# ---------------------------------------------------------------------------------------------------------------
# Otsu criterion from the book's equations (§3.1.1.1)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class OtsuCurves:
    """All §3.1.1.1 quantities as functions of the threshold ``t = 0..L-1`` (index = ``t``).

    Attributes
    ----------
    p : (L,) normalised histogram, Eq. (3.3).
    P0, P1 : (L,) class probabilities, Eqs. (3.4)–(3.5) (``P1`` summed directly over ``i > t``, exactly 0 for an
        empty C1).
    m : (L,) cumulative mean ``m(t)``, Eq. (3.8).
    mG : global mean, Eq. (3.9).
    m0, m1 : (L,) class means, Eqs. (3.6)–(3.7) (NaN where the class is empty).
    sigma0_2, sigma1_2 : (L,) class variances, Eqs. (3.12)–(3.13).
    sigma_W2 : (L,) within-class variance, Eq. (3.15).
    sigma_B2 : (L,) between-class variance, Eq. (3.16) last form (NaN where ``P0 (1 - P0) == 0``).
    sigma_G2 : global variance, Eq. (3.17).
    eta : (L,) separability ``σ_B²(t) / σ_G²``, Eq. (3.20).
    t_star : Otsu threshold, Eq. (3.21) — mean of the tied maximisers (float).
    eta_star : ``η(t*)``, Eq. (3.22).
    """

    p: np.ndarray
    P0: np.ndarray
    P1: np.ndarray
    m: np.ndarray
    mG: float
    m0: np.ndarray
    m1: np.ndarray
    sigma0_2: np.ndarray
    sigma1_2: np.ndarray
    sigma_W2: np.ndarray
    sigma_B2: np.ndarray
    sigma_G2: float
    eta: np.ndarray
    t_star: float
    eta_star: float


def otsu_criterion(counts: np.ndarray) -> OtsuCurves:
    """Evaluate Otsu's discriminant analysis for every threshold from a gray-level histogram.

    Book: §3.1.1.1, Eqs. (3.2)–(3.22) (levels ``i = 0..L-1``, class C0 = ``0..t``, C1 = ``t+1..L-1``).
    MATLAB source: ``MATLAB_ROOT/ch3/separability.m`` lines 20–40 (single ``t``; this function vectorises all ``t``),
    MATLAB ``otsuthresh`` (same σ_B² formula on 1-based bins).

    Parameters
    ----------
    counts : (L,) array
        Histogram ``n_i`` (Eq. 3.2), e.g. ``imhist(gray)[0]``.

    Returns
    -------
    OtsuCurves
        See the dataclass; ``t_star`` averages tied maximisers exactly like MATLAB ``otsuthresh``
        (``mean(find(sigma_b == max))``), so it may be a half-integer.  For a constant image every ``σ_B²`` is NaN:
        ``t_star = 0`` and ``eta_star = 0`` (MATLAB ``graythresh`` → ``level 0, em 0``); for a two-valued
        ``{0, L-1}`` image ``eta_star = 1`` (Eq. 3.22 bounds).

    Notes
    -----
    Identities Eq. (3.10) ``P0 m0 + P1 m1 = mG``, Eq. (3.11) ``P0 + P1 = 1`` and Eq. (3.18) ``σ_W² + σ_B² = σ_G²``
    hold to round-off wherever both classes are non-empty.  Parity: ``reimplemented`` from the equations (the
    Otsu threshold itself is checked ``exact`` against ``graythresh`` through :func:`graythresh`).
    """
    counts = np.asarray(counts, dtype=np.float64).ravel()
    L = counts.size
    n = counts.sum()  # Eq. (3.2)
    if n <= 0:
        raise ValueError("otsu_criterion: histogram is empty")
    i = np.arange(L, dtype=np.float64)
    p = counts / n  # Eq. (3.3)
    P0 = np.cumsum(p)  # Eq. (3.4)
    m = np.cumsum(i * p)  # Eq. (3.8)
    mG = float(m[-1])  # Eq. (3.9)
    sigma_G2 = float(np.sum((i - mG) ** 2 * p))  # Eq. (3.17)

    def tail_sum(v: np.ndarray) -> np.ndarray:
        """``Σ_{i>t} v_i`` for every ``t`` (reverse cumulative sum; exactly 0 where the tail is empty)."""
        out = np.zeros(L, dtype=np.float64)
        out[:-1] = np.cumsum(v[::-1])[::-1][1:]
        return out

    # C1 moments summed directly over i > t (Eqs. 3.5, 3.7, 3.13) instead of ``1 − P0`` / ``mG − m``: the
    # subtraction form leaves ~1e-15 residues for an empty C1 (finite junk instead of MATLAB's NaN) and cancels
    # digits for tiny classes.
    P1 = tail_sum(p)  # Eq. (3.5)
    m1_num = tail_sum(i * p)
    s2_1 = tail_sum(i ** 2 * p)
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = m / P0  # Eq. (3.6)
        m1 = m1_num / P1  # Eq. (3.7), NaN where C1 is empty
        # class variances, Eqs. (3.12)–(3.13): Σ_{i≤t} i² p_i / P0 − m0², etc.
        s2 = np.cumsum(i ** 2 * p)
        sigma0_2 = s2 / P0 - m0 ** 2
        sigma1_2 = s2_1 / P1 - m1 ** 2
        sigma_W2 = P0 * sigma0_2 + P1 * sigma1_2  # Eq. (3.15)
        # Eq. (3.16), last form — the exact expression of otsuthresh / separability.m, kept for the t* search
        sigma_B2_otsu = (mG * P0 - m) ** 2 / (P0 * (1.0 - P0))
        sigma_B2 = np.where(P1 == 0, np.nan, sigma_B2_otsu)  # NaN where a class is empty (P0 == 0 is already 0/0)
        eta = sigma_B2 / sigma_G2 if sigma_G2 > 0 else np.full(L, np.nan)  # Eq. (3.20)
    # Eq. (3.21): exhaustive search over 0 ≤ t < L-1 (otsuthresh loops k = 1..num_bins-1), NaN never wins
    cand = sigma_B2_otsu[: L - 1]
    finite = np.isfinite(cand)
    if finite.any():
        maxval = np.max(cand[finite])
        ties = np.flatnonzero(finite & (cand == maxval))
        t_star = float(ties.mean())
        eta_star = float(maxval / sigma_G2) if sigma_G2 > 0 else 0.0
    else:
        t_star, eta_star = 0.0, 0.0
    return OtsuCurves(p=p, P0=P0, P1=P1, m=m, mG=mG, m0=m0, m1=m1, sigma0_2=sigma0_2, sigma1_2=sigma1_2,
                      sigma_W2=sigma_W2, sigma_B2=sigma_B2, sigma_G2=sigma_G2, eta=eta, t_star=t_star,
                      eta_star=eta_star)


def separability(gray: np.ndarray, t: float) -> float:
    """Separability ``η(t) = σ_B²(t) / σ_G²`` of the two classes ``I ≤ t`` / ``I > t`` — Book Eq. (3.20), (3.22).

    Book convention (levels ``0..255``, class C0 = levels ``0..floor(t)``).  MATLAB source: the quantity
    ``separability.m`` computes, but that script evaluates it with a one-level offset (its ``C0 = bins 1..k`` =
    levels ``0..k-1``) — see :func:`seaice.ch03_ice_pixel_detection.separability_script` for the literal port.
    ``separability(gray, graythresh(gray)[0] * 255)`` equals ``graythresh``'s second output ``em``.

    Returns ``0.0`` for a constant image (``σ_G² = 0``, as MATLAB's ``em``) and ``0.0`` when one class is empty.
    """
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = rgb2gray_matlab(gray)
    counts, _ = imhist(im2uint8(gray) if gray.dtype != np.uint8 else gray, 256)
    cur = otsu_criterion(counts)
    idx = int(np.floor(t))
    if idx < 0 or idx > 254 or cur.sigma_G2 <= 0:
        return 0.0
    val = cur.sigma_B2[idx] / cur.sigma_G2
    return float(val) if np.isfinite(val) else 0.0


# ---------------------------------------------------------------------------------------------------------------
# MATLAB graythresh / otsuthresh / im2bw
# ---------------------------------------------------------------------------------------------------------------


def _im2uint8_any(I: np.ndarray) -> np.ndarray:
    """``im2uint8`` for every class ``graythresh`` accepts (uint8, uint16, logical, single/double in [0, 1])."""
    I = np.asarray(I)
    if I.dtype == np.uint16:  # MATLAB grayto8: (v + 128) / 257 in integer arithmetic == round(v / 257)
        return np.floor((I.astype(np.float64) + 128.0) / 257.0).astype(np.uint8)
    if I.dtype == np.int16:  # int16touint16 then /257
        return np.floor((I.astype(np.float64) + 32768.0 + 128.0) / 257.0).astype(np.uint8)
    return im2uint8(I)


def otsuthresh(counts: np.ndarray) -> tuple[float, float]:
    """MATLAB ``[t, em] = otsuthresh(counts)`` — Otsu's threshold from a histogram, level in ``[0, 1]``.

    Book: §3.1.1.1 Eqs. (3.16), (3.20)–(3.22).  MATLAB source: R2025a ``otsuthresh.m`` (called by ``graythresh``),
    ported line by line: cumulative ``omega``/``mu`` on 1-based bins, ``σ_B² = (mu_t ω − mu)² / (ω (1 − ω))`` for
    ``k = 1..num_bins-1``, strict ``>`` maximum with **tie averaging** (``idx = mean of tied k``),
    ``t = (idx − 1)/(num_bins − 1)``, ``em = maxval / (Σ p k² − mu_t²)`` (= η(t*)).  Constant histogram → ``(0, 0)``.

    Parity: exact (same arithmetic order; sums are of ≤ 256 terms).
    """
    counts = np.asarray(counts, dtype=np.float64).ravel()
    num_bins = counts.size
    num_elems = counts.sum()
    if num_elems <= 0:
        return 0.0, 0.0
    p = counts / num_elems
    omega = np.cumsum(p)
    k = np.arange(1, num_bins + 1, dtype=np.float64)
    mu = np.cumsum(p * k)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    cand = sigma_b_squared[: num_bins - 1]
    finite = np.isfinite(cand)
    if not finite.any():  # maxval stays -Inf in MATLAB
        return 0.0, 0.0
    maxval = np.max(cand[finite])
    ties = np.flatnonzero(finite & (cand == maxval)) + 1  # 1-based k
    idx = ties.sum() / ties.size
    t = (idx - 1.0) / (num_bins - 1.0)
    d = np.sum(p * k ** 2)
    em = float(maxval / (d - mu_t ** 2))
    return float(t), em


def graythresh(I: np.ndarray) -> tuple[float, float]:
    """MATLAB ``[level, em] = graythresh(I)`` — global Otsu threshold (normalised level) and its effectiveness.

    Book: §3.1.1.1, Eqs. (3.20)–(3.22), Fig. 3.3; used by ``MATLAB_ROOT/ch3/Otsu.m`` line 9 and
    ``local_Otsu.m`` line 18 (and by ch4–ch9 wherever ``im2bw(I, graythresh(I))`` appears).
    MATLAB source: R2025a ``graythresh.m`` = ``im2uint8(I(:))`` → ``imhist(I, 256)`` → :func:`otsuthresh`.

    Parameters
    ----------
    I : ndarray
        uint8 / uint16 / int16 / logical / float image (floats are expected in ``[0, 1]`` and are converted with
        ``im2uint8``, i.e. ``round(255 x)`` clipped).  RGB input is **not** converted (MATLAB would histogram the
        three planes together) — pass ``rgb2gray_matlab(rgb)``.

    Returns
    -------
    level : float in [0, 1]
        ``t* / 255``; the gray-level threshold is ``255 * level`` (may be a half-integer on ties).
    em : float in [0, 1]
        Effectiveness metric = book's separability ``η(t*)`` (Eq. 3.22); 0 for a constant or empty image.

    Parity: exact vs MATLAB R2025a (chapter 3 verification).
    """
    I = np.asarray(I)
    if I.size == 0:
        return 0.0, 0.0
    u8 = _im2uint8_any(I.ravel())
    counts, _ = imhist(u8, 256)
    return otsuthresh(counts)


def im2bw(I: np.ndarray, level: float = 0.5) -> np.ndarray:
    """MATLAB ``BW = im2bw(I, level)`` — binarise by the normalised luminance ``level`` (Book Eq. 3.1).

    MATLAB source: R2025a ``im2bw.m``: RGB → ``rgb2gray``; integer classes → ``I > intmax * level`` (uint8: ``255 *
    level``, uint16: ``65535 * level``); logical → returned unchanged (with a warning); float → ``I > level``.
    Used by ``Otsu.m`` line 10 and ``local_Otsu.m`` line 35.  ``level`` must lie in ``[0, 1]`` (MATLAB errors).

    Note the strict ``>``: with ``level = t*/255`` pixels equal to ``t*`` are water/background.  Parity: exact.
    """
    I = np.asarray(I)
    if not (0.0 <= float(level) <= 1.0):
        raise ValueError("im2bw: level must be in [0, 1]")
    level = float(level)
    if I.ndim == 3:
        I = rgb2gray_matlab(I)
    if I.dtype == np.bool_:
        warnings.warn("im2bw: input is already binary (MATLAB images:im2bw:binaryInput)", RuntimeWarning)
        return I.copy()
    if I.dtype == np.uint8:
        return I > 255.0 * level
    if I.dtype == np.uint16:
        return I > 65535.0 * level
    if I.dtype == np.int16:
        return (I.astype(np.float64) + 32768.0) > 65535.0 * level
    return I.astype(np.float64) > level


# ---------------------------------------------------------------------------------------------------------------
# MATLAB multithresh / imquantize (§3.1.3)
# ---------------------------------------------------------------------------------------------------------------


def _multithresh_pdf(A: np.ndarray) -> tuple[np.ndarray | None, float, float]:
    """``getpdf`` of R2025a ``multithresh.m``: normalise to ``[minA, maxA]`` → ``grayto8`` → 256-bin pdf.

    Integer classes: ``single(A - minA) / single(maxA - minA)`` (single precision!) and ``grayto8`` then forms
    ``x * 255`` **in single** too (so e.g. 212.4999949 rounds to the float32 212.5 → bin 213); floats: double
    arithmetic throughout.  NaNs dropped.  Returns ``(p, minA, maxA)`` with ``p = None`` when the image is
    constant / empty.  Parity: exact (Eq. 3.28 metric ≤ 1e-12 vs MATLAB on the shipped images).
    """
    A = np.asarray(A).ravel()
    if np.issubdtype(A.dtype, np.floating):
        A = A[~np.isnan(A)]
        if A.size == 0:
            return None, np.nan, np.nan
        finite = np.isfinite(A)
        if not finite.any():
            return None, float(A.min()), float(A.max())
        minA, maxA = float(A[finite].min()), float(A[finite].max())
        if minA == maxA:
            return None, minA, maxA
        x = (A.astype(np.float64) - minA) / (maxA - minA)
    else:
        minA, maxA = float(A.min()), float(A.max())
        if minA == maxA:
            return None, minA, maxA
        # MATLAB normalises integer images in *single* precision; float32 reproduces it bit for bit.
        x = (np.float32(A.astype(np.float64) - minA)) / np.float32(maxA - minA)
        x = x.astype(np.float32)
    # grayto8: uint8(x * 255) with the product in the class of x (single for integer inputs, double for floats),
    # then MATLAB rounding (half away from zero) and saturation (values outside [0, 1] clipped; ±Inf → 255 / 0)
    y = (x * x.dtype.type(255)).astype(x.dtype)
    u8 = np.clip(np.floor(y.astype(np.float64) + 0.5), 0, 255).astype(np.uint8)
    counts, _ = imhist(u8, 256)
    p = counts.astype(np.float64) / counts.sum()
    return p, minA, maxA


def _degenerate_thresholds(unique_vals: np.ndarray, N: int) -> np.ndarray:
    """``getDegenerateThresholds`` of ``multithresh.m`` (fewer distinct values than ``N + 1``)."""
    u = np.asarray(unique_vals, dtype=np.float64).ravel()
    if u.size == 0:
        return np.arange(1, N + 1, dtype=np.float64)
    need1 = N - u.size
    if need1 <= 0:
        return u.copy()
    if u[0] > 1:
        lead = np.arange(1, min(need1, int(np.ceil(u[0])) - 1) + 1, dtype=np.float64)
        th = np.concatenate([lead, u])
    else:
        th = u.copy()
    need2 = N - th.size
    if need2 > 0:
        extra = []
        cand = max(np.floor(u[0]), 0.0)
        while len(extra) < need2:
            cand += 1.0
            if np.any(np.abs(u - cand) < np.spacing(cand)):
                continue
            extra.append(cand)
        th = np.sort(np.concatenate([th, extra]))
    return th


def _map_to_original_scale(thresh: np.ndarray, minA: float, maxA: float, dtype: np.dtype) -> np.ndarray:
    """``map2OriginalScale``: ``minA + thresh/255 (maxA − minA)`` cast back to the input class (MATLAB rounding)."""
    scl = minA + np.asarray(thresh, dtype=np.float64) / 255.0 * (maxA - minA)
    if dtype == np.uint8:
        return to_uint8_saturating(scl)
    if dtype == np.uint16:
        return np.clip(matlab_round(scl), 0, 65535).astype(np.uint16)
    if dtype == np.int16:
        return np.clip(matlab_round(scl), -32768, 32767).astype(np.int16)
    if dtype == np.float32:
        return scl.astype(np.float32)
    return scl


def _sigma_b2_matrix_n2(omega: np.ndarray, mu: np.ndarray, mu_t: float) -> np.ndarray:
    """``calcFullObjCriteriaMatrix`` for ``N = 2``: σ_B²(t1, t2) for all pairs (NaN where ``t1 >= t2``)."""
    nb = omega.size
    with np.errstate(divide="ignore", invalid="ignore"):
        omega0 = np.repeat(omega[:, None], nb, axis=1)  # ω(i)
        mu_0_t = np.repeat((mu_t - mu / omega)[:, None], nb, axis=1)
        omega1 = omega[None, :] - omega[:, None]  # ω(j) − ω(i)
        mu_1_t = mu_t - (mu[None, :] - mu[:, None]) / omega1
        R, C = np.mgrid[1:nb + 1, 1:nb + 1]
        pix_nan = R >= C
        omega0 = omega0.copy()
        omega1 = omega1.copy()
        omega0[pix_nan] = np.nan
        omega1[pix_nan] = np.nan
        term1 = omega0 * mu_0_t ** 2
        term2 = omega1 * mu_1_t ** 2
        omega2 = 1.0 - (omega0 + omega1)
        omega2[omega2 <= 0] = np.nan
        term3 = (omega0 * mu_0_t + omega1 * mu_1_t) ** 2 / omega2
        return term1 + term2 + term3


def _sigma_b2_exhaustive_n3(omega: np.ndarray, mu: np.ndarray, mu_t: float) -> tuple[float, np.ndarray]:
    """Exhaustive Eq. (3.27) for three thresholds (``0 < t1 < t2 < t3 < L-1``) — vectorised over ``t3``.

    Returns ``(maxval, thresholds)`` with tied maximisers averaged like the N ≤ 2 branches.
    """
    nb = omega.size
    best = -np.inf
    tied: list[np.ndarray] = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(2, nb - 2):  # 1-based bin index of the last bin in class 1 (0 < t1, Eq. 3.27)
            w0 = omega[i - 1]
            if w0 <= 0:
                continue
            term1 = w0 * (mu[i - 1] / w0 - mu_t) ** 2
            j = np.arange(i + 1, nb - 1)  # class 2 ends at bin j
            w1 = omega[j - 1] - w0
            m1 = (mu[j - 1] - mu[i - 1]) / w1
            term2 = w1 * (m1 - mu_t) ** 2
            # class 3 ends at bin k, class 4 is the rest
            k = np.arange(i + 2, nb)
            w2 = omega[k - 1][None, :] - omega[j - 1][:, None]
            m2 = (mu[k - 1][None, :] - mu[j - 1][:, None]) / w2
            term3 = w2 * (m2 - mu_t) ** 2
            w3 = 1.0 - omega[k - 1][None, :]
            m3 = (mu_t - mu[k - 1][None, :]) / w3
            term4 = w3 * (m3 - mu_t) ** 2
            val = term1 + term2[:, None] + term3 + term4
            val[k[None, :] <= j[:, None]] = np.nan
            val[~np.isfinite(val)] = np.nan
            if np.all(np.isnan(val)):
                continue
            vmax = np.nanmax(val)
            if vmax > best:
                best = vmax
                tied = []
            if vmax == best:
                jj, kk = np.nonzero(val == vmax)
                tied.append(np.column_stack([np.full(jj.size, i), j[jj], k[kk]]).astype(np.float64))
    if not np.isfinite(best):
        return best, np.array([np.nan, np.nan, np.nan])
    allt = np.vstack(tied)
    return float(best), allt.mean(axis=0) - 1.0


def multithresh(A: np.ndarray, N: int = 1) -> tuple[np.ndarray, float]:
    """MATLAB ``[thresh, metric] = multithresh(A, N)`` — ``N`` Otsu thresholds by maximising Eq. (3.24).

    Book: §3.1.3, Eqs. (3.23)–(3.28), Fig. 3.5 (N = 2 → 3 classes), Fig. 3.12(a) / Tables 3.2–3.3.
    MATLAB source: ``MATLAB_ROOT/ch3/Otsu.m`` line 31 (``multithresh(I, 2)``); ported from R2025a ``multithresh.m``:

    1. ``getpdf``: normalise ``A`` to ``[minA, maxA]`` (integer classes in **single** precision), ``grayto8``
       (``round(255 x)``), 256-bin pdf ``p``; constant image → degenerate thresholds and ``metric = 0``.
    2. ``omega = cumsum(p)``, ``mu = cumsum(p .* (1:256)')``.
    3. ``N = 1``: σ_B² vector, tie-averaged index; ``N = 2``: full 256×256 σ_B²(t1, t2) matrix with NaN masking
       (``t1 < t2``), ties averaged per coordinate (``mean([maxR maxC], 1) - 1``).
    4. ``map2OriginalScale``: ``minA + t/255 (maxA − minA)`` cast to the input class (uint8 → rounded, saturated).
    5. ``metric = maxval / Σ p (k − mu_t)²`` (Eq. 3.28).

    Parameters
    ----------
    A : ndarray
        uint8 / uint16 / int16 / float image (any shape; NaNs ignored for floats).
    N : int
        Number of thresholds (1 ≤ N ≤ 3 here; MATLAB allows ≤ 20).

    Returns
    -------
    thresh : (N,) array in the input class/units (uint8 for uint8 input, like MATLAB)
    metric : float, Eq. (3.28) separability of the N + 1 classes

    Deviation from MATLAB
    ---------------------
    * ``N = 3``: MATLAB runs ``fminsearch`` (Nelder–Mead, ``TolX = 1``, rounded) from equally spaced starts — a
      *local* search.  Here Eq. (3.27) is maximised **exhaustively**, so the result can legitimately be better
      than MATLAB's (parity ``reimplemented``).  ``N > 3`` raises ``NotImplementedError``
      (use ``skimage.filters.threshold_multiotsu`` for an approximation).
    * The degenerate-input branches (fewer distinct values than ``N + 1``) reproduce ``getDegenerateThresholds``
      and emit a ``RuntimeWarning`` instead of MATLAB's ``images:multithresh:degenerateInput`` warning.

    Parity: exact for N ≤ 2 (uint8 inputs), near if a single-precision bin edge differs.
    """
    A = np.asarray(A)
    N = int(N)
    if N < 1:
        raise ValueError("multithresh: N must be a positive integer")
    if A.size == 0:
        warnings.warn("multithresh: degenerate (empty) input", RuntimeWarning)
        return _degenerate_thresholds(np.array([]), N).astype(A.dtype, copy=False), 0.0
    p, minA, maxA = _multithresh_pdf(A)
    if p is None:  # constant image / no finite values
        warnings.warn(f"multithresh: degenerate input, fewer than {N + 1} distinct values", RuntimeWarning)
        if N == 1:
            th = np.array([minA])
        elif minA == maxA:
            th = _degenerate_thresholds(np.array([minA]), N)
        else:
            th = _degenerate_thresholds(np.array([minA, maxA]), N)
        return _cast_like(th, A.dtype), 0.0

    num_bins = 256
    omega = np.cumsum(p)
    k = np.arange(1, num_bins + 1, dtype=np.float64)
    mu = np.cumsum(p * k)
    mu_t = mu[-1]

    if N == 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            sb = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
        finite = np.isfinite(sb)
        valid = finite.any()
        if valid:
            maxval = float(np.max(sb[finite]))
            idx = np.flatnonzero(finite & (sb == maxval)) + 1
            thresh_raw = np.array([idx.mean() - 1.0])
    elif N == 2:
        sb = _sigma_b2_matrix_n2(omega, mu, mu_t)
        finite = np.isfinite(sb)
        valid = finite.any()
        if valid:
            maxval = float(np.max(sb[finite]))
            maxR, maxC = np.nonzero(finite & (sb == maxval))
            # MATLAB find() is column-major, but the per-coordinate mean is order-independent
            thresh_raw = np.array([maxR.mean() + 1.0, maxC.mean() + 1.0]) - 1.0
    elif N == 3:
        # PARITY: reimplemented — exhaustive Eq. (3.27) instead of MATLAB's fminsearch local search.
        maxval, thresh_raw = _sigma_b2_exhaustive_n3(omega, mu, mu_t)
        valid = np.isfinite(maxval)
    else:
        raise NotImplementedError(
            f"multithresh: N = {N} > 3 is not ported (MATLAB uses fminsearch); "
            "use skimage.filters.threshold_multiotsu(A, classes=N + 1) as an approximation."
        )

    if valid:
        thresh = _map_to_original_scale(thresh_raw, minA, maxA, A.dtype)
        metric = float(maxval / np.sum(p * (k - mu_t) ** 2))
        return thresh, metric
    uniq = np.unique(A[np.isfinite(A)] if np.issubdtype(A.dtype, np.floating) else A)
    warnings.warn(f"multithresh: degenerate input, only {uniq.size} distinct values for N = {N}", RuntimeWarning)
    return _cast_like(_degenerate_thresholds(uniq, N), A.dtype), 0.0


def _cast_like(th: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype == np.uint8:
        return to_uint8_saturating(th)
    if dtype == np.uint16:
        return np.clip(matlab_round(th), 0, 65535).astype(np.uint16)
    if dtype == np.int16:
        return np.clip(matlab_round(th), -32768, 32767).astype(np.int16)
    return np.asarray(th, dtype=np.float64)


def imquantize(A: np.ndarray, levels: np.ndarray | float, values: np.ndarray | None = None) -> np.ndarray:
    """MATLAB ``imquantize(A, levels[, values])`` — multilevel quantisation, Book Eq. (3.23).

    ``index = 1 + Σ_i (A > levels_i)`` (classes ``1..N+1``, strictly increasing ``levels``); with ``values`` the
    class index is replaced by ``values[index - 1]``.  MATLAB source: ``Otsu.m`` line 32 (``seg = imquantize(I,
    thresh)``), R2025a ``imquantize.m``.  Comparison is done in float64 so uint8 thresholds and float images mix
    as in MATLAB.  Parity: exact.
    """
    A = np.asarray(A)
    levels = np.atleast_1d(np.asarray(levels, dtype=np.float64))
    if levels.ndim != 1 or np.any(np.diff(levels) <= 0):
        raise ValueError("imquantize: levels must be a strictly increasing vector")
    x = A.astype(np.float64)
    index = np.ones(A.shape, dtype=np.int64)
    for lv in levels:
        index += (x > lv)
    if values is None:
        return index
    values = np.asarray(values)
    if values.size != levels.size + 1:
        raise ValueError("imquantize: numel(values) must equal numel(levels) + 1")
    return values[index - 1]


# ---------------------------------------------------------------------------------------------------------------
# Block (local) Otsu — local_Otsu.m / ch9 block_threshold.m (§3.1.2)
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockOtsu:
    """Result of :func:`block_otsu` (block index ``b = i * n_c + j`` in row-major order, like the script's
    ``(i-1)*n_c + j``).

    Attributes
    ----------
    thresholds : (n_r*n_c,) gray-level thresholds ``th = 255 * graythresh(block)`` (float, may be half-integer)
    levels : (n_r*n_c,) normalised levels
    counts : (n_r*n_c,) number of ice pixels ``block > th`` (script ``num``)
    ic_local : (n_r*n_c,) per-block ice concentration ``num / (r_t c_t)``
    ic : overall ice concentration ``Σ num / (r c)`` (script ``IC``)
    bw : (r, c) bool — the block-wise binary image assembled from the ``im2bw(temp, t)`` tiles
    slices : list of ``(slice_rows, slice_cols)`` per block (0-based)
    """

    thresholds: np.ndarray
    levels: np.ndarray
    counts: np.ndarray
    ic_local: np.ndarray
    ic: float
    bw: np.ndarray
    slices: list[tuple[slice, slice]]


def block_otsu(gray: np.ndarray, n_r: int = 2, n_c: int = 3) -> BlockOtsu:
    """Local Otsu thresholding on an ``n_r × n_c`` grid of equal blocks — Book §3.1.2, Fig. 3.4(c).

    MATLAB source: ``MATLAB_ROOT/ch3/local_Otsu.m`` lines 6–36 and 50 (identical algorithm in
    ``MATLAB_ROOT/ch9/block_threshold.m``): ``c_r = r/n_r``, ``t1 = (0:n_r-1)*c_r + 1``, ``t2 = (1:n_r)*c_r`` (same
    for columns), per block ``t = graythresh(temp)``, ``th = t*255``, ``n = #(temp > th)``, ``IC_local = n/(r_t c_t)``,
    finally ``IC = Σ num / (r c)``.

    ``r`` must be divisible by ``n_r`` and ``c`` by ``n_c`` — MATLAB's non-integer block indices raise an error;
    so does this port (``ValueError``).  Parity: exact.
    """
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = rgb2gray_matlab(gray)
    r, c = gray.shape
    if r % n_r or c % n_c:
        raise ValueError(f"block_otsu: image {r}x{c} is not divisible into {n_r}x{n_c} equal blocks "
                         "(MATLAB errors on the non-integer indices)")
    c_r, c_c = r // n_r, c // n_c
    nb = n_r * n_c
    thresholds = np.zeros(nb)
    levels = np.zeros(nb)
    counts = np.zeros(nb, dtype=np.int64)
    ic_local = np.zeros(nb)
    bw = np.zeros((r, c), dtype=bool)
    slices: list[tuple[slice, slice]] = []
    for i in range(n_r):
        for j in range(n_c):
            b = i * n_c + j
            sl = (slice(i * c_r, (i + 1) * c_r), slice(j * c_c, (j + 1) * c_c))
            temp = gray[sl]
            t, _ = graythresh(temp)
            th = t * 255.0
            thresholds[b], levels[b] = th, t
            n = int(np.count_nonzero(temp.astype(np.float64) > th))
            counts[b] = n
            ic_local[b] = n / temp.size
            bw[sl] = im2bw(temp, t)
            slices.append(sl)
    return BlockOtsu(thresholds=thresholds, levels=levels, counts=counts, ic_local=ic_local,
                     ic=float(counts.sum() / (r * c)), bw=bw, slices=slices)


# ---------------------------------------------------------------------------------------------------------------
# Ice-concentration bookkeeping (Tables 3.1–3.2)
# ---------------------------------------------------------------------------------------------------------------


def ice_concentration(mask: np.ndarray) -> float:
    """Ice concentration ``IC = #ice pixels / (M N)`` of a binary ice mask (Book Ch. 1 definition; ``Otsu.m``
    lines 13–14 ``ic = length(find(bw == 1)) / (r*c)``).  Returns a fraction in ``[0, 1]``."""
    mask = np.asarray(mask)
    return float(np.count_nonzero(mask) / mask.size)


def class_coverage(labels: np.ndarray, k: int) -> np.ndarray:
    """Fraction of pixels in each class ``1..k`` of a label image (``Otsu.m`` line 37 ``coverage(i)``,
    ``kmeans.m`` lines 81/88 ``n(i)``, ``ic = n / (M N)``; Table 3.2).  Returns ``(k,)`` fractions."""
    labels = np.asarray(labels)
    return np.array([np.count_nonzero(labels == i) / labels.size for i in range(1, k + 1)], dtype=np.float64)


def class_mean_intensity(gray: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Mean gray level of each class ``1..k`` (Table 3.3 "average intensity"; the *correct* form of ``Otsu.m``
    lines 39–42 / ``kmeans.m`` lines 76–79, whose ``s``/``ss`` buffers are never cleared — see
    :func:`seaice.ch03_ice_pixel_detection.stale_mean_intensity`).  Empty classes give NaN."""
    g = np.asarray(gray).astype(np.float64)
    labels = np.asarray(labels)
    out = np.full(k, np.nan)
    for i in range(1, k + 1):
        sel = labels == i
        if sel.any():
            out[i - 1] = g[sel].sum() / sel.sum()
    return out
