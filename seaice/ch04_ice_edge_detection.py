"""Chapter 4 — Ice Edge Detection: book-equation functions and the two script pipelines.

Book: Zhang & Skjetne (2018), Chapter 4, pp. 59–82.  The chapter's output is a binary edge map — from the first
derivative (§4.1.1 Sobel/Prewitt, Eqs. 4.1–4.8), from zero-crossings of the second derivative (§4.1.2 Laplacian /
LoG, Eqs. 4.9–4.15) or from morphological gradients of the ch03 ice mask or of the gray image (§4.2.4,
Eqs. 4.39–4.42).  Everything reusable (``edge``, ``fspecial``, ``strel``, ``imerode``/``imdilate``, opening /
closing, reconstruction, ``morphological_gradient``) lives in :mod:`seaice.core.edges`, :mod:`seaice.core.filters`
and :mod:`seaice.core.morphology`; this module holds the equation-level teaching forms and the pipelines of the
two MATLAB scripts.

MATLAB ↔ Python map (analysis/ch04.md §3):

* ``derivative.m`` → :func:`sobel_edges_script` (driver ``scripts/ch04_derivative.py``)
* ``morphology.m`` → :func:`morphological_edges` (driver ``scripts/ch04_morphology.py``)
* text-only Eqs. (4.1)–(4.8) → :func:`gradient_operator`, :func:`gradient_magnitude`, :func:`gradient_direction`,
  :func:`threshold_gradient`
* text-only Eqs. (4.9)–(4.13) + the p. 64 zero-crossing rule → :func:`laplacian`, :func:`laplacian_zero_crossings`
* text-only Eqs. (4.14)–(4.15) / Fig. 4.5 → :func:`gaussian_kernel`, :func:`log_kernel`
* Fig. 4.8 walk-through → :func:`fig_4_8_demo`; Figs. 4.11–4.14 1-D sketches → :func:`profile_open_close_demo`,
  :func:`profile_reconstruction_demo`
* §4.3 experiments (unshipped image) → :func:`experiment_sobel_thresholds`, :func:`experiment_internal_gradient`
  (driver ``scripts/ch04_experiments.py``)
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from .core import synth
from .core.edges import EdgeResult, edge
from .core.filters import conv2, fspecial, imfilter
from .core.matlab_compat import rgb2gray_matlab
from .core.morphology import imdilate, imerode, imclose, imopen, imreconstruct, morphological_gradient, \
    reconstruct_by_erosion, reconstruct_iterative, strel
from .core.threshold import graythresh, ice_concentration, im2bw

#: Parameters quoted in the chapter text (page numbers in analysis/ch04.md §6).
BOOK_PARAMS = {
    "sobel_T": 0.05,  # Fig. 4.3, 4.17(b), 4.18(b), 4.19(a) (pp. 63, 80, 81)
    "sobel_T_low": 0.03,  # Fig. 4.19(b) (p. 81)
    "log_sigma": 2.0,  # Fig. 4.6 (p. 66): 13×13 kernel
    "log_T": 0.005,  # Fig. 4.6
    "disk_r_figs": 15,  # Figs. 4.9, 4.10, 4.15, 4.16 (pp. 69–79; Fig. 4.15 caption "157" is a typo)
    "disk_r_script": 7,  # morphology.m line 6: strel('dis', 7)
    "disk_r_thin": 5,  # Figs. 4.17(c), 4.18(c), 4.20(a) (pp. 80–82)
    "disk_r_thick": 16,  # Fig. 4.20(b) (p. 82)
}

#: Fig. 4.3(a) = crop of ``test.jpg``: MATLAB ``im(1600:2151, 1979:2552)`` → 0-based rows 1599..2150, cols 1978..2551
#: (552×574), located by template matching (NCC 0.998, analysis/ch04.md header; verification may refine ±2 px).
FIG_4_3A_CROP = (slice(1599, 2151), slice(1978, 2552))

#: MATLAB 1-based, inclusive form of :data:`FIG_4_3A_CROP` for the reports.
FIG_4_3A_CROP_MATLAB = ((1600, 2151), (1979, 2552))


def fig_4_3a(im: np.ndarray) -> np.ndarray:
    """The two-floe region of Fig. 4.3(a): ``im[FIG_4_3A_CROP]`` (gray or RGB ``test.jpg``)."""
    im = np.asarray(im)
    if im.shape[0] < FIG_4_3A_CROP[0].stop or im.shape[1] < FIG_4_3A_CROP[1].stop:
        raise ValueError(f"fig_4_3a: image {im.shape[:2]} is smaller than the crop window (needs test.jpg, 2856×4290)")
    return im[FIG_4_3A_CROP[0], FIG_4_3A_CROP[1]]


def _as_gray(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    return rgb2gray_matlab(img) if img.ndim == 3 else img


# ---------------------------------------------------------------------------------------------------------------
# §4.1.1 Gradient operator — Eqs. (4.1)–(4.8), Figs. 4.1–4.2 (text only)
# ---------------------------------------------------------------------------------------------------------------

#: Fig. 4.1: 1-D running-difference kernels of Eqs. (4.6a) ``f(x+1,y) − f(x,y)`` (x = row) and (4.6b) (y = col),
#: written as correlation kernels centred on ``(x, y)``.
FORWARD_KERNELS = {
    "x": np.array([[0.0], [-1.0], [1.0]]),
    "y": np.array([[0.0, -1.0, 1.0]]),
}
#: Fig. 4.2(b): Sobel kernels as printed (Eq. 4.7a ``Gx``: rows below minus rows above; Eq. 4.7b ``Gy`` = transpose).
SOBEL_KERNELS = {
    "x": np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]),
    "y": np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]),
}
#: Fig. 4.2(c): Prewitt kernels as printed (Eq. 4.8a/b).
PREWITT_KERNELS = {
    "x": np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    "y": np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]),
}


def gradient_operator(f: np.ndarray, kind: str = "sobel", padding: str = "replicate") -> tuple[np.ndarray, np.ndarray]:
    """Gradient vector ``∇f = [Gx, Gy]ᵀ`` (Eq. 4.1) with the book's kernels — Eqs. (4.6) forward differences,
    (4.7) Sobel or (4.8) Prewitt, book convention ``x`` = row, ``y`` = column.

    Book: §4.1.1, Figs. 4.1–4.2.  Text only (no ``.m``); MATLAB's ``edge`` (:func:`seaice.core.edges.edge`) uses
    ``fspecial('sobel') = −(Fig. 4.2(b) left)`` scaled by ``1/8`` (Prewitt ``1/6``), so ``Gx_book = −8·gh`` and
    ``Gy_book = −8·gv`` in terms of ``edge``'s gradient outputs (``−6·`` for Prewitt); the sign is irrelevant for
    the magnitude (Eqs. 4.2/4.4/4.5).

    Parameters
    ----------
    f : ndarray (M, N)
        Gray image (any numeric dtype; computed in float64 without rescaling).
    kind : {'sobel', 'prewitt', 'forward'}
    padding : imfilter padding option (default ``'replicate'`` = what ``edge`` does; the text does not say).

    Returns ``(Gx, Gy)`` float64 — the kernels are applied as **correlation** (``imfilter``), exactly as the
    equations are written out (``f(x+1, ·)`` terms carry the positive weights).
    Parity: reimplemented (checked against ``edge(..., 'nothinning')`` after scaling and hand-coded MATLAB).
    """
    f = np.asarray(f, dtype=np.float64)
    kernels = {"sobel": SOBEL_KERNELS, "prewitt": PREWITT_KERNELS, "forward": FORWARD_KERNELS}[kind]
    # PARITY: reimplemented — book-equation form (unscaled, book sign, replicate padding by choice); MATLAB's edge
    # uses fspecial('sobel')/8 = −(this kernel)/8, so Gx_book = −8·gh, Gy_book = −8·gv exactly.
    Gx = imfilter(f, kernels["x"], padding)  # Eq. (4.7a) / (4.8a) / (4.6a)
    Gy = imfilter(f, kernels["y"], padding)  # Eq. (4.7b) / (4.8b) / (4.6b)
    return Gx, Gy


def gradient_magnitude(Gx: np.ndarray, Gy: np.ndarray, kind: str = "l2") -> np.ndarray:
    """Gradient magnitude — Eq. (4.2) ``sqrt(Gx² + Gy²)`` (``'l2'``), Eq. (4.4) ``Gx² + Gy²`` (``'squared'``, what
    MATLAB ``edge`` thresholds against ``T²``) or Eq. (4.5) ``|Gx| + |Gy|`` (``'l1'``)."""
    Gx = np.asarray(Gx, dtype=np.float64)
    Gy = np.asarray(Gy, dtype=np.float64)
    if kind == "l2":
        return np.sqrt(Gx * Gx + Gy * Gy)  # Eq. (4.2)
    if kind == "squared":
        return Gx * Gx + Gy * Gy  # Eq. (4.4)
    if kind == "l1":
        return np.abs(Gx) + np.abs(Gy)  # Eq. (4.5)
    raise ValueError("kind must be 'l2', 'squared' or 'l1'")


def gradient_direction(Gx: np.ndarray, Gy: np.ndarray, convention: str = "book") -> np.ndarray:
    """Gradient direction — Eq. (4.3) as printed, ``θ = arctan(Gx / Gy)`` (``convention='book'``, quadrant-correct
    via ``atan2(Gx, Gy)``), or the usual ``atan2(Gy, Gx)`` measured from the x (row) axis (``'math'``).

    The printed ratio ``Gx/Gy`` is the inverse of the textbook ``Gy/Gx`` (analysis/ch04.md risk 10); no figure of
    the chapter depends on it, so both readings are offered and the choice is explicit.  Radians in ``(−π, π]``.
    """
    Gx = np.asarray(Gx, dtype=np.float64)
    Gy = np.asarray(Gy, dtype=np.float64)
    # PARITY: reimplemented — Eq. (4.3) is printed as arctan(Gx/Gy) (row/col ratio); the quadrant-correct atan2 is a
    # choice the text does not make, and no figure or .m file uses the direction.
    if convention == "book":
        return np.arctan2(Gx, Gy)  # Eq. (4.3) literally
    if convention == "math":
        return np.arctan2(Gy, Gx)
    raise ValueError("convention must be 'book' or 'math'")


def threshold_gradient(mag: np.ndarray, T: float) -> np.ndarray:
    """Edge decision ``|∇f| > T`` (§4.1.1, p. 62) without MATLAB's thinning."""
    return np.asarray(mag, dtype=np.float64) > float(T)


# ---------------------------------------------------------------------------------------------------------------
# §4.1.2 Laplacian, Gaussian, LoG — Eqs. (4.9)–(4.15), Figs. 4.4–4.5 (text only)
# ---------------------------------------------------------------------------------------------------------------

#: Fig. 4.4(a): 4-neighbour Laplacian kernel of Eq. (4.13); Fig. 4.4(b): 8-neighbour variant.
LAPLACIAN_KERNELS = {
    4: np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]),
    8: np.array([[1.0, 1.0, 1.0], [1.0, -8.0, 1.0], [1.0, 1.0, 1.0]]),
}


def laplacian(f: np.ndarray, neighbors: int = 4, padding: str = "replicate") -> np.ndarray:
    """Discrete Laplacian ``∇²f`` — Eq. (4.9) with the second differences Eqs. (4.11)–(4.12), i.e. Eq. (4.13)
    ``f(x−1,y) + f(x+1,y) + f(x,y−1) + f(x,y+1) − 4 f(x,y)`` (``neighbors=4``, Fig. 4.4(a)) or the 8-neighbour
    kernel of Fig. 4.4(b).  Both kernels are symmetric, so correlation and convolution coincide.
    Parity: reimplemented (vs hand-coded MATLAB ``imfilter`` with the same kernels)."""
    f = np.asarray(f, dtype=np.float64)
    return imfilter(f, LAPLACIAN_KERNELS[int(neighbors)], padding)


def second_difference_forward(f: np.ndarray, axis: int = 0) -> np.ndarray:
    """Eq. (4.10): ``∂²f/∂x² ≈ f(x+2, y) − 2 f(x+1, y) + f(x, y)`` (differencing Eq. 4.6a; centred at ``x+1``), or the
    same along ``y`` (``axis=1``).  Replicate padding at the far border."""
    f = np.asarray(f, dtype=np.float64)
    k = np.array([[0.0], [0.0], [1.0], [-2.0], [1.0]]) if axis == 0 else np.array([[0.0, 0.0, 1.0, -2.0, 1.0]])
    return imfilter(f, k, "replicate")


def laplacian_zero_crossings(lap: np.ndarray, T: float) -> np.ndarray:
    """Zero-crossing edge rule of p. 64 [121]: in a 3×3 window take the maximum of the positive Laplacian responses
    and the minimum of the negative ones; an edge is present where both exist and ``max − min > T``.

    Book: §4.1.2 (text only; MATLAB's own rule is different — see :func:`seaice.core.edges.log_zero_crossings`).
    Border pixels use replicate padding of the response.  Parity: reimplemented (text).
    """
    lap = np.asarray(lap, dtype=np.float64)
    # PARITY: reimplemented — text-only rule (p. 64, [121]); MATLAB's edge('log') uses a different zero-crossing test
    # (core.edges.log_zero_crossings), so no reference exists for this form.
    pos = np.where(lap > 0, lap, 0.0)
    neg = np.where(lap < 0, lap, 0.0)
    max_pos = ndimage.maximum_filter(pos, size=3, mode="nearest")
    min_neg = ndimage.minimum_filter(neg, size=3, mode="nearest")
    return (max_pos > 0) & (min_neg < 0) & (max_pos - min_neg > float(T))


def gaussian_kernel(size: int = 5, sigma: float = 1.0, normalize: bool = True) -> np.ndarray:
    """Gaussian kernel of Eq. (4.14) ``G_σ(x, y) = exp(−(x²+y²)/(2σ²)) / (2πσ²)`` sampled on the ``size×size`` grid.

    Book: §4.1.2, Fig. 4.5(a) (5×5, σ = 1: 0.0030 … 0.1621).  ``normalize=True`` rescales to unit sum, which is
    what the printed figure shows (= ``fspecial('gaussian', 5, 1)`` up to MATLAB's ``eps`` clipping);
    ``normalize=False`` returns the analytic values (sum < 1 because of the truncation at 3σ, p. 65).
    """
    h = (size - 1) / 2.0
    y, x = np.mgrid[-h:h + 1, -h:h + 1]
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma)) / (2.0 * np.pi * sigma * sigma)  # Eq. (4.14)
    return g / g.sum() if normalize else g


def log_kernel(size: int = 5, sigma: float = 1.0, mode: str = "matlab") -> np.ndarray:
    """Laplacian-of-Gaussian kernel — Eq. (4.15)
    ``∇²G_σ = (x² + y² − 2σ²) / (2πσ⁶) · exp(−(x²+y²)/(2σ²))``.

    Book: §4.1.2, Fig. 4.5(b) (5×5, σ = 1: 0.0239, 0.0460, 0.0499, 0.0061, −0.0923, −0.3182).

    ``mode='matlab'`` (default; = the printed Fig. 4.5(b)): ``fspecial('log', size, sigma)`` — unit-sum Gaussian
    times ``(x²+y²−2σ²)/σ⁴``, then mean-subtracted so the kernel sums to zero.  ``mode='analytic'`` (alias
    ``'book'``): Eq. (4.15) sampled literally (carries ``1/(2πσ⁶)``; does not sum to zero).
    ``mode='gauss_conv_laplacian'``: the alternative mentioned on p. 65 [48] — ``conv2(gaussian, Fig. 4.4(a)
    Laplacian, 'same')`` on the normalised Gaussian.
    """
    if mode == "matlab":
        return fspecial("log", size, sigma)
    h = (size - 1) / 2.0
    y, x = np.mgrid[-h:h + 1, -h:h + 1]
    s2 = sigma * sigma
    if mode in ("analytic", "book"):
        return (x * x + y * y - 2.0 * s2) / (2.0 * np.pi * s2 ** 3) * np.exp(-(x * x + y * y) / (2.0 * s2))  # Eq. (4.15)
    if mode == "gauss_conv_laplacian":
        return conv2(gaussian_kernel(size, sigma, normalize=True), LAPLACIAN_KERNELS[4], "same")
    raise ValueError("mode must be 'matlab', 'analytic'/'book' or 'gauss_conv_laplacian'")


# ---------------------------------------------------------------------------------------------------------------
# derivative.m  (§4.1.1, Fig. 4.3(b); §4.3 Figs. 4.17(b)–4.19)
# ---------------------------------------------------------------------------------------------------------------


def bwareaopen(bw: np.ndarray, P: int, conn: int = 8) -> np.ndarray:
    """MATLAB ``bwareaopen(BW, P, conn)`` — remove connected components with fewer than ``P`` pixels (default 8-conn).

    Used by the commented line 8 of ``derivative.m`` (``bwareaopen(BW, 20)``).  Built on
    :func:`seaice.core.connectivity.label_components` (= ``bwlabel``), so the semantics are MATLAB's by construction
    (``skimage.morphology.remove_small_objects`` changed its threshold parameter in 0.26 and is not used).
    """
    from .core.connectivity import label_components

    bw = np.asarray(bw) != 0
    labels = label_components(bw, conn)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return sizes[labels] >= int(P)


def sobel_edges_script(gray: np.ndarray, T: float | None = 0.05, method: str = "sobel", sigma: float = 2.0,
                       median: bool = False, min_area: int | None = None, smooth: bool = False) -> dict[str, Any]:
    """Port of ``MATLAB_ROOT/ch4/derivative.m``: ``im = double(rgb2gray(im))/256; BW = edge(im, 'sobel', 0.05)``.

    Book: §4.1.1 Eqs. (4.2)/(4.4)/(4.7), Fig. 4.3(b) (Sobel, T = 0.05, on the Fig. 4.3(a) crop), Fig. 4.3(c)
    (``method='prewitt'``), Fig. 4.6 (``method='log'``, σ = 2, T = 0.005), Figs. 4.17(b)/4.18(b)/4.19 (T = 0.05 /
    0.03 on the unshipped §4.3 image).

    Note the literal ``/256`` (line 5) — **not** ``im2double`` (``/255``): gray levels map to ``[0, 0.996]`` and the
    book's ``T = 0.05`` refers to that scaling.  The three post-processing steps commented out in the script are
    reproduced behind flags, off by default: ``median`` = ``medfilt2(im, [3 3])`` *before* ``edge`` (zero-padded
    median), ``min_area`` = ``bwareaopen(BW, 20)`` (8-connected), ``smooth`` = ``conv2(double(BW), double(msk))``
    with the 5×5 ``msk`` holding a 3×3 block of ones — MATLAB's default ``'full'`` output, so the result grows to
    ``(r+4)×(c+4)`` and is no longer binary.

    Returns
    -------
    dict with ``im`` (the ``/256`` image), ``bw`` (bool), ``thresh`` (T used, automatic if ``T is None``), ``gv``,
    ``gh``, ``n_edge`` (edge-pixel count), ``msk``, ``r``, ``c`` (the script's unused variables) and, when
    requested, ``bw_smooth`` (float, ``'full'`` size).
    """
    gray = _as_gray(gray)
    im = gray.astype(np.float64) / 256.0  # line 5: im = double(im)/256
    if median:  # line 6 (commented): medfilt2(im, [3 3])
        # PARITY: exact (expected) — medfilt2 zero-pads; scipy's median_filter with mode='constant', cval=0 does the
        # same (scipy's default 'reflect' would not); checked by the verifier's BW_med reference.
        im = ndimage.median_filter(im, size=3, mode="constant", cval=0.0)
    res: EdgeResult = edge(im, method, T, sigma=sigma)  # line 7
    bw = res.bw
    if min_area is not None:  # line 8 (commented): bwareaopen(BW, 20)
        bw = bwareaopen(bw, int(min_area))
    r, c = bw.shape  # line 9
    msk = np.zeros((5, 5))  # lines 10–14
    msk[1:4, 1:4] = 1.0
    out = {"im": im, "bw": bw, "thresh": res.thresh, "gv": res.gv, "gh": res.gh, "n_edge": int(bw.sum()),
           "msk": msk, "r": r, "c": c, "method": method}
    if smooth:  # line 15 (commented): conv2(double(BW), double(msk)) — 'full' → (r+4)×(c+4), not binary
        out["bw_smooth"] = conv2(bw.astype(np.float64), msk, "full")
    return out


# ---------------------------------------------------------------------------------------------------------------
# morphology.m  (§4.2.1, §4.2.4; Figs. 4.9, 4.10, 4.15, 4.16)
# ---------------------------------------------------------------------------------------------------------------


def morphological_edges(gray: np.ndarray, se: np.ndarray | None = None, radius: int = 7) -> dict[str, Any]:
    """Port of ``MATLAB_ROOT/ch4/morphology.m`` — binary and grayscale erosion, dilation and the three
    morphological gradients with a disk structuring element.

    Book: §4.2.1 Eqs. (4.16)–(4.21) (Figs. 4.9–4.10), §4.2.4 Eqs. (4.39)–(4.42) (Figs. 4.15–4.16).  The script uses
    ``strel('dis', 7)`` on the full ``test.jpg``; the figures use r = 15 on the Fig. 4.3(a) crop (pass the crop
    and ``radius=15``).

    Script lines → keys of the returned dict (names = the MATLAB variables):

    * 3 ``im = rgb2gray(im)`` → ``im`` (uint8)
    * 6 ``SE = strel('dis', 7)`` → ``SE`` (bool 13×13, 157 px)
    * 10 ``I = im2bw(im, graythresh(im))`` → ``I`` (bool), plus ``level`` and ``ic``
    * 13 ``J = imerode(I, SE)``; 16 ``K = imdilate(I, SE)`` → ``J``, ``K`` (bool)
    * 14 ``BW1 = I − J`` (internal, Eq. 4.40); 17 ``BW2 = K − I`` (external, Eq. 4.41); 19 ``BW = K − J`` (basic,
      Eq. 4.39) → float64 0/1 (MATLAB: logical − logical = double)
    * 31 ``X = imerode(im, SE)``; 32 ``Y = imdilate(im, SE)`` → uint8
    * 34–36 ``internal = im − X``, ``external = Y − im``, ``basic = Y − X`` → uint8 (saturating subtraction; every
      difference is ≥ 0 so the values are exact)

    Parity: exact (all 12 arrays 0 px / 0 levels vs MATLAB; ``level`` identical).
    """
    im = _as_gray(gray)  # lines 2–3
    SE = strel("dis", radius) if se is None else (np.asarray(se) != 0)  # line 6: 'dis' → 'disk' (prefix match)
    level, _ = graythresh(im)
    I = im2bw(im, level)  # line 10
    J = imerode(I, SE)  # line 13
    BW1 = morphological_gradient(I, SE, "internal")  # line 14: I - J   (Eq. 4.40)
    K = imdilate(I, SE)  # line 16
    BW2 = morphological_gradient(I, SE, "external")  # line 17: K - I   (Eq. 4.41)
    BW = morphological_gradient(I, SE, "basic")  # line 19: K - J   (Eq. 4.39)
    X = imerode(im, SE)  # line 31
    Y = imdilate(im, SE)  # line 32
    internal = morphological_gradient(im, SE, "internal")  # line 34: im - X
    external = morphological_gradient(im, SE, "external")  # line 35: Y - im
    basic = morphological_gradient(im, SE, "basic")  # line 36: Y - X
    return {"im": im, "SE": SE, "level": level, "threshold": 255.0 * level, "ic": ice_concentration(I),
            "I": I, "J": J, "K": K, "BW1": BW1, "BW2": BW2, "BW": BW,
            "X": X, "Y": Y, "internal": internal, "external": external, "basic": basic}


def gradient_identity_check(res: dict[str, Any]) -> dict[str, bool]:
    """Eq. (4.42) ``ρ_int + ρ_ext = ρ`` on the binary and grayscale outputs of :func:`morphological_edges`."""
    return {
        "binary": bool(np.array_equal(res["BW1"] + res["BW2"], res["BW"])),
        "gray": bool(np.array_equal(res["internal"].astype(np.int64) + res["external"].astype(np.int64),
                                    res["basic"].astype(np.int64))),
    }


# ---------------------------------------------------------------------------------------------------------------
# Text-only demonstrations: Fig. 4.8 matrices, Figs. 4.11–4.14 1-D profiles
# ---------------------------------------------------------------------------------------------------------------


def fig_4_8_demo() -> dict[str, Any]:
    """Erosion and dilation of the printed Fig. 4.8(a) rectangle by the cross SE Fig. 4.8(b) (§4.2.1.1).

    Returns the input, SE, computed erosion/dilation and the printed results (d)/(f) with equality flags —
    ``imerode``/``imdilate`` must reproduce the book matrices exactly.
    """
    A, B = synth.FIG_4_8_IMAGE, synth.FIG_4_8_SE
    er, di = imerode(A, B), imdilate(A, B)
    return {"image": A, "se": B, "eroded": er, "dilated": di, "eroded_book": synth.FIG_4_8_ERODED,
            "dilated_book": synth.FIG_4_8_DILATED, "erosion_matches": bool(np.array_equal(er, synth.FIG_4_8_ERODED)),
            "dilation_matches": bool(np.array_equal(di, synth.FIG_4_8_DILATED))}


def profile_open_close_demo(length: int = 9) -> dict[str, np.ndarray]:
    """1-D grayscale closing (Eq. 4.22, Fig. 4.11 "push down from above") and opening (Eq. 4.23, Fig. 4.12 "push
    up from below") of the synthetic :func:`seaice.core.synth.two_floes_profile` with a flat line SE of ``length``
    pixels.  Returns ``f``, ``erosion``, ``dilation``, ``opening``, ``closing`` as 1-D arrays."""
    f = synth.two_floes_profile()
    B = strel("line", length, 0)
    return {"f": f[0], "erosion": imerode(f, B)[0], "dilation": imdilate(f, B)[0],
            "opening": imopen(f, B)[0], "closing": imclose(f, B)[0], "se_length": np.array(length)}


def profile_reconstruction_demo(h: int = 40) -> dict[str, Any]:
    """1-D grayscale reconstruction by dilation (Eqs. 4.31–4.34, Fig. 4.13: marker ``f − h`` under mask ``f`` — the
    h-maxima idea) and by erosion (Eqs. 4.35–4.38, Fig. 4.14: marker ``f + h`` over mask ``f``) on
    :func:`seaice.core.synth.two_floes_profile`; also runs the literal iteration Eq. (4.34)/(4.38) and returns the
    number of steps ``k`` until ``X_k = X_{k−1}``."""
    f = synth.two_floes_profile().astype(np.int32)
    B = strel("line", 3, 0)  # 1-D 3-pixel neighbourhood = the "B" of Eq. 4.25 in one dimension
    marker_d = np.maximum(f - h, 0)
    rec_d = imreconstruct(marker_d, f, B)
    it_d, k_d = reconstruct_iterative(marker_d, f, B, "dilation")
    marker_e = f + h
    rec_e = reconstruct_by_erosion(marker_e, f, B)
    it_e, k_e = reconstruct_iterative(marker_e, f, B, "erosion")
    return {"f": f[0], "marker_dilation": marker_d[0], "rec_dilation": rec_d[0], "k_dilation": k_d,
            "iterative_equals_dilation": bool(np.array_equal(it_d, rec_d)),
            "marker_erosion": marker_e[0], "rec_erosion": rec_e[0], "k_erosion": k_e,
            "iterative_equals_erosion": bool(np.array_equal(it_e, rec_e)), "h": h}


# ---------------------------------------------------------------------------------------------------------------
# §4.3 Experimental results (Figs. 4.17–4.20; the book image is not shipped)
# ---------------------------------------------------------------------------------------------------------------


def experiment_sobel_thresholds(gray: np.ndarray, thresholds: tuple[float, ...] = (0.05, 0.03)) -> dict[float, dict]:
    """§4.3 Fig. 4.19: Sobel edges (``derivative.m`` pipeline) at several thresholds.  Returns ``{T: result}`` with
    the :func:`sobel_edges_script` dicts."""
    return {T: sobel_edges_script(gray, T) for T in thresholds}


def experiment_internal_gradient(gray: np.ndarray, radii: tuple[int, ...] = (5, 16)) -> dict[int, dict]:
    """§4.3 Figs. 4.17(c)/4.18(c)/4.20: binary internal gradient (Eq. 4.40) of the Otsu ice mask with disks of
    ``radii`` (5 = thin closed edges that keep touching floes connected; 16 = thick edges that break weak
    connections but shrink the floes).  Returns ``{r: {'I', 'SE', 'BW1', 'level'}}``."""
    im = _as_gray(gray)
    level, _ = graythresh(im)
    I = im2bw(im, level)
    out = {}
    for r in radii:
        SE = strel("disk", r)
        BW1 = morphological_gradient(I, SE, "internal")
        out[r] = {"I": I, "SE": SE, "BW1": BW1, "level": level, "n_edge": int(BW1.sum())}
    return out
