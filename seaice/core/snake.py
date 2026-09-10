"""The Xu & Prince GVF-snake toolbox: gradient vector flow, the pentadiagonal snake solver and its helpers.

Book: Chapter 6 "GVF Snake-Based Ice Floe Boundary Identification and Ice Image Segmentation" —
§6.1.2 (the discretisation, Eqs. 6.31–6.40), §6.2 (gradient vector flow, Eqs. 6.41–6.56),
Eq. (6.10) (Gaussian smoothing) and Eq. (6.46) (the natural boundary condition ``∇u·dσ = 0``).

MATLAB sources (all in ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/``, byte-identical copies in
``ch6/for test/``, ``ch7/Sea_Ice_Floe_Identification/`` and ``ch9/Model_Ice_Floe_Identification/``):
``GVF.m``, ``BoundMirrorExpand.m``, ``BoundMirrorEnsure.m``, ``BoundMirrorShrink.m``, ``gradient2.m``,
``xconv2.m``, ``gaussianMask.m``, ``gaussianBlur.m``, ``snakedeform.m``, ``snakeinterp.m``, ``snakeindex.m``.

Attribution
-----------
``GVF.m``, ``BoundMirror*.m``, ``snakedeform.m``, ``snakeinterp.m``, ``snakeindex.m``, ``snakedisp.m``,
``xconv2.m``, ``gaussianMask.m`` and ``gaussianBlur.m`` are part of the **GVF snake toolbox freely distributed
for academic use by Chenyang Xu and Jerry L. Prince**, Image Analysis and Communications Laboratory, Johns
Hopkins University — ``http://iacl.ece.jhu.edu/projects/gvf`` (headers "Chenyang Xu and Jerry L. Prince,
4/1/95, 6/17/97, 9/9/1999; Copyright (c) 1995-99").  Reference: C. Xu and J. L. Prince, "Snakes, Shapes, and
Gradient Vector Flow", *IEEE Transactions on Image Processing* **7**(3):359–369, 1998 (book ref. [174]; the
book's own footnote 1 on p. 110 points at the same URL).  The functions below are a Python re-implementation of
that toolbox written from the published equations and the shipped M-code; every function names its origin.

``gradient2.m`` is a renamed copy of the MathWorks' MATLAB-4 ``gradient.m`` (the authors kept it to dodge the
MATLAB-5 signature change).  It is **not** redistributed here: :func:`gradient2` implements the same
"MATLAB-4 ``gradient`` semantics", which for unit spacing are identical to modern ``gradient`` / ``np.gradient``.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .interp import interp2
from .matlab_compat import del2

#: Contour length from which :func:`snakedeform` switches to the FFT (circulant) solver by default.
CIRCULANT_MIN_N = 32

__all__ = [
    "CIRCULANT_MIN_N", "snake_first_column",
    "bound_mirror_expand", "bound_mirror_ensure", "bound_mirror_shrink",
    "gradient2", "gradient2_complex", "gradient2_magnitude",
    "gvf", "xconv2", "gaussian_mask", "gaussian_blur",
    "snake_matrix", "snakedeform", "snakeinterp", "snakeindex",
]


# --------------------------------------------------------------------------------------------------------------
# Mirror boundary condition — the natural boundary condition ∇u·dσ = 0 of Eq. (6.46)
# --------------------------------------------------------------------------------------------------------------

def bound_mirror_expand(A: np.ndarray) -> np.ndarray:
    """Pad ``A`` by one pixel on every side, mirroring about the **second** row/column (Xu & Prince
    ``BoundMirrorExpand.m``).

    Book: §6.2, Eq. (6.46) — the natural boundary condition ``∇û·dσ = 0`` on ``∂Ω`` is exactly a zero normal
    derivative, which a mirror about the border row/column realises.  MATLAB source:
    ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/BoundMirrorExpand.m``.

    Equivalent to ``np.pad(A, 1, mode='reflect')`` (asserted in the tests).  Parity: exact.
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    if m < 2 or n < 2:
        raise ValueError("BoundMirrorExpand needs at least a 2x2 matrix")
    B = np.zeros((m + 2, n + 2), dtype=np.float64)
    B[1:m + 1, 1:n + 1] = A
    # MATLAB: B([1 m+2],[1 n+2]) = B([3 m],[3 n]) -- corners first, then the four sides.
    B[np.ix_([0, m + 1], [0, n + 1])] = B[np.ix_([2, m - 1], [2, n - 1])]
    B[np.ix_([0, m + 1], np.arange(1, n + 1))] = B[np.ix_([2, m - 1], np.arange(1, n + 1))]
    B[np.ix_(np.arange(1, m + 1), [0, n + 1])] = B[np.ix_(np.arange(1, m + 1), [2, n - 1])]
    return B


def bound_mirror_ensure(A: np.ndarray) -> np.ndarray:
    """Re-impose the mirror boundary condition on an **already padded** array (Xu & Prince ``BoundMirrorEnsure.m``).

    Book: §6.2, Eq. (6.46).  MATLAB source:
    ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/BoundMirrorEnsure.m`` (errors when ``m < 3`` or ``n < 3`` —
    reproduced as a ``ValueError``).  Parity: exact.
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    if m < 3 or n < 3:
        raise ValueError("either the number of rows or columns is smaller than 3")
    B = A.copy()
    yi = np.arange(1, m - 1)
    xi = np.arange(1, n - 1)
    B[np.ix_([0, m - 1], [0, n - 1])] = B[np.ix_([2, m - 3], [2, n - 3])]  # mirror corners
    B[np.ix_([0, m - 1], xi)] = B[np.ix_([2, m - 3], xi)]
    B[np.ix_(yi, [0, n - 1])] = B[np.ix_(yi, [2, n - 3])]
    return B


def bound_mirror_shrink(A: np.ndarray) -> np.ndarray:
    """Drop the one-pixel mirrored border (Xu & Prince ``BoundMirrorShrink.m``): ``A[1:-1, 1:-1]``.  Parity: exact."""
    A = np.asarray(A)
    return A[1:-1, 1:-1].copy()


# --------------------------------------------------------------------------------------------------------------
# gradient2 — the MATLAB-4 ``gradient`` semantics used to build the edge map f = |∇I|
# --------------------------------------------------------------------------------------------------------------

def _grad_first_axis(a: np.ndarray, ax: np.ndarray) -> np.ndarray:
    """Derivative of ``a`` along its **columns** (MATLAB's inner loop of ``gradient2.m``): one-sided at the two
    edges, centred over ``ax(k+2) - ax(k)`` in the middle."""
    m, n = a.shape
    y = np.zeros((m, n), dtype=np.float64)
    if n > 1:
        y[:, 0] = (a[:, 1] - a[:, 0]) / (ax[1] - ax[0])
        y[:, n - 1] = (a[:, n - 1] - a[:, n - 2]) / (ax[n - 1] - ax[n - 2])
    if n > 2:
        y[:, 1:n - 1] = (a[:, 2:n] - a[:, 0:n - 2]) / (ax[2:n] - ax[0:n - 2])[None, :]
    return y


def gradient2(a: np.ndarray, xax: float | np.ndarray = 1.0,
              yax: float | np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``[xx, yy] = gradient2(a, xax, yax)`` — MATLAB-4 ``gradient``: ``xx = dZ/dx`` (**columns**), ``yy = dZ/dy``
    (**rows**).

    Book: §6.1.1.2 Eq. (6.7) / §6.2 — the ch6 drivers build the edge map as ``f2 = abs(gradient2(double(f)))``
    (see :func:`gradient2_magnitude`) and, with ``GVFOn = 0``, use ``[u, v] = gradient2(f2)`` as the plain
    external force field.  MATLAB source: ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/gradient2.m`` (itself a
    renamed MATLAB-4 ``gradient.m``, MathWorks 1984-94, kept by the authors so the MATLAB-5 argument order could
    not change under them).

    ``xax``/``yax`` may be scalar spacings or explicit coordinate vectors, exactly like MATLAB's; the default
    unit spacing makes this identical to ``np.gradient(a)[::-1]``.  Parity: exact.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("gradient2 expects a 2-D matrix")
    m, n = a.shape
    if yax is None:
        yax = xax
    xa = np.asarray(xax, dtype=np.float64)
    ya = np.asarray(yax, dtype=np.float64)
    xa = xa * np.arange(n, dtype=np.float64) if xa.ndim == 0 else xa.ravel()
    ya = ya * np.arange(m, dtype=np.float64) if ya.ndim == 0 else ya.ravel()
    xx = _grad_first_axis(a, xa)          # d/dcolumn  (first pass of the MATLAB loop)
    yy = _grad_first_axis(a.T, ya).T      # d/drow     (second pass, on the transpose)
    return xx, yy


def gradient2_complex(a: np.ndarray, xax: float | np.ndarray = 1.0,
                      yax: float | np.ndarray | None = None) -> np.ndarray:
    """The **one-output** form of ``gradient2.m``: ``dZ/dx + i·dZ/dy`` (``z = x + sqrt(-1).*y.'`` in the M-code).

    This is why ``abs(gradient2(double(f)))`` in ``GVF_distance.m`` / ``for_test.m`` is the gradient
    *magnitude* ``|∇I|`` of Eq. (6.7) and not a component.  Parity: exact.
    """
    xx, yy = gradient2(a, xax, yax)
    return xx + 1j * yy


def gradient2_magnitude(a: np.ndarray) -> np.ndarray:
    """``abs(gradient2(a))`` = ``|∇a|`` — the edge map ``f`` of Eq. (6.7) used by every ch6 driver.  Parity: exact."""
    xx, yy = gradient2(a)
    return np.hypot(xx, yy)


# --------------------------------------------------------------------------------------------------------------
# Gradient vector flow — Eqs. (6.41), (6.50)-(6.55)
# --------------------------------------------------------------------------------------------------------------

def gvf(f: np.ndarray, mu: float, iters: int, *, check_cfl: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Gradient vector flow of the edge map ``f`` — Xu & Prince ``GVF.m``.

    Book: §6.2.  Minimises Eq. (6.41) ``E = ∬ μ|∇v|² + |∇f|²|v − ∇f|² dx dy`` through the Euler equations
    (6.50a/b) ``μ∇²u − (u − f_x)(f_x² + f_y²) = 0``, solved by the explicit generalized-diffusion iteration
    (6.51)–(6.53):

    ``u^{t+1} = [1 − (f_x²+f_y²)Δt] u^t + r ∇²_5 u^t + f_x (f_x²+f_y²) Δt``

    with ``r = μΔt/(ΔxΔy)`` (Eq. 6.54).  ``GVF.m`` runs at ``Δt = Δx = Δy = 1`` so ``r = μ``, and writes the
    five-point Laplacian of Eq. (6.52c) as ``4·del2`` (see :func:`seaice.core.matlab_compat.del2`).  Footnote 3
    on p. 125: **μ = 0.1 for all GVF snakes in this book**; the CFL condition (6.55) is ``Δt ≤ ΔxΔy/(4μ)``,
    i.e. ``r = μ ≤ 1/4`` at unit steps.

    MATLAB source: ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/GVF.m``.

    Parameters
    ----------
    f : ndarray (M, N)
        Edge map, larger on edges.  Normalised to ``[0, 1]`` **inside** the function, exactly as ``GVF.m`` does.
    mu : float
        Regularization coefficient of Eq. (6.41).
    iters : int
        Number of diffusion iterations (§6.5.2: this is what sets the capture range).
    check_cfl : bool
        Raise when ``mu > 0.25`` violates the stability condition (6.55) at ``Δt = Δx = Δy = 1``.

    Returns
    -------
    (u, v) : float64 arrays of the shape of ``f``.

    Notes
    -----
    Boundary handling is the mirror condition of Eq. (6.46) (``BoundMirrorExpand`` once, ``BoundMirrorEnsure``
    at the top of every iteration, ``BoundMirrorShrink`` at the end).  A constant ``f`` makes the
    normalisation ``(f − fmin)/(fmax − fmin)`` a ``0/0``; that is MATLAB's behaviour too (NaN field) and is
    reported through ``np.errstate`` rather than silently patched.  Parity target: exact.
    """
    f = np.asarray(f, dtype=np.float64)
    if check_cfl and mu > 0.25:
        raise ValueError(
            f"mu = {mu} violates the CFL condition (6.55) Δt ≤ ΔxΔy/(4μ) at Δt = Δx = Δy = 1 (needs μ ≤ 0.25); "
            "pass check_cfl=False to override")
    fmin = float(f.min())
    fmax = float(f.max())
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (f - fmin) / (fmax - fmin)  # normalise f to [0, 1]
    f = bound_mirror_expand(f)
    fx, fy = gradient2(f)  # MATLAB `gradient`; identical to gradient2 at unit spacing
    u = fx.copy()
    v = fy.copy()
    sqr_mag_f = fx * fx + fy * fy
    for _ in range(int(iters)):
        u = bound_mirror_ensure(u)
        v = bound_mirror_ensure(v)
        u = u + mu * 4.0 * del2(u) - sqr_mag_f * (u - fx)  # Eq. (6.53a)
        v = v + mu * 4.0 * del2(v) - sqr_mag_f * (v - fy)  # Eq. (6.53b)
    return bound_mirror_shrink(u), bound_mirror_shrink(v)


# --------------------------------------------------------------------------------------------------------------
# Gaussian smoothing — Eqs. (6.10) / (6.13)
# --------------------------------------------------------------------------------------------------------------

def xconv2(I: np.ndarray, G: np.ndarray) -> np.ndarray:
    """FFT version of ``conv2(I, G, 'same')`` — Xu & Prince ``xconv2.m``.

    Zero-pads both operands to ``(n + n1 − 1, m + m1 − 1)``, multiplies the spectra and crops
    ``YT(1+floor(n1/2) : n+floor(n1/2), …)``.  The header promises agreement with ``conv2(..., 'same')`` "under
    1e-10"; :func:`seaice.core.filters.conv2` is the direct cross-check.  MATLAB source:
    ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/xconv2.m``.  Parity: exact (same crop, FFT round-off ≤ 1e-10).
    """
    I = np.asarray(I, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    n, m = I.shape
    n1, m1 = G.shape
    shape = (n + n1 - 1, m + m1 - 1)
    FI = np.fft.fft2(I, shape)
    FG = np.fft.fft2(G, shape)
    YT = np.real(np.fft.ifft2(FI * FG))
    nl = n1 // 2
    ml = m1 // 2
    return YT[nl:n + nl, ml:m + ml]


def gaussian_mask(k: float, s: float) -> np.ndarray:
    """``M = gaussianMask(k, s)`` — Xu & Prince ``gaussianMask.m``, the sampled Gaussian ``G_σ`` of Eq. (6.10).

    ``R = ceil(3 s)``; ``M(i+R+1, j+R+1) = k · exp(−(i² + j²)/2σ²)/(2πσ²)`` on ``i, j ∈ [−R, R]`` → a
    ``(2R+1)×(2R+1)`` kernel.  Deliberately **not** ``fspecial('gaussian')``, which uses ``2·ceil(2σ)+1``,
    zeroes entries below ``eps·max`` and normalises to unit sum.  Parity: exact.
    """
    if s <= 0:
        raise ValueError("s (standard deviation) must be positive")
    R = int(np.ceil(3.0 * s))
    i = np.arange(-R, R + 1, dtype=np.float64)
    ii, jj = np.meshgrid(i, i, indexing="ij")
    return k * np.exp(-(ii * ii + jj * jj) / 2.0 / s / s) / (2.0 * np.pi * s * s)


def gaussian_blur(I: np.ndarray, s: float) -> np.ndarray:
    """``GI = gaussianBlur(I, s)`` — Xu & Prince ``gaussianBlur.m``: normalise :func:`gaussian_mask` to unit sum
    and convolve with :func:`xconv2`.

    Book: Eq. (6.10) ``C = G_σ * I`` (and Eq. 6.13 ``E_ext = −γ G_σ * I``); Fig. 6.4(b) uses σ = 5, Fig. 6.7(b)
    σ = 4.  **Never executed by the ch6 drivers** (all of them set ``sigma = 0``), but needed to reproduce those
    figures.  MATLAB source: ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/gaussianBlur.m``.  Parity: exact.
    """
    M = gaussian_mask(1.0, s)
    M = M / M.sum()
    return xconv2(np.asarray(I, dtype=np.float64), M)


# --------------------------------------------------------------------------------------------------------------
# The snake solver — Eqs. (6.34)-(6.40)
# --------------------------------------------------------------------------------------------------------------

def snake_matrix(N: int, alpha: float, beta: float, *, book_index: bool = False) -> np.ndarray:
    """The pentadiagonal **circulant** matrix ``A`` of Eq. (6.37) (printed on book p. 120).

    Book: §6.1.2, Eqs. (6.34)–(6.37).  With ``h = 1`` (Eqs. 6.35/6.36) the five diagonals are
    ``a = β_{i−1}``, ``b = −2(β_i + β_{i−1}) − α_i``, ``c = β_{i+1} + 4β_i + β_{i−1} + α_{i+1} + α_i``,
    ``d = −2(β_{i+1} + β_i) − α_{i+1}``, ``e = β_{i+1}``, wrapped around cyclically because the snake is closed.

    MATLAB source: ``snakedeform.m`` lines 22–38.  **Note (Risk R9 of `analysis/ch06.md`)**: the shipped code
    names ``alpham1 = [alpha(2:N) alpha(1)]`` — that is ``α_{i+1}``, the *mirror* of the book's subscript — so
    its ``a`` holds the book's ``e`` and vice versa.  For the constant ``α``, ``β`` used everywhere in the book
    (footnote 2, p. 121: α = 0.05, β = 0) the two conventions give the **same symmetric circulant** matrix;
    ``book_index=True`` selects the book's Eq. (6.34) subscripts, ``False`` (default) the shipped script's.

    Parity: exact (identical to ``snakedeform.m``'s ``A`` for the script convention).
    """
    N = int(N)
    if N < 3:
        raise ValueError("a snake needs at least 3 points (the diag(..., N-2) terms require it)")
    al = np.full(N, float(alpha))
    be = np.full(N, float(beta))
    # MATLAB's names: alpham1(i) = alpha(i+1), alphap1(i) = alpha(i-1)  (cyclic)
    alpha_next = np.roll(al, -1)
    alpha_prev = np.roll(al, 1)
    beta_next = np.roll(be, -1)
    beta_prev = np.roll(be, 1)
    if book_index:  # Eq. (6.36) as printed
        a = beta_prev
        b = -2.0 * (be + beta_prev) - al
        c = beta_next + 4.0 * be + beta_prev + alpha_next + al
        d = -2.0 * (beta_next + be) - alpha_next
        e = beta_next
    else:  # snakedeform.m lines 29-33 verbatim (betam1 = beta(i+1) etc.)
        a = beta_next
        b = -al - 2.0 * be - 2.0 * beta_next
        c = al + alpha_prev + beta_next + 4.0 * be + beta_prev
        d = -alpha_prev - 2.0 * be - 2.0 * beta_prev
        e = beta_prev
    A = np.diag(a[0:N - 2], -2) + np.diag(a[N - 2:N], N - 2)
    A = A + np.diag(b[0:N - 1], -1) + np.diag(b[N - 1:N], N - 1)
    A = A + np.diag(c)
    A = A + np.diag(d[0:N - 1], 1) + np.diag(d[N - 1:N], -(N - 1))
    A = A + np.diag(e[0:N - 2], 2) + np.diag(e[N - 2:N], -(N - 2))
    return A


@lru_cache(maxsize=256)
def _inv_A_gamma(N: int, alpha: float, beta: float, gamma: float, book_index: bool) -> np.ndarray:
    """``inv(A + γI)`` cached per ``(N, α, β, γ)`` — the literal ``invAI = inv(A + gamma*eye(N))`` of the M-code.

    Caching is mathematically identical to recomputing it (``A`` depends only on ``N``, ``α``, ``β``) and is the
    first of the two speed-ups the chapter needs: ``GVF_distance`` calls :func:`snakedeform` ``ceil(iter/5)`` =
    20 times per contour, each of which would otherwise invert a dense ``N×N`` matrix (Risk R4 of
    ``analysis/ch06.md``).
    """
    A = snake_matrix(N, alpha, beta, book_index=book_index)
    return np.linalg.inv(A + gamma * np.eye(N))


def snake_first_column(N: int, alpha: float, beta: float, gamma: float = 0.0) -> np.ndarray:
    """First column of ``A + γI`` for **constant** ``α``, ``β``, when ``A`` is a symmetric circulant matrix.

    With constant coefficients Eq. (6.36) collapses to ``a = e = β``, ``b = d = −α − 4β``, ``c = 2α + 6β``, so
    ``A`` is the circulant generated by ``[c, b, a, 0, …, 0, a, b]`` — which is why the book can call ``A + λI``
    "pentadiagonal and constant" (p. 121).  The wrap-around additions matter only for ``N <= 4``.
    """
    N = int(N)
    a = e = float(beta)
    b = d = -float(alpha) - 4.0 * float(beta)
    c = 2.0 * float(alpha) + 6.0 * float(beta)
    col = np.zeros(N, dtype=np.float64)
    col[0 % N] += c + float(gamma)
    col[1 % N] += b     # sub-diagonal:  A[i, i-1] = b
    col[2 % N] += a     # A[i, i-2] = a
    col[(-1) % N] += d  # super-diagonal, wrapped
    col[(-2) % N] += e
    return col


@lru_cache(maxsize=512)
def _circulant_eigs(N: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Eigenvalues of the circulant ``A + γI`` (its DFT), cached per ``(N, α, β, γ)``."""
    return np.fft.fft(snake_first_column(N, alpha, beta, gamma))


def _solve_circulant(eigs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``(A + γI) z = rhs`` by FFT for the circulant system (``rhs`` is ``(N,)`` or ``(N, k)``)."""
    denom = eigs[:, None] if rhs.ndim == 2 else eigs
    return np.real(np.fft.ifft(np.fft.fft(rhs, axis=0) / denom, axis=0))


def snakedeform(x, y, alpha: float, beta: float, gamma: float, kappa: float,
                fx: np.ndarray, fy: np.ndarray, iters: int, *,
                book_index: bool = False, solver: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Deform a closed snake in the external force field ``(fx, fy)`` — Xu & Prince ``snakedeform.m``.

    Book: §6.1.2, Eqs. (6.39)–(6.40): the implicit-in-``A``, explicit-in-``f`` Euler step

    ``x^{t+1} = (A + λI)^{-1}(λ x^t − f_x)``,  ``y^{t+1} = (A + λI)^{-1}(λ y^t − f_y)``

    with ``λ = gamma`` (the "viscosity"/step size).  The code writes ``+κ·v`` instead of ``−f`` because for a
    GVF snake the external force *is* the GVF field (Eq. 6.56a/b replaces ``∂E_ext/∂x`` by ``−u``), and κ weights
    it.  Force values at the snake's non-integer positions come from **bilinear** interpolation of the field
    (Eq. 6.33): MATLAB ``interp2(fx, x, y, '*linear', 0)`` with extrapolation value 0 →
    :func:`seaice.core.interp.interp2` with ``u = y − 1``, ``v = x − 1``, ``fill = 0.0``.

    Footnote 2, p. 121 fixes **α = 0.05, β = 0.0** for every snake in the book.
    MATLAB source: ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/snakedeform.m``.

    Parameters
    ----------
    x, y : array_like
        Contour points, **1-based MATLAB image coordinates** (``x`` horizontal/column, ``y`` vertical/row).
    alpha, beta, gamma, kappa : float
        Elasticity, rigidity, step size, external-force weight.
    fx, fy : ndarray
        External force field sampled on the image grid.
    iters : int
        Number of Euler steps.
    book_index : bool
        Passed to :func:`snake_matrix` (see its note on Eq. 6.36's subscripts).
    solver : {'auto', 'dense', 'circulant'}
        ``'dense'`` is the literal ``inv(A + gamma*eye(N))`` of the M-code (cached per ``(N, α, β, γ)``).
        ``'circulant'`` solves the same linear system by FFT; for the constant ``α``, ``β`` of this book
        ``A + γI`` is a symmetric circulant matrix, so the two are mathematically identical.
        ``'auto'`` (default) picks ``'circulant'`` from ``N >= CIRCULANT_MIN_N`` upwards, ``'dense'`` below.

    Returns
    -------
    (x, y) : float64 1-D arrays of the same length as the input.

    Notes
    -----
    ``inv(A + γI)`` is computed once per ``(N, α, β, γ)`` and cached; the book (p. 121) makes the same point
    ("``A + λI`` is pentadiagonal and constant"), recommending a single LU decomposition.

    # DEVIATION: `near` - the 'circulant' fast path replaces MATLAB's dense `inv` (LAPACK dgetri, O(N^3)) by an
    # O(N log N) FFT solve of the *same* system.  Exact in exact arithmetic, ~1e-12 relative in float64, but not
    # the identical sequence of floating-point operations.  It is not optional in practice: the snakes of
    # `GVF_distance` grow past 1500 points and the dense inverses alone cost > 5 minutes per image (measured),
    # so `solver='dense'` is kept for the parity test and for short contours.

    Parity: exact (``solver='dense'``) / near (``solver='circulant'``, <= 1e-9 relative).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    N = x.size
    if y.size != N:
        raise ValueError("x and y must have the same length")
    if solver == "auto":
        solver = "circulant" if N >= CIRCULANT_MIN_N else "dense"
    if solver == "circulant":
        eigs = _circulant_eigs(N, float(alpha), float(beta), float(gamma))
    elif solver == "dense":
        inv_ai = _inv_A_gamma(N, float(alpha), float(beta), float(gamma), bool(book_index))
    else:
        raise ValueError("solver must be 'auto', 'dense' or 'circulant'")
    fx = np.asarray(fx, dtype=np.float64)
    fy = np.asarray(fy, dtype=np.float64)
    for _ in range(int(iters)):
        vfx = interp2(fx, y - 1.0, x - 1.0, method="bilinear", fill=0.0)
        vfy = interp2(fy, y - 1.0, x - 1.0, method="bilinear", fill=0.0)
        rhs = np.column_stack([gamma * x + kappa * vfx, gamma * y + kappa * vfy])  # Eq. (6.40)
        z = _solve_circulant(eigs, rhs) if solver == "circulant" else inv_ai @ rhs
        x, y = z[:, 0].copy(), z[:, 1].copy()
    return x, y


def snakeindex(idx) -> np.ndarray:
    """``y = snakeindex(IDX)`` — Xu & Prince ``snakeindex.m``: the half-integer index vector used to insert a new
    point between ``i`` and ``i+1`` wherever ``IDX(i)`` is true.

    ``y = 1:0.5:N+0.5`` (2N entries) with the midpoint ``k + 0.5`` **deleted** wherever ``IDX(k)`` is false.
    Book: §6.1.2 (the arc-length/point-spacing constraint that α enforces).  MATLAB source: ``snakeindex.m``.
    Parity: exact.
    """
    idx = np.asarray(idx).ravel().astype(bool)
    N = idx.size
    y = np.arange(1.0, N + 0.5 + 0.25, 0.5)  # 1, 1.5, ..., N, N+0.5  -> 2N entries
    keep = np.ones(y.size, dtype=bool)
    k = np.nonzero(~idx)[0]                  # MATLAB x(IDX == 0), 0-based
    keep[2 * k + 1] = False                  # MATLAB y(2*x(IDX==0)) = []  (1-based 2k -> 0-based 2k-1)
    return y[keep]


def snakeinterp(x, y, dmax: float, dmin: float, *, max_passes: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Adaptively re-space the snake points — Xu & Prince ``snakeinterp.m``.

    Book: §6.1.2 — with ``h = 1`` (Eq. 6.35) the discretisation assumes unit spacing between snake points, so
    the contour is resampled after every deformation block: a point closer than ``dmin`` to its successor is
    dropped, and a new point is linearly interpolated wherever the spacing exceeds ``dmax``.  The spacing
    metric is **city-block** ``d = |Δx| + |Δy|`` and cyclic (the snake is closed).  The ch6 drivers use
    ``Dmax = 1``, ``Dmin = 0`` — so the removal branch never fires there.

    MATLAB source: ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/snakeinterp.m``.

    Parameters
    ----------
    max_passes : int
        Cap on the ``while max(d) > dmax`` loop, which the shipped code leaves unbounded.  Its own header admits
        "there is a bug in the program for points removal", and a degenerate contour (coincident vertices after
        clipping) could otherwise spin forever.  Exceeding the cap raises ``RuntimeError`` with a diagnostic.

    Returns
    -------
    (xi, yi) : float64 1-D arrays.

    Parity: exact (literal port; ``interp1`` linear → ``np.interp``).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    def cyclic_d(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        nxt = np.r_[np.arange(1, px.size), 0]
        return np.abs(px[nxt] - px) + np.abs(py[nxt] - py)

    d = cyclic_d(x, y)
    keep = ~(d < dmin)           # remove points closer than dmin to their neighbour
    x = x[keep]
    y = y[keep]
    if x.size < 3:
        raise ValueError(f"snakeinterp: only {x.size} points left after the dmin removal pass")

    def refine(px: np.ndarray, py: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = px.size
        z = snakeindex(idx)
        p = np.arange(1.0, n + 2.0)                     # 1 : N+1
        xi = np.interp(z, p, np.r_[px, px[0]])          # interp1(p, [x; x(1)], z')
        yi = np.interp(z, p, np.r_[py, py[0]])
        return xi, yi

    xi, yi = refine(x, y, cyclic_d(x, y) > dmax)
    d = cyclic_d(xi, yi)
    passes = 0
    while d.max() > dmax:
        passes += 1
        if passes > max_passes:
            raise RuntimeError(
                f"snakeinterp did not converge in {max_passes} passes (N = {xi.size}, max spacing "
                f"{d.max():g} > dmax = {dmax}); the shipped snakeinterp.m has no iteration cap")
        xi, yi = refine(xi, yi, d > dmax)
        d = cyclic_d(xi, yi)
    return xi, yi
