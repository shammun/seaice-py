"""2-D convolution / correlation — Book §2.5, Eqs. (2.13)–(2.15), Fig. 2.15.

Text-only in ch2 (no ``.m`` file); MATLAB equivalents are ``conv2`` and ``imfilter`` (used by ch4–ch6).

Important semantics: the book's Eq. (2.14) is **true convolution** (kernel flipped),

    h(x, y) = Σ_s Σ_t ω(s, t) f(x - s, y - t),

whereas MATLAB ``imfilter(f, w)`` performs **correlation** by default (``imfilter(..., 'conv')`` flips).
Both are provided; every later chapter should call :func:`imfilter` for ``imfilter`` and :func:`conv2` for
``conv2`` so the semantics stay explicit.

Note that the book's nine-term expansion Eq. (2.15) is printed in *correlation* form, ``Σ ω(s,t) f(x+s, y+t)``,
which is inconsistent with Eq. (2.14); see :func:`conv_at` (``correlate`` flag) for both readings.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

_PAD_MODES = {"zeros": "constant", "replicate": "nearest", "symmetric": "reflect", "circular": "wrap"}


def conv2(f: np.ndarray, w: np.ndarray, mode: str = "same", boundary: str = "fill") -> np.ndarray:
    """MATLAB ``conv2(f, w, mode)`` — discrete 2-D convolution of Book Eq. (2.14) (kernel flipped, zero padding).

    Book: §2.5, Eqs. (2.13)–(2.14).  Parameters follow MATLAB: ``mode`` in ``{'full', 'same', 'valid'}``;
    ``boundary='fill'`` pads with zeros (MATLAB's only option; scipy also offers ``'wrap'``/``'symm'``).

    Parity: exact vs MATLAB ``conv2`` for odd-sized kernels.  For **even-sized** kernels MATLAB keeps the
    central part starting at row/col ``ceil((size-1)/2)+1`` (1-based) while scipy's ``'same'`` starts at
    ``floor((size-1)/2)+1``: this wrapper reproduces MATLAB by computing ``'full'`` and cropping explicitly.
    """
    f = np.asarray(f, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if mode == "same":
        full = signal.convolve2d(f, w, mode="full", boundary=boundary)
        r0 = int(np.ceil((w.shape[0] - 1) / 2))
        c0 = int(np.ceil((w.shape[1] - 1) / 2))
        return full[r0:r0 + f.shape[0], c0:c0 + f.shape[1]]
    return signal.convolve2d(f, w, mode=mode, boundary=boundary)


_PADDING_OPTIONS = ("zeros", "replicate", "symmetric", "circular")
_MODE_OPTIONS = ("corr", "conv")
_OUTPUT_OPTIONS = ("same", "full")


def _parse_imfilter_options(options: tuple, mode: str | None, padding: str | float | None,
                            output: str | None) -> tuple[str, str | float, str]:
    """Classify MATLAB-style positional ``imfilter`` options (any order) and merge them with the keywords."""
    for opt in options:
        if isinstance(opt, str):
            key = opt.lower()
            if key in _MODE_OPTIONS:
                kind, val = "mode", key
            elif key in _PADDING_OPTIONS:
                kind, val = "padding", key
            elif key in _OUTPUT_OPTIONS:
                kind, val = "output", key
            else:
                raise ValueError(f"unknown imfilter option {opt!r}; expected one of "
                                 f"{_MODE_OPTIONS + _PADDING_OPTIONS + _OUTPUT_OPTIONS} or a number")
        elif isinstance(opt, (int, float, np.integer, np.floating)) and not isinstance(opt, bool):
            kind, val = "padding", float(opt)  # imfilter(f, w, X): pad with the constant X
        else:
            raise ValueError(f"unsupported imfilter option {opt!r}")
        current = {"mode": mode, "padding": padding, "output": output}[kind]
        if current is not None and current != val:
            raise ValueError(f"conflicting imfilter {kind} options: {current!r} and {val!r}")
        if kind == "mode":
            mode = val
        elif kind == "padding":
            padding = val
        else:
            output = val
    return (mode if mode is not None else "corr", padding if padding is not None else "zeros",
            output if output is not None else "same")


def imfilter(f: np.ndarray, w: np.ndarray, *options: str | float, mode: str | None = None,
             padding: str | float | None = None, output: str | None = None) -> np.ndarray:
    """MATLAB ``imfilter(f, w, ...)``: correlation (default) or convolution with a chosen padding rule.

    Book: §2.5 (the "convolution" of the text is what MATLAB scripts in ch4–ch6 do with ``imfilter``).

    Options may be given exactly as in MATLAB, positionally and in any order — e.g. the ch4–ch6 idiom
    ``imfilter(I, h, 'replicate')`` becomes ``imfilter(I, h, "replicate")`` — or by keyword.  A positional
    string is classified by its value (``'corr'``/``'conv'`` → ``mode``; ``'zeros'``/``'replicate'``/
    ``'symmetric'``/``'circular'`` → ``padding``; ``'same'``/``'full'`` → ``output``) and a positional number is
    a constant padding value (``imfilter(f, w, 2.5)``).  Giving the same option both ways with different values
    raises ``ValueError``.  The legacy positional order ``imfilter(f, w, mode, padding, output)`` therefore still
    works.

    Parameters
    ----------
    f : ndarray (M, N) or (M, N, C)
        Filtered in float64; the caller converts back to uint8 if needed (MATLAB would saturate).
    w : ndarray (m, n)
        Kernel.  ``m``, ``n`` may be even: MATLAB centres the kernel at element ``floor((size+1)/2)`` (1-based),
        reproduced with scipy's ``origin`` argument.
    *options : str or float
        MATLAB-style positional options, see above.
    mode : {'corr', 'conv'}, default 'corr'
        ``'conv'`` flips the kernel (= MATLAB ``imfilter(..., 'conv')`` = Eq. 2.14).
    padding : {'zeros', 'replicate', 'symmetric', 'circular'} or float, default 'zeros'
        MATLAB boundary options; a number pads with that constant.
    output : {'same', 'full'}, default 'same'

    Examples
    --------
    >>> g = imfilter(I, h, "replicate")            # MATLAB: imfilter(I, h, 'replicate')
    >>> g = imfilter(I, h, "conv", "symmetric")    # MATLAB: imfilter(I, h, 'conv', 'symmetric')
    >>> g = imfilter(I, h, padding="replicate")    # keyword form, identical result

    Parity: exact (float64) vs MATLAB ``imfilter`` on double inputs.
    """
    mode, padding, output = _parse_imfilter_options(options, mode, padding, output)
    f = np.asarray(f, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError("kernel must be 2-D")
    if mode == "conv":
        w = w[::-1, ::-1]
    elif mode != "corr":
        raise ValueError("mode must be 'corr' or 'conv'")
    if isinstance(padding, str):
        if padding not in _PAD_MODES:
            raise ValueError(f"padding must be one of {list(_PAD_MODES)} or a number")
        nd_mode, cval = _PAD_MODES[padding], 0.0
    else:
        nd_mode, cval = "constant", float(padding)
    # MATLAB centre index (0-based) = (size - 1) // 2 ; ndimage centre = size // 2 + origin
    origin = tuple(((n - 1) // 2) - (n // 2) for n in w.shape)
    M, N = f.shape[:2]
    m, n = w.shape
    if output == "full":
        # pad by (size-1) with the requested boundary rule, filter 'same', keep the M+m-1 x N+n-1 part where
        # the kernel overlaps the original image.
        np_mode = {"constant": "constant", "nearest": "edge", "reflect": "symmetric", "wrap": "wrap"}[nd_mode]
        pad = [(m - 1, m - 1), (n - 1, n - 1)] + [(0, 0)] * (f.ndim - 2)
        kw = {"constant_values": cval} if np_mode == "constant" else {}
        f = np.pad(f, pad, mode=np_mode, **kw)
    elif output != "same":
        raise ValueError("output must be 'same' or 'full'")
    if f.ndim == 3:
        out = np.empty_like(f)
        for k in range(f.shape[2]):
            out[..., k] = ndimage.correlate(f[..., k], w, mode=nd_mode, cval=cval, origin=origin)
    else:
        out = ndimage.correlate(f, w, mode=nd_mode, cval=cval, origin=origin)
    if output == "full":
        r0, c0 = (m - 1) // 2, (n - 1) // 2
        out = out[r0:r0 + M + m - 1, c0:c0 + N + n - 1]
    return out


def conv_at(f: np.ndarray, w: np.ndarray, x: int, y: int, correlate: bool = False) -> float:
    """Response of a 3×3 (or any odd) kernel at one pixel, written out term by term (Book Eq. 2.14 / 2.15).

    By default (``correlate=False``) this implements Book **Eq. (2.14)**, true convolution,

        h(x, y) = Σ_{s=-1..1} Σ_{t=-1..1} ω(s, t) f(x - s, y - t),

    with ``ω`` indexed from its centre (``ω(0, 0)`` = centre element) and pixels outside ``f`` counted as 0;
    it equals ``conv2(f, w, 'same')[x, y]`` and ``imfilter(f, w, 'conv')[x, y]``.

    Book text inconsistency (§2.5, p. 24): the nine-term expansion printed as **Eq. (2.15)** reads
    ``ω(−1,−1) f(x−1,y−1) + ω(−1,0) f(x−1,y) + … + ω(1,1) f(x+1,y+1)``, i.e. ``Σ ω(s,t) f(x+s, y+t)`` — the
    *correlation* form (kernel not flipped), which is what MATLAB ``imfilter(f, w)`` computes and which
    contradicts the sign convention of Eq. (2.14).  For symmetric kernels the two agree; for the antisymmetric
    Sobel-type kernel of :func:`seaice.ch02_preliminaries.convolution_example` they differ in sign.  Pass
    ``correlate=True`` to evaluate Eq. (2.15) exactly as printed (= ``imfilter(f, w)[x, y]``).  The package follows
    Eq. (2.14) / ``conv2`` as the definition of convolution.

    Parameters
    ----------
    f, w : ndarray
        Image and odd-sized kernel.
    x, y : int
        0-based (row, col) of the pixel whose response is wanted.
    correlate : bool, default False
        ``False`` → Eq. (2.14) (convolution); ``True`` → Eq. (2.15) as printed (correlation).

    Demonstration helper for Fig. 2.15.
    """
    f = np.asarray(f, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    m, n = w.shape
    if m % 2 == 0 or n % 2 == 0:
        raise ValueError("Eqs. (2.14)-(2.15) as written out assume an odd-sized kernel with a unique centre")
    hm, hn = m // 2, n // 2
    sign = 1 if correlate else -1
    total = 0.0
    for s in range(-hm, hm + 1):
        for t in range(-hn, hn + 1):
            # Eq. (2.14): f(x - s, y - t)   |   Eq. (2.15) as printed (correlation): f(x + s, y + t)
            xx, yy = x + sign * s, y + sign * t
            if 0 <= xx < f.shape[0] and 0 <= yy < f.shape[1]:
                total += w[s + hm, t + hn] * f[xx, yy]
    return float(total)


# ---------------------------------------------------------------------------------------------------------------
# MATLAB fspecial (chapter 4: Figs. 4.2, 4.4, 4.5; used by edge('log'); ch5/ch6/ch7 Gaussian smoothing)
# ---------------------------------------------------------------------------------------------------------------

_FSPECIAL_KINDS = ("average", "disk", "gaussian", "laplacian", "log", "prewitt", "sobel", "unsharp")


def _hsize_pair(hsize) -> tuple[int, int]:
    """MATLAB ``HSIZE``: scalar → ``[h h]``; two-element → ``[rows cols]``."""
    arr = np.atleast_1d(np.asarray(hsize, dtype=np.float64)).ravel()
    if arr.size == 1:
        arr = np.array([arr[0], arr[0]])
    if arr.size != 2 or np.any(arr <= 0) or np.any(arr != np.floor(arr)):
        raise ValueError("fspecial: HSIZE must be a positive integer scalar or a 2-element vector")
    return int(arr[0]), int(arr[1])


def fspecial(kind: str, p2=None, p3=None, *, hsize=None, sigma: float | None = None,
             alpha: float | None = None, radius: float | None = None) -> np.ndarray:
    """MATLAB ``fspecial(TYPE, ...)`` — the predefined 2-D filter kernels, ported from R2025a ``fspecial.m``.

    Book: §4.1.1 Fig. 4.2 (``'sobel'``, ``'prewitt'``), §4.1.2 Fig. 4.4 (``'laplacian'``), Eqs. (4.14)–(4.15) /
    Fig. 4.5 (``'gaussian'``, ``'log'``).  No ``.m`` file of the book calls ``fspecial`` directly; MATLAB's ``edge``
    (ported in :mod:`seaice.core.edges`) uses ``fspecial('sobel')``, ``fspecial('prewitt')`` and
    ``fspecial('log', 2*ceil(3*sigma)+1, sigma)``; the Gaussian is used by ch5–ch7.

    Parameters follow MATLAB positionally (``fspecial('gaussian', 5, 1)``, ``fspecial('log', 13, 2)``,
    ``fspecial('laplacian', 0.2)``, ``fspecial('disk', 5)``, ``fspecial('average', [3, 3])``) or by keyword
    (``hsize``, ``sigma``, ``alpha``, ``radius``).  Defaults are MATLAB's: ``average`` 3×3; ``disk`` r = 5;
    ``gaussian`` 3×3, σ = 0.5; ``laplacian``/``unsharp`` α = 0.2; ``log`` 5×5, σ = 0.5.  For ``gaussian``/``log``
    an omitted ``hsize`` with a given ``sigma`` gives ``2*ceil(2*sigma)+1`` (MATLAB rule).

    Formulas (MATLAB):

    * ``sobel = [1 2 1; 0 0 0; -1 -2 -1]`` (= −(Fig. 4.2(b) left)); ``prewitt = [1 1 1; 0 0 0; -1 -1 -1]``.
    * ``laplacian(α) = [α 1−α α; 1−α −4 1−α; α 1−α α] / (α + 1)``; α = 0 gives Fig. 4.4(a).
    * ``gaussian``: ``exp(−(x²+y²)/(2σ²))`` on the centred grid, entries ``< eps·max`` zeroed, divided by the sum
      (Eq. 4.14 normalised to unit sum — Fig. 4.5(a)).
    * ``log``: ``h = gaussian`` (unit sum), ``h1 = h·(x²+y²−2σ²)/σ⁴``, ``h = h1 − mean(h1)`` (sums to zero — Fig. 4.5(b);
      *not* Eq. 4.15 sampled, which carries ``1/(2πσ⁶)`` and does not sum to zero).
    * ``average``: ``ones(hsize)/prod(hsize)``; ``disk``: area-weighted pillbox of radius r, unit sum;
      ``unsharp``: ``[0 0 0; 0 1 0; 0 0 0] − laplacian(α)``.

    Returns float64.  Parity: exact (formula port; verified in the chapter-4 report).
    """
    key = str(kind).lower()
    matches = [key] if key in _FSPECIAL_KINDS else [k for k in _FSPECIAL_KINDS if k.startswith(key)]
    if len(matches) != 1:
        raise ValueError(f"fspecial: unknown or ambiguous TYPE {kind!r}; expected one of {_FSPECIAL_KINDS}")
    kind = matches[0]
    # merge keyword aliases into MATLAB's positional (p2, p3) slots
    if kind in ("gaussian", "log", "average"):
        if hsize is not None:
            p2 = hsize
        if sigma is not None:
            p3 = sigma
    elif kind in ("laplacian", "unsharp"):
        if alpha is not None:
            p2 = alpha
    elif kind == "disk":
        if radius is not None:
            p2 = radius
    elif p2 is not None or p3 is not None or hsize is not None or sigma is not None:
        raise ValueError(f"fspecial('{kind}') takes no further arguments")

    if kind == "sobel":
        return np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    if kind == "prewitt":
        return np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]])
    if kind in ("laplacian", "unsharp"):
        a = 0.2 if p2 is None else float(p2)
        if a < 0 or a > 1:
            raise ValueError("fspecial: ALPHA must be in [0, 1]")
        h1, h2 = a / (a + 1.0), (1.0 - a) / (a + 1.0)
        lap = np.array([[h1, h2, h1], [h2, -4.0 / (a + 1.0), h2], [h1, h2, h1]])
        if kind == "laplacian":
            return lap
        return np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]) - lap
    if kind == "average":
        rows, cols = _hsize_pair([3, 3] if p2 is None else p2)
        return np.ones((rows, cols)) / float(rows * cols)
    if kind in ("gaussian", "log"):
        std = 0.5 if p3 is None else float(p3)
        if std <= 0:
            raise ValueError("fspecial: SIGMA must be positive")
        if p2 is None:
            if p3 is not None:
                p2 = [2 * int(np.ceil(2 * std)) + 1] * 2
            else:
                p2 = [3, 3] if kind == "gaussian" else [5, 5]
        rows, cols = _hsize_pair(p2)
        sr, sc = (rows - 1) / 2.0, (cols - 1) / 2.0
        x, y = np.meshgrid(np.arange(-sc, sc + 1), np.arange(-sr, sr + 1))
        std2 = std * std
        h = np.exp(-(x * x + y * y) / (2.0 * std2))
        h[h < np.finfo(float).eps * h.max()] = 0.0
        s = h.sum()
        if s != 0:
            h = h / s
        if kind == "gaussian":
            return h
        h1 = h * (x * x + y * y - 2.0 * std2) / (std2 * std2)
        return h1 - h1.sum() / float(rows * cols)  # make the filter sum to zero
    # 'disk': MATLAB's exact-area pillbox (fspecial.m 'disk' case, transcribed)
    rad = 5.0 if p2 is None else float(p2)
    if rad <= 0:
        raise ValueError("fspecial: RADIUS must be positive")
    crad = int(np.ceil(rad - 0.5))
    x, y = np.meshgrid(np.arange(-crad, crad + 1), np.arange(-crad, crad + 1))
    maxxy = np.maximum(np.abs(x), np.abs(y)).astype(np.float64)
    minxy = np.minimum(np.abs(x), np.abs(y)).astype(np.float64)
    r2 = rad * rad
    with np.errstate(invalid="ignore"):
        m1 = ((r2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * (minxy - 0.5)
              + (r2 >= (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * np.sqrt(np.maximum(r2 - (maxxy + 0.5) ** 2, 0.0)))
        m2 = ((r2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * (minxy + 0.5)
              + (r2 <= (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * np.sqrt(np.maximum(r2 - (maxxy - 0.5) ** 2, 0.0)))
        a2, a1 = np.arcsin(np.clip(m2 / rad, -1, 1)), np.arcsin(np.clip(m1 / rad, -1, 1))
        sgrid = ((r2 * (0.5 * (a2 - a1) + 0.25 * (np.sin(2 * a2) - np.sin(2 * a1)))
                  - (maxxy - 0.5) * (m2 - m1) + (m1 - minxy + 0.5))
                 * ((((r2 < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2) & (r2 > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2))
                     | ((minxy == 0) & (maxxy - 0.5 < rad) & (maxxy + 0.5 >= rad)))))
    sgrid = sgrid + ((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2 < r2)
    sgrid[crad, crad] = min(np.pi * r2, np.pi / 2)
    if crad > 0 and rad > crad - 0.5 and r2 < (crad - 0.5) ** 2 + 0.25:
        m1 = np.sqrt(r2 - (crad - 0.5) ** 2)
        m1n = m1 / rad
        sg0 = 2 * (r2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) - m1 * (crad - 0.5))
        sgrid[2 * crad, crad] = sg0
        sgrid[crad, 2 * crad] = sg0
        sgrid[crad, 0] = sg0
        sgrid[0, crad] = sg0
        sgrid[2 * crad - 1, crad] -= sg0
        sgrid[crad, 2 * crad - 1] -= sg0
        sgrid[crad, 1] -= sg0
        sgrid[1, crad] -= sg0
    sgrid[crad, crad] = min(sgrid[crad, crad], 1.0)
    return sgrid / sgrid.sum()


def homomorphic_butterworth(im: np.ndarray, d: float, n: float = 1.0, *, shape: tuple[int, int] | None = None,
                            alpha_l: float = 0.0999, alpha_h: float = 1.01,
                            matlab_bug: bool = True) -> np.ndarray:
    """Homomorphic Butterworth filter — port of ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/homofil.m``.

    ``H = 1/(1 + (d/A)^{2n})`` with ``A(i, j) = sqrt((i − r/2)² + (j − c/2)²)``, rescaled to
    ``H ← 1 − ((α_H − α_L)H + α_L)`` and applied to ``log2(1 + im)`` in the Fourier domain,
    the result being ``exp(|ifft2(H · fft2(log2(1 + im)))|)``.

    **No book section describes this file** — grepping ch2–ch9 for "homomorphic" / "Butterworth" gives zero
    hits and no `.m` file calls it.  It is ported as a documented orphan utility so the chapter's file
    inventory is complete.

    # DEVIATION: `exact` — measured against MATLAB R2025a at ≤ 1.14e-12 (`reports/ch06_verification.md`,
    # `tests/test_ch06.py::TestL2Extras::test_homofil`).  Nothing in the book or the shipped code *calls* this
    # function and it matches no book section, so its traceability is incomplete — but its parity is not
    # (corrected 2026-09-10, review finding S6; it was labelled `unverified` before it was measured).

    Parameters
    ----------
    matlab_bug : bool
        ``True`` (default) reproduces the shipped file literally: the frequency response is built on the
        **unshifted** DFT grid (``homofil.m`` never calls ``fftshift``, so the "centre" ``(r/2, c/2)`` of the
        filter is not the DC bin), and the author's typo'd variable ``aplhaH`` is used as ``α_H``.
        ``False`` centres the response on the DC bin the way the textbook filter is defined.
    shape : (int, int), optional
        ``(r, c)`` of the filter grid — the M-file takes them as arguments; defaults to the image's own shape.
    """
    im = np.asarray(im, dtype=np.float64)
    r, c = shape if shape is not None else im.shape[:2]
    i = np.arange(1, r + 1, dtype=np.float64)[:, None]
    j = np.arange(1, c + 1, dtype=np.float64)[None, :]
    A = np.sqrt((i - r / 2.0) ** 2 + (j - c / 2.0) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = 1.0 / (1.0 + (d / A) ** (2.0 * n))
    H = np.nan_to_num(H, nan=0.0, posinf=1.0, neginf=0.0)
    H = (alpha_h - alpha_l) * H + alpha_l
    H = 1.0 - H
    if not matlab_bug:
        H = np.fft.ifftshift(H)  # put the designed centre on the DC bin
    im_l = np.log2(1.0 + im)
    im_n = np.abs(np.fft.ifft2(H * np.fft.fft2(im_l)))
    return np.exp(im_n)
