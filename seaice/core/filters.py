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
