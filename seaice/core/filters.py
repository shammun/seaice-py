"""2-D convolution / correlation — Book §2.5, Eqs. (2.13)–(2.15), Fig. 2.15.

Text-only in ch2 (no ``.m`` file); MATLAB equivalents are ``conv2`` and ``imfilter`` (used by ch4–ch6).

Important semantics: the book's Eq. (2.14) is **true convolution** (kernel flipped),

    h(x, y) = Σ_s Σ_t ω(s, t) f(x - s, y - t),

whereas MATLAB ``imfilter(f, w)`` performs **correlation** by default (``imfilter(..., 'conv')`` flips).
Both are provided; every later chapter should call :func:`imfilter` for ``imfilter`` and :func:`conv2` for
``conv2`` so the semantics stay explicit.
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


def imfilter(f: np.ndarray, w: np.ndarray, mode: str = "corr", padding: str | float = "zeros",
             output: str = "same") -> np.ndarray:
    """MATLAB ``imfilter(f, w, ...)``: correlation (default) or convolution with a chosen padding rule.

    Book: §2.5 (the "convolution" of the text is what MATLAB scripts in ch4–ch6 do with ``imfilter``).

    Parameters
    ----------
    f : ndarray (M, N) or (M, N, C)
        Filtered in float64; the caller converts back to uint8 if needed (MATLAB would saturate).
    w : ndarray (m, n)
        Kernel.  ``m``, ``n`` may be even: MATLAB centres the kernel at element ``floor((size+1)/2)`` (1-based),
        reproduced with scipy's ``origin`` argument.
    mode : {'corr', 'conv'}
        ``'conv'`` flips the kernel (= MATLAB ``imfilter(..., 'conv')`` = Eq. 2.14).
    padding : {'zeros', 'replicate', 'symmetric', 'circular'} or float
        MATLAB boundary options; a number pads with that constant.
    output : {'same', 'full'}

    Parity: exact (float64) vs MATLAB ``imfilter`` on double inputs.
    """
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


def conv_at(f: np.ndarray, w: np.ndarray, x: int, y: int) -> float:
    """Response of a 3×3 (or any odd) kernel at one pixel, written out as Book Eq. (2.15).

    ``h(x, y) = Σ_{s=-1..1} Σ_{t=-1..1} ω(s, t) f(x - s, y - t)`` with ``ω`` indexed from its centre
    (``ω(0, 0)`` = centre element).  Pixels outside ``f`` count as 0.  Demonstration helper for Fig. 2.15.
    """
    f = np.asarray(f, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    m, n = w.shape
    if m % 2 == 0 or n % 2 == 0:
        raise ValueError("Eq. (2.15) assumes an odd-sized kernel with a unique centre")
    hm, hn = m // 2, n // 2
    total = 0.0
    for s in range(-hm, hm + 1):
        for t in range(-hn, hn + 1):
            xx, yy = x - s, y - t  # Eq. (2.14): f(x - s, y - t)
            if 0 <= xx < f.shape[0] and 0 <= yy < f.shape[1]:
                total += w[s + hm, t + hn] * f[xx, yy]
    return float(total)
