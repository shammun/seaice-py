"""MATLAB-compatible scalar/array helpers whose numpy/skimage equivalents differ silently.

Book: §2.1 (image types, uint8 range), §2.1.2.2 Eq. (2.3) (``imcomplement``), §2.2 (``rgb2gray`` before the
histogram).  MATLAB sources: ``MATLAB_ROOT/ch2/histogram.m`` (``rgb2gray``), ``MATLAB_ROOT/ch2/color_image.m``
(``imcomplement``); the functions are reused by every later chapter.

Semantics reproduced here (see the matlab-to-python skill table):

* ``round`` — MATLAB rounds half *away from zero*; numpy rounds half to even.
* uint8 arithmetic saturates in MATLAB; numpy wraps.  All maths is done in float64 and converted back with
  :func:`im2uint8` (round + clip).
* ``rgb2gray`` — MATLAB uses the NTSC luminance coefficients
  ``[0.298936021293775, 0.587043074451121, 0.114020904255103]`` (the book text quotes the rounded
  ``0.2989/0.5870/0.1140``; ``histogram.m`` line 60 quotes ``0.299/0.587/0.114``), computes in double and rounds.
  ``skimage.color.rgb2gray`` uses Rec.709 weights and returns float — *not* the same image.
"""
from __future__ import annotations

import numpy as np

#: MATLAB ``rgb2gray`` coefficients (``coef = [0.298936021293775 0.587043074451121 0.114020904255103]`` in rgb2gray.m).
RGB2GRAY_COEF = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103], dtype=np.float64)


def matlab_round(x: np.ndarray | float) -> np.ndarray:
    """MATLAB ``round``: halves rounded away from zero (numpy ``np.round`` is half-to-even).

    Book: used implicitly wherever MATLAB converts to integer classes (§2.1.1).  MATLAB source: any ``round`` /
    ``uint8()`` cast in later chapters (``bound2im.m`` uses ``round`` on coordinates).
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def im2double(img: np.ndarray) -> np.ndarray:
    """MATLAB ``im2double``: uint8 → ``/255``, uint16 → ``/65535``, logical → 0/1, float → unchanged (float64).

    Book: §2.1.1 (B-bit integers in ``[0, 2^B - 1]``).  MATLAB source: ``color_image.m`` uses ``double(I)`` (no
    scaling) — see :func:`seaice.core.color.rgb2hsi` for how that is handled.
    """
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float64) / 65535.0
    if img.dtype == np.bool_:
        return img.astype(np.float64)
    return img.astype(np.float64)


def im2uint8(img: np.ndarray) -> np.ndarray:
    """MATLAB ``im2uint8``: float in ``[0, 1]`` → ``round(255 x)`` clipped; uint8 unchanged; logical → 0/255.

    Rounding is MATLAB-style (half away from zero) and the result saturates instead of wrapping (§2.1.1).
    """
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.bool_:
        return img.astype(np.uint8) * 255
    x = matlab_round(np.asarray(img, dtype=np.float64) * 255.0)
    return np.clip(x, 0, 255).astype(np.uint8)


def to_uint8_saturating(x: np.ndarray) -> np.ndarray:
    """Convert a float array already in *0–255 units* to uint8 the way MATLAB ``uint8(x)`` does (round, saturate)."""
    return np.clip(matlab_round(np.asarray(x, dtype=np.float64)), 0, 255).astype(np.uint8)


def saturate_to_class(x: np.ndarray | float, dtype) -> np.ndarray:
    """The result of one MATLAB **integer-class** arithmetic operation: round half away from zero, then saturate.

    MATLAB evaluates ``a - b``, ``a ./ b`` etc. *in the class of the operands* when one of them is an integer
    type: ``uint8(3) - uint8(200)`` is ``0`` (saturation, not wrap-around) and ``uint8(3) / uint8(2)`` is ``2``
    (rounding, not truncation).  numpy wraps and truncates instead, so every step of a chain that MATLAB keeps
    in an integer class has to be pushed through this function separately — the intermediate values are *not*
    the float64 ones.  See ``GVF.m`` lines 21–23 and ``gradient2.m`` lines 38–48, both of which are handed a
    ``uint8`` image by ``GVF_distance.m`` when ``GradientOn = 0``.

    The values are returned as **float64 carrying the integer class's values**, which is what MATLAB itself
    produces as soon as the result is stored in a ``double`` array (``B = zeros(...)`` in ``BoundMirrorExpand.m``,
    ``y = zeros(m, n)`` in ``gradient2.m``).

    Parameters
    ----------
    x : array_like
        The exact (float64) value of the operation.
    dtype : numpy integer dtype
        The MATLAB class the operation is evaluated in.

    Notes
    -----
    The arithmetic is carried out in float64, which represents every ``int8``…``uint32`` value exactly; for
    ``int64``/``uint64`` operands above ``2**53`` the intermediate would lose precision (MATLAB would not).  No
    code in this project uses 64-bit integer images.
    """
    info = np.iinfo(dtype)
    return np.clip(matlab_round(np.asarray(x, dtype=np.float64)), info.min, info.max)


def rgb2gray_matlab(rgb: np.ndarray) -> np.ndarray:
    """MATLAB ``rgb2gray`` (NTSC luminance, double precision, rounded back to the input integer class).

    Book: §2.2, used to build the grayscale image of Fig. 2.7 / the ``imhist`` input.  MATLAB source:
    ``MATLAB_ROOT/ch2/histogram.m`` line 22 (``I = rgb2gray(I)``).

    Parameters
    ----------
    rgb : ndarray, shape (M, N, 3)
        uint8 (returns uint8) or float (returns float64, no rounding — MATLAB keeps class double).

    Notes
    -----
    Parity target: exact (0 differing pixels) vs MATLAB.  ``skimage.color.rgb2gray`` (Rec. 709 weights) would
    differ by several gray levels.  Rounding uses :func:`matlab_round` (half away from zero) as MATLAB's
    ``imlincomb`` does.
    """
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        return rgb  # already gray (MATLAB errors; we are lenient for scripts that accept either)
    if rgb.shape[-1] != 3:
        raise ValueError(f"expected (M, N, 3) RGB, got shape {rgb.shape}")
    f = rgb.astype(np.float64)
    gray = f[..., 0] * RGB2GRAY_COEF[0] + f[..., 1] * RGB2GRAY_COEF[1] + f[..., 2] * RGB2GRAY_COEF[2]
    if rgb.dtype == np.uint8:
        return np.clip(matlab_round(gray), 0, 255).astype(np.uint8)
    if rgb.dtype == np.uint16:
        return np.clip(matlab_round(gray), 0, 65535).astype(np.uint16)
    return gray


def imcomplement(img: np.ndarray) -> np.ndarray:
    """MATLAB ``imcomplement``: logical → ``~I``; unsigned → ``intmax - I``; signed → ``bitcmp(I)``; float → ``1 - I``.

    Book: §2.1.2.2, Eq. (2.3) ``[C; M; Y] = [1; 1; 1] - [R; G; B]`` (RGB normalised to [0, 1]); Eq. (2.21) set
    complement for binary images.  MATLAB source: ``MATLAB_ROOT/ch2/color_image.m`` lines 10–13
    (``I_cmy = imcomplement(I)``; ``Ic = imcomplement(Ir)`` on *doubles in 0–255*, giving ``1 - Ir``, i.e.
    negative values that ``imshow(Ic, [])`` rescales).

    The class rules are R2025a ``toolbox/images/images/imcomplement.m`` lines 39–54 — "IM2 has the same class
    and size as IM" (line 4).  The branches are exactly MATLAB's: ``islogical`` → ``~im``;
    ``uint8/uint16/uint32/uint64`` → ``intmax(class(im)) - im``; ``int8/int16/int32/int64`` → ``bitcmp(im)``
    (= ``-1 - im`` in two's complement); otherwise "should be a float" → ``1 - im`` **evaluated in the input
    class**.  Keeping the float branch in the input class matters: ``imfill``'s grayscale branch
    (``imfill.m`` lines 128–140) complements twice around the reconstruction, and MATLAB round-trips a
    ``single`` image through ``1 - (1 - x)`` in single, which is *not* the float64 value
    (``imcomplement(single(1e-8))`` is exactly ``1``, so ``imfill`` returns ``0`` there, and ``single(0.1)``
    comes back as ``0.100000024``).  ch06 established the same rule for ``imimposemin``.

    Parity: exact (branch for branch vs ``imcomplement.m``; class-preserving).
    """
    img = np.asarray(img)
    if img.dtype == np.bool_:
        return ~img
    if np.issubdtype(img.dtype, np.unsignedinteger):
        # intmax(class) - im, computed in the input class (uint8/uint16/uint32/uint64)
        return np.asarray(np.iinfo(img.dtype).max, dtype=img.dtype) - img
    if np.issubdtype(img.dtype, np.signedinteger):
        return -1 - img  # MATLAB bitcmp(im) == intmax + intmin - I == -1 - I for signed types
    if np.issubdtype(img.dtype, np.floating):
        return img.dtype.type(1) - img  # float32 stays float32, float64 stays float64
    return 1.0 - img.astype(np.float64)  # anything else (e.g. object/Python ints) -> MATLAB's double default


def _del2_along_columns(f: np.ndarray, x: np.ndarray) -> np.ndarray:
    """One pass of MATLAB ``del2``: half the centred second difference along the **first** axis of ``f``.

    ``x`` holds the sample locations of that axis (``h = diff(x)``).  Interior points get
    ``g(i) = (df(i)/h(i) - df(i-1)/h(i-1)) / (h(i) + h(i-1))`` (= ``(f(i+1) - 2 f(i) + f(i-1)) / 2`` for unit
    spacing); the two border rows are **linearly extrapolated from the interior**
    (``g(1) = g(2)(h1+h2)/h2 - g(3) h1/h2``), ``n == 3`` copies ``g(2)`` to both borders and ``n <= 2`` gives 0.
    """
    n = f.shape[0]
    g = np.zeros_like(f)
    h = np.diff(x)
    # NOTE (ch06 verification, open item 4): on input containing +/-Inf the differences below evaluate Inf - Inf
    # and numpy warns, where MATLAB silently produces NaN.  Suppress the warning so the values still match
    # MATLAB exactly (ch05 precedent S8: `imimposemin`'s errstate).
    with np.errstate(invalid="ignore"):
        if n > 2:
            df = np.diff(f, axis=0)  # df[i] = f[i+1] - f[i]
            hi = h[1:n - 1][:, None]  # h(2:n-1) in MATLAB 1-based terms
            hm = h[0:n - 2][:, None]
            g[1:n - 1] = (df[1:n - 1] / hi - df[0:n - 2] / hm) / (hi + hm)
        if n > 3:
            g[0] = g[1] * (h[0] + h[1]) / h[1] - g[2] * h[0] / h[1]
            g[n - 1] = -g[n - 3] * h[n - 2] / h[n - 3] + g[n - 2] * (h[n - 2] + h[n - 3]) / h[n - 3]
        elif n == 3:
            g[0] = g[1]
            g[2] = g[1]
    return g


def del2(f: np.ndarray, hx: float | np.ndarray = 1.0, hy: float | np.ndarray | None = None) -> np.ndarray:
    """MATLAB ``del2``: the **discrete Laplacian divided by 2·ndims** (``∇²/4`` for a 2-D matrix).

    Book: §6.2, Eq. (6.52c) — ``GVF.m`` writes ``mu*4*del2(u)`` precisely because ``4·del2`` restores the
    five-point Laplacian ``u(x+1,y) + u(x,y+1) + u(x-1,y) + u(x,y-1) - 4u(x,y)`` of Eq. (6.52c).
    MATLAB source: ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/GVF.m`` line 37 (``del2``); ported from
    R2025a ``toolbox/matlab/datafun/del2.m``.

    Parameters
    ----------
    f : ndarray
        2-D matrix (a 1-D vector is treated as a column, as MATLAB does; a row vector is transposed back).
    hx, hy : float or ndarray
        Spacings (scalar) or explicit sample locations (vector).  ``del2(f, h)`` uses ``h`` for **both**
        dimensions; ``del2(f, hx, hy)`` swaps them the way MATLAB does (``x`` is the second dimension).

    Notes
    -----
    The border values are *not* the Laplacian: MATLAB linearly extrapolates the interior second differences
    (``g(1) = 2 g(2) - g(3)`` for unit spacing).  Inside ``GVF.m`` those border values are overwritten by
    :func:`seaice.core.snake.bound_mirror_ensure` at the top of the next iteration and dropped by
    ``BoundMirrorShrink`` at the end, but the rule is reproduced here anyway.  Parity: exact.
    """
    a = np.asarray(f, dtype=np.float64)
    rflag = False
    if a.ndim == 1:
        a = a[:, None]
    elif a.ndim == 2 and a.shape[0] == 1:  # MATLAB treats a row vector as a column vector and transposes back
        a = a.T
        rflag = True
    if a.ndim != 2:
        raise ValueError("del2 supports 1-D and 2-D input only")
    if a.shape[1] == 1:  # column vector: one-dimensional case, v = g, divided by ndims(f) = 2
        # NOTE (ch06 review nit): this branch uses `hx` and ignores `hy`.  R2025a's `del2.m` builds `loc` from
        # `parse_inputs`, which for a vector input keeps only one location vector, so `del2(colvec, hx, hy)`
        # would take `v{2}` (i.e. `hy`) as `loc{1}` instead.  Unreachable from ch06 -- `GVF.m` only ever calls
        # `del2(u)` on a 2-D field -- and no other chapter calls the 3-argument form on a vector, so the
        # divergence is documented here rather than reproduced.
        loc0 = hx * np.arange(1, a.shape[0] + 1) if np.isscalar(hx) else np.asarray(hx, dtype=np.float64)
        v = _del2_along_columns(a, np.asarray(loc0, dtype=np.float64)) / 2.0
        return (v.T if rflag else v).reshape(np.shape(f))
    m, n = a.shape
    if hy is None:  # del2(f) or del2(f, h): the same spacing for both dimensions
        loc_rows = hx * np.arange(1, m + 1) if np.isscalar(hx) else np.asarray(hx, dtype=np.float64)
        loc_cols = hx * np.arange(1, n + 1) if np.isscalar(hx) else np.asarray(hx, dtype=np.float64)
    else:  # del2(f, hx, hy): MATLAB swaps 1 and 2 because x is the second dimension
        loc_cols = hx * np.arange(1, n + 1) if np.isscalar(hx) else np.asarray(hx, dtype=np.float64)
        loc_rows = hy * np.arange(1, m + 1) if np.isscalar(hy) else np.asarray(hy, dtype=np.float64)
    v = _del2_along_columns(a, np.asarray(loc_rows, dtype=np.float64))
    v = v + _del2_along_columns(a.T, np.asarray(loc_cols, dtype=np.float64)).T
    v = v / 2.0  # ndims(f) == 2
    return v.T if rflag else v


def matlab_colon(start: float, step: float, stop: float) -> np.ndarray:
    """MATLAB's colon operator ``start:step:stop`` — ``floor((stop-start)/step) + 1`` elements.

    ``np.arange`` decides its length by floating-point accumulation and can add or drop the last element; MATLAB
    computes the count once and multiplies.  Needed literally for ``sea_ice_model.m`` line 25
    ``t = 0:0.05:6.28`` (**126** points, last value **6.25** — the "circle" is a 125-gon with a gap, ch08 risk
    R18 / erratum E10) and for ``min_x : inter : max_x`` in ``color_hist.m`` and ``SeaIce_Image_Structure.m``.

    Book: no equation — this is a language primitive.  MATLAB source: the ``:`` operator as used by
    ``MATLAB_ROOT/ch{6,7}/Sea_Ice_Floe_Identification/sea_ice_model.m`` line 25,
    ``SeaIce_Image_Structure.m`` line 98 and ``ch8/SIFI/color_hist.m`` line 36.

    Parity: **near** (``reports/ch08_verification.md``, deviation D1).  ``0:0.05:6.28`` agrees with MATLAB in
    **length (126) and last value (6.25) exactly**, but **30 of the 126** non-integer angles differ by exactly
    **≤ 1 ulp** (max abs diff 8.9e-16 — MATLAB's colon is more accurate than ``start + k·step``).  Bounded:
    substituting MATLAB's own ``t`` vector into ``sea_ice_model`` changes the brash raster by **0 px**; the
    integer colon vectors (``20:70:3500``, ``21:79:3971``) are **bit-exact**.
    """
    start, step, stop = float(start), float(step), float(stop)
    if step == 0:
        return np.zeros(0, dtype=np.float64)
    q = (stop - start) / step
    # MATLAB's colon snaps `q` to the nearest integer when it is within a few ulps (so `0:0.1:1` has 11 elements
    # and its last value is exactly 1) and truncates otherwise (`0:0.05:6.28` -> 126 elements, last 6.25).
    nearest = np.round(q)
    n = int(nearest) if abs(q - nearest) <= 3.0 * np.finfo(float).eps * max(abs(q), 1.0) else int(np.floor(q))
    if n < 0:
        return np.zeros(0, dtype=np.float64)
    v = start + step * np.arange(n + 1, dtype=np.float64)
    if n >= 1 and abs(v[-1] - stop) <= 3.0 * np.finfo(float).eps * max(abs(stop), 1.0):
        v[-1] = stop
    return v
