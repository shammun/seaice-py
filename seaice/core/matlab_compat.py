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
    """MATLAB ``imcomplement``: uint8 → ``255 - I``; logical → ``~I``; float → ``1 - I``.

    Book: §2.1.2.2, Eq. (2.3) ``[C; M; Y] = [1; 1; 1] - [R; G; B]`` (RGB normalised to [0, 1]); Eq. (2.21) set
    complement for binary images.  MATLAB source: ``MATLAB_ROOT/ch2/color_image.m`` lines 10–13
    (``I_cmy = imcomplement(I)``; ``Ic = imcomplement(Ir)`` on *doubles in 0–255*, giving ``1 - Ir``, i.e.
    negative values that ``imshow(Ic, [])`` rescales).

    Parity: exact.
    """
    img = np.asarray(img)
    if img.dtype == np.bool_:
        return ~img
    if img.dtype == np.uint8:
        return (255 - img.astype(np.int16)).astype(np.uint8)
    if img.dtype == np.uint16:
        return (65535 - img.astype(np.int32)).astype(np.uint16)
    if np.issubdtype(img.dtype, np.signedinteger):
        return -1 - img  # MATLAB: intmax + intmin - I  == -1 - I for signed types
    return 1.0 - img.astype(np.float64)
