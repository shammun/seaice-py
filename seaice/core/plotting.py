"""Figure helpers that mimic MATLAB ``imshow`` display conventions and save arrays as PNGs.

Book: no equation; reproduces ``imshow(I)`` (uint8 → 0..255, logical → 0/1, double → 0..1) and ``imshow(I, [])``
(autoscale min..max) used throughout ``MATLAB_ROOT/ch2/*.m``.  Numerics stay float64; conversion to uint8 happens
only here, with MATLAB-style rounding (half away from zero) and clipping.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")  # non-interactive backend: scripts never block on plt.show()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .matlab_compat import matlab_round  # noqa: E402


def imshow_scale(img: np.ndarray, autoscale: bool = False) -> np.ndarray:
    """Map an image to float64 in ``[0, 1]`` the way MATLAB ``imshow`` would before display.

    ``autoscale=False`` reproduces ``imshow(I)``: uint8 → ``I/255``; logical → 0/1; float assumed in ``[0, 1]``
    (values outside are clipped).  ``autoscale=True`` reproduces ``imshow(I, [])``: linear stretch of
    ``[nanmin, nanmax]`` to ``[0, 1]`` (constant images map to 0, like MATLAB).
    """
    img = np.asarray(img)
    if autoscale:
        x = img.astype(np.float64)
        lo, hi = np.nanmin(x), np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(x)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float64) / 65535.0
    if img.dtype == np.bool_:
        return img.astype(np.float64)
    return np.clip(img.astype(np.float64), 0.0, 1.0)


def to_display_uint8(img: np.ndarray, autoscale: bool = False) -> np.ndarray:
    """``imshow``-scaled image converted to uint8 with MATLAB rounding/clipping (for ``imwrite``-like saving)."""
    x = imshow_scale(img, autoscale=autoscale)
    x = np.where(np.isnan(x), 0.0, x)
    return np.clip(matlab_round(x * 255.0), 0, 255).astype(np.uint8)


def save_image(path: str | Path, img: np.ndarray, autoscale: bool = False) -> Path:
    """Save an array as a PNG exactly as ``imshow`` would display it (gray or RGB), returning the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, to_display_uint8(img, autoscale=autoscale))
    return path


def imshow_matlab(ax: plt.Axes, img: np.ndarray, autoscale: bool = False, title: str | None = None,
                  cmap: str = "gray", interpolation: str = "nearest") -> None:
    """Draw ``img`` on ``ax`` with MATLAB ``imshow`` scaling (see :func:`imshow_scale`) and no axes ticks."""
    x = imshow_scale(img, autoscale=autoscale)
    if x.ndim == 2:
        ax.imshow(x, cmap=cmap, vmin=0.0, vmax=1.0, interpolation=interpolation)
    else:
        ax.imshow(x, interpolation=interpolation)
    ax.set_axis_off()
    if title:
        ax.set_title(title)


def show_matrix(ax: plt.Axes, mat: np.ndarray, fmt: str = "{:g}", cmap: str = "Blues", title: str | None = None,
                fontsize: float = 8.0, highlight: np.ndarray | None = None) -> None:
    """Draw a small matrix as a grid with the values printed in each cell (book-style Figs. 2.11–2.13, 2.16, 2.19)."""
    mat = np.asarray(mat)
    shade = highlight if highlight is not None else (mat != 0)
    ax.imshow(np.asarray(shade, dtype=float), cmap=cmap, vmin=0, vmax=1.6, interpolation="nearest")
    for (r, c), v in np.ndenumerate(mat):
        if isinstance(v, (bool, np.bool_)):
            v = int(v)
        ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=fontsize)
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="k", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    if title:
        ax.set_title(title)


def finish_figure(fig: plt.Figure, path: str | Path, show: bool = False, dpi: int = 120) -> Path:
    """Save ``fig`` to ``path`` (creating folders), optionally ``plt.show()`` (non-blocking on Agg), then close."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:  # pragma: no cover - interactive use only
        plt.show(block=False)
    plt.close(fig)
    return path
