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


# ---------------------------------------------------------------------------------------------------------------
# Chapter 5 display helpers (label2rgb, surf, imcontour) — display only, never compared numerically
# ---------------------------------------------------------------------------------------------------------------


def label2rgb(L: np.ndarray, cmap: str = "jet", background: str | tuple = "k", shuffle: bool = True,
              seed: int = 0) -> np.ndarray:
    """MATLAB ``label2rgb(L, 'jet', 'k', 'shuffle')`` — colour a label image (uint8 RGB).

    Used by ``direct_watershed.m``, ``distance_watershed.m``, ``marker_watershed.m`` to display ``bwlabel``
    results.  MATLAB samples the colormap at ``max(L)`` levels and, with ``'shuffle'``, permutes the colours with a
    private fixed-seed stream; here the permutation comes from ``numpy.random.default_rng(seed)`` - the assignment
    of colours to labels is therefore *not* MATLAB's (display only).
    """
    import matplotlib.colors as mcolors

    L = np.asarray(L).astype(np.int64)
    n = int(L.max()) if L.size else 0
    out = np.zeros(L.shape + (3,), dtype=np.uint8)
    bg = np.asarray(mcolors.to_rgb(background)) if isinstance(background, str) else np.asarray(background, float)
    out[...] = np.clip(matlab_round(bg * 255.0), 0, 255).astype(np.uint8)
    if n == 0:
        return out
    colours = plt.get_cmap(cmap)(np.linspace(0.0, 1.0, n))[:, :3]
    if shuffle:
        colours = colours[np.random.default_rng(seed).permutation(n)]
    lut = np.clip(matlab_round(colours * 255.0), 0, 255).astype(np.uint8)
    fg = L > 0
    out[fg] = lut[L[fg] - 1]
    return out


def surface_plot(Z: np.ndarray, path: str | Path | None = None, cmap: str = "copper",
                 xlim: tuple[float, float] | None = None, title: str | None = None, show: bool = False,
                 elev: float = 30.0, azim: float = -60.0, dpi: int = 120):
    """MATLAB ``h = surf(double(Z)); set(h, 'FaceColor', 'texturemap', 'EdgeColor', 'none'); colormap(copper);
    xlim([...])`` (``topological_surface.m``, Figs. 5.1(c), 5.5(b), 5.8(c)) with matplotlib ``plot_surface``.

    ``x`` = column index (1-based like MATLAB's ``surf``), ``y`` = row index, ``z`` = value.  Returns the figure
    (closed after saving when ``path`` is given).
    """
    Z = np.asarray(Z, dtype=np.float64)
    M, N = Z.shape
    X, Y = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False, shade=False, rstride=1, cstride=1)
    ax.view_init(elev=elev, azim=azim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(M, 1)  # MATLAB's surf of an image keeps the row axis increasing away from the viewer
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    if title:
        ax.set_title(title)
    if path is not None:
        finish_figure(fig, path, show, dpi=dpi)
    return fig


def contour_overlay(img: np.ndarray, Z: np.ndarray, path: str | Path | None = None, levels: int = 10,
                    cmap: str = "viridis", title: str | None = None, show: bool = False,
                    xlim: tuple[float, float] | None = None, ylim: tuple[float, float] | None = None,
                    dpi: int = 120):
    """MATLAB ``image(dist, 'CDataMapping', 'scaled'); hold all; imcontour(imgDist)`` (``distance_propagation.m``,
    Fig. 5.10): the scaled image with the iso-distance contours of ``Z`` drawn on top.  ``imcontour``'s automatic
    level count is display-only; ``levels`` chooses matplotlib's.  Axes are 1-based pixel centres like MATLAB.
    """
    img = np.asarray(img)
    Z = np.asarray(Z, dtype=np.float64)
    M, N = Z.shape
    fig, ax = plt.subplots(figsize=(6.5, 6.5 * M / N))
    ax.imshow(imshow_scale(img, autoscale=True), cmap="gray", vmin=0, vmax=1, extent=(0.5, N + 0.5, M + 0.5, 0.5),
              interpolation="nearest")
    X, Y = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))
    ax.contour(X, Y, Z, levels=levels, cmap=cmap, linewidths=0.8)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(max(ylim), min(ylim))
    if title:
        ax.set_title(title)
    if path is not None:
        finish_figure(fig, path, show, dpi=dpi)
    return fig


def snake_plot(ax: plt.Axes, x, y, style: str = "r", **kwargs) -> None:
    """Draw a **closed** snake — port of Xu & Prince ``snakedisp.m`` (``plot([x; x(1)], [y; y(1)], style)``).

    Book: the snake overlays of Figs. 6.5–6.13 and 6.17.  MATLAB source:
    ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/snakedisp.m``.  Coordinates are MATLAB 1-based ``(x, y)`` =
    (column, row); the axes are assumed to show the image with :func:`imshow_matlab`, whose extent is 0-based,
    so 1 is subtracted here.  Display only.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ax.plot(np.r_[x, x[0]] - 1.0, np.r_[y, y[0]] - 1.0, style, **kwargs)


def quiver_field(ax: plt.Axes, u: np.ndarray, v: np.ndarray, step: int = 1, scale: float | None = None,
                 color: str = "b", **kwargs) -> None:
    """Quiver plot of a 2-D vector field on image axes — the display of Figs. 6.4(c), 6.8, 6.9 and 6.16(b)–(e).

    ``u`` is the horizontal (column) component and ``v`` the vertical (row) component, as MATLAB's
    ``quiver(x, y, qx, qy)`` with ``axis('ij')`` expects.  ``step`` subsamples the grid.  Display only.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    M, N = u.shape
    rr, cc = np.mgrid[0:M:step, 0:N:step]
    ax.quiver(cc, rr, u[::step, ::step], v[::step, ::step], color=color, scale=scale, **kwargs)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(M - 0.5, -0.5)  # axis('ij')
    ax.set_aspect("equal")
    ax.axis("off")
