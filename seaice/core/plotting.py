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

    ch7 also calls it as ``label2rgb(index, @jet, [1, 1, 1])`` (``ice_shape_enhancement.m`` line 183), where
    ``index`` is **not** a ``1..N`` label matrix but the Eq. (7.6) size-coded colour values
    ``fix((1 - exp(-area/1000)) * 10000)``, so ``max(L)`` is of order 10^4 and MATLAB builds a 10 000-entry
    ``jet`` LUT.  That is exactly what happens here (one LUT row per integer value, background where ``L == 0``,
    ``shuffle=False``); the ``background=(1, 1, 1)`` RGB-triple form is the ``[1 1 1]`` argument.  It is still a
    **display-only** routine - compare the ``index`` array, never the colour PNG.
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


def size_colorbar(fig, mappable, colors, n: int = 6, *, C1: float = 10000.0, C2: float = 1000.0,
                  ax=None, label: str | None = None):
    """The Eq. (7.6) size-coded colour bar of Figs. 7.13/7.19/7.20/7.26/7.28 (and the FSD bars of Fig. 7.15/7.21).

    MATLAB source: ``ice_shape_enhancement.m`` lines 195-206 (``n = 6``, on the map) and 226-237 (``nn = 8``, on
    the histogram)::

        d   = fix((max(area_ice) - min(area_ice))/n);
        ysh = min(area_ice) : d : max(area_ice);
        set(colorbar, 'YTick', linspace(min(ytic), max(ytic), length(ysh)));
        YT{1,i} = -round(1000 * log(1 - ysh(i)/10000));      % Eq. (7.6) inverted
        set(colorbar, 'YTickLabel', YT)

    The tick **positions** are ``length(ysh)`` equally spaced points across the colour axis and the tick
    **labels** are the areas that produced those colour values, i.e. Eq. (7.6) solved for ``area``.  The label
    integers are book truths (Fig. 7.13: 3, 131, 277, 448, 656, 917, 1273), so the arithmetic lives in the pure
    :func:`seaice.ch07_ice_type.colorbar_area_ticks`; this function only draws it.  Display only.
    """
    from seaice.ch07_ice_type import colorbar_area_ticks

    values, labels = colorbar_area_ticks(colors, n, C1=C1, C2=C2)
    cb = fig.colorbar(mappable, ax=ax)
    lo, hi = cb.mappable.get_clim()
    cb.set_ticks(np.linspace(lo, hi, len(values)))
    cb.set_ticklabels([str(v) for v in labels])
    if label:
        cb.set_label(label)
    return cb


def matlab_jet(m: int) -> np.ndarray:
    """MATLAB ``jet(m)`` — the ``(m, 3)`` colour table, ported line by line from R2025a ``jet.m``.

    Book: Ch. 8 §8.3 uses ``jet(color_limit_N)`` with ``color_limit_N = 30``
    (``MATLAB_ROOT/ch8/MCD/plot_color_bar_and_floe.m`` line 17, ``main_WL_new.m`` line 43) and ``jet(255)`` /
    ``jet`` in ``color_hist.m`` lines 28/35.  R2025a ``toolbox/matlab/graphics/graphics/color/jet.m``::

        n = ceil(m/4);
        u = [(1:1:n)/n ones(1,n-1) (n:-1:1)/n]';
        g = ceil(n/2) - (mod(m,4)==1) + (1:length(u))';
        r = g + n;  b = g - n;
        g(g>m) = [];  r(r>m) = [];  b(b<1) = [];
        J = zeros(m,3);  J(r,1) = u(1:length(r));  J(g,2) = u(1:length(g));  J(b,3) = u(end-length(b)+1:end);

    Matplotlib's ``"jet"`` descends from this map but is a piecewise-linear *continuous* colormap sampled at
    ``m`` points, which is not the same table (it differs in the third decimal for small ``m``), so the M-code is
    reproduced rather than approximated.

    # DEVIATION: older MATLAB releases wrote ``ceil(n/2) - (mod(m,2)==1)`` where R2025a writes ``mod(m,4)==1``.
    # The two differ only for ``m`` congruent to 3 (mod 4) — e.g. ``jet(255)``, which this chapter uses for a
    # colormap and never for a numeric value.  ``jet(30)``, the table whose rows *are* consumed numerically
    # (``color_M(index,:)`` painted into ``rgbImage``), is identical under both rules.
    """
    m = int(m)
    if m <= 0:
        return np.zeros((0, 3))
    n = int(np.ceil(m / 4))
    u = np.concatenate([np.arange(1, n + 1) / n, np.ones(max(n - 1, 0)), np.arange(n, 0, -1) / n])
    base = int(np.ceil(n / 2)) - (1 if m % 4 == 1 else 0)
    g = base + np.arange(1, u.size + 1)
    r = g + n
    b = g - n
    g = g[g <= m]
    r = r[r <= m]
    b = b[b >= 1]
    J = np.zeros((m, 3), dtype=np.float64)
    J[r - 1, 0] = u[:r.size]
    J[g - 1, 1] = u[:g.size]
    J[b - 1, 2] = u[u.size - b.size:]
    return J


def mcd_colorbar(fig, ax, color_m: np.ndarray, clim: tuple[float, float], *,
                 saturate_label: str = "\u2265", label: str | None = "[m]", mappable=None):
    """The Ch. 8 §8.3 MCD colour bar of Figs. 8.19/8.20 — ``jet(N)`` clamped at the ``N``-th colour.

    MATLAB source: ``MATLAB_ROOT/ch8/MCD/plot_color_bar_and_floe.m`` lines 34-36 and 113-115
    (and ``main_WL_new.m`` lines 57-59)::

        colormap(color_M)                                              % color_M = jet(color_limit_N), N = 30
        caxis([histogram_centers(1) histogram_centers(color_limit_N)])  % = [1 30]
        colorbar

    Every piece whose MCD exceeds ``histogram_centers(color_limit_N)`` is drawn in the **last** colour
    (``plot_color_bar_and_floe.m`` lines 66-70/78-82), so the top tick means "greater than or equal to".  The
    book prints the resulting ticks as ``5, 10, 15, 20, 25, >=30 [m]`` (Figs. 8.19/8.20, pp. 191-192), which is
    MATLAB's default 5:5:30 tick set over ``caxis([1 30])`` with the last label annotated.

    This is a **different** rule from ch7's Eq. (7.6) colour bar (:func:`size_colorbar`), whose ticks are the
    *inverted areas* of a saturating exponential; both are needed, so this is a second function rather than an
    overload (rule 9 applies to duplication, not to genuinely different mappings).

    Parameters
    ----------
    fig, ax : Figure, Axes
        Where to attach the bar.
    color_m : ndarray
        ``(N, 3)`` colour table, i.e. ``jet(color_limit_N)``.
    clim : (float, float)
        ``caxis`` limits — ``(histogram_centers[0], histogram_centers[color_limit_n - 1])``.
    saturate_label : str
        Prefix put in front of the top tick label (MATLAB draws a plain number; the book's caption writes ">=").
    label : str, optional
        Colour-bar label; the book uses ``[m]``.
    mappable : ScalarMappable, optional
        Use an existing mappable (e.g. the one returned by ``imshow``) instead of building one.

    Returns
    -------
    matplotlib.colorbar.Colorbar

    Display only — the numbers behind it are :func:`seaice.ch08_applications.plot_color_bar_and_floe`'s
    ``counts``/``centers``/``color_index``.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize

    cmap = ListedColormap(np.asarray(color_m, dtype=np.float64)[:, :3])
    lo, hi = float(clim[0]), float(clim[1])
    sm = mappable if mappable is not None else ScalarMappable(norm=Normalize(vmin=lo, vmax=hi), cmap=cmap)
    if mappable is None:
        sm.set_array([])
    cb = fig.colorbar(sm, ax=ax)
    ticks = [t for t in cb.get_ticks() if lo <= t <= hi]
    if ticks:
        cb.set_ticks(ticks)
        texts = [f"{t:g}" for t in ticks]
        if abs(ticks[-1] - hi) <= 0.5 * (ticks[-1] - ticks[0]) / max(len(ticks) - 1, 1):
            texts[-1] = f"{saturate_label}{ticks[-1]:g}"
        cb.set_ticklabels(texts)
    if label:
        cb.set_label(label)
    return cb
