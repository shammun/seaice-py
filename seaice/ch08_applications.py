"""Chapter 8 — Sea Ice Image Processing Applications (Zhang & Skjetne, CRC Press 2018, pp. 175–194).

Three applications of the ch2–ch7 pipeline:

* **§8.1 A shipborne camera system to acquire sea ice concentration at engineering scale** (pp. 176–181) — an
  oblique 4-lens camera on IB *Frej* (OATRC'15), one frame per 10 s, linearly rectified (Appendix A.1.2 → ch10)
  and thresholded with **Otsu** (ch3 §3.1) *and* **k-means with k = 3** (ch3 §3.2), the white *and* the grey
  cluster counting as ice.  **No MATLAB code and no data ship for §8.1** — :func:`shipborne_ice_concentration`
  is the procedure, and every number the book prints for it stays ``unverified``.
* **§8.2 Numerical characterization of a real ice field for parametrization of an ice simulator** (pp. 181–189)
  — Fig. 8.8 (which is ch6/ch7's ``sea_ice_test.jpg``, printed transposed and inverted) is run through
  Algorithms 3/4/5, then each floe is replaced by its **convex hull** and each brash piece by an
  **area-equivalent disk** (:func:`sea_ice_model`), and the result is stored in the Appendix-B structure
  (:func:`sea_ice_image_structure`, :mod:`seaice.core.icestruct`).  Figs. 8.11/8.14/8.15 are
  :func:`color_hist` and :func:`color_hist_comparison`.
* **§8.3 Sea ice floe size statistic** (pp. 189–194) — 2888 floes from a helicopter frame, sized by
  **Eq. (8.1)** ``L = sqrt(4A/π)`` (the "mean clipper diameter", MCD), mapped and histogrammed by
  :func:`mcd_analysis` / :func:`plot_color_bar_and_floe` (Figs. 8.19/8.20) and fitted with a power law through
  **Eqs. (8.2)/(8.3)** (:func:`cumulative_fsd_powerlaw`, Fig. 8.21, α = 1.3704).

MATLAB sources ported here (one Python home each, ``analysis/ch08.md`` §5).  ``MCD`` = ``ch8/MCD/``,
``SIFI`` = ``ch{6,7}/Sea_Ice_Floe_Identification/`` (the four ``SIFI`` files are **byte-identical** in ch6 and
ch7 — ``md5sum`` 4/4 — and were formally deferred to this chapter by ch07):

=============================================================  =========================================================
``MCD/main_WL_new.m`` (67 l)                                   :func:`mcd_analysis`; ``scripts/ch08_main_WL_new.py``
``MCD/plot_color_bar_and_floe.m`` (116 l)                      :func:`plot_color_bar_and_floe`
``MCD/fitting_iceFloes_distribution.m`` (63 l)                 :func:`cumulative_fsd_powerlaw`; ``scripts/ch08_fitting_ice_floes_distribution.py``
``MCD/PowerLaw_fitting_method_and_plotting.m`` (34 l)          :func:`power_law_fit` (orphan: no caller in ``MATLAB_ROOT``)
``MCD/three_fitting_method_and_plotting.m`` (65 l)             :func:`three_distribution_fits`; ``scripts/ch08_three_fitting.py``
``SIFI/sea_ice_model.m`` (170 l)                               :func:`sea_ice_model`; ``scripts/ch08_sea_ice_model.py``
``SIFI/SeaIce_Image_Structure.m`` (115 l)                      :func:`sea_ice_image_structure`; ``scripts/ch08_sea_ice_image_structure.py``
``SIFI/color_hist.m`` (46 l)                                   :func:`color_hist`; ``scripts/ch08_color_hist.py``
``SIFI/color_hist_comparison.m`` (55 l)                        :func:`color_hist_comparison`; ``scripts/ch08_color_hist_comparison.py``
=============================================================  =========================================================

Reused, not re-implemented (CLAUDE.md rule 9): :func:`seaice.core.histogram.hist` (bin **centres**),
:mod:`seaice.core.polygon` (``convhull``/``roipoly``/``polygeom``/``polyxpoly``), :mod:`seaice.core.icestruct`,
:mod:`seaice.core.fitting`, :mod:`seaice.core.stats`,
:func:`seaice.ch07_ice_type.size_color` / :func:`~seaice.ch07_ice_type.colorbar_area_ticks` (Eq. 7.6) and
:func:`seaice.core.plotting.matlab_jet` / :func:`~seaice.core.plotting.mcd_colorbar`.

This code is a port of material that is "only available for academic non-commercial use"; the authors ask that
users cite Q. Zhang and R. Skjetne, *Image processing for identification of sea-ice floes and the floe size
distributions*, IEEE TGRS 53(5):2913–2924, 2015.  ``polygeom.m`` is by H. J. Sommer III.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .ch07_ice_type import IcePiece, colorbar_area_ticks, size_color
from .core.fitting import (LsqOptions, LsqResult, lsqcurvefit, optimset, power_law, truncated_power_law,
                           weibull_survival)
from .core.histogram import hist as matlab_hist
from .core.icestruct import Brash, Circle, Field, Floe, IceImage, Intersect, Param, Polygon
from .core.matlab_compat import matlab_colon
from .core.polygon import convhull, poly2mask, polygeom, polyxpoly, roipoly
from .core.stats import cumulative_size_distribution, mean_caliper_diameter

__all__ = [
    "LENGTH_OVER_PIXEL", "COLOR_LIMIT_N", "MCD_X_BIN", "CIRCLE_ANGLES", "COLOR_HIST_PARAMS",
    "matlab_colon", "FloeModel", "BrashModel", "SeaIceModel", "PlotColorBarFloe", "MCDAnalysis",
    "PowerLawFit", "ThreeFits", "ColorHist", "ColorHistComparison",
    "plot_color_bar_and_floe", "mcd_analysis", "power_law_fit", "cumulative_fsd_powerlaw",
    "three_distribution_fits", "sea_ice_model", "sea_ice_image_structure", "color_hist",
    "color_hist_comparison", "iceimage_to_pieces", "shipborne_ice_concentration", "ShipborneIC",
    "sea_ice_field", "FloeModel", "BrashModel",
]

# ==================================================================================================================
# Book / script constants
# ==================================================================================================================

#: ``main_WL_new.m`` line 13 — metres per pixel of the §8.3 helicopter frame.  The book (p. 190) says the scale
#: comes from IB *Oden*'s known length in frames taken at an equivalent height, but **never prints the number**;
#: ``1114 px × 1.1794 = 1313.8 m`` is suspiciously close to the quoted 1314 m filming height (risk R15).
LENGTH_OVER_PIXEL = 1.1794

#: ``main_WL_new.m`` line 21 / ``plot_color_bar_and_floe.m`` line 10 — "the first 30 bars are colored";
#: every MCD above the 30th histogram centre gets the 30th colour, which is why Figs. 8.19/8.20 print "≥30 [m]".
COLOR_LIMIT_N = 30

#: ``plot_color_bar_and_floe.m`` line 4 — ``x_bin = [1:1:100]``, the MCD histogram **centres** in metres.
MCD_X_BIN = np.arange(1, 101, dtype=np.float64)

#: ``color_hist.m`` lines 5–7 / 36 and ``color_hist_comparison.m`` lines 8–10 / 45.  ``max_x`` is the only
#: difference between the two files (3500 vs 6000) — see risk R9.
COLOR_HIST_PARAMS = {"min_x": 20, "inter": 70, "max_x": 3500, "nn": 8}


# `matlab_colon` lives in `core/matlab_compat.py` (CLAUDE.md rule 9 — the colon operator's element-count /
# endpoint rule is a general MATLAB-semantics primitive, not a chapter-8 one; ch09's model-ice work needs it
# too).  It is imported above and re-exported in `__all__`, so `from seaice.ch08_applications import
# matlab_colon` keeps working.

#: ``sea_ice_model.m`` line 25 — ``t = 0:0.05:6.28``.  126 angles, the last one 6.25 rad: 6.28 < 2π, so the
#: sampled "circle" does **not** close (the gap spans ≈ 0.033 rad).  Reproduced exactly; an off-by-one here
#: changes the ``roipoly`` raster and every ``polyxpoly`` overlap test.
CIRCLE_ANGLES = matlab_colon(0.0, 0.05, 6.28)


# ==================================================================================================================
# Small MATLAB idioms this chapter needs (analysis/ch08.md §6 "small helpers")
# ==================================================================================================================

def _nearest_centre_index(value: float, centers: np.ndarray) -> int:
    """``[~, index] = min(abs(MCD(i) - histogram_centers))`` — 1-based, **first** index on a tie (analysis C11).

    ``plot_color_bar_and_floe.m`` line 51.  This is *not* the ``hist`` bin assignment: for the integer centres
    ``1:100`` it is MATLAB's ``round`` with MATLAB's own tie rule (``min`` keeps the first, i.e. the *lower*
    centre, where ``round`` would go away from zero) — the two differ at exact half-metres.
    """
    return int(np.argmin(np.abs(float(value) - np.asarray(centers, dtype=np.float64)))) + 1


def _matlab_isfield(struct_array: Sequence[Any], name: str) -> bool:
    """``isfield(Poly_struc, 'Vertices')`` — does the struct array have this field?

    ``plot_color_bar_and_floe.m`` lines 53/73/105/107 choose the drawing branch with it.  In Python the
    "struct array" is a list of objects, so the test is "does the first element have this attribute".
    """
    if len(struct_array) == 0:
        return False
    return hasattr(struct_array[0], name)


def _segments_cross(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, tol: float = 1e-12) -> bool:
    """``~isempty(polyxpoly(x1, y1, x2, y2))`` — do two polylines share at least one point?

    A vectorised boolean twin of :func:`seaice.core.polygon.polyxpoly` (same segment/segment predicate, same
    ``1e-12`` tolerances, same collinear-overlap rule), used because ``sea_ice_model.m`` calls ``polyxpoly``
    ``O(N²)`` times and only ever looks at whether the result is empty.  Pass
    ``intersect_impl='polyxpoly'`` to :func:`sea_ice_model` to route through the full routine instead.
    """
    px = np.asarray(x1, dtype=np.float64)[:-1, None]
    py = np.asarray(y1, dtype=np.float64)[:-1, None]
    rx = np.asarray(x1, dtype=np.float64)[1:, None] - px
    ry = np.asarray(y1, dtype=np.float64)[1:, None] - py
    qx = np.asarray(x2, dtype=np.float64)[None, :-1]
    qy = np.asarray(y2, dtype=np.float64)[None, :-1]
    sx = np.asarray(x2, dtype=np.float64)[None, 1:] - qx
    sy = np.asarray(y2, dtype=np.float64)[None, 1:] - qy
    if px.size == 0 or qx.size == 0:
        return False

    denom = rx * sy - ry * sx
    qpx = qx - px
    qpy = qy - py
    proper = denom != 0
    if proper.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qpx * sy - qpy * sx) / denom
            u = (qpx * ry - qpy * rx) / denom
        hit = proper & (t >= -tol) & (t <= 1 + tol) & (u >= -tol) & (u <= 1 + tol)
        if hit.any():
            return True
    collinear = (~proper) & (np.abs(qpx * ry - qpy * rx) <= tol)
    if collinear.any():
        rr = rx * rx + ry * ry
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = (qpx * rx + qpy * ry) / rr
            t1 = t0 + (sx * rx + sy * ry) / rr
        lo = np.maximum(np.minimum(t0, t1), 0.0)
        hi = np.minimum(np.maximum(t0, t1), 1.0)
        if (collinear & (rr > 0) & (lo <= hi)).any():
            return True
    return False


def _crosses(v1: np.ndarray, v2: np.ndarray, impl: str = "fast") -> bool:
    """The ``if xx ~= NaN`` test of ``sea_ice_model.m`` lines 65, 81 and 135.

    # PARITY: literal — ``xx ~= NaN`` is ``true`` for *every* element of ``xx`` (nothing equals NaN in IEEE
    # arithmetic), and MATLAB's ``if`` on an **empty** array is false, so the statement reduces to
    # ``if ~isempty(xx)``: "the two boundaries **cross**".  Consequence, reproduced deliberately: a polygon
    # lying entirely **inside** another one (containment without a boundary crossing) is reported as
    # *non*-overlapping.  Do not "fix" it — pass ``strict_containment=True`` to :func:`sea_ice_model` for the
    # geometrically correct test (off by default, analysis/ch08.md risk R6).
    """
    x1, y1 = np.asarray(v1, dtype=np.float64)[:, 0], np.asarray(v1, dtype=np.float64)[:, 1]
    x2, y2 = np.asarray(v2, dtype=np.float64)[:, 0], np.asarray(v2, dtype=np.float64)[:, 1]
    if impl == "polyxpoly":
        xx, _, _ = polyxpoly(x1, y1, x2, y2)
        return xx.size > 0
    return _segments_cross(x1, y1, x2, y2)


def _point_in_polygon(px: float, py: float, x: np.ndarray, y: np.ndarray) -> bool:
    """Ray-crossing test, used **only** by the optional ``strict_containment`` branch (not by the M-file)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inside = False
    n = x.size
    for i in range(n):
        j = (i + 1) % n
        if (y[i] > py) != (y[j] > py):
            xin = (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i]) + x[i]
            if px < xin:
                inside = not inside
    return inside


def _bbox(v: np.ndarray) -> tuple[float, float, float, float]:
    """``(xmin, xmax, ymin, ymax)`` of a vertex list."""
    v = np.asarray(v, dtype=np.float64)
    return float(v[:, 0].min()), float(v[:, 0].max()), float(v[:, 1].min()), float(v[:, 1].max())


def _roipoly_into(bw: np.ndarray, x: np.ndarray, y: np.ndarray, crop: bool = True) -> None:
    """``bw(roipoly(double(img), x, y) == 1) = 1`` — accumulate one polygon into the model raster.

    ``sea_ice_model.m`` lines 46–47 (floes) and 155–156 (brash).  With ``crop=True`` the polygon is rasterised
    inside its own bounding box instead of the whole image, which is **provably identical**:

    1. :func:`seaice.core.polygon.poly2mask` snaps vertices with ``floor(5(v − 0.5) + 0.5) + 1``.  Shifting a
       vertex by an **integer** ``d`` shifts the snapped value by exactly ``5d``, so an integer-offset window
       produces the identically shifted edge walk.
    2. The interior is filled by a **per-column** parity scan running downwards.  Columns outside
       ``[floor(xmin) − 1, ceil(xmax) + 1]`` contain no edge crossing, so their parity never toggles and they
       stay 0; inside the window, the scan starts one row above the topmost edge, where the parity is 0 — the
       same state the full-image scan is in when it reaches that row.
    3. The window is clipped to the image, so wherever ``poly2mask`` would clip the polygon at an image border
       the cropped call clips it at the same border.

    Pass ``crop=False`` to run the M-file's literal full-image ``roipoly`` (≈ 18 ms per call on the 627×1114
    §8.3 frame, i.e. ≈ 2 min for 2888 floes + 3452 brash pieces).
    """
    M, N = bw.shape[:2]
    if not crop:
        bw[roipoly(np.zeros((M, N)), x, y)] = 1
        return
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    c0 = max(int(np.floor(x.min())) - 2, 0)
    c1 = min(int(np.ceil(x.max())) + 2, N)
    r0 = max(int(np.floor(y.min())) - 2, 0)
    r1 = min(int(np.ceil(y.max())) + 2, M)
    if c0 >= c1 or r0 >= r1:
        return
    sub = poly2mask(x - c0, y - r0, r1 - r0, c1 - c0)
    bw[r0:r1, c0:c1][sub] = 1


# ==================================================================================================================
# §8.3 — plot_color_bar_and_floe.m
# ==================================================================================================================

@dataclass
class PlotColorBarFloe:
    """Everything ``plot_color_bar_and_floe.m`` computes (its two outputs plus what its two figures draw)."""

    counts: np.ndarray            #: ``[counts, histogram_centers] = hist(MCD, 1:100)`` — the Fig. 8.20 bars
    centers: np.ndarray           #: the 100 bin centres (metres)
    color_index: np.ndarray       #: ``index`` per piece, **1-based**, *before* the ``color_limit_N`` clamp
    color_clamped: np.ndarray     #: ``min(index, color_limit_N)`` — the row of ``color_M`` actually used
    color_m: np.ndarray           #: ``jet(color_limit_N)``
    centres_xy: np.ndarray        #: ``centre(i,:) = Poly_struc(i).Center`` — 1-based ``[x, y]``, the white dots
    kind: str                     #: ``'Vertices'`` or ``'Pixels'`` — which ``isfield`` branch ran
    rgb_image: np.ndarray | None = None     #: the ``Pixels`` branch's painted map (Fig. 8.19); ``None`` otherwise
    polygons: list[np.ndarray] = field(default_factory=list)
    #: the ``Vertices`` branch's ``fill(x, Y_limi - y)`` polygons, **in metres**; empty otherwise


def plot_color_bar_and_floe(color_limit_n: int, mcd, y_limit: float, n: int, pieces: Sequence[Any],
                            length_over_pixel: float = LENGTH_OVER_PIXEL, *,
                            kind: str = "auto", image_shape: tuple[int, int] | None = None,
                            x_bin: np.ndarray | None = None) -> PlotColorBarFloe:
    """``[counts, histogram_centers] = plot_color_bar_and_floe(color_limit_N, MCD, Y_limi, N, Poly_struc, lop)``.

    Book: §8.3, **Figs. 8.19 and 8.20** (pp. 191–192).  MATLAB source:
    ``MATLAB_ROOT/ch8/MCD/plot_color_bar_and_floe.m`` lines 1–116.

    Three blocks:

    1. **lines 4–43** — the MCD histogram ``hist(MCD, 1:100)`` drawn as 100 bars, bar ``i`` coloured
       ``color_M(min(i, color_limit_N), :)``, with ``caxis([centers(1) centers(color_limit_N)])`` = ``[1, 30]``
       (Fig. 8.20).
    2. **lines 47–100** — the floe map, with **two branches** chosen by ``isfield``:

       * ``Vertices`` (called with ``[IceImage.Floe.Polygon]``): ``fill(x·lop, (Y_limi − y)·lop)`` — the map is
         in **metres** and the ``y`` axis is flipped;
       * ``Pixels`` (called with ``[IceImage.Floe]``): the floe's own pixels are painted into ``rgbImage`` — in
         **pixels**, with **no** flip.  This branch is **Fig. 8.19**.

       The colour of piece ``i`` is ``color_M(index, :)`` with ``index`` = the *nearest* histogram centre
       (:func:`_nearest_centre_index`), clamped at ``color_limit_N``.
    3. **lines 102–115** — a white dot at every piece's ``Center`` (flipped and scaled in the ``Vertices``
       branch, raw pixels in the ``Pixels`` branch) plus the colour bar.

    Parameters
    ----------
    color_limit_n : int
        ``color_limit_N`` (30): pieces above the 30th centre share the last colour.
    mcd : array_like
        One MCD per piece, in metres (Eq. 8.1).
    y_limit : float
        ``Y_limi = IceImage.Param.NumPix_y`` — used only by the ``Vertices`` branch's flip.
    n : int
        Number of pieces to draw (``N``); the M-file loops ``1:N``.
    pieces : sequence
        The "struct array": objects with ``.Vertices``/``.Center`` (polygons) or ``.Pixels``/``.Center`` (floes).
    length_over_pixel : float
        Metres per pixel (``1.1794``).
    kind : {'auto', 'Vertices', 'Pixels'}
        Override the ``isfield`` branch selection.
    image_shape : (rows, cols), optional
        Force the size of ``rgbImage``.  The M-file **never pre-allocates it** (line 79 grows it from the first
        assignment), so MATLAB's array ends up ``(max y, max x, 3)`` over all painted pixels — which is what
        the default reproduces.  On the shipped §8.3 data both maxima are attained (627×1114), so it coincides
        with the image size; pass ``image_shape`` when it must not (risk R17).
    x_bin : ndarray, optional
        The histogram centres; default :data:`MCD_X_BIN` (``1:100``).

    Returns
    -------
    PlotColorBarFloe

    Notes
    -----
    Parity: **exact** for ``counts``/``centers``/``color_index``/``rgb_image``; the *figures* are display only
    (MATLAB calls ``imshow`` inside the loop 2888 times — that is a drawing artefact, not a computation).
    """
    mcd = np.asarray(mcd, dtype=np.float64).ravel()
    centers_bin = MCD_X_BIN if x_bin is None else np.asarray(x_bin, dtype=np.float64)

    # --- lines 4-5: the histogram (bin CENTRES, ch07's `hist` port) --------------------------------------------
    counts, centers = matlab_hist(mcd, centers_bin)

    # --- line 17: the colour table -----------------------------------------------------------------------------
    from .core.plotting import matlab_jet
    color_m = matlab_jet(int(color_limit_n))

    if kind == "auto":
        kind = "Vertices" if _matlab_isfield(pieces, "Vertices") else \
               ("Pixels" if _matlab_isfield(pieces, "Pixels") else "")
    if kind not in ("Vertices", "Pixels"):
        raise ValueError("the field name is incorrect")   # plot_color_bar_and_floe.m line 88

    n = int(n)
    index = np.empty(n, dtype=np.int64)
    for i in range(n):
        index[i] = _nearest_centre_index(mcd[i], centers)      # line 51
    clamped = np.minimum(index, int(color_limit_n))            # lines 66-70 / 78-82

    polygons: list[np.ndarray] = []
    rgb_image: np.ndarray | None = None
    if kind == "Vertices":
        for i in range(n):
            v = np.atleast_2d(np.asarray(pieces[i].Vertices, dtype=np.float64))
            x = v[:, 0] * length_over_pixel                                  # lines 54, 57
            y = (float(y_limit) - v[:, 1]) * length_over_pixel               # lines 55, 58
            polygons.append(np.column_stack([x, y]))
    else:
        if image_shape is None:
            max_x = max(int(np.asarray(pieces[i].Pixels)[:, 0].max()) for i in range(n))
            max_y = max(int(np.asarray(pieces[i].Pixels)[:, 1].max()) for i in range(n))
            image_shape = (max_y, max_x)
        rgb_image = np.zeros((int(image_shape[0]), int(image_shape[1]), 3), dtype=np.float64)
        for i in range(n):
            px = np.atleast_2d(np.asarray(pieces[i].Pixels, dtype=np.int64))
            rgb_image[px[:, 1] - 1, px[:, 0] - 1, :] = color_m[clamped[i] - 1]   # lines 77-83 (1-based → 0-based)

    centres_xy = np.array([np.asarray(pieces[i].Center, dtype=np.float64).ravel() for i in range(n)]) \
        if n else np.zeros((0, 2))

    return PlotColorBarFloe(counts=counts, centers=centers, color_index=index, color_clamped=clamped,
                            color_m=color_m, centres_xy=centres_xy, kind=kind,
                            rgb_image=rgb_image, polygons=polygons)


# ==================================================================================================================
# §8.3 — main_WL_new.m
# ==================================================================================================================

@dataclass
class MCDAnalysis:
    """Every array ``main_WL_new.m`` computes (its four figures are drawn by ``scripts/ch08_main_WL_new.py``)."""

    poly_area: np.ndarray          #: line 17 — polygon areas in m²
    poly_mcd: np.ndarray           #: line 18 — Eq. (8.1) on the polygon areas
    raw_area: np.ndarray           #: line 30 — pixel-count areas in m²
    raw_mcd: np.ndarray            #: line 31 — Eq. (8.1) on the pixel areas (== the shipped ``MCD_results.mat``)
    poly: PlotColorBarFloe         #: line 25 — the ``Vertices`` branch
    raw: PlotColorBarFloe          #: line 35 — the ``Pixels`` branch = **Fig. 8.19**
    count_error: np.ndarray        #: line 41 — ``Poly_counts - [Raw_counts(2:end) 0]``
    length_over_pixel: float
    color_limit_n: int

    @property
    def poly_counts(self) -> np.ndarray:
        return self.poly.counts

    @property
    def poly_centers(self) -> np.ndarray:
        return self.poly.centers

    @property
    def raw_counts(self) -> np.ndarray:
        return self.raw.counts

    @property
    def raw_centers(self) -> np.ndarray:
        return self.raw.centers


def mcd_analysis(ice: IceImage, length_over_pixel: float = LENGTH_OVER_PIXEL,
                 color_limit_n: int = COLOR_LIMIT_N, **kwargs: Any) -> MCDAnalysis:
    """``main_WL_new.m`` — the §8.3 driver: MCD of every floe, twice, plus their difference.

    Book: §8.3, **Eq. (8.1)** and **Figs. 8.19/8.20** (pp. 190–192).  MATLAB source:
    ``MATLAB_ROOT/ch8/MCD/main_WL_new.m`` lines 7–41::

        load('IceImage_290915_2_jpg.0000179.mat')
        N = size(IceImage.Floe,1);
        length_over_Pixel = 1.1794;
        for i = 1:N
            Poly_Area(i) = IceImage.Floe(i).Polygon.Area*length_over_Pixel^2;
            Poly_MCD(i)  = sqrt(Poly_Area(i)*4/pi);
        end
        color_limit_N = 30;  Y_limi = IceImage.Param.NumPix_y;
        Poly_struc = [IceImage.Floe.Polygon];
        [Poly_counts,Poly_centers] = plot_color_bar_and_floe(color_limit_N,Poly_MCD,Y_limi,N,Poly_struc,lop)
        for i = 1:N
            Raw_Area(i) = IceImage.Floe(i).Area*length_over_Pixel^2;
            Raw_MCD(i)  = sqrt(Raw_Area(i)*4/pi);
        end
        Raw_floe_struc = [IceImage.Floe];
        [Raw_counts,Raw_centers] = plot_color_bar_and_floe(color_limit_N,Raw_MCD,Y_limi,N,Raw_floe_struc,lop)
        count_error = Poly_counts-[Raw_counts(2:end) 0];

    ``cd('E:\\NTNU\\CRC\\latex\\matlab\\ch8\\MCD')`` (line 5) is dropped; the caller resolves the ``.mat``
    through :func:`seaice.core.io.book_data_dir` (risk R3).

    ``count_error`` (analysis C1) is ``ΔN_k = P_k − R_{k+1}``: the polygon histogram minus the **one-bin
    shifted** raw histogram, because polygonization enlarges the MCD by roughly one 1-metre bin.  It is *not*
    the quantity Fig. 8.15 plots (that is :func:`color_hist_comparison`'s unshifted ``z − z0``) and the figure
    it produces is **not printed in the book** (risk R20).

    Parity: **exact** — ``raw_mcd`` reproduces the authors' own ``MCD_results.mat`` at max \\|Δ\\| = 0.0.
    """
    n = len(ice.Floe)
    poly_px = np.array([f.Polygon.Area for f in ice.Floe], dtype=np.float64)
    raw_px = np.array([f.Area for f in ice.Floe], dtype=np.float64)
    poly_area = poly_px * length_over_pixel ** 2                      # line 17
    raw_area = raw_px * length_over_pixel ** 2                        # line 30
    poly_mcd = mean_caliper_diameter(poly_px, length_over_pixel)      # line 18 = Eq. (8.1)
    raw_mcd = mean_caliper_diameter(raw_px, length_over_pixel)        # line 31 = Eq. (8.1)

    y_limit = ice.Param.NumPix_y
    poly_struct = [f.Polygon for f in ice.Floe]                     # line 24: [IceImage.Floe.Polygon]
    poly = plot_color_bar_and_floe(color_limit_n, poly_mcd, y_limit, n, poly_struct, length_over_pixel,
                                   kind="Vertices", **kwargs)
    raw = plot_color_bar_and_floe(color_limit_n, raw_mcd, y_limit, n, list(ice.Floe), length_over_pixel,
                                  kind="Pixels", **kwargs)

    shifted = np.append(raw.counts[1:], 0.0)                        # line 41: [Raw_counts(2:end) 0]
    count_error = poly.counts - shifted

    return MCDAnalysis(poly_area=poly_area, poly_mcd=poly_mcd, raw_area=raw_area, raw_mcd=raw_mcd,
                       poly=poly, raw=raw, count_error=count_error,
                       length_over_pixel=length_over_pixel, color_limit_n=int(color_limit_n))


# ==================================================================================================================
# §8.3 — the distribution fits
# ==================================================================================================================

@dataclass
class PowerLawFit:
    """``eta = [ε₁, ε₂]`` plus the outputs the M-files capture and then discard."""

    eta: np.ndarray                #: ``[ε₁, ε₂]``; ``α = ε₂`` is the book's 1.3704
    resnorm: float
    exitflag: int
    output: dict
    residual: np.ndarray
    jacobian: np.ndarray
    x: np.ndarray                  #: the fitted ``xdata`` (sorted MCD)
    y: np.ndarray                  #: the fitted ``ydata`` (``N_c``)
    x_plot: np.ndarray             #: ``min(x):0.001:max(x)*1.5`` — the curve's abscissa
    y_plot: np.ndarray             #: the fitted curve on ``x_plot``

    @property
    def alpha(self) -> float:
        """The power-law exponent ``α = ε₂`` (p. 192: "estimated to be 1.3704")."""
        return float(self.eta[1])


def _fit_curve(model, p0, x, y, lb=None, ub=None, options: LsqOptions | None = None) -> tuple[LsqResult, np.ndarray, np.ndarray]:
    """``lsqcurvefit`` + the M-files' shared ``x_plot = [min(x):0.001:max(x)*1.5]`` evaluation."""
    res = lsqcurvefit(model, p0, x, y, lb, ub, options)
    x_plot = matlab_colon(float(np.min(x)), 0.001, float(np.max(x)) * 1.5)
    return res, x_plot, model(res.x, x_plot)


def power_law_fit(x, y, *, options: LsqOptions | None = None) -> PowerLawFit:
    """``eta = PowerLaw_fitting_method_and_plotting(fig_num, x, y)`` — fit **Eq. (8.3)** to ``(x, y)``.

    Book: §8.3, **Eq. (8.3)** ``N_c(L) ∝ L^{-α}`` (p. 192).  MATLAB source:
    ``MATLAB_ROOT/ch8/MCD/PowerLaw_fitting_method_and_plotting.m`` lines 12–21::

        F_powerlaw0 = @(epsilong,x)(epsilong(1)*(x.^(-1*epsilong(2))));
        epsilong0   = [min(x) 1];
        [epsilong,resnorm,residual,exitflag,output,~,jacobian] = lsqcurvefit(F_powerlaw0,epsilong0,x,y);
        x_plot = [min(x):0.001:max(x)*1.5];
        eta = epsilong;

    That file is an **orphan** — no caller exists anywhere in ``MATLAB_ROOT`` (grep) — but
    ``fitting_iceFloes_distribution.m`` lines 42–51 duplicate it verbatim, so this one function serves both
    (rule 9: the fit is not written twice).  Its axis labels (``$L/h$``, ``$N(>=L)/N_{total}$``) use a
    *normalised* size, suggesting it was reused from the model-ice study of ch9 (risk R16).

    Parity: **near** — see :func:`seaice.core.fitting.lsqcurvefit`.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    p0 = np.array([float(x.min()), 1.0])                      # line 13: epsilong0 = [min(x) 1]
    res, x_plot, y_plot = _fit_curve(power_law, p0, x, y, options=options)
    return PowerLawFit(eta=res.x, resnorm=res.resnorm, exitflag=res.exitflag, output=res.output,
                       residual=res.residual, jacobian=res.jacobian, x=x, y=y, x_plot=x_plot, y_plot=y_plot)


def cumulative_fsd_powerlaw(mcd, *, options: LsqOptions | None = None) -> PowerLawFit:
    """``fitting_iceFloes_distribution.m`` — **Eq. (8.2)** then **Eq. (8.3)**, i.e. **Fig. 8.21**.

    Book: §8.3, pp. 192–193.  MATLAB source:
    ``MATLAB_ROOT/ch8/MCD/fitting_iceFloes_distribution.m`` lines 7–51::

        load MCD_results.mat
        [sorted_floe_size,index] = sort(Raw_MCD);
        a_sorted_floe_size = Raw_MCD(index);       % == sorted_floe_size (dead code, line 17)
        MCD = a_sorted_floe_size;   N_total = size(MCD,2);
        for i = 1:size(MCD,2),  N_L(i) = size(find(MCD>=MCD(i)),2)/N_total;  end
        ... lsqcurvefit(F_powerlaw0, [min(x) 1], x, y) ...

    Lines 37–39 (the three-distribution call) are **commented out** in the shipped file; use
    :func:`three_distribution_fits` for them.  ``cd('C:\\Users\\qinz\\Desktop\\sent_to_Qin')`` (line 5) is
    dropped (risk R3), and line 57's legend string has a stray quote (``'Power law fitting''``) which renders as
    ``Power law fitting'`` — reproduced in the script's legend.

    Note the text/code discrepancy (analysis §1): the book says α "is the slope of the power law curve on
    log-log plot", but the code runs **non-linear least squares on the untransformed pairs**.  A log-log OLS
    line on the same data gives ``α = 1.87``; the code's route gives **1.37036**, which is what the book prints.

    Parity: **near** (α to ≥ 5 s.f. against MATLAB's own ``lsqcurvefit``).
    """
    L, Nc = cumulative_size_distribution(mcd)          # lines 16-22 (Eq. 8.2)
    return power_law_fit(L, Nc, options=options)


@dataclass
class ThreeFits:
    """``[eta1, eta2, eta3] = three_fitting_method_and_plotting(fig_num, x, y)``."""

    truncated_power: PowerLawFit   #: ``eta1`` = ``[ε₁, ε₂, ε₃]`` (upper-truncated power law, analysis C9)
    weibull: PowerLawFit           #: ``eta2`` = ``[shape, scale]`` (Weibull survival, analysis C10)
    power: PowerLawFit             #: ``eta3`` = ``[ε₁, ε₂]`` (the plain power law — the one the book uses)


def three_distribution_fits(x, y) -> ThreeFits:
    """``three_fitting_method_and_plotting.m`` — the **three** candidate distributions of §8.3.

    Book: §8.3 (only the power law is printed).  MATLAB source:
    ``MATLAB_ROOT/ch8/MCD/three_fitting_method_and_plotting.m`` lines 14–52::

        F_powerlaw = @(epsilong,x)(epsilong(1)*(x.^(-1*epsilong(2))-epsilong(3).^(-1*epsilong(2))));
        epsilong0 = [min(x) 1 x(end)];   lb = [min(x)/10 0 1];   ub = [10^6 10^6 10^6];
        options = optimset('LargeScale','on','MaxFunEvals',100000,'TolFun',1e-5,'MaxIter',10000);
        [epsilong,...] = lsqcurvefit(F_powerlaw,epsilong0,x,y,lb,ub,options);         % eta1
        F_weibull  = @(epsilong,x)(exp(-(x./epsilong(2)).^(epsilong(1))));
        epsilong0  = [1 mean(x)];
        [epsilong,...] = lsqcurvefit(F_weibull,epsilong0,x,y);                        % eta2
        F_powerlaw0= @(epsilong,x)(epsilong(1)*(x.^(-1*epsilong(2))));
        epsilong0  = [min(x) 1];
        [epsilong,...] = lsqcurvefit(F_powerlaw0,epsilong0,x,y);                      % eta3

    Its **only call site is commented out** (``fitting_iceFloes_distribution.m`` line 37), so this is never run
    by the shipped driver; ``scripts/ch08_three_fitting.py`` exposes it (risk R16).  No goodness-of-fit
    statistic is computed for any of the three — neither here nor anywhere in the chapter.

    Parity: **near**; the truncated-power fit is the most start-point sensitive of the three — report
    ``resnorm``/``exitflag`` beside the parameters and, if MATLAB lands elsewhere, report **both** minima
    rather than tuning (risk R1).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    # --- lines 14-26: upper-truncated power law (bounded, legacy optimset) --------------------------------------
    p0 = np.array([float(x.min()), 1.0, float(x[-1])])
    lb = np.array([float(x.min()) / 10.0, 0.0, 1.0])
    ub = np.array([1e6, 1e6, 1e6])
    opts = optimset(LargeScale="on", MaxFunEvals=100000, TolFun=1e-5, MaxIter=10000)
    r1, xp1, yp1 = _fit_curve(truncated_power_law, p0, x, y, lb, ub, opts)
    trunc = PowerLawFit(eta=r1.x, resnorm=r1.resnorm, exitflag=r1.exitflag, output=r1.output,
                        residual=r1.residual, jacobian=r1.jacobian, x=x, y=y, x_plot=xp1, y_plot=yp1)

    # --- lines 30-39: Weibull survival function ------------------------------------------------------------------
    p0 = np.array([1.0, float(x.mean())])
    r2, xp2, yp2 = _fit_curve(weibull_survival, p0, x, y)
    wb = PowerLawFit(eta=r2.x, resnorm=r2.resnorm, exitflag=r2.exitflag, output=r2.output,
                     residual=r2.residual, jacobian=r2.jacobian, x=x, y=y, x_plot=xp2, y_plot=yp2)

    # --- lines 43-52: the plain power law (identical to power_law_fit) ------------------------------------------
    pw = power_law_fit(x, y)
    return ThreeFits(truncated_power=trunc, weibull=wb, power=pw)


# ==================================================================================================================
# §8.2.1 — sea_ice_model.m
# ==================================================================================================================

@dataclass
class FloeModel:
    """``floe0 = struct('Vertices','Center','Area','Perimeter','Intersect')`` — ``sea_ice_model.m`` line 89.

    ``Vertices`` is the **closed** convex-hull ring (``v = [x(k), y(k)]`` with ``k`` from ``convhull``);
    ``SeaIce_Image_Structure.m`` lines 15–16 delete the duplicated last vertex when storing it in the
    Appendix-B structure, so the two forms are deliberately kept distinct.
    """

    Vertices: np.ndarray
    Center: np.ndarray
    Area: float
    Perimeter: float
    Intersect: Intersect = field(default_factory=Intersect)


@dataclass
class BrashModel:
    """``brash0 = struct('Radius','Center','Area','Perimeter','Intersect')`` — ``sea_ice_model.m`` line 143.

    ``Center`` is the pair ``(c(1), c(2))`` the M-file actually uses everywhere (the circle centre at lines
    112-113 and 152-153, the distance test at line 122, the marker at line 165), i.e. MATLAB's **column-major**
    reading of ``cat(1, brash_ice(i).Center)`` — see :func:`_matlab_c1c2`.  For the usual ``1x2`` input that is
    just ``[x, y]``.  The Appendix-B ``Brash(i).Center`` is *not* taken from here:
    ``SeaIce_Image_Structure.m`` line 34 copies ``brash_ice(i).Center`` in its original shape.
    """

    Radius: float
    Center: np.ndarray
    Area: float
    Perimeter: float
    Intersect: Intersect = field(default_factory=Intersect)


@dataclass
class SeaIceModel:
    """``[floe, brash] = sea_ice_model(...)`` plus the two rasters its figures show."""

    floe: list[FloeModel]
    brash: list[BrashModel]
    bw_floe: np.ndarray            #: line 29/47 — union of the polygonized floes (Fig. 8.12(a))
    bw_brash: np.ndarray           #: line 148/156 — union of the circularized brash pieces (Fig. 8.12(b))
    n_pairs_tested: int = 0        #: how many polygon pairs actually reached the crossing test (prefilter stats)
    n_pairs_total: int = 0         #: how many the M-file's literal double loop would test


def _matlab_c1c2(center: np.ndarray) -> tuple[float, float]:
    """``c(1)``, ``c(2)`` of ``c = cat(1, brash_ice(i).Center)`` — **column-major linear indexing**.

    # PARITY: literal.  ``Center`` is normally the ``1×2`` row ``[x, y]``, so ``c(1), c(2)`` is ``(x, y)``.
    # But ch7's ``cat(1, cen.Centroid)`` appends one row per connected component of ``out == i``, and **4 of
    # the shipped 3452 brash pieces have a 2×2 Center** (ch07 review S4, observed here for the first time).
    # MATLAB's linear indexing is column-major, so for ``[x1 y1; x2 y2]`` the M-file's ``c(1)`` is ``x1`` and
    # its ``c(2)`` is **``x2``, not ``y1``** — the circle for such a piece is centred at ``(x1, x2)``.  That is
    # what produced the shipped ``Brash(i).Circle``, so it is reproduced rather than corrected.
    """
    flat = np.asarray(center, dtype=np.float64).ravel(order="F")
    return float(flat[0]), float(flat[1])


def _circle(center: np.ndarray, area: float, t: np.ndarray = CIRCLE_ANGLES) -> tuple[np.ndarray, np.ndarray, float]:
    """``r = sqrt(a/pi); x = c(1) + r*cos(t); y = c(2) + r*sin(t)`` (analysis C2) — lines 76–78 / 109–113."""
    r = float(np.sqrt(float(area) / np.pi))
    c1, c2 = _matlab_c1c2(center)
    return c1 + r * np.cos(t), c2 + r * np.sin(t), r


def sea_ice_model(ice_floe: Sequence[Any], brash_ice: Sequence[Any], img: np.ndarray, *,
                  prefilter: bool = True, strict_containment: bool = False,
                  raster: bool = True, raster_crop: bool = True,
                  intersect_impl: str = "fast") -> SeaIceModel:
    """``[floe, brash] = sea_ice_model(ice_floe, brash_ice, img)`` — polygon fit + disk fit + overlap flags.

    Book: **§8.2.1 "Sea ice numerical modeling"**, pp. 184–189, **Figs. 8.12(a)(b) and 8.13**.  MATLAB source:
    ``MATLAB_ROOT/ch{6,7}/Sea_Ice_Floe_Identification/sea_ice_model.m`` lines 25–157 (byte-identical copies).

    "MIZ floes are generally rounded, but the segmented floes are not convex, so for numerical representation
    each ice floe is replaced by its bounding minimum-area polygon and each brash piece by an area-equivalent
    disk.  The polygonized floes will not be smaller than the actual identified floes, and they may overlap
    other floes and brash pieces" (p. 184).  The "bounding minimum-area polygon" is coded as the **convex
    hull** (line 37, ``convhull(x, y, 'simplify', true)``), which is the minimum-area **convex**
    bounding polygon (the book's phrase omits "convex": a non-convex bounding polygon can be
    smaller).

    Structure of the M-file:

    * **lines 32–41** — one convex hull per floe from its ``PixelsPosition``, closed ring ``v = [x(k), y(k)]``;
    * **lines 46–47** — ``roipoly`` raster union ``bw_floe`` (Fig. 8.12(a));
    * **lines 54–57** — ``polygeom`` for ``Area``/``Center``/``Perimeter`` (the commented-out alternative at
      lines 49–53 would have used ``regionprops`` + ``polyarea``; the authors chose ``polygeom``);
    * **lines 61–69 / 73–85** — floe–floe (``O(N²)``) and floe–brash (``O(N·M)``) overlap by ``polyxpoly``;
    * **lines 104–146** — the brash model: ``r = sqrt(A/π)``, ``p = 2πr`` (analysis C2), brash–brash overlap by
      the pure circle test ``d < r + r'`` (analysis C3, lines 122–124) and brash–floe by ``polyxpoly``.

    Parameters
    ----------
    ice_floe, brash_ice : sequence
        ch7 :class:`seaice.ch07_ice_type.IcePiece` objects (``Center``, ``Area``, ``Perimeter``,
        ``PixelsPosition``) — the field names ``sea_ice_model.m`` reads.  ``brash_ice`` needs only
        ``Center``/``Area``.
    img : ndarray
        Used **only for its size** (``roipoly(double(img), ...)``, ``zeros(size(img))``).
    prefilter : bool
        Skip pairs whose axis-aligned bounding boxes are disjoint.  **Provably equivalent**: every point of a
        polyline lies inside that polyline's AABB (a segment is a convex combination of its endpoints, and an
        AABB is convex), so a common point of two polylines lies in **both** AABBs; if the AABBs are disjoint,
        no common point can exist and ``polyxpoly`` must return empty.  The prefilter therefore only removes
        pairs the crossing test would have rejected anyway.  Set ``False`` for the M-file's literal double loop
        (``n_pairs_tested`` then equals ``n_pairs_total``).
    strict_containment : bool
        ``False`` (default, the M-file): overlap means the boundaries **cross** (see :func:`_crosses`).
        ``True``: also flag a piece whose polygon lies entirely inside another — geometrically correct, and
        **not** what the shipped code does.
    raster : bool
        Compute ``bw_floe``/``bw_brash`` (the M-file always does; they are only used by its two figures).
    raster_crop : bool
        Rasterise each polygon in its bounding box instead of the whole image (see :func:`_roipoly_into` for
        the equivalence proof).  ``False`` = the literal full-image ``roipoly`` call.
    intersect_impl : {'fast', 'polyxpoly'}
        Which crossing test to use; ``'polyxpoly'`` routes through
        :func:`seaice.core.polygon.polyxpoly` exactly as the M-file does (much slower).

    Returns
    -------
    SeaIceModel

    Notes
    -----
    Parity: ``Area``/``Center``/``Perimeter`` **exact** (``polygeom``: ``Area`` ≤ **1.8e-12** against
    the shipped L3 structure of all 2888 floes, ``Center`` ≤ 1.1e-13, ``Perimeter`` ≤ 5.7e-14;
    the test asserts < 1e-11);
    ``Vertices`` **near** — MATLAB's ``convhull`` keeps collinear hull points and starts at a different vertex,
    so the contract is on the vertex **set** and on the rasterised mask (ch06 lesson, risk R5);
    ``Intersect`` lists **exact as sets**.
    """
    shape = np.asarray(img).shape[:2]
    t = CIRCLE_ANGLES

    # ---- lines 32-41: floe polygonization ----------------------------------------------------------------------
    ver: list[np.ndarray] = []
    for piece in ice_floe:
        pixels = np.atleast_2d(np.asarray(piece.PixelsPosition, dtype=np.float64))
        x = pixels[:, 0]
        y = pixels[:, 1]
        k = convhull(x, y, simplify=True)              # line 37 — closed index list
        ver.append(np.column_stack([x[k], y[k]]))      # line 38 — v = [x(k), y(k)]

    # ---- brash circles (lines 73-78 use them inside the floe loop; precomputed once here) -----------------------
    brash_xy: list[tuple[np.ndarray, np.ndarray, float, np.ndarray]] = []
    for piece in brash_ice:
        # `Area` is read column-major for consistency with `_matlab_c1c2`'s `c(1), c(2)`.  Inert on the
        # shipped data (all 3452 `Brash.Area` values are scalars, including the 4 pieces with a 2x2
        # `Center`); a non-scalar would make MATLAB build a k x 126 circle matrix and `polyxpoly` error.
        a = float(np.asarray(piece.Area).ravel(order="F")[0]) if np.size(piece.Area) else float(piece.Area)
        cx, cy, r = _circle(piece.Center, a, t)
        # `c` below is the pair the M-file's `c(1)`, `c(2)` actually read (see `_matlab_c1c2`), which is what
        # the brash-brash distance test of lines 122-124 compares.
        brash_xy.append((cx, cy, r, np.array(_matlab_c1c2(piece.Center))))

    floe_box = np.array([_bbox(v) for v in ver]) if ver else np.zeros((0, 4))
    brash_box = np.array([[cx.min(), cx.max(), cy.min(), cy.max()] for cx, cy, _, _ in brash_xy]) \
        if brash_xy else np.zeros((0, 4))

    def boxes_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Row-wise AABB overlap of one box against a table of boxes (inclusive — touching counts)."""
        return (a[0] <= b[:, 1]) & (b[:, 0] <= a[1]) & (a[2] <= b[:, 3]) & (b[:, 2] <= a[3])

    bw_floe = np.zeros(shape, dtype=np.uint8)
    floe: list[FloeModel] = []
    n_tested = 0
    n_total = 0
    for i, v in enumerate(ver):
        if raster:
            _roipoly_into(bw_floe, v[:, 0], v[:, 1], crop=raster_crop)      # lines 46-47

        geom, _, _ = polygeom(v[:, 0], v[:, 1])                             # line 54
        a, c, p = float(geom[0]), np.array([geom[1], geom[2]]), float(geom[3])   # lines 55-57

        # -- lines 61-69: floe-floe overlapping --------------------------------------------------------------
        inter_floe: list[int] = []
        cand = boxes_overlap(floe_box[i], floe_box) if prefilter else np.ones(len(ver), dtype=bool)
        n_total += len(ver) - 1
        for j in range(len(ver)):
            if j == i:
                continue                                                     # line 62: if j ~= i
            if not cand[j]:
                continue
            n_tested += 1
            if _crosses(v, ver[j], intersect_impl) or \
                    (strict_containment and _contained(v, ver[j])):
                inter_floe.append(j + 1)                                     # line 66 — 1-based serial number

        # -- lines 73-85: floe-brash overlapping -------------------------------------------------------------
        inter_brash: list[int] = []
        candb = boxes_overlap(floe_box[i], brash_box) if (prefilter and len(brash_xy)) \
            else np.ones(len(brash_xy), dtype=bool)
        n_total += len(brash_xy)
        for kk in range(len(brash_xy)):
            if not candb[kk]:
                continue
            n_tested += 1
            cx, cy, _, _ = brash_xy[kk]
            circle = np.column_stack([cx, cy])
            if _crosses(v, circle, intersect_impl) or (strict_containment and _contained(v, circle)):
                inter_brash.append(kk + 1)                                   # line 82

        floe.append(FloeModel(Vertices=v, Center=c, Area=a, Perimeter=p,
                              Intersect=Intersect(floe=np.array(inter_floe, dtype=np.int64),
                                                  brash=np.array(inter_brash, dtype=np.int64))))

    # ---- lines 104-146: brash ice model --------------------------------------------------------------------------
    brash: list[BrashModel] = []
    centers = np.array([c for _, _, _, c in brash_xy]) if brash_xy else np.zeros((0, 2))
    radii = np.array([r for _, _, r, _ in brash_xy]) if brash_xy else np.zeros(0)
    for i, (cx, cy, r, c) in enumerate(brash_xy):
        # column-major, as above (inert: every shipped `Brash.Area` is a scalar)
        a = float(np.asarray(brash_ice[i].Area).ravel(order="F")[0]) if np.size(brash_ice[i].Area) \
            else float(brash_ice[i].Area)
        p = 2.0 * np.pi * r                                                  # line 110 (analysis C2)

        # -- lines 117-128: brash-brash overlapping (pure circle test, analysis C3) ---------------------------
        d = np.hypot(centers[:, 0] - c[0], centers[:, 1] - c[1])
        rr = radii + r
        hit = d < rr                                                         # line 124: if d < rr
        hit[i] = False                                                       # line 118: if j ~= i
        inter_brash = (np.flatnonzero(hit) + 1).astype(np.int64)

        # -- lines 132-139: brash-floe overlapping -----------------------------------------------------------
        circle = np.column_stack([cx, cy])
        inter_floe_list: list[int] = []
        cand = boxes_overlap(brash_box[i], floe_box) if (prefilter and len(ver)) else np.ones(len(ver), dtype=bool)
        n_total += len(ver)
        for kk in range(len(ver)):
            if not cand[kk]:
                continue
            n_tested += 1
            if _crosses(ver[kk], circle, intersect_impl) or (strict_containment and _contained(ver[kk], circle)):
                inter_floe_list.append(kk + 1)                               # line 136

        brash.append(BrashModel(Radius=r, Center=c, Area=a, Perimeter=p,
                                Intersect=Intersect(floe=np.array(inter_floe_list, dtype=np.int64),
                                                    brash=inter_brash)))

    # ---- lines 148-157: the brash raster ------------------------------------------------------------------------
    bw_brash = np.zeros(shape, dtype=np.uint8)
    if raster:
        for i, b in enumerate(brash):
            cx, cy, _, _ = brash_xy[i]
            _roipoly_into(bw_brash, cx, cy, crop=raster_crop)

    return SeaIceModel(floe=floe, brash=brash, bw_floe=bw_floe.astype(bool), bw_brash=bw_brash.astype(bool),
                       n_pairs_tested=n_tested, n_pairs_total=n_total)


def _contained(v1: np.ndarray, v2: np.ndarray) -> bool:
    """Is either polygon entirely inside the other?  **Only** used by ``strict_containment=True`` (risk R6)."""
    v1 = np.atleast_2d(np.asarray(v1, dtype=np.float64))
    v2 = np.atleast_2d(np.asarray(v2, dtype=np.float64))
    return (_point_in_polygon(v1[0, 0], v1[0, 1], v2[:, 0], v2[:, 1]) or
            _point_in_polygon(v2[0, 0], v2[0, 1], v1[:, 0], v1[:, 1]))


# ==================================================================================================================
# Appendix B — SeaIce_Image_Structure.m
# ==================================================================================================================

def sea_ice_image_structure(ice_floe: Sequence[Any], brash_ice: Sequence[Any],
                            floe: Sequence[FloeModel], brash: Sequence[BrashModel],
                            coverage: Any, index_floe: np.ndarray, index_residue: np.ndarray, *,
                            nbins: int = 50,
                            TiltAngle: Any = 90, PanAngle: Any = 0,
                            Location: Any = "Ny-Alesund", Creator: Any = "UAV",
                            LengthSI_x: Any = 50, LengthSI_y: Any = 18,
                            **param_overrides: Any) -> IceImage:
    """``SeaIce_Image_Structure.m`` — pack ch7's pieces and ch8's models into the **Appendix B** structure.

    Book: **Appendix B**, pp. 221–225 (Figs. B.1–B.3 are IDE screenshots of exactly this structure, derived
    from Fig. 8.8); the ``Field.FSD`` block is also §8.2's floe-size distribution.  MATLAB source:
    ``MATLAB_ROOT/ch{6,7}/Sea_Ice_Floe_Identification/SeaIce_Image_Structure.m`` lines 3–115.

    Four blocks:

    * **lines 3–27** ``Floe(i)`` = ``{Center, Area, Perimeter, Polygon{Vertices, Center, Area, Perimeter,
      Intersect}, Pixels}``.  Lines 15–16 ``n = length(v1); v1(n,:) = [];`` delete the duplicated closing vertex
      — which is why the shipped rings are **open** ("There shall be no duplicated vertices", p. 222).
    * **lines 30–46** ``Brash(i)`` = ``{Center, Area, Circle{Radius, Perimeter, Intersect}, Pixels}``; note
      ``Brash.Center`` is the **pixel** centroid from ch7, not the circle centre (they coincide), and
      ``p1 = 2*pi*r1`` is recomputed here rather than copied from ``brash(i).Perimeter``.
    * **lines 49–72** ``Param`` (17 fields).
    * **lines 75–111** ``Field`` (15 fields) with ``CovOther = length(find(index_residue ~= 0))/(NumPix_x*NumPix_y)``
      and the ``FSD`` cell array (analysis C5)::

          inter = fix((max_x - min_x)/nbins);
          [z, n] = hist(floe_area, min_x : inter : max_x);
          int_min = n;   int_max = [n(2:end) - 1, max_x];   num = z;
          FSD{i} = [int_min(i), int_max(i), num(i)];

      **The printed interval labels do not describe the counting rule**: ``n`` are ``hist`` bin *centres*, so
      the real edges are their midpoints and both outer bins are unbounded.  Re-counting the areas from
      ``[int_min, int_max]`` reproduces **none** of the shipped triplets (1861 vs 1411 in the first interval) —
      risk R11.  Reproducing them requires :func:`seaice.core.histogram.hist`, not the labels.

    Parameters
    ----------
    ice_floe, brash_ice : sequence
        ch7 :class:`~seaice.ch07_ice_type.IcePiece` lists (the *real* pieces).
    floe, brash : sequence
        :func:`sea_ice_model` output (the *modelled* pieces).
    coverage : object
        ch7 :class:`~seaice.ch07_ice_type.Coverage` (``IceFloe``/``BrashIce``/``Slush``/``Water``).
    index_floe : ndarray
        Any full-size layer — used only for ``size()`` (lines 50–51).
    index_residue : ndarray
        Fig. 7.16's residue layer, for ``CovOther`` (line 89).
    nbins : int
        ``nbins = 50`` (line 96).
    TiltAngle, PanAngle, Location, Creator, LengthSI_x, LengthSI_y : any
        The metadata the script hard-codes (lines 54–55, 62–63, 78–79).  They are **arguments** here because
        the shipped ``.mat`` contradicts them — it holds ``Creator = 'Helicopter'``, ``PrjName = 'OATRC 2015'``
        and empty lengths, i.e. it came from a §8.3 variant of this §8.2 script (risk R12).
    **param_overrides
        Any other :class:`~seaice.core.icestruct.Param` field (``PrjName``, ``FileName``, ``Time``, …).

    Returns
    -------
    IceImage

    Parity: **exact** — every field of the shipped structure is a direct target, and its 51 ``FSD`` triplets are
    reproduced at 0.
    """
    Floes: list[Floe] = []
    for i, piece in enumerate(ice_floe):
        v1 = np.atleast_2d(np.asarray(floe[i].Vertices, dtype=np.float64))
        v1 = v1[:-1, :]                                     # lines 15-16: v1(n,:) = []
        polygon = Polygon(Vertices=v1, Center=np.asarray(floe[i].Center, dtype=np.float64).ravel(),
                          Area=floe[i].Area, Perimeter=floe[i].Perimeter, Intersect=floe[i].Intersect)
        # `cat(1, ice_floe(i).Center)` / `.Perimeter` keep whatever shape ch7 stored: normally a 1x2 row and a
        # scalar, but a (k, 2) / (k,) block when `regionprops(out == i, ...)` returned k > 1 components
        # (ch07 review S4; 4 of the shipped 3452 brash pieces are such a case).  The shape is preserved.
        per = np.atleast_1d(np.asarray(piece.Perimeter, dtype=np.float64))
        Floes.append(Floe(Center=np.atleast_1d(np.asarray(piece.Center, dtype=np.float64)),
                          Area=int(piece.Area), Perimeter=float(per[0]) if per.size == 1 else per,
                          Polygon=polygon,
                          Pixels=np.atleast_2d(np.asarray(piece.PixelsPosition))))

    Brashes: list[Brash] = []
    for i, piece in enumerate(brash_ice):
        r1 = float(brash[i].Radius)
        circle = Circle(Radius=r1, Perimeter=2.0 * np.pi * r1, Intersect=brash[i].Intersect)   # line 38
        Brashes.append(Brash(Center=np.atleast_1d(np.asarray(piece.Center, dtype=np.float64)),
                             Area=int(piece.Area), Circle=circle,
                             Pixels=np.atleast_2d(np.asarray(piece.PixelsPosition))))

    NumPix_y, NumPix_x = np.asarray(index_floe).shape[:2]    # lines 50-51 (size(.,2) = columns = x)
    param = Param(NumPix_x=int(NumPix_x), NumPix_y=int(NumPix_y),
                  TiltAngle=TiltAngle, PanAngle=PanAngle, Location=Location, Creator=Creator)
    for name, value in param_overrides.items():
        if not hasattr(param, name):
            raise TypeError(f"Param has no field {name!r}")
        setattr(param, name, value)

    # --- lines 91-105: the FSD ------------------------------------------------------------------------------------
    floe_area = np.array([int(p.Area) for p in ice_floe], dtype=np.float64)
    fsd: list[np.ndarray] = []
    if floe_area.size:
        min_x, max_x = float(floe_area.min()), float(floe_area.max())
        inter = float(np.trunc((max_x - min_x) / int(nbins)))         # line 97: fix(...)
        centres = matlab_colon(min_x, inter, max_x)
        z, n_c = matlab_hist(floe_area, centres)
        int_min = n_c
        int_max = np.append(n_c[1:] - 1, max_x)                       # lines 100-101
        for i in range(int_min.size):
            fsd.append(np.array([int_min[i], int_max[i], z[i]], dtype=np.int64))

    pix_scale_x = (LengthSI_x / NumPix_x) if _is_number(LengthSI_x) else None
    pix_scale_y = (LengthSI_y / NumPix_y) if _is_number(LengthSI_y) else None
    pix_area = (pix_scale_x * pix_scale_y) if (pix_scale_x is not None and pix_scale_y is not None) else None
    cov_other = float(np.count_nonzero(np.asarray(index_residue) != 0)) / float(NumPix_x * NumPix_y)

    fld = Field(LengthSI_x=LengthSI_x if _is_number(LengthSI_x) else None,
                LengthSI_y=LengthSI_y if _is_number(LengthSI_y) else None,
                PixScale_x=pix_scale_x, PixScale_y=pix_scale_y, PixArea=pix_area,
                NumFloes=len(floe), NumBrash=len(brash),
                CovFloes=float(coverage.IceFloe), CovBrash=float(coverage.BrashIce),
                CovSlush=float(coverage.Slush), CovWater=float(coverage.Water),
                CovOther=cov_other, FSD=fsd)
    return IceImage(Param=param, Field=fld, Floe=Floes, Brash=Brashes)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def iceimage_to_pieces(ice: IceImage) -> tuple[list[IcePiece], list[IcePiece]]:
    """Turn a loaded :class:`~seaice.core.icestruct.IceImage` back into ch7 ``IcePiece`` lists.

    Not in any ``.m`` file — it is the inverse of :func:`sea_ice_image_structure`'s ``Floe``/``Brash`` blocks
    and lets :func:`sea_ice_model` be re-run on the authors' **own** stored pieces (the §8.3 field), which is
    the strongest available reference for it: the shipped ``Polygon``/``Circle``/``Intersect`` sub-structures
    are then the expected output.

    ``Floe.Pixels`` / ``Brash.Pixels`` are exactly ch7's ``PixelsPosition`` (1-based ``[x, y]``), and
    ``Brash`` has no ``Perimeter`` field in Appendix B, so its ``IcePiece.Perimeter`` is set to ``nan``
    (``sea_ice_model`` never reads it).
    """
    floes = [IcePiece(Center=f.Center, Area=int(f.Area), Perimeter=float(f.Perimeter),
                      PixelsPosition=np.asarray(f.Pixels), label=i + 1) for i, f in enumerate(ice.Floe)]
    brash = [IcePiece(Center=b.Center, Area=int(b.Area), Perimeter=float("nan"),
                      PixelsPosition=np.asarray(b.Pixels), label=i + 1) for i, b in enumerate(ice.Brash)]
    return floes, brash


# ==================================================================================================================
# §8.2 / §8.3 — color_hist.m and color_hist_comparison.m
# ==================================================================================================================

@dataclass
class ColorHist:
    """Everything ``color_hist.m`` computes (the HG1 bar-colouring block is display; see the DEVIATION note)."""

    z: np.ndarray                  #: ``[z, n] = hist(floe_area, min_x:inter:max_x)`` — the bar heights
    centers: np.ndarray            #: ``n`` **as returned by hist** (the bin centres)
    n: np.ndarray                  #: ``n + inter/2`` — the *shifted* bar positions the M-file plots and colours
    color: np.ndarray              #: ``color(i) = fix((1 - exp(-n(i)/1000))*10000)`` on the **shifted** centres
    color_floe: np.ndarray         #: Eq. (7.6) colour of every input area (line 3; used by the map, not the bars)
    color_min: int                 #: ``fix((1 - exp(-min_x/1000))*10000)``
    color_max: int
    tick_values: np.ndarray        #: ``ysh = color_min : fix((color_max-color_min)/nn) : color_max``
    tick_labels: np.ndarray        #: ``YT{1,i} = -round(1000*log(1 - ysh(i)/10000))`` — the printed integers
    nbins: int
    params: dict


@dataclass
class ColorHistComparison(ColorHist):
    """``color_hist_comparison.m``: the same block twice plus ``z_d = z - z0``."""

    z0: np.ndarray = field(default_factory=lambda: np.zeros(0))      #: histogram of the **raw** (pixel) areas
    n0: np.ndarray = field(default_factory=lambda: np.zeros(0))      #: its shifted centres (identical to ``n``)
    z_d: np.ndarray = field(default_factory=lambda: np.zeros(0))     #: ``z - z0`` — Fig. 8.15


def _color_hist_core(areas, min_x: float, inter: float, max_x: float, nn: int) -> dict:
    """The arithmetic shared by ``color_hist.m`` lines 2–46 and ``color_hist_comparison.m`` lines 2–55."""
    a = np.asarray(areas, dtype=np.float64).ravel()
    color_floe = size_color(a)                                        # line 3 (Eq. 7.6, analysis C6)
    color_min = int(size_color(min_x))                                # line 9
    color_max = int(size_color(max_x))                                # line 10
    z, centers = matlab_hist(a, matlab_colon(min_x, inter, max_x))     # line 18
    n = centers + inter / 2.0                                          # line 20 — the +35 shift
    color = size_color(n)                                              # line 30
    values, labels = colorbar_area_ticks(np.array([color_min, color_max], dtype=np.float64), nn)  # lines 36-45
    return {"z": z, "centers": centers, "n": n, "color": color, "color_floe": color_floe,
            "color_min": color_min, "color_max": color_max,
            "tick_values": values, "tick_labels": labels, "nbins": int(z.size)}


def color_hist(areas, *, min_x: float = 20, inter: float = 70, max_x: float = 3500, nn: int = 8) -> ColorHist:
    """``color_hist.m`` — the floe-size histogram of **Figs. 8.11 and 8.14**, coloured by Eq. (7.6).

    Book: §8.2 (pp. 186–188).  MATLAB source:
    ``MATLAB_ROOT/ch{6,7}/Sea_Ice_Floe_Identification/color_hist.m`` lines 2–46::

        floe_area  = cat(1, floe.Area);
        color_floe = fix( (1 - exp(-floe_area/1000)) * 10000 );
        min_x = 20;  inter = 70;  max_x = 3500;
        color_min = fix((1 - exp(-min_x/1000))*10000);   color_max = fix((1 - exp(-max_x/1000))*10000);
        [z, n] = hist(floe_area, min_x : inter : max_x);
        nbins = length(z);   n = n + inter/2;
        h = bar(n(1:nbins), z(1:nbins));  ...
        for i = 1:nbins,  color(i) = fix((1 - exp(-n(i)/1000))*10000);  end
        nn = 8;  d = fix((color_max-color_min)/nn);  ysh = min(color_min) : d : max(color_max);
        YT{1,i} = -round(1000*log(1 - ysh(i)/10000));

    The **same file drives both histograms**: pass the *raw* piece areas (ch7 ``ice_floe``) for Fig. 8.11 and
    the *model* polygon areas (:func:`sea_ice_model`'s ``floe``) for Fig. 8.14.

    ``n = n + inter/2`` (line 20) shifts the bar positions by **+35**, so the colour of bar ``i`` is computed
    from the *shifted* centre — reproduced.  ``[zs, izs] = sortrows(z', 1)`` (line 26) is dead code and is not
    ported.

    # DEVIATION: display only — lines 23-25 and 31-33 use **HG1** handle graphics
    # (``ch = get(h,'Children'); fvd = get(ch,'Faces'); fvcd = get(ch,'FaceVertexCData'); ...
    # set(ch,'FaceVertexCData',fvcd)``) to colour the individual bars.  ``get(bar,'Children')`` returns an
    # empty ``GraphicsPlaceholder`` in R2025a and the next ``get`` errors, so the block cannot run at all in a
    # current MATLAB (the same failure ch07 hit in ``ice_shape_enhancement.m``).  Every **number** it consumes
    # (``color``, ``z``, ``n``) is computed here and returned; the drawing is done with matplotlib
    # (``scripts/ch08_color_hist.py``), which colours each bar with ``jet`` at the same ``color(i)``.

    Notes
    -----
    The printed tick list of Figs. 8.11/8.14/8.15 (20, 149, 297, 471, 682, 950, 1317, 1902, 3487) follows from
    the **literals** ``min_x = 20``, ``max_x = 3500``, ``nn = 8`` alone and contains no information about the
    data (risk R10, ch07 lesson 5) — never quote it as parity evidence.

    Parity: **exact** for ``z``, ``centers``, ``n``, ``color`` and the tick integers.
    """
    d = _color_hist_core(areas, min_x, inter, max_x, nn)
    return ColorHist(params={"min_x": min_x, "inter": inter, "max_x": max_x, "nn": nn}, **d)


def color_hist_comparison(areas_model, areas_raw, *, min_x: float = 20, inter: float = 70,
                          max_x: float = 6000, nn: int = 8) -> ColorHistComparison:
    """``color_hist_comparison.m`` — **Fig. 8.15**, the polygon histogram *minus* the pixel histogram.

    Book: §8.2, p. 188 ("floe size distribution error … due to the shape simplification").  MATLAB source:
    ``MATLAB_ROOT/ch{6,7}/Sea_Ice_Floe_Identification/color_hist_comparison.m`` lines 2–55.  It is
    ``color_hist.m`` with a second ``hist`` (line 25) and ``z_d = z - z0`` (line 29), on the **same** centres —
    unlike ``main_WL_new.m``'s ``count_error``, which shifts by one bin (analysis C1, risk R20).

    Parameters
    ----------
    areas_model : array_like
        ``cat(1, floe.Area)`` — the polygonized (``sea_ice_model``) areas.
    areas_raw : array_like
        ``cat(1, ice_floe.Area)`` — the identified pixel-count areas.
    max_x : float
        **6000 in the shipped file** (line 10), against ``color_hist.m``'s 3500.  The tick list printed under
        Fig. 8.15 (20 … 3487) is the one ``max_x = 3500`` produces, so the figure was made with the *other*
        file's constant — a shipped-code / printed-figure discrepancy (risk R9).  The default here is the
        shipped 6000; pass ``max_x=3500`` to reproduce the printed colour bar.

    Parity: **exact** for ``z``, ``z0``, ``z_d``, ``n`` and the tick integers.  The same HG1 DEVIATION as
    :func:`color_hist` applies (lines 32-34, 40-42).
    """
    d = _color_hist_core(areas_model, min_x, inter, max_x, nn)
    a0 = np.asarray(areas_raw, dtype=np.float64).ravel()
    z0, n0 = matlab_hist(a0, matlab_colon(min_x, inter, max_x))         # line 25
    n0 = n0 + inter / 2.0                                               # line 27
    return ColorHistComparison(params={"min_x": min_x, "inter": inter, "max_x": max_x, "nn": nn},
                               z0=z0, n0=n0, z_d=d["z"] - z0, **d)


# ==================================================================================================================
# §8.1 — the shipborne camera system (text only: no MATLAB code and no data ship for this section)
# ==================================================================================================================

@dataclass
class ShipborneIC:
    """One frame's ice concentration by one method (§8.1.2)."""

    concentration: float           #: ice pixels / total pixels
    mask: np.ndarray               #: the ice mask
    method: str
    level: float | None = None     #: the Otsu level (``graythresh``), when applicable
    centers: np.ndarray | None = None   #: the k-means cluster centres, sorted ascending
    labels: np.ndarray | None = None    #: the k-means label image (0 = darkest cluster)


def shipborne_ice_concentration(frame: np.ndarray, method: str = "otsu", *, k: int = 3,
                                ice_clusters: str | int = "top2", impl: str = "authors",
                                shift_bug: bool = False, seed: int = 0) -> ShipborneIC:
    """§8.1.2 "Methods", pp. 176–179 — ice concentration of one rectified shipborne frame.

    Book text only — **§8.1 ships no MATLAB code and no images** (every figure is credited to Lu, Zhang,
    Lubbad, Løset & Skjetne, OTC Arctic Technology Conference 2016).  The two methods it names are already
    ported: the global Otsu threshold of §3.1 and the k-means of §3.2.

    * ``method='otsu'`` — ``bw = im2bw(gray, graythresh(gray))``; **white = ice** (:mod:`seaice.core.threshold`).
    * ``method='kmeans'`` — ``k = 3`` clusters read as water (black) / **wet ice** (grey: rubble, young ice,
      melt ponds) / dry ice (white); ``ice_clusters='top2'`` counts **both the white and the grey cluster** as
      ice, which is why the book's k-means series sits systematically above the Otsu one (Fig. 8.5, p. 180).
      Pass an integer to count the brightest ``m`` clusters instead.  "If we choose two clusters for the
      k-means method (k = 2), then this method is approximately reduced to the Otsu thresholding method"
      (p. 179).

    Ice concentration is ice pixels / total pixels (there is no area weighting: the frame is already
    rectified, and the two white triangles are cropped away before this is called — p. 178).

    Parameters
    ----------
    frame : ndarray
        One rectified frame, RGB or gray (``rgb2gray_matlab`` is applied to RGB).
    method : {'otsu', 'kmeans'}
    k : int
        Number of clusters for ``method='kmeans'`` (the book's §8.1 choice is 3, p. 179).
    ice_clusters : {'top2', 'top1'} or int
        How many of the **brightest** clusters count as ice.  Identical rule in both ``impl`` branches, so the
        two are interchangeable.
    impl : {'authors', 'lloyd'}
        Which k-means.  ``'authors'`` (**default**) is the book's own §3.2/§3.3 routine,
        :func:`seaice.core.clustering.kmeans_gray` — a line-by-line port of ``MATLAB_ROOT/ch3/kmeans.m``,
        **deterministic** (equal-division initialisation ``mu = (1:k)·m/(k+1)``, histogram-weighted centroid
        update, exact ``mu == oldmu`` stop), so no seed is involved.  It requires an **integer-valued** gray
        image, exactly as ``kmeans.m`` does (it indexes ``h(ima(i))``).  ``'lloyd'`` routes through
        :func:`seaice.core.clustering.kmeans_lloyd` with ``init='kmeans++'`` and ``seed`` — the generic
        Lloyd/Statistics-Toolbox-style routine, kept for continuous data and for comparison.
    shift_bug : bool
        Only for ``impl='authors'``: ``False`` (default, ch03's default) is the text-consistent branch;
        ``True`` reproduces ``kmeans.m`` lines 64–70 literally (the unshifted image compared with the shifted
        centroids), which is what every k-means number *printed in ch3* requires.
    seed : int
        Meaningful **only** for ``impl='lloyd'`` (the ``'authors'`` routine has no RNG).

    Parity
    ------
    ``method='otsu'`` — **exact** (ch03 ``graythresh``/``im2bw``).
    ``method='kmeans', impl='authors'`` — **exact** (ch03's port of the authors' ``kmeans.m``; MATLAB source
    ``MATLAB_ROOT/ch3/kmeans.m``, verified in ``reports/ch03_verification.md``).
    ``method='kmeans', impl='lloyd'`` — **approx** (a different algorithm from the authors' routine, with an
    RNG-seeded k-means++ initialisation).
    The §8.1 **results** stay **unverified** in either case: no §8.1 data and no §8.1 ``.m`` file ship, so the
    6-hour series of Fig. 8.5, Event #1 and Event #2 cannot be reproduced (``reports/ch08_verification.md``,
    open item O1).  The linear 4-corner rectification of Appendix A.1.2 that precedes this step is **deferred
    to ch10** — do not write a second rectifier here.
    """
    from .core.clustering import kmeans_gray, kmeans_lloyd
    from .core.matlab_compat import rgb2gray_matlab
    from .core.threshold import graythresh, im2bw

    img = np.asarray(frame)
    gray = rgb2gray_matlab(img) if img.ndim == 3 else img

    if method == "otsu":
        level, _ = graythresh(gray)
        mask = im2bw(gray, level)
        return ShipborneIC(concentration=float(mask.mean()), mask=mask, method="otsu", level=float(level))

    if method != "kmeans":
        raise ValueError("method must be 'otsu' or 'kmeans'")

    if impl == "authors":
        # The book's own routine (§3.2/§3.3, `MATLAB_ROOT/ch3/kmeans.m`): deterministic, no seed.
        res = kmeans_gray(gray, k=k, shift_bug=shift_bug)
        centers = np.asarray(res.centroids, dtype=np.float64).ravel()
        raw_labels = np.asarray(res.mask).ravel() - 1          # `mask` is 1..k
    elif impl == "lloyd":
        # PARITY: approx — generic Lloyd with a seeded k-means++ start; not the authors' routine.
        X = gray.astype(np.float64).reshape(-1, 1)
        res = kmeans_lloyd(X, k, init="kmeans++", seed=seed)
        centers = np.asarray(res.centers, dtype=np.float64).ravel()
        raw_labels = np.asarray(res.labels).ravel()
    else:
        raise ValueError("impl must be 'authors' or 'lloyd'")

    order = np.argsort(centers)                       # darkest ... brightest
    rank = np.empty_like(order)
    rank[order] = np.arange(k)
    labels = rank[raw_labels].reshape(gray.shape)
    m = 2 if ice_clusters == "top2" else (1 if ice_clusters == "top1" else int(ice_clusters))
    mask = labels >= (k - m)          # the `m` brightest clusters count as ice
    return ShipborneIC(concentration=float(mask.mean()), mask=mask, method=f"kmeans(k={k}, {impl})",
                       centers=np.sort(centers), labels=labels)


# ==================================================================================================================
# §8.2 — "Algorithms 3, 4 and 5 are carried out directly" (p. 182)
# ==================================================================================================================

def sea_ice_field(rgb: np.ndarray, *, cache: Any = None, verbose: bool = True, **params: Any):
    r"""Run the ch6/ch7 pipeline on the §8.2 image — the input every §8.2 figure needs.

    Book: §8.2, p. 182 — "Since the image is undistorted, **Algorithms 3, 4 and 5 are carried out directly**",
    i.e. the whole ch7 chain with no rectification and no tiling.  There is no ``.m`` file for this step in
    ``ch8/``: it is ``sea_ice_demo.m``'s first two calls, already ported
    (:func:`seaice.ch07_ice_type.sea_ice_edge_detection` → :func:`seaice.ch07_ice_type.ice_shape_enhancement`).
    The result feeds :func:`sea_ice_model`, :func:`sea_ice_image_structure` and :func:`color_hist`.

    Parameters
    ----------
    rgb : ndarray
        Figure 8.8 — which is ch6/ch7's ``sea_ice_test.jpg`` (analysis finding 5: the printed bitmap correlates
        NCC +0.982 with the **inverted transpose** of it, and +0.974 with the Fig. 7.22 bitmap).
    cache : path-like, optional
        ``.npz`` holding Algorithm 3's ``seg``/``bk`` (it costs ~45 s and three ch8 scripts need it).
    **params
        Overrides for :data:`seaice.ch06_gvf_snake.BOOK_PARAMS`\ ``["sea_ice_demo"]``.

    Returns
    -------
    seaice.ch07_ice_type.IceShapeEnhancement

    Notes
    -----
    The book's §8.2 counts (**498 ice floes, 201 brash pieces**, 76.73 / 0.46 / 9.05 / 13.76 %) are expected
    **not** to reproduce: ch07 measured 433 floes / 274 brash on the same image with the shipped
    ``sea_ice_demo.m`` parameters, and the authors' §8.2 parameter set is not printed anywhere (risk R7).
    Report both, do not search parameters until they match.

    Parity: **unverified** — no ``.m`` file ships for this driver and the printed counts do not reproduce
    (``reports/ch08_verification.md`` open item O2); its two constituents are ch07's
    :func:`~seaice.ch07_ice_type.sea_ice_edge_detection` and
    :func:`~seaice.ch07_ice_type.ice_shape_enhancement`, verified in ch07.
    """
    from pathlib import Path as _Path

    from .ch06_gvf_snake import BOOK_PARAMS
    from .ch07_ice_type import ice_shape_enhancement, sea_ice_edge_detection

    P = dict(BOOK_PARAMS["sea_ice_demo"])
    P.update(params)
    seg = bk = None
    if cache is not None and _Path(cache).exists():
        z = np.load(cache)
        seg, bk = z["seg"], z["bk"]
        if verbose:
            print(f"Algorithm 3: reused the cached seg/bk from {_Path(cache).name}")
    if seg is None:
        res = sea_ice_edge_detection(rgb, kms0=P["kms0"], sigma=P["sigma"], GradientOn=P["GradientOn"],
                                     GVFOn=P["GVFOn"], Num=P["Num"], mu=P["mu"], iter=P["iter"],
                                     alpha=P["alpha"], beta=P["beta"], gamma=P["gamma"], kappa=P["kappa"],
                                     Dmin=P["Dmin"], Dmax=P["Dmax"], Ra_min=P["Ra_min"], Ra=P["Ra"],
                                     Rc=P["Rc"], Rl=P["Rl"], se_radius=P["se_radius"], timer=P["timer"],
                                     keep_history=False)
        seg, bk = res.out, res.bk
        if cache is not None:
            _Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, seg=seg, bk=bk)
    return ice_shape_enhancement(bk, seg, min_floe=P["min_floe"], min_brash=P["min_brash"], se_th=P["se_th"])
