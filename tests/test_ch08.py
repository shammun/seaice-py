"""Chapter 8 verification tests — sea ice image processing applications.

Evidence levels (see the ``verify-port`` skill):

* **L1** synthetic truth — small arrays whose answer follows from the book's own equations or from a
  geometric fact, plus fixtures **constructed to discriminate** a fix from the bug it replaces.
* **L2** MATLAB parity — the ORIGINAL ``.m`` code of ``MATLAB_ROOT/ch8/MCD`` and of
  ``MATLAB_ROOT/ch7/Sea_Ice_Floe_Identification`` run through ``reference/ch08/make_refs.py``
  (MATLAB R2025a, ``matlab -batch``, graphics-only patches recorded in ``reference/ch08/patches.json``).
* **L3** the authors' **own shipped output** — ``data/book/ch08/MCD/IceImage_290915_2_jpg.0000179.mat``
  (the Appendix-B structure of the §8.3 field) and ``MCD_results.mat``.
* **L4** numbers quoted in the chapter text (2888 floes, 58.00/4.85/21.21/15.94 %, α = 1.3704, …).

Run: ``.venv/Scripts/python.exe -m pytest tests/test_ch08.py -q -p no:cacheprovider``
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.ch08 import fixtures as FX  # noqa: E402
from seaice import ch08_applications as ch8  # noqa: E402
from seaice.ch07_ice_type import Coverage, IcePiece  # noqa: E402
from seaice.core.fitting import (LsqOptions, MATLAB_LSQ_DEFAULTS, lsqcurvefit, optimset, power_law,  # noqa: E402
                                 truncated_power_law, weibull_survival)
from seaice.core.histogram import hist as mhist  # noqa: E402
from seaice.core.icestruct import load_iceimage_mat, overlap_graph, save_iceimage_mat  # noqa: E402
from seaice.core.matlab_compat import matlab_colon  # noqa: E402  (review S5: moved out of ch08)
from seaice.core.plotting import matlab_jet  # noqa: E402
from seaice.core.polygon import polygeom, polyxpoly  # noqa: E402
from seaice.core.stats import cumulative_size_distribution, mean_caliper_diameter  # noqa: E402

REF = ROOT / "reference/ch08"
DATA = ROOT / "data/book/ch08/MCD"
VERIFY = ROOT / "outputs/ch08/verify"
PY = ROOT / ".venv/Scripts/python.exe"
SHIPPED = DATA / "IceImage_290915_2_jpg.0000179.mat"
MCD_RESULTS = DATA / "MCD_results.mat"

_CACHE: dict[str, dict] = {}
_ICE = {}

needs_book = pytest.mark.skipif(not SHIPPED.exists(),
                                reason="private book data absent (data/book/ch08/MCD/IceImage_*.mat)")


def ref(name: str) -> dict:
    """Load ``reference/ch08/<name>.mat`` once, or skip when it has not been generated."""
    if name not in _CACHE:
        p = REF / f"{name}.mat"
        if not p.exists():
            pytest.skip(f"run reference/ch08/make_refs.py (needs MATLAB R2025a): {p.name} missing")
        _CACHE[name] = loadmat(str(p))
    return _CACHE[name]


def flat(d: dict, key: str) -> np.ndarray:
    return np.asarray(d[key]).ravel()


def scal(d: dict, key: str) -> float:
    return float(np.asarray(d[key]).ravel()[0])


def cells(v) -> list[np.ndarray]:
    """A MATLAB cell array of matrices → a list of 2-D float arrays."""
    return [np.atleast_2d(np.asarray(x, dtype=np.float64)) for x in np.asarray(v).ravel()]


def index_sets(v) -> list[set]:
    """A MATLAB cell array of index columns → a list of sets of 1-based ints."""
    return [set(np.asarray(x).ravel().astype(int).tolist()) for x in np.asarray(v).ravel()]


def shipped_ice():
    """The Appendix-B structure of the §8.3 field, loaded once (L3 gold reference)."""
    if "ice" not in _ICE:
        if not SHIPPED.exists():
            pytest.skip("private book data absent")
        _ICE["ice"] = load_iceimage_mat(SHIPPED)
    return _ICE["ice"]


def pieces_of(fx: dict) -> tuple[list[IcePiece], list[IcePiece]]:
    def mk(p):
        return IcePiece(Center=p["Center"], Area=int(p["Area"]), Perimeter=float(np.ravel(p["Perimeter"])[0]),
                        PixelsPosition=p["PixelsPosition"])
    return [mk(p) for p in fx["ice_floe"]], [mk(p) for p in fx["brash_ice"]]


# =====================================================================================================================
# L1 — Eq. (8.1) and Eq. (8.2):  seaice.core.stats
# =====================================================================================================================

def test_mcd_is_the_area_equivalent_circle_diameter():
    """L1 — Eq. (8.1) ``L = sqrt(4A/pi)`` must invert ``A = pi r^2`` exactly (the book calls it a *caliper*)."""
    for r in (0.5, 1.0, 3.0, 17.25):
        assert mean_caliper_diameter(np.pi * r ** 2) == pytest.approx(2 * r, rel=0, abs=1e-12)
    # `scale` multiplies the AREA by scale**2 before the root -> the diameter scales linearly
    assert mean_caliper_diameter(100.0, 1.1794) == pytest.approx(mean_caliper_diameter(100.0) * 1.1794, abs=1e-12)
    # and it is NOT a caliper (support-function) diameter: for a 1xN pixel bar the caliper diameter is N
    assert mean_caliper_diameter(100.0) == pytest.approx(11.283791670955126, abs=1e-12)


def test_cumulative_fsd_ties_and_endpoints():
    """L1 — Eq. (8.2): monotone, ``Nc(min) = 1``, ``Nc(max) = 1/N``, and **ties repeat** (Fig. 8.21's stacks)."""
    L, Nc = cumulative_size_distribution([3.0, 1.0, 2.0, 2.0, 5.0])
    assert np.array_equal(L, [1.0, 2.0, 2.0, 3.0, 5.0])
    assert np.allclose(Nc, [1.0, 0.8, 0.8, 0.4, 0.2])
    assert Nc[0] == 1.0 and Nc[-1] == pytest.approx(1 / 5)
    assert np.all(np.diff(Nc) <= 0)
    # the O(N^2) loop of the M-file, literally, must agree with the searchsorted shortcut
    m = np.sort(np.array([3.0, 1.0, 2.0, 2.0, 5.0]))
    slow = np.array([(m >= v).sum() / m.size for v in m])
    assert np.array_equal(Nc, slow)


def test_cumulative_fsd_empty():
    L, Nc = cumulative_size_distribution([])
    assert L.size == 0 and Nc.size == 0


# =====================================================================================================================
# L1/L2 — the MATLAB colon operator (sea_ice_model.m line 25, risk R18)
# =====================================================================================================================

def test_matlab_colon_counts_and_last_value():
    """L1 — ``0:0.05:6.28`` is **126** points ending at **6.25**; ``np.arange`` is not a substitute."""
    t = ch8.matlab_colon(0.0, 0.05, 6.28)
    assert t.size == 126 and t[-1] == pytest.approx(6.25, abs=1e-12)
    assert t[-1] < 2 * np.pi          # the "circle" never closes: 6.28 < 2*pi
    # the discriminating case: a naive np.arange DROPS the endpoint of 0:0.1:1
    assert ch8.matlab_colon(0.0, 0.1, 1.0).size == 11
    assert np.arange(0.0, 1.0, 0.1).size == 10, "fixture no longer discriminates np.arange from the colon"
    assert ch8.matlab_colon(0.0, 0.1, 1.0)[-1] == 1.0     # snapped exactly, as MATLAB does
    assert ch8.matlab_colon(5.0, 1.0, 1.0).size == 0      # descending with a positive step -> empty
    assert ch8.matlab_colon(1.0, 0.0, 5.0).size == 0      # zero step -> empty (color_hist's d = 0 case)


def test_matlab_colon_parity():
    """L2 — element-wise against MATLAB's own colon vectors."""
    d = ref("misc")
    assert int(scal(d, "t_n")) == 126 and scal(d, "t_last") == pytest.approx(6.25, abs=0)
    t = ch8.matlab_colon(0.0, 0.05, 6.28)
    ml = flat(d, "t_circle")
    assert t.size == ml.size
    # DEVIATION: <= 1 ulp.  MATLAB's colon is more accurate than `start + k*step`; 30 of the 126 angles differ
    # by exactly one ulp.  test_circle_angles_ulp_is_harmless bounds the consequence.
    assert np.abs(t - ml).max() <= 1e-15
    ulps = np.abs(t.view(np.int64) - ml.view(np.int64))
    assert ulps.max() <= 1
    # the two INTEGER colon vectors this chapter uses are bit-exact
    assert np.array_equal(ch8.matlab_colon(20, 70, 3500), flat(d, "colon_20_70_3500"))
    assert np.array_equal(ch8.matlab_colon(21, 79, 3971), flat(d, "colon_21_79_3971"))


def test_matlab_colon_lives_in_core_and_is_re_exported():
    """L1 — review item S5: the colon primitive now lives in ``core.matlab_compat``; ``ch8.matlab_colon`` is
    the *same object*, so every existing call site and test keeps testing the one implementation."""
    assert ch8.matlab_colon is matlab_colon
    assert "matlab_colon" in ch8.__all__


def test_fsd_colon_endpoint_is_truncated_parity():
    """L2 — review item N4: the line ``SeaIce_Image_Structure.m`` line 98 really evaluates for the shipped
    field is ``min_x : inter : max_x`` = **21:79:3979**, not ``21:79:3971``.

    ``(3979-21)/79 = 50.1`` is **not** integral, so this case — unlike the 3971 one, whose quotient is exactly
    50 — exercises the colon's truncation rule: MATLAB returns **51** elements ending at **3971**, i.e. the
    stated endpoint 3979 is *never reached*.  Asserted against MATLAB's own vector and pinned in the other
    direction (a rule that snapped the last element to `stop` would give 3979).
    """
    d = ref("misc")
    ml = flat(d, "colon_21_79_3979")
    ours = matlab_colon(21, 79, 3979)
    assert int(scal(d, "colon_21_79_3979_n")) == 51 and scal(d, "colon_21_79_3979_last") == 3971.0
    assert np.array_equal(ours, ml)
    assert ours.size == 51 and ours[-1] == 3971.0 and ours[-1] != 3979.0
    # ... and it is the vector the port feeds to `hist` for the shipped FSD (51 triplets, test below)
    assert np.array_equal(ours, matlab_colon(21.0, float(np.trunc((3979 - 21) / 50)), 3979.0))


def test_x_plot_colon_parity():
    """L2 — review item N9: ``fitting_iceFloes_distribution.m`` line 45 ``x_plot = min(x):0.001:max(x)*1.5``.

    119 822 elements with a non-integer step — by far the longest colon in the chapter and the only one whose
    length is decided by floating-point rounding of ``(stop-start)/step``.  It was saved by ``make_refs.py``
    but asserted by no test until now.
    """
    d = ref("fit")
    ml = flat(d, "x_plot")
    mcd = flat(d, "MCD")
    ours = matlab_colon(mcd.min(), 0.001, mcd.max() * 1.5)
    assert ours.size == ml.size == 119822
    assert ours[0] == ml[0]
    ulps = np.abs(ours.view(np.int64) - ml.view(np.int64))
    assert ulps.max() <= 1, f"max {ulps.max()} ulp"
    assert np.abs(ours - ml).max() <= 1e-13


@needs_book
def test_circle_angles_ulp_is_harmless():
    """L2 — replacing our ``t`` by MATLAB's own must change **nothing** in ``sea_ice_model`` (bounds the ulp)."""
    d = ref("misc")
    fx = FX.model_fixtures()["nested"]
    ifl, ibr = pieces_of(fx)
    img = np.zeros(fx["shape"])
    a = ch8.sea_ice_model(ifl, ibr, img)
    ml_t = flat(d, "t_circle")
    saved = ch8.CIRCLE_ANGLES
    try:
        ch8.CIRCLE_ANGLES = ml_t
        b = ch8.sea_ice_model(ifl, ibr, img, )
    finally:
        ch8.CIRCLE_ANGLES = saved
    # the default argument of `_circle` was bound at import time, so pass t explicitly through the raster too
    assert int((a.bw_brash != b.bw_brash).sum()) == 0
    xa, ya, ra = ch8._circle(np.array([10.0, 20.0]), 100.0, ch8.CIRCLE_ANGLES)
    xb, yb, rb = ch8._circle(np.array([10.0, 20.0]), 100.0, ml_t)
    assert max(np.abs(xa - xb).max(), np.abs(ya - yb).max()) < 1e-14


# =====================================================================================================================
# L1/L2 — jet(m):  seaice.core.plotting.matlab_jet
# =====================================================================================================================

@pytest.mark.parametrize("m", [1, 3, 4, 7, 30, 255])
def test_matlab_jet_parity(m):
    """L2 — ``jet(m)`` bit-for-bit against R2025a, including ``jet(30)`` whose rows are painted into Fig. 8.19."""
    d = ref("misc")
    assert np.array_equal(matlab_jet(m), np.asarray(d[f"jet{m}"], dtype=np.float64))


def test_matlab_jet_mod4_rule_is_the_discriminating_one():
    """L1 — the port uses R2025a's ``mod(m,4)==1``; assert the **old** ``mod(m,2)==1`` rule gives a *different*
    table for ``m = 255`` (so :func:`test_matlab_jet_parity` at 255 really tests the rule, ch07 lesson)."""
    def jet_old(m: int) -> np.ndarray:
        n = int(np.ceil(m / 4))
        u = np.concatenate([np.arange(1, n + 1) / n, np.ones(max(n - 1, 0)), np.arange(n, 0, -1) / n])
        base = int(np.ceil(n / 2)) - (1 if m % 2 == 1 else 0)       # the pre-R2014 rule
        g = base + np.arange(1, u.size + 1)
        r, b = g + n, g - n
        g, r, b = g[g <= m], r[r <= m], b[b >= 1]
        J = np.zeros((m, 3))
        J[r - 1, 0] = u[:r.size]
        J[g - 1, 1] = u[:g.size]
        J[b - 1, 2] = u[u.size - b.size:]
        return J
    assert not np.array_equal(jet_old(255), matlab_jet(255)), "255 does not discriminate the two rules"
    assert np.array_equal(jet_old(30), matlab_jet(30)), "m = 30 is congruent to 2 (mod 4): both rules agree"


# =====================================================================================================================
# L1/L2 — plot_color_bar_and_floe.m: the nearest-centre colour index (analysis C11)
# =====================================================================================================================

def test_nearest_centre_index_tie_goes_to_the_lower_centre():
    """L1 — ``[~, i] = min(abs(v - centres))`` keeps the **first** minimiser, where ``round`` would go up."""
    c = np.arange(1.0, 101.0)
    assert ch8._nearest_centre_index(6.5, c) == 6            # MATLAB round(6.5) = 7 -> would be centre 7
    assert ch8._nearest_centre_index(6.4999, c) == 6
    assert ch8._nearest_centre_index(6.5001, c) == 7
    assert ch8._nearest_centre_index(-5.0, c) == 1           # below the first centre
    assert ch8._nearest_centre_index(1000.0, c) == 100       # above the last
    # the discriminating statement: MATLAB's own round would disagree at the exact half-metre
    assert int(np.floor(6.5 + 0.5)) == 7 != ch8._nearest_centre_index(6.5, c)


def test_matlab_isfield_branch_selection():
    """L1 — ``isfield(Poly_struc,'Vertices')`` picks the branch; an object with neither field raises."""
    class V:
        Vertices = np.zeros((3, 2))
        Center = np.zeros(2)

    class P:
        Pixels = np.ones((3, 2))
        Center = np.zeros(2)

    class N:
        Center = np.zeros(2)

    assert ch8._matlab_isfield([V()], "Vertices") and not ch8._matlab_isfield([P()], "Vertices")
    assert not ch8._matlab_isfield([], "Vertices")
    with pytest.raises(ValueError, match="field name is incorrect"):
        ch8.plot_color_bar_and_floe(30, [1.0], 10, 1, [N()])


# =====================================================================================================================
# L2/L3 — main_WL_new.m + plot_color_bar_and_floe.m on the shipped §8.3 structure
# =====================================================================================================================

@needs_book
def test_mcd_analysis_parity():
    """L2 — every array ``main_WL_new.m`` computes, against MATLAB running the (cd-patched) original."""
    d = ref("mcd")
    a = ch8.mcd_analysis(shipped_ice())
    assert int(scal(d, "N")) == 2888 and int(scal(d, "Y_limi")) == 627
    for key, py in (("Poly_Area", a.poly_area), ("Poly_MCD", a.poly_mcd),
                    ("Raw_Area", a.raw_area), ("Raw_MCD", a.raw_mcd),
                    ("Poly_counts", a.poly_counts), ("Poly_centers", a.poly_centers),
                    ("Raw_counts", a.raw_counts), ("Raw_centers", a.raw_centers),
                    ("count_error", a.count_error),
                    ("Poly_index", a.poly.color_index), ("Raw_index", a.raw.color_index)):
        ml = flat(d, key)
        assert np.array_equal(np.asarray(py, dtype=np.float64).ravel(), ml), f"{key} differs"
    assert np.array_equal(a.raw.color_m, np.asarray(d["color_M"], dtype=np.float64))
    # caxis([centers(1) centers(30)]) = [1 30] -> the "5, 10, ... >=30 [m]" bar of Figs. 8.19/8.20 (N16)
    for key in ("Poly_caxis", "Raw_caxis", "caxis_lim_err"):
        assert np.array_equal(flat(d, key), [1.0, 30.0])


@needs_book
def test_plot_color_bar_and_floe_pixels_branch_parity():
    """L2 — Figure 8.19: the grown ``rgbImage``, its size, the white-dot coordinates, all bit-exact."""
    d = ref("mcd")
    a = ch8.mcd_analysis(shipped_ice())
    # Risk R17.  On the FULL field both pixel maxima are attained, so this assertion alone cannot distinguish
    # "grown from the painted pixels" from "pre-allocated to the image size" (review item S4) — that is what
    # `test_rgb_image_is_grown_not_preallocated` does on two subsets whose maxima are *not* attained.
    assert np.array_equal(flat(d, "rgb_size"), [627, 1114, 3])
    assert a.raw.rgb_image.shape == (627, 1114, 3)
    p = REF / "mcd_rgb.mat"
    if p.exists():
        ml = np.asarray(loadmat(str(p))["Raw_rgb_u8"])
        py = np.clip(np.round(a.raw.rgb_image * 255), 0, 255).astype(np.uint8)
        assert int((ml != py).sum()) == 0
    assert np.array_equal(np.asarray(d["Raw_dotxy"], dtype=np.float64), a.raw.centres_xy)
    assert np.array_equal(np.asarray(d["Raw_centre"], dtype=np.float64), a.raw.centres_xy)


@needs_book
def test_plot_color_bar_and_floe_vertices_branch_parity():
    """L2 — the ``Vertices`` branch: ``fill(x*lop, (Y_limi-y)*lop)`` coordinates, colours and white dots."""
    d = ref("mcd")
    a = ch8.mcd_analysis(shipped_ice())
    assert scal(d, "poly_rgb_isempty") == 1.0            # the Vertices branch never touches rgbImage
    assert np.array_equal(np.asarray(d["Poly_polyc"], dtype=np.float64),
                          a.poly.color_m[a.poly.color_clamped - 1])
    px = [np.asarray(x, dtype=np.float64).ravel() for x in np.asarray(d["Poly_polyx20"]).ravel()]
    py_ = [np.asarray(x, dtype=np.float64).ravel() for x in np.asarray(d["Poly_polyy20"]).ravel()]
    for i in range(len(px)):
        assert np.array_equal(px[i], a.poly.polygons[i][:, 0])
        assert np.array_equal(py_[i], a.poly.polygons[i][:, 1])
    lop = ch8.LENGTH_OVER_PIXEL
    dots = np.column_stack([a.poly.centres_xy[:, 0] * lop,
                            shipped_ice().Param.NumPix_y * lop - a.poly.centres_xy[:, 1] * lop])
    assert np.array_equal(np.asarray(d["Poly_dotxy"], dtype=np.float64), dots)


def _grow_subset(ice, which: str):
    """The two S4 subsets of ``IceImage.Floe`` — identical selection on both sides (``fixtures.grow_fixture``)."""
    if which == "g50":
        idx = np.arange(FX.GROW_FIRST_N)
    else:
        sel = FX.grow_fixture()
        if sel is None:
            pytest.skip("book data absent")
        idx = sel
    return [ice.Floe[int(i)] for i in idx]


@needs_book
@pytest.mark.parametrize("tag", ["g50", "gin"])
def test_rgb_image_is_grown_not_preallocated(tag):
    """L2 — review item S4: ``size(rgbImage)`` on floes that do **not** reach the last row/column.

    ``plot_color_bar_and_floe.m`` line 79 never pre-allocates ``rgbImage``; MATLAB grows it from the
    assignments, so its size is ``(max y, max x, 3)`` over the **painted** pixels.  On the shipped 2888-floe
    field both maxima are attained (627x1114 = the image size), so that run — the one the report used to call
    "confirmed" — cannot tell the grown rule from a pre-allocation.  These two subsets can:

    * ``g50`` = ``IceImage.Floe(1:50)`` -> MATLAB returns **627 x 1096 x 3** (1096 < 1114 columns);
    * ``gin`` = the 50 floes inside ``x <= 900, y <= 500`` -> **489 x 865 x 3**, strictly smaller in **both**
      axes than the 627 x 1114 image.

    Asserted: our size equals MATLAB's exactly, it is strictly smaller than the image, and every byte of the
    painted map agrees.
    """
    d = ref("grow")
    ice = shipped_ice()
    pieces = _grow_subset(ice, tag)
    mcd = mean_caliper_diameter(np.array([f.Area for f in pieces], dtype=np.float64), ch8.LENGTH_OVER_PIXEL)
    assert np.array_equal(mcd, flat(d, f"{tag}_mcd"))            # main_WL_new.m lines 30-31 on the subset
    r = ch8.plot_color_bar_and_floe(ch8.COLOR_LIMIT_N, mcd, ice.Param.NumPix_y, len(pieces), pieces,
                                    ch8.LENGTH_OVER_PIXEL)
    assert r.kind == "Pixels"
    ml_size = flat(d, f"{tag}_size").astype(int)
    full = flat(d, "full_size").astype(int)                      # [627, 1114]
    assert np.array_equal(full, [627, 1114])
    assert tuple(r.rgb_image.shape) == tuple(ml_size), f"{r.rgb_image.shape} vs MATLAB {ml_size}"
    # the discriminating clause: MATLAB's array is NOT the image size
    assert not np.array_equal(ml_size[:2], full), "fixture no longer discriminates grown from pre-allocated"
    assert ml_size[1] < full[1]
    if tag == "gin":
        assert ml_size[0] < full[0] and ml_size[1] < full[1]     # strictly smaller in BOTH axes
    py = np.clip(np.round(r.rgb_image * 255), 0, 255).astype(np.uint8)
    assert int((np.asarray(d[f"{tag}_rgb_u8"]) != py).sum()) == 0
    assert np.array_equal(r.counts, flat(d, f"{tag}_counts"))
    assert np.array_equal(r.centers, flat(d, f"{tag}_centers"))
    assert np.array_equal(r.color_index.astype(float), flat(d, f"{tag}_index"))
    assert np.array_equal(r.centres_xy, np.asarray(d[f"{tag}_centre"], dtype=np.float64))


@needs_book
def test_image_shape_override_preallocates_without_moving_a_pixel():
    """L1 — the ``image_shape`` override branch (never executed before this test, review item S4).

    Forcing the full image size must give exactly that array and leave the painted content unchanged inside
    the grown sub-window — i.e. the override changes the canvas, not the drawing.
    """
    ice = shipped_ice()
    pieces = _grow_subset(ice, "gin")
    mcd = mean_caliper_diameter(np.array([f.Area for f in pieces], dtype=np.float64), ch8.LENGTH_OVER_PIXEL)
    grown = ch8.plot_color_bar_and_floe(ch8.COLOR_LIMIT_N, mcd, ice.Param.NumPix_y, len(pieces), pieces,
                                        ch8.LENGTH_OVER_PIXEL)
    forced = ch8.plot_color_bar_and_floe(ch8.COLOR_LIMIT_N, mcd, ice.Param.NumPix_y, len(pieces), pieces,
                                         ch8.LENGTH_OVER_PIXEL, image_shape=(627, 1114))
    assert forced.rgb_image.shape == (627, 1114, 3)
    assert grown.rgb_image.shape == (489, 865, 3)
    h, w = grown.rgb_image.shape[:2]
    assert np.array_equal(forced.rgb_image[:h, :w], grown.rgb_image)
    assert float(forced.rgb_image[h:, :].max()) == 0.0 and float(forced.rgb_image[:, w:].max()) == 0.0
    assert np.array_equal(forced.color_index, grown.color_index)


@needs_book
def test_raw_mcd_is_the_shipped_MCD_results():
    """L3 — Eq. (8.1) on ``Floe.Area`` reproduces the authors' own ``MCD_results.mat`` at max |d| = 0."""
    if not MCD_RESULTS.exists():
        pytest.skip("MCD_results.mat absent")
    ref_mcd = np.asarray(loadmat(str(MCD_RESULTS))["Raw_MCD"]).ravel()
    ice = shipped_ice()
    ours = mean_caliper_diameter(np.array([f.Area for f in ice.Floe], dtype=np.float64), ch8.LENGTH_OVER_PIXEL)
    assert ref_mcd.size == 2888
    assert np.abs(ours - ref_mcd).max() == 0.0


@needs_book
def test_count_error_is_the_one_bin_shifted_difference():
    """L1/L3 — analysis C1: ``count_error = Poly_counts - [Raw_counts(2:end) 0]``, *not* ``z - z0``."""
    a = ch8.mcd_analysis(shipped_ice())
    shifted = np.append(a.raw_counts[1:], 0.0)
    assert np.array_equal(a.count_error, a.poly_counts - shifted)
    # and it is a DIFFERENT quantity from the unshifted difference (so the shift is really tested)
    assert not np.array_equal(a.count_error, a.poly_counts - a.raw_counts)


# =====================================================================================================================
# L2/L3/L4 — the distribution fits (Eqs. 8.2/8.3, Fig. 8.21)
# =====================================================================================================================

def test_model_functions_are_the_m_file_expressions():
    """L1 — the three model functions, evaluated by hand."""
    assert power_law([3.0, 1.5], np.array([1.0, 4.0])) == pytest.approx([3.0, 3.0 * 4 ** -1.5])
    assert truncated_power_law([2.0, 1.0, 10.0], np.array([1.0])) == pytest.approx([2.0 * (1.0 - 0.1)])
    assert weibull_survival([1.0, 2.0], np.array([0.0, 2.0])) == pytest.approx([1.0, np.exp(-1.0)])
    # Eq. (8.3) is scale-free in the exponent: doubling eps1 doubles the curve
    assert power_law([6.0, 1.5], np.array([4.0])) == pytest.approx(2 * power_law([3.0, 1.5], np.array([4.0])))


def test_lsqcurvefit_defaults_match_matlab():
    """L2 — MATLAB's own ``optimoptions('lsqcurvefit')`` defaults, as the port's table claims."""
    d = ref("misc")
    assert MATLAB_LSQ_DEFAULTS["TolFun"] == scal(d, "opt_tolfun") == 1e-6
    assert MATLAB_LSQ_DEFAULTS["TolX"] == scal(d, "opt_tolx") == 1e-6
    assert MATLAB_LSQ_DEFAULTS["MaxIter"] == int(scal(d, "opt_maxiter")) == 400
    assert MATLAB_LSQ_DEFAULTS["Algorithm"] == str(np.asarray(d["opt_alg"]).ravel()[0])
    assert MATLAB_LSQ_DEFAULTS["MaxFunEvals"] is None
    assert str(np.asarray(d["opt_maxfun"]).ravel()[0]) == "100*numberOfVariables"


def test_optimset_maps_the_m_file_call():
    """L1 — ``optimset('LargeScale','on','MaxFunEvals',1e5,'TolFun',1e-5,'MaxIter',1e4)`` (three_fitting l. 18)."""
    o = optimset(LargeScale="on", MaxFunEvals=100000, TolFun=1e-5, MaxIter=10000)
    assert (o.LargeScale, o.MaxFunEvals, o.TolFun, o.MaxIter) == ("on", 100000, 1e-5, 10000)
    assert o.TolX == 1e-6 and o.Algorithm == "trust-region-reflective"   # untouched defaults
    o2 = optimset(Display="off")
    assert o2.extra == {"Display": "off"}                                 # unknown names are kept, not raised


def test_lsqcurvefit_recovers_exact_parameters():
    """L1 — noiseless data drawn from the model must come back with resnorm ~ 0."""
    x = np.arange(1.0, 21.0)
    y = 4.0 * x ** -2.0
    r = lsqcurvefit(power_law, [x.min(), 1.0], x, y)
    assert r.x == pytest.approx([4.0, 2.0], rel=1e-6)
    assert r.resnorm < 1e-18
    assert r.residual.shape == x.shape and r.jacobian.shape == (x.size, 2)


def test_lsqcurvefit_honours_bounds():
    """L1 — with ``ub[1] = 1`` the exponent cannot reach 2 (the bounded branch of three_fitting)."""
    x = np.arange(1.0, 21.0)
    y = 4.0 * x ** -2.0
    r = lsqcurvefit(power_law, [1.0, 0.5], x, y, lb=[0.0, 0.0], ub=[1e6, 1.0])
    assert r.x[1] <= 1.0 + 1e-9
    assert lsqcurvefit(power_law, [x.min(), 1.0], x, y).resnorm < r.resnorm


@needs_book
def test_cumulative_fsd_powerlaw_parity():
    """L2/L4 — Fig. 8.21 and the printed α = 1.3704, against MATLAB's own ``lsqcurvefit``."""
    d = ref("fit")
    ref_mcd = np.asarray(loadmat(str(MCD_RESULTS))["Raw_MCD"]).ravel()
    f = ch8.cumulative_fsd_powerlaw(ref_mcd)
    assert np.array_equal(f.x, flat(d, "MCD"))            # Eq. (8.2)'s sorted sizes: exact
    assert np.array_equal(f.y, flat(d, "N_L"))            # and its Nc values: exact
    ml = flat(d, "epsilong")
    rel = np.abs(f.eta - ml) / np.abs(ml)
    assert rel.max() < 1e-8, f"eta {f.eta} vs MATLAB {ml}"
    assert f.alpha == pytest.approx(round(ml[1], 4), abs=5e-5)
    assert round(f.alpha, 4) == 1.3704                     # the book's printed value, p. 192
    assert f.resnorm == pytest.approx(scal(d, "resnorm"), rel=1e-10)
    assert f.exitflag == int(scal(d, "exitflag")) == 3


@needs_book
def test_book_text_describes_an_estimator_its_own_code_does_not_use():
    """L2/L4 — the log-log OLS slope the *text* describes is 1.87, not the printed 1.3704 (book erratum).

    Pinned in both directions: the code's estimator reproduces 1.3704 and the text's does not.
    """
    d = ref("fit")
    L, Nc = cumulative_size_distribution(np.asarray(loadmat(str(MCD_RESULTS))["Raw_MCD"]).ravel())
    slope = np.polyfit(np.log10(L), np.log10(Nc), 1)
    ml = flat(d, "p_loglog")
    assert slope == pytest.approx(ml, rel=1e-8)            # MATLAB's own polyfit agrees with ours
    assert -slope[0] == pytest.approx(1.8697, abs=5e-5)
    assert round(-slope[0], 4) != 1.3704


def test_power_law_fit_parity_on_fixtures():
    """L2 — ``PowerLaw_fitting_method_and_plotting.m`` (the orphan), verbatim, on three fixtures."""
    d = ref("orphan")
    for name, (x, y) in FX.fit_fixtures().items():
        f = ch8.power_law_fit(x, y)
        ml = flat(d, f"eta_pl_{name}")
        assert np.abs(f.eta - ml).max() / np.abs(ml).max() < 1e-4, f"{name}: {f.eta} vs {ml}"
        assert f.resnorm == pytest.approx(scal(d, f"pl_resnorm_{name}"), rel=1e-6, abs=1e-15)
        assert f.exitflag == int(scal(d, f"pl_exitflag_{name}"))


def test_three_distribution_fits_parity_on_fixtures():
    """L2 — the three fits of ``three_fitting_method_and_plotting.m``.

    The **plain power law** and the **Weibull** agree with MATLAB to 1e-4 relative or better.  The
    **upper-truncated power law** does *not*: its ``eps3`` is unidentifiable on these data (the model tends to
    the plain power law as ``eps3 -> inf``), so the two solvers stop at different points of the same flat
    valley.  What is asserted is therefore the *objective*: our parameters must be at least as good as
    MATLAB's, up to MATLAB's own ``TolFun = 1e-5``.  Both minima are reported in reports/ch08_verification.md
    (analysis risk R1: report the two minima, do not tune).
    """
    d = ref("orphan")
    for name, (x, y) in FX.fit_fixtures().items():
        t = ch8.three_distribution_fits(x, y)
        ml_pow = flat(d, f"eta3_{name}")
        assert np.abs(t.power.eta - ml_pow).max() / np.abs(ml_pow).max() < 1e-4
        ml_wb = flat(d, f"eta2_{name}")
        rn_ml = float(((weibull_survival(ml_wb, x) - y) ** 2).sum())
        assert t.weibull.resnorm <= rn_ml * (1 + 1e-5) + 1e-12, f"{name}: weibull worse than MATLAB"
        assert t.weibull.resnorm == pytest.approx(scal(d, f"rn2_{name}"), rel=1e-5)
        ml_tp = flat(d, f"eta1_{name}")
        rn_tp_ml = float(((truncated_power_law(ml_tp, x) - y) ** 2).sum())
        assert t.truncated_power.resnorm <= rn_tp_ml * (1 + 1e-3) + 1e-12, \
            f"{name}: truncated power law worse than MATLAB ({t.truncated_power.resnorm} > {rn_tp_ml})"
        assert rn_tp_ml == pytest.approx(scal(d, f"rn1_{name}"), rel=1e-6, abs=1e-12)


def test_truncated_power_law_third_parameter_is_unidentifiable():
    """L1 — the deviation of the previous test is a property of the MODEL, not of the solver.

    On the ``exact`` fixture, changing ``eps3`` from MATLAB's 766 to 10^6 changes the residual sum by less than
    1e-6 once ``eps1``/``eps2`` are refitted — that is why the two solvers disagree by three orders of
    magnitude in ``eps3`` and by ~1e-3 in the objective.
    """
    x, y = FX.fit_fixtures()["exact"]
    best = []
    for e3 in (766.0, 1e4, 1e6):
        r = lsqcurvefit(lambda p, xx: truncated_power_law([p[0], p[1], e3], xx), [3.0, 1.5], x, y)
        best.append(r.resnorm)
    assert max(best) < 1e-6, f"eps3 is not flat after all: {best}"
    assert max(best) - min(best) < 1e-6


# =====================================================================================================================
# L1/L2 — sea_ice_model.m
# =====================================================================================================================

def test_matlab_c1c2_reads_a_2x2_center_column_major():
    """L1 — ``c = cat(1, brash_ice(i).Center); c(1), c(2)`` on a 2x2 Center is ``(x1, x2)``, **not** ``(x1, y1)``.

    Pinned in both directions: the 1x2 case is the plain ``(x, y)``, the 2x2 case is not.
    """
    assert ch8._matlab_c1c2(np.array([7.0, 9.0])) == (7.0, 9.0)
    c = np.array([[100.0, 300.0], [200.0, 400.0]])
    assert ch8._matlab_c1c2(c) == (100.0, 200.0)
    assert ch8._matlab_c1c2(c) != (100.0, 300.0), "the naive row reading must NOT be what the port does"


def test_brash_center_field_is_the_documented_deviation_D4():
    """L2 — review item N9: assert the reference variable that deviation **D4** rests on (``center2x2_bC``).

    ``sea_ice_model.m`` line 143 stores ``c = cat(1, brash_ice(i).Center)`` — the *whole matrix* — so for the
    ``center2x2`` fixture (3 brash pieces, the first with a 2x2 ``Center``) MATLAB's
    ``cat(1, brash.Center)`` is **4x2**, not 3x2.  The port stores the **pair** ``(c(1), c(2))`` that every
    downstream line re-reads, so ours is 3x2.  D4's claim was true but rested on a human loading the ``.mat``.
    """
    d = ref("model")
    bC = np.asarray(d["center2x2_bC"], dtype=np.float64)
    assert bC.shape == (4, 2), bC.shape                       # 4 rows for 3 pieces = the 2x2 Center
    assert np.array_equal(bC, [[100.0, 300.0], [200.0, 400.0], [100.0, 200.0], [100.0, 300.0]])
    fx = FX.model_fixtures()["center2x2"]
    ifl, ibr = pieces_of(fx)
    m = ch8.sea_ice_model(ifl, ibr, np.zeros(fx["shape"]))
    ours = np.array([np.asarray(b.Center, dtype=np.float64).ravel() for b in m.brash])
    assert ours.shape == (3, 2)
    assert np.array_equal(ours[0], [100.0, 200.0])            # the `c(1), c(2)` pair, not the matrix
    # ... and the value MATLAB itself *uses* (rows 1 and 3 of its own stack read as c(1), c(2)) is the same
    assert (bC.ravel(order="F")[0], bC.ravel(order="F")[1]) == (100.0, 200.0)
    # the shipped Brash.Center of the Appendix-B structure is the pair too (D4: functionally inert)
    assert np.array_equal(np.asarray(d["center2x2_bA"], dtype=np.float64).ravel(), [100.0, 100.0, 100.0])


def test_crosses_containment_is_not_detected():
    """L1 — ``if xx ~= NaN`` is ``~isempty(polyxpoly(...))``: containment without a crossing is NOT overlap.

    Both directions: the strictly-nested pair is *not* crossing, the genuinely overlapping pair is.
    """
    big = np.array([[10.0, 10.0], [70.0, 10.0], [70.0, 70.0], [10.0, 70.0], [10.0, 10.0]])
    inner = np.array([[30.0, 30.0], [40.0, 30.0], [40.0, 40.0], [30.0, 40.0], [30.0, 30.0]])
    crossing = np.array([[60.0, 60.0], [90.0, 60.0], [90.0, 90.0], [60.0, 90.0], [60.0, 60.0]])
    assert ch8._crosses(big, inner) is False
    assert ch8._crosses(big, crossing) is True
    # ... and the geometrically correct test (off by default) DOES see the containment
    assert ch8._contained(big, inner) is True
    # the same answers through the full polyxpoly routine
    assert ch8._crosses(big, inner, "polyxpoly") is False
    assert ch8._crosses(big, crossing, "polyxpoly") is True
    xx, _, _ = polyxpoly(big[:, 0], big[:, 1], inner[:, 0], inner[:, 1])
    assert xx.size == 0


def test_crosses_containment_parity():
    """L2 — MATLAB's own ``polyxpoly`` returns empty for the nested pair and a point for the touching pair."""
    d = ref("misc")
    assert scal(d, "contain_empty") == 1.0
    assert np.asarray(d["touch_pts"]).shape[0] > 0
    assert np.allclose(np.asarray(d["touch_pts"], dtype=np.float64)[0], [29.0, 29.0])
    # ... and MATLAB's `if xx ~= NaN` is exactly `~isempty(xx)`
    assert scal(d, "idiom_branch_nonempty") == 1.0
    assert scal(d, "idiom_branch_empty") == 0.0
    assert scal(d, "idiom_branch_withnan") == 1.0          # even a NaN element takes the branch
    assert np.array_equal(flat(d, "idiom_nonempty"), [1, 1, 1])


@pytest.mark.parametrize("name", ["nested", "center2x2", "touching"])
def test_sea_ice_model_parity_on_fixtures(name):
    """L2 — the whole of ``sea_ice_model.m`` on the constructed fixtures, element by element."""
    d = ref("model")
    fx = FX.model_fixtures()[name]
    ifl, ibr = pieces_of(fx)
    m = ch8.sea_ice_model(ifl, ibr, np.zeros(fx["shape"]))
    assert len(m.floe) == int(scal(d, f"{name}_nfloe"))
    assert len(m.brash) == int(scal(d, f"{name}_nbrash"))

    assert np.allclose([f.Area for f in m.floe], flat(d, f"{name}_A"), atol=1e-10, rtol=0)
    assert np.allclose(np.array([f.Center for f in m.floe]), np.atleast_2d(d[f"{name}_C"]), atol=1e-10, rtol=0)
    assert np.allclose([f.Perimeter for f in m.floe], flat(d, f"{name}_P"), atol=1e-10, rtol=0)

    # Vertices: SET equality (MATLAB convhull's start vertex/direction is not reproducible - ch06 lesson)
    for pv, mv in zip([f.Vertices for f in m.floe], cells(d[f"{name}_V"])):
        assert set(map(tuple, np.round(pv, 9))) == set(map(tuple, np.round(mv, 9)))
        assert np.array_equal(pv[0], pv[-1]) and np.array_equal(mv[0], mv[-1])   # both rings are CLOSED

    for i, f in enumerate(m.floe):
        assert set(f.Intersect.floe.tolist()) == index_sets(d[f"{name}_IF"])[i]
        assert set(f.Intersect.brash.tolist()) == index_sets(d[f"{name}_IB"])[i]
    for i, b in enumerate(m.brash):
        assert set(b.Intersect.floe.tolist()) == index_sets(d[f"{name}_bIF"])[i]
        assert set(b.Intersect.brash.tolist()) == index_sets(d[f"{name}_bIB"])[i]

    assert np.allclose([b.Radius for b in m.brash], flat(d, f"{name}_bR"), atol=0, rtol=0)
    assert np.allclose([b.Perimeter for b in m.brash], flat(d, f"{name}_bP"), atol=0, rtol=0)
    assert int((m.bw_floe != np.asarray(d[f"{name}_bwf"]).astype(bool)).sum()) == 0
    assert int((m.bw_brash != np.asarray(d[f"{name}_bwb"]).astype(bool)).sum()) == 0


def test_sea_ice_model_reproduces_the_2x2_center_reading():
    """L2 — the ``center2x2`` fixture settles ``c(1), c(2)`` by *which brash piece* MATLAB flags as overlapping.

    Brash 1 has ``Center = [100 300; 200 400]``.  Brash 2 sits at ``(100, 200)`` (the column-major reading) and
    brash 3 at ``(100, 300)`` (the naive row reading); the two discs are 100 px apart, so exactly one of them
    can be within ``r + r'`` = 11.3 px.  MATLAB flags **brash 2** — and so must we.
    """
    d = ref("model")
    ml_1 = index_sets(d["center2x2_bIB"])[0]
    assert ml_1 == {2}, f"the reference itself changed: {ml_1}"
    fx = FX.model_fixtures()["center2x2"]
    m = ch8.sea_ice_model(*pieces_of(fx), np.zeros(fx["shape"]))
    assert set(m.brash[0].Intersect.brash.tolist()) == {2}
    assert 3 not in m.brash[0].Intersect.brash.tolist()


@needs_book
def test_sea_ice_model_parity_on_a_real_window():
    """L2 — 227 real floes and 240 real brash pieces (a 200x200 px window of the §8.3 field) vs MATLAB."""
    d = ref("window")
    fx = FX.window_fixture()
    if fx is None:
        pytest.skip("book data absent")
    m = ch8.sea_ice_model(*pieces_of(fx), np.zeros(fx["shape"]))
    assert len(m.floe) == int(scal(d, "win_nfloe")) == 227
    assert len(m.brash) == int(scal(d, "win_nbrash")) == 240
    assert np.abs(np.array([f.Area for f in m.floe]) - flat(d, "win_A")).max() < 1e-11
    assert np.abs(np.array([f.Center for f in m.floe]) - np.atleast_2d(d["win_C"])).max() < 1e-11
    assert np.abs(np.array([f.Perimeter for f in m.floe]) - flat(d, "win_P")).max() < 1e-11
    V = cells(d["win_V"])
    same_set = sum(set(map(tuple, np.round(pv, 9))) == set(map(tuple, np.round(mv, 9)))
                   for pv, mv in zip([f.Vertices for f in m.floe], V))
    assert same_set == 227
    for i, f in enumerate(m.floe):
        assert set(f.Intersect.floe.tolist()) == index_sets(d["win_IF"])[i]
        assert set(f.Intersect.brash.tolist()) == index_sets(d["win_IB"])[i]
    for i, b in enumerate(m.brash):
        assert set(b.Intersect.floe.tolist()) == index_sets(d["win_bIF"])[i]
        assert set(b.Intersect.brash.tolist()) == index_sets(d["win_bIB"])[i]
    assert np.abs(np.array([b.Radius for b in m.brash]) - flat(d, "win_bR")).max() == 0.0
    assert int((m.bw_floe != np.asarray(d["win_bwf"]).astype(bool)).sum()) == 0
    assert int((m.bw_brash != np.asarray(d["win_bwb"]).astype(bool)).sum()) == 0


@needs_book
def test_aabb_prefilter_and_raster_crop_are_exact_on_real_data():
    """L1 — the two speed-ups must change **nothing**: prefilter vs the literal double loop, cropped vs full
    ``roipoly``, and the fast crossing test vs :func:`seaice.core.polygon.polyxpoly`.

    The fixture must actually exercise the prefilter, so the test also asserts that it *skipped* work.
    """
    fx = FX.window_fixture()
    if fx is None:
        pytest.skip("book data absent")
    ifl, ibr = pieces_of(fx)
    ifl, ibr = ifl[:60], ibr[:60]
    img = np.zeros(fx["shape"])
    fast = ch8.sea_ice_model(ifl, ibr, img)
    brute = ch8.sea_ice_model(ifl, ibr, img, prefilter=False)
    full = ch8.sea_ice_model(ifl, ibr, img, raster_crop=False)
    exact_xy = ch8.sea_ice_model(ifl, ibr, img, intersect_impl="polyxpoly")
    assert fast.n_pairs_tested < fast.n_pairs_total, "the prefilter skipped nothing - fixture is useless"
    assert brute.n_pairs_tested == brute.n_pairs_total
    for other, what in ((brute, "prefilter"), (full, "raster crop"), (exact_xy, "polyxpoly")):
        for i, f in enumerate(fast.floe):
            assert set(f.Intersect.floe.tolist()) == set(other.floe[i].Intersect.floe.tolist()), what
            assert set(f.Intersect.brash.tolist()) == set(other.floe[i].Intersect.brash.tolist()), what
        for i, b in enumerate(fast.brash):
            assert set(b.Intersect.floe.tolist()) == set(other.brash[i].Intersect.floe.tolist()), what
        assert int((fast.bw_floe != other.bw_floe).sum()) == 0, what
        assert int((fast.bw_brash != other.bw_brash).sum()) == 0, what


def test_strict_containment_flag_changes_the_answer():
    """L1 — the opt-in correct test is genuinely different from the M-file's literal one (both directions)."""
    fx = FX.model_fixtures()["nested"]
    ifl, ibr = pieces_of(fx)
    img = np.zeros(fx["shape"])
    literal = ch8.sea_ice_model(ifl, ibr, img)
    strict = ch8.sea_ice_model(ifl, ibr, img, strict_containment=True)
    assert literal.floe[0].Intersect.floe.tolist() == [3]          # floe 2 is inside floe 1 and NOT reported
    assert 2 in strict.floe[0].Intersect.floe.tolist()             # ... but strict containment sees it
    assert literal.floe[0].Intersect.brash.tolist() == [2]
    assert 1 in strict.floe[0].Intersect.brash.tolist()


@needs_book
def test_sea_ice_model_reproduces_the_shipped_structure():
    """L3 — the gold check: re-run the model on the authors' own 2888 floes + 3452 brash pieces.

    ``Polygon.Area/Center/Perimeter`` (polygeom), ``Circle.Radius/Perimeter``, the convex-hull vertex **sets**
    and all four ``Intersect`` lists (1106 / 1171 / 1171 / 544 entries) must come back.  ~20 s.
    """
    ice = shipped_ice()
    ifl, ibr = ch8.iceimage_to_pieces(ice)
    m = ch8.sea_ice_model(ifl, ibr, np.zeros((ice.Param.NumPix_y, ice.Param.NumPix_x)))
    assert len(m.floe) == 2888 and len(m.brash) == 3452
    for i, f in enumerate(m.floe):
        pg = ice.Floe[i].Polygon
        assert abs(f.Area - pg.Area) < 1e-11
        assert np.abs(np.asarray(f.Center) - np.asarray(pg.Center)).max() < 1e-11
        assert abs(f.Perimeter - pg.Perimeter) < 1e-11
        # the shipped ring is OPEN (SeaIce_Image_Structure.m l. 15-16 deletes the duplicate); ours is CLOSED
        assert set(map(tuple, np.round(f.Vertices[:-1], 9))) == \
               set(map(tuple, np.round(np.asarray(pg.Vertices, dtype=np.float64), 9)))
        assert set(f.Intersect.floe.tolist()) == set(np.asarray(pg.Intersect.floe).tolist())
        assert set(f.Intersect.brash.tolist()) == set(np.asarray(pg.Intersect.brash).tolist())
    for i, b in enumerate(m.brash):
        cr = ice.Brash[i].Circle
        assert b.Radius == cr.Radius and b.Perimeter == cr.Perimeter
        assert set(b.Intersect.floe.tolist()) == set(np.asarray(cr.Intersect.floe).tolist())
        assert set(b.Intersect.brash.tolist()) == set(np.asarray(cr.Intersect.brash).tolist())


@needs_book
def test_polygonized_floes_are_not_bigger_the_way_the_book_claims():
    """L2/L3 — the book's p. 184 claim ("the polygonized floes will **not be smaller** than the actual
    identified floes") is false in **both** readings, and the failure is MATLAB's, not the port's.

    1. ``Polygon.Area`` (the continuous ``polygeom`` area of the hull through pixel *centres*) is **smaller**
       than the pixel count for 2383 of the 2888 shipped floes.
    2. The rasterised hull (``roipoly``) does not contain every pixel of the floe either: 2079 of the 19194
       floe pixels of the 227-floe window are dropped, every one of them within **0.083 px** of a hull edge --
       exactly the band ``poly2mask``'s 1/5-pixel snapping can round the wrong way.  MATLAB's own ``bw_floe``
       (``reference/ch08/window.mat``) is identical to ours to **0 px**, so this is the shipped code's
       behaviour, not a port defect.
    """
    ice = shipped_ice()
    smaller = sum(1 for f in ice.Floe if f.Polygon.Area < f.Area)
    assert smaller == 2383 and sum(1 for f in ice.Floe if f.Polygon.Area > f.Area) == 490

    fx = FX.window_fixture()
    if fx is None:
        pytest.skip("book data absent")
    ifl, ibr = pieces_of(fx)
    m = ch8.sea_ice_model(ifl, [], np.zeros(fx["shape"]))
    missing = 0
    for i, p in enumerate(ifl):
        px = np.asarray(p.PixelsPosition, dtype=np.int64)
        inside = m.bw_floe[px[:, 1] - 1, px[:, 0] - 1]
        for q in px[~inside]:
            v = m.floe[i].Vertices
            a, b = v[:-1], v[1:]
            dv = b - a
            t = np.clip(((q - a) * dv).sum(1) / np.maximum((dv * dv).sum(1), 1e-30), 0.0, 1.0)
            proj = a + t[:, None] * dv
            assert np.min(np.hypot(*(proj - q).T)) < 0.1, "an INTERIOR pixel escaped the polygonized mask"
        missing += int((~inside).sum())
    # 2079 of the 19194 floe pixels of this window (10.8 %), every one of them within 0.083 px of a hull edge
    # -- i.e. exactly the band `poly2mask`'s 1/5-pixel snapping can round the wrong way
    assert missing == 2079
    # ... and the raster our port builds is MATLAB's own (the same fixture is compared at 0 px elsewhere)
    d = ref("window")
    ml = np.asarray(d["win_bwf"]).astype(bool)
    full = ch8.sea_ice_model(ifl, ibr, np.zeros(fx["shape"]))
    assert int((full.bw_floe != ml).sum()) == 0


# =====================================================================================================================
# L1/L2/L3 — SeaIce_Image_Structure.m (Appendix B)
# =====================================================================================================================

@pytest.mark.parametrize("name", ["nested", "center2x2", "touching"])
def test_sea_ice_image_structure_parity(name):
    """L2 — every ``Param``/``Field``/``Floe``/``Brash`` field against MATLAB running the original script."""
    d = loadmat(str(REF / "model.mat"), struct_as_record=False, squeeze_me=True) if (REF / "model.mat").exists() \
        else pytest.skip("run reference/ch08/make_refs.py model")
    fx = FX.model_fixtures()[name]
    ifl, ibr = pieces_of(fx)
    m = ch8.sea_ice_model(ifl, ibr, np.zeros(fx["shape"]))
    index_floe = np.zeros(fx["shape"])
    residue = np.zeros(fx["shape"])
    flatv = residue.ravel(order="F")
    flatv[:7] = 1                                 # MATLAB `index_residue(1:7) = 1` is column-major
    residue = flatv.reshape(fx["shape"], order="F")
    cov = Coverage(IceFloe=0.25, BrashIce=0.1, Slush=0.3, Water=0.35)
    ice = ch8.sea_ice_image_structure(ifl, ibr, m.floe, m.brash, cov, index_floe, residue)

    F = d[f"{name}_st_Field"]
    P = d[f"{name}_st_Param"]
    for f in ("LengthSI_x", "LengthSI_y", "PixScale_x", "PixScale_y", "PixArea", "NumFloes", "NumBrash",
              "CovFloes", "CovBrash", "CovSlush", "CovWater", "CovOther"):
        assert float(getattr(ice.Field, f)) == pytest.approx(float(getattr(F, f)), rel=0, abs=1e-15), f
    for f in ("NumPix_x", "NumPix_y", "TiltAngle", "PanAngle"):
        assert float(getattr(ice.Param, f)) == float(getattr(P, f)), f
    for f in ("Location", "Creator"):
        assert getattr(ice.Param, f) == str(getattr(P, f))
    # Review item S1 — the (row, col) <-> (x, y) mapping of lines 50-51.  On a square image swapping the two
    # names leaves every assertion above true; the `touching` fixture is 80 rows x 137 columns, so MATLAB's
    # own NumPix_x = 137 / NumPix_y = 80 (and PixScale_x = 50/137, PixScale_y = 18/80) pin it.  The wrong
    # variant is computed explicitly and asserted to DIFFER, which is the proof that the fixture discriminates.
    rows, cols = fx["shape"]
    assert (int(ice.Param.NumPix_x), int(ice.Param.NumPix_y)) == (cols, rows)
    if rows != cols:
        assert (int(P.NumPix_x), int(P.NumPix_y)) == (cols, rows) != (rows, cols)
        swapped = (float(ice.Param.NumPix_y), float(ice.Param.NumPix_x))
        assert swapped != (float(P.NumPix_x), float(P.NumPix_y)), "transposing rows/cols must be visible"
        assert float(F.PixScale_x) != float(F.PixScale_y)      # 50/137 vs 18/80 - a swap moves both
        assert float(ice.Field.PixScale_x) == pytest.approx(50.0 / cols, rel=0, abs=1e-15)
        assert float(ice.Field.PixScale_y) == pytest.approx(18.0 / rows, rel=0, abs=1e-15)
    else:
        assert rows == cols  # nested / center2x2 are square: they cannot see the transposition (S1)
    assert int(scal(ref("model"), f"{name}_st_nFloe")) == len(ice.Floe)
    assert int(scal(ref("model"), f"{name}_st_nBrash")) == len(ice.Brash)

    fsd_ml = [np.asarray(x).ravel().astype(np.int64) for x in np.atleast_1d(F.FSD)]
    assert len(fsd_ml) == len(ice.Field.FSD)
    for a, b in zip(fsd_ml, ice.Field.FSD):
        assert np.array_equal(a, np.asarray(b, dtype=np.int64))

    vert_ml = cells(d[f"{name}_st_fVert"])
    for a, b in zip(vert_ml, [f.Polygon.Vertices for f in ice.Floe]):
        b = np.atleast_2d(np.asarray(b, dtype=np.float64))
        assert a.shape == b.shape and np.allclose(a, b)                  # the OPEN ring
        assert not np.array_equal(b[0], b[-1]), "the duplicate closing vertex was not deleted"


@needs_book
def test_shipped_structure_invariants():
    """L3 — the shipped ``.mat``'s own consistency: Pixels/Area, Center = mean(Pixels), coverages, counts."""
    ice = shipped_ice()
    assert len(ice.Floe) == 2888 and len(ice.Brash) == 3452
    assert ice.Field.NumFloes == 2888 and ice.Field.NumBrash == 3452
    total = ice.Param.NumPix_x * ice.Param.NumPix_y
    assert total == 1114 * 627
    floe_px = sum(int(f.Area) for f in ice.Floe)
    brash_px = sum(int(b.Area) for b in ice.Brash)
    assert ice.Field.CovFloes == pytest.approx(floe_px / total, rel=0, abs=1e-15)
    assert ice.Field.CovBrash == pytest.approx(brash_px / total, rel=0, abs=1e-15)
    assert (ice.Field.CovFloes + ice.Field.CovBrash + ice.Field.CovSlush
            + ice.Field.CovWater) == pytest.approx(1.0, abs=1e-12)
    for p in list(ice.Floe) + list(ice.Brash):
        px = np.atleast_2d(np.asarray(p.Pixels))
        assert px.shape[0] == int(p.Area)
        assert px[:, 0].min() >= 1 and px[:, 0].max() <= 1114
        assert px[:, 1].min() >= 1 and px[:, 1].max() <= 627
    for f in ice.Floe:
        assert np.abs(np.asarray(f.Center) - np.asarray(f.Pixels, float).mean(axis=0)).max() < 1e-9


@needs_book
def test_shipped_fsd_is_reproduced_and_its_labels_are_not_the_bins():
    """L3 + risk R11 — the 51 ``FSD`` triplets are reproduced by ``hist`` on **centres**, and a recount from
    the printed ``[int_min, int_max]`` labels gives a *different* answer (1861 vs 1411 in the first interval).

    Both directions, so the fixture provably discriminates the two counting rules.
    """
    ice = shipped_ice()
    areas = np.array([float(f.Area) for f in ice.Floe])
    mn, mx = areas.min(), areas.max()
    inter = float(np.trunc((mx - mn) / 50))
    assert inter == 79.0
    centres = ch8.matlab_colon(mn, inter, mx)
    z, n = mhist(areas, centres)
    int_max = np.append(n[1:] - 1, mx)
    ours = np.column_stack([n, int_max, z]).astype(np.int64)
    shipped = np.array([np.asarray(t, dtype=np.int64) for t in ice.Field.FSD])
    assert shipped.shape == (51, 3)
    assert np.array_equal(ours, shipped)
    # the labels say [21, 99] but the bin is (-inf, 60.5]
    recount = np.array([int(((areas >= t[0]) & (areas <= t[1])).sum()) for t in shipped])
    assert recount[0] == 1861 and shipped[0, 2] == 1411
    assert int((recount != shipped[:, 2]).sum()) == 24, "the label recount no longer contradicts the counts"


@needs_book
def test_overlap_graph_counts_and_symmetry():
    """L3 — §8.2.1's "overlap flag": 1106 floe-floe, 1171 floe-brash (= brash-floe), 544 brash-brash, symmetric."""
    g = overlap_graph(shipped_ice())
    assert (g["n_floe_floe"], g["n_floe_brash"], g["n_brash_floe"], g["n_brash_brash"]) == (1106, 1171, 1171, 544)
    assert g["floe_floe_symmetric"] and g["brash_brash_symmetric"] and g["floe_brash_consistent"]


@needs_book
def test_iceimage_mat_roundtrip(tmp_path):
    """L1 — ``save_iceimage_mat`` / ``load_iceimage_mat`` round-trip a subset of the shipped structure."""
    ice = shipped_ice()
    from seaice.core.icestruct import IceImage
    small = IceImage(Param=ice.Param, Field=ice.Field, Floe=list(ice.Floe[:5]), Brash=list(ice.Brash[:5]))
    p = tmp_path / "roundtrip.mat"
    save_iceimage_mat(p, small)
    back = load_iceimage_mat(p)
    assert back.Param.NumPix_x == small.Param.NumPix_x and back.Param.Creator == small.Param.Creator
    assert len(back.Floe) == 5 and len(back.Brash) == 5
    assert back.Field.CovFloes == pytest.approx(small.Field.CovFloes)
    assert len(back.Field.FSD) == len(small.Field.FSD)
    for a, b in zip(back.Floe, small.Floe):
        assert a.Area == b.Area and np.allclose(np.asarray(a.Pixels, float), np.asarray(b.Pixels, float))
        assert np.allclose(np.asarray(a.Polygon.Vertices, float), np.asarray(b.Polygon.Vertices, float))
        assert np.array_equal(a.Polygon.Intersect.floe, b.Polygon.Intersect.floe)


@needs_book
def test_polygeom_round_trip_on_all_2888_shipped_rings():
    """L3 — ``polygeom`` on each stored (open) ring must return the stored Area/Center/Perimeter."""
    ice = shipped_ice()
    dA = dC = dP = 0.0
    for f in ice.Floe:
        v = np.atleast_2d(np.asarray(f.Polygon.Vertices, dtype=np.float64))
        geom, _, _ = polygeom(v[:, 0], v[:, 1])
        dA = max(dA, abs(geom[0] - f.Polygon.Area))
        dC = max(dC, abs(geom[1] - f.Polygon.Center[0]), abs(geom[2] - f.Polygon.Center[1]))
        dP = max(dP, abs(geom[3] - f.Polygon.Perimeter))
    assert dA < 1e-11 and dC < 1e-11 and dP < 1e-11, (dA, dC, dP)


@needs_book
def test_brash_circles_are_area_equivalent_disks():
    """L3 — analysis C2: ``Radius = sqrt(Area/pi)``, ``Perimeter = 2*pi*r`` for all 3452, at max |d| = 0."""
    ice = shipped_ice()
    a = np.array([float(b.Area) for b in ice.Brash])
    r = np.array([b.Circle.Radius for b in ice.Brash])
    p = np.array([b.Circle.Perimeter for b in ice.Brash])
    assert np.abs(np.sqrt(a / np.pi) - r).max() == 0.0
    assert np.abs(2 * np.pi * r - p).max() == 0.0


# =====================================================================================================================
# L2 — color_hist.m and color_hist_comparison.m
# =====================================================================================================================

def _hist_case_areas():
    cases = {k: (v, v) for k, v in FX.hist_fixtures().items()}
    if SHIPPED.exists():
        ice = load_iceimage_mat(SHIPPED)
        cases["real"] = (np.array([f.Polygon.Area for f in ice.Floe], dtype=np.float64),
                         np.array([float(f.Area) for f in ice.Floe]))
    return cases


@pytest.mark.parametrize("name", list(FX.hist_fixtures()) + ["real"])
def test_color_hist_parity(name):
    """L2 — ``z``, the shifted centres ``n``, ``color`` and the nine tick integers, against MATLAB."""
    d = ref("colorhist")
    cases = _hist_case_areas()
    if name not in cases:
        pytest.skip("book data absent")
    areas, _ = cases[name]
    ch = ch8.color_hist(areas)
    assert np.array_equal(ch.z, flat(d, f"ch_z_{name}"))
    assert np.array_equal(ch.n, flat(d, f"ch_n_{name}"))
    assert np.array_equal(ch.color, flat(d, f"ch_color_{name}").astype(np.int64))
    assert np.array_equal(ch.tick_values, flat(d, f"ch_ysh_{name}"))
    YT = np.array([np.asarray(x).ravel()[0] for x in np.asarray(d[f"ch_YT_{name}"]).ravel()], dtype=np.int64)
    assert np.array_equal(ch.tick_labels, YT)
    assert ch.color_min == int(scal(d, f"ch_cmin_{name}")) == 198
    assert ch.color_max == int(scal(d, f"ch_cmax_{name}")) == 9698
    assert ch.nbins == int(scal(d, f"ch_nbins_{name}")) == 50
    assert np.array_equal(ch.centers + 35.0, ch.n)          # `n = n + inter/2` (line 20)


@pytest.mark.parametrize("name", list(FX.hist_fixtures()) + ["real"])
def test_color_hist_comparison_parity(name):
    """L2 — the second histogram ``z0`` and the difference ``z_d = z - z0`` of Fig. 8.15."""
    d = ref("colorhist")
    cases = _hist_case_areas()
    if name not in cases:
        pytest.skip("book data absent")
    model, raw = cases[name]
    cc = ch8.color_hist_comparison(model, raw)
    assert np.array_equal(cc.z, flat(d, f"cc_z_{name}"))
    assert np.array_equal(cc.z0, flat(d, f"cc_z0_{name}"))
    assert np.array_equal(cc.z_d, flat(d, f"cc_zd_{name}"))
    assert np.array_equal(cc.n, flat(d, f"cc_n_{name}"))
    assert np.array_equal(cc.n0, flat(d, f"cc_n0_{name}"))
    assert np.array_equal(cc.tick_values, flat(d, f"cc_ysh_{name}"))
    assert cc.color_max == int(scal(d, f"cc_cmax_{name}"))     # max_x = 6000 here, 3500 in color_hist.m
    assert np.array_equal(cc.n, cc.n0)                          # same centres, unlike main_WL_new's shift


def test_fig_8_15_tick_list_belongs_to_color_hist_not_to_the_comparison():
    """L2/L4 + risk R9 — the list printed under Fig. 8.15 (20 ... 3487) is ``max_x = 3500``'s, but the shipped
    ``color_hist_comparison.m`` has ``max_x = 6000`` (20 ... 5952).  Both against MATLAB, both directions."""
    d = ref("colorhist")
    printed = np.array([20, 149, 297, 471, 682, 950, 1317, 1902, 3487])
    ml3500 = np.array([np.asarray(x).ravel()[0] for x in np.asarray(d["YT3500"]).ravel()], dtype=np.int64)
    ml6000 = np.array([np.asarray(x).ravel()[0] for x in np.asarray(d["YT6000"]).ravel()], dtype=np.int64)
    assert np.array_equal(ml3500, printed)
    assert not np.array_equal(ml6000, printed)
    assert np.array_equal(ml6000, [20, 153, 307, 488, 710, 996, 1398, 2081, 5952])
    a = np.array([100.0, 500.0, 2000.0])
    assert np.array_equal(ch8.color_hist_comparison(a, a, max_x=3500).tick_labels, printed)
    assert np.array_equal(ch8.color_hist_comparison(a, a).tick_labels, ml6000)   # the shipped default is 6000


def test_color_hist_tick_list_is_constants_only():
    """L4 + risk R10 — the Fig. 8.11/8.14/8.15 tick list carries **no** information about the data.

    Three wildly different area vectors give the identical nine integers, so a tick-list match must never be
    quoted as parity evidence (ch07 lesson 60).
    """
    lists = [ch8.color_hist(a).tick_labels
             for a in (np.array([1.0]), np.arange(1.0, 9000.0, 37.0), np.full(500, 3499.0))]
    assert all(np.array_equal(l, lists[0]) for l in lists)
    assert np.array_equal(lists[0], [20, 149, 297, 471, 682, 950, 1317, 1902, 3487])


# =====================================================================================================================
# L1 — §8.1 (text only: no MATLAB code and no data ship for this section)
# =====================================================================================================================

def _two_tone_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((80, 120), 40, dtype=np.uint8)
    img[:, 60:] = 200
    img = np.clip(img.astype(np.int16) + rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)
    return img


def test_shipborne_ice_concentration_otsu_on_a_known_frame():
    """L1 — half the frame is bright: the Otsu concentration must be 0.5 to within the noise."""
    r = ch8.shipborne_ice_concentration(_two_tone_frame(), "otsu")
    assert r.concentration == pytest.approx(0.5, abs=0.01)
    assert 40 < 255 * r.level < 200
    assert r.mask.dtype == bool


def _ramp_frame(seed: int = 1) -> np.ndarray:
    """A **discriminating** section-8.1 frame: water 30 | a broad grey ramp 55...205 | dry ice 225, +/-5 noise.

    The two-tone frame above cannot test "k = 2 k-means is approximately Otsu": its histogram has a 150-level
    empty gap, so *every* sensible threshold produces the identical mask and a 100 % agreement is inevitable
    (ch07's saturating-agreement lesson).  This frame has pixels at **every** gray level between the classes —
    :func:`test_kmeans_k2_is_approximately_otsu` asserts that a one-level move of either threshold would
    displace 11 / 30 pixels, so the agreement it measures is a result and not an artefact of the fixture.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((80, 120), dtype=np.uint8)
    img[:, :50] = 30                                             # open water
    img[:, 50:100] = np.linspace(55, 205, 50).astype(np.uint8)[None, :]   # rubble / wet ice, every level
    img[:, 100:] = 225                                           # dry ice
    return np.clip(img.astype(np.int16) + rng.integers(-5, 6, img.shape), 0, 255).astype(np.uint8)


def test_kmeans_k2_is_approximately_otsu():
    """L1 — p. 179: "if we choose two clusters ... this method is approximately reduced to Otsu".

    Measured on the ramp frame (the authors' own k-means, ``impl='authors'``): Otsu's level is
    ``graythresh`` = 116/255 and the k = 2 centres are ``[45.746939, 186.758134]``, whose midpoint is
    **116.2525** — a *different* threshold that happens to select the same pixels, so the masks agree on
    **100.00 %**.  The fixture is not degenerate: 11 pixels sit at level 116 and 30 at 117, so a one-level
    move of either threshold would break the agreement (asserted).  With **k = 3 and ice = the two brightest
    clusters** the concentration is *higher* (51.10 % vs 41.30 %), the mechanism the book gives for Fig. 8.5.
    """
    img = _ramp_frame()
    otsu = ch8.shipborne_ice_concentration(img, "otsu")
    k2 = ch8.shipborne_ice_concentration(img, "kmeans", k=2, ice_clusters=1)
    assert 255 * otsu.level == pytest.approx(116.0, abs=1e-12)
    mid = float((k2.centers[0] + k2.centers[1]) / 2)
    assert mid == pytest.approx(116.25253622255963, rel=1e-9)
    assert mid != 255 * otsu.level                 # the two methods do NOT pick the same threshold
    agree = float((otsu.mask == k2.mask).mean())
    assert agree == 1.0, agree
    assert k2.concentration == pytest.approx(otsu.concentration, abs=1e-12)
    assert k2.concentration == pytest.approx(0.4130208333333333, rel=0, abs=1e-12)
    # the fixture discriminates: the gray levels straddling the two thresholds are populated, so a one-level
    # move of either threshold would move 11 / 30 of the 9600 pixels and the 100 % agreement would be lost.
    h = np.bincount(img.ravel(), minlength=256)
    assert int(h[116]) == 11 and int(h[117]) == 30
    assert float((img > 117).mean()) != float((img > 116).mean())
    k3 = ch8.shipborne_ice_concentration(img, "kmeans", k=3, ice_clusters="top2")
    assert k3.concentration > otsu.concentration + 0.02
    assert k3.centers.size == 3 and np.all(np.diff(k3.centers) > 0)


def test_shipborne_kmeans_impl_branches_are_both_pinned():
    """L1 — both ``impl`` branches, with numbers (review item S6: the default moved to the authors' k-means).

    ``impl='authors'`` is ch03's line-by-line port of the book's own ``MATLAB_ROOT/ch3/kmeans.m``
    (deterministic, parity **exact**); ``impl='lloyd'`` is the generic seeded k-means++/Lloyd routine
    (parity **approx**).  Measured on the ramp frame:

    ==========  ==========================================  ==============
    k           centres                                     concentration
    ==========  ==========================================  ==============
    2 (both)    ``[45.746939, 186.758134]``                 41.3021 %
    3 authors   ``[35.364295, 122.842687, 209.718880]``     51.1042 % (top2)
    3 lloyd     ``[34.834807, 120.788106, 208.897883]``     51.6979 % (top2, seed 0)
    ==========  ==========================================  ==============

    The k = 3 row is the discriminating one: the two implementations give **different** centres, so this test
    would fail if ``impl`` silently routed both branches to the same routine.  ``'lloyd'`` is also shown to be
    **seed-dependent** (seed 1 lands on the authors' answer, seed 7 on a third one) while ``'authors'`` has no
    RNG at all — the reason the default was changed.
    """
    img = _ramp_frame()
    a2 = ch8.shipborne_ice_concentration(img, "kmeans", k=2, ice_clusters=1, impl="authors")
    l2 = ch8.shipborne_ice_concentration(img, "kmeans", k=2, ice_clusters=1, impl="lloyd", seed=0)
    assert np.allclose(a2.centers, [45.746939, 186.758134], rtol=0, atol=1e-6)
    assert np.allclose(l2.centers, a2.centers, rtol=0, atol=1e-9)          # k = 2: the two agree exactly
    assert a2.concentration == l2.concentration == pytest.approx(0.4130208333333333, rel=0, abs=1e-12)
    assert a2.method == "kmeans(k=2, authors)" and l2.method == "kmeans(k=2, lloyd)"

    a3 = ch8.shipborne_ice_concentration(img, "kmeans", k=3, impl="authors")
    l3 = ch8.shipborne_ice_concentration(img, "kmeans", k=3, impl="lloyd", seed=0)
    assert np.allclose(a3.centers, [35.364295, 122.842687, 209.718880], rtol=0, atol=1e-6)
    assert np.allclose(l3.centers, [34.834807, 120.788106, 208.897883], rtol=0, atol=1e-6)
    assert a3.concentration == pytest.approx(0.5110416666666666, rel=0, abs=1e-12)
    assert l3.concentration == pytest.approx(0.5169791666666667, rel=0, abs=1e-12)
    assert not np.allclose(a3.centers, l3.centers, atol=1e-3), "the two impl branches must be distinguishable"

    # determinism of the default vs seed-dependence of 'lloyd'
    assert np.array_equal(a3.centers,
                          ch8.shipborne_ice_concentration(img, "kmeans", k=3, impl="authors", seed=7).centers)
    seeds = {tuple(np.round(ch8.shipborne_ice_concentration(img, "kmeans", k=3, impl="lloyd", seed=s).centers, 6))
             for s in (0, 1, 7)}
    assert len(seeds) == 3, "the 'lloyd' branch is expected to depend on its seed"
    with pytest.raises(ValueError):
        ch8.shipborne_ice_concentration(img, "kmeans", impl="magic")


def test_shipborne_rejects_unknown_method():
    with pytest.raises(ValueError):
        ch8.shipborne_ice_concentration(_two_tone_frame(), "magic")


# =====================================================================================================================
# L1 — section 8.2's driver (review item S7: `sea_ice_field` was executed by no test)
# =====================================================================================================================

CH07_IMAGE = ROOT / "data/book/ch07/Sea_Ice_Floe_Identification/sea_ice_test.jpg"


@pytest.mark.skipif(not CH07_IMAGE.exists(), reason="private book data absent (ch07 sea_ice_test.jpg)")
def test_sea_ice_field_runs_the_ch07_pipeline(tmp_path):
    """L1 — review item S7: the section-8.2 driver itself, on a 160x240 crop (~4 s), including its cache branch.

    ``sea_ice_field`` is Algorithm 3 (``sea_ice_edge_detection``) followed by Algorithms 4/5
    (``ice_shape_enhancement``), both verified in ch07; this test exercises **the driver** — the
    ``BOOK_PARAMS['sea_ice_demo']`` defaults, the ``seg``/``bk`` hand-off, the ``.npz`` cache write and the
    cache **reuse** — which the script tests bypassed by passing ``--source iceimage``.  Its parity label
    stays **unverified** (no ``.m`` file, printed counts do not reproduce: open item O2); this is coverage,
    not parity.
    """
    import imageio.v3 as iio

    rgb = iio.imread(CH07_IMAGE)
    crop = np.ascontiguousarray(rgb[120:280, 200:440])
    cache = tmp_path / "stage3_crop.npz"
    first = ch8.sea_ice_field(crop, cache=cache, verbose=False)      # cache MISS -> runs Algorithm 3
    assert cache.exists(), "the cache branch never wrote its .npz"
    assert len(first.ice_floe) > 0 and len(first.brash_ice) > 0
    assert first.index_floe.shape == first.out.shape == crop.shape[:2]
    areas = np.array([p.Area for p in first.ice_floe], dtype=np.float64)
    # the Fig. 7.14(a) floe layer carries exactly the floe pixels (it is colour-valued, not labelled)
    assert int(np.count_nonzero(first.index_floe)) == int(areas.sum())
    assert first.t == len(first.ice_floe) + len(first.brash_ice)
    cov = first.coverage
    assert cov.IceFloe + cov.BrashIce + cov.Slush + cov.Water == pytest.approx(1.0, abs=1e-12)

    second = ch8.sea_ice_field(crop, cache=cache, verbose=False)     # cache HIT -> identical result
    assert len(second.ice_floe) == len(first.ice_floe)
    assert np.array_equal(np.array([p.Area for p in second.ice_floe]), areas)
    assert np.array_equal(second.index_floe, first.index_floe)

    # the pieces it returns are exactly what `sea_ice_model` consumes downstream (the section-8.2 chain)
    m = ch8.sea_ice_model(first.ice_floe[:8], first.brash_ice[:8], first.index_floe, raster=False)
    assert len(m.floe) == 8 and len(m.brash) == 8


# =====================================================================================================================
# L4 — numbers quoted in the chapter text
# =====================================================================================================================

@needs_book
def test_book_numbers_section_8_3():
    """L4 — p. 190: 2888 floes, 3452 brash, 58.00 / 4.85 / 21.21 / **15.94** % (a truncation, not a rounding)."""
    ice = shipped_ice()
    assert (len(ice.Floe), len(ice.Brash)) == (2888, 3452)
    pct = [100 * ice.Field.CovFloes, 100 * ice.Field.CovBrash,
           100 * ice.Field.CovSlush, 100 * ice.Field.CovWater]
    assert [float(f"{v:.2f}") for v in pct[:3]] == [58.00, 4.85, 21.21]
    # the water number is 15.945957 %: TRUNCATED to 15.94 in the book, where rounding would print 15.95
    assert pct[3] == pytest.approx(15.945957, abs=1e-5)
    assert np.trunc(pct[3] * 100) / 100 == 15.94
    assert round(pct[3], 2) == 15.95


@needs_book
def test_book_numbers_figure_8_20():
    """L4 — Fig. 8.20's histogram: peak **454 at 7 m**, 0 below 6 m, total 2888, y-limit 450 clips the peak."""
    ref_mcd = np.asarray(loadmat(str(MCD_RESULTS))["Raw_MCD"]).ravel()
    counts, centers = mhist(ref_mcd, np.arange(1.0, 101.0))
    assert counts.sum() == 2888
    assert counts.max() == 454 and centers[int(counts.argmax())] == 7.0
    assert counts[:5].sum() == 0                    # nothing below 6 m
    assert counts.max() > 450                       # ... which is why the printed axis clips it
    top = {int(centers[i]): int(counts[i]) for i in np.argsort(counts)[::-1][:6]}
    assert top == {7: 454, 6: 311, 8: 246, 9: 233, 10: 198, 12: 161}


@needs_book
def test_book_number_alpha():
    """L4 — p. 192: α = 1.3704 (to the four printed decimals)."""
    f = ch8.cumulative_fsd_powerlaw(np.asarray(loadmat(str(MCD_RESULTS))["Raw_MCD"]).ravel())
    assert round(f.alpha, 4) == 1.3704
    assert f.x.min() == pytest.approx(6.0985, abs=1e-3)     # Fig. 8.21's x range 10^0.8 ... 10^2.1
    assert f.x.max() == pytest.approx(83.9466, abs=1e-3)
    assert f.y.min() == pytest.approx(1 / 2888, abs=1e-12)


@needs_book
def test_shipped_mat_contradicts_the_shipped_script_metadata():
    """L3 + risk R12 — the ``.mat`` holds Helicopter/OATRC 2015/empty lengths, the script hard-codes UAV/
    Ny-Alesund/50x18: the shipped structure came from a §8.3 variant of the §8.2 script."""
    p = shipped_ice().Param
    f = shipped_ice().Field
    assert p.Creator == "Helicopter" and p.PrjName == "OATRC 2015" and p.Location is None
    assert f.LengthSI_x is None and f.LengthSI_y is None and f.PixScale_x is None
    # the port's defaults are the SCRIPT's values, so they must differ from the .mat
    kw = ch8.sea_ice_image_structure.__kwdefaults__
    assert kw["Location"] == "Ny-Alesund" and kw["Creator"] == "UAV"
    assert kw["LengthSI_x"] == 50 and kw["LengthSI_y"] == 18
    assert kw["Location"] != p.Location and kw["Creator"] != p.Creator


# =====================================================================================================================
# scripts run headless
# =====================================================================================================================
SCRIPTS = ["ch08_main_WL_new", "ch08_fitting_ice_floes_distribution", "ch08_three_fitting",
           "ch08_sea_ice_model", "ch08_sea_ice_image_structure", "ch08_color_hist",
           "ch08_color_hist_comparison", "ch08_ice_concentration_series"]

SCRIPT_FLAGS = {
    "ch08_sea_ice_model": ["--source", "iceimage", "--limit", "40"],
    "ch08_sea_ice_image_structure": ["--source", "iceimage"],
    "ch08_color_hist": ["--input", "iceimage"],
    "ch08_color_hist_comparison": ["--input", "iceimage"],
}
NEEDS_BOOK = {"ch08_main_WL_new", "ch08_fitting_ice_floes_distribution", "ch08_sea_ice_model",
              "ch08_sea_ice_image_structure", "ch08_color_hist", "ch08_color_hist_comparison"}


def _run_script(script: str, out: Path, *flags: str) -> subprocess.CompletedProcess:
    cmd = [str(PY), str(ROOT / "scripts" / f"{script}.py"), "--no-show", "--out", str(out), *flags]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=3600)


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs(script):
    out = VERIFY / "pytest_scripts" / script
    out.mkdir(parents=True, exist_ok=True)
    proc = _run_script(script, out, *SCRIPT_FLAGS.get(script, []))
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    if script in NEEDS_BOOK and not SHIPPED.exists():
        assert "SKIP" in proc.stdout
    elif script == "ch08_ice_concentration_series":
        assert "SKIP" in proc.stdout or list(out.glob("*.png"))
    else:
        assert list(out.glob("*.png")), "no figure written"


CLI_CASES = [
    ("ch08_color_hist_comparison", ["--max-x", "3500", "--input", "iceimage"]),
    ("ch08_sea_ice_model", ["--source", "iceimage", "--limit", "25", "--brute-force"]),
    ("ch08_sea_ice_model", ["--source", "iceimage", "--limit", "25", "--strict-containment"]),
]


@pytest.mark.parametrize("script,flags", CLI_CASES, ids=[f"{s}:{' '.join(f)}" for s, f in CLI_CASES])
def test_script_cli_flags(script, flags):
    safe = "_".join(f.strip("-").replace(".", "p") for f in flags)[:60]
    out = VERIFY / "pytest_scripts" / f"{script}_{safe}"
    out.mkdir(parents=True, exist_ok=True)
    proc = _run_script(script, out, *flags)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
