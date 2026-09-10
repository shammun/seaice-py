"""Chapter 6 verification tests — GVF snake-based ice floe boundary identification.

Evidence levels (see the ``verify-port`` skill):

* **L1** synthetic truth — small arrays whose answer is known analytically or from the book's own equations.
* **L2** MATLAB parity — the ORIGINAL ``.m`` code of ``MATLAB_ROOT/ch6`` run through
  ``reference/ch06/make_refs.py`` (MATLAB R2025a, ``matlab -batch``) and compared element-wise.
* **L3** book-figure reproduction (rendered by ``reference/ch06/make_compare_figures.py``; verdicts in the report).
* **L4** numbers quoted in the chapter text.

Run: ``.venv/Scripts/python.exe -m pytest tests/test_ch06.py -q -p no:cacheprovider``
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

from reference.ch06 import fixtures as FX  # noqa: E402
from seaice import ch06_gvf_snake as ch6  # noqa: E402
from seaice.core import polygon as POLY  # noqa: E402
from seaice.core import snake as SN  # noqa: E402
from seaice.core import synth  # noqa: E402
from seaice.core.connectivity import bwareaopen, bwperim, label_components  # noqa: E402
from seaice.core.distance import bwdist  # noqa: E402
from seaice.core.filters import conv2, homomorphic_butterworth  # noqa: E402
from seaice.core.interp import interp2  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.matlab_compat import del2, rgb2gray_matlab  # noqa: E402
from seaice.core.morphology import imregionalmax, regional_maxima_by_reconstruction  # noqa: E402
from seaice.core.regionprops import region_table, regionprops  # noqa: E402
from seaice.core.threshold import graythresh, im2bw  # noqa: E402

REF = ROOT / "reference/ch06"
DATA = ROOT / "data/book/ch06"
VERIFY = ROOT / "outputs/ch06/verify"
PY = ROOT / ".venv/Scripts/python.exe"
CH = "ch06"

RP_PROPS = ("Area", "Centroid", "BoundingBox", "ConvexArea", "Solidity", "MajorAxisLength",
            "MinorAxisLength", "Eccentricity", "Orientation", "Perimeter")

_CACHE: dict[str, dict] = {}


def ref(name: str) -> dict:
    """Load ``reference/ch06/<name>.mat`` once, or skip the test when it has not been generated."""
    if name not in _CACHE:
        p = REF / f"{name}.mat"
        if not p.exists():
            pytest.skip(f"run reference/ch06/make_refs.py {name} (needs MATLAB R2025a)")
        _CACHE[name] = loadmat(str(p))
    return _CACHE[name]


def maxdiff(a, b) -> float:
    """Max |a - b| after squeezing.

    Equal values count as 0 even when they are +-Inf (``Inf - Inf`` is NaN), and NaN at the same position on
    both sides counts as 0 -- MATLAB produces NaNs and infinities here too (constant ``f`` in ``GVF.m``, the
    ``+-Inf`` plants of the ``del2`` fixtures).  The NaN *patterns* must still match exactly.
    """
    a = np.atleast_1d(np.asarray(a, dtype=float).squeeze())
    b = np.atleast_1d(np.asarray(b, dtype=float).squeeze())
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    if a.size == 0:
        return 0.0
    assert np.array_equal(np.isnan(a), np.isnan(b)), "NaN patterns differ"
    with np.errstate(invalid="ignore"):
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        e = np.where(same, 0.0, np.abs(a - b))
    return float(np.max(e))


def mstr(x) -> str:
    """A MATLAB char array loaded by scipy as a 1x1 object/str array."""
    a = np.asarray(x)
    if a.size == 0:
        return ""
    return str(a.ravel()[0]).strip()


def book_image(name: str) -> np.ndarray:
    """The private book image, or skip (CLAUDE.md rule 12: never fall back to a substitute in a parity test)."""
    try:
        img, _ = load_image(CH, name, allow_fallback=False, verbose=False)
    except FileNotFoundError as exc:
        pytest.skip(f"private book image absent: {exc}")
    return img


# =====================================================================================================================
# L1 — synthetic truth
# =====================================================================================================================
class TestL1Del2:
    def test_four_del2_is_the_five_point_laplacian_interior(self):
        """Book Eq. (6.52c): ``GVF.m`` writes ``mu*4*del2(u)`` because ``4 del2`` restores the 5-point Laplacian."""
        rng = np.random.default_rng(0)
        u = rng.normal(size=(9, 11))
        lap = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1])
        assert np.max(np.abs(4.0 * del2(u)[1:-1, 1:-1] - lap)) < 1e-12

    def test_border_rule_is_linear_extrapolation(self):
        """MATLAB extrapolates the border second difference of **each pass**: ``g(1) = 2 g(2) - g(3)``.

        The 2-D result is ``(g_rows + g_cols)/2``, so the identity is isolated with a field that varies along
        one dimension only (then the other pass contributes exactly 0).
        """
        rng = np.random.default_rng(1)
        col = rng.normal(size=(8, 1))
        u = np.tile(col, (1, 9))                       # constant along columns -> only the row pass is non-zero
        g = del2(u)
        assert np.max(np.abs(g[0] - (2 * g[1] - g[2]))) < 1e-12
        row = rng.normal(size=(1, 9))
        w = np.tile(row, (8, 1))
        h = del2(w)
        assert np.max(np.abs(h[:, 0] - (2 * h[:, 1] - h[:, 2]))) < 1e-12
        v = del2(rng.normal(size=(9, 1)))              # a column vector: the 1-D case, v = g/2
        assert abs(v[0, 0] - (2 * v[1, 0] - v[2, 0])) < 1e-12

    @pytest.mark.parametrize("shape", [(2, 2), (2, 5), (5, 2), (1, 1), (1, 4), (4, 1)])
    def test_small_matrices_are_zero(self, shape):
        """``n <= 2`` along a dimension gives 0 there; a 1-D input is the 1-D second difference / 2."""
        a = np.arange(1.0, shape[0] * shape[1] + 1).reshape(shape)
        g = del2(a)
        if min(shape) <= 2 and (shape[0] <= 2 and shape[1] <= 2):
            assert np.all(g == 0)
        assert g.shape == a.shape

    def test_three_row_rule(self):
        """``n == 3`` copies the single interior value of that pass to both borders."""
        col = np.array([[1.0], [4.0], [7.5]])
        a = np.tile(col, (1, 6))                       # only the row pass contributes
        g = del2(a)
        assert np.allclose(g[0], g[1]) and np.allclose(g[2], g[1])

    def test_constant_is_zero(self):
        assert np.all(del2(np.full((6, 7), 3.25)) == 0.0)


class TestL1Snake:
    def test_bound_mirror_expand_equals_numpy_reflect(self):
        rng = np.random.default_rng(2)
        a = rng.normal(size=(7, 5))
        assert np.array_equal(SN.bound_mirror_expand(a), np.pad(a, 1, mode="reflect"))

    def test_bound_mirror_round_trip(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(6, 8))
        assert np.array_equal(SN.bound_mirror_shrink(SN.bound_mirror_expand(a)), a)

    def test_bound_mirror_ensure_is_idempotent(self):
        rng = np.random.default_rng(4)
        b = SN.bound_mirror_expand(rng.normal(size=(6, 8)))
        assert np.array_equal(SN.bound_mirror_ensure(SN.bound_mirror_ensure(b)), SN.bound_mirror_ensure(b))

    def test_bound_mirror_ensure_rejects_small(self):
        with pytest.raises(ValueError):
            SN.bound_mirror_ensure(np.zeros((2, 5)))

    def test_gvf_refuses_cfl_violation(self):
        """Eqs. (6.54)/(6.55): ``r = mu dt/(dx dy) <= 1/4`` at unit steps."""
        f = np.zeros((6, 6))
        f[2:4, 2:4] = 1.0
        with pytest.raises(ValueError):
            SN.gvf(f, 0.3, 5)
        SN.gvf(f, 0.25, 5)  # exactly at the bound is allowed
        SN.gvf(f, 0.3, 5, check_cfl=False)

    def test_gvf_field_points_at_the_edge(self):
        """§6.2: the GVF field of an edge map points towards the edges and reaches far from them."""
        bw = np.zeros((60, 60))
        bw[20:40, 20:40] = 1.0
        f = SN.gradient2_magnitude(bw)
        u, v = SN.gvf(f, 0.1, 100)
        mag = np.hypot(u, v)
        assert mag[30, 10] > 1e-3       # far from the edge the field is non-zero (large capture range)
        assert u[30, 12] > 0            # and points right, towards the square

    def test_snake_matrix_is_symmetric_circulant_for_constant_coefficients(self):
        A = SN.snake_matrix(16, 0.05, 0.03)
        assert np.allclose(A, A.T)
        first = A[:, 0]
        for k in range(16):
            assert np.allclose(np.roll(first, k), A[:, k])
        assert np.allclose(A, SN.snake_matrix(16, 0.05, 0.03, book_index=True))

    def test_snake_matrix_beta0_eigenvalues(self):
        """With beta = 0 the matrix is alpha*(2I - S - S^T); eigenvalues alpha(2 - 2cos(2 pi k/N)) >= 0."""
        N, al = 12, 0.05
        A = SN.snake_matrix(N, al, 0.0)
        k = np.arange(N)
        assert np.allclose(np.sort(np.linalg.eigvalsh(A)), np.sort(al * (2 - 2 * np.cos(2 * np.pi * k / N))))

    def test_snake_first_column_matches_snake_matrix(self):
        for N in (3, 4, 5, 8, 37, 126):
            for al, be in ((0.05, 0.0), (0.05, 0.1), (0.0, 0.1), (0.1, 0.2)):
                col = SN.snake_first_column(N, al, be, 1.0)
                A = SN.snake_matrix(N, al, be) + np.eye(N)
                assert np.allclose(col, A[:, 0]), (N, al, be)

    def test_snake_shrinks_without_external_force(self):
        """p. 115: with kappa = 0 the snake evolves under its own energy and keeps shrinking."""
        t = np.arange(0.0, 6.28 + 1e-12, 0.05)
        x, y = 50 + 20 * np.cos(t), 50 + 20 * np.sin(t)
        z = np.zeros((100, 100))
        areas = []
        for _ in range(4):
            x, y = SN.snakedeform(x, y, 0.05, 0.0, 1.0, 0.0, z, z, 20, solver="dense")
            areas.append(POLY.polyarea(x, y))
        assert all(a2 < a1 for a1, a2 in zip(areas, areas[1:]))

    def test_snake_is_static_with_zero_parameters(self):
        t = np.arange(0.0, 6.28 + 1e-12, 0.05)
        x, y = 50 + 20 * np.cos(t), 50 + 20 * np.sin(t)
        z = np.zeros((100, 100))
        x2, y2 = SN.snakedeform(x, y, 0.0, 0.0, 1.0, 0.0, z, z, 10, solver="dense")
        assert np.max(np.abs(x2 - x)) < 1e-10 and np.max(np.abs(y2 - y)) < 1e-10

    def test_snakeinterp_respects_dmax(self):
        x = np.array([10.0, 40.0, 40.0, 10.0])
        y = np.array([10.0, 10.0, 35.0, 35.0])
        for dmax in (1.0, 2.0, 4.0):
            xi, yi = SN.snakeinterp(x, y, dmax, 0.0)
            d = np.abs(np.roll(xi, -1) - xi) + np.abs(np.roll(yi, -1) - yi)  # city-block, cyclic
            assert d.max() <= dmax + 1e-9

    def test_snakeinterp_caps_the_unbounded_while_loop(self):
        """Risk R10: the shipped ``snakeinterp.m`` has no iteration cap (``dmax = 0`` never converges: each pass
        only halves the spacing).  The port raises after ``max_passes`` instead of hanging."""
        t = ch6.CIRCLE_T
        x, y = 50 + 20 * np.cos(t), 50 + 20 * np.sin(t)
        with pytest.raises(RuntimeError):
            SN.snakeinterp(x, y, 0.0, 0.0, max_passes=5)

    def test_snakeindex_inserts_only_where_flagged(self):
        idx = np.array([1, 0, 1, 0, 0], dtype=bool)
        z = SN.snakeindex(idx)
        assert np.allclose(z, [1, 1.5, 2, 3, 3.5, 4, 5])

    def test_gaussian_mask_shape_and_normalisation(self):
        """``gaussianMask.m``: R = ceil(3 s) -> (2R+1)^2; ``gaussianBlur`` normalises the mask to unit sum."""
        for s in (0.5, 1.0, 2.0, 4.0):
            M = SN.gaussian_mask(1.0, s)
            R = int(np.ceil(3 * s))
            assert M.shape == (2 * R + 1, 2 * R + 1)
            assert M.argmax() == (M.size - 1) // 2
        flat = SN.gaussian_blur(np.full((20, 20), 7.0), 1.0)
        assert abs(flat[10, 10] - 7.0) < 1e-9

    def test_xconv2_agrees_with_conv2_same(self):
        rng = np.random.default_rng(5)
        I = rng.random((23, 19))
        for G in (rng.random((3, 3)), rng.random((4, 4)), rng.random((5, 3))):
            assert np.max(np.abs(SN.xconv2(I, G) - conv2(I, G, "same"))) < 1e-10

    def test_gradient2_complex_form(self):
        rng = np.random.default_rng(6)
        a = rng.normal(size=(7, 9))
        gx, gy = SN.gradient2(a)
        z = SN.gradient2_complex(a)
        assert np.allclose(np.real(z), gx) and np.allclose(np.imag(z), gy)
        assert np.allclose(np.abs(z), SN.gradient2_magnitude(a))

    def test_circulant_and_dense_solvers_agree(self):
        """The 'circulant' fast path solves the same system; measure the difference (report: `near`)."""
        rng = np.random.default_rng(7)
        fx, fy = rng.normal(size=(60, 70)), rng.normal(size=(60, 70))
        t = np.arange(0.0, 6.28 + 1e-12, 0.05)
        x, y = 40 + 15 * np.cos(t), 30 + 12 * np.sin(t)
        xd, yd = SN.snakedeform(x, y, 0.05, 0.0, 1.0, 0.5, fx, fy, 20, solver="dense")
        xc, yc = SN.snakedeform(x, y, 0.05, 0.0, 1.0, 0.5, fx, fy, 20, solver="circulant")
        assert max(np.max(np.abs(xd - xc)), np.max(np.abs(yd - yc))) < 1e-9


class TestL1ContourInit:
    def test_fig_6_14_printed_matrices(self):
        """Fig. 6.14(a)/(b): the printed 8x8 blob and its city-block distance transform (L4 printed truth)."""
        A = synth.FIG_6_14_IMAGE
        D = bwdist(~A, "cityblock")
        assert np.array_equal(D.astype(int), synth.FIG_6_14_DISTANCE.astype(int))
        assert np.array_equal(synth.FIG_6_14_IMAGE, synth.FIG_6_14_DISTANCE > 0)

    def test_fig_6_14_one_regional_maximum_of_three_local_maxima(self):
        """p. 134: "a regional maximum consisting of three local maxima" (numeral 3 in Fig. 6.14(b))."""
        init = ch6.initialize_contours(synth.FIG_6_14_IMAGE, se_radius=3)
        maxima = init.minima_map & synth.FIG_6_14_IMAGE
        assert int(maxima.sum()) == 3
        assert int(label_components(maxima, 8).max()) == 1
        assert set(np.asarray(init.img_dist)[maxima].tolist()) == {3.0}

    def test_fig_6_14_radius_is_three_over_sqrt_two(self):
        """Footnote 4, p. 135: the city-block value at the seed is divided by sqrt(2) -> 3/sqrt(2) = 2.1213."""
        init = ch6.initialize_contours(synth.FIG_6_14_IMAGE, se_radius=3)
        assert init.num == 1
        assert abs(float(init.radii[0]) - 3.0 / np.sqrt(2.0)) < 1e-12
        assert abs(float(init.radii[0]) - 2.1213) < 1e-4

    def test_book_and_script_forms_agree(self):
        """Risk R12: the text's regional maxima of D (Eq. 6.57) = the code's imregionalmin(-D) with -Inf outside."""
        for bw in (synth.FIG_6_14_IMAGE, synth.fig_6_16_circles()):
            s = ch6.initialize_contours(bw, form="script")
            b = ch6.initialize_contours(bw, form="book")
            assert np.array_equal(s.minima_map & bw, b.minima_map & bw)

    def test_circle_discretisation(self):
        """``t = 0:0.05:6.28`` -> 126 points, last 6.25: the initial circle is NOT closed."""
        assert ch6.CIRCLE_T.size == 126
        assert abs(ch6.CIRCLE_T[-1] - 6.25) < 1e-12
        assert ch6.CIRCLE_T[-1] < 2 * np.pi

    def test_radius_floor(self):
        """``if r == 0, r = 2``: two blocks whose dilated maxima merge put the seed centroid on background."""
        bw = np.zeros((20, 30), dtype=bool)
        bw[8:11, 5:8] = True
        bw[8:11, 10:13] = True
        init = ch6.initialize_contours(bw, se_radius=3)
        assert init.num == 1
        assert np.allclose(init.centroids[0], [9.5, 10.0])
        assert float(init.img_dist[9, 9]) == 0.0           # round(9.5) = 10 (1-based) lands on background
        assert float(init.radii[0]) == 2.0


class TestL1Polygon:
    def test_polygeom_header_self_test(self):
        """``polygeom.m``'s own header: a 3x5 rectangle at 30 deg -> area 15, centroid (3.415, 6.549), P 16."""
        x = np.array([2.0, 0.5, 4.83, 6.33])
        y = np.array([4.0, 6.598, 9.098, 6.5])
        geom, iner, cpmo = POLY.polygeom(x, y)
        assert abs(geom[0] - 15.0) < 1e-3
        assert abs(geom[1] - 3.415) < 1e-3 and abs(geom[2] - 6.549) < 1e-3
        assert abs(geom[3] - 16.0) < 1e-3
        assert abs(iner[0] - 659.561) < 1e-2 and abs(iner[1] - 201.173) < 1e-2
        assert abs(iner[2] - 344.117) < 1e-2 and abs(iner[3] - 16.249) < 1e-2
        assert abs(iner[4] - 26.247) < 1e-2 and abs(iner[5] - 8.660) < 1e-2
        assert abs(cpmo[0] - 11.249) < 1e-2 and abs(cpmo[2] - 31.247) < 1e-2
        assert abs(cpmo[4] - 42.496) < 1e-2
        assert abs(np.degrees(cpmo[1]) - 30.0) < 1e-2 and abs(np.degrees(cpmo[3]) - 120.0) < 1e-2

    def test_polyarea_square(self):
        assert abs(POLY.polyarea([0, 4, 4, 0], [0, 0, 3, 3]) - 12.0) < 1e-12

    def test_minboundrect_unit_square_example(self):
        """The author's own example: 50 000 uniform points in the unit square -> area ~ 0.99994."""
        rng = np.random.default_rng(11)
        p = rng.random((50000, 2))
        _, _, area, _ = POLY.minboundrect(p[:, 0], p[:, 1], "a")
        assert 0.99 < area <= 1.02

    def test_minboundrect_rotated_rectangle(self):
        th = np.deg2rad(37.0)
        base = np.array([[0, 0], [6, 0], [6, 2], [0, 2]], dtype=float)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        p = base @ R.T
        rx, ry, area, per = POLY.minboundrect(p[:, 0], p[:, 1], "a")
        assert abs(area - 12.0) < 1e-9 and abs(per - 16.0) < 1e-9
        assert rx.size == 5 and ry.size == 5

    def test_minboundrect_special_cases(self):
        rx, ry, a, p = POLY.minboundrect([3.0], [4.0])
        assert rx.size == 5 and a == 0.0 and p == 0.0
        rx, ry, a, p = POLY.minboundrect([1.0, 5.0], [2.0, 7.0])
        assert a == 0.0 and abs(p - 2 * np.hypot(4, 5)) < 1e-12
        rx, ry, a, p = POLY.minboundrect([], [])
        assert rx.size == 0 and np.isnan(a)

    def test_clip_polygon_rect_is_a_no_op_inside(self):
        t = ch6.CIRCLE_T
        x, y = 20 + 8 * np.cos(t), 15 + 8 * np.sin(t)
        cx, cy = POLY.clip_polygon_rect(x, y, (0.0, 40.0), (0.0, 30.0))
        assert set(zip(np.round(cx, 9), np.round(cy, 9))) == set(zip(np.round(x, 9), np.round(y, 9)))

    def test_clip_polygon_rect_clips_and_keeps_the_corner(self):
        t = ch6.CIRCLE_T
        x, y = 3 + 8 * np.cos(t), 4 + 8 * np.sin(t)
        cx, cy = POLY.clip_polygon_rect(x, y, (0.0, 40.0), (0.0, 30.0))
        assert cx.min() >= -1e-12 and cy.min() >= -1e-12
        assert any(abs(a) < 1e-12 and abs(b) < 1e-12 for a, b in zip(cx, cy))  # the (0, 0) corner is inserted

    def test_snake_is_invariant_under_rotation_and_reversal(self):
        """Risk R1: ``polybool`` rotates the start vertex and reverses the order; the snake must not care."""
        rng = np.random.default_rng(12)
        fx, fy = rng.normal(size=(60, 70)), rng.normal(size=(60, 70))
        t = ch6.CIRCLE_T
        x, y = 35 + 12 * np.cos(t), 30 + 10 * np.sin(t)
        base = SN.snakedeform(x, y, 0.05, 0.0, 1.0, 0.5, fx, fy, 15, solver="dense")
        for k in (1, 37, 90):
            xr, yr = np.roll(x, k), np.roll(y, k)
            rx, ry = SN.snakedeform(xr, yr, 0.05, 0.0, 1.0, 0.5, fx, fy, 15, solver="dense")
            assert np.max(np.abs(np.roll(rx, -k) - base[0])) < 1e-9
        xv, yv = x[::-1].copy(), y[::-1].copy()
        vx, vy = SN.snakedeform(xv, yv, 0.05, 0.0, 1.0, 0.5, fx, fy, 15, solver="dense")
        assert np.max(np.abs(vx[::-1] - base[0])) < 1e-9

    def test_poly2mask_unit_square(self):
        """A 3x2 axis-aligned rectangle: MATLAB includes the pixels whose centres fall inside."""
        m = POLY.poly2mask([2.0, 5.0, 5.0, 2.0], [2.0, 2.0, 4.0, 4.0], 8, 8)
        assert m.sum() == 6
        assert m[2:4, 2:5].all()
        assert not m[1, :].any() and not m[:, 1].any()

    def test_roipoly_three_argument_form(self):
        img = np.zeros((10, 12))
        a = POLY.roipoly(img, [2.0, 8.0, 8.0, 2.0], [2.0, 2.0, 6.0, 6.0])
        b = POLY.poly2mask([2.0, 8.0, 8.0, 2.0], [2.0, 2.0, 6.0, 6.0], 10, 12)
        assert np.array_equal(a, b)

    def test_polyxpoly_crossing_squares(self):
        xi, yi, ii = POLY.polyxpoly([0, 10, 10, 0, 0], [0, 0, 10, 10, 0],
                                    [5, 15, 15, 5, 5], [5, 5, 15, 15, 5])
        pts = {(round(a, 9), round(b, 9)) for a, b in zip(xi, yi)}
        assert pts == {(10.0, 5.0), (5.0, 10.0)}
        assert ii.shape == (2, 2)


class TestL1Regionprops:
    def test_single_pixel(self):
        bw = np.zeros((5, 5), bool)
        bw[2, 3] = True
        s = regionprops(bw, "all")[0]
        assert s.Area == 1
        assert np.allclose(s.Centroid, [4.0, 3.0])            # MATLAB (x, y), 1-based
        assert np.allclose(s.BoundingBox, [3.5, 2.5, 1, 1])
        assert s.Perimeter == 0.0                              # <= 1 boundary pixel -> 0
        assert s.ConvexArea == 1 and s.Solidity == 1.0

    def test_square_solidity_is_one(self):
        bw = np.zeros((12, 12), bool)
        bw[2:10, 2:10] = True
        s = regionprops(bw, ("Area", "ConvexArea", "Solidity", "Orientation"))[0]
        assert s.Area == 64 and s.ConvexArea == 64 and s.Solidity == 1.0

    def test_axis_ratio_of_a_rectangle_is_near_its_aspect(self):
        bw = np.zeros((16, 26), bool)
        bw[3:13, 3:23] = True                                  # 10 x 20
        s = regionprops(bw, ("MajorAxisLength", "MinorAxisLength"))[0]
        assert 1.9 < s.MajorAxisLength / s.MinorAxisLength < 2.1

    def test_concave_shape_has_solidity_below_one(self):
        bw = np.zeros((14, 14), bool)
        bw[2:12, 2:6] = True
        bw[8:12, 6:12] = True
        s = regionprops(bw, ("Area", "ConvexArea", "Solidity"))[0]
        assert s.ConvexArea > s.Area and s.Solidity < 0.9

    def test_label_order_is_matlab_column_major(self):
        bw = np.zeros((10, 10), bool)
        bw[1:3, 6:8] = True      # right block: later in column-major order
        bw[6:8, 1:3] = True      # left block: earlier
        stats = regionprops(bw, ("Centroid",))
        assert stats[0].Centroid[0] < stats[1].Centroid[0]

    def test_region_table_shape(self):
        bw = np.zeros((10, 20), bool)
        bw[2:5, 2:5] = True
        bw[2:5, 10:18] = True
        tab = region_table(bw)
        assert set(tab) == {"Area", "Solidity", "MajorAxisLength", "MinorAxisLength"}
        assert tab["Area"].shape == (2,)


class TestL1Criteria:
    def test_component_criteria_thresholds(self):
        """ch9 p. 205 / ``GVF_distance.m`` lines 95-102: Area > Ra, Solidity < Rc, Major/Minor > Rl."""
        bw = np.zeros((60, 90), bool)
        bw[5:15, 5:15] = True                    # small compact square: passes everything
        bw[20:50, 20:60] = True                  # large: fails Ra
        bw[52:56, 5:60] = True                   # long thin bar: fails Rl
        label, num, area, sol, major, minor, rl, k = ch6.component_criteria(bw, Ra=500, Rc=0.9, Rl=2, conn=4)
        assert num == 3
        assert 0 not in k.tolist()
        assert sorted(k.tolist()) == [1, 2]

    def test_empty_mask(self):
        label, num, *_rest, k = ch6.component_criteria(np.zeros((10, 10), bool), 100, 0.9, 2)
        assert num == 0 and k.size == 0


class TestL1Misc:
    def test_regional_maxima_by_reconstruction_equals_imregionalmax(self):
        """Eqs. (6.57)/(6.58): both forms must equal ``imregionalmax``."""
        rng = np.random.default_rng(13)
        for img in (rng.integers(0, 5, size=(20, 25)).astype(float),
                    bwdist(~synth.fig_6_16_circles(), "cityblock").astype(float)):
            a = regional_maxima_by_reconstruction(img, 8, form="6.57")
            b = regional_maxima_by_reconstruction(img, 8, form="6.58")
            assert np.array_equal(a, imregionalmax(img, 8))
            assert np.array_equal(b, imregionalmax(img, 8))

    def test_bwperim_of_a_square(self):
        bw = np.zeros((9, 9), bool)
        bw[2:7, 2:7] = True
        p8 = bwperim(bw, 8)
        assert p8.sum() == 16 and not p8[4, 4]

    def test_fig_6_16_matches_the_printed_specification(self):
        """§6.5.2 caption: a 110x186 binary image, a 61-px and a 9-px diameter circle."""
        c = synth.fig_6_16_circles()
        assert c.shape == (110, 186)
        L = label_components(c, 8)
        assert L.max() == 2
        sizes = []
        for k in (1, 2):
            r, cc = np.nonzero(L == k)
            sizes.append((r.max() - r.min() + 1, cc.max() - cc.min() + 1))
        assert sorted(sizes) == [(9, 9), (61, 61)]

    def test_homomorphic_butterworth_runs(self):
        rng = np.random.default_rng(14)
        im = rng.random((32, 40)) * 255.0
        out = homomorphic_butterworth(im, 10.0, 1.0, matlab_bug=True)
        assert out.shape == im.shape and np.all(np.isfinite(out))


# =====================================================================================================================
# L2 — MATLAB parity: the Xu & Prince toolbox (unit.mat)
# =====================================================================================================================
class TestL2Del2:
    @pytest.mark.parametrize("name", sorted(FX.del2_fixtures()))
    def test_del2(self, name):
        assert maxdiff(del2(FX.del2_fixtures()[name]), ref("unit")[f"d_{name}"]) == 0.0

    def test_del2_spacings(self):
        a = FX.del2_fixtures()["del2_4x5"]
        d = ref("unit")
        assert maxdiff(del2(a, 2), d["dh_del2_4x5"]) == 0.0
        assert maxdiff(del2(a, 2, 3), d["dxy_del2_4x5"]) == 0.0
        assert maxdiff(del2(a, np.array([0., 1, 3, 6, 10]), np.array([0., 2, 5, 9])),
                       d["dvec_del2_4x5"]) == 0.0
        assert maxdiff(del2(FX.del2_fixtures()["del2_rand"], 0.5), d["dh_del2_rand"]) == 0.0

    def test_four_del2_matlab(self):
        assert maxdiff(4 * del2(FX.del2_fixtures()["del2_rand"]), ref("unit")["d5_del2_rand"]) == 0.0


class TestL2BoundMirror:
    @pytest.mark.parametrize("name", sorted(FX.boundmirror_fixtures()))
    def test_expand_shrink_ensure(self, name):
        a = FX.boundmirror_fixtures()[name]
        d = ref("unit")
        e = SN.bound_mirror_expand(a)
        assert maxdiff(e, d[f"bme_{name}"]) == 0.0
        assert maxdiff(SN.bound_mirror_shrink(e), d[f"bms_{name}"]) == 0.0
        err = mstr(d[f"bmnerr_{name}"])
        if err:
            with pytest.raises(ValueError):
                SN.bound_mirror_ensure(e)
        else:
            assert maxdiff(SN.bound_mirror_ensure(e), d[f"bmn_{name}"]) == 0.0

    def test_matlab_error_text_matches(self):
        assert "smaller than 3" in mstr(ref("unit")["bmn_small_err"])


class TestL2Gradient2:
    @pytest.mark.parametrize("name", sorted(FX.gradient2_fixtures()))
    def test_gradient2(self, name):
        a = FX.gradient2_fixtures()[name]
        d = ref("unit")
        gx, gy = SN.gradient2(a)
        assert maxdiff(gx, d[f"gx_{name}"]) == 0.0
        assert maxdiff(gy, d[f"gy_{name}"]) == 0.0
        z = SN.gradient2_complex(a)
        assert maxdiff(np.real(z), np.real(d[f"gc_{name}"])) == 0.0
        assert maxdiff(np.imag(z), np.imag(d[f"gc_{name}"])) == 0.0
        assert maxdiff(SN.gradient2_magnitude(a), d[f"gm_{name}"]) <= 1e-12

    def test_spacing_arguments(self):
        d = ref("unit")
        gx, gy = SN.gradient2(FX.gradient2_fixtures()["g2_9x11"], 0.5, 2)
        assert maxdiff(gx, d["gsx"]) == 0.0 and maxdiff(gy, d["gsy"]) == 0.0
        gx, gy = SN.gradient2(FX.gradient2_fixtures()["g2_3x5"],
                              np.array([0., 1, 3, 6, 10]), np.array([0., 2, 5]))
        assert maxdiff(gx, d["gvx"]) == 0.0 and maxdiff(gy, d["gvy"]) == 0.0

    def test_gradient2_equals_matlab_gradient(self):
        """``gradient2.m`` is MATLAB-4 ``gradient``; at unit spacing it equals the modern builtin."""
        d = ref("unit")
        gx, gy = SN.gradient2(FX.gradient2_fixtures()["g2_9x11"])
        assert maxdiff(gx, d["gmx"]) == 0.0 and maxdiff(gy, d["gmy"]) == 0.0


class TestL2Convolution:
    @pytest.mark.parametrize("iname", ["xc_I", "xc_I2"])
    @pytest.mark.parametrize("gname", ["xc_G3", "xc_G4", "xc_G5x3"])
    def test_xconv2(self, iname, gname):
        f = FX.xconv_fixtures()
        d = ref("unit")
        got = SN.xconv2(f[iname], f[gname])
        assert maxdiff(got, d[f"xc_{iname}_{gname}"]) < 1e-9
        assert maxdiff(got, d[f"cv_{iname}_{gname}"]) < 1e-9   # the header's "differences under 1e-10"

    @pytest.mark.parametrize("k,s", list(enumerate((0.5, 1.0, 2.0, 4.0, 5.0))))
    def test_gaussian_mask_and_blur(self, k, s):
        d = ref("unit")
        assert maxdiff(SN.gaussian_mask(1.0, s), d[f"gmask{k}"]) < 1e-18
        assert maxdiff(SN.gaussian_mask(2.5, s), d[f"gmask{k}b"]) < 1e-17
        assert maxdiff(SN.gaussian_blur(FX.xconv_fixtures()["xc_I"], s), d[f"gblur{k}"]) < 1e-10


class TestL2SnakeUnit:
    @pytest.mark.parametrize("name", sorted(FX.snakeindex_fixtures()))
    def test_snakeindex(self, name):
        got = SN.snakeindex(FX.snakeindex_fixtures()[name].ravel())
        assert maxdiff(got, np.asarray(ref("unit")[f"si_{name}"]).ravel()) == 0.0

    @pytest.mark.parametrize("tag,xn,yn", [("circle", "snk_circle_x", "snk_circle_y"),
                                           ("small", "snk_small_x", "snk_small_y"),
                                           ("square", "snk_square_x", "snk_square_y"),
                                           ("dup", "snk_dup_x", "snk_dup_y"),
                                           ("star", "snk_star_x", "snk_star_y"),
                                           ("n8", "snk_n8_x", "snk_n8_y")])
    @pytest.mark.parametrize("j,dmax,dmin", [(0, 1.0, 0.0), (1, 2.0, 0.5), (2, 4.0, 1.0)])
    def test_snakeinterp(self, tag, xn, yn, j, dmax, dmin):
        f = FX.snake_fixtures()
        d = ref("unit")
        err = mstr(d[f"serr_{tag}_{j}"])
        if err:
            # the shipped snakeinterp.m's own admitted removal bug: MATLAB raises "Index exceeds array bounds"
            with pytest.raises((ValueError, RuntimeError)):
                SN.snakeinterp(f[xn].ravel(), f[yn].ravel(), dmax, dmin)
            return
        xi, yi = SN.snakeinterp(f[xn].ravel(), f[yn].ravel(), dmax, dmin)
        assert maxdiff(xi, np.asarray(d[f"sx_{tag}_{j}"]).ravel()) == 0.0
        assert maxdiff(yi, np.asarray(d[f"sy_{tag}_{j}"]).ravel()) == 0.0

    @pytest.mark.parametrize("i,N,al,be", [(i, *c) for i, c in enumerate(
        [(8, 0.05, 0.0), (8, 0.05, 0.1), (8, 0.0, 0.1), (12, 0.05, 0.0), (37, 0.05, 0.0), (37, 0.05, 0.1),
         (126, 0.05, 0.0), (126, 0.1, 0.2), (3, 0.05, 0.0), (4, 0.05, 0.1), (5, 0.05, 0.1)])])
    def test_snake_matrix_and_inverse(self, i, N, al, be):
        d = ref("unit")
        A = SN.snake_matrix(N, al, be)
        assert maxdiff(A, d[f"Amat{i}"]) == 0.0
        assert maxdiff(np.linalg.inv(A + np.eye(N)), d[f"invAI{i}"]) < 1e-12

    def test_snake_matrix_nonunit_gamma(self):
        d = ref("unit")
        A = SN.snake_matrix(20, 0.05, 0.03)
        assert maxdiff(A, d["Amat_g"]) == 0.0
        assert maxdiff(np.linalg.inv(A + 2.5 * np.eye(20)), d["invAI_g"]) < 1e-12

    @pytest.mark.parametrize("j,iters", [(0, 1), (1, 5), (2, 20)])
    def test_snakedeform_dense(self, j, iters):
        f = FX.snake_fixtures()
        d = ref("unit")
        x, y = SN.snakedeform(f["snk_circle_x"].ravel(), f["snk_circle_y"].ravel(), 0.05, 0.0, 1.0, 0.5,
                              f["fld_px"], f["fld_py"], iters, solver="dense")
        assert maxdiff(x, np.asarray(d[f"dfx{j}"]).ravel()) < 1e-10
        assert maxdiff(y, np.asarray(d[f"dfy{j}"]).ravel()) < 1e-10
        x, y = SN.snakedeform(f["snk_n8_x"].ravel(), f["snk_n8_y"].ravel(), 0.05, 0.1, 1.0, 0.5,
                              f["fld_px"], f["fld_py"], iters, solver="dense")
        assert maxdiff(x, np.asarray(d[f"dgx{j}"]).ravel()) < 1e-12
        assert maxdiff(y, np.asarray(d[f"dgy{j}"]).ravel()) < 1e-12

    @pytest.mark.parametrize("j,iters", [(0, 1), (1, 5), (2, 20)])
    def test_snakedeform_circulant(self, j, iters):
        """The FFT solver against the *same MATLAB reference* — the `near` label's measured bound."""
        f = FX.snake_fixtures()
        d = ref("unit")
        x, y = SN.snakedeform(f["snk_circle_x"].ravel(), f["snk_circle_y"].ravel(), 0.05, 0.0, 1.0, 0.5,
                              f["fld_px"], f["fld_py"], iters, solver="circulant")
        assert maxdiff(x, np.asarray(d[f"dfx{j}"]).ravel()) < 1e-9
        assert maxdiff(y, np.asarray(d[f"dfy{j}"]).ravel()) < 1e-9

    def test_snakedeform_ring_field(self):
        f = FX.snake_fixtures()
        d = ref("unit")
        for solver in ("dense", "circulant"):
            x, y = SN.snakedeform(f["snk_circle_x"].ravel(), f["snk_circle_y"].ravel(), 0.05, 0.0, 1.0, 0.5,
                                  f["fld_ring_px"], f["fld_ring_py"], 30, solver=solver)
            assert maxdiff(x, np.asarray(d["drx"]).ravel()) < 1e-9
            assert maxdiff(y, np.asarray(d["dry"]).ravel()) < 1e-9

    def test_interp2_star_linear(self):
        """``interp2(fx, x, y, '*linear', 0)`` — bilinear with extrapolation value 0."""
        f = FX.snake_fixtures()
        d = ref("unit")
        cx, cy = f["snk_circle_x"].ravel(), f["snk_circle_y"].ravel()
        assert maxdiff(interp2(f["fld_px"], cy - 1, cx - 1, method="bilinear", fill=0.0), d["ip_in"]) == 0.0
        assert maxdiff(interp2(f["fld_px"], cy * 3 - 21, cx * 3 - 41, method="bilinear", fill=0.0),
                       d["ip_out"]) == 0.0


# =====================================================================================================================
# L2 — GVF (gvf.mat)
# =====================================================================================================================
class TestL2GVF:
    @pytest.mark.parametrize("name", sorted(FX.gvf_fixtures()))
    @pytest.mark.parametrize("iters", [1, 2, 5, 30])
    def test_gvf_fixture(self, name, iters):
        u, v = SN.gvf(FX.gvf_fixtures()[name], 0.1, iters)
        d = ref("gvf")
        assert maxdiff(u, d[f"u_{name}_{iters}"]) == 0.0
        assert maxdiff(v, d[f"v_{name}_{iters}"]) == 0.0

    @pytest.mark.parametrize("iters", [100, 250, 500])
    def test_gvf_many_iterations(self, iters):
        u, v = SN.gvf(FX.gvf_fixtures()["gvf_u"], 0.1, iters)
        d = ref("gvf")
        assert maxdiff(u, d[f"u_gvf_u_{iters}"]) == 0.0
        assert maxdiff(v, d[f"v_gvf_u_{iters}"]) == 0.0

    @pytest.mark.parametrize("mu,key", [(0.25, "mu"), (0.02, "mu2")])
    def test_gvf_mu(self, mu, key):
        u, v = SN.gvf(FX.gvf_fixtures()["gvf_u"], mu, 20)
        d = ref("gvf")
        assert maxdiff(u, d[f"u_{key}"]) == 0.0 and maxdiff(v, d[f"v_{key}"]) == 0.0

    def test_gvf_zero_iterations_is_the_gradient(self):
        u, v = SN.gvf(FX.gvf_fixtures()["gvf_u"], 0.1, 0)
        d = ref("gvf")
        assert maxdiff(u, d["u_it0"]) == 0.0 and maxdiff(v, d["v_it0"]) == 0.0

    @pytest.mark.parametrize("name,tag,num", [("test8.jpg", "t8", 150), ("alg_seg_gray.jpg", "a", 500)])
    def test_gvf_force_field_on_book_images(self, name, tag, num):
        """Stages 1 and 4-7 of ``GVF_distance.m`` on the two images the chapter's scripts read."""
        d = ref("gvf")
        rgb = book_image(name)
        gray = rgb2gray_matlab(rgb)
        assert maxdiff(gray.astype(float), d["I8" if tag == "t8" else "Ia"].astype(float)) == 0.0
        f2, u, v, px, py = ch6.gvf_force_field(gray, num=num)
        assert maxdiff(f2, d["f2_t8" if tag == "t8" else "f2_a"]) == 0.0
        assert maxdiff(u, d[f"u_{tag}"]) == 0.0
        assert maxdiff(v, d[f"v_{tag}"]) == 0.0
        assert maxdiff(px, d[f"px_{tag}"]) == 0.0
        assert maxdiff(py, d[f"py_{tag}"]) == 0.0

    @pytest.mark.parametrize("iters", [5, 30, 100, 250])
    def test_gvf_fig_6_16_iterations(self, iters):
        """Figs. 6.16(b)-(e): the capture-range experiment, on the port's own 110x186 fixture."""
        d = ref("extra")
        f = SN.gradient2_magnitude(synth.fig_6_16_circles().astype(float))
        assert maxdiff(f, d["f616p"]) == 0.0
        u, v = SN.gvf(f, 0.1, iters)
        assert maxdiff(u, d[f"u616p_{iters}"]) == 0.0
        assert maxdiff(v, d[f"v616p_{iters}"]) == 0.0


# =====================================================================================================================
# L2 — regionprops (regionprops.mat)  -- Risk R2, the chapter's decision path
# =====================================================================================================================
class TestL2Regionprops:
    @pytest.mark.parametrize("name", sorted(FX.shape_fixtures()))
    def test_all_properties_on_a_shape(self, name):
        d = ref("regionprops")
        bw = FX.shape_fixtures()[name].astype(bool)
        got = regionprops(bw, RP_PROPS, conn=8)
        assert len(got) == int(d[f"n_{name}"][0, 0])
        if not got:
            return
        for p in RP_PROPS:
            mine = np.array([np.atleast_1d(getattr(s, p)) for s in got], dtype=float)
            assert maxdiff(mine, np.atleast_2d(d[f"{p.lower()}_{name}"])) < 1e-12, (name, p)

    def test_probe_axis_lengths_settle_the_analysis_dispute(self):
        """analysis/ch06.md §0.5 recorded 10.35814 / 6.76639; the port computes 10.358106 / 6.766408.

        MATLAB R2025a itself (``sprintf('%.17g', ...)``) is the arbiter.
        """
        d = ref("regionprops")
        s = regionprops(FX.shape_fixtures()["sh_probe"].astype(bool), "all")[0]
        ml_major = float(mstr(d["probe_major_str"]))
        ml_minor = float(mstr(d["probe_minor_str"]))
        assert abs(ml_major - 10.358106483626367) < 1e-12
        assert abs(ml_minor - 6.7664075482213342) < 1e-12
        assert abs(s.MajorAxisLength - ml_major) < 1e-12
        assert abs(s.MinorAxisLength - ml_minor) < 1e-12
        assert abs(ml_major - 10.35814) > 1e-5      # the analysis figure is NOT what MATLAB returns
        # everything else of the §0.5 probe is reproduced exactly
        assert s.Area == 53 and s.ConvexArea == 56
        assert abs(s.Solidity - 0.94643) < 5e-6
        assert abs(s.Perimeter - 25.42200) < 5e-6
        assert abs(s.Orientation - (-26.04715)) < 1e-4
        assert np.allclose(s.Centroid, [6.92453, 6.09434], atol=1e-5)
        assert np.allclose(s.BoundingBox, [2.5, 2.5, 9, 7])

    @pytest.mark.parametrize("name", ["sh_L", "sh_peanut", "sh_ring", "sh_diag", "sh_single", "sh_hline"])
    def test_convex_image_and_hull(self, name):
        d = ref("regionprops")
        s = regionprops(FX.shape_fixtures()[name].astype(bool), ("ConvexImage", "ConvexHull"))[0]
        assert np.array_equal(s.ConvexImage, np.asarray(d[f"ci_{name}"]).astype(bool))
        mine = {(round(a, 9), round(b, 9)) for a, b in s.ConvexHull}
        ml = np.atleast_2d(d[f"ch_{name}"])
        theirs = {(round(a, 9), round(b, 9)) for a, b in ml}
        # DEVIATION `near`: MATLAB's convhull (Qhull 'Qt') keeps the **collinear** points of a hull edge, the
        # port keeps only the extreme vertices.  The polygon -- and therefore ConvexImage / ConvexArea /
        # Solidity, the quantities the chapter thresholds -- is identical.
        assert mine <= theirs
        assert abs(POLY.polyarea(s.ConvexHull[:, 0], s.ConvexHull[:, 1])
                   - POLY.polyarea(ml[:, 0], ml[:, 1])) < 1e-9

    @pytest.mark.parametrize("name", ["sh_multi", "sh_rand0", "sh_rand3"])
    @pytest.mark.parametrize("conn", [4, 8])
    def test_label_matrix_and_loop_form(self, name, conn):
        """``regionprops(L, ...)`` and the shipped ``regionprops(label == n, ...)`` loop must agree."""
        d = ref("regionprops")
        L = label_components(FX.shape_fixtures()[name].astype(bool), conn)
        got = regionprops(L, RP_PROPS)
        tag = f"{name}_L{conn}"
        assert len(got) == int(d[f"n_{tag}"][0, 0])
        for p in RP_PROPS:
            mine = np.array([np.atleast_1d(getattr(s, p)) for s in got], dtype=float)
            assert maxdiff(mine, np.atleast_2d(d[f"{p.lower()}_{tag}"])) < 1e-12, (tag, p)
        for key in RP_PROPS:
            mine = np.array([np.atleast_1d(getattr(s, key)) for s in got], dtype=float)
            assert maxdiff(mine, np.atleast_2d(d[f"loop{key}_{name}_{conn}"])) < 1e-12, key

    @pytest.mark.parametrize("name,tag", [("alg_seg_gray.jpg", "alg"), ("test8.jpg", "t8")])
    def test_real_ch6_masks(self, name, tag):
        """Every component of the Otsu masks the chapter actually processes (83 / 30 components)."""
        d = ref("regionprops")
        gray = rgb2gray_matlab(book_image(name))
        level, _ = graythresh(gray)
        bw = im2bw(gray, level)
        assert int((bw != np.asarray(d[f"bwm_{tag}"]).astype(bool)).sum()) == 0
        L = label_components(bw, 4)
        assert int((L != np.asarray(d[f"lab_{tag}"])).sum()) == 0
        got = regionprops(L, RP_PROPS)
        assert len(got) == int(d[f"n_real_{tag}"][0, 0])
        for p in RP_PROPS:
            mine = np.array([np.atleast_1d(getattr(s, p)) for s in got], dtype=float)
            assert maxdiff(mine, np.atleast_2d(d[f"{p.lower()}_real_{tag}"])) < 1e-9, p
            assert maxdiff(mine, np.atleast_2d(d[f"loop{p}_{tag}"])) < 1e-9, p

    @pytest.mark.parametrize("name,tag", [("alg_seg_gray.jpg", "alg"), ("test8.jpg", "t8")])
    def test_decision_set_k_matches(self, name, tag):
        """The criteria of ch9 p. 205 select the same components as MATLAB (Rc = 0.9, Rl = 2 boundaries)."""
        d = ref("regionprops")
        gray = rgb2gray_matlab(book_image(name))
        bw = im2bw(gray, graythresh(gray)[0])
        _, _, area, sol, major, minor, rl, k = ch6.component_criteria(bw, 2500, 0.9, 2, conn=4)
        a_ml = np.asarray(d[f"area_real_{tag}"]).ravel()
        s_ml = np.asarray(d[f"solidity_real_{tag}"]).ravel()
        l_ml = np.asarray(d[f"majoraxislength_real_{tag}"]).ravel()
        w_ml = np.asarray(d[f"minoraxislength_real_{tag}"]).ravel()
        k_ml = np.unique(np.concatenate([np.nonzero(a_ml > 2500)[0], np.nonzero(s_ml < 0.9)[0],
                                         np.nonzero(l_ml / w_ml > 2)[0]]))
        assert k.tolist() == k_ml.tolist()


# =====================================================================================================================
# L2 — polygon geometry (geom.mat)
# =====================================================================================================================
class TestL2Polygon:
    PG = ["pg_test", "pg_cw", "pg_tri", "pg_sq"] + [f"pg_rand{k}" for k in range(10)]

    @pytest.mark.parametrize("name", PG)
    def test_polygeom(self, name):
        d = ref("geom")
        f = FX.polygon_fixtures()
        geom, iner, cpmo = POLY.polygeom(f[f"{name}_x"].ravel(), f[f"{name}_y"].ravel())
        assert maxdiff(geom, d[f"geom_{name}"]) < 1e-9
        assert maxdiff(iner, d[f"iner_{name}"]) < 1e-8
        cref = np.asarray(d[f"cpmo_{name}"]).ravel()
        assert abs(cpmo[0] - cref[0]) < 1e-9 and abs(cpmo[2] - cref[2]) < 1e-9
        assert abs(cpmo[4] - cref[4]) < 1e-9
        # DEVIATION `near`: an eigenvector's sign is arbitrary, so the principal-axis angles agree modulo pi.
        for j in (1, 3):
            dd = (cpmo[j] - cref[j]) % np.pi
            assert min(dd, np.pi - dd) < 1e-9, (name, j, cpmo[j], cref[j])

    def test_polygeom_angle_convention_differs_from_matlab_eig(self):
        """Measured: R2025a's ``eig`` picks the opposite eigenvector, so its ang1 is the port's minus pi."""
        d = ref("geom")
        f = FX.polygon_fixtures()
        flips = 0
        for name in self.PG:
            _, _, c = POLY.polygeom(f[f"{name}_x"].ravel(), f[f"{name}_y"].ravel())
            cref = np.asarray(d[f"cpmo_{name}"]).ravel()
            if abs(abs(c[1] - cref[1]) - np.pi) < 1e-9:
                flips += 1
        assert flips >= 10                    # the majority of cases flip; the axes are identical mod pi
        _, _, c = POLY.polygeom(f["pg_test_x"].ravel(), f["pg_test_y"].ravel())
        assert abs(np.degrees(c[1]) - 30.0) < 1e-2   # the port matches the M-file's own documented self-test

    def test_original_minboundrect_does_not_run_in_r2025a(self):
        """Evidence for the deviation: the shipped 2007 FEX file calls ``convhull(x,y,{'Qt'})``."""
        assert "CONVHULL no longer supports" in mstr(ref("geom")["mbr_orig_err"])

    @pytest.mark.parametrize("name", ["pc_one", "pc_two", "pc_three", "pc_square"] +
                             [f"pc_rand{k}" for k in range(6)])
    @pytest.mark.parametrize("metric", ["a", "p"])
    def test_minboundrect(self, name, metric):
        d = ref("geom")
        f = FX.polygon_fixtures()
        rx, ry, area, per = POLY.minboundrect(f[f"{name}_x"].ravel(), f[f"{name}_y"].ravel(), metric)
        assert maxdiff(rx, np.asarray(d[f"rx_{name}_{metric}"]).ravel()) < 1e-12
        assert maxdiff(ry, np.asarray(d[f"ry_{name}_{metric}"]).ravel()) < 1e-12
        assert abs(area - float(np.asarray(d[f"ar_{name}_{metric}"]).ravel()[0])) < 1e-12
        # for two points MATLAB returns the perimeter as a 1x2 vector (diff of a 3-element x)
        assert abs(per - float(np.asarray(d[f"pe_{name}_{metric}"]).ravel()[0])) < 1e-12

    def test_minboundrect_unit_square_matlab(self):
        d = ref("geom")
        assert abs(float(d["aru"][0, 0]) - 1.0) < 0.02

    @pytest.mark.parametrize("name", ["pc_collinear", "pc_dup"])
    def test_degenerate_clouds_matlab_errors(self, name):
        """MATLAB's modern ``convhull`` rejects collinear / duplicate clouds; the port returns a degenerate rect."""
        d = ref("geom")
        assert "convex hull" in mstr(d[f"mbrerr_{name}_a"])
        f = FX.polygon_fixtures()
        rx, ry, area, per = POLY.minboundrect(f[f"{name}_x"].ravel(), f[f"{name}_y"].ravel())
        assert rx.size == 5 and area == 0.0

    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_convhull_vertex_set(self, k):
        d = ref("geom")
        f = FX.polygon_fixtures()
        mine = POLY.convhull(f[f"pc_rand{k}_x"].ravel(), f[f"pc_rand{k}_y"].ravel())
        theirs = np.asarray(d[f"kh{k}"]).ravel().astype(int) - 1
        assert set(mine.tolist()) == set(theirs.tolist())
        assert mine[0] == mine[-1]

    @pytest.mark.parametrize("name", [f"pm{k}" for k in range(5)] + ["pm_tri", "pm_int", "pm_half"])
    def test_poly2mask_and_roipoly(self, name):
        d = ref("geom")
        f = FX.polygon_fixtures()
        x, y = f[f"{name}_x"].ravel(), f[f"{name}_y"].ravel()
        assert int((POLY.poly2mask(x, y, 21, 21) != np.asarray(d[f"mask_{name}"]).astype(bool)).sum()) == 0
        assert int((POLY.poly2mask(x, y, 15, 25) != np.asarray(d[f"mask2_{name}"]).astype(bool)).sum()) == 0
        assert int((POLY.roipoly(np.zeros((21, 21)), x, y)
                    != np.asarray(d[f"rmask_{name}"]).astype(bool)).sum()) == 0

    def test_poly2mask_of_the_regionprops_hull(self):
        """The ``ConvexImage`` chain that produces ``Solidity`` (the Rc criterion)."""
        d = ref("geom")
        got = POLY.poly2mask(np.asarray(d["ch_probe_x"]).ravel(), np.asarray(d["ch_probe_y"]).ravel(), 7, 9)
        assert int((got != np.asarray(d["mask_hull"]).astype(bool)).sum()) == 0

    @pytest.mark.parametrize("k", list(range(7)))
    def test_clip_polygon_rect_vs_polybool(self, k):
        """``polybool`` rotates the start vertex and reverses the traversal: compare **sets**, never order.

        Both close the ring: the port appends a copy of the first vertex as ``polybool`` does (ch06 verification
        open item 2 -- with the chapter's ``Dmin = 0`` that duplicate survives ``snakeinterp``, and closing it took
        ``GVF_distance.m``'s ``bw1`` from 61 to 7 differing pixels of 31 730).
        """
        d = ref("geom")
        f = FX.polygon_fixtures()
        x, y = f[f"pb{k}_x"].ravel(), f[f"pb{k}_y"].ravel()
        cx, cy = POLY.clip_polygon_rect(x, y, (0.0, 40.0), (0.0, 30.0))
        rx = np.asarray(d[f"pbx{k}"]).ravel()
        ry = np.asarray(d[f"pby{k}"]).ravel()
        mine = {(round(a, 9), round(b, 9)) for a, b in zip(cx, cy)}
        theirs = {(round(a, 9), round(b, 9)) for a, b in zip(rx, ry)}
        assert mine == theirs
        # Both rings are closed, so the vertex counts now agree exactly.
        assert rx.size == cx.size
        if cx.size:
            assert (cx[0], cy[0]) == (cx[-1], cy[-1])
            assert (rx[0], ry[0]) == (rx[-1], ry[-1])
        m_py = POLY.poly2mask(cx, cy, 30, 40) if cx.size >= 3 else np.zeros((30, 40), bool)
        assert int((m_py != np.asarray(d[f"pbm{k}"]).astype(bool)).sum()) == 0

    def test_clip_after_snakeinterp_matches_the_ch6_call_order(self):
        d = ref("geom")
        f = FX.polygon_fixtures()
        xi, yi = SN.snakeinterp(f["pb1_x"].ravel(), f["pb1_y"].ravel(), 1.0, 0.0)
        assert maxdiff(xi, np.asarray(d["cxi"]).ravel()) == 0.0
        cx, cy = POLY.clip_polygon_rect(xi, yi, (0.0, 40.0), (0.0, 30.0))
        theirs = {(round(a, 9), round(b, 9))
                  for a, b in zip(np.asarray(d["pbix"]).ravel(), np.asarray(d["pbiy"]).ravel())}
        assert {(round(a, 9), round(b, 9)) for a, b in zip(cx, cy)} == theirs

    @pytest.mark.parametrize("name", ["pg_test", "pg_cw", "pg_tri", "pg_sq", "pg_bow"]
                             + [f"pg_rand{k}" for k in range(10)]
                             + ["pc_collinear", "pc_dup", "pc_two", "pm_int", "pm_half"])
    def test_polyarea(self, name):
        """MATLAB ``polyarea`` on 20 polygons, each open and closed (reviewer finding: this had no L2 row).

        Covers the clockwise case, a self-intersecting bowtie (0 on both sides), collinear and duplicate point
        sets, a two-point and a one-point degenerate input.  A repeated first vertex must not change the area.
        """
        d = ref("polyarea")
        f = FX.polygon_fixtures()
        x = f[f"{name}_x"].ravel()
        y = f[f"{name}_y"].ravel()
        assert abs(POLY.polyarea(x, y) - float(np.asarray(d[f"pa_{name}"]).ravel()[0])) < 1e-12
        assert abs(POLY.polyarea(np.r_[x, x[0]], np.r_[y, y[0]])
                   - float(np.asarray(d[f"pac_{name}"]).ravel()[0])) < 1e-12

    def test_polyarea_degenerate(self):
        d = ref("polyarea")
        assert POLY.polyarea([3.0], [4.0]) == float(np.asarray(d["pa_one"]).ravel()[0]) == 0.0
        assert abs(POLY.polyarea(np.asarray(d["ch_probe_x"]).ravel(), np.asarray(d["ch_probe_y"]).ravel())
                   - float(np.asarray(d["pa_hull"]).ravel()[0])) < 1e-12

    def test_polyxpoly(self):
        d = ref("geom")
        f = FX.polygon_fixtures()
        xi, yi, ii = POLY.polyxpoly(f["px1_x"].ravel(), f["px1_y"].ravel(),
                                    f["px2_x"].ravel(), f["px2_y"].ravel())
        assert maxdiff(np.sort(xi), np.sort(np.asarray(d["xi12"]).ravel())) < 1e-9
        assert maxdiff(np.sort(yi), np.sort(np.asarray(d["yi12"]).ravel())) < 1e-9
        assert np.array_equal(np.sort(ii, axis=0), np.sort(np.asarray(d["ii12"]) - 1, axis=0))
        xi, _, _ = POLY.polyxpoly(f["px1_x"].ravel(), f["px1_y"].ravel(),
                                  f["px3_x"].ravel(), f["px3_y"].ravel())
        assert xi.size == 0 and np.asarray(d["xi13"]).size == 0


# =====================================================================================================================
# L2 — extras (extra.mat): bwperim, Eq. 6.57/6.58, homofil.m, Fig. 6.14 recipe
# =====================================================================================================================
class TestL2Extras:
    @pytest.mark.parametrize("name", ["sh_probe", "sh_ring", "sh_L", "sh_disc", "sh_border", "sh_rand0",
                                      "sh_diag", "sh_single", "sh_multi"])
    @pytest.mark.parametrize("conn", [4, 8])
    def test_bwperim(self, name, conn):
        d = ref("extra")
        bw = FX.shape_fixtures()[name].astype(bool)
        got = bwperim(bw, conn)
        assert int((got != np.asarray(d[f"bp{conn}_p_{name}"]).astype(bool)).sum()) == 0

    def test_fig_6_14_recipe_against_matlab(self):
        d = ref("extra")
        A = synth.FIG_6_14_IMAGE
        assert int((A != np.asarray(d["A614"]).astype(bool)).sum()) == 0
        D = bwdist(~A, "cityblock")
        assert maxdiff(D, d["D614"]) == 0.0
        init = ch6.initialize_contours(A, se_radius=3)
        assert int((init.minima_map != np.asarray(d["M614"]).astype(bool)).sum()) == 0
        assert int((init.dis != np.asarray(d["dis614"]).astype(float)).sum()) == 0
        assert init.num == int(d["n614"][0, 0])
        assert np.allclose(init.centroids, np.atleast_2d(d["cen614"]))
        assert int((imregionalmax(D.astype(float) * A) != np.asarray(d["rmax614"]).astype(bool)).sum()) == 0

    def test_eq_6_57_and_6_58_vs_matlab(self):
        d = ref("extra")
        D = bwdist(~FX.shape_fixtures()["sh_peanut"].astype(bool), "cityblock")
        assert maxdiff(D, d["Dff"]) == 0.0
        assert int((imregionalmax(D.astype(float)) != np.asarray(d["rmx"]).astype(bool)).sum()) == 0
        a = regional_maxima_by_reconstruction(D.astype(float), 8, form="6.57")
        b = regional_maxima_by_reconstruction(D.astype(float), 8, form="6.58")
        assert int((a != np.asarray(d["rmx"]).astype(bool)).sum()) == 0
        assert int((b != np.asarray(d["rmx"]).astype(bool)).sum()) == 0
        F = np.asarray(d["Frnd"], dtype=float)
        assert int((regional_maxima_by_reconstruction(F, 8) != np.asarray(d["rmxf"]).astype(bool)).sum()) == 0

    @pytest.mark.parametrize("k,dd,nn", [(0, 10.0, 1.0), (1, 30.0, 2.0), (2, 5.0, 4.0)])
    def test_homofil(self, k, dd, nn):
        """``homofil.m`` has no book section and no caller — but MATLAB can still run it (L2)."""
        d = ref("extra")
        rng = np.random.default_rng(609)
        _ = rng.random((110, 186)), rng.random((110, 186))  # keep the fixture RNG stream aligned
        im = np.asarray(loadmat(str(REF / "inputs2.mat"))["hf_im"], dtype=float)
        im2 = np.asarray(loadmat(str(REF / "inputs2.mat"))["hf_im2"], dtype=float)
        got = homomorphic_butterworth(im, dd, nn, matlab_bug=True)
        assert maxdiff(got, d[f"hf{k}"]) < 1e-9
        got2 = homomorphic_butterworth(im2, dd, nn, matlab_bug=True)
        assert maxdiff(got2, d[f"hg{k}"]) < 1e-9

    def test_contour_initialisation_on_a_synthetic_floe_field(self):
        d = ref("extra")
        img = synth.synthetic_floe_field()
        level, _ = graythresh(img)
        bw = im2bw(img, level)
        assert int((bw != np.asarray(d["bwf"]).astype(bool)).sum()) == 0
        init = ch6.initialize_contours(bw, se_radius=3)
        assert maxdiff(init.img_dist, d["Df"]) == 0.0
        assert int((init.minima_map != np.asarray(d["Mf"]).astype(bool)).sum()) == 0
        assert int((init.dis_dilated != np.asarray(d["disf"]).astype(float)).sum()) == 0
        assert init.num == int(d["nf"][0, 0])
        assert np.allclose(init.centroids, np.atleast_2d(d["cenf"]))
        assert np.allclose(init.radii, np.asarray(d["rf"]).ravel())


# =====================================================================================================================
# L2 — the original scripts
# =====================================================================================================================
class TestL2DistScript:
    """``for test/dist.m`` run verbatim by MATLAB (in the transposed frame it creates)."""

    @staticmethod
    def _pipeline():
        rgb = book_image("sea_ice_test.jpg")
        gray = rgb2gray_matlab(rgb).T                     # bw = im2bw(I', graythresh(I'))
        level, _ = graythresh(gray)
        bw = im2bw(gray, level)
        init = ch6.initialize_contours(bw, se_radius=5)
        return gray, level, bw, init

    def test_masks_and_distance(self):
        d = ref("dist_script")
        _, _, bw, init = self._pipeline()
        assert int((bw != np.asarray(d["bw"]).astype(bool)).sum()) == 0
        assert maxdiff(init.img_dist, d["img_Dist"]) == 0.0
        assert int((init.minima_map != np.asarray(d["Dis_img"]).astype(bool)).sum()) == 0
        assert int((init.dis != np.asarray(d["dis"]).astype(float)).sum()) == 0
        assert int((init.dis_dilated != np.asarray(d["dis1"]).astype(float)).sum()) == 0

    def test_find_order_is_column_major(self):
        """``[p, q] = find(dis == 1)`` uses MATLAB's column-major linear order."""
        d = ref("dist_script")
        _, _, _, init = self._pipeline()
        qq, pp = np.nonzero(init.dis.T == 1.0)            # column-major (row, col)
        assert maxdiff(pp + 1, np.asarray(d["p"]).ravel()) == 0.0
        assert maxdiff(qq + 1, np.asarray(d["q"]).ravel()) == 0.0

    def test_seeds_centroids_and_radii(self):
        d = ref("dist_script")
        _, _, _, init = self._pipeline()
        assert init.num == int(np.asarray(d["num_out"]).ravel()[0])
        assert int((init.label != np.asarray(d["label"])).sum()) == 0
        assert maxdiff(init.centroids, np.atleast_2d(d["CEN2"])) < 1e-12
        assert maxdiff(np.atleast_2d(d["CEN1"]), np.atleast_2d(d["CEN2"])) == 0.0
        # DEVIATION `near`: bwdist returns **single**, so MATLAB evaluates `img_Dist(...)/sqrt(2)` in single
        # precision (the reference array is float32); the port divides the promoted double.  <= 1e-6 absolute.
        assert maxdiff(init.radii, np.asarray(d["R0"]).ravel()) < 1e-6
        assert int((init.radii == 2.0).sum()) == int((np.asarray(d["R0"]).ravel() == 2.0).sum())

    def test_initial_circles(self):
        d = ref("dist_script")
        _, _, _, init = self._pipeline()
        x0 = np.asarray(d["X0"]).ravel()
        assert len(init.contours) == x0.size
        for k in (0, init.num // 2, init.num - 1):
            # single-precision r0 (see test_seeds_centroids_and_radii) -> the circle coordinates are float32
            assert np.asarray(x0[k]).dtype == np.float32          # MATLAB single, cf. the radii
            assert maxdiff(init.contours[k][0], np.asarray(x0[k]).ravel()) < 1e-4, k

    def test_transpose_relationship(self):
        """Risk R8: the script's mask is exactly the transpose of the normally-binarised one."""
        rgb = book_image("sea_ice_test.jpg")
        gray = rgb2gray_matlab(rgb)
        a = im2bw(gray.T, graythresh(gray.T)[0])
        b = im2bw(gray, graythresh(gray)[0])
        assert np.array_equal(a, b.T)

    def test_component_count(self):
        d = ref("dist_script")
        rgb = book_image("sea_ice_test.jpg")
        gray = rgb2gray_matlab(rgb).T
        bw = im2bw(gray, graythresh(gray)[0])
        assert int(label_components(bw, 4).max()) == int(np.asarray(d["l_num"]).ravel()[0])


class TestL2ForTest:
    """``for test/for_test.m`` -- the single manual contour on ``test8.jpg`` (Figs. 6.8/6.9 style)."""

    P = ch6.BOOK_PARAMS["for_test"]

    @classmethod
    def _pipeline(cls, solver="dense", iters=None):
        P = cls.P
        II = book_image("test8.jpg")
        I = rgb2gray_matlab(II)
        level, _ = graythresh(I)
        bw = im2bw(I, level)
        s1, s2 = bw.shape
        f2, u, v, px, py = ch6.gvf_force_field(I, sigma=P["sigma"], num=P["Num"], mu=P["mu"])
        x = P["x0"] + P["r"] * np.cos(ch6.CIRCLE_T)
        y = P["y0"] + P["r"] * np.sin(ch6.CIRCLE_T)
        xc, yc = POLY.clip_polygon_rect(x, y, (0.0, float(s2)), (0.0, float(s1)))
        xi, yi = SN.snakeinterp(xc, yc, P["Dmax"], P["Dmin"])
        it = P["iter"] if iters is None else iters
        blocks = []
        xs, ys = xi, yi
        for i in range(1, int(np.ceil(it / 5)) + 1):
            steps = 5 if i <= int(np.floor(it / 5)) else it - int(np.floor(it / 5)) * 5
            xs, ys = SN.snakedeform(xs, ys, P["alpha"], P["beta"], P["gamma"], P["kappa"], px, py, steps,
                                    solver=solver)
            xs, ys = SN.snakeinterp(xs, ys, P["Dmax"], P["Dmin"])
            blocks.append((xs.copy(), ys.copy()))
        bw_out = bw.copy()
        xx = np.ceil(xs).astype(np.int64)
        yy = np.ceil(ys).astype(np.int64)
        ok = (xx <= s2) & (yy <= s1) & (xx >= 1) & (yy >= 1)
        bw_out[yy[ok] - 1, xx[ok] - 1] = False
        return dict(bw=bw, f2=f2, u=u, v=v, px=px, py=py, xc=xc, yc=yc, xi=xi, yi=yi, blocks=blocks,
                    bw_out=bw_out, xx=xx, yy=yy, s1=s1, s2=s2)

    def test_field(self):
        d = ref("for_test")
        r = self._pipeline()
        assert int((r["bw"] != np.asarray(d["bw_top"]).astype(bool)).sum()) == 0
        assert maxdiff(r["f2"], d["f2"]) == 0.0
        # NOTE: `for_test.m` line 119 `[v, h] = size(bw)` **overwrites** the GVF field variable `v`; the
        # reference keeps the field under `u_gvf` / `v_gvf` (saved just before the shadowing statement).
        assert maxdiff(r["u"], d["u_gvf"]) == 0.0 and maxdiff(r["v"], d["v_gvf"]) == 0.0
        assert maxdiff(r["px"], d["px"]) == 0.0 and maxdiff(r["py"], d["py"]) == 0.0

    def test_quiver_grid(self):
        """The script's own 64x64 ``interp2`` grid, with the authors' swapped xSpace/ySpace roles."""
        d = ref("for_test")
        r = self._pipeline()
        s1, s2 = r["s1"], r["s2"]
        x_space = np.arange(1.0, s1 + 1e-9, s1 / 64.0)
        y_space = np.arange(1.0, s2 + 1e-9, s2 / 64.0)
        assert maxdiff(x_space, np.asarray(d["xSpace"]).ravel()) < 1e-12
        assert maxdiff(y_space, np.asarray(d["ySpace"]).ravel()) < 1e-12
        XX, YY = np.meshgrid(x_space, y_space)
        qx = interp2(r["px"], YY - 1.0, XX - 1.0, method="bilinear")
        qy = interp2(r["py"], YY - 1.0, XX - 1.0, method="bilinear")
        assert maxdiff(qx, d["qx"]) < 1e-12
        assert maxdiff(qy, d["qy"]) < 1e-12

    def test_clip_and_interp(self):
        """``clip_polygon_rect`` reproduces ``polybool``'s **closed** ring: same vertex set, same point count.

        (Contract fixed by ch06 verification open item 2; before the fix the port returned the open ring and
        was one point short here and after ``snakeinterp``.)
        """
        d = ref("for_test")
        r = self._pipeline()
        XPB = np.asarray(d["XPB"]).ravel()
        YPB = np.asarray(d["YPB"]).ravel()
        theirs = {(round(a, 9), round(b, 9)) for a, b in zip(XPB, YPB)}
        assert {(round(a, 9), round(b, 9)) for a, b in zip(r["xc"], r["yc"])} == theirs
        assert XPB[0] == XPB[-1] and YPB[0] == YPB[-1]          # polybool returns a closed ring
        assert r["xc"][0] == r["xc"][-1] and r["yc"][0] == r["yc"][-1]   # ...and so does the port now
        assert XPB.size == r["xc"].size == 127
        # the zero-length wrap segment sits at the same cyclic position on both sides
        def zero_segments(px, py):
            dd = np.abs(np.roll(px, -1) - px) + np.abs(np.roll(py, -1) - py)
            return np.nonzero(dd == 0)[0].tolist()
        assert zero_segments(r["xc"], r["yc"]) == zero_segments(XPB, YPB) == [126]
        # with Dmin = 0 the duplicate survives snakeinterp -> 252 points on both sides
        assert np.asarray(d["XI0"]).size == r["xi"].size == 252
        xi2, yi2 = SN.snakeinterp(XPB, YPB, 1.0, 0.0)           # start from MATLAB's own closed ring
        assert maxdiff(xi2, np.asarray(d["XI0"]).ravel()) == 0.0
        assert maxdiff(yi2, np.asarray(d["YI0"]).ravel()) == 0.0

    def test_contour_after_every_block_from_the_matlab_clip(self):
        """Fed MATLAB's own (closed) clip output, the port reproduces every 5-iteration block element-wise."""
        d = ref("for_test")
        r = self._pipeline()
        XPB = np.asarray(d["XPB"]).ravel()
        YPB = np.asarray(d["YPB"]).ravel()
        bx = np.asarray(d["XB"]).ravel()
        by = np.asarray(d["YB"]).ravel()
        xs, ys = SN.snakeinterp(XPB, YPB, 1.0, 0.0)
        worst = 0.0
        for i in range(bx.size):
            xs, ys = SN.snakedeform(xs, ys, 0.05, 0.0, 1.0, 0.5, r["px"], r["py"], 5, solver="dense")
            xs, ys = SN.snakeinterp(xs, ys, 1.0, 0.0)
            ref_x = np.asarray(bx[i]).ravel()
            ref_y = np.asarray(by[i]).ravel()
            assert xs.size == ref_x.size, (i, xs.size, ref_x.size)
            worst = max(worst, maxdiff(xs, ref_x), maxdiff(ys, ref_y))
        assert worst < 1e-6, worst

    def test_final_contour_is_the_same_curve(self):
        """After 50 iterations the two snakes are the same **curve**; the point count can differ by one.

        The port's clip is ``polybool``'s ring *rotated and reversed* (the compiled GPC library normalises the
        contour, see :meth:`test_residual_is_the_polybool_start_vertex`), so the float summation order inside
        ``inv(A + gammaI) @ rhs`` differs and one ``d > dmax`` insertion flips at block 2: 778 points instead
        of 777.  The curve is within 1e-3 px and the burnt mask is identical (``test_burnt_mask``).
        """
        d = ref("for_test")
        r = self._pipeline()
        mlx = np.asarray(np.asarray(d["XB"]).ravel()[-1]).ravel()
        mly = np.asarray(np.asarray(d["YB"]).ravel()[-1]).ravel()
        x, y = r["blocks"][-1]
        assert abs(x.size - mlx.size) <= 2, (x.size, mlx.size)   # measured 778 vs 777
        dist = np.hypot(x[:, None] - mlx[None, :], y[:, None] - mly[None, :])
        hausdorff = max(dist.min(axis=1).max(), dist.min(axis=0).max())
        assert hausdorff < 1e-3, hausdorff

    def test_residual_is_the_polybool_start_vertex(self):
        """Proof that the remaining difference is `polybool`'s vertex ordering, not the clip's geometry.

        The port's closed ring is MATLAB's rotated by 30 and reversed.  Realigned to MATLAB's start vertex it
        is bit-identical, and the evolution then reproduces MATLAB's 777 points to ~2e-6 px.
        """
        d = ref("for_test")
        r = self._pipeline()
        XPB = np.asarray(d["XPB"]).ravel()
        YPB = np.asarray(d["YPB"]).ravel()
        pts_ml = [(round(a, 12), round(b, 12)) for a, b in zip(XPB[:-1], YPB[:-1])]
        pts_py = [(round(a, 12), round(b, 12)) for a, b in zip(r["xc"][:-1], r["yc"][:-1])]
        k = pts_py.index(pts_ml[0])
        rot = pts_py[k:] + pts_py[:k]
        rev = rot[:1] + rot[1:][::-1]
        assert rev == pts_ml, "the port's ring is not a rotation+reversal of polybool's"
        ax = np.array([q[0] for q in rev] + [rev[0][0]])
        ay = np.array([q[1] for q in rev] + [rev[0][1]])
        assert maxdiff(ax, XPB) < 1e-9 and maxdiff(ay, YPB) < 1e-9
        xs, ys = SN.snakeinterp(ax, ay, 1.0, 0.0)
        for _ in range(10):
            xs, ys = SN.snakedeform(xs, ys, 0.05, 0.0, 1.0, 0.5, r["px"], r["py"], 5, solver="dense")
            xs, ys = SN.snakeinterp(xs, ys, 1.0, 0.0)
        mlx = np.asarray(np.asarray(d["XB"]).ravel()[-1]).ravel()
        mly = np.asarray(np.asarray(d["YB"]).ravel()[-1]).ravel()
        assert xs.size == mlx.size == 777
        dist = np.hypot(xs[:, None] - mlx[None, :], ys[:, None] - mly[None, :])
        assert max(dist.min(axis=1).max(), dist.min(axis=0).max()) < 1e-5

    def test_burnt_mask(self):
        """The burnt pixel **set** and the resulting mask are identical (the point order is shifted by one)."""
        d = ref("for_test")
        r = self._pipeline()
        mine = set(zip(r["xx"].tolist(), r["yy"].tolist()))
        theirs = set(zip(np.asarray(d["xx_out"]).ravel().astype(int).tolist(),
                         np.asarray(d["yy_out"]).ravel().astype(int).tolist()))
        assert mine == theirs
        assert int((r["bw_out"] != np.asarray(d["bw_burn"]).astype(bool)).sum()) == 0

    def test_circulant_solver_on_the_full_evolution(self):
        """The FFT fast path over 50 snake iterations: same curve, identical burnt mask."""
        d = ref("for_test")
        r = self._pipeline(solver="circulant")
        mlx = np.asarray(np.asarray(d["XB"]).ravel()[-1]).ravel()
        mly = np.asarray(np.asarray(d["YB"]).ravel()[-1]).ravel()
        x, y = r["blocks"][-1]
        dist = np.hypot(x[:, None] - mlx[None, :], y[:, None] - mly[None, :])
        assert max(dist.min(axis=1).max(), dist.min(axis=0).max()) < 1e-3
        assert int((r["bw_out"] != np.asarray(d["bw_burn"]).astype(bool)).sum()) == 0


class TestL2GVFDistance:
    """``GVF_distance.m`` on ``alg_seg_gray.jpg`` with the ``sea_ice_demo.m`` parameters (Fig. 6.15)."""

    _RESULT: dict = {}

    @classmethod
    def result(cls, solver="dense"):
        if solver not in cls._RESULT:
            ref("gvf_alg")                      # skip early when the reference is missing
            rgb = book_image("alg_seg_gray.jpg")
            cls._RESULT[solver] = ch6.gvf_distance(rgb, solver=solver, keep_history=False)
        return cls._RESULT[solver]

    def test_stages_before_the_snakes(self):
        d = ref("gvf_alg")
        r = self.result()
        p0 = r.passes[0]
        assert maxdiff(r.gray.astype(float), np.asarray(d["gray"]).astype(float)) == 0.0
        assert abs(r.level - float(np.asarray(d["level"]).ravel()[0])) < 1e-12
        assert int((r.bw != np.asarray(d["bw"]).astype(bool)).sum()) == 0
        assert maxdiff(r.f2, d["f2"]) == 0.0
        assert maxdiff(r.u, d["u"]) == 0.0 and maxdiff(r.v, d["v"]) == 0.0
        assert maxdiff(r.px, d["px"]) == 0.0 and maxdiff(r.py, d["py"]) == 0.0
        assert int((p0.label != np.asarray(d["label"])).sum()) == 0
        assert p0.num == int(np.asarray(d["num"]).ravel()[0])

    def test_criteria_and_selection(self):
        d = ref("gvf_alg")
        p0 = self.result().passes[0]
        assert maxdiff(p0.area, np.asarray(d["a"]).ravel()) == 0.0
        assert maxdiff(p0.solidity, np.asarray(d["rc"]).ravel()) < 1e-12
        assert maxdiff(p0.major, np.asarray(d["l"]).ravel()) < 1e-9
        assert maxdiff(p0.minor, np.asarray(d["w"]).ravel()) < 1e-9
        assert maxdiff(p0.rl, np.asarray(d["rl"]).ravel()) < 1e-9
        assert (p0.k + 1).tolist() == np.asarray(d["k"]).ravel().astype(int).tolist()
        assert int((p0.bw2 != np.asarray(d["bw2"]).astype(bool)).sum()) == 0

    def test_seed_initialisation(self):
        d = ref("gvf_alg")
        init = self.result().passes[0].init
        assert maxdiff(init.img_dist, d["img_Dist"]) == 0.0
        assert int((init.minima_map != np.asarray(d["Dis_img"]).astype(bool)).sum()) == 0
        assert int((init.dis_dilated != np.asarray(d["dis"]).astype(float)).sum()) == 0
        assert int((init.label != np.asarray(d["label1"])).sum()) == 0
        assert init.num == int(np.asarray(d["num1"]).ravel()[0])
        assert maxdiff(init.centroids, np.atleast_2d(d["cen"])) < 1e-12
        assert maxdiff(init.radii, np.asarray(d["r"]).ravel()) < 1e-6    # single-precision r (see dist.m)

    def test_per_seed_interpolated_contours_are_exact(self):
        """``snakeinterp`` of every initial circle: identical point count and values for all 46 seeds."""
        d = ref("gvf_alg")
        r = self.result()
        xi = np.asarray(d["XI"]).ravel()
        yi = np.asarray(d["YI"]).ravel()
        seeds = r.passes[0].seeds
        assert len(seeds) == xi.size
        for i, s in enumerate(seeds):
            assert s.x_interp.size == np.asarray(xi[i]).ravel().size, i
            assert maxdiff(s.x_interp, np.asarray(xi[i]).ravel()) < 1e-5, i   # single-precision radius
            assert maxdiff(s.y_interp, np.asarray(yi[i]).ravel()) < 1e-5, i

    def test_per_seed_clip_vertex_sets(self):
        """All 46 clips: same vertex multiset, same point count, both rings closed (open item 2 contract)."""
        d = ref("gvf_alg")
        r = self.result()
        xc = np.asarray(d["XC"]).ravel()
        yc = np.asarray(d["YC"]).ravel()
        n = 0
        for i, s in enumerate(r.passes[0].seeds):
            mx = np.asarray(xc[i]).ravel()
            my = np.asarray(yc[i]).ravel()
            assert mx.size == s.x_clip.size, (i, s.x_clip.size, mx.size)
            assert mx[0] == mx[-1] and my[0] == my[-1], i               # polybool closes the ring
            assert s.x_clip[0] == s.x_clip[-1] and s.y_clip[0] == s.y_clip[-1], i   # ...and so does the port
            # Compare the vertex multisets of the **open** rings: each side closes its ring on its *own*
            # first vertex, and `polybool` rotates the start and reverses the traversal, so the two closed
            # rings duplicate different vertices.  The coordinates carry the single-precision radius, hence
            # the 1e-4 tolerance.
            px_, py_ = s.x_clip[:-1], s.y_clip[:-1]
            mx_, my_ = mx[:-1], my[:-1]
            order_py = np.lexsort((py_, px_))
            order_ml = np.lexsort((my_, mx_))
            assert maxdiff(px_[order_py], mx_[order_ml]) < 1e-4, i
            assert maxdiff(py_[order_py], my_[order_ml]) < 1e-4, i
            n += 1
        assert n == 46

    def test_per_seed_final_curves(self):
        """After 100 snake iterations the two contours are the same **curve** (Hausdorff), not the same list.

        The point count of a long snake is numerically unstable: ``snakeinterp`` inserts where ``d > dmax``
        and hundreds of spacings sit at exactly ``dmax``, so a 1e-13 perturbation of the starting contour
        already changes the final count (measured on seed 22: 2266 -> 2262 for 1e-13 noise, MATLAB 2256).
        """
        d = ref("gvf_alg")
        r = self.result()
        xf = np.asarray(d["XF"]).ravel()
        yf = np.asarray(d["YF"]).ravel()
        worst = 0.0
        n_same = 0
        for i, s in enumerate(r.passes[0].seeds):
            a = np.asarray(xf[i]).ravel()
            b = np.asarray(yf[i]).ravel()
            n_same += int(s.x_final.size == a.size)
            dist = np.hypot(s.x_final[:, None] - a[None, :], s.y_final[:, None] - b[None, :])
            worst = max(worst, max(dist.min(axis=1).max(), dist.min(axis=0).max()))
        assert worst < 1.0, worst          # measured 0.4794 px over the 46 seeds
        # closing the ring (open item 2) took the exact-point-count agreement from 2/46 to 13/46
        assert n_same >= 13, n_same

    def test_burnt_mask(self):
        """The segmentation result: at most 0.5 % of the ice pixels differ (measured 61 of 31 730 = 0.19 %)."""
        d = ref("gvf_alg")
        r = self.result()
        ml = np.asarray(d["bw1"]).astype(bool)
        n_diff = int((r.bw1 != ml).sum())
        n_ice = int(ml.sum())
        assert n_diff <= 0.005 * n_ice, (n_diff, n_ice)
        burnt_py = int(r.bw.sum() - r.bw1.sum())
        burnt_ml = int(np.asarray(d["bw"]).astype(bool).sum() - ml.sum())
        assert abs(burnt_py - burnt_ml) <= 5, (burnt_py, burnt_ml)

    def test_circulant_solver_difference(self):
        """The default fast path vs MATLAB: same order of difference as the dense solver."""
        d = ref("gvf_alg")
        r = self.result("circulant")
        ml = np.asarray(d["bw1"]).astype(bool)
        n_diff = int((r.bw1 != ml).sum())
        assert n_diff <= 0.005 * int(ml.sum()), n_diff


class TestL2KmeanGVF:
    """``seaice_kmean_GVF_forenhancement.m`` on ``alg_seg_gray.jpg`` (MATLAB run under ``rng(0)``)."""

    _R: dict = {}

    @classmethod
    def result(cls):
        if "r" not in cls._R:
            ref("kmean_alg")
            rgb = book_image("alg_seg_gray.jpg")
            cls._R["r"] = ch6.seaice_kmean_gvf(rgb, solver="dense", keep_history=False)
        return cls._R["r"]

    def test_pass1_equals_gvf_distance(self):
        d = ref("kmean_alg")
        r = self.result()
        assert int((r.bw != np.asarray(d["bw"]).astype(bool)).sum()) == 0
        assert maxdiff(r.f2, d["f2"]) == 0.0
        assert maxdiff(r.px, d["px"]) == 0.0 and maxdiff(r.py, d["py"]) == 0.0
        ml = np.asarray(d["bw1"]).astype(bool)
        # pass 1 is literally GVF_distance: the same 0.19 % snake-discretisation difference (TestL2GVFDistance)
        assert int((r.pass1.bw1 != ml).sum()) <= 0.005 * int(ml.sum())

    def test_kmeans_cluster_centres(self):
        """Statistics Toolbox ``kmeans`` is randomly seeded: compare **sorted centres**, never the labels.

        Measured on ``alg_seg_gray.jpg``: MATLAB under ``rng(0)`` and ``core.clustering.kmeans_lloyd`` with
        k-means++ / seed 0 reach the *same* optimum -- the centres agree to double precision.
        """
        d = ref("kmean_alg")
        r = self.result()
        mine = np.sort(np.asarray(r.s0, dtype=float))
        theirs = np.sort(np.asarray(d["s0"]).ravel())
        assert np.max(np.abs(mine - theirs)) < 1e-9, (mine, theirs)

    def test_kmeans_optimum_is_not_unique_in_matlab(self):
        """20 MATLAB restarts (different seeds): the optimum is **not** unique -- this is why k-means is `approx`.

        Per-centre spread over the restarts: about 2.20 / 2.00 / 0.09 gray levels.  The reference run
        (``rng(0)``) and the port land on the same optimum, and at least one restart reproduces it.
        """
        d = ref("kmean_alg")
        C = np.asarray(d["C"], dtype=float)
        assert C.shape[0] >= 10
        spread = C.max(axis=0) - C.min(axis=0)
        assert spread.max() > 1e-3
        ref_centres = np.sort(np.asarray(d["s0"]).ravel())
        assert np.min(np.abs(C - ref_centres[None, :]).max(axis=1)) < 1e-6
        mine = np.sort(np.asarray(self.result().s0, dtype=float))
        assert np.max(np.abs(mine - ref_centres)) < 1e-9

    def test_kmeans_mask_and_residual(self):
        """``bk``, the ``bw_kmeans - bw`` residual and ``bwareaopen(.., Ra_min, 4)`` are pixel-identical."""
        d = ref("kmean_alg")
        r = self.result()
        bk_ml = np.asarray(d["bk"]).astype(bool)
        assert int((r.bk.astype(bool) != bk_ml).sum()) == 0
        assert r.n_negative == int(np.asarray(d["n_negative"]).ravel()[0]) == 0   # Risk R7: no -1 pixels
        assert int((r.bw0 != np.asarray(d["bw0"]).astype(bool)).sum()) == 0

    def test_three_level_output(self):
        """``out = bw1 + 0.5*bw0``: same three levels; 0.18 % of the pixels move with the snake boundaries."""
        d = ref("kmean_alg")
        r = self.result()
        out_ml = np.asarray(d["out"], dtype=float)
        assert set(np.unique(r.out).tolist()) == set(np.unique(out_ml).tolist()) == {0.0, 0.5, 1.0}
        assert float(np.mean(np.abs(r.out - out_ml) > 1e-9)) <= 0.005
        assert int((r.pass2.bw1 != np.asarray(d["bw0_final"]).astype(bool)).sum()) <= 50


class TestL2KmeanStage:
    """The stages of ``sea_ice_demo.m`` on the full 1038x394 ``sea_ice_test.jpg`` (no snake loop)."""

    _S: dict = {}

    @classmethod
    def _stages(cls):
        ref("kmean_stage")
        if "s" not in cls._S:
            rgb = book_image("sea_ice_test.jpg")
            gray = rgb2gray_matlab(rgb)
            level, _ = graythresh(gray)
            bw = im2bw(gray, level)
            f2, u, v, px, py = ch6.gvf_force_field(gray, num=500)
            cls._S["s"] = (rgb, gray, level, bw, f2, u, v, px, py)
        return cls._S["s"]

    def test_field_and_criteria(self):
        d = ref("kmean_stage")
        rgb, gray, level, bw, f2, u, v, px, py = self._stages()
        assert maxdiff(gray.astype(float), np.asarray(d["I"]).astype(float)) == 0.0
        assert abs(level - float(np.asarray(d["level"]).ravel()[0])) < 1e-12
        assert int((bw != np.asarray(d["bw"]).astype(bool)).sum()) == 0
        assert maxdiff(f2, d["f2"]) == 0.0
        assert maxdiff(u, d["u"]) == 0.0 and maxdiff(v, d["v"]) == 0.0
        assert maxdiff(px, d["px"]) == 0.0 and maxdiff(py, d["py"]) == 0.0
        label, num, area, sol, major, minor, rl, k = ch6.component_criteria(bw, 2500, 0.9, 2, conn=4)
        assert num == int(np.asarray(d["num"]).ravel()[0])
        assert int((label != np.asarray(d["label"])).sum()) == 0
        assert maxdiff(area, np.asarray(d["a"]).ravel()) == 0.0
        assert maxdiff(sol, np.asarray(d["rc"]).ravel()) < 1e-12
        assert maxdiff(major, np.asarray(d["l"]).ravel()) < 1e-9
        assert maxdiff(minor, np.asarray(d["w"]).ravel()) < 1e-9
        assert (k + 1).tolist() == np.asarray(d["k"]).ravel().astype(int).tolist()

    def test_seed_initialisation_on_the_demo_image(self):
        d = ref("kmean_stage")
        rgb, gray, level, bw = self._stages()[:4]
        label, num, a_, s_, l_, w_, rl_, k = ch6.component_criteria(bw, 2500, 0.9, 2, conn=4)
        bw2 = np.zeros(bw.shape, bool)
        for m in k:
            bw2 |= (label == m + 1)
        bw2 = bwareaopen(bw2, 10)
        assert int((bw2 != np.asarray(d["bw2"]).astype(bool)).sum()) == 0
        init = ch6.initialize_contours(bw2, se_radius=3)
        assert maxdiff(init.img_dist, d["img_Dist"]) == 0.0
        assert int((init.minima_map != np.asarray(d["Dis_img"]).astype(bool)).sum()) == 0
        assert int((init.dis_dilated != np.asarray(d["dis"]).astype(float)).sum()) == 0
        assert init.num == int(np.asarray(d["num1"]).ravel()[0])
        assert maxdiff(init.centroids, np.atleast_2d(d["cenA"])) < 1e-12
        assert maxdiff(init.radii, np.asarray(d["rA"]).ravel()) < 1e-6   # single-precision r (see dist.m)

    def test_kmeans_stage_on_the_demo_image(self):
        d = ref("kmean_stage")
        rgb, gray, level, bw = self._stages()[:4]
        from seaice.core.clustering import kmeans_lloyd
        ima = gray.astype(np.float64).ravel(order="F")
        map0 = kmeans_lloyd(ima[:, None], 3, init="kmeans++", seed=0, max_iter=100).labels
        s0 = np.array([ima[map0 == i].mean() for i in range(3)])
        assert np.max(np.abs(np.sort(s0) - np.sort(np.asarray(d["s0"]).ravel()))) < 1.0
        ind0 = np.argsort(s0, kind="stable")
        bk = (map0 != ind0[0]).astype(float).reshape(gray.shape, order="F")
        bk_ml = np.asarray(d["bk"]).astype(bool)
        assert int((bk.astype(bool) != bk_ml).sum()) <= 0.001 * bk_ml.size
        bw0_raw = bk - bw.astype(float)
        assert int((bw0_raw < 0).sum()) == int(np.asarray(d["n_negative"]).ravel()[0])


# =====================================================================================================================
# L4 -- numbers quoted in the chapter text
# =====================================================================================================================
class TestL4BookNumbers:
    def test_footnote_parameters(self):
        """Footnote 2 p. 121 (alpha = 0.05, beta = 0), footnote 3 p. 125 (mu = 0.1), the drivers' blocks."""
        S = ch6.BOOK_PARAMS["sea_ice_demo"]
        assert S["alpha"] == 0.05 and S["beta"] == 0.0 and S["mu"] == 0.1
        assert S["kms0"] == 3 and S["Ra_min"] == 10 and S["Ra"] == 2500
        assert S["Rc"] == 0.9 and S["Rl"] == 2 and S["Num"] == 500 and S["iter"] == 100
        assert S["se_radius"] == 3 and S["gamma"] == 1.0 and S["kappa"] == 0.5
        assert S["Dmin"] == 0.0 and S["Dmax"] == 1.0 and S["timer"] == 1 and S["sigma"] == 0
        F = ch6.BOOK_PARAMS["for_test"]
        assert F["Num"] == 150 and F["iter"] == 50 and F["r"] == 20 and (F["x0"], F["y0"]) == (80, 40)
        assert ch6.BOOK_PARAMS["dist"] == {"se_radius": 5, "r": 15}

    def test_cfl_number_of_the_book_settings(self):
        """Eqs. (6.54)/(6.55): ``GVF.m`` runs at dt = dx = dy = 1, so r = mu = 0.1 <= 1/4."""
        assert ch6.BOOK_PARAMS["sea_ice_demo"]["mu"] * 1.0 / (1.0 * 1.0) <= 0.25

    def test_strel_disk_sizes(self):
        """The T_seed / merging structuring elements, against ``getnhood(strel('disk', r))`` in MATLAB.

        NOTE: ``analysis/ch06.md`` (§2.2 stage 15 and §6.4) states ``strel('disk', 3)`` is "7x7 / 37 px".
        MATLAB R2025a returns **5x5 / 25 px** (``reference/ch06/strel.mat``); the port matches MATLAB.
        ``strel('disk', 5)`` really is 9x9 / 69 px.
        """
        from seaice.core.morphology import strel
        d = ref("strel")
        se3 = strel("disk", 3)
        se5 = strel("disk", 5)
        assert se3.shape == (5, 5) and int(se3.sum()) == 25
        assert se5.shape == (9, 9) and int(se5.sum()) == 69
        assert np.array_equal(se3, np.asarray(d["se3"]).astype(bool))
        assert np.array_equal(se5, np.asarray(d["se5"]).astype(bool))

    def test_fig_6_16_capture_range_claims(self):
        """p. 139: 5 iterations suffice for the 9-px circle; 30 are not enough for the 61-px one."""
        bw = synth.fig_6_16_circles()
        f = SN.gradient2_magnitude(bw.astype(float))
        L = label_components(bw, 8)
        sizes = np.array([int((L == k).sum()) for k in (1, 2)])
        big, small = (1, 2) if sizes[0] > sizes[1] else (2, 1)

        def filled(u, v, mask, thresh=0.02):
            return float((np.hypot(u, v)[mask] > thresh).mean())

        u5, v5 = SN.gvf(f, 0.1, 5)
        u30, v30 = SN.gvf(f, 0.1, 30)
        u250, v250 = SN.gvf(f, 0.1, 250)
        assert filled(u5, v5, L == small) > 0.9
        assert filled(u30, v30, L == big) < 0.6
        assert filled(u250, v250, L == big) > filled(u30, v30, L == big)

    def test_fig_6_14_text_claims(self):
        """p. 134: one regional maximum, three local maxima of value 3; radius 3/sqrt(2)."""
        init = ch6.initialize_contours(synth.FIG_6_14_IMAGE, se_radius=3)
        maxima = init.minima_map & synth.FIG_6_14_IMAGE
        assert int(maxima.sum()) == 3 and int(label_components(maxima, 8).max()) == 1
        assert init.num == 1 and abs(init.radii[0] - 3 / np.sqrt(2)) < 1e-12

    def test_criterion_three_uses_the_ellipse_not_minboundrect(self):
        """ch9 p. 205 criterion 3 is the min-area **rectangle** ratio; the shipped code uses the ellipse axes."""
        bw = np.zeros((40, 60), bool)
        bw[10:30, 10:50] = True
        s = regionprops(bw, ("MajorAxisLength", "MinorAxisLength"))[0]
        ell = s.MajorAxisLength / s.MinorAxisLength
        r, c = np.nonzero(bw)
        _, _, area, _ = POLY.minboundrect(c + 1.0, r + 1.0, "a")
        assert abs(ell - 2.0) < 0.05
        assert abs(area - 19.0 * 39.0) < 1e-6

    def test_t_seed_is_the_disk_radius_three(self):
        """Risk R12: the book never gives T_seed; the code's only evidence is strel('disk', 3)."""
        assert ch6.BOOK_PARAMS["sea_ice_demo"]["se_radius"] == 3


# =====================================================================================================================
# scripts run headless
# =====================================================================================================================
SCRIPTS = ["ch06_contour_init", "ch06_snake_parameters", "ch06_ushape", "ch06_capture_range",
           "ch06_dist", "ch06_for_test", "ch06_gvf_distance", "ch06_sea_ice_demo"]


def _run_script(script: str, out: Path, *flags: str) -> subprocess.CompletedProcess:
    cmd = [str(PY), str(ROOT / "scripts" / f"{script}.py"), "--no-show", "--out", str(out), *flags]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs(script):
    out = VERIFY / "pytest_scripts" / script
    out.mkdir(parents=True, exist_ok=True)
    flags = []
    if script in ("ch06_gvf_distance", "ch06_sea_ice_demo"):
        flags = ["--max-seeds", "3"]
    proc = _run_script(script, out, *flags)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    needs_book = script in ("ch06_dist", "ch06_for_test", "ch06_gvf_distance", "ch06_sea_ice_demo")
    if needs_book and not DATA.exists():
        assert "SKIP" in proc.stdout
    else:
        assert list(out.glob("*.png")), "no figure written"


CLI_CASES = [
    ("ch06_dist", ["--no-transpose", "--se-radius", "3"]),
    ("ch06_for_test", ["--num", "30", "--iter", "10"]),
    ("ch06_gvf_distance", ["--image", "test8.jpg", "--num", "50", "--iter", "20", "--max-seeds", "2",
                           "--solver", "dense"]),
    ("ch06_capture_range", ["--iters", "5", "30", "--snake-iters", "30", "--snake-blocks", "5",
                            "--edge-map", "binary"]),
    ("ch06_contour_init", ["--se-radius", "1"]),
    ("ch06_ushape", ["--gvf-iters", "40", "--blocks", "5", "--sigma-sweep", "2"]),
    ("ch06_snake_parameters", ["--alphas", "0.05", "--betas", "0.0", "--iters", "20", "--blocks", "3",
                               "--gvf-iters", "30"]),
    ("ch06_sea_ice_demo", ["--image", "alg_seg_gray.jpg", "--num", "50", "--iter", "20", "--max-seeds", "2"]),
]


@pytest.mark.parametrize("script,flags", CLI_CASES, ids=[f"{s}:{' '.join(f)}" for s, f in CLI_CASES])
def test_script_cli_flags(script, flags):
    safe = "_".join(f.strip("-").replace(".", "p") for f in flags)[:60]
    out = VERIFY / "pytest_scripts" / f"{script}_{safe}"
    out.mkdir(parents=True, exist_ok=True)
    proc = _run_script(script, out, *flags)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
