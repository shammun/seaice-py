"""Chapter 5 verification — Watershed-Based Ice Floe Segmentation (Book pp. 83–108).

Evidence levels (verify-port skill):
  L1  synthetic truth (no external tools)            -> ``TestL1*``
  L2  parity vs MATLAB R2025a references              -> ``TestL2*`` (skip if ``reference/ch05/*.mat`` missing;
      regenerate with ``.venv/Scripts/python.exe reference/ch05/make_refs.py``)
  L4  numbers quoted in the book text                 -> ``TestL4BookNumbers``
  scripts: every ``scripts/ch05_*.py`` must run headless and exit 0 -> ``test_script_runs`` / ``test_script_cli_flags``

The MATLAB references were produced by the ORIGINAL ``.m`` files (verbatim copies with only literals patched, see
``reference/ch05/make_refs.py``) and by the toolbox functions (``watershed``, ``imregionalmin``, ``imregionalmax``,
``imimposemin``, ``imreconstruct``, ``imfilter``, ``regionprops``) on the controlled fixtures of
``reference/ch05/inputs.mat`` (plateaus, corridors, ties, random fields, > 255 minima, ±Inf, every dtype).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from seaice import ch05_watershed as ch5
from seaice.core import synth
from seaice.core.chaincode import boundaries, fchcode
from seaice.core.connectivity import label_components
from seaice.core.distance import bwdist
from seaice.core.filters import imfilter
from seaice.core.io import load_image
from seaice.core.morphology import imimposemin, imreconstruct, imregionalmax, imregionalmin
from seaice.core.plotting import contour_overlay, label2rgb, surface_plot
from seaice.core.threshold import graythresh
from seaice.core.watershed import neighbour_offsets, watershed, watershed_skimage
from tools.compare_arrays import assert_parity, label_agreement

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "ch05"
DATA = ROOT / "data" / "book" / "ch05"
DATA4 = ROOT / "data" / "book" / "ch04"
VERIFY = ROOT / "outputs" / "ch05" / "verify"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

needs_image = pytest.mark.skipif(not (DATA / "q.jpg").exists(), reason="data/book/ch05/q.jpg missing (private book image)")


def needs_ref(name: str):
    return pytest.mark.skipif(not (REF / f"{name}.mat").exists(),
                              reason=f"reference/ch05/{name}.mat missing — run reference/ch05/make_refs.py (MATLAB)")


_REF_CACHE: dict[str, dict] = {}


def ref(name: str) -> dict:
    if name not in _REF_CACHE:
        d = loadmat(str(REF / f"{name}.mat"))
        _REF_CACHE[name] = {k: v for k, v in d.items() if not k.startswith("__")}
    return _REF_CACHE[name]


def inputs() -> dict:
    return ref("inputs")


def sc(x) -> float:
    return float(np.asarray(x).ravel()[0])


def ref_str(d: dict, key: str) -> str:
    v = np.asarray(d[key]).ravel()
    return "" if v.size == 0 else str(v[0])


def cell(d: dict, key: str) -> list[np.ndarray]:
    """A MATLAB cell row ``{…}`` as a list of arrays."""
    c = d[key]
    if c.size == 0:
        return []
    return [np.asarray(x) for x in c.ravel()]


def rows2(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.reshape(-1, 2) if a.size else np.zeros((0, 2), dtype=np.int64)


def q_image() -> np.ndarray:
    rgb, _ = load_image("ch05", "q.jpg", allow_fallback=False, verbose=False)
    return rgb


def same_labels(py: np.ndarray, ml: np.ndarray, name: str = "") -> dict:
    """Label images equal *including label values* (MATLAB uint8/uint16 cast to int32)."""
    ml = np.asarray(ml)
    py = np.asarray(py)
    assert py.shape == ml.shape, f"{name}: shape {py.shape} vs {ml.shape}"
    n = int((py.astype(np.int64) != ml.astype(np.int64)).sum())
    assert n == 0, f"{name}: {n} label pixels differ (max label py {py.max()} / ml {ml.max()})"
    return {"n_diff": n}


# fixture name lists shared with make_refs (kept in sync by loading inputs.mat keys)
def _names(prefix: str) -> list[str]:
    if not (REF / "inputs.mat").exists():
        return []
    return [k for k in inputs() if k.startswith(prefix)]


WS_NAMES = [k for k in _names("pf_") + _names("ws_")]
RM_NAMES = _names("rm_")
# the (image × marker) grid MATLAB ran in make_refs (im_u8_plateau / im_bw_in_plateau are a separate probe)
IM_IMAGES = [k for k in ("im_u8", "im_u8_sat", "im_i16", "im_i16_max", "im_single", "im_single_neg", "im_double",
                         "im_const_single", "im_const_u8", "im_double_pinf", "im_double_ninf") if k in _names("im_")]
IM_MARKERS = [k for k in ("im_bw", "im_bw_all", "im_bw_none", "im_bw_adjacent") if k in _names("im_bw")]
METRICS = ("chessboard", "cityblock", "euclidean", "quasi-euclidean")
DW_FILES = {"chessboard": "distance_watershed", "cityblock": "distance_watershed_cityblock",
            "euclidean": "distance_watershed_euclidean", "quasi-euclidean": "distance_watershed_quasi"}


# =====================================================================================================================
# L1 — synthetic truth
# =====================================================================================================================
class TestL1Watershed:
    def test_neighbour_offsets_are_ascending_linear_indices(self):
        assert neighbour_offsets(8, 10) == [-11, -10, -9, -1, 1, 9, 10, 11]
        assert neighbour_offsets(4, 10) == [-10, -1, 1, 10]
        with pytest.raises(ValueError):
            neighbour_offsets(6, 10)

    def test_1d_profile_two_basins(self):
        prof = np.array([[3, 2, 1, 2, 3, 4, 3, 2, 1, 2, 3]], dtype=np.uint8)
        L = watershed(prof)
        assert L.dtype == np.int32
        assert L[0, 2] == 1 and L[0, 8] == 2  # basins numbered in bwlabel (find) order
        assert L[0, 5] == 0  # the ridge at the maximum
        assert (L == 0).sum() == 1
        assert set(np.unique(L)) == {0, 1, 2}
        Lc = watershed(prof.T)
        assert np.array_equal(Lc, L.T)

    def test_two_touching_discs_split_by_distance_watershed(self):
        rr, cc = np.mgrid[0:40, 0:60]
        two = ((rr - 20) ** 2 + (cc - 18) ** 2 <= 12 ** 2) | ((rr - 20) ** 2 + (cc - 40) ** 2 <= 12 ** 2)
        assert label_components(two).max() == 1  # one connected object
        D = -bwdist(~two, "cityblock")
        L = watershed(D)
        seg = two & (L != 0)
        assert label_components(seg).max() == 2
        assert L.max() == 2
        assert np.all(seg[:, :15][two[:, :15]]) and np.all(seg[:, 45:][two[:, 45:]])

    def test_constant_image_is_one_basin(self):
        assert np.array_equal(watershed(np.full((4, 6), 7, dtype=np.uint8)), np.ones((4, 6), dtype=np.int32))

    def test_bool_input_and_single_pixel(self):
        bw = np.zeros((5, 5), dtype=bool)
        bw[2, 2] = True
        L = watershed(bw)  # 0 plateau (8-connected ring) is one minimum, the centre a maximum
        assert L.max() == 1 and L[2, 2] == 1  # no second minimum → no ridge: the centre is flooded too
        assert watershed(np.array([[5]], dtype=np.uint8)).tolist() == [[1]]

    def test_rejections(self):
        with pytest.raises(ValueError):
            watershed(np.array([[1.0, np.nan], [2.0, 3.0]]))
        with pytest.raises(ValueError):
            watershed(np.zeros((3, 3, 3)))
        with pytest.raises(ValueError):
            watershed(np.zeros((3, 3)), conn=6)
        assert watershed(np.zeros((0, 5))).shape == (0, 5)

    def test_ridges_are_4_connected_paths_between_basins(self):
        fx = synth.plateau_fixtures()
        L = watershed(fx["four_minima_plateau"])
        assert L.max() == 5  # four corner minima + the central level-2 plateau (itself a regional minimum)
        ridge = L == 0
        # every ridge pixel touches at least two different basins within its 8-neighbourhood
        M, N = L.shape
        for r, c in zip(*np.nonzero(ridge)):
            nb = L[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            assert len(set(np.unique(nb)) - {0}) >= 2 or ridge.sum() == 1

    def test_neg_inf_minima_are_basins(self):
        fx = synth.plateau_fixtures()["neg_inf_minima"]
        L = watershed(fx)
        assert L[1, 1] == 1 and L[4, 6] == 2 and L.max() == 2

    def test_immersion_agrees_with_meyer_on_1d_profiles(self):
        prof = np.array([[5, 3, 1, 2, 4, 6, 5, 3, 2, 3, 7, 4, 1, 2, 5]], dtype=np.float64)
        Li = ch5.watershed_immersion(prof)
        Lm = watershed(prof)
        # same partition and same dams; the immersion numbers basins in level order, Meyer in bwlabel order
        assert label_agreement(Li, Lm) == 1.0 and np.array_equal(Li == 0, Lm == 0)
        assert Lm.max() == 3 == Li.max()
        # a plateau maximum between two basins: both put the dam on the plateau, basins agree
        prof2 = np.array([[3, 1, 2, 4, 4, 4, 2, 1, 3]], dtype=np.float64)
        assert label_agreement(ch5.watershed_immersion(prof2), watershed(prof2)) >= 0.75
        assert ch5.watershed_immersion(prof2).max() == 2 == watershed(prof2).max()

    def test_threshold_set(self):
        I = np.array([[1, 2], [3, 4]], dtype=np.float64)
        assert ch5.threshold_set(I, 3).tolist() == [[True, True], [False, False]]

    def test_skimage_cross_check_is_approx_only(self):
        rr, cc = np.mgrid[0:40, 0:60]
        two = ((rr - 20) ** 2 + (cc - 18) ** 2 <= 12 ** 2) | ((rr - 20) ** 2 + (cc - 40) ** 2 <= 12 ** 2)
        D = -bwdist(~two, "cityblock")
        Lm, Ls = watershed(D), watershed_skimage(D)
        assert Ls.dtype == np.int32 and Ls.max() == Lm.max()
        assert label_agreement(Ls, Lm) > 0.9  # same basins, ridge pixels may differ (documented `approx`)


class TestL1RegionalExtrema:
    def test_constant_image_all_true(self):
        assert imregionalmin(np.full((3, 4), 2.0)).all()
        assert imregionalmax(np.full((3, 4), 2, dtype=np.uint8)).all()

    def test_plateau_touching_border_and_strict_neighbours(self):
        I = np.full((6, 8), 5, dtype=np.uint8)
        I[0:2, 0:3] = 2  # border plateau minimum
        I[4, 6] = 1
        I[4, 4] = 5
        m = imregionalmin(I)
        assert m[0:2, 0:3].all() and m[4, 6] and m.sum() == 7
        assert imregionalmax(I).sum() == 6 * 8 - 7  # the level-5 plateau is the maximum

    def test_inf_values(self):
        I = np.array([[1.0, 2.0, np.inf], [-np.inf, 3.0, 2.0]])
        m = imregionalmin(I)
        assert m[1, 0] and not m[0, 0] and m.sum() == 1
        x = imregionalmax(I)
        assert x[0, 2] and x.sum() == 1

    def test_conn_4_vs_8(self):
        I = np.array([[0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=np.uint8)
        assert imregionalmin(I, 8).sum() == 2 and imregionalmin(I, 4).sum() == 2
        J = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        assert imregionalmin(J, 4).sum() == 5  # five isolated zeros with 4-connectivity
        assert imregionalmin(J, 8).sum() == 5  # the zeros form one 8-connected plateau: still a minimum
        assert np.array_equal(imregionalmin(J, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])), imregionalmin(J, 4))

    def test_rejections(self):
        with pytest.raises(ValueError):
            imregionalmin(np.array([[1.0, np.nan]]))
        with pytest.raises(ValueError):
            imregionalmin(np.zeros((2, 2)), 6)

    @pytest.mark.parametrize("name", list(synth.plateau_fixtures()))
    def test_eq_5_2_identity_on_plateau_fixtures(self, name):
        I = synth.plateau_fixtures()[name]
        for conn in (4, 8):
            assert np.array_equal(ch5.regional_minima_by_reconstruction(I, conn), imregionalmin(I, conn)), (name, conn)

    def test_eq_5_2_identity_random_and_inf(self):
        rng = np.random.default_rng(1)
        I = rng.integers(0, 5, size=(30, 40)).astype(np.float64)
        assert np.array_equal(ch5.regional_minima_by_reconstruction(I), imregionalmin(I))
        I[3, 3] = -np.inf
        I[10, 10] = np.inf
        assert np.array_equal(ch5.regional_minima_by_reconstruction(I), imregionalmin(I))
        assert np.array_equal(ch5.regional_minima_by_reconstruction(-I), imregionalmax(I))


class TestL1Imposemin:
    @pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32, np.float64])
    def test_markers_become_the_only_minima_and_dtype_is_kept(self, dtype):
        rng = np.random.default_rng(3)
        I = rng.integers(0, 50, size=(12, 15)).astype(dtype)
        BW = np.zeros((12, 15), dtype=bool)
        BW[2:4, 2:4] = True
        BW[9, 11] = True
        J = imimposemin(I, BW)
        assert J.dtype == I.dtype
        assert np.array_equal(imregionalmin(J), BW)
        if np.issubdtype(dtype, np.floating):
            assert np.all(J[BW] == -np.inf) and np.all(np.isfinite(J[~BW]))
        else:
            assert np.all(J[BW] == np.iinfo(dtype).min)

    def test_h_rule(self):
        I = np.full((5, 5), -3.0, dtype=np.float32)
        BW = np.zeros((5, 5), dtype=bool)
        BW[2, 2] = True
        J = imimposemin(I, BW)  # constant float → h = 0.1
        assert J.dtype == np.float32 and np.isclose(J[0, 0], -2.9) and J[2, 2] == -np.inf
        I8 = np.full((5, 5), 255, dtype=np.uint8)
        J8 = imimposemin(I8, BW)  # I + 1 saturates at 255
        assert J8[0, 0] == 255 and J8[2, 2] == 0

    def test_rejections(self):
        with pytest.raises(TypeError):
            imimposemin(np.zeros((3, 3), dtype=bool), np.zeros((3, 3), dtype=bool))
        with pytest.raises(ValueError):
            imimposemin(np.zeros((3, 3)), np.zeros((2, 2), dtype=bool))
        with pytest.raises(ValueError):
            imimposemin(np.array([[np.nan, 1.0]]), np.array([[True, False]]))

    def test_book_form_gives_the_same_watershed(self):
        rr, cc = np.mgrid[0:40, 0:60]
        two = ((rr - 20) ** 2 + (cc - 18) ** 2 <= 12 ** 2) | ((rr - 20) ** 2 + (cc - 40) ** 2 <= 12 ** 2)
        g = -bwdist(~two, "cityblock").astype(np.float32)
        mk = np.zeros_like(two)
        mk[18:23, 15:21] = True
        mk[18:23, 38:43] = True
        gm = imimposemin(g, mk)
        gb = ch5.impose_minima_book(g, mk)
        assert np.array_equal(imregionalmin(gb), mk) and np.array_equal(imregionalmin(gm), mk)
        assert np.array_equal(watershed(gm), watershed(gb))
        assert watershed(gb).max() == 2


class TestL1ChainCode:
    @pytest.mark.parametrize("k", range(12))
    def test_fig_5_15_patterns_respond_3(self, k):
        P = np.zeros((5, 5))
        P[1:4, 1:4] = synth.FIG_5_15_ENDPOINT_PATTERNS[k]
        r = imfilter(P, ch5.ENDPOINT_KERNEL)
        assert r[2, 2] == 3
        for rule in ("max", "ge3"):
            ep = ch5.junction_endpoints(P > 0, rule)
            assert [2, 2] in ep.tolist()
        # each pattern: the centre has exactly one 4-neighbour (patterns 5-12 add one diagonal pixel)
        pat = synth.FIG_5_15_ENDPOINT_PATTERNS[k]
        assert pat[1, 1] == 1 and pat.sum() in (2, 3) and pat[0, 1] + pat[1, 0] + pat[1, 2] + pat[2, 1] == 1

    def test_isolated_pixel_responds_4_and_loop_rule_difference(self):
        P = np.zeros((5, 5))
        P[2, 2] = 1
        assert imfilter(P, ch5.ENDPOINT_KERNEL)[2, 2] == 4
        assert ch5.junction_endpoints(P > 0, "max").tolist() == [[2, 2]]
        assert ch5.junction_endpoints(P > 0, "ge3").tolist() == [[2, 2]]
        loop = np.zeros((6, 6), dtype=bool)
        loop[1:5, 1] = loop[1:5, 4] = loop[1, 1:5] = loop[4, 1:5] = True
        # max = 2 → every line pixel qualifies, and (abs) the four enclosed background pixels with two line
        # 4-neighbours (response -2) as well — the literal rule of main.m
        assert ch5.junction_endpoints(loop, "max").shape[0] == loop.sum() + 4 == 16
        assert ch5.junction_endpoints(loop, "ge3").shape[0] == 0
        with pytest.raises(ValueError):
            ch5.junction_endpoints(loop, "other")

    def test_fig_5_16_trace_is_a_contiguous_subsequence(self):
        b = boundaries(synth.FIG_5_16_IMAGE, 8, "cw")[0] + 1  # 1-based
        seq = [tuple(int(v) for v in r) for r in b]
        want = list(synth.FIG_5_16_TRACE_MATLAB)
        assert seq[0] == (3, 2)  # DIPUM starts at the first object pixel in column-major order
        assert any(seq[i:i + 3] == want for i in range(len(seq) - 2)), seq

    def _codes(self, mask):
        b = boundaries(mask, 8, "cw")[0]
        return b, fchcode(b).fcc

    def test_eq_5_11_and_5_15_identities(self):
        sq = np.zeros((12, 12), dtype=bool)
        sq[3:9, 3:9] = True
        b, fcc = self._codes(sq)
        R, A, S, D, Diff = ch5.differential_chain_code(fcc)
        assert A[-1] == -8  # Eq. (5.11): A(0) - A(N) = 8 for a clockwise trace
        assert S[-1] == -24  # Eq. (5.15): S(0) - S(N) = 24 with S(0) normalised to 0
        assert Diff[0] == D[-1]  # D0 = Sum(3) - Sum0 equals D(l) = Sum(3) - 24 - Sum(l) since Sum(l) = -24
        assert len(R) == len(fcc)

    def test_square_has_no_concave_points_L_has_inner_corner(self):
        sq = np.zeros((12, 12), dtype=bool)
        sq[3:9, 3:9] = True
        r = ch5.freeman_concave(sq)
        assert r.points.shape[0] == 0
        assert (r.Diff == -6).sum() == 4 and r.Diff.max() < 0  # four convex 90° corners (-6 × 15°), all clockwise
        L = np.zeros((14, 14), dtype=bool)
        L[2:12, 2:6] = True
        L[8:12, 2:12] = True
        r = ch5.freeman_concave(L)
        pts = {tuple(p) for p in r.points.tolist()}
        # the 3-code window of Eq. (5.12) spreads the +90° turn over the pixels around the inner corner (8, 6)
        assert len(pts) >= 1 and all(abs(p[0] - 7.5) <= 1.5 and abs(p[1] - 5.5) <= 1.5 for p in pts), pts

    def test_circle_has_small_differential_code(self):
        rr, cc = np.mgrid[0:41, 0:41]
        disc = (rr - 20) ** 2 + (cc - 20) ** 2 <= 15 ** 2
        r = ch5.freeman_concave(disc)
        assert r.points.shape[0] == 0 and r.Diff.max() < 3

    def test_rejections_and_object_choice(self):
        with pytest.raises(ValueError):
            ch5.differential_chain_code(np.array([1, 2]))
        with pytest.raises(ValueError):
            ch5.freeman_concave(np.zeros((5, 5), dtype=bool))
        m = np.zeros((10, 12), dtype=bool)
        m[1, 1] = True  # 1-px object: boundary of 2 identical points, 1 code → MATLAB errors on A(l-2)
        with pytest.raises(ValueError):
            ch5.freeman_concave(m)
        m2 = np.zeros((14, 20), dtype=bool)
        m2[1:4, 1:4] = True
        m2[6:13, 8:19] = True
        assert ch5.freeman_concave(m2, "first").boundary.shape[0] < ch5.freeman_concave(m2, "longest").boundary.shape[0]
        with pytest.raises(ValueError):
            ch5.freeman_concave(m2, "biggest")


class TestL1Merging:
    def test_two_touching_floes_keep_the_junction(self):
        rgb = synth.two_touching_floes()
        assert rgb.shape == (96, 81, 3) and rgb.dtype == np.uint8
        assert np.array_equal(rgb, synth.two_touching_floes())  # deterministic
        bw = ch5.otsu_mask(rgb)
        assert label_components(bw).max() == 1
        res = ch5.neighboring_region_merging(bw)
        assert res.num >= 1 and res.n_floes_before >= 2
        kept = [ln for ln in res.lines if not ln.removed]
        assert len(kept) >= 1, "the junction line with concave notch endpoints must survive"
        assert res.n_floes_after == 2
        assert all(ln.concave_endpoints.shape[0] >= 1 for ln in kept)
        assert res.f.sum() == sum(ln.n_pixels for ln in res.lines)
        assert np.array_equal(res.seg0 | res.f, res.bw)

    def test_convex_blob_spurious_lines_are_removed(self):
        rr, cc = np.mgrid[0:70, 0:90]
        ell = ((rr - 35) / 30.0) ** 2 + ((cc - 45) / 40.0) ** 2 <= 1.0
        res = ch5.neighboring_region_merging(ell, "euclidean")
        assert res.n_floes_after == 1
        assert all(ln.removed for ln in res.lines)

    def test_non_sequential_and_rules(self):
        bw = ch5.otsu_mask(synth.two_touching_floes())
        a = ch5.neighboring_region_merging(bw, sequential=False)
        b = ch5.neighboring_region_merging(bw, endpoint_rule="ge3")
        assert a.n_floes_after == b.n_floes_after == 2
        with pytest.raises(ValueError):
            ch5.junction_endpoints(bw, "x")

    def test_component_centroids(self):
        m = np.zeros((10, 10), dtype=bool)
        m[1:4, 1:4] = True  # centre row 2, col 2 (0-based) → MATLAB (x, y) = (3, 3)
        m[7, 8] = True
        c = ch5.component_centroids(m)
        assert c.tolist() == [[3.0, 3.0], [9.0, 8.0]]


class TestL1Pipelines:
    def test_point_image_inverse_distance(self):
        img = synth.point_image(201)
        r = ch5.inverse_distance_map(~img, "cityblock")
        assert r.imgDist.dtype == np.float32 and r.imgDist[100, 100] == 0 and r.imgDist[0, 0] == -200
        assert r.dist.dtype == np.uint8 and r.dist[100, 100] == 255 and r.dist[0, 0] == 0
        assert r.dist0.min() == 0
        with pytest.raises(ValueError):
            ch5.inverse_distance_map(~img, "manhattan")

    def test_synthetic_end_to_end(self):
        rgb = synth.two_touching_floes()
        bw = ch5.otsu_mask(rgb)
        d = ch5.distance_watershed(bw, "chessboard")
        assert d.n_basins == d.L.max() and d.n_minimum_px == d.minima.sum() and d.seg_ao.sum() <= d.seg.sum()
        m = ch5.marker_watershed(bw, radius=5)
        assert m.imposed.dtype == np.float32 and np.all(m.imposed[m.marker] == -np.inf)
        assert np.array_equal(imregionalmin(m.imposed), m.marker)
        assert m.n_floes >= 2
        mp = ch5.marker_watershed(bw, point_markers=True)
        assert mp.marker0.sum() == mp.n_markers == m.n_markers
        g = ch5.gradient_watershed(rgb, smooth=7)
        assert g.g.shape == bw.shape and g.n_basins2 <= g.n_basins
        assert ch5.gradient_watershed(rgb, smooth=None).g2 is None
        dw = ch5.direct_watershed(rgb)
        assert dw.n_basins == dw.L.max() > 1
        s = ch5.topographic_surfaces(rgb)
        assert np.array_equal(s["complement"], 255 - s["gray"]) and s["neg_chessboard_dt"].max() == 0

    def test_display_helpers(self, tmp_path):
        L = np.array([[0, 1, 1], [2, 2, 0]])
        rgb = label2rgb(L)
        assert rgb.shape == (2, 3, 3) and rgb.dtype == np.uint8 and rgb[0, 0].tolist() == [0, 0, 0]
        assert np.array_equal(label2rgb(L), label2rgb(L))  # seeded shuffle
        assert label2rgb(np.zeros((2, 2), int)).sum() == 0
        surface_plot(np.arange(12.0).reshape(3, 4), tmp_path / "s.png", xlim=(0, 5))
        contour_overlay(np.arange(12.0).reshape(3, 4), np.arange(12.0).reshape(3, 4), tmp_path / "c.png", levels=3)
        assert (tmp_path / "s.png").exists() and (tmp_path / "c.png").exists()


# =====================================================================================================================
# L2 — parity against MATLAB R2025a
# =====================================================================================================================
@needs_ref("compat")
class TestL2Watershed:
    @pytest.mark.parametrize("name", WS_NAMES)
    @pytest.mark.parametrize("conn", [8, 4])
    def test_watershed_fixture(self, name, conn):
        d, ins = ref("compat"), inputs()
        cls = ref_str(d, f"cls_ws{conn}_{name}")
        X = ins[name]
        if name == "ws_logical":
            X = X.astype(bool)
        if cls.startswith("ERR"):
            pytest.skip(f"MATLAB rejected {name}: {cls}")
        ml = d[f"ws{conn}_{name}"]
        py = watershed(X, conn)  # the port accepts int16/int32 too; MATLAB ran double(X) for those (same values)
        same_labels(py, ml, f"watershed({name}, {conn})")
        assert cls in ("uint8", "uint16", "uint32", "double"), cls
        if py.max() > 255:
            assert cls == "uint16", f"{name}: {py.max()} basins but MATLAB class {cls}"
        rej = ref_str(d, f"rej_{name}")
        cls_in = ref_str(d, f"cls_in_{name}")
        if cls_in in ("int16", "int32"):
            assert "int" in rej and "Expected input number 1" in rej  # MATLAB rejects signed integer classes
        else:
            assert rej == "", f"MATLAB rejected {cls_in}: {rej}"

    @pytest.mark.parametrize("name", WS_NAMES)
    @pytest.mark.parametrize("conn", [8, 4])
    def test_imregionalmin_fixture(self, name, conn):
        d, ins = ref("compat"), inputs()
        ml = d[f"rm{conn}_{name}"]
        if ml.size == 0:
            pytest.skip("MATLAB errored on this fixture")
        X = ins[name].astype(bool) if name == "ws_logical" else ins[name]
        assert_parity(imregionalmin(X, conn), ml, "binary", name=f"imregionalmin({name},{conn})")

    def test_default_and_matrix_connectivity(self):
        d, ins = ref("compat"), inputs()
        same_labels(watershed(ins["pf_even_plateau"]), d["wsd_pf_even_plateau"], "default conn")
        same_labels(watershed(ins["pf_even_plateau"], 4), d["wsc_pf_even_plateau"], "cross conn matrix = 4")
        same_labels(watershed(ins["ws_rand_u8_30x40_ties"], 8), d["wsc8_ws_rand_u8_30x40_ties"], "ones(3) = 8")

    def test_matlab_rejections_match(self):
        d = ref("compat")
        assert sc(d["ws_nan_ok"]) == 0 and "NaN" in ref_str(d, "ws_nan_msg")
        # MATLAB's watershed is N-D: conn = 6 (a 3-D connectivity) and 3-D arrays are accepted there; the port is
        # 2-D only and raises for both (Deviation in the report)
        assert sc(d["ws_c6_ok"]) == 1 and sc(d["ws_3d_ok"]) == 1
        assert ref_str(d, "rej_ws_logical") == "" and ref_str(d, "cls_ws8_ws_logical") == "uint8"  # logical accepted
        cls = ref_str(d, "cls_ws_empty")
        if not cls.startswith("ERR"):
            assert watershed(np.zeros((0, 5))).shape == d["ws_empty"].shape

    def test_neighbour_scan_order_is_irrelevant_but_initial_scan_order_is_pinned(self):
        """Analysis risk 1.  (a) The neighbour scan order of the flooding loop cannot change the result: a pixel's
        label depends only on the *set* of labels around it, and the neighbours pushed by one pop are consecutive
        same-label FIFO entries that no foreign-label pop can interleave at equal priority — checked with random
        permutations of the offsets.  (b) The *initial* column-major seed scan does matter: running the same
        algorithm on the transpose (= a row-major scan) changes the ridges / partition on many fixtures, and those
        fixtures are the ones matched 0 px against MATLAB in ``test_watershed_fixture`` — so the order is pinned."""
        import seaice.core.watershed as wsmod

        ins = inputs()
        names = [nm for nm in WS_NAMES if nm != "ws_logical"]
        orig = wsmod.neighbour_offsets
        base = {(nm, conn): watershed(ins[nm], conn) for nm in names for conn in (8, 4)}
        rng = np.random.default_rng(0)
        try:
            for _ in range(4):
                p8, p4 = rng.permutation(8), rng.permutation(4)
                wsmod.neighbour_offsets = lambda conn, Mp, p8=p8, p4=p4: [orig(conn, Mp)[i] for i in (p8 if conn == 8 else p4)]
                for (nm, conn), L in base.items():
                    assert np.array_equal(watershed(ins[nm], conn), L), f"neighbour order changed {nm}/{conn}"
        finally:
            wsmod.neighbour_offsets = orig
        sensitive = []
        for (nm, conn), L in base.items():
            Lt = watershed(np.ascontiguousarray(ins[nm].T), conn).T
            if ((L == 0) != (Lt == 0)).any() or label_agreement(L, Lt) < 1.0:
                sensitive.append((nm, conn))
        assert len(sensitive) >= 20, sensitive
        for nm in ("ws_asym_adjacency", "ws_asym_adjacency_T", "ws_two_corridors", "ws_rand_u8_30x40_ties",
                   "ws_rand_u8_200x300_ties", "pf_four_minima_plateau"):
            assert (nm, 8) in sensitive, nm
        # and MATLAB agrees with the column-major scan on every one of them (label values included)
        d = ref("compat")
        for nm, conn in sensitive:
            same_labels(base[(nm, conn)], d[f"ws{conn}_{nm}"], f"{nm}/{conn}")


@needs_ref("compat")
class TestL2RegionalExtrema:
    @pytest.mark.parametrize("name", RM_NAMES)
    @pytest.mark.parametrize("conn", [8, 4])
    def test_min_max(self, name, conn):
        d, ins = ref("compat"), inputs()
        X = ins[name].astype(bool) if name == "rm_logical" else ins[name]
        assert_parity(imregionalmin(X, conn), d[f"rmn{conn}_{name}"], "binary", name=f"imregionalmin({name},{conn})")
        assert_parity(imregionalmax(X, conn), d[f"rmx{conn}_{name}"], "binary", name=f"imregionalmax({name},{conn})")
        assert ref_str(d, f"cls_rmn_{name}") == "logical"

    @pytest.mark.parametrize("name", RM_NAMES)
    def test_eq_5_2_identity_vs_matlab(self, name):
        d, ins = ref("compat"), inputs()
        X = ins[name].astype(bool) if name == "rm_logical" else ins[name]
        if name == "rm_all_inf":
            # Open item: the teaching form M = R^E_I(I + 1) - I is undefined on an all-+Inf image (Inf - Inf); the
            # port returns all-False where MATLAB's imregionalmin (and core.imregionalmin) return all-True.
            pytest.xfail("regional_minima_by_reconstruction(all +Inf) returns all-False; MATLAB imregionalmin all-True")
        assert_parity(ch5.regional_minima_by_reconstruction(X, 8), d[f"rmn8_{name}"], "binary", name=name)

    def test_connectivity_forms_and_nan(self):
        d, ins = ref("compat"), inputs()
        X = ins["rm_u8_ties"]
        assert_parity(imregionalmin(X, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])), d["rmn_conn_cross"], "binary")
        assert_parity(imregionalmin(X, np.ones((3, 3))), d["rmn_conn_ones"], "binary")
        assert_parity(imregionalmin(X), d["rmn_default"], "binary")
        assert sc(d["rm_nan_ok"]) == 0


@needs_ref("compat")
class TestL2Imposemin:
    @pytest.mark.parametrize("img", IM_IMAGES)
    @pytest.mark.parametrize("mk", IM_MARKERS)
    @pytest.mark.parametrize("conn", [8, 4])
    def test_imimposemin(self, img, mk, conn):
        d, ins = ref("compat"), inputs()
        I, BW = ins[img], ins[mk].astype(bool)
        key = f"ii_{img}_{mk}" if conn == 8 else f"ii4_{img}_{mk}"
        ml = d[key]
        py = imimposemin(I, BW, conn)
        assert py.dtype == I.dtype and ml.dtype == I.dtype, (py.dtype, ml.dtype, ref_str(d, f"cls_ii_{img}_{mk}"))
        # bit-identical, ±Inf included (np.array_equal treats inf == inf)
        neq = ~(np.equal(py, ml) | (np.isnan(py.astype(float)) & np.isnan(ml.astype(float))))
        assert neq.sum() == 0, f"{key}: {int(neq.sum())} px differ, max |diff| {np.nanmax(np.abs(py.astype(float) - ml.astype(float))[neq]) if neq.any() else 0}"

    @pytest.mark.parametrize("img", IM_IMAGES)
    def test_h_rule_vs_matlab(self, img):
        d, ins = ref("compat"), inputs()
        I = ins[img]
        if np.issubdtype(I.dtype, np.floating):
            rng = float(I.max()) - float(I.min())
            h = 0.1 if rng == 0 else 0.001 * rng
        else:
            h = 1
        assert np.isclose(sc(d[f"h_{img}"]), h)

    def test_plateau_markers_and_marker_classes(self):
        d, ins = ref("compat"), inputs()
        J = imimposemin(ins["im_u8_plateau"], ins["im_bw_in_plateau"].astype(bool))
        assert np.array_equal(J, d["ii_plateau"])
        assert_parity(imregionalmin(J), d["ii_plateau_rm"], "binary")
        assert sc(d["ii_plateau_n"]) == 2 == label_components(imregionalmin(J)).max()  # two markers stay apart
        same_labels(watershed(J), d["ii_plateau_ws"], "watershed(imposed)")
        assert np.array_equal(imimposemin(ins["im_double"], ins["im_bw"].astype(np.float64)), d["ii_dbl_marker"])
        assert np.array_equal(imimposemin(ins["im_double"], ins["im_bw"]), d["ii_u8_marker"])
        assert sc(d["ii_log_ok"]) == 0  # MATLAB rejects a logical I (the port raises TypeError)
        # MATLAB's imimposemin has no 'nonnan' check and silently accepts NaN (ii_nan_ok = 1); the port refuses NaN
        # like imregionalmin/watershed do (Deviation in the report) — record what MATLAB did
        assert sc(d["ii_nan_ok"]) == 1
        for k, img in (("ii_rm_u8", "im_u8"), ("ii_rm_single", "im_single"), ("ii_rm_double", "im_double")):
            assert_parity(imregionalmin(imimposemin(ins[img], ins["im_bw"].astype(bool))), d[k], "binary", name=k)

    def test_imreconstruct_with_inf(self):
        d, ins = ref("compat"), inputs()
        assert np.array_equal(imreconstruct(ins["rc_mk_inf"], ins["rc_mask_inf"]), d["rc_inf"])
        assert np.array_equal(imreconstruct(ins["rc_mk_inf"], ins["rc_mask_inf"], 4), d["rc_inf4"])


@needs_ref("compat")
class TestL2Misc:
    def test_endpoint_kernel_responses(self):
        d = ref("compat")
        EPR, EPC = d["EPR"], d["EPC"].ravel()
        assert EPC.tolist() == [3.0] * 12
        for k in range(12):
            P = np.zeros((5, 5))
            P[1:4, 1:4] = synth.FIG_5_15_ENDPOINT_PATTERNS[k]
            assert np.array_equal(imfilter(P, ch5.ENDPOINT_KERNEL), EPR[k])
        assert sc(d["ep_iso_c"]) == 4
        iso = np.zeros((5, 5))
        iso[2, 2] = 1
        assert np.array_equal(imfilter(iso, ch5.ENDPOINT_KERNEL), d["ep_iso"])
        loop = np.zeros((6, 6), dtype=bool)
        loop[1:5, 1] = loop[1:5, 4] = loop[1, 1:5] = loop[4, 1:5] = True
        assert sc(d["ep_loop_max"]) == 2
        hit = d["ep_loop_hit"].astype(bool)
        assert np.array_equal(ch5.junction_endpoints(loop, "max"), ch5._find_rows_cols(hit))

    def test_regionprops_centroid_bitand_intersect(self):
        d, ins = ref("compat"), inputs()
        bw = ins["cen_bw"].astype(bool)
        assert np.array_equal(label_components(bw), d["cen_lab"].astype(np.int32))
        assert np.allclose(ch5.component_centroids(bw), d["cen_xy"])
        assert ref_str(d, "cls_bitand") == "logical" and ref_str(d, "mul_cls") == "double"
        assert d["bitand_tf"].astype(bool).ravel().tolist() == [True, False, False, False]
        assert np.array_equal(ch5._intersect_rows(ins["ix_A"], ins["ix_B"]), d["ix_rows"].astype(np.int64))
        assert sc(d["ix_empty_n"]) == 0 and ch5._intersect_rows(ins["ix_A"], np.array([[7, 7]])).shape == (0, 2)

    def test_two_discs(self):
        d, ins = ref("compat"), inputs()
        td = ins["two_discs"].astype(bool)
        D = ch5.inverse_distance(td, "cityblock")
        assert_parity(D, d["td_D"], "float", atol=0, name="td_D")
        same_labels(watershed(D), d["td_L"], "td_L")
        assert_parity(imregionalmin(D), d["td_rm"], "binary")
        seg = td & (watershed(D) != 0)
        assert_parity(seg, d["td_seg"], "binary")
        assert sc(d["td_n"]) == 2 == label_components(seg).max()
        De = ch5.inverse_distance(td, "euclidean")
        assert_parity(De, d["td_De"], "float", atol=1e-4, name="td_De")
        same_labels(watershed(De), d["td_Le"], "td_Le (euclidean, single)")


@needs_image
@needs_ref("direct_watershed")
class TestL2DirectWatershed:
    def test_arrays(self):
        d = ref("direct_watershed")
        r = ch5.direct_watershed(q_image())
        assert np.array_equal(r.gray, d["im0"])
        assert np.isclose(sc(d["level"]), graythresh(r.gray)[0]) and sc(d["level"]) == 128 / 255
        assert_parity(r.bw, d["img"], "binary", name="img (gray Otsu)")
        same_labels(r.L, d["imgLabel"], "imgLabel")
        assert ref_str(d, "cls_L") == "uint8" and r.n_basins == sc(d["n_basins"]) == 160
        assert_parity(r.ridge, d["bgm"], "binary", name="bgm")
        assert np.array_equal(r.overlay, d["im"])
        same_labels(r.labels, d["colorimg"], "colorimg")
        assert r.n_components == sc(d["n_comp"])


@needs_image
@needs_ref("distance_watershed")
class TestL2DistanceWatershed:
    @pytest.mark.parametrize("metric", METRICS)
    def test_arrays(self, metric):
        name = DW_FILES[metric]
        if not (REF / f"{name}.mat").exists():
            pytest.skip(f"{name}.mat missing")
        d = ref(name)
        rgb = q_image()
        bw = ch5.otsu_mask(rgb)
        assert sc(d["level"]) == 130 / 255 and np.isclose(graythresh(rgb)[0], 130 / 255)
        assert_parity(bw, d["img0"], "binary", name="img0")
        r = ch5.distance_watershed(bw, metric)
        assert ref_str(d, "cls_dist") == "single" and r.imgDist.dtype == np.float32
        atol = 0.0 if metric in ("chessboard", "cityblock") else 1e-4
        assert_parity(r.imgDist, d["imgDist"], "float", atol=atol, rtol=0, name=f"imgDist {metric}")
        assert_parity(r.minima, d["Dis_img"], "binary", name="Dis_img")
        assert_parity(r.dis, d["dis"], "binary", name="dis")
        assert ref_str(d, "cls_dis") == "double"
        pq = np.column_stack([d["p"].ravel(), d["q"].ravel()]).astype(np.int64) - 1
        assert np.array_equal(r.minima_points, pq)  # same find order
        assert_parity(r.minima_overlay, d["bw0"], "binary", name="bw0")
        same_labels(r.L, d["imgLabel"], f"imgLabel {metric}")
        assert ref_str(d, "cls_L") == "uint8"
        assert_parity(r.ridge, d["bgm"], "binary", name="bgm")
        assert_parity(r.seg, d["seg_pre"], "binary", name="seg before bwareaopen")
        assert_parity(r.seg_ao, d["img"], "binary", name="img after bwareaopen")
        same_labels(r.labels, d["colorimg"], "colorimg")
        assert (r.n_minima, r.n_minimum_px, r.n_basins, r.n_floes) == tuple(int(sc(d[k])) for k in ("n_min", "n_min_px", "n_basins", "n_floes"))
        assert label_components(r.seg).max() == sc(d["n_floes_pre"])

    def test_book_counts_cityblock(self):
        if not (REF / "distance_watershed_cityblock.mat").exists():
            pytest.skip("cityblock reference missing")
        d = ref("distance_watershed_cityblock")
        assert (sc(d["n_min"]), sc(d["n_min_px"]), sc(d["n_basins"])) == (4, 18, 4)


@needs_image
@needs_ref("gradients_watershed")
class TestL2GradientsWatershed:
    def test_arrays(self):
        d = ref("gradients_watershed")
        r = ch5.gradient_watershed(q_image(), smooth=7)
        assert np.array_equal(r.gray, d["I"])
        assert_parity(r.bw, d["bw"], "binary", name="bw")
        assert_parity(ch5.sobel_magnitude(r.gray), d["g"], "float", atol=1e-12, rtol=0, name="g")
        assert_parity(r.g, d["g"], "float", atol=1e-12, rtol=0, name="g")
        same_labels(r.L, d["l"], "l")
        assert r.n_basins == sc(d["n1"]) == 350
        assert_parity(r.ridge, d["wr"], "binary", name="wr")
        assert np.array_equal(r.overlay, d["f"])
        assert_parity(r.g2, d["g2"], "float", atol=1e-12, rtol=0, name="g2")
        same_labels(r.L2, d["l2"], "l2")
        assert r.n_basins2 == sc(d["n2"]) == 12
        assert_parity(r.ridge2, d["wr2"], "binary", name="wr2")
        assert np.array_equal(r.overlay2, d["f2"])
        assert ref_str(d, "cls_l") == "uint16" and ref_str(d, "cls_l2") == "uint8"


@needs_image
@needs_ref("marker_watershed")
class TestL2MarkerWatershed:
    def test_arrays(self):
        d = ref("marker_watershed")
        bw = ch5.otsu_mask(q_image())
        assert_parity(bw, d["img0"], "binary", name="img0")
        r = ch5.marker_watershed(bw, "cityblock", 5)
        assert_parity(r.imgDist0, d["imgDist0"], "float", atol=0, rtol=0, name="imgDist0")
        assert_parity(r.minima, d["Dis_img"], "binary", name="Dis_img")
        assert (r.n_minima, r.n_minimum_px) == (sc(d["n_min"]), sc(d["n_min_px"])) == (4, 18)
        from seaice.core.morphology import strel
        assert np.array_equal(strel("disk", 5), d["se_nhood"].astype(bool))
        assert_parity(r.marker, d["marker"], "binary", name="marker")
        assert r.n_markers == sc(d["n_marker"]) == 2
        ml = d["imgDist"]
        assert ref_str(d, "cls_imp") == "single" and ml.dtype == np.float32 and r.imposed.dtype == np.float32
        assert np.array_equal(r.imposed, ml), f"imposed map differs: {int((r.imposed != ml).sum())} px"
        assert_parity(imregionalmin(r.imposed), d["rm_imp"], "binary", name="imregionalmin(imposed)")
        assert_parity(r.marker_overlay, d["bw0"], "binary", name="bw0")
        same_labels(r.L, d["imgLabel"], "imgLabel")
        assert_parity(r.ridge, d["bgm"], "binary", name="bgm")
        assert_parity(r.seg_ao, d["img"], "binary", name="img")
        same_labels(r.labels, d["colorimg"], "colorimg")
        assert (r.n_basins, r.n_floes) == (sc(d["n_basins"]), sc(d["n_floes"])) == (2, 2)

    @needs_ref("marker_watershed_centroid")
    def test_centroid_markers_variant(self):
        d = ref("marker_watershed_centroid")
        bw = ch5.otsu_mask(q_image())
        r = ch5.marker_watershed(bw, "cityblock", 5, point_markers=True)
        assert np.allclose(r.centroids, d["CENR"]) and np.allclose(r.centroids, d["CENR2"])
        assert np.array_equal(np.floor(r.centroids), d["CENF"])
        assert_parity(r.marker0, d["marker0"], "binary", name="marker0")
        assert r.n_markers == sc(d["n_marker0"]) == 2
        assert np.array_equal(r.imposed, d["imgDist"])
        same_labels(r.L, d["imgLabel"], "imgLabel (point markers)")
        assert_parity(r.seg_ao, d["img"], "binary", name="img (point markers)")
        assert_parity(r.marker_overlay, d["bw0"], "binary", name="bw0 (point markers)")
        assert r.n_floes == sc(d["n_floes"])


@needs_image
@needs_ref("topological_surface")
class TestL2TopologicalSurface:
    def test_arrays(self):
        d = ref("topological_surface")
        s = ch5.topographic_surfaces(q_image())
        assert np.array_equal(s["gray"], d["I"]) and np.array_equal(s["complement"], d["x"])
        assert ref_str(d, "cls_x") == "uint8" and s["complement"].dtype == np.uint8
        assert_parity(s["gradient"], d["g"], "float", atol=1e-12, rtol=0, name="g")
        assert_parity(s["bw"], d["img"], "binary", name="img (gray Otsu)")
        assert np.isclose(sc(d["level"]), 128 / 255)
        assert_parity(s["neg_chessboard_dt"], d["d"], "float", atol=0, name="d")
        assert ref_str(d, "cls_d") == "single"


@needs_ref("distance_propagation")
class TestL2DistancePropagation:
    def test_point_image(self):
        d = ref("distance_propagation")
        img = synth.point_image(201)
        assert_parity(~img, d["img"], "binary", name="img (= ~point)")
        r = ch5.inverse_distance_map(~img, "cityblock")
        assert ref_str(d, "cls_dist") == "single"
        assert_parity(r.imgDist, d["imgDist"], "float", atol=0, name="imgDist")
        assert_parity(r.dist0, d["dist0"], "float", atol=1e-4, name="dist0")
        assert np.array_equal(r.dist, d["dist"]) and r.dist.dtype == np.uint8

    @needs_image
    def test_q_variants(self):
        d = ref("distance_propagation")
        bw = ch5.otsu_mask(q_image())
        assert_parity(bw, d["bwq"], "binary", name="bwq")
        for m, key, atol in (("cityblock", "D_city", 0), ("euclidean", "D_euc", 1e-4), ("chessboard", "D_chess", 0),
                             ("quasi-euclidean", "D_quasi", 1e-4)):
            assert_parity(ch5.inverse_distance(bw, m), d[key], "float", atol=atol, rtol=0, name=key)
        for m, key in (("cityblock", "dist_city"), ("euclidean", "dist_euc"), ("chessboard", "dist_chess")):
            r = ch5.inverse_distance_map(bw, m)
            assert np.array_equal(r.dist, d[key]), f"{key}: {int((r.dist != d[key]).sum())} px"
        # the script's literal (commented) path: img = ~bw; -bwdist(~img) = distance of the water to the ice
        assert_parity(-bwdist(bw, "cityblock"), d["Dlit_city"], "float", atol=0, name="Dlit_city")


@needs_image
@needs_ref("chaincode_corner")
class TestL2ChainCode:
    def test_arrays(self):
        d = ref("chaincode_corner")
        rgb = q_image()
        r = ch5.freeman_concave(rgb)
        assert_parity(r.bw, d["B"], "binary", name="B")
        assert np.array_equal(r.boundary + 1, d["b"])
        assert r.boundary.shape[0] == 263 and sc(d["l"]) == 262 == r.code.fcc.size
        assert r.n_boundaries == sc(d["n_boundaries"]) == 1 and r.longest + 1 == sc(d["k"])
        assert np.array_equal(r.code.fcc, d["fcc"].ravel()) and np.array_equal(np.asarray(r.code.x0y0) + 1, d["x0y0"].ravel())
        assert np.array_equal(r.code.diff, d["diff_c"].ravel())
        assert_parity(r.bim, d["bim"], "binary", name="bim")
        for key, arr in (("R", r.R), ("A", r.A), ("Sum", r.S), ("D", r.D), ("Diff", r.Diff)):
            assert np.array_equal(arr, d[key].ravel().astype(np.int64)), key
        assert sc(d["A_last"]) == -8 == r.A[-1] and sc(d["Sum0_final"]) == 0
        assert sc(d["D0"]) == r.Diff[0]
        assert np.array_equal(r.index + 1, d["p"].ravel())
        assert np.array_equal(r.points_matlab, d["concave"])
        assert r.points.shape[0] == 9
        assert r.Diff.min() == -5 and r.Diff.max() == 7

    def test_fig_5_16_tracing_vs_matlab(self):
        d = ref("chaincode_corner")
        b = boundaries(synth.FIG_5_16_IMAGE, 8, "cw")[0]
        assert np.array_equal(b + 1, d["b516"]) and sc(d["n516"]) == 1
        c = fchcode(b)
        assert np.array_equal(c.fcc, d["f516"].ravel()) and np.array_equal(np.asarray(c.x0y0) + 1, d["x0y0_516"].ravel())
        assert np.array_equal(boundaries(synth.FIG_5_16_IMAGE, 8, "ccw")[0] + 1, d["b516_ccw"])


def _check_main(d: dict, rgb: np.ndarray, tag: str) -> ch5.MergeResult:
    bw = ch5.otsu_mask(rgb)
    assert_parity(bw, d["bw"], "binary", name=f"{tag} bw")
    assert (bw.shape[0], bw.shape[1]) == (sc(d["m"]), sc(d["n"]))
    r = ch5.neighboring_region_merging(bw)
    assert_parity(r.D, d["D"], "float", atol=0, name=f"{tag} D")
    same_labels(r.L, d["L"], f"{tag} L")
    assert ref_str(d, "cls_L") in ("uint8", "uint16")
    assert_parity(r.w, d["w"], "binary", name=f"{tag} w")
    assert_parity(r.f, d["f"], "binary", name=f"{tag} f")
    assert ref_str(d, "cls_f") == "logical"
    assert_parity(r.seg0, d["seg0"], "binary", name=f"{tag} seg0")
    assert_parity(r.seg, d["seg"], "binary", name=f"{tag} seg")
    same_labels(r.label, d["label"], f"{tag} label (bwlabel(f,4))")
    assert r.num == sc(d["num"])
    assert np.array_equal(ch5.ENDPOINT_KERNEL, d["wr"])
    EP, CONC, CC, REG, NBR, CONNECT, G2, SEGB = (cell(d, k) for k in ("EP", "CONC", "CC", "REG", "NBR", "CONNECT", "G2", "SEGB"))
    removed = np.asarray(d["REMOVED"]).ravel().astype(bool)
    TT = np.asarray(d["TT"]).ravel()
    assert len(r.lines) == len(EP) == r.num
    for i, ln in enumerate(r.lines):
        assert np.array_equal(ln.endpoints + 1, rows2(EP[i]).astype(np.int64)), f"{tag} line {i + 1} endpoints"
        assert np.array_equal(ln.concave + 1, rows2(CONC[i]).astype(np.int64)), f"{tag} line {i + 1} concave"
        assert np.array_equal(ln.concave_endpoints + 1, rows2(CC[i]).astype(np.int64)), f"{tag} line {i + 1} c"
        assert ln.removed == bool(removed[i])
        assert np.array_equal(ln.region, REG[i]), f"{tag} line {i + 1} im"
        assert np.array_equal(label_components(NBR[i].astype(bool), 8).astype(np.float64), CONNECT[i]), f"{tag} line {i + 1} connect"
        g = ln.pixels
        gm = np.zeros(bw.shape)
        gm[g[:, 0], g[:, 1]] = 1
        assert np.array_equal(np.abs(imfilter(gm, ch5.ENDPOINT_KERNEL)) >= TT[i], G2[i].astype(bool)), f"{tag} line {i + 1} g2"
        assert TT[i] == np.abs(imfilter(gm, ch5.ENDPOINT_KERNEL)).max()
    assert r.n_floes_before == sc(d["n_floes0"]) and r.n_floes_after == sc(d["n_floes"])
    assert r.n_removed == sc(d["n_removed"]) and int(r.f.sum()) == sc(d["n_lines_px"])
    return r


@needs_ref("main")
class TestL2Main:
    @needs_image
    def test_q_image(self):
        d = ref("main")
        r = _check_main(d, q_image(), "main")
        assert r.num == 3 and r.n_removed == 2 and r.n_floes_before == 4 and r.n_floes_after == 2
        assert [ln.n_pixels for ln in r.lines] == [22, 22, 19]
        assert [tuple(map(tuple, ln.endpoints + 1)) for ln in r.lines] == [((64, 11), (64, 32)), ((70, 12), (70, 33)), ((48, 29), (62, 33))]
        assert [ln.removed for ln in r.lines] == [True, True, False]
        assert r.lines[2].concave_endpoints.shape[0] == 2

    @needs_ref("main_synth")
    def test_synthetic_substitute(self):
        d = ref("main_synth")
        r = _check_main(d, synth.two_touching_floes(), "main_synth")
        assert r.n_floes_after == 2

    @needs_ref("main_crop")
    def test_ch04_crop(self):
        if not (DATA4 / "test.jpg").exists():
            pytest.skip("data/book/ch04/test.jpg missing")
        from seaice.ch04_ice_edge_detection import fig_4_3a
        import imageio.v3 as iio
        rgb = fig_4_3a(iio.imread(DATA4 / "test.jpg"))
        _check_main(ref("main_crop"), rgb, "main_crop")


@needs_image
@needs_ref("fig_5_14f")
@needs_ref("main")
class TestL2AuthorsFigFile:
    def test_fig_5_14f_matches_f_and_endpoints(self):
        d = ref("fig_5_14f")
        r = ch5.neighboring_region_merging(ch5.otsu_mask(q_image()))
        cdata = d["FIG_CDATA"]
        assert cdata.shape == r.f.shape and ref_str(d, "cls_cdata") in ("uint8", "logical")
        assert_parity(cdata > 0, r.f, "binary", name="fig CData vs f")
        nl = int(sc(d["nl"]))
        assert nl == 3
        pts = set()
        for k in range(nl):
            for x, y in zip(d["FIG_X"][k], d["FIG_Y"][k]):
                pts.add((int(round(y)), int(round(x))))  # (row, col) 1-based
        ours = {tuple(p) for ln in r.lines for p in (ln.endpoints + 1).tolist()}
        assert pts == ours, (pts, ours)


# =====================================================================================================================
# L4 — numbers quoted in the book text
# =====================================================================================================================
@needs_image
class TestL4BookNumbers:
    def test_p96_four_minima_18_px_four_regions_two_markers(self):
        bw = ch5.otsu_mask(q_image())
        dw = ch5.distance_watershed(bw, "cityblock")
        assert dw.n_minima == 4 and dw.n_minimum_px == 18 and dw.n_basins == 4 and dw.n_floes == 4
        mk = ch5.marker_watershed(bw, "cityblock", 5)
        assert mk.n_markers == 2 and mk.n_basins == 2 and mk.n_floes == 2

    def test_p99_fig_5_14_three_lines_two_removed(self):
        r = ch5.neighboring_region_merging(ch5.otsu_mask(q_image()))
        assert r.num == 3 and r.n_removed == 2 and (r.n_floes_before, r.n_floes_after) == (4, 2)

    def test_p104_fig_5_17_nine_concave_points_at_two_notches(self):
        r = ch5.freeman_concave(q_image())
        pts = r.points_matlab.tolist()
        assert len(pts) == 9
        assert sorted(pts) == sorted([[48, 27], [49, 28], [48, 29], [47, 30], [62, 35], [62, 34], [62, 33], [63, 32], [64, 32]])

    def test_fig_5_6_7x7_and_fig_5_5_counts(self):
        g = ch5.gradient_watershed(q_image(), smooth=7)
        assert g.n_basins == 350 and g.n_basins2 == 12
        assert ch5.BOOK_PARAMS["close_open_size"] == 7 and ch5.BOOK_PARAMS["marker_disk_radius"] == 5
        assert (ch5.BOOK_PARAMS["concave_min"], ch5.BOOK_PARAMS["concave_max"]) == (3, 10)


# =====================================================================================================================
# scripts run headless
# =====================================================================================================================
SCRIPTS = ["ch05_direct_watershed", "ch05_distance_propagation", "ch05_distance_watershed", "ch05_gradients_watershed",
           "ch05_marker_watershed", "ch05_topological_surface", "ch05_chaincode_corner", "ch05_main", "ch05_experiments"]


def _run(script: str, out: Path, *flags: str) -> subprocess.CompletedProcess:
    cmd = [str(PY), str(ROOT / "scripts" / f"{script}.py"), "--no-show", "--out", str(out), *flags]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs(script):
    out = VERIFY / "pytest_scripts" / script
    out.mkdir(parents=True, exist_ok=True)
    proc = _run(script, out)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    if not (DATA / "q.jpg").exists() and script not in ("ch05_distance_propagation", "ch05_experiments"):
        assert "SKIP" in proc.stdout
    else:
        assert list(out.glob("*.png")), "no figure written"


CLI_CASES = [
    ("ch05_distance_watershed", ["--metric", "all", "--min-area", "3"]),
    ("ch05_marker_watershed", ["--metric", "chessboard", "--radius", "3", "--centroid-markers"]),
    ("ch05_main", ["--metric", "euclidean", "--endpoint-rule", "ge3", "--no-sequential"]),
    ("ch05_distance_propagation", ["--source", "q.jpg"]),
    ("ch05_distance_propagation", ["--metric", "all", "--size", "101"]),
    ("ch05_gradients_watershed", ["--smooth", "0"]),
    ("ch05_chaincode_corner", ["--object", "longest"]),
    ("ch05_experiments", ["--crop", "0:300,0:400", "--metric", "chessboard", "--endpoint-rule", "ge3"]),
]


@pytest.mark.parametrize("script,flags", CLI_CASES, ids=[f"{s}:{' '.join(f)}" for s, f in CLI_CASES])
def test_script_cli_flags(script, flags):
    safe = "_".join(f.strip("-").replace(":", "-").replace(",", "_") for f in flags)  # no ':' on Windows paths
    out = VERIFY / "pytest_scripts" / f"{script}_{safe}"
    out.mkdir(parents=True, exist_ok=True)
    proc = _run(script, out, *flags)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
