"""Chapter 4 verification — Ice Edge Detection (Book pp. 59–82).

Evidence levels (verify-port skill):
  L1  synthetic truth (no external tools)            -> ``TestL1*``
  L2  parity vs MATLAB R2025a references              -> ``TestL2*`` (skip if ``reference/ch04/*.mat`` missing;
      regenerate with ``.venv/Scripts/python.exe reference/ch04/make_refs.py``)
  L4  numbers quoted in the book text                 -> ``TestL4BookNumbers``
  scripts: every ``scripts/ch04_*.py`` must run headless and exit 0 -> ``test_script_runs``

The MATLAB references were produced by the ORIGINAL ``derivative.m`` / ``morphology.m`` (verbatim copies with only
the ``imread('test.jpg')`` literal patched, see ``reference/ch04/make_refs.py``) and by the toolbox functions
(``edge``, ``fspecial``, ``strel``, ``imerode``, ``imdilate``, ``imopen``, ``imclose``, ``imreconstruct``,
``medfilt2``, ``bwareaopen``, ``conv2``) on the controlled fixtures of ``reference/ch04/inputs.mat``.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage
from scipy.io import loadmat

from seaice import ch04_ice_edge_detection as ch4
from seaice.core import synth
from seaice.core.edges import EdgeResult, edge, gradient_roberts, gradient_sobel_prewitt, log_zero_crossings, \
    thin_gradient
from seaice.core.filters import conv2, fspecial, imfilter
from seaice.core.io import load_image
from seaice.core.matlab_compat import rgb2gray_matlab
from seaice.core.morphology import _border_values, _matlab_minus, disk_decomposition, geodesic_dilation, geodesic_erosion, imclose, \
    imdilate, imerode, imopen, imreconstruct, intline, line_strel, minkowski_sum, morphological_gradient, \
    periodic_line, reconstruct_by_erosion, reconstruct_iterative, se_origin, strel
from seaice.core.threshold import graythresh, im2bw
from tools.compare_arrays import assert_parity

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "ch04"
DATA = ROOT / "data" / "book" / "ch04"
VERIFY = ROOT / "outputs" / "ch04" / "verify"

needs_image = pytest.mark.skipif(not (DATA / "test.jpg").exists(),
                                 reason="data/book/ch04/test.jpg missing (run /setup-project)")


def needs_ref(name: str):
    return pytest.mark.skipif(not (REF / f"{name}.mat").exists(),
                              reason=f"reference/ch04/{name}.mat missing — run reference/ch04/make_refs.py (MATLAB)")


_REF_CACHE: dict[str, dict] = {}


def ref(name: str) -> dict:
    if name not in _REF_CACHE:
        d = loadmat(str(REF / f"{name}.mat"))
        _REF_CACHE[name] = {k: v for k, v in d.items() if not k.startswith("__")}
    return _REF_CACHE[name]


def sc(x) -> float:
    """Scalar stored by MATLAB as a 1x1 array."""
    return float(np.asarray(x).ravel()[0])


def ref_str(d: dict, key: str) -> str:
    v = np.asarray(d[key]).ravel()
    return "" if v.size == 0 else str(v[0])


_BOOL_INPUTS = ("Bw", "BwB", "Mk", "Bw2", "F48", "F48SE")


def inputs() -> dict:
    """The controlled fixtures shared with MATLAB (``inputs.mat``; bool arrays stored as uint8 0/1)."""
    if "inputs" not in _REF_CACHE:
        d = loadmat(str(REF / "inputs.mat"))
        d = {k: v for k, v in d.items() if not k.startswith("__")}
        for k in _BOOL_INPUTS:
            d[k] = d[k].astype(bool)
        _REF_CACHE["inputs"] = d
    return _REF_CACHE["inputs"]


_GRAY_CACHE: dict[str, np.ndarray] = {}


def gray_image() -> np.ndarray:
    """MATLAB ``rgb2gray(imread('test.jpg'))`` (4290x2856)."""
    if "test" not in _GRAY_CACHE:
        if not (DATA / "test.jpg").exists():
            pytest.skip("test.jpg missing")
        rgb, _ = load_image("ch04", "test.jpg", allow_fallback=False, verbose=False)
        _GRAY_CACHE["test"] = rgb2gray_matlab(rgb)
    return _GRAY_CACHE["test"]


def se_by_name(name: str) -> np.ndarray:
    """The Python structuring element for a ``compat.mat`` SE key (see ``make_refs.MORPH_SES``)."""
    inp = inputs()
    return {"disk7": strel("disk", 7), "asym": inp["AsymSE"] != 0, "even": inp["EvenSE"] != 0,
            "line5": strel("line", 5, 0), "line7_45": strel("line", 7, 45), "sq3": strel("square", 3),
            "dia3": strel("diamond", 3), "disk2": strel("disk", 2), "pair": strel("pair", [2, -1])}[name]


def _assert_same(out: np.ndarray, m: np.ndarray, dtype, name: str) -> None:
    assert out.dtype == dtype, name
    if dtype == np.bool_:
        n = int((out != m.astype(bool)).sum())
        assert n == 0, f"{name}: {n} px differ"
    else:
        diff = out.astype(np.float64) != m.astype(np.float64)
        assert not diff.any(), (f"{name}: {int(diff.sum())} px differ, max abs "
                                f"{np.abs(out.astype(np.float64) - m.astype(np.float64)).max()}")


def _imclose_matlab_padding(I: np.ndarray, se: np.ndarray, pad_value=0) -> np.ndarray:
    """R2025a imclose.m: padSize = ceil(size(nhood)/2); Ap = padarray(A, padSize, 'both') (zeros);
    B = imerode(imdilate(Ap, se), se) cropped back.  ``pad_value`` lets the test probe the -Inf/intmin variant."""
    pr, pc = (int(np.ceil(n / 2)) for n in se.shape)
    Ap = np.pad(I, ((pr, pr), (pc, pc)), mode="constant", constant_values=pad_value)
    Bp = imerode(imdilate(Ap, se), se)
    return Bp[pr:pr + I.shape[0], pc:pc + I.shape[1]]


# printed matrices (book pages 66, 67)
FIG_4_5A = np.array([[0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
                     [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                     [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
                     [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                     [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]])
FIG_4_5B = np.array([[0.0239, 0.0460, 0.0499, 0.0460, 0.0239],
                     [0.0460, 0.0061, -0.0923, 0.0061, 0.0460],
                     [0.0499, -0.0923, -0.3182, -0.0923, 0.0499],
                     [0.0460, 0.0061, -0.0923, 0.0061, 0.0460],
                     [0.0239, 0.0460, 0.0499, 0.0460, 0.0239]])
FIG_4_7C = np.array([[0, 0, 1, 1, 1, 1, 1, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 1, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1, 1, 1, 0, 0]], dtype=bool)


def _literal_erode(I: np.ndarray, se: np.ndarray, pad_val) -> np.ndarray:
    """Eq. (4.20) written out: min over the SE placed at each pixel, outside pixels = ``pad_val``."""
    M, N = se.shape
    r0, c0 = se_origin(se)
    out = np.empty(I.shape, dtype=np.float64)
    for r in range(I.shape[0]):
        for c in range(I.shape[1]):
            vals = []
            for i, j in zip(*np.nonzero(se)):
                rr, cc = r + i - r0, c + j - c0
                vals.append(I[rr, cc] if 0 <= rr < I.shape[0] and 0 <= cc < I.shape[1] else pad_val)
            out[r, c] = min(vals)
    return out


def _literal_dilate(I: np.ndarray, se: np.ndarray, pad_val) -> np.ndarray:
    """Eq. (4.21) written out: max of ``f(x-s, y-t)`` over the SE (reflected placement), outside = ``pad_val``."""
    M, N = se.shape
    r0, c0 = se_origin(se)
    out = np.empty(I.shape, dtype=np.float64)
    for r in range(I.shape[0]):
        for c in range(I.shape[1]):
            vals = []
            for i, j in zip(*np.nonzero(se)):
                rr, cc = r - (i - r0), c - (j - c0)
                vals.append(I[rr, cc] if 0 <= rr < I.shape[0] and 0 <= cc < I.shape[1] else pad_val)
            out[r, c] = max(vals)
    return out


# ================================================================================================================
# L1 — synthetic truth
# ================================================================================================================
class TestL1Kernels:
    def test_fspecial_sobel_prewitt_are_minus_fig_4_2(self):
        assert np.array_equal(fspecial("sobel"), -ch4.SOBEL_KERNELS["x"])
        assert np.array_equal(fspecial("prewitt"), -ch4.PREWITT_KERNELS["x"])
        assert np.array_equal(ch4.SOBEL_KERNELS["y"], ch4.SOBEL_KERNELS["x"].T)
        assert np.array_equal(ch4.PREWITT_KERNELS["y"], ch4.PREWITT_KERNELS["x"].T)
        for k in (ch4.SOBEL_KERNELS, ch4.PREWITT_KERNELS, ch4.FORWARD_KERNELS):
            assert k["x"].sum() == 0 and k["y"].sum() == 0  # flat regions → 0 (p. 61)

    def test_fspecial_laplacian_fig_4_4(self):
        assert np.array_equal(fspecial("laplacian", 0), ch4.LAPLACIAN_KERNELS[4])  # Eq. (4.13)
        lap = fspecial("laplacian")  # alpha 0.2 default
        assert lap[1, 1] == pytest.approx(-4 / 1.2) and lap[0, 0] == pytest.approx(0.2 / 1.2)
        assert abs(lap.sum()) < 1e-15
        assert np.array_equal(fspecial("laplacian", 1), np.array([[.5, 0, .5], [0, -2, 0], [.5, 0, .5]]))

    def test_fspecial_gaussian_and_log_equal_fig_4_5(self):
        g = fspecial("gaussian", 5, 1)
        lg = fspecial("log", 5, 1)
        assert np.allclose(np.round(g, 4), FIG_4_5A) and np.abs(g - FIG_4_5A).max() < 5e-5
        assert np.allclose(np.round(lg, 4), FIG_4_5B) and np.abs(lg - FIG_4_5B).max() < 5e-5
        assert g.sum() == pytest.approx(1.0, abs=1e-15)
        assert abs(lg.sum()) < 1e-15
        assert np.array_equal(g, g.T) and np.array_equal(lg, lg.T)
        # positional / keyword / default forms
        assert np.array_equal(fspecial("gaussian", hsize=5, sigma=1), g)
        assert fspecial("gaussian").shape == (3, 3) and fspecial("log").shape == (5, 5)
        assert fspecial("gaussian", [5, 7], 1.3).shape == (5, 7)
        assert fspecial("gaussian", None, 1.5).shape == (7, 7)  # 2*ceil(2*sigma)+1

    def test_fspecial_average_disk_unsharp(self):
        assert np.array_equal(fspecial("average"), np.ones((3, 3)) / 9)
        assert fspecial("average", [3, 5]).shape == (3, 5) and fspecial("average", [3, 5]).sum() == pytest.approx(1)
        d = fspecial("disk", 5)
        assert d.shape == (11, 11) and d.sum() == pytest.approx(1.0) and d[5, 5] == d.max()
        assert np.array_equal(d, d.T) and np.array_equal(d, d[::-1])
        u = fspecial("unsharp")
        assert np.allclose(u, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) - fspecial("laplacian", 0.2))
        with pytest.raises(ValueError):
            fspecial("nosuch")
        with pytest.raises(ValueError):
            fspecial("laplacian", 2.0)

    def test_gaussian_kernel_eq_4_14_and_log_kernel_eq_4_15(self):
        g = ch4.gaussian_kernel(5, 1.0, normalize=False)
        assert g[2, 2] == pytest.approx(1 / (2 * np.pi))  # Eq. (4.14) at the origin
        assert 0.9 < g.sum() < 1.0  # truncation at 2 sigma (5x5) loses mass
        assert np.allclose(ch4.gaussian_kernel(5, 1.0), fspecial("gaussian", 5, 1), atol=1e-15)
        g7 = ch4.gaussian_kernel(7, 1.0, normalize=False)
        assert g7.sum() > 0.99  # "3σ from the mean covers more than 99 %" (p. 65)
        lg = ch4.log_kernel(5, 1.0, "analytic")
        assert lg[2, 2] == pytest.approx(-1 / np.pi)  # Eq. (4.15) at the origin: -2σ²/(2πσ⁶) = -1/(πσ⁴)
        assert lg.sum() != 0  # the analytic sampling does not sum to zero
        assert np.array_equal(ch4.log_kernel(5, 1.0, "matlab"), fspecial("log", 5, 1))
        assert np.array_equal(ch4.log_kernel(5, 1.0), fspecial("log", 5, 1))  # default = Fig. 4.5(b)
        gl = ch4.log_kernel(5, 1.0, "gauss_conv_laplacian")
        assert np.allclose(gl, conv2(fspecial("gaussian", 5, 1), ch4.LAPLACIAN_KERNELS[4], "same"), atol=1e-15)
        assert gl[2, 2] < 0
        with pytest.raises(ValueError):
            ch4.log_kernel(5, 1.0, "bogus")


class TestL1Strel:
    def test_disk_5_equals_fig_4_7c_and_sizes(self):
        assert np.array_equal(strel("disk", 5), FIG_4_7C)  # printed 9x9 / 69 px (n = 4 approximation)
        assert int(FIG_4_7C.sum()) == 69
        for r, shape, n_px in ((1, (3, 3), 5), (2, (5, 5), 13), (3, (5, 5), 25), (5, (9, 9), 69), (7, (13, 13), 157),
                               (15, (29, 29), 697), (16, (31, 31), 817)):
            s = strel("disk", r)
            assert s.shape == shape and int(s.sum()) == n_px, (r, s.shape, s.sum())
        # r < 3 and n = 0 → exact Euclidean disk
        yy, xx = np.mgrid[-5:6, -5:6]
        assert np.array_equal(strel("disk", 5, 0), xx * xx + yy * yy <= 25)
        assert int(strel("disk", 5, 0).sum()) == 81
        assert np.array_equal(strel("disk", 2), np.array([[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1],
                                                          [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]], dtype=bool))
        assert np.array_equal(strel("disk", 0), np.ones((1, 1), bool))

    def test_prefix_matching_like_matlab(self):
        assert np.array_equal(strel("dis", 7), strel("disk", 7))  # morphology.m line 6
        assert np.array_equal(strel("DISK", 7), strel("disk", 7))
        with pytest.raises(ValueError):
            strel("di", 7)  # ambiguous: disk / diamond (MATLAB errors too)
        with pytest.raises(ValueError):
            strel("nosuch", 3)

    def test_diamond_square_rectangle_octagon_pair_arbitrary(self):
        d5 = strel("diamond", 5)
        rr, cc = np.mgrid[-5:6, -5:6]
        assert d5.shape == (11, 11) and int(d5.sum()) == 61 and np.array_equal(d5, np.abs(rr) + np.abs(cc) <= 5)
        assert d5[0, 5] and d5[5, 0] and d5[5, 10] and d5[10, 5] and not d5[0, 0]  # Fig. 4.7(d) extreme points
        assert np.array_equal(strel("square", 3), np.ones((3, 3), bool))  # Fig. 4.7(a)
        assert np.array_equal(strel("rectangle", [3, 5]), np.ones((3, 5), bool))  # Fig. 4.7(b)
        assert strel("octagon", 6).shape == (13, 13) and strel("octagon", 0).shape == (1, 1)
        with pytest.raises(ValueError):
            strel("octagon", 4)
        p = strel("pair", [2, -1])
        assert p.shape == (5, 3) and int(p.sum()) == 2 and p[2, 1] and p[4, 0]
        a = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
        assert np.array_equal(strel(a), a != 0) and np.array_equal(strel("arbitrary", a), a != 0)
        with pytest.raises(NotImplementedError):
            strel("sphere", 3)

    def test_line_periodicline_intline(self):
        assert np.array_equal(strel("line", 5, 0), np.ones((1, 5), bool))
        assert np.array_equal(strel("line", 5, 90), np.ones((5, 1), bool))
        # len 7 at 45 deg: x = round(3 cos 45) = 2 -> 5x5 anti-diagonal (5 px, not 7; MATLAB identical, TestL2Strel.test_line)
        assert np.array_equal(strel("line", 7, 45), np.eye(5, dtype=bool)[::-1])  # counter-clockwise from horizontal
        assert np.array_equal(strel("line", 7, 135), np.eye(5, dtype=bool))
        assert np.array_equal(strel("line", 9, 45), np.eye(7, dtype=bool)[::-1])  # round(4 cos 45) = 3
        assert strel("line", 1, 30).shape == (1, 1)
        assert np.array_equal(strel("line", 7, 180), strel("line", 7, 0))
        pl = periodic_line(2, (1, 2))
        assert pl.shape == (5, 9) and int(pl.sum()) == 5 and pl[2, 4] and pl[0, 0] and pl[4, 8]
        assert np.array_equal(strel("periodicline", 2, [1, 2]), pl)
        x, y = intline(0, 4, 0, 2)
        assert np.array_equal(x, [0, 1, 2, 3, 4]) and np.array_equal(y, [0, 1, 1, 2, 2])  # MATLAB round half up
        assert se_origin(np.ones((3, 5))) == (1, 2) and se_origin(np.ones((2, 4))) == (0, 1)

    def test_disk_decomposition_minkowski_sum_equals_strel(self):
        for r in (3, 5, 7, 10, 15, 16, 20):
            seq = disk_decomposition(r, 4)
            assert len(seq) == 6  # 4 periodic lines + horizontal + vertical line
            assert np.array_equal(minkowski_sum(seq), strel("disk", r))
        assert disk_decomposition(2, 4) == [] and disk_decomposition(7, 0) == []
        assert len(disk_decomposition(7, 6)) == 8 and len(disk_decomposition(7, 8)) == 10
        shapes = [s.shape for s in disk_decomposition(7, 4)]
        assert shapes == [(5, 1), (3, 3), (1, 5), (3, 3), (1, 5), (5, 1)]
        with pytest.raises(ValueError):
            disk_decomposition(7, 5)


class TestL1ErodeDilate:
    def test_fig_4_8_matrices(self):
        d = ch4.fig_4_8_demo()
        assert d["erosion_matches"] and d["dilation_matches"]
        assert np.array_equal(imerode(synth.FIG_4_8_IMAGE, synth.FIG_4_8_SE), synth.FIG_4_8_ERODED)
        assert np.array_equal(imdilate(synth.FIG_4_8_IMAGE, synth.FIG_4_8_SE), synth.FIG_4_8_DILATED)
        assert int(synth.FIG_4_8_IMAGE.sum()) == 21 and int(synth.FIG_4_8_ERODED.sum()) == 5
        assert int(synth.FIG_4_8_DILATED.sum()) == 41  # 5 rows x 9 cols minus 4 corners
        assert imerode(synth.FIG_4_8_IMAGE, synth.FIG_4_8_SE).dtype == np.bool_

    def test_border_rule(self):
        ones = np.ones((6, 7), dtype=bool)
        se = strel("disk", 3)
        assert imerode(ones, se).all()  # outside counts as 1 for erosion
        assert imdilate(ones, se).all()
        zeros = np.zeros((6, 7), dtype=bool)
        assert not imdilate(zeros, se).any()  # outside counts as 0 for dilation
        obj = np.zeros((8, 8), dtype=bool)
        obj[:3, :3] = True  # object touching the top-left border
        er = imerode(obj, np.ones((3, 3), bool))
        assert er[0, 0] and er[1, 1] and not er[2, 2] and int(er.sum()) == 4  # eroded only from the inside
        u8 = np.full((5, 5), 100, dtype=np.uint8)
        assert (imerode(u8, se) == 100).all() and (imdilate(u8, se) == 100).all()  # intmax / intmin pads
        f = np.full((5, 5), -3.5)
        assert (imerode(f, se) == -3.5).all() and (imdilate(f, se) == -3.5).all()  # +Inf / -Inf pads
        i16 = np.full((5, 5), -1000, dtype=np.int16)
        assert (imerode(i16, se) == -1000).all() and imerode(i16, se).dtype == np.int16

    def test_reflection_with_asymmetric_se_literal_eqs_4_20_4_21(self):
        rng = np.random.default_rng(0)
        I = rng.integers(0, 256, size=(9, 11)).astype(np.uint8)
        B = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)
        assert np.array_equal(imerode(I, B), _literal_erode(I, B, 255).astype(np.uint8))
        assert np.array_equal(imdilate(I, B), _literal_dilate(I, B, 0).astype(np.uint8))
        # the non-reflected ("correlation") dilation is a different image → the reflection is load-bearing
        assert not np.array_equal(imdilate(I, B), imdilate(I, B[::-1, ::-1]))
        bw = I > 128
        assert np.array_equal(imerode(bw, B), _literal_erode(bw, B, True).astype(bool))
        assert np.array_equal(imdilate(bw, B), _literal_dilate(bw, B, False).astype(bool))
        # even-sized SE: origin floor((size+1)/2) → (0, 1)
        E = np.array([[1, 1, 1], [1, 0, 1]], dtype=bool)
        assert np.array_equal(imerode(I, E), _literal_erode(I, E, 255).astype(np.uint8))
        assert np.array_equal(imdilate(I, E), _literal_dilate(I, E, 0).astype(np.uint8))
        F = rng.random((9, 11)) - 0.5
        assert np.array_equal(imerode(F, B), _literal_erode(F, B, np.inf))
        assert np.array_equal(imdilate(F, B), _literal_dilate(F, B, -np.inf))

    def test_decomposed_sequence_equals_full_neighbourhood(self):
        rng = np.random.default_rng(1)
        I = rng.integers(0, 256, size=(40, 50)).astype(np.uint8)
        bw = I > 100
        for r in (3, 7, 15):
            seq = disk_decomposition(r)
            se = strel("disk", r)
            assert np.array_equal(imerode(I, seq), imerode(I, se))
            assert np.array_equal(imdilate(I, seq), imdilate(I, se))
            assert np.array_equal(imerode(bw, seq), imerode(bw, se))
            assert np.array_equal(imdilate(bw, seq), imdilate(bw, se))
        with pytest.raises(ValueError):
            imerode(I, [])
        with pytest.raises(ValueError):
            imerode(np.zeros((2, 2, 2)), se)

    def test_open_close_properties_and_1d_hand_truth(self):
        d = ch4.profile_open_close_demo(9)
        f = d["f"].astype(int)
        assert np.all(d["opening"] <= f) and np.all(d["closing"] >= f)  # anti-extensive / extensive
        assert np.all(d["erosion"] <= d["opening"]) and np.all(d["dilation"] >= d["closing"])
        assert np.all(d["opening"][10:12] == 40)  # 2-px bright speck removed (Fig. 4.12 "push up from below")
        assert np.all(d["opening"][30:100] == 170) and np.all(d["opening"][103:140] == 200)  # wide floes kept
        assert np.all(d["closing"][100:103] == 170)  # 3-px crack filled to the lower floe (Fig. 4.11)
        assert np.all(d["closing"][140:143] == 200)  # 3-px dark pit filled
        assert np.all(d["closing"][0:8] == 40) and np.all(d["closing"][200:] == 40)  # water unchanged
        B = strel("line", 9, 0)
        f2 = synth.two_floes_profile()
        assert np.array_equal(imopen(imopen(f2, B), B), imopen(f2, B))  # idempotence
        assert np.array_equal(imclose(imclose(f2, B), B), imclose(f2, B))


class TestL1Reconstruction:
    def test_two_blobs_marker_keeps_only_marked_blob(self):
        marker, mask = synth.two_blobs_with_marker()
        rec = imreconstruct(marker, mask)
        rr, cc = np.mgrid[0:40, 0:60]
        disc = (rr - 18) ** 2 + (cc - 16) ** 2 <= 100
        assert np.array_equal(rec, disc) and rec.dtype == np.bool_
        assert int(mask.sum()) > int(rec.sum())
        it, k = reconstruct_iterative(marker, mask, None, "dilation")
        assert np.array_equal(it, rec) and k == 11  # 10 geodesic steps reach the rim (chessboard radius) + the stability check
        assert np.array_equal(imreconstruct(marker, mask, 4), disc)
        with pytest.raises(ValueError):
            imreconstruct(mask, marker)  # marker > mask
        with pytest.raises(ValueError):
            imreconstruct(marker, mask[:, :50])

    def test_geodesic_steps_eqs_4_25_4_26(self):
        marker, mask = synth.two_blobs_with_marker()
        d1 = geodesic_dilation(marker, mask, None, 1)
        assert int(d1.sum()) == 9  # 3x3 around the seed, all inside the mask
        d2 = geodesic_dilation(d1, mask, None, 1)
        assert np.array_equal(d2, geodesic_dilation(marker, mask, None, 2))  # D^(2) = D^(1)[D^(1)]
        e1 = geodesic_erosion(mask, marker, None, 1)
        assert np.all(e1 >= marker) and int(e1.sum()) < int(mask.sum())

    def test_1d_reconstruction_hand_truth(self):
        d = ch4.profile_reconstruction_demo(40)
        f = d["f"].astype(int)
        # reconstruction by dilation of f-h under f: every dome is cut down to (its top - h)
        cap = np.full(f.shape, 255)
        cap[25:100] = 130
        cap[10:12] = 110
        cap[103:196] = 160
        assert np.array_equal(d["rec_dilation"].astype(int), np.minimum(f, cap))
        # reconstruction by erosion of f+h over f: every valley is filled up to (its bottom + h)
        floor = np.full(f.shape, 80)
        floor[100:103] = 100
        floor[140:143] = 130
        assert np.array_equal(d["rec_erosion"].astype(int), np.maximum(f, floor))
        assert d["iterative_equals_dilation"] and d["iterative_equals_erosion"]
        assert d["k_dilation"] > 1 and d["k_erosion"] > 1
        with pytest.raises(ValueError):
            reconstruct_by_erosion(f[None, :], (f + 1)[None, :])
        with pytest.raises(ValueError):
            reconstruct_iterative(f[None, :], f[None, :], None, "bogus")


class TestL1Gradients:
    def test_disc_gradients_ring_widths_and_eq_4_42(self):
        rr, cc = np.mgrid[0:41, 0:41]
        I = (rr - 20) ** 2 + (cc - 20) ** 2 <= 100
        se = strel("disk", 2)  # exact Euclidean r = 2
        gi = morphological_gradient(I, se, "internal")
        ge = morphological_gradient(I, se, "external")
        gb = morphological_gradient(I, se, "basic")
        assert gi.dtype == np.float64 and set(np.unique(gi)) <= {0.0, 1.0}  # logical - logical → double 0/1
        assert np.array_equal(gi + ge, gb)  # Eq. (4.42)
        assert np.all(I[gi > 0]) and not np.any(I[ge > 0])  # internal inside, external outside the object
        dist = np.sqrt((rr - 20) ** 2 + (cc - 20) ** 2)
        assert np.all(dist[gi > 0] > 10 - 2 - 0.01) and np.all(dist[ge > 0] < 10 + 2 + 0.01)  # ring width r
        assert gb[20, 20] == 0 and gb[0, 0] == 0
        u8 = (I * 200).astype(np.uint8) + 20
        for kind in ("internal", "external", "basic"):
            g = morphological_gradient(u8, se, kind)
            assert g.dtype == np.uint8
        assert np.array_equal(morphological_gradient(u8, se, "internal").astype(int)
                              + morphological_gradient(u8, se, "external").astype(int),
                              morphological_gradient(u8, se, "basic").astype(int))
        assert np.array_equal(morphological_gradient(u8, se, "standard"), morphological_gradient(u8, se, "basic"))
        with pytest.raises(ValueError):
            morphological_gradient(u8, se, "sideways")

    def test_matlab_minus_saturation(self):
        A = np.array([[10, 200, 255]], dtype=np.uint8)
        B = np.array([[20, 100, 255]], dtype=np.uint8)
        assert np.array_equal(_matlab_minus(A, B, A.dtype), np.array([[0, 100, 0]], dtype=np.uint8))
        assert _matlab_minus(np.array([True]), np.array([False]), np.dtype(bool)).dtype == np.float64
        assert _matlab_minus(np.array([0.25]), np.array([0.5]), np.dtype(float))[0] == -0.25


class TestL1Edge:
    def test_step_edge_sobel_one_column(self):
        inp_step = np.full((9, 11), 0.2)
        inp_step[:, 5:] = 0.8
        res = edge(inp_step, "sobel", 0.1)
        assert isinstance(res, EdgeResult) and res.bw.dtype == np.bool_
        assert np.array_equal(res.bw.sum(axis=0), [0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0])  # thinning keeps one column
        no = edge(inp_step, "sobel", 0.1, thinning=False)
        assert np.array_equal(no.bw.sum(axis=0), [0, 0, 0, 0, 9, 9, 0, 0, 0, 0, 0])  # both tie columns without it
        assert np.array_equal(no.bw, res.gv ** 2 + res.gh ** 2 > 0.01)
        assert res.gv[0, 4] == pytest.approx(-0.3) and res.gv[0, 5] == pytest.approx(-0.3)  # (0.2-0.8)*4/8
        assert np.abs(res.gh).max() < 1e-15  # constant rows: rounding noise only (MATLAB shows the same ~1e-17)
        assert res.thresh == 0.1
        # direction: a vertical step has no horizontal edges
        assert edge(inp_step, "sobel", 0.1, "horizontal").bw.sum() == 0
        # 'vertical' (kx=1, ky=0): the up/down thinning test is always allowed (by >= 0*bx - eps), so on the
        # zero-padded bottom row the tie column 4 also survives -> 10 px (MATLAB identical, TestL2Edge Step/vertical)
        v = edge(inp_step, "sobel", 0.1, "vertical").bw
        assert v[:, 5].all() and v.sum() == 10 and v[8, 4]
        # automatic threshold = sqrt(4 mean(b))
        auto = edge(inp_step, "sobel")
        assert auto.thresh == pytest.approx(np.sqrt(4 * np.mean(auto.gv ** 2 + auto.gh ** 2)))
        # Prewitt: (0.2-0.8)*3/6 = -0.3 as well
        assert edge(inp_step, "prewitt", 0.1).gv[0, 4] == pytest.approx(-0.3)
        # Roberts runs (2x2 masks), edge lands on the step too
        rob = edge(inp_step, "roberts", 0.1)
        assert rob.bw.sum() > 0 and rob.bw[:, 7:].sum() == 0 and rob.bw[:, :3].sum() == 0

    def test_book_kernels_vs_edge_gradients(self):
        rng = np.random.default_rng(2)
        f = rng.random((12, 15))
        Gx, Gy = ch4.gradient_operator(f, "sobel")
        bx, by, b = gradient_sobel_prewitt(f, "sobel")
        assert np.allclose(Gx, -8 * by, atol=1e-14) and np.allclose(Gy, -8 * bx, atol=1e-14)
        Gx, Gy = ch4.gradient_operator(f, "prewitt")
        bx, by, _ = gradient_sobel_prewitt(f, "prewitt")
        assert np.allclose(Gx, -6 * by, atol=1e-14) and np.allclose(Gy, -6 * bx, atol=1e-14)
        Gx, Gy = ch4.gradient_operator(f, "forward")
        assert np.allclose(Gx[:-1, :], f[1:, :] - f[:-1, :]) and np.allclose(Gy[:, :-1], f[:, 1:] - f[:, :-1])  # Eq. (4.6)
        mag = ch4.gradient_magnitude(Gx, Gy)
        assert np.allclose(mag, np.sqrt(Gx ** 2 + Gy ** 2))
        assert np.allclose(ch4.gradient_magnitude(Gx, Gy, "squared"), Gx ** 2 + Gy ** 2)
        assert np.allclose(ch4.gradient_magnitude(Gx, Gy, "l1"), np.abs(Gx) + np.abs(Gy))
        assert np.all(ch4.gradient_magnitude(Gx, Gy, "l1") >= mag - 1e-15)  # |Gx|+|Gy| >= sqrt(Gx²+Gy²)
        th = ch4.gradient_direction(np.array([1.0]), np.array([0.0]))
        assert th[0] == pytest.approx(np.pi / 2) and ch4.gradient_direction(np.array([1.0]), np.array([0.0]), "math")[0] == 0
        assert np.array_equal(ch4.threshold_gradient(mag, 0.5), mag > 0.5)
        with pytest.raises(ValueError):
            ch4.gradient_magnitude(Gx, Gy, "l3")
        with pytest.raises(ValueError):
            ch4.gradient_direction(Gx, Gy, "polar")
        with pytest.raises(KeyError):
            ch4.gradient_operator(f, "scharr")

    def test_thin_gradient_rule_ties(self):
        # 1-px ridge: kept; 2-px plateau ridge: the pixel whose right neighbour is smaller survives
        b = np.array([[0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]])
        bx = np.ones_like(b)
        by = np.zeros_like(b)
        e = thin_gradient(b, bx, by, 1.0, 1.0, (0, 0, 0, 0), 0.5)
        assert np.array_equal(e[0], [0, 1, 0, 0, 0, 1, 0])
        # dominant direction: with |bx| == |by| both branches may fire (eps tolerance)
        b2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        assert thin_gradient(b2, np.ones((3, 3)), np.ones((3, 3)), 1, 1, cutoff=0.5)[1, 1]
        assert not thin_gradient(b2, np.ones((3, 3)), np.ones((3, 3)), 1, 1, cutoff=2.0).any()

    def test_log_step_and_zero_crossing_rule(self):
        inp_step = np.full((9, 11), 0.2)
        inp_step[:, 5:] = 0.8
        res = edge(inp_step, "log", 0.001, sigma=1.0)
        assert res.gv is None and res.gh is None
        # the negative side of the crossing at the step (col 5, bright side: the LoG response is negative there),
        # interior rows only (MATLAB: rr = 2:m-1); col 7 may carry a spurious crossing against the +2.7e-17 rounding
        # noise of the flat region (see TestL2Edge.test_log_step_noise_sensitivity) and is excluded here
        cols = res.bw.sum(axis=0)
        assert cols[5] == 7 and cols[[0, 1, 2, 3, 4, 6, 8, 9, 10]].sum() == 0
        assert res.bw[0].sum() == 0 and res.bw[-1].sum() == 0  # MATLAB: rr = 2:m-1 only
        auto = edge(inp_step, "log")
        assert auto.thresh == pytest.approx(0.75 * np.mean(np.abs(imfilter(inp_step, fspecial("log", 13, 2) - fspecial("log", 13, 2).mean(), "replicate"))))
        # exact-zero case: opposite signs across a zero pixel, jump > 2T
        b = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        assert log_zero_crossings(b, 0.5)[1, 1] and not log_zero_crossings(b, 1.0)[1, 1]
        bt = b.T.copy()
        assert log_zero_crossings(bt, 0.5)[1, 1]
        # [- +] pattern flags the negative pixel only
        b3 = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        e3 = log_zero_crossings(b3, 0.5)
        assert e3[1, 1] and not e3[1, 2]
        assert not log_zero_crossings(np.ones((2, 2)), 0).any()
        # zerocross with a user kernel: plateaus of an integer image give exact zeros inside
        z = np.zeros((10, 12))
        z[3:7, 3:9] = 5.0
        zc = edge(z, "zerocross", 0.5, H=ch4.LAPLACIAN_KERNELS[4])
        assert zc.bw[4:6, 4:8].sum() == 0 and zc.bw.sum() > 0
        assert zc.thresh == 0.5

    def test_edge_argument_errors(self):
        a = np.zeros((5, 5))
        with pytest.raises(NotImplementedError):
            edge(a, "canny")
        with pytest.raises(TypeError):
            edge(np.zeros((5, 5), dtype=np.uint8), "sobel")
        with pytest.raises(ValueError):
            edge(a, "hough")
        with pytest.raises(ValueError):
            edge(a, "sobel", 0.1, "diagonal")
        with pytest.raises(ValueError):
            edge(np.zeros((5, 5, 3)), "sobel")
        with pytest.raises(ValueError):
            gradient_sobel_prewitt(a, "roberts")
        bx, by, b = gradient_roberts(a)
        assert b.shape == (5, 5)


class TestL1Chapter:
    def test_laplacian_on_quadratic_surface(self):
        y, x = np.mgrid[0:9, 0:9].astype(float)
        f = x * x + y * y
        l4 = ch4.laplacian(f, 4)
        l8 = ch4.laplacian(f, 8)
        assert np.allclose(l4[1:-1, 1:-1], 4.0) and np.allclose(l8[1:-1, 1:-1], 12.0)
        d2 = ch4.second_difference_forward(f, 0)
        assert np.allclose(d2[:-2, :], 2.0)  # Eq. (4.10) on y²
        d2y = ch4.second_difference_forward(f, 1)
        assert np.allclose(d2y[:, :-2], 2.0)
        # zero-crossing rule of p. 64 on a step: edges hug the step on both sides, nothing elsewhere
        s = np.zeros((7, 12))
        s[:, 6:] = 1.0
        zc = ch4.laplacian_zero_crossings(ch4.laplacian(s, 4), 0.5)
        assert zc[:, 5:7].all() and not zc[:, :4].any() and not zc[:, 8:].any()

    def test_bwareaopen(self):
        bw = np.zeros((10, 10), dtype=bool)
        bw[0:5, 0:5] = True  # 25 px
        bw[8, 8] = bw[9, 9] = bw[7, 7] = True  # 3 px diagonal chain (8-connected)
        assert int(ch4.bwareaopen(bw, 10).sum()) == 25
        assert int(ch4.bwareaopen(bw, 3).sum()) == 28
        assert int(ch4.bwareaopen(bw, 3, conn=4).sum()) == 25  # the chain splits into 1-px objects with 4-conn
        assert int(ch4.bwareaopen(bw, 26).sum()) == 0

    def test_sobel_edges_script_on_synthetic_gray(self):
        g = np.full((30, 40), 50, dtype=np.uint8)
        g[:, 20:] = 200
        res = ch4.sobel_edges_script(g, 0.05)
        assert np.array_equal(res["im"], g / 256.0)  # literal /256, not im2double
        assert res["bw"].sum() == 30 and res["n_edge"] == 30 and res["thresh"] == 0.05
        assert res["r"] == 30 and res["c"] == 40 and res["msk"].shape == (5, 5) and res["msk"].sum() == 9
        assert res["method"] == "sobel"
        res2 = ch4.sobel_edges_script(g, None, smooth=True, median=True, min_area=5)
        assert res2["bw_smooth"].shape == (34, 44) and res2["bw_smooth"].max() > 1  # conv2 'full', not binary
        assert res2["thresh"] > 0
        rgb = np.stack([g, g, g], axis=-1)
        assert np.array_equal(ch4.sobel_edges_script(rgb, 0.05)["bw"], res["bw"])
        assert ch4.experiment_sobel_thresholds(g, (0.05, 0.03))[0.03]["n_edge"] == 30

    def test_morphological_edges_on_synthetic(self):
        rr, cc = np.mgrid[0:60, 0:80]
        g = np.where((rr - 30) ** 2 + (cc - 40) ** 2 <= 400, 210, 30).astype(np.uint8)
        res = ch4.morphological_edges(g, radius=3)
        assert set(res) >= {"im", "SE", "level", "threshold", "ic", "I", "J", "K", "BW1", "BW2", "BW", "X", "Y",
                            "internal", "external", "basic"}
        assert res["SE"].shape == (5, 5) and res["I"].dtype == np.bool_ and res["BW1"].dtype == np.float64
        assert res["X"].dtype == np.uint8 and res["internal"].dtype == np.uint8
        assert np.all(res["J"] <= res["I"]) and np.all(res["I"] <= res["K"])
        chk = ch4.gradient_identity_check(res)
        assert chk["binary"] and chk["gray"]
        assert res["ic"] == pytest.approx(res["I"].mean())  # fraction, as core.threshold.ice_concentration
        assert 30 < res["threshold"] < 210
        res2 = ch4.morphological_edges(g, se=strel("disk", 3))
        assert np.array_equal(res2["BW"], res["BW"])
        ex = ch4.experiment_internal_gradient(g, (2, 4))
        assert ex[4]["n_edge"] > ex[2]["n_edge"] and ex[2]["SE"].shape == (5, 5)

    def test_crop_constants_and_book_params(self):
        assert ch4.FIG_4_3A_CROP == (slice(1599, 2151), slice(1978, 2552))
        assert ch4.FIG_4_3A_CROP_MATLAB == ((1600, 2151), (1979, 2552))
        big = np.zeros((2856, 4290), dtype=np.uint8)
        assert ch4.fig_4_3a(big).shape == (552, 574)
        with pytest.raises(ValueError):
            ch4.fig_4_3a(np.zeros((100, 100)))
        assert ch4.BOOK_PARAMS == {"sobel_T": 0.05, "sobel_T_low": 0.03, "log_sigma": 2.0, "log_T": 0.005,
                                   "disk_r_figs": 15, "disk_r_script": 7, "disk_r_thin": 5, "disk_r_thick": 16}
        assert 2 * int(np.ceil(3 * ch4.BOOK_PARAMS["log_sigma"])) + 1 == 13  # "13x13 LoG kernel" (p. 66)


# ================================================================================================================
# L2 — parity vs MATLAB R2025a on the controlled fixtures (compat.mat)
# ================================================================================================================
@needs_ref("compat")
class TestL2Fspecial:
    @pytest.mark.parametrize("key, args", [
        ("fs_sobel", ("sobel",)), ("fs_prewitt", ("prewitt",)),
        ("fs_lap0", ("laplacian", 0)), ("fs_lap02", ("laplacian", 0.2)), ("fs_lap1", ("laplacian", 1)),
        ("fs_lapd", ("laplacian",)),
        ("fs_g51", ("gaussian", 5, 1)), ("fs_g305", ("gaussian", 3, 0.5)), ("fs_g132", ("gaussian", 13, 2)),
        ("fs_g57", ("gaussian", [5, 7], 1.3)), ("fs_gd", ("gaussian",)), ("fs_gs", ("gaussian", None, 1.5)),
        ("fs_l51", ("log", 5, 1)), ("fs_l132", ("log", 13, 2)), ("fs_l505", ("log", 5, 0.5)), ("fs_ld", ("log",)),
        ("fs_l79", ("log", [7, 9], 1.2)),
        ("fs_avg", ("average",)), ("fs_avg35", ("average", [3, 5])), ("fs_avg7", ("average", 7)),
        ("fs_disk5", ("disk", 5)), ("fs_disk25", ("disk", 2.5)), ("fs_diskd", ("disk",)), ("fs_disk1", ("disk", 1)),
        ("fs_unsharp", ("unsharp",)), ("fs_unsharp05", ("unsharp", 0.5)),
    ])
    def test_fspecial(self, key, args):
        d = ref("compat")
        m = assert_parity(fspecial(*args), d[key], "float", atol=1e-15, rtol=0, name=key)
        assert m["max_abs_err"] <= 1e-15


@needs_ref("compat")
class TestL2Strel:
    @pytest.mark.parametrize("r", list(range(1, 21)))
    @pytest.mark.parametrize("n", [0, 4, 6, 8])
    def test_disk_nhoods(self, r, n):
        d = ref("compat")
        assert np.array_equal(strel("disk", r, n), d[f"sd_r{r}_n{n}"].astype(bool)), (r, n)

    def test_prefix_and_default_n(self):
        d = ref("compat")
        assert np.array_equal(strel("dis", 7), d["sdis7"].astype(bool))
        assert np.array_equal(strel("disk", 7), d["sd_default7"].astype(bool))
        assert np.array_equal(d["sd_default7"], d["sd_r7_n4"])  # MATLAB default N = 4
        assert np.array_equal(strel("DISK", 7), d["sdi_s7"].astype(bool))
        assert sc(d["amb_err"]) == 1  # MATLAB errors on the ambiguous 'di' — so does the port
        with pytest.raises(ValueError):
            strel("di", 7)
        assert int(d["sd_r7_n4"].sum()) == 157 and d["sd_r7_n4"].shape == (13, 13)

    @pytest.mark.parametrize("M", list(range(1, 11)))
    def test_diamond(self, M):
        assert np.array_equal(strel("diamond", M), ref("compat")[f"sdia_{M}"].astype(bool))

    def test_square_rectangle_octagon_periodicline_pair_arbitrary(self):
        d = ref("compat")
        assert np.array_equal(strel("square", 3), d["ssq_3"].astype(bool))
        assert np.array_equal(strel("square", 4), d["ssq_4"].astype(bool))
        assert np.array_equal(strel("rectangle", [3, 5]), d["srect_3_5"].astype(bool))
        for M in (0, 3, 6, 9):
            assert np.array_equal(strel("octagon", M), d[f"soct_{M}"].astype(bool)), M
        assert np.array_equal(strel("periodicline", 2, [1, 2]), d["spl_2_1_2"].astype(bool))
        assert np.array_equal(strel("periodicline", 3, [-1, 2]), d["spl_3_m1_2"].astype(bool))
        assert np.array_equal(strel("periodicline", 1, [1, 0]), d["spl_1_1_0"].astype(bool))
        assert np.array_equal(strel("pair", [2, -1]), d["spair_2_m1"].astype(bool))
        assert np.array_equal(strel("pair", [-1, 3]), d["spair_m1_3"].astype(bool))
        assert np.array_equal(strel(inputs()["AsymSE"]), d["sarb"].astype(bool))
        assert np.array_equal(strel("arbitrary", inputs()["EvenSE"]), d["sarb2"].astype(bool))

    @pytest.mark.parametrize("ln, deg", [(7, 0), (7, 45), (7, 90), (9, 30), (6, 120), (5, 0), (3, 0), (8, 60),
                                         (10, 135), (7, 180), (7, 200)])
    def test_line(self, ln, deg):
        assert np.array_equal(strel("line", ln, deg), ref("compat")[f"sline_{ln}_{deg}"].astype(bool)), (ln, deg)

    @pytest.mark.parametrize("r, n", [(3, 4), (5, 4), (7, 4), (10, 4), (15, 4), (16, 4), (20, 4), (7, 6), (7, 8),
                                      (12, 6), (12, 8)])
    def test_decomposition_sequence(self, r, n):
        d = ref("compat")
        seq = disk_decomposition(r, n)
        assert len(seq) == int(sc(d[f"ndec_r{r}_n{n}"]))
        for k, s in enumerate(seq, start=1):
            m = d[f"dec_r{r}_n{n}_{k}"]
            assert np.array_equal(s, m.astype(bool)), (r, n, k, s.shape, m.shape)  # same element, same order


@needs_ref("compat")
class TestL2ErodeDilate:
    @pytest.mark.parametrize("img", ["Bw", "BwB", "Rnd8", "Rnd", "Neg", "I16"])
    @pytest.mark.parametrize("se_name", ["disk7", "asym", "even", "line5", "line7_45", "sq3", "dia3", "disk2", "pair"])
    def test_erode_dilate_open_close(self, img, se_name):
        d = ref("compat")
        I = inputs()[img]
        se = se_by_name(se_name)
        for op, fn in (("er", imerode), ("di", imdilate), ("op", imopen)):
            _assert_same(fn(I, se), d[f"{op}_{img}_{se_name}"], I.dtype, f"{op}_{img}_{se_name}")

    @pytest.mark.parametrize("img", ["Bw", "BwB", "Rnd8", "Rnd", "Neg", "I16"])
    @pytest.mark.parametrize("se_name", ["disk7", "asym", "even", "line5", "line7_45", "sq3", "dia3", "disk2", "pair"])
    def test_imclose(self, img, se_name):
        """MATLAB ``imclose`` = ``imerode(imdilate(padarray(A, ceil(size(nhood)/2), 'both'), se), se)`` cropped
        (R2025a ``imclose.m`` pad 0, or the Halide kernel's class-minimum border); the port reproduces both rules."""
        d = ref("compat")
        I = inputs()[img]
        se = se_by_name(se_name)
        _assert_same(imclose(I, se), d[f"cl_{img}_{se_name}"], I.dtype, f"cl_{img}_{se_name}")

    @pytest.mark.parametrize("img", ["Bw", "BwB", "Rnd8", "Rnd", "Neg", "I16"])
    @pytest.mark.parametrize("se_name", ["disk7", "asym", "even", "line5", "line7_45", "sq3", "dia3", "disk2", "pair"])
    def test_imclose_matlab_padding_hypothesis(self, img, se_name):
        """Proof of the cause: pre-padding by ceil(size(nhood)/2) on top of the port's imdilate/imerode reproduces
        MATLAB's closing exactly on every fixture.  Pad value: the dilation border value (0 / false / intmin / -Inf,
        = imclose.m's ``padarray(..., 0)`` for uint8, logical and non-negative double); MATLAB's 3x3-square path
        zero-pads signed images for all-ones rectangular SEs (observed: Neg/I16 with sq3 and line5 match pad 0, not
        pad -Inf/intmin; every other SE matches pad -Inf/intmin)."""
        d = ref("compat")
        I = inputs()[img]
        se = se_by_name(se_name)
        # observed: MATLAB zero-pads signed images for all-ones rectangular SEs (sq3, line5) and pads with
        # intmin/-Inf otherwise; for uint8 / logical / non-negative double the two coincide (0 = the minimum)
        pad = 0 if se.all() else _border_values(I.dtype)[1]
        _assert_same(_imclose_matlab_padding(I, se, pad), d[f"cl_{img}_{se_name}"], I.dtype, f"cl_pad_{img}_{se_name}")

    def test_imclose_padding_reference_identity(self):
        """MATLAB's own ``imclose`` equals the half-SE-padded ``erode(dilate(.))`` and differs from the unpadded
        composition; the port equals MATLAB (and therefore the padded form), not the unpadded Eq. (4.22) composition."""
        d = ref("compat")
        se = strel("disk", 7)
        assert np.array_equal(d["cl_pad_Rnd8_disk7"], d["cl_Rnd8_disk7"])  # MATLAB: imclose == padded erode(dilate)
        assert not np.array_equal(d["cl_nopad_Rnd8_disk7"], d["cl_Rnd8_disk7"])  # and != the unpadded composition
        out = imclose(inputs()["Rnd8"], se)
        assert np.array_equal(out, d["cl_Rnd8_disk7"])  # the port == MATLAB imclose
        assert np.array_equal(out, d["cl_pad_Rnd8_disk7"])  # == the padded form
        assert not np.array_equal(out, d["cl_nopad_Rnd8_disk7"])  # != the unpadded composition (the old defect)
        assert np.array_equal(imdilate(imerode(inputs()["Rnd8"], se), se), d["op_Rnd8_disk7"])  # imopen: no padding


PAD_SES = {"dia7": lambda: strel("diamond", 7), "disk8": lambda: strel("disk", 8), "dia8": lambda: strel("diamond", 8),
           "disk10": lambda: strel("disk", 10), "line31_45": lambda: strel("line", 31, 45),
           "rect3_17": lambda: strel("rectangle", [3, 17])}


@needs_ref("imclose_pad")
class TestL2ImcloseBorderRule:
    """R2025a ``imclose`` takes two paths (``morphop_fast.m``): SEs with nnz < 600, every side <= 15 and not an all-ones
    rectangle go to the Halide kernel (border = class minimum, -Inf/intmin); all others fall through to ``imclose.m``
    (``padarray(A, ceil(size(nhood)/2), 'both')`` = zeros).  The two differ only on signed images with negative
    values, which ``compat.mat`` (SEs <= 15 px) covered for the Halide path only; ``imclose_pad.mat`` adds the
    side > 15 / all-ones cases on ``Neg`` (double in [-0.5, 0.5)) and ``I16``."""

    @pytest.mark.parametrize("img", ["Neg", "I16", "Rnd8"])
    @pytest.mark.parametrize("se_name", list(PAD_SES))
    def test_imclose_signed_large_se(self, img, se_name):
        d = ref("imclose_pad")
        I = inputs()[img]
        se = PAD_SES[se_name]()
        _assert_same(imclose(I, se), d[f"cl_{img}_{se_name}"], I.dtype, f"cl_{img}_{se_name}")

    @pytest.mark.parametrize("se_name", ["disk10"])
    def test_decomposed_se_list(self, se_name):
        d = ref("imclose_pad")
        I = inputs()["Neg"]
        out = imclose(I, disk_decomposition(10))  # decomposed list: padded by half the Minkowski-sum nhood
        _assert_same(out, d["cl_Neg_disk10"], I.dtype, "cl_Neg_disk10 (decomposed)")
        assert np.array_equal(out, imclose(I, strel("disk", 10)))

    def test_which_pad_value_matlab_used(self):
        """Proof of the routing rule on the signed double image: side > 15 -> imclose.m zero pad (dia8, disk10);
        side <= 15 -> Halide class-minimum pad (dia7).  Both differ from the unpadded composition."""
        d = ref("imclose_pad")
        assert sc(d["halide_on"]) == 1.0  # the reference was made with the default images.UseHalide setting
        for se_name in ("dia8", "disk10"):
            assert np.array_equal(d[f"cl_Neg_{se_name}"], d[f"cl0_Neg_{se_name}"]), se_name  # == pad 0
            assert not np.array_equal(d[f"cl_Neg_{se_name}"], d[f"clm_Neg_{se_name}"]), se_name  # != pad -Inf
            assert not np.array_equal(d[f"cl_Neg_{se_name}"], d[f"cln_Neg_{se_name}"]), se_name  # != unpadded
        assert np.array_equal(d["cl_Neg_dia7"], d["clm_Neg_dia7"])  # Halide path == pad -Inf
        assert not np.array_equal(d["cl_Neg_dia7"], d["cl0_Neg_dia7"])  # != pad 0
        assert not np.array_equal(d["cl_Neg_dia7"], d["cln_Neg_dia7"])  # != unpadded

    def test_decomposed_path_vs_matlab_and_nhood_form(self):
        d = ref("compat")
        inp = inputs()
        assert np.array_equal(imerode(inp["Bw"], disk_decomposition(7)), d["er_Bw_disk7"].astype(bool))
        assert np.array_equal(imdilate(inp["Rnd8"], disk_decomposition(7)), d["di_Rnd8_disk7"])
        assert np.array_equal(d["er_Bw_disk7nh"], d["er_Bw_disk7"])  # MATLAB: decomposed strel == getnhood
        assert np.array_equal(d["di_Rnd8_disk7nh"], d["di_Rnd8_disk7"])

    def test_fig_4_8_and_profile_vs_matlab(self):
        d = ref("compat")
        assert np.array_equal(d["f48_er"].astype(bool), synth.FIG_4_8_ERODED)  # MATLAB reproduces the printed (d)
        assert np.array_equal(d["f48_di"].astype(bool), synth.FIG_4_8_DILATED)  # and (f)
        demo = ch4.profile_open_close_demo(9)
        for key, mkey in (("opening", "po"), ("closing", "pc"), ("erosion", "pe"), ("dilation", "pd")):
            assert np.array_equal(demo[key], d[mkey].ravel()), key


@needs_ref("compat")
class TestL2Reconstruction:
    def test_binary_and_grayscale(self):
        d = ref("compat")
        inp = inputs()
        assert np.array_equal(imreconstruct(inp["Mk"], inp["Bw"]), d["rc_bw8"].astype(bool))
        assert np.array_equal(imreconstruct(inp["Mk"], inp["Bw"], 4), d["rc_bw4"].astype(bool))
        assert np.array_equal(imreconstruct(inp["Mk8"], inp["Rnd8"]), d["rc_g8"])
        assert np.array_equal(imreconstruct(inp["Mk8"], inp["Rnd8"], 4), d["rc_g4"])
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        assert np.array_equal(imreconstruct(inp["Mk8"], inp["Rnd8"], cross), d["rc_gc"])
        assert np.array_equal(d["rc_g4"], d["rc_gc"])
        it, _ = reconstruct_iterative(inp["Mk8"], inp["Rnd8"], np.ones((3, 3), bool), "dilation")
        assert np.array_equal(it, d["rc_g8"])
        assert np.array_equal(imreconstruct(np.maximum(inp["Rnd"] - 0.3, 0), inp["Rnd"]), d["rc_dbl"])

    def test_1d_profile_vs_matlab(self):
        d = ref("compat")
        demo = ch4.profile_reconstruction_demo(40)
        assert np.array_equal(demo["marker_dilation"], d["ProfMkD"].ravel())
        assert np.array_equal(demo["rec_dilation"], d["rc_prof_d"].ravel())
        assert np.array_equal(demo["marker_erosion"], inputs()["ProfMkE"].ravel())
        assert np.array_equal(demo["rec_erosion"], d["rc_prof_e"].ravel())  # imcomplement(imreconstruct(...))


@needs_ref("compat")
class TestL2Gradients:
    def test_gradients_and_minus(self):
        d = ref("compat")
        inp = inputs()
        se = strel("disk", 7)
        assert ref_str(d, "cls_mg_Bw") == "double"
        for kind, key in (("basic", "mg_b_Bw"), ("internal", "mg_i_Bw"), ("external", "mg_e_Bw")):
            g = morphological_gradient(inp["Bw"], se, kind)
            assert g.dtype == np.float64 and np.array_equal(g, d[key].astype(np.float64)), key
        for kind, key in (("basic", "mg_b_Rnd8"), ("internal", "mg_i_Rnd8"), ("external", "mg_e_Rnd8")):
            g = morphological_gradient(inp["Rnd8"], se, kind)
            assert g.dtype == np.uint8 and np.array_equal(g, d[key]), key
        assert np.array_equal(morphological_gradient(inp["Neg"], se, "basic"), d["mg_b_Neg"])
        gi = morphological_gradient(inp["I16"], se, "internal")
        assert gi.dtype == np.int16 and np.array_equal(gi, d["mg_i_I16"])
        assert np.array_equal(_matlab_minus(inp["A8"], inp["B8"], np.dtype(np.uint8)), d["sat_ab"])
        assert np.array_equal(_matlab_minus(inp["B8"], inp["A8"], np.dtype(np.uint8)), d["sat_ba"])


_EDGE_CASES = []
for _img, _T in (("Rnd", 0.2), ("Ramp", 0.1), ("RampD", 0.1), ("Step", 0.1), ("StepH", 0.1), ("Diag", 0.1),
                 ("Cst", 0.1)):
    for _m in ("sobel", "prewitt", "roberts"):
        for _d in (("both", "horizontal", "vertical") if _m != "roberts" else ("both",)):
            for _thin in ("thin", "nothin"):
                _EDGE_CASES.append((_img, _T, _m, _d, _thin))


@needs_ref("compat")
class TestL2Edge:
    @pytest.mark.parametrize("img, T, m, d, thin", _EDGE_CASES)
    def test_gradient_methods(self, img, T, m, d, thin):
        r = ref("compat")
        I = inputs()[img]
        base = f"e_{m}_{img}_{d}_{thin}"
        given = edge(I, m, T, d, thinning=thin == "thin")
        assert not np.any(given.bw != r[f"{base}_T"].astype(bool)), f"{base}_T: {int((given.bw != r[f'{base}_T'].astype(bool)).sum())} px differ"
        auto = edge(I, m, None, d, thinning=thin == "thin")
        t_ml = sc(r[f"t_{m}_{img}_{d}_{thin}_auto"])
        if t_ml < 1e-15:
            # b is floating-point noise (~1e-34: constant image, or a direction with no gradient component);
            # thresholds ~1e-17 / 0 on the two sides, maps all-zero
            assert auto.thresh < 1e-15 and int(auto.bw.sum()) == 0 and int(r[f"{base}_auto"].sum()) == 0
        else:
            assert auto.thresh == pytest.approx(t_ml, rel=1e-12, abs=0), (auto.thresh, t_ml)
        assert not np.any(auto.bw != r[f"{base}_auto"].astype(bool)), f"{base}_auto: {int((auto.bw != r[f'{base}_auto'].astype(bool)).sum())} px differ"

    @pytest.mark.parametrize("img", ["Rnd", "Ramp", "RampD", "Step", "StepH", "Diag", "Cst"])
    def test_gv_gh(self, img):
        r = ref("compat")
        I = inputs()[img]
        for m in ("sobel", "prewitt", "roberts"):
            res = edge(I, m)
            for out, key in ((res.gv, f"gv_{m}_{img}"), (res.gh, f"gh_{m}_{img}")):
                err = np.abs(out - r[key]).max()
                assert err <= 2e-16, (key, err)  # 1 ulp: the builtin filters then divides (measured 1.1e-16 on Rnd)

    def test_log_and_zerocross(self):
        r = ref("compat")
        inp = inputs()
        res = edge(inp["Rnd"], "log")
        assert res.thresh == pytest.approx(sc(r["t_log_Rnd_auto"]), rel=1e-12)
        assert not np.any(res.bw != r["e_log_Rnd_auto"].astype(bool))
        assert not np.any(edge(inp["Rnd"], "log", 0.02).bw != r["e_log_Rnd_002"].astype(bool))
        res = edge(inp["Rnd"], "log", None, sigma=1.5)
        assert res.thresh == pytest.approx(sc(r["t_log_Rnd_s15"]), rel=1e-12)
        assert not np.any(res.bw != r["e_log_Rnd_s15"].astype(bool))
        assert not np.any(edge(inp["Rnd"], "log", 0.01, sigma=3).bw != r["e_log_Rnd_s3"].astype(bool))
        res = edge(inp["Ramp"], "log")
        assert res.thresh == pytest.approx(sc(r["t_log_Ramp"]), rel=1e-12)
        assert not np.any(res.bw != r["e_log_Ramp"].astype(bool))
        assert not np.any(edge(inp["Rnd8"].astype(np.float64) / 256, "log", 0.005, sigma=2).bw != r["e_log_Rnd8d"].astype(bool))
        H4, H8 = inp["H4"], inp["H8"]
        assert not np.any(edge(inp["Zc"], "zerocross", 0.5, H=H4).bw != r["e_zc_Zc_H4"].astype(bool))
        assert not np.any(edge(inp["Zc"], "zerocross", 0, H=H4).bw != r["e_zc_Zc_H4_0"].astype(bool))
        res = edge(inp["Zc"], "zerocross", None, H=H8)
        assert res.thresh == pytest.approx(sc(r["t_zc_Zc_H8"]), rel=1e-12)
        assert not np.any(res.bw != r["e_zc_Zc_H8"].astype(bool))
        assert not np.any(edge(inp["Rnd"], "zerocross", 0.05, H=fspecial("log", 7, 1)).bw != r["e_zc_Rnd_log7"].astype(bool))
        assert not np.any(edge(inp["Rnd8"].astype(np.float64), "zerocross", 3, H=H4).bw != r["e_zc_Rnd8d_H4"].astype(bool))

    def test_log_step_noise_sensitivity(self):
        """``edge(Step, 'log', 0.001, 1)``: MATLAB flags col 5 only, the port cols 5 and 7.  Cause: in the flat region
        right of the step the LoG response is rounding noise (Python +2.7e-17, MATLAB <= 0, ``b_log_Step``), and
        the strict ``b(r,c+1) > 0`` test of the ``[- +]`` pattern turns a +noise neighbour into a crossing.  The
        maps agree wherever no neighbour of the pixel is below 1e-12 in magnitude."""
        r = ref("compat")
        inp = inputs()
        res = edge(inp["Step"], "log", 0.001, sigma=1)
        ml = r["e_log_Step"].astype(bool)
        b_ml = r["b_log_Step"]
        op = fspecial("log", 7, 1)
        b_py = imfilter(inp["Step"], op - op.mean(), "replicate")
        assert np.abs(b_py - b_ml).max() < 1e-12  # the responses agree to rounding
        flat = np.abs(b_ml) < 1e-12
        assert flat[:, 8:].all() and flat[:, :2].all()
        assert (b_py[:, 8:] > 0).all() and (b_ml[:, 8:] <= 0).all()  # opposite noise signs -> the extra column
        noisy_nb = ndimage.binary_dilation(flat, np.ones((3, 3), bool))
        assert not np.any((res.bw != ml) & ~noisy_nb)  # identical away from the noise
        diff_cols = np.nonzero(res.bw != ml)[1]
        assert diff_cols.size == 7 and np.all(diff_cols == 7)

    def test_log_constant_image_is_floating_noise(self):
        """MATLAB flags 14 'zero crossings' of a constant image at T = 4.7e-18 (its LoG kernel sums to ~1e-17, not 0);
        the port's kernel noise differs (T = 1.6e-17, 0 px).  Not a comparable quantity — recorded, not asserted."""
        r = ref("compat")
        res = edge(inputs()["Cst"], "log")
        assert res.thresh < 1e-15 and sc(r["t_log_Cst"]) < 1e-15
        assert int(r["e_log_Cst"].sum()) <= 20 and int(res.bw.sum()) <= 20


@needs_ref("compat")
class TestL2HandCodedEquations:
    def test_gradient_kernels_eqs_4_6_4_8(self):
        r = ref("compat")
        f = inputs()["Rnd"]
        Gx, Gy = ch4.gradient_operator(f, "sobel")
        assert np.abs(Gx - r["Gx_s"]).max() <= 1e-15 and np.abs(Gy - r["Gy_s"]).max() <= 1e-15
        Gx, Gy = ch4.gradient_operator(f, "prewitt")
        assert np.abs(Gx - r["Gx_p"]).max() <= 1e-15 and np.abs(Gy - r["Gy_p"]).max() <= 1e-15
        Gx, Gy = ch4.gradient_operator(f, "forward")
        assert np.abs(Gx - r["Gx_f"]).max() <= 1e-15 and np.abs(Gy - r["Gy_f"]).max() <= 1e-15
        Gx0, _ = ch4.gradient_operator(f, "sobel", "zeros")
        assert np.abs(Gx0 - r["Gx_s0"]).max() <= 1e-15
        Gx, Gy = r["Gx_s"], r["Gy_s"]
        assert np.abs(ch4.gradient_magnitude(Gx, Gy) - r["mag_l2"]).max() <= 1e-15
        assert np.abs(ch4.gradient_magnitude(Gx, Gy, "squared") - r["mag_sq"]).max() <= 1e-15
        assert np.abs(ch4.gradient_magnitude(Gx, Gy, "l1") - r["mag_l1"]).max() <= 1e-15
        assert np.abs(ch4.gradient_direction(Gx, Gy, "book") - r["dir_book"]).max() <= 1e-15
        assert np.abs(ch4.gradient_direction(Gx, Gy, "math") - r["dir_math"]).max() <= 1e-15
        assert np.array_equal(ch4.threshold_gradient(r["mag_l2"], 0.5), r["thr_mag"].astype(bool))

    def test_laplacian_gaussian_log_eqs_4_9_4_15(self):
        r = ref("compat")
        f = inputs()["Rnd"]
        assert np.abs(ch4.laplacian(f, 4) - r["Lap4"]).max() <= 1e-15
        assert np.abs(ch4.laplacian(f, 8) - r["Lap8"]).max() <= 1e-14  # 9-term sums, values ~3: measured 2.7e-15
        assert np.abs(ch4.second_difference_forward(f, 0) - r["d2x"]).max() <= 1e-15
        assert np.abs(ch4.second_difference_forward(f, 1) - r["d2y"]).max() <= 1e-15
        assert np.abs(ch4.gaussian_kernel(5, 1, normalize=False) - r["Gau_an"]).max() <= 1e-16
        assert np.abs(ch4.gaussian_kernel(5, 1) - r["Gau_n"]).max() <= 1e-16
        assert np.abs(ch4.log_kernel(5, 1, "analytic") - r["LoG_an"]).max() <= 1e-16
        assert np.abs(ch4.log_kernel(5, 1, "gauss_conv_laplacian") - r["gauss_conv_lap"]).max() <= 1e-16
        assert np.array_equal(ch4.laplacian_zero_crossings(r["Lap4"], 0.3), r["zc_text"].astype(bool))

    def test_commented_post_processing_medfilt2_bwareaopen_conv2(self):
        r = ref("compat")
        inp = inputs()
        med = ndimage.median_filter(inp["Rnd"], size=3, mode="constant", cval=0.0)
        assert np.array_equal(med, r["med_Rnd"])  # medfilt2 zero-pads
        med8 = ndimage.median_filter(inp["Rnd8"], size=3, mode="constant", cval=0)
        assert np.array_equal(med8, r["med_Rnd8"])
        assert np.array_equal(ndimage.median_filter(inp["Rnd"], size=3, mode="reflect"), r["med_Rnd_sym"])
        assert np.array_equal(ch4.bwareaopen(inp["Bw2"], 20), r["ao_20"].astype(bool))
        assert np.array_equal(ch4.bwareaopen(inp["Bw2"], 20, 4), r["ao_20_4"].astype(bool))
        assert np.array_equal(ch4.bwareaopen(inp["Bw2"], 5), r["ao_5"].astype(bool))
        assert int(r["ao_20"].sum()) == int(sc(r["n_ao"]))
        bw = inp["Bw2"].astype(np.float64)
        assert np.array_equal(conv2(bw, inp["Msk5"], "full"), r["cf"])
        assert np.array_equal(conv2(bw, inp["Msk5"], "same"), r["cs"])
        assert np.array_equal(conv2(bw, inp["Msk5"], "valid"), r["cv"])


# ================================================================================================================
# L2 — the original scripts on test.jpg (derivative.mat, morphology.mat, crop_fig4_3.mat)
# ================================================================================================================
@needs_image
@needs_ref("derivative")
class TestL2Derivative:
    @pytest.fixture(scope="class")
    def res(self):
        return ch4.sobel_edges_script(gray_image(), 0.05)

    def test_input_and_bw(self, res):
        d = ref("derivative")
        assert np.array_equal(res["im"], d["im"])  # double(im)/256, 0 diff
        m = assert_parity(res["bw"], d["BW"], "binary", name="BW")
        assert m["n_diff"] == 0
        assert res["n_edge"] == int(sc(d["n_edge"])) == int(d["BW"].sum())
        assert res["r"] == int(sc(d["r"])) and res["c"] == int(sc(d["c"]))
        assert np.array_equal(res["msk"], d["msk"])
        assert np.abs(res["gv"] - d["gv"]).max() <= 1e-16 and np.abs(res["gh"] - d["gh"]).max() <= 1e-16

    def test_variants(self, res):
        d = ref("derivative")
        im = res["im"]
        auto = edge(im, "sobel")
        assert auto.thresh == pytest.approx(sc(d["t_auto"]), rel=1e-12)
        assert int((auto.bw != d["BW_auto"].astype(bool)).sum()) == 0
        assert int((edge(im, "sobel", 0.03).bw != d["BW_003"].astype(bool)).sum()) == 0
        pre = edge(im, "prewitt", 0.05)
        assert int((pre.bw != d["BW_prewitt"].astype(bool)).sum()) == 0
        assert np.abs(pre.gv - d["gv_p"]).max() <= 1e-16 and np.abs(pre.gh - d["gh_p"]).max() <= 1e-16
        pa = edge(im, "prewitt")
        assert pa.thresh == pytest.approx(sc(d["t_prewitt_auto"]), rel=1e-12)
        assert int((pa.bw != d["BW_prewitt_auto"].astype(bool)).sum()) == 0
        assert int((edge(im, "sobel", 0.05, thinning=False).bw != d["BW_nothin"].astype(bool)).sum()) == 0
        assert int((edge(im, "sobel", 0.05, "horizontal").bw != d["BW_h"].astype(bool)).sum()) == 0
        assert int((edge(im, "sobel", 0.05, "vertical").bw != d["BW_v"].astype(bool)).sum()) == 0
        rob = edge(im, "roberts", 0.05)
        assert int((rob.bw != d["BW_rob"].astype(bool)).sum()) == 0
        ra = edge(im, "roberts")
        assert ra.thresh == pytest.approx(sc(d["t_rob"]), rel=1e-12)
        assert int((ra.bw != d["BW_rob_auto"].astype(bool)).sum()) == 0

    def test_log(self, res):
        d = ref("derivative")
        im = res["im"]
        lg = edge(im, "log", 0.005, sigma=2)
        assert lg.thresh == 0.005 == sc(d["t_log"])
        assert int((lg.bw != d["BW_log"].astype(bool)).sum()) == 0
        la = edge(im, "log", None, sigma=2)
        assert la.thresh == pytest.approx(sc(d["t_log_auto"]), rel=1e-12)
        assert int((la.bw != d["BW_log_auto"].astype(bool)).sum()) == 0

    def test_commented_lines_flags(self, res):
        d = ref("derivative")
        g = gray_image()
        med = ch4.sobel_edges_script(g, 0.05, median=True)
        assert np.array_equal(med["im"], d["im_med"])
        assert int((med["bw"] != d["BW_med"].astype(bool)).sum()) == 0
        ao = ch4.sobel_edges_script(g, 0.05, min_area=20)
        assert int((ao["bw"] != d["BW_ao"].astype(bool)).sum()) == 0
        sm = ch4.sobel_edges_script(g, 0.05, smooth=True)
        assert sm["bw_smooth"].shape == d["BWc"].shape == (2860, 4294)
        assert np.array_equal(sm["bw_smooth"], d["BWc"])


@needs_image
@needs_ref("morphology")
class TestL2Morphology:
    @pytest.fixture(scope="class")
    def res(self):
        return ch4.morphological_edges(gray_image())

    def test_all_twelve_arrays(self, res):
        d = ref("morphology")
        assert np.array_equal(res["im"], d["im"])
        assert res["level"] == sc(d["level"])  # graythresh bit-identical
        assert ref_str(d, "cls_BW1") == "double" and ref_str(d, "cls_internal") == "uint8"
        for key in ("I", "J", "K"):
            m = assert_parity(res[key], d[key], "binary", name=key)
            assert m["n_diff"] == 0, key
        for key in ("BW1", "BW2", "BW"):
            assert res[key].dtype == np.float64 and np.array_equal(res[key], d[key].astype(np.float64)), key
        for key in ("X", "Y", "internal", "external", "basic"):
            assert res[key].dtype == np.uint8 and np.array_equal(res[key], d[key]), key
        assert np.array_equal(res["SE"], d["nh"].astype(bool)) and int(sc(d["nseq"])) == 6
        assert ch4.gradient_identity_check(res) == {"binary": True, "gray": True}

    def test_counts_match_analysis_precheck(self, res):
        assert int(res["J"].sum()) == 3097387 and int(res["K"].sum()) == 4607838
        assert int(res["BW1"].sum()) == 698679 and int(res["BW2"].sum()) == 811772 and int(res["BW"].sum()) == 1510451
        assert res["threshold"] == 113.0


@needs_image
@needs_ref("crop_fig4_3")
class TestL2CropFig4_3:
    @pytest.fixture(scope="class")
    def crop(self):
        return ch4.fig_4_3a(gray_image())

    def test_crop_and_edges(self, crop):
        d = ref("crop_fig4_3")
        assert crop.shape == (552, 574) and np.array_equal(crop, d["crop_gray"])
        im = crop.astype(np.float64) / 256.0
        assert np.array_equal(im, d["im_d"])
        assert int((edge(im, "sobel", 0.05).bw != d["BW_sobel"].astype(bool)).sum()) == 0  # Fig. 4.3(b)
        assert int((edge(im, "prewitt", 0.05).bw != d["BW_prewitt"].astype(bool)).sum()) == 0  # Fig. 4.3(c)
        lg = edge(im, "log", 0.005, sigma=2)
        assert int((lg.bw != d["BW_log"].astype(bool)).sum()) == 0  # Fig. 4.6
        a = edge(im, "sobel")
        assert a.thresh == pytest.approx(sc(d["t_sobel_auto"]), rel=1e-12)
        assert int((a.bw != d["BW_sobel_auto"].astype(bool)).sum()) == 0
        assert int((edge(im, "sobel", 0.03).bw != d["BW_sobel_003"].astype(bool)).sum()) == 0
        assert int(d["BW_sobel"].sum()) == 1736 and int(d["BW_prewitt"].sum()) == 1737 and int(d["BW_log"].sum()) == 1885

    def test_morphology_r15(self, crop):
        d = ref("crop_fig4_3")
        res = ch4.morphological_edges(crop, radius=15)
        assert res["level"] == sc(d["level"])
        assert np.array_equal(res["SE"], d["nh15"].astype(bool)) and res["SE"].shape == (29, 29)
        for key in ("I", "J", "K"):
            assert int((res[key] != d[key].astype(bool)).sum()) == 0, key
        for key in ("BW1", "BW2", "BW"):
            assert np.array_equal(res[key], d[key].astype(np.float64)), key
        for key in ("X", "Y", "internal", "external", "basic"):
            assert np.array_equal(res[key], d[key]), key


# ================================================================================================================
# L4 — numbers quoted in the text
# ================================================================================================================
class TestL4BookNumbers:
    def test_fig_4_5_kernel_values(self):
        assert np.allclose(np.round(fspecial("gaussian", 5, 1), 4), FIG_4_5A)
        assert np.allclose(np.round(fspecial("log", 5, 1), 4), FIG_4_5B)

    def test_fig_4_7_matrices(self):
        assert np.array_equal(strel("disk", 5), FIG_4_7C)
        assert strel("diamond", 5).shape == (11, 11) and int(strel("diamond", 5).sum()) == 61

    def test_fig_4_15_caption_157_is_a_typo_for_15(self):
        # "157-pixel-radius" (p. 79) — the r = 15 disk of Figs. 4.9–4.16 has 697 px; 157 is the pixel count of the
        # script's strel('dis', 7), and a 157-px radius would exceed the 552x574 crop
        assert int(strel("disk", 15).sum()) == 697
        assert int(strel("dis", 7).sum()) == 157
        assert strel("disk", 157).shape == (313, 313)  # a 157-px radius disk would cover most of the 552x574 crop

    @needs_image
    @needs_ref("crop_fig4_3")
    def test_book_parameters_reproduce_fig_4_3_4_6(self):
        d = ref("crop_fig4_3")
        assert sc(d["t_log"]) == 0.005
        assert d["BW_sobel"].shape == (552, 574)

    def test_crop_refinement_result(self):
        p = VERIFY / "crop_refine.json"
        if not p.exists():
            pytest.skip("run reference/ch04/refine_crop.py")
        j = json.loads(p.read_text())
        assert j["best_within_pm2"] == {"dr": 0, "dc": 0, "ncc": j["ncc_nominal"]}
        assert j["ncc_nominal"] > 0.99
        assert j["nominal_crop_matlab"] == {"rows": [1600, 2151], "cols": [1979, 2552]}


# ================================================================================================================
# scripts
# ================================================================================================================
SCRIPTS = ["derivative", "morphology", "experiments"]
SCRIPTS_SYNTHETIC_ONLY: set[str] = set()


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_runs(name: str, tmp_path: Path):
    script = ROOT / "scripts" / f"ch04_{name}.py"
    out = tmp_path / name
    proc = subprocess.run([sys.executable, str(script), "--no-show", "--out", str(out)], capture_output=True,
                          text=True, cwd=str(ROOT), timeout=900)
    assert proc.returncode == 0, f"{script.name} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    if name not in SCRIPTS_SYNTHETIC_ONLY and not DATA.exists():
        assert "SKIP" in proc.stdout, f"{script.name} should print SKIP without data/book/ch04"
        if name != "morphology":
            pytest.skip(f"{script.name}: data/book/ch04 absent — graceful SKIP verified")
    files = list(out.glob("*.png"))
    assert files, f"{script.name} wrote nothing to {out}"
