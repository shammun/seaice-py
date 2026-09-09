"""Chapter 2 verification — Digital Image Processing Preliminaries (Book pp. 11–36).

Evidence levels (verify-port skill):
  L1  synthetic truth (no external tools)            -> ``test_l1_*``
  L2  parity vs MATLAB R2025a references              -> ``test_l2_*`` (skip if ``reference/ch02/*.mat`` missing;
      regenerate with ``.venv/Scripts/python.exe reference/ch02/make_refs.py``)
  L4  numbers quoted in the book text                 -> ``test_l4_*``
  scripts: every ``scripts/ch02_*.py`` must run headless and exit 0 -> ``test_script_*``

Coordinate conventions: Python is 0-based (row, col); MATLAB references are 1-based, so ``+1`` is applied on the
Python side before comparing boundaries / start points.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from seaice import ch02_preliminaries as ch2
from seaice.core import chaincode as cc
from seaice.core import color, connectivity as conn, distance as dist, filters, histogram as hist, interp, setops, synth
from seaice.core.io import load_book_image
from seaice.core.matlab_compat import im2double, im2uint8, imcomplement, matlab_round, rgb2gray_matlab
from tools.compare_arrays import assert_parity

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "ch02"
RGB_PATH = ROOT / "data" / "book" / "ch02" / "rgb.JPG"

needs_image = pytest.mark.skipif(not RGB_PATH.exists(), reason="data/book/ch02/rgb.JPG missing (run /setup-project)")


def needs_ref(name: str):
    return pytest.mark.skipif(not (REF / f"{name}.mat").exists(),
                              reason=f"reference/ch02/{name}.mat missing — run reference/ch02/make_refs.py (MATLAB)")


@pytest.fixture(scope="module")
def rgb() -> np.ndarray:
    if not RGB_PATH.exists():
        pytest.skip("rgb.JPG missing")
    return load_book_image("ch02", "rgb.jpg")


def ref(name: str) -> dict:
    d = loadmat(str(REF / f"{name}.mat"))
    return {k: v for k, v in d.items() if not k.startswith("__")}


def ref_str(d: dict, key: str) -> str:
    v = np.asarray(d[key]).ravel()
    return "" if v.size == 0 else str(v[0])


def vec(x) -> np.ndarray:
    return np.asarray(x).astype(np.int64).ravel()


def sc(x) -> int:
    """Scalar stored by MATLAB as a 1x1 array."""
    return int(np.asarray(x).ravel()[0])


# ================================================================================================================
# L1 — synthetic truth
# ================================================================================================================
class TestL1Color:
    def test_split_and_cmy(self):
        rgb = np.array([[[10, 20, 30], [255, 0, 128]]], dtype=np.uint8)
        R, G, B = color.split_rgb(rgb)
        assert R.tolist() == [[10, 255]] and G.tolist() == [[20, 0]] and B.tolist() == [[30, 128]]
        cmy = color.rgb2cmy(rgb)  # Eq. (2.3): 255 - I on uint8
        assert cmy.dtype == np.uint8 and cmy.tolist() == [[[245, 235, 225], [0, 255, 127]]]
        assert np.allclose(color.rgb2cmy(rgb.astype(float) / 255), 1 - rgb / 255.0)

    def test_cmyk_pure_colours(self):
        px = np.array([[[255, 0, 0], [255, 255, 255], [0, 0, 0], [0, 255, 0], [128, 128, 128]]], dtype=np.uint8)
        cmyk = color.rgb2cmyk(px, u=1.0, b=1.0)  # Eqs. (2.4)-(2.5)
        assert np.allclose(cmyk[0, 0], [0, 1, 1, 0])        # red   -> M, Y
        assert np.allclose(cmyk[0, 1], [0, 0, 0, 0])        # white -> nothing
        assert np.allclose(cmyk[0, 2], [0, 0, 0, 1])        # black -> K only
        assert np.allclose(cmyk[0, 3], [1, 0, 1, 0])        # green -> C, Y
        g = 1 - 128 / 255
        assert np.allclose(cmyk[0, 4], [0, 0, 0, g])        # gray  -> K = Kb
        with pytest.raises(ValueError):
            color.rgb2cmyk(px, u=1.5)

    def test_hsi_equations(self):
        px = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [100, 100, 100], [200, 50, 50]]], dtype=np.uint8)
        H, S, I = color.rgb2hsi(px)
        f = px[0].astype(float) / 255
        assert np.allclose(I[0], f.mean(axis=1))                         # Eq. (2.6a) row 1
        assert S[0, 3] == 0 and np.isnan(H[0, 3])                       # gray: V1 = V2 = 0 -> atan(0/0)
        v1 = (-f[:, 0] - f[:, 1] + 2 * f[:, 2]) / np.sqrt(6)
        v2 = (f[:, 0] - f[:, 1]) / np.sqrt(6)
        assert np.allclose(S[0], np.hypot(v1, v2))                        # Eq. (2.6c)
        with np.errstate(all="ignore"):
            assert np.allclose(H[0, :3], np.arctan(v2[:3] / v1[:3]))     # Eq. (2.6b)
        H2, _, _ = color.rgb2hsi(px, use_atan2=True)
        assert np.allclose(H2[0, :3], np.arctan2(v2[:3], v1[:3]))
        Hb, Sb, Ib = color.rgb2hsi(px, matlab_bug=True)
        assert np.allclose(Hb[0, [0, 1, 4]], -np.pi / 4) and np.isnan(Hb[0, 3])   # V2 = -V1 -> atan(-1)
        assert np.isnan(Hb[0, 2])                                                  # R == G == 0 -> NaN
        Hs, Ss, Is = color.rgb2hsi(px, scale=255.0)
        assert np.allclose(Ss, 255 * S) and np.allclose(Is, 255 * I)

    def test_indexed_to_rgb(self):
        cmap = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        idx = np.array([[0, 1], [2, 7]])
        out = color.indexed_to_rgb(idx, cmap)
        assert out.shape == (2, 2, 3) and np.allclose(out[1, 1], [0, 1, 0])  # clipped to last row
        assert np.allclose(color.indexed_to_rgb(idx + 1, cmap, one_based=True), out)


class TestL1MatlabCompat:
    def test_matlab_round_half_away(self):
        x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 0.49999, -0.49999])
        assert matlab_round(x).tolist() == [-3, -2, -1, 1, 2, 3, 0, 0]
        assert np.round(x).tolist() != matlab_round(x).tolist()  # numpy is half-to-even, so this matters

    def test_im2uint8_im2double(self):
        assert im2uint8(np.array([-0.1, 0.0, 0.5, 0.998, 1.0, 1.7])).tolist() == [0, 0, 128, 254, 255, 255]
        assert im2uint8(np.array([True, False])).tolist() == [255, 0]
        assert np.allclose(im2double(np.array([0, 51, 255], np.uint8)), [0, 0.2, 1])
        assert im2double(np.array([65535], np.uint16))[0] == 1.0

    def test_imcomplement_dtypes(self):
        assert imcomplement(np.array([0, 1, 255], np.uint8)).tolist() == [255, 254, 0]
        assert imcomplement(np.array([True, False])).tolist() == [False, True]
        assert np.allclose(imcomplement(np.array([0.0, 0.25, 300.0])), [1, 0.75, -299])
        assert imcomplement(np.array([-128, 0, 127], np.int8)).tolist() == [127, -1, -128]

    def test_rgb2gray_matlab_pure_colours(self):
        px = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [1, 1, 1]]], dtype=np.uint8)
        g = rgb2gray_matlab(px)
        assert g.dtype == np.uint8 and g[0].tolist() == [76, 150, 29, 255, 1]  # NTSC weights, rounded
        gf = rgb2gray_matlab(px.astype(float) / 255)
        assert gf.dtype == np.float64 and abs(gf[0, 0] - 0.298936021293775) < 1e-12


class TestL1Histogram:
    def test_imhist_exact_counts(self):
        img = np.array([[0, 0, 1], [255, 255, 255]], dtype=np.uint8)
        counts, x = hist.imhist(img)
        assert counts.sum() == img.size and counts[0] == 2 and counts[1] == 1 and counts[255] == 3
        assert x.tolist()[:3] == [0, 1, 2] and x[-1] == 255
        p, _ = hist.normalized_histogram(img)  # Eq. (2.8)
        assert abs(p.sum() - 1) < 1e-12 and p[255] == 0.5

    def test_imhist_bins_and_float(self):
        counts, x = hist.imhist(np.array([0, 1, 2, 3, 4, 5, 250, 251, 252, 253, 254, 255], np.uint8), 5)
        assert counts.tolist() == [6, 0, 0, 0, 6] and np.allclose(x, [0, 63.75, 127.5, 191.25, 255])
        c2, x2 = hist.imhist(np.array([0.0, 0.5, 1.0]), 3)
        assert c2.tolist() == [1, 1, 1] and np.allclose(x2, [0, 0.5, 1])


class TestL1Connectivity:
    def test_neighbourhood_sizes(self):
        shape = (5, 5)
        assert len(conn.n4((2, 2), shape)) == 4 and len(conn.nd((2, 2), shape)) == 4 and len(conn.n8((2, 2), shape)) == 8
        assert len(conn.n4((0, 0), shape)) == 2 and len(conn.nd((0, 0), shape)) == 1 and len(conn.n8((0, 0), shape)) == 3
        assert set(conn.n4((2, 2))) == {(3, 2), (1, 2), (2, 3), (2, 1)}
        assert set(conn.nd((2, 2))) == {(3, 3), (3, 1), (1, 3), (1, 1)}

    def test_fig_2_10_paths(self):
        img = synth.FIG_2_10_BC
        assert len(conn.find_paths(img, (0, 2), (2, 2), 4)) == 0       # no 4-path
        assert len(conn.find_paths(img, (0, 2), (2, 2), 8)) > 1        # 8-adjacency is ambiguous
        assert len(conn.find_paths(img, (0, 2), (2, 2), "m")) == 1     # m-adjacency: unique path
        assert conn.is_adjacent((0, 1), (1, 1), 4, img) and not conn.is_adjacent((0, 2), (1, 1), 4, img)
        # (0,2) and (1,1) are diagonal; N4 common = {(0,1),(1,2)}, (0,1) is in V -> NOT m-adjacent
        assert not conn.is_m_adjacent((0, 2), (1, 1), img)
        assert conn.is_m_adjacent((1, 1), (2, 2), img)

    def test_fig_2_11_component_counts(self):
        bw = synth.FIG_2_11_COMPONENTS
        assert conn.count_components(bw, 4) == 5 and conn.count_components(bw, 8) == 2  # p. 20

    def test_bwlabel_column_major_numbering(self):
        bw = np.zeros((6, 8), bool)
        bw[0, 6] = True   # first in row-major order
        bw[4, 0] = True   # first in column-major order -> MATLAB label 1
        bw[2, 3] = True
        L = conn.label_components(bw, 8)
        assert L[4, 0] == 1 and L[2, 3] == 2 and L[0, 6] == 3
        assert conn.label_components(np.zeros((3, 3), bool)).max() == 0

    def test_region_boundary_mask(self):
        bw = np.zeros((7, 7), bool)
        bw[1:6, 1:6] = True
        b8 = conn.region_boundary_mask(bw, 8)
        assert b8.sum() == 16 and not b8[3, 3]
        ring = conn.region_boundary_mask(np.ones((3, 3), bool), 4)   # image border counts as outside
        assert ring.sum() == 8 and not ring[1, 1]
        # notched square: (1,1) sees the missing corner only diagonally -> boundary under 8-conn, not under 4-conn
        notch = np.ones((5, 5), bool); notch[0, 0] = False
        b4, b8 = conn.region_boundary_mask(notch, 4), conn.region_boundary_mask(notch, 8)
        assert b4.sum() == 15 and b8.sum() == 16
        assert not b4[1, 1] and b8[1, 1]


class TestL1Distance:
    def test_fig_2_12_b(self):
        D = dist.distance_transform(synth.FIG_2_12_A, "euclidean")  # Eq. (2.9) + (2.10)
        assert np.allclose(D, synth.FIG_2_12_B, atol=1e-4)
        assert {round(float(v), 4) for v in np.unique(D)} == {0, 1, 1.4142, 2, 2.2361}

    def test_fig_2_13_centre_distances(self):
        C4 = dist.center_distance_map(7, "cityblock")   # Eq. (2.11)
        C8 = dist.center_distance_map(7, "chessboard")  # Eq. (2.12)
        assert C4[0, 0] == 6 and C4.max() == 6 and C8[0, 0] == 3 and C8.max() == 3
        assert (C4 == 1).sum() == 4 and (C8 == 1).sum() == 8   # r = 1 sets are N4 / N8
        assert np.allclose(dist.bwdist(synth.point_image(7, (3, 3)), "cityblock"), C4)

    def test_point_and_complement_semantics(self):
        bw = np.zeros((5, 5), bool)
        bw[2, 2] = True
        d = dist.distance_transform(bw, "chessboard")
        assert d[2, 2] == 1 and d[0, 0] == 0                      # object pixel -> distance to background = 1
        assert dist.bwdist(bw, "chessboard")[0, 0] == 2 and dist.bwdist(bw, "chessboard")[2, 3] == 1
        assert np.isinf(dist.distance_transform(np.ones((3, 3), bool))).all()
        assert (dist.distance_transform(np.zeros((3, 3), bool)) == 0).all()
        assert np.allclose(dist.bwdist(bw), dist.distance_transform(~bw))

    def test_metric_axioms(self):
        rng = np.random.default_rng(0)
        p, q, r = (rng.integers(0, 50, size=(200, 2)) for _ in range(3))
        for m in dist.METRICS:
            dpq, dqp = dist.pixel_distance(p, q, m), dist.pixel_distance(q, p, m)
            assert np.allclose(dpq, dqp)                                        # symmetry
            assert (dist.pixel_distance(p, p, m) == 0).all()                     # positive definite
            assert (dpq <= dist.pixel_distance(p, r, m) + dist.pixel_distance(r, q, m) + 1e-9).all()  # triangle
        assert dist.pixel_distance((0, 0), (3, 4)) == 5 and dist.pixel_distance((0, 0), (3, 4), "cityblock") == 7

    def test_quasi_euclidean_point(self):
        bw = synth.point_image(21, (10, 10))
        Dq = dist.bwdist(bw, "quasi-euclidean")
        rr, ccx = np.mgrid[0:21, 0:21]
        expect = dist.pixel_distance(np.stack([rr, ccx], -1), np.array([10, 10]), "quasi-euclidean")
        assert np.allclose(Dq, expect, atol=1e-9)


class TestL1Filters:
    def test_conv_at_eq_2_14_vs_eq_2_15_as_printed(self):
        """Book Eq. (2.14) is convolution f(x-s, y-t); the printed Eq. (2.15) expands to f(x+s, y+t) (correlation,
        = imfilter).  `conv_at` follows (2.14) by default and reproduces the printed (2.15) with correlate=True."""
        ex = ch2.convolution_example()
        f, w, x, y = ex["f"], ex["w"], ex["x"], ex["y"]
        assert ex["h_xy_conv"] == filters.conv_at(f, w, x, y) == ex["h_conv2"][x, y]
        assert ex["h_xy_corr"] == filters.conv_at(f, w, x, y, correlate=True) == filters.imfilter(f, w)[x, y]
        assert ex["h_xy_conv"] == 1.0 and ex["h_xy_corr"] == -1.0       # antisymmetric kernel: opposite signs
        assert ex["h_xy"] == ex["h_xy_conv"]                            # backward-compatible key = Eq. (2.14)
        sym = np.ones((3, 3)) / 9.0
        assert np.isclose(filters.conv_at(f, sym, x, y), filters.conv_at(f, sym, x, y, correlate=True))

    def test_imfilter_matlab_positional_options(self):
        f = np.arange(20, dtype=float).reshape(4, 5)
        w = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float)
        assert np.array_equal(filters.imfilter(f, w, "replicate"), filters.imfilter(f, w, padding="replicate"))
        assert np.array_equal(filters.imfilter(f, w, "symmetric", "full"),
                              filters.imfilter(f, w, padding="symmetric", output="full"))
        assert np.array_equal(filters.imfilter(f, w, "conv", "replicate"),
                              filters.imfilter(f, w, mode="conv", padding="replicate"))
        assert np.array_equal(filters.imfilter(f, w, 2.5), filters.imfilter(f, w, padding=2.5))
        with pytest.raises(ValueError):
            filters.imfilter(f, w, "replicate", padding="symmetric")
        with pytest.raises(ValueError):
            cc.fchcode(np.array([[0, 0], [0, 1], [1, 1], [1, 0]]), conn=5)   # validated before any computation

    def test_conv2_eq_2_15(self):
        f = np.arange(36, dtype=float).reshape(6, 6)
        w = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float)
        h = filters.conv2(f, w, "same")
        for (x, y) in [(2, 3), (0, 0), (5, 5), (1, 4)]:
            assert abs(h[x, y] - filters.conv_at(f, w, x, y)) < 1e-12
        assert np.allclose(filters.imfilter(f, w, "corr"), filters.conv2(f, w[::-1, ::-1], "same"))
        assert np.allclose(filters.imfilter(f, w, "conv"), h)
        delta = np.zeros((3, 3)); delta[1, 1] = 1
        assert np.allclose(filters.conv2(f, delta), f) and np.allclose(filters.imfilter(f, delta), f)
        shift = np.zeros((3, 3)); shift[0, 1] = 1          # w(-1, 0) -> h(x, y) = f(x + 1, y)
        assert np.allclose(filters.conv2(f, shift)[:-1], f[1:])

    def test_imfilter_padding_modes(self):
        f = np.arange(20, dtype=float).reshape(4, 5)
        w = np.ones((3, 3)) / 9
        z = filters.imfilter(f, w, padding="zeros")
        r = filters.imfilter(f, w, padding="replicate")
        assert z[0, 0] < r[0, 0] and np.allclose(z[1:-1, 1:-1], r[1:-1, 1:-1])
        assert filters.imfilter(f, w, output="full").shape == (6, 7)


class TestL1SetOps:
    def test_truth_tables_and_bitwise(self):
        t = setops.truth_tables()
        assert t["NOT"] == [(0, 1), (1, 0)]
        assert t["OR"] == [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)]
        assert t["AND"] == [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
        assert int(setops.bitwise_and(np.uint8(57), np.uint8(207))) == 9  # p. 29
        assert int(setops.bitwise_or(np.uint8(57), np.uint8(207))) == 255
        assert setops.bitwise_not(np.array([0, 57], np.uint8)).tolist() == [255, 198]

    def test_set_identities(self):
        rng = np.random.default_rng(1)
        A, B = rng.random((16, 16)) > 0.5, rng.random((16, 16)) > 0.5
        assert np.array_equal(setops.difference(A, B), A & ~B)                                  # Eq. (2.24)
        assert np.array_equal(setops.complement(setops.union(A, B)),
                              setops.intersection(setops.complement(A), setops.complement(B)))  # De Morgan
        assert np.array_equal(setops.complement(setops.complement(A)), A)
        assert np.array_equal(setops.union(A, setops.complement(A)), np.ones_like(A))

    def test_reflection_and_translation(self):
        L = synth.set_operation_fixtures()["L"]
        origin = synth.set_operation_fixtures()["L_origin"]
        Lh, oh = setops.reflect(L, origin)                          # Eq. (2.25)
        assert np.array_equal(setops.reflect(Lh, oh)[0], L)
        for (r, c) in zip(*np.nonzero(L)):
            assert Lh[oh[0] - (r - origin[0]), oh[1] - (c - origin[1])]   # a -> -a relative to the origin
        A = np.zeros((6, 6), bool); A[1, 2] = True
        Az = setops.translate(A, (2, 3))                             # Eq. (2.26)
        assert Az.sum() == 1 and Az[3, 5]
        assert setops.translate(A, (5, 0)).sum() == 0                # shifted out of the rectangle
        assert setops.translate(A, (5, 0), shape=(8, 6))[6, 2]

    def test_gray_set_ops(self):
        I = np.array([[0, 100, 255]], np.uint8)
        assert setops.gray_complement(I).tolist() == [[255, 155, 0]]                   # Eq. (2.28)
        assert np.array_equal(setops.gray_union(I, setops.gray_complement(I)), np.maximum(I, 255 - I))
        assert np.array_equal(setops.gray_intersection(I, setops.gray_complement(I)), np.minimum(I, 255 - I))


class TestL1ChainCode:
    def test_first_difference_eq_2_31(self):
        assert cc.first_difference(np.array([0, 1, 2, 0]), 8).tolist() == [1, 1, 6, 0]
        assert cc.first_difference(np.array([0, 6, 4, 2]), 8).tolist() == [6, 6, 6, 6]   # square, cw
        assert cc.first_difference(np.array([0, 3, 2, 1]), 4).tolist() == [3, 3, 3, 3]

    def test_min_magnitude_bruteforce(self):
        rng = np.random.default_rng(3)
        for n in (1, 2, 5, 9, 18):
            for _ in range(20):
                code = rng.integers(0, 8, n)
                rots = [np.roll(code, -k).tolist() for k in range(n)]
                assert cc.min_magnitude(code).tolist() == min(rots)
        periodic = np.array([0, 6, 1, 1, 6, 0, 7, 1, 6] * 2)
        assert cc.min_magnitude(np.roll(periodic, 4)).tolist() == periodic.tolist()

    def test_fchcode_small_shapes(self):
        sq = np.zeros((4, 4), bool); sq[1:3, 1:3] = True
        b = cc.boundaries(sq)[0]
        assert b.shape == (5, 2) and np.array_equal(b[0], b[-1]) and tuple(b[0]) == (1, 1)
        c = cc.fchcode(b)
        assert c.fcc.tolist() == [0, 6, 4, 2] and c.diff.tolist() == [6, 6, 6, 6] and c.x0y0 == (1, 1)
        assert c.x0y0_matlab == (2, 2)
        diag = np.zeros((5, 5), bool); diag[[1, 2, 3], [1, 2, 3]] = True
        assert cc.fchcode(cc.boundaries(diag)[0]).fcc.tolist() == [7, 7, 3, 3]
        assert cc.fchcode(cc.boundaries(sq, 4)[0], 4).fcc.tolist() == [0, 3, 2, 1]
        assert cc.code_reverse(np.array([0, 6, 4, 2])).tolist() == [6, 0, 2, 4]
        assert cc.fchcode(b, 8, "reverse").fcc.tolist() == [6, 0, 2, 4]
        with pytest.raises(ValueError):
            cc.fchcode(np.array([[0, 0], [0, 2]]))
        with pytest.warns(UserWarning):
            cc.fchcode(cc.boundaries(diag)[0], 4)

    def test_rotation_invariance_of_first_difference(self):
        r1 = np.zeros((6, 6), bool); r1[1:3, 1:4] = True     # 2 x 3
        r2 = np.zeros((6, 6), bool); r2[1:4, 1:3] = True     # 3 x 2 (rotated 90 deg)
        d1 = cc.fchcode(cc.boundaries(r1)[0]).diff
        d2 = cc.fchcode(cc.boundaries(r2)[0]).diff
        assert any(np.array_equal(np.roll(d1, k), d2) for k in range(len(d1)))
        assert np.array_equal(cc.min_magnitude(d1), cc.min_magnitude(d2))

    def test_boundaries_special_cases(self):
        single = np.zeros((5, 5), bool); single[2, 2] = True
        b = cc.boundaries(single)[0]
        assert b.tolist() == [[2, 2], [2, 2]]
        assert cc.fchcode(b).fcc.tolist() == [0]                     # MATLAB errors here (minmag); documented
        two = np.zeros((3, 3), bool)
        two[0, 0] = two[1, 1] = True
        B = cc.boundaries(two)
        assert len(B) == 1 and len(cc.boundaries(two, 4)) == 2      # 8-connected diagonal pair is one object
        assert cc.boundaries(np.zeros((3, 3), bool)) == []
        hole = np.ones((7, 7), bool); hole[2:5, 2:5] = False
        assert len(cc.boundaries(hole)) == 1 and cc.boundaries(hole)[0].shape[0] == 25   # exterior only
        spur = cc.boundaries(synth.spur_shape())[0]
        assert (spur[:, 1] == 7).sum() == 1 and (spur[:, 1] == 6).sum() == 2   # spur traversed twice
        ccw = cc.boundaries(synth.FIG_2_19_OBJECT, 8, "ccw")[0]
        assert np.array_equal(ccw, cc.boundaries(synth.FIG_2_19_OBJECT)[0][::-1])

    def test_chain_to_points_roundtrip(self):
        b = cc.boundaries(synth.FIG_2_19_OBJECT)[0]
        c = cc.fchcode(b)
        pts = cc.chain_to_points(c.x0y0, c.fcc)
        assert np.array_equal(pts, b)

    def test_bound2im_conventions(self):
        b = cc.boundaries(synth.FIG_2_19_OBJECT)[0]
        t = cc.bound2im(b)
        assert t.shape == (6, 7) and t.sum() == 18
        c3 = cc.bound2im(b, 12, 14)
        assert c3.shape == (12, 14) and c3.sum() == 18 and c3[3:9, 4:11].sum() == 18
        full = cc.bound2im(b, 8, 9, int(b[:, 0].min()), int(b[:, 1].min()))
        assert np.array_equal(full, cc.bound2im(b.T, 8, 9, int(b[:, 0].min()), int(b[:, 1].min())))
        assert full.shape == (8, 9) and full[b[:, 0], b[:, 1]].all()
        with pytest.raises(ValueError):
            cc.bound2im(b, 5, 5)
        with pytest.raises(ValueError):
            cc.bound2im(b, 8, 9, 5, 0)


class TestL1Interp:
    def test_keys_kernel(self):
        assert interp.keys_kernel([0, 1, -1, 2, -2, 3]).tolist() == [1, 0, 0, 0, 0, 0]   # interpolating
        x = np.linspace(-2, 2, 41)
        assert abs(interp.keys_kernel(x).sum() * 0.1 - 1.0) < 1e-9      # partition of unity on the lattice
        assert np.allclose(interp.keys_kernel(0.5), 0.5625) and np.allclose(interp.keys_kernel(1.5), -0.0625)
        assert np.allclose(interp.keys_kernel(0.5, a=-0.75), 0.59375)

    def test_lattice_reproduction_and_weights(self):
        rng = np.random.default_rng(0)
        img = rng.random((8, 9)) * 255
        rr, ccx = np.mgrid[0:8, 0:9].astype(float)
        for m in ("nearest", "bilinear", "bicubic"):
            assert np.allclose(interp.interp2(img, rr, ccx, m), img)
        const = np.full((6, 6), 7.0)
        u, v = np.meshgrid(np.linspace(0, 5, 13), np.linspace(0, 5, 17), indexing="ij")
        assert np.allclose(interp.interp2(const, u, v, "bilinear"), 7.0)   # weights sum to 1
        assert np.allclose(interp.interp2(const, u, v, "bicubic"), 7.0)
        # bilinear midpoint = mean of the 4 neighbours (Eqs. 2.35-2.37)
        assert np.isclose(interp.interp_bilinear(img, 2.5, 3.5), img[2:4, 3:5].mean())
        assert np.isclose(interp.interp_bilinear(img, 2.0, 3.5), img[2, 3:5].mean())

    def test_nearest_tie_rule_and_border(self):
        def fresh():
            return np.arange(16, dtype=float).reshape(4, 4)
        assert interp.interp_nearest(fresh(), 1.5, 2.5) == 11.0          # half away from zero -> (2, 3)
        assert interp.interp_nearest(fresh(), 1.49, 2.49) == 6.0         # -> (1, 2)
        assert np.isnan(interp.interp_nearest(fresh(), -0.1, 0.0))
        assert np.isnan(interp.interp_bilinear(fresh(), 0, 3.01))
        assert interp.interp_nearest(fresh(), -0.1, 0.0, border="replicate") == 0.0
        assert interp.interp_bilinear(fresh(), 5.0, 5.0, border="replicate") == 15.0

    def test_interp_nearest_scalar_query_must_not_mutate_input(self):
        """PORT BUG (open item): with 0-d (scalar) query coordinates ``img[i, j]`` is a basic-indexing *view* of the
        float64 input, and ``out[out_mask] = fill`` then writes NaN into the caller's image.  Array queries use
        fancy indexing (a copy) and are unaffected."""
        img = np.arange(16, dtype=float).reshape(4, 4)
        before = img.copy()
        interp.interp_nearest(img, -0.1, 0.0)             # out-of-range scalar query
        assert np.array_equal(img, before), f"interp_nearest mutated its input: {img[0, 0]!r} at (0, 0)"
        interp.interp2(img, np.array([-1.0]), np.array([0.0]), "nearest")   # 1-element array query is safe
        assert np.array_equal(img, before)

    def test_resize_and_warp(self):
        img = np.arange(30, dtype=float).reshape(5, 6)
        assert interp.resize(img, 2, "nearest").shape == (10, 12)
        assert interp.resize(img, (0.5, 1.5), "bilinear").shape == (3, 9)
        ident = interp.warp_image(img, lambda m, n: (m, n), img.shape, "bicubic")
        assert np.allclose(ident, img)
        assert interp.resize(img, 3, "nearest")[::3, ::3].shape == (5, 6)
        with pytest.raises(ValueError):
            interp.interp2(img, 0, 0, "spline")


# ================================================================================================================
# L2 — parity vs MATLAB R2025a references
# ================================================================================================================
@needs_ref("histogram")
class TestL2Histogram:
    @needs_image
    def test_jpeg_decode(self, rgb):
        d = ref("histogram")
        I_ml = d["I"]
        assert I_ml.shape == rgb.shape == (1536, 2048, 3)
        diff = np.abs(I_ml.astype(int) - rgb.astype(int))
        n_diff = int((diff > 0).sum())
        assert diff.max() <= 1, f"JPEG decode differs by up to {diff.max()} levels"
        assert n_diff == 0, f"JPEG decode: {n_diff} of {diff.size} samples differ (max {diff.max()})"

    def test_rgb2gray_exact(self):
        d = ref("histogram")
        G = rgb2gray_matlab(d["I"])                                  # same decoded input as MATLAB
        m = assert_parity(G, d["G"], "float", atol=0, rtol=0, name="rgb2gray")
        assert int((G != d["G"]).sum()) == 0 and m["max_abs_err"] == 0

    @needs_image
    def test_rgb2gray_from_python_decode(self, rgb):
        d = ref("histogram")
        G = rgb2gray_matlab(rgb)
        assert int((G != d["G"]).sum()) == 0

    def test_manual_and_imhist_counts(self):
        d = ref("histogram")
        G = d["G"]
        h = ch2.gray_histogram(G)
        assert np.array_equal(h["num"], vec(d["num"])) and np.array_equal(h["counts"], vec(d["cnt"]))
        assert_parity(h["GP"], d["GP"].ravel(), "float", atol=1e-15, name="GP (Eq. 2.8)")
        assert h["counts"].sum() == sc(d["m"]) * sc(d["n"])
        assert np.array_equal(h["x"], vec(d["x"]))

    def test_channel_histograms(self):
        d = ref("histogram")
        ch = ch2.channel_histograms(d["I"])
        for k in ("y_r", "y_g", "y_b"):
            assert np.array_equal(ch[k], vec(d[k])), k
        peaks = d["peaks"].astype(int)
        assert (int(ch["y_r"].max()), int(ch["y_r"].argmax())) == tuple(peaks[1])
        assert (int(ch["y_g"].max()), int(ch["y_g"].argmax())) == tuple(peaks[2])
        assert (int(ch["y_b"].max()), int(ch["y_b"].argmax())) == tuple(peaks[3])
        assert ch["pixel_1076_675"].tolist() == vec(d["pix"]).tolist() == [28, 76, 114]


@needs_ref("color_image")
class TestL2ColorImage:
    def test_cmy_exact(self):
        d = ref("color_image")
        res = ch2.color_components(d["I"])
        assert np.array_equal(res["I_cmy"], d["I_cmy"])
        for k in ("Ic", "Im", "Iy"):
            assert_parity(res[k], d[k], "float", atol=0, rtol=0, name=k)   # 1 - double(plane)

    def test_hsi_matlab_script_values(self):
        d = ref("color_image")
        res = ch2.color_components(d["I"])
        assert_parity(res["I_matlab"], d["Ii"], "float", atol=1e-9, rtol=0, name="Ii")
        assert_parity(res["S_matlab"], d["Is"], "float", atol=1e-9, rtol=0, name="Is (buggy V1)")
        H = res["H_matlab"]
        assert np.array_equal(np.isnan(H), np.isnan(d["Ih"])), "NaN pattern of the buggy hue differs"
        assert_parity(H, d["Ih"], "float", atol=1e-9, rtol=0, name="Ih (buggy V1)")
        assert sc(d["nanH"]) == int(np.isnan(H).sum()) == 0   # no pixel of rgb.JPG has R == G (MATLAB agrees)
        assert np.allclose(np.nanmax(np.abs(d["Ih"] + np.pi / 4)), 0, atol=1e-9)  # the script's H is -pi/4 flat

    def test_hsi_book_equation_vs_corrected_matlab(self):
        d = ref("color_image")
        I8 = d["I"]
        H, S, I = color.rgb2hsi(I8)                                      # book Eq. (2.6), RGB in [0, 1]
        assert np.array_equal(np.isnan(H), np.isnan(d["Ihc"]))
        assert sc(d["nanHc"]) == int(np.isnan(H).sum())
        assert_parity(255 * S, d["Isc"], "float", atol=1e-9, rtol=0, name="S")
        assert_parity(255 * I, d["Ii"], "float", atol=1e-9, rtol=0, name="I")
        # Same integer-valued doubles as the script (scale=255): bit-level agreement expected.
        H255, _, _ = color.rgb2hsi(I8, scale=255.0)
        assert_parity(H255, d["Ihc"], "float", atol=1e-12, rtol=0, name="H (scale=255 vs corrected MATLAB)")
        # RGB in [0, 1] (the text's convention): identical except where V1 == 0 mathematically (R + G == 2B):
        # there atan(V2 / V1) = ±pi/2 with the sign fixed by ~1e-16 rounding, on both sides.  Not a port error.
        diff = np.nan_to_num(np.abs(H - d["Ihc"]))
        bad = diff > 1e-9
        R, G, B = (I8[..., k].astype(int) for k in range(3))
        assert np.all((R + G == 2 * B)[bad]), "hue mismatch at a pixel where V1 != 0"
        assert np.allclose(np.abs(H[bad]), np.pi / 2, atol=1e-6) and np.allclose(np.abs(d["Ihc"][bad]), np.pi / 2, atol=1e-6)
        assert bad.sum() <= 1e-4 * bad.size
        assert diff[~bad].max() < 1e-9
        H2, _, _ = color.rgb2hsi(I8, use_atan2=True)
        assert_parity(H2, d["Ihc2"], "float", atol=1e-9, rtol=0, name="H atan2")   # atan2 is continuous across V1 = 0

    def test_cmyk(self):
        d = ref("color_image")
        cmyk = color.rgb2cmyk(d["I"], u=1.0, b=1.0)
        for i, k in enumerate(("C", "M", "Y", "K")):
            assert_parity(cmyk[..., i], d[k], "float", atol=1e-12, rtol=0, name=k)
        cmyk2 = color.rgb2cmyk(d["I"], u=0.5, b=0.7)
        for i, k in enumerate(("C2", "M2", "Y2", "K2")):
            assert_parity(cmyk2[..., i], d[k], "float", atol=1e-12, rtol=0, name=k)


@needs_ref("distance_transform")
class TestL2DistanceTransform:
    def test_script_point_maps(self):
        d = ref("distance_transform")
        maps = ch2.point_distance_maps(201)
        assert_parity(maps["euclidean"], d["De"], "float", atol=1e-4, rtol=0, name="De")
        assert_parity(maps["cityblock"], d["D4"], "float", atol=0, rtol=0, name="D4")
        assert_parity(maps["chessboard"], d["D8"], "float", atol=0, rtol=0, name="D8")
        # MATLAB bwdist returns single and accumulates the quasi-euclidean chamfer in single precision
        # (error ~1e-4 at r ~ 140); the port is exact to 1e-12 vs the analytic value -> compare with rtol.
        rr, ccx = np.mgrid[0:201, 0:201]
        exact = -dist.pixel_distance(np.stack([rr, ccx], -1), np.array([100, 100]), "quasi-euclidean")
        assert np.abs(maps["quasi-euclidean"] - exact).max() < 1e-9
        assert_parity(maps["quasi-euclidean"], d["Dq"], "float", atol=1e-6, rtol=1e-5, name="Dq")
        assert_parity(maps["chessboard"], d["D8_script"], "float", atol=0, rtol=0, name="script imgDist")
        assert np.array_equal(~maps["point"], d["img_script"].astype(bool))

    def test_fig_2_12_and_2_13(self):
        d = ref("distance_transform")
        assert np.array_equal(d["A12"].astype(bool), synth.FIG_2_12_A)
        ex = ch2.distance_fixture_examples()
        assert_parity(ex["D12"], d["D12"], "float", atol=1e-4, rtol=0, name="D12")
        assert_parity(d["D12"], synth.FIG_2_12_B, "float", atol=1e-4, rtol=0, name="MATLAB vs book Fig 2.12b")
        assert_parity(dist.distance_transform(synth.FIG_2_12_A, "cityblock"), d["D12_4"], "float", atol=0, name="D12_4")
        assert_parity(dist.distance_transform(synth.FIG_2_12_A, "chessboard"), d["D12_8"], "float", atol=0, name="D12_8")
        assert_parity(dist.distance_transform(synth.FIG_2_12_A, "quasi-euclidean"), d["D12_q"], "float", atol=1e-4,
                      name="D12_q")
        assert_parity(ex["C4"], d["C4"], "float", atol=0, name="C4")
        assert_parity(ex["C8"], d["C8"], "float", atol=0, name="C8")
        assert_parity(ex["Ce"], d["Ce"], "float", atol=1e-4, name="Ce")

    def test_random_mask_all_metrics(self):
        d = ref("distance_transform")
        R = d["R"].astype(bool)
        assert_parity(dist.bwdist(R, "euclidean"), d["Re"], "float", atol=1e-4, rtol=0, name="Re")
        assert_parity(dist.bwdist(R, "cityblock"), d["R4"], "float", atol=0, rtol=0, name="R4")
        assert_parity(dist.bwdist(R, "chessboard"), d["R8"], "float", atol=0, rtol=0, name="R8")
        assert_parity(dist.bwdist(R, "quasi-euclidean"), d["Rq"], "float", atol=1e-4, rtol=0, name="Rq")
        assert np.isinf(d["Dz"]).all() and np.isinf(dist.bwdist(np.zeros((5, 5), bool))).all()


@needs_ref("chain_diff")
class TestL2ChainDiff:
    def test_boundary_and_image(self):
        d = ref("chain_diff")
        res = ch2.chain_code_demo()
        assert np.array_equal(res["B"], d["B"].astype(bool))
        assert np.array_equal(res["b_matlab"], d["b"].astype(int))          # 19 x 2, 1-based
        assert res["b"].shape[0] == 19 == sc(d["d"])
        assert np.array_equal(res["bim"], d["bim"].astype(bool))
        assert np.array_equal(res["b4"] + 1, d["b4"].astype(int))
        assert np.array_equal(cc.boundaries(res["B"], 8, "ccw")[0] + 1, d["bccw"].astype(int))

    def test_fchcode_fields(self):
        d = ref("chain_diff")
        c = ch2.chain_code_demo()["c"]
        for py, ml in (("fcc", "fcc"), ("diff", "dif"), ("mm", "mm"), ("diffmm", "diffmm")):
            assert getattr(c, py).tolist() == vec(d[ml]).tolist(), py
        assert c.x0y0_matlab == tuple(vec(d["x0y0"])) == (4, 2)
        assert vec(d["minmag_ok"]).tolist() == c.mm.tolist()

    def test_fchcode_4conn_reverse_ccw(self):
        d = ref("chain_diff")
        res = ch2.chain_code_demo()
        assert ref_str(d, "err4") == "" and ref_str(d, "errrev") == "" and ref_str(d, "errccw") == ""
        c4 = res["c4"]
        assert c4.fcc.tolist() == vec(d["fcc4"]).tolist() and len(c4.fcc) == 22      # Fig. 2.19(b)
        assert c4.diff.tolist() == vec(d["dif4"]).tolist()
        assert c4.mm.tolist() == vec(d["mm4"]).tolist() and c4.diffmm.tolist() == vec(d["diffmm4"]).tolist()
        crev = cc.fchcode(res["b"], 8, "reverse")
        assert crev.fcc.tolist() == vec(d["fccrev"]).tolist() and crev.diff.tolist() == vec(d["difrev"]).tolist()
        assert crev.mm.tolist() == vec(d["mmrev"]).tolist()
        bccw = cc.boundaries(res["B"], 8, "ccw")[0]
        assert cc.fchcode(bccw).fcc.tolist() == vec(d["fccccw"]).tolist()

    def test_minmag_periodic_matlab_errors_python_tiebreaks(self):
        """MATLAB's minmag never assigns z for a periodic code; the port returns the lexicographic minimum."""
        d = ref("chain_diff")
        assert "not assigned" in ref_str(d, "minmag_err")
        assert np.asarray(d["mm_of_diff"]).size == 0
        nfd = cc.normalized_first_difference(vec(d["fcc"]))
        assert nfd.tolist() == synth.FIG_2_21["normalized_first_difference"].tolist()   # book p. 31

    def test_bound2im_conventions(self):
        d = ref("chain_diff")
        b = ch2.chain_code_demo()["b"]
        assert np.array_equal(cc.bound2im(b), d["bim1"].astype(bool))
        assert np.array_equal(cc.bound2im(b.T), d["bimT"].astype(bool))
        assert np.array_equal(cc.bound2im(b, 12, 14), d["bim3"].astype(bool))
        assert np.array_equal(cc.bound2im(b, 8, 9, 1, 2), d["bim5"].astype(bool))      # MATLAB x0=2, y0=3


@needs_ref("boundaries_multi")
class TestL2BoundariesMulti:
    def test_bwlabel_numbering_fig_2_11(self):
        d = ref("boundaries_multi")
        bw = d["BW11"].astype(bool)
        assert np.array_equal(bw, synth.FIG_2_11_COMPONENTS)
        L4, L8 = conn.label_components(bw, 4), conn.label_components(bw, 8)
        assert np.array_equal(L4, d["L4"]) and np.array_equal(L8, d["L8"])      # numbering included
        assert L4.max() == 5 == sc(d["n4"]) and L8.max() == 2 == sc(d["n8"])

    def test_bwlabel_numbering_real_mask(self):
        d = ref("boundaries_multi")
        Q = d["Q"].astype(bool)
        assert np.array_equal(conn.label_components(Q, 4), d["LQ4"])
        assert np.array_equal(conn.label_components(Q, 8), d["LQ8"])
        assert_parity(conn.label_components(Q, 8), d["LQ8"], "label", name="LQ8 partition")

    def test_all_boundaries_fig_2_11(self):
        d = ref("boundaries_multi")
        bw = d["BW11"].astype(bool)
        for c_, tag in ((8, "B8"), (4, "B4")):
            for direction, suffix in (("cw", ""), ("ccw", "c")):
                B = cc.boundaries(bw, c_, direction)
                n = 2 if c_ == 8 else 5
                assert len(B) == n
                for k in range(n):
                    assert np.array_equal(B[k] + 1, d[f"{tag}{suffix}_{k + 1}"].astype(int)), f"{tag}{suffix}_{k + 1}"

    def test_single_pixel_spur_hole(self):
        d = ref("boundaries_multi")
        s = np.zeros((5, 5), bool); s[2, 2] = True
        assert np.array_equal(cc.boundaries(s)[0] + 1, d["Bs_1"].astype(int))           # two identical rows
        assert "not assigned" in ref_str(d, "errs")                                       # MATLAB fchcode fails
        assert cc.fchcode(cc.boundaries(s)[0]).fcc.tolist() == [0]                        # port: zero step -> 0
        assert np.array_equal(d["Sp"].astype(bool), synth.spur_shape())
        bsp = cc.boundaries(synth.spur_shape())[0]
        assert np.array_equal(bsp + 1, d["Bsp_1"].astype(int))
        csp = cc.fchcode(bsp)
        assert csp.fcc.tolist() == vec(d["fccsp"]).tolist() and csp.mm.tolist() == vec(d["mmsp"]).tolist()
        H = d["H"].astype(bool)
        BH = cc.boundaries(H)
        assert len(BH) == sc(d["nH"]) == 1 and np.array_equal(BH[0] + 1, d["BH_1"].astype(int))
        assert sc(d["nE"]) == 0 and cc.boundaries(np.zeros((4, 6), bool)) == []

    def test_real_mask_boundaries_and_codes(self):
        d = ref("boundaries_multi")
        Q = d["Q"].astype(bool)
        for c_, len_key, cat_key in ((8, "BQlen", "BQcat"), (4, "BQ4len", "BQ4cat")):
            B = cc.boundaries(Q, c_)
            lens = vec(d[len_key])
            assert [len(b) for b in B] == lens.tolist()
            assert np.array_equal(np.vstack(B) + 1, d[cat_key].astype(int))
        B8 = cc.boundaries(Q, 8)
        err = vec(d["fccQerr"]); lens = vec(d["fccQlen"])
        fcc_cat, mm_cat = vec(d["fccQcat"]), vec(d["mmQcat"])
        pos = 0
        n_checked = 0
        for k, b in enumerate(B8):
            c = cc.fchcode(b)
            if err[k]:
                continue  # MATLAB minmag failed (periodic / single pixel) — port tie-breaks, nothing to compare
            assert c.fcc.tolist() == fcc_cat[pos:pos + lens[k]].tolist(), f"object {k + 1} fcc"
            assert c.mm.tolist() == mm_cat[pos:pos + lens[k]].tolist(), f"object {k + 1} mm"
            pos += lens[k]
            n_checked += 1
        assert n_checked >= 1 and pos == fcc_cat.size


@needs_ref("conv2")
class TestL2Conv2:
    def test_conv2(self):
        d = ref("conv2")
        f = d["f"]
        for w, key in (("w3", "h3"), ("w5", "h5"), ("w4", "h4"), ("w23", "h23")):
            assert_parity(filters.conv2(f, d[w], "same"), d[key], "float", atol=1e-12, rtol=0, name=key)
        assert_parity(filters.conv2(f, d["w3"], "full"), d["h3full"], "float", atol=1e-12, rtol=0, name="h3full")
        assert_parity(filters.conv2(f, d["w5"], "full"), d["h5full"], "float", atol=1e-12, rtol=0, name="h5full")
        assert_parity(filters.conv2(f, d["w3"], "valid"), d["h3valid"], "float", atol=1e-12, rtol=0, name="h3valid")

    def test_imfilter(self):
        d = ref("conv2")
        f, w3, w5, w4, w23 = d["f"], d["w3"], d["w5"], d["w4"], d["w23"]
        cases = {
            "g3": filters.imfilter(f, w3), "g3c": filters.imfilter(f, w3, "conv"),
            "g3r": filters.imfilter(f, w3, padding="replicate"), "g3s": filters.imfilter(f, w3, padding="symmetric"),
            "g3w": filters.imfilter(f, w3, padding="circular"), "g3k": filters.imfilter(f, w3, padding=2.5),
            "g4": filters.imfilter(f, w4), "g23": filters.imfilter(f, w23),
            "g5r": filters.imfilter(f, w5, padding="replicate"), "g5s": filters.imfilter(f, w5, padding="symmetric"),
            "g3full": filters.imfilter(f, w3, output="full"), "g5full": filters.imfilter(f, w5, output="full"),
            "g4full": filters.imfilter(f, w4, output="full"), "g4r": filters.imfilter(f, w4, padding="replicate"),
            "g3ch": filters.imfilter(d["f3"], w3),
            # 'full' output combined with non-zero padding (MATLAB positional idiom)
            "g3rfull": filters.imfilter(f, w3, "replicate", "full"), "g3sfull": filters.imfilter(f, w3, "symmetric", "full"),
            "g3wfull": filters.imfilter(f, w3, "circular", "full"), "g5rfull": filters.imfilter(f, w5, "replicate", "full"),
            "g5sfull": filters.imfilter(f, w5, "symmetric", "full"), "g4rfull": filters.imfilter(f, w4, "replicate", "full"),
            "g23sfull": filters.imfilter(f, w23, "symmetric", "full"),
        }
        for key, val in cases.items():
            assert_parity(val, d[key], "float", atol=1e-12, rtol=0, name=key)


@needs_ref("interp2")
class TestL2Interp2:
    def test_grid_queries(self):
        d = ref("interp2")
        P = d["P"]
        U, V = d["Yq"] - 1, d["Xq"] - 1
        assert_parity(interp.interp2(P, U, V, "nearest"), d["Zn"], "float", atol=0, rtol=0, name="Zn")
        assert_parity(interp.interp2(P, U, V, "linear"), d["Zl"], "float", atol=1e-10, rtol=0, name="Zl")
        Zc = interp.interp2(P, U, V, "cubic")
        assert_parity(Zc[5:-5, 5:-5], d["Zc"][5:-5, 5:-5], "float", atol=1e-8, rtol=0, name="Zc interior")
        assert_parity(Zc, d["Zc"], "float", atol=1e-8, rtol=0, name="Zc incl. border")
        assert_parity(interp.interp2(P, U, V, "cubic", pad="replicate")[5:-5, 5:-5], d["Zc"][5:-5, 5:-5], "float",
                      atol=1e-8, rtol=0, name="Zc replicate-pad interior")

    def test_scattered_queries_with_ties_and_nan(self):
        d = ref("interp2")
        P = d["P"]
        u, v = d["uq"].ravel() - 1, d["vq"].ravel() - 1
        for m, key, tol in (("nearest", "Sn", 0), ("linear", "Sl", 1e-10), ("cubic", "Sc", 1e-8)):
            py = interp.interp2(P, u, v, m)
            ml = d[key].ravel()
            assert np.array_equal(np.isnan(py), np.isnan(ml)), f"{key}: NaN (out-of-range) pattern"
            assert_parity(py, ml, "float", atol=tol, rtol=0, name=key)
        assert np.isnan(d["Sn"].ravel()[6]) and np.isnan(d["Sn"].ravel()[7])   # 0.999 and 32.001 are outside

    def test_resize_vs_imresize_documented_approx(self):
        """`resize` is labelled approx vs imresize: same pixel-centre mapping, but imresize pads by replication
        *before* the kernel while interp2-style cubic uses quadratic extrapolation, and imresize antialiases when
        shrinking.  The interior of nearest / bilinear enlargements must agree exactly; errors are recorded."""
        d = ref("interp2")
        P = d["P"]
        # nearest / bilinear: full arrays, bit-identical (label `exact`)
        for s, kn, kl in ((4, "R4n", "R4l"), (2, "R2n", "R2l"), (0.5, "Rhn", "Rhl")):
            Rn, Rl = interp.resize(P, s, "nearest"), interp.resize(P, s, "bilinear")
            assert Rn.shape == d[kn].shape and Rl.shape == d[kl].shape
            assert_parity(Rn, d[kn], "float", atol=0, rtol=0, name=kn)
            assert_parity(Rl, d[kl], "float", atol=0, rtol=0, name=kl)
        # bicubic: interior (>= 2 source px from the border) identical; border differs (label `approx`)
        R4c, R2c = interp.resize(P, 4, "bicubic"), interp.resize(P, 2, "bicubic")
        assert_parity(R4c[8:-8, 8:-8], d["R4c"][8:-8, 8:-8], "float", atol=1e-9, rtol=0, name="R4c interior")
        assert_parity(R2c[4:-4, 4:-4], d["R2c"][4:-4, 4:-4], "float", atol=1e-9, rtol=0, name="R2c interior")
        e4 = np.abs(R4c - d["R4c"])
        bad = np.argwhere(e4 > 1e-9)
        n = e4.shape[0]
        dist = np.minimum.reduce([bad[:, 0], bad[:, 1], n - 1 - bad[:, 0], n - 1 - bad[:, 1]])
        assert dist.max() <= 5                       # every differing output px lies within 1.25 source px of the border
        # measured (R2025a): max 8.21, 2896 px > 1e-9, 742 px > 0.5, 921 px differ after uint8 rounding (5.62 %)
        assert e4.max() < 9 and int((e4 > 1e-9).sum()) <= 2896 and int((e4 > 0.5).sum()) <= 742


@needs_ref("compat")
class TestL2Compat:
    def test_round_im2uint8_im2double_imcomplement(self):
        d = ref("compat")
        assert np.array_equal(matlab_round(d["xr"].ravel()), d["rx"].ravel().astype(float))
        assert np.array_equal(im2uint8(d["v"].ravel()), d["u8"].ravel())
        assert np.array_equal(im2uint8(np.array([False, True, False])), d["u8b"].ravel())
        assert np.allclose(im2double(np.arange(256, dtype=np.uint8)), d["d8"].ravel(), atol=1e-15)
        assert np.allclose(im2double(np.array([0, 1, 65535], np.uint16)), d["d16"].ravel(), atol=1e-15)
        assert np.array_equal(im2double(np.array([False, True])), d["dl"].ravel())
        assert np.array_equal(imcomplement(np.array([0, 1, 127, 128, 254, 255], np.uint8)), d["ic8"].ravel())
        assert np.allclose(imcomplement(np.array([0, 0.25, 1, 300, -2.0])), d["icd"].ravel())
        assert np.array_equal(imcomplement(np.array([False, True])), d["icl"].ravel().astype(bool))
        assert np.array_equal(imcomplement(np.array([0, 65535, 1000], np.uint16)), d["ic16"].ravel())
        assert np.array_equal(imcomplement(np.array([-128, 0, 127], np.int8)), d["ici8"].ravel())

    def test_rgb2gray_double_and_imhist_variants(self):
        d = ref("compat")
        I = ref("histogram")["I"] if (REF / "histogram.mat").exists() else None
        if I is None:
            pytest.skip("histogram.mat missing")
        Gd = rgb2gray_matlab(im2double(I))[:300, :400]
        assert_parity(Gd, d["Gd"], "float", atol=1e-12, rtol=0, name="rgb2gray double")
        Gc = d["Gc"]
        h64, x64 = hist.imhist(Gc, 64)
        assert np.array_equal(h64, vec(d["h64"])) and np.allclose(x64, d["x64"].ravel())
        h256, x256 = hist.imhist(Gc)
        assert np.array_equal(h256, vec(d["h256"])) and np.allclose(x256, d["x256"].ravel())
        hd, xd = hist.imhist(im2double(Gc))
        assert np.array_equal(hd, vec(d["hd"])) and np.allclose(xd, d["xd"].ravel())
        hd100, xd100 = hist.imhist(im2double(Gc), 100)
        assert np.array_equal(hd100, vec(d["hd100"])) and np.allclose(xd100, d["xd100"].ravel())
        hl, xl = hist.imhist(Gc > 128, 2)
        assert np.array_equal(hl, vec(d["hl"])) and np.allclose(xl, d["xl"].ravel())
        h16, x16 = hist.imhist(Gc.astype(np.uint16) * 257)
        assert np.array_equal(h16, vec(d["h16"])) and np.allclose(x16, d["x16"].ravel())
        h5, x5 = hist.imhist(np.array([0, 1, 2, 3, 4, 5, 250, 251, 252, 253, 254, 255], np.uint8), 5)
        assert np.array_equal(h5, vec(d["h5"])) and np.allclose(x5, d["x5"].ravel())
        assert vec(d["sat"]).tolist() == [255, 200, 100] and vec(d["satm"]).tolist() == [0, 50]  # MATLAB saturates

    def test_ind2rgb(self):
        """MATLAB ind2rgb: integer classes are 0-based, double is 1-based, both clipped to [1, m]."""
        d = ref("compat")
        cmap = d["cmap"]
        assert np.array_equal(color.indexed_to_rgb(d["idx8"], cmap), d["rgb8"])                          # uint8, 7 clipped
        assert np.array_equal(color.indexed_to_rgb(d["idxd"], cmap, one_based=True), d["rgbd"])           # double 0 / 9 clipped
        assert np.array_equal(color.indexed_to_rgb(d["idx16"], cmap), d["rgb16"])
        assert np.array_equal(color.indexed_to_rgb(d["idxl"].astype(np.uint8), cmap), d["rgbl"])
        # non-integer double indices: MATLAB raises; the port truncates (documented, no parity claim)
        assert ref_str(d, "errf") != ""

    def test_imhist_logical_default_is_two_bins(self):
        """PORT DISCREPANCY (open item): MATLAB ``imhist(BW)`` on a logical image defaults to n = 2 bins
        (``[355; 119645]`` here); the port's default ``nbins=256`` puts the counts in bins 0 and 255."""
        d = ref("compat")
        hl, xl = hist.imhist(d["Gc"] > 128)
        assert hl.shape == (2,), f"imhist(logical) returned {hl.shape[0]} bins, MATLAB returns 2"
        assert np.array_equal(hl, vec(d["hl"])) and np.allclose(xl, d["xl"].ravel())


# ================================================================================================================
# L4 — numbers quoted in the text
# ================================================================================================================
class TestL4BookNumbers:
    @needs_image
    def test_fig_2_3_pixel(self, rgb):
        assert rgb[1075, 674].tolist() == [28, 76, 114]           # R(1076,675)=28, G=76, B=114 (p. 14)

    @needs_image
    def test_histogram_peaks(self, rgb):
        ch = ch2.channel_histograms(rgb)
        g = ch2.gray_histogram(rgb)
        assert (int(g["counts"].max()), int(g["counts"].argmax())) == (47840, 208)
        assert (int(ch["y_r"].max()), int(ch["y_r"].argmax())) == (77513, 5)
        assert (int(ch["y_g"].max()), int(ch["y_g"].argmax())) == (64866, 212)
        assert (int(ch["y_b"].max()), int(ch["y_b"].argmax())) == (71090, 227)
        assert g["counts"].max() < 80000 and ch["y_r"].max() < 80000   # Fig. 2.8 y-limit 8e4

    def test_fig_2_11_counts(self):
        ex = ch2.pixel_relationship_examples()
        assert ex["n_components_4"] == 5 and ex["n_components_8"] == 2

    def test_fig_2_12_2_13_values(self):
        ex = ch2.distance_fixture_examples()
        assert np.allclose(ex["D12"], ex["D12_book"], atol=1e-4)
        assert ex["C4"][0, 0] == 6 and ex["C8"][0, 0] == 3

    def test_p29_bitwise(self):
        assert ch2.set_operation_examples()["bitwise_57_and_207"] == 9

    def test_fig_2_20_2_21_sequences(self):
        res = ch2.chain_code_demo()
        assert res["x0y0_matlab"] == (4, 2) and res["b"].shape[0] == 19
        assert res["table"][0] == (4, 2, 0) and res["table"][1] == (4, 3, 1) and len(res["table"]) == 18
        seq = res["sequences"]
        assert seq["Original chain code"].tolist() == synth.FIG_2_21["original"].tolist()
        assert seq["First difference"].tolist() == synth.FIG_2_21["first_difference"].tolist()
        assert seq["Normalized chain code"].tolist() == synth.FIG_2_21["normalized"].tolist()
        assert seq["First difference (of normalized)"].tolist() == \
            synth.FIG_2_21["normalized_first_difference_of_normalized"].tolist()
        assert seq["Normalized first difference"].tolist() == synth.FIG_2_21["normalized_first_difference"].tolist()
        assert len(res["c4"].fcc) == 22 and len(res["c"].fcc) == 18   # Fig. 2.19(b)/(c) code lengths


# ================================================================================================================
# Scripts — every ported .m must run headless
# ================================================================================================================
SCRIPTS = ["color_image", "histogram", "distance_transform", "chain_diff", "image_types", "pixel_relationships",
           "convolution", "set_operations", "interpolation"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_runs(name: str, tmp_path: Path):
    script = ROOT / "scripts" / f"ch02_{name}.py"
    if name in ("color_image", "histogram", "image_types", "interpolation") and not RGB_PATH.exists():
        pytest.skip("rgb.JPG missing")
    out = tmp_path / name
    proc = subprocess.run([sys.executable, str(script), "--no-show", "--out", str(out)], capture_output=True,
                          text=True, cwd=str(ROOT), timeout=600)
    assert proc.returncode == 0, f"{script.name} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    files = list(out.glob("*"))
    assert files, f"{script.name} wrote nothing to {out}"
