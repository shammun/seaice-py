"""Chapter 3 verification — Ice Pixel Detection (Book pp. 37–58).

Evidence levels (verify-port skill):
  L1  synthetic truth (no external tools)            -> ``TestL1*``
  L2  parity vs MATLAB R2025a references              -> ``TestL2*`` (skip if ``reference/ch03/*.mat`` missing;
      regenerate with ``.venv/Scripts/python.exe reference/ch03/make_refs.py``)
  L4  numbers quoted in the book text                 -> ``TestL4BookNumbers``
  scripts: every ``scripts/ch03_*.py`` must run headless and exit 0 -> ``test_script_runs``

The MATLAB references were produced by the ORIGINAL ``Otsu.m`` / ``local_Otsu.m`` / ``separability.m`` / ``kmeans.m``
(verbatim copies with only the ``imread('...')`` / ``k = ...`` literals patched, see ``reference/ch03/make_refs.py``).
``t.jpg`` / ``ch3ice.jpg`` are not shipped: the references use the labelled substitutes (``2.jpg`` and the synthetic
uneven-illumination JPEG ``data/synthetic/ch03/t_uneven_2_g0.5_b40.jpg``).
"""
from __future__ import annotations

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from seaice import ch03_ice_pixel_detection as ch3
from seaice.core import clustering as cl
from seaice.core import synth
from seaice.core import threshold as th
from seaice.core.histogram import imhist
from seaice.core.io import load_image, read_image
from seaice.core.matlab_compat import rgb2gray_matlab
from tools.compare_arrays import assert_parity

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "ch03"
DATA = ROOT / "data" / "book" / "ch03"
T_SYNTH = ROOT / "data" / "synthetic" / "ch03" / "t_uneven_2_g0.5_b40.jpg"
IMAGES = {"1": "1.jpg", "2": "2.jpg", "test": "test.jpg"}

needs_image = pytest.mark.skipif(not all((DATA / f).exists() for f in IMAGES.values()),
                                 reason="data/book/ch03/{1,2,test}.jpg missing (run /setup-project)")


def needs_ref(name: str):
    return pytest.mark.skipif(not (REF / f"{name}.mat").exists(),
                              reason=f"reference/ch03/{name}.mat missing — run reference/ch03/make_refs.py (MATLAB)")


def ref(name: str) -> dict:
    d = loadmat(str(REF / f"{name}.mat"))
    return {k: v for k, v in d.items() if not k.startswith("__")}


def sc(x) -> float:
    """Scalar stored by MATLAB as a 1x1 array."""
    return float(np.asarray(x).ravel()[0])


def vec(x) -> np.ndarray:
    return np.asarray(x).ravel()


def ref_str(d: dict, key: str) -> str:
    v = np.asarray(d[key]).ravel()
    return "" if v.size == 0 else str(v[0])


_GRAY_CACHE: dict[str, np.ndarray] = {}


def _matlab_multithresh_pdf(A: np.ndarray) -> np.ndarray:
    """Test-side emulation of MATLAB ``multithresh``'s ``getpdf`` for uint8 input **including** the single-precision
    product of ``grayto8`` (``uint8(x * 255)`` evaluated in single, round half away from zero)."""
    A = np.asarray(A).ravel()
    minA, maxA = int(A.min()), int(A.max())
    x = (np.float32(A.astype(np.float64) - minA) / np.float32(maxA - minA)).astype(np.float32)
    y = (x * np.float32(255)).astype(np.float32)                        # single-precision product
    u8 = np.clip(np.floor(y.astype(np.float64) + 0.5), 0, 255).astype(np.uint8)
    counts = np.bincount(u8, minlength=256).astype(np.float64)
    return counts / counts.sum()


def gray_image(key: str) -> np.ndarray:
    """MATLAB ``rgb2gray(imread(<file>))`` of a shipped book image (cached: 4290x2856 each)."""
    if key not in _GRAY_CACHE:
        if key == "t":
            if not T_SYNTH.exists():
                pytest.skip("synthetic t.jpg substitute missing (run scripts/ch03_local_otsu.py)")
            _GRAY_CACHE[key] = rgb2gray_matlab(read_image(T_SYNTH))
        else:
            if not (DATA / IMAGES[key]).exists():
                pytest.skip(f"{IMAGES[key]} missing")
            rgb, _ = load_image("ch03", IMAGES[key], allow_fallback=False, verbose=False)
            _GRAY_CACHE[key] = rgb2gray_matlab(rgb)
    return _GRAY_CACHE[key]


# ================================================================================================================
# L1 — synthetic truth
# ================================================================================================================
class TestL1Otsu:
    def test_constant_image(self):
        img = np.full((10, 12), 77, dtype=np.uint8)
        assert th.graythresh(img) == (0.0, 0.0)                       # MATLAB graythresh -> level 0, em 0
        assert th.separability(img, 100) == 0.0                        # Eq. (3.22) lower bound
        cur = th.otsu_criterion(imhist(img)[0])
        assert cur.t_star == 0.0 and cur.eta_star == 0.0 and cur.sigma_G2 == 0.0

    def test_two_valued_image_eta_is_one(self):
        rng = np.random.default_rng(1)
        img = (rng.random((20, 25)) > 0.6).astype(np.uint8) * 255
        level, em = th.graythresh(img)
        assert em == pytest.approx(1.0, abs=1e-12)                     # Eq. (3.22) upper bound (L = 2 case)
        assert 255 * level == pytest.approx(127.0)                     # ties at t = 0..254 -> mean 127
        assert th.separability(img, 100) == pytest.approx(1.0, abs=1e-12)
        assert np.array_equal(th.im2bw(img, level), img == 255)

    def test_tie_averaging_half_integer_threshold(self):
        tie = np.where((np.indices((8, 8)).sum(axis=0) % 2) == 0, 100, 110).astype(np.uint8)
        level, em = th.graythresh(tie)
        assert 255 * level == pytest.approx(104.5)                     # mean(find(sigma_b == max)) = 104.5
        assert em == pytest.approx(1.0)
        assert np.array_equal(th.im2bw(tie, level), tie == 110)        # I > 104.5

    def test_otsu_criterion_identities(self):
        img = synth.bimodal_image(seed=0)
        counts, _ = imhist(img)
        cur = th.otsu_criterion(counts)
        both = (cur.P0 > 0) & (cur.P1 > 0)
        assert np.allclose(cur.P0 + cur.P1, 1.0, atol=1e-12)                                  # Eq. (3.11)
        assert np.allclose((cur.P0 * cur.m0 + cur.P1 * cur.m1)[both], cur.mG, atol=1e-9)      # Eq. (3.10)
        assert np.allclose((cur.sigma_W2 + cur.sigma_B2)[both], cur.sigma_G2, atol=1e-8)      # Eq. (3.18)
        assert np.all((cur.eta[both] >= -1e-12) & (cur.eta[both] <= 1 + 1e-12))              # Eq. (3.22)
        level, em = th.graythresh(img)
        assert cur.t_star == 255 * level and cur.eta_star == pytest.approx(em, abs=1e-12)     # Eq. (3.21)
        assert cur.eta_star >= np.nanmax(cur.eta[:255]) - 1e-12
        assert th.separability(img, cur.t_star) == pytest.approx(em, abs=1e-12)

    def test_bimodal_synthetic_truth(self):
        img = synth.bimodal_image(seed=0, dark=60.0, bright=190.0, sigma=12.0, ice_fraction=0.4)
        res = ch3.otsu_segmentation(img)
        assert 100 <= res["threshold"] <= 150                          # valley between the two modes
        assert abs(res["ic"] - 0.4) < 0.02                             # rectangle covers 40 % of the area
        assert res["bw"].dtype == np.bool_ and res["em"] > 0.9
        assert th.ice_concentration(res["bw"]) == res["ic"]

    def test_im2bw_strict_and_dtypes(self):
        u8 = np.array([[100, 101, 0, 255]], dtype=np.uint8)
        assert th.im2bw(u8, 100 / 255).tolist() == [[False, True, False, True]]   # strict '>'
        assert th.im2bw(np.array([[0.5, 0.51]]), 0.5).tolist() == [[False, True]]
        assert th.im2bw(np.array([[32767, 32768]], dtype=np.uint16), 0.5).tolist() == [[False, True]]
        with pytest.warns(RuntimeWarning):
            assert th.im2bw(np.array([[True, False]])).tolist() == [[True, False]]
        with pytest.raises(ValueError):
            th.im2bw(u8, 1.5)
        rgb = np.stack([u8, u8, u8], axis=-1)
        assert np.array_equal(th.im2bw(rgb, 100 / 255), th.im2bw(u8, 100 / 255))

    def test_imquantize(self):
        ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
        q = th.imquantize(ramp, [50, 200])                             # Eq. (3.23)
        assert q.min() == 1 and q.max() == 3
        assert (q == 1).sum() == 51 and (q == 2).sum() == 150 and (q == 3).sum() == 55
        qv = th.imquantize(ramp, [50, 200], values=np.array([7, 8, 9]))
        assert qv[0, 0] == 7 and qv[15, 15] == 9
        assert th.imquantize(np.array([[0.2, 0.7]]), 0.5).tolist() == [[1, 2]]
        with pytest.raises(ValueError):
            th.imquantize(ramp, [200, 50])
        with pytest.raises(ValueError):
            th.imquantize(ramp, [50, 200], values=[1, 2])

    def test_multithresh_synthetic(self):
        rng = np.random.default_rng(2)
        img = rng.choice(np.array([20, 120, 220], dtype=np.uint8), size=(30, 30))
        t2, m2 = th.multithresh(img, 2)
        assert t2.dtype == np.uint8 and 20 <= t2[0] < 120 and 120 <= t2[1] < 220
        assert np.array_equal(th.imquantize(img, t2), 1 + (img > 20) + (img > 120))
        assert m2 == pytest.approx(1.0, abs=1e-12)                     # perfectly separable -> Eq. (3.28) = 1
        two = np.where(rng.random((10, 10)) > 0.5, 0, 255).astype(np.uint8)
        t1, m1 = th.multithresh(two, 1)
        assert 0 <= t1[0] < 255 and m1 == pytest.approx(1.0, abs=1e-12)
        with pytest.raises(NotImplementedError):
            th.multithresh(img, 4)
        with pytest.raises(ValueError):
            th.multithresh(img, 0)
        with pytest.warns(RuntimeWarning):
            tc, mc = th.multithresh(np.full((5, 5), 9, dtype=np.uint8), 1)
        assert tc.tolist() == [9] and mc == 0.0

    def test_block_otsu_synthetic(self):
        rng = np.random.default_rng(3)
        left = rng.choice(np.array([30, 90], dtype=np.uint8), size=(40, 30))
        right = rng.choice(np.array([150, 210], dtype=np.uint8), size=(40, 30))
        img = np.hstack([left, right])
        blk = th.block_otsu(img, 1, 2)
        assert 30 <= blk.thresholds[0] < 90 and 150 <= blk.thresholds[1] < 210
        assert blk.counts.tolist() == [int((left == 90).sum()), int((right == 210).sum())]
        assert np.allclose(blk.ic_local, blk.counts / left.size) and blk.ic == blk.counts.sum() / img.size
        assert np.array_equal(blk.bw[:, :30], left == 90) and np.array_equal(blk.bw[:, 30:], right == 210)
        assert th.graythresh(img)[0] * 255 > 90                        # global Otsu would merge the two dark levels
        with pytest.raises(ValueError):
            th.block_otsu(img, 3, 2)

    def test_class_helpers(self):
        lab = np.array([[1, 1, 2], [3, 1, 2]])
        g = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
        assert np.allclose(th.class_coverage(lab, 3), [3 / 6, 2 / 6, 1 / 6])
        assert np.allclose(th.class_mean_intensity(g, lab, 3), [(10 + 20 + 50) / 3, 45.0, 40.0])
        assert np.isnan(th.class_mean_intensity(g, lab, 4)[3])
        assert th.ice_concentration(lab == 1) == 0.5

    def test_stale_mean_intensity_emulation(self):
        # column-major fill: class 1 = 4 px, class 2 = 2 px -> s keeps the last two class-1 values
        g = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.uint8)          # column-major order: 1 2 3 4 5 6
        lab = np.array([[1, 1, 2], [1, 1, 2]])
        stale = ch3.stale_mean_intensity(g, lab, 2)
        assert stale[0] == pytest.approx((1 + 2 + 3 + 4) / 4)
        assert stale[1] == pytest.approx((5 + 6 + 3 + 4) / 2)         # sum(s(:)) / length(p) with stale 3, 4
        assert np.allclose(th.class_mean_intensity(g, lab, 2), [2.5, 5.5])


class TestL1Clustering:
    def test_pairwise_distance_hand_values(self):
        a, b = np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]])
        assert cl.pairwise_distance(a, b, "euclidean")[0, 0] == 5.0                     # Eq. (3.29)
        assert cl.pairwise_distance(a, b, "sqeuclidean")[0, 0] == 25.0                  # Eq. (3.30)
        assert cl.pairwise_distance(a, b, "chebyshev")[0, 0] == 4.0                     # Eq. (3.31)
        assert cl.pairwise_distance(a, b, "cityblock")[0, 0] == 7.0                     # Eq. (3.32)
        x, y = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])
        assert cl.pairwise_distance(x, y, "cosine")[0, 0] == pytest.approx(1.0)         # 1 - cos (Eq. 3.33)
        assert cl.pairwise_distance(x, 2 * x, "cosine")[0, 0] == pytest.approx(0.0)
        d = cl.pairwise_distance(a, b, "mahalanobis", cov=np.eye(2))[0, 0]              # Eq. (3.34), S = I
        assert d == pytest.approx(5.0)
        assert cl.pairwise_distance(np.array([1.0, 4.0]), np.array([2.0]), "cityblock").tolist() == [[1.0], [2.0]]
        with pytest.raises(ValueError):
            cl.pairwise_distance(a, b, "minkowski")

    def test_kmeans_lloyd_separated_blobs(self):
        X = synth.two_clusters_2d(seed=0)
        res = cl.kmeans_lloyd(X, 2, init="random", seed=0)
        assert res.converged and res.n_iter >= 1
        c = res.centers[np.argsort(res.centers[:, 0])]
        assert np.allclose(c, [[2.5, 2.5], [6.5, 6.5]], atol=0.6)     # Eq. (3.37) cluster means
        assert len(set(res.labels[:15])) == 1 and len(set(res.labels[15:])) == 1 and res.labels[0] != res.labels[-1]
        J = [h[2] for h in res.history]
        assert all(J[i + 1] <= J[i] + 1e-12 for i in range(len(J) - 1))  # J non-increasing (Eq. 3.35)
        assert res.J == pytest.approx(cl.objective_J(X, res.centers, res.labels))
        for init in ("equal", "kmeans++", np.array([[1.0, 1.0], [9.0, 9.0]])):
            r = cl.kmeans_lloyd(X, 2, init=init, seed=0)
            assert len(set(r.labels[:15])) == 1 and len(set(r.labels[15:])) == 1 and r.labels[0] != r.labels[-1]
        with pytest.raises(ValueError):
            cl.kmeans_lloyd(X, 40)
        with pytest.raises(ValueError):
            cl.kmeans_lloyd(X, 2, init="bogus")

    def test_kmeans_gray_equals_lloyd_with_same_init(self):
        img = synth.bimodal_image(seed=1)
        res = cl.kmeans_gray(img, 2)
        mi = float(img.min())
        m = int(img.max() - mi + 1) + 1
        init = np.arange(1, 3) * m / 3 + mi - 1.0                     # kmeans.m equal-division start, gray units
        ll = cl.kmeans_lloyd(img.ravel(order="F").astype(float), 2, init=init.reshape(2, 1))
        assert np.allclose(np.sort(res.centroids), np.sort(ll.centers.ravel()), atol=1e-9)
        assert np.array_equal(res.mask.ravel(order="F"), ll.labels + 1)  # consistent Eqs. (3.36)-(3.37)
        assert res.converged and ll.converged

    def test_kmeans_gray_shift_bug_semantics(self):
        img = np.array([[50, 100, 200]], dtype=np.uint8)              # min 50 -> shifted values 1, 51, 151
        ok = cl.kmeans_gray(img, 2, shift_bug=False)
        bug = cl.kmeans_gray(img, 2, shift_bug=True)
        assert np.allclose(ok.centroids_shifted, [26.0, 151.0]) and np.allclose(ok.centroids, [75.0, 200.0])
        assert ok.mask.tolist() == [[1, 1, 2]]                         # 100 is nearer 75 than 200
        assert bug.mask.tolist() == [[1, 2, 2]]                        # unshifted 100 vs shifted mu 26 / 151
        assert np.allclose(bug.centroids, ok.centroids) and bug.offset == 50.0
        assert np.allclose(ok.coverage, [2 / 3, 1 / 3]) and ok.ic == pytest.approx(1 / 3)
        assert np.allclose(ok.average_intensity, [75.0, 200.0]) and np.allclose(ok.mask1, ok.mask / 2)

    def test_kmeans_gray_empty_cluster_raises(self):
        img = np.where(np.indices((6, 6)).sum(axis=0) % 2 == 0, 0, 255).astype(np.uint8)
        with pytest.raises(ValueError, match="empty"):
            cl.kmeans_gray(img, 3)                                      # MATLAB: NaN centroid -> infinite loop
        with pytest.raises(ValueError):
            cl.kmeans_gray(img, 0)
        with pytest.raises(ValueError):
            cl.kmeans_gray(np.array([[0.0, 0.5, 1.0, 7.0]]), 2)       # non-integer spacing: h(ima(i)) undefined


class TestL1Synth:
    def test_uneven_illumination(self):
        g = np.full((20, 30), 100, dtype=np.uint8)
        u = synth.uneven_illumination(g, gain=0.5, bias=40.0, axis=1)
        assert u.dtype == np.uint8 and u.shape == g.shape
        assert u[0, 0] == 190 and u[0, -1] == 10                       # 100*1.5+40 / 100*0.5-40
        assert np.all(np.diff(u[0].astype(int)) <= 0)
        rgb = np.stack([g, g, g], axis=-1)
        assert synth.uneven_illumination(rgb, gain=0.5, bias=40.0).shape == rgb.shape
        assert synth.uneven_illumination(g, kind="spot").shape == g.shape
        with pytest.raises(ValueError):
            synth.uneven_illumination(g, kind="bogus")

    def test_two_clusters_and_bimodal(self):
        X = synth.two_clusters_2d(seed=0)
        assert X.shape == (30, 2) and X.min() >= 0 and X.max() <= 10
        Xo = synth.two_clusters_2d(seed=0, outlier=(9.8, 0.4))
        assert Xo.shape == (31, 2) and np.array_equal(Xo[:-1], X) and Xo[-1].tolist() == [9.8, 0.4]
        img = synth.bimodal_image(seed=0)
        assert img.dtype == np.uint8 and img.shape == (120, 160)
        assert np.array_equal(img, synth.bimodal_image(seed=0))        # deterministic


class TestL1Chapter:
    def test_separability_script_vs_book_convention(self):
        img = synth.bimodal_image(seed=0)
        res = ch3.separability_script(img, k=108)
        assert res["t_eval"] == 107
        assert res["eta"] == pytest.approx(th.separability(img, 107), abs=1e-12)   # C0 = bins 1..k = levels 0..k-1
        assert res["eta_book"] == pytest.approx(th.separability(img, 108), abs=1e-12)
        assert res["IC"] == pytest.approx(float((img > 108).mean()))
        assert res["bw"].dtype == np.float64 and set(np.unique(res["bw"])) <= {0.0, 1.0}
        assert res["mg"] == pytest.approx(res["curves"].mG + 1.0)     # 1-based bin index as intensity
        assert res["eta_star"] >= res["eta"] and res["t_star"] == 255 * th.graythresh(img)[0]

    def test_fixed_threshold_and_histogram_plot(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img = synth.bimodal_image(seed=0)
        bw, ic = ch3.fixed_threshold(img, 125)
        assert np.array_equal(bw, img > 125) and ic == float(bw.mean())
        fig, ax = plt.subplots()
        ch3.plot_histogram_with_threshold(ax, img, 125)
        assert len(ax.lines) >= 1 and ax.get_xlim() == (-0.5, 255.5)
        plt.close(fig)

    def test_local_otsu_titles_num2str(self):
        rng = np.random.default_rng(4)
        img = rng.integers(0, 256, size=(40, 60)).astype(np.uint8)
        res = ch3.local_otsu(img, 2, 3)
        assert len(res["titles"]) == 6 and res["titles"][0].startswith("IC=") and "Threshold=" in res["titles"][0]
        # MATLAB num2str(x) for a non-integer double scalar uses floor(log10(|x|)) + 5 significant digits
        # ('%.6g' for 10 <= x < 100): the book's Fig. 3.4(c) titles read "IC =73.8472%", "IC =86.496%"
        assert ch3._num2str(92.0) == "92" and ch3._num2str(107.5) == "107.5"
        assert ch3._num2str(3.14159265) == "3.1416" and ch3._num2str(0.5) == "0.5"
        assert ch3._num2str(73.8472) == "73.8472" and ch3._num2str(86.496) == "86.496"
        assert ch3._num2str(20.9869) == "20.9869" and ch3._num2str(123.4567) == "123.4567"
        assert ch3._num2str(0.123456789) == "0.12346" and ch3._num2str(1234.56789) == "1234.5679"
        # (all ten strings verified in MATLAB R2025a: reference/ch03/local_otsu.mat, variable n2s)
        assert res["ic"] == th.block_otsu(img, 2, 3).ic and res["global_bw"].shape == img.shape

    def test_multi_otsu_and_kmeans_segmentation_dicts(self):
        rng = np.random.default_rng(5)
        img = rng.choice(np.array([20, 120, 220], dtype=np.uint8), size=(30, 30))
        mo = ch3.multi_otsu_segmentation(img, 2)
        assert np.allclose(mo["coverage"], [(img == v).mean() for v in (20, 120, 220)])
        assert np.allclose(mo["average_intensity"], [20, 120, 220]) and mo["ic"] == pytest.approx(1 - mo["coverage"][0])
        km = ch3.kmeans_segmentation(img, 3, shift_bug=False)
        assert np.allclose(np.sort(km["centroids"]), [20, 120, 220]) and km["ic"] == pytest.approx(1 - (img == 20).mean())
        assert np.allclose(km["boundaries"], [70, 170])

    def test_kmeans_demo_2d(self):
        demo = ch3.kmeans_demo_2d(seed=0)
        assert demo["result"].converged and demo["X"].shape == (30, 2)
        assert set(demo["distances"]) == {"euclidean", "sqeuclidean", "chebyshev", "cityblock", "cosine", "mahalanobis"}
        assert demo["distances"]["euclidean"] ** 2 == pytest.approx(demo["distances"]["sqeuclidean"])
        ro = demo["result_outlier"]
        assert ro.labels[-1] != ro.labels[0] and (ro.labels[:-1] == ro.labels[0]).sum() >= 25   # outlier isolates


# ================================================================================================================
# L2 — parity vs MATLAB R2025a (reference/ch03/*.mat)
# ================================================================================================================
@needs_ref("compat")
class TestL2Compat:
    @pytest.fixture(scope="class")
    def d(self):
        return ref("compat")

    @pytest.mark.parametrize("name", ["Cst", "Two", "Tie", "U16", "Dbl", "Ramp", "Four", "Rnd", "I16"])
    def test_graythresh(self, d, name):
        level, em = th.graythresh(d[name])
        assert level == sc(d[f"lv_{name}"]), name                       # bit-identical level
        assert em == pytest.approx(sc(d[f"em_{name}"]), abs=1e-12), name

    def test_graythresh_double_out_of_range_and_otsuthresh(self, d):
        level, em = th.graythresh(d["Dbl"] * 255)                      # im2uint8 clips floats > 1
        assert level == sc(d["lv_Dbl255"]) and em == pytest.approx(sc(d["em_Dbl255"]), abs=1e-12)
        t, em = th.otsuthresh(np.bincount(d["Rnd"].ravel(), minlength=256))
        assert t == sc(d["ot_t"]) and em == pytest.approx(sc(d["ot_em"]), abs=1e-12)
        t10, em10 = th.otsuthresh(vec(d["Hc"]))
        assert t10 == sc(d["ot10_t"]) and em10 == pytest.approx(sc(d["ot10_em"]), abs=1e-12)

    def test_im2bw_every_level_and_classes(self, d):
        nbw = np.array([np.count_nonzero(th.im2bw(d["Rnd"], i / 255)) for i in range(256)])
        assert np.array_equal(nbw, vec(d["nbw"]))
        for key, arr, lvk in (("bw_tie", "Tie", "lv_Tie"), ("bw_u16", "U16", "lv_U16"), ("bw_dbl", "Dbl", "lv_Dbl"),
                              ("bw_i16", "I16", "lv_I16")):
            assert_parity(th.im2bw(d[arr], sc(d[lvk])), d[key], "binary", name=key)
        assert_parity(th.im2bw(d["Rnd"]), d["bw_default"], "binary", name="bw_default")
        rgb = np.stack([d["Rnd"]] * 3, axis=-1)
        assert_parity(th.im2bw(rgb, sc(d["lv_Rnd"])), d["bw_rgb"], "binary", name="bw_rgb")

    @pytest.mark.parametrize("name,N", [("Ramp", 1), ("Ramp", 2), ("Rnd", 1), ("Rnd", 2), ("U16", 1), ("U16", 2),
                                        ("Dbl", 1), ("Dbl", 2), ("Four", 1), ("Four", 2), ("Tie", 1)])
    def test_multithresh_n_le_2(self, d, name, N):
        t, m = th.multithresh(d[name], N)
        r = vec(d[f"mt{N}_{name}"])
        assert t.dtype == r.dtype, (t.dtype, r.dtype)
        if np.issubdtype(r.dtype, np.floating):
            assert np.allclose(t, r, atol=1e-12, rtol=0), (name, N, t, r)
        else:
            assert np.array_equal(t, r), (name, N, t, r)
        assert m == pytest.approx(sc(d[f"mm{N}_{name}"]), abs=1e-12)

    @pytest.mark.parametrize("name,N", [("Two", 2), ("Cst", 1), ("Cst", 2)])
    def test_multithresh_degenerate(self, d, name, N):
        with pytest.warns(RuntimeWarning):
            t, m = th.multithresh(d[name], N)
        assert np.array_equal(t, vec(d[f"mt{N}_{name}"])) and m == 0.0 == sc(d[f"mm{N}_{name}"])

    def test_multithresh_n3_reimplemented(self, d):
        """N = 3: MATLAB uses fminsearch (local search, TolX = 1); the port maximises Eq. (3.27) exhaustively."""
        t, m = th.multithresh(d["Ramp"], 3)
        assert np.array_equal(t, vec(d["mt3_Ramp"])) and m == pytest.approx(sc(d["mm3_Ramp"]), abs=1e-12)
        t, m = th.multithresh(d["Rnd"], 3)
        # plateau: the histogram is empty between the two modes, so every t2 in the gap has the same sigma_B^2;
        # MATLAB stops at 135, the port averages the tied bins (124) — the criterion value is identical
        assert m == pytest.approx(sc(d["mm3_Rnd"]), abs=1e-12)
        assert t[0] == vec(d["mt3_Rnd"])[0] and t[2] == vec(d["mt3_Rnd"])[2]
        assert np.array_equal(th.imquantize(d["Rnd"], t), th.imquantize(d["Rnd"], vec(d["mt3_Rnd"])))
        # 4-valued image, N = 3: MATLAB's fminsearch never finds a finite objective and returns its initial guess
        # with metric -Inf; the exhaustive search returns the perfect split (metric 1)
        t, m = th.multithresh(d["Four"], 3)
        assert np.isinf(sc(d["mm3_Four"])) and m == pytest.approx(1.0, abs=1e-12)
        assert np.array_equal(np.unique(th.imquantize(d["Four"], t)), [1, 2, 3, 4])

    @pytest.mark.xfail(strict=True, reason="int16 input: MATLAB multithresh.m line 274 evaluates single(A - minA) "
                                          "in saturating int16 arithmetic; the port normalises in float64 "
                                          "(int16 images do not occur in the book) — documented deviation")
    def test_multithresh_int16_known_divergence(self, d):
        t, m = th.multithresh(d["I16"], 2)
        assert np.array_equal(t, vec(d["mt2_I16"]))

    def test_imquantize_and_im2uint8(self, d):
        assert np.array_equal(th.imquantize(d["Ramp"], vec(d["mt2_Ramp"])), d["q_Ramp"])
        assert np.array_equal(th.imquantize(d["Ramp"], vec(d["mt2_Ramp"]), np.array([10, 20, 30])), d["qv_Ramp"])
        assert np.array_equal(th.imquantize(d["Dbl"], vec(d["mt2_Dbl"])), d["q_Dbl"])
        assert np.array_equal(th.imquantize(d["Rnd"], vec(d["mt3_Rnd"])), d["q_Rnd3"])
        assert np.array_equal(th.imquantize(d["Rnd"], vec(d["mt1_Rnd"])), d["q_Rnd1"])
        u = th._im2uint8_any(np.arange(65536, dtype=np.uint16))
        assert np.array_equal(u, vec(d["u16to8"])) and u.dtype == np.uint8   # round(v / 257) for all 65536 values
        assert np.array_equal(th._im2uint8_any(d["I16"]), d["i16to8"])
        assert np.array_equal(imhist(th._im2uint8_any(d["Rnd"].ravel()), 256)[0], vec(d["gt_Rnd_pdf"]))


@needs_image
class TestL2Otsu:
    """``Otsu.m`` run verbatim on test.jpg / 1.jpg / 2.jpg (Figs. 3.9(b)–3.11(b), 3.12(a), Tables 3.1–3.3)."""

    @pytest.fixture(scope="class", params=["test", "1", "2"])
    def case(self, request):
        key = request.param
        if not (REF / f"otsu_{key}.mat").exists():
            pytest.skip(f"reference/ch03/otsu_{key}.mat missing — run make_refs.py")
        return key, ref(f"otsu_{key}"), gray_image(key)

    def test_jpeg_decode_and_rgb2gray(self, case):
        key, d, I = case
        assert I.shape == d["I"].shape == (2856, 4290)
        n = int((I != d["I"]).sum())
        assert n == 0, f"{key}: {n} gray pixels differ from MATLAB imread+rgb2gray"

    def test_graythresh_im2bw_ic(self, case):
        key, d, I = case
        res = ch3.otsu_segmentation(I)
        assert res["level"] == sc(d["t"]) == sc(d["t2"])
        assert res["em"] == pytest.approx(sc(d["em"]), abs=1e-12)
        assert res["threshold"] == 255 * sc(d["t"])
        m = assert_parity(res["bw"], d["bw"], "binary", name=f"bw_{key}")
        assert m["n_diff"] == 0
        assert res["ic"] == pytest.approx(sc(d["ic"]), abs=1e-12)

    def test_multithresh_imquantize_coverage(self, case):
        key, d, I = case
        mo = ch3.multi_otsu_segmentation(I, 2)
        assert mo["thresh"].dtype == np.uint8 and np.array_equal(mo["thresh"], vec(d["thresh"]))
        assert np.array_equal(mo["thresh"], vec(d["th2"]))
        assert np.array_equal(mo["seg"], d["seg"])                     # uint8 class image, 1..3
        assert np.allclose(mo["coverage"], vec(d["coverage"]), atol=1e-12)
        assert np.array_equal([int((mo["seg"] == i).sum()) for i in (1, 2, 3)], vec(d["ncls"]))
        assert np.array_equal(th.imquantize(I, mo["thresh"], np.array([0, 128, 255], dtype=np.uint8)), d["seg_vals"])
        t1, _ = th.multithresh(I, 1)
        assert np.array_equal(t1, vec(d["th1"]))
        assert t1[0] == pytest.approx(255 * sc(d["t"]), abs=1.0)      # multithresh(I,1) ~ graythresh (grayto8 rebinning)

    def test_multithresh_metric_strict(self, case):
        """Eq. (3.28) metric must equal MATLAB's to 1e-12 on all three images (test.jpg was 1.1e-4 off before the
        single-precision ``x * 255`` fix in ``_multithresh_pdf`` — see test_multithresh_metric_single_precision_hypothesis)."""
        key, d, I = case
        _, m2 = th.multithresh(I, 2)
        assert m2 == pytest.approx(sc(d["metric"]), abs=1e-12), (key, m2, sc(d["metric"]))

    def test_multithresh_metric_single_precision_hypothesis(self, case):
        """MATLAB's grayto8 multiplies the *single* normalised value by 255 in single precision; the port multiplies
        in float64.  For test.jpg (range 21..255) gray 216 -> x = float32(195/234) -> x*255 = 212.4999949, which
        rounds to the float32 212.5 and then to 213 in MATLAB but to 212 in float64.  Emulating the single product
        on the test side must reproduce MATLAB's metric exactly (N = 2) and MATLAB's N = 3 objective ordering."""
        key, d, I = case
        p = _matlab_multithresh_pdf(I)
        k = np.arange(1, 257, dtype=np.float64)
        omega, mu = np.cumsum(p), np.cumsum(p * k)
        mu_t = mu[-1]
        sb = th._sigma_b2_matrix_n2(omega, mu, mu_t)
        metric_ml_emulated = float(np.nanmax(sb) / np.sum(p * (k - mu_t) ** 2))
        assert metric_ml_emulated == pytest.approx(sc(d["metric"]), abs=1e-12), (key, metric_ml_emulated, sc(d["metric"]))
        # the port now forms x * 255 in single as well, so its pdf must equal the emulation bin for bin on every
        # image (before the fix test.jpg's gray 216 landed in bin 212 instead of MATLAB's 213)
        p_port, minA, maxA = th._multithresh_pdf(I)
        moved = np.flatnonzero(p_port != p)
        assert moved.size == 0, (key, moved.tolist())
        # N = 3 (reimplemented, exhaustive vs fminsearch): under MATLAB's own pdf the port's maximiser is not worse
        best, t_raw = th._sigma_b2_exhaustive_n3(omega, mu, mu_t)
        metric3_port_on_ml_pdf = best / np.sum(p * (k - mu_t) ** 2)
        assert metric3_port_on_ml_pdf >= sc(d["metric3"]) - 1e-12, (key, metric3_port_on_ml_pdf, sc(d["metric3"]))
        # (the N = 3 thresholds themselves may differ: fminsearch is a local search — 1.jpg: MATLAB 50/101/165 vs
        # exhaustive 49/86/160; test.jpg and 2.jpg coincide.  Values tabulated in reports/ch03_verification.md.)

    def test_average_intensity_stale_buffer(self, case):
        key, d, I = case
        mo = ch3.multi_otsu_segmentation(I, 2)
        ml = vec(d["average_intensity"])
        assert np.allclose(mo["average_intensity_script"], ml, atol=1e-9)   # the script's stale 's' emulated
        if key == "test":  # class sizes increase (3.5 < 42.1 < 54.4 %) -> the bug is harmless, Table 3.3 correct
            assert np.allclose(mo["average_intensity"], ml, atol=1e-9)
        else:  # 1.jpg / 2.jpg: class 2 or 3 is smaller than an earlier class -> MATLAB prints wrong means
            assert not np.allclose(mo["average_intensity"], ml, atol=1e-6)
            assert np.all(mo["average_intensity"] > 0)


@needs_image
class TestL2KMeans:
    """``kmeans.m`` (patched k and file name only, plus an iteration counter) on the three images."""

    @pytest.fixture(scope="class", params=[("test", 2), ("test", 3), ("1", 2), ("1", 3), ("2", 2), ("2", 3)],
                    ids=lambda p: f"{p[0]}_k{p[1]}")
    def case(self, request):
        key, k = request.param
        if not (REF / f"kmeans_{key}_k{k}.mat").exists():
            pytest.skip(f"reference/ch03/kmeans_{key}_k{k}.mat missing — run make_refs.py")
        return key, k, ref(f"kmeans_{key}_k{k}"), gray_image(key)

    def test_centroids_iterations_mask(self, case):
        key, k, d, I = case
        res = cl.kmeans_gray(I, k, shift_bug=True)
        assert np.allclose(res.centroids_shifted, vec(d["mu"]), atol=1e-9), (res.centroids_shifted, vec(d["mu"]))
        assert res.n_iter == int(sc(d["n_iter"]))
        assert res.offset == sc(d["mi"])
        assert np.array_equal(np.bincount((I.astype(int) - int(res.offset) + 1).ravel(), minlength=int(sc(d["m"])) + 1)[1:],
                              vec(d["h"]))                              # histogram h(1..m)
        n_diff = int((res.mask != d["mask"].astype(int)).sum())
        assert n_diff == 0, f"{key} k={k}: {n_diff} mask pixels differ"
        assert np.array_equal(res.counts, vec(d["n"]))
        assert np.allclose(res.coverage, vec(d["ic"]), atol=1e-12)

    def test_average_intensity_stale_buffer(self, case):
        key, k, d, I = case
        seg = ch3.kmeans_segmentation(I, k, shift_bug=True)
        ml = vec(d["average_intensity"])
        assert np.allclose(seg["average_intensity_script"], ml, atol=1e-9)
        sizes = seg["counts"]
        if np.all(np.diff(sizes) > 0):   # sizes increasing -> stale buffer harmless (test.jpg, both k)
            assert np.allclose(seg["average_intensity"], ml, atol=1e-9)
        else:
            assert not np.allclose(seg["average_intensity"], ml, atol=1e-6)

    def test_correct_units_differ_only_between_boundaries(self, case):
        key, k, d, I = case
        ok = cl.kmeans_gray(I, k, shift_bug=False)
        bug = d["mask"].astype(int)
        assert np.allclose(ok.centroids_shifted, vec(d["mu"]), atol=1e-9)   # same iteration, only the mask differs
        assert np.all(np.diff(ok.centroids_shifted) > 0)                # centroids stay ordered (index = class)
        diff = ok.mask != bug
        assert diff.any()                                               # the bug matters on every image
        assert np.all(ok.mask[diff] == bug[diff] - 1)                  # the bug pushes pixels to the next brighter class
        mids = (ok.centroids_shifted[:-1] + ok.centroids_shifted[1:]) / 2   # class boundaries in shifted units
        c = ok.mask[diff] - 1                                           # 0-based class of the consistent mask
        g = I[diff].astype(float)
        # the script compares the UNSHIFTED gray value with the shifted boundary: affected pixels lie in
        # (mid, mid + (min - 1)] where `min - 1` is the shift (20 on test.jpg, 28 on 1.jpg, 12 on 2.jpg)
        assert np.all((g > mids[c]) & (g <= mids[c] + (ok.offset - 1.0) + 1e-9))


@needs_ref("local_otsu")
class TestL2LocalOtsu:
    """``local_Otsu.m`` verbatim on the synthetic substitute for the unshipped ``t.jpg`` (Fig. 3.4)."""

    @pytest.fixture(scope="class")
    def case(self):
        return ref("local_otsu"), gray_image("t")

    def test_input_identical(self, case):
        d, I = case
        assert I.shape == d["I"].shape and int((I != d["I"]).sum()) == 0

    def test_block_thresholds_counts_ic(self, case):
        d, I = case
        res = ch3.local_otsu(I, int(sc(d["n_r"])), int(sc(d["n_c"])))
        assert np.array_equal(res["thresholds"], vec(d["thresh"]))     # th = t*255 (float, exact)
        assert np.array_equal(res["levels"], vec(d["levels"]))
        assert np.array_equal(res["counts"], vec(d["num"]))
        assert np.allclose(res["ic_local"], vec(d["IC_local"]), atol=1e-12)
        assert res["ic"] == pytest.approx(sc(d["IC"]), abs=1e-12)
        m = assert_parity(res["bw"], d["bwl"], "binary", name="local_otsu_bw")
        assert m["n_diff"] == 0
        assert res["global_threshold"] == 255 * sc(d["tg"]) and res["global_ic"] == pytest.approx(sc(d["ICg"]), abs=1e-12)
        assert assert_parity(res["global_bw"], d["bwg"], "binary", name="global_bw")["n_diff"] == 0

    def test_subplot_titles_num2str(self, case):
        """The 2x3 subplot titles of local_Otsu.m: ``num2str(ic0)`` / ``num2str(th)`` (cosmetic; the book prints
        the MATLAB strings, e.g. "IC =73.8472%")."""
        d, I = case
        res = ch3.local_otsu(I, int(sc(d["n_r"])), int(sc(d["n_c"])))
        ic_str = ref_str(d, "tstr").split("|")
        th_str = ref_str(d, "thstr").split("|")
        for title, a, b in zip(res["titles"], ic_str, th_str):
            assert title == f"IC={a}%\nThreshold={b}", (title, a, b)


@needs_ref("separability_108")
class TestL2Separability:
    """``separability.m`` verbatim (k = 108) and with k = 107 on ``2.jpg`` (substitute for ``ch3ice.jpg``)."""

    @pytest.mark.parametrize("k", [108, 107])
    def test_script_variables(self, k):
        if not (REF / f"separability_{k}.mat").exists():
            pytest.skip("reference missing")
        d = ref(f"separability_{k}")
        I = gray_image("2")
        assert int((I != d["I"]).sum()) == 0
        res = ch3.separability_script(I, k)
        assert assert_parity(res["bw"] > 0, d["bw"], "binary", name=f"bw_k{k}")["n_diff"] == 0
        assert res["IC"] == pytest.approx(sc(d["IC"]), abs=1e-12)
        assert np.allclose(res["p"], vec(d["p"]), atol=1e-15)
        assert res["mg"] == pytest.approx(sc(d["mg"]), abs=1e-9)
        assert res["sigma2_g"] == pytest.approx(sc(d["sigma2_g"]), abs=1e-9)
        assert res["sigma2_b"] == pytest.approx(sc(d["sigma2_b"]), abs=1e-9)
        assert res["eta"] == pytest.approx(sc(d["eta"]), abs=1e-12)
        # the script's eta is eta(t = k - 1) in the book convention (C0 = bins 1..k = levels 0..k-1)
        assert th.separability(I, k - 1) == pytest.approx(sc(d["eta"]), abs=1e-12)

    def test_hand_coded_equations_vs_otsu_criterion(self):
        d = ref("separability_108")
        I = gray_image("2")
        cur = th.otsu_criterion(imhist(I)[0])
        assert np.array_equal(imhist(I)[0], vec(d["cnt"]))
        assert cur.mG == pytest.approx(sc(d["mGv"]), abs=1e-9) and cur.sigma_G2 == pytest.approx(sc(d["sGv"]), abs=1e-6)
        P1 = vec(d["P1v"]).astype(float)
        both = (vec(d["P0v"]).astype(float) > 0) & (P1 > 0)            # both classes non-empty (t = 12..217 here)
        for py, ml, tol in ((cur.P0, "P0v", 1e-12), (cur.P1, "P1v", 1e-12), (cur.m, "mv", 1e-9),
                            (cur.m0, "m0v", 1e-9), (cur.sigma0_2, "s0v", 1e-8), (cur.sigma_W2, "sWv", 1e-6),
                            (cur.sigma_B2, "sBv", 1e-6), (cur.sigma_B2, "sB16", 1e-6), (cur.eta, "etav", 1e-12)):
            r = vec(d[ml]).astype(float)
            assert np.allclose(py[both], r[both], atol=tol, rtol=1e-9), ml
        # C1 moments: the port uses cumulative differences ((mG - m) / P1, E[i^2] - m1^2); MATLAB sums the class
        # directly.  Mass-weighted moments agree to round-off; the ratios lose digits only when C1 is tiny
        # (t = 215: P1 = 8e-8 = one pixel -> m1 rel. error 1.4e-8, sigma1^2 6.6e-4 instead of exactly 0).
        m1, s1 = vec(d["m1v"]).astype(float), vec(d["s1v"]).astype(float)
        assert np.allclose((cur.P1 * cur.m1)[both], (P1 * m1)[both], atol=1e-12)
        big = both & (P1 >= 1e-6)
        assert np.allclose(cur.m1[big], m1[big], rtol=1e-7, atol=0)
        assert np.allclose(cur.sigma1_2[big], s1[big], rtol=1e-4, atol=1e-3)
        assert cur.t_star == sc(d["tstar"]) == 255 * sc(d["lv"])
        assert cur.eta_star == pytest.approx(sc(d["etastar"]), abs=1e-12) == pytest.approx(sc(d["emv"]), abs=1e-12)
        assert th.separability(I, 125) == pytest.approx(sc(d["eta125"]), abs=1e-12)

    def test_otsu_criterion_empty_class_tail_matches_matlab(self):
        """C1 moments are summed directly over i > t (reverse cumulative sums): P1 is exactly 0 for an empty C1, so
        m1 / sigma_B2 are NaN exactly where MATLAB's direct sums give NaN, and sigma1^2 of a one-pixel class is 0
        (was 6.6e-4 from cancellation in the old E[i^2]/P1 - m1^2 with P1 = 1 - P0)."""
        d = ref("separability_108")
        cur = th.otsu_criterion(imhist(gray_image("2"))[0])
        m1, s1 = vec(d["m1v"]).astype(float), vec(d["s1v"]).astype(float)
        assert np.array_equal(np.isfinite(cur.m1), np.isfinite(m1))
        assert np.array_equal(np.isfinite(cur.sigma_B2), np.isfinite(vec(d["sBv"]).astype(float)))
        ok = np.isfinite(s1)
        assert np.allclose(cur.sigma1_2[ok], s1[ok], atol=1e-6)


# ================================================================================================================
# L4 — numbers quoted in the book (only for the shipped images 1.jpg / 2.jpg / test.jpg)
# ================================================================================================================
@needs_image
class TestL4BookNumbers:
    @pytest.mark.parametrize("key,t_star,ic", [("1", 123, 15.36), ("2", 107, 32.05), ("test", 182, 72.63)])
    def test_table_3_1_otsu(self, key, t_star, ic):
        res = ch3.otsu_segmentation(gray_image(key))
        assert res["threshold"] == t_star
        assert round(100 * res["ic"], 2) == ic                          # Figs. 3.9(b)-3.11(b), Table 3.1 p. 56

    @pytest.mark.parametrize("key,ic", [("1", 15.65), ("2", 32.49), ("test", 96.50)])
    def test_table_3_1_kmeans2(self, key, ic):
        res = ch3.kmeans_segmentation(gray_image(key), 2, shift_bug=True)   # book numbers need the script's units bug
        assert round(100 * res["ic"], 2) == ic                          # Figs. 3.9(c)-3.11(c), Table 3.1

    def test_fig_3_12a_tables_3_2_3_3_multi_otsu(self):
        mo = ch3.multi_otsu_segmentation(gray_image("test"), 2)
        assert mo["thresh"].tolist() == [120, 197]
        assert [round(100 * v, 2) for v in mo["coverage"]] == [3.50, 42.11, 54.39]    # water / ice 2 / ice 1
        assert round(100 * mo["ic"], 2) == 96.50
        assert [round(v, 4) for v in mo["average_intensity"]] == [63.8908, 177.0690, 218.1751]   # Table 3.3

    def test_fig_3_12b_tables_3_2_3_3_kmeans3(self):
        km = ch3.kmeans_segmentation(gray_image("test"), 3, shift_bug=True)
        assert [round(100 * v, 2) for v in km["coverage"]] == [2.89, 19.20, 77.91]
        assert round(100 * km["ic"], 2) == 97.11
        assert [round(v, 4) for v in km["average_intensity"]] == [53.8234, 161.6657, 209.0405]  # "209,0405" [sic]
        # book's remark (p. 56): both methods minimise the within-class variance — with consistent units the k-means
        # class boundaries reproduce multi-Otsu's 120 / 197 and the centroids equal Table 3.3's multi-Otsu means
        ok = ch3.kmeans_segmentation(gray_image("test"), 3, shift_bug=False)
        assert np.allclose(ok["boundaries"], [120.5, 197.6], atol=0.05)
        assert [round(v, 4) for v in ok["centroids"]] == [63.8908, 177.0690, 218.1751]

    def test_substitute_numbers_are_not_book_numbers(self):
        """2.jpg stands in for the unshipped Fig. 3.2(a): its values are recorded, not claimed as book matches."""
        I = gray_image("2")
        res = ch3.separability_script(I, 108)
        bw125, ic125 = ch3.fixed_threshold(I, 125)
        assert 0.3 < res["IC"] < 0.35 and 0.3 < ic125 < 0.35            # book: 42.14 % / 41.47 % (other image)
        assert 0.95 < res["eta"] < 0.98                                # book: 0.9643 (other image)
        mo = ch3.multi_otsu_segmentation(I, 2)
        assert mo["thresh"].tolist() == [61, 142]                      # coincides with Fig. 3.5's 61 / 142


# ================================================================================================================
# Scripts — every ported .m must run headless
# ================================================================================================================
SCRIPTS = ["otsu", "local_otsu", "separability", "kmeans", "global_threshold", "kmeans_demo_2d"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_runs(name: str, tmp_path: Path):
    script = ROOT / "scripts" / f"ch03_{name}.py"
    out = tmp_path / name
    proc = subprocess.run([sys.executable, str(script), "--no-show", "--out", str(out)], capture_output=True,
                          text=True, cwd=str(ROOT), timeout=900)
    assert proc.returncode == 0, f"{script.name} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    files = list(out.glob("*"))
    assert files, f"{script.name} wrote nothing to {out}"
