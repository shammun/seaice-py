"""Chapter 7 verification tests — sea ice type identification.

Evidence levels (see the ``verify-port`` skill):

* **L1** synthetic truth — small arrays whose answer is known analytically or from the book's own equations.
* **L2** MATLAB parity — the ORIGINAL ``.m`` code of ``MATLAB_ROOT/ch7`` run through
  ``reference/ch07/make_refs.py`` (MATLAB R2025a, ``matlab -batch``) and compared element-wise.
* **L3** book-figure reproduction (the printed matrices of Figs. 7.2–7.8; verdicts in the report).
* **L4** numbers quoted in the chapter text.

Run: ``.venv/Scripts/python.exe -m pytest tests/test_ch07.py -q -p no:cacheprovider``
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat
from scipy.ndimage import binary_fill_holes

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reference.ch07 import fixtures as FX  # noqa: E402
from seaice import ch07_ice_type as ch7  # noqa: E402
from seaice.core import synth  # noqa: E402
from seaice.core.connectivity import label_components  # noqa: E402
from seaice.core.histogram import hist as mhist  # noqa: E402
from seaice.core.io import load_image  # noqa: E402
from seaice.core.morphology import imclose, imfill, imopen, strel  # noqa: E402
from seaice.core.plotting import label2rgb  # noqa: E402
from seaice.core.threshold import graythresh, im2bw  # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab  # noqa: E402

REF = ROOT / "reference/ch07"
DATA = ROOT / "data/book/ch07"
VERIFY = ROOT / "outputs/ch07/verify"
PY = ROOT / ".venv/Scripts/python.exe"
CH = "ch07"

_CACHE: dict[str, dict] = {}


def ref(name: str) -> dict:
    """Load ``reference/ch07/<name>.mat`` once, or skip when it has not been generated."""
    if name not in _CACHE:
        p = REF / f"{name}.mat"
        if not p.exists():
            pytest.skip(f"run reference/ch07/make_refs.py (needs MATLAB R2025a): {p.name} missing")
        _CACHE[name] = loadmat(str(p))
    return _CACHE[name]


def eq(a, b) -> bool:
    """Bit equality after a float cast (used for the 0/1 and label images this chapter produces)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float).squeeze() if np.asarray(b).ndim > a.ndim else np.asarray(b, dtype=float)
    return a.shape == b.shape and np.array_equal(a, b)


def npx(a, b) -> int:
    """Number of differing elements."""
    return int(np.count_nonzero(np.asarray(a, float) != np.asarray(b, float)))


def flat(d, key) -> np.ndarray:
    return np.ravel(np.asarray(d[key]))


def _touches_border(m: np.ndarray) -> bool:
    return bool(m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any())


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
class TestL1Morphology:
    def test_cleaning_is_opening_of_the_closing(self):
        """§7.1.1 p. 145: morphological cleaning = closing **then** opening."""
        rng = np.random.default_rng(0)
        I = (rng.random((30, 40)) > 0.5).astype(float)
        se = np.ones((2, 2), bool)
        r = ch7.morphological_cleaning(I, se)
        assert eq(r.f0, imopen(imclose(I, se), se))
        assert eq(r.f1, imclose(I, se)) and eq(r.f2, imopen(I, se))

    def test_closing_is_extensive_opening_antiextensive(self):
        """Set-theoretic truth: ``A subset A.B`` and ``A o B subset A`` for any SE."""
        rng = np.random.default_rng(1)
        I = (rng.random((25, 25)) > 0.6).astype(float)
        se = np.ones((2, 2), bool)
        r = ch7.morphological_cleaning(I, se)
        assert np.all(r.f1 >= I), "closing must be extensive"
        assert np.all(r.f2 <= I), "opening must be anti-extensive"

    def test_dilate_erode_chain_reproduces_closing_and_cleaning(self):
        """``morphology_cleaning.m`` l. 24-27: c2 == f1 (closing) and c4 == f0 (cleaning) for a 2x2 SE."""
        r = ch7.morphological_cleaning(synth.FIG_7_2_IMAGE.astype(float), np.ones((2, 2), bool))
        assert eq(r.c2, r.f1) and eq(r.c4, r.f0)


class TestL1Extraction:
    @pytest.mark.parametrize("se,conn", [("square3", 8), ("diamond1", 4)])
    def test_eq_7_1_equals_the_bwlabel_component(self, se, conn):
        """Eq. (7.1) must converge to exactly the ``bwlabel`` component containing the seed."""
        A = synth.FIG_7_3_IMAGE
        seed = synth.FIG_7_3_SEED
        r = ch7.connected_component_extract(A, seed, se, max_iter=20)
        L = label_components(A.astype(bool), conn)
        lab = L[np.nonzero(seed)][0]
        assert r.converged
        assert eq(r.component, L == lab)

    def test_recursion_is_monotone_and_terminates(self):
        r = ch7.connected_component_extract(synth.FIG_7_3_IMAGE, synth.FIG_7_3_SEED, "square3", max_iter=20)
        sums = [int(b.sum()) for b in r.blocks[1:]]
        assert sums == sorted(sums), "X_k must grow monotonically"
        assert r.converged and r.n_iter == 8

    def test_fig_7_4_needs_one_iteration_less_than_fig_7_3(self):
        """Book p. 148/149: Fig. 7.3 (8-conn) completes at the 8th block, Fig. 7.4 (4-conn) has 7."""
        r8 = ch7.connected_component_extract(synth.FIG_7_3_IMAGE, synth.FIG_7_3_SEED, "square3", max_iter=20)
        r4 = ch7.connected_component_extract(synth.FIG_7_3_IMAGE, synth.FIG_7_3_SEED, "diamond1", max_iter=20)
        assert len(synth.FIG_7_3_STEPS) == 8 and len(synth.FIG_7_4_STEPS) == 7
        assert r8.n_iter == 8 and r4.n_iter == 7

    def test_seed_accepts_a_row_col_pair(self):
        A = np.zeros((5, 5), bool)
        A[1:4, 1:4] = True
        r = ch7.connected_component_extract(A, (2, 2), "square3", max_iter=5)
        assert eq(r.component, A)


class TestL1Filling:
    def test_eq_7_4_equals_imfill_and_binary_fill_holes_for_logical(self):
        """Eq. (7.4) ``H = [R^D_{F^c}(F_m)]^c`` == ``imfill(F, 'holes')`` == ``binary_fill_holes`` (logical)."""
        F = synth.FIG_7_8_IMAGE.astype(bool)
        H = ch7.hole_fill_reconstruct(F, conn=4)
        assert eq(H, imfill(F, "holes", conn=4))
        assert eq(H, binary_fill_holes(F))

    def test_border_marker_is_eq_7_3(self):
        """Eq. (7.3): ``F_m = 1 - F`` on the border, 0 inside."""
        rng = np.random.default_rng(3)
        F = rng.random((9, 11)) > 0.5
        m = ch7.border_marker(F)
        assert not m[1:-1, 1:-1].any()
        assert np.array_equal(m[0, :], ~F[0, :]) and np.array_equal(m[-1, :], ~F[-1, :])
        assert np.array_equal(m[:, 0], ~F[:, 0]) and np.array_equal(m[:, -1], ~F[:, -1])

    def test_fig_7_7_failure_case_floods_the_whole_background(self):
        """§7.1.3.1: with a 3x3 square SE the hole is 8-connected to the background, so nothing is filled."""
        A = synth.FIG_7_6_IMAGE
        r = ch7.hole_fill_dilation(A, synth.FIG_7_5_SEED, "square3", max_iter=15)
        assert eq(r.component, ~A.astype(bool)), "the recursion must reach the whole complement"
        assert eq(r.filled, np.ones_like(A, dtype=float))

    def test_fig_7_6_failure_case_fills_only_the_seeded_hole(self):
        """§7.1.3.1 / Fig. 7.6: the cross SE finds only the hole that contains the seed."""
        A = synth.FIG_7_6_IMAGE
        r = ch7.hole_fill_dilation(A, synth.FIG_7_5_SEED, "diamond1", max_iter=15)
        auto = ch7.hole_fill_reconstruct(A.astype(bool), conn=4)
        assert not eq(r.filled, auto), "Eq. (7.2) must miss the second hole that Eq. (7.4) fills"
        assert int(np.asarray(auto, float).sum()) > int(np.asarray(r.filled, float).sum())


class TestL1Equations:
    @pytest.mark.parametrize("area,expect", [(0, 1), (1, 1), (49, 1), (50, 2), (51, 2), (10 ** 6, 2)])
    def test_eq_7_5_radius(self, area, expect):
        """Eq. (7.5): ``r = r1 if size < size_th else r2`` with ``size_th = 50``, ``r1 = 1``, ``r2 = 2``."""
        assert ch7.adaptive_se_radius(area, 50) == expect

    def test_eq_7_6_round_trip(self):
        """Eq. (7.6) and its printed inverse must agree within the ``fix``/``round`` quantisation."""
        areas = np.array([1, 2, 3, 40, 41, 154, 1000, 2500, 8112])
        back = ch7.color_to_area(ch7.size_color(areas))
        assert np.all(np.abs(back - areas) <= 1)

    def test_eq_7_6_is_monotone_and_uses_fix(self):
        a = np.arange(1, 5000)
        c = ch7.size_color(a)
        assert np.all(np.diff(c) >= 0)
        assert ch7.size_color(2) == int(np.trunc((1 - np.exp(-2 / 1000)) * 10000)) == 19

    def test_sorting_small_to_large_is_necessary(self):
        """p. 151: "the arrangement of ice pieces in order of increasing size is required".

        Processing the same fixture largest-first must change ``out`` — otherwise the sentence is untestable.
        """
        bk, seg = FX.seg_fixtures()["swallowed"]
        r = ch7.ice_shape_enhancement(bk, seg, nbins=0)
        # reverse order: run the pieces largest-first by relabelling the areas
        rows, cols = np.nonzero(seg == 1)
        assert r.t >= 1
        rev = _enhancement_reverse_order(bk, seg)
        assert not eq(rev, r.out), "large-first must give a different identification"

    def test_coverage_identity(self):
        """floe + brash + slush + water == 1 only when ``bk`` covers every identified pixel."""
        bk, seg = FX.seg_fixtures()["mixed"]
        r = ch7.ice_shape_enhancement(bk, seg, nbins=0)
        c = r.coverage
        # `floe`/`brash` are *pixel-area sums of the kept pieces*, not layer masks, so the identity is only
        # approximate when a piece is dropped by `> min_brash`; slush + water + identified == 1 exactly.
        ident = float(np.count_nonzero(r.index != 0)) / seg.size
        assert abs(c.Slush + c.Water + ident - 1.0) < 1e-12
        b0, _ = FX.seg_fixtures()["bk_partial"]
        r2 = ch7.ice_shape_enhancement(b0, seg, nbins=0)
        assert abs(r2.coverage.Slush) < 1e-12 and r2.coverage.Water < 1.0


def _enhancement_reverse_order(bk, seg):
    """Algorithm 2 with the pieces superimposed **largest first** (the order p. 151 warns against)."""
    seg = np.asarray(seg, float)
    bw = (seg == 1).astype(float)
    k = (seg == 0.5).astype(float)
    k = k - k * bw
    lab_bw = label_components(bw != 0, 4)
    lab_k = label_components(k != 0, 4)
    lists = [np.flatnonzero((lab_bw == i).ravel(order="F")) for i in range(1, int(lab_bw.max()) + 1)]
    lists += [np.flatnonzero((lab_k == i).ravel(order="F")) for i in range(1, int(lab_k.max()) + 1)]
    order = np.argsort([p.size for p in lists], kind="stable")[::-1]
    M, N = seg.shape
    out = np.zeros((M, N))
    t = 0
    for i in order:
        p = lists[i]
        rows, cols = np.unravel_index(p, (M, N), order="F")
        r = ch7.adaptive_se_radius(p.size, 50)
        b = np.zeros((M, N))
        b[rows, cols] = 1.0
        b = imfill(b, "hole", conn=4)
        b = imopen(imclose(b, strel("disk", r)), strel("disk", r))
        b = imfill(b, "hole", conn=4)
        L = label_components(b != 0, 4)
        for j in range(1, int(L.max()) + 1):
            t += 1
            out[L == j] = float(t)
    return out


# =====================================================================================================================
# L2 — MATLAB parity: cleaning & labeling & filling
# =====================================================================================================================
class TestL2CleaningLabelingFilling:
    def test_morphology_cleaning_m(self):
        """``morphology_cleaning.m`` — Fig. 7.2(a)-(d) and the explicit dilate/erode chain."""
        d = ref("clf")
        assert eq(synth.FIG_7_2_IMAGE, d["cl_I"]), "the printed 13x23 matrix is the script's own literal"
        r = ch7.morphological_cleaning(synth.FIG_7_2_IMAGE.astype(float), np.ones((2, 2), bool))
        for name in ("f1", "f2", "f0", "c1", "c2", "c3", "c4"):
            assert eq(getattr(r, name), d[f"cl_{name}"]), name
        assert [int(np.sum(d[f"cl_{n}"])) for n in ("I", "f1", "f2", "f0")] == [111, 115, 104, 108]

    def test_fig_7_2_printed_blocks(self):
        """L3: the book's printed (b)/(c)/(d) blocks == MATLAB's own output."""
        d = ref("clf")
        assert eq(synth.FIG_7_2_CLOSED, d["cl_f1"])
        assert eq(synth.FIG_7_2_OPENED, d["cl_f2"])
        assert eq(synth.FIG_7_2_CLEANED, d["cl_f0"])

    @pytest.mark.parametrize("pre,se,steps", [("lb3", "square3", synth.FIG_7_3_STEPS),
                                              ("lb1", "diamond1", synth.FIG_7_4_STEPS)])
    def test_labeling_m(self, pre, se, steps):
        """``labeling.m`` (Fig. 7.3) and its commented ``diamond`` variant (Fig. 7.4)."""
        d = ref("clf")
        assert eq(synth.FIG_7_3_IMAGE, d[f"{pre}_I"]) and eq(synth.FIG_7_3_SEED, d[f"{pre}_x0"])
        r = ch7.connected_component_extract(synth.FIG_7_3_IMAGE, synth.FIG_7_3_SEED, se, max_iter=20)
        ml = [d[f"{pre}_xx"]] + [d[f"{pre}_x{i}"] for i in range(1, 10)]
        for i, (p, m) in enumerate(zip(r.blocks, ml)):
            assert eq(p, m), f"block {i}"
        for m in ml[len(r.blocks):]:
            assert eq(m, r.blocks[-1]), "MATLAB's unrolled tail must be the fixed point"
        for i, b in enumerate(steps):
            assert eq(b, ml[i]), f"printed block {i}"

    def test_filling_m(self):
        """``filling.m`` — Eq. (7.2), Fig. 7.5(e)."""
        d = ref("clf")
        assert eq(synth.FIG_7_3_IMAGE, d["fi_I0"]) and eq(synth.FIG_7_5_SEED, d["fi_x0"])
        r = ch7.hole_fill_dilation(synth.FIG_7_3_IMAGE, synth.FIG_7_5_SEED, "diamond1", max_iter=20)
        ml = [d["fi_xx"]] + [d[f"fi_x{i}"] for i in range(1, 10)]
        for i, (p, m) in enumerate(zip(r.blocks, ml)):
            assert eq(p, m), f"block {i}"
        assert eq(r.filled, d["fi_I1"])
        assert eq(r.complement, d["fi_I"])
        for i, b in enumerate(synth.FIG_7_5_STEPS[:-1]):
            assert eq(b, ml[i]), f"printed block {i}"
        assert eq(synth.FIG_7_5_STEPS[-1], d["fi_I1"]), "the last printed block is X u A"

    @pytest.mark.parametrize("pre,conn", [("fr1", 4), ("fr3", 8), ("fra", 4)])
    def test_filling_reconstruct_m(self, pre, conn):
        """``filling_reconstruct.m`` — Eqs. (7.3)/(7.4), Fig. 7.8, incl. the commented SE and I0 variants."""
        d = ref("clf")
        I0 = d[f"{pre}_I0"].astype(bool)
        assert eq(ch7.border_marker(I0), d[f"{pre}_x0"]), "Eq. (7.3) == the script's hand-written x0"
        H = ch7.hole_fill_reconstruct(I0, conn=conn)
        assert eq(H, d[f"{pre}_x10"])
        assert eq(H | I0, d[f"{pre}_I1"])
        assert eq(H & ~I0, d[f"{pre}_I2"])
        assert eq(~I0, d[f"{pre}_I"])

    def test_fig_7_8_printed_blocks(self):
        d = ref("clf")
        assert eq(synth.FIG_7_8_IMAGE, d["fr1_I0"]) and eq(synth.FIG_7_8_MARKER, d["fr1_x0"])
        ml = [d["fr1_xx"]] + [d[f"fr1_x{i}"] for i in range(1, 11)] + [d["fr1_I1"], d["fr1_I2"]]
        for i, b in enumerate(synth.FIG_7_8_STEPS):
            assert any(eq(b, m) for m in ml), f"printed block {i} matches no MATLAB block"

    def test_se_neighbourhoods(self):
        """The SEs the four scripts and Eq. (7.5) use, against MATLAB's own ``getnhood``."""
        d = ref("clf")
        assert eq(strel("square", 2), d["se_sq2"]) and eq(strel("square", 3), d["se_sq3"])
        assert eq(strel("diamond", 1), d["se_dm1"])
        for r in (1, 2, 3):
            assert eq(strel("disk", r), d[f"se_d{r}"]), f"disk {r}"
        assert int(np.sum(d["se_d1"])) == 5 and d["se_d1"].shape == (3, 3)
        assert int(np.sum(d["se_d2"])) == 13 and d["se_d2"].shape == (5, 5)
        assert int(np.sum(d["se_d3"])) == 25 and d["se_d3"].shape == (5, 5)


class TestL2Figs76And77:
    """The book's two failure cases — the variant image is in **no** shipped ``.m``."""

    def test_variant_image(self):
        d = ref("fig767")
        assert eq(synth.FIG_7_6_IMAGE, d["f76_I0"])
        assert int(np.asarray(d["f76_I0"], float).sum()) == 25
        assert npx(synth.FIG_7_6_IMAGE, synth.FIG_7_3_IMAGE) == 1

    @pytest.mark.parametrize("pre,se", [("f76", "diamond1"), ("f77", "square3")])
    def test_blocks(self, pre, se):
        d = ref("fig767")
        r = ch7.hole_fill_dilation(synth.FIG_7_6_IMAGE, synth.FIG_7_5_SEED, se, max_iter=20)
        ml = [d[f"{pre}_xx"]] + [d[f"{pre}_x{i}"] for i in range(1, 10)]
        for i, (p, m) in enumerate(zip(r.blocks, ml)):
            assert eq(p, m), f"block {i}"
        for m in ml[len(r.blocks):]:
            assert eq(m, r.blocks[-1])
        assert eq(r.filled, d[f"{pre}_I1"])

    def test_fig_7_7_x8_is_a_book_typo_matlab_adjudicates(self):
        """MATLAB's own ``x8`` has 55 object pixels; the book prints 53 (2 px at 1-based (8,9) and (9,9))."""
        d = ref("fig767")
        ml_x8 = np.asarray(d["f77_x8"], bool)
        assert int(ml_x8.sum()) == 55
        assert int(synth.FIG_7_7_STEPS_BOOK[8].sum()) == 53
        assert eq(ml_x8, synth.FIG_7_7_STEPS[8]), "the corrected block must be MATLAB's"
        assert not eq(ml_x8, synth.FIG_7_7_STEPS_BOOK[8])
        diff = {tuple(p) for p in np.argwhere(ml_x8 != synth.FIG_7_7_STEPS_BOOK[8])}
        assert diff == set(synth.FIG_7_7_X8_TYPO)

    def test_all_other_printed_blocks_match(self):
        d = ref("fig767")
        ml76 = [d["f76_xx"]] + [d[f"f76_x{i}"] for i in range(1, 10)]
        for i, b in enumerate(synth.FIG_7_6_STEPS[:-1]):
            assert any(eq(b, m) for m in ml76), f"Fig. 7.6 printed block {i}"
        assert eq(synth.FIG_7_6_STEPS[-1], d["f76_I1"])
        ml77 = [d["f77_xx"]] + [d[f"f77_x{i}"] for i in range(1, 10)]
        for i, b in enumerate(synth.FIG_7_7_STEPS):
            assert eq(b, ml77[i]), f"Fig. 7.7 printed block {i}"


# =====================================================================================================================
# L2 — new core primitive: imfill
# =====================================================================================================================
IMFILL_CASES = sorted(FX.imfill_cases())
MATLAB_CLASS = {np.dtype(bool): "logical", np.dtype("uint8"): "uint8", np.dtype("float64"): "double",
                np.dtype("float32"): "single", np.dtype("int16"): "int16"}


class TestL2Imfill:
    @pytest.mark.parametrize("name", IMFILL_CASES)
    @pytest.mark.parametrize("conn", [4, 8])
    def test_values(self, name, conn):
        d = ref("imfill")
        a = FX.imfill_cases()[name]
        got = imfill(a, "hole", conn=conn)
        m = np.asarray(d[f"out_{name}_{conn}"])
        assert got.shape == m.shape, f"{name}: {got.shape} vs {m.shape}"
        assert np.array_equal(np.asarray(got, float), np.asarray(m, float)), \
            f"{name} conn{conn}: {npx(got, m)} elements differ"

    @pytest.mark.parametrize("name", IMFILL_CASES)
    def test_output_class(self, name):
        """R2: the ch7 call passes a **double**, so MATLAB takes the grayscale branch and returns a double."""
        d = ref("imfill")
        got = imfill(FX.imfill_cases()[name], "hole")
        mcls = str(np.ravel(d[f"cls_{name}"])[0]).strip()
        assert MATLAB_CLASS[got.dtype] == mcls, f"{name}: python {got.dtype} vs MATLAB {mcls}"

    def test_prefix_matching(self):
        """``validatestring``: 'hole', 'holes' and 'h' are the same option."""
        d = ref("imfill")
        a = FX.imfill_cases()["double_hole"]
        assert eq(imfill(a, "hole"), d["pref_full"]) and eq(imfill(a, "holes"), d["pref_h"])
        assert eq(imfill(a, "h"), d["pref_full"])
        with pytest.raises(ValueError):
            imfill(a, "pixels")

    def test_default_connectivity_is_4(self):
        """``conndef(2, 'minimal')`` is the cross; the default must therefore be conn 4."""
        d = ref("imfill")
        assert eq(d["conndef_min"], np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        assert eq(imfill(FX.imfill_cases()["diagonal_hole"], "holes"), d["def_conn"])

    def test_binary_fill_holes_agrees_only_on_the_logical_conn4_branch(self):
        """The justification for re-implementing ``imfill.m`` (analysis R2) — pinned, not assumed."""
        d = ref("imfill")
        cases = FX.imfill_cases()
        logical_conn4, logical_conn8, nonlogical = [], [], []
        for name, a in cases.items():
            if a.dtype == bool:
                bfh = binary_fill_holes(a)
                logical_conn4.append(np.array_equal(bfh, np.asarray(d[f"out_{name}_4"], bool)))
                logical_conn8.append(np.array_equal(bfh, np.asarray(d[f"out_{name}_8"], bool)))
            else:
                bfh = binary_fill_holes(np.asarray(a) != 0)
                same = (bfh.dtype == a.dtype
                        and np.array_equal(np.asarray(bfh, float), np.asarray(d[f"out_{name}_4"], float)))
                nonlogical.append(same)
        assert all(logical_conn4), "binary_fill_holes must equal MATLAB on the logical conn-4 branch"
        assert not all(logical_conn8), "and must differ for conn 8 (scipy's default structure is the cross)"
        assert not any(nonlogical), "and must never reproduce a non-logical input (wrong class and/or values)"

    def test_ch7_call_form_returns_double(self):
        """``ice_shape_enhancement.m`` l. 73/93: ``b`` is a double 0/1 array; the class must survive."""
        b = np.zeros((9, 9))
        b[1:8, 1:8] = 1.0
        b[3:6, 3:6] = 0.0
        got = imfill(b, "hole")
        assert got.dtype == np.float64 and set(np.unique(got)) == {0.0, 1.0}


# =====================================================================================================================
# L2 — new core primitive: hist (MATLAB centre semantics)
# =====================================================================================================================
HIST_CASES = sorted(FX.hist_cases())


class TestL2Hist:
    @pytest.mark.parametrize("name", [n for n in HIST_CASES if n != "with_nan_inf"])
    def test_counts_and_centres(self, name):
        d = ref("hist")
        y, b = FX.hist_cases()[name]
        z, n = mhist(np.asarray(y, float), b)
        assert np.array_equal(np.asarray(z, float), flat(d, f"z_{name}").astype(float)), \
            f"{name}: counts {z.tolist()} vs {flat(d, f'z_{name}').tolist()}"
        assert np.allclose(n, flat(d, f"n_{name}"), atol=1e-12), f"{name}: centres"

    def test_empty_input(self):
        d = ref("hist")
        z, n = mhist(np.zeros(0), 5)
        assert np.array_equal(z, flat(d, "z_empty")) and np.allclose(n, flat(d, "n_empty"))
        z, n = mhist(np.zeros(0), np.array([1.0, 2.0, 3.0]))
        assert np.array_equal(z, flat(d, "z_empty_c")) and np.allclose(n, flat(d, "n_empty_c"))

    def test_centres_not_edges(self):
        """L1: ``hist(y, n)`` returns bin **centres**; ``np.histogram`` returns edges (R7)."""
        y = np.array([0.0, 10.0])
        z, n = mhist(y, 2)
        assert np.allclose(n, [2.5, 7.5]) and z.tolist() == [1, 1]
        np_counts, np_edges = np.histogram(y, 2)
        assert not np.allclose(np_edges[:-1], n)

    def test_outer_bins_are_unbounded_with_explicit_centres(self):
        """L1: values below/above the centre range are *counted*, not dropped (R7)."""
        z, n = mhist(np.array([-1000.0, 0.0, 5.0, 1e6]), np.array([0.0, 5.0, 10.0]))
        assert int(z.sum()) == 4 and z.tolist() == [2, 1, 1]

    @pytest.mark.xfail(strict=True, reason="VERIFIER FINDING (ch07 open item 1): core.histogram.hist drops "
                                           "-Inf. MATLAB's hist counts -Inf in the first bin and +Inf in the "
                                           "last; the port's line 139 filter `np.isfinite(y) | (y == np.inf)` "
                                           "keeps +Inf but discards -Inf. Latent for the chapter (floe areas "
                                           "are finite) but the docstring claims 'exact'.")
    def test_minus_inf_is_counted_in_the_first_bin(self):
        d = ref("hist")
        y, b = FX.hist_cases()["with_nan_inf"]
        z, _ = mhist(np.asarray(y, float), b)
        assert np.array_equal(np.asarray(z, float), flat(d, "z_with_nan_inf").astype(float)), \
            f"counts {z.tolist()} vs MATLAB {flat(d, 'z_with_nan_inf').tolist()}"


# =====================================================================================================================
# L2 — ice_shape_enhancement.m (the chapter's core file)
# =====================================================================================================================
SEG_CASES = sorted(FX.seg_fixtures())


def _iceenh_ref(name: str, tag: str) -> dict:
    p = REF / f"iceenh_{name}_{tag}.mat"
    if not p.exists():
        pytest.skip(f"run reference/ch07/make_refs.py iceenh: {p.name} missing")
    key = f"iceenh_{name}_{tag}"
    if key not in _CACHE:
        _CACHE[key] = loadmat(str(p))
    return _CACHE[key]


class TestL2IceShapeEnhancement:
    """Every array of ``ice_shape_enhancement.m`` on 9 controlled fixtures x 2 threshold forms x 2 crop modes."""

    @pytest.mark.parametrize("name", SEG_CASES)
    @pytest.mark.parametrize("tag,book", [("code", False), ("book", True)])
    @pytest.mark.parametrize("crop", [True, False])
    def test_all_outputs(self, name, tag, book, crop):
        d = _iceenh_ref(name, tag)
        bk, seg = FX.seg_fixtures()[name]
        r = ch7.ice_shape_enhancement(bk, seg, min_floe=40, min_brash=1, se_th=50,
                                      nbins=50, crop=crop, book_threshold=book)
        for attr, key in (("out", "out"), ("l", "l"), ("fill", "fill"), ("index", "index"),
                          ("index_floe", "index_floe"), ("index_brash", "index_brash"),
                          ("index_slush", "index_slush"), ("index_water", "index_water"),
                          ("index_residue", "index_residue")):
            assert eq(getattr(r, attr), d[key]), f"{name}/{tag}/crop={crop}: {key}"
        assert int(r.t) == int(np.ravel(d["t"])[0])
        assert int(r.nn_bw) == int(np.ravel(d["nn_bw"])[0])
        assert int(r.nn_k) == int(np.ravel(d["nn_k"])[0])
        assert np.array_equal(np.sort(r.ice_area), flat(d, "A").astype(np.int64)), "sorted areas A"
        assert np.array_equal(r.order + 1, flat(d, "ind").astype(np.int64)), "sort index (stable ties)"
        assert np.array_equal(r.color_floe, flat(d, "color_floe").astype(np.int64))
        assert np.array_equal(r.color_brash, flat(d, "color_brash").astype(np.int64))
        assert np.array_equal([p.Area for p in r.ice_floe], flat(d, "floe_area").astype(np.int64))
        assert np.array_equal([p.Area for p in r.brash_ice], flat(d, "brash_area").astype(np.int64))
        assert np.allclose(r.floe_cen.ravel(), np.asarray(d["floe_cen"], float).ravel(), atol=1e-12)
        assert np.allclose(r.brash_cen.ravel(), np.asarray(d["brash_cen"], float).ravel(), atol=1e-12)
        cov = d["coverage"][0, 0]
        for field in ("IceFloe", "BrashIce", "Slush", "Water"):
            assert abs(getattr(r.coverage, field) - float(cov[field][0, 0])) < 1e-12, field

    @pytest.mark.parametrize("name", SEG_CASES)
    def test_perimeters(self, name):
        """``regionprops(out == i, 'perimeter')`` — the Vossepoel-Smeulders perimeter of ch06's port."""
        d = _iceenh_ref(name, "code")
        bk, seg = FX.seg_fixtures()[name]
        r = ch7.ice_shape_enhancement(bk, seg, nbins=0)
        ml = np.asarray(d["ice_floe"])
        if ml.size == 0:
            assert not r.ice_floe
            return
        per_ml = np.array([float(np.ravel(s["Perimeter"])[0]) for s in ml.ravel()])
        per_py = np.array([p.Perimeter for p in r.ice_floe])
        assert np.allclose(per_py, per_ml, atol=1e-9), f"{name}: {per_py} vs {per_ml}"
        cen_ml = np.array([np.ravel(s["Center"]) for s in ml.ravel()])
        cen_py = np.array([p.Center for p in r.ice_floe])
        assert np.allclose(cen_py, cen_ml, atol=1e-12)
        # the four field names Appendix B's `SeaIce_Image_Structure.m` and ch8's `sea_ice_model.m` read
        assert ml.ravel()[0].dtype.names == ("Center", "Area", "Perimeter", "PixelsPosition")
        for sml, ppy in zip(ml.ravel(), r.ice_floe):
            assert np.array_equal(np.asarray(sml["PixelsPosition"]), ppy.PixelsPosition), name
            assert int(np.ravel(sml["Area"])[0]) == ppy.Area

    @pytest.mark.parametrize("name", SEG_CASES)
    def test_fsd_histogram(self, name):
        """The 50-bin FSD of lines 210-223 and the Eq. (7.6) bar colours of line 221."""
        d = _iceenh_ref(name, "code")
        bk, seg = FX.seg_fixtures()[name]
        r = ch7.ice_shape_enhancement(bk, seg, nbins=50)
        if r.fsd is None:
            assert np.asarray(d["z"]).size == 0
            return
        assert np.array_equal(np.asarray(r.fsd.counts, float), flat(d, "z").astype(float))
        assert np.allclose(r.fsd.centers, flat(d, "n"), atol=1e-12)
        assert np.array_equal(np.asarray(r.fsd.colors, float), flat(d, "color").astype(float))

    @pytest.mark.parametrize("name", SEG_CASES)
    def test_colorbar_ticks(self, name):
        """Lines 194-206 (``n = 6``, map) and 226-237 (``nn = 8``, histogram) — the printed integers."""
        d = _iceenh_ref(name, "code")
        bk, seg = FX.seg_fixtures()[name]
        r = ch7.ice_shape_enhancement(bk, seg, nbins=50)
        area_ice = np.concatenate([r.color_floe, r.color_brash]) if (r.color_floe.size or r.color_brash.size) \
            else np.zeros(0)
        vals, labels = ch7.colorbar_area_ticks(area_ice, 6)
        ml_vals = flat(d, "ysh")
        if ml_vals.size == 0:
            assert vals.size <= 1
        else:
            assert np.allclose(vals, ml_vals), f"{name}: ysh {vals} vs {ml_vals}"
            ml_lab = np.array([float(np.ravel(x)[0]) for x in np.asarray(d["YT"]).ravel()])
            assert np.array_equal(np.asarray(labels, float), ml_lab)
        if r.color_floe.size:
            v2, l2 = ch7.colorbar_area_ticks(r.color_floe, 8)
            ml2 = flat(d, "ysh2")
            if ml2.size:
                assert np.allclose(v2, ml2)
                assert np.array_equal(np.asarray(l2, float),
                                      np.array([float(np.ravel(x)[0]) for x in np.asarray(d["YT2"]).ravel()]))

    def test_crop_equals_the_literal_full_image_form(self):
        """R11: the bounding-box crop must be bit-identical to the M-file's full-image scratch arrays.

        The adversarial fixture is ``border_notch``: every piece touches an image border and carries a notch
        cut in from it, which is exactly where ``imclose``'s 0 pre-pad (ch04) could differ under a crop.
        """
        for name in ("border_notch", "mixed", "ring"):
            bk, seg = FX.seg_fixtures()[name]
            a = ch7.ice_shape_enhancement(bk, seg, nbins=0, crop=True)
            b = ch7.ice_shape_enhancement(bk, seg, nbins=0, crop=False)
            assert eq(a.out, b.out) and eq(a.fill, b.fill), name
            d = _iceenh_ref(name, "code")
            assert eq(a.out, d["out"]) and eq(b.out, d["out"]), f"{name} vs MATLAB"

    def test_book_vs_code_threshold_differ_only_on_the_boundary_areas(self):
        """R3: ``>`` (code) vs ``>=`` (book Algorithm 5) — only pieces of exactly ``T_floe`` move.

        The ``thresholds`` fixture's raw areas are 1, 2, 20, 39, 40, 49, 50, 51 px, but the cleaning changes
        them (a solid rectangle loses its four corners to ``imopen`` with ``strel('disk', 1)``, the cross), so
        the *identified* areas are 35, 36, 38 and 45.  The thresholds are therefore set to those values — the
        branch under test is the comparison operator, not the arithmetic that produced the areas.
        """
        bk, seg = FX.seg_fixtures()["thresholds"]
        a = ch7.ice_shape_enhancement(bk, seg, nbins=0, min_floe=38, min_brash=35, book_threshold=False)
        b = ch7.ice_shape_enhancement(bk, seg, nbins=0, min_floe=38, min_brash=35, book_threshold=True)
        assert sorted(p.Area for p in a.ice_floe) == [45]
        assert sorted(p.Area for p in a.brash_ice) == [36, 38]       # 35 is dropped by the strict `>`
        assert sorted(p.Area for p in b.ice_floe) == [38, 45]        # 38 is promoted by the book's `>=`
        assert sorted(p.Area for p in b.brash_ice) == [35, 36]       # 35 survives
        moved = {p.label for p in b.ice_floe} - {p.label for p in a.ice_floe}
        assert {next(p.Area for p in b.ice_floe if p.label == lb) for lb in moved} == {38}

    def test_label2rgb_survives_eq_7_6_colour_values(self):
        """R6: ``index`` holds colour values up to ~10 000, not 1..N labels — display only, must not blow up."""
        bk, seg = FX.seg_fixtures()["mixed"]
        r = ch7.ice_shape_enhancement(bk, seg, nbins=0)
        rgb = label2rgb(r.index, cmap="jet", background=(1, 1, 1), shuffle=False)
        assert rgb.shape == seg.shape + (3,)
        assert np.all(rgb[r.index == 0] == 1.0) or np.all(rgb[r.index == 0] == 255)


# =====================================================================================================================
# L2 — sea_ice_demo.m end to end on ch7's own sea_ice_test.jpg
# =====================================================================================================================
def demo_ref() -> dict:
    p = REF / "demo.mat"
    if not p.exists():
        pytest.skip("run reference/ch07/make_refs.py demo (needs MATLAB R2025a, ~20 min)")
    if "demo" not in _CACHE:
        _CACHE["demo"] = loadmat(str(p))
    return _CACHE["demo"]


class TestL2Demo:
    """Algorithms 4 and 5 fed **MATLAB's own** ``seg``/``bk``, so the chapter's contribution is isolated."""

    def test_stage3_inputs_exist(self):
        d = demo_ref()
        assert d["seg"].shape == (1038, 394) and set(np.unique(d["seg"])) <= {0.0, 0.5, 1.0}

    def test_ice_shape_enhancement_on_the_real_image(self):
        d = demo_ref()
        r = ch7.ice_shape_enhancement(d["bk"], d["seg"], min_floe=40, min_brash=1, se_th=50, nbins=50)
        for key in ("out", "l", "fill", "index", "index_floe", "index_brash", "index_slush",
                    "index_water", "index_residue"):
            assert eq(getattr(r, key), d[key]), f"{key}: {npx(getattr(r, key), d[key])} px differ"
        assert int(r.t) == int(np.ravel(d["t"])[0])
        assert np.array_equal(r.order + 1, flat(d, "ind").astype(np.int64))
        assert np.array_equal(np.sort(r.ice_area), flat(d, "A").astype(np.int64))
        assert np.array_equal(r.color_floe, flat(d, "color_floe").astype(np.int64))
        assert np.array_equal(r.color_brash, flat(d, "color_brash").astype(np.int64))
        cov = d["coverage"][0, 0]
        for field in ("IceFloe", "BrashIce", "Slush", "Water"):
            assert abs(getattr(r.coverage, field) - float(cov[field][0, 0])) < 1e-12

    def test_crop_equivalence_on_the_real_image(self):
        """R11 on **MATLAB's own** segmentation: 1211 pieces (982 light + 229 dark), 712 identified labels.

        ``crop=True`` (the port's default, a bounding-box optimisation) must be bit-identical to the M-file's
        per-piece **full-image** scratch arrays.  This is the independent re-verification of the porter's claim,
        measured against MATLAB rather than against the port's own ``crop=False`` path.
        """
        d = demo_ref()
        r = ch7.ice_shape_enhancement(d["bk"], d["seg"], nbins=0, crop=True)
        assert eq(r.out, d["out"]), f"{npx(r.out, d['out'])} px differ from MATLAB's full-image form"
        assert eq(r.fill, d["fill"])
        assert (int(r.ice_area.size), int(r.nn_bw), int(r.nn_k)) == (1211, 982, 229)
        border = sum(1 for i in range(r.ice_area.size)
                     if _touches_border(r.l == (i + 1)))
        assert border > 0, "the fixture must contain border-touching pieces"

    def test_fsd_and_colorbar_on_the_real_image(self):
        d = demo_ref()
        r = ch7.ice_shape_enhancement(d["bk"], d["seg"], nbins=50)
        assert np.array_equal(np.asarray(r.fsd.counts, float), flat(d, "z").astype(float))
        assert np.allclose(r.fsd.centers, flat(d, "n"), atol=1e-9)
        area_ice = np.concatenate([r.color_floe, r.color_brash])
        vals, labels = ch7.colorbar_area_ticks(area_ice, 6)
        assert np.allclose(vals, flat(d, "ysh"))
        assert np.array_equal(np.asarray(labels, float),
                              np.array([float(np.ravel(x)[0]) for x in np.asarray(d["YT"]).ravel()]))

    def test_piece_counts(self):
        """On MATLAB's own segmentation: 433 floes / 274 brash pieces with the script's strict ``>``."""
        d = demo_ref()
        r = ch7.ice_shape_enhancement(d["bk"], d["seg"], nbins=0)
        assert len(r.ice_floe) == int(np.asarray(d["ice_floe"]).size)
        assert len(r.brash_ice) == int(np.asarray(d["brash_ice"]).size)
        assert (len(r.ice_floe), len(r.brash_ice)) == (433, 274)

    def test_algorithm_3_residual_propagates_to_the_piece_counts(self):
        """The chapter's only inexactness: ch06's Algorithm 3 on **ch7's** JPEG (`near`, analysis R9).

        ``outputs/ch07/verify/alg3_python.npz`` is the port's own ``seaice_kmean_gvf`` run with the
        ``sea_ice_demo.m`` parameters; MATLAB's is in ``demo.mat``.  The measured gap is quoted in the report,
        so it must be pinned here rather than left as prose.
        """
        p = VERIFY / "alg3_python.npz"
        if not p.exists():
            pytest.skip("run reference/ch07/make_python_runs.py alg3")
        d = demo_ref()
        z = np.load(p)
        seg_diff = float(np.mean(np.asarray(z["seg"]) != np.asarray(d["seg"])))
        bk_diff = float(np.mean(np.asarray(z["bk"]) != np.asarray(d["bk"])))
        assert seg_diff < 0.005, f"seg differs on {100 * seg_diff:.3f} % of pixels"
        assert bk_diff < 0.005, f"bk differs on {100 * bk_diff:.3f} % of pixels"
        rp = ch7.ice_shape_enhancement(z["bk"], z["seg"], nbins=0)
        rm = ch7.ice_shape_enhancement(d["bk"], d["seg"], nbins=0)
        assert len(rp.ice_floe) == len(rm.ice_floe) == 433
        assert (len(rp.brash_ice), len(rm.brash_ice)) == (290, 274)
        for f in ("IceFloe", "BrashIce", "Slush", "Water"):
            assert abs(getattr(rp.coverage, f) - getattr(rm.coverage, f)) < 0.005, f
        tp = ch7.colorbar_area_ticks(np.concatenate([rp.color_floe, rp.color_brash]), 6)[1]
        tm = ch7.colorbar_area_ticks(np.concatenate([rm.color_floe, rm.color_brash]), 6)[1]
        assert tp.tolist() == tm.tolist() == [2, 173, 379, 640, 993, 1544, 2870]


class TestL2SeaIceTestJpg:
    """The ch7 copy of ``sea_ice_test.jpg`` is a *different encoding* from ch6's (analysis R4)."""

    def test_otsu_and_bwlabel_counts(self):
        d = ref("misc")
        assert abs(float(np.ravel(d["lev7"])[0]) - 162 / 255) < 1e-12
        assert abs(float(np.ravel(d["lev6"])[0]) - 162 / 255) < 1e-12
        assert int(np.ravel(d["n7"])[0]) == 215, "ch7's copy"
        assert int(np.ravel(d["n6"])[0]) == 231, "ch6's copy (the number every ch06 row was measured on)"

    def test_the_two_copies_really_differ(self):
        d = ref("misc")
        assert abs(float(np.ravel(d["frac_diff"])[0]) - 0.8413) < 1e-3
        assert int(np.ravel(d["maxabs"])[0]) == 43

    def test_python_reproduces_the_215(self):
        rgb = book_image("sea_ice_test.jpg")
        d = ref("misc")
        gray = rgb2gray_matlab(rgb)
        assert eq(gray, d["g7"]), "the JPEG decodes identically in MATLAB and imageio"
        level, _em = graythresh(gray)
        assert abs(level - float(np.ravel(d["lev7"])[0])) < 1e-12
        bw = im2bw(gray, level)
        assert eq(bw, d["bw7"])
        L = label_components(bw, 4)
        assert int(L.max()) == 215

    def test_matlab_sort_is_stable(self):
        """R8: ties keep their original order, which decides which label survives in ``out``."""
        d = ref("misc")
        ties = np.ravel(loadmat(str(REF / "misc_inputs.mat"))["ties"])
        order = np.argsort(ties, kind="stable")
        assert np.array_equal(order + 1, flat(d, "ties_ind").astype(np.int64))
        assert np.array_equal(ties[order], flat(d, "ties_A"))


class TestL2KMeans:
    """R9 — Algorithm 3's k-means on **ch7's** JPEG (ch06 measured 0 px on ch6's *different* encoding)."""

    def test_sorted_centres_and_pixel_agreement(self):
        d = ref("kmeans")
        from seaice.core.clustering import kmeans_lloyd
        rgb = book_image("sea_ice_test.jpg")
        g = rgb2gray_matlab(rgb)
        X = g.astype(float).ravel(order="F")[:, None]
        r = kmeans_lloyd(X, 3, init="kmeans++", seed=0)
        ml = np.sort(flat(d, "centres"))
        py = np.sort(np.ravel(r.centers))
        assert np.max(np.abs(ml - py)) < 1.5, f"sorted centres {py} vs {ml}"
        means = np.array([X[r.labels == i].mean() for i in range(3)])
        bk = np.ones(X.shape[0])
        bk[r.labels == int(np.argsort(means)[0])] = 0.0
        bk = bk.reshape(g.shape, order="F")
        agree = float(np.mean(bk == np.asarray(d["bk_ml"])))
        assert agree > 0.99, f"bk agreement {100 * agree:.3f} %"
        # ...and this is *not* the 0 px ch06 measured on ch6's copy of the same photograph
        assert agree < 1.0, "if this ever becomes 0 px the report's R4 note must be updated"

    def test_demo_bk_is_the_same_kmeans_mask(self):
        d, dd = ref("kmeans"), demo_ref()
        assert eq(dd["bk"], d["bk_ml"])


# =====================================================================================================================
# L4 — numbers quoted in the chapter text
# =====================================================================================================================
class TestL4Numbers:
    def test_script_parameter_block(self):
        """``sea_ice_demo.m`` lines 10-46 (ch7 adds ``se_th``, ``min_floe``, ``min_brash``)."""
        from seaice.ch06_gvf_snake import BOOK_PARAMS
        p = BOOK_PARAMS["sea_ice_demo"]
        assert (p["kms0"], p["Ra_min"], p["Ra"], p["Rc"], p["Rl"]) == (3, 10, 2500, 0.9, 2)
        assert (p["Num"], p["iter"]) == (500, 100)
        assert (p["se_th"], p["min_floe"], p["min_brash"]) == (50, 40, 1)
        assert (p["sigma"], p["GradientOn"], p["GVFOn"], p["mu"]) == (0, 1, 1, 0.1)
        assert (p["alpha"], p["beta"], p["gamma"], p["kappa"]) == (0.05, 0, 1, 0.5)
        assert (p["Dmin"], p["Dmax"], p["timer"]) == (0, 1, 1)
        assert (ch7.SE_TH, ch7.MIN_FLOE, ch7.MIN_BRASH) == (50, 40, 1)

    def test_eq_7_6_constants(self):
        assert (ch7.C1_COLOR, ch7.C2_AREA) == (10000, 1000)

    #: Every colour-bar tick list the chapter prints, with the page.  Each is **uniquely** invertible under the
    #: Eq. (7.6) tick arithmetic to a single ``(min(colors), d)`` pair — 5 (or 7) degrees of freedom of
    #: agreement per figure, which is what makes this a real check and not a fit.
    BOOK_TICKS = {
        "7.13": (3, 131, 277, 448, 656, 917, 1273),
        "7.15": (21, 114, 218, 333, 463, 612, 788),
        "7.19": (21, 203, 426, 713, 1118, 1809, 7264),
        "7.20": (21, 203, 426, 714, 1119, 1812, 9210),
        "7.21": (201, 334, 488, 671, 894, 1181, 1586, 2277, 7824),
        "7.26a": (2, 184, 407, 695, 1100, 1792, 8112),
        "7.26b": (2, 165, 359, 601, 921, 1394, 2322),
        "7.26c": (2, 173, 378, 638, 990, 1537, 2839),
        "7.28a": (2, 178, 391, 663, 1038, 1645, 3439),
        "7.28b": (2, 165, 361, 604, 926, 1404, 2353),
        "7.28c": (2, 165, 361, 605, 927, 1406, 2359),
    }

    @pytest.mark.parametrize("fig", sorted(BOOK_TICKS))
    def test_printed_colorbar_ticks_are_reproducible_from_two_integers(self, fig):
        """L4: the ``-round(1000 log(1 - c/10000))`` tick arithmetic reproduces every printed list exactly.

        The source images of Figs. 7.13/7.15/7.19/7.20/7.21 are **not shipped**, so the *piece areas* behind
        them cannot be reproduced (Open item).  What *is* verifiable, and is verified here, is the arithmetic:
        given only ``lo = min(colors)`` and the truncated step ``d``, the port must print exactly the book's
        integers — and the search below shows the ``(lo, d)`` pair is unique, so this is not a fit.
        """
        want = self.BOOK_TICKS[fig]
        n = 8 if len(want) == 9 else 6
        sols = []
        for lo in range(0, 10000):
            if int(ch7.color_to_area(np.array([float(lo)]))[0]) != want[0]:
                continue
            for step in range(1, 10000):
                vals = lo + step * np.arange(len(want), dtype=float)
                if vals[-1] >= 10000:
                    break
                if tuple(ch7.color_to_area(vals).tolist()) == tuple(want):
                    sols.append((lo, step))
        assert len(sols) == 1, f"{fig}: {len(sols)} (lo, d) pairs reproduce the printed ticks"
        lo, step = sols[0]
        colors = np.array([lo, lo + step * n], dtype=float)   # min and max that give exactly this ysh
        vals, labels = ch7.colorbar_area_ticks(colors, n)
        assert labels.tolist() == list(want), f"{fig}: {labels.tolist()}"

    #: How wide the *largest-piece area band* is that reproduces each printed tick list (measured by
    #: enumeration over areas 1..59 999 with the printed ``min`` colour held fixed).  Eq. (7.6) **saturates**:
    #: ``fix(10000 (1 - exp(-A/1000)))`` is 9997 for every A >= 8112, so four of the eleven printed lists are
    #: reproduced by an *unbounded* range of areas and therefore carry no information about the floe size.
    TICK_BANDS = {"7.13": (1273, 1274), "7.15": (788, 788), "7.19": (7265, 9210),
                  "7.20": (9211, None), "7.21": (7825, None), "7.26a": (8112, None),
                  "7.26b": (2322, 2327), "7.26c": (2839, 2849), "7.28a": (3439, 3457),
                  "7.28b": (2353, 2359), "7.28c": (2360, 2365)}
    TICK_LO = {"7.13": 29, "7.15": 207, "7.19": 207, "7.20": 207, "7.21": 1820,
               "7.26a": 19, "7.26b": 19, "7.26c": 19, "7.28a": 19, "7.28b": 19, "7.28c": 19}

    @pytest.mark.parametrize("fig", ["7.13", "7.15", "7.19", "7.26a", "7.26b", "7.28a"])
    def test_tick_lists_are_saturating_and_how_much(self, fig):
        """Which printed tick lists actually constrain the largest floe area, and which do not."""
        want, lo = self.BOOK_TICKS[fig], self.TICK_LO[fig]
        n = 8 if len(want) == 9 else 6
        lo_band, hi_band = self.TICK_BANDS[fig]
        hi_test = hi_band if hi_band is not None else 59999

        def ticks_for(area):
            c = float(ch7.size_color(area))
            return tuple(ch7.colorbar_area_ticks(np.array([float(lo), c]), n)[1].tolist())

        assert ticks_for(lo_band) == tuple(want)
        assert ticks_for(hi_test) == tuple(want)
        assert ticks_for(lo_band - 1) != tuple(want)
        if hi_band is not None:
            assert ticks_for(hi_band + 1) != tuple(want)

    def test_fig_7_26a_tick_match_is_a_saturation_coincidence(self):
        """The §7.3.2 snake=1 run reproduces Fig. 7.26(a)'s ticks at **both** downscalings — and would for any
        largest piece >= 8112 px, so the agreement is not evidence of parity (analysis R10 / open item)."""
        import json
        p = REF / "sensitivity_ticks.json"
        if not p.exists():
            pytest.skip("run reference/ch07/make_python_runs.py sens")
        rows = json.loads(p.read_text())
        want = list(self.BOOK_TICKS["7.26a"])
        assert [r["ticks"] for r in rows] == [want, want]
        assert {r["downscale"] for r in rows} == {1, 2}
        assert rows[0]["n_floe"] != rows[1]["n_floe"], "the two runs identify different numbers of floes"

    def test_fig_7_13_min_piece_is_3_px_and_max_1273_px(self):
        """The unique inversion of Fig. 7.13's ticks says the smallest identified piece is 3 px, the largest 1273."""
        lo, step = 29, 1195
        assert int(ch7.size_color(3)) == lo
        assert int(ch7.color_to_area(np.array([float(lo + 6 * step)]))[0]) == 1273

    def test_fig_7_2_sums(self):
        """L3/L4: the printed 13x23 matrix has 111 object pixels; closing 115, opening 104, cleaning 108."""
        r = ch7.morphological_cleaning(synth.FIG_7_2_IMAGE.astype(float), np.ones((2, 2), bool))
        assert int(synth.FIG_7_2_IMAGE.sum()) == 111
        assert [int(x.sum()) for x in (r.f1, r.f2, r.f0)] == [115, 104, 108]

    def test_fig_7_3_and_7_5_iteration_counts(self):
        """p. 148 "completes at the 8th iteration"; p. 150 "finishes at the 7th iteration"."""
        r3 = ch7.connected_component_extract(synth.FIG_7_3_IMAGE, synth.FIG_7_3_SEED, "square3", max_iter=20)
        r5 = ch7.hole_fill_dilation(synth.FIG_7_3_IMAGE, synth.FIG_7_5_SEED, "diamond1", max_iter=20)
        assert r3.n_iter == 8, "Fig. 7.3 — 8 blocks, X7 == A"
        assert r5.n_iter == 7, "Fig. 7.5 — X7 == X6, the printed blocks stop at X6 before the union"
        assert eq(r3.component, synth.FIG_7_3_IMAGE)

    def test_image_is_394_by_1038(self):
        """p. 168: "a 394 x 1038 MIZ image" — MATLAB reads it as 1038 x 394 x 3."""
        d = ref("misc")
        assert list(flat(d, "sz7").astype(int)) == [1038, 394, 3]

    def test_brash_ice_threshold_is_a_parameter_not_a_constant(self):
        """p. 160: brash ice is "no more than 2 m across"; the code's proxy is ``min_floe`` in **pixels**."""
        assert ch7.MIN_FLOE == 40
        bk, seg = FX.seg_fixtures()["thresholds"]
        a = ch7.ice_shape_enhancement(bk, seg, min_floe=40, nbins=0)
        b = ch7.ice_shape_enhancement(bk, seg, min_floe=100, nbins=0)
        assert len(b.ice_floe) < len(a.ice_floe)


# =====================================================================================================================
# Algorithm 5 written out on its own (reimplemented from the p. 160 pseudocode)
# =====================================================================================================================
class TestL1Algorithm5:
    def test_layers_partition_the_image(self):
        """Steps 3-5: ``FLOE u BRASH u SLUSH u WATER`` must be the whole image, pairwise disjoint."""
        bk, seg = FX.seg_fixtures()["mixed"]
        r = ch7.ice_shape_enhancement(bk, seg, nbins=0)
        floe, brash, slush, water = ch7.ice_types_classification(r.out, bk, 40)
        total = floe + brash + slush + water
        assert np.array_equal(total, np.ones_like(total)), "the four layers must tile the image exactly once"

    def test_script_form_reproduces_the_m_files_layers(self):
        """``book_threshold=False`` must give exactly the layers ``ice_shape_enhancement.m`` computes."""
        for name in ("mixed", "thresholds", "ring"):
            bk, seg = FX.seg_fixtures()[name]
            r = ch7.ice_shape_enhancement(bk, seg, nbins=0)
            floe, brash, slush, water = ch7.ice_types_classification(r.out, bk, 40, book_threshold=False)
            assert eq(floe, (r.index_floe != 0).astype(float)), name
            assert eq(brash, (r.index_brash != 0).astype(float)), name
            assert eq(slush, r.index_slush), name
            assert eq(water, r.index_water), name

    def test_book_form_differs_only_on_the_boundary_areas(self):
        """A hand-made label image with pieces of exactly 39/40/41 and 1/2 px isolates the ``>`` vs ``>=`` rule."""
        L = np.zeros((10, 40))
        areas = {1: 39, 2: 40, 3: 41, 4: 1, 5: 2}
        col = 0
        for lab, a in areas.items():
            idx = np.arange(a)
            L[idx % 8, col + idx // 8] = float(lab)
            col += a // 8 + 1
        assert [int((L == lab).sum()) for lab in areas] == list(areas.values())
        ice = (L != 0).astype(float)
        code = ch7.ice_types_classification(L, ice, 40, min_brash=1, book_threshold=False)
        book = ch7.ice_types_classification(L, ice, 40, min_brash=1, book_threshold=True)
        moved = code[0] != book[0]
        assert {int((L == lab).sum()) for lab in np.unique(L[moved]) if lab} == {40}
        newbrash = code[1] != book[1]
        assert {int((L == lab).sum()) for lab in np.unique(L[newbrash]) if lab} == {1, 40}
        # the code form drops every 1-pixel piece from BOTH ice layers (analysis R3)
        assert not code[0][L == 4].any() and not code[1][L == 4].any()
        assert book[1][L == 4].all()

    def test_pixel_union_with_ice(self):
        """Step 3 ``PIXEL = IDENTIFICATION U ICE`` — cleaning can turn water pixels into ice (p. 160)."""
        bk, seg = FX.seg_fixtures()["mixed"]
        r = ch7.ice_shape_enhancement(bk, seg, nbins=0)
        grew = int(np.count_nonzero((r.index != 0) & (bk == 0)))
        assert grew > 0, "the fixture must exercise the union (otherwise the step is untested)"
        assert np.all(r.index_water[r.index != 0] == 0)


# =====================================================================================================================
# Algorithm 6 / §7.3.1 (reimplemented from prose — no .m exists)
# =====================================================================================================================
class TestL1Algorithm6:
    def test_tile_grid_covers_the_image_with_the_requested_overlap(self):
        tiles, nr, nc = ch7.tile_grid((100, 140), 40, 10)
        cover = np.zeros((100, 140), bool)
        for t in tiles:
            r0, r1, c0, c1 = t.bounds
            cover[r0:r1, c0:c1] = True
        assert cover.all(), "the tiling must cover every pixel"
        assert nr * nc == len(tiles)

    def test_crop_regions_partition_the_image(self):
        tiles, _, _ = ch7.tile_grid((80, 96), 32, 8)
        count = np.zeros((80, 96), int)
        for t in tiles:
            r0, r1, c0, c1 = t.core
            count[r0:r1, c0:c1] += 1
        assert count.min() == 1 and count.max() == 1, "the kept (non-overlapping) parts must tile exactly once"

    def test_nearest_neighbour_resampling_preserves_the_label_set(self):
        """§7.3.1.2: the segmented image is categorical, so only nearest-neighbour may be used."""
        L = np.zeros((20, 20))
        L[4:9, 4:9] = 1.0
        L[12:18, 3:8] = 0.5
        u, v = np.meshgrid(np.linspace(0, 19, 40), np.linspace(0, 19, 40))
        R = ch7.resample_categorical(L, u, v)
        assert set(np.unique(R)) <= {0.0, 0.5, 1.0}


# =====================================================================================================================
# scripts run headless
# =====================================================================================================================
SCRIPTS = ["ch07_morphology_cleaning", "ch07_labeling", "ch07_filling", "ch07_filling_reconstruct",
           "ch07_ice_shape_enhancement", "ch07_sea_ice_demo", "ch07_local_processing", "ch07_sensitivity"]


def _run_script(script: str, out: Path, *flags: str) -> subprocess.CompletedProcess:
    cmd = [str(PY), str(ROOT / "scripts" / f"{script}.py"), "--no-show", "--out", str(out), *flags]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=3600)


SCRIPT_FLAGS = {
    "ch07_sea_ice_demo": ["--downscale", "4", "--max-seeds", "3"],
    "ch07_local_processing": ["--downscale", "4", "--tile", "80", "--overlap", "16", "--num", "20",
                              "--iter", "5", "--no-global"],
    "ch07_sensitivity": ["--downscale", "4", "--param", "snake", "--values", "1", "10"],
}


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_runs(script):
    out = VERIFY / "pytest_scripts" / script
    out.mkdir(parents=True, exist_ok=True)
    proc = _run_script(script, out, *SCRIPT_FLAGS.get(script, []))
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    needs_book = script in ("ch07_sea_ice_demo", "ch07_local_processing", "ch07_sensitivity")
    if needs_book and not DATA.exists():
        assert "SKIP" in proc.stdout
    else:
        assert list(out.glob("*.png")), "no figure written"


CLI_CASES = [
    ("ch07_labeling", ["--se", "diamond1"]),
    ("ch07_filling", ["--image", "fig76", "--se", "square3"]),
    ("ch07_filling_reconstruct", ["--se", "square3", "--i0", "alternative"]),
    ("ch07_morphology_cleaning", ["--se-size", "3"]),
    ("ch07_ice_shape_enhancement", ["--book-threshold", "--no-crop", "--min-floe", "10"]),
]


@pytest.mark.parametrize("script,flags", CLI_CASES, ids=[f"{s}:{' '.join(f)}" for s, f in CLI_CASES])
def test_script_cli_flags(script, flags):
    safe = "_".join(f.strip("-").replace(".", "p") for f in flags)[:60]
    out = VERIFY / "pytest_scripts" / f"{script}_{safe}"
    out.mkdir(parents=True, exist_ok=True)
    proc = _run_script(script, out, *flags)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
