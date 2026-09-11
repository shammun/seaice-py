"""Chapter 9 — model sea ice image processing applications: L1 synthetic truth + L2 MATLAB R2025a parity.

Reference generation: ``.venv/Scripts/python.exe reference/ch09/make_refs.py`` (10 sessions, ``matlab -batch``).

**Honesty contract for this chapter.**  Three of chapter 9's four external inputs (``04100_analyse.jpg``,
``dypic_05100_cam1_top.avi``, ``05100.avi``) do not ship and are unobtainable, so every MATLAB comparison below
that runs on a ``data/synthetic/ch09`` fixture is a **parity** result against MATLAB and **never** a book number.
The only book-derived inputs are ``model_ice.jpg`` (shipped, but not a printed figure) and the printed constants
of Tables 9.1/9.4 and Figs. 9.4/9.12/9.13.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.io import loadmat                                                    # noqa: E402

from reference.ch09 import fixtures as FX                                       # noqa: E402
from seaice import ch09_model_ice as ch9                                        # noqa: E402
from seaice.ch06_gvf_snake import component_criteria, gvf_distance              # noqa: E402
from seaice.ch07_ice_type import colorbar_area_ticks, size_color                # noqa: E402
from seaice.core.connectivity import bwareaopen, label_components               # noqa: E402
from seaice.core.io import read_image                                           # noqa: E402
from seaice.core.matlab_compat import rgb2gray_matlab                           # noqa: E402
from seaice.core.polygon import clip_polygon_convex, clip_polygon_rect, minboundrect   # noqa: E402
from seaice.core.regionprops import regionprops                                 # noqa: E402
from seaice.core.synth import model_ice_tank, model_ice_tank_video, segmented_floe_video  # noqa: E402
from seaice.core.threshold import BlockOtsu, block_otsu, graythresh, ice_concentration, im2bw  # noqa: E402
from seaice.core.video import read_video, write_video                           # noqa: E402
from tools.compare_arrays import assert_parity                                  # noqa: E402

REF = ROOT / "reference/ch09"
SCRATCH = ROOT / "outputs/ch09/verify/scratch"
MATLAB_ROOT = ROOT / "K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab"
CH9M = MATLAB_ROOT / "ch9"
SIFI = CH9M / "Model_Ice_Floe_Identification"
BOOK_IMG = ROOT / "data/book/ch09/Model_Ice_Floe_Identification/model_ice.jpg"


def needs(*names: str):
    """Skip unless every named reference ``.mat`` (and the scratch input it was produced from) exists."""
    missing = [n for n in names if not (REF / n).exists()]
    return pytest.mark.skipif(bool(missing),
                              reason=f"run reference/ch09/make_refs.py (missing {', '.join(missing)})")


needs_book = pytest.mark.skipif(not BOOK_IMG.exists(), reason="data/book/ch09 not present")


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def sc(x) -> float:
    """MATLAB saves a scalar as a 1x1 array; numpy 2 refuses ``int()`` on it."""
    return float(np.asarray(x).ravel()[0])


# ===============================================================================================================
# L1 — provenance and transcription (no MATLAB, no data)
# ===============================================================================================================

SHARED_14 = {
    "BoundMirrorEnsure.m": "3ccfe086d647712c1e817c5f5277e5d1",
    "BoundMirrorExpand.m": "7508640bdc9c0fda507197cd10273f30",
    "BoundMirrorShrink.m": "a4e165feb1655f0950487c08fb21f137",
    "gaussianBlur.m": "6310139f9903b20708ea369b34052d88",
    "gaussianMask.m": "c7c3f9ee8201fe8bef081b3901b02a36",
    "gradient2.m": "552231a16e602e25b50f682c3894735a",
    "GVF.m": "a0e17d489c4f24d12e9d7b65c0c615ac",
    "GVF_distance.m": "39cca98a5e06ac8e4700caf586444fc9",
    "minboundrect.m": "3e179fb6c5bff5c52d704277550d3ddf",
    "snakedeform.m": "1b7c415463840a66cf0a18693b164ad8",
    "snakedisp.m": "5eb085051d492ea14368d8af54c13c1d",
    "snakeindex.m": "351fe854ca7ea472db106356f179ee04",
    "snakeinterp.m": "9a7e7bc36dfd27bf83f97fe97b6c8a8c",
    "xconv2.m": "c4a1a1cc4675695b6e58c4de0572947e",
}


@pytest.mark.skipif(not SIFI.exists(), reason="MATLAB_ROOT not present")
def test_fourteen_ch9_files_are_byte_identical_to_ch6_and_ch7():
    """`analysis/ch09.md` §0.2 — 14 of the 18 ``Model_Ice_Floe_Identification/`` files are byte-identical to their
    ch6 **and** ch7 namesakes, so ch9 re-ports nothing (project rule 9).  CUMULATIVE said 13; ``minboundrect.m``
    is the 14th."""
    ch6 = MATLAB_ROOT / "ch6/Sea_Ice_Floe_Identification"
    ch7 = MATLAB_ROOT / "ch7/Sea_Ice_Floe_Identification"
    for name, md5 in SHARED_14.items():
        h9 = _md5(SIFI / name)
        assert h9 == md5, f"{name}: md5 {h9} != recorded {md5}"
        assert _md5(ch6 / name) == h9, f"{name} differs from ch6's copy"
        assert _md5(ch7 / name) == h9, f"{name} differs from ch7's copy"
    assert len(SHARED_14) == 14


@pytest.mark.skipif(not SIFI.exists(), reason="MATLAB_ROOT not present")
def test_the_four_new_ch9_files_are_not_shared():
    """The four genuinely new files must NOT match any ch6/ch7 namesake, and ``block_threshold.m`` must NOT be
    ``ch3/local_Otsu.m`` (its one numeric difference is the ``>=`` of `analysis/ch09.md` §0.3)."""
    assert _md5(SIFI / "model_ice_demo.m") == "4aad1d5502537cfb27505701972e9e9c"
    assert _md5(SIFI / "model_ice_model.m") == "921537ba37efd00d0a3c11dba1eb4537"
    assert _md5(SIFI / "rect.m") == "22d8f774679af6c47e6cd4a45a42dbe9"
    assert _md5(SIFI / "movie_floe.m") == "785fa42f8cfc6e227f2e1abe9ea0ad32"
    assert _md5(CH9M / "block_threshold.m") == "d0bbee029865e079c3765fd0edb731ae"
    assert _md5(CH9M / "block_threshold.m") != _md5(MATLAB_ROOT / "ch3/local_Otsu.m")
    src = (CH9M / "block_threshold.m").read_text(errors="replace").replace(" ", "")
    ch3 = (MATLAB_ROOT / "ch3/local_Otsu.m").read_text(errors="replace").replace(" ", "")
    assert "iftemp(r1,c1)>=th" in src and "iftemp(r1,c1)>th" not in src
    assert "iftemp(r1,c1)>th" in ch3 and "iftemp(r1,c1)>=th" not in ch3


@needs_book
def test_model_ice_jpg_is_the_shipped_image_and_not_a_printed_figure():
    """§0.4: the only shipped ch9 image, 181 x 76 x 3, md5 unique.  It is **not** a book figure (the analyst's
    whole-book NCC was 0.167 and the best ``matchTemplate`` 0.373), so every §9.3 "reproduction" is procedural."""
    assert _md5(BOOK_IMG) == "4d10f3030fb6b4c53827c91a0c9defa6"
    rgb = read_image(BOOK_IMG)
    assert rgb.shape == (181, 76, 3) and rgb.dtype == np.uint8


def test_N4_the_fig_9_4_caption_is_the_unweighted_mean_of_the_six_blocks():
    """Book number **N4** (p. 198).  ``block_threshold.m`` deleted ch3's pixel-weighted ``IC = sum(num)/(r*c)``,
    so the caption's "average IC = 83.14 %, average threshold = 84" can only be the plain mean of the six
    printed block values — and it is."""
    assert abs(float(np.mean(ch9.FIG_9_4_IC)) - 83.1467) < 5e-5
    assert f"{np.mean(ch9.FIG_9_4_IC):.2f}" == "83.15"          # 83.1467 -> the book *truncates* to 83.14
    assert int(np.floor(np.mean(ch9.FIG_9_4_IC) * 100)) / 100 == 83.14
    assert abs(float(np.mean(ch9.FIG_9_4_THRESH)) - 505 / 6) < 1e-12
    assert round(float(np.mean(ch9.FIG_9_4_THRESH))) == 84


def test_ic_mean_and_ic_coincide_for_equal_blocks_but_not_for_unequal_ones():
    """``BlockOtsu.ic_mean`` (the Fig. 9.4 caption) vs ``BlockOtsu.ic`` (ch3's pixel-weighted total).

    ``block_otsu`` **refuses** a grid that does not divide the image, so through that entry point the two are
    algebraically identical and a test using them interchangeably would prove nothing.  The distinction is
    exercised on an explicitly unequal-block record."""
    img = model_ice_tank(shape=(60, 90), seed=3)
    b = block_otsu(rgb2gray_matlab(img), 2, 3, compare="ge")
    assert b.ic_mean == pytest.approx(b.ic, abs=1e-15)          # equal blocks: identical by construction
    with pytest.raises(ValueError):
        block_otsu(rgb2gray_matlab(img), 2, 7)                  # 90 is not divisible by 7

    # an unequal-block record: one huge block at 100 % ice and one tiny block at 0 %
    uneq = BlockOtsu(thresholds=np.array([10.0, 20.0]), levels=np.array([0.04, 0.08]),
                     counts=np.array([9000, 0]), ic_local=np.array([1.0, 0.0]),
                     ic=9000 / 9100, bw=np.zeros((1, 1), bool),
                     slices=[(slice(0, 90), slice(0, 100)), (slice(90, 91), slice(0, 100))], compare="ge")
    assert uneq.ic_mean == pytest.approx(0.5)
    assert uneq.ic == pytest.approx(9000 / 9100)
    assert abs(uneq.ic_mean - uneq.ic) > 0.48                   # the two really are different quantities


def test_N8_crop_box_and_ice_concentration_denominator():
    """Book number **N8** — pure arithmetic on ``movie_otsu.m``/``movie_kmeans.m`` literals (lines 19-27, 46)."""
    y1, y2, x1, x2 = ch9.CROP_5100
    y3, _y2, x3, x4 = ch9.BOX_5100
    assert (y2 - y1 + 1, x2 - x1 + 1) == (221, 397)
    assert (y2 - y3 + 1, x4 - x3 + 1) == (94, 113)
    assert 94 * 113 == 10622
    assert ch9.IC_DENOMINATOR_5100 == 221 * 397 - 10622 == 77115


def test_table_9_4_transcription():
    """Book number **N13** — the 20 per-sub-image GVF iteration counts of Table 9.4 (p. 207). They are *inputs*,
    so only the transcription can be asserted."""
    assert ch9.TABLE_9_4 == (150, 65, 65, 65, 160, 60, 110, 130, 90, 130,
                             170, 160, 100, 90, 90, 100, 90, 80, 90, 80)
    assert len(ch9.TABLE_9_4) == 20


def test_tables_9_2_and_9_3_are_transcriptions_only():
    """Book numbers **N5/N6** (Table 9.2) and **N11** (Table 9.3).  ``04100_analyse.jpg`` and the four HSVA
    videos do not ship, so these can only be *transcribed*; nothing in this chapter reproduces them and the
    report lists them as `unverified`."""
    assert ch9.TABLE_9_2[5100] == dict(target=86.00, global_otsu=83.17, local_otsu=83.14, kmeans=82.86)
    assert ch9.TABLE_9_2[5200] == dict(target=70.00, global_otsu=62.50, local_otsu=62.51, kmeans=62.00)
    assert ch9.TABLE_9_3 == {5100: (200, 88.93), 5200: (300, 80.39), 5300: (600, 81.69), 5400: (300, 84.83)}
    # Table 9.2's local-Otsu column IS the Fig. 9.4 caption (N4), i.e. the mean of the six printed block values
    assert round(float(np.mean(ch9.FIG_9_4_IC)), 2) - ch9.TABLE_9_2[5100]["local_otsu"] < 0.02


def test_R8_table_9_1_strip_arithmetic_all_three_readings():
    """Risk **R8** — "one strip of 1.50 m, four of 1.00 m and nine of 0.50 m coincide with the percentages
    recorded in Table 9.1" (p. 196).

    Three readings of "percentage", computed here so the report can state which matches:

    * **by total strip width** (= by ice **area**, since every strip spans the same tank length):
      9x0.50 : 4x1.00 : 1x1.50 = 4.5 : 4.0 : 1.5 of 10.0 m = **45 / 40 / 15 %** -> **matches Table 9.1 exactly**
      in the printed column order (Floe size 1 = 0.50 m at 45 %, size 2 = 1.00 m at 40 %, size 3 = 1.50 m at 15 %);
    * by **strip count**: 9 : 4 : 1 of 14 = 64.3 / 28.6 / 7.1 %;
    * by **floe count** after the cross-cut into squares (a strip of width ``w`` and length ``L`` yields ``L/w``
      squares): 18 : 4 : 2/3 = 79.4 / 17.6 / 2.9 %.
    """
    widths = np.array([0.50, 1.00, 1.50])
    n_strips = np.array([9, 4, 1])
    by_width = n_strips * widths / (n_strips * widths).sum()
    by_strip_count = n_strips / n_strips.sum()
    per_len = n_strips / widths
    by_floe_count = per_len / per_len.sum()

    assert np.allclose(by_width * 100, [45.0, 40.0, 15.0])
    assert np.allclose(ch9.TABLE_9_1[5100]["shares"], [0.45, 0.40, 0.15])
    assert np.allclose(ch9.TABLE_9_1[5100]["sizes_m"], [0.50, 1.00, 1.50])
    # the printed shares ARE the width/area reading, in the printed order
    assert np.allclose(by_width, ch9.TABLE_9_1[5100]["shares"])
    # the other two readings are different, so the agreement above is a result and not an identity
    assert not np.allclose(by_strip_count, ch9.TABLE_9_1[5100]["shares"])
    assert not np.allclose(by_floe_count, ch9.TABLE_9_1[5100]["shares"])
    assert np.allclose(np.round(by_strip_count * 100, 1), [64.3, 28.6, 7.1])
    assert np.allclose(np.round(by_floe_count * 100, 1), [79.4, 17.6, 2.9])


def test_R7_fig_9_12_and_fig_9_13_tick_lists_come_from_ONE_colour_range():
    """Risk **R7** adjudicated (book numbers **N14/N15**).

    Fig. 9.12 prints **7** colour ticks ending 5878, Figs. 9.13/9.16 print **9** ending 5952, for data the text
    calls the same.  Both lists are reproduced *exactly* from a **single** colour range ``[198, 9974]`` by ch07's
    two ``ice_shape_enhancement.m`` conventions — ``n = 6`` on the map (line 196) and ``nn = 8`` on the histogram
    (line 227) — because ``ysh = min : fix((max-min)/n) : max`` **does not reach** ``max`` and the two step sizes
    stop at 9972 and 9974 respectively.  So the difference is the colon truncation, not different data.

    CUMULATIVE pitfall 60: the last label pins the largest floe only to a **band**, which is measured here.
    """
    colours = np.array([198.0, 9974.0])
    v6, lab6 = colorbar_area_ticks(colours, 6)
    v8, lab8 = colorbar_area_ticks(colours, 8)
    assert list(map(int, lab6)) == [20, 202, 424, 710, 1113, 1798, 5878]          # book Fig. 9.12
    assert list(map(int, lab8)) == [20, 153, 307, 488, 710, 996, 1398, 2081, 5952]  # book Figs. 9.13/9.16
    assert int(v6[-1]) == 9972 and int(v8[-1]) == 9974        # the colon stops short, by different amounts

    # the fixture discriminates: swapping the two conventions does NOT reproduce the printed lists
    assert list(map(int, colorbar_area_ticks(colours, 8)[1])) != list(map(int, lab6))
    assert list(map(int, colorbar_area_ticks(colours, 6)[1])) != list(map(int, lab8))

    # pre-image bands (pitfall 60): colour 198 pins the SMALLEST floe to exactly 20 px (= Ra_min), colour 9974
    # pins the LARGEST only to [5953, 5991] px -- so 5952 is a label, not the largest floe.
    areas = np.arange(1, 20000)
    c = np.asarray(size_color(areas.astype(float)))
    assert (areas[c == 198].min(), areas[c == 198].max()) == (20, 20)
    assert (areas[c == 9974].min(), areas[c == 9974].max()) == (5953, 5991)

    # The hypothesis is over-determined (16 printed integers from 2 free parameters), but the fit is a BAND, not
    # a point (pitfall 60 again): exactly four colour maxima reproduce BOTH printed lists.
    fits = [m for m in range(9000, 10000)
            if list(map(int, colorbar_area_ticks(np.array([198.0, float(m)]), 6)[1])) == list(map(int, lab6))
            and list(map(int, colorbar_area_ticks(np.array([198.0, float(m)]), 8)[1])) == list(map(int, lab8))]
    assert fits == [9974, 9975, 9976, 9977], f"colour-maximum band moved: {fits}"
    band = np.concatenate([areas[c == m] for m in fits])
    assert (int(band.min()), int(band.max())) == (5953, 6119)     # <- all the printed lists pin about the largest floe
    # and no colour minimum other than 198 fits either
    mins = [m for m in range(150, 260)
            if list(map(int, colorbar_area_ticks(np.array([float(m), 9974.0]), 6)[1])) == list(map(int, lab6))]
    assert mins == [198]


# ===============================================================================================================
# L1 — the errata, pinned in BOTH directions
# ===============================================================================================================

def test_E4_running_max_threshold_changes_the_answer():
    """Erratum **E4** (``movie_otsu.m`` line 39).  ``if I(i,j) >= t*255`` compares the pixel with the whole
    growing ``t`` vector, so ``if`` means ``all(...)`` and the effective threshold is ``max_{j<=k} t(j)``.

    The fixture is built so the per-frame threshold **falls** after frame 2 — without that, ``max = t(k)`` and
    the bug would be invisible.  The corrected variant is computed explicitly and asserted to differ."""
    frames = []
    for lvl in (60, 200, 60):                     # bright, then dark again -> a DECREASING Otsu threshold
        f = np.zeros((420, 560, 3), np.uint8)
        f[:, :, :] = 20
        f[180:300, 130:400, :] = lvl
        frames.append(f)
    stack = np.stack(frames)
    buggy = ch9.movie_otsu(stack)
    fixed = ch9.movie_otsu(stack, running_max_bug=False)
    assert buggy.effective_level[2] == buggy.t.max() > buggy.t[2]      # frame 3 uses frame 2's threshold
    assert fixed.effective_level[2] == fixed.t[2]
    assert buggy.counts[2] != fixed.counts[2], "fixture cannot distinguish E4 from the per-frame rule"
    assert buggy.IC[2] != fixed.IC[2]
    assert np.array_equal(buggy.counts[:2], fixed.counts[:2])          # before the drop they agree


def test_E5_n_is_the_reverse_cumulative_sum_and_IC_escapes_it():
    """Erratum **E5** (``movie_otsu.m`` line 40).  The body is ``n = n+1``, not ``n(k) = n(k)+1``, so every
    element of the growing vector is incremented: ``n_final[j] = sum_{m>=j} count_m``.  ``IC(k)`` reads ``n(k)``
    *inside* the same iteration, so ``IC`` is unharmed — pinned in both directions."""
    counts = np.array([7, 6, 4])

    def literal(counts):
        n: list[int] = []
        ic = []
        for k, c in enumerate(counts):
            n.append(0)
            for _ in range(int(c)):
                n = [v + 1 for v in n]
            ic.append(n[k])
        return np.array(n), np.array(ic)

    n, ic = literal(counts)
    assert list(n) == [17, 10, 4]
    assert list(ic) == list(counts)
    assert np.array_equal(n, np.cumsum(counts[::-1])[::-1])
    assert not np.array_equal(n, counts), "the closed form must differ from the un-buggy vector"


def test_E5_flag_is_honoured_by_movie_otsu():
    stack = segmented_floe_video(3, shape=(420, 560))
    buggy = ch9.movie_otsu(stack)
    fixed = ch9.movie_otsu(stack, increment_all_bug=False)
    assert np.array_equal(buggy.n, np.cumsum(buggy.counts[::-1])[::-1])
    assert np.array_equal(fixed.n, fixed.counts)
    assert np.array_equal(buggy.IC, fixed.IC)                     # E5 never reaches IC
    assert not np.array_equal(buggy.n, fixed.n)


def test_E7_the_aspect_band_is_symmetric_and_strict():
    """Erratum **E7** — p. 207 says rectangles with a ratio *less than* a threshold are removed, but
    ``model_ice_model.m`` line 34 is ``if k < k2 && k > k1`` with ``k`` **not** normalised to >= 1, i.e. a
    symmetric band with *strict* inequalities at both ends."""
    V, shape, k_expected = FX.ratio_rect_set()
    S = [ch9.RectFloe(Vertices=V[:, :, i], Center=V[:4, :, i].mean(axis=0), Area=0.0, Perimeter=0.0)
         for i in range(V.shape[2])]
    m = ch9.model_ice_model(S, np.zeros(shape), 0.4, 2.5)
    assert np.allclose(m.ratios, k_expected)
    assert list(m.accepted) == [2, 3, 4]              # 0-based: k = 0.50, 1.00, 2.40
    assert 0.39 not in k_expected[m.accepted] and 0.4 not in k_expected[m.accepted]
    assert 2.5 not in k_expected[m.accepted]


def test_E8_rect_metric_is_dead_code_in_matlab_but_validated_here():
    """Erratum **E8** — ``if (nargin<3)`` in a **two**-argument function is always true, so ``rect.m`` forces
    ``metric = 'a'`` and its ``strmatch`` validation branch is unreachable.  MATLAB therefore accepts a bogus
    metric silently (probe: ``probes.mat`` ``e8_ok == 1``); this port raises instead — a documented deviation."""
    bw = FX.rect_mask_a()
    a = ch9.rect(bw)
    assert [s.Area for s in a] == [s.Area for s in ch9.rect(bw, "p")]     # 'p' is ignored, as in MATLAB
    assert [s.Area for s in a] == [s.Area for s in ch9.rect(bw, "AREA")]
    with pytest.raises(ValueError):
        ch9.rect(bw, "not-a-metric")                                     # DEVIATION: MATLAB does not raise


def test_R13_movie_floe_empty_frame_modes():
    """Risk **R13** as *corrected* during the port: ``floe(k) = max([])`` with ``numel(floe) == k-1`` is a null
    assignment **past the end**, which MATLAB R2025a rejects; delete-and-shift only happens for
    ``k <= numel(x)``.  All three port modes are exercised."""
    frames = segmented_floe_video(4, blank_frame=2)
    with pytest.raises(ValueError, match="no component"):
        ch9.movie_floe(frames)
    deleted = ch9.movie_floe(frames, empty="delete")
    nanned = ch9.movie_floe(frames, empty="nan")
    assert deleted.floe.size == 3 and nanned.floe.size == 4
    assert np.isnan(nanned.floe[2])
    assert np.array_equal(deleted.floe, nanned.floe[~np.isnan(nanned.floe)])
    assert list(deleted.empty_frames) == [2]


# ===============================================================================================================
# L1 — the additive edits to ch02/ch03/ch06 primitives: defaults unmoved, opt-in branches exercised
# ===============================================================================================================

def test_block_otsu_default_is_gt_and_ge_really_differs():
    """Extension X1.  The ch3 default must be untouched; the ch9 rule must change the count on a fixture that
    *has* pixels sitting exactly on the threshold (a fixture without them cannot discriminate)."""
    import inspect
    assert inspect.signature(block_otsu).parameters["compare"].default == "gt"
    tie = FX.tie_block_image()
    gt = block_otsu(tie, 2, 3)                      # default
    ge = block_otsu(tie, 2, 3, compare="ge")
    assert gt.compare == "gt"
    assert list(ge.counts - gt.counts) == [0, 1, 0, 0, 2, 1]
    assert np.array_equal(ge.bw, gt.bw)             # the DISPLAYED tiles are im2bw's strict `>` either way
    with pytest.raises(ValueError):
        block_otsu(tie, 2, 3, compare="lt")


def test_component_criteria_default_is_ellipse_and_minrect_changes_the_failing_set():
    """Extension of ch06's ``component_criteria`` (risk R6).  ``ratio='ellipse'`` is the shipped code and must
    stay the default; ``ratio='minrect'`` is the book's p.-205 wording and must actually change something."""
    import inspect
    assert inspect.signature(component_criteria).parameters["ratio"].default == "ellipse"
    bw = np.zeros((40, 60), bool)
    bw[5:15, 5:35] = True
    rr, cc = np.mgrid[0:40, 0:60]
    bw |= (np.abs(rr - 28) + np.abs(cc - 15)) <= 6
    bw[20:26, 40:52] = True
    _l, n_e, _a, _s, _M, _m, rl_e, k_e = component_criteria(bw, 1e9, 0.0, 2.0, 4)
    _l, n_m, _a, _s, _M, _m, rl_m, k_m = component_criteria(bw, 1e9, 0.0, 2.0, 4, ratio="minrect")
    assert n_e == n_m == 3
    assert list(k_e) == [0] and list(k_m) == [0, 2], "the two ratio rules must disagree on this fixture"
    assert not np.allclose(rl_e, rl_m)
    with pytest.raises(ValueError):
        component_criteria(bw, 1e9, 0.0, 2.0, 4, ratio="bbox")


@needs_book
def test_gvf_distance_default_stop_is_criteria_and_count_stops_earlier():
    """Gap **G1** — Algorithm 7's ``N0 != N1`` convergence test is **not** in ``GVF_distance.m`` (line 80
    computes ``num`` and never compares it).  ``stop='count'`` is opt-in; the default must be unchanged, and the
    opt-in branch must actually fire."""
    import inspect
    assert inspect.signature(gvf_distance).parameters["stop"].default == "criteria"
    I = read_image(BOOK_IMG)
    kw = dict(sigma=0, GradientOn=1, GVFOn=1, Num=150, mu=0.1, iter=150, alpha=0.05, beta=0.0, gamma=1.0,
              kappa=0.6, Dmin=0.0, Dmax=1.0, Ra_min=20, Ra=1000, Rc=0.9, Rl=2, se_radius=3, timer=4,
              keep_history=False)
    base = gvf_distance(I, **kw)
    cnt = gvf_distance(I, **kw, stop="count")
    assert [p.num for p in base.passes] == [3, 36, 39, 39]
    assert not any(p.stopped_on_count for p in base.passes)
    assert cnt.passes[-1].stopped_on_count is True
    assert cnt.n_seeds == 55 and base.n_seeds == 67          # the opt-in really does less work
    assert int(cnt.bw1.sum()) == int(base.bw1.sum()) == 11407  # ... and the result is unchanged here


def test_clip_polygon_convex_refactor_did_not_move_clip_polygon_rect():
    """New primitive **P1**.  ``clip_polygon_rect`` (ch06, 46 verified clips) was refactored onto a shared
    Sutherland-Hodgman engine; the ch06 numbers are guarded by ``tests/test_ch06.py`` (the whole suite still
    passes) and by the fact that ``clip_polygon_rect`` still uses the **exact** axis-aligned intersector.

    Against the new general convex engine on the *same* rectangle the contract is the ch06/ch08 ``polybool``
    rule — vertex **sets** and rasterised masks, never vertex order: the two apply the four half-planes in a
    different order, which rotates the ring and moves where the rectangle corners are inserted.
    """
    import inspect
    from seaice.core.polygon import poly2mask
    assert "_intersect_axis" in inspect.getsource(clip_polygon_rect)
    rng = np.random.default_rng(0)
    worst = 0.0
    n_nonempty = 0
    for _ in range(200):
        n = int(rng.integers(3, 12))
        t = np.sort(rng.uniform(0, 2 * np.pi, n))
        r = rng.uniform(1.0, 9.0, n)
        x = 5.0 + r * np.cos(t)
        y = 5.0 + r * np.sin(t)
        a = clip_polygon_rect(x, y, (0.0, 8.0), (0.0, 7.0))
        b = clip_polygon_convex(x, y, [0.0, 8.0, 8.0, 0.0], [0.0, 0.0, 7.0, 7.0], drop_degenerate=False)
        assert a[0].shape == b[0].shape                     # same vertex count, same closing vertex
        if not a[0].size:
            continue
        n_nonempty += 1
        sa = {(round(u, 9), round(v, 9)) for u, v in zip(a[0][:-1], a[1][:-1])}
        sb = {(round(u, 9), round(v, 9)) for u, v in zip(b[0][:-1], b[1][:-1])}
        assert sa == sb, "vertex SETS differ"
        ma = poly2mask(a[0], a[1], 12, 12)
        mb = poly2mask(b[0], b[1], 12, 12)
        worst = max(worst, int((ma != mb).sum()))
    assert n_nonempty == 199
    # The two engines emit the SAME vertex set on all 199 non-empty clips but a rotated ring on 134 of them,
    # and ch06's documented poly2mask sensitivity to the start vertex then moves at most 4 px of a 12x12 raster
    # (9 clips of 199, mean 0.08 px).  That is the residual, measured -- not assumed away.
    assert worst <= 4, f"ring-rotation raster sensitivity grew to {worst} px"


def test_video_round_trip_is_lossless():
    """New primitive **P2**.  The Uncompressed-AVI contract (risk R2) on the Python side alone."""
    frames = segmented_floe_video(5, shape=(48, 64))
    p = ROOT / "outputs/ch09/verify/roundtrip.avi"
    write_video(p, frames, 12.0)
    back = read_video(p)
    assert back.shape == (48, 64, 3, 5)
    assert np.array_equal(back, np.transpose(frames, (1, 2, 3, 0)))


def test_synthetic_fixtures_are_deterministic():
    """The three Tier-3 generators are seeded; a re-run must give the identical array, otherwise the MATLAB
    references and the Python side would silently drift apart."""
    assert np.array_equal(model_ice_tank(shape=(60, 90), seed=1), model_ice_tank(shape=(60, 90), seed=1))
    assert np.array_equal(model_ice_tank_video(2), model_ice_tank_video(2))
    assert np.array_equal(segmented_floe_video(3), segmented_floe_video(3))
    assert not np.array_equal(model_ice_tank(shape=(60, 90), seed=1),
                              model_ice_tank(shape=(60, 90), seed=2))


def test_segmented_floe_video_carries_its_load_bearing_fixtures():
    """The ``05100.avi`` stand-in must pin ``bwareaopen``'s ``>= P`` rule (a 19-px and a 20-px blob) and
    ``bwlabel(.., 4)`` (a diagonally touching pair)."""
    f = segmented_floe_video(1)
    bw = im2bw(f[0])
    lab4 = label_components(bw, 4)
    lab8 = label_components(bw, 8)
    areas4 = np.bincount(lab4.ravel())[1:]
    assert 19 in areas4 and 20 in areas4
    kept = bwareaopen(bw, 20, 4)
    a_kept = np.bincount(label_components(kept, 4).ravel())[1:]
    assert 20 in a_kept and 19 not in a_kept              # `>= P`, not `> P`
    assert lab4.max() > lab8.max()                        # the diagonal pair splits under 4-connectivity


def test_model_ice_tank_blurs_before_the_noise_so_gt_and_ge_can_differ():
    """The tank generator applies a Gaussian **before** the noise on purpose.

    Hard-edged rectangles on a two-tone background leave an *empty band* in the histogram, and every threshold
    inside that band gives the same mask -- the ch08 degenerate-fixture trap, where a `gt` vs `ge` (or a
    global-vs-local Otsu) agreement would be forced by the fixture rather than by the algorithms.  Measured
    directly: with ``blur=0`` the 41 gray levels around the block threshold are almost all empty and moving the
    threshold by one level changes nothing; with the shipped default they are populated and it does.
    """
    def occupancy(blur):
        g = rgb2gray_matlab(model_ice_tank(shape=(120, 180), seed=5, blur=blur, noise=0.0))
        b = block_otsu(g, 2, 3, compare="ge")
        th = b.thresholds[0]
        blk = g[b.slices[0]].astype(float)
        window = np.arange(np.floor(th) - 20, np.floor(th) + 21)
        occupied = int(sum(1 for v in window if (blk == v).any()))
        masks = {(blk > v).tobytes() for v in window}
        return occupied, len(masks)

    flat_occ, flat_masks = occupancy(0.0)
    blur_occ, blur_masks = occupancy(0.9)
    assert flat_occ <= 3 and flat_masks <= 3, f"the un-blurred fixture is not degenerate: {flat_occ}, {flat_masks}"
    assert blur_occ >= 20 and blur_masks >= 20, f"the blurred fixture is still degenerate: {blur_occ}, {blur_masks}"
    assert blur_occ >= 5 * max(flat_occ, 1)

    # and on the real Tier-3 tank the `gt` / `ge` difference is a genuine 35-59 px per block (see the L2 test)
    g = rgb2gray_matlab(read_image(SCRATCH / "04100_analyse.jpg")) if (SCRATCH / "04100_analyse.jpg").exists() \
        else rgb2gray_matlab(model_ice_tank())
    b = block_otsu(g, 2, 3, compare="ge")
    at_threshold = [int((g[sl].astype(float) == th).sum()) for sl, th in zip(b.slices, b.thresholds)]
    assert all(v > 0 for v in at_threshold), at_threshold
    assert list(b.counts - block_otsu(g, 2, 3).counts) == at_threshold


def test_rect_ice_concentration_counts_overlaps_twice():
    """Book §9.3.2.1 p. 208 — the third ice concentration (87.75 % there) is ``sum(rect area)/(M*N)`` and the
    double counting of overlapping rectangles is the book's whole point, so it must NOT de-duplicate."""
    v1 = FX.ring(1.0, 1.0, 11.0, 11.0)
    v2 = FX.ring(6.0, 6.0, 16.0, 16.0)
    S = [ch9.RectFloe(Vertices=v, Center=v[:4].mean(axis=0), Area=100.0, Perimeter=40.0) for v in (v1, v2)]
    m = ch9.model_ice_model(S, np.zeros((40, 40)), 0.4, 2.5)
    assert ch9.rect_ice_concentration(m, (40, 40)) == pytest.approx(200.0 / 1600.0)
    assert ch9.rect_ice_concentration(m, (40, 40)) > ice_concentration(m.bw)   # union < sum of areas
    assert list(m.s_model[0].Intersection) == [2] and list(m.s_model[1].Intersection) == [1]


def test_fsd_error_is_a_bin_wise_difference_on_shared_centres():
    a, _ = ch9.floe_size_histogram([1, 2, 3, 40, 41], 5)
    b, _ = ch9.floe_size_histogram([1, 1, 1, 40, 41], 5)
    assert np.array_equal(ch9.fsd_error(a, b), a.astype(float) - b.astype(float))
    with pytest.raises(ValueError):
        ch9.fsd_error(np.zeros(4), np.zeros(5))


def test_preprocess_frame_blanks_before_it_crops():
    """``movie_otsu.m``/``movie_kmeans.m`` lines 24-30: ``I1``/``I2`` are stored and never read again, the vessel
    box is zeroed **before** the crop, and ``r2*c2`` is removed from the IC denominator but not the numerator."""
    f = np.full((420, 560, 3), 200, np.uint8)
    cropped, I1, I2, r2, c2 = ch9.preprocess_frame(f)
    assert cropped.shape == (221, 397, 3) and I1.shape == (221, 397, 3) and I2.shape == (94, 113, 3)
    assert (r2, c2) == (94, 113)
    assert (I1 == 200).all() and (I2 == 200).all()               # captured BEFORE the blanking
    y3, y2, x3, x4 = ch9.BOX_5100
    y1, _y2, x1, _x2 = ch9.CROP_5100
    assert (cropped[y3 - y1:y2 - y1 + 1, x3 - x1:x4 - x1 + 1] == 0).all()
    assert int((cropped == 0).sum()) == 94 * 113 * 3


# ===============================================================================================================
# L2 — MATLAB R2025a parity.  Session 1 (video) is a PRECONDITION for every algorithm claim below.
# ===============================================================================================================

@needs("video.mat")
def test_uncompressed_avi_decodes_bit_identically_in_matlab_and_python():
    """**Risk R2 closed — this is a precondition, not a result.**  Every ice-concentration and floe-size number
    in §9.2.2/§9.3.3 below rests on MATLAB's ``VideoReader`` and ffmpeg seeing the *same* frames.  Both Tier-3
    fixtures are Uncompressed AVI (``codec='rawvideo'``), and MATLAB's ``read(VideoReader(...))`` is compared
    byte for byte with :func:`seaice.core.video.read_video` — including the ``(H, W, 3, N)`` **axis order**,
    which is pinned against MATLAB's own ``size()``, not merely against the pixel values."""
    d = loadmat(REF / "video.mat")
    for key, n, name in (("F1", "n1", "dypic_synth_top.avi"), ("F2", "n2", "05100_synth_segmented.avi")):
        ml = d[key]
        py = read_video(SCRATCH / name)
        assert py.dtype == np.uint8 == ml.dtype
        assert py.shape == ml.shape, f"{name}: axis order {py.shape} vs MATLAB {ml.shape}"
        assert np.array_equal(py, ml), f"{name}: decode is NOT bit-identical"
        assert int(d[n].ravel()[0]) == py.shape[3]
    # the axis order really is MATLAB's (H, W, 3, N) and not imageio's (N, H, W, 3)
    assert tuple(d["sz1"].ravel()) == (480, 640, 3, 24)
    assert tuple(d["sz2"].ravel()) == (240, 320, 3, 40)
    assert int(d["h1"].ravel()[0]) == 480 and int(d["w1"].ravel()[0]) == 640
    assert sorted(d["u2"].ravel().tolist()) == [0, 255]          # the segmented fixture really is binary


@needs("probes.mat")
def test_matlab_confirms_the_errata_and_the_removed_functions():
    d = loadmat(REF / "probes.mat")
    # E4
    assert list(d["e4_vec"].ravel()) == [1, 0]
    assert int(sc(d["e4_taken"])) == 0 and int(sc(d["e4_taken_all_below"])) == 1
    assert int(sc(d["e4_empty_taken"])) == 0              # `if <empty>` is false, as ch08's `if xx ~= NaN` relies on
    # E5 -- MATLAB's own numbers, and the closed form the port uses
    counts = d["e5_counts"].ravel().astype(int)
    assert list(d["e5_n"].ravel()) == [17, 10, 4]
    assert list(d["e5_ic"].ravel()) == list(counts)
    assert np.array_equal(d["e5_n"].ravel(), np.cumsum(counts[::-1])[::-1])
    # R13 -- MATLAB RAISES (see the dedicated r13.mat probe for the two error forms)
    assert str(d["r13_id"][0]) == "MATLAB:subsdeldimmismatch"
    assert "out of range for deletion" in str(d["r13_msg"][0])
    assert list(d["r13_delete"].ravel()) == [1, 3]   # delete-and-shift only for k <= numel(x)
    # the ch3 `>` vs ch9 `>=` rule and im2bw's own strict `>`
    assert list(d["cmp_gt"].ravel()) == [0, 0, 1]
    assert list(d["cmp_ge"].ravel()) == [0, 1, 1]
    assert list(d["cmp_im2bw"].ravel()) == [0, 0, 1]
    assert list(d["cmp_half"].ravel()) == [0, 1, 1]  # a half-integer threshold is compared in double
    # regionprops 'basic' (extension X2)
    assert [str(x[0]) for x in d["rp_fields"].ravel()] == ["Area", "Centroid", "BoundingBox"]
    from seaice.core.regionprops import BASIC_PROPERTIES
    assert list(BASIC_PROPERTIES) == ["Area", "Centroid", "BoundingBox"]
    # R3: mmreader / movie2avi / avifile are GONE; polybool / strmatch / VideoReader survive
    assert int(sc(d["ex_mmreader"])) == 0 and int(sc(d["ex_movie2avi"])) == 0 and int(sc(d["ex_avifile"])) == 0
    assert int(sc(d["ex_polybool"])) == 2 and int(sc(d["ex_strmatch"])) == 2 and int(sc(d["ex_videoreader"])) == 2
    # E8: a bogus metric does NOT raise in MATLAB (the validation branch is unreachable)
    assert int(sc(d["e8_ok"])) == 1 and str(d["e8_id"][0] if d["e8_id"].size else "") == ""
    assert int(sc(d["e8_same_as_p"])) == 1
    # level-free im2bw on RGB = rgb2gray then > 127.5
    gray = d["gray_of_rgbtest"]
    assert np.array_equal(d["im2bw_nolevel"].astype(bool), gray > 127.5)
    rgb = np.repeat(np.array([[0, 127, 128, 255], [10, 20, 30, 40]], np.uint8)[:, :, None], 3, axis=2)
    assert np.array_equal(rgb2gray_matlab(rgb), gray)
    assert np.array_equal(im2bw(rgb), d["im2bw_nolevel"].astype(bool))


@needs("probes.mat")
def test_minboundrect_ring_matches_matlab_on_the_sliver():
    """``minboundrect`` is reused from ch06 (`exact`); this re-measures it on ch9's own fixture and pins the ring
    contract ``rect.m`` depends on: **5 points, closed**, with a positive cross product (counter-clockwise in the
    ``(x = column, y = row)`` frame ``polybool`` then warns about)."""
    d = loadmat(REF / "probes.mat")
    lab = label_components(FX.rect_mask_a(), 4)
    r, c = np.nonzero(lab == 2)
    x, y, a, p = minboundrect(c.astype(float) + 1, r.astype(float) + 1, "a")
    assert x.size == 5 and x[0] == x[-1] and y[0] == y[-1]
    assert int(sc(d["mr_closed"])) == 1
    assert sc(d["mr_cross"]) > 0
    assert_parity(x, d["mrx"].ravel(), "float", atol=1e-12, name="minboundrect x")
    assert_parity(y, d["mry"].ravel(), "float", atol=1e-12, name="minboundrect y")
    assert a == pytest.approx(sc(d["mra"]), abs=1e-12)
    assert p == pytest.approx(sc(d["mrp"]), abs=1e-12)


@needs("ch09_block.mat", "block.mat")
def test_block_threshold_parity_on_the_tier3_tank():
    """``block_threshold.m`` (§9.2.1, Fig. 9.4) run verbatim in MATLAB on the **Tier-3** tank image.

    Parity against MATLAB, **not** a book number: ``04100_analyse.jpg`` does not ship (risk R1), so N2/N3/N5/N6
    stay `unverified`."""
    d = loadmat(REF / "ch09_block.mat")
    e = loadmat(REF / "block.mat")
    rgb = read_image(SCRATCH / "04100_analyse.jpg")
    gray = rgb2gray_matlab(rgb)
    assert np.array_equal(gray, e["Itank"]), "the two engines decoded the same JPEG differently"
    b = ch9.block_threshold(rgb)
    assert (int(sc(d["r"])), int(sc(d["c"]))) == gray.shape == (348, 1770)
    assert_parity(b.thresholds, d["thresh"].ravel(), "float", atol=0.0, name="block thresh")
    assert_parity(b.ic_local, d["IC"].ravel(), "float", atol=0.0, name="block IC")
    assert_parity(b.bw, e["bw_tank"].astype(bool), "binary", name="block tiles")
    # the fixture is NOT degenerate: 35-59 pixels sit exactly ON each block threshold, so `>` and `>=` differ
    at_th = e["at_th"].ravel().astype(int)
    assert at_th.min() >= 35 and at_th.max() <= 59 and (at_th > 0).all()
    gt = block_otsu(gray, 2, 3)
    assert list(b.counts - gt.counts) == list(at_th)


@needs("block.mat")
def test_gt_versus_ge_on_the_constructed_tie_fixture_matches_matlab():
    """`analysis/ch09.md` §0.3 — the one numeric difference between ch3's ``local_Otsu.m`` and ch9's
    ``block_threshold.m``.  MATLAB runs **both** comparison rules on a fixture with pixels exactly on the
    threshold (and one block with a half-integer level), so the two cannot agree by luck."""
    e = loadmat(REF / "block.mat")
    tie = FX.tie_block_image()
    ge = block_otsu(tie, 2, 3, compare="ge")
    gt = block_otsu(tie, 2, 3, compare="gt")
    assert_parity(ge.thresholds, e["thresh_tie"].ravel(), "float", atol=0.0, name="tie thresholds")
    assert list(ge.counts) == list(e["n_ge_tie"].ravel().astype(int))
    assert list(gt.counts) == list(e["n_gt_tie"].ravel().astype(int))
    assert list(e["n_ge_tie"].ravel() - e["n_gt_tie"].ravel()) == [0, 1, 0, 0, 2, 1]
    assert 104.5 in list(ge.thresholds)                  # graythresh tie averaging is exercised
    assert_parity(ge.bw, e["bw_tie"].astype(bool), "binary", name="tie tiles")


@needs("block.mat")
def test_global_otsu_tank_ice_concentration_parity():
    """§9.2.1 Fig. 9.3 has no ``.m``; ``tank_ice_concentration(method='otsu')`` is a two-line composition of
    ch03 primitives and is compared with MATLAB's ``graythresh``/``im2bw`` on the same image."""
    e = loadmat(REF / "block.mat")
    rgb = read_image(SCRATCH / "04100_analyse.jpg")
    ic, mask, th = ch9.tank_ice_concentration(rgb, "otsu")
    assert th / 255.0 == pytest.approx(sc(e["glevel_tank"]), abs=0.0)
    assert ic == pytest.approx(sc(e["IC_global"]), abs=0.0)
    ic_k, mask_k, th_k = ch9.tank_ice_concentration(rgb, "kmeans")
    assert np.isnan(th_k)
    assert abs(ic_k - ic) < 0.05                        # k=2 and Otsu land close on a bimodal field
    assert (mask_k == mask).mean() > 0.95


@needs("ch09_movie_otsu.mat")
def test_movie_otsu_parity():
    """``movie_otsu.m`` (§9.2.2, Figs. 9.6-9.8/9.10) run verbatim in MATLAB on the Tier-3 uncompressed AVI —
    including both errata.  Every array the script leaves in the workspace is compared."""
    d = loadmat(REF / "ch09_movie_otsu.mat")
    F = read_video(SCRATCH / "dypic_synth_top.avi")
    r = ch9.movie_otsu(F)
    assert (int(sc(d["r"])), int(sc(d["c"])), int(sc(d["r2"])), int(sc(d["c2"]))) == (221, 397, 94, 113)
    assert r.denominator == 221 * 397 - 94 * 113 == 77115
    assert_parity(r.t, d["t"].ravel(), "float", atol=0.0, name="graythresh level t(k)")
    assert list(r.n) == list(d["n"].ravel().astype(np.int64)), "the E5-corrupted n vector"
    assert_parity(r.IC, d["IC"].ravel(), "float", atol=0.0, name="IC(k)")
    assert_parity(r.bw, np.transpose(d["bwstack"], (2, 0, 1)).astype(bool), "binary", name="bw(k).cadata")
    assert np.array_equal(r.gray, np.transpose(d["graystack"], (2, 0, 1)))
    c0, I1, I2, _r2, _c2 = ch9.preprocess_frame(F[:, :, :, 0])
    assert np.array_equal(I1, d["I1first"]) and np.array_equal(I2, d["I2first"])


@needs("ch09_movie_otsu.mat")
def test_the_video_fixture_makes_E4_observable_and_IC_disagrees_with_the_written_avi():
    """Two claims the report rests on, measured on MATLAB's own reference.

    1. the fixture's Otsu thresholds **fall** after frame 7 (143 -> 100), so the running maximum is not
       ``t(k)`` and E4 is observable — 17 of 24 frames change when the bug is switched off;
    2. the plotted ``IC`` and the written ``otsu.avi`` legitimately **disagree**: the count uses ``>=`` at the
       running maximum, ``im2bw`` uses strict ``>`` at the *per-frame* level."""
    d = loadmat(REF / "ch09_movie_otsu.mat")
    t255 = d["t"].ravel() * 255.0
    assert t255.max() > t255[-1] and np.any(np.diff(t255) < 0), "monotone thresholds would hide E4"
    F = read_video(SCRATCH / "dypic_synth_top.avi")
    buggy = ch9.movie_otsu(F)
    fixed = ch9.movie_otsu(F, running_max_bug=False)
    assert int((buggy.counts != fixed.counts).sum()) == 17
    assert np.allclose(buggy.effective_level, np.maximum.accumulate(buggy.t))
    # the AVI's masks are not the counted pixels
    avi_counts = buggy.bw.reshape(buggy.bw.shape[0], -1).sum(axis=1)
    assert not np.array_equal(avi_counts, buggy.counts)
    gt = ch9.movie_otsu(F, running_max_bug=False, count_rule="gt")
    assert np.array_equal(gt.counts, avi_counts), "with t(k) and strict > the two rules must coincide"


@needs("ch09_movie_kmeans.mat")
def test_movie_kmeans_parity():
    """``movie_kmeans.m`` (§9.2.2).  Line 44 calls the **Statistics Toolbox** ``kmeans`` (its own RNG), so the
    label *numbers* are not reproducible; the contract is on the sorted cluster means, the masks and ``IC``.
    Everything around the clustering (crop, blanking, ``si``, denominator, column-major ``reshape``) is exact."""
    d = loadmat(REF / "ch09_movie_kmeans.mat")
    F = read_video(SCRATCH / "dypic_synth_top.avi")
    r = ch9.movie_kmeans(F)
    assert tuple(d["si"].ravel()) == (221, 397)
    assert r.denominator == 77115 == int(d["si"].ravel()[0]) * int(d["si"].ravel()[1]) - 94 * 113
    assert_parity(r.IC, d["IC"].ravel(), "float", atol=0.0, name="k-means IC(k)")
    assert_parity(np.sort(r.s, axis=1), np.sort(d["smeans"], axis=0).T, "float", atol=1e-12,
                  name="sorted cluster means")
    out_ml = np.transpose(d["outstack"], (2, 0, 1))
    agree = float((out_ml == r.out).mean())
    assert agree >= 0.99, f"mask agreement {agree}"
    # label numbers are NOT reproducible -- this is the documented `approx` part
    assert not np.array_equal(r.a, d["a"].ravel())


@needs("ch09_movie_floe.mat")
def test_movie_floe_parity():
    """``movie_floe.m`` (§9.3.3, Fig. 9.18) run verbatim in MATLAB (``mmreader`` -> ``VideoReader``, the single
    IO patch) on the Tier-3 segmented AVI."""
    d = loadmat(REF / "ch09_movie_floe.mat")
    F = read_video(SCRATCH / "05100_synth_segmented.avi")
    r = ch9.movie_floe(F, keep_labels=True)
    assert list(r.floe.astype(int)) == list(d["floe"].ravel().astype(int))
    ims = np.array([im2bw(F[:, :, :, k]) for k in range(F.shape[3])])
    outs = np.array([bwareaopen(x, 20, 4) for x in ims])
    assert_parity(ims, np.transpose(d["imstack"], (2, 0, 1)).astype(bool), "binary", name="im2bw(mov)")
    assert_parity(outs, np.transpose(d["bwstack"], (2, 0, 1)).astype(bool), "binary", name="bwareaopen(.,20,4)")
    assert [int(l.max()) for l in r.labels] == list(d["nc"].ravel().astype(int))
    assert np.array_equal(r.labels[4], d["lab5"].astype(np.int64))    # bwlabel numbering, not just the partition
    rp = regionprops(r.labels[4], "basic")
    assert list(np.array([s.Area for s in rp])) == list(d["area5"].ravel())
    assert_parity(np.array([s.Centroid for s in rp]), d["cent5"], "float", atol=0.0, name="Centroid")
    assert_parity(np.array([s.BoundingBox for s in rp]), d["bbox5"], "float", atol=0.0, name="BoundingBox")


@needs("rect.mat")
def test_rect_parity():
    """``rect.m`` (§9.3.2.1, Fig. 9.15(a)) on two **non-square** constructed masks, including a 5 : 1 sliver, a
    45-degree square, an exact square, an L and a border-touching blob."""
    d = loadmat(REF / "rect.mat")
    for mask, key in ((FX.rect_mask_a(), "A"), (FX.rect_mask_b(), "B")):
        S = ch9.rect(mask)
        assert len(S) == int(sc(d["n" + key])) == int(sc(d["num" + key]))
        assert np.array_equal(label_components(mask, 4), d["lab" + key].astype(np.int64))
        assert_parity(np.stack([s.Vertices for s in S], axis=2), d["V" + key], "float", atol=1e-12,
                      name=f"rect {key} Vertices")
        assert_parity(np.array([s.Center for s in S]), d["C" + key], "float", atol=1e-12, name=f"rect {key} Center")
        assert_parity(np.array([s.Area for s in S]), d["A" + key].ravel(), "float", atol=1e-12,
                      name=f"rect {key} Area")
        assert_parity(np.array([s.Perimeter for s in S]), d["P" + key].ravel(), "float", atol=1e-12,
                      name=f"rect {key} Perimeter")


@needs("rect.mat")
def test_the_rect_m_row_column_swap_is_caught_by_the_fixture_but_not_by_Area():
    """``rect.m`` line 52 calls ``minboundrect(c, r, 'a')`` — **column first**.  MATLAB computed the swapped call
    for every component: the rectangle **Areas are identical** under the swap (a transposition preserves area),
    so an Area-only test proves nothing; the non-square fixture catches it on ``Vertices``/``Center``."""
    d = loadmat(REF / "rect.mat")
    assert np.allclose(d["AAswap"].ravel(), d["AA"].ravel()), "Area cannot discriminate the swap"
    assert not np.allclose(d["VAswap"], d["VA"]), "the fixture does not exercise the swap"
    S = ch9.rect(FX.rect_mask_a())
    V = np.stack([s.Vertices for s in S], axis=2)
    assert np.allclose(V, d["VA"], atol=1e-12)
    assert not np.allclose(V, d["VAswap"], atol=1e-6)
    # and the port really does pass (column, row): swapping it reproduces MATLAB's swapped answer
    lab = label_components(FX.rect_mask_a(), 4)
    r, c = np.nonzero(lab == 2)
    sx, sy, _a, _p = minboundrect(r.astype(float) + 1, c.astype(float) + 1, "a")
    assert np.allclose(np.column_stack([sx, sy]), d["VAswap"][:, :, 1], atol=1e-12)


@needs("rect.mat")
def test_matlab_refuses_a_collinear_component_but_this_port_does_not():
    """Deviation (new, found by this verification).  ``minboundrect.m`` line 102 calls ``convhull`` whenever the
    component has more than 3 pixels; MATLAB R2025a **errors** on a collinear cloud
    (``MATLAB:convhull:EmptyConvhull2DErrId``), so ``rect.m`` cannot run on a mask containing a 1-pixel line.
    The Python port returns a degenerate rectangle instead — more permissive than the original."""
    d = loadmat(REF / "rect.mat")
    assert int(sc(d["deg_ok"])) == 0
    assert str(d["deg_id"][0]) == "MATLAB:convhull:EmptyConvhull2DErrId"
    assert "collinear" in str(d["deg_msg"][0])
    assert str(d["deg_line_id"][0]) == "MATLAB:convhull:EmptyConvhull2DErrId"
    assert d["deg_single_id"].size == 0 or str(d["deg_single_id"][0]) == ""   # a SINGLE point is fine (n <= 1)
    S = ch9.rect(FX.rect_mask_degenerate())                                   # the port does not raise
    assert len(S) == 2 and S[1].Area == pytest.approx(0.0)


@needs("polybool.mat")
def test_polybool_emptiness_parity_and_the_drop_degenerate_contract():
    """New primitive **P1** (risk R4).  ``polybool`` is compiled GPC with no readable source and an
    unreproducible vertex order, but ``model_ice_model.m`` consumes only ``isempty(xx)``.  MATLAB was probed on
    ten pairs — contained, disjoint, partial, **edge-touching**, **vertex-touching**, **partial-edge-overlap**,
    rotated/rotated, identical, a 1e-9 gap and a 1e-9 overlap — and every emptiness verdict is reproduced.

    The test also proves ``drop_degenerate`` is load-bearing: with ``False`` the port disagrees with MATLAB on
    3 of the 10 pairs."""
    d = loadmat(REF / "polybool.mat")
    A, B, names = FX.polybool_pairs()
    ml_empty = d["pb_empty"].ravel().astype(bool)
    ml_taken = d["pb_taken"].ravel().astype(bool)
    assert np.array_equal(ml_empty, ~ml_taken)            # `if xx ~= NaN` IS `~isempty(xx)` (erratum E9)
    wrong_without = 0
    for i, nm in enumerate(names):
        x, _y = clip_polygon_convex(A[:, 0, i], A[:, 1, i], B[:, 0, i], B[:, 1, i])
        assert (x.size == 0) == bool(ml_empty[i]), f"{nm}: emptiness differs from polybool"
        if x.size:
            assert x.size == int(d["pb_nv"].ravel()[i]), f"{nm}: vertex count"
            px = np.round(np.column_stack([x, _y])[:-1], 9)
            ref = np.round(np.column_stack([d["PX"][:int(d["pb_nv"].ravel()[i]), i],
                                            d["PY"][:int(d["pb_nv"].ravel()[i]), i]])[:-1], 9)
            assert {tuple(p) for p in px} == {tuple(p) for p in ref}, f"{nm}: vertex SET differs"
        xr, _ = clip_polygon_convex(A[:, 0, i], A[:, 1, i], B[:, 0, i], B[:, 1, i], drop_degenerate=False)
        if (xr.size == 0) != bool(ml_empty[i]):
            wrong_without += 1
    assert wrong_without == 3, f"drop_degenerate must matter on exactly the 3 zero-area cases, got {wrong_without}"
    # the three zero-area relations MATLAB drops
    for nm in ("edge_touching", "vertex_touching", "partial_edge"):
        assert bool(ml_empty[names.index(nm)]) is True


@needs("model.mat")
def test_model_ice_model_parity_on_the_constructed_rectangle_set():
    """``model_ice_model.m`` (§9.3.2.1, Fig. 9.15(b)) on eleven rectangles realising all six overlap relations."""
    d = loadmat(REF / "model.mat")
    V, shape, rel = FX.model_rect_set()
    S = []
    for i in range(V.shape[2]):
        v = V[:, :, i]
        S.append(ch9.RectFloe(Vertices=v, Center=v[:4].mean(axis=0),
                              Area=float(abs((v[1,0]-v[0,0])*(v[3,1]-v[0,1]) - (v[1,1]-v[0,1])*(v[3,0]-v[0,0]))),
                              Perimeter=float(np.hypot(*(np.diff(v, axis=0)).T).sum())))
    m = ch9.model_ice_model(S, np.zeros(shape), 0.4, 2.5)
    assert len(m.s_model) == int(sc(d["nM"])) == 11
    assert list(m.accepted + 1) == list(d["accepted"].ravel().astype(int))
    assert_parity(m.ratios, d["ratios"].ravel(), "float", atol=1e-12, name="k ratios")
    assert_parity(np.stack([s.Vertices for s in m.s_model], axis=2), d["MV"], "float", atol=1e-12, name="MV")
    assert_parity(np.array([s.Center for s in m.s_model]), d["MC"], "float", atol=1e-12, name="MC")
    assert_parity(m.bw, d["bw_model"].astype(bool), "binary", name="roipoly union bw")
    MI = d["MI"].astype(bool)
    py = np.zeros_like(MI)
    for i, f in enumerate(m.s_model):
        for j in f.Intersection:
            py[i, j - 1] = True
    assert np.array_equal(py, MI), "Intersection index sets differ from polybool's"
    # the relations are all present, and only the two positive-area ones are flagged
    assert MI[rel["contained"][0] - 1, rel["contained"][1] - 1]
    assert MI[rel["partial"][0] - 1, rel["partial"][1] - 1]
    for nm in ("edge_touching", "vertex_touching", "partial_edge", "disjoint"):
        i, j = rel[nm]
        assert not MI[i - 1, j - 1], f"{nm} must NOT be flagged"


@needs("model.mat")
def test_model_ice_model_k_band_boundaries_are_strict_in_matlab_too():
    d = loadmat(REF / "model.mat")
    assert list(d["ratio_kept"].ravel().astype(int)) == [3, 4, 5]      # 1-based: k = 0.50, 1.00, 2.40
    assert int(sc(d["ratio_nM"])) == 3
    assert np.allclose(d["ratio_k"].ravel(), [0.39, 0.40, 0.50, 1.00, 2.40, 2.50])


@needs("ch09_demo.mat")
@needs_book
def test_model_ice_demo_chain_against_matlab():
    """``model_ice_demo.m`` (§9.3.1/§9.3.2) end to end on the shipped ``model_ice.jpg``, with the authors' own
    commented lines 53-66 un-commented on both sides.

    Parity **near**, inherited from ch06's ``gvf_distance`` (the ``polybool`` start-vertex residual).  ch06
    measured 16 px of 31 730 on *its* image; CUMULATIVE pitfall 59 says re-measure on ch9's own data, and the
    number is recorded here so a regression cannot hide."""
    d = loadmat(REF / "ch09_demo.mat")
    I = read_image(BOOK_IMG)
    gray = rgb2gray_matlab(I)
    assert np.array_equal(gray, d["Igray"])
    level, _ = graythresh(gray)
    assert level == pytest.approx(sc(d["glevel"]), abs=0.0)
    assert_parity(im2bw(gray, level), d["bw0"].astype(bool), "binary", name="im2bw(I, graythresh)")

    res = ch9.model_ice_demo(I, full=True)
    diffs = {}
    for nm in ("bw1", "bw2", "bw3", "bw4"):
        ml = d[nm].astype(bool)
        py = getattr(res, nm)
        diffs[nm] = int((ml ^ py).sum())
    assert diffs["bw1"] <= 35, f"GVF chain residual grew: {diffs}"          # measured 28 of 13 756 = 0.204 %
    assert diffs["bw4"] <= 35, f"{diffs}"
    assert max(diffs.values()) / gray.size < 0.005                          # within the `near` band
    assert len(res.S) == int(sc(d["nS"])) == 30
    assert len(res.model.s_model) == int(sc(d["nM"])) == 23


@needs("ch09_demo.mat")
def test_rect_and_model_ice_model_are_EXACT_on_matlabs_own_bw4():
    """The isolated L2 measurement for ``rect.m`` and ``model_ice_model.m``: both fed **MATLAB's own ``bw4``**,
    so the GVF residual of the previous test cannot leak in (ch07 lesson — a cited measurement must name its
    input).  This is the number the parity table quotes for those two files."""
    d = loadmat(REF / "ch09_demo.mat")
    bw4 = d["bw4"].astype(bool)
    S = ch9.rect(bw4)
    assert len(S) == int(sc(d["nS"])) == 30
    assert_parity(np.stack([s.Vertices for s in S], axis=2), d["SV"], "float", atol=1e-12, name="S.Vertices")
    assert_parity(np.array([s.Center for s in S]), d["SC"], "float", atol=1e-12, name="S.Center")
    assert_parity(np.array([s.Area for s in S]), d["SA"].ravel(), "float", atol=1e-11, name="S.Area")
    assert_parity(np.array([s.Perimeter for s in S]), d["SP"].ravel(), "float", atol=1e-12, name="S.Perimeter")

    m = ch9.model_ice_model(S, bw4, 0.4, 2.5)
    assert len(m.s_model) == int(sc(d["nM"])) == 23
    assert_parity(np.stack([s.Vertices for s in m.s_model], axis=2), d["MV"], "float", atol=1e-12, name="MV")
    assert_parity(np.array([s.Center for s in m.s_model]), d["MC"], "float", atol=1e-12, name="MC")
    assert_parity(np.array([s.Area for s in m.s_model]), d["MA"].ravel(), "float", atol=1e-11, name="MA")
    assert_parity(np.array([s.Perimeter for s in m.s_model]), d["MP"].ravel(), "float", atol=1e-12, name="MP")
    MI = d["MI"].astype(bool)
    py = np.zeros_like(MI)
    for i, f in enumerate(m.s_model):
        for j in f.Intersection:
            py[i, j - 1] = True
    assert np.array_equal(py, MI)
    assert int(MI.sum()) == 34
    # the `k` band really removes 7 of the 30 rectangles on real data (so the filter is exercised)
    assert len(S) - len(m.s_model) == 7
    assert m.ratios.min() < 0.4 and m.ratios.max() > 2.5


# ===============================================================================================================
# Ours, not the book's — gaps G1/G2 must never carry a parity claim
# ===============================================================================================================

def test_segment_video_and_tiled_segmentation_are_labelled_as_ours():
    """Gaps **G2** (§9.3.3 per-frame segmentation) and the §9.3.2 20-sub-image tiling have **no shipped ``.m``**;
    both carry an explicit ``# DEVIATION`` marker in the source and neither appears in any parity assertion."""
    import inspect
    for fn in (ch9.segment_video, ch9.tiled_segmentation):
        src = inspect.getsource(fn)
        assert "DEVIATION" in src
    assert "OURS, not the book's" in ch9.segment_video.__doc__
    with pytest.raises(ValueError):
        ch9.tiled_segmentation(np.zeros((20, 20, 3), np.uint8), grid=(2, 3), gvf_iters=ch9.TABLE_9_4)


def test_segment_video_runs_on_a_tiny_stack():
    frames = np.stack([segmented_floe_video(1, shape=(40, 60))[0] for _ in range(2)])
    out = ch9.segment_video(frames, params=dict(Num=5, iter=10, timer=1), max_seeds=2)
    assert out.shape == (2, 40, 60) and out.dtype == np.bool_


@needs("r13.mat")
def test_R13_matlab_raises_in_two_different_ways_and_never_deletes_past_the_end():
    """Risk **R13**, settled precisely.  ``movie_floe.m`` line 25 is ``floe(k) = max(ice_areas)`` with
    ``numel(floe) == k-1``; MATLAB R2025a raises, it does **not** delete-and-shift.  The *identifier* depends on
    the form of the right-hand side, which is why the port's docstring and the earlier generic probe disagree and
    both are right:

    * ``x(k) = []`` with a **literal** ``[]`` is parsed as a deletion -> ``MATLAB:subsdeldimmismatch``
      ("Matrix index is out of range for deletion.");
    * ``x(k) = max(ice_areas)`` — a *function call* returning ``0x0`` — is an **assignment** ->
      ``MATLAB:matrix:singleSubscriptNumelMismatch`` ("Unable to perform assignment because the left and right
      sides have a different number of elements"), which is exactly what the shipped line does and exactly what
      ``seaice/ch09_model_ice.py`` records.
    * ``v = [1 2 3]; v(2) = []`` **does** delete-and-shift — but ``movie_floe.m``'s loop never reaches that case.
    """
    d = loadmat(REF / "r13.mat")
    def s_(k):
        v = d[k]
        return str(v[0]) if v.size else ""
    assert s_("id1") == "MATLAB:subsdeldimmismatch"
    assert s_("id2") == s_("id3") == "MATLAB:matrix:singleSubscriptNumelMismatch"
    assert "different number of elements" in s_("m3")
    assert s_("id4") == "" and s_("m4") == "[1 3]"
    assert s_("maxempty") == "[0 0]"                      # max([]) really is 0x0, so the RHS is empty
