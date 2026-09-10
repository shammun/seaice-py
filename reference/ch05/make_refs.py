"""Generate the MATLAB (R2025a) reference outputs for chapter 5 (Book §5.1–§5.3, Watershed-Based Ice Floe Segmentation).

Runs the ORIGINAL ``.m`` files of ``MATLAB_ROOT/ch5`` (+ ``watershed_based/``) through ``tools.run_matlab_ref.run_ref``
(``matlab -batch``, figures invisible) and saves the workspace variables listed in ``analysis/ch05.md`` §6 as
``reference/ch05/*.mat`` (v7).  Every launch also ``imwrite``s / ``print``s the images MATLAB would *display* into
``outputs/ch05/verify/matlab/`` for the side-by-side figure comparisons (L3).

Usage:  .venv/Scripts/python.exe reference/ch05/make_refs.py [name ...]
        (no arguments = all references; names = direct, distance_propagation, distance_watershed, dw_cityblock,
         dw_euclidean, dw_quasi, gradients, marker, marker_centroid, topological, chaincode, main, main_synth,
         main_crop, fig, compat)

Notes
-----
* Every script hard-codes ``imread('q.jpg')`` and reads from the *current folder*.  Each is copied verbatim into a
  scratch folder (``outputs/ch05/verify/scratch``) under a non-shadowing ``*_ref.m`` name together with ``q.jpg`` and
  the four helper functions (``boundaries.m``, ``fchcode.m``, ``bound2im.m``, ``freeman_concave.m`` — none shadows a
  toolbox function) and only literals are patched (metric name for the ``distance_watershed`` variants, the
  ``imread`` file name for the synthetic / crop variants, the commented centroid block of ``marker_watershed.m``,
  accumulator statements appended *after* the verbatim statements of ``main.m``'s loop).  Nothing in MATLAB_ROOT is
  touched and the ch5 folder is never put on the MATLAB path (cwd = scratch resolves the helpers).
* ``compat`` calls the toolbox functions (``watershed``, ``imregionalmin``, ``imregionalmax``, ``imimposemin``,
  ``imreconstruct``, ``imfilter``, ``regionprops``, ``bitand``, ``intersect``) on controlled arrays written from
  Python (``reference/ch05/inputs.mat``): the plateau / tie fixtures of ``seaice.core.synth.plateau_fixtures`` plus
  order-discriminating ones (2-px corridors, asymmetric adjacency, equal-priority plateaus, random uint8 fields with
  ties, > 255 minima → uint16 output, int16 / int32 / uint16 / single / double / logical inputs, ±Inf, border minima,
  1×N / N×1 / 1×1 / constant images) so the neighbour-scan and initial-push order of the flooding is pinned.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.run_matlab_ref import run_ref  # noqa: E402
from seaice.core import synth  # noqa: E402

MATLAB_ROOT = ROOT / "K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab"
CH = MATLAB_ROOT / "ch5"
WB = CH / "watershed_based"
REF = ROOT / "reference/ch05"
VERIFY = ROOT / "outputs/ch05/verify"
ML_IMG = VERIFY / "matlab"
SCRATCH = VERIFY / "scratch"
DATA = ROOT / "data/book/ch05"
IMAGE = "q.jpg"
SYNTH_PNG = "synth_two_floes.png"
CROP_PNG = "ch04_test_fig4_3a.png"
HELPERS = ("boundaries.m", "fchcode.m", "bound2im.m")  # top-level copies (byte-identical to watershed_based/)

IMREAD_TOK = f"imread('{IMAGE}')"


def q(p: Path | str) -> str:
    """MATLAB single-quoted literal with forward slashes (MATLAB on Windows accepts them)."""
    return "'" + str(p).replace("\\", "/").replace("'", "''") + "'"


def _read_m(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _patch(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        if old not in text:
            raise RuntimeError(f"patch token {old!r} not found in script")
        text = text.replace(old, new, 1)
    return text


def _imw(var: str, name: str, autoscale: bool = False) -> str:
    p = q(ML_IMG / name)
    if autoscale:
        return f"tmp__ = mat2gray(double({var})); imwrite(tmp__, {p});"
    return f"imwrite({var}, {p});"


def _imw_finite(var: str, name: str) -> str:
    """``imshow(I, [])`` of an image holding ``-Inf`` markers: display convention = finite range, -Inf → min."""
    p = q(ML_IMG / name)
    return (f"tmp__ = double({var}); fin__ = isfinite(tmp__); tmp__(~fin__) = min(tmp__(fin__)); "
            f"imwrite(mat2gray(tmp__), {p});")


def _print_figs(prefix: str) -> str:
    """Print every open (invisible) figure to ``<prefix>_fig<k>.png`` in creation order, then close them."""
    p = q(ML_IMG)
    return (f"hs__ = findobj(0, 'Type', 'figure'); if ~isempty(hs__), [~, ix__] = sort([hs__.Number]); hs__ = hs__(ix__); "
            f"for k__ = 1:numel(hs__), print(hs__(k__), '-dpng', '-r100', fullfile({p}, sprintf('{prefix}_fig%d.png', k__))); end; end; "
            f"close all;")


# ------------------------------------------------------------------------------------------------------------
# Controlled fixtures (identical arrays on both sides through inputs.mat)
# ------------------------------------------------------------------------------------------------------------
def _watershed_fixtures() -> dict[str, np.ndarray]:
    fx: dict[str, np.ndarray] = {f"pf_{k}": v for k, v in synth.plateau_fixtures().items()}
    rng = np.random.default_rng(55)
    # asymmetric adjacency: plateau (2) touched by basin A through a column neighbour (left) and by basin B through
    # a row neighbour (above); A is earlier in column-major order.  Transposed variant swaps the roles.
    t = np.full((8, 9), 5, dtype=np.int16)
    t[2:6, 3:7] = 2
    t[4, 2] = 1  # A: left of (4,3)
    t[1, 5] = 1  # B: above (2,5)
    fx["ws_asym_adjacency"] = t
    fx["ws_asym_adjacency_T"] = np.ascontiguousarray(t.T)
    # equal-priority plateau reached simultaneously from both sides (odd / even widths, minima at the same level)
    t = np.full((7, 13), 4, dtype=np.int16)
    t[3, 0] = 1
    t[3, 12] = 1
    t[1:6, 1:12] = 2
    fx["ws_equal_plateau_odd"] = t
    t = np.full((6, 12), 4, dtype=np.int16)
    t[2, 0] = 1
    t[3, 11] = 1
    t[1:5, 1:11] = 2
    fx["ws_equal_plateau_even"] = t
    # two basins joined by two 2-px corridors of different lengths (which corridor is dammed first?)
    t = np.full((12, 14), 7, dtype=np.int16)
    t[1:11, 1:4] = 1
    t[1:11, 10:13] = 1
    t[2:4, 4:10] = 3
    t[7:9, 4:10] = 3
    t[8, 6] = 2
    fx["ws_two_corridors"] = t
    # 2-px corridor with one side pushed first (minimum at different levels)
    t = np.full((9, 12), 6, dtype=np.int16)
    t[1:8, 1:4] = 2
    t[1:8, 8:11] = 1
    t[3:5, 4:8] = 3
    fx["ws_corridor_unequal"] = t
    # random uint8 fields: ties everywhere / full range (> 255 basins → uint16 output) / ternary 200×300
    fx["ws_rand_u8_30x40_ties"] = rng.integers(0, 4, size=(30, 40)).astype(np.uint8)
    fx["ws_rand_u8_200x300"] = rng.integers(0, 256, size=(200, 300)).astype(np.uint8)
    fx["ws_rand_u8_200x300_ties"] = rng.integers(0, 3, size=(200, 300)).astype(np.uint8)
    fx["ws_rand_dbl_50x60"] = rng.random((50, 60))
    fx["ws_rand_single_25x35"] = rng.random((25, 35)).astype(np.float32)
    # > 255 isolated minima on a grid (900) → uint16 class in MATLAB
    t = np.ones((60, 60), dtype=np.uint8)
    t[1::2, 1::2] = 0
    fx["ws_grid_900_minima"] = t
    # signed / wide integer classes
    fx["ws_int16_neg"] = rng.integers(-300, 300, size=(30, 30)).astype(np.int16)
    fx["ws_int32"] = rng.integers(-5, 5, size=(20, 24)).astype(np.int32)
    fx["ws_uint16"] = rng.integers(0, 6, size=(20, 24)).astype(np.uint16)
    # double with -Inf minima and +Inf walls
    t = rng.integers(1, 4, size=(15, 20)).astype(np.float64)
    t[3, 4] = -np.inf
    t[10, 15] = -np.inf
    t[:, 9] = np.inf
    t[7, 9] = 2.0  # one gap in the +Inf wall
    fx["ws_dbl_inf"] = t
    # minima on the border / corners plus a ramp
    t = np.add.outer(np.arange(10), np.arange(12)).astype(np.float64)
    t[0, 11] = -1.0
    t[9, 0] = -1.0
    t[5, 0] = -2.0
    t[0, 6] = 0.0
    fx["ws_border_minima"] = t
    fx["ws_ramp"] = np.tile(np.arange(10, dtype=np.float64), (8, 1))
    fx["ws_1x1"] = np.array([[3]], dtype=np.uint8)
    fx["ws_2x2_ties"] = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    fx["ws_2x2"] = np.array([[1, 2], [2, 1]], dtype=np.uint8)
    fx["ws_const"] = np.full((5, 7), 3, dtype=np.uint8)
    fx["ws_logical"] = (rng.random((12, 15)) > 0.6).astype(np.uint8)  # cast to logical in MATLAB
    fx["ws_all_zero"] = np.zeros((6, 6), dtype=np.uint8)
    # a checkerboard: every pixel is a regional minimum or a ridge candidate
    fx["ws_checker"] = (np.indices((9, 11)).sum(axis=0) % 2).astype(np.uint8)
    return fx


def _extrema_fixtures() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(56)
    fx: dict[str, np.ndarray] = {}
    t = np.full((8, 10), 5, dtype=np.uint8)
    t[0:3, 0:4] = 2  # plateau touching the top-left border
    t[6:8, 7:10] = 2  # plateau touching the bottom-right corner
    t[3, 5] = 1
    t[4:6, 2:4] = 9  # a plateau maximum
    fx["rm_plateau_border"] = t
    fx["rm_const_u8"] = np.full((4, 6), 7, dtype=np.uint8)
    fx["rm_const_dbl"] = np.full((4, 6), 0.3)
    t = rng.integers(1, 5, size=(9, 12)).astype(np.float64)
    t[2, 2] = -np.inf
    t[2, 3] = -np.inf
    t[6, 8] = np.inf
    t[0, 11] = np.inf
    fx["rm_inf"] = t
    fx["rm_all_inf"] = np.full((3, 4), np.inf)
    fx["rm_all_neg_inf"] = np.full((3, 4), -np.inf)
    fx["rm_i16"] = rng.integers(-20, 20, size=(15, 18)).astype(np.int16)
    fx["rm_single"] = rng.integers(0, 4, size=(14, 16)).astype(np.float32)
    fx["rm_u8_ties"] = rng.integers(0, 3, size=(25, 30)).astype(np.uint8)
    fx["rm_logical"] = (rng.random((10, 12)) > 0.5).astype(np.uint8)
    fx["rm_row"] = np.array([[3, 1, 1, 2, 5, 5, 4, 0, 0, 1]], dtype=np.uint8)
    fx["rm_col"] = fx["rm_row"].T.copy()
    fx["rm_1x1"] = np.array([[4]], dtype=np.uint8)
    return fx


def _imposemin_fixtures() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(57)
    fx: dict[str, np.ndarray] = {}
    shape = (16, 20)
    bw = np.zeros(shape, dtype=np.uint8)
    bw[3:6, 3:6] = 1
    bw[10:13, 12:17] = 1
    bw[0, 19] = 1  # marker on the border
    fx["im_bw"] = bw
    fx["im_u8"] = rng.integers(0, 256, size=shape).astype(np.uint8)
    fx["im_u8_sat"] = np.full(shape, 255, dtype=np.uint8)  # I + 1 saturates
    fx["im_i16"] = rng.integers(-500, 500, size=shape).astype(np.int16)
    fx["im_i16_max"] = np.full(shape, np.iinfo(np.int16).max, dtype=np.int16)
    fx["im_single"] = (rng.random(shape) * 40 - 20).astype(np.float32)
    fx["im_single_neg"] = (-rng.integers(0, 12, size=shape)).astype(np.float32)  # an inverse-distance-like map
    fx["im_double"] = rng.random(shape)
    fx["im_const_single"] = np.full(shape, -3.0, dtype=np.float32)  # h = 0.1 branch
    fx["im_const_u8"] = np.full(shape, 9, dtype=np.uint8)
    t = rng.random(shape) * 10
    t[5:8, 8:11] = np.inf
    t[14, 2] = np.inf
    fx["im_double_pinf"] = t
    t = rng.random(shape) * 10
    t[1, 1] = -np.inf
    t[12, 12] = -np.inf
    fx["im_double_ninf"] = t
    fx["im_bw_all"] = np.ones(shape, dtype=np.uint8)
    fx["im_bw_none"] = np.zeros(shape, dtype=np.uint8)
    adj = np.zeros(shape, dtype=np.uint8)
    adj[6:9, 5:8] = 1
    adj[6:9, 8:11] = 1  # two markers sharing a 4-neighbour edge (one 8-connected component)
    adj[1, 1] = 1
    adj[2, 2] = 1  # two markers touching diagonally only
    fx["im_bw_adjacent"] = adj
    t = rng.integers(2, 6, size=shape).astype(np.uint8)
    t[4:10, 4:12] = 0  # a level-0 plateau ...
    fx["im_u8_plateau"] = t
    m = np.zeros(shape, dtype=np.uint8)
    m[6, 6] = 1
    m[7, 10] = 1  # ... holding two distinct markers (the book's reason for the "+1")
    fx["im_bw_in_plateau"] = m
    return fx


def _misc_fixtures() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(58)
    fx: dict[str, np.ndarray] = {}
    fx["ep_patterns"] = synth.FIG_5_15_ENDPOINT_PATTERNS.astype(np.float64)  # (12, 3, 3)
    fx["ep_kernel"] = synth.FIG_5_15_KERNEL
    fx["F516"] = synth.FIG_5_16_IMAGE.astype(np.uint8)
    marker, mask = synth.two_blobs_with_marker()
    mk = marker.astype(np.float64) * 3.0
    mk[0, 0] = -np.inf
    msk = mask.astype(np.float64) * 3.0
    msk[10:14, 10:14] = np.inf
    msk[0, 0] = 0.0
    fx["rc_mk_inf"] = mk
    fx["rc_mask_inf"] = np.maximum(msk, mk)  # marker <= mask required by MATLAB
    fx["cen_bw"] = (rng.random((20, 30)) > 0.7).astype(np.uint8)
    fx["cen_bw"][5:12, 3:9] = 1
    fx["ix_A"] = np.array([[1, 2], [3, 4], [5, 6], [3, 4], [9, 9]], dtype=np.float64)
    fx["ix_B"] = np.array([[9, 9], [3, 4], [0, 0]], dtype=np.float64)
    # a few extra regional-min / watershed probes on a bool image of two touching discs (L1-style truth)
    rr, cc = np.mgrid[0:40, 0:60]
    two = ((rr - 20) ** 2 + (cc - 18) ** 2 <= 12 ** 2) | ((rr - 20) ** 2 + (cc - 40) ** 2 <= 12 ** 2)
    fx["two_discs"] = two.astype(np.uint8)
    return fx


def make_inputs() -> dict[str, np.ndarray]:
    return {**_watershed_fixtures(), **_extrema_fixtures(), **_imposemin_fixtures(), **_misc_fixtures()}


WS_NAMES = list(_watershed_fixtures())
RM_NAMES = list(_extrema_fixtures())
IM_IMAGES = ("im_u8", "im_u8_sat", "im_i16", "im_i16_max", "im_single", "im_single_neg", "im_double", "im_const_single",
             "im_const_u8", "im_double_pinf", "im_double_ninf")
IM_MARKERS = ("im_bw", "im_bw_all", "im_bw_none", "im_bw_adjacent")


# ------------------------------------------------------------------------------------------------------------
def _prepare() -> None:
    for d in (REF, ML_IMG, SCRATCH):
        d.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA / IMAGE, SCRATCH / IMAGE)
    for h in HELPERS:
        shutil.copy(CH / h, SCRATCH / h)
    shutil.copy(WB / "freeman_concave.m", SCRATCH / "freeman_concave.m")
    shutil.copy(WB / "nrm_junction_ending.fig", SCRATCH / "nrm_junction_ending.fig")
    # verbatim copies (imread token re-substituted by itself to assert its presence)
    for name in ("direct_watershed", "distance_propagation", "gradients_watershed", "topological_surface",
                 "chaincode_corner", "marker_watershed"):
        (SCRATCH / f"{name}_ref.m").write_text(_patch(_read_m(CH / f"{name}.m"), [(IMREAD_TOK, IMREAD_TOK)]),
                                               encoding="utf-8")
    dw = _read_m(CH / "distance_watershed.m")
    (SCRATCH / "distance_watershed_ref.m").write_text(_patch(dw, [(IMREAD_TOK, IMREAD_TOK)]), encoding="utf-8")
    for metric in ("cityblock", "euclidean", "quasi-euclidean"):
        (SCRATCH / f"distance_watershed_ref_{metric.split('-')[0]}.m").write_text(
            _patch(dw, [("'chessboard'", f"'{metric}'")]), encoding="utf-8")
    # marker_watershed with the commented centroid block un-commented AND used as the marker (the block's intent)
    mw = _read_m(CH / "marker_watershed.m")
    (SCRATCH / "marker_watershed_ref_centroid.m").write_text(_patch(mw, [
        ("% marker0 = zeros(size(marker));", "marker0 = zeros(size(marker));"),
        ("% [label, num] = bwlabel(marker, 8);", "[label, num] = bwlabel(marker, 8);"),
        ("% for i = 1 : num", "for i = 1 : num"),
        ("%     cen = regionprops(label == i, 'centroid');", "    cen = regionprops(label == i, 'centroid');"),
        ("%     cen = cat(1, cen.Centroid);", "    cen = cat(1, cen.Centroid); CENR(i, :) = cen;"),
        ("%     cen = floor(cen);", "    cen = floor(cen); CENF(i, :) = cen;"),
        ("%     marker0(cen(2), cen(1)) = 1;", "    marker0(cen(2), cen(1)) = 1;"),
        ("% end", "end"),
        ("% figure, imshow(marker0);", "figure, imshow(marker0);"),
        ("imimposemin(imgDist, marker)", "imimposemin(imgDist, marker0)"),
        ("dis = marker .* bw;", "dis = marker0 .* bw;"),
    ]), encoding="utf-8")
    # main.m with per-iteration accumulators appended after the verbatim statements
    mn = _read_m(WB / "main.m")
    acc_init = ("EP = {}; CONC = {}; REG = {}; CC = {}; NBR = {}; CONNECT = {}; G2 = {}; TT = []; REMOVED = []; SEGB = {};\n"
                "for i = 1:num")
    acc = ("c = intersect(ep,concave, 'rows'); EP{i} = ep; CONC{i} = concave; REG{i} = im; CC{i} = c; NBR{i} = neighbor; "
           "CONNECT{i} = connect; G2{i} = g2; TT(i) = T; REMOVED(i) = double(numel(c) == 0); SEGB{i} = seg;")
    main_patched = _patch(mn, [("for i = 1:num", acc_init), ("c = intersect(ep,concave, 'rows');", acc)])
    (SCRATCH / "main_ref.m").write_text(main_patched, encoding="utf-8")
    # synthetic substitute image (deterministic) and the Fig. 4.3(a) crop of ch04's test.jpg as lossless PNGs
    iio.imwrite(SCRATCH / SYNTH_PNG, synth.two_touching_floes())
    (SCRATCH / "main_ref_synth.m").write_text(_patch(main_patched, [(IMREAD_TOK, f"imread('{SYNTH_PNG}')")]),
                                              encoding="utf-8")
    test4 = ROOT / "data/book/ch04/test.jpg"
    if test4.exists():
        from seaice.ch04_ice_edge_detection import fig_4_3a
        iio.imwrite(SCRATCH / CROP_PNG, fig_4_3a(iio.imread(test4)))
        (SCRATCH / "main_ref_crop.m").write_text(_patch(main_patched, [(IMREAD_TOK, f"imread('{CROP_PNG}')")]),
                                                 encoding="utf-8")
    savemat(REF / "inputs.mat", make_inputs())


# ------------------------------------------------------------------------------------------------------------
def ref_direct():
    code = f"""
run({q(SCRATCH / 'direct_watershed_ref.m')});
im0 = rgb2gray(imread('{IMAGE}')); level = graythresh(im0); cls_L = class(imgLabel); n_basins = double(max(imgLabel(:)));
n_comp = double(max(colorimg(:))); rgb_color = label2rgb(colorimg, 'jet', 'k', 'shuffle');
{_imw('bgm', 'direct_bgm.png')}
{_imw('im', 'direct_overlay.png')}
{_imw('rgb_color', 'direct_color.png')}
{_print_figs('direct')}
"""
    return run_ref(code, ["im0", "level", "img", "imgLabel", "cls_L", "n_basins", "bgm", "im", "colorimg", "n_comp",
                          "rgb_color"], REF / "direct_watershed.mat", workdir=SCRATCH, timeout=900)


def ref_distance_propagation():
    code = f"""
run({q(SCRATCH / 'distance_propagation_ref.m')});
cls_dist = class(imgDist);
{_imw('imgDist', 'dprop_imgDist.png', autoscale=True)}
{_imw('dist', 'dprop_dist.png')}
{_print_figs('dprop')}
rgb__ = imread('{IMAGE}'); bwq = im2bw(rgb__, graythresh(rgb__));
imgq = ~bwq; Dlit_city = -bwdist(~imgq, 'cityblock');
D_city = -bwdist(~bwq, 'cityblock'); D_euc = -bwdist(~bwq, 'euclidean'); D_chess = -bwdist(~bwq, 'chessboard');
D_quasi = -bwdist(~bwq, 'quasi-euclidean');
d0__ = D_city + abs(min(min(D_city))); dist_city = uint8(d0__ .* 255 / max(max(d0__)));
d0__ = D_euc + abs(min(min(D_euc))); dist_euc = uint8(d0__ .* 255 / max(max(d0__)));
d0__ = D_chess + abs(min(min(D_chess))); dist_chess = uint8(d0__ .* 255 / max(max(d0__)));
figure, image(dist_city, 'CDataMapping','scaled'); hold all; imcontour(D_city); title('cityblock');
figure, image(dist_euc, 'CDataMapping','scaled'); hold all; imcontour(D_euc); title('euclidean');
figure, image(dist_chess, 'CDataMapping','scaled'); hold all; imcontour(D_chess); title('chessboard');
{_print_figs('dprop_q')}
"""
    return run_ref(code, ["img", "imgDist", "cls_dist", "dist0", "dist", "bwq", "Dlit_city", "D_city", "D_euc", "D_chess",
                          "D_quasi", "dist_city", "dist_euc", "dist_chess"],
                   REF / "distance_propagation.mat", workdir=SCRATCH, timeout=900)


def _dw_code(script: str, tag: str) -> str:
    return f"""
run({q(SCRATCH / script)});
rgb__ = imread('{IMAGE}'); level = graythresh(rgb__); img0 = im2bw(rgb__, level);
[~, n_min] = bwlabel(Dis_img); n_min_px = double(nnz(Dis_img)); cls_L = class(imgLabel); cls_dist = class(imgDist);
n_basins = double(max(imgLabel(:))); n_floes = double(max(colorimg(:))); cls_dis = class(dis);
seg_pre = img0; seg_pre(bgm) = 0; n_floes_pre = double(max(max(bwlabel(seg_pre))));
rgb_color = label2rgb(colorimg, 'jet', 'k', 'shuffle');
{_imw('img0', tag + '_a_binary.png')}
{_imw('imgDist', tag + '_b_imgDist.png', autoscale=True)}
{_imw('~Dis_img', tag + '_minima_mask.png')}
{_imw('bw0', tag + '_d_minima_overlay.png')}
{_imw('bgm', tag + '_e_ridge.png')}
{_imw('img', tag + '_f_segmented.png')}
{_imw('rgb_color', tag + '_color.png')}
{_print_figs(tag)}
"""


_DW_VARS = ["img0", "level", "imgDist", "cls_dist", "Dis_img", "bw", "dis", "cls_dis", "p", "q", "bw0", "imgLabel", "cls_L",
            "bgm", "img", "colorimg", "n_min", "n_min_px", "n_basins", "n_floes", "seg_pre", "n_floes_pre", "rgb_color"]


def ref_distance_watershed():
    return run_ref(_dw_code("distance_watershed_ref.m", "dw_chess"), _DW_VARS, REF / "distance_watershed.mat",
                   workdir=SCRATCH, timeout=900)


def ref_dw_cityblock():
    return run_ref(_dw_code("distance_watershed_ref_cityblock.m", "dw_city"), _DW_VARS,
                   REF / "distance_watershed_cityblock.mat", workdir=SCRATCH, timeout=900)


def ref_dw_euclidean():
    return run_ref(_dw_code("distance_watershed_ref_euclidean.m", "dw_euc"), _DW_VARS,
                   REF / "distance_watershed_euclidean.mat", workdir=SCRATCH, timeout=900)


def ref_dw_quasi():
    return run_ref(_dw_code("distance_watershed_ref_quasi.m", "dw_quasi"), _DW_VARS,
                   REF / "distance_watershed_quasi.mat", workdir=SCRATCH, timeout=900)


def ref_gradients():
    code = f"""
run({q(SCRATCH / 'gradients_watershed_ref.m')});
level = graythresh(I); n1 = double(max(l(:))); n2 = double(max(l2(:))); cls_l = class(l); cls_l2 = class(l2);
{_imw('g', 'grad_a_g.png', autoscale=True)}
{_imw('l', 'grad_c_l.png', autoscale=True)}
{_imw('wr', 'grad_wr.png')}
{_imw('f', 'grad_d_f.png')}
{_imw('g2', 'grad_6a_g2.png', autoscale=True)}
{_imw('l2', 'grad_6b_l2.png', autoscale=True)}
{_imw('wr2', 'grad_wr2.png')}
{_imw('f2', 'grad_6c_f2.png')}
{_print_figs('grad')}
"""
    return run_ref(code, ["I", "level", "bw", "hy", "hx", "Ix", "Iy", "g", "l", "cls_l", "wr", "f", "g2", "l2", "cls_l2",
                          "wr2", "f2", "n1", "n2"], REF / "gradients_watershed.mat", workdir=SCRATCH, timeout=900)


def _marker_code(script: str, tag: str, extra: str = "") -> str:
    return f"""
run({q(SCRATCH / script)});
rgb__ = imread('{IMAGE}'); level = graythresh(rgb__); img0 = im2bw(rgb__, level);
imgDist0 = -bwdist(~img0, 'cityblock'); [~, n_min] = bwlabel(Dis_img); n_min_px = double(nnz(Dis_img));
[~, n_marker] = bwlabel(marker); rm_imp = imregionalmin(imgDist); cls_imp = class(imgDist); cls_L = class(imgLabel);
n_basins = double(max(imgLabel(:))); n_floes = double(max(colorimg(:))); se_nhood = getnhood(se);
rgb_color = label2rgb(colorimg, 'jet', 'k', 'shuffle');
{extra}
{_imw('imgDist0', tag + '_imgDist0.png', autoscale=True)}
{_imw('~Dis_img', tag + '_a_minima.png')}
{_imw('marker', tag + '_b_marker.png')}
{_imw_finite('imgDist', tag + '_c_imposed.png')}
{_imw('bw0', tag + '_marker_overlay.png')}
{_imw('bgm', tag + '_d_ridge.png')}
{_imw('img', tag + '_e_segmented.png')}
{_imw('rgb_color', tag + '_color.png')}
{_print_figs(tag)}
"""


_MK_VARS = ["img0", "level", "imgDist0", "Dis_img", "n_min", "n_min_px", "se_nhood", "marker", "n_marker", "imgDist",
            "cls_imp", "rm_imp", "bw", "dis", "bw0", "imgLabel", "cls_L", "bgm", "img", "colorimg", "n_basins", "n_floes",
            "rgb_color"]


def ref_marker():
    return run_ref(_marker_code("marker_watershed_ref.m", "marker"), _MK_VARS, REF / "marker_watershed.mat",
                   workdir=SCRATCH, timeout=900)


def ref_marker_centroid():
    extra = (f"rp__ = regionprops(bwlabel(marker, 8), 'Centroid'); CENR2 = cat(1, rp__.Centroid); "
             f"[~, n_marker0] = bwlabel(marker0); {_imw('marker0', 'marker_cen_marker0.png')}")
    return run_ref(_marker_code("marker_watershed_ref_centroid.m", "marker_cen", extra),
                   _MK_VARS + ["marker0", "CENR", "CENF", "CENR2", "n_marker0"],
                   REF / "marker_watershed_centroid.mat", workdir=SCRATCH, timeout=900)


def ref_topological():
    code = f"""
run({q(SCRATCH / 'topological_surface_ref.m')});
level = graythresh(I); cls_x = class(x); cls_d = class(d);
{_imw('I', 'topo_gray.png')}
{_imw('x', 'topo_complement.png')}
{_imw('img', 'topo_bw.png')}
{_print_figs('topo')}
"""
    return run_ref(code, ["I", "level", "x", "cls_x", "g", "img", "d", "cls_d"], REF / "topological_surface.mat",
                   workdir=SCRATCH, timeout=900)


def ref_chaincode():
    code = f"""
run({q(SCRATCH / 'chaincode_corner_ref.m')});
fcc = c.fcc; x0y0 = c.x0y0; mm = c.mm; diff_c = c.diff; l = double(length(c.fcc)); n_boundaries = double(numel(boundaries(B, 8, 'cww')));
Sum0_final = Sum0; A_last = A(l); level = graythresh(I);
{_imw('B', 'cc_B.png')}
{_imw('bim', 'cc_bim.png')}
{_print_figs('cc')}
load({q(REF / 'inputs.mat')});
b516 = boundaries(logical(F516), 8, 'cww'); n516 = double(numel(b516)); b516 = b516{{1}}; c516 = fchcode(b516); f516 = c516.fcc; x0y0_516 = c516.x0y0;
b516_ccw = boundaries(logical(F516), 8, 'ccw'); b516_ccw = b516_ccw{{1}};
"""
    return run_ref(code, ["I", "level", "B", "b", "d", "max_d", "k", "bim", "fcc", "x0y0", "mm", "diff_c", "l", "n_boundaries",
                          "R", "A", "A_last", "Sum", "Sum0_final", "D", "D0", "Diff", "p", "concave", "b516", "n516",
                          "f516", "x0y0_516", "b516_ccw"],
                   REF / "chaincode_corner.mat", workdir=SCRATCH, timeout=900)


_MAIN_VARS = ["img", "bw", "m", "n", "D", "L", "cls_L", "w", "f", "cls_f", "seg0", "seg", "label", "num", "wr", "EP", "CONC",
              "REG", "CC", "NBR", "CONNECT", "G2", "TT", "REMOVED", "SEGB", "n_floes0", "n_floes", "n_lines_px",
              "n_removed"]


def _main_code(script: str, tag: str, image: str) -> str:
    return f"""
run({q(SCRATCH / script)});
cls_L = class(L); cls_f = class(f); n_floes0 = double(max(max(bwlabel(seg0)))); n_floes = double(max(max(bwlabel(seg))));
n_lines_px = double(nnz(f)); n_removed = double(sum(REMOVED));
img = imread('{image}');
{_imw('bw', tag + '_b_binary.png')}
{_imw('-double(D)', tag + '_c_negD.png', autoscale=True)}
{_imw('w', tag + '_d_ridges.png')}
{_imw('seg0', tag + '_e_seg0.png')}
{_imw('f', tag + '_f_lines.png')}
{_imw('seg', tag + '_i_seg.png')}
{_print_figs(tag)}
"""


def ref_main():
    return run_ref(_main_code("main_ref.m", "main", IMAGE), _MAIN_VARS, REF / "main.mat", workdir=SCRATCH, timeout=900)


def ref_main_synth():
    return run_ref(_main_code("main_ref_synth.m", "main_synth", SYNTH_PNG), _MAIN_VARS, REF / "main_synth.mat",
                   workdir=SCRATCH, timeout=900)


def ref_main_crop():
    if not (SCRATCH / "main_ref_crop.m").exists():
        raise RuntimeError("data/book/ch04/test.jpg absent — crop variant skipped")
    return run_ref(_main_code("main_ref_crop.m", "main_crop", CROP_PNG), _MAIN_VARS, REF / "main_crop.mat",
                   workdir=SCRATCH, timeout=1800)


def ref_fig():
    """The authors' ``nrm_junction_ending.fig`` (Fig. 5.14(f)): image CData + the three end-point line series."""
    code = f"""
h = openfig('nrm_junction_ending.fig', 'new', 'invisible');
im__ = findobj(h, 'Type', 'image'); FIG_CDATA = get(im__(1), 'CData'); cls_cdata = class(FIG_CDATA);
ln__ = findobj(h, 'Type', 'line'); nl = double(numel(ln__));
FIG_X = zeros(nl, 2); FIG_Y = zeros(nl, 2);
for k = 1:nl, xd = get(ln__(k), 'XData'); yd = get(ln__(k), 'YData'); FIG_X(k, 1:numel(xd)) = xd; FIG_Y(k, 1:numel(yd)) = yd; end
tx__ = findobj(h, 'Type', 'text'); nt = double(numel(tx__)); FIG_TEXT = cell(nt, 1); FIG_TPOS = zeros(nt, 3);
for k = 1:nt, s = get(tx__(k), 'String'); if iscell(s), s = strjoin(s, ' '); end; FIG_TEXT{{k}} = s; FIG_TPOS(k, :) = get(tx__(k), 'Position'); end
print(h, '-dpng', '-r100', {q(ML_IMG / 'fig_5_14f_authors.png')});
{_imw('FIG_CDATA', 'fig_5_14f_cdata.png')}
close(h);
"""
    return run_ref(code, ["FIG_CDATA", "cls_cdata", "nl", "FIG_X", "FIG_Y", "nt", "FIG_TEXT", "FIG_TPOS"],
                   REF / "fig_5_14f.mat", workdir=SCRATCH, timeout=600)


def _compat_code() -> tuple[str, list[str]]:
    lines: list[str] = [f"load({q(REF / 'inputs.mat')});", "ws_logical = logical(ws_logical); rm_logical = logical(rm_logical);"]
    v: list[str] = []

    def add(line: str, *names: str) -> None:
        lines.append(line)
        v.extend(names)

    # --- watershed on every fixture, 8- and 4-connectivity, output class recorded ------------------------------------
    for nm in WS_NAMES:
        # MATLAB's watershed accepts only uint8/uint16/single/double/logical: the raw call is probed (rej_<name> keeps
        # the error text) and int16/int32 fixtures are run as double (same values → same flooding) for the reference.
        add(f"try, r__ = watershed({nm}); rej_{nm} = ''; catch e__, rej_{nm} = e__.message; end; cls_in_{nm} = class({nm}); "
            f"X__ = {nm}; if isa(X__, 'int16') || isa(X__, 'int32'), X__ = double(X__); end; "
            f"try, ws8_{nm} = watershed(X__, 8); ws4_{nm} = watershed(X__, 4); cls_ws8_{nm} = class(ws8_{nm}); "
            f"cls_ws4_{nm} = class(ws4_{nm}); catch e__, ws8_{nm} = []; ws4_{nm} = []; cls_ws8_{nm} = ['ERR: ' e__.message]; "
            f"cls_ws4_{nm} = cls_ws8_{nm}; end", f"rej_{nm}", f"cls_in_{nm}", f"ws8_{nm}", f"ws4_{nm}", f"cls_ws8_{nm}", f"cls_ws4_{nm}")
        add(f"try, rm8_{nm} = imregionalmin({nm}, 8); rm4_{nm} = imregionalmin({nm}, 4); catch, rm8_{nm} = []; rm4_{nm} = []; end",
            f"rm8_{nm}", f"rm4_{nm}")
    add("wsd_pf_even_plateau = watershed(double(pf_even_plateau)); wsc_pf_even_plateau = watershed(double(pf_even_plateau), [0 1 0; 1 1 1; 0 1 0]); "
        "wsc8_ws_rand_u8_30x40_ties = watershed(ws_rand_u8_30x40_ties, ones(3));",
        "wsd_pf_even_plateau", "wsc_pf_even_plateau", "wsc8_ws_rand_u8_30x40_ties")
    add("try, ws_nan__ = watershed([1 NaN; 2 3]); ws_nan_ok = 1; ws_nan_msg = ''; catch e__, ws_nan_ok = 0; ws_nan_msg = e__.message; end",
        "ws_nan_ok", "ws_nan_msg")
    add("try, ws_3d__ = watershed(rand(3, 3, 3)); ws_3d_ok = 1; catch e__, ws_3d_ok = 0; end", "ws_3d_ok")
    add("try, ws_c6__ = watershed(double(pf_even_plateau), 6); ws_c6_ok = 1; ws_c6_msg = ''; catch e__, ws_c6_ok = 0; ws_c6_msg = e__.message; end",
        "ws_c6_ok", "ws_c6_msg")
    add("try, ws_empty = watershed(zeros(0, 5)); cls_ws_empty = class(ws_empty); catch e__, ws_empty = []; cls_ws_empty = ['ERR: ' e__.message]; end",
        "ws_empty", "cls_ws_empty")
    # --- imregionalmin / imregionalmax on the extrema fixtures ---------------------------------------------------------
    for nm in RM_NAMES:
        add(f"rmn8_{nm} = imregionalmin({nm}, 8); rmn4_{nm} = imregionalmin({nm}, 4); rmx8_{nm} = imregionalmax({nm}, 8); "
            f"rmx4_{nm} = imregionalmax({nm}, 4); cls_rmn_{nm} = class(rmn8_{nm});",
            f"rmn8_{nm}", f"rmn4_{nm}", f"rmx8_{nm}", f"rmx4_{nm}", f"cls_rmn_{nm}")
    add("rmn_conn_cross = imregionalmin(rm_u8_ties, [0 1 0; 1 1 1; 0 1 0]); rmn_conn_ones = imregionalmin(rm_u8_ties, ones(3)); "
        "rmn_default = imregionalmin(rm_u8_ties);", "rmn_conn_cross", "rmn_conn_ones", "rmn_default")
    add("try, rm_nan__ = imregionalmin([1 NaN; 2 3]); rm_nan_ok = 1; rm_nan_msg = ''; catch e__, rm_nan_ok = 0; rm_nan_msg = e__.message; end",
        "rm_nan_ok", "rm_nan_msg")
    # --- imimposemin ------------------------------------------------------------------------------------------------
    for img in IM_IMAGES:
        for mk in IM_MARKERS:
            add(f"ii_{img}_{mk} = imimposemin({img}, logical({mk})); ii4_{img}_{mk} = imimposemin({img}, logical({mk}), 4); "
                f"cls_ii_{img}_{mk} = class(ii_{img}_{mk});", f"ii_{img}_{mk}", f"ii4_{img}_{mk}", f"cls_ii_{img}_{mk}")
        add(f"h_{img} = 0; if isfloat({img}), h_{img} = 0.001 * (double(max({img}(:))) - double(min({img}(:)))); if h_{img} == 0, h_{img} = 0.1; end; else, h_{img} = 1; end",
            f"h_{img}")
    add("ii_plateau = imimposemin(im_u8_plateau, logical(im_bw_in_plateau)); ii_plateau_rm = imregionalmin(ii_plateau); "
        "ii_plateau_n = double(max(max(bwlabel(ii_plateau_rm)))); ii_plateau_ws = watershed(ii_plateau);",
        "ii_plateau", "ii_plateau_rm", "ii_plateau_n", "ii_plateau_ws")
    add("ii_dbl_marker = imimposemin(im_double, double(im_bw)); ii_u8_marker = imimposemin(im_double, im_bw);", "ii_dbl_marker", "ii_u8_marker")
    add("try, ii_log__ = imimposemin(logical(im_bw), logical(im_bw)); ii_log_ok = 1; ii_log_msg = ''; catch e__, ii_log_ok = 0; ii_log_msg = e__.message; end",
        "ii_log_ok", "ii_log_msg")
    add("try, ii_nan__ = imimposemin([1 NaN; 2 3], logical([1 0; 0 0])); ii_nan_ok = 1; ii_nan_msg = ''; catch e__, ii_nan_ok = 0; ii_nan_msg = e__.message; end",
        "ii_nan_ok", "ii_nan_msg")
    # regional minima of the imposed maps equal the markers (up to connectivity) — MATLAB's own statement
    add("ii_rm_u8 = imregionalmin(ii_im_u8_im_bw); ii_rm_single = imregionalmin(ii_im_single_im_bw); ii_rm_double = imregionalmin(ii_im_double_im_bw);",
        "ii_rm_u8", "ii_rm_single", "ii_rm_double")
    # --- imreconstruct with ±Inf ------------------------------------------------------------------------------------
    add("rc_inf = imreconstruct(rc_mk_inf, rc_mask_inf); rc_inf4 = imreconstruct(rc_mk_inf, rc_mask_inf, 4);", "rc_inf", "rc_inf4")
    # --- ending-point kernel on the 12 Fig. 5.15 patterns and an isolated pixel --------------------------------------
    add("EPR = zeros(12, 5, 5); EPC = zeros(12, 1); for k = 1:12, P = zeros(5); P(2:4, 2:4) = squeeze(ep_patterns(k, :, :)); "
        "r = imfilter(P, ep_kernel); EPR(k, :, :) = r; EPC(k) = r(3, 3); end", "EPR", "EPC")
    add("iso__ = zeros(5); iso__(3, 3) = 1; ep_iso = imfilter(iso__, ep_kernel); ep_iso_c = ep_iso(3, 3); "
        "loop__ = zeros(6); loop__(2:5, 2) = 1; loop__(2:5, 5) = 1; loop__(2, 2:5) = 1; loop__(5, 2:5) = 1; ep_loop = abs(imfilter(loop__, ep_kernel)); "
        "ep_loop_max = max(ep_loop(:)); ep_loop_hit = ep_loop >= ep_loop_max;", "ep_iso", "ep_iso_c", "ep_loop", "ep_loop_max", "ep_loop_hit")
    # --- misc: regionprops Centroid, bitand class, intersect rows --------------------------------------------------
    add("cen_lab = bwlabel(logical(cen_bw)); rp__ = regionprops(cen_lab, 'Centroid'); cen_xy = cat(1, rp__.Centroid); cen_n = double(max(cen_lab(:)));",
        "cen_lab", "cen_xy", "cen_n")
    add("cls_bitand = class(bitand(true, false)); bitand_tf = bitand(logical([1 1 0 0]), logical([1 0 1 0]));", "cls_bitand", "bitand_tf")
    add("ix_rows = intersect(ix_A, ix_B, 'rows'); ix_empty = intersect(ix_A, [7 7], 'rows'); ix_empty_n = double(numel(ix_empty));",
        "ix_rows", "ix_empty", "ix_empty_n")
    add("mul_cls = class(logical([1 0]) .* logical([1 1]));", "mul_cls")
    # --- two touching discs (bool → distance → watershed) ------------------------------------------------------------
    add("td = logical(two_discs); td_D = -bwdist(~td, 'cityblock'); td_L = watershed(td_D); td_rm = imregionalmin(td_D); "
        "td_seg = td & ~(td_L == 0); td_n = double(max(max(bwlabel(td_seg)))); td_De = -bwdist(~td); td_Le = watershed(td_De);",
        "td_D", "td_L", "td_rm", "td_seg", "td_n", "td_De", "td_Le")
    return "\n".join(lines) + "\n", v


def ref_compat():
    code, vars_ = _compat_code()
    (VERIFY / "compat_code.m").write_text(code, encoding="utf-8")
    return run_ref(code, vars_, REF / "compat.mat", timeout=1800)


ALL = {
    "direct": ref_direct,
    "distance_propagation": ref_distance_propagation,
    "distance_watershed": ref_distance_watershed,
    "dw_cityblock": ref_dw_cityblock,
    "dw_euclidean": ref_dw_euclidean,
    "dw_quasi": ref_dw_quasi,
    "gradients": ref_gradients,
    "marker": ref_marker,
    "marker_centroid": ref_marker_centroid,
    "topological": ref_topological,
    "chaincode": ref_chaincode,
    "main": ref_main,
    "main_synth": ref_main_synth,
    "main_crop": ref_main_crop,
    "fig": ref_fig,
    "compat": ref_compat,
}


def main(argv: list[str]) -> int:
    _prepare()
    names = argv or list(ALL)
    log_path = REF / "refs_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    rc = 0
    for name in names:
        print(f"[make_refs] {name} ...", flush=True)
        try:
            res = ALL[name]()
            entry = {"engine": res.engine, "version": res.version, "out": res.out_mat.name, "status": "ok"}
            if "NOTE:" in (res.stdout or ""):
                entry["note"] = res.stdout[res.stdout.index("NOTE:"):][:500]
            log[name] = entry
            print(f"  ok: {res.engine} {res.version} -> {res.out_mat.name}", flush=True)
        except Exception as exc:  # keep going; the report lists what failed
            msg = str(exc)
            log[name] = {"status": "error", "error": msg[:4000]}
            print(f"  ERROR: {msg[:3000]}", flush=True)
            rc = 1
        log_path.write_text(json.dumps(log, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
