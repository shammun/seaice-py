"""Generate the MATLAB (R2025a) reference outputs for chapter 4 (Book §4.1–§4.3, Ice Edge Detection).

Runs the ORIGINAL ``.m`` files of ``MATLAB_ROOT/ch4`` (``derivative.m``, ``morphology.m``) through
``tools.run_matlab_ref.run_ref`` (``matlab -batch``, figures invisible) and saves the workspace variables listed in
``analysis/ch04.md`` §6 as ``reference/ch04/*.mat`` (v7).  Every launch also writes the images MATLAB would *display*
(``imwrite``) into ``outputs/ch04/verify/matlab/`` for the side-by-side figure comparisons (L3).

Usage:  .venv/Scripts/python.exe reference/ch04/make_refs.py [name ...]
        (no arguments = all references; names = derivative, morphology, crop_fig4_3, compat, imclose_pad)

Notes
-----
* Both scripts hard-code ``imread('test.jpg')`` and read from the *current folder*.  They are therefore copied
  verbatim into a scratch folder (``outputs/ch04/verify/scratch``) under non-shadowing names and only the literal
  ``imread('test.jpg')`` token is patched (for the Fig. 4.3(a) crop variants a crop statement is appended to that
  line and ``strel('dis', 7)`` becomes ``strel('dis', 15)``).  Nothing in MATLAB_ROOT is touched and the ch4 folder
  is never put on the MATLAB path.
* ``compat`` calls the toolbox functions (``edge``, ``fspecial``, ``strel``/``getnhood``/``decompose``, ``imerode``,
  ``imdilate``, ``imopen``, ``imclose``, ``imreconstruct``, ``medfilt2``, ``bwareaopen``, ``conv2``, ``imfilter``)
  on controlled arrays saved from Python (``reference/ch04/inputs.mat``) and hand-codes Eqs. (4.7)–(4.15) in
  MATLAB.  The fixtures contain exact gradient ties (linear ramps, diagonal steps) so the compiled thinning rule
  and the ``|bx| >= |by|`` dominant-direction test are probed at their tie cases.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.run_matlab_ref import run_ref  # noqa: E402
from seaice.core import synth  # noqa: E402

MATLAB_ROOT = ROOT / "K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab"
CH = MATLAB_ROOT / "ch4"
REF = ROOT / "reference/ch04"
VERIFY = ROOT / "outputs/ch04/verify"
ML_IMG = VERIFY / "matlab"
SCRATCH = VERIFY / "scratch"
DATA = ROOT / "data/book/ch04"
IMAGE = "test.jpg"

#: Fig. 4.3(a) crop, MATLAB 1-based inclusive (analysis/ch04.md header; = seaice.ch04_ice_edge_detection.FIG_4_3A_CROP_MATLAB)
CROP = ((1600, 2151), (1979, 2552))
CROP_STMT = f"im = im({CROP[0][0]}:{CROP[0][1]}, {CROP[1][0]}:{CROP[1][1]}, :);"

# --- edge fixtures: (name, given threshold literal) --------------------------------------------------------------
EDGE_IMAGES = {"Rnd": "0.2", "Ramp": "0.1", "RampD": "0.1", "Step": "0.1", "StepH": "0.1", "Diag": "0.1", "Cst": "0.1"}
EDGE_METHODS = ("sobel", "prewitt", "roberts")
EDGE_DIRS = ("both", "horizontal", "vertical")
DISK_RADII = range(1, 21)
DISK_N = (0, 4, 6, 8)
DECOMP_CASES = [(3, 4), (5, 4), (7, 4), (10, 4), (15, 4), (16, 4), (20, 4), (7, 6), (7, 8), (12, 6), (12, 8)]
MORPH_IMAGES = ("Bw", "BwB", "Rnd8", "Rnd", "Neg", "I16")
MORPH_SES = {"disk7": "strel('disk', 7)", "asym": "strel(AsymSE)", "even": "strel(EvenSE)",
             "line5": "strel('line', 5, 0)", "line7_45": "strel('line', 7, 45)", "sq3": "strel('square', 3)",
             "dia3": "strel('diamond', 3)", "disk2": "strel('disk', 2)", "pair": "strel('pair', [2 -1])"}


def q(p: Path | str) -> str:
    """MATLAB single-quoted literal with forward slashes (MATLAB on Windows accepts them)."""
    return "'" + str(p).replace("\\", "/").replace("'", "''") + "'"


def _read_m(name: str) -> str:
    return (CH / name).read_text(encoding="utf-8", errors="replace")


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


def make_inputs() -> dict[str, np.ndarray]:
    """Controlled fixtures shared by MATLAB (``inputs.mat``) and the tests (identical arrays on both sides)."""
    rng = np.random.default_rng(4)
    ramp = np.tile(np.arange(9) / 8.0, (7, 1))  # constant column gradient → every interior |bx| equal (ties)
    ramp_d = (np.add.outer(np.arange(9), np.arange(9)) / 16.0)  # |bx| == |by| everywhere (dominant-direction tie)
    step = np.full((9, 11), 0.2)
    step[:, 5:] = 0.8  # vertical step edge
    step_h = np.full((11, 9), 0.2)
    step_h[5:, :] = 0.8  # horizontal step edge
    diag = np.triu(np.ones((9, 9)))  # diagonal step: |bx| == |by| along the edge
    zc = np.zeros((12, 14))  # integer-valued plateaus → exact zeros of an integer Laplacian (zerocross zero case)
    zc[3:8, 4:10] = 5.0
    zc[5:7, 6:8] = 2.0
    zc[9:, :3] = 3.0
    rnd8 = rng.integers(0, 256, size=(64, 96)).astype(np.uint8)
    marker, mask = synth.two_blobs_with_marker()
    bw2 = rng.random((50, 70)) > 0.7  # noisy binary with many small components (bwareaopen)
    bw2[10:30, 10:40] = True
    bwb = np.zeros((20, 24), dtype=bool)
    bwb[2:17, 2:21] = True
    bwb[8:12, 0:2] = True  # a 4-row bridge to the left border leaves 2-px notches above/below it
    msk = np.zeros((5, 5))
    msk[1:4, 1:4] = 1.0
    prof = synth.two_floes_profile()
    prof_mk_e = np.clip(prof.astype(np.int64) + 40, 0, 255).astype(np.uint8)  # uint8-saturating f + h
    return {
        "Rnd": rng.random((64, 96)),
        "Rnd8": rnd8,
        "Ramp": ramp, "RampD": ramp_d, "Step": step, "StepH": step_h, "Diag": diag,
        "Cst": np.full((8, 10), 0.37),
        "Zc": zc,
        "Neg": rng.random((32, 40)) - 0.5,
        "I16": rng.integers(-3000, 3000, size=(20, 30)).astype(np.int16),
        "Bw": mask.astype(np.uint8), "Mk": marker.astype(np.uint8),
        "Bw2": bw2.astype(np.uint8),
        "BwB": bwb.astype(np.uint8),  # object 2 px from the left/top border (imclose border rule)
        "Mk8": np.clip(rnd8.astype(np.int64) - 40, 0, 255).astype(np.uint8),
        "AsymSE": np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8),
        "EvenSE": np.array([[1, 1, 1], [1, 0, 1]], dtype=np.uint8),
        "F48": synth.FIG_4_8_IMAGE.astype(np.uint8), "F48SE": synth.FIG_4_8_SE.astype(np.uint8),
        "Prof": prof, "ProfMkE": prof_mk_e,
        "Msk5": msk,
        "H4": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64),
        "H8": np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64),
        "A8": rng.integers(0, 256, size=(10, 12)).astype(np.uint8),
        "B8": rng.integers(0, 256, size=(10, 12)).astype(np.uint8),
    }


def _prepare() -> None:
    for d in (REF, ML_IMG, SCRATCH):
        d.mkdir(parents=True, exist_ok=True)
    if not (SCRATCH / IMAGE).exists():
        shutil.copy(DATA / IMAGE, SCRATCH / IMAGE)
    tok = f"imread('{IMAGE}')"
    deriv = _read_m("derivative.m")
    morph = _read_m("morphology.m")
    (SCRATCH / "derivative_ref.m").write_text(_patch(deriv, [(tok, tok)]), encoding="utf-8")
    (SCRATCH / "morphology_ref.m").write_text(_patch(morph, [(tok, tok)]), encoding="utf-8")
    (SCRATCH / "derivative_ref_crop.m").write_text(
        _patch(deriv, [(tok + ";", tok + "; " + CROP_STMT)]), encoding="utf-8")
    (SCRATCH / "morphology_ref_crop.m").write_text(
        _patch(morph, [(tok + ";", tok + "; " + CROP_STMT), ("strel('dis', 7)", "strel('dis', 15)")]),
        encoding="utf-8")
    savemat(REF / "inputs.mat", make_inputs())


# ------------------------------------------------------------------------------------------------------------
def ref_derivative():
    code = f"""
run({q(SCRATCH / 'derivative_ref.m')});
[BW_auto, t_auto] = edge(im, 'sobel');
BW_003 = edge(im, 'sobel', 0.03);
BW_prewitt = edge(im, 'prewitt', 0.05);
[BW_prewitt_auto, t_prewitt_auto] = edge(im, 'prewitt');
BW_nothin = edge(im, 'sobel', 0.05, 'nothinning');
[~, ~, gv, gh] = edge(im, 'sobel', 0.05);
[~, ~, gv_p, gh_p] = edge(im, 'prewitt', 0.05);
BW_h = edge(im, 'sobel', 0.05, 'horizontal');
BW_v = edge(im, 'sobel', 0.05, 'vertical');
[BW_log, t_log] = edge(im, 'log', 0.005, 2);
[BW_log_auto, t_log_auto] = edge(im, 'log', [], 2);
BW_rob = edge(im, 'roberts', 0.05);
[BW_rob_auto, t_rob] = edge(im, 'roberts');
BW_med = edge(medfilt2(im, [3 3]), 'sobel', 0.05);
im_med = medfilt2(im, [3 3]);
BW_ao = bwareaopen(BW, 20);
BWc = conv2(double(BW), double(msk));
n_edge = nnz(BW);
{_imw('BW', 'derivative_bw.png')}
{_imw('BW_prewitt', 'derivative_bw_prewitt.png')}
{_imw('BW_003', 'derivative_bw_T0.03.png')}
"""
    return run_ref(code, ["im", "BW", "n_edge", "BW_auto", "t_auto", "BW_003", "BW_prewitt", "BW_prewitt_auto",
                          "t_prewitt_auto", "BW_nothin", "gv", "gh", "gv_p", "gh_p", "BW_h", "BW_v", "BW_log", "t_log",
                          "BW_log_auto", "t_log_auto", "BW_rob", "BW_rob_auto", "t_rob", "BW_med", "im_med", "BW_ao",
                          "BWc", "msk", "r", "c"],
                   REF / "derivative.mat", workdir=SCRATCH, timeout=1800)


def ref_morphology():
    code = f"""
run({q(SCRATCH / 'morphology_ref.m')});
[level, em] = graythresh(im);
nh = getnhood(SE);
seq = decompose(SE); nseq = numel(seq);
{_imw('im', 'morphology_im.png')}
{_imw('I', 'morphology_I.png')}
{_imw('J', 'morphology_J.png')}
{_imw('K', 'morphology_K.png')}
{_imw('BW1', 'morphology_BW1.png')}
{_imw('BW2', 'morphology_BW2.png')}
{_imw('BW', 'morphology_BW.png')}
{_imw('X', 'morphology_X.png')}
{_imw('Y', 'morphology_Y.png')}
{_imw('internal', 'morphology_internal.png')}
{_imw('external', 'morphology_external.png')}
{_imw('basic', 'morphology_basic.png')}
cls_BW1 = class(BW1); cls_internal = class(internal);
"""
    return run_ref(code, ["im", "level", "em", "I", "J", "K", "BW1", "BW2", "BW", "X", "Y", "internal", "external",
                          "basic", "nh", "nseq", "cls_BW1", "cls_internal"],
                   REF / "morphology.mat", workdir=SCRATCH, timeout=1800)


def ref_crop():
    code = f"""
run({q(SCRATCH / 'derivative_ref_crop.m')});
im_d = im; BW_sobel = BW;
[BW_sobel_auto, t_sobel_auto] = edge(im, 'sobel');
BW_prewitt = edge(im, 'prewitt', 0.05);
[BW_log, t_log] = edge(im, 'log', 0.005, 2);
BW_sobel_003 = edge(im, 'sobel', 0.03);
{_imw('BW_sobel', 'crop_bw_sobel.png')}
{_imw('BW_prewitt', 'crop_bw_prewitt.png')}
{_imw('BW_log', 'crop_bw_log.png')}
run({q(SCRATCH / 'morphology_ref_crop.m')});
crop_gray = im;
[level, em] = graythresh(im);
nh15 = getnhood(SE);
{_imw('crop_gray', 'crop_gray.png')}
{_imw('I', 'crop_I.png')}
{_imw('J', 'crop_J.png')}
{_imw('K', 'crop_K.png')}
{_imw('BW1', 'crop_BW1.png')}
{_imw('BW2', 'crop_BW2.png')}
{_imw('BW', 'crop_BW.png')}
{_imw('X', 'crop_X.png')}
{_imw('Y', 'crop_Y.png')}
{_imw('internal', 'crop_internal.png')}
{_imw('external', 'crop_external.png')}
{_imw('basic', 'crop_basic.png')}
"""
    return run_ref(code, ["im_d", "crop_gray", "BW_sobel", "BW_sobel_auto", "t_sobel_auto", "BW_prewitt", "BW_log",
                          "t_log", "BW_sobel_003", "level", "em", "I", "J", "K", "BW1", "BW2", "BW", "X", "Y",
                          "internal", "external", "basic", "nh15"],
                   REF / "crop_fig4_3.mat", workdir=SCRATCH, timeout=1800)


def _compat_code() -> tuple[str, list[str]]:
    lines: list[str] = [f"load({q(REF / 'inputs.mat')});",
                        "Bw = logical(Bw); BwB = logical(BwB); Mk = logical(Mk); Bw2 = logical(Bw2); F48 = logical(F48); F48SE = logical(F48SE);"]
    v: list[str] = []

    def add(line: str, *names: str) -> None:
        lines.append(line)
        v.extend(names)

    # --- fspecial ------------------------------------------------------------------------------------------------
    add("fs_sobel = fspecial('sobel'); fs_prewitt = fspecial('prewitt');", "fs_sobel", "fs_prewitt")
    add("fs_lap0 = fspecial('laplacian', 0); fs_lap02 = fspecial('laplacian', 0.2); fs_lap1 = fspecial('laplacian', 1); "
        "fs_lapd = fspecial('laplacian');", "fs_lap0", "fs_lap02", "fs_lap1", "fs_lapd")
    add("fs_g51 = fspecial('gaussian', 5, 1); fs_g305 = fspecial('gaussian', 3, 0.5); fs_g132 = fspecial('gaussian', 13, 2); "
        "fs_g57 = fspecial('gaussian', [5 7], 1.3); fs_gd = fspecial('gaussian'); fs_gs = fspecial('gaussian', [], 1.5);",
        "fs_g51", "fs_g305", "fs_g132", "fs_g57", "fs_gd", "fs_gs")
    add("fs_l51 = fspecial('log', 5, 1); fs_l132 = fspecial('log', 13, 2); fs_l505 = fspecial('log', 5, 0.5); "
        "fs_ld = fspecial('log'); fs_l79 = fspecial('log', [7 9], 1.2);", "fs_l51", "fs_l132", "fs_l505", "fs_ld", "fs_l79")
    add("fs_avg = fspecial('average'); fs_avg35 = fspecial('average', [3 5]); fs_avg7 = fspecial('average', 7);",
        "fs_avg", "fs_avg35", "fs_avg7")
    add("fs_disk5 = fspecial('disk', 5); fs_disk25 = fspecial('disk', 2.5); fs_diskd = fspecial('disk'); fs_disk1 = fspecial('disk', 1);",
        "fs_disk5", "fs_disk25", "fs_diskd", "fs_disk1")
    add("fs_unsharp = fspecial('unsharp'); fs_unsharp05 = fspecial('unsharp', 0.5);", "fs_unsharp", "fs_unsharp05")
    # --- strel neighbourhoods ------------------------------------------------------------------------------------
    for r in DISK_RADII:
        for n in DISK_N:
            add(f"sd_r{r}_n{n} = getnhood(strel('disk', {r}, {n}));", f"sd_r{r}_n{n}")
    add("sd_default7 = getnhood(strel('disk', 7)); sdis7 = getnhood(strel('dis', 7)); sdi_s7 = getnhood(strel('DISK', 7));",
        "sd_default7", "sdis7", "sdi_s7")
    add("try, strel('di', 7); amb_err = 0; catch, amb_err = 1; end", "amb_err")
    for M in range(1, 11):
        add(f"sdia_{M} = getnhood(strel('diamond', {M}));", f"sdia_{M}")
    add("ssq_3 = getnhood(strel('square', 3)); ssq_4 = getnhood(strel('square', 4)); srect_3_5 = getnhood(strel('rectangle', [3 5]));",
        "ssq_3", "ssq_4", "srect_3_5")
    for (ln, deg) in ((7, 0), (7, 45), (7, 90), (9, 30), (6, 120), (5, 0), (3, 0), (8, 60), (10, 135), (7, 180), (7, 200)):
        add(f"sline_{ln}_{deg} = getnhood(strel('line', {ln}, {deg}));", f"sline_{ln}_{deg}")
    add("soct_0 = getnhood(strel('octagon', 0)); soct_3 = getnhood(strel('octagon', 3)); soct_6 = getnhood(strel('octagon', 6)); "
        "soct_9 = getnhood(strel('octagon', 9));", "soct_0", "soct_3", "soct_6", "soct_9")
    add("spl_2_1_2 = getnhood(strel('periodicline', 2, [1 2])); spl_3_m1_2 = getnhood(strel('periodicline', 3, [-1 2])); "
        "spl_1_1_0 = getnhood(strel('periodicline', 1, [1 0]));", "spl_2_1_2", "spl_3_m1_2", "spl_1_1_0")
    add("spair_2_m1 = getnhood(strel('pair', [2 -1])); spair_m1_3 = getnhood(strel('pair', [-1 3]));", "spair_2_m1", "spair_m1_3")
    add("sarb = getnhood(strel(AsymSE)); sarb2 = getnhood(strel('arbitrary', EvenSE));", "sarb", "sarb2")
    for (r, n) in DECOMP_CASES:
        add(f"seq__ = decompose(strel('disk', {r}, {n})); ndec_r{r}_n{n} = numel(seq__);", f"ndec_r{r}_n{n}")
        for k in range(1, 11):
            add(f"if numel(seq__) >= {k}, dec_r{r}_n{n}_{k} = getnhood(seq__({k})); else, dec_r{r}_n{n}_{k} = []; end",
                f"dec_r{r}_n{n}_{k}")
    # --- erosion / dilation / opening / closing on the fixtures -------------------------------------------------
    for img in MORPH_IMAGES:
        for se_name, se_expr in MORPH_SES.items():
            add(f"er_{img}_{se_name} = imerode({img}, {se_expr}); di_{img}_{se_name} = imdilate({img}, {se_expr}); "
                f"op_{img}_{se_name} = imopen({img}, {se_expr}); cl_{img}_{se_name} = imclose({img}, {se_expr});",
                f"er_{img}_{se_name}", f"di_{img}_{se_name}", f"op_{img}_{se_name}", f"cl_{img}_{se_name}")
    add("er_Bw_disk7nh = imerode(Bw, getnhood(strel('disk', 7))); di_Rnd8_disk7nh = imdilate(Rnd8, getnhood(strel('disk', 7)));",
        "er_Bw_disk7nh", "di_Rnd8_disk7nh")
    add("f48_er = imerode(F48, F48SE); f48_di = imdilate(F48, F48SE);", "f48_er", "f48_di")
    add("po = imopen(Prof, strel('line', 9, 0)); pc = imclose(Prof, strel('line', 9, 0)); "
        "pe = imerode(Prof, strel('line', 9, 0)); pd = imdilate(Prof, strel('line', 9, 0));", "po", "pc", "pe", "pd")
    # --- reconstruction --------------------------------------------------------------------------------------------
    add("rc_bw8 = imreconstruct(Mk, Bw); rc_bw4 = imreconstruct(Mk, Bw, 4); rc_g8 = imreconstruct(Mk8, Rnd8); "
        "rc_g4 = imreconstruct(Mk8, Rnd8, 4); rc_gc = imreconstruct(Mk8, Rnd8, [0 1 0; 1 1 1; 0 1 0]);",
        "rc_bw8", "rc_bw4", "rc_g8", "rc_g4", "rc_gc")
    add("ProfMkD = uint8(max(double(Prof) - 40, 0)); rc_prof_d = imreconstruct(ProfMkD, Prof, [0 0 0; 1 1 1; 0 0 0]); "
        "rc_prof_e = imcomplement(imreconstruct(imcomplement(ProfMkE), imcomplement(Prof), [0 0 0; 1 1 1; 0 0 0]));",
        "ProfMkD", "rc_prof_d", "rc_prof_e")
    add("rc_dbl = imreconstruct(max(Rnd - 0.3, 0), Rnd);", "rc_dbl")
    # --- morphological gradients / MATLAB minus ---------------------------------------------------------------------
    add("SE7 = strel('disk', 7); mg_b_Bw = imdilate(Bw, SE7) - imerode(Bw, SE7); mg_i_Bw = Bw - imerode(Bw, SE7); "
        "mg_e_Bw = imdilate(Bw, SE7) - Bw; cls_mg_Bw = class(mg_b_Bw);", "mg_b_Bw", "mg_i_Bw", "mg_e_Bw", "cls_mg_Bw")
    add("mg_b_Rnd8 = imdilate(Rnd8, SE7) - imerode(Rnd8, SE7); mg_i_Rnd8 = Rnd8 - imerode(Rnd8, SE7); "
        "mg_e_Rnd8 = imdilate(Rnd8, SE7) - Rnd8;", "mg_b_Rnd8", "mg_i_Rnd8", "mg_e_Rnd8")
    add("mg_b_Neg = imdilate(Neg, SE7) - imerode(Neg, SE7); mg_i_I16 = I16 - imerode(I16, SE7);", "mg_b_Neg", "mg_i_I16")
    add("sat_ab = A8 - B8; sat_ba = B8 - A8;", "sat_ab", "sat_ba")
    # --- edge on the fixtures ---------------------------------------------------------------------------------------
    for img, T in EDGE_IMAGES.items():
        for m in EDGE_METHODS:
            dirs = EDGE_DIRS if m != "roberts" else ("both",)
            for d in dirs:
                for thin in ("thin", "nothin"):
                    opt = "'thinning'" if thin == "thin" else "'nothinning'"
                    dopt = f", '{d}'" if m != "roberts" else ""
                    base = f"e_{m}_{img}_{d}_{thin}"
                    add(f"[{base}_auto, t_{m}_{img}_{d}_{thin}_auto] = edge({img}, '{m}', []{dopt}, {opt});",
                        f"{base}_auto", f"t_{m}_{img}_{d}_{thin}_auto")
                    add(f"{base}_T = edge({img}, '{m}', {T}{dopt}, {opt});", f"{base}_T")
        add(f"[~, ~, gv_sobel_{img}, gh_sobel_{img}] = edge({img}, 'sobel'); [~, ~, gv_prewitt_{img}, gh_prewitt_{img}] = edge({img}, 'prewitt'); "
            f"[~, ~, gv_roberts_{img}, gh_roberts_{img}] = edge({img}, 'roberts');",
            f"gv_sobel_{img}", f"gh_sobel_{img}", f"gv_prewitt_{img}", f"gh_prewitt_{img}", f"gv_roberts_{img}", f"gh_roberts_{img}")
    add("[e_log_Rnd_auto, t_log_Rnd_auto] = edge(Rnd, 'log'); e_log_Rnd_002 = edge(Rnd, 'log', 0.02); "
        "[e_log_Rnd_s15, t_log_Rnd_s15] = edge(Rnd, 'log', [], 1.5); e_log_Rnd_s3 = edge(Rnd, 'log', 0.01, 3); "
        "e_log_Step = edge(Step, 'log', 0.001, 1); [e_log_Ramp, t_log_Ramp] = edge(Ramp, 'log'); "
        "[e_log_Cst, t_log_Cst] = edge(Cst, 'log'); e_log_Rnd8d = edge(double(Rnd8)/256, 'log', 0.005, 2);",
        "e_log_Rnd_auto", "t_log_Rnd_auto", "e_log_Rnd_002", "e_log_Rnd_s15", "t_log_Rnd_s15", "e_log_Rnd_s3",
        "e_log_Step", "e_log_Ramp", "t_log_Ramp", "e_log_Cst", "t_log_Cst", "e_log_Rnd8d")
    add("e_zc_Zc_H4 = edge(Zc, 'zerocross', 0.5, H4); [e_zc_Zc_H8, t_zc_Zc_H8] = edge(Zc, 'zerocross', [], H8); "
        "e_zc_Zc_H4_0 = edge(Zc, 'zerocross', 0, H4); e_zc_Rnd_log7 = edge(Rnd, 'zerocross', 0.05, fspecial('log', 7, 1)); "
        "e_zc_Rnd8d_H4 = edge(double(Rnd8), 'zerocross', 3, H4);",
        "e_zc_Zc_H4", "e_zc_Zc_H8", "t_zc_Zc_H8", "e_zc_Zc_H4_0", "e_zc_Rnd_log7", "e_zc_Rnd8d_H4")
    add("op_log7 = fspecial('log', 7, 1); op_log7 = op_log7 - sum(op_log7(:))/numel(op_log7); b_log_Step = imfilter(Step, op_log7, 'replicate'); "
        "op_log13 = fspecial('log', 13, 2); op_log13 = op_log13 - sum(op_log13(:))/numel(op_log13); b_log_Cst = imfilter(Cst, op_log13, 'replicate');",
        "b_log_Step", "b_log_Cst")
    # MATLAB imclose.m pads with padarray(A, ceil(size(nhood)/2), 'both') (value 0) before erode(dilate(.)); imopen does not
    add("cl_pad_Rnd8_disk7 = imerode(imdilate(padarray(Rnd8, [7 7], 'both'), SE7), SE7); cl_pad_Rnd8_disk7 = cl_pad_Rnd8_disk7(8:end-7, 8:end-7); "
        "cl_nopad_Rnd8_disk7 = imerode(imdilate(Rnd8, SE7), SE7);", "cl_pad_Rnd8_disk7", "cl_nopad_Rnd8_disk7")
    # --- hand-coded book equations ------------------------------------------------------------------------------
    add("Gx_s = imfilter(Rnd, [-1 -2 -1; 0 0 0; 1 2 1], 'replicate'); Gy_s = imfilter(Rnd, [-1 0 1; -2 0 2; -1 0 1], 'replicate'); "
        "Gx_p = imfilter(Rnd, [-1 -1 -1; 0 0 0; 1 1 1], 'replicate'); Gy_p = imfilter(Rnd, [-1 0 1; -1 0 1; -1 0 1], 'replicate'); "
        "Gx_f = imfilter(Rnd, [0; -1; 1], 'replicate'); Gy_f = imfilter(Rnd, [0 -1 1], 'replicate');",
        "Gx_s", "Gy_s", "Gx_p", "Gy_p", "Gx_f", "Gy_f")
    add("Gx_s0 = imfilter(Rnd, [-1 -2 -1; 0 0 0; 1 2 1]); mag_l2 = sqrt(Gx_s.^2 + Gy_s.^2); mag_sq = Gx_s.^2 + Gy_s.^2; "
        "mag_l1 = abs(Gx_s) + abs(Gy_s); dir_book = atan2(Gx_s, Gy_s); dir_math = atan2(Gy_s, Gx_s); thr_mag = mag_l2 > 0.5;",
        "Gx_s0", "mag_l2", "mag_sq", "mag_l1", "dir_book", "dir_math", "thr_mag")
    add("Lap4 = imfilter(Rnd, H4, 'replicate'); Lap8 = imfilter(Rnd, H8, 'replicate'); "
        "d2x = imfilter(Rnd, [0; 0; 1; -2; 1], 'replicate'); d2y = imfilter(Rnd, [0 0 1 -2 1], 'replicate');",
        "Lap4", "Lap8", "d2x", "d2y")
    add("[xg, yg] = meshgrid(-2:2, -2:2); s = 1; Gau_an = exp(-(xg.^2 + yg.^2) / (2*s^2)) / (2*pi*s^2); "
        "LoG_an = (xg.^2 + yg.^2 - 2*s^2) / (2*pi*s^6) .* exp(-(xg.^2 + yg.^2) / (2*s^2)); "
        "Gau_n = Gau_an / sum(Gau_an(:)); gauss_conv_lap = conv2(fspecial('gaussian', 5, 1), H4, 'same');",
        "Gau_an", "LoG_an", "Gau_n", "gauss_conv_lap")
    add("pos = max(Lap4, 0); neg = min(Lap4, 0); mxp = imdilate(pos, ones(3)); mnn = -imdilate(-neg, ones(3)); "
        "zc_text = (mxp > 0) & (mnn < 0) & (mxp - mnn > 0.3);", "zc_text")
    # --- commented-out derivative.m post-processing --------------------------------------------------------------
    add("med_Rnd = medfilt2(Rnd, [3 3]); med_Rnd8 = medfilt2(Rnd8, [3 3]); med_Rnd_sym = medfilt2(Rnd, [3 3], 'symmetric');",
        "med_Rnd", "med_Rnd8", "med_Rnd_sym")
    add("ao_20 = bwareaopen(Bw2, 20); ao_20_4 = bwareaopen(Bw2, 20, 4); ao_5 = bwareaopen(Bw2, 5); n_ao = nnz(ao_20);",
        "ao_20", "ao_20_4", "ao_5", "n_ao")
    add("cf = conv2(double(Bw2), double(Msk5)); cs = conv2(double(Bw2), double(Msk5), 'same'); cv = conv2(double(Bw2), double(Msk5), 'valid');",
        "cf", "cs", "cv")
    return "\n".join(lines) + "\n", v


def ref_compat():
    code, vars_ = _compat_code()
    (VERIFY / "compat_code.m").write_text(code, encoding="utf-8")
    return run_ref(code, vars_, REF / "compat.mat", timeout=1800)


#: SEs that exercise both R2025a closing paths on SIGNED images (Neg = double in [-0.5, 0.5), I16):
#:   morphop_fast.m routes an SE with nnz < 600 and every side <= 15 (and not an all-ones rectangle) to the Halide
#:   kernel, whose border value is the class minimum (-Inf / intmin); everything else falls through to imclose.m,
#:   which pads by ceil(size(nhood)/2) with ZEROS.  The 9 compat SEs are all <= 15 px, so the "side > 15, not
#:   all-ones" branch (pad 0 on a signed image) was not covered by compat.mat.
PAD_SES = {"dia7": "strel('diamond', 7)",           # 15x15, 113 px  -> Halide, class minimum
           "disk8": "strel('disk', 8)",              # 15x15, 185 px  -> Halide, class minimum
           "dia8": "strel('diamond', 8)",            # 17x17, 145 px  -> imclose.m, pad 0
           "disk10": "strel('disk', 10)",            # 19x19, 321 px, decomposed -> imclose.m, pad 0
           "line31_45": "strel('line', 31, 45)",     # 23x23, 23 px   -> imclose.m, pad 0
           "rect3_17": "strel('rectangle', [3 17])"}  # 3x17 all ones -> imclose.m, pad 0
PAD_IMAGES = ("Neg", "I16", "Rnd8")


def _imclose_pad_code() -> tuple[str, list[str]]:
    lines: list[str] = [f"load({q(REF / 'inputs.mat')});"]
    v: list[str] = []

    def add(line: str, *names: str) -> None:
        lines.append(line)
        v.extend(names)

    add("s__ = settings; halide_on = double(s__.images.UseHalide.ActiveValue);", "halide_on")
    for img in PAD_IMAGES:
        for se_name, se_expr in PAD_SES.items():
            add(f"cl_{img}_{se_name} = imclose({img}, {se_expr});", f"cl_{img}_{se_name}")
    # hand-coded variants that prove which pad value MATLAB used on the signed double image
    for se_name, se_expr in (("dia8", PAD_SES["dia8"]), ("dia7", PAD_SES["dia7"]), ("disk10", PAD_SES["disk10"])):
        add(f"se__ = {se_expr}; ps__ = ceil(size(getnhood(se__))/2); "
            f"t0__ = imerode(imdilate(padarray(Neg, ps__, 0, 'both'), se__), se__); "
            f"cl0_Neg_{se_name} = t0__(ps__(1)+1:end-ps__(1), ps__(2)+1:end-ps__(2)); "
            f"tm__ = imerode(imdilate(padarray(Neg, ps__, -Inf, 'both'), se__), se__); "
            f"clm_Neg_{se_name} = tm__(ps__(1)+1:end-ps__(1), ps__(2)+1:end-ps__(2)); "
            f"tn__ = imerode(imdilate(Neg, se__), se__); cln_Neg_{se_name} = tn__;",
            f"cl0_Neg_{se_name}", f"clm_Neg_{se_name}", f"cln_Neg_{se_name}")
    return "\n".join(lines) + "\n", v


def ref_imclose_pad():
    code, vars_ = _imclose_pad_code()
    (VERIFY / "imclose_pad_code.m").write_text(code, encoding="utf-8")
    return run_ref(code, vars_, REF / "imclose_pad.mat", timeout=600)


ALL = {
    "derivative": ref_derivative,
    "morphology": ref_morphology,
    "crop_fig4_3": ref_crop,
    "compat": ref_compat,
    "imclose_pad": ref_imclose_pad,
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
            log[name] = {"engine": res.engine, "version": res.version, "out": res.out_mat.name, "status": "ok"}
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
