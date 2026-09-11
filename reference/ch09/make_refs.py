"""Generate the MATLAB (R2025a) reference outputs for chapter 9 (model sea ice image processing applications).

Runs the ORIGINAL ``.m`` files of ``MATLAB_ROOT/ch9/`` (and ``ch9/Model_Ice_Floe_Identification/``) through
``tools.run_matlab_ref.run_ref`` (``matlab -batch``, ``set(0,'DefaultFigureVisible','off')``,
``save(..., '-v7')``) and writes ``reference/ch09/*.mat``.

Usage::

    .venv/Scripts/python.exe reference/ch09/make_refs.py [name ...]
    names = video, probes, block, rect, model, polybool, r13, movie_otsu, movie_kmeans, movie_floe, demo

Scratch policy (CLAUDE.md rule 8 — nothing under ``MATLAB_ROOT`` is ever touched)
---------------------------------------------------------------------------------
Every ``.m`` file is **copied** into ``outputs/ch09/verify/scratch/`` under a non-shadowing name and MATLAB runs
with that folder as its cwd; the ch6/ch7/ch9 folders are never put on the MATLAB path (they ship
``minboundrect.m``, and the three ``movie_*.m`` / ``block_threshold.m`` are scripts that write into the cwd).
The copies are line-for-line the originals except for the patches produced by :func:`_patch`, which are recorded
verbatim (removed line beside its replacement) in ``reference/ch09/patches_verified.json``.  Only two classes of
patch exist in this chapter, and **no numeric expression is altered anywhere**:

1. **IO** — ``mmreader`` (``movie_floe.m`` line 5) and ``movie2avi`` (``movie_otsu.m`` line 65,
   ``movie_kmeans.m`` line 79) were **removed** from R2025a; and ``minboundrect.m``'s
   ``convhull(x, y, {'Qt'})`` is rejected by R2025a ("CONVHULL no longer supports or requires Qhull-specific
   options").  The input file names are re-pointed at the Tier-3 stand-ins because the book's
   ``04100_analyse.jpg`` / ``dypic_05100_cam1_top.avi`` / ``05100.avi`` do not ship and are unobtainable (risk R1).
2. **graphics / output plumbing** — the ``plot``/``getframe`` replay blocks, and appended ``save`` statements that
   dump the struct arrays into plain numeric arrays.  ``model_ice_demo.m``'s lines 53–66 are **un-commented**
   (they are the authors' own lines, reproduced verbatim) so that ``rect.m`` and ``model_ice_model.m`` run.

Data provenance
---------------
``model_ice.jpg`` is the shipped book image.  ``04100_analyse.jpg``, ``dypic_synth_top.avi`` and
``05100_synth_segmented.avi`` are **Tier-3 synthetic stand-ins** written once by
``seaice.ch09_model_ice.ensure_synthetic_*`` and read back by both engines, so no number computed from them is a
book number — they buy parity against MATLAB, never an L4 claim.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.run_matlab_ref import run_ref  # noqa: E402
from reference.ch09 import fixtures as FX  # noqa: E402

MATLAB_ROOT = ROOT / "K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab"
CH9 = MATLAB_ROOT / "ch9"
SIFI = CH9 / "Model_Ice_Floe_Identification"
REF = ROOT / "reference/ch09"
VERIFY = ROOT / "outputs/ch09/verify"
SCRATCH = VERIFY / "scratch"
SYNTH = ROOT / "data/synthetic/ch09"
BOOK = ROOT / "data/book/ch09/Model_Ice_Floe_Identification"

PATCHES: list[dict] = []

VERBATIM = ["GVF_distance.m", "GVF.m", "gradient2.m", "xconv2.m", "gaussianMask.m", "gaussianBlur.m",
            "BoundMirrorExpand.m", "BoundMirrorEnsure.m", "BoundMirrorShrink.m", "snakedeform.m",
            "snakeinterp.m", "snakeindex.m", "snakedisp.m", "rect.m", "model_ice_model.m"]


def _read_lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def _patch(src: Path, dst_name: str, edits: dict[int, str | None], note: str, append: str = "") -> Path:
    """Copy ``src`` to ``SCRATCH/dst_name`` applying 1-based line ``edits`` (``None`` deletes the line)."""
    lines = _read_lines(src)
    record, out = [], []
    for i, line in enumerate(lines, start=1):
        if i in edits:
            new = edits[i]
            record.append({"line": i, "removed": line, "replaced_by": new})
            if new is not None:
                out.append(new)
        else:
            out.append(line)
    if append:
        record.append({"line": "appended", "removed": None, "replaced_by": append})
        out.append(append)
    dst = SCRATCH / dst_name
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    PATCHES.append({"source": str(src.relative_to(MATLAB_ROOT)).replace("\\", "/"), "target": dst_name,
                    "note": note, "edits": record})
    return dst


def _copy_verbatim(src: Path, dst_name: str | None = None) -> Path:
    dst = SCRATCH / (dst_name or src.name)
    shutil.copyfile(src, dst)
    return dst


def prepare_scratch() -> None:
    """Populate ``outputs/ch09/verify/scratch`` with the (patched) ``.m`` copies and the input data files."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for name in VERBATIM:
        _copy_verbatim(SIFI / name)
    PATCHES.append({"source": "ch9/Model_Ice_Floe_Identification/{" + ", ".join(VERBATIM) + "}",
                    "target": "(same name)", "note": "Copied verbatim, zero edits.", "edits": []})

    # --- minboundrect.m: the one literal R2025a rejects (the same patch ch06's reference already carries) ----
    mb_lines = _read_lines(SIFI / "minboundrect.m")
    mb_edits = {}
    for i, line in enumerate(mb_lines, start=1):
        if "convhull" in line and "{'Qt'}" in line:
            mb_edits[i] = line.replace(",{'Qt'}", "").replace(", {'Qt'}", "")
    _patch(SIFI / "minboundrect.m", "minboundrect.m", mb_edits,
           "R2025a: 'CONVHULL no longer supports or requires Qhull-specific options'. The option list is dropped; "
           "the hull point set is unchanged. Graphics/API only.")

    # --- block_threshold.m: rename + save; the input is re-pointed at the Tier-3 tank image -------------------
    bt = _read_lines(CH9 / "block_threshold.m")
    bt_edits = {}
    for i, line in enumerate(bt, start=1):
        if "imread(" in line:
            bt_edits[i] = "I = imread('04100_analyse.jpg');   % Tier-3 stand-in, same file name"
    _patch(CH9 / "block_threshold.m", "ch09_block_threshold_ref.m", bt_edits,
           "Renamed only (non-shadowing scratch copy) and a `save` appended. The imread line keeps the SAME file "
           "name; the scratch cwd holds the Tier-3 04100_analyse.jpg (the book's is unobtainable, risk R1). "
           "Every numeric statement -- graythresh, th = t*255, the `if temp(r1,c1) >= th` double loop, "
           "im2bw(temp, t), IC = n/(r_t*c_t) -- is verbatim.",
           append="save('ch09_block.mat','thresh','IC','r','c','n_r','n_c','c_r','c_c','t1','t2','t3','t4');")

    # --- movie_otsu.m ----------------------------------------------------------------------------------------
    _patch(CH9 / "movie_otsu.m", "ch09_movie_otsu_ref.m",
           {7: "readerobj = VideoReader('dypic_synth_top.avi');",
            51: None, 52: None, 53: None,
            59: None, 60: None, 61: None, 62: None, 63: None, 65: None},
           "IO/graphics only. `movie2avi` is REMOVED in R2025a (exist == 0), so the getframe/movie2avi replay of "
           "lines 59-65 is replaced by a VideoWriter('Uncompressed AVI') block that writes the SAME frames "
           "(bw(k).cadata) at the SAME 12 fps; the IC plot of lines 51-53 is display only and IC is saved instead. "
           "The input name points at the Tier-3 uncompressed AVI. NO numeric expression is touched: t(k) = "
           "graythresh(...), the `if I(k).cdata(i,j) >= t*255` double loop (E4), `n = n+1` (E5), "
           "bw(k).cadata = im2bw(I(k).cdata, t(k)) and IC(k) = n(k)/(r*c-r2*c2) are verbatim.",
           append=("bwstack = false(size(bw(1).cadata,1), size(bw(1).cadata,2), numFrames);\n"
                   "graystack = zeros(size(bw(1).cadata,1), size(bw(1).cadata,2), numFrames, 'uint8');\n"
                   "for k = 1 : numFrames\n"
                   "  bwstack(:,:,k) = bw(k).cadata;\n"
                   "  graystack(:,:,k) = I(k).cdata;\n"
                   "end\n"
                   "vw = VideoWriter('otsu.avi','Uncompressed AVI'); vw.FrameRate = 12; open(vw);\n"
                   "for k = 1 : numFrames, writeVideo(vw, uint8(bw(k).cadata)*255); end\n"
                   "close(vw);\n"
                   "I1first = I1(1).cadata; I2first = I2(1).cadata;\n"
                   "save('ch09_movie_otsu.mat','t','n','IC','bwstack','graystack','numFrames','r','c','r2','c2',"
                   "'I1first','I2first','-v7');"))

    # --- movie_kmeans.m --------------------------------------------------------------------------------------
    _patch(CH9 / "movie_kmeans.m", "ch09_movie_kmeans_ref.m",
           {7: "readerobj = VideoReader('dypic_synth_top.avi');",
            65: None, 67: None, 68: None, 69: None,
            73: None, 74: None, 75: None, 76: None, 77: None, 79: None},
           "The same two IO patches as movie_otsu.m plus the display-only plot. Line 44 "
           "`kmeans(ima, 2, 'EmptyAction','singleton')` is the Statistics Toolbox routine with its own RNG, so "
           "this reference pins `si`, the denominator, the column-major reshape and the cluster MEANS; the label "
           "numbers themselves are not reproducible (label `approx`). `rng(0)` is set by the driver, not by a "
           "patch to the file.",
           append=("outstack = zeros(si(1), si(2), numFrames, 'uint8');\n"
                   "mapstack = zeros(si(1)*si(2), numFrames);\n"
                   "smeans = zeros(2, numFrames);\n"
                   "for k = 1 : numFrames\n"
                   "  outstack(:,:,k) = out(k).cdata;\n"
                   "  mapstack(:,k) = map1(k).cdata;\n"
                   "  imak = ima(k).cdata;\n"
                   "  for i = 1:2, smeans(i,k) = sum(imak(map1(k).cdata==i))/sum(map1(k).cdata==i); end\n"
                   "end\n"
                   "vw = VideoWriter('05400_kmeans.avi','Uncompressed AVI'); vw.FrameRate = 12; open(vw);\n"
                   "for k = 1 : numFrames, writeVideo(vw, out(k).cdata); end\n"
                   "close(vw);\n"
                   "save('ch09_movie_kmeans.mat','IC','a','outstack','mapstack','smeans','si','r2','c2',"
                   "'numFrames','-v7');"))

    # --- movie_floe.m ----------------------------------------------------------------------------------------
    _patch(SIFI / "movie_floe.m", "ch09_movie_floe_ref.m",
           {5: "readerobj = VideoReader('05100_synth_segmented.avi');"},
           "ONE IO patch: `mmreader` is REMOVED in R2025a (exist == 0, 'Undefined function mmreader for input "
           "arguments of type char'), so line 5 becomes VideoReader; `read(v)` and `get(v,'numberOfFrames')` still "
           "work unchanged. The file has NO plotting code, so nothing else is removed. NO numeric expression is "
           "touched: im2bw(mov(k).cdata) (level-free), bwareaopen(.., 20, 4), bwlabel(.., 4), "
           "regionprops(cc,'basic') and floe(k) = max(ice_areas) are verbatim.",
           append=("bwstack = false(size(out(1).cdata,1), size(out(1).cdata,2), numFrames);\n"
                   "imstack = false(size(im(1).cdata,1), size(im(1).cdata,2), numFrames);\n"
                   "nc = zeros(1,numFrames);\n"
                   "for k = 1 : numFrames\n"
                   "  bwstack(:,:,k) = out(k).cdata;\n"
                   "  imstack(:,:,k) = im(k).cdata;\n"
                   "  nc(k) = max(max(bwlabel(out(k).cdata,4)));\n"
                   "end\n"
                   "lab5 = bwlabel(out(5).cdata,4); lab5 = double(lab5);\n"
                   "rp5 = regionprops(bwlabel(out(5).cdata,4),'basic');\n"
                   "area5 = [rp5.Area]; cent5 = cat(1, rp5.Centroid); bbox5 = cat(1, rp5.BoundingBox);\n"
                   "save('ch09_movie_floe.mat','floe','bwstack','imstack','nc','numFrames','lab5','area5',"
                   "'cent5','bbox5','-v7');"))

    # --- model_ice_demo.m: un-comment the authors' own lines 53-66 -------------------------------------------
    demo_edits = {i: None for i in range(53, 67)}
    demo_edits[53] = "% % rectanglarize   [lines 54-66 un-commented verbatim below]"
    demo_edits[54] = "S = rect(bw4);"
    demo_edits[65] = "k1 = 0.4; k2 = 2.5;"
    demo_edits[66] = "s_model = model_ice_model( S, bw4, k1, k2 );"
    _patch(SIFI / "model_ice_demo.m", "ch09_model_ice_demo_ref.m", demo_edits,
           "Renamed only; `model_ice.jpg` is copied into the scratch cwd, so line 9's imread is untouched. "
           "Lines 53-66 are COMMENTED OUT in the shipped file; this copy un-comments the three executable ones "
           "(`S = rect(bw4);`, `k1 = 0.4; k2 = 2.5;`, `s_model = model_ice_model(S, bw4, k1, k2);`) verbatim and "
           "drops the commented display loop of lines 55-62. `figure, imshow(bw2)` on line 46 runs headless under "
           "-batch. NO numeric expression changed.",
           append=("nS = numel(S); SV = zeros(5,2,nS); SC = zeros(nS,2); SA = zeros(nS,1); SP = zeros(nS,1);\n"
                   "for i = 1:nS, SV(:,:,i)=S(i).Vertices; SC(i,:)=S(i).Center; SA(i)=S(i).Area; "
                   "SP(i)=S(i).Perimeter; end\n"
                   "nM = numel(s_model); MV = zeros(5,2,nM); MC = zeros(nM,2); MA = zeros(nM,1); "
                   "MP = zeros(nM,1); MI = false(nM,nM);\n"
                   "for i = 1:nM\n"
                   "  MV(:,:,i)=s_model(i).Vertices; MC(i,:)=s_model(i).Center; MA(i)=s_model(i).Area;\n"
                   "  MP(i)=s_model(i).Perimeter; idx = s_model(i).Intersection;\n"
                   "  if ~isempty(idx), MI(i, idx) = true; end\n"
                   "end\n"
                   "Igray = rgb2gray(I); glevel = graythresh(Igray); bw0 = im2bw(Igray, glevel);\n"
                   "save('ch09_demo.mat','bw1','bw2','bw3','bw4','SV','SC','SA','SP','MV','MC','MA','MP','MI',"
                   "'nS','nM','Igray','glevel','bw0','-v7');"))

    # --- data files ------------------------------------------------------------------------------------------
    for src in (BOOK / "model_ice.jpg", SYNTH / "04100_analyse.jpg",
                SYNTH / "dypic_synth_top.avi", SYNTH / "05100_synth_segmented.avi"):
        shutil.copyfile(src, SCRATCH / src.name)

    (REF / "patches_verified.json").write_text(json.dumps(PATCHES, indent=1), encoding="utf-8")


# ===================================================================================================
# Fixture .mat written for MATLAB to load
# ===================================================================================================

def write_inputs() -> None:
    V, shape, _rel = FX.model_rect_set()
    RV, rshape, _k = FX.ratio_rect_set()
    A, B, names = FX.polybool_pairs()
    savemat(str(SCRATCH / "ch09_inputs.mat"), {
        "rect_mask_a": FX.rect_mask_a().astype(np.uint8),
        "rect_mask_b": FX.rect_mask_b().astype(np.uint8),
        "rect_mask_degenerate": FX.rect_mask_degenerate().astype(np.uint8),
        "model_V": V, "model_shape": np.array(shape, dtype=float),
        "ratio_V": RV, "ratio_shape": np.array(rshape, dtype=float),
        "pb_A": A, "pb_B": B, "pb_names": np.array(names, dtype=object),
        "tie_block": FX.tie_block_image(),
    }, do_compression=True)


# ===================================================================================================
# Sessions
# ===================================================================================================

def ref_video():
    code = """
v1 = VideoReader('dypic_synth_top.avi');
F1 = read(v1); n1 = get(v1, 'numberOfFrames'); sz1 = size(F1); cls1 = class(F1);
h1 = v1.Height; w1 = v1.Width; fps1 = v1.FrameRate;
v2 = VideoReader('05100_synth_segmented.avi');
F2 = read(v2); n2 = get(v2, 'numberOfFrames'); sz2 = size(F2); cls2 = class(F2);
h2 = v2.Height; w2 = v2.Width; fps2 = v2.FrameRate;
u2 = unique(F2(:));
"""
    return run_ref(code, ["F1", "n1", "sz1", "cls1", "h1", "w1", "fps1",
                          "F2", "n2", "sz2", "cls2", "h2", "w2", "fps2", "u2"],
                   REF / "video.mat", workdir=SCRATCH, timeout=1800)


def ref_probes():
    code = r"""
% ---- erratum E4: `if scalar >= vector` is `all(...)` ------------------------------------------------
e4_vec = uint8(50) >= [25 51];
e4_taken = false; if uint8(50) >= [25 51], e4_taken = true; end
e4_taken_all_below = false; if uint8(50) >= [25 40], e4_taken_all_below = true; end
e4_empty_taken = false; if uint8(50) >= [], e4_empty_taken = true; end
% ---- erratum E5: `n = n+1` increments EVERY element of the growing vector ---------------------------
clear n ic
counts = [7 6 4];
for k = 1 : 3
    n(k) = 0;
    for q = 1 : counts(k)
        n = n + 1;
    end
    ic(k) = n(k);
end
e5_n = n; e5_ic = ic; e5_counts = counts;
% ---- R13: `floe(k) = []` past the end ERRORS; only k <= numel(x) deletes ----------------------------
r13_id = ''; r13_msg = '';
try
    fl = []; fl(1) = 5; fl(2) = [];
catch err
    r13_id = err.identifier; r13_msg = err.message;
end
r13_delete = [1 2 3]; r13_delete(2) = [];
% ---- the ch3 `>` vs ch9 `>=` rule, and im2bw's own strict `>` ---------------------------------------
cmp_gt = uint8([100 101 102]) > 101;
cmp_ge = uint8([100 101 102]) >= 101;
cmp_im2bw = im2bw(uint8([100 101 102]), 101/255);
cmp_half = uint8([104 105 106]) >= 104.5;
% ---- regionprops 'basic' shorthand ------------------------------------------------------------------
rp = regionprops(logical([1 1 0; 0 1 1]), 'basic');
rp_fields = fieldnames(rp);
% ---- removed / surviving functions in R2025a --------------------------------------------------------
ex_mmreader = exist('mmreader'); ex_movie2avi = exist('movie2avi'); ex_avifile = exist('avifile');
ex_polybool = exist('polybool'); ex_strmatch = exist('strmatch'); ex_videoreader = exist('VideoReader');
% ---- im2bw with NO level on an RGB frame = rgb2gray then > 127.5 ------------------------------------
rgbtest = uint8(cat(3, [0 127 128 255; 10 20 30 40], [0 127 128 255; 10 20 30 40], ...
                       [0 127 128 255; 10 20 30 40]));
im2bw_nolevel = im2bw(rgbtest);
gray_of_rgbtest = rgb2gray(rgbtest);
% ---- rect.m's dead `metric` (erratum E8): a bogus metric must NOT error -----------------------------
S = load('ch09_inputs.mat');
bwA = logical(S.rect_mask_a);
e8_id = ''; e8_ok = false;
try
    rr = rect(bwA, 'not-a-metric'); e8_ok = true; e8_n = numel(rr);
catch err2
    e8_id = err2.identifier;
end
rA = rect(bwA); rP = rect(bwA, 'p');
e8_same_as_p = isequal(rA, rP);
% ---- minboundrect's ring: 5 points, closed, orientation --------------------------------------------
[r_, c_] = find(bwlabel(bwA,4) == 2);   % the 4 x 20 sliver
[mrx, mry, mra, mrp] = minboundrect(c_, r_, 'a');
mr_closed = (mrx(1) == mrx(5)) && (mry(1) == mry(5));
mr_signed_area = polyarea(mrx(1:4), mry(1:4));
mr_cross = (mrx(2)-mrx(1))*(mry(3)-mry(2)) - (mry(2)-mry(1))*(mrx(3)-mrx(2));
"""
    return run_ref(code, ["e4_vec", "e4_taken", "e4_taken_all_below", "e4_empty_taken",
                          "e5_n", "e5_ic", "e5_counts", "r13_id", "r13_msg", "r13_delete",
                          "cmp_gt", "cmp_ge", "cmp_im2bw", "cmp_half", "rp_fields",
                          "ex_mmreader", "ex_movie2avi", "ex_avifile", "ex_polybool", "ex_strmatch",
                          "ex_videoreader", "im2bw_nolevel", "gray_of_rgbtest",
                          "e8_id", "e8_ok", "e8_same_as_p", "mrx", "mry", "mra", "mrp",
                          "mr_closed", "mr_signed_area", "mr_cross"],
                   REF / "probes.mat", workdir=SCRATCH, timeout=900)


def ref_block():
    code = r"""
run('ch09_block_threshold_ref.m');
thresh_tank = thresh; IC_tank = IC; r_tank = r; c_tank = c;
Itank = rgb2gray(imread('04100_analyse.jpg'));
% the six displayed tiles, assembled (im2bw's strict `>` at the per-block level)
bw_tank = false(r_tank, c_tank);
for i = 1 : n_r
    for j = 1 : n_c
        temp = Itank(t1(i):t2(i), t3(j):t4(j));
        bw_tank(t1(i):t2(i), t3(j):t4(j)) = im2bw(temp, graythresh(temp));
    end
end
% the same loop on the constructed tie fixture, with BOTH comparison rules
S = load('ch09_inputs.mat');
tie = uint8(S.tie_block);
[rt, ct] = size(tie); nr2 = 2; nc2 = 3; cr2 = rt/nr2; cc2 = ct/nc2;
u1 = (0:nr2-1)*cr2 + 1; u2 = (1:nr2)*cr2; u3 = (0:nc2-1)*cc2 + 1; u4 = (1:nc2)*cc2;
thresh_tie = zeros(1, nr2*nc2); n_ge_tie = zeros(1, nr2*nc2); n_gt_tie = zeros(1, nr2*nc2);
IC_tie_ge = zeros(1, nr2*nc2); bw_tie = false(rt, ct);
for i = 1 : nr2
    for j = 1 : nc2
        temp = tie(u1(i):u2(i), u3(j):u4(j));
        t = graythresh(temp); th = t*255;
        thresh_tie((i-1)*nc2+j) = th;
        [r_t, c_t] = size(temp);
        nge = 0; ngt = 0;
        for r1 = 1:r_t
            for c1 = 1:c_t
                if temp(r1,c1) >= th, nge = nge + 1; end
                if temp(r1,c1) >  th, ngt = ngt + 1; end
            end
        end
        n_ge_tie((i-1)*nc2+j) = nge; n_gt_tie((i-1)*nc2+j) = ngt;
        IC_tie_ge((i-1)*nc2+j) = nge/(r_t*c_t);
        bw_tie(u1(i):u2(i), u3(j):u4(j)) = im2bw(temp, t);
    end
end
% global Otsu and the count of pixels sitting exactly ON each block threshold (fixture non-degeneracy)
glevel_tank = graythresh(Itank); bw_global = im2bw(Itank, glevel_tank);
IC_global = sum(bw_global(:))/numel(bw_global);
at_th = zeros(1, n_r*n_c);
for i = 1 : n_r
    for j = 1 : n_c
        temp = double(Itank(t1(i):t2(i), t3(j):t4(j)));
        at_th((i-1)*n_c+j) = sum(temp(:) == thresh_tank((i-1)*n_c+j));
    end
end
"""
    return run_ref(code, ["thresh_tank", "IC_tank", "r_tank", "c_tank", "bw_tank", "Itank",
                          "thresh_tie", "n_ge_tie", "n_gt_tie", "IC_tie_ge", "bw_tie",
                          "glevel_tank", "IC_global", "at_th"],
                   REF / "block.mat", workdir=SCRATCH, timeout=1800)


def ref_rect():
    code = r"""
S = load('ch09_inputs.mat');
bwA = logical(S.rect_mask_a); bwB = logical(S.rect_mask_b); bwD = logical(S.rect_mask_degenerate);
[labA, numA] = bwlabel(bwA, 4); [labB, numB] = bwlabel(bwB, 4);
labA = double(labA); labB = double(labB);
RA = rect(bwA); nA = numel(RA);
VA = zeros(5,2,nA); CA = zeros(nA,2); AA = zeros(nA,1); PA = zeros(nA,1);
for i = 1:nA, VA(:,:,i)=RA(i).Vertices; CA(i,:)=RA(i).Center; AA(i)=RA(i).Area; PA(i)=RA(i).Perimeter; end
RB = rect(bwB); nB = numel(RB);
VB = zeros(5,2,nB); CB = zeros(nB,2); AB = zeros(nB,1); PB = zeros(nB,1);
for i = 1:nB, VB(:,:,i)=RB(i).Vertices; CB(i,:)=RB(i).Center; AB(i)=RB(i).Area; PB(i)=RB(i).Perimeter; end
% the SWAPPED call for EVERY component, to prove the (c, r) argument order of rect.m line 52 is load-bearing
VAswap = zeros(5,2,nA); AAswap = zeros(nA,1);
for i = 1:nA
    [rr, cc] = find(labA == i);
    [sx, sy, sa, sp] = minboundrect(rr, cc, 'a');    % deliberately (r, c) -- the WRONG order
    VAswap(:,:,i) = [sx, sy]; AAswap(i) = sa;
end
% MATLAB REFUSES a collinear point cloud: convhull errors, so rect.m cannot run on a 1-px blob or a 1-px line
deg_id = ''; deg_msg = ''; deg_ok = false;
try
    RD = rect(bwD); deg_ok = true;
catch errd
    deg_id = errd.identifier; deg_msg = errd.message;
end
deg_single_id = ''; deg_line_id = '';
try, minboundrect(4, 3, 'a'); catch e1, deg_single_id = e1.identifier; end
try, minboundrect((6:12)', repmat(8,7,1), 'a'); catch e2, deg_line_id = e2.identifier; end
"""
    return run_ref(code, ["labA", "numA", "labB", "numB", "VA", "CA", "AA", "PA", "nA",
                          "VB", "CB", "AB", "PB", "nB", "VAswap", "AAswap",
                          "deg_id", "deg_msg", "deg_ok", "deg_single_id", "deg_line_id"],
                   REF / "rect.mat", workdir=SCRATCH, timeout=900)


def ref_model():
    code = r"""
S = load('ch09_inputs.mat');
V = S.model_V; shp = S.model_shape;
Sarr = [];
for i = 1 : size(V,3)
    v = V(:,:,i);
    s0 = struct('Vertices', v, 'Center', [sum(v(1:4,1))/4, sum(v(1:4,2))/4], ...
                'Area', polyarea(v(1:4,1), v(1:4,2)), ...
                'Perimeter', sum(sqrt(sum(diff(v).^2, 2))));
    Sarr = [Sarr; s0];
end
img = zeros(shp(1), shp(2));
sm = model_ice_model(Sarr, img, 0.4, 2.5);
nM = numel(sm);
MV = zeros(5,2,nM); MC = zeros(nM,2); MA = zeros(nM,1); MP = zeros(nM,1); MI = false(nM,nM);
for i = 1:nM
    MV(:,:,i)=sm(i).Vertices; MC(i,:)=sm(i).Center; MA(i)=sm(i).Area; MP(i)=sm(i).Perimeter;
    idx = sm(i).Intersection;
    if ~isempty(idx), MI(i, idx) = true; end
end
% the rasterised union `bw`, rebuilt with the file's own two expressions (lines 32-36) so it can be compared
bw = zeros(shp(1), shp(2)); kk = zeros(1, size(V,3)); acc = [];
for i = 1 : size(V,3)
    v = V(:,:,i);
    k = (sqrt((v(1,1,1)-v(2,1,1))^2+(v(1,2,1)-v(2,2,1))^2))/ ...
        (sqrt((v(3,1,1)-v(2,1,1))^2+(v(3,2,1)-v(2,2,1))^2));
    kk(i) = k;
    if k < 2.5 && k > 0.4
        bb = roipoly(double(bw), v(:, 1), v(:, 2));
        bw(bb == 1) = 1;
        acc = [acc, i];
    end
end
bw_model = logical(bw); accepted = acc; ratios = kk;
% ---- the k1 < k < k2 band with k EXACTLY on both boundaries ----------------------------------------
RV = S.ratio_V; rshp = S.ratio_shape;
Rarr = [];
for i = 1 : size(RV,3)
    v = RV(:,:,i);
    Rarr = [Rarr; struct('Vertices', v, 'Center', [sum(v(1:4,1))/4, sum(v(1:4,2))/4], ...
                         'Area', polyarea(v(1:4,1), v(1:4,2)), 'Perimeter', 0)];
end
rm = model_ice_model(Rarr, zeros(rshp(1), rshp(2)), 0.4, 2.5);
ratio_nM = numel(rm);
ratio_kept = zeros(ratio_nM, 1);
for i = 1:ratio_nM
    for j = 1 : size(RV,3)
        if isequal(rm(i).Vertices, RV(:,:,j)), ratio_kept(i) = j; end
    end
end
ratio_k = zeros(1, size(RV,3));
for i = 1 : size(RV,3)
    v = RV(:,:,i);
    ratio_k(i) = (sqrt((v(1,1)-v(2,1))^2+(v(1,2)-v(2,2))^2))/(sqrt((v(3,1)-v(2,1))^2+(v(3,2)-v(2,2))^2));
end
"""
    return run_ref(code, ["MV", "MC", "MA", "MP", "MI", "nM", "bw_model", "accepted", "ratios",
                          "ratio_nM", "ratio_kept", "ratio_k"],
                   REF / "model.mat", workdir=SCRATCH, timeout=900)


def ref_polybool():
    code = r"""
S = load('ch09_inputs.mat');
A = S.pb_A; B = S.pb_B; n = size(A,3);
pb_empty = false(1,n); pb_nv = zeros(1,n); pb_taken = false(1,n); pb_area = zeros(1,n);
PX = NaN(24, n); PY = NaN(24, n);
for i = 1 : n
    [xx, yy] = polybool('intersection', A(:,1,i), A(:,2,i), B(:,1,i), B(:,2,i));
    pb_empty(i) = isempty(xx);
    pb_nv(i) = numel(xx);
    if xx ~= NaN, pb_taken(i) = true; end            % the literal `if xx ~= NaN` of model_ice_model.m:54
    if numel(xx) >= 3, pb_area(i) = polyarea(xx, yy); end
    if ~isempty(xx), PX(1:numel(xx), i) = xx(:); PY(1:numel(yy), i) = yy(:); end
end
"""
    return run_ref(code, ["pb_empty", "pb_nv", "pb_taken", "pb_area", "PX", "PY"],
                   REF / "polybool.mat", workdir=SCRATCH, timeout=900)


def ref_r13():
    """Risk **R13** — what MATLAB R2025a really does at ``movie_floe.m`` line 25 on a blank frame.

    The answer depends on the *form* of the right-hand side, which is why this needs its own probe:
    ``x(k) = []`` with a **literal** ``[]`` is parsed as a deletion, while ``x(k) = max(ice_areas)`` (a function
    call that returns ``0x0``) is an **assignment**.  Neither deletes-and-shifts past the end; both raise."""
    code = r"""
ids = cell(1,4); msgs = cell(1,4);
for q = 1:4, ids{q} = ''; msgs{q} = ''; end
try, fl = []; fl(1) = 5; fl(2) = [];              catch e, ids{1}=e.identifier; msgs{1}=e.message; end
try, fl = []; fl(1) = 5; z = []; fl(2) = z;       catch e, ids{2}=e.identifier; msgs{2}=e.message; end
try, floe = []; floe(1) = 5; ice_areas = []; floe(2) = max(ice_areas);
                                                  catch e, ids{3}=e.identifier; msgs{3}=e.message; end
try, v = [1 2 3]; v(2) = []; msgs{4} = mat2str(v); catch e, ids{4}=e.identifier; msgs{4}=e.message; end
id1=ids{1}; id2=ids{2}; id3=ids{3}; id4=ids{4};
m1=msgs{1}; m2=msgs{2}; m3=msgs{3}; m4=msgs{4};
maxempty = mat2str(size(max([])));
"""
    return run_ref(code, ["id1", "id2", "id3", "id4", "m1", "m2", "m3", "m4", "maxempty"],
                   REF / "r13.mat", workdir=SCRATCH, timeout=600)


def ref_movie_otsu():
    return run_ref("run('ch09_movie_otsu_ref.m'); done_otsu = 1;", ["done_otsu"],
                   REF / "movie_otsu_done.mat", workdir=SCRATCH, timeout=3600)


def ref_movie_kmeans():
    return run_ref("rng(0); run('ch09_movie_kmeans_ref.m'); done_kmeans = 1;", ["done_kmeans"],
                   REF / "movie_kmeans_done.mat", workdir=SCRATCH, timeout=3600)


def ref_movie_floe():
    return run_ref("run('ch09_movie_floe_ref.m'); done_floe = 1;", ["done_floe"],
                   REF / "movie_floe_done.mat", workdir=SCRATCH, timeout=3600)


def ref_demo():
    return run_ref("run('ch09_model_ice_demo_ref.m'); done_demo = 1;", ["done_demo"],
                   REF / "demo_done.mat", workdir=SCRATCH, timeout=5400)


SESSIONS = {
    "video": ref_video,
    "probes": ref_probes,
    "block": ref_block,
    "rect": ref_rect,
    "model": ref_model,
    "polybool": ref_polybool,
    "r13": ref_r13,
    "movie_otsu": ref_movie_otsu,
    "movie_kmeans": ref_movie_kmeans,
    "movie_floe": ref_movie_floe,
    "demo": ref_demo,
}

#: sessions whose `.m` writes its own `.mat` into the scratch cwd; it is moved into reference/ch09 afterwards
SCRATCH_MAT = {"block": "ch09_block.mat", "movie_otsu": "ch09_movie_otsu.mat",
               "movie_kmeans": "ch09_movie_kmeans.mat", "movie_floe": "ch09_movie_floe.mat",
               "demo": "ch09_demo.mat"}


def main(argv: list[str]) -> int:
    names = argv[1:] or list(SESSIONS)
    VERIFY.mkdir(parents=True, exist_ok=True)
    prepare_scratch()
    write_inputs()
    log_path = VERIFY / "refs_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    for name in names:
        if name not in SESSIONS:
            print(f"unknown session {name!r}; known: {', '.join(SESSIONS)}")
            return 2
        print(f"=== {name} ...", flush=True)
        t0 = time.time()
        try:
            res = SESSIONS[name]()
            dt = time.time() - t0
            extra = SCRATCH_MAT.get(name)
            if extra:
                src = SCRATCH / extra
                if not src.exists():
                    raise RuntimeError(f"{name}: MATLAB did not write {extra}")
                shutil.copyfile(src, REF / extra)
            log[name] = {"status": "ok", "seconds": round(dt, 1), "engine": res.engine,
                         "version": res.version, "mat": str(Path(res.out_mat).name),
                         "extra_mat": extra, "note": res.stdout[-400:] if res.stdout else ""}
            print(f"    ok in {dt:.1f}s  ({res.engine} {res.version})")
        except Exception as exc:                                   # noqa: BLE001
            dt = time.time() - t0
            log[name] = {"status": "fail", "seconds": round(dt, 1), "error": str(exc)[:4000]}
            print(f"    FAILED after {dt:.1f}s: {str(exc)[:2000]}")
        log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
    # provenance of the inputs both engines read
    prov = {p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(SCRATCH.glob("*")) if p.suffix in (".jpg", ".avi", ".mat") and p.is_file()}
    (VERIFY / "input_md5.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
