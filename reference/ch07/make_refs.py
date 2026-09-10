"""Generate the MATLAB (R2025a) reference outputs for chapter 7 (sea ice type identification).

Runs the ORIGINAL ``.m`` files of ``MATLAB_ROOT/ch7/`` through ``tools.run_matlab_ref.run_ref``
(``matlab -batch``, figures invisible, ``save(..., '-v7')``) and writes ``reference/ch07/*.mat``.

Usage::

    .venv/Scripts/python.exe reference/ch07/make_refs.py [name ...]
    names = clf, fig767, imfill, hist, misc, kmeans, iceenh, demo

Scratch policy (CLAUDE.md rule 8: nothing under ``MATLAB_ROOT`` is touched)
--------------------------------------------------------------------------
All ``.m`` files are **copied** into ``outputs/ch07/verify/scratch/`` and MATLAB runs with that folder as cwd;
the ch7 folders are never put on the MATLAB path.  The copies are verbatim except for the patches listed in
``PATCHES`` below, which are reproduced in ``reports/ch07_verification.md``:

* the four ``cleaning & labeling & filling`` scripts lose only their leading ``clear all, clc;`` (so that several
  of them can run in one session without wiping the accumulated variables) and are renamed
  ``ch07_*_ref.m`` (a file called ``filling.m`` or ``labeling.m`` is a generic name that could shadow);
  their SE-variant copies additionally swap the shipped ``se = strel(...)`` line for the **commented** variant
  that the book's Figs. 7.4 / 7.7 use — no other literal is changed.
* ``ice_shape_enhancement.m`` → ``ice_shape_enhancement_ref.m``: the function is renamed, its output list is
  extended with the intermediates, and the **figure** statements of lines 183–237 are removed because
  lines 214–216/222/224 are HG1 code that errors in R2025a.  Every numeric expression is verbatim.
* ``sea_ice_demo.m`` → ``ch7_demo_ref.m``: ``rng(0)`` before the k-means stage (reproducibility), the call to
  ``ice_shape_enhancement`` retargeted at the ref copy, and line 57 ``sea_ice_model`` dropped (book §8.2, ch8).
"""
from __future__ import annotations

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
from reference.ch07.fixtures import hist_cases, imfill_cases, seg_fixtures  # noqa: E402

MATLAB_ROOT = ROOT / "K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab"
CH7 = MATLAB_ROOT / "ch7"
CLF = CH7 / "cleaning & labeling & filling"
SIFI = CH7 / "Sea_Ice_Floe_Identification"
CH6_SIFI = MATLAB_ROOT / "ch6/Sea_Ice_Floe_Identification"
REF = ROOT / "reference/ch07"
VERIFY = ROOT / "outputs/ch07/verify"
SCRATCH = VERIFY / "scratch"
ML_IMG = VERIFY / "matlab"

PATCHES: list[tuple[str, str]] = []


def q(p) -> str:
    return "'" + str(p).replace("\\", "/").replace("'", "''") + "'"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _replace(text: str, old: str, new: str, what: str) -> str:
    if old not in text:
        raise RuntimeError(f"patch token not found ({what}): {old!r}")
    return text.replace(old, new, 1)


# ==============================================================================================================
# Scratch preparation
# ==============================================================================================================
def prepare_scratch() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    ML_IMG.mkdir(parents=True, exist_ok=True)
    REF.mkdir(parents=True, exist_ok=True)
    PATCHES.clear()

    # ---- the four `cleaning & labeling & filling` scripts + their SE variants --------------------------------
    lead = "clear all, clc;\n"

    def strip_lead(name: str) -> str:
        t = _read(CLF / name)
        return _replace(t, lead, "", f"{name}: leading clear all")

    cleaning = strip_lead("morphology_cleaning.m")
    (SCRATCH / "ch07_cleaning_ref.m").write_text(cleaning, encoding="utf-8")
    PATCHES.append(("morphology_cleaning.m -> ch07_cleaning_ref.m", "removed the leading `clear all, clc;`"))

    labeling = strip_lead("labeling.m")
    (SCRATCH / "ch07_labeling_sq3_ref.m").write_text(labeling, encoding="utf-8")
    PATCHES.append(("labeling.m -> ch07_labeling_sq3_ref.m", "removed the leading `clear all, clc;`"))
    lab_dm = _replace(labeling, "se = strel('square', 3);\n% se = strel('diamond', 1);",
                      "% se = strel('square', 3);\nse = strel('diamond', 1);", "labeling SE swap")
    (SCRATCH / "ch07_labeling_dm1_ref.m").write_text(lab_dm, encoding="utf-8")
    PATCHES.append(("labeling.m -> ch07_labeling_dm1_ref.m (Fig. 7.4)",
                    "removed `clear all, clc;`; un-commented the shipped `se = strel('diamond', 1);` line and "
                    "commented `se = strel('square', 3);` (the file's own alternative, book Fig. 7.4)"))

    filling = strip_lead("filling.m")
    (SCRATCH / "ch07_filling_ref.m").write_text(filling, encoding="utf-8")
    PATCHES.append(("filling.m -> ch07_filling_ref.m", "removed the leading `clear all, clc;`"))

    fillrec = strip_lead("filling_reconstruct.m")
    (SCRATCH / "ch07_fillrec_dm1_ref.m").write_text(fillrec, encoding="utf-8")
    PATCHES.append(("filling_reconstruct.m -> ch07_fillrec_dm1_ref.m", "removed the leading `clear all, clc;`"))
    fr_sq = _replace(fillrec, "se = strel('diamond', 1);\n% se = strel('square', 3);",
                     "% se = strel('diamond', 1);\nse = strel('square', 3);", "fillrec SE swap")
    (SCRATCH / "ch07_fillrec_sq3_ref.m").write_text(fr_sq, encoding="utf-8")
    PATCHES.append(("filling_reconstruct.m -> ch07_fillrec_sq3_ref.m",
                    "removed `clear all, clc;`; swapped in the file's own commented `se = strel('square', 3);`"))

    # the file's commented alternative I0 (lines 13-21)
    lines = fillrec.splitlines(keepends=True)
    i_ship = next(i for i, s in enumerate(lines) if s.startswith("I0 ="))
    ship_block = "".join(lines[i_ship:i_ship + 9])
    alt_block = "".join(lines[i_ship + 10:i_ship + 19])
    if not alt_block.strip().startswith("%  I0"):
        raise RuntimeError("could not locate the commented alternative I0 block")
    alt_uncommented = "\n".join(s.lstrip().lstrip("%").lstrip() if s.strip().startswith("%") else s
                                for s in alt_block.splitlines())
    alt_uncommented = alt_uncommented.replace("I0 =[", "I0 =[")  # cosmetic no-op, keeps the literal identical
    fr_alt = fillrec.replace(ship_block, alt_uncommented + "\n", 1)
    (SCRATCH / "ch07_fillrec_alt_ref.m").write_text(fr_alt, encoding="utf-8")
    PATCHES.append(("filling_reconstruct.m -> ch07_fillrec_alt_ref.m",
                    "removed `clear all, clc;`; replaced the active `I0` literal with the file's own **commented** "
                    "alternative `I0` (lines 13-21), character for character"))

    # ---- Figs. 7.6 / 7.7: filling.m's image + one pixel at 1-based (5,5); the book prints no .m -------------
    fig76 = _replace(filling, "se = strel('diamond', 1);",
                     "I0(5, 5) = 1;   % Fig. 7.6/7.7: the book's variant image (one extra object pixel)\n"
                     "se = strel('diamond', 1);", "fig76 pixel")
    (SCRATCH / "ch07_filling76_ref.m").write_text(fig76, encoding="utf-8")
    fig77 = _replace(fig76, "se = strel('diamond', 1);", "se = strel('square', 3);", "fig77 SE")
    (SCRATCH / "ch07_filling77_ref.m").write_text(fig77, encoding="utf-8")
    PATCHES.append(("filling.m -> ch07_filling76_ref.m / ch07_filling77_ref.m (book Figs. 7.6 / 7.7)",
                    "removed `clear all, clc;`; added `I0(5, 5) = 1;` (the book's variant image, which no shipped "
                    ".m contains) and, for Fig. 7.7, `strel('square', 3)` instead of `strel('diamond', 1)`"))

    # ---- Sea_Ice_Floe_Identification: verbatim copies + the patched ice_shape_enhancement -------------------
    for m in sorted(SIFI.glob("*.m")):
        shutil.copyfile(m, SCRATCH / m.name)
    shutil.copyfile(SIFI / "sea_ice_test.jpg", SCRATCH / "sea_ice_test.jpg")
    shutil.copyfile(CH6_SIFI / "sea_ice_test.jpg", SCRATCH / "sea_ice_test_ch6.jpg")

    (SCRATCH / "ice_shape_enhancement_ref.m").write_text(_patched_enhancement(), encoding="utf-8")
    (SCRATCH / "ch7_demo_ref.m").write_text(_patched_demo(), encoding="utf-8")

    (VERIFY / "patches.json").write_text(json.dumps(PATCHES, indent=1), encoding="utf-8")


ENH_HEADER_OLD = """function [out, index_floe, ice_floe, index_brash, brash_ice, index_slush, ...
    index_water, index_residue, coverage] = ice_shape_enhancement(bk, seg, min_floe, min_brash, se_th)"""

ENH_HEADER_NEW = """function [out, index_floe, ice_floe, index_brash, brash_ice, index_slush, ...
    index_water, index_residue, coverage, l, ice_area, A, ind, fill, t, ...
    floe_area, brash_area, color_floe, color_brash, floe_cen, brash_cen, index, ...
    nn_bw, nn_k, ysh, YT, z, n, color, zs, izs, ysh2, YT2] = ...
    ice_shape_enhancement_ref(bk, seg, min_floe, min_brash, se_th)"""

#: Everything from line 182 of the original, with the *graphics* statements removed and the second `ysh`/`YT`
#: renamed `ysh2`/`YT2` so both survive.  Every numeric expression is character-for-character the original's.
ENH_TAIL = """%% rgb  (label2rgb kept, the figure/plot/axis statements removed - display only)
ysh = []; YT = []; z = []; n = []; color = []; zs = []; izs = []; ysh2 = []; YT2 = [];

area_ice = [color_floe, color_brash];
if ~isempty(area_ice)
n = 6;
d = fix((max(area_ice)-min(area_ice))/n);
ysh = min(area_ice) : d : max(area_ice);
YT = [];
for i = 1 : length(ysh)
    YT{1, i} = -round(1000 * log(1 - ysh(i)/10000));
end
end

%% histogram  (the bar()/HG1 FaceVertexCData statements of lines 213-224 removed - they error in R2025a)
if ~isempty(floe_area)
nbins = 50;
[z, n] = hist(floe_area, nbins);
[zs, izs] = sortrows(z', 1);
k = 255;
for i = 1 : nbins
    color(i) = fix( (1 - exp(-n(i)/1000)) * 10000 );
end

nn = 8;
d = fix((max(color_floe)-min(color_floe))/nn);
ysh2 = min(color_floe) : d : max(color_floe);
YT2 = [];
for i = 1 : length(ysh2)
    YT2{1, i} = -round(1000 * log(1 - ysh2(i)/10000));
end
end

end
"""


def _patched_enhancement() -> str:
    text = _read(SIFI / "ice_shape_enhancement.m")
    text = _replace(text, ENH_HEADER_OLD, ENH_HEADER_NEW, "enhancement header")
    cut = text.index("%% rgb")
    body = text[:cut]
    PATCHES.append((
        "ice_shape_enhancement.m -> ice_shape_enhancement_ref.m",
        "function renamed and the output list extended with the intermediates "
        "(l, ice_area, A, ind, fill, t, floe_area, brash_area, color_floe, color_brash, floe_cen, brash_cen, "
        "index, nn_bw, nn_k, ysh, YT, z, n, color, zs, izs, ysh2, YT2); lines 39-181 verbatim; "
        "lines 182-237 replaced by the same arithmetic with the graphics statements removed "
        "(`figure`, `imshow`, `hold on`, the two `plot` loops, `axis off`, `colormap`, `colorbar`, `get/set` on "
        "the colorbar, `h = bar(...)`, and the HG1 block `ch = get(h,'Children'); fvd = get(ch,'Faces'); "
        "fvcd = get(ch,'FaceVertexCData'); fvcd(fvd(i,:)) = color(i); set(ch,'FaceVertexCData',fvcd)` which "
        "ERRORS in R2025a) and the second `ysh`/`YT` renamed `ysh2`/`YT2`; guarded by `if ~isempty(...)` so an "
        "ice-free fixture returns empties instead of erroring.  No numeric literal or expression changed."))
    return body + ENH_TAIL


def _patched_demo() -> str:
    text = _read(SIFI / "sea_ice_demo.m")
    text = _replace(text, "[seg, bk] = seaice_kmean_GVF_forenhancement(",
                    "rng(0);\n[seg, bk] = seaice_kmean_GVF_forenhancement(", "demo rng")
    old_call = """[out, index_floe, ice_floe, index_brash, brash_ice, index_slush, index_water, index_residue, coverage] = ...
    ice_shape_enhancement(bk, seg, min_floe, min_brash, se_th);"""
    new_call = """[out, index_floe, ice_floe, index_brash, brash_ice, index_slush, index_water, index_residue, ...
    coverage, l, ice_area, A, ind, fill, t, floe_area, brash_area, color_floe, color_brash, floe_cen, ...
    brash_cen, index, nn_bw, nn_k, ysh, YT, z, n, color, zs, izs, ysh2, YT2] = ...
    ice_shape_enhancement_ref(bk, seg, min_floe, min_brash, se_th);"""
    text = _replace(text, old_call, new_call, "demo enhancement call")
    text = _replace(text, "[floe, brash] = sea_ice_model( ice_floe, brash_ice, index_floe );",
                    "% [floe, brash] = sea_ice_model(...);  % book section 8.2 -> ch08", "demo sea_ice_model")
    PATCHES.append(("sea_ice_demo.m -> ch7_demo_ref.m",
                    "`rng(0);` inserted before the k-means stage (reproducibility, as in ch06); the "
                    "`ice_shape_enhancement` call retargeted at `ice_shape_enhancement_ref` with the extended "
                    "output list; line 57 `sea_ice_model` commented out (book section 8.2, ported in ch08). "
                    "Every parameter literal of lines 9-46 is verbatim."))
    return text


# ==============================================================================================================
# Reference runs
# ==============================================================================================================
def ref_clf() -> None:
    """The four `cleaning & labeling & filling` scripts + the SE / I0 variants, one MATLAB session."""
    blocks = [
        ("cl", "ch07_cleaning_ref", ["I", "f1", "f2", "f0", "c1", "c2", "c3", "c4"]),
        ("lb3", "ch07_labeling_sq3_ref", ["I", "x0", "xx"] + [f"x{i}" for i in range(1, 10)]),
        ("lb1", "ch07_labeling_dm1_ref", ["I", "x0", "xx"] + [f"x{i}" for i in range(1, 10)]),
        ("fi", "ch07_filling_ref", ["I0", "I", "x0", "xx"] + [f"x{i}" for i in range(1, 10)] + ["I1"]),
        ("fr1", "ch07_fillrec_dm1_ref",
         ["I0", "I", "x0", "xx"] + [f"x{i}" for i in range(1, 11)] + ["I1", "I2"]),
        ("fr3", "ch07_fillrec_sq3_ref",
         ["I0", "I", "x0", "xx"] + [f"x{i}" for i in range(1, 11)] + ["I1", "I2"]),
        ("fra", "ch07_fillrec_alt_ref",
         ["I0", "I", "x0", "xx"] + [f"x{i}" for i in range(1, 11)] + ["I1", "I2"]),
    ]
    code, save_vars = [], []
    for pre, script, vs in blocks:
        code.append(f"run('{script}.m');")
        for v in vs:
            code.append(f"{pre}_{v} = {v};")
            save_vars.append(f"{pre}_{v}")
        code.append("clearvars " + " ".join(dict.fromkeys(vs)) + " se;")
    # the SE neighbourhoods the scripts use
    code += ["se_sq2 = getnhood(strel('square', 2));", "se_sq3 = getnhood(strel('square', 3));",
             "se_dm1 = getnhood(strel('diamond', 1));", "se_d1 = getnhood(strel('disk', 1));",
             "se_d2 = getnhood(strel('disk', 2));", "se_d3 = getnhood(strel('disk', 3));"]
    save_vars += ["se_sq2", "se_sq3", "se_dm1", "se_d1", "se_d2", "se_d3"]
    _run("\n".join(code), save_vars, "clf.mat")


def ref_fig767() -> None:
    """Figs. 7.6 / 7.7 — the book's variant image, cross SE and 3x3 square SE (adjudicates the X8 typo)."""
    code, save_vars = [], []
    for pre, script in (("f76", "ch07_filling76_ref"), ("f77", "ch07_filling77_ref")):
        vs = ["I0", "I", "x0", "xx"] + [f"x{i}" for i in range(1, 10)] + ["I1"]
        code.append(f"run('{script}.m');")
        for v in vs:
            code.append(f"{pre}_{v} = {v};")
            save_vars.append(f"{pre}_{v}")
        code.append("clearvars " + " ".join(vs) + " se;")
    _run("\n".join(code), save_vars, "fig767.mat")


def ref_imfill() -> None:
    cases = imfill_cases()
    savemat(str(REF / "imfill_inputs.mat"), {f"in_{k}": v for k, v in cases.items()}, do_compression=False)
    code = [f"S = load({q(REF / 'imfill_inputs.mat')});"]
    save_vars = []
    for k in cases:
        for c in (4, 8):
            code.append(f"out_{k}_{c} = imfill(S.in_{k}, {c}, 'hole');")
            save_vars.append(f"out_{k}_{c}")
        code.append(f"cls_{k} = class(imfill(S.in_{k}, 'hole'));")
        save_vars.append(f"cls_{k}")
    # `imfill(BW, 'holes')` vs the book's prefix `'hole'` vs the default connectivity
    code += ["pref_full = imfill(S.in_double_hole, 'holes');",
             "pref_h    = imfill(S.in_double_hole, 'h');",
             "def_conn  = imfill(S.in_diagonal_hole, 'holes');",
             "conndef_min = conndef(2, 'minimal');"]
    save_vars += ["pref_full", "pref_h", "def_conn", "conndef_min"]
    _run("\n".join(code), save_vars, "imfill.mat")


def ref_hist() -> None:
    cases = hist_cases()
    payload = {}
    for k, (y, b) in cases.items():
        payload[f"y_{k}"] = np.asarray(y, dtype=np.float64).reshape(1, -1)
        payload[f"b_{k}"] = np.asarray(b, dtype=np.float64).reshape(1, -1) if np.ndim(b) else float(b)
    savemat(str(REF / "hist_inputs.mat"), payload, do_compression=False)
    code = [f"S = load({q(REF / 'hist_inputs.mat')});"]
    save_vars = []
    for k in cases:
        code.append(f"[z_{k}, n_{k}] = hist(S.y_{k}, S.b_{k});")
        save_vars += [f"z_{k}", f"n_{k}"]
    # the empty-input branch (an empty array cannot round-trip through savemat reliably)
    code += ["[z_empty, n_empty] = hist([], 5);", "[z_empty_c, n_empty_c] = hist([], [1 2 3]);"]
    save_vars += ["z_empty", "n_empty", "z_empty_c", "n_empty_c"]
    _run("\n".join(code), save_vars, "hist.mat")


def ref_misc() -> None:
    """The `sea_ice_test.jpg` pre-check (ch7 vs ch6), MATLAB `sort` tie order, `strel('disk', r)`."""
    ties = np.array([[4, 4, 6, 4, 1, 6, 1, 9, 4, 6, 1, 2]], dtype=np.float64)
    savemat(str(REF / "misc_inputs.mat"), {"ties": ties})
    code = f"""
S = load({q(REF / 'misc_inputs.mat')});
[ties_A, ties_ind] = sort(S.ties);

I7 = imread('sea_ice_test.jpg');
I6 = imread('sea_ice_test_ch6.jpg');
g7 = rgb2gray(I7);  g6 = rgb2gray(I6);
lev7 = graythresh(g7);  lev6 = graythresh(g6);
bw7 = im2bw(g7, lev7);  bw6 = im2bw(g6, lev6);
[L7, n7] = bwlabel(bw7, 4);
[L6, n6] = bwlabel(bw6, 4);
[L7e, n7e] = bwlabel(bw7, 8);
sz7 = size(I7);
diff_jpeg = sum(sum(sum(abs(double(I7) - double(I6)))));
frac_diff = nnz(I7 ~= I6) / numel(I7);
maxabs = max(max(max(abs(double(I7) - double(I6)))));

d1 = getnhood(strel('disk', 1));  d2 = getnhood(strel('disk', 2));
"""
    save_vars = ["ties_A", "ties_ind", "g7", "lev7", "lev6", "bw7", "n7", "n6", "n7e", "sz7",
                 "diff_jpeg", "frac_diff", "maxabs", "d1", "d2"]
    _run(code, save_vars, "misc.mat")


def ref_iceenh() -> None:
    """`ice_shape_enhancement_ref.m` on the controlled (bk, seg) fixtures, one .mat per fixture."""
    fixtures = seg_fixtures()
    payload = {}
    for k, (bk, seg) in fixtures.items():
        payload[f"bk_{k}"] = np.asarray(bk, dtype=np.float64)
        payload[f"seg_{k}"] = np.asarray(seg, dtype=np.float64)
    savemat(str(REF / "seg_inputs.mat"), payload, do_compression=False)

    outs = ("out, index_floe, ice_floe, index_brash, brash_ice, index_slush, index_water, index_residue, "
            "coverage, l, ice_area, A, ind, fill, t, floe_area, brash_area, color_floe, color_brash, "
            "floe_cen, brash_cen, index, nn_bw, nn_k, ysh, YT, z, n, color, zs, izs, ysh2, YT2")
    names = [s.strip() for s in outs.split(",")]
    code = [f"S = load({q(REF / 'seg_inputs.mat')});"]
    for k in fixtures:
        # the book's `>=` form of Algorithm 5 is obtained by shifting the thresholds by one pixel, which is
        # exactly what `area0 > min_floe - 1` means for integer areas.
        for tag, mf, mb in (("code", "40", "1"), ("book", "39", "0")):
            code.append(f"[{outs}] = ice_shape_enhancement_ref(S.bk_{k}, S.seg_{k}, {mf}, {mb}, 50);")
            code.append(f"save({q(REF / f'iceenh_{k}_{tag}.mat')}, " +
                        ", ".join(f"'{n}'" for n in names) + ", '-v7');")
            code.append("clearvars " + " ".join(names) + ";")
    code.append("done = 1;")
    _run("\n".join(code), ["done"], "iceenh_done.mat")


def ref_kmeans() -> None:
    """Algorithm 3's k-means stage alone on ch7's own JPEG (R9): sorted centres and the `bk` mask.

    Lines 189-205 of ``seaice_kmean_GVF_forenhancement.m``, verbatim, with ``rng(0)`` in front.
    """
    code = """
rng(0);
I = rgb2gray(imread('sea_ice_test.jpg'));
ima = double(I(:));
map0 = kmeans(ima, 3, 'EmptyAction', 'singleton');
s0 = zeros(1,3);
for i = 1 : 3
  pp = zeros(size(map0)); pp(map0 == i) = 1;
  s0(i) = sum(ima.*pp)/sum(pp);
end
[A0, ind0] = sort(s0);
bw_kmeans = ones(size(map0));
p = find(map0 == ind0(1));
bw_kmeans(p) = 0;
bk_ml = double(reshape(bw_kmeans, size(I,1), size(I,2)));
centres = A0;
"""
    _run(code, ["centres", "bk_ml", "A0", "ind0"], "kmeans.mat")


def ref_demo() -> None:
    """The whole chapter driver on ch7's own `sea_ice_test.jpg` (Algorithm 3 -> 4 -> 5).  ~10-20 minutes."""
    names = ("seg, bk, out, index_floe, index_brash, index_slush, index_water, index_residue, coverage, "
             "ice_floe, brash_ice, "
             "l, ice_area, A, ind, fill, t, floe_area, brash_area, color_floe, color_brash, floe_cen, "
             "brash_cen, index, nn_bw, nn_k, ysh, YT, z, n, color, zs, izs, ysh2, YT2")
    save_vars = [s.strip() for s in names.split(",")]
    code = "run('ch7_demo_ref.m');\n" + \
           f"imwrite(label2rgb(index, @jet, [1,1,1]), {q(ML_IMG / 'demo_index_rgb.png')});\n" + \
           f"imwrite(logical(index_floe > 0), {q(ML_IMG / 'demo_index_floe.png')});\n" + \
           f"imwrite(logical(index_brash > 0), {q(ML_IMG / 'demo_index_brash.png')});\n" + \
           f"imwrite(logical(index_slush), {q(ML_IMG / 'demo_index_slush.png')});\n" + \
           f"imwrite(logical(index_water), {q(ML_IMG / 'demo_index_water.png')});\n" + \
           f"imwrite(logical(index_residue), {q(ML_IMG / 'demo_index_residue.png')});\n"
    _run(code, save_vars, "demo.mat", timeout=7200)


# ==============================================================================================================
def _run(code: str, save_vars, out_name: str, timeout: int = 1800) -> None:
    t0 = time.time()
    print(f"--- {out_name}: {len(save_vars)} variables ...", flush=True)
    (VERIFY / f"{Path(out_name).stem}_code.m").write_text(code, encoding="utf-8")
    res = run_ref(code, save_vars, REF / out_name, addpath=None, workdir=SCRATCH, timeout=timeout)
    dt = time.time() - t0
    print(f"    {out_name}: engine={res.engine} version={res.version} {dt:.1f} s", flush=True)
    log = VERIFY / "refs_log.json"
    entries = json.loads(log.read_text()) if log.exists() else {}
    entries[out_name] = {"engine": res.engine, "version": res.version, "seconds": round(dt, 1),
                         "status": "ok", "vars": list(save_vars), "stdout_tail": res.stdout[-800:]}
    log.write_text(json.dumps(entries, indent=1), encoding="utf-8")


TARGETS = {"clf": ref_clf, "fig767": ref_fig767, "imfill": ref_imfill, "hist": ref_hist,
           "misc": ref_misc, "kmeans": ref_kmeans, "iceenh": ref_iceenh, "demo": ref_demo}


def main(argv: list[str]) -> int:
    prepare_scratch()
    wanted = argv[1:] or list(TARGETS)
    for name in wanted:
        if name not in TARGETS:
            raise SystemExit(f"unknown target {name!r}; choose from {list(TARGETS)}")
        TARGETS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
