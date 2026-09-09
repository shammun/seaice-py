"""Generate the MATLAB (R2025a) reference outputs for chapter 2 (Book §2.1–§2.8).

Runs the ORIGINAL ``.m`` files of ``MATLAB_ROOT/ch2`` (+ ``chain code``) through ``tools.run_matlab_ref.run_ref``
(``matlab -batch``, figures invisible) and saves the workspace variables listed in ``analysis/ch02.md`` §6.1 as
``reference/ch02/*.mat`` (v7).  Every launch also writes the images MATLAB would *display* (``imwrite`` /
``print``) into ``outputs/ch02/verify/matlab/`` for the side-by-side figure comparisons (L3).

Usage:  .venv/Scripts/python.exe reference/ch02/make_refs.py [name ...]
        (no arguments = all references; names = histogram, color_image, distance_transform, chain_diff,
         boundaries_multi, conv2, interp2, compat)

Notes
-----
* ``histogram.m`` / ``color_image.m`` read ``rgb.jpg`` from the *current folder* and ``histogram.m`` writes
  ``red.png/green.png/blue.png`` there, so those scripts run with the working directory set to a scratch folder
  (``outputs/ch02/verify/scratch``) holding a copy of ``data/book/ch02/rgb.JPG``.  Nothing is ever written into
  MATLAB_ROOT.
* ``minmag`` (local function of ``fchcode.m``) is copied verbatim into the scratch folder as ``minmag_ref.m`` so
  its behaviour on periodic codes can be demonstrated directly.
"""
from __future__ import annotations

import json
import re
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
CH = MATLAB_ROOT / "ch2"
CHAIN = CH / "chain code"
REF = ROOT / "reference/ch02"
VERIFY = ROOT / "outputs/ch02/verify"
ML_IMG = VERIFY / "matlab"
SCRATCH = VERIFY / "scratch"
RGB = ROOT / "data/book/ch02/rgb.JPG"


def q(p: Path | str) -> str:
    """MATLAB single-quoted literal with forward slashes (MATLAB on Windows accepts them)."""
    return "'" + str(p).replace("\\", "/").replace("'", "''") + "'"


def _prepare() -> None:
    for d in (REF, ML_IMG, SCRATCH):
        d.mkdir(parents=True, exist_ok=True)
    if not (SCRATCH / "rgb.JPG").exists():
        shutil.copy(RGB, SCRATCH / "rgb.JPG")
    # verbatim copy of fchcode.m's local minmag -> callable function minmag_ref (evidence for the periodic-code case)
    src = (CHAIN / "fchcode.m").read_text(encoding="utf-8", errors="replace")
    m = re.search(r"function z = minmag\(c\)(.*?)(?=\n%-+%\nfunction d = codediff)", src, re.S)
    if m is None:
        raise RuntimeError("could not locate minmag in fchcode.m")
    (SCRATCH / "minmag_ref.m").write_text("function z = minmag_ref(c)" + m.group(1) + "\n", encoding="utf-8")
    # fixtures shared with the tests (identical arrays on both sides)
    savemat(REF / "inputs.mat", {
        "A12": synth.FIG_2_12_A.astype(np.uint8),
        "BW11": synth.FIG_2_11_COMPONENTS.astype(np.uint8),
        "Sp": synth.spur_shape().astype(np.uint8),
        "B19": synth.FIG_2_19_OBJECT.astype(np.uint8),
    })


def _imw(var: str, name: str, autoscale: bool = True) -> str:
    """MATLAB statement writing `var` as imshow would display it (imshow(x) or imshow(x, []))."""
    p = q(ML_IMG / name)
    if autoscale:
        return f"tmp__ = mat2gray(double({var})); tmp__(isnan(tmp__)) = 0; imwrite(tmp__, {p});"
    return f"imwrite({var}, {p});"


# ------------------------------------------------------------------------------------------------------------
def ref_histogram():
    code = f"""
run('histogram.m');
G = I; I = imread('rgb.jpg'); [m, n] = size(G);
cnt = imhist(G);
imwrite(Ir, [[0:1/255:1]', zeros(256,1), zeros(256,1)], {q(ML_IMG / 'fig_2_03_red.png')});
imwrite(Ig, [zeros(256,1), [0:1/255:1]', zeros(256,1)], {q(ML_IMG / 'fig_2_03_green.png')});
imwrite(Ib, [zeros(256,1), zeros(256,1), [0:1/255:1]'], {q(ML_IMG / 'fig_2_03_blue.png')});
imwrite(G, {q(ML_IMG / 'fig_2_07_gray.png')});
f7 = figure('Visible','off'); imhist(G); xlabel('Intensity value'); ylabel('Number of pixels');
print(f7, '-dpng', '-r100', {q(ML_IMG / 'fig_2_07_imhist.png')});
f8 = figure('Visible','off'); plot(x, y_r, 'r', x, y_g, '--g', x, y_b, ':b', 'LineWidth', 2.5);
legend('Red', 'Green', 'Blue', 'Location', 'best'); legend boxoff; xlim([0 255]); ylim([0 80000]);
xlabel('Intensity value'); ylabel('Number of pixels');
print(f8, '-dpng', '-r100', {q(ML_IMG / 'fig_2_08_rgb_hist.png')});
[gmax, gpk] = max(cnt); [rmax, rpk] = max(y_r); [gmax2, gpk2] = max(y_g); [bmax, bpk] = max(y_b);
peaks = [gmax gpk-1; rmax rpk-1; gmax2 gpk2-1; bmax bpk-1];
pix = double(squeeze(I(1076, 675, :)))';
"""
    return run_ref(code, ["I", "G", "Ir", "Ig", "Ib", "m", "n", "num", "GP", "cnt", "x", "y_r", "y_g", "y_b",
                          "peaks", "pix"],
                   REF / "histogram.mat", addpath=[CH], workdir=SCRATCH)


def ref_color_image():
    code = f"""
run('color_image.m');
V1c = -Ir/sqrt(6) - Ig/sqrt(6) + 2*Ib/sqrt(6);
V2c = Ir/sqrt(6) - Ig/sqrt(6);
Ihc = atan(V2c ./ V1c);
Isc = sqrt(V1c.^2 + V2c.^2);
Ihc2 = atan2(V2c, V1c);
Rn = Ir/255; Gn = Ig/255; Bn = Ib/255;
Kb = min(cat(3, 1-Rn, 1-Gn, 1-Bn), [], 3);
u = 1; bb = 1;
C = 1 - Rn - u*Kb; M = 1 - Gn - u*Kb; Y = 1 - Bn - u*Kb; K = bb*Kb;
u2 = 0.5; b2 = 0.7;
C2 = 1 - Rn - u2*Kb; M2 = 1 - Gn - u2*Kb; Y2 = 1 - Bn - u2*Kb; K2 = b2*Kb;
nanH = nnz(isnan(Ih)); nanHc = nnz(isnan(Ihc));
{_imw('I_cmy', 'fig_2_04_I_cmy.png', False)}
{_imw('Ic', 'fig_2_04_Ic.png')}
{_imw('Im', 'fig_2_04_Im.png')}
{_imw('Iy', 'fig_2_04_Iy.png')}
{_imw('Ih', 'fig_2_05_Ih.png')}
{_imw('Is', 'fig_2_05_Is.png')}
{_imw('Ii', 'fig_2_05_Ii.png')}
{_imw('Ihc', 'fig_2_05_Ih_corrected.png')}
{_imw('Isc', 'fig_2_05_Is_corrected.png')}
"""
    return run_ref(code, ["I", "I_cmy", "Ic", "Im", "Iy", "Ii", "Ih", "Is", "V1", "V2", "Ihc", "Isc", "Ihc2",
                          "C", "M", "Y", "K", "Kb", "C2", "M2", "Y2", "K2", "nanH", "nanHc"],
                   REF / "color_image.mat", addpath=[CH], workdir=SCRATCH)


def ref_distance_transform():
    code = f"""
run('distance_transform.m');
D8_script = imgDist; img_script = img;
img = zeros(201, 201); img(101, 101) = 1;
De = -bwdist(img, 'euclidean'); D4 = -bwdist(img, 'cityblock'); D8 = -bwdist(img, 'chessboard');
Dq = -bwdist(img, 'quasi-euclidean');
load({q(REF / 'inputs.mat')});
A12 = logical(A12);
D12 = bwdist(~A12); D12_4 = bwdist(~A12, 'cityblock'); D12_8 = bwdist(~A12, 'chessboard');
D12_q = bwdist(~A12, 'quasi-euclidean');
c7 = zeros(7); c7(4, 4) = 1;
C4 = bwdist(c7, 'cityblock'); C8 = bwdist(c7, 'chessboard'); Ce = bwdist(c7, 'euclidean');
rng(7); R = rand(64, 80) > 0.9;
Re = bwdist(R); R4 = bwdist(R, 'cityblock'); R8 = bwdist(R, 'chessboard'); Rq = bwdist(R, 'quasi-euclidean');
Z = false(5); Dz = bwdist(Z);
{_imw('img', 'fig_2_14_point.png', False)}
{_imw('De', 'fig_2_14_euclidean.png')}
{_imw('D4', 'fig_2_14_cityblock.png')}
{_imw('D8', 'fig_2_14_chessboard.png')}
"""
    return run_ref(code, ["D8_script", "img_script", "De", "D4", "D8", "Dq", "A12", "D12", "D12_4", "D12_8", "D12_q",
                          "C4", "C8", "Ce", "R", "Re", "R4", "R8", "Rq", "Dz"],
                   REF / "distance_transform.mat", addpath=[CH])


def ref_chain_diff():
    code = f"""
run('chain_diff.m');
print(gcf, '-dpng', '-r100', {q(ML_IMG / 'fig_2_19_chain_diff_figure.png')});
fcc = c.fcc; dif = c.diff; mm = c.mm; diffmm = c.diffmm; x0y0 = c.x0y0;
b4c = boundaries(B, 4); b4 = b4c{{1}};
err4 = ''; fcc4 = []; dif4 = []; mm4 = []; diffmm4 = [];
try, c4 = fchcode(b4, 4); fcc4 = c4.fcc; dif4 = c4.diff; mm4 = c4.mm; diffmm4 = c4.diffmm; catch e, err4 = e.message; end
bccwc = boundaries(B, 8, 'ccw'); bccw = bccwc{{1}};
errrev = ''; fccrev = []; difrev = []; mmrev = [];
try, crev = fchcode(b, 8, 'reverse'); fccrev = crev.fcc; difrev = crev.diff; mmrev = crev.mm; catch e, errrev = e.message; end
errccw = ''; fccccw = [];
try, cccw = fchcode(bccw); fccccw = cccw.fcc; catch e, errccw = e.message; end
addpath({q(SCRATCH)});
minmag_err = ''; mm_of_diff = [];
try, mm_of_diff = minmag_ref(dif); catch e, minmag_err = e.message; end
minmag_ok = minmag_ref(fcc);
bim1 = bound2im(b); bim3 = bound2im(b, 12, 14); bim5 = bound2im(b, M, N, 2, 3);
bimT = bound2im(b');
{_imw('bim', 'fig_2_19_bim.png', False)}
"""
    return run_ref(code, ["B", "b", "d", "bim", "fcc", "dif", "mm", "diffmm", "x0y0", "b4", "fcc4", "dif4", "mm4",
                          "diffmm4", "err4", "bccw", "fccrev", "difrev", "mmrev", "errrev", "fccccw", "errccw",
                          "minmag_err", "mm_of_diff", "minmag_ok", "bim1", "bim3", "bim5", "bimT"],
                   REF / "chain_diff.mat", addpath=[CH, CHAIN])


def ref_boundaries_multi():
    code = f"""
load({q(REF / 'inputs.mat')});
BW11 = logical(BW11); Sp = logical(Sp);
L4 = bwlabel(BW11, 4); L8 = bwlabel(BW11, 8);
B8 = boundaries(BW11, 8); B4 = boundaries(BW11, 4); B8c = boundaries(BW11, 8, 'ccw'); B4c = boundaries(BW11, 4, 'ccw');
n8 = numel(B8); n4 = numel(B4);
for k = 1:numel(B8), eval(sprintf('B8_%d = B8{{k}};', k)); eval(sprintf('B8c_%d = B8c{{k}};', k)); end
for k = 1:numel(B4), eval(sprintf('B4_%d = B4{{k}};', k)); eval(sprintf('B4c_%d = B4c{{k}};', k)); end
S = false(5); S(3, 3) = true; Bs = boundaries(S); Bs_1 = Bs{{1}};
errs = ''; fccs = [];
try, cs = fchcode(Bs_1); fccs = cs.fcc; catch e, errs = e.message; end
Bsp = boundaries(Sp); Bsp_1 = Bsp{{1}}; csp = fchcode(Bsp_1); fccsp = csp.fcc; mmsp = csp.mm;
H = true(7, 7); H(3:5, 3:5) = false; BH = boundaries(H); BH_1 = BH{{1}}; nH = numel(BH);
G = rgb2gray(imread({q(RGB)}));
Q = G(1:200, 1:300) > 128;
LQ4 = bwlabel(Q, 4); LQ8 = bwlabel(Q, 8);
BQ = boundaries(Q); BQlen = cellfun(@(x) size(x, 1), BQ); BQcat = vertcat(BQ{{:}});
BQ4 = boundaries(Q, 4); BQ4len = cellfun(@(x) size(x, 1), BQ4); BQ4cat = vertcat(BQ4{{:}});
nQ = numel(BQ); fccQcat = []; fccQlen = zeros(nQ, 1); mmQcat = []; fccQerr = zeros(nQ, 1);
for k = 1:nQ
    try, ck = fchcode(BQ{{k}}); fccQcat = [fccQcat, ck.fcc]; fccQlen(k) = numel(ck.fcc); mmQcat = [mmQcat, ck.mm];
    catch e, fccQerr(k) = 1; end
end
E = false(4, 6); BE = boundaries(E); nE = numel(BE);
"""
    vars_ = ["BW11", "L4", "L8", "n8", "n4", "Bs_1", "fccs", "errs", "Sp", "Bsp_1", "fccsp", "mmsp", "H", "BH_1",
             "nH", "Q", "LQ4", "LQ8", "BQlen", "BQcat", "BQ4len", "BQ4cat", "fccQcat", "fccQlen", "mmQcat",
             "fccQerr", "nE"]
    vars_ += [f"B8_{k}" for k in (1, 2)] + [f"B8c_{k}" for k in (1, 2)]
    vars_ += [f"B4_{k}" for k in range(1, 6)] + [f"B4c_{k}" for k in range(1, 6)]
    return run_ref(code, vars_, REF / "boundaries_multi.mat", addpath=[CH, CHAIN])


def ref_conv2():
    code = """
rng(0); f = rand(12, 15); w3 = [1 2 1; 0 0 0; -1 -2 -1]; w5 = rand(5); w4 = rand(4); w23 = rand(2, 3);
h3 = conv2(f, w3, 'same'); h5 = conv2(f, w5, 'same'); h4 = conv2(f, w4, 'same'); h23 = conv2(f, w23, 'same');
h3full = conv2(f, w3, 'full'); h3valid = conv2(f, w3, 'valid'); h5full = conv2(f, w5, 'full');
g3 = imfilter(f, w3); g3c = imfilter(f, w3, 'conv'); g3r = imfilter(f, w3, 'replicate');
g3s = imfilter(f, w3, 'symmetric'); g3w = imfilter(f, w3, 'circular'); g3k = imfilter(f, w3, 2.5);
g4 = imfilter(f, w4); g23 = imfilter(f, w23); g5r = imfilter(f, w5, 'replicate'); g5s = imfilter(f, w5, 'symmetric');
g3full = imfilter(f, w3, 'full'); g5full = imfilter(f, w5, 'full'); g4full = imfilter(f, w4, 'full');
g4r = imfilter(f, w4, 'replicate');
f3 = rand(9, 11, 3); g3ch = imfilter(f3, w3);
"""
    return run_ref(code, ["f", "w3", "w5", "w4", "w23", "h3", "h5", "h4", "h23", "h3full", "h3valid", "h5full",
                          "g3", "g3c", "g3r", "g3s", "g3w", "g3k", "g4", "g23", "g5r", "g5s", "g3full", "g5full",
                          "g4full", "g4r", "f3", "g3ch"],
                   REF / "conv2.mat", addpath=[CH])


def ref_interp2():
    code = f"""
G = double(rgb2gray(imread({q(RGB)})));
P = G(700:731, 900:931);
[X, Y] = meshgrid(1:32); [Xq, Yq] = meshgrid(1:0.4:32);
Zn = interp2(X, Y, P, Xq, Yq, 'nearest'); Zl = interp2(X, Y, P, Xq, Yq, 'linear'); Zc = interp2(X, Y, P, Xq, Yq, 'cubic');
rng(1); uq = 0.5 + 32*rand(200, 1); vq = 0.5 + 32*rand(200, 1);
uq(1:8) = [1.5 2.5 10.5 31.5 1 32 0.999 32.001]'; vq(1:8) = [3.5 1 2.5 32 1 32 5 5]';
Sn = interp2(X, Y, P, vq, uq, 'nearest'); Sl = interp2(X, Y, P, vq, uq, 'linear'); Sc = interp2(X, Y, P, vq, uq, 'cubic');
R2n = imresize(P, 2, 'nearest'); R2l = imresize(P, 2, 'bilinear', 'Antialiasing', false);
R2c = imresize(P, 2, 'bicubic', 'Antialiasing', false);
R4n = imresize(P, 4, 'nearest'); R4l = imresize(P, 4, 'bilinear'); R4c = imresize(P, 4, 'bicubic');
Rhn = imresize(P, 0.5, 'nearest'); Rhl = imresize(P, 0.5, 'bilinear', 'Antialiasing', false);
imwrite(uint8(Zn), {q(ML_IMG / 'fig_2_22_nearest.png')});
imwrite(uint8(Zl), {q(ML_IMG / 'fig_2_22_bilinear.png')});
imwrite(uint8(Zc), {q(ML_IMG / 'fig_2_22_bicubic.png')});
imwrite(uint8(R4n), {q(ML_IMG / 'fig_2_23_resize4_nearest.png')});
imwrite(uint8(R4l), {q(ML_IMG / 'fig_2_23_resize4_bilinear.png')});
imwrite(uint8(R4c), {q(ML_IMG / 'fig_2_23_resize4_bicubic.png')});
"""
    return run_ref(code, ["P", "Xq", "Yq", "Zn", "Zl", "Zc", "uq", "vq", "Sn", "Sl", "Sc", "R2n", "R2l", "R2c",
                          "R4n", "R4l", "R4c", "Rhn", "Rhl"],
                   REF / "interp2.mat", addpath=[CH])


def ref_compat():
    code = f"""
xr = [-2.5 -1.5 -0.5 -0.4999 0.4999 0.5 1.5 2.5 2.4999999 -0.0 1e-9 100.5];
rx = round(xr);
v = linspace(-0.1, 1.1, 25); u8 = im2uint8(v); u8b = im2uint8(logical([0 1 0]));
d8 = im2double(uint8(0:255)); d16 = im2double(uint16([0 1 65535])); dl = im2double(logical([0 1]));
ic8 = imcomplement(uint8([0 1 127 128 254 255])); icd = imcomplement([0 0.25 1 300 -2]);
icl = imcomplement(logical([0 1])); ic16 = imcomplement(uint16([0 65535 1000])); ici8 = imcomplement(int8([-128 0 127]));
I = imread({q(RGB)}); G = rgb2gray(I); Gd = rgb2gray(im2double(I));
Gc = G(1:300, 1:400); Gd = Gd(1:300, 1:400);
[h64, x64] = imhist(Gc, 64); [h256, x256] = imhist(Gc); [hd, xd] = imhist(im2double(Gc));
[hd100, xd100] = imhist(im2double(Gc), 100); [hl, xl] = imhist(Gc > 128); [h16, x16] = imhist(uint16(Gc) * 257);
[h5, x5] = imhist(uint8([0 1 2 3 4 5 250 251 252 253 254 255]), 5);
sat = uint8([200 100 50]) + uint8([100 100 50]); satm = uint8([10 100]) - uint8([20 50]);
"""
    return run_ref(code, ["xr", "rx", "v", "u8", "u8b", "d8", "d16", "dl", "ic8", "icd", "icl", "ic16", "ici8",
                          "Gc", "Gd", "h64", "x64", "h256", "x256", "hd", "xd", "hd100", "xd100", "hl", "xl",
                          "h16", "x16", "h5", "x5", "sat", "satm"],
                   REF / "compat.mat", addpath=[CH])


ALL = {
    "histogram": ref_histogram,
    "color_image": ref_color_image,
    "distance_transform": ref_distance_transform,
    "chain_diff": ref_chain_diff,
    "boundaries_multi": ref_boundaries_multi,
    "conv2": ref_conv2,
    "interp2": ref_interp2,
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
            log[name] = {"engine": res.engine, "version": res.version, "out": res.out_mat.name, "status": "ok"}
            print(f"  ok: {res.engine} {res.version} -> {res.out_mat.name}")
        except Exception as exc:  # keep going; the report lists what failed
            msg = str(exc)
            log[name] = {"status": "error", "error": msg[:4000]}
            print(f"  ERROR: {msg[:3000]}")
            rc = 1
        log_path.write_text(json.dumps(log, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
