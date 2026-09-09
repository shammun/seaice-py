"""Generate the MATLAB (R2025a) reference outputs for chapter 3 (Book §3.1–§3.3, Ice Pixel Detection).

Runs the ORIGINAL ``.m`` files of ``MATLAB_ROOT/ch3`` (``Otsu.m``, ``local_Otsu.m``, ``separability.m``,
``kmeans.m``) through ``tools.run_matlab_ref.run_ref`` (``matlab -batch``, figures invisible) and saves the workspace
variables listed in ``analysis/ch03.md`` §6 as ``reference/ch03/*.mat`` (v7).  Every launch also writes the images
MATLAB would *display* (``imwrite`` / ``print``) into ``outputs/ch03/verify/matlab/`` for the side-by-side figure
comparisons (L3).

Usage:  .venv/Scripts/python.exe reference/ch03/make_refs.py [name ...]
        (no arguments = all references; names = otsu_test, otsu_1, otsu_2, kmeans_test_k2, kmeans_test_k3,
         kmeans_1_k2, kmeans_1_k3, kmeans_2_k2, kmeans_2_k3, local_otsu, separability_108, separability_107, compat)

Notes
-----
* The scripts hard-code ``imread('test.jpg')`` / ``k = 3`` / ``k = 108`` and read from the *current folder*, and
  ``kmeans.m`` **shadows** the Statistics Toolbox function of the same name.  Therefore the four scripts are copied
  verbatim into a scratch folder (``outputs/ch03/verify/scratch``) under non-shadowing names, and only the literal
  ``imread('...')`` / ``k = ...`` tokens are patched (plus an iteration counter ``n_iter`` in ``kmeans.m``).  Nothing
  in MATLAB_ROOT is touched and the ch3 folder is never put on the MATLAB path.
* ``t.jpg`` (Fig. 3.4(a)) and ``ch3ice.jpg`` (Fig. 3.2(a)) are **not shipped**.  Following ``analysis/ch03.md`` §5 the
  scratch folder holds ``ch3ice.jpg`` = copy of ``2.jpg`` and ``t.jpg`` = the synthetic uneven-illumination JPEG
  ``data/synthetic/ch03/t_uneven_2_g0.5_b40.jpg`` written by ``scripts/ch03_local_otsu.py`` (so MATLAB and Python
  decode the same bytes).  The book's Fig. 3.2–3.5 / 3.7 numbers stay ``unverified``; MATLAB-vs-Python parity on the
  substitutes is still measured.
* ``compat`` runs the toolbox functions (``graythresh``, ``otsuthresh``, ``im2bw``, ``multithresh``, ``imquantize``,
  ``im2uint8``) on controlled arrays saved from Python (``reference/ch03/inputs.mat``) and hand-codes Eqs. (3.3)–(3.22)
  in MATLAB for the ``otsu_criterion`` curves.
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
CH = MATLAB_ROOT / "ch3"
REF = ROOT / "reference/ch03"
VERIFY = ROOT / "outputs/ch03/verify"
ML_IMG = VERIFY / "matlab"
SCRATCH = VERIFY / "scratch"
DATA = ROOT / "data/book/ch03"
T_SYNTH = ROOT / "data/synthetic/ch03/t_uneven_2_g0.5_b40.jpg"

IMAGES = {"test": "test.jpg", "1": "1.jpg", "2": "2.jpg"}


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


def _prepare() -> None:
    for d in (REF, ML_IMG, SCRATCH):
        d.mkdir(parents=True, exist_ok=True)
    for fname in IMAGES.values():
        if not (SCRATCH / fname).exists():
            shutil.copy(DATA / fname, SCRATCH / fname)
    if not (SCRATCH / "ch3ice.jpg").exists():  # substitute for the unshipped Fig. 3.2(a)
        shutil.copy(DATA / "2.jpg", SCRATCH / "ch3ice.jpg")
    if not T_SYNTH.exists():  # written once by scripts/ch03_local_otsu.py (deterministic)
        sys.path.insert(0, str(ROOT / "scripts"))
        from ch03_local_otsu import ramped_substitute  # noqa: E402
        from seaice.core.io import read_image  # noqa: E402
        ramped_substitute(read_image(DATA / "2.jpg"), T_SYNTH.parent, 0.5, 40.0)
    shutil.copy(T_SYNTH, SCRATCH / "t.jpg")

    # --- patched verbatim copies (only the literals change) -------------------------------------------------------
    otsu = _read_m("Otsu.m")
    for key, fname in IMAGES.items():
        (SCRATCH / f"Otsu_ref_{key}.m").write_text(
            _patch(otsu, [("imread('test.jpg')", f"imread('{fname}')")]), encoding="utf-8")
    km = _read_m("kmeans.m")
    for key, fname in IMAGES.items():
        for k in (2, 3):
            (SCRATCH / f"kmeans_ref_{key}_k{k}.m").write_text(
                _patch(km, [("imread('test.jpg')", f"imread('{fname}')"), ("k = 3;", f"k = {k};"),
                            ("while(true)", "n_iter = 0;\nwhile(true)"),
                            ("oldmu = mu;", "oldmu = mu; n_iter = n_iter + 1;")]), encoding="utf-8")
    (SCRATCH / "local_Otsu_ref.m").write_text(_read_m("local_Otsu.m"), encoding="utf-8")
    sep = _read_m("separability.m")
    (SCRATCH / "separability_ref_108.m").write_text(sep, encoding="utf-8")
    (SCRATCH / "separability_ref_107.m").write_text(_patch(sep, [("k = 108;", "k = 107;")]), encoding="utf-8")

    # --- controlled inputs shared with the tests (identical arrays on both sides) ------------------------------------
    rng = np.random.default_rng(3)
    ramp = np.round(np.linspace(5, 250, 30 * 41)).astype(np.uint8).reshape(30, 41)
    tie = np.where((np.indices((8, 8)).sum(axis=0) % 2) == 0, 100, 110).astype(np.uint8)
    savemat(REF / "inputs.mat", {
        "Cst": np.full((10, 12), 77, dtype=np.uint8),
        "Two": (rng.random((20, 25)) > 0.6).astype(np.uint8) * 255,
        "Tie": tie,
        "U16": np.round(rng.random((30, 40)) * 65535).astype(np.uint16),
        "Dbl": rng.random((30, 40)),
        "Ramp": ramp,
        "Four": rng.choice(np.array([10, 60, 150, 240], dtype=np.uint8), size=(20, 20)),
        "Rnd": synth.bimodal_image(seed=0),
        "I16": np.array([[-32768, -32767, -1, 0, 1, 32767, 1000, -1000]], dtype=np.int16),
        "Hc": np.array([[0, 0, 5, 0, 0, 0, 5, 0, 0, 0]], dtype=np.float64),
        # review follow-ups (reports/ch03_review.md Must-fix 1, Should-fix 2-3): degenerate multithresh inputs,
        # the four-spike N = 3 case and a half-integer im2bw / block-Otsu level
        "Z44": np.zeros((4, 4), dtype=np.uint8),                                  # multithresh(., 2) -> [0 1]
        "B2": np.array([[0, 255, 0, 255]], dtype=np.uint8),                       # multithresh(., 3) -> [0 1 255]
        "Tri": np.tile(np.array([100, 101, 102], dtype=np.uint8), (3, 5)),        # multithresh(., 3) -> [100 101 102]
        "HL": np.tile(np.array([100, 255], dtype=np.uint8), (2, 6)),              # multithresh(., 3) -> [1 100 255]
        "Spk": np.tile(np.array([0, 1, 128, 255], dtype=np.uint8), 100),          # 4 spikes, N = 3 needs t1 = 0
        "Half": np.array([[104, 105, 106]], dtype=np.uint8),                      # graythresh tie -> 104.5 / 255
    })


def _imw(var: str, name: str, autoscale: bool = True) -> str:
    p = q(ML_IMG / name)
    if autoscale:
        return f"tmp__ = mat2gray(double({var})); imwrite(tmp__, {p});"
    return f"imwrite({var}, {p});"


# ------------------------------------------------------------------------------------------------------------
def ref_otsu(key: str):
    fname = IMAGES[key]
    code = f"""
run({q(SCRATCH / f'Otsu_ref_{key}.m')});
[t2, em] = graythresh(I);
[th2, metric] = multithresh(I, 2);
th1 = multithresh(I, 1);
[th3, metric3] = multithresh(I, 3);
seg_vals = imquantize(I, thresh, [0 128 255]);
ncls = zeros(1, 3); for ii = 1:3, ncls(ii) = nnz(seg == ii); end
seg = uint8(seg); seg_vals = uint8(seg_vals); bw = logical(bw);
Irgb = imread('{fname}');
{_imw('bw', f'otsu_bw_{key}.png', False)}
{_imw('seg', f'multi_otsu_seg_{key}.png')}
"""
    return run_ref(code, ["I", "r", "c", "t", "bw", "ic", "thresh", "seg", "coverage", "average_intensity", "t2",
                          "em", "th2", "metric", "th1", "th3", "metric3", "seg_vals", "ncls", "Irgb"],
                   REF / f"otsu_{key}.mat", workdir=SCRATCH, timeout=1800)


def ref_kmeans(key: str, k: int):
    code = f"""
run({q(SCRATCH / f'kmeans_ref_{key}_k{k}.m')});
mask = uint8(mask);
{_imw('mask1', f'kmeans{k}_mask_{key}.png', False)}
imwrite(ind2rgb(gray2ind(mask1, 256), parula(256)), {q(ML_IMG / f'kmeans{k}_mask_{key}_parula.png')});
"""
    return run_ref(code, ["mu", "mask", "ic", "n", "average_intensity", "n_iter", "mi", "m", "h", "hl", "hc", "k"],
                   REF / f"kmeans_{key}_k{k}.mat", workdir=SCRATCH, timeout=1800)


def ref_local_otsu():
    code = f"""
run({q(SCRATCH / 'local_Otsu_ref.m')});
print(gcf, '-dpng', '-r60', {q(ML_IMG / 'fig_3_04c_local_otsu_blocks.png')});
[tg, emg] = graythresh(I); bwg = im2bw(I, tg); ICg = nnz(bwg) / (r*c);
bwl = false(r, c);
for i = 1 : n_r
    for j = 1 : n_c
        tmp = I(t1(i):t2(i), t3(j):t4(j));
        bwl(t1(i):t2(i), t3(j):t4(j)) = im2bw(tmp, graythresh(tmp));
    end
end
levels = thresh / 255;
tstr = strjoin(arrayfun(@(x) num2str(x), IC_local * 100, 'UniformOutput', false), '|');
thstr = strjoin(arrayfun(@(x) num2str(x), thresh, 'UniformOutput', false), '|');
n2s = strjoin({{num2str(73.8472), num2str(86.496), num2str(pi), num2str(0.5), num2str(107.5), num2str(92), ...
               num2str(20.9869), num2str(123.4567), num2str(0.123456789), num2str(1234.56789)}}, '|');
{_imw('bwg', 'local_otsu_global_bw.png', False)}
{_imw('bwl', 'local_otsu_bw.png', False)}
{_imw('I', 'local_otsu_input_gray.png', False)}
"""
    return run_ref(code, ["I", "r", "c", "n_r", "n_c", "thresh", "levels", "num", "IC_local", "IC", "tg", "emg",
                          "bwg", "ICg", "bwl", "tstr", "thstr", "n2s"],
                   REF / "local_otsu.mat", workdir=SCRATCH, timeout=1800)


def ref_separability(k: int):
    extra = ""
    if k == 108:
        # Book Eqs. (3.3)-(3.22) hand-coded in MATLAB, levels i = 0..255, class C0 = 0..t (book convention)
        extra = f"""
cnt = imhist(I, 256); pv = cnt / sum(cnt); iv = (0:255)';
mGv = sum(iv .* pv); sGv = sum((iv - mGv).^2 .* pv);
P0v = zeros(256,1); P1v = P0v; mv = P0v; m0v = P0v; m1v = P0v; s0v = P0v; s1v = P0v; sWv = P0v; sBv = P0v; etav = P0v;
for tt = 0:255
    c0 = iv <= tt; c1 = ~c0;
    P0v(tt+1) = sum(pv(c0)); P1v(tt+1) = sum(pv(c1)); mv(tt+1) = sum(iv(c0) .* pv(c0));
    m0v(tt+1) = mv(tt+1) / P0v(tt+1); m1v(tt+1) = sum(iv(c1) .* pv(c1)) / P1v(tt+1);
    s0v(tt+1) = sum((iv(c0) - m0v(tt+1)).^2 .* pv(c0)) / P0v(tt+1);
    s1v(tt+1) = sum((iv(c1) - m1v(tt+1)).^2 .* pv(c1)) / P1v(tt+1);
    sWv(tt+1) = P0v(tt+1) * s0v(tt+1) + P1v(tt+1) * s1v(tt+1);
    sBv(tt+1) = P0v(tt+1) * (m0v(tt+1) - mGv)^2 + P1v(tt+1) * (m1v(tt+1) - mGv)^2;
    etav(tt+1) = sBv(tt+1) / sGv;
end
sB16 = (mGv * P0v - mv).^2 ./ (P0v .* (1 - P0v));
[etastar, ix] = max(etav(1:255)); tstar = ix - 1;
[lv, emv] = graythresh(I); eta125 = etav(126);
"""
    code = f"""
run({q(SCRATCH / f'separability_ref_{k}.m')});
bw = logical(bw);
{_imw('bw', f'separability_bw_k{k}.png', False)}
{extra}
"""
    vars_ = ["I", "k", "bw", "IC", "p", "mg", "sigma2_g", "sigma2_b", "eta", "m", "p0"]
    if k == 108:
        vars_ += ["cnt", "pv", "mGv", "sGv", "P0v", "P1v", "mv", "m0v", "m1v", "s0v", "s1v", "sWv", "sBv", "sB16",
                  "etav", "etastar", "tstar", "lv", "emv", "eta125"]
    return run_ref(code, vars_, REF / f"separability_{k}.mat", workdir=SCRATCH, timeout=1800)


def ref_compat():
    code = f"""
load({q(REF / 'inputs.mat')});
[lv_Cst, em_Cst] = graythresh(Cst); [lv_Two, em_Two] = graythresh(Two); [lv_Tie, em_Tie] = graythresh(Tie);
[lv_U16, em_U16] = graythresh(U16); [lv_Dbl, em_Dbl] = graythresh(Dbl); [lv_Ramp, em_Ramp] = graythresh(Ramp);
[lv_Four, em_Four] = graythresh(Four); [lv_Rnd, em_Rnd] = graythresh(Rnd); [lv_I16, em_I16] = graythresh(I16);
[lv_Dbl255, em_Dbl255] = graythresh(Dbl * 255);
[ot_t, ot_em] = otsuthresh(imhist(Rnd)); [ot10_t, ot10_em] = otsuthresh(Hc);
nbw = zeros(1, 256); for ii = 0:255, nbw(ii+1) = nnz(im2bw(Rnd, ii/255)); end
bw_tie = im2bw(Tie, lv_Tie); bw_u16 = im2bw(U16, lv_U16); bw_dbl = im2bw(Dbl, lv_Dbl); bw_default = im2bw(Rnd);
bw_i16 = im2bw(I16, lv_I16); bw_rgb = im2bw(cat(3, Rnd, Rnd, Rnd), lv_Rnd);
[mt1_Ramp, mm1_Ramp] = multithresh(Ramp, 1); [mt2_Ramp, mm2_Ramp] = multithresh(Ramp, 2); [mt3_Ramp, mm3_Ramp] = multithresh(Ramp, 3);
[mt1_Rnd, mm1_Rnd] = multithresh(Rnd, 1); [mt2_Rnd, mm2_Rnd] = multithresh(Rnd, 2); [mt3_Rnd, mm3_Rnd] = multithresh(Rnd, 3);
[mt1_U16, mm1_U16] = multithresh(U16, 1); [mt2_U16, mm2_U16] = multithresh(U16, 2);
[mt1_Dbl, mm1_Dbl] = multithresh(Dbl, 1); [mt2_Dbl, mm2_Dbl] = multithresh(Dbl, 2);
[mt1_Four, mm1_Four] = multithresh(Four, 1); [mt2_Four, mm2_Four] = multithresh(Four, 2); [mt3_Four, mm3_Four] = multithresh(Four, 3);
[mt2_Two, mm2_Two] = multithresh(Two, 2); [mt1_Cst, mm1_Cst] = multithresh(Cst, 1); [mt2_Cst, mm2_Cst] = multithresh(Cst, 2);
[mt1_Tie, mm1_Tie] = multithresh(Tie, 1); [mt2_I16, mm2_I16] = multithresh(I16, 2);
[mt2_Z44, mm2_Z44] = multithresh(Z44, 2); [mt3_B2, mm3_B2] = multithresh(B2, 3);
[mt3_Tri, mm3_Tri] = multithresh(Tri, 3); [mt3_HL, mm3_HL] = multithresh(HL, 3); [mt3_Spk, mm3_Spk] = multithresh(Spk, 3);
[lv_Half, em_Half] = graythresh(Half); gt_half = Half > 104.5; bw_half = im2bw(Half, 104.5/255);
n_half = nnz(Half > lv_Half * 255);
q_Ramp = imquantize(Ramp, mt2_Ramp); qv_Ramp = imquantize(Ramp, mt2_Ramp, [10 20 30]); q_Dbl = imquantize(Dbl, mt2_Dbl);
q_Rnd3 = imquantize(Rnd, mt3_Rnd); q_Rnd1 = imquantize(Rnd, mt1_Rnd);
u16to8 = im2uint8(uint16(0:65535)); i16to8 = im2uint8(I16);
gt_Rnd_pdf = imhist(im2uint8(Rnd(:)), 256);
"""
    vars_ = ["Cst", "Two", "Tie", "U16", "Dbl", "Ramp", "Four", "Rnd", "I16", "Hc",
             "Z44", "B2", "Tri", "HL", "Spk", "Half", "mt2_Z44", "mm2_Z44", "mt3_B2", "mm3_B2", "mt3_Tri", "mm3_Tri",
             "mt3_HL", "mm3_HL", "mt3_Spk", "mm3_Spk", "lv_Half", "em_Half", "gt_half", "bw_half", "n_half",
             "lv_Cst", "em_Cst", "lv_Two", "em_Two", "lv_Tie", "em_Tie", "lv_U16", "em_U16", "lv_Dbl", "em_Dbl",
             "lv_Ramp", "em_Ramp", "lv_Four", "em_Four", "lv_Rnd", "em_Rnd", "lv_I16", "em_I16", "lv_Dbl255", "em_Dbl255",
             "ot_t", "ot_em", "ot10_t", "ot10_em", "nbw", "bw_tie", "bw_u16", "bw_dbl", "bw_default", "bw_i16", "bw_rgb",
             "mt1_Ramp", "mm1_Ramp", "mt2_Ramp", "mm2_Ramp", "mt3_Ramp", "mm3_Ramp",
             "mt1_Rnd", "mm1_Rnd", "mt2_Rnd", "mm2_Rnd", "mt3_Rnd", "mm3_Rnd",
             "mt1_U16", "mm1_U16", "mt2_U16", "mm2_U16", "mt1_Dbl", "mm1_Dbl", "mt2_Dbl", "mm2_Dbl",
             "mt1_Four", "mm1_Four", "mt2_Four", "mm2_Four", "mt3_Four", "mm3_Four",
             "mt2_Two", "mm2_Two", "mt1_Cst", "mm1_Cst", "mt2_Cst", "mm2_Cst", "mt1_Tie", "mm1_Tie", "mt2_I16", "mm2_I16",
             "q_Ramp", "qv_Ramp", "q_Dbl", "q_Rnd3", "q_Rnd1", "u16to8", "i16to8", "gt_Rnd_pdf"]
    return run_ref(code, vars_, REF / "compat.mat", timeout=1800)


ALL = {
    "otsu_test": lambda: ref_otsu("test"),
    "otsu_1": lambda: ref_otsu("1"),
    "otsu_2": lambda: ref_otsu("2"),
    "kmeans_test_k2": lambda: ref_kmeans("test", 2),
    "kmeans_test_k3": lambda: ref_kmeans("test", 3),
    "kmeans_1_k2": lambda: ref_kmeans("1", 2),
    "kmeans_1_k3": lambda: ref_kmeans("1", 3),
    "kmeans_2_k2": lambda: ref_kmeans("2", 2),
    "kmeans_2_k3": lambda: ref_kmeans("2", 3),
    "local_otsu": ref_local_otsu,
    "separability_108": lambda: ref_separability(108),
    "separability_107": lambda: ref_separability(107),
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
