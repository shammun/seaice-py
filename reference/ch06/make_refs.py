"""Generate the MATLAB (R2025a) reference outputs for chapter 6 (GVF snake-based ice floe identification).

Runs the ORIGINAL ``.m`` files of ``MATLAB_ROOT/ch6/`` (both sub-folders) through
``tools.run_matlab_ref.run_ref`` (``matlab -batch``, figures invisible) and saves the workspace variables listed
in ``analysis/ch06.md`` §6.1 as ``reference/ch06/*.mat`` (v7).

Usage:  .venv/Scripts/python.exe reference/ch06/make_refs.py [name ...]
        names = unit, gvf, regionprops, geom, dist_script, for_test, gvf_alg, kmean_alg, kmean_stage

Notes
-----
* Every ``.m`` file of ``ch6/Sea_Ice_Floe_Identification/`` is copied **verbatim** into the scratch folder
  ``outputs/ch06/verify/scratch/`` together with the three JPEGs, and MATLAB is run with that folder as cwd.
  The ch6 folders are never put on the MATLAB path and nothing under ``MATLAB_ROOT`` is touched.
* ``for test/dist.m`` is copied as ``ch6_dist_script.m`` (a file called ``dist.m`` would shadow the Deep
  Learning Toolbox function of that name) with two literal patches: the author's absolute ``imread`` path
  becomes the local ``sea_ice_test.jpg`` and accumulator statements are appended *after* the verbatim
  statements of its three display loops.
* ``for test/for_test.m`` is copied as ``for_test_ref.m`` with accumulators appended and the two ``pause``
  calls removed (display-only).
* ``GVF_distance.m`` and ``seaice_kmean_GVF_forenhancement.m`` are copied as ``gvf_distance_ref.m`` /
  ``kmean_gvf_ref.m``: the function line is renamed, a second output ``REC`` is added and accumulator
  statements are appended after the verbatim statements.  No expression of the original is modified.
* ``snake_matrix_ref.m`` is a small helper holding the **verbatim** matrix-building lines 22–38 of
  ``snakedeform.m`` so ``A``/``invAI`` can be compared directly (they are local variables of the original).
* Controlled fixtures come from ``reference/ch06/fixtures.py`` via ``reference/ch06/inputs.mat`` so both sides
  operate on identical arrays.
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
from reference.ch06.fixtures import all_fixtures  # noqa: E402

MATLAB_ROOT = ROOT / "K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab"
CH = MATLAB_ROOT / "ch6"
SIFI = CH / "Sea_Ice_Floe_Identification"
FORTEST = CH / "for test"
REF = ROOT / "reference/ch06"
VERIFY = ROOT / "outputs/ch06/verify"
ML_IMG = VERIFY / "matlab"
SCRATCH = VERIFY / "scratch"
DATA = ROOT / "data/book/ch06"


def q(p) -> str:
    return "'" + str(p).replace("\\", "/").replace("'", "''") + "'"


def _imw(var: str, name: str, autoscale: bool = False) -> str:
    p = q(ML_IMG / name)
    if autoscale:
        return f"tmp__ = mat2gray(double({var})); imwrite(tmp__, {p});"
    return f"imwrite({var}, {p});"


# ------------------------------------------------------------------------------------------------------------
# Scratch preparation
# ------------------------------------------------------------------------------------------------------------
def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _patch(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        if old not in text:
            raise RuntimeError(f"patch token not found: {old!r}")
        text = text.replace(old, new, 1)
    return text


SNAKE_MATRIX_HELPER = """function [A, invAI] = snake_matrix_ref(N, alpha, beta, gamma)
% Lines 22-38 of the shipped snakedeform.m, verbatim, exposed so the matrix can be compared.
alpha = alpha* ones(1,N);
beta = beta*ones(1,N);
alpham1 = [alpha(2:N) alpha(1)];
alphap1 = [alpha(N) alpha(1:N-1)];
betam1 = [beta(2:N) beta(1)];
betap1 = [beta(N) beta(1:N-1)];
a = betam1;
b = -alpha - 2*beta - 2*betam1;
c = alpha + alphap1 +betam1 + 4*beta + betap1;
d = -alphap1 - 2*beta - 2*betap1;
e = betap1;
A = diag(a(1:N-2),-2) + diag(a(N-1:N),N-2);
A = A + diag(b(1:N-1),-1) + diag(b(N), N-1);
A = A + diag(c);
A = A + diag(d(1:N-1),1) + diag(d(N),-(N-1));
A = A + diag(e(1:N-2),2) + diag(e(N-1:N),-(N-2));
invAI = inv(A + gamma * diag(ones(1,N)));
end
"""


def _prepare() -> None:
    for d in (REF, VERIFY, ML_IMG, SCRATCH):
        d.mkdir(parents=True, exist_ok=True)
    for m in sorted(SIFI.glob("*.m")):
        shutil.copy2(m, SCRATCH / m.name)
    for img in ("sea_ice_test.jpg",):
        shutil.copy2(SIFI / img, SCRATCH / img)
    for img in ("test8.jpg", "alg_seg_gray.jpg"):
        shutil.copy2(FORTEST / img, SCRATCH / img)
    (SCRATCH / "snake_matrix_ref.m").write_text(SNAKE_MATRIX_HELPER, encoding="utf-8")

    # --- minboundrect.m -> minboundrect_ref.m (one obsolete Qhull option removed; see ref_geom) ---------
    t = _read(SIFI / "minboundrect.m")
    t = _patch(t, [
        ("function [rectx,recty,area,perimeter] = minboundrect(x,y,metric)",
         "function [rectx,recty,area,perimeter] = minboundrect_ref(x,y,metric)"),
        ("  edges = convhull(x,y,{'Qt'});  % 'Pp' will silence the warnings",
         "  edges = convhull(x,y);  % R2025a rejects the 2007 Qhull option list"),
    ])
    (SCRATCH / "minboundrect_ref.m").write_text(t, encoding="utf-8")

    # --- dist.m -> ch6_dist_script.m -------------------------------------------------------------------
    t = _read(FORTEST / "dist.m")
    t = _patch(t, [
        ("[I1,MAP] = imread('C:\\Users\\qinz\\Desktop\\OneDrive\\OneDrive for Business\\CRC\\latex\\matlab\\"
         "ch6\\Sea_Ice_Floe_Identification\\sea_ice_test.jpg');",
         "[I1,MAP] = imread('sea_ice_test.jpg');"),
    ])
    # accumulators appended after the verbatim statements of the three loops
    t = t.replace("    plot(cen(1), cen(2), 'r+');\n    plot(x, y, 'b');\nend",
                  "    plot(cen(1), cen(2), 'r+');\n    plot(x, y, 'b');\n    CEN1(n,:) = cen;\nend", 1)
    t = t.replace("    plot(cen(1), cen(2), 'r+');\n    plot(x, y, 'b');\nend",
                  "    plot(cen(1), cen(2), 'r+');\n    plot(x, y, 'b');\n    CEN2(n,:) = cen; R0(n) = r0;"
                  " X0{n} = x(:)'; Y0{n} = y(:)';\nend", 1)
    t = ("CEN1 = []; CEN2 = []; R0 = []; X0 = {}; Y0 = {};\n" + t
         + "\nnum_out = num; l_num = nn;\n")
    (SCRATCH / "ch6_dist_script.m").write_text(t, encoding="utf-8")

    # --- for_test.m -> for_test_ref.m ------------------------------------------------------------------
    t = _read(FORTEST / "for_test.m")
    t = _patch(t, [
        ("bw = im2bw(I, graythresh(I));\n[s1, s2] = size(bw);",
         "bw = im2bw(I, graythresh(I));\nbw_top = bw;\n[s1, s2] = size(bw);"),
        ("[x, y] = polybool('intersection', s_2, s_1, x, y);",
         "[x, y] = polybool('intersection', s_2, s_1, x, y);\nXPB = x(:)'; YPB = y(:)'; SZPB = size(x);"),
        ("[x,y] = snakeinterp(x, y, Dmax, Dmin);",
         "[x,y] = snakeinterp(x, y, Dmax, Dmin);\nXI0 = x(:)'; YI0 = y(:)';"),
        ("   [x, y] = snakeinterp(x, y, Dmax, Dmin);",
         "   [x, y] = snakeinterp(x, y, Dmax, Dmin);\n   XB{i} = x(:)'; YB{i} = y(:)';"),
        # `for_test.m` line 119 `[v, h] = size(bw)` shadows the GVF field variable `v`: keep a copy first.
        ("bw = im2bw(I, graythresh(I));\n[v, h] = size(bw);",
         "bw = im2bw(I, graythresh(I));\nbw_pre = bw; u_gvf = u; v_gvf = v;\n[v, h] = size(bw);"),
    ])
    t = t.replace("pause(1);", ";").replace("pause(0.1);", ";")
    t = "XB = {}; YB = {};\n" + t + "\nxx_out = xx(:)'; yy_out = yy(:)'; bw_burn = bw;\n"
    (SCRATCH / "for_test_ref.m").write_text(t, encoding="utf-8")

    # --- GVF_distance.m -> gvf_distance_ref.m ----------------------------------------------------------
    t = _read(SIFI / "GVF_distance.m")
    t = _patch(t, [
        ("function bw1 = GVF_distance( I, sigma", "function [bw1, REC] = gvf_distance_ref( I, sigma"),
        ("py = v ./ (mag + 1e-10);",
         "py = v ./ (mag + 1e-10);\nREC.bw = bw; REC.f2 = f2; REC.u = u; REC.v = v; REC.px = px; REC.py = py;"
         "\nREC.gray = I; REC.level = graythresh(I);"),
        ("    rl = l ./ w;    % ratio between length and width ",
         "    rl = l ./ w;    % ratio between length and width \n"
         "    REC.label = label; REC.num = num; REC.a = a; REC.rc = rc; REC.l = l; REC.w = w; REC.rl = rl;"),
        ("    k = unique(k);", "    k = unique(k);\n    REC.k = k;"),
        ("    bw2 = bwareaopen(bw2, Ra_min);", "    bw2 = bwareaopen(bw2, Ra_min);\n    REC.bw2 = bw2;"),
        ("    dis = imdilate(dis, se);",
         "    dis = imdilate(dis, se);\n    REC.img_Dist = img_Dist; REC.Dis_img = Dis_img; REC.dis = dis;"),
        ("    [label1, num1] = bwlabel(dis, 8);",
         "    [label1, num1] = bwlabel(dis, 8);\n    REC.label1 = label1; REC.num1 = num1;"
         " REC.cen = zeros(num1,2); REC.r = zeros(num1,1);"),
        ("        [x, y] = snakeinterp(x, y, Dmax, Dmin);\n"
         "        [x, y] = polybool('intersection', s_2, s_1, x, y);",
         "        [x, y] = snakeinterp(x, y, Dmax, Dmin);\n"
         "        REC.XI{n1} = x(:)'; REC.YI{n1} = y(:)';\n"
         "        [x, y] = polybool('intersection', s_2, s_1, x, y);"),
        ("         x = x';\n         y = y';",
         "         x = x';\n         y = y';\n         REC.XC{n1} = x(:)'; REC.YC{n1} = y(:)';"
         "\n         REC.cen(n1,:) = cen; REC.r(n1) = r;"),
        ("         xx = ceil(x); yy = ceil(y);",
         "         REC.XF{n1} = x(:)'; REC.YF{n1} = y(:)';\n         xx = ceil(x); yy = ceil(y);"
         "\n         REC.XX{n1} = xx(:)'; REC.YY{n1} = yy(:)';"),
    ])
    (SCRATCH / "gvf_distance_ref.m").write_text(t, encoding="utf-8")

    # --- seaice_kmean_GVF_forenhancement.m -> kmean_gvf_ref.m ------------------------------------------
    t = _read(SIFI / "seaice_kmean_GVF_forenhancement.m")
    t = t.replace("function [out, bk] = seaice_kmean_GVF_forenhancement( I, kms0",
                  "function [out, bk, REC] = kmean_gvf_ref( I, kms0", 1)
    t = _patch(t, [
        ("py = v ./ (mag + 1e-10);",
         "py = v ./ (mag + 1e-10);\nREC.bw = bw; REC.f2 = f2; REC.u = u; REC.v = v; REC.px = px; REC.py = py;"),
        ("bw0 = bwareaopen(bw0, Ra_min, 4);",
         "bw0 = bwareaopen(bw0, Ra_min, 4);\nREC.map0 = map0; REC.s0 = s0; REC.ind0 = ind0; REC.bk = bk;"
         "\nREC.bw0_raw = bw_kmeans - bw; REC.bw0 = bw0; REC.n_negative = sum(sum((bw_kmeans - bw) < 0));"),
        ("out = bw1 + bw0 * 0.5;", "REC.bw1 = bw1; REC.bw0_final = bw0;\nout = bw1 + bw0 * 0.5;"),
    ])
    # per-pass records (the body appears twice; tag them 1 and 2)
    for tag in (1, 2):
        t = t.replace("    rl = l ./ w;    % ratio between length and width ",
                      f"    rl = l ./ w;    % ratio between length and width \n"
                      f"    REC.label{tag} = label; REC.num{tag} = num; REC.a{tag} = a; REC.rc{tag} = rc;"
                      f" REC.l{tag} = l; REC.w{tag} = w; REC.rl{tag} = rl;", 1)
        t = t.replace("    k = unique(k);", f"    k = unique(k);\n    REC.k{tag} = k;", 1)
        t = t.replace("    bw2 = bwareaopen(bw2, Ra_min);",
                      f"    bw2 = bwareaopen(bw2, Ra_min);\n    REC.bw2_{tag} = bw2;", 1)
        t = t.replace("        dis = imdilate(dis, se);",
                      f"        dis = imdilate(dis, se);\n        REC.imgDist{tag} = img_Dist;"
                      f" REC.Dis{tag} = Dis_img; REC.dis{tag} = dis;", 1)
        t = t.replace("        [label1, num1] = bwlabel(dis, 8);",
                      f"        [label1, num1] = bwlabel(dis, 8);\n        REC.lab1_{tag} = label1;"
                      f" REC.num1_{tag} = num1; REC.cen{tag} = zeros(num1,2); REC.r{tag} = zeros(num1,1);", 1)
        t = t.replace("            x = x';\n            y = y';",
                      f"            x = x';\n            y = y';\n"
                      f"            REC.cen{tag}(n1,:) = cen; REC.r{tag}(n1) = r;", 1)
    (SCRATCH / "kmean_gvf_ref.m").write_text(t, encoding="utf-8")

    # --- fixtures --------------------------------------------------------------------------------------
    fx = all_fixtures()
    savemat(str(REF / "inputs.mat"), fx, do_compression=True)
    print(f"[make_refs] {len(fx)} fixtures -> inputs.mat")


# ------------------------------------------------------------------------------------------------------------
# Group 1: unit references (del2, BoundMirror*, gradient2, xconv2, gaussian*, snakeindex/interp/deform)
# ------------------------------------------------------------------------------------------------------------
def ref_unit():
    from reference.ch06 import fixtures as F
    lines = [f"load({q(REF / 'inputs.mat')});"]
    v: list[str] = []

    def add(code: str, *names: str) -> None:
        lines.append(code)
        v.extend(names)

    # --- del2 -------------------------------------------------------------------------------------------
    for name in F.del2_fixtures():
        add(f"d_{name} = del2({name});", f"d_{name}")
    add("dh_del2_4x5 = del2(del2_4x5, 2);", "dh_del2_4x5")
    add("dxy_del2_4x5 = del2(del2_4x5, 2, 3);", "dxy_del2_4x5")
    add("dvec_del2_4x5 = del2(del2_4x5, [0 1 3 6 10], [0 2 5 9]);", "dvec_del2_4x5")
    add("dh_del2_rand = del2(del2_rand, 0.5);", "dh_del2_rand")
    add("d5_del2_rand = 4*del2(del2_rand);", "d5_del2_rand")
    # --- BoundMirror ------------------------------------------------------------------------------------
    for name in F.boundmirror_fixtures():
        add(f"bme_{name} = BoundMirrorExpand({name});", f"bme_{name}")
        add(f"bms_{name} = BoundMirrorShrink(BoundMirrorExpand({name}));", f"bms_{name}")
        add(f"try, bmn_{name} = BoundMirrorEnsure(BoundMirrorExpand({name})); bmnerr_{name} = '';"
            f" catch e, bmn_{name} = []; bmnerr_{name} = e.message; end",
            f"bmn_{name}", f"bmnerr_{name}")
    add("try, bmn_small = BoundMirrorEnsure(bm_2x2); bmn_small_err = ''; "
        "catch e, bmn_small = []; bmn_small_err = e.message; end", "bmn_small", "bmn_small_err")
    # --- gradient2 --------------------------------------------------------------------------------------
    for name in F.gradient2_fixtures():
        add(f"[gx_{name}, gy_{name}] = gradient2({name});", f"gx_{name}", f"gy_{name}")
        add(f"gc_{name} = gradient2({name});", f"gc_{name}")
        add(f"gm_{name} = abs(gradient2({name}));", f"gm_{name}")
    add("[gsx, gsy] = gradient2(g2_9x11, 0.5, 2);", "gsx", "gsy")
    add("[gvx, gvy] = gradient2(g2_3x5, [0 1 3 6 10], [0 2 5]);", "gvx", "gvy")
    add("[gmx, gmy] = gradient(g2_9x11); gdiff = max(max(abs(gmx - gsx*0)));", "gmx", "gmy")
    # --- xconv2 / gaussian ------------------------------------------------------------------------------
    for gname in ("xc_G3", "xc_G4", "xc_G5x3"):
        for iname in ("xc_I", "xc_I2"):
            add(f"xc_{iname}_{gname} = xconv2({iname}, {gname});", f"xc_{iname}_{gname}")
            add(f"cv_{iname}_{gname} = conv2({iname}, {gname}, 'same');", f"cv_{iname}_{gname}")
    for k, s in enumerate((0.5, 1.0, 2.0, 4.0, 5.0)):
        add(f"gmask{k} = gaussianMask(1, {s});", f"gmask{k}")
        add(f"gmask{k}b = gaussianMask(2.5, {s});", f"gmask{k}b")
        add(f"gblur{k} = gaussianBlur(xc_I, {s});", f"gblur{k}")
    add("gblur_u8 = gaussianBlur(uint8(xc_I2), 2);", "gblur_u8")
    # --- snakeindex -------------------------------------------------------------------------------------
    for name in F.snakeindex_fixtures():
        add(f"si_{name} = snakeindex({name});", f"si_{name}")
    # --- snakeinterp ------------------------------------------------------------------------------------
    pairs = [("circle", "snk_circle_x", "snk_circle_y"), ("small", "snk_small_x", "snk_small_y"),
             ("square", "snk_square_x", "snk_square_y"), ("dup", "snk_dup_x", "snk_dup_y"),
             ("star", "snk_star_x", "snk_star_y"), ("n8", "snk_n8_x", "snk_n8_y")]
    for tag, xn, yn in pairs:
        for j, (dmax, dmin) in enumerate(((1.0, 0.0), (2.0, 0.5), (4.0, 1.0))):
            add(f"try, [sx_{tag}_{j}, sy_{tag}_{j}] = snakeinterp({xn}, {yn}, {dmax}, {dmin});"
                f" serr_{tag}_{j} = ''; catch e, sx_{tag}_{j} = []; sy_{tag}_{j} = []; serr_{tag}_{j} ="
                f" e.message; end", f"sx_{tag}_{j}", f"sy_{tag}_{j}", f"serr_{tag}_{j}")
    # --- snake matrix + snakedeform ---------------------------------------------------------------------
    combos = [(8, 0.05, 0.0), (8, 0.05, 0.1), (8, 0.0, 0.1), (12, 0.05, 0.0), (37, 0.05, 0.0),
              (37, 0.05, 0.1), (126, 0.05, 0.0), (126, 0.1, 0.2), (3, 0.05, 0.0), (4, 0.05, 0.1),
              (5, 0.05, 0.1)]
    for i, (N, al, be) in enumerate(combos):
        add(f"[Amat{i}, invAI{i}] = snake_matrix_ref({N}, {al}, {be}, 1);", f"Amat{i}", f"invAI{i}")
    add("[Amat_g, invAI_g] = snake_matrix_ref(20, 0.05, 0.03, 2.5);", "Amat_g", "invAI_g")
    # one deform run on a controlled field, several iteration counts
    for j, it in enumerate((1, 5, 20)):
        add(f"[dfx{j}, dfy{j}] = snakedeform(snk_circle_x', snk_circle_y', 0.05, 0, 1, 0.5, fld_px, fld_py,"
            f" {it});", f"dfx{j}", f"dfy{j}")
        add(f"[dgx{j}, dgy{j}] = snakedeform(snk_n8_x', snk_n8_y', 0.05, 0.1, 1, 0.5, fld_px, fld_py,"
            f" {it});", f"dgx{j}", f"dgy{j}")
    add("[drx, dry] = snakedeform(snk_circle_x', snk_circle_y', 0.05, 0, 1, 0.5, fld_ring_px, fld_ring_py,"
        " 30);", "drx", "dry")
    # --- interp2('*linear', 0) --------------------------------------------------------------------------
    add("ip_in = interp2(fld_px, snk_circle_x, snk_circle_y, '*linear', 0);", "ip_in")
    add("ip_out = interp2(fld_px, snk_circle_x*3 - 40, snk_circle_y*3 - 20, '*linear', 0);", "ip_out")
    code = "\n".join(lines) + "\n"
    (VERIFY / "unit_code.m").write_text(code, encoding="utf-8")
    return run_ref(code, v, REF / "unit.mat", workdir=SCRATCH, timeout=1800)


# ------------------------------------------------------------------------------------------------------------
# Group 2: GVF fields
# ------------------------------------------------------------------------------------------------------------
def ref_gvf():
    from reference.ch06 import fixtures as F
    lines = [f"load({q(REF / 'inputs.mat')});"]
    v: list[str] = []

    def add(code: str, *names: str) -> None:
        lines.append(code)
        v.extend(names)

    for name in F.gvf_fixtures():
        for it in (1, 2, 5, 30):
            add(f"[u_{name}_{it}, v_{name}_{it}] = GVF({name}, 0.1, {it});",
                f"u_{name}_{it}", f"v_{name}_{it}")
    for it in (100, 250, 500):
        add(f"[u_gvf_u_{it}, v_gvf_u_{it}] = GVF(gvf_u, 0.1, {it});", f"u_gvf_u_{it}", f"v_gvf_u_{it}")
    add("[u_mu, v_mu] = GVF(gvf_u, 0.25, 20);", "u_mu", "v_mu")
    add("[u_mu2, v_mu2] = GVF(gvf_u, 0.02, 20);", "u_mu2", "v_mu2")
    add("[u_it0, v_it0] = GVF(gvf_u, 0.1, 0);", "u_it0", "v_it0")
    # the two book images' own edge maps
    add("I8 = rgb2gray(imread('test8.jpg')); f2_t8 = abs(gradient2(double(I8)));", "I8", "f2_t8")
    add("[u_t8, v_t8] = GVF(f2_t8, 0.1, 150); mag_t8 = sqrt(u_t8.*u_t8 + v_t8.*v_t8);"
        " px_t8 = u_t8 ./ (mag_t8 + 1e-10); py_t8 = v_t8 ./ (mag_t8 + 1e-10);",
        "u_t8", "v_t8", "px_t8", "py_t8")
    add("Ia = rgb2gray(imread('alg_seg_gray.jpg')); f2_a = abs(gradient2(double(Ia)));", "Ia", "f2_a")
    add("[u_a, v_a] = GVF(f2_a, 0.1, 500); mag_a = sqrt(u_a.*u_a + v_a.*v_a);"
        " px_a = u_a ./ (mag_a + 1e-10); py_a = v_a ./ (mag_a + 1e-10);", "u_a", "v_a", "px_a", "py_a")
    # Fig. 6.16: 110 x 186 binary with a 61-px and a 9-px circle, GVF at 5/30/100/250 iterations
    add("[YY, XX] = ndgrid(1:110, 1:186); c1 = ((XX - 60).^2 + (YY - 55).^2) <= 30.5^2;"
        " c2 = ((XX - 150).^2 + (YY - 55).^2) <= 4.5^2; fig616 = double(c1 | c2);"
        " f616 = abs(gradient2(fig616));", "fig616", "f616")
    for it in (5, 30, 100, 250):
        add(f"[u616_{it}, v616_{it}] = GVF(f616, 0.1, {it});", f"u616_{it}", f"v616_{it}")
    code = "\n".join(lines) + "\n"
    (VERIFY / "gvf_code.m").write_text(code, encoding="utf-8")
    return run_ref(code, v, REF / "gvf.mat", workdir=SCRATCH, timeout=2400)


# ------------------------------------------------------------------------------------------------------------
# Group 3: regionprops
# ------------------------------------------------------------------------------------------------------------
RP_PROPS = ("Area", "Centroid", "BoundingBox", "ConvexArea", "Solidity", "MajorAxisLength",
            "MinorAxisLength", "Eccentricity", "Orientation", "Perimeter")


def _rp_block(src: str, tag: str, v: list[str], label: bool = False) -> str:
    """MATLAB code computing every property of ``src`` and flattening the struct array into matrices.

    ``label=True`` passes ``src`` through as a **label matrix** (one struct per label, whatever connectivity
    produced it); ``False`` binarises it, which is what ``regionprops(label == n, ...)`` does in the ch6 code.
    """
    props = ", ".join(f"'{p}'" for p in RP_PROPS)
    call = f"regionprops({src}, {props})" if label else f"regionprops(logical({src}), {props})"
    out = [f"s__ = {call};"]
    for p in RP_PROPS:
        out.append(f"if isempty(s__), {p.lower()}_{tag} = []; else, {p.lower()}_{tag} = cat(1, s__.{p}); end")
        v.append(f"{p.lower()}_{tag}")
    out.append(f"n_{tag} = numel(s__);")
    v.append(f"n_{tag}")
    return "\n".join(out)


def _loop_form(src: str, tag: str, v: list[str]) -> str:
    """The shipped ch6 call form: ``for n = 1:num, regionprops(label == n, prop)``, accumulated per property."""
    nprops = ", ".join(f"'{q_}'" for q_ in RP_PROPS)
    out = [f"nn__ = max({src}(:));"]
    for pname in RP_PROPS:
        out.append(f"loop{pname}_{tag} = [];")
    out.append("for n__ = 1:nn__")
    out.append(f"  ss__ = regionprops({src} == n__, {nprops});")
    for pname in RP_PROPS:
        out.append(f"  loop{pname}_{tag} = [loop{pname}_{tag}; cat(1, ss__.{pname})];")
    out.append("end")
    v.extend([f"loop{pname}_{tag}" for pname in RP_PROPS])
    return "\n".join(out)


def ref_regionprops():
    from reference.ch06 import fixtures as F
    lines = [f"load({q(REF / 'inputs.mat')});"]
    v: list[str] = []

    def add(code: str, *names: str) -> None:
        lines.append(code)
        v.extend(names)

    for name in F.shape_fixtures():
        lines.append(_rp_block(name, name, v))
    # the 12x14 probe: full precision of the two disputed axis lengths + the convex image / hull
    add("sp__ = regionprops(logical(sh_probe), 'MajorAxisLength', 'MinorAxisLength', 'ConvexImage',"
        " 'ConvexHull', 'Image', 'PixelList');"
        " probe_major = sp__(1).MajorAxisLength; probe_minor = sp__(1).MinorAxisLength;"
        " probe_convimg = sp__(1).ConvexImage; probe_hull = sp__(1).ConvexHull;"
        " probe_img = sp__(1).Image; probe_pixels = sp__(1).PixelList;"
        " probe_major_str = sprintf('%.17g', probe_major);"
        " probe_minor_str = sprintf('%.17g', probe_minor);",
        "probe_major", "probe_minor", "probe_convimg", "probe_hull", "probe_img", "probe_pixels",
        "probe_major_str", "probe_minor_str")
    # ConvexImage / ConvexHull for a few more shapes
    for name in ("sh_L", "sh_peanut", "sh_ring", "sh_diag", "sh_single", "sh_hline"):
        add(f"sc__ = regionprops(logical({name}), 'ConvexImage', 'ConvexHull');"
            f" ci_{name} = sc__(1).ConvexImage; ch_{name} = sc__(1).ConvexHull;",
            f"ci_{name}", f"ch_{name}")
    # multi-component: the label matrix vs the ch6 per-component loop form
    for name in ("sh_multi", "sh_rand0", "sh_rand3"):
        for conn in (4, 8):
            lines.append(f"L__ = bwlabel(logical({name}), {conn});")
            lines.append(_rp_block("L__", f"{name}_L{conn}", v, label=True))
            lines.append(_loop_form("L__", f"{name}_{conn}", v))
    # the real ch6 masks: alg_seg_gray and test8 Otsu masks, 4-connected labelling (the shipped call)
    for tag, fname in (("alg", "alg_seg_gray.jpg"), ("t8", "test8.jpg")):
        lines.append(f"I__ = rgb2gray(imread('{fname}')); bw__ = im2bw(I__, graythresh(I__));"
                     f" L__ = bwlabel(bw__, 4); lab_{tag} = L__; bwm_{tag} = bw__;")
        v.extend([f"lab_{tag}", f"bwm_{tag}"])
        lines.append(_rp_block("L__", f"real_{tag}", v, label=True))
        lines.append(_loop_form("L__", tag, v))
    code = "\n".join(lines) + "\n"
    (VERIFY / "regionprops_code.m").write_text(code, encoding="utf-8")
    return run_ref(code, v, REF / "regionprops.mat", workdir=SCRATCH, timeout=3600)


# ------------------------------------------------------------------------------------------------------------
# Group 4: polygon geometry (polygeom, minboundrect, poly2mask/roipoly, polyxpoly, polybool, convhull)
# ------------------------------------------------------------------------------------------------------------
def ref_geom():
    lines = [f"load({q(REF / 'inputs.mat')});"]
    v: list[str] = []

    def add(code: str, *names: str) -> None:
        lines.append(code)
        v.extend(names)

    # --- polygeom ---------------------------------------------------------------------------------------
    names = ["pg_test", "pg_cw", "pg_tri", "pg_sq", "pg_bow"] + [f"pg_rand{k}" for k in range(10)]
    for nm in names:
        add(f"[geom_{nm}, iner_{nm}, cpmo_{nm}] = polygeom({nm}_x, {nm}_y);",
            f"geom_{nm}", f"iner_{nm}", f"cpmo_{nm}")
    # --- minboundrect -----------------------------------------------------------------------------------
    # DEVIATION (recorded): the shipped 2007 File-Exchange minboundrect.m calls `convhull(x,y,{'Qt'})`,
    # which R2025a rejects ("CONVHULL no longer supports or requires Qhull-specific options").  The original
    # is probed once so the report can quote MATLAB's own refusal; the references come from
    # `minboundrect_ref.m`, a verbatim copy whose only change is dropping that obsolete option.
    add("try, [z1, z2, z3, z4] = minboundrect(pc_rand0_x(:), pc_rand0_y(:), 'a'); mbr_orig_err = '';"
        " catch e, z1 = []; z2 = []; z3 = []; z4 = []; mbr_orig_err = e.message; end", "mbr_orig_err")
    pcs = ["pc_one", "pc_two", "pc_three", "pc_collinear", "pc_dup", "pc_square"] + \
          [f"pc_rand{k}" for k in range(6)]
    for nm in pcs:
        for met in ("a", "p"):
            add(f"try, [rx_{nm}_{met}, ry_{nm}_{met}, ar_{nm}_{met}, pe_{nm}_{met}] ="
                f" minboundrect_ref({nm}_x(:), {nm}_y(:), '{met}'); mbrerr_{nm}_{met} = '';"
                f" catch e, rx_{nm}_{met} = []; ry_{nm}_{met} = []; ar_{nm}_{met} = NaN;"
                f" pe_{nm}_{met} = NaN; mbrerr_{nm}_{met} = e.message; end",
                f"rx_{nm}_{met}", f"ry_{nm}_{met}", f"ar_{nm}_{met}", f"pe_{nm}_{met}", f"mbrerr_{nm}_{met}")
    add("rng(7); pts = rand(50000, 2); [rxu, ryu, aru, peu] ="
        " minboundrect_ref(pts(:,1), pts(:,2), 'a');", "rxu", "ryu", "aru", "peu")
    add("[rxd, ryd, ard, ped] = minboundrect_ref(pc_rand0_x(:), pc_rand0_y(:));",
        "rxd", "ryd", "ard", "ped")
    # --- convhull ---------------------------------------------------------------------------------------
    for k in range(3):
        add(f"kh{k} = convhull(pc_rand{k}_x, pc_rand{k}_y); khs{k} ="
            f" convhull(pc_rand{k}_x, pc_rand{k}_y, 'simplify', true);", f"kh{k}", f"khs{k}")
    # --- poly2mask / roipoly ----------------------------------------------------------------------------
    pms = [f"pm{k}" for k in range(5)] + ["pm_tri", "pm_int", "pm_half"]
    for nm in pms:
        add(f"mask_{nm} = poly2mask({nm}_x, {nm}_y, 21, 21);", f"mask_{nm}")
        add(f"rmask_{nm} = roipoly(zeros(21, 21), {nm}_x, {nm}_y);", f"rmask_{nm}")
        add(f"mask2_{nm} = poly2mask({nm}_x, {nm}_y, 15, 25);", f"mask2_{nm}")
    add("mask_sq = poly2mask(pg_sq_x, pg_sq_y, 12, 12);", "mask_sq")
    add("mask_hull = poly2mask(ch_probe_x, ch_probe_y, 7, 9);", "mask_hull")
    lines.insert(1, "sp__ = regionprops(logical(sh_probe), 'ConvexHull'); h__ = sp__(1).ConvexHull;"
                    " ch_probe_x = h__(:,1) - 2; ch_probe_y = h__(:,2) - 2;")
    v.extend(["ch_probe_x", "ch_probe_y"])
    # --- polybool ---------------------------------------------------------------------------------------
    add("s1 = 30; s2 = 40; s_1 = [0, s1, s1, 0]; s_2 = [0, 0, s2, s2];", "s_1", "s_2")
    for k in range(7):
        add(f"try, [pbx{k}, pby{k}] = polybool('intersection', s_2, s_1, pb{k}_x, pb{k}_y);"
            f" pberr{k} = ''; catch e, pbx{k} = []; pby{k} = []; pberr{k} = e.message; end"
            f"\npbsz{k} = size(pbx{k});", f"pbx{k}", f"pby{k}", f"pberr{k}", f"pbsz{k}")
        add(f"if isempty(pbx{k}), pbm{k} = false(30, 40); else,"
            f" pbm{k} = poly2mask(pbx{k}, pby{k}, 30, 40); end", f"pbm{k}")
    # the same clip after snakeinterp (the actual ch6 call order)
    add("[cxi, cyi] = snakeinterp(pb1_x, pb1_y, 1, 0); [pbix, pbiy] ="
        " polybool('intersection', s_2, s_1, cxi, cyi); pbi_sz = size(pbix);",
        "cxi", "cyi", "pbix", "pbiy", "pbi_sz")
    # --- polyxpoly --------------------------------------------------------------------------------------
    add("[xi12, yi12, ii12] = polyxpoly(px1_x, px1_y, px2_x, px2_y);", "xi12", "yi12", "ii12")
    add("[xi13, yi13, ii13] = polyxpoly(px1_x, px1_y, px3_x, px3_y);", "xi13", "yi13", "ii13")
    add("[xi11, yi11, ii11] = polyxpoly(px1_x, px1_y, px1_x, px1_y);", "xi11", "yi11", "ii11")
    code = "\n".join(lines) + "\n"
    (VERIFY / "geom_code.m").write_text(code, encoding="utf-8")
    return run_ref(code, v, REF / "geom.mat", workdir=SCRATCH, timeout=1800)


# ------------------------------------------------------------------------------------------------------------
# Group 5: for test/dist.m (verbatim, transposed frame)
# ------------------------------------------------------------------------------------------------------------
def ref_dist_script():
    code = "\n".join([
        "run('ch6_dist_script.m');",
        _imw("bw", "dist_bw.png"),
        _imw("dis", "dist_dis.png"),
        _imw("dis1", "dist_dis1.png"),
        _imw("Dis_img", "dist_Dis_img.png"),
        _imw("img_Dist", "dist_img_Dist.png", autoscale=True),
        _imw("rgb", "dist_label2rgb.png"),
        # NOTE: the script's eight `figure` windows carry 540 seeds x 126-point circles each; `print`ing them
        # never finished inside the 30-min budget (first attempt killed at 1800 s, images already on disk), so
        # only the raw `imwrite` images are kept.  The L3 comparison uses those.
        "close all;",
    ])
    v = ["I", "bw", "img_Dist", "Dis_img", "dis", "dis1", "p", "q", "label", "num_out", "CEN1", "CEN2",
         "R0", "X0", "Y0", "l", "l_num", "t"]
    return run_ref(code, v, REF / "dist_script.mat", workdir=SCRATCH, timeout=1800)


# ------------------------------------------------------------------------------------------------------------
# Group 6: for test/for_test.m
# ------------------------------------------------------------------------------------------------------------
def ref_for_test():
    code = "\n".join([
        "run('for_test_ref.m');",
        _imw("bw_top", "ft_bw.png"),
        _imw("bw_burn", "ft_bw_burn.png"),
        _imw("f2", "ft_f2.png", autoscale=True),
        "hs__ = findobj(0, 'Type', 'figure'); if ~isempty(hs__), [~, ix__] = sort([hs__.Number]);"
        " hs__ = hs__(ix__); for k__ = 1:numel(hs__),"
        f" print(hs__(k__), '-dpng', '-r100', fullfile({q(ML_IMG)}, sprintf('ft_fig%d.png', k__)));"
        " end; end; close all;",
    ])
    v = ["I", "bw_top", "f2", "u_gvf", "v_gvf", "px", "py", "xSpace", "ySpace", "qx", "qy", "XPB",
         "YPB", "SZPB",
         "XI0", "YI0", "XB", "YB", "xx_out", "yy_out", "bw_burn", "bw_pre", "s1", "s2"]
    return run_ref(code, v, REF / "for_test.mat", workdir=SCRATCH, timeout=1800)


# ------------------------------------------------------------------------------------------------------------
# Group 7: GVF_distance on alg_seg_gray.jpg with the sea_ice_demo parameters (Fig. 6.15)
# ------------------------------------------------------------------------------------------------------------
GVF_ALG_CALL = ("I = imread('alg_seg_gray.jpg');\n"
                "se = strel('disk', 3);\n"
                "tic; [bw1, REC] = gvf_distance_ref(I, 0, 1, 1, 500, 0.1, 100, 0.05, 0, 1, 0.5, 0, 1,"
                " 10, 2500, 0.9, 2, se, 1); elapsed = toc;\n")

REC_UNPACK = ("gray = REC.gray; level = REC.level; bw = REC.bw; f2 = REC.f2; u = REC.u; v = REC.v;"
              " px = REC.px; py = REC.py;\n"
              "label = REC.label; num = REC.num; a = REC.a; rc = REC.rc; l = REC.l; w = REC.w; rl = REC.rl;"
              " k = REC.k; bw2 = REC.bw2;\n"
              "img_Dist = REC.img_Dist; Dis_img = REC.Dis_img; dis = REC.dis; label1 = REC.label1;"
              " num1 = REC.num1; cen = REC.cen; r = REC.r;\n"
              "XI = REC.XI; YI = REC.YI; XC = REC.XC; YC = REC.YC; XF = REC.XF; YF = REC.YF;"
              " XX = REC.XX; YY = REC.YY;\n")


def ref_gvf_alg():
    code = "\n".join([
        GVF_ALG_CALL, REC_UNPACK,
        _imw("bw", "alg_bw.png"),
        _imw("bw1", "alg_bw1.png"),
        _imw("bw2", "alg_bw2.png"),
        _imw("Dis_img", "alg_maxima.png"),
        _imw("dis", "alg_seeds.png"),
        _imw("img_Dist", "alg_dist.png", autoscale=True),
        _imw("gray", "alg_gray.png"),
    ])
    v = ["bw1", "gray", "level", "bw", "f2", "u", "v", "px", "py", "label", "num", "a", "rc", "l", "w", "rl",
         "k", "bw2", "img_Dist", "Dis_img", "dis", "label1", "num1", "cen", "r", "XI", "YI", "XC", "YC",
         "XF", "YF", "XX", "YY", "elapsed"]
    return run_ref(code, v, REF / "gvf_alg.mat", workdir=SCRATCH, timeout=10800)


# ------------------------------------------------------------------------------------------------------------
# Group 8: seaice_kmean_GVF_forenhancement on alg_seg_gray.jpg (rng(0) before the kmeans call)
# ------------------------------------------------------------------------------------------------------------
def ref_kmean_alg():
    code = "\n".join([
        "I = imread('alg_seg_gray.jpg');",
        "se = strel('disk', 3);",
        "rng(0);",
        "tic; [out, bk, REC] = kmean_gvf_ref(I, 3, 0, 1, 1, 500, 0.1, 100, 0.05, 0, 1, 0.5, 0, 1,"
        " 10, 2500, 0.9, 2, se, 1); elapsed = toc;",
        "bw = REC.bw; f2 = REC.f2; u = REC.u; v = REC.v; px = REC.px; py = REC.py;",
        "map0 = REC.map0; s0 = REC.s0; ind0 = REC.ind0; bw0_raw = REC.bw0_raw; bw0 = REC.bw0;",
        "n_negative = REC.n_negative; bw1 = REC.bw1; bw0_final = REC.bw0_final;",
        "lab1 = REC.label1; num_1 = REC.num1; a1 = REC.a1; rc1 = REC.rc1; l1 = REC.l1; w1 = REC.w1;"
        " rl1 = REC.rl1; k1v = REC.k1; cen1 = REC.cen1; r1 = REC.r1;",
        "lab2 = REC.label2; num_2 = REC.num2; a2 = REC.a2; rc2 = REC.rc2; l2 = REC.l2; w2 = REC.w2;"
        " rl2 = REC.rl2; k2v = REC.k2; cen2 = REC.cen2; r2 = REC.r2;",
        "imgDist1 = REC.imgDist1; Dis1 = REC.Dis1; dis1_ = REC.dis1; imgDist2 = REC.imgDist2;"
        " Dis2 = REC.Dis2; dis2_ = REC.dis2; bw2_1 = REC.bw2_1; bw2_2 = REC.bw2_2;",
        # cluster-stability probe: 20 restarts of kmeans on the same data
        "ima = double(reshape(rgb2gray(I), [], 1)); C = zeros(20, 3);",
        "for it = 1:20, rng(it); [~, cc] = kmeans(ima, 3, 'EmptyAction', 'singleton');"
        " C(it, :) = sort(cc)'; end",
        _imw("bk", "kmean_bk.png"),
        _imw("bw0", "kmean_bw0.png"),
        _imw("bw1", "kmean_bw1.png"),
        "tmp__ = mat2gray(out); " + _imw("tmp__", "kmean_out.png").replace("imwrite(tmp__", "imwrite(tmp__"),
    ])
    v = ["out", "bk", "bw", "f2", "px", "py", "map0", "s0", "ind0", "bw0_raw", "bw0", "n_negative", "bw1",
         "bw0_final", "lab1", "num_1", "a1", "rc1", "l1", "w1", "rl1", "k1v", "cen1", "r1", "lab2", "num_2",
         "a2", "rc2", "l2", "w2", "rl2", "k2v", "cen2", "r2", "imgDist1", "Dis1", "dis1_", "imgDist2",
         "Dis2", "dis2_", "bw2_1", "bw2_2", "C", "elapsed"]
    return run_ref(code, v, REF / "kmean_alg.mat", workdir=SCRATCH, timeout=14400)


# ------------------------------------------------------------------------------------------------------------
# Group 9: the stages of sea_ice_demo.m on sea_ice_test.jpg (no snake loop -- too slow in MATLAB)
# ------------------------------------------------------------------------------------------------------------
def ref_kmean_stage():
    code = "\n".join([
        "I0 = imread('sea_ice_test.jpg'); I = rgb2gray(I0);",
        "level = graythresh(I); bw = im2bw(I, level); [s1, s2] = size(bw);",
        "f2 = abs(gradient2(double(I)));",
        "tic; [u, v] = GVF(f2, 0.1, 500); gvf_time = toc;",
        "mag = sqrt(u.*u + v.*v); px = u ./ (mag + 1e-10); py = v ./ (mag + 1e-10);",
        "[label, num] = bwlabel(bw, 4);",
        "a = zeros(num,1); rc = zeros(num,1); l = zeros(num,1); w = zeros(num,1);",
        "for n = 1:num",
        "  aa = regionprops(label == n, 'Area'); a(n) = cat(1, aa.Area);",
        "  rcc = regionprops(label == n, 'Solidity'); rc(n) = cat(1, rcc.Solidity);",
        "  ll = regionprops(label == n, 'MajorAxisLength'); l(n) = cat(1, ll.MajorAxisLength);",
        "  ww = regionprops(label == n, 'MinorAxisLength'); w(n) = cat(1, ww.MinorAxisLength);",
        "end",
        "rl = l ./ w; k = unique([find(a > 2500); find(rc < 0.9); find(rl > 2)]);",
        "bw2 = zeros(s1, s2); for m = 1:length(k), bw2(find(label == k(m))) = 1; end",
        "bw2 = bwareaopen(bw2, 10);",
        "img_Dist = bwdist(~bw2, 'cityblock'); imgDist = -img_Dist; imgDist(~bw2) = -inf;",
        "Dis_img = imregionalmin(imgDist); dis = Dis_img .* bw2; se = strel('disk', 3);",
        "dis = imdilate(dis, se); [label1, num1] = bwlabel(dis, 8);",
        "cenA = zeros(num1, 2); rA = zeros(num1, 1); t = 0:0.05:6.28;",
        "for n1 = 1:num1",
        "  cn = regionprops(label1 == n1, 'centroid'); cn = cat(1, cn.Centroid); cenA(n1,:) = cn;",
        "  rr = img_Dist(round(cn(2)), round(cn(1))) / sqrt(2); if rr == 0, rr = 2; end; rA(n1) = rr;",
        "end",
        "ima = double(I(:)); rng(0); map0 = kmeans(ima, 3, 'EmptyAction', 'singleton');",
        "s0 = zeros(1,3); for i = 1:3, pp = zeros(size(map0)); pp(map0 == i) = 1;"
        " s0(i) = sum(ima.*pp)/sum(pp); end",
        "[A0, ind0] = sort(s0); bw_kmeans = ones(size(map0)); bw_kmeans(find(map0 == ind0(1))) = 0;",
        "bk = double(reshape(bw_kmeans, size(I,1), size(I,2)));",
        "bw0_raw = bk - bw; n_negative = sum(sum(bw0_raw < 0));",
        "bw0 = bwareaopen(bw0_raw, 10, 4);",
        _imw("bw", "sit_bw.png"),
        _imw("bk", "sit_bk.png"),
        _imw("bw0", "sit_bw0.png"),
        _imw("bw2", "sit_bw2.png"),
        _imw("dis", "sit_seeds.png"),
    ])
    v = ["I0", "I", "level", "bw", "f2", "u", "v", "px", "py", "label", "num", "a", "rc", "l", "w", "rl",
         "k", "bw2", "img_Dist", "Dis_img", "dis", "label1", "num1", "cenA", "rA", "map0", "s0", "ind0",
         "bk", "bw0_raw", "bw0", "n_negative", "gvf_time"]
    return run_ref(code, v, REF / "kmean_stage.mat", workdir=SCRATCH, timeout=10800)


# ------------------------------------------------------------------------------------------------------------
# Group 10: extras -- homofil.m, bwperim, regional maxima, Fig. 6.14/6.16 fixtures, contour-init recipe
# ------------------------------------------------------------------------------------------------------------
def ref_extra():
    """Second fixture file (inputs2.mat) written from the PORT's own synthetic fixtures, so the book-figure
    reproductions (Fig. 6.14, Fig. 6.16) and the remaining primitives get an L2 reference too."""
    import numpy as _np
    from seaice.core import synth as _synth
    rng = _np.random.default_rng(609)
    fx2 = {
        "fig616": _synth.fig_6_16_circles().astype(_np.uint8),
        "ushape": _synth.u_shape().astype(_np.uint8),
        "fig614": _np.asarray(_synth.FIG_6_14_IMAGE, dtype=_np.uint8),
        "floefield": _synth.synthetic_floe_field(),
        "hf_im": (rng.random((32, 40)) * 255.0),
        "hf_im2": _np.arange(1, 32 * 40 + 1, dtype=_np.float64).reshape(32, 40) % 251,
    }
    from reference.ch06.fixtures import shape_fixtures as _sf
    for k, vv in _sf().items():
        fx2["p_" + k] = vv
    savemat(str(REF / "inputs2.mat"), fx2, do_compression=True)

    lines = [f"load({q(REF / 'inputs2.mat')});"]
    v: list[str] = []

    def add(code: str, *names: str) -> None:
        lines.append(code)
        v.extend(names)

    # --- GVF of the port's own Fig. 6.16(a) and U-shape ------------------------------------------------
    add("f616p = abs(gradient2(double(fig616)));", "f616p")
    for it in (5, 30, 100, 250):
        add(f"[u616p_{it}, v616p_{it}] = GVF(f616p, 0.1, {it});", f"u616p_{it}", f"v616p_{it}")
    add("fup = abs(gradient2(double(ushape)));", "fup")
    for it in (30, 250):
        add(f"[uup_{it}, vup_{it}] = GVF(fup, 0.1, {it});", f"uup_{it}", f"vup_{it}")
    add("gbu = gaussianBlur(double(ushape), 4); [gbx, gby] = gradient2(gbu);"
        " eext = -(gbx.^2 + gby.^2);", "gbu", "gbx", "gby", "eext")
    # --- Fig. 6.14: the printed 8x8 matrices and the seed recipe ---------------------------------------
    add("A614 = logical(fig614); D614 = bwdist(~A614, 'cityblock'); Dn = -D614; Dn(~A614) = -inf;"
        " M614 = imregionalmin(Dn); dis614 = M614 .* A614; dis614d = imdilate(dis614, strel('disk', 3));"
        " [L614, n614] = bwlabel(dis614d, 8);"
        " cc614 = regionprops(L614, 'Centroid'); cen614 = cat(1, cc614.Centroid);"
        " rmax614 = imregionalmax(double(D614) .* A614);",
        "A614", "D614", "M614", "dis614", "dis614d", "L614", "n614", "cen614", "rmax614")
    # --- bwperim (used by regionprops's ConvexHull) ----------------------------------------------------
    for k in ("p_sh_probe", "p_sh_ring", "p_sh_L", "p_sh_disc", "p_sh_border", "p_sh_rand0", "p_sh_diag",
              "p_sh_single", "p_sh_multi"):
        add(f"bp4_{k} = bwperim(logical({k}), 4); bp8_{k} = bwperim(logical({k}), 8);",
            f"bp4_{k}", f"bp8_{k}")
    # --- regional maxima by reconstruction (Eqs. 6.57 / 6.58) ------------------------------------------
    add("Dff = bwdist(~logical(p_sh_peanut), 'cityblock'); rmx = imregionalmax(Dff);"
        " rc57 = double(Dff) - imreconstruct(double(Dff) - 1, double(Dff));"
        " rc58 = double(Dff) + 1 - imreconstruct(double(Dff), double(Dff) + 1);"
        " Dfe = bwdist(~logical(p_sh_peanut)); rmxe = imregionalmax(Dfe);",
        "Dff", "rmx", "rc57", "rc58", "Dfe", "rmxe")
    add("Frnd = double(floefield); rmxf = imregionalmax(Frnd);"
        " rcf = Frnd - imreconstruct(Frnd - 1, Frnd);", "Frnd", "rmxf", "rcf")
    # --- homofil.m (the orphan homomorphic Butterworth filter) -----------------------------------------
    for k, (dd, nn) in enumerate(((10.0, 1.0), (30.0, 2.0), (5.0, 4.0))):
        add(f"hf{k} = homofil(hf_im, {dd}, 32, 40, {nn});", f"hf{k}")
        add(f"hg{k} = homofil(hf_im2, {dd}, 32, 40, {nn});", f"hg{k}")
    add("hfu = homofil(double(uint8(hf_im)), 20, 32, 40, 2);", "hfu")
    # --- the contour-initialisation recipe on the port's own floe field --------------------------------
    add("Igf = double(floefield); lvl = graythresh(uint8(Igf)); bwf = im2bw(uint8(Igf), lvl);"
        " Df = bwdist(~bwf, 'cityblock'); Dnf = -Df; Dnf(~bwf) = -inf; Mf = imregionalmin(Dnf);"
        " disf = Mf .* bwf; sef = strel('disk', 3); disf = imdilate(disf, sef);"
        " [Lf, nf] = bwlabel(disf, 8); cf = regionprops(Lf, 'Centroid'); cenf = cat(1, cf.Centroid);"
        " rf = zeros(nf, 1); for n1 = 1:nf, cn = cenf(n1,:);"
        " rr = Df(round(cn(2)), round(cn(1))) / sqrt(2); if rr == 0, rr = 2; end; rf(n1) = rr; end",
        "bwf", "lvl", "Df", "Mf", "disf", "Lf", "nf", "cenf", "rf")
    code = "\n".join(lines) + "\n"
    (VERIFY / "extra_code.m").write_text(code, encoding="utf-8")
    return run_ref(code, v, REF / "extra.mat", workdir=SCRATCH, timeout=2400)


# ------------------------------------------------------------------------------------------------------------
# Group 11: strel -- the T_seed / merging structuring elements (analysis Sec. 2.2 claims 7x7/37 px for disk 3)
# ------------------------------------------------------------------------------------------------------------
def ref_strel():
    code = ("se3 = getnhood(strel('disk',3)); se5 = getnhood(strel('disk',5));"
            " n3 = nnz(se3); n5 = nnz(se5); s3 = size(se3); s5 = size(se5);")
    return run_ref(code, ["se3", "se5", "n3", "n5", "s3", "s5"], REF / "strel.mat", workdir=SCRATCH)


ALL = {
    "unit": ref_unit,
    "gvf": ref_gvf,
    "regionprops": ref_regionprops,
    "geom": ref_geom,
    "dist_script": ref_dist_script,
    "for_test": ref_for_test,
    "gvf_alg": ref_gvf_alg,
    "kmean_stage": ref_kmean_stage,
    "kmean_alg": ref_kmean_alg,
    "extra": ref_extra,
    "strel": ref_strel,
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
        except Exception as exc:
            msg = str(exc)
            log[name] = {"status": "error", "error": msg[:4000]}
            print(f"  ERROR: {msg[:3000]}", flush=True)
            rc = 1
        log_path.write_text(json.dumps(log, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
