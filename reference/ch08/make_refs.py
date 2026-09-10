"""Generate the MATLAB (R2025a) reference outputs for chapter 8 (sea ice image processing applications).

Runs the ORIGINAL ``.m`` files of ``MATLAB_ROOT/ch8/MCD/`` and the four ``Sea_Ice_Floe_Identification`` files
that ch07 deferred here, through ``tools.run_matlab_ref.run_ref`` (``matlab -batch``, figures invisible,
``save(..., '-v7')``), and writes ``reference/ch08/*.mat``.

Usage::

    .venv/Scripts/python.exe reference/ch08/make_refs.py [name ...]
    names = misc, mcd, fit, orphan, colorhist, model, window

Scratch policy (CLAUDE.md rule 8: nothing under ``MATLAB_ROOT`` is touched)
--------------------------------------------------------------------------
Every ``.m`` file is **copied** into ``outputs/ch08/verify/scratch/`` under a non-shadowing name and MATLAB runs
with that folder as cwd; the ch6/ch7/ch8 folders are never put on the MATLAB path (they ship ``polygeom.m`` and
``minboundrect.m``, and ``ch3/kmeans.m`` shadows the toolbox).  The copies are **line-for-line the originals**
except for the patches listed in ``reference/ch08/patches.json``, which are produced by :func:`_patch` and
recorded there verbatim (the removed source line is stored beside the replacement).  Three classes of patch,
and nothing else:

1. **hard-coded ``cd``/``load``** — ``main_WL_new.m`` line 5 (``E:\\NTNU\\CRC\\latex\\matlab\\ch8\\MCD``) and
   ``fitting_iceFloes_distribution.m`` line 5 (``C:\\Users\\qinz\\Desktop\\sent_to_Qin``) do not exist on this
   host; the two ``load`` lines are re-pointed at ``data/book/ch08/MCD/``.
2. **graphics** — every ``figure``/``imshow``/``fill``/``plot``/``line``/``bar``/``colormap``/``colorbar``/
   ``axis``/``xlabel``/``set(gcf|gca|h(i))`` statement.  ``color_hist.m`` lines 23–25/31/33 and
   ``color_hist_comparison.m`` lines 32–34/40/42 are **HG1** (``get(h,'Children')`` →
   ``GraphicsPlaceholder``), which *errors* in R2025a, so those files cannot run at all unpatched.  Where a
   graphics statement consumed a number that is evidence (``caxis`` limits, the ``fill`` colour, the white dot
   coordinates), the number is **assigned to a variable and saved** instead of being dropped.
3. **output plumbing** — function headers renamed (non-shadowing) and their output lists extended with the
   intermediates.  No numeric literal or expression is changed anywhere.
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
from reference.ch08 import fixtures as FX  # noqa: E402

MATLAB_ROOT = ROOT / "K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab"
MCD = MATLAB_ROOT / "ch8/MCD"
SIFI = MATLAB_ROOT / "ch7/Sea_Ice_Floe_Identification"
REF = ROOT / "reference/ch08"
VERIFY = ROOT / "outputs/ch08/verify"
SCRATCH = VERIFY / "scratch"
DATA = ROOT / "data/book/ch08/MCD"

PATCHES: list[dict] = []


def q(p) -> str:
    return "'" + str(p).replace("\\", "/").replace("'", "''") + "'"


def _read_lines(p: Path) -> list[str]:
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


def _patch(src: Path, dst_name: str, edits: dict[int, str | None], note: str) -> Path:
    """Copy ``src`` to ``SCRATCH/dst_name`` applying 1-based line ``edits`` (``None`` = delete the line).

    Every removed / replaced line is recorded verbatim in ``patches.json`` together with its replacement, so
    the report can show exactly what was changed and a reader can check that no numeric expression moved.
    """
    lines = _read_lines(src)
    record = []
    out = []
    for i, line in enumerate(lines, start=1):
        if i in edits:
            new = edits[i]
            record.append({"line": i, "removed": line, "replaced_by": new})
            if new is not None:
                out.append(new)
        else:
            out.append(line)
    missing = [i for i in edits if i > len(lines)]
    if missing:
        raise RuntimeError(f"{src.name}: patch lines {missing} beyond EOF ({len(lines)} lines)")
    dst = SCRATCH / dst_name
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    PATCHES.append({"source": src.name, "target": dst_name, "note": note, "edits": record})
    return dst


# ==============================================================================================================
# Scratch preparation — the patched copies
# ==============================================================================================================

def prepare_scratch() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    REF.mkdir(parents=True, exist_ok=True)
    PATCHES.clear()

    # ---- verbatim copies (no patch at all) -------------------------------------------------------------------
    for name in ("PowerLaw_fitting_method_and_plotting.m", "three_fitting_method_and_plotting.m"):
        shutil.copyfile(MCD / name, SCRATCH / name)
    shutil.copyfile(SIFI / "polygeom.m", SCRATCH / "polygeom.m")
    shutil.copyfile(SIFI / "SeaIce_Image_Structure.m", SCRATCH / "ch08_struct_ref.m")
    PATCHES.append({"source": "SeaIce_Image_Structure.m", "target": "ch08_struct_ref.m",
                    "note": "verbatim copy, renamed only (a script called `SeaIce_Image_Structure` would be "
                            "fine, but every scratch file is renamed by policy)", "edits": []})
    PATCHES.append({"source": "PowerLaw_fitting_method_and_plotting.m / "
                              "three_fitting_method_and_plotting.m / polygeom.m",
                    "target": "(same name)", "note": "verbatim copies, no edit", "edits": []})

    # ---- plot_color_bar_and_floe.m -> ch08_pcbf_ref.m --------------------------------------------------------
    hdr = ("function [counts,histogram_centers,index_all,color_M,centre,rgbImage,polyx,polyy,polyc,dotxy,"
           "caxis_lim]=ch08_pcbf_ref(color_limit_N,MCD,Y_limi,N,Poly_struc,length_over_Pixel)\n"
           "index_all = zeros(N,1); rgbImage = []; polyx = {}; polyy = {}; polyc = []; dotxy = []; centre = [];")
    edits = {2: hdr}
    for i in range(21, 35):                      # figure, hold on, the 100-bar loop, hold off, colormap
        edits[i] = None
    edits[35] = "caxis_lim = [histogram_centers(1) histogram_centers(color_limit_N)];"   # was caxis(...)
    for i in list(range(36, 44)) + [47, 48]:     # colorbar, set(gcf/gca), xlabel, ylabel, grid, axis; figure/hold
        edits[i] = None
    edits[51] = "    [~,index]=min(abs(MCD(i)-histogram_centers));\n    index_all(i) = index;"
    edits[67] = "            polyx{i} = x; polyy{i} = y; polyc(i,:) = color_M(index,:);"
    edits[69] = "            polyx{i} = x; polyy{i} = y; polyc(i,:) = color_M(color_limit_N,:);"
    edits[84] = None                             # imshow(rgbImage) inside the 2888-iteration loop
    for i in range(91, 101):                     # hold off, set(gcf/gca), xlabel, ylabel, axis equal/tight
        edits[i] = None
    edits[104] = None                            # hold on
    edits[106] = ("        dotxy(i,:) = [centre(i,1)*length_over_Pixel,"
                  "Y_limi*length_over_Pixel-centre(i,2)*length_over_Pixel];")
    edits[108] = "        dotxy(i,:) = [centre(i,1),centre(i,2)];"
    for i in (113, 114, 115):                    # colormap, caxis, colorbar
        edits[i] = None
    _patch(MCD / "plot_color_bar_and_floe.m", "ch08_pcbf_ref.m", edits,
           "renamed (non-shadowing) and the output list extended with index_all/color_M/centre/rgbImage/"
           "polyx/polyy/polyc/dotxy/caxis_lim; every graphics statement removed, with the three that carried "
           "numbers turned into assignments: caxis(...) -> caxis_lim, fill(x,y,color_M(...)) -> polyx/polyy/"
           "polyc, plot(...,'w.') -> dotxy.  The rgbImage(y(j),x(j),:) assignment (a computation, not a "
           "drawing) is kept verbatim; only the imshow() inside the loop is dropped (risk R17).  No numeric "
           "expression changed.")

    # ---- main_WL_new.m -> ch08_main_ref.m --------------------------------------------------------------------
    call = ("[{0}_counts,{0}_centers,{0}_index,color_M0,{0}_centre,{0}_rgb,{0}_polyx,{0}_polyy,{0}_polyc,"
            "{0}_dotxy,{0}_caxis]=ch08_pcbf_ref(color_limit_N,{1},Y_limi,N,{2},length_over_Pixel);")
    edits = {
        5: None,                                                     # cd('E:\NTNU\CRC\latex\matlab\ch8\MCD')
        7: f"load({q(DATA / 'IceImage_290915_2_jpg.0000179.mat')})",
        25: call.format("Poly", "Poly_MCD", "Poly_struc"),
        35: call.format("Raw", "Raw_MCD", "Raw_floe_struc"),
    }
    for i in range(44, 58):                                          # figure ... hold off, colormap
        edits[i] = None
    edits[58] = "caxis_lim_err = [Raw_centers(1) Raw_centers(color_limit_N)];"
    for i in range(59, 68):                                          # colorbar, set, xlabel, ylabel, grid, axis
        edits[i] = None
    _patch(MCD / "main_WL_new.m", "ch08_main_ref.m", edits,
           "line 5 cd() to a path that does not exist on this host deleted and line 7's load re-pointed at "
           "data/book/ch08/MCD/ (risk R3); the two plot_color_bar_and_floe calls retargeted at ch08_pcbf_ref "
           "with the extended output list and terminated with a semicolon (the originals end without one, "
           "which only makes MATLAB echo the 100-element vectors); the delta-N bar figure of lines 44-66 "
           "removed, with its caxis limits kept as caxis_lim_err.  count_error (line 41) and every other "
           "numeric line are verbatim.")

    # ---- fitting_iceFloes_distribution.m -> ch08_fit_ref.m ---------------------------------------------------
    src = MCD / "fitting_iceFloes_distribution.m"
    legend_line = _read_lines(src)[56]
    (SCRATCH / "ch08_legend_probe.m").write_text(legend_line + "\n", encoding="utf-8")
    _patch(src, "ch08_fit_ref.m",
           {5: None, 7: f"load({q(DATA / 'MCD_results.mat')})", 57: None},
           "line 5 cd('C:\\Users\\qinz\\Desktop\\sent_to_Qin') deleted and line 7's load re-pointed at "
           "data/book/ch08/MCD/MCD_results.mat (risk R3).  Line 57, the legend call, is deleted because its "
           "stray quote (`'Power law fitting''`) is a SYNTAX ERROR, not a rendering quirk: the shipped file "
           "cannot be parsed by MATLAB at all (the error MATLAB reports is captured as `legend_err` in "
           "fit.mat, from the one-line copy ch08_legend_probe.m).  Everything else, including the loglog "
           "plotting, is verbatim.")

    # ---- color_hist.m -> ch08_colorhist_ref.m ----------------------------------------------------------------
    edits = {14: None, 22: None, 23: None, 24: None, 25: None, 28: None, 31: None, 33: None, 35: None,
             39: None, 40: None, 41: None, 46: None}
    _patch(SIFI / "color_hist.m", "ch08_colorhist_ref.m", edits,
           "graphics only.  Lines 23-25 (`ch = get(h,'Children'); fvd = get(ch,'Faces'); "
           "fvcd = get(ch,'FaceVertexCData')`), 31 (`fvcd(fvd(i,:)) = color(i);`) and 33 "
           "(`set(ch,'FaceVertexCData',fvcd)`) are HG1 and ERROR in R2025a (get(bar,'Children') returns an "
           "empty GraphicsPlaceholder), so the file cannot run unpatched; lines 14 (figure), 22 (bar), 28/35 "
           "(colormap), 39-41 (colorbar) and 46 (set YTickLabel) are the rest of the drawing.  Every quantity "
           "the removed lines consumed is still computed: z, n, color, zs, izs, ysh, YT, color_floe, "
           "color_min, color_max, nbins, k, d.")

    # ---- color_hist_comparison.m -> ch08_colorhistcmp_ref.m --------------------------------------------------
    edits = {17: None, 31: None, 32: None, 33: None, 34: None, 37: None, 40: None, 42: None, 44: None,
             48: None, 49: None, 50: None, 55: None}
    _patch(SIFI / "color_hist_comparison.m", "ch08_colorhistcmp_ref.m", edits,
           "the same graphics-only patch as color_hist.m: the HG1 block is lines 32-34/40/42 here, plus "
           "17 (figure), 31 (bar), 37/44 (colormap), 48-50 (colorbar) and 55 (set YTickLabel).  z, z0, z_d, "
           "n, n0, color, ysh and YT are all still computed and saved.")

    # ---- sea_ice_model.m -> ch08_model_ref.m -----------------------------------------------------------------
    edits = {1: "function [floe, brash, bw_floe, bw_brash, ver, t] = ch08_model_ref( ice_floe, brash_ice, img )"}
    for i in range(94, 102):                    # figure, imshow(bw_floe), the line/plot loop, hold off
        edits[i] = None
    for i in range(159, 169):                   # figure, imshow(bw_brash), the plot loop, hold off
        edits[i] = None
    _patch(SIFI / "sea_ice_model.m", "ch08_model_ref.m", edits,
           "function renamed (non-shadowing) and its output list extended with bw_floe/bw_brash/ver/t; the two "
           "display blocks (lines 94-101 and 159-168: figure, imshow, line, plot, hold) removed.  The whole "
           "computation - convhull, roipoly, polygeom, polyxpoly, the `if xx ~= NaN` tests, the circle "
           "sampling on t = 0:0.05:6.28 and the d < r+r0 test - is verbatim.")

    (REF / "patches.json").write_text(json.dumps(PATCHES, indent=1), encoding="utf-8")
    (VERIFY / "patches.json").write_text(json.dumps(PATCHES, indent=1), encoding="utf-8")


# ==============================================================================================================
# Fixture -> .mat helpers
# ==============================================================================================================

#: The four fields ``sea_ice_model.m`` / ``SeaIce_Image_Structure.m`` read off an ``ice_floe``/``brash_ice``
#: element.  ``scipy.io.savemat`` writes an object array of dicts as a MATLAB *cell* array, not a struct array,
#: so each field is written as its own 1xN cell and MATLAB rebuilds the struct array with
#: ``struct('Center', C, 'Area', A, ...)`` - which is exactly what ch7's ``ice_shape_enhancement.m`` produces.
PIECE_FIELDS = ("Center", "Area", "Perimeter", "PixelsPosition")


def _cells(items, field: str) -> np.ndarray:
    arr = np.empty((1, len(items)), dtype=object)
    for i, d in enumerate(items):
        arr[0, i] = d[field]
    return arr


def _struct_payload(prefix: str, items) -> dict[str, np.ndarray]:
    return {f"{prefix}_{f}": _cells(items, f) for f in PIECE_FIELDS}


def _struct_matlab(var: str, prefix: str) -> str:
    args = ", ".join(f"'{f}', S.{prefix}_{f}" for f in PIECE_FIELDS)
    return f"{var} = struct({args});"


def _write_model_inputs() -> dict[str, dict]:
    """Write every ``sea_ice_model`` fixture into ``reference/ch08/model_inputs.mat``."""
    fixtures = FX.model_fixtures()
    payload: dict[str, np.ndarray] = {}
    for name, fx in fixtures.items():
        payload.update(_struct_payload(f"floe_{name}", fx["ice_floe"]))
        payload.update(_struct_payload(f"brash_{name}", fx["brash_ice"]))
        payload[f"img_{name}"] = np.zeros(fx["shape"], dtype=np.float64)
    savemat(str(REF / "model_inputs.mat"), payload, do_compression=True)
    return fixtures


# ==============================================================================================================
# Reference runs
# ==============================================================================================================

def ref_misc() -> None:
    """Colour tables, the colon operator, MATLAB's own ``lsqcurvefit`` defaults and the ``xx ~= NaN`` idiom."""
    code = """
jet30 = jet(30);
jet255 = jet(255);
jet3 = jet(3);
jet7 = jet(7);
jet1 = jet(1);
jet4 = jet(4);
t_circle = 0:0.05:6.28;
t_n = numel(t_circle);
t_last = t_circle(end);
colon_20_70_3500 = 20:70:3500;
colon_21_79_3971 = 21:79:3971;
colon_0_1e3_1 = 0:0.001:1;
o = optimoptions('lsqcurvefit');
opt_tolfun = o.FunctionTolerance;
opt_tolx = o.StepTolerance;
opt_maxiter = o.MaxIterations;
opt_alg = o.Algorithm;
opt_maxfun = o.MaxFunctionEvaluations;
% the `if xx ~= NaN` idiom of sea_ice_model.m lines 65/81/135, isolated
xx = [1 2 3];
idiom_nonempty = double(xx ~= NaN);
if xx ~= NaN
    idiom_branch_nonempty = 1;
else
    idiom_branch_nonempty = 0;
end
xx = [];
if xx ~= NaN
    idiom_branch_empty = 1;
else
    idiom_branch_empty = 0;
end
xx = [1 NaN];
if xx ~= NaN
    idiom_branch_withnan = 1;
else
    idiom_branch_withnan = 0;
end
% polyxpoly on the containment case: a small square strictly inside a big one
bx = [10 70 70 10 10]; by = [10 10 70 70 10];
sx = [30 40 40 30 30]; sy = [30 30 40 40 30];
[cx, cy] = polyxpoly(bx, by, sx, sy);
contain_empty = double(isempty(cx));
% and on two squares sharing exactly one corner point
ax = [10 29 29 10 10]; ay = [10 10 29 29 10];
cx2 = [29 48 48 29 29]; cy2 = [29 29 48 48 29];
[tx, ty] = polyxpoly(ax, ay, cx2, cy2);
touch_pts = [tx, ty];
"""
    _run(code, ["jet30", "jet255", "jet3", "jet7", "jet1", "jet4", "t_circle", "t_n", "t_last",
                "colon_20_70_3500", "colon_21_79_3971", "colon_0_1e3_1",
                "opt_tolfun", "opt_tolx", "opt_maxiter", "opt_alg", "opt_maxfun",
                "idiom_nonempty", "idiom_branch_nonempty", "idiom_branch_empty", "idiom_branch_withnan",
                "contain_empty", "touch_pts"], "misc.mat")


def ref_mcd() -> None:
    """``main_WL_new.m`` on the shipped §8.3 structure (Figs. 8.19/8.20 + the two unprinted figures)."""
    if not (DATA / "IceImage_290915_2_jpg.0000179.mat").exists():
        print("SKIP mcd: data/book/ch08/MCD/IceImage_290915_2_jpg.0000179.mat is absent")
        return
    code = ("run('ch08_main_ref.m');\n"
            "rgb_size = size(Raw_rgb);\n"
            "poly_rgb_isempty = double(isempty(Poly_rgb));\n"
            "Poly_polyx20 = Poly_polyx(1:20); Poly_polyy20 = Poly_polyy(1:20);\n"
            # the painted map is ~17 MB as double; store it once, quantised, in its own file
            "Raw_rgb_u8 = uint8(round(Raw_rgb*255));\n"
            f"save({q(REF / 'mcd_rgb.mat')}, 'Raw_rgb_u8', '-v7');\n")
    _run(code, ["Poly_Area", "Poly_MCD", "Raw_Area", "Raw_MCD", "Poly_counts", "Poly_centers",
                "Raw_counts", "Raw_centers", "count_error", "color_M", "N", "Y_limi",
                "Poly_index", "Raw_index", "Poly_caxis", "Raw_caxis", "caxis_lim_err",
                "Poly_polyc", "Poly_dotxy", "Raw_dotxy", "Raw_centre", "Poly_centre", "rgb_size",
                "poly_rgb_isempty", "Poly_polyx20", "Poly_polyy20"],
         "mcd.mat", timeout=5400)


def ref_fit() -> None:
    """``fitting_iceFloes_distribution.m`` on the shipped ``MCD_results.mat`` (Eq. 8.2/8.3, Fig. 8.21)."""
    if not (DATA / "MCD_results.mat").exists():
        print("SKIP fit: data/book/ch08/MCD/MCD_results.mat is absent")
        return
    code = """
run('ch08_fit_ref.m');
out_iterations = output.iterations;
out_funcCount = output.funcCount;
out_firstorderopt = output.firstorderopt;
out_algorithm = output.algorithm;
jac = full(jacobian);
% the estimator the TEXT describes (a straight line on the log-log plot), for comparison
p_loglog = polyfit(log10(MCD), log10(N_L), 1);
% does the shipped line 57 (the legend with the stray quote) parse at all?
legend_err = '';
try
    figure(99);
    run('ch08_legend_probe.m');
catch err
    legend_err = err.message;
end
"""
    _run(code, ["MCD", "N_L", "N_total", "epsilong", "eta", "resnorm", "residual", "exitflag",
                "out_iterations", "out_funcCount", "out_firstorderopt", "out_algorithm", "jac",
                "x_plot", "y_fit_powerlaw0", "sorted_floe_size", "a_sorted_floe_size", "p_loglog",
                "legend_err"],
         "fit.mat", timeout=1800)


def ref_orphan() -> None:
    """The two orphans, verbatim, on the fixtures and (for the power law) on the real §8.3 data."""
    fx = FX.fit_fixtures()
    payload = {}
    for name, (x, y) in fx.items():
        payload[f"x_{name}"] = np.asarray(x, dtype=np.float64).reshape(1, -1)
        payload[f"y_{name}"] = np.asarray(y, dtype=np.float64).reshape(1, -1)
    savemat(str(REF / "fit_inputs.mat"), payload, do_compression=True)

    lines = [f"S = load({q(REF / 'fit_inputs.mat')});"]
    save_vars = []
    for i, name in enumerate(fx):
        lines += [
            f"x = S.x_{name}; y = S.y_{name};",
            f"eta_pl_{name} = PowerLaw_fitting_method_and_plotting({10 + i}, x, y);",
            f"[eta1_{name}, eta2_{name}, eta3_{name}] = three_fitting_method_and_plotting({20 + i}, x, y);",
            # the resnorm/exitflag of the three fits, which the file computes and discards
            "F_tp=@(e,x)(e(1)*(x.^(-1*e(2))-e(3).^(-1*e(2))));",
            "lb=[min(x)/10 0 1]; ub=[10^6 10^6 10^6];",
            "op3=optimset('LargeScale','on','MaxFunEvals',100000,'TolFun',1e-5,'MaxIter',10000);",
            "[~,rn1,~,ef1] = lsqcurvefit(F_tp,[min(x) 1 x(end)],x,y,lb,ub,op3);",
            "F_wb=@(e,x)(exp(-(x./e(2)).^(e(1))));",
            "[~,rn2,~,ef2] = lsqcurvefit(F_wb,[1 mean(x)],x,y);",
            f"rn1_{name}=rn1; ef1_{name}=ef1; rn2_{name}=rn2; ef2_{name}=ef2;",
            # an inline repeat of the file's own lsqcurvefit call, to capture the outputs it discards
            "F_powerlaw0=@(epsilong,x)(epsilong(1)*(x.^(-1*epsilong(2))));",
            "epsilong0 = [min(x) 1];",
            "[ep,rn,rs,ef,op,~,jc] = lsqcurvefit(F_powerlaw0,epsilong0,x,y);",
            f"pl_x_{name} = ep; pl_resnorm_{name} = rn; pl_exitflag_{name} = ef;",
            f"pl_iter_{name} = op.iterations; pl_nfev_{name} = op.funcCount;",
        ]
        save_vars += [f"rn1_{name}", f"ef1_{name}", f"rn2_{name}", f"ef2_{name}",
                      f"eta_pl_{name}", f"eta1_{name}", f"eta2_{name}", f"eta3_{name}",
                      f"pl_x_{name}", f"pl_resnorm_{name}", f"pl_exitflag_{name}",
                      f"pl_iter_{name}", f"pl_nfev_{name}"]
    _run("\n".join(lines), save_vars, "orphan.mat", timeout=3600)


def ref_colorhist() -> None:
    """``color_hist.m`` and ``color_hist_comparison.m`` on the fixtures and on the real §8.3 areas."""
    payload: dict[str, np.ndarray] = {}
    for name, a in FX.hist_fixtures().items():
        payload[f"a_{name}"] = np.asarray(a, dtype=np.float64).reshape(-1, 1)
    have_book = (DATA / "IceImage_290915_2_jpg.0000179.mat").exists()
    if have_book:
        from seaice.core.icestruct import load_iceimage_mat
        ice = load_iceimage_mat(DATA / "IceImage_290915_2_jpg.0000179.mat")
        payload["a_real_poly"] = np.array([f.Polygon.Area for f in ice.Floe], dtype=np.float64).reshape(-1, 1)
        payload["a_real_pix"] = np.array([f.Area for f in ice.Floe], dtype=np.float64).reshape(-1, 1)
    savemat(str(REF / "hist_inputs.mat"), payload, do_compression=True)

    names = list(FX.hist_fixtures()) + (["real"] if have_book else [])
    lines = [f"S = load({q(REF / 'hist_inputs.mat')});"]
    save_vars: list[str] = []
    for name in names:
        model_key = "a_real_poly" if name == "real" else f"a_{name}"
        raw_key = "a_real_pix" if name == "real" else f"a_{name}"
        lines += [
            f"floe = struct('Area', num2cell(S.{model_key}));",
            f"ice_floe = struct('Area', num2cell(S.{raw_key}));",
            "clearvars z n color zs izs ysh YT color_floe color_min color_max nbins k d;",
            "run('ch08_colorhist_ref.m');",
            f"ch_z_{name} = z; ch_n_{name} = n; ch_color_{name} = color; ch_zs_{name} = zs;",
            f"ch_izs_{name} = izs; ch_ysh_{name} = ysh; ch_YT_{name} = YT;",
            f"ch_cf_{name} = color_floe; ch_cmin_{name} = color_min; ch_cmax_{name} = color_max;",
            f"ch_nbins_{name} = nbins;",
            "clearvars z n z0 n0 z_d color zs izs ysh YT color_floe color_floe0 color_min color_max nbins "
            "nbins0 k d floe_area floe_area0;",
            "run('ch08_colorhistcmp_ref.m');",
            f"cc_z_{name} = z; cc_z0_{name} = z0; cc_zd_{name} = z_d; cc_n_{name} = n; cc_n0_{name} = n0;",
            f"cc_color_{name} = color; cc_ysh_{name} = ysh; cc_YT_{name} = YT;",
            f"cc_cmin_{name} = color_min; cc_cmax_{name} = color_max;",
        ]
        save_vars += [f"ch_z_{name}", f"ch_n_{name}", f"ch_color_{name}", f"ch_zs_{name}", f"ch_izs_{name}",
                      f"ch_ysh_{name}", f"ch_YT_{name}", f"ch_cf_{name}", f"ch_cmin_{name}", f"ch_cmax_{name}",
                      f"ch_nbins_{name}", f"cc_z_{name}", f"cc_z0_{name}", f"cc_zd_{name}", f"cc_n_{name}",
                      f"cc_n0_{name}", f"cc_color_{name}", f"cc_ysh_{name}", f"cc_YT_{name}",
                      f"cc_cmin_{name}", f"cc_cmax_{name}"]
    # `color_hist_comparison.m` with color_hist.m's max_x = 3500 (risk R9: the list printed under Fig. 8.15)
    lines += ["mx3500_min = fix((1 - exp(-20/1000))*10000);",
              "mx3500_max = fix((1 - exp(-3500/1000))*10000);",
              "mx6000_max = fix((1 - exp(-6000/1000))*10000);",
              "d3500 = fix((mx3500_max-mx3500_min)/8); ysh3500 = mx3500_min : d3500 : mx3500_max;",
              "d6000 = fix((mx6000_max-mx3500_min)/8); ysh6000 = mx3500_min : d6000 : mx6000_max;",
              "YT3500 = []; for i = 1:length(ysh3500); YT3500{1,i} = -round(1000*log(1-ysh3500(i)/10000)); end",
              "YT6000 = []; for i = 1:length(ysh6000); YT6000{1,i} = -round(1000*log(1-ysh6000(i)/10000)); end"]
    save_vars += ["ysh3500", "ysh6000", "YT3500", "YT6000", "mx3500_min", "mx3500_max", "mx6000_max"]
    _run("\n".join(lines), save_vars, "colorhist.mat", timeout=1800)


def _model_block(fx_name: str, tag: str) -> tuple[list[str], list[str]]:
    lines = [
        _struct_matlab("ice_floe", f"floe_{fx_name}"),
        _struct_matlab("brash_ice", f"brash_{fx_name}"),
        f"img = S.img_{fx_name};",
        f"[floe, brash, bw_floe, bw_brash, ver, t] = ch08_model_ref(ice_floe, brash_ice, img);",
        f"{tag}_nfloe = numel(floe); {tag}_nbrash = numel(brash);",
        f"{tag}_V = {{floe.Vertices}}; {tag}_C = cat(1, floe.Center); {tag}_A = cat(1, floe.Area);",
        f"{tag}_P = cat(1, floe.Perimeter);",
        f"{tag}_IF = cell(1, numel(floe)); {tag}_IB = cell(1, numel(floe));",
        "for i = 1:numel(floe)",
        f"  {tag}_IF{{i}} = floe(i).Intersect.floe; {tag}_IB{{i}} = floe(i).Intersect.brash;",
        "end",
        f"{tag}_bR = cat(1, brash.Radius); {tag}_bC = cat(1, brash.Center); {tag}_bA = cat(1, brash.Area);",
        f"{tag}_bP = cat(1, brash.Perimeter);",
        f"{tag}_bIF = cell(1, numel(brash)); {tag}_bIB = cell(1, numel(brash));",
        "for i = 1:numel(brash)",
        f"  {tag}_bIF{{i}} = brash(i).Intersect.floe; {tag}_bIB{{i}} = brash(i).Intersect.brash;",
        "end",
        f"{tag}_bwf = logical(bw_floe); {tag}_bwb = logical(bw_brash);",
        f"{tag}_ver = {{ver.Vertices}};",
        # the Appendix-B structure on the same pieces
        "coverage = struct('IceFloe', 0.25, 'BrashIce', 0.1, 'Slush', 0.3, 'Water', 0.35);",
        "index_floe = zeros(size(img)); index_residue = zeros(size(img)); index_residue(1:7) = 1;",
        "clearvars Floe Brash Param Field IceImage FSD z n int_min int_max num;",
        "run('ch08_struct_ref.m');",
        f"{tag}_st_polyV = {{IceImage.Floe.Polygon}};",
        f"{tag}_st_FSD = IceImage.Field.FSD; {tag}_st_Field = IceImage.Field;",
        f"{tag}_st_Param = IceImage.Param;",
        f"{tag}_st_nFloe = numel(IceImage.Floe); {tag}_st_nBrash = numel(IceImage.Brash);",
        f"{tag}_st_bCircle = cell(1, numel(IceImage.Brash));",
        "for i = 1:numel(IceImage.Brash)",
        f"  {tag}_st_bCircle{{i}} = [IceImage.Brash(i).Circle.Radius, IceImage.Brash(i).Circle.Perimeter];",
        "end",
        f"{tag}_st_fVert = cell(1, numel(IceImage.Floe));",
        "for i = 1:numel(IceImage.Floe)",
        f"  {tag}_st_fVert{{i}} = IceImage.Floe(i).Polygon.Vertices;",
        "end",
        "clearvars floe brash bw_floe bw_brash ver t ice_floe brash_ice img;",
    ]
    save = [f"{tag}_nfloe", f"{tag}_nbrash", f"{tag}_V", f"{tag}_C", f"{tag}_A", f"{tag}_P",
            f"{tag}_IF", f"{tag}_IB", f"{tag}_bR", f"{tag}_bC", f"{tag}_bA", f"{tag}_bP",
            f"{tag}_bIF", f"{tag}_bIB", f"{tag}_bwf", f"{tag}_bwb", f"{tag}_ver",
            f"{tag}_st_FSD", f"{tag}_st_Field", f"{tag}_st_Param", f"{tag}_st_nFloe", f"{tag}_st_nBrash",
            f"{tag}_st_bCircle", f"{tag}_st_fVert"]
    return lines, save


def ref_model() -> None:
    """``sea_ice_model.m`` + ``SeaIce_Image_Structure.m`` on the constructed fixtures."""
    fixtures = _write_model_inputs()
    lines = [f"S = load({q(REF / 'model_inputs.mat')});"]
    save_vars: list[str] = []
    for name in fixtures:
        blk, sv = _model_block(name, name)
        lines += blk
        save_vars += sv
    _run("\n".join(lines), save_vars, "model.mat", timeout=3600)


def ref_window() -> None:
    """``sea_ice_model.m`` on a real 200x200-pixel window of the shipped §8.3 field (227 floes, 240 brash)."""
    fx = FX.window_fixture()
    if fx is None:
        print("SKIP window: the shipped IceImage .mat is absent")
        return
    payload = {"img_win": np.zeros(fx["shape"], dtype=np.float64),
               "floe_indices": fx["floe_indices"].reshape(1, -1),
               "brash_indices": fx["brash_indices"].reshape(1, -1)}
    payload.update(_struct_payload("floe_win", fx["ice_floe"]))
    payload.update(_struct_payload("brash_win", fx["brash_ice"]))
    savemat(str(REF / "window_inputs.mat"), payload, do_compression=True)
    print(f"    window fixture: {len(fx['ice_floe'])} floes, {len(fx['brash_ice'])} brash, "
          f"image {fx['shape']}")
    lines = [f"S = load({q(REF / 'window_inputs.mat')});"]
    blk, sv = _model_block("win", "win")
    lines += blk
    _run("\n".join(lines), sv, "window.mat", timeout=7200)


# ==============================================================================================================
def _run(code: str, save_vars, out_name: str, timeout: int = 1800) -> None:
    t0 = time.time()
    print(f"--- {out_name}: {len(save_vars)} variables ...", flush=True)
    VERIFY.mkdir(parents=True, exist_ok=True)
    (VERIFY / f"{Path(out_name).stem}_code.m").write_text(code, encoding="utf-8")
    res = run_ref(code, save_vars, REF / out_name, addpath=None, workdir=SCRATCH, timeout=timeout)
    dt = time.time() - t0
    print(f"    {out_name}: engine={res.engine} version={res.version} {dt:.1f} s", flush=True)
    log = VERIFY / "refs_log.json"
    entries = json.loads(log.read_text()) if log.exists() else {}
    entries[out_name] = {"engine": res.engine, "version": res.version, "seconds": round(dt, 1),
                         "status": "ok", "vars": list(save_vars), "stdout_tail": res.stdout[-800:]}
    log.write_text(json.dumps(entries, indent=1), encoding="utf-8")


TARGETS = {"misc": ref_misc, "mcd": ref_mcd, "fit": ref_fit, "orphan": ref_orphan,
           "colorhist": ref_colorhist, "model": ref_model, "window": ref_window}


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
