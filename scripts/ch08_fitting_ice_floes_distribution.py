"""Port of ``MATLAB_ROOT/ch8/MCD/fitting_iceFloes_distribution.m`` — Figure 8.21, Eqs. (8.2) and (8.3).

The M-file loads the authors' own saved ``Raw_MCD`` (2888 floe sizes in metres), builds the cumulative floe size
distribution and fits a power law to it::

    load MCD_results.mat
    [sorted_floe_size,index] = sort(Raw_MCD);
    MCD = Raw_MCD(index);              N_total = size(MCD,2);
    N_L(i) = size(find(MCD>=MCD(i)),2)/N_total;                      % Eq. (8.2)
    F_powerlaw0 = @(epsilong,x)(epsilong(1)*(x.^(-1*epsilong(2))));  % Eq. (8.3)
    [epsilong,resnorm,residual,exitflag,output,~,jacobian] = lsqcurvefit(F_powerlaw0,[min(x) 1],x,y);

The book says alpha "is the slope of the power law curve on log-log plot" and prints **1.3704** (p. 192), but the
code fits the **untransformed** pairs with ``lsqcurvefit``; a log-log ordinary-least-squares line on the same data
gives 1.87.  The script prints both so the discrepancy is visible.  ``cd('C:\\Users\\qinz\\Desktop\\sent_to_Qin')``
(line 5) is dropped.  Lines 37-39 (the three-distribution call) are commented out in the shipped file — see
``scripts/ch08_three_fitting.py``.

Usage: ``python scripts/ch08_fitting_ice_floes_distribution.py [--mat MCD_results.mat]
[--source mcd-results|iceimage] [--data data/book/ch08] [--out outputs/ch08] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch08_applications import LENGTH_OVER_PIXEL, cumulative_fsd_powerlaw  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.icestruct import load_iceimage_mat  # noqa: E402
from seaice.core.plotting import finish_figure  # noqa: E402
from seaice.core.stats import cumulative_size_distribution, mean_caliper_diameter  # noqa: E402

CH = "ch08"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--mat", default="MCD_results.mat", help="fitting_iceFloes_distribution.m line 7")
    p.add_argument("--source", default="mcd-results", choices=["mcd-results", "iceimage"],
                   help="use the authors' saved Raw_MCD, or recompute it from the IceImage structure")
    p.add_argument("--iceimage", default="IceImage_290915_2_jpg.0000179.mat")
    p.add_argument("--length-over-pixel", type=float, default=LENGTH_OVER_PIXEL)
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    name = args.mat if args.source == "mcd-results" else args.iceimage
    hits = sorted(data.rglob(name))
    if not hits:
        print(f"SKIP {Path(__file__).name}: private data absent ({name} not found under {data.as_posix()}; "
              "the two ch8 .mat files ship with the book's MATLAB archive and are not in the public repo)")
        return 0
    path = hits[0]

    if args.source == "mcd-results":
        import scipy.io as sio
        raw_mcd = np.asarray(sio.loadmat(path)["Raw_MCD"], dtype=np.float64).ravel()
        origin = "the authors' saved Raw_MCD"
    else:
        ice = load_iceimage_mat(path)
        raw_mcd = mean_caliper_diameter([f.Area for f in ice.Floe], args.length_over_pixel)
        origin = "Eq. (8.1) recomputed from IceImage.Floe(i).Area"

    print(f"=== fitting_iceFloes_distribution.m on {path.name} ({origin}) ===")
    print(f"  N_total = {raw_mcd.size} floes; MCD {raw_mcd.min():.4f} .. {raw_mcd.max():.4f} m")

    L, Nc = cumulative_size_distribution(raw_mcd)
    print(f"  Eq. (8.2): N_c from {Nc.max():.6f} down to {Nc.min():.3e} = 1/{raw_mcd.size} "
          f"(Fig. 8.21 axes: 10^0.8..10^2.1 m, 10^-3.5..10^0)")

    fit = cumulative_fsd_powerlaw(raw_mcd)
    print(f"  Eq. (8.3) lsqcurvefit: eps = [{fit.eta[0]:.8f}, {fit.eta[1]:.8f}]  ->  alpha = {fit.alpha:.6f}")
    print(f"    resnorm = {fit.resnorm:.10f}, exitflag = {fit.exitflag}, funcCount = {fit.output['funcCount']}, "
          f"first-order optimality = {fit.output['firstorderopt']:.3e}")
    print(f"    book (p. 192): alpha = 1.3704  ->  {'MATCH' if abs(fit.alpha - 1.3704) < 5e-5 else 'DIFFERS'} "
          "to the printed digits")

    # The estimator the book *describes* (a straight line on the log-log plot) is NOT the one it runs.
    slope, intercept = np.polyfit(np.log10(L), np.log10(Nc), 1)
    print(f"  log-log OLS line (the estimator the TEXT describes): alpha = {-slope:.4f}, "
          f"eps1 = {10 ** intercept:.4f}  -> the book's own description of its estimator does not match its code")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(L, Nc, "ro", markersize=4, markerfacecolor="none", label="Observed data")
    ax.loglog(fit.x_plot, fit.y_plot, "k", linewidth=2, label="Power law fitting'")
    ax.set_xlabel("MCD[m]", fontsize=13)
    ax.set_ylabel("Cumulative Frequency", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", frameon=True)
    ax.set_title(f"Figure 8.21 — cumulative FSD and the Eq. (8.3) power law ($\\alpha$ = {fit.alpha:.4f})")
    written = [finish_figure(fig, out / "fig_8_21_powerlaw_fit.png", args.show)]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(L, Nc, "ro", markersize=4, markerfacecolor="none", label="Observed data")
    ax.loglog(fit.x_plot, fit.y_plot, "k", linewidth=2, label=f"lsqcurvefit (code): $\\alpha$ = {fit.alpha:.4f}")
    ax.loglog(fit.x_plot, 10 ** intercept * fit.x_plot ** slope, "b--", linewidth=2,
              label=f"log-log OLS (text): $\\alpha$ = {-slope:.4f}")
    ax.set_xlabel("MCD[m]")
    ax.set_ylabel("Cumulative Frequency")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title("The estimator the text describes vs the one the code runs")
    written.append(finish_figure(fig, out / "sec_8_3_powerlaw_estimators.png", args.show))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
