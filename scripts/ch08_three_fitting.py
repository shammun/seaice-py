"""Port of ``MATLAB_ROOT/ch8/MCD/three_fitting_method_and_plotting.m`` — the three candidate distributions.

The M-file fits **three** models to the same cumulative floe size distribution and over-plots them::

    F_powerlaw = @(e,x)(e(1)*(x.^(-e(2)) - e(3).^(-e(2))));   % upper-truncated power law, bounded
    F_weibull  = @(e,x)(exp(-(x./e(2)).^(e(1))));             % Weibull survival function
    F_powerlaw0= @(e,x)(e(1)*(x.^(-e(2))));                   % plain power law (the one the book prints)

Its **only call site is commented out** (``fitting_iceFloes_distribution.m`` line 37), so the shipped driver never
runs it and the book never prints its results; this script exposes it (analysis/ch08.md risk R16).  No
goodness-of-fit statistic is computed for any of the three, here or anywhere in the chapter — ``resnorm`` and
``exitflag`` are printed so that a different local minimum is visible rather than silent.

The orphan ``PowerLaw_fitting_method_and_plotting.m`` (no caller anywhere in ``MATLAB_ROOT``) is the same power-law
fit and is ported as ``seaice.ch08_applications.power_law_fit``; ``--orphan`` runs it on its own.

Usage: ``python scripts/ch08_three_fitting.py [--mat MCD_results.mat] [--orphan]
[--data data/book/ch08] [--out outputs/ch08] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch08_applications import power_law_fit, three_distribution_fits  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.plotting import finish_figure  # noqa: E402
from seaice.core.stats import cumulative_size_distribution  # noqa: E402

CH = "ch08"


def main(argv: list[str] | None = None) -> int:
    p = chapter_argparser(CH, __doc__.splitlines()[0])
    p.add_argument("--mat", default="MCD_results.mat", help="the floe sizes to fit")
    p.add_argument("--orphan", action="store_true",
                   help="also run PowerLaw_fitting_method_and_plotting.m on its own (it has no caller)")
    args = p.parse_args(argv)
    data, out = resolve_dirs(args)

    hits = sorted(data.rglob(args.mat))
    if not hits:
        print(f"SKIP {Path(__file__).name}: private data absent ({args.mat} not found under {data.as_posix()})")
        return 0
    import scipy.io as sio
    raw_mcd = np.asarray(sio.loadmat(hits[0])["Raw_MCD"], dtype=np.float64).ravel()
    x, y = cumulative_size_distribution(raw_mcd)

    print(f"=== three_fitting_method_and_plotting.m on {hits[0].name} ({x.size} floes) ===")
    print("  (called only from a commented-out line; no book numbers exist for these fits)")
    r = three_distribution_fits(x, y)

    for tag, fit, names in (
        ("Upper Truncated Power-Law (eta1)", r.truncated_power, ("eps1", "eps2 = alpha", "eps3 = L_max")),
        ("Weibull                   (eta2)", r.weibull, ("shape", "scale", "")),
        ("Power-Law                 (eta3)", r.power, ("eps1", "eps2 = alpha", "")),
    ):
        pretty = ", ".join(f"{n} = {v:.8f}" for n, v in zip(names, fit.eta) if n)
        print(f"  {tag}: {pretty}")
        print(f"        resnorm = {fit.resnorm:.10f}, exitflag = {fit.exitflag}, "
              f"funcCount = {fit.output['funcCount']}, algorithm = {fit.output['algorithm']}")
    print("  bounds used for eta1 (line 16-17): lb = [min(x)/10, 0, 1], ub = [1e6, 1e6, 1e6]; "
          "optimset('LargeScale','on','MaxFunEvals',1e5,'TolFun',1e-5,'MaxIter',1e4)")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(x, y, "ro", markersize=4, markerfacecolor="none", label="Test data")
    ax.loglog(r.truncated_power.x_plot, r.truncated_power.y_plot, "--g", linewidth=2,
              label="Upper Truncated Power-Law")
    ax.loglog(r.weibull.x_plot, r.weibull.y_plot, ":b", linewidth=2, label="Weibull")
    ax.loglog(r.power.x_plot, r.power.y_plot, "k", linewidth=2, label="Power-Law")
    ax.set_xlabel("MCD[m]", fontsize=13)
    ax.set_ylabel("Cumulative Frequency", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title("three_fitting_method_and_plotting.m — the three candidate FSD models (not printed in the book)")
    written = [finish_figure(fig, out / "sec_8_3_three_fits.png", args.show)]

    if args.orphan:
        o = power_law_fit(x, y)
        print(f"\n  PowerLaw_fitting_method_and_plotting.m (orphan): eps = [{o.eta[0]:.8f}, {o.eta[1]:.8f}], "
              f"resnorm = {o.resnorm:.10f} - identical to eta3 by construction "
              f"({'same' if np.allclose(o.eta, r.power.eta) else 'DIFFERENT'})")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(x, y, "ro", markersize=4, markerfacecolor="none", label="Test data")
        ax.loglog(o.x_plot, o.y_plot, "k", linewidth=2, label="Power-Law")
        ax.set_xlabel("$L/h$ [-]", fontsize=13)          # the orphan's own axis labels (lines 25-26)
        ax.set_ylabel("$N(>=L)/N_{total}$", fontsize=13)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower left")
        ax.set_title("PowerLaw_fitting_method_and_plotting.m — the orphan file, with its own axis labels")
        written.append(finish_figure(fig, out / "sec_8_3_powerlaw_orphan.png", args.show))

    print("\nfigures written:")
    for w in written:
        print("  ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
