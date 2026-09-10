"""Non-linear least-squares curve fitting — the Optimization Toolbox ``lsqcurvefit`` and the book's three models.

Book: Ch. 8 §8.3 "Sea ice floe size statistic" (pp. 189–194).  **Eq. (8.3)** ``N_c(L) ∝ L^{-α}`` is fitted with
``lsqcurvefit`` on the *untransformed* ``(L, N_c)`` pairs — not, as the text says, as a straight line on a log-log
plot (a log-log OLS line on the same data gives ``α = 1.87``, the code gives ``α = 1.3704``).  Two further models
appear only in the code: the upper-truncated power law and the Weibull survival function.

MATLAB sources that call ``lsqcurvefit`` (4 call sites, all in ``MATLAB_ROOT/ch8/MCD/``):

============================================================  ==============================================
``fitting_iceFloes_distribution.m`` line 44                   :func:`lsqcurvefit` + :func:`power_law`
``PowerLaw_fitting_method_and_plotting.m`` line 14            :func:`lsqcurvefit` + :func:`power_law`
``three_fitting_method_and_plotting.m`` line 19               bounded + ``optimset`` → :func:`truncated_power_law`
``three_fitting_method_and_plotting.m`` line 32               :func:`lsqcurvefit` + :func:`weibull_survival`
``three_fitting_method_and_plotting.m`` line 45               :func:`lsqcurvefit` + :func:`power_law`
============================================================  ==============================================

# DEVIATION: `near` — `lsqcurvefit` (Optimization Toolbox) has no exact Python twin.  MATLAB's default
# `Algorithm = 'trust-region-reflective'` is the Coleman–Li interior trust-region method; SciPy's
# `least_squares(method='trf')` is the same *family* but a different implementation (different subproblem
# solver, different trust-region update, different scaling).  The tolerances are mapped one for one
# (`TolFun -> ftol`, `TolX -> xtol`, `MaxFunEvals -> max_nfev`) and MATLAB's own defaults are the defaults here
# (read from R2025a `toolbox/shared/optimlib/lsqcurvefit.m` line 150 and lines 345-377:
# `Algorithm = 'trust-region-reflective'`, `TolFun = TolFunValue = TolX = 1e-6`, `MaxIter = 400`,
# `MaxFunEvals = []` = 100*numberOfVariables, which is also SciPy's default), so the *parameters* agree to
# roughly the square root of the tolerance, not bit for bit.  `MaxIter` has no SciPy analogue and is only
# recorded.  Always report `resnorm`/`exitflag` beside the parameters: a difference beyond ~1e-5 relative means
# a *different local minimum*, which must be reported rather than tuned away (analysis/ch08.md risk R1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "MATLAB_LSQ_DEFAULTS", "LsqOptions", "LsqResult", "optimset", "lsqcurvefit",
    "power_law", "truncated_power_law", "weibull_survival",
]

#: MATLAB's own ``lsqcurvefit`` defaults (R2025a ``toolbox/shared/optimlib/lsqcurvefit.m`` l. 150, 345–377).
MATLAB_LSQ_DEFAULTS: dict[str, Any] = {
    "Algorithm": "trust-region-reflective",
    "TolFun": 1e-6,
    "TolFunValue": 1e-6,
    "TolX": 1e-6,
    "MaxIter": 400,
    "MaxFunEvals": None,   # MATLAB: [] = 100 * numberOfVariables (SciPy's `trf` default is the same)
}


@dataclass
class LsqOptions:
    """A parsed ``optimset``/``optimoptions`` structure (only the fields the book's four call sites set)."""

    Algorithm: str = MATLAB_LSQ_DEFAULTS["Algorithm"]
    TolFun: float = MATLAB_LSQ_DEFAULTS["TolFun"]
    TolX: float = MATLAB_LSQ_DEFAULTS["TolX"]
    MaxIter: int = MATLAB_LSQ_DEFAULTS["MaxIter"]
    MaxFunEvals: int | None = MATLAB_LSQ_DEFAULTS["MaxFunEvals"]
    LargeScale: str = "on"     #: legacy switch; ``'on'`` *is* trust-region-reflective, so it changes nothing here
    extra: dict[str, Any] = field(default_factory=dict)   #: any other ``optimset`` name, kept for the record


def optimset(**kwargs: Any) -> LsqOptions:
    """MATLAB ``optimset('Name', value, ...)`` restricted to the names this chapter uses.

    ``three_fitting_method_and_plotting.m`` line 18 writes::

        options = optimset('LargeScale','on','MaxFunEvals',100000,'TolFun',1e-5,'MaxIter',10000);

    Unknown names are accepted and stored in :attr:`LsqOptions.extra` rather than raising, because MATLAB's
    legacy ``optimset`` accepts (and silently ignores) options that the chosen algorithm does not read.
    """
    known = {f for f in LsqOptions.__dataclass_fields__ if f != "extra"}
    opts = LsqOptions()
    for name, value in kwargs.items():
        if name in known:
            setattr(opts, name, value)
        else:
            opts.extra[name] = value
    return opts


@dataclass
class LsqResult:
    """The seven outputs of ``[x, resnorm, residual, exitflag, output, lambda, jacobian] = lsqcurvefit(...)``.

    ``lambda`` (the Lagrange multipliers) is the one output every book call site discards with ``~``; it is not
    reproduced.  ``resnorm`` is MATLAB's ``sum(residual.^2)`` = ``2 * scipy_result.cost``.
    """

    x: np.ndarray                 #: the fitted parameter vector (MATLAB's ``epsilong``)
    resnorm: float                #: ``sum((F(x, xdata) - ydata).^2)``
    residual: np.ndarray          #: ``F(x, xdata) - ydata``
    exitflag: int                 #: MATLAB exit flag (see :func:`_exitflag`)
    output: dict[str, Any]        #: ``iterations``, ``funcCount``, ``algorithm``, ``firstorderopt``, ``message``
    jacobian: np.ndarray          #: ``∂residual/∂x`` at the solution
    scipy_result: Any = None      #: the raw ``scipy.optimize.OptimizeResult`` (not a MATLAB output)


#: SciPy ``status`` → MATLAB ``exitflag``.  MATLAB: 1 = converged (first-order optimality ≤ TolFun),
#: 2 = change in ``x`` ≤ TolX, 3 = change in the residual ≤ TolFun, 0 = iteration/function-evaluation limit,
#: −2 = problem infeasible.  SciPy: 1 = gtol, 2 = ftol, 3 = xtol, 4 = ftol *and* xtol, 0 = max_nfev, −1 = failure.
_EXITFLAG = {1: 1, 2: 3, 3: 2, 4: 3, 0: 0, -1: -2}


def _exitflag(status: int) -> int:
    """Translate SciPy's termination ``status`` into MATLAB's ``exitflag`` (documented mapping, not a MATLAB run)."""
    return _EXITFLAG.get(int(status), 0)


def lsqcurvefit(
    fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x0: Sequence[float] | np.ndarray,
    xdata: np.ndarray,
    ydata: np.ndarray,
    lb: Sequence[float] | np.ndarray | None = None,
    ub: Sequence[float] | np.ndarray | None = None,
    options: LsqOptions | None = None,
) -> LsqResult:
    """``[x, resnorm, residual, exitflag, output, ~, jacobian] = lsqcurvefit(FUN, X0, XDATA, YDATA, LB, UB, OPTIONS)``.

    Minimises ``sum((FUN(x, xdata) - ydata).^2)`` over ``x``, subject to ``lb <= x <= ub``.

    Book: §8.3, **Eq. (8.3)**; MATLAB source: ``MATLAB_ROOT/ch8/MCD/fitting_iceFloes_distribution.m`` line 44
    (unbounded, default options) and ``three_fitting_method_and_plotting.m`` line 19 (with ``lb``/``ub`` and an
    ``optimset`` structure).

    Parameters
    ----------
    fun : callable
        ``fun(params, xdata) -> ydata_hat`` — MATLAB's ``FUN`` takes the *parameters first*, exactly like the
        anonymous functions ``@(epsilong, x) ...`` the M-files build.
    x0 : array_like
        Initial parameter vector (``epsilong0``).
    xdata, ydata : ndarray
        The observations.  Flattened, as MATLAB does for vector data.
    lb, ub : array_like, optional
        Element-wise bounds.  ``None`` (MATLAB ``[]``) means unbounded.
    options : LsqOptions, optional
        From :func:`optimset`; MATLAB's own defaults are used when omitted.

    Returns
    -------
    LsqResult

    Notes
    -----
    Parity **near** — see the module docstring's ``DEVIATION`` note.  ``exitflag`` follows a documented
    translation of SciPy's ``status``, not a MATLAB run.
    """
    from scipy.optimize import least_squares

    opts = options or LsqOptions()
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    xd = np.asarray(xdata, dtype=np.float64).ravel()
    yd = np.asarray(ydata, dtype=np.float64).ravel()
    if xd.size != yd.size:
        raise ValueError("lsqcurvefit: xdata and ydata must have the same number of elements")

    lo = -np.inf * np.ones_like(x0) if lb is None else np.asarray(lb, dtype=np.float64).ravel()
    hi = np.inf * np.ones_like(x0) if ub is None else np.asarray(ub, dtype=np.float64).ravel()

    def residual(p: np.ndarray) -> np.ndarray:
        return np.asarray(fun(p, xd), dtype=np.float64).ravel() - yd

    # MATLAB switches to 'trust-region-reflective' by default and keeps it when bounds are present; SciPy's
    # 'trf' is the same family and is also the only SciPy method that accepts bounds.
    res = least_squares(residual, x0, bounds=(lo, hi), method="trf",
                        ftol=opts.TolFun, xtol=opts.TolX,
                        max_nfev=opts.MaxFunEvals)
    r = np.asarray(res.fun, dtype=np.float64)
    return LsqResult(
        x=np.asarray(res.x, dtype=np.float64),
        resnorm=float(r @ r),
        residual=r,
        exitflag=_exitflag(res.status),
        output={
            "iterations": int(getattr(res, "njev", 0) or 0),
            "funcCount": int(res.nfev),
            "algorithm": opts.Algorithm,
            "firstorderopt": float(np.max(np.abs(res.grad))) if np.size(res.grad) else 0.0,
            "message": str(res.message),
            "MaxIter": opts.MaxIter,      # recorded only: SciPy has no iteration cap
        },
        jacobian=np.asarray(res.jac, dtype=np.float64),
        scipy_result=res,
    )


# ==================================================================================================================
# The three model functions (pure; the fit and the plotting both call them)
# ==================================================================================================================

def power_law(p: Sequence[float] | np.ndarray, x: np.ndarray) -> np.ndarray:
    r"""**Eq. (8.3)** as coded: :math:`\hat N_c(L) = \varepsilon_1 L^{-\varepsilon_2}`.

    MATLAB source: ``fitting_iceFloes_distribution.m`` line 42 =
    ``PowerLaw_fitting_method_and_plotting.m`` line 12 = ``three_fitting_method_and_plotting.m`` line 43::

        F_powerlaw0 = @(epsilong, x)(epsilong(1)*(x.^(-1*epsilong(2))));

    The book writes Eq. (8.3) as a proportionality ``N_c(L) ∝ L^{-α}``; the code fits the constant too, so
    ``α = ε₂`` (p. 192, "estimated to be 1.3704").  Parity: **exact** (one expression).
    """
    p = np.asarray(p, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return p[0] * (x ** (-1.0 * p[1]))


def truncated_power_law(p: Sequence[float] | np.ndarray, x: np.ndarray) -> np.ndarray:
    r"""Upper-truncated power law (analysis C9): :math:`\hat N_c(L)=\varepsilon_1(L^{-\varepsilon_2}-\varepsilon_3^{-\varepsilon_2})`.

    MATLAB source: ``three_fitting_method_and_plotting.m`` line 14::

        F_powerlaw = @(epsilong, x)(epsilong(1)*(x.^(-1*epsilong(2))-epsilong(3).^(-1*epsilong(2))));

    ``ε₃`` is the upper truncation size.  Not printed in the book — its call site (line 37 of
    ``fitting_iceFloes_distribution.m``) is commented out.  Parity: **exact** (one expression).
    """
    p = np.asarray(p, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return p[0] * (x ** (-1.0 * p[1]) - p[2] ** (-1.0 * p[1]))


def weibull_survival(p: Sequence[float] | np.ndarray, x: np.ndarray) -> np.ndarray:
    r"""Weibull survival function (analysis C10): :math:`\hat N_c(L)=\exp(-(L/\varepsilon_2)^{\varepsilon_1})`.

    MATLAB source: ``three_fitting_method_and_plotting.m`` line 30::

        F_weibull = @(epsilong, x)(exp(-(x./epsilong(2)).^(epsilong(1))));

    ``ε₁`` is the shape and ``ε₂`` the scale; the start point is ``[1, mean(x)]`` (line 31).
    Parity: **exact** (one expression).
    """
    p = np.asarray(p, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-((x / p[1]) ** p[0]))
