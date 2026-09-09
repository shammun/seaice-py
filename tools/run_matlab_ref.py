"""Run original MATLAB `.m` code non-interactively and save reference variables to a `.mat` file.

This is the L2 "reference parity" engine for seaice-py. MATLAB R2025a is installed on this host, so
references are generated against **real MATLAB** — the same product the book's authors used — which makes
`imbinarize`, `adaptthresh`, `activecontour`, `graythresh`, `regionprops` etc. directly testable.

Use from a chapter's ``reference/chNN/make_refs.py``::

    from tools.run_matlab_ref import run_ref
    run_ref(
        "I = imread('data/book/ch02/ice.jpg'); BW = imbinarize(rgb2gray(I));",
        save_vars=["I", "BW"],
        out_mat="reference/ch02/imbinarize.mat",
        addpath="K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab/ch2",
    )

or run a whole original script and keep whatever it leaves in the workspace::

    run_ref("run('histogram.m')", ["h", "I"], "reference/ch02/histogram.mat", addpath=MATLAB_CH)

`run_ref` prefers MATLAB and falls back to oct2py/Octave only when MATLAB is absent; the returned
:class:`RefResult` records which engine actually produced the file so the verification report can say so.

CLI::

    .venv/Scripts/python.exe tools/run_matlab_ref.py --out reference/ch02/x.mat \\
        --vars I,BW --addpath "<MATLAB_ROOT>/ch2" --code "I = imread('a.png'); BW = I > 128;"
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Checked in this order; the first that exists wins. `MATLAB_EXE` overrides everything.
MATLAB_CANDIDATES = (
    r"C:\Program Files\MATLAB\R2025a\bin\matlab.exe",
    r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe",
    r"C:\Program Files\MATLAB\R2024a\bin\matlab.exe",
)


def matlab_exe() -> str | None:
    """Absolute path to the MATLAB executable, or None if MATLAB is not installed."""
    env = os.environ.get("MATLAB_EXE")
    if env and Path(env).exists():
        return env
    for cand in MATLAB_CANDIDATES:
        if Path(cand).exists():
            return cand
    return shutil.which("matlab")


def matlab_available() -> bool:
    return matlab_exe() is not None


def matlab_version(timeout: int = 300) -> str | None:
    """Run ``matlab -batch "disp(version)"`` and return the version string (None if unavailable)."""
    exe = matlab_exe()
    if exe is None:
        return None
    proc = subprocess.run([exe, "-batch", "disp(version)"], capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        return None
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.strip():
            return line.strip()
    return None


@dataclass
class RefResult:
    """What actually happened, so the verification report can cite the engine."""

    engine: str  # "matlab" | "octave"
    version: str
    out_mat: Path
    stdout: str


def _as_paths(addpath: str | Path | Sequence[str | Path] | None) -> list[Path]:
    if addpath is None:
        return []
    if isinstance(addpath, (str, Path)):
        addpath = [addpath]
    return [Path(p) if Path(p).is_absolute() else REPO_ROOT / p for p in addpath]


def _quote(p: Path) -> str:
    """MATLAB single-quoted string literal (single quotes are doubled)."""
    return "'" + str(p).replace("'", "''") + "'"


def _mat_loads(out_mat: Path, save_vars: Sequence[str]) -> bool:
    """True if ``out_mat`` is a complete v7 .mat holding every name in ``save_vars`` (used after a MATLAB timeout)."""
    try:
        from scipy.io import loadmat
        d = loadmat(str(out_mat))
    except Exception:
        return False
    return all(v in d for v in save_vars)


def run_matlab_ref(
    code: str,
    save_vars: Sequence[str],
    out_mat: str | Path,
    addpath: str | Path | Sequence[str | Path] | None = None,
    workdir: str | Path | None = None,
    timeout: int = 900,
) -> RefResult:
    """Execute `code` in a headless MATLAB session and save `save_vars` to `out_mat` (v7 .mat).

    Parameters
    ----------
    code
        MATLAB statements. May be a single call, several statements, or ``run('script.m')`` to execute an
        original book script verbatim.
    save_vars
        Names of workspace variables to write into the ``.mat`` file. Must all exist when `code` finishes.
    out_mat
        Destination ``.mat`` path (relative paths resolve against the repo root). Parents are created.
    addpath
        Directory (or directories) added to the MATLAB path first — normally ``MATLAB_ROOT/chN``.
    workdir
        Working directory for the session; defaults to the repo root so relative data paths work.
    timeout
        Seconds before the MATLAB process is killed.

    Notes
    -----
    Figures are suppressed with ``set(0,'DefaultFigureVisible','off')`` so scripts that call ``imshow`` /
    ``figure`` run unattended. ``-batch`` implies ``-nodesktop -nosplash -nodisplay`` and returns a non-zero
    exit code on error, which is raised here as :class:`RuntimeError` with the MATLAB error report.
    """
    exe = matlab_exe()
    if exe is None:
        raise RuntimeError("MATLAB not found; set MATLAB_EXE or use run_ref() to fall back to Octave")
    if not save_vars:
        raise ValueError("save_vars must name at least one variable")

    out_mat = Path(out_mat)
    if not out_mat.is_absolute():
        out_mat = REPO_ROOT / out_mat
    out_mat.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir) if workdir else REPO_ROOT

    var_list = ", ".join(f"'{v}'" for v in save_vars)
    lines = [
        "set(0,'DefaultFigureVisible','off');",
        "warning('off','all');",
        f"cd({_quote(workdir)});",
        *[f"addpath({_quote(p)});" for p in _as_paths(addpath)],
        "try",
        code,
        # MATLAB's functional save() takes the filename first; '-v7' is the format flag.
        f"    save({_quote(out_mat)}, {var_list}, '-v7');",
        "catch err",
        "    disp(getReport(err, 'extended', 'hyperlinks', 'off'));",
        "    exit(1);",
        "end",
        "exit(0);",
    ]
    script = "\n".join(lines) + "\n"

    tmpdir = Path(tempfile.mkdtemp(prefix="seaice_matlab_"))
    try:
        script_path = tmpdir / "seaice_ref.m"
        script_path.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [exe, "-batch", f"run({_quote(script_path)})"],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            # Seen in ch04 (2026-09-09): MATLAB had written the complete .mat but matlab.exe did not exit within the
            # timeout (host overloaded by orphaned MATLABWindow processes). A complete, loadable .mat is a valid
            # reference — accept it and say so in stdout instead of failing the whole make_refs run.
            if out_mat.exists() and _mat_loads(out_mat, save_vars):
                note = (f"NOTE: matlab.exe exceeded the {timeout} s timeout after saving {out_mat.name} "
                        f"(all {len(save_vars)} variables present); reference accepted. "
                        f"Check for stale MATLABWindow/matlab processes (tasklist | findstr -i matlab).")
                out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                return RefResult("matlab", matlab_version() or "unknown", out_mat, out + "\n" + note)
            raise
        if proc.returncode != 0 or not out_mat.exists():
            raise RuntimeError(
                f"MATLAB failed (exit {proc.returncode}) for {out_mat.name}\n"
                f"--- script ---\n{script}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
            )
        return RefResult("matlab", matlab_version() or "unknown", out_mat, proc.stdout)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_octave_ref(
    code: str,
    save_vars: Sequence[str],
    out_mat: str | Path,
    addpath: str | Path | Sequence[str | Path] | None = None,
    workdir: str | Path | None = None,
    timeout: int = 900,
) -> RefResult:
    """Octave/oct2py fallback with the same contract as :func:`run_matlab_ref`.

    Only used when MATLAB is absent. Octave lacks several Image Processing Toolbox functions
    (``imbinarize``, ``adaptthresh``, ``activecontour``); those items must then drop to L3/L4 evidence.
    """
    from oct2py import octave  # imported lazily: oct2py is not needed when MATLAB is present

    out_mat = Path(out_mat)
    if not out_mat.is_absolute():
        out_mat = REPO_ROOT / out_mat
    out_mat.parent.mkdir(parents=True, exist_ok=True)
    octave.eval(f"cd {_quote(Path(workdir) if workdir else REPO_ROOT)}")
    octave.eval("pkg load image;")
    octave.eval("set(0,'DefaultFigureVisible','off');")
    for p in _as_paths(addpath):
        octave.addpath(str(p))
    octave.eval(code)
    var_list = ", ".join(f"'{v}'" for v in save_vars)
    octave.eval(f"save('-v7', {_quote(out_mat)}, {var_list});")
    return RefResult("octave", str(octave.eval("version", nout=1)).strip(), out_mat, "")


def run_ref(
    code: str,
    save_vars: Sequence[str],
    out_mat: str | Path,
    addpath: str | Path | Sequence[str | Path] | None = None,
    workdir: str | Path | None = None,
    timeout: int = 900,
    engine: str = "auto",
) -> RefResult:
    """Preferred entry point: run in MATLAB, falling back to Octave only if MATLAB is absent.

    `engine` may be ``"auto"`` (default), ``"matlab"`` or ``"octave"`` to force one engine.
    """
    if engine not in ("auto", "matlab", "octave"):
        raise ValueError(f"unknown engine {engine!r}")
    use_matlab = engine == "matlab" or (engine == "auto" and matlab_available())
    fn = run_matlab_ref if use_matlab else run_octave_ref
    return fn(code, save_vars, out_mat, addpath=addpath, workdir=workdir, timeout=timeout)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code", help="MATLAB statements to run (or use --script)")
    ap.add_argument("--script", help="a .m file to run with run('...')")
    ap.add_argument("--vars", help="comma-separated workspace variables to save")
    ap.add_argument("--out", help="destination .mat path")
    ap.add_argument("--addpath", action="append", default=[], help="directory to addpath (repeatable)")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--engine", default="auto", choices=["auto", "matlab", "octave"])
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--check", action="store_true", help="just print the MATLAB version and exit")
    args = ap.parse_args(argv)

    if args.check:
        exe = matlab_exe()
        print(f"matlab_exe: {exe}")
        print(f"version:    {matlab_version()}")
        return 0 if exe else 1

    if not args.out or not args.vars or not (args.code or args.script):
        ap.error("--out, --vars and one of --code/--script are required (or use --check)")
    code = args.code if args.code else f"run({_quote(Path(args.script))})"
    res = run_ref(code, [v.strip() for v in args.vars.split(",") if v.strip()], args.out,
                  addpath=args.addpath or None, workdir=args.workdir, timeout=args.timeout,
                  engine=args.engine)
    print(f"{res.engine} {res.version} -> {res.out_mat}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
