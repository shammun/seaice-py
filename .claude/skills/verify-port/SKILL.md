---
name: verify-port
description: How to verify that ported Python code reproduces the book's MATLAB results — MATLAB reference generation with tools/run_matlab_ref.py (matlab -batch; oct2py/Octave only as fallback), parity tests, synthetic-truth tests, book-figure reproduction, and the verification report format. Load when testing, verifying, or reviewing any chapter.
---

# verify-port — evidence that a port is correct

## The four evidence levels (do all that apply, in order)
**L1 Synthetic truth.** Small hand-made arrays whose correct answer is known analytically (a 5×5 square → distance
transform values; a 3-pixel diagonal → chain code `[7,7]`; two touching discs → watershed splits them). Put these in
`tests/test_chNN.py`. They never depend on external tools.

**L2 Reference parity (the gold standard).** Run the *original* `.m` code in **MATLAB** (R2025a is installed at
`C:\Program Files\MATLAB\R2025a\bin\matlab.exe`) on the *same input*, save outputs to `reference/chNN/<name>.mat`, load in
pytest and compare with `tools/compare_arrays.py`. MATLAB is the primary engine: references are produced by the very
product the authors used, so `exact`/`near` labels are measured against MATLAB itself and MATLAB-only functions
(`imbinarize`, `adaptthresh`, `activecontour`, `graythresh`, `regionprops` names …) are fully testable.
- Check availability once: `python tools/run_matlab_ref.py --check` (runs `matlab -batch "disp(version)"`), or read
  `progress.json → environment.matlab`. Never call `matlab` directly from a make_refs script — always go through the helper.
- Generating references (write `reference/chNN/make_refs.py`) with `tools/run_matlab_ref.py`. It takes a MATLAB command
  string and the list of workspace variables to save; it runs `matlab -batch "run(<tmp script>)"` non-interactively with
  `set(0,'DefaultFigureVisible','off')`, `addpath(MATLAB_ROOT/chN)`, the repo root as cwd, and `save(<out>, vars…, '-v7')`;
  a non-zero MATLAB exit raises `RuntimeError` carrying the MATLAB error report:
  ```python
  from tools.run_matlab_ref import run_ref
  CH = "K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab/ch2"      # MATLAB_ROOT/ch2
  # single function on a controlled input (write the input as PNG/.mat first so both sides load the identical array)
  run_ref("BW = imread('reference/ch02/in_bw.png') > 0; D = bwdist(BW, 'euclidean');",
          save_vars=["BW", "D"], out_mat="reference/ch02/bwdist_euclid.mat", addpath=CH)
  # or run a whole original script verbatim and keep what it leaves in the workspace
  run_ref("run('histogram.m')", ["h", "I"], "reference/ch02/histogram.mat", addpath=CH)
  ```
  `run_ref` returns a `RefResult(engine, version, out_mat, stdout)`; write `engine`/`version` into the report's Environment
  line. Each MATLAB launch costs ~10–20 s, so batch several variables into one call per `.m` file rather than one call per
  variable. Inputs to MATLAB must be files on disk (PNG/JPG/.mat) — there is no `push()`; to pass a numpy array save it with
  `scipy.io.savemat` and `load(...)` it in the command string.
- **MATLAB run hygiene (ch04 lessons):** (a) write Python-built fixtures to `reference/chNN/inputs.mat` (constructed
  ties, `intmax`/`intmin` plants, asymmetric and even SEs, signed images) — a real photograph never exercises the
  border/tie/dtype branches where compiled builtins differ; (b) use `try/catch` probes that record MATLAB's own error
  text for classes it rejects (e.g. `imerode` on int64) so the report can label those cases `L1-only` honestly;
  (c) `run_ref` accepts a complete `.mat` even if `matlab.exe` fails to exit within the timeout (it says so in
  `RefResult.stdout`; record the note in `refs_log.json`); (d) before long runs, kill stale `MATLABWindow` processes
  from earlier timed-out launches (`tasklist | findstr -i matlab`) — they slow the whole host; (e) when you start pytest
  or a reference run in the background, **wait for it inside the same task** (poll the log in one Bash loop) and only
  return with the final numbers — never hand back a report with placeholders.
- **Fallback (only if MATLAB is absent):** `run_ref(..., engine="auto")` transparently uses oct2py/Octave
  (`pkg load image`, same contract). Octave lacks `imbinarize`, `adaptthresh`, `activecontour`, some `regionprops` names
  and `imgaussfilt` in older versions. When a function is missing there: (a) find the authors' own `.m` implementation in the
  chapter folder — most of the book's *algorithms* are plain MATLAB; (b) drop to L3/L4 and label `unverified`→`approx`.
  Say `Octave` (not MATLAB) in the report whenever the fallback produced a reference.
- `.mat` files are git-ignored; commit `make_refs.py` so they can be regenerated.

**L3 Book-figure reproduction.** For every figure in the chapter that the MATLAB code produces, generate the Python
version with the same input image and save a side-by-side PNG in `reports/chNN/figures/fig_X_Y_compare.png`
(left: Python; right: MATLAB output image saved via `run_ref` (`imwrite(...)`) if available, else the cropped figure page from `chapters/chNN.pdf`
rendered with pymupdf at 150 dpi). Then *look at it* (Read the PNG) and write one sentence per figure.

**L4 Numbers quoted in the text.** The book quotes numbers (ice concentration %, thresholds, floe counts). Grep
`chapters/chNN.txt` for `%`, `threshold`, `= ` near results sections; reproduce them and tabulate `book value / our value`.

## Tolerances (defaults; tighten if the algorithm is deterministic)
| Output type | Comparator | Pass |
|---|---|---|
| binary mask | `assert_parity(kind="binary")` | ≤0.1% pixels differ (`exact` if 0) |
| label image | `assert_parity(kind="label", min_agreement=0.995)` | partition agreement ≥99.5% |
| float image (filters, distance, gradients) | `kind="float", atol=1e-6, rtol=1e-5` | |
| float image from JPEG-decoded input | `atol=2` (uint8 units) or compare after both sides load the same PNG | |
| scalar (threshold, concentration) | `abs diff ≤ 1/255` for thresholds, ≤0.5 pp for concentrations | |
| contours / chain codes | identical sequence after aligning the start point and direction | |
| k-means | sorted cluster centres within 1 gray level; pixel agreement ≥99% | |

## Test file skeleton (`tests/test_chNN.py`)
```python
import numpy as np, pytest
from pathlib import Path
from tools.compare_arrays import assert_parity, load_ref
from seaice import ch02_preliminaries as ch2
REF = Path("reference/ch02")
needs_ref = pytest.mark.skipif(not REF.exists(), reason="run reference/ch02/make_refs.py (needs MATLAB, or Octave as fallback)")

def test_distance_transform_synthetic():          # L1
    bw = np.zeros((5, 5), bool); bw[2, 2] = True
    d = ch2.distance_transform(bw, "chessboard")
    assert d[0, 0] == 2 and d[2, 3] == 1

@needs_ref
def test_distance_transform_parity():             # L2
    bw = load_ref(REF / "bwdist.mat", "BW").astype(bool)
    for m in ("euclidean", "cityblock", "chessboard", "quasi"):
        assert_parity(ch2.distance_transform(bw, m), load_ref(REF / "bwdist.mat", f"D_{m}"), "float", name=m)
```
Run with `python -m pytest tests/test_chNN.py -q -p no:cacheprovider` (add `-n auto` for speed).

## Report format (`reports/chNN_verification.md`)
```
# Chapter N verification — <title>            date, commit hash
## Environment: python x.y, numpy, scikit-image | reference engine: MATLAB 25.1 (R2025a) via tools/run_matlab_ref.py [or: Octave x.y (image pkg z) fallback — MATLAB absent]
## Parity table
| MATLAB file / function | Python | Evidence levels | Result | Parity label | Notes |
## Figures reproduced   (one row per book figure; link to compare PNG; one-sentence visual verdict)
## Numbers from the text (book vs ours)
## Deviations & justifications
## Open items (anything `unverified`, missing data, needs user)
## Verdict: PASS / FAIL  (PASS requires: all scripts run, all tests pass, every .m has a row, no `unverified` without an open item)
```

## Failure loop
If a parity test fails: (1) confirm the *reference* is right (rerun MATLAB via `run_ref`, check dtype/complement/1-based issues);
(2) fix the port; (3) rerun; max 3 rounds, then write the discrepancy in "Open items" with your best hypothesis and stop —
do not loosen tolerances to make tests pass.

## Public-repo rule
The repo is public. Never commit book text, book-shipped images, PDF page crops, or executed notebooks that contain them. Published notebooks load data through seaice.core.io.load_image(), which prefers the reader's private Drive copy and falls back to public-domain imagery. Book-figure comparisons stay local in reports/**/figures/ (git-ignored).
