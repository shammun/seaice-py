# CUMULATIVE knowledge — seaice-py
(Rewritten by the knowledge-keeper after every chapter. Read this first at the start of any chapter.)

## Pipeline so far
(none yet) — target: Ch2 primitives → Ch3 ice mask → Ch4 edges → Ch5 watershed floes → Ch6 GVF boundaries → Ch7 ice types → Ch8/9 applications → App. A calibration

## Available primitives in seaice/core/
| function | module | book § | used by chapters | parity |
|---|---|---|---|---|

## Global pitfalls (MATLAB → Python) confirmed in this project
- (add as discovered; keep the list deduplicated)

## Data inventory
| file | chapter | tier | shows |
|---|---|---|---|

## Parity summary per chapter
| chapter | exact | near | approx | reimplemented | unverified | verdict |
|---|---|---|---|---|---|---|

## Setup findings (2026-09-09, /setup-project)
- Reference engine: **MATLAB R2025a** (25.1.0.2833191, prerelease) via `tools/run_matlab_ref.py`; Octave absent (fine).
- PDF offset = 31 (pdf 0-based index = printed page − 1 + 31); each `chapters/chNN.txt` starts on its chapter title.
- `MATLAB_ROOT/ch10` = Appendix A (`fisheye_calibration.m` → A.2 lens distortion, `orthoretification.m` → A.1).
- `ch6/Sea_Ice_Floe_Identification` and `ch7/Sea_Ice_Floe_Identification` hold the same 24 `.m` files (byte-identical);
  only `sea_ice_test.jpg` differs. Port once (ch6), reuse in ch7. `ch9/Model_Ice_Floe_Identification` shares the GVF/snake
  files too. ch5 duplicates ch2's `bound2im.m` / `boundaries.m` / `fchcode.m`.
- Data gaps: ch9 movie scripts need `dypic_05100_cam1_top.avi` (not shipped); ch8 MCD ships only `.mat` results.
  See `data/online/SOURCES.md` (git-ignored, on disk).
