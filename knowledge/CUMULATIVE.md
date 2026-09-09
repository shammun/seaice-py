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
- Reference engine: **MATLAB R2025a** (25.1.0.2833191, prerelease) via ; Octave absent (fine).
- PDF offset = 31 (pdf 0-based index = printed page − 1 + 31);  start on each chapter title.
-  = Appendix A (fisheye_calibration.m → A.2 lens distortion, orthoretification.m → A.1).
-  and  are the same 24 .m files (byte-identical);
  only  differs. Port once (ch6), reuse in ch7. ch9  shares the GVF/snake
  files too. ch5 duplicates ch2 .
- Data gaps: ch9 movie scripts need  (not shipped); ch8 MCD ships only  results.
  See .
