# Chapter 5 — independent review (port-reviewer, 2026-09-09)

Reviewed against `chapters/ch05.txt` §5.1–5.3, the authors' `.m` files, R2025a `eml/watershed.m` + `FifoPriorityQueue.m` +
`imimposemin.m`, the rendered Fig. 5.15/5.16 pages, probes, and a re-run of `tests/test_ch05.py` (394 passed, 1 xfailed).
**No correctness bug was found in the numerical port.** Disposition column filled in by the orchestrator.

## Must fix
| # | Finding | Owner | Disposition |
|---|---|---|---|
| M1 | `tests/test_ch05.py` `test_convex_blob_spurious_lines_are_removed` is vacuous: a convex blob's distance transform has one minimum, so 0 junction lines → `all([])`. The "line removed" branch of `neighboring_region_merging` had no L1 coverage. Replace with a near-circular peanut (two r=20 discs centred (35,42)/(35,48) on 70×90) → 2 basins, 1 line, 0 concave endpoints, removed, floes 2 → 1, `seg == bw`. | verifier | fixed |

## Should fix
| # | Finding | Owner | Disposition |
|---|---|---|---|
| S1 | Docstrings in `seaice/ch05_watershed.py` (`JunctionLine.region`, `neighboring_region_merging`, `freeman_concave`) and the print in `scripts/ch05_main.py` claim the reconstructed region is label-valued `{0,k}`; `imreconstruct(g, connect)` with marker `{0,1}` gives `{0,1}`. | porter | fixed |
| S2 | `junction_endpoints(rule='ge3')` applies `abs` before the `≥ 3` threshold, so it is not exactly the text's rule (admits background pixels with ≥ 3 line 4-neighbours). Drop `abs` for `'ge3'` or document. | porter | fixed |
| S3 | No L1 test pins `watershed`'s default 8-connectivity (`[[1,5,5],[5,1,5],[5,5,5]]` → 1 basin at 8, 2 at 4). | verifier | fixed |
| S4 | `test_non_sequential_and_rules` only checks the floe count; assert per-line `removed`/endpoints equal the default run. | verifier | fixed |
| S5 | `core/morphology.py` 3×3 `conn` heuristic maps any matrix with a corner to 8; raise for non-cross/ones patterns. | porter | fixed |
| S6 | `core/watershed.py` rejects 3×3 `conn` matrices while the report says the matrix forms were compared at 0 px (they were compared against scalar calls). Accept the matrix form or reword the report + Deviation 2. | porter + verifier | fixed |
| S7 | Report "Numbers" row / analysis risk 9: a 4-connected diagonal staircase *is* what the book predicts (p. 99). The real departure from "1-pixel-thick" is thick ridges on plateaus (10 2×2 ridge blocks on the gray-image watershed, 0 on the distance maps). | verifier | fixed |
| S8 | Report Open items: add the former L1 coverage gap, and the `RuntimeWarning: invalid value encountered in add` in `imimposemin` for `±Inf` images (`h = Inf`, MATLAB identical; suppress/document). | porter (warning) + verifier (report) | fixed |
| S9 | Report parity count: 24 `exact` rows, not 23. | verifier | fixed |

## Verifier open item folded in
| # | Finding | Owner | Disposition |
|---|---|---|---|
| V1 | `regional_minima_by_reconstruction` on an all-`+Inf` (constant) image returns all-False; MATLAB `imregionalmin` returns all-True. Return all-True when `I` is constant; flip the strict xfail to a pass. | porter + verifier | fixed |
| V2 | Docstrings: MATLAB `watershed` rejects int16/int32 input (MATLAB cross-check of the int16 fixtures is on `double(X)`); `imimposemin` MATLAB accepts NaN, port raises. | porter | fixed |

## Fine / verified (reviewer)
- `core/watershed.py` vs `eml/watershed.m`: column-major initial scan, push-only-unvisited, `(priority, order)` FIFO heap
  equivalent to `FifoPriorityQueue.m`, set-based label decision, no propagation from ridge pixels, `max(A(nb), p)`, one push per
  pixel; `neighbour_offsets` = ascending linear index; label dtype documented.
- `imimposemin`: `h` rule, class arithmetic, saturating integer add, `intmin/intmax` markers, per-class complement, ±Inf fixtures identical.
- Eq. 5.2, §5.1.3 Steps 1–2 (also for g ≥ 0), Eqs. 5.1/5.3–5.8 immersion, Eqs. 5.9–5.18 indices and wrap terms, `3 ≤ D ≤ 10`
  inclusive, `'cww'` = clockwise, `b{1}` = bwlabel-order first object.
- Fig. 5.15 patterns/kernel and Fig. 5.16 matrix transcriptions match the rendered pages; `≥ max` vs `≥ 3` documented.
- `main.m`: `bitand`, `bwlabel(f,4)`, sequential `seg` update, reconstruct, `intersect(...,'rows')`, (row, col) conventions, `bound2im` placement.
- `marker_watershed` centroid floor/index order, `distance_watershed` `find` order, `inverse_distance_map` single arithmetic,
  `sobel_magnitude` unscaled/replicate, `gradient_watershed` close∘open with `ones(7)`, `direct_watershed` bwlabel of the overlay.
- Tests catch flipped `bwdist` complement, transposed offsets, wrong `-D` sign, concave-wrap off-by-one; every script CLI option runs.
