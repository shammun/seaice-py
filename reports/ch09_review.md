# Chapter 9 — independent port review

Reviewer: `port-reviewer` (fresh context, read-only), 2026-09-11.
Inputs read: `chapters/ch09.txt`, all 7 new `.m` + `minboundrect.m`, `seaice/ch09_model_ice.py`, `seaice/core/video.py`,
the additive edits to `core/polygon.py` / `threshold.py` / `regionprops.py` / `synth.py` / `matlab_compat.py` /
`ch03_ice_pixel_detection.py` / `ch06_gvf_snake.py`, all 7 scripts, `tests/test_ch09.py`,
`reference/ch09/{fixtures,make_refs}.py`, both patch JSONs, `reports/ch09_verification.md`.
Independently re-ran the ch09 suite (**48/48 pass, 403 s**), three Python probes and one MATLAB R2025a probe.

Tally: **1 MUST-FIX · 11 SHOULD-FIX · 4 NITS.**

---

## MUST-FIX

### M1 — `rect()` diverges from `rect.m` on the two degenerate `minboundrect` branches, on inputs MATLAB *accepts*, while the docstring claims `Parity: exact`
`seaice/core/polygon.py:552-557` · label `seaice/ch09_model_ice.py:560` · report claim `reports/ch09_verification.md:239`
("a **single** pixel is fine").

MATLAB R2025a probe:

```
bw = false(12,20); bw(3,4)=true; s = rect(bw);
size(s(1).Vertices)   ->  1   10          % NOT 5x2
s(1).Vertices         ->  4 4 4 4 4 3 3 3 3 3
bw2(5,5)=true; bw2(5,6)=true; s2 = rect(bw2);
size(s2(1).Perimeter) ->  2    1          % NOT a scalar;  P2 = [2; 2]
```

Cause, in `minboundrect.m`:

* **`case 1`** — `rectx = repmat(x,1,5)` on a 1x1 column gives a **1x5 row**, so `rect.m:56` `v = [rectx, recty]` is a
  **1x10**, not a 5x2 ring. Consequence: MATLAB's `model_ice_model.m:32` `v(2,1,1)` then **errors** ("index out of
  bounds"); our port silently gets `0/0 = NaN`, rejects the floe and carries on.
* **`case 2`** — `perimeter = 2*sqrt(diff(x).^2 + diff(y).^2)` is evaluated on the **3-element closed** `x = [x1;x2;x1]`,
  so `diff` has two entries and `Perimeter` is a **2x1 vector**. The port at `polygon.py:556` takes `np.diff(x[:2])[0]`
  and returns a scalar.

Both branches are unreachable from `model_ice_demo.m` (`bwareaopen(.,20)` precedes `rect`), so **no shipped number
moves** — but `rect()` / `minboundrect()` are public API, the label is an unqualified `exact`, and D3/O7 currently
assert the single-pixel case is clean. `tests/test_ch09.py:824`
(`test_matlab_refuses_a_collinear_component_but_this_port_does_not`) asserts only `len(S)==2 and S[1].Area==0.0` and
never touches `S[0]`.

**Fix (code + docs + test).** Preferred: reproduce MATLAB's literal shapes in the two degenerate branches — but only if
ch06's 541 tests stay green, since `minboundrect` is a ch06-verified `exact` primitive. If reproducing them would
disturb ch06, instead qualify both docstrings, record a new deviation **D9** and extend **O7**, and pin MATLAB's actual
degenerate shapes in a test.

---

## SHOULD-FIX

### S1 (docs, 4 places) — the N4 "proof" is vacuous
`core/threshold.py:616-623` ("Proof that the caption means this and not `ic`") · `ch09_model_ice.py:209-212` ·
`reports/ch09_verification.md:194` · `analysis/ch09.md:268`.
With six **equal** blocks, `sum(num)/(r*c) = sum(ic_b * rc/6)/(rc) = mean(ic_b)` **exactly**. Measured on the Tier-3
tank: `ic = ic_mean = 0.8673501526073122`, `ic == ic_mean` is `True`. N4 is reproduced arithmetic, but it **cannot
discriminate the two definitions**. ch9's deletion of `local_Otsu.m:50` is the only evidence, and it is evidence about
what the script computes, not about what the caption means. The test at `tests/test_ch09.py:140` already says this
honestly — **the report contradicts its own test.** Rewrite the three "proves/proof" sentences.

### S2 (code) — `np.hypot` is not MATLAB's `sqrt(dx^2+dy^2)`, and it feeds a strict discontinuous test
`ch09_model_ice.py:673-674`. `model_ice_model.m:32-33` writes `sqrt((..)^2+(..)^2)`; `np.hypot` is a different, more
accurate algorithm and can differ in the last ulp — and `k` is then compared with **strict** `<`/`>` against 0.4/2.5
(E7). Use `np.sqrt(dx*dx + dy*dy)`. Zero cost, removes a class of doubt.
Same at `ch06_gvf_snake.py:549-550` (`_minrect_sides`), where it is only the `reimplemented` branch.

### S3 (code) — `gvf_distance(stop=...)` is never validated
`ch06_gvf_snake.py:655` (signature); consumed only at `:580` as `if stop == "count"`, so `stop='counts'` / `stop='N0'`
silently selects the **shipped** behaviour with no error. The neighbouring `ratio=` **is** validated (`:513`).
Add `if stop not in ("criteria", "count"): raise ValueError(...)` plus a `pytest.raises` beside the existing
`ratio='bbox'` case (`tests/test_ch09.py:400`).

### S4 (docs + test) — the end-to-end consequence of the 28-px `bw1` residual is unmeasured
`reports/ch09_verification.md:258-261` (D6) records only `bw1` = 28 px. Reviewer ran the full chain from our own
`gvf_distance` against `ch09_demo.mat`:

```
bw4 xor 28 / 13756        nS 30 = 30      nM 23 = 23
max |Area diff| (sorted)  12.34 px^2
Sum rect area   py 12257.80   ml 12243.54
Intersection flag matrix  NOT equal:  py 36 entries, MATLAB 34
```

Both orchestrator statements hold: the `exact` rows 5/6 are correct **only** on MATLAB's own `bw4`, and end-to-end the
overlap flags move (36 vs 34), which also moves `rect_ice_concentration`. Add these numbers to D6 with a ceiling
assertion (as `bw1 <= 35` is pinned at `tests/test_ch09.py:935`), so a regression in the model stage cannot hide behind
the count-only assertions at `:938-939`.

### S5 (docs) — `segment_video` / `tiled_segmentation` are smoke-only but labelled `reimplemented`; G2 has no Open item
`reports/ch09_verification.md:148-149`, count line `:154` ("`unverified` 0"). `tiled_segmentation` is never executed
successfully by any test (`tests/test_ch09.py:978-987` checks only the `DEVIATION` marker and a `ValueError`);
`segment_video` has one smoke run. By the project vocabulary that is **`unverified`**, not `reimplemented`. Either
relabel both and add an Open item for **G2** (currently only under "Gaps", not in O1-O9), or state explicitly why a
no-reference wiring counts as `reimplemented`.

### S6 (docs) — a retracted R13 claim is still live in a shipped docstring
`core/synth.py:1584-1585`: "`floe(k) = []` **delete** the element in MATLAB (risk R13)". R13 was corrected in this very
chapter — MATLAB **raises** (`MATLAB:matrix:singleSubscriptNumelMismatch`), and `ch09_model_ice.movie_floe`'s docstring
says so. Fix the synth docstring to match.

### S7 (docs — for the KNOWLEDGE phase) — `knowledge/` still carries the two statements ch09 disproved
`knowledge/CUMULATIVE.md:11` ("Ch9 block_threshold.m (same code)"), `:40` ("block_otsu (= block_threshold.m = ch3
local_Otsu.m)"), `:307-308` ("shares 13" and "ch9 `block_threshold.m` = ch3 `local_Otsu.m`"), and
`knowledge/function_map.md:54` (the `local_Otsu.m / ch9/block_threshold.m` row, no `compare=` parameter).
The reuse count is **14**, not 13, and the two files differ in one **numeric** hunk (`>=` vs `>`).

### S8 (docs) — `reference/ch09/patches.json` is not a faithful record of the patch actually applied
`patches.json:32` records the `movie_otsu` replacement block without the two lines `make_refs.py:154` really writes
(`I1first = I1(1).cadata; I2first = I2(1).cadata;`) and without them in the `save` list — yet `tests/test_ch09.py:720`
**consumes** `d["I1first"]` / `d["I2first"]`. It also records the removed range as "50-53" where line 50
(`k = 1 : numFrames;`) is kept (`make_refs.py:137` deletes 51-53 only). Same class as the `minboundrect` indentation nit
already logged at `:64`. Since `patches_verified.json` is generated and correct, either delete the hand-written
`patches.json` and point the report at `patches_verified.json`, or regenerate it.

### S9 (test) — `component_criteria(ratio='minrect')` has no known-answer assertion
`tests/test_ch09.py:385-401` asserts only that the two rules **disagree** (`k_e=[0]` vs `k_m=[0,2]`,
`not allclose(rl_e, rl_m)`). The fixture's first component is `bw[5:15, 5:35]`, a pixel-perfect rectangle whose
min-area-rectangle ratio is exactly `29/9 = 3.2222...`. Assert that value (and `1.0` for the diamond) — otherwise a
`_minrect_sides` returning the bounding-box ratio, or `short/long`, would still pass.

### S10 (docs) — Fig. 9.15's report row does not name its input
`reports/ch09_verification.md:183` — "reproduces MATLAB's accepted set and all 34 overlap flags".
`reference/ch09/make_compare_figures.py:155-157` builds that panel from **MATLAB's own `bw4`** (the figure's suptitle
says so; the report row does not). Per the ch07 lesson, add "on MATLAB's own `bw4`"; from our `bw4` it is 36 flags (S4).

### S11 (docs) — bookkeeping in the report
`:164` / `:154` say "9 permanently `unverified` (N2, N3, N5, N6, N9-N12, N19)" while the table at `:205` also marks
**N18** unverified (O4) — **10** by the report's own reckoning. `:312` says "24 L1 and 24 L2/L3"; the file actually has
48 tests of which **19** carry `@needs(...)` and 29 do not. N1, N7, N16, N17 and N20 (analysis §4) appear nowhere in the
"Numbers from the text" table.

---

## NITS

* **N-a — Algorithm-7 ordering.** `ch06_gvf_snake.py:576-586` tests `k.size == 0` *before* the `N0 == N1` count test;
  Algorithm 7 (p. 206) puts line 7 (`N0 != N1`) *before* lines 8-10 (`k`). Only the recorded `stopped_on_count` flag can
  differ; `bw1` is identical either way.
* **N-b — layout ambiguity.** `ch09_model_ice.py:1013-1016` and `core/video.py:83` both resolve `(H,W,3,3)` / `(3,H,W,3)`
  to `NHWC`, documented in both places — but with slightly different tests (`shape[2]==3 and shape[3]!=3` vs
  `shape[3]==3`). Worth unifying.
* **N-c — `rect(bw, "")`.** `ch09_model_ice.py:562-564` raises; `rect.m:32` is `if (nargin<3) || isempty(metric)`, so
  MATLAB defaults to `'a'`. Sub-case of D2, not recorded.
* **N-d — `test_N4...`** `tests/test_ch09.py:134` asserts `f"{mean:.2f}" == "83.15"` and comments "the book *truncates*
  to 83.14". `num2str(83.146666...)` is `'83.1467'`, so the caption was hand-typed; "truncates" is a hypothesis, not a
  finding.

---

## Checked and found CORRECT (coverage record)

* **Every `movie_otsu.m` semantic**: E4 (`if scalar >= vector` => `all` => running max, and `max(t)*255 == max(t*255)`
  under IEEE monotone scaling), E5 (`n_final[j] = sum_{m>=j} count_m` = reverse cumsum; `IC` escapes E5 but not E4),
  `r2/c2` from crop∩box, denominator 77 115, blank-before-crop, `im2bw` at the **per-frame** level with strict `>`.
  The default `count_rule='ge'` is genuinely pinned: on the Tier-3 video `ge - gt` is **18-271 px on all 24 frames**, so
  the `atol=0.0` IC comparison at `tests/test_ch09.py:716` discriminates it. Thresholds run 122->143->100, so E4 is
  observable — not a monotone fixture.
* **`movie_kmeans.m`**: `si` taken before flattening, column-major `(:)` and `reshape`, `uint8()` cast not `im2uint8`,
  the `find(s==max(s))` scalar-assignment error reproduced, no stale-`s` bug (unlike ch3). `approx` label honest (D4).
* **`movie_floe.m`**: level-free `im2bw` (RGB->rgb2gray->`>127.5`, MATLAB-probed), `bwareaopen(.,20,4)`, `bwlabel(.,4)`,
  `regionprops('basic') = {Area,Centroid,BoundingBox}`, R13 corrected to **raise** and pinned against `r13.mat` in both
  error forms.
* **`rect.m`'s `(c, r)` swap** is real and is caught. The reviewer independently verified that the row-major vs
  column-major `find` order is **provably irrelevant** here (`np.unique(np.mod(edgeangles, pi/2))` sorts, so the caliper
  loop and its tie-break are order-invariant) — max diff over all 30 components of `bw4` is **0.0**. The swap test at
  `:805` is honest (Area cannot discriminate; `Vertices` on a non-square canvas can).
* **`model_ice_model.m`**: `k = |v1-v2|/|v3-v2|` indices, the strict symmetric band (E7, MATLAB-pinned at
  0.39/0.40/2.40/2.50), `roipoly` union accumulation, `Intersection` indices 1-based into `ss` (not `S`), field name
  `Intersection` vs ch8's `Intersect`, E9's opposite consequence vs ch08.
* **`clip_polygon_convex`**'s emptiness contract is genuinely equivalent to `polybool` for what the code consumes:
  10/10 geometries including the three zero-area relations; `drop_degenerate=False` wrong on exactly 3; the degeneracy
  filter (consecutive-dup removal + `|signed area| <= 0`) handles hair-gap / hair-overlap correctly.
* **`block_otsu(compare=)`**: default still `'gt'`, ch3 untouched; `block_threshold` wires `'ge'`; the tie fixture has
  pixels exactly on the threshold **and** a half-integer level (104.5), and the real Tier-3 tank has 35-59 such pixels
  per block — the ch08 empty-band trap is excluded.
* **Additive-edit safety**: `clip_polygon_rect` keeps the exact axis-aligned intersector and the
  (xmin,xmax,ymin,ymax) order; `_num2str` is a pure alias of the promoted `core.matlab_compat.num2str`;
  `regionprops('basic')` and `BASIC_PROPERTIES` are additive; `component_criteria` / `gvf_distance` defaults unchanged.
* **R8 / Table 9.1**: verified against the book text (`chapters/ch09.txt:41-49`; table headers "Floe size 1 (45%) /
  2 (40%) / 3 (15%)" = 0.50 / 1.00 / 1.50 m). 9x0.50 : 4x1.00 : 1x1.50 = 4.5 : 4.0 : 1.5 of 10.0 m = **45/40/15 % in the
  printed order**, and because the strips are cross-cut into squares the area share equals the width share.
  **The orchestrator's correction is right**, and the test at `:191` also proves the other two readings do not match.
* **Patch audit (numerics)**: `make_refs.py`'s edit dictionaries diffed against the shipped files line by line.
  `movie_otsu` deletes 51-53 and 59-63, 65 (keeps 50 and the blank 64); `movie_kmeans` deletes 65, 67-69, 73-77, 79
  (keeps 66); `movie_floe` only line 5; `block_threshold` only the `imread` literal (same file name);
  `model_ice_demo` un-comments the authors' own three statements. **No numeric expression is altered anywhere.**
* **Honesty**: all seven scripts print an explicit "SYNTHETIC (Tier 3) ... the book's numbers are NOT reproducible and
  are printed for orientation only" banner; no book number is computed from synthetic data;
  `segment_video` / `tiled_segmentation` carry `# DEVIATION` markers and appear in no parity assertion.

---

## Dispatch

| Finding | Owner |
|---|---|
| M1 (shapes + docstrings), S2, S3, S6, N-b, N-c | **porter** (`seaice/`) |
| M1 (test), S4 (test+D6), S5, S8, S9, S10, S11, S1 (report) | **verifier** (`tests/`, `reports/`, `reference/`) |
| S1 (analysis wording), N-d | **orchestrator** |
| S7 | **knowledge-keeper** (phase 6) |
| N-a | accepted as-is (flag-only, `bw1` identical) — recorded, not changed |
