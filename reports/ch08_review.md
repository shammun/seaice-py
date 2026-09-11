# Chapter 8 — independent port review

**Reviewer:** `port-reviewer` agent, fresh context, read-only.
**Date:** 2026-09-10. **Reviewed at commit:** `46e6c21` (after the verify phase).
**Inputs:** `CLAUDE.md`, `knowledge/CUMULATIVE.md`, `analysis/ch08.md`, `reports/ch08_verification.md`,
`chapters/ch08.txt`, the nine `.m` files (`MATLAB_ROOT/ch8/MCD/*.m`, `ch8/SIFI/*.m` and the four deferred from
`ch6/`/`ch7/`), `seaice/ch08_applications.py`, `seaice/core/{fitting,icestruct,stats,plotting,polygon,
clustering,matlab_compat}.py`, `scripts/ch08_*.py`, `tests/test_ch08.py`, `reference/ch08/make_refs.py`,
`reference/ch08/patches.json` and the reference `.mat` files loaded directly.

**Independent work done, not just reading:** re-ran `tests/test_ch08.py` (68 passed, 11 script tests
deselected); loaded `misc.mat`, `fit.mat`, `model.mat`, `window.mat`, `mcd.mat`, `colorhist.mat` and
`orphan.mat` by hand to check quoted numbers against the stored arrays; ran a 4 000-case randomised
comparison of the port's fast crossing test `_segments_cross` against `core.polygon.polyxpoly` on integer
lattice rings including collinear and single-point-touch cases (**4 000/4 000 agreement**); redid the
report's arithmetic spot-checks.

**Outcome: 0 must-fix, 7 should-fix, 10 nits.**

---

## Must fix

**None.** No defect was found that changes any computed value.

Everything numeric in the §8.3 chain is pinned at 0.0 against MATLAB on the real 2 888-floe field, and each of
the chapter's MATLAB-semantics traps is reproduced *and* pinned in both directions: the `if xx ~= NaN` idiom,
the column-major `c(1), c(2)` read of a 2x2 `Center`, the 126-point non-closing `t = 0:0.05:6.28`, the
`[Raw_counts(2:end) 0]` one-bin shift, the `n + inter/2` bar shift, and the `hist`-on-centres FSD.

The MATLAB-side patch set in `reference/ch08/patches.json` was verified **edit by edit**: all 46 + 28 + 3 + 13
+ 13 + 19 edits are `figure` / `bar` / `colormap` / `colorbar` / `set` / `axis` / `imshow` / HG1 removals, two
`cd`/`load` re-points, the syntax-error `legend` line, and three graphics statements turned into assignments.
**No numeric expression was altered:** `count_error`, `caxis`, `rgbImage(y(j),x(j),:)`, `convhull`, `roipoly`,
`polygeom`, `polyxpoly`, `t = 0:0.05:6.28` and `d < r+r0` are verbatim.

---

## Should fix

### S1 — a (row,col) <-> (x,y) transposition in the Appendix-B `Param` block would pass all 79 tests

`seaice/ch08_applications.py:978`

```python
NumPix_y, NumPix_x = np.asarray(index_floe).shape[:2]    # lines 50-51 (size(.,2) = columns = x)
```

This is **correct** against `SeaIce_Image_Structure.m:50-51`
(`NumPix_x = size(index_floe, 2); NumPix_y = size(index_floe, 1);`) — but nothing tests it. All four fixtures
that reach this line are square: `reference/ch08/fixtures.py:78` `(110, 110)`, `:92` `(450, 450)`, `:100`
`(80, 80)`, `:140` `shape=(y1, x1) = (200, 200)`. `test_sea_ice_image_structure_parity`
(`tests/test_ch08.py:732-733`) compares `NumPix_x`/`NumPix_y` against MATLAB's, and `PixScale_x = 50/N`,
`PixScale_y = 18/N` are unaffected by a swap when `N` is the same in both axes. Swapping the two names on
line 978 leaves every assertion true.

*Verify the fix:* change one `model_fixtures()` entry to a non-square `shape` (e.g. `(110, 137)`), regenerate
`reference/ch08/model.mat`, and confirm the test fails when the names are swapped.

### S2 — the book's description of Fig. 8.15 has the opposite sign from the shipped code, and this is nowhere recorded

Book p. 188 (`chapters/ch08.txt`, PDF p. 219): *"By subtracting the histograms of Figure 8.12(a) from Figure
8.10, we get the floe size distribution error"* — i.e. identified - polygonized = Fig. 8.11 - Fig. 8.14.
`color_hist_comparison.m:29` computes `z_d = z - z0` with `z = hist(floe.Area)` (polygonized, Fig. 8.14) and
`z0 = hist(ice_floe.Area)` (identified, Fig. 8.11), i.e. **Fig. 8.14 - Fig. 8.11**.

The port follows the code (`seaice/ch08_applications.py:1155`; the docstring at `:1128` says "the polygon
histogram *minus* the pixel histogram"), which is right. But `analysis/ch08.md:207` states the opposite
(`Fig. 8.15 | 188 | Fig. 8.11 - Fig. 8.14 histogram difference`), contradicting `analysis/ch08.md:239` in the
same file, and `reports/ch08_verification.md` has no erratum for it (E1-E10 do not cover it). The Fig. 8.15
figure verdict ("a signed difference oscillating within roughly +/-10 counts about zero") would not notice a
global sign flip.

Reading `reports/ch08_verification/figures/fig_8_15_compare.png` (book panel, right), the printed first bar is
**positive**, matching the code's `z - z0`, so the *text* is the erratum.

*Fix:* add erratum E11 with this quotation, correct `analysis/ch08.md:207`, and make the Fig. 8.15 verdict
state the sign of the first bin explicitly (ours is `z_d[0] = +10` on the §8.2 run) so a flip would be caught.

### S3 — the parity counts in the report do not match its own table

`reports/ch08_verification.md:77-78`: *"Parity counts (17 rows with a label): exact 10 · near 4 · ...
unverified 2 ... plus 1 row labelled display"*. Parsed programmatically, the table has **19 rows**:
9 pure-`exact` + 1 `exact (arrays)/display` = 10 exact, **5** pure-`near` rows (`fitting_iceFloes_distribution`,
`PowerLaw...`, `three_fitting...`, `core.fitting.lsqcurvefit`, `matlab_colon`), 1 pure-`display`,
2 `unverified`, plus the mixed `sea_ice_model` row. So "17 rows" and "near 4" are both wrong.

*Verify:* rerun the same count; 9 + 5 + 1 + 1 + 2 + 1 = 19.

### S4 — risk R17's own mitigation was not carried out, and the report calls the grown-array rule "confirmed" in the one case where it cannot be

`seaice/ch08_applications.py:375-378` computes the default `image_shape` as `(max y, max x)` over the painted
pixels, reproducing MATLAB's array growth at `plot_color_bar_and_floe.m:79`. `tests/test_ch08.py:281-282`
asserts `rgb_size == [627, 1114, 3]` — but on the shipped field **both maxima are attained**, so that
assertion cannot distinguish "grown to the pixel maxima" from "pre-allocated to the image size", and the
`image_shape` override branch is never executed. `reports/ch08_verification.md:58` nonetheless says *"R17's
grown-array claim is confirmed, both maxima are attained"* — the second clause is exactly why the first is not
evidence. This is the ch07 lesson "a test that cannot distinguish the fix from the bug is not a test".

*Fix:* add an L2 fixture whose floes never touch the last row/column (e.g. re-run `ch08_pcbf_ref` on
`IceImage.Floe(1:50)`), save `size(rgbImage)`, and assert it equals the pixel maxima, not the image size.
List it as an open item until then.

### S5 — `matlab_colon` is a general MATLAB-semantics primitive living in a chapter module (rule 9)

`seaice/ch08_applications.py:93-114`. The colon operator's element-count/endpoint rule is not chapter-8
specific: it is already needed by `core`-level code paths (`_color_hist_core` at `:1073`, the Appendix-B FSD at
`:992`, `_fit_curve` at `:513`) and ch09's model-ice work will need `0:step:stop` vectors.
`seaice/core/matlab_compat.py` has no colon helper.

*Fix:* move it to `core/matlab_compat.py` (re-exported from `ch08_applications` so existing tests keep
working) and add it to `knowledge/function_map.md` during the knowledge phase.

### S6 — §8.1's k-means uses ch06's Lloyd, not the book's own §3.2 k-means, without saying why

`seaice/ch08_applications.py:1199` / `:1215` call `core.clustering.kmeans_lloyd(..., init="kmeans++",
seed=seed)`, and the docstring at `:1182-1188` / `:1193` cites *"k-means of §3.2"* while labelling it `approx`
"because of its RNG". But ch03 ships the authors' own deterministic `kmeans.m` port,
`core.clustering.kmeans_gray`, labelled `exact`, and that is precisely the §3.2 routine the book refers to on
p. 179 ("the k-means clustering method ... we identify three clusters (k = 3)"). Using the `approx` RNG
routine where an `exact` deterministic one exists is a silent downgrade under rule 4.

*Fix:* either route `method='kmeans'` through `kmeans_gray` (grayscale, k = 3) and keep `kmeans_lloyd` behind
an option, or state in the docstring **and** in the report why Lloyd was preferred.
*Verify:* repeat `test_kmeans_k2_is_approximately_otsu` with `kmeans_gray` and report the agreement for both.

### S7 — `sea_ice_field` (the §8.2 driver) is never executed by any test

`seaice/ch08_applications.py:1231`. `tests/test_ch08.py:1038-1041` runs every book-data script with
`--source iceimage` / `--input iceimage`, which bypasses `sea_ice_field` entirely; no unit test calls it. The
`unverified` label in the parity table (report line 75) and O2 cover the *counts*, not the fact that the
function is unexercised.

*Fix:* add one `CLI_CASES` entry `("ch08_color_hist", ["--input", "demo", "--source", "raw"])` (it reuses the
cached `stage3_*.npz`, so it is cheap after the first run), or a direct smoke test on a small crop, and say so
in the report.

---

## Nits

**N1** — `core.plotting.matlab_jet`'s docstring says "Display only." (`seaice/core/plotting.py:272`) while its
next paragraph correctly notes that `jet(30)`'s rows *are* consumed numerically: `color_M(index,:)` is painted
into `rgbImage`, which the report compares byte for byte, and the report labels the row `exact`, not
`display`. Drop the "Display only." sentence.

**N2** — over-tight number in a docstring. `seaice/ch08_applications.py:759` claims "`polygeom` <= 1e-12
against the shipped structure"; the report (line 62) measured `Area <= 1.8e-12` on the L3 field and the test
asserts `< 1e-11` (`tests/test_ch08.py:648`, `:837`). Quote 1.8e-12 or 1e-11.

**N3** — over-claim about the convex hull. `seaice/ch08_applications.py:713` says the hull "is indeed the
minimum-area bounding polygon". The book (p. 184) says "bounding minimum-area-polygons"; the hull is the
minimum-area **convex** bounding polygon only. Add "convex".

**N4** — the colon fixture pins the wrong endpoint. `reference/ch08/make_refs.py:273` saves
`colon_21_79_3971`, but the FSD line the port actually evaluates is `min_x : inter : max_x` = `21:79:3979`
(`SeaIce_Image_Structure.m:98` with `min = 21, max = 3979, inter = 79`). `21:79:3971` has `(b-a)/d` exactly
integral, so it exercises none of `matlab_colon`'s truncation logic. The real case is covered transitively by
the 51/51 FSD triplet match, but the fixture as named proves less than it appears to.

**N5** — column-major inconsistency for `Area`. `seaice/ch08_applications.py:841` and `:779` read
`brash_ice[i].Area` with `.ravel()[0]` (C-order) while `Center` is read column-major via `_matlab_c1c2`
(`:689`). All 3 452 shipped `Brash.Area` values are scalars (the 4 pieces with a 2x2 `Center` have scalar areas
11, 11, 12, 13), so it is inert — but use `ravel(order="F")[0]` for consistency, or raise on a non-scalar
(MATLAB would produce a `k x 126` matrix and `polyxpoly` would error).

**N6** — display-only float ordering. `scripts/ch08_main_WL_new.py:120-121` draws the polygon white dots as
`(NumPix_y - c_y) * lop`, whereas `plot_color_bar_and_floe.m:106` writes `Y_limi*length_over_Pixel -
centre(i,2)*length_over_Pixel`. The test (`tests/test_ch08.py:306-308`) uses the M-file's form and matches
MATLAB at 0.0; only the script's drawing differs (~1e-13). Harmless, but the script should use the same
expression.

**N7** — stale pre-check number in `analysis/ch08.md`. Lines 181 / 371 (N18, L3-4) quote
`eta = [11.85792898, 1.37036153]`; the verified answer against MATLAB is `[11.85785671, 1.37035854]`
(`epsilong` in `reference/ch08/fit.mat` is `[11.8578567, 1.37035854]`). O4 records a superseded pre-check for
Fig. 8.19 but not this one.

**N8** — D5 mixes inputs in one sentence. `reports/ch08_verification.md:161-166` says the speed-ups were
proven inert "on 60 real floes and 60 real brash pieces" and then quotes "900 of 160 262 pairs ... on the
227/240 window". The 160 262 = 227*226 + 2*227*240 figure belongs to the full-window run (which *is* the
stronger evidence: `test_sea_ice_model_parity_on_a_real_window` runs the defaults against MATLAB's literal
loop). Separate the two sentences.

**N9** — unused reference variables. `x_plot`, `y_fit_powerlaw0`, `sorted_floe_size`, `a_sorted_floe_size`,
`jac`, `out_iterations/funcCount/firstorderopt`, `legend_err`, `{tag}_bC/_bA/_ver`, `ch_zs_/ch_izs_/ch_cf_`,
`cc_color_/cc_cmin_`, `st_polyV`, `st_bCircle` are saved by `reference/ch08/make_refs.py` but asserted by no
test. Two are worth promoting: `{tag}_bC` is the only evidence behind deviation D4 (`center2x2_bC` is 4x2 for
3 pieces — the report's claim is true, but only because the reviewer loaded the `.mat` by hand), and `x_plot`
is an untested `matlab_colon` case (checked by hand: 119 822 elements, <= 1 ulp).

**N10** — stray artefact. `reference/ch08/probe.mat` (`has_map`, `has_opt`, `lsq_defaults`, `ver_str`) is not
produced by `make_refs.py`. Git-ignored, so harmless, but it makes the "seven MATLAB sessions" table look
incomplete.

---

## Verified (checked and correct)

* **Eq. (8.1)** `L = sqrt(4A/pi)` — `core/stats.py:46` evaluates `a * scale**2 * 4.0 / np.pi` in the same
  left-to-right order as `main_WL_new.m:17-18/30-31`; bit-identical to the shipped `MCD_results.mat`
  (2 888 values, max |delta| = 0.0).
* **Eq. (8.2)** `Nc = N(>=L)/N_total` — `core/stats.py:82` uses `n - searchsorted(L, L, 'left')`, exactly
  `size(find(MCD>=MCD(i)),2)`; ties repeat, `Nc(min) = 1`, `Nc(max) = 1/N`; 0.0 vs MATLAB's `N_L`.
* **Eq. (8.3)** `eps1 * L^(-eps2)` — `core/fitting.py:210` matches `F_powerlaw0` character for character;
  `alpha = 1.37035854` vs MATLAB `1.37035854` (rel. 2.4e-10), rounds to the printed 1.3704. The text/code
  estimator mismatch (log-log OLS gives 1.8697; MATLAB's own `polyfit` = `[-1.86967809, 1.55640427]` in
  `fit.mat`) is correctly recorded and pinned in both directions.
* **`if xx ~= NaN`** — `_crosses` at `:191-206` is `~isempty`; MATLAB's three branches (`[1 2 3]` taken, `[]`
  not taken, `[1 NaN]` taken) are in `misc.mat` and asserted; containment is deliberately not detected and
  `strict_containment=True` is asserted to see it.
* **Column-major `c(1), c(2)`** — `_matlab_c1c2` at `:679-690`. Exactly 4 of 3 452 brash pieces have a 2x2
  `Center` (indices 2110, 2358, 3071, 3129), and the `center2x2` fixture discriminates the two readings by
  *which brash piece* MATLAB flags (`{2}`, and `3` absent).
* **`t = 0:0.05:6.28`** — 126 angles, last 6.25 rad, <= 1 ulp vs MATLAB's own vector, and the ulp proven inert
  (0 px on `bw_brash`).
* **AABB prefilter / cropped `roipoly` / fast crossing test** — the *defaults* (`prefilter=True`,
  `raster_crop=True`, `intersect_impl='fast'`, `strict_containment=False`) are the configuration compared
  against MATLAB on 227 real floes + 240 real brash (all four `Intersect` lists equal as sets,
  `bw_floe`/`bw_brash` 0 px) and on the whole 2 888/3 452 field. The crop-equivalence proof was traced
  independently through `core/polygon.py:203-226` (the `x_use` in `[2, N+1]` bound, the `y_use >= 3` guarantee
  when `r0 > 0`, the overflow clamp, and the per-column cumulative XOR restricted to `[minY-1, maxY-2]`) and
  it holds. 4 000 random integer-lattice ring pairs: `_segments_cross` and `core.polygon.polyxpoly` agreed
  4 000/4 000.
* **`SeaIce_Image_Structure.m`** — `v1(n,:) = []` (open ring), `int_max = [n(2:end)-1, max_x]`,
  `CovOther = nnz(index_residue)/(NumPix_x*NumPix_y)`, `NumFloes = length(floe)`; the `index_residue(1:7) = 1`
  fixture is written column-major on the Python side (`tests/test_ch08.py:721-723`). The shipped 51 FSD
  triplets reproduce exactly and the label recount is asserted to give 1861 != 1411 in 24 of 51 intervals.
* **`hist` on centres, `n + inter/2`, Eq. (7.6) colours, the nine tick integers** — `colorbar_area_ticks` is
  reused from ch07, not re-implemented; `d = fix((9698-198)/8) = 1187`, `ysh = 198:1187:9698` gives 9 values
  -> `20, 149, ..., 3487`, and `test_color_hist_tick_list_is_constants_only` explicitly disqualifies that
  match as parity evidence (the ch07 saturating-agreement lesson, correctly applied).
* **`matlab_jet`** — the R2025a `mod(m,4)==1` rule, bit-exact for `m = 1, 3, 4, 7, 30, 255`, with the
  pre-R2014 `mod(m,2)` variant computed explicitly and asserted to differ at 255 and agree at 30.
* **Patches** — graphics + plumbing only, as claimed; every quantity the removed lines consumed (`caxis`
  limits -> `caxis_lim`, `fill` colours -> `polyc`, `plot('w.')` -> `dotxy`,
  `z/n/color/zs/izs/ysh/YT/nbins/k/d`) is still computed and saved. `rgbImage(y(j),x(j),:)` is verbatim; only
  the in-loop `imshow` was dropped.
* **Reference log** — `outputs/ch08/verify/refs_log.json` shows 7 sessions, all `status: ok`, engine
  `matlab 25.1.0.2833191 (R2025a) Prerelease Update 2`, seconds 63.7/56.3/53.8/73.1/42.2/160.3/33.2 — exactly
  the report's table.
* **Arithmetic spot-checks redone**: 227*226 + 2*227*240 = 160 262; 2888*2887 + 2*2888*3452 = 28 276 408;
  2888 - 2383 - 490 = 15; Fig. 8.10 pre-image `C_min = 9` => `A = 1 px`, `C_max` in `{9693..9698}` =>
  `A` in `[3484, 3503]`.
* **Traceability (rule 3)** — all nine `.m` files have exactly one Python home, each docstring cites the file
  and line range, and the duplicated fit body (`fitting_iceFloes_distribution.m:42-51` =
  `PowerLaw_fitting_method_and_plotting.m:12-21`) is written once (rule 9) with the duplication documented.
