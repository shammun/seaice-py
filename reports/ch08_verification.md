# Chapter 8 verification — Sea Ice Image Processing Applications (book pp. 175–194)

Date 2026-09-10 · verify phase at commit `a400a03` (`ch08: port — 9/9 .m ported, 8 scripts exit 0, MCD 0.0
vs the gold .mat, alpha 1.370359`), **revised after the independent review** (`reports/ch08_review.md`, commit
`46e6c21`): see the "Review revision" section for every item and its disposition.

## Environment

python 3.11.5 · numpy 2.4.6 · scipy 1.17.1 · scikit-image 0.26.0 · opencv 5.0.0 · matplotlib 3.11.1
**Reference engine: MATLAB 25.1.0.2833191 (R2025a) Prerelease Update 2** via `tools/run_matlab_ref.py`
(`matlab -batch`, `set(0,'DefaultFigureVisible','off')`, `save(..., '-v7')`), driven by
`reference/ch08/make_refs.py`.  Octave was **not** used anywhere in this chapter.
Toolboxes exercised and licensed on this host: Image Processing, **Optimization** (`lsqcurvefit`),
**Mapping** (`polyxpoly`).

**Eight** MATLAB sessions, all `status: ok` (`outputs/ch08/verify/refs_log.json`).  Three of them were
re-run for the review revision (marked *): `misc` gained the real FSD colon case (N4), `model` was regenerated
after one fixture was made **non-square** (S1), and `grow` is new (S4).

| `.mat` | vars | seconds | |
|---|---|---|---|
| `misc.mat` | 26 | 213.0 | * re-run (N4; the host was loaded, hence 213 s against the first run's 63.7 s) |
| `fit.mat` | 19 | 56.3 | |
| `colorhist.mat` | 91 | 53.8 | |
| `model.mat` | 72 | 64.0 | * re-run (S1: the `touching` fixture is now 80x137, not 80x80) |
| `orphan.mat` | 39 | 42.2 | |
| `window.mat` | 24 | 160.3 | |
| `mcd.mat` (+ `mcd_rgb.mat`) | 26 | 33.2 | |
| `grow.mat` | 19 | 57.9 | * **new** (S4: `size(rgbImage)` on two subsets of `IceImage.Floe`) |

The `grow` session introduced **no new patch**: it calls the very same `ch08_pcbf_ref.m` copy the `mcd` session
uses.  The patch set is unchanged from the one the review audited edit by edit — 122 edits in total
(46 + 28 + 3 + 13 + 13 + 19), all graphics/plumbing.

(`reference/ch08/probe.mat` — `has_map`, `has_opt`, `lsq_defaults`, `ver_str` — also sits in that folder but is
**not** produced by `make_refs.py`: it is the one-off toolbox/licence probe run before the reference script was
written, kept because it is the provenance of the `optimoptions` default table quoted below.  It is not one of
the eight sessions above.  Review item N10.)

### Patches applied to the original `.m` files (all recorded verbatim in `reference/ch08/patches.json`)

`reference/ch08/make_refs.py` copies every `.m` into `outputs/ch08/verify/scratch/` under a non-shadowing name
and patches it **line by line**, storing the removed source line beside its replacement.  Three classes only:

| original | scratch copy | edited lines | what was changed |
|---|---|---|---|
| `SeaIce_Image_Structure.m` | `ch08_struct_ref.m` | 0 | verbatim, renamed only |
| `PowerLaw_fitting_method_and_plotting.m`, `three_fitting_method_and_plotting.m`, `polygeom.m` | same name | 0 | verbatim |
| `plot_color_bar_and_floe.m` | `ch08_pcbf_ref.m` | 46 | graphics only + output plumbing. Removed: `figure`, `hold`, the 100-bar `bar`/`set` loop, `colormap`, `colorbar`, `set(gcf/gca)`, `xlabel`/`ylabel`/`grid`/`axis`, and the `imshow(rgbImage)` **inside** the 2888-iteration loop (risk R17). The three graphics statements that carried numbers were turned into assignments: `caxis(...)` → `caxis_lim`, `fill(x,y,color_M(index,:))` → `polyx`/`polyy`/`polyc`, `plot(...,'w.')` → `dotxy`. The `rgbImage(y(j),x(j),:) = ...` assignment (a computation) is verbatim. |
| `main_WL_new.m` | `ch08_main_ref.m` | 28 | line 5 `cd('E:\NTNU\CRC\latex\matlab\ch8\MCD')` deleted, line 7's `load` re-pointed at `data/book/ch08/MCD/` (risk R3); the two calls retargeted at `ch08_pcbf_ref` with the extended output list and a terminating semicolon; the ΔN bar figure of lines 44–66 removed, keeping its `caxis` limits as `caxis_lim_err`. `count_error` (line 41) verbatim. |
| `fitting_iceFloes_distribution.m` | `ch08_fit_ref.m` | 3 | line 5 `cd('C:\Users\qinz\Desktop\sent_to_Qin')` deleted, line 7's `load` re-pointed, **line 57 (the `legend` with the stray quote) deleted — it is a SYNTAX ERROR** (see erratum E1). |
| `color_hist.m` | `ch08_colorhist_ref.m` | 13 | graphics only, incl. the **HG1** block `ch = get(h,'Children'); fvd = get(ch,'Faces'); fvcd = get(ch,'FaceVertexCData'); … fvcd(fvd(i,:)) = color(i); … set(ch,'FaceVertexCData',fvcd)` (lines 23–25/31/33), which *errors* in R2025a. |
| `color_hist_comparison.m` | `ch08_colorhistcmp_ref.m` | 13 | the same HG1 block at lines 32–34/40/42, plus `figure`/`bar`/`colormap`/`colorbar`/`set`. |
| `sea_ice_model.m` | `ch08_model_ref.m` | 19 | function renamed, outputs extended with `bw_floe`/`bw_brash`/`ver`/`t`; the two display blocks (lines 94–101, 159–168: `figure`, `imshow`, `line`, `plot`, `hold`) removed. Every numeric line — `convhull`, `roipoly`, `polygeom`, `polyxpoly`, the `if xx ~= NaN` tests, `t = 0:0.05:6.28`, `d < r+r0` — is verbatim. |

**Every quantity the removed lines consumed is still computed and saved.**  For `color_hist*.m` that is
`z, n, color, zs, izs, ysh, YT, color_floe, color_min, color_max, nbins, k, d` (and `z0, n0, z_d`); for
`plot_color_bar_and_floe.m` it is `counts, histogram_centers, index_all, color_M, centre, rgbImage, polyx,
polyy, polyc, dotxy, caxis_lim`.  No numeric literal or expression was altered anywhere.

---

## Parity table

One row per port-plan row (`analysis/ch08.md` §5) plus one per new core primitive.
Evidence levels: **L1** synthetic truth · **L2** MATLAB R2025a parity · **L3** the authors' own shipped output ·
**L4** numbers printed in the book.

| MATLAB file / function | Python | Evidence | Result | Parity | Notes |
|---|---|---|---|---|---|
| `MCD/main_WL_new.m` (67 l) | `ch08_applications.mcd_analysis`, `scripts/ch08_main_WL_new.py` | L2 `mcd.mat`, L3 `MCD_results.mat`, L4 | `Poly_Area`, `Poly_MCD`, `Raw_Area`, `Raw_MCD`, `Poly_counts/centers`, `Raw_counts/centers`, `count_error`, `color_M`, `N`, `Y_limi`, all three `caxis` limits: **max abs diff = 0.0** on all 2888 values / 100 bins | **exact** | `Raw_MCD` also matches the shipped `MCD_results.mat` at **0.0** |
| `MCD/plot_color_bar_and_floe.m` (116 l) | `ch08_applications.plot_color_bar_and_floe`, `core.plotting.mcd_colorbar` | L2 `mcd.mat` + `mcd_rgb.mat`, L4 Figs. 8.19/8.20 | `counts`/`centers`/`index_all` **0.0**; the grown `rgbImage` **627×1114×3 with 0 differing bytes**, its size exactly MATLAB's; `Vertices`-branch `fill` coordinates (20 floes) **0.0**; `polyc` colours **0.0**; white-dot `dotxy` **0.0** in both branches | **exact** (arrays); **display** (the figures) | R17: MATLAB never pre-allocates `rgbImage`. On the full field **both pixel maxima are attained**, so that run alone cannot tell "grown" from "pre-allocated" (review S4); the new `grow.mat` fixtures do: `Floe(1:50)` -> **627x1096x3** and the 50 floes inside `x<=900, y<=500` -> **489x865x3**, both reproduced exactly and byte-identically, and the `image_shape` override branch is now executed |
| `MCD/fitting_iceFloes_distribution.m` (63 l) | `ch08_applications.cumulative_fsd_powerlaw`, `scripts/ch08_fitting_ice_floes_distribution.py` | L2 `fit.mat`, L4 α | `MCD`/`N_L` (Eq. 8.2) **0.0**; `eta` = `[11.85785671, 1.37035854]` vs MATLAB `[11.85785670, 1.37035854]` → **rel. diff 6.6e-10 / 2.4e-10 (10 s.f.)**; `resnorm` 2.2425018400 vs 2.2425018400 (rel. 5.9e-14); `exitflag` **3 = 3** | **near** | the file as shipped is unrunnable (erratum E1) |
| `MCD/PowerLaw_fitting_method_and_plotting.m` (34 l, orphan) | `ch08_applications.power_law_fit` | L2 `orphan.mat` | `eta` agrees to ≤ 4.1e-5 relative on 3 fixtures (best 3.3e-10); `resnorm` to ≤ 1e-6 relative; `exitflag` identical (1/3/3) | **near** | no caller exists anywhere in `MATLAB_ROOT` |
| `MCD/three_fitting_method_and_plotting.m` (65 l, commented call site) | `ch08_applications.three_distribution_fits`, `scripts/ch08_three_fitting.py` | L2 `orphan.mat` | plain power law ≤ 4.1e-5 rel.; Weibull `resnorm` ≤ 1e-6 rel. of MATLAB's; **upper-truncated power law: `ε₃` differs by up to 3 orders of magnitude** — see deviation D2 | **near** | its only call site (`fitting_iceFloes_distribution.m:37`) is commented out |
| `SIFI/sea_ice_model.m` (170 l) | `ch08_applications.sea_ice_model`, `scripts/ch08_sea_ice_model.py` | L1 (3 constructed fixtures), L2 `model.mat` + `window.mat`, L3 the whole shipped field | **227 real floes / 240 real brash (L2)**: `Area` ≤ 1.14e-13, `Center` ≤ 1.42e-14, `Perimeter` ≤ 1.42e-14, vertex **sets** 227/227, all four `Intersect` lists **227/227 and 240/240 equal as sets**, `bw_floe`/`bw_brash` **0 px**, `Radius`/`Perimeter` **0.0**. **L3 whole field (2888 + 3452)**: `Area` ≤ 1.8e-12, `Center` ≤ 1.1e-13, `Perimeter` ≤ 5.7e-14, hull vertex sets **2888/2888**, `Intersect` **1106 / 1171 / 1171 / 544 entries, every list equal as a set**, `Radius`/`Perimeter` **0.0** | `Area`/`Center`/`Perimeter`/`Radius`/`Intersect` **exact**; `Vertices` **near** | vertex **order** is identical for only 105 of 227 (MATLAB `convhull`'s start vertex and traversal direction are not reproducible — ch06 lesson); the contract is on the vertex set and the rasterised mask |
| `SIFI/SeaIce_Image_Structure.m` (115 l) | `core.icestruct.*`, `ch08_applications.sea_ice_image_structure`, `scripts/ch08_sea_ice_image_structure.py` | L1, L2 `model.mat`, L3 the shipped `.mat` | all 12 numeric `Field` members and `Param.NumPix_x/y`, `TiltAngle`, `PanAngle`, `Location`, `Creator` **identical**; `FSD` **51/51 and 65/65 triplets identical**; `Polygon.Vertices` open rings **identical incl. order**; `Circle.Radius`/`Perimeter` identical | **exact** | shipped `Field.FSD` also reproduced **51/51** on the real §8.3 field |
| `SIFI/color_hist.m` (46 l) | `ch08_applications.color_hist`, `scripts/ch08_color_hist.py` | L2 `colorhist.mat` (4 inputs incl. the real 2888 areas), L4 | `z`, `centers`, `n = centers+35`, `color`, `ysh`, the nine `YT` integers, `color_min = 198`, `color_max = 9698`, `nbins = 50`: **identical on every input** | **exact** | HG1 block is display only (patched) |
| `SIFI/color_hist_comparison.m` (55 l) | `ch08_applications.color_hist_comparison`, `scripts/ch08_color_hist_comparison.py` | L2 `colorhist.mat`, L4 | `z`, `z0`, `z_d`, `n`, `n0`, `ysh`, `YT`, `color_max`: **identical on every input**; both tick lists (`max_x` 3500 and 6000) reproduced | **exact** | risk R9 adjudicated below (erratum E3) |
| **new** `core.stats.mean_caliper_diameter` (Eq. 8.1) | — | L1, L3 | inverts `A = πr²` to 1e-12; on `Floe.Area` reproduces `MCD_results.mat` at **0.0** (2888 values) | **exact** | |
| **new** `core.stats.cumulative_size_distribution` (Eq. 8.2) | — | L1, L2 | equals the M-file's literal O(N²) loop; `Nc(min)=1`, `Nc(max)=1/N`, ties repeat; **0.0** vs MATLAB's `N_L` (2888 values) | **exact** | |
| **new** `core.fitting.lsqcurvefit` | — | L1, L2 `misc.mat`/`fit.mat`/`orphan.mat` | MATLAB's own `optimoptions('lsqcurvefit')` defaults confirmed: `Algorithm = 'trust-region-reflective'`, `FunctionTolerance = StepTolerance = 1e-6`, `MaxIterations = 400`, `MaxFunctionEvaluations = '100*numberOfVariables'` — exactly the table in `core/fitting.py`. Parameters agree to 5–10 s.f. on the well-posed fits; `exitflag` agreed in 4 of 4 comparisons | **near** | SciPy `trf` is the same Coleman–Li family, not the same code |
| **new** `core.fitting.{power_law, truncated_power_law, weibull_survival}` | — | L1, L2 | the three anonymous-function expressions, evaluated by hand and through MATLAB's fits | **exact** | |
| **new** `core.icestruct` (Appendix B) | — | L1, L2, L3 | field-by-field against MATLAB (above) plus a `save_iceimage_mat`/`load_iceimage_mat` round-trip; `overlap_graph` = **1106 / 1171 / 1171 / 544**, all three symmetry checks `True` | **exact** | |
| **new** `core.plotting.matlab_jet` | — | L2 `misc.mat` | `jet(1)`, `jet(3)`, `jet(4)`, `jet(7)`, `jet(30)`, `jet(255)`: **max abs diff = 0.0** | **exact** | the porter's recorded `mod(m,4)` vs `mod(m,2)` deviation is a *historical* note only — R2025a uses `mod(m,4)==1` and `jet(255)` matches; a test computes the old rule and asserts it differs (D3) |
| **new** `core.plotting.mcd_colorbar` | — | L2 (its `caxis` limits) / L4 | `caxis([centers(1) centers(30)]) = [1, 30]` reproduced in all three call sites; the book's printed ticks 5,10,…,≥30 follow | **display** | drawing only; the numbers behind it are exact |
| **new** `core.matlab_compat.matlab_colon` (re-exported as `ch08_applications.matlab_colon`) | — | L1, L2 `misc.mat`, `fit.mat` | `0:0.05:6.28` → **126** elements, last **6.25** (MATLAB agrees exactly on both); integer colons `20:70:3500`, `21:79:3971` **and the real FSD line `21:79:3979`** (51 elements, last **3971** — the stated endpoint is never reached) **bit-exact**; `x_plot = min(MCD):0.001:max(MCD)*1.5` **119 822** elements, ≤ 1 ulp | **near** | ≤ **1 ulp** on 30 of the 126 non-integer angles (D1). Moved to `core/matlab_compat.py` (review S5); `test_matlab_colon_lives_in_core_and_is_re_exported` asserts the two names are the same object |
| §8.1.2 Otsu / k-means ice concentration (text only) | `ch08_applications.shipborne_ice_concentration`, `scripts/ch08_ice_concentration_series.py` | L1 only | **methods**: ch03 `graythresh`/`im2bw` (`exact`); k-means now has two branches (review S6) — `impl='authors'` (**the default**) is ch03's port of the authors' own `MATLAB_ROOT/ch3/kmeans.m`, deterministic, parity **exact**, and `impl='lloyd'` is the seeded k-means++/Lloyd routine, parity **approx**. Both pinned on the new ramp fixture: k=2 centres `[45.746939, 186.758134]` in *both* branches → IC **41.3021 %** = Otsu's, masks agree **100.00 %**; k=3/top-2 `authors` `[35.364295, 122.842687, 209.718880]` → **51.1042 %** vs `lloyd` (seed 0) `[34.834807, 120.788106, 208.897883]` → **51.6979 %** | **unverified** as a §8.1 *result* (the *method* is `exact` with the default `impl`) | see Open item O1 |
| §8.2 pipeline driver (no `.m` in `ch8/`) | `ch08_applications.sea_ice_field` | L1 (smoke), L4 | **does not reproduce the printed 498 floes / 201 brash** — see Open item O2. The function itself is now **executed by a test** (review S7): `test_sea_ice_field_runs_the_ch07_pipeline` runs it on a 160x240 crop of `sea_ice_test.jpg` (40 floes / 36 brash, coverages summing to 1.0, `index_floe` carrying exactly the 21 174 floe pixels) and exercises **both** cache branches (write, then reuse → identical arrays) | **unverified** | its two constituents are ch07's `sea_ice_edge_detection` / `ice_shape_enhancement`, verified there |

**Parity counts — 19 rows: exact 10 · near 5 · approx 0 · reimplemented 0 · unverified 2 · display 1 ·
deferred 0, plus 1 mixed row.**  Itemised so the count can be checked against the table above:
9 rows are pure `exact` (`main_WL_new`, `SeaIce_Image_Structure`, `color_hist`, `color_hist_comparison`,
`mean_caliper_diameter`, `cumulative_size_distribution`, `core.fitting.{power_law, truncated_power_law,
weibull_survival}`, `core.icestruct`, `matlab_jet`) and `plot_color_bar_and_floe` is `exact` for its arrays and
`display` for its figures — 10 exact; 5 rows are pure `near` (`fitting_iceFloes_distribution`,
`PowerLaw_fitting_method_and_plotting`, `three_fitting_method_and_plotting`, `core.fitting.lsqcurvefit`,
`matlab_colon`); 1 is pure `display` (`mcd_colorbar`); 2 are `unverified` (§8.1's results, §8.2's counts); and
`sea_ice_model` is **mixed** — `exact` for `Area`/`Center`/`Perimeter`/`Radius`/`Intersect`, `near` for
`Vertices` (hull vertex *order*).  9 + 1 + 5 + 1 + 2 + 1 = 19.
(Review item S3: the previous wording said "17 rows · near 4", which did not match this table.)
The review revision moved **no label**: the §8.1 row's parity *label* is still `unverified` (it is a statement
about §8.1's printed *results*), even though its Result cell now says that the **method** is `exact` under the
new default `impl='authors'` and `approx` under `impl='lloyd'` (review item S6).  Counts therefore stay at 19.
The `deferred` item of this chapter is the Appendix-A.1.2 linear rectification, which belongs to ch10 and is
not a ch08 port-plan row.

---

## Figures reproduced

Comparison PNGs in `reports/ch08_verification/figures/` (git-ignored, CLAUDE.md rule 12); metrics in
`figure_metrics.json`; generator `reference/ch08/make_compare_figures.py`.

| figure | comparison | verdict |
|---|---|---|
| **Fig. 8.19** (p. 191) | `fig_8_19_compare.png` — our painted `rgbImage` (627×1114, byte-identical to MATLAB's) vs the embedded CMYK bitmap of `chapters/ch08.pdf` page 16 xref 429, best of ~6 000 crops (389×219 at (32,16)) | Unmistakably the same ice field: the same large dark-red floes, the same black voids in the upper-middle and left, the same cyan/yellow brash texture; **RGB NCC 0.8187 raw, 0.9312 after a 1.2 px Gaussian** (the printed plate is screened, JPEG-compressed and ~3× downsampled). The analysis's pre-check of 0.9045 sits inside the smoothed band; I could not reproduce it as a *raw* NCC and record the lower raw figure instead. |
| **Fig. 8.20** (p. 192) | `fig_8_20_compare.png` — our `hist(Raw_MCD, 1:100)` bar chart vs the rendered page | Same shape, same colours, same axes: a jet-coloured spike clipped by the 450 y-limit at 7 m, a long red tail to ~58 m and two isolated bars near 76 and 84 m; our peak is **454 at 7 m**, sum 2888. |
| **Fig. 8.21** (p. 193) | `fig_8_21_compare.png` — our log-log cumulative FSD + fit vs the rendered page | Same curve: red circles from (6.1, 1) bending down to ~4e-4 at 84 m, with the straight black power-law line crossing the data near 20 m and lying above it thereafter — exactly the printed "good fit for the smaller floe sizes". |
| Fig. 8.11 (p. 186) | `fig_8_11_compare.png` | Same *form* (50 bins over 0–3500 px, jet colouring, the 20…3487 colour bar) but a visibly different bar height profile — our run identifies 433 floes, the book's 498 (see O2). |
| Fig. 8.12 (p. 187) | `fig_8_12_compare.png` | Same two-panel structure — a dense packing of white convex-hull polygons over black, and a sparse field of small bright disks — at the same aspect; the printed (a) panel is denser, consistent with the larger floe count. |
| Fig. 8.13 (p. 187) | `fig_8_13_compare.png` | Our close-up shows exactly the printed content: closed polygon boundaries drawn over the floes with the identified centres (`*`) and the modelled centres (`+`) nearly coincident; the crop region is our choice (rows 409–609, cols 88–288) because the book does not state it. |
| Fig. 8.14 (p. 188) | `fig_8_14_compare.png` | Same shape as the printed polygonized histogram (a taller first bin, a shifted body, a thin tail); heights differ for the same reason as Fig. 8.11. |
| Fig. 8.15 (p. 188) | `fig_8_15_compare.png` | Same character: a signed difference oscillating within roughly ±10 counts about zero, largest at the small-size end — the printed panel is within ±5. **Sign checked, not just shape:** our first bin is `z_d[0] = +10` (positive) and the book panel's leftmost bar also rises above the zero line, so the plate plots `z − z0` = polygonized − identified, the direction the code computes and the *opposite* of the p. 188 sentence (erratum E11). A global sign flip would now be caught. Produced with `--max-x 3500` to match the printed colour bar (erratum E3). |
| Figs. 8.1–8.7, 8.16–8.18 | — | **Not reproducible**: third-party photographs, a schematic, an external DEM simulator, and the raw §8.3 frame, none of which ship (O1, O3). |

---

## Numbers from the text (book vs ours)

| # | book | ours | input | status |
|---|---|---|---|---|
| N14 | 2888 ice floes, 3452 brash pieces | 2888 / 3452 | shipped `IceImage_290915_2_jpg.0000179.mat` | **reproduced** |
| N15 | 58.00 / 4.85 / 21.21 / **15.94** % | 57.99753 / 4.84911 / 21.20740 / **15.94596** % | same `.mat` (`Field.Cov*`; `CovFloes` = 405 100/698 478 and `CovBrash` = 33 870/698 478 exactly) | **reproduced with caveat** — the water figure is **truncated**, not rounded: 15.945957 → the book prints 15.94 where rounding gives 15.95. The other three are unaffected by the choice. Erratum E2 (minor). |
| N16 | colour bar 5, 10, 15, 20, 25, ≥30 [m] | `caxis = [1, 30]` in all three call sites (MATLAB's own value) → MATLAB's default 5:5:30 | `mcd.mat` | **reproduced** (explained by the code) |
| N17 | Fig. 8.20 axes MCD 0–100 m, N 0–450 | peak **454 at 7 m** (clipped by the 450 limit), then 311@6, 246@8, 233@9, 198@10, 161@12; Σ = 2888; nothing below 6 m | `MCD_results.mat` | **reproduced** |
| N18 | α = **1.3704** | **1.3703585384** (MATLAB `lsqcurvefit`: 1.3703585381) → rounds to 1.3704 | `MCD_results.mat` | **reproduced** |
| N19 | Fig. 8.21 axes 10^0.8–10^2.1 m, 10^-3.5–10^0 | MCD ∈ [6.0985, 83.9466], N_c ∈ [1/2888 = 3.4626e-4, 1] | `MCD_results.mat` | **reproduced** |
| N10 | Figs. 8.11/8.14/8.15 ticks 20, 149, 297, 471, 682, 950, 1317, 1902, 3487 | identical, from MATLAB and from the port | `colorhist.mat`, any input | **reproduced — but NOT evidence.** The list is a function of the three literals `min_x = 20`, `max_x = 3500`, `nn = 8` only. Three deliberately different area vectors (a single 1-px piece; 1…9000 step 37; 500 copies of 3499) give the identical nine integers (test `test_color_hist_tick_list_is_constants_only`). Never quote it as parity (risk R10). |
| N9 | Fig. 8.10 ticks 1, 177, 391, 664, 1040, 1650, 3483 | **not reproduced**: our §8.2 run gives 2, 173, 379, 640, 993, 1544, **2870** | our §8.2 pipeline on `sea_ice_test.jpg` (433 floes + 290 brash) | **not reproducible with the shipped parameters.** This list *is* data-dependent: enumerating every `(C_min, C_max)` that yields the printed list gives `C_min = 9` uniquely and `C_max ∈ {9693…9698}`, i.e. the book's smallest piece is **exactly 1 px** and its largest is in **[3484, 3503] px**; ours are 2 px and 2871 px. Independent confirmation of O2. |
| N7 | 76.73 / 0.46 / 9.05 / 13.76 % | 76.23 / 0.99 / 9.21 / 13.57 % | our §8.2 pipeline on ch07's `sea_ice_test.jpg`, ch07's shipped `sea_ice_demo.m` parameters, `kmeans_lloyd(seed=0)` | **not reproducible** — the authors' §8.2 parameters are not printed (O2) |
| N8 | 498 ice floes, 201 brash pieces | **433 floes / 290 brash** | same input as N7. (ch07's MATLAB **reference** stage on the same image gave 433 / 274 — a different number from a different input stage, recorded here so the two are never mixed.) | **not reproducible** (O2) |
| §8.1 p. 179 | "k = 2 k-means ≈ Otsu" | **on a fixture built so that the agreement is not forced** (review S4's lesson applied to §8.1): water 30 / a grey ramp covering *every* level 55…205 / dry ice 225, ±5 noise. Otsu's threshold is **116**, the k = 2 centre midpoint is **116.2525** — *different numbers* — and the masks nevertheless agree on **100.00 %** of pixels (IC **41.3021 %** both). The fixture discriminates: 11 pixels sit at level 116 and 30 at 117, so a one-level move of either threshold would break the agreement. k = 3 with ice = the two brightest clusters is **51.1042 %**, higher by 9.8 pp — the mechanism the book gives for Fig. 8.5 | synthetic fixture (`_ramp_frame`) | **reproduced as a relation**, not as the book's numbers (no data). The earlier two-tone fixture had a 150-level empty gap in its histogram, which makes *any* agreement inevitable; it is kept only for the trivial 50 %-Otsu check |
| N1–N6, N11–N13, N20, N21 | camera/platform specifications and code literals | transcribed; `length_over_Pixel = 1.1794`, `color_limit_N = 30`, `x_bin = 1:100` used verbatim | — | context |

---

## Deviations & justifications

**D1 — `matlab_colon` is 1 ulp off MATLAB's colon on non-integer steps.**  `0:0.05:6.28` agrees with MATLAB in
**length (126) and last value (6.25) exactly**, but 30 of the 126 angles differ by exactly one ulp (max abs
diff 8.9e-16; MATLAB's colon is more accurate than `start + k·step`).  Bounded: substituting MATLAB's own `t`
vector into `sea_ice_model` changes the brash raster by **0 px** and the circle coordinates by < 1e-14, and
the integer colon vectors this chapter uses (`20:70:3500`, `21:79:3971` and the FSD line's actual
`21:79:3979`) are **bit-exact**.  Review item N4: the fixture originally pinned `21:79:3971`, whose `(b-a)/d`
is exactly 50, so it exercised none of the truncation logic; the line `SeaIce_Image_Structure.m:98` really
evaluates for the shipped field is `21:79:3979` with `(b-a)/d = 50.1`, and MATLAB's own vector confirms **51**
elements ending at **3971** — the stated endpoint is never reached.  Both are now asserted, as is the
119 822-element `x_plot` of `fitting_iceFloes_distribution.m:45` (≤ 1 ulp).  Label `near`.

**D2 — the upper-truncated power law lands in a different point of the same flat valley.**
`three_fitting_method_and_plotting.m`'s first fit has an *unidentifiable* third parameter: as `ε₃ → ∞` the
model degenerates to the plain power law, so the objective is flat along `ε₃`.  Measured, on the three
fixtures (Python `ε₃` vs MATLAB `ε₃`, then the residual sum each parameter vector produces):

| fixture | MATLAB `ε₁, ε₂, ε₃` | Python `ε₁, ε₂, ε₃` | resnorm (MATLAB) | resnorm (Python) |
|---|---|---|---|---|
| `exact` | 3.00000006, 1.49999949, **766.1** | 3.00000000, 1.49999999, **9.94e5** | 9.998e-07 | **2.573e-16** |
| `noisy` | 3.152781, 1.546689, **7565.5** | 3.149153, 1.539385, **6.95e4** | 1.13712e-03 | **1.04142e-03** |
| `survival` | 25.5927, 0.0126714, 35.9517 | 21.1707, 0.0154009, 36.0378 | **0.1140375** | 0.1140534 |

Python's solution is **better** in two of three cases and worse by 1.4e-4 relative in the third; MATLAB stops
earlier because the file sets `optimset('TolFun',1e-5)`.  Both minima are reported here rather than tuned away
(risk R1).  A dedicated test (`test_truncated_power_law_third_parameter_is_unidentifiable`) shows that fixing
`ε₃` anywhere in [766, 1e6] and refitting `ε₁, ε₂` changes the residual sum by < 1e-6, i.e. the disagreement is
a property of the model, not of the port.  Label `near`; the other two of the three fits are ≤ 1e-4 relative.

**D3 — `matlab_jet`'s recorded `mod(m,4)` deviation is historical only.**  R2025a's `jet.m` uses
`ceil(n/2) - (mod(m,4)==1)`, which is what the port implements, and `jet(255)` matches MATLAB **bit for bit**.
The pre-R2014 rule `mod(m,2)==1` gives a *different* table for `m ≡ 3 (mod 4)`; a test computes that variant
explicitly and asserts it differs at `m = 255` and agrees at `m = 30`, so the parity test at 255 really tests
the rule.  No open item.

**D4 — `sea_ice_model`'s `brash(i).Center` field holds a different object from MATLAB's when `Center` is
`k×2`.**  MATLAB's line 143 stores `c = cat(1, brash_ice(i).Center)` — the **whole matrix** — while the port
stores the pair `(c(1), c(2))` it is read as everywhere downstream.  Measured on the `center2x2` fixture:
MATLAB's `cat(1, brash.Center)` is **4×2 for 3 pieces** — `[[100, 300], [200, 400], [100, 200], [100, 300]]` —
while ours is 3×2 (`[[100, 200], [100, 200], [100, 300]]`).  Review item N9: this is now **asserted**
(`test_brash_center_field_is_the_documented_deviation_D4` on the reference variable `center2x2_bC`) instead of
resting on a human loading the `.mat`.  Functionally inert: every *use* of that
field (`sea_ice_model.m` lines 150–153, and `SeaIce_Image_Structure.m` line 34, which copies
`brash_ice(i).Center` rather than `brash(i).Center`) reads `c(1)`, `c(2)` again, and both the circle raster and
the Appendix-B `Brash.Center` are identical to MATLAB's.  Recorded, not filed as a defect.

**D5 — the AABB prefilter, the cropped `roipoly` and the fast crossing test are speed-ups, proven inert.**
Two separate measurements, on two different inputs — do not merge them (review item N8).
**(a) Flag-by-flag equivalence, on a 60-floe + 60-brash subset:** `prefilter=True/False`,
`raster_crop=True/False` and `intersect_impl='fast'/'polyxpoly'` give **identical** `Intersect` lists and
**0 px** raster differences in every combination.
**(b) The prefilter demonstrably skipped work, on the full runs:** 900 of 160 262 candidate pairs
(= 227·226 + 2·227·240) reached the crossing test on the 227/240 MATLAB-parity window, and 13 466 of
28 276 408 on the whole 2 888/3 452 field.  The defaults used for (b) are the configuration
`test_sea_ice_model_parity_on_a_real_window` compares against MATLAB's literal loop, so (b) is the stronger
evidence: the speed-ups are inert on exactly the run that is pinned at 0 px against MATLAB.
The equivalence is also an argument, not just a measurement: a common point of two polylines lies inside both
AABBs, so disjoint AABBs cannot cross.  (An independent 4 000-case randomised check of `_segments_cross`
against `core.polygon.polyxpoly` on integer-lattice rings, including collinear and single-point-touch cases,
agreed 4 000/4 000 — `reports/ch08_review.md`.)

**D6 — `colorbar_area_ticks` returns one tick where MATLAB returns none when `d = 0`.**  Inherited from ch07
(review S6(a)); it cannot arise for `color_hist*.m`, whose `d = fix((9698−198)/8) = 1187` is a constant.

**D7 — the port's `exitflag` is a documented translation of SciPy's `status`, not a MATLAB run.**  It was
nevertheless checked against MATLAB in 4 of 4 comparisons (1, 3, 3, 3) and agreed each time.

---

## Book / shipped-code errata confirmed

**E1 (new, blocking) — `fitting_iceFloes_distribution.m` does not run.**  Line 57
`legend(gca,{'Observed data','Power law fitting''},'Box','on')` is not a rendering quirk but a **syntax
error**: the doubled quote escapes into the string and the parser fails.  MATLAB R2025a's own message, captured
in `reference/ch08/fit.mat` as `legend_err` from a one-line copy of that line:

> `File: …\ch08_legend_probe.m Line: 1 Column: 52` · *Invalid expression. Check for missing multiplication
> operator, missing or unbalanced delimiters, or other syntax error. To construct matrices, use brackets
> instead of parentheses.*

The script therefore cannot have produced Fig. 8.21 in the form it ships; the line is deleted in the reference
copy (recorded in `patches.json`).  Everything before line 57 — including the whole fit — runs unchanged.

**E2 (text) — the book's estimator description does not match its own code.**  p. 192: "The exponent α … is the
slope of the power law curve on log-log plot (which is a straight line), and it is estimated to be 1.3704".
The code runs `lsqcurvefit` on the **untransformed** `(L, N_c)` pairs.  MATLAB's own `polyfit(log10(MCD),
log10(N_L), 1)` on the same data gives a slope of **−1.86968** (α = 1.8697), not 1.3704; the `lsqcurvefit`
route gives 1.37036.  Pinned in both directions by `test_book_text_describes_an_estimator_its_own_code_does_not_use`.
Also: the chapter computes **no goodness-of-fit statistic** for any of its three candidate distributions —
`resnorm`, `residual`, `exitflag`, `output` and `jacobian` are all requested from `lsqcurvefit` and discarded.

**E3 (figure/code mismatch) — Fig. 8.15's colour bar is `color_hist.m`'s, not the shipped
`color_hist_comparison.m`'s.**  MATLAB confirms both lists: `max_x = 3500` gives **20, 149, 297, 471, 682, 950,
1317, 1902, 3487** (the printed list) and the shipped `max_x = 6000` gives **20, 153, 307, 488, 710, 996, 1398,
2081, 5952**.  The port defaults to the shipped 6000 and exposes `--max-x 3500` to reproduce the figure.

**E4 (minor, arithmetic) — the water coverage is truncated.**  `Field.CovWater = 0.1594595678` = 15.945957 %;
the book prints **15.94 %** where correct rounding gives 15.95 %.  The other three percentages are the same
either way.

**E5 (terminology) — "mean clipper diameter".**  p. 190's acronym is a typo for *mean caliper diameter*
(Rothrock & Thorndike 1984), and the quantity Eq. (8.1) actually defines, `L = sqrt(4A/π)`, is the
**area-equivalent circle diameter**, which is not a caliper (support-function) diameter at all.  The acronym is
kept because the code, the folder and the figures all use it; the docstring records both discrepancies.

**E6 (claim) — "the polygonized floes will not be smaller than the actual identified floes" (p. 184) is false
in both readings.**  The stored continuous `Polygon.Area` is **smaller** than the pixel count for **2383 of the
2888** shipped floes (equal for 15, larger for 490), because `polygeom` measures the hull through pixel
*centres*.  The rasterised hull does not contain the whole floe either: on the 227-floe window, **2079 of
19 194** floe pixels (10.8 %) fall outside `roipoly`'s mask, every one of them within **0.083 px** of a hull
edge — exactly the band `poly2mask`'s 1/5-pixel snapping can round the wrong way.  This is the shipped code's
behaviour, not a port defect: MATLAB's own `bw_floe` on that fixture is identical to ours to **0 px**.

**E7 (data) — the shipped `.mat` contradicts the shipped `SeaIce_Image_Structure.m`.**  The script hard-codes
`Location = 'Ny-Alesund'`, `Creator = 'UAV'`, `LengthSI_x = 50`, `LengthSI_y = 18` (the §8.2 UAV field); the
`.mat` holds `Creator = 'Helicopter'`, `PrjName = 'OATRC 2015'`, `Location = []` and empty lengths (the §8.3
helicopter field).  The `.mat` was produced by a §8.3 variant of the §8.2 script.  The port makes them
arguments with the script's values as defaults and compares them against the *script*.

**E8 (risk R11, adjudicated) — the shipped `Field.FSD` interval labels do not describe its counts, and the
counts are the correct ones.**  `SeaIce_Image_Structure.m` line 98 passes `min_x : inter : max_x` to `hist`,
which reads them as bin **centres**; the true edges are their midpoints and both outer bins are unbounded.
So the first triplet is labelled `[21, 99]` but actually counts everything ≤ 60.5.  Reproducing the shipped
counts through `core.histogram.hist` gives **51/51 triplets identical**; re-counting the areas from the printed
`[int_min, int_max]` labels gives **1861 instead of 1411** in the first interval and disagrees in **24 of the
51** intervals.  **Verdict: the shipped labels are wrong (misleading), the shipped counts are right**, and the
port reproduces the counts.  Both directions are pinned in
`test_shipped_fsd_is_reproduced_and_its_labels_are_not_the_bins`.

**E9 (risk R6, adjudicated) — `if xx ~= NaN` is `if ~isempty(xx)`, and containment is therefore not detected.**
MATLAB confirms all three branches (`[1 2 3]` → taken, `[]` → not taken, `[1 NaN]` → taken) and that
`polyxpoly` on a square strictly inside another returns **empty**.  On the `nested` fixture MATLAB reports the
inner floe and the enclosed brash disc as **non**-overlapping, and the port reproduces that; the opt-in
`strict_containment=True` flag reports them, so the fixture discriminates in both directions.  This is a
genuine limitation of the published algorithm, faithfully reproduced.

**E10 (risk R18, adjudicated) — `t = 0:0.05:6.28` really is 126 points ending at 6.25 rad.**  MATLAB's own
`numel` and `t(end)` confirm it, so every brash "circle" in the book's results is an open 125-gon with a
0.033 rad gap.

**E11 (new, review item S2, minor) — p. 188 states the Fig. 8.15 subtraction in the wrong order.**  The text
reads *"By subtracting the histograms of Figure 8.12(a) from Figure 8.10, we get the floe size distribution
error"* (`chapters/ch08.txt` l. 364-365).  Figure 8.11 is the histogram *of* Fig. 8.10 (the identified field)
and Figure 8.14 is the histogram *of* Fig. 8.12(a) (the polygonized field), so the sentence says
**identified − polygonized**.  The shipped code computes the opposite: `color_hist_comparison.m` l. 29 is
`z_d = z - z0` with `z = hist(floe.Area)` (polygonized, Fig. 8.14) and `z0 = hist(ice_floe.Area)` (identified,
Fig. 8.11), i.e. **polygonized − identified**.  The *printed figure* follows the code, not the sentence: in
`reports/ch08_verification/figures/fig_8_15_compare.png` the book panel's leftmost bar rises **above** the
zero line, the same sign as our `z_d[0] = +10`.  Fig. 8.15's own caption ("Error between the floe size
distributions of Figure 8.11 and Figure 8.14") is order-neutral and correct.  **The code and the plate are the
authority; the body sentence is the erratum.**  The port follows the code
(`seaice/ch08_applications.py` `color_hist_comparison`, docstring: "the polygon histogram *minus* the pixel
histogram").

---

## Open items

**O1 — §8.1 has no code and no data (`unverified`).**  All six §8.1 figures (ship photos, checkerboard
calibration, rectified frames, the 6-hour Otsu-vs-k-means concentration series, Events #1/#2) are credited to
Lu, Zhang, Lubbad, Løset & Skjetne (OTC 2016) and neither the images nor a `.m` file ship.
`shipborne_ice_concentration` is therefore verified only as a *method* (L1 on synthetic frames, and its
constituents `graythresh`/`im2bw` are `exact` from ch03 and, after review item S6, its **default** k-means is
the authors' own deterministic `kmeans_gray` — `exact` from ch03 — with the seeded `kmeans_lloyd` (`approx`)
kept behind `impl='lloyd'`); every number the book prints for §8.1 stays `unverified`.  Both branches are
pinned by `test_shipborne_kmeans_impl_branches_are_both_pinned`, and the fixture they run on was rebuilt so
that the "k = 2 ≈ Otsu" agreement is not forced by an empty band in the histogram (see the Review revision).
The linear 4-corner rectification of Appendix A.1.2 that precedes it is **deferred to ch10** — do not write a
second rectifier here.
`scripts/ch08_ice_concentration_series.py` prints `SKIP` without a `--frames` folder.

**O2 — §8.2's printed counts do not reproduce (`unverified`), and the authors' parameters are not printed.**
The book states 498 ice floes / 201 brash pieces and 76.73 / 0.46 / 9.05 / 13.76 % for Fig. 8.8.  Running the
ch06→ch07 pipeline (Algorithms 3/4/5) on ch07's own `sea_ice_test.jpg` with the shipped `sea_ice_demo.m`
parameter set and `kmeans_lloyd(seed=0)` gives **433 floes / 290 brash** and **76.23 / 0.99 / 9.21 / 13.57 %**.
Fig. 8.10's colour-bar list corroborates the gap from the book's side: its pre-image pins the book's smallest
piece at **exactly 1 px** and its largest at **[3484, 3503] px**, against our 2 px and 2871 px.  Nothing in the
chapter prints the parameter set that would close this, and **no parameter search was performed** (the brief
and risk R7 both forbid it).  Consequences: Figs. 8.9–8.15 are reproduced *as a pipeline* and compared
qualitatively, never numerically, and N7/N8/N9 stay `unverified`.  Note the two numbers must not be mixed:
ch07's MATLAB **reference** stage on the same image gave 433 / 274, our Python stage gives 433 / 290; the
former is a parity number, the latter is our pipeline's own output.

**O3 — Fig. 8.18 (the raw §8.3 photograph) does not ship.**  Only the derived Appendix-B structure does, so the
§8.3 pipeline cannot be re-run from the image; everything downstream of it (Figs. 8.19–8.21) is fully verified
from the shipped `.mat`, and the Fig. 8.19 comparison uses the *reconstruction* from `Floe.Pixels`, labelled as
such.  Fig. 8.16 (the DEM simulator) is out of scope — external software, no data.

**O4 — the Fig. 8.19 NCC of 0.9045 quoted in `analysis/ch08.md` (pre-check L4-3) was not reproduced as a raw
correlation.**  My best-of-~6 000-crops search gives **0.8187** raw RGB NCC at full resolution; the value rises
to 0.9100 / 0.9312 / 0.9444 under a 0.8 / 1.2 / 1.8 px Gaussian, so 0.9045 is consistent with a mildly
smoothed comparison.  Since the pre-check's exact procedure is not recorded, the *raw* figure is the one this
report stands behind.  No action needed; the visual verdict is unaffected.

**O5 — `resnorm`/`exitflag` for `PowerLaw_fitting_method_and_plotting.m` and
`three_fitting_method_and_plotting.m` had to be captured by an inline repeat of the files' own `lsqcurvefit`
calls**, because those files return only `eta`.  The repeats are character-identical to the files' lines
(`reference/ch08/make_refs.py::ref_orphan`) and are *additional* probes, not patches; the `eta` values compared
against the port come from the **verbatim** files.

---

## Test suite

`tests/test_ch08.py` — **88 tests, 88 passed** in 84.95 s (`.venv/Scripts/python.exe -m pytest
tests/test_ch08.py -q -p no:cacheprovider`), 77 parity/L1/L3/L4 tests plus 11 headless script runs.
The nine tests added in the review revision are:

| test | item | what it pins |
|---|---|---|
| `test_sea_ice_image_structure_parity[touching]` (extended) | S1 | the (row, col) ↔ (x, y) mapping, on an **80 × 137** image |
| `test_rgb_image_is_grown_not_preallocated[g50]` / `[gin]` | S4 | `size(rgbImage)` = **627×1096×3** / **489×865×3**, 0 differing bytes |
| `test_image_shape_override_preallocates_without_moving_a_pixel` | S4 | the `image_shape` branch (previously never executed) |
| `test_sea_ice_field_runs_the_ch07_pipeline` | S7 | the §8.2 driver + both cache branches |
| `test_matlab_colon_lives_in_core_and_is_re_exported` | S5 | `ch8.matlab_colon is core.matlab_compat.matlab_colon` |
| `test_fsd_colon_endpoint_is_truncated_parity` | N4 | `21:79:3979` → 51 elements ending **3971** |
| `test_x_plot_colon_parity` | N9 | the 119 822-element `x_plot`, ≤ 1 ulp |
| `test_brash_center_field_is_the_documented_deviation_D4` | N9 | `center2x2_bC` is **4×2 for 3 pieces** (D4) |
| `test_shipborne_kmeans_impl_branches_are_both_pinned` | S6 | both k-means branches, and that they are distinguishable |

Test-design rules followed (ch07's recorded lessons):

* every "fixed defect" is pinned **in both directions** — `matlab_jet`'s old `mod(m,2)` rule is computed
  explicitly and asserted to differ; `np.arange` is asserted to drop the endpoint that `matlab_colon` keeps;
  the FSD label recount is asserted to give 1861 where the shipped count is 1411; `strict_containment=True` is
  asserted to see the containment the literal `if xx ~= NaN` misses; MATLAB's `round(6.5) = 7` is asserted to
  differ from `min`'s first-index tie-break;
* the `center2x2` fixture is designed so that the two candidate readings of a 2×2 `Center` reach **different**
  brash pieces, and the test asserts MATLAB's choice (`{2}`) and the absence of the other (`3 not in …`);
* the saturating-agreement trap is met head-on: `test_color_hist_tick_list_is_constants_only` shows three
  wildly different inputs give the identical nine tick integers, so that match is explicitly disqualified as
  parity evidence;
* every cited measurement names its input (the parity table's `sea_ice_model` row separates the 227/240 L2
  window from the 2888/3452 L3 field; O2 separates our pipeline's counts from ch07's reference-stage counts);
* the prefilter test asserts that the prefilter actually *skipped* work, so the fixture cannot pass vacuously;
* **(new, S1/S4)** the two fixtures that could not distinguish a bug from its fix were replaced rather than
  re-asserted: one `sea_ice_model` fixture is now **non-square** (80 × 137) and two new `plot_color_bar_and_floe`
  fixtures paint floes that reach neither the last row nor the last column.  Both were verified to discriminate
  — swapping `NumPix_x`/`NumPix_y` in `seaice/ch08_applications.py:968` is *accepted* by the two square fixtures
  and **rejected** by the non-square one (`PixScale_x` 0.625 vs MATLAB's 50/137 = 0.36496350364963503); the swap
  was applied locally, observed to fail, and reverted (the file's MD5 is unchanged, `4e2adf8ced39f2abc015b1eb3d2def67`);
* **(new, S6)** the §8.1 k-means fixture is a ramp rather than a two-tone frame, because a two-tone histogram
  makes "k = 2 ≈ Otsu" true for *every* threshold — ch07's saturating-agreement lesson applied to a claim the
  chapter makes in words.

Full suite `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider` → **`1937 passed, 2 skipped,
1 xfailed, 15 warnings in 1 622.45 s`** (`outputs/ch08/verify/pytest_full.txt`, re-run after the review
revision; 1849 + 88 = 1937) = the ch02–ch07 baseline (**1849 passed, 2 skipped, 1 xfailed** — the surviving
xfail is ch03's) **unchanged** plus the 88 ch08 tests.  (The verify phase measured 1928 = 1849 + 79.)
**No regression.**  15 warnings: the 14 pre-existing ch03/ch04 pytest deprecations ch06/ch07 reported, plus one
new `RuntimeWarning: invalid value encountered in power` raised inside
`core.fitting.weibull_survival` during `test_three_distribution_fits_parity_on_fixtures`: the trust-region
solver evaluates the model at a trial `p[1] < 0`, so `(x/p[1])**p[0]` takes a negative base to a non-integer
power and numpy returns `nan`.  The step is then rejected, and the `survival` fixture's `resnorm` still lands
within 1.4e-4 relative of MATLAB's (D2 table).  Cosmetic, but it marks the one place in the chapter where the
port and MATLAB could in principle diverge on a *rejected* trial point rather than on the answer.

---

## Review revision (`reports/ch08_review.md`, 2026-09-10, commit `46e6c21`)

The independent review found **0 must-fix**, 7 should-fix and 10 nits.  Every item and its disposition, so this
report is a complete record of what changed after the verify phase:

| item | what it said | disposition | by |
|---|---|---|---|
| **S1** | the Appendix-B (row,col)↔(x,y) mapping is untested — all four fixtures are square | **applied.** The `touching` fixture is now **80 rows × 137 columns**; `model.mat` was regenerated in MATLAB R2025a (`NumPix_x = 137`, `NumPix_y = 80`, `PixScale_x = 50/137`, `PixScale_y = 18/80`) and `test_sea_ice_image_structure_parity` asserts the mapping *and* that the swapped variant differs. **Verified by doing the swap:** with the swap, the two square fixtures still pass and `touching` **fails** (`PixScale_x` 0.625 vs MATLAB's 0.36496350364963503); the edit was reverted and `seaice/ch08_applications.py` is byte-identical to the porter's version (MD5 `4e2adf8ced39f2abc015b1eb3d2def67`) | verifier |
| **S2** | p. 188 states the Fig. 8.15 subtraction in the wrong order, and nothing records it | **applied** — erratum **E11**, `analysis/ch08.md:207` corrected, the Fig. 8.15 verdict now states the sign of the first bin | main session |
| **S3** | the parity counts did not match the table | **applied** — recounted: **19 rows**, exact 10 · near 5 · display 1 · unverified 2 · 1 mixed. Unchanged by this revision (no label moved) | main session |
| **S4** | R17's mitigation was never carried out; the "grown array" claim cannot be evidence where both maxima are attained | **applied.** New MATLAB session `grow.mat` runs `ch08_pcbf_ref` on two subsets of the *same* `IceImage.Floe` struct array: `Floe(1:50)` → `size(rgbImage)` = **627×1096×3** and the 50 floes inside `x ≤ 900, y ≤ 500` → **489×865×3**, i.e. strictly smaller than the 627×1114 image in one and in **both** axes respectively. Our arrays match those sizes exactly and the painted maps are byte-identical (0 differing bytes of 2 061 576 and 1 268 955). The `image_shape` override branch is exercised by a separate test. The parity-table sentence that called the rule "confirmed" is replaced | verifier |
| **S5** | `matlab_colon` is a general primitive living in a chapter module (rule 9) | **applied** — moved to `seaice/core/matlab_compat.py`, re-exported from `ch08_applications`; a test asserts the two names are the *same object* | porter (+ verifier test) |
| **S6** | §8.1 used ch06's seeded Lloyd where the book's own §3.2 k-means exists | **applied** — `shipborne_ice_concentration(..., impl='authors')` is now the default (`core.clustering.kmeans_gray`, the authors' `ch3/kmeans.m`, deterministic, parity **exact**); `impl='lloyd'` keeps the seeded path (**approx**). Both branches are pinned with numbers (below) | porter (+ verifier test) |
| **S7** | `sea_ice_field`, the §8.2 driver, is executed by no test | **applied** — `test_sea_ice_field_runs_the_ch07_pipeline` runs it on a 160×240 crop of `sea_ice_test.jpg` (~4 s): 40 floes / 36 brash, `index_floe` carrying exactly the 21 174 floe pixels, coverages summing to 1.0, and **both** cache branches (write then reuse → identical arrays). Its parity label stays **unverified** (O2 is about the counts) | verifier |
| **N1** | `matlab_jet`'s "Display only." contradicts its own next paragraph | **applied** — sentence dropped | porter |
| **N2** | over-tight `polygeom ≤ 1e-12` in a docstring | **applied** — now quotes 1.8e-12 / the test's 1e-11 | porter |
| **N3** | "minimum-area bounding polygon" should say **convex** | **applied** | porter |
| **N4** | the colon fixture pinned `21:79:3971`, not the line the port evaluates | **applied** — `misc.mat` re-run with `21:79:3979`; MATLAB returns **51** elements ending at **3971** (`(b−a)/d = 50.1`, so the truncation rule really is exercised) and our vector is bit-identical | verifier |
| **N5** | `Area` read C-order while `Center` is read column-major | **applied** — `ravel(order="F")[0]` in both places (inert on the shipped data) | porter |
| **N6** | the script's white-dot expression differed from the M-file's (~1e-13) | **applied** — `scripts/ch08_main_WL_new.py` now writes `Y_limi*lop − centre*lop` | porter |
| **N7** | stale pre-check `eta` in `analysis/ch08.md` | **applied** | main session |
| **N8** | D5 mixed two inputs in one sentence | **applied** — D5 is now (a) the 60+60 flag-by-flag run and (b) the full-window/full-field prefilter counts | main session |
| **N9** | unused reference variables; two carry real evidence | **applied for the two named**: `center2x2_bC` (deviation **D4** — MATLAB's `cat(1, brash.Center)` is **4×2 for 3 pieces**) and `x_plot` (**119 822** elements, ≤ 1 ulp) are now asserted. **The rest stay unasserted, deliberately:** `y_fit_powerlaw0`, `sorted_floe_size`, `a_sorted_floe_size`, `jac`, `out_iterations`/`funcCount`/`firstorderopt`, `legend_err` (quoted verbatim in erratum E1 but not machine-checked), `{tag}_bA` (except `center2x2_bA`), `{tag}_ver`, `ch_zs_`/`ch_izs_`/`ch_cf_`, `cc_color_`/`cc_cmin_`, `st_polyV`, `st_bCircle` — each is either a plotting by-product or a quantity already covered transitively by an asserted one; they are kept in the `.mat` files because re-running MATLAB is expensive | verifier |
| **N10** | `probe.mat` is not produced by `make_refs.py` | **applied** — explained in the Environment section | main session |

**Not applicable / deferred: none.**  No review item was declined.

**Two measurements the revision adds, quoted here so they can be audited:**

1. **The `impl` split (S6).**  On the ramp fixture `_ramp_frame()` (water 30 | a grey ramp covering every level
   55…205 | dry ice 225, ±5 noise, 80×120 = 9 600 px):

   | method | centres | ice concentration |
   |---|---|---|
   | Otsu (`graythresh` = 116/255) | — | **41.3021 %** |
   | k = 2, `impl='authors'` | `[45.746939, 186.758134]` (midpoint **116.2525**) | **41.3021 %**, mask = Otsu's on **100.00 %** of pixels |
   | k = 2, `impl='lloyd'` (seed 0) | identical to `authors` to 1e-9 | **41.3021 %** |
   | k = 3 / top-2, `impl='authors'` | `[35.364295, 122.842687, 209.718880]` | **51.1042 %** |
   | k = 3 / top-2, `impl='lloyd'` (seed 0) | `[34.834807, 120.788106, 208.897883]` | **51.6979 %** |

   The k = 3 row is what makes the test meaningful: the two branches give **different** centres, so a test in
   which `impl` silently routed both to the same routine would fail.  `'lloyd'` is also seed-dependent — seeds
   0, 1 and 7 give three different centre triplets (seed 1 happens to land on the authors' answer) — while
   `'authors'` has no RNG at all, which is the justification for the default change.
   **On the degeneracy question:** the *previous* fixture (a two-tone frame plus a mid-grey patch) made the
   k = 2 ≈ Otsu agreement **inevitable** — its histogram has a 150-level empty gap, so every threshold in that
   gap produces the same mask.  The ramp fixture removes that: Otsu picks **116**, k-means' centre midpoint is
   **116.2525**, they are *different thresholds*, and 11 / 30 pixels sit at levels 116 / 117, so a one-level
   move would destroy the 100 % agreement.  The agreement is therefore a result, not an artefact — but note it
   is still a statement about *this* synthetic frame; §8.1's own numbers remain **unverified** (O1).

2. **The grown `rgbImage` (S4).**  `full_size = [627 1114]`; `size(rgbImage)` = `[627 1096 3]` for
   `Floe(1:50)` and `[489 865 3]` for the inside-`x ≤ 900, y ≤ 500` subset, both reproduced exactly with
   0 differing bytes.  The old assertion (`[627 1114 3]` on the full field) is kept, with a comment saying what
   it can and cannot prove.

---

## Verdict: **PASS**

All nine `.m` files have a row and a MATLAB reference; all eight `scripts/ch08_*.py` exit 0; **88/88** chapter
tests pass (79 from the verify phase + 9 added by the review revision); the two `unverified` rows (§8.1's results, §8.2's printed counts) each have an explicit open item
(O1, O2) explaining exactly why no reference exists, and no tolerance was loosened anywhere.  The chapter's
strongest evidence is that the §8.3 chain — `main_WL_new.m` → `plot_color_bar_and_floe.m` → Eq. (8.1) →
Eq. (8.2)/(8.3) — is **bit-exact against MATLAB** down to the 627×1114×3 painted map, and that re-running
`sea_ice_model.m` on the authors' own 2888 floes and 3452 brash pieces reproduces their shipped Appendix-B
structure including all 3992 overlap-list entries.
