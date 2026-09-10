# Chapter 7 verification — Sea Ice Type Identification

Date 2026-09-10 · port commit `541ac23` (`ch07: port — ch07_ice_type.py (1171 l) + 8 scripts, all exit 0, 63 figures`) ·
**revised 2026-09-10 at commit `5186e9d`, after the independent review** (`reports/ch07_review.md`): `9927dfa` made
`core.histogram.hist` count `-Inf` (the original Open item 1) and `5186e9d` made `core.matlab_compat.imcomplement`
return the **input class** (review M1) — both are re-measured against MATLAB below and both original open items are
closed ·
verifier artefacts: `tests/test_ch07.py` (**275 tests**), `reference/ch07/make_refs.py` + `reference/ch07/fixtures.py`
(9 `(bk, seg)` fixtures, **21** `imfill` cases, **23** `hist` cases, **11** `imcomplement` classes → `*_inputs.mat`;
7 `.mat` reference groups + 18 per-fixture `iceenh_*.mat`, `refs_log.json`), `reference/ch07/make_python_runs.py` (`sensitivity_ticks.json`; the Algorithm-3 segmentation
`alg3_python.npz` stays under the git-ignored `outputs/` tree, CLAUDE.md rule 12),
`reference/ch07/make_compare_figures.py`, `outputs/ch07/verify/` (patched script copies in `scratch/`, MATLAB `imwrite`
images in `matlab/`, generated MATLAB code in `*_code.m`, `patches.json`, `figure_diffs.json`, `make_refs_demo.log`,
`make_python_runs.log`, `pytest_ch07.txt`, `pytest_full.txt`, `pytest_scripts/<case>/`), `reports/ch07/figures/`
(9 compare PNGs, git-ignored).

## Environment
python 3.11.5, numpy 2.4.6, scipy 1.17.1, scikit-image 0.26.0, opencv 5.0.0, matplotlib 3.11.1 |
reference engine: **MATLAB 25.1.0.2833191 (R2025a) Prerelease Update 2** via `tools/run_matlab_ref.py`
(`matlab -batch`, `set(0,'DefaultFigureVisible','off')`, `save(..., '-v7')`). **Every** reference run reports
`engine: matlab`, `status: ok` (`outputs/ch07/verify/refs_log.json`) — 8 in the first pass, 11 counting this
revision's three: `clf` 69 s, `fig767` 52 s, `imfill` 24 s,
`hist` 24 s, `misc` 26 s, `iceenh` 21 s, **`demo` 2 680.8 s (44.7 min)**, plus a 30 s `kmeans` probe.
The revision pass re-ran three of them against R2025a (same engine/version, `outputs/ch07/verify/make_refs_s1b.log`):
`imfill` **39.6 s** (the 21st case, `single_eps`, added), the new `imcomp` → `imcomplement.mat` **16.5 s** (11 classes
through `imcomplement`, each in a `try/catch` that would have stored MATLAB's refusal — none refused), and `hist`
**16.9 s** (the seven new cases plus MATLAB's own `eps([-1 -2 -0.5 1 2 0 -3])`).
Octave is absent (`progress.json → environment.octave = null`); **no fallback was used and none was needed** —
the Statistics & ML Toolbox (`kmeans`) is licensed and every ch7 toolbox call ran in MATLAB itself.

How the original code was run (CLAUDE.md rule 8: nothing under `MATLAB_ROOT` was touched). All `.m` files were
**copied** into `outputs/ch07/verify/scratch/` and MATLAB ran with that folder as cwd; the ch7 folders were never put
on the MATLAB path. The patches are recorded verbatim in `outputs/ch07/verify/patches.json` and reproduced here:

| copy | changes |
|---|---|
| `ch07_cleaning_ref.m`, `ch07_labeling_sq3_ref.m`, `ch07_filling_ref.m`, `ch07_fillrec_dm1_ref.m` | the leading `clear all, clc;` removed (so several scripts can run in one session without wiping the accumulated variables). **Nothing else.** |
| `ch07_labeling_dm1_ref.m` (book Fig. 7.4) | same, plus the file's own **commented** `se = strel('diamond', 1);` un-commented and `se = strel('square', 3);` commented |
| `ch07_fillrec_sq3_ref.m` | same, plus the file's own commented `se = strel('square', 3);` swapped in |
| `ch07_fillrec_alt_ref.m` | same, plus the active `I0` literal replaced by the file's own **commented** alternative `I0` (lines 13–21), character for character |
| `ch07_filling76_ref.m` / `ch07_filling77_ref.m` (book Figs. 7.6/7.7) | same, plus `I0(5, 5) = 1;` — the book's variant image, which **no shipped `.m` contains** — and, for Fig. 7.7, `strel('square', 3)` instead of `strel('diamond', 1)` |
| `ice_shape_enhancement_ref.m` | function renamed; output list extended with the intermediates (`l, ice_area, A, ind, fill, t, floe_area, brash_area, color_floe, color_brash, floe_cen, brash_cen, index, nn_bw, nn_k, ysh, YT, z, n, color, zs, izs, ysh2, YT2`); **lines 39–181 verbatim**; lines 182–237 replaced by the same arithmetic with the graphics statements removed and the second `ysh`/`YT` renamed `ysh2`/`YT2`; the two tail blocks guarded by `if ~isempty(...)` so an ice-free fixture returns empties. **No numeric literal or expression changed.** |
| `ch7_demo_ref.m` (`sea_ice_demo.m`) | `rng(0);` inserted before the k-means stage (reproducibility, as in ch06); the `ice_shape_enhancement` call retargeted at the ref copy; line 57 `sea_ice_model` commented out (book §8.2 → ch08). Every parameter literal of lines 9–46 is verbatim. |

### R1 confirmed — the chapter's core file does **not** run unpatched in R2025a
`ice_shape_enhancement.m` lines 214–216/222/224 are HG1 code:

```matlab
h = bar(n(1 : nbins), z(1 : nbins));
ch = get(h,'Children');
fvd = get(ch,'Faces');
fvcd = get(ch,'FaceVertexCData');
...
    fvcd(fvd(i,:)) = color(i);
set(ch,'FaceVertexCData',fvcd)
```

In R2025a `get(barHandle,'Children')` returns an empty `GraphicsPlaceholder`, so `get(ch,'Faces')` errors. The
**exact statements removed** from the reference copy are: `rgb = label2rgb(...)`'s `figure,imshow(rgb)`, `hold on`,
the two `plot` loops over `floe_cen`/`brash_cen`, `axis off`, `colormap(jet)`, `colormap(jet(k))`, `hcb = colorbar;`,
`ytic = get(colorbar,'Ytick');`, the two `set(colorbar, …)` pairs, `figure,`, `h = bar(...)` and the four HG1
statements above. Everything they consume — `area_ice`, `n = 6`, `d = fix(...)`, `ysh`, the `YT` loop, `nbins = 50`,
`[z, n] = hist(floe_area, nbins)`, `[zs, izs] = sortrows(z', 1)`, `k = 255`, the `color(i) = fix((1 - exp(-n(i)/1000))
* 10000)` loop, `nn = 8`, the second `d`/`ysh`/`YT` block — is kept verbatim and saved. **The removed lines colour
bars; they compute nothing that is not saved.**

## pytest
`.venv/Scripts/python.exe -m pytest tests/test_ch07.py -q -p no:cacheprovider` → **`275 passed in 92.3 s`**,
0 failed, **0 xfailed**, 0 warnings (`outputs/ch07/verify/pytest_ch07.txt`). Nothing in this chapter is xfailed or
skipped: the strict `xfail` of the first pass (`hist` dropping `-Inf`) became a passing L2 test in `9927dfa`.
Breakdown by evidence level (re-counted test by test, by whether the body loads a MATLAB reference):
**35 L1** (synthetic truth), **202 L2** (MATLAB parity: 10 cleaning/labeling/filling, 5 Figs. 7.6/7.7, 67 `imfill`,
12 `imcomplement`, 32 `hist`, 64 `ice_shape_enhancement`, 6 end-to-end demo, 4 `sea_ice_test.jpg`, 2 k-means),
**25 L4** (quoted numbers), **13 script runs** (8 default + 5 non-default CLI flag combinations, all exit 0);
35 + 202 + 25 + 13 = 275.
The 32 tests added by this revision pass: 3 for the `single_eps` `imfill` case (conn 4, conn 8, class), 1 that
re-measures review M1 before/after, 13 for `imcomplement` (11 classes + the `uint32`/`uint64` guard + the `single`
round trip), 7 new `hist` fixtures in the parametrised parity test, 6 that pin the non-finite probes against the
review's MATLAB table, and 2 for `eps(edges)`.

Full suite `.venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider` → **`1849 passed, 2 skipped,
1 xfailed in 1 222.7 s`** (`outputs/ch07/verify/pytest_full.txt`; 1574 + 275 = 1849) = the ch02–ch06 baseline (**1574 passed, 2 skipped,
1 xfailed** — the surviving xfail is ch03's) **unchanged** plus the 275 ch07 tests. **No regression.** The 14
warnings are the same pre-existing ch03/ch04 pytest deprecations ch06 reported.

Parity label counts over the **44 rows** below (**27 `.m` files** + 7 new-primitive rows + 8 text-only algorithm
rows + 2 supporting rows): **exact 22 · near 5 · approx 1 · reimplemented 3 · unverified 3 · deferred 4**
(+ 3 mixed rows — `snakedeform.m` `exact`/`near`, Algorithm 5 `reimplemented` (book form) / `exact` (script form) and
the Eq. (7.6) colour-bar row `exact` / `near` for its degenerate `d == 0` branch — 2 display-only rows and 1
evidence-only row, the `binary_fill_holes` comparison).
Every one of the **27 `.m` files** has a row: 6 ported here, **17** byte-identical duplicates of ch6 files carried
forward (`ch7/Sea_Ice_Floe_Identification/` holds **23** `.m`, all 23 byte-identical to ch6's — 2 of them are ported
here and 4 deferred, leaving 17 reuse rows; 6 + 17 + 4 = 27) and 4 deferred to ch08/Appendix B.
The three `unverified` rows all have an Open item.

## Parity table
Levels: L1 synthetic truth · L2 MATLAB reference · L3 figure · L4 quoted number. "0 px"/"0.0" = identical arrays.

| MATLAB file / function | Python | Evidence | Result (measured) | Parity | Notes |
|---|---|---|---|---|---|
| `cleaning & labeling & filling/morphology_cleaning.m` | `ch07_ice_type.morphological_cleaning`, `scripts/ch07_morphology_cleaning.py` | L1 L2 L3 L4 | `I`, `f1`, `f2`, `f0`, `c1`, `c2`, `c3`, `c4` all **0 px** vs MATLAB; sums 111 → **115 / 104 / 108**; the printed Fig. 7.2(a)–(d) blocks in `core.synth` are **0 px** vs MATLAB too; `c2 == f1`, `c4 == f0` (L1) | exact | 2×2 SE ⇒ even/asymmetric origin — exercises `imdilate`'s SE reflection and `imclose`'s 0 pre-pad (ch04 rules) |
| `cleaning & labeling & filling/labeling.m` | `ch07_ice_type.connected_component_extract`, `scripts/ch07_labeling.py` | L1 L2 L3 L4 | `square,3` (Fig. 7.3): all blocks `xx, x1…x9` **0 px**, sums 9, 5, 8, 12, 17, 20, 23, 24, 24, 24, `X7 == A`; the commented `diamond,1` variant (Fig. 7.4): **0 px**, sums 5, 4, 7, 9, 12, 13, 14, 14, 14, 14. The port stops at the fixed point (9 / 8 blocks) and MATLAB's unrolled tail is verified to be that fixed point | exact | Eq. (7.1) itself is `reimplemented` (row below); the *file* is reproduced exactly |
| `cleaning & labeling & filling/filling.m` | `ch07_ice_type.hole_fill_dilation`, `scripts/ch07_filling.py` | L1 L2 L3 L4 | `I = ~I0`, `xx`, `x1…x9`, `I1 = x9 \| I0` all **0 px**; sums 5, 3, 4, 5, 6, 7, 8, 8, 8, 8 and `I1` 32 px; the 8 printed Fig. 7.5(e) blocks **0 px** | exact | |
| `cleaning & labeling & filling/filling_reconstruct.m` | `ch07_ice_type.border_marker` + `hole_fill_reconstruct` (+ `core.morphology.imfill`), `scripts/ch07_filling_reconstruct.py` | L1 L2 L3 | 3 variants (shipped `I0` × cross, shipped `I0` × `square,3`, the file's **commented alternative** `I0` × cross): `I`, `x0`, `x10 = H`, `I1`, `I2` all **0 px**; the Eq. (7.3) `border_marker(I0)` **equals the script's hand-written `x0`** in all three; sums 39 → 48 → 9 | exact | `H == binary_fill_holes(F)` and `== imfill(F,'holes')` for logical input (L1) |
| `Sea_Ice_Floe_Identification/ice_shape_enhancement.m` | **`ch07_ice_type.ice_shape_enhancement`**, `scripts/ch07_ice_shape_enhancement.py` | L1 L2 L3 L4 | **The chapter's core port. 9 controlled fixtures × 2 threshold forms × 2 crop modes = 36 runs; every array 0 px**: `out`, `l`, `fill`, `index`, `index_floe`, `index_brash`, `index_slush`, `index_water`, `index_residue`; and `t`, `nn_bw`, `nn_k`, the sorted areas `A`, the **stable** sort index `ind`, `color_floe`, `color_brash`, `floe_area`, `brash_area`, `floe_cen`/`brash_cen` (≤ 1e-12), the `coverage` struct (≤ 1e-12), the `regionprops` `Centroid`/`Perimeter` of every piece (≤ 1e-9), the 50-bin FSD (`z`, `n`, `color`) and both colour-bar blocks (`ysh`/`YT`, `ysh2`/`YT2`). **On the real image** (MATLAB's own `seg`/`bk` from the 44.7-min `demo` run, 1038×394, **1211 pieces = 982 light + 229 dark, 712 labels**): all nine arrays **0 px**, `ind` identical, coverage ≤ 1e-12, FSD and ticks identical, 433 floes / 274 brash | exact | R1's HG1 block is display-only (see Environment); the numbers it colours are saved and verified |
| `Sea_Ice_Floe_Identification/sea_ice_demo.m` | `scripts/ch07_sea_ice_demo.py` | L2 L3 L4 | the ch7 parameter block asserted value-by-value (`se_th 50`, `min_floe 40`, `min_brash 1`). The file is **byte-identical to ch6's `sea_ice_demo.m`**, so ch7 adds no parameter — those three literals are in ch6's copy too; ch06 simply never consumed them (review N4). End to end on ch7's own `sea_ice_test.jpg`: Algorithm 3 (`seaice_kmean_gvf`) differs from MATLAB on **0.245 %** of `seg` and **0.125 %** of `bk`; that propagates to **1232 vs 1211 pieces**, **433 / 290 vs 433 / 274** floe/brash, coverage within **0.13 pp** and **identical** Eq. (7.6) colour-bar ticks `[2, 173, 379, 640, 993, 1544, 2870]`. Algorithms 4–5 are exact given the same input (row above) | near | the residual is ch06's, not ch07's — see Deviation 2 |
| `SIFI/seaice_kmean_GVF_forenhancement.m` | `ch06_gvf_snake.seaice_kmean_gvf` (Algorithm 3) | L2 | **byte-identical to ch6's file** (`cmp -s`, re-verified: 23/23 SAME) — verified in ch06 (`near`, 0.049 % of `out`). **Re-measured here on ch7's own JPEG** (a different encoding, see the `sea_ice_test.jpg` row): `seg` 0.245 %, `bk` 0.125 % | near | ch06 row carried forward **and re-measured**, because ch7 calls it on a different input |
| `SIFI/GVF_distance.m` | `ch06_gvf_snake.gvf_distance` | — | byte-identical to ch6's file; verified in ch06 (`near`, `bw1` 16 of 31 730 px) | near | inherited |
| `SIFI/GVF.m` | `core.snake.gvf` | — | byte-identical; ch06 **exact** (44 cases 0.0) | exact | inherited |
| `SIFI/gradient2.m` | `core.snake.gradient2` | — | byte-identical; ch06 **exact** (7 fixtures 0.0) | exact | inherited |
| `SIFI/BoundMirrorExpand.m` | `core.snake.bound_mirror_expand` | — | byte-identical; ch06 **exact** | exact | inherited |
| `SIFI/BoundMirrorEnsure.m` | `core.snake.bound_mirror_ensure` | — | byte-identical; ch06 **exact** | exact | inherited |
| `SIFI/BoundMirrorShrink.m` | `core.snake.bound_mirror_shrink` | — | byte-identical; ch06 **exact** | exact | inherited |
| `SIFI/xconv2.m` | `core.snake.xconv2` | — | byte-identical; ch06 **exact** (≤ 9.1e-13) | exact | inherited |
| `SIFI/gaussianMask.m` | `core.snake.gaussian_mask` | — | byte-identical; ch06 **exact** (≤ 1.4e-20) | exact | inherited |
| `SIFI/gaussianBlur.m` | `core.snake.gaussian_blur` | — | byte-identical; ch06 **exact** (≤ 6.4e-14) | exact | inherited |
| `SIFI/snakedeform.m` | `core.snake.snakedeform` | — | byte-identical; ch06 **exact** (dense) / `near` (circulant) | exact / near | inherited |
| `SIFI/snakeinterp.m` | `core.snake.snakeinterp` | — | byte-identical; ch06 **exact** (16 of 18, 2 are the header's own admitted bug) | exact | inherited |
| `SIFI/snakeindex.m` | `core.snake.snakeindex` | — | byte-identical; ch06 **exact** | exact | inherited |
| `SIFI/snakedisp.m` | `core.plotting.snake_plot` | — | byte-identical; display only | — (display) | inherited |
| `SIFI/minboundrect.m` | `core.polygon.minboundrect` | — | byte-identical; ch06 **exact** (the shipped file needs `convhull(x,y,{'Qt'})` → `convhull(x,y)` to run in R2025a) | exact | inherited; no ch7 caller |
| `SIFI/polygeom.m` | `core.polygon.polygeom` | — | byte-identical; ch06 `near` (`ang1` ±π) | near | inherited; no ch7 caller |
| `SIFI/homofil.m` | `core.filters.homomorphic_butterworth` | — | byte-identical; ch06 **exact** (≤ 1.14e-12). Re-checked against the ch07 text: §7.3.1.1 prescribes local sub-image processing, not homomorphic filtering | exact | **still an orphan** — no caller, no book section |
| `SIFI/sea_ice_model.m` | → `ch08_applications.sea_ice_model` | — | not ported here (book §8.2); ch7 *feeds* it (`ice_floe`, `brash_ice`, `index_floe`) and the struct MATLAB builds is verified field by field — the names are exactly `('Center', 'Area', 'Perimeter', 'PixelsPosition')` and `Center`, `Area`, `Perimeter` and the full `PixelsPosition` matrix are identical for every piece | deferred | Open item 3 |
| `SIFI/SeaIce_Image_Structure.m` | → ch08 / Appendix B | — | not ported here | deferred | Open item 3 |
| `SIFI/color_hist.m` | → `scripts/ch08_color_hist.py` | — | not ported here; its input `floe` is `sea_ice_model`'s output. It contains the **same HG1 block** that breaks in R2025a | deferred | Open item 3 |
| `SIFI/color_hist_comparison.m` | → `scripts/ch08_color_hist_comparison.py` | — | not ported here | deferred | Open item 3 |
| **New core primitives** | | | | | |
| MATLAB `imfill(I[, conn], 'holes')` — l. 73/93, **Eq. (7.4)** | **`core.morphology.imfill`** | L1 L2 | **21 constructed cases × conn {4, 8} = 42 comparisons, all 0 px**, and the **output class** matches MATLAB on all 21 (`logical→logical`, `double→double`, `uint8→uint8`, `single→single`, `int16→int16`). Cases: square-with-hole in 5 classes, nested holes, a hole open to the border, a diagonal-only hole, an 8-connected ring, all-true, all-false, a single pixel, random binary, a checkerboard, three genuinely **grayscale** surfaces with two basins, a signed `int16` image, an `±Inf` plant and — added by this revision — `single_eps`, the review's 5×5 `single` probe (`1e-8`, `1+1e-8`, `0.1`, rest `2`) that **only** the class-preserving `imcomplement` gets right. `'hole'`/`'holes'`/`'h'` prefix matching == MATLAB's `validatestring`; the default connectivity == `conndef(2,'minimal')` = 4 | exact | line-by-line port of R2025a `imfill.m` l. 124–145. **Defect found and fixed** (review M1, re-measured here, Deviation 11): before `5186e9d` the `single` branch differed from MATLAB on **2 of 25** elements, max **2.2351742e-08** |
| — the justification, pinned | vs `scipy.ndimage.binary_fill_holes` | L2 | `binary_fill_holes` equals MATLAB on **every logical conn-4 case** but **differs for conn 8** (3 of 13 logical cases: `ring8`, `random_binary`, `checker`) and **never** reproduces a non-logical input — wrong values for `uint8` 0/255, wrong values *and* wrong class for all five grayscale/`int16`/`±Inf` cases. ch7 calls `imfill(b,'hole')` on a **double**, so the library call would have been wrong here | — | analysis R2 closed with measurements, not assertion |
| MATLAB `imcomplement(I)` — inside `imfill.m` l. 128/140 (**revised in ch07**, review M1) | `core.matlab_compat.imcomplement` | L1 L2 | **11 classes, all values and all classes identical to MATLAB** (`logical, uint8, uint16, uint32, uint64, int8, int16, int32, int64, single, double`; every call wrapped in a MATLAB `try/catch` — **none** was refused). The two branches ch02 never fixtured are now pinned: `uint32`/`uint64` take `intmax − im` (`imcomplement(uint32([0 1 2]))` = `[4294967295, 4294967294, 4294967293]`, **not** the float64 `[1, 0, −1]` the port returned before `5186e9d`), and the float branch stays in the input class — MATLAB `imcomplement(single(1e-8))` = **exactly 1**, `imcomplement(imcomplement(single(0.1)))` = **0.100000024** ≠ the double `0.09999999999999998`, both reproduced bit for bit | exact | ch02 primitive **re-verified and corrected here**; it decides `imfill`'s `single` branch (row above) |
| MATLAB `hist(y, n)` / `hist(y, centres)` — l. 212 | `core.histogram.hist` | L1 L2 | **23 of 23 cases 0 px** (counts *and* centres) + both empty-input forms: scalar `n` ∈ {1, 2, 3, 4, 5, 7, 10, 50}, `min == max` (the `±n/2 − 0.5` widening), a single value, negative data, values exactly on the internal edges, explicit uniform and non-uniform **centre** vectors, values far outside the centre range (counted in the unbounded outer bins, not dropped), a 2-centre vector, 403 floe-like areas, the mixed `NaN`/`±Inf` vector (`[2 1 1 2]`, the first pass's Open item 1, **fixed in `9927dfa`**), the **six all-/mixed-non-finite probes** of review S3 — `[Inf −Inf NaN]`→`[1 0 0 1]`, `[NaN NaN]`→`[0 0 0]`, `[Inf Inf]`→`[0 0 2]`, `[−Inf −Inf]`→`[2 0 0]`, `[1 −Inf]`→`[1 1]`, `[−Inf 0 Inf]`→`[1 0 1 0 1]`, which exercise the `finite.size == 0` (`miny = maxy = 0`) branch — and the `edges + eps(edges)` discriminator `y = −2 + eps(−2)/2` with centres `[−3 −1 1]`: MATLAB **`[1 0 0]`**, the port `[1 0 0]`, a `np.nextafter` port `[0 1 0]`. MATLAB's own `eps([−1 −2 −0.5 1 2 0 −3])` equals `abs(np.spacing(·))` bit for bit on all seven | exact | `np.histogram` reproduces neither the centres nor the outer-bin rule. Known unreachable deviation: `n < 1` raises (Deviation 11c) |
| **Eq. (7.6)** colour map + inverse-labelled colour bar, l. 127 / 194–206 / 226–237 | `ch07_ice_type.size_color`, `color_to_area`, `colorbar_area_ticks`, `core.plotting.size_colorbar` | L1 L2 L4 | `ysh`/`YT` (`n = 6`, map) and `ysh2`/`YT2` (`nn = 8`, histogram) **identical to MATLAB on all 9 fixtures and on the real image**. **All 11 printed tick lists** (Figs. 7.13, 7.15, 7.19, 7.20, 7.21, 7.26(a)–(c), 7.28(a)–(c)) are reproduced by the port's arithmetic, and the `(min colour, step)` pair that produces each is **unique** — 5 (or 7) degrees of freedom of agreement per figure. `colorbar_area_ticks` is `near`, not `exact`, on the degenerate `d == 0` branch (Deviation 11a) | exact / near (`d == 0`) | but see Deviation 5: for four of the eleven the tick list **saturates** and constrains nothing |
| `label2rgb(index, @jet, [1 1 1])` with `max(index) ≈ 10⁴` (R6) | `core.plotting.label2rgb` | L2 L3 | does not blow up on ~10 000 colour values; background is the white triple; the **arrays** (`index`) are 0 px vs MATLAB, and the two rendered maps are visually identical (`sec_7_2_identification_compare.png`) | — (display) | colour PNGs deliberately **not** compared (CUMULATIVE pitfall 25) |
| `core.synth.FIG_7_2_*`, `FIG_7_3_*`, `FIG_7_4_STEPS`, `FIG_7_5_*`, `FIG_7_6_*`, `FIG_7_7_STEPS(_BOOK)`, `FIG_7_8_*` | `core.synth` | L2 L3 | every printed matrix and every printed block of Figs. 7.2–7.8 compared against MATLAB running the original `.m` — 4 + 8 + 7 + 8 + 4 + 10 + 7 = **48 blocks, 47 of them 0 px**; the 48th (Fig. 7.7(e) `X8`) differs by exactly **2 px**, the **book typo** of Deviation 1 | exact (printed truths) | `FIG_7_6_IMAGE` is a *derivation* — no shipped `.m` contains it — and is confirmed by running the patched copy |
| **Text-only algorithms** | | | | | |
| **Eq. (7.1)** general `n`-iteration extraction | `ch07_ice_type.connected_component_extract` | L1 L2 | equals the `bwlabel` component containing the seed for **both** connectivities; monotone; converges at k = 8 (8-conn) and k = 7 (4-conn); every block equals MATLAB's | reimplemented | from the equation, on top of the exact `core.morphology.imdilate` |
| **Eq. (7.2)** dilation hole filling incl. both failure cases | `ch07_ice_type.hole_fill_dilation` | L1 L2 L3 | Fig. 7.5 8/8 blocks, Fig. 7.6 4/4, Fig. 7.7 10/10 (against MATLAB); the cross SE provably misses the second hole and the 3×3 square provably floods the whole complement | reimplemented | |
| **Algorithm 3** (p. 157) | `ch07_ice_type.sea_ice_edge_detection` | L2 | a thin wrapper over `ch06_gvf_snake.seaice_kmean_gvf`; measured on ch7's image above | near | inherits ch06 |
| **Algorithm 4** (p. 158) | `ch07_ice_type.sea_ice_shape_enhancement` | L2 | an alias of `ice_shape_enhancement` (whose l. 39–61 already build one labelling out of the light **and** dark layers, which is what the algorithm requires) — exact by the `ice_shape_enhancement.m` row | exact | |
| **Algorithm 5** (p. 160) written out on its own | `ch07_ice_type.ice_types_classification` | L1 | the four layers **tile the image exactly once**; `book_threshold=False` reproduces the M-file's `index_floe`/`index_brash`/`index_slush`/`index_water` **0 px** on 3 fixtures; the `>=` form differs **only** on pieces of exactly `T_floe` (and, with `min_brash = 1`, on 1-pixel pieces the code drops entirely); `PIXEL = IDENTIFICATION ∪ ICE` measurably grows the ice set | reimplemented (book form) + exact (script form) | Deviation 3 |
| **Algorithm 6** (p. 168) + §7.3.1.1 tiling/stitching | `ch07_ice_type.tile_grid`, `local_segmentation`, `scripts/ch07_local_processing.py` | L1 | **no `.m` exists.** The tiling covers every pixel; the kept (overlap-removed) cores are an exact partition (every pixel claimed once); the script runs end to end | reimplemented | Open item 1 — the book fixes neither the tile size nor the overlap |
| **§7.3.1.2** orthorectification of a *labelled* image | `ch07_ice_type.resample_categorical` (+ ch10) | L1 | nearest-neighbour resampling provably preserves the label set {0, 0.5, 1}; the camera model (shooting angle 20°, FOV 46°) is **Appendix A / ch10** | unverified (rectifier) | Open item 2 |
| **§7.3.2** sensitivity sweeps (Figs. 7.24, 7.27) | `scripts/ch07_sensitivity.py` | L3 | the book prints **no counts**, only axis ranges and curve shapes, so this is `unverified` **by construction**. Measured here: the port's snake = 1 run reproduces Fig. 7.26(a)'s printed ticks at `--downscale` 2 **and** 1 — and so would *any* run whose largest piece is ≥ 8112 px (Deviation 5), so that agreement is **not** recorded as parity | unverified | Open item 5 |
| Figs. 7.1, 7.9, 7.10–7.21 and every number attached to them | — | — | the source images (155×125, 205×263, and the aerial scene) are **not shipped anywhere in `MATLAB_ROOT`** | unverified | Open item 6 — **154/189, 2511/2624 and the eight coverage percentages cannot be reproduced and were not fabricated** |
| Statistics TB `kmeans` on **ch7's** JPEG (R9) | `core.clustering.kmeans_lloyd(init='kmeans++', seed=0)` | L2 | sorted centres MATLAB `[77.774, 183.265, 240.027]` vs port `[77.298, 182.223, 239.728]`, **max 1.04 gray levels**; `bk` agrees on **99.875 %** of the 408 972 pixels (510 px differ). ch06 measured **0 px** for this stage — but on **ch6's** copy of the same photograph | approx | never compare labels; Deviation 2 |

## Figures reproduced
Compare PNGs: `outputs/ch07/verify/<name>_compare.png`, copies in `reports/ch07/figures/` (git-ignored, CLAUDE.md
rule 12). Layout: Python row | MATLAB row (the arrays the original `.m` produced) | book page rendered from
`chapters/ch07.pdf` at 110 dpi. Per-block pixel differences in `outputs/ch07/verify/figure_diffs.json`.

| Figure | File | Verdict |
|---|---|---|
| 7.2(a)–(d) (p. 146) | `fig_7_2_compare.png` | The 13×23 matrix, its 2×2-square closing (+4 px), opening (−7 px) and cleaning (108 px) are pixel-identical across all three rows and match the book's printed matrices panel for panel: the closing thickens the horizontal bridge between the two blocks and keeps the isolated corner pixel, the opening breaks that bridge and deletes the isolated pixel while leaving a 1-px hole, and the cleaning (opening of the closing) leaves two solid blocks. |
| 7.3(d) (p. 148) | `fig_7_3_compare.png` | All eight printed blocks of the 8-connected extraction agree between Python, MATLAB and the book (sums 9, 5, 8, 12, 17, 20, 23, 24), and the last one is the whole object, exactly as the text's "completes at the 8th iteration" says. |
| 7.4(d) (p. 149) | `fig_7_4_compare.png` | The 4-connected variant reaches its fixed point one block earlier (7 blocks, sums 5, 4, 7, 9, 12, 13, 14) and stops at the 4-connected component, leaving the diagonal tail of Fig. 7.3 out — identical in all three rows. |
| 7.5(e) (p. 150) | `fig_7_5_compare.png` | The seeded hole grows one cross-step at a time (sums 5, 3, 4, 5, 6, 7, 8) to its fixed point `X6` and the union with `A` gives the 32-pixel filled object; Python, MATLAB and the book agree pixel for pixel. |
| 7.6(e) (p. 151) | `fig_7_6_compare.png` | With the cross SE the recursion saturates after two steps at a 4-pixel region — only the hole containing the seed is filled, the second 4-connected hole survives in the union (29 px), which is precisely the failure the book illustrates. |
| 7.7(e) (p. 152) | `fig_7_7_compare.png` | The 3×3 square SE lets the recursion escape into the background and flood it (sums 9, 4, 6, 13, 19, 29, 38, 47, **55**, 56); **the printed `X8` shows 53 pixels where MATLAB and the port both give 55** — the two red-ringed cells at 1-based (8, 9) and (9, 9) are a book typo (Deviation 1). |
| 7.8 (p. 153) | `fig_7_8_compare.png` | The complement (60 px), the Eq. (7.3) border marker (27 px — and the port's `border_marker` equals the script's hand-written `x0` exactly), the reconstruction complement `H` (48 px), `H ∪ F` and the 9 filled hole pixels are all identical between Python and MATLAB and match the book's panels. |
| 7.22 (p. 168) | `fig_7_22_compare.png` | ch7's own `sea_ice_test.jpg`, displayed **transposed** (394 × 1038), is unmistakably the marginal-ice-zone scene the book prints as Fig. 7.22 — same floe field, same aspect, same brightness gradient (the analysis' NCC 0.9787 confirmed visually). |
| §7.2 identification layers (**not** a book figure) | `sec_7_2_identification_compare.png` | On ch7's `sea_ice_test.jpg` the Eq. (7.6) size-coloured identification map and all five binary layers (`index_floe`, `index_brash`, `index_slush`, `index_water`, `index_residue`) are **0 px** against MATLAB's own `imwrite` output; Figs. 7.13–7.16 themselves use a 205×263 image the book does not ship, so this is a *procedure* reproduction and is labelled as such in the figure's own title. |
| 7.1, 7.9, 7.10–7.21, 7.23–7.28 | — | Not reproducible as figures: the source images are not shipped and no `.m` file produces them (Open item 6). The mechanisms are shown on `sea_ice_test.jpg` and on synthetic fixtures by `scripts/ch07_*.py` (63 PNGs in `outputs/ch07/`). |

## Numbers from the text (book vs ours)
| Item | Page | Book | Ours (Python) | MATLAB (original .m) | Status |
|---|---|---|---|---|---|
| Fig. 7.2 SE = 2×2 square; matrix has 111 object px | 146 | — | 111 → 115 / 104 / 108 | identical | match |
| Fig. 7.3 "completes at the 8th iteration" | 148 | 8 | `n_iter = 8`, `X7 == A` | identical blocks | match |
| Fig. 7.4 4-connected, 7 printed blocks | 149 | 7 | `n_iter = 7` | identical | match |
| Fig. 7.5 "finishes at the 7th iteration … union of `A` with `X7`" | 150 | 7 | `n_iter = 7` (`X7 == X6`); the printed blocks stop at `X6` before the union — both readings give the same 32-px image | identical | match (convention stated) |
| Fig. 7.7(e) `X8` | 152 | 53 px | **55 px** | **55 px** | **book erratum** (Deviation 1) |
| `C1 = 10000`, `C2 = 1000` (Eq. 7.6) | 159 | 10000 / 1000 | `C1_COLOR`, `C2_AREA` | identical | match |
| Eq. (7.5) `r1 = 1`, `r2 = 2`, `size_th = 50` | 158 | — | `adaptive_se_radius` at 49 → 1, 50 → 2 | `if r < se_th, r = 1; else r = 2` | match (the README's "≤ se_th" is the odd one out) |
| brash ice "not more than 2 m across" | 160 | 2 m | `T_floe` is a **pixel** proxy (40); exposed as a parameter | `min_floe = 40` | match (no scale is given) |
| Fig. 7.13 ticks 3, 131, 277, 448, 656, 917, 1273 | 159 | those 7 | reproduced exactly from a **unique** `(lo, d) = (29, 1195)`; implies smallest piece 3 px, largest 1273–1274 px | (arithmetic) | match (arithmetic); the *image* is unshipped |
| Fig. 7.15 ticks 21, 114, 218, 333, 463, 612, 788 | 162 | those 7 | unique `(207, 874)`; largest floe **exactly 788 px** | (arithmetic) | match (arithmetic) |
| Figs. 7.19 / 7.20 / 7.21 ticks | 165–167 | 7 / 7 / 9 | unique `(207, 1631)`, `(207, 1632)`, `(1820, 1022)` | (arithmetic) | match (arithmetic) |
| Figs. 7.26(a)–(c) / 7.28(a)–(c) ticks | 171, 173 | 6 × 7 | unique `(19, 1663/1500/1566/1610/1505/1506)` | (arithmetic) | match (arithmetic) — but 7.26(a) is **saturated**, Deviation 5 |
| "a total of **154** ice floes and **189** brash ice pieces … 60.52 / 3.34 / 16.03 / 20.11 %" (Fig. 7.10(a)) | 160 | those | **not reproduced** — the 205×263 source image is not shipped | — | **unverified**, Open item 6 |
| "A total of **2511** ice floes and **2624** brash ice pieces … 65.98 / 5.03 / 17.52 / 11.47 %" (Fig. 7.17) | 165 | those | **not reproduced** — the aerial source image is not shipped | — | **unverified**, Open item 6 |
| image "394 × 1038" | 168 | 394×1038 | `imread` gives 1038×394×3 | `size` = [1038 394 3] | match |
| snake sweep 1 → 122 with GVF fixed at 500; GVF sweep 1 → 1500 step 20 with snake fixed at 100 | 168–172 | those | `FULL_SWEEP`/`BOOK_PANELS` in `scripts/ch07_sensitivity.py` | — | match (ranges); counts unverified |
| shooting angle 20°, FOV 46° | 166 | those | not used in ch07 (→ ch10) | — | deferred |
| `sea_ice_demo.m` parameters `se_th 50`, `min_floe 40`, `min_brash 1` | script | — | asserted value-by-value | identical | match — the file is **byte-identical to ch6's**, so these are not ch7 additions; ch06 just did not consume them (review N4) |
| `ice_shape_enhancement.m` `nbins = 50`, colour bar `n = 6` / `nn = 8`, `k = 255` | script | — | asserted; the 50-bin FSD is 0 px vs MATLAB | identical | match |
| `bwlabel(bw, 4)` on `sea_ice_test.jpg` after Otsu | analysis pre-check | **215** (ch7) vs 231 (ch6) | **215** (Python, same JPEG decode as MATLAB) | **215** / 231 | **analyst's pre-check confirmed** |
| Otsu level on both copies | analysis | 162/255 | 0.635294 | 0.635294 on **both** | match |
| ch7's vs ch6's `sea_ice_test.jpg` | analysis | "84 % of samples differ, max 43" | — | **84.13 %**, max **43** | **analyst confirmed** |
| `strel('disk', r)` neighbourhoods used by Eq. (7.5) | — | — | r = 1 → 3×3 **cross**/5 px, r = 2 → 5×5/13 px, r = 3 → 5×5/25 px | identical | match (r = 1 is the cross, **not** a 3×3 square) |

## Deviations & justifications
1. **Book erratum — Fig. 7.7(e) `X8`.** The printed block has **53** object pixels; MATLAB running the original
   recursion (`ch07_filling77_ref.m`, the shipped `filling.m` with the book's variant image and `strel('square',3)`)
   gives **55**. The two extra pixels are at 1-based **(8, 9)** and **(9, 9)**, and they are forced: `X7` already
   contains (7, 9) and (8, 8), the 3×3 dilation of either reaches both cells, and `A^c` has both set (the printed
   `X9 = X10 = A^c` has them set too). The port reproduces the **recursion's** answer
   (`core.synth.FIG_7_7_STEPS`) and keeps the printed variant for the record
   (`FIG_7_7_STEPS_BOOK`, `FIG_7_7_X8_TYPO`); both are tested, and the compare figure rings the two cells in red.
   The analyst's and porter's reading is confirmed **by MATLAB**, not by argument.
2. **ch7's `sea_ice_test.jpg` is a different JPEG encoding from ch6's, and ch06's numbers do not carry over.**
   Same photograph, same 1038×394×3 shape, but **84.13 %** of samples differ (max |Δ| 43). Confirmed consequences:
   `graythresh` is 162/255 on **both**, but `bwlabel(bw, 4)` gives **215** components here against ch6's **231**; and
   the k-means stage, which ch06 measured as **0 px** on ch6's copy, now differs on **510 px (0.125 %)** with sorted
   centres up to **1.04 gray levels** apart. That is the whole of the chapter's inexactness: `seg` differs on 0.245 %
   of pixels, which changes 1232 → 1211 identified pieces and 290 → 274 brash pieces (the floe count, 433, and the
   colour-bar ticks are unchanged). Fed **MATLAB's own** `seg`/`bk`, `ice_shape_enhancement` is 0 px on every array.
   `core.io.load_image('ch07', 'sea_ice_test.jpg')` resolves to `data/book/ch07/...` (md5 `54e56aa8…`, verified
   equal to `MATLAB_ROOT/ch7/...` and different from ch6's `fca33143…`).
3. **The book's `≥` vs the code's `>` (Algorithm 5).** p. 160 says "sizes equal to or larger than `T_floe`"; the
   shipped code writes `if area0 > min_floe` / `elseif area0 > min_brash`. With `min_brash = 1` the code's form
   **drops every 1-pixel piece** from FLOE *and* BRASH (it survives only as slush). Measured on the real image with
   the port's own Algorithm 3: **433 floes / 290 brash (code)** vs **436 / 289 (book)** — 3 pieces of exactly 40 px
   move up and 2 pieces of exactly 1 px reappear. The porter's numbers are confirmed. The default follows the
   **code** (`book_threshold=False`); `ice_types_classification`, which *is* the pseudocode, defaults to the book's
   `>=`. Both forms have MATLAB references (the reference run uses `min_floe = 39`, `min_brash = 0`, which for
   integer areas is exactly the `>=` form) and both are 0 px on all 9 fixtures.
4. **The bounding-box crop is a proven-equivalent optimisation, re-verified independently.** The M-file allocates a
   full-image scratch array per piece (`b = zeros(size(out)); b(p) = 1`). `crop=True` runs the same four operations
   on a bounding box padded by `4r + 6` and clipped to the image. Re-measured here against **MATLAB**, not against
   the port's own `crop=False`: on the real image, **1211 pieces of which 106 touch an image border**, both
   `crop=True` (**1.6 s**) and `crop=False` (**243.7 s**) reproduce MATLAB's `out` and `fill` **bit-exactly**; and on
   the purpose-built adversarial fixture `border_notch` — every piece touching a border and carrying a notch cut in
   from it, which is exactly where ch04's "`imclose` pre-pads with 0" rule could bite — both modes are 0 px against
   MATLAB. The crop window is clipped to the image, so a border-touching piece keeps the *real* border in both cases.
5. **Four of the eleven printed colour-bar tick lists are saturated and constrain nothing.** Eq. (7.6) is
   `fix(10000 (1 − e^{−A/1000}))`, which is **9997 for every `A ≥ 8112`** and never exceeds 9999. Enumerating the
   largest-piece area that reproduces each printed list (with the printed minimum colour held fixed) gives the
   bands: 7.13 **[1273, 1274]**, 7.15 **[788, 788]**, 7.19 [7265, 9210], 7.26(b) [2322, 2327], 7.26(c) [2839, 2849],
   7.28(a) [3439, 3457], 7.28(b) [2353, 2359], 7.28(c) [2360, 2365] — but 7.20 **[9211, ∞)**, 7.21 **[7825, ∞)** and
   7.26(a) **[8112, ∞)**. Consequence: the porter's flagged oddity is **a coincidence, and is now proved to be one**.
   Re-run at `--downscale 1` as well as 2, the port's snake = 1 sweep gives ticks
   `[2, 184, 407, 695, 1100, 1792, 8112]` — byte-identical to printed Fig. 7.26(a) — at **both** scalings, even
   though the two runs identify **319/260** and **393/336** pieces respectively. Any run whose smallest kept piece is
   2 px and whose largest is ≥ 8112 px prints that list. **It is not recorded as parity.** What *is* verified is the
   tick arithmetic itself: for each of the eleven figures the `(min colour, truncated step)` pair that reproduces the
   printed integers is **unique**, so reproducing 7 (or 9) integers from 2 is a real check of
   `color_to_area`/`colorbar_area_ticks` — of the formula, not of the floe areas.
   The porter's commit message claim "Eq. 7.6 colour-bar ticks match printed Fig. 7.13 exactly" is **not** supported
   as an image-level result: on ch7's own image the map ticks are `[2, 173, 379, 640, 993, 1544, 2870]`, not Fig.
   7.13's. (The shipped `scripts/ch07_sea_ice_demo.py` already prints "a different image, not comparable", so the
   code is right and only the commit message overstates it.)
6. **MATLAB's `sort` is stable and the tie order decides which label survives** (R8). Verified directly
   (`[A, ind] = sort([4 4 6 4 1 6 1 9 4 6 1 2])` → `ind = [5 7 11 12 1 2 4 9 3 6 10 8]`, i.e.
   `np.argsort(kind='stable')`), and end to end: the port's `order + 1` equals MATLAB's `ind` on all 9 fixtures
   (including the deliberate 6-way and 4-way ties) **and on the 1211 pieces of the real image**.
7. **`imfill` returns the input class, and that matters here.** `ice_shape_enhancement.m` passes a **double** 0/1
   array, so MATLAB takes the *grayscale* branch of `imfill.m` and returns a double which `bwlabel`, `imclose` and
   `out(pp) = t` then see. `scipy.ndimage.binary_fill_holes` would have returned `bool`. Both the values and the
   class are verified against MATLAB for all five classes exercised.
8. **`regionprops` is `core.regionprops`, not skimage.** `ice_shape_enhancement.m` l. 129–133 asks for `centroid`
   and `perimeter` in the `regionprops(out == i, …)` per-label call form; the port evaluates them on a 1-px-padded
   bounding-box crop. Both match MATLAB to ≤ 1e-12 (centroid) and ≤ 1e-9 (Vossepoel–Smeulders perimeter) on every
   piece of every fixture. skimage's perimeter is 5.5 % different (ch06 Deviation 9) and would have been wrong.
9. **`strel('disk', 1)` is the 4-connected cross, not a 3×3 square** (5 px, 3×3), and `strel('disk', 2)` is 13 px in
   5×5. This is what Eq. (7.5) actually applies. Verified against MATLAB's `getnhood`. A consequence worth stating:
   opening a solid rectangle with `strel('disk',1)` removes its four corners, so a piece's *identified* area is not
   its labelled area — which is why the `thresholds` fixture's raw areas (1, 2, 20, 39, 40, 49, 50, 51) become
   35, 36, 38, 45 after cleaning.
10. **`ice_area` is reassigned inside the M-file.** Line 107 clears it and refills it with the *identified* pieces,
    so the variable MATLAB returns is not the pre-sort vector. The port exposes the pre-sort vector as
    `ice_area` and the sorted/permuted forms as `np.sort(ice_area)` / `order`; the comparison in the tests is
    against MATLAB's `A` and `ind`, which are the unambiguous witnesses.
11. **Three deviations on branches the book's parameters never reach** (added by the review pass, S6 — they were
    in the code but in neither list). All three are *stated choices*, not unknowns:
    (a) `ch07_ice_type.colorbar_area_ticks` when `d = fix((max − min)/n)` is **0**: MATLAB's `min : 0 : max` is the
    **empty** vector, so its `for i = 1:length(ysh)` loop never runs and the colour bar gets no ticks; the port
    returns the single tick `min` so the axis is still labelled. The function is therefore labelled **`near`**, not
    `exact` (the parity row above says so, and `tests/test_ch07.py` asserts `vals.size <= 1` for this input).
    (b) `ice_shape_enhancement(...).fsd` is `None` when `floe_area` is empty, where MATLAB `hist([], 50)`
    (`hist.m` l. 81–89) returns `zeros(1, 50)` counts at centres `1:50`. This was never *compared* because the
    reference wrapper `ice_shape_enhancement_ref.m` guards the tail block with `if ~isempty(floe_area)` — an
    honest gap in the L2 evidence, closed by argument rather than by measurement: `core.histogram.hist` itself
    **is** compared against MATLAB on both empty-input forms (`z_empty`, `z_empty_c`) and is 0 px, so only the
    wrapper's `None` differs, and only for an ice-free image.
    (c) `core.histogram.hist` **raises** for a scalar `n < 1`, where MATLAB takes the `binwidth = Inf` branch
    (`hist.m` l. 124–125) and returns a degenerate histogram. `nbins` is 50 or 8 everywhere in the book.
12. **Two defects found by the independent review and fixed under this report; both re-measured here.**
    (a) **M1 — `imcomplement` promoted `single` to `double`.** `imfill.m` l. 128–140 complements twice around the
    reconstruction and MATLAB evaluates both complements **in the input class**, so a `single` image round-trips
    through `1 − (1 − x)` in single. Measured against MATLAB on the review's 5×5 `single` probe (`I(1,1) = 1e-8`,
    `I(1,3) = 1 + 1e-8`, `I(5,5) = 0.1`, rest `2`), exchanged as a `.mat` so both sides see identical bits:
    **before `5186e9d`: 2 of 25 elements differ, max 2.2351742e-08** ((1,1) `1e-08` vs MATLAB `0`; (5,5) `0.1` vs
    MATLAB `0.100000024`); **after: 0 of 25, conn 4 and conn 8, class `single`**. The verifier reproduced the
    pre-fix behaviour by monkey-patching the old expression back in
    (`TestL2Imfill::test_single_branch_needs_the_class_preserving_imcomplement`), so the regression is pinned in
    both directions. The same commit fixed `uint32`/`uint64`, which fell through to the float branch
    (`imcomplement(uint32([0 1 2]))` returned `float64 [1, 0, −1]`); all 11 MATLAB classes now have a reference.
    (b) **`hist` dropped `−Inf`** (the first pass's Open item 1, fixed in `9927dfa`): the filter kept `+Inf` but
    not `−Inf`, so `hist([1 2 NaN 3 Inf −Inf 4], 4)` gave `[1 1 1 2]` against MATLAB's `[2 1 1 2]`. The strict
    `xfail` that recorded it is now a passing L2 test, and six further all-/mixed-non-finite probes (review S3)
    plus the `edges + eps(edges)` discriminator (review S2) were added as real fixtures.

## Open items
1. **Algorithm 6 and §7.3.1.1 have no MATLAB code at all** and the book fixes neither the sub-image size, the
   overlap size, nor the stitching rule (p. 163–164 is prose). `tile_grid`/`local_segmentation` are
   `reimplemented` with `tile`, `overlap` and `merge` as parameters; only the structural properties are provable
   (the tiling covers the image; the overlap-removed cores partition it exactly once — 700 configurations). There
   is nothing to compare against, and no book number for §7.3.1.1 exists.
2. **§7.3.1.2 orthorectification is deferred to ch10 (Appendix A).** ch07 exposes only the categorical
   (nearest-neighbour) resampling, which is verified to preserve the label set; the camera model (shooting angle
   20°, FOV 46°) and the rectification itself belong to Appendix A. Nothing in ch07 should implement a second
   rectifier (analysis R14).
3. **Four `.m` files are deliberately not ported here** — `sea_ice_model.m` (book §8.2), `SeaIce_Image_Structure.m`
   (Appendix B), `color_hist.m` and `color_hist_comparison.m` (§8.3). They consume ch7's outputs; all four field names
   (`Center`, `Area`, `Perimeter`, `PixelsPosition`) are present in `IcePiece` and every one of them was compared
   element-wise against MATLAB's own struct array in the reference. `color_hist*.m` contain the **same HG1 bar-colouring
   block** that breaks in R2025a, so ch08 will need the same patch. Resolves when ch08 lands.
4. **Three deliberate deviations on unreachable branches** (Deviation 11), listed here so this section is complete:
   (a) `colorbar_area_ticks` keeps one tick where MATLAB's `min:0:max` is empty — the function is labelled `near`
   for that branch; (b) `ice_shape_enhancement(...).fsd` is `None` for an ice-free image where MATLAB
   `hist([], 50)` returns `zeros(1,50)`/`1:50` — never compared, because the reference wrapper guards the tail
   with `if ~isempty(floe_area)` (the underlying `core.histogram.hist` **is** compared on both empty forms and is
   0 px); (c) `core.histogram.hist` raises for `n < 1` where MATLAB takes the `binwidth = Inf` branch. None is
   reachable from the book's parameters (`nbins` is 50 or 8; the colour spread of any real ice field exceeds `n`).
5. **§7.3.2 is `unverified` by construction.** The book prints only axis ranges (Fig. 7.24 y 200–1600, x ticks
   1, 6, …, 121; Fig. 7.27 y 200–2000, x ticks 1, 61, …, 1441) and the panel colour-bar ticks, no counts. The port
   reproduces the *procedure* and the qualitative shape; the tick agreement at snake = 1 is a saturation
   coincidence (Deviation 5) and must not be quoted as parity. A full sweep is 25 + 75 pipeline runs.
6. **Figs. 7.1, 7.9 and 7.10–7.21 are `unverified` as figures and as numbers.** The 155×125 §7.1 crops, the
   205×263 §7.2 image and the aerial §7.3.1 scene are not shipped anywhere in `MATLAB_ROOT` (the analysis searched
   every non-`.m` file in the tree; `ch7/` itself ships only `README.docx` and `sea_ice_test.jpg`, re-checked here)
   and no `.m` produces them. **154/189 pieces, 2511/2624 pieces and the eight coverage percentages
   therefore cannot be reproduced, and no substitute image was used to invent them.** Needs the source images from
   the authors; not a port failure. The *mechanism* is demonstrated on ch7's own `sea_ice_test.jpg` and on
   `core.synth` fixtures.

### Closed since the first pass (kept for the record)
* ~~`core.histogram.hist` drops `-Inf`~~ — **fixed** in `9927dfa`; the strict `xfail` is now a passing L2 test and
  six more non-finite probes back it (Deviation 12b). The parity label is now **`exact`**.
* ~~Two documentation corrections for the porter~~ — **done** in `9927dfa`: the `hist` docstring's parity claim, and
  the input attribution of the real-image numbers (**1211 pieces = 982 + 229, 712 labels, 433/274, 106 border
  pieces** on *MATLAB's* segmentation; 1232 and 433/290 are the port's own Algorithm-3 output).
* ~~"the brief asks for 19 REUSE rows; there are 17"~~ — **corrected** in `progress.json` and `analysis/ch07.md`:
  `ch7/Sea_Ice_Floe_Identification/` holds **23** `.m`, **23/23 byte-identical** to ch6's, giving 6 ported + **17**
  reused + 4 deferred = **27**.
* ~~`imcomplement` promotes `single` to `double` (review M1)~~ — **fixed** in `5186e9d` and re-measured by the
  verifier: 2/25 elements at max 2.235e-8 before, **0/25** after (Deviation 12a); `uint32`/`uint64` now take
  `intmax − im` and all 11 MATLAB classes have an L2 reference.

## Verdict: PASS  (revised — the review's findings are closed, the numbers below are the current ones)

All 8 scripts run headless (8 default + 5 non-default CLI combinations, exit 0), **every one of the 27 `.m` files has
a row** — 6 ported and measured against MATLAB R2025a running the original code, **17** byte-identical duplicates of
ch6 files carried forward (`ch7/Sea_Ice_Floe_Identification/` holds 23 `.m`, 23/23 byte-identical to ch6's; 6 + 17 + 4
= 27) with `seaice_kmean_GVF_forenhancement.m` **re-measured** because ch7 calls it on a different JPEG, and 4
explicitly deferred to ch08/Appendix B with Open item 3 — and every new `seaice/core` primitive has its own row. The chapter's high-risk targets are closed with numbers, not opinion:

* **`ice_shape_enhancement.m`, the chapter's one new algorithm, is exact.** It had to be patched to run at all in
  R2025a (R1 confirmed: the HG1 `get(h,'Children')`/`'Faces'`/`'FaceVertexCData'` block errors), but the patch
  removes only graphics statements and changes no numeric expression, and every quantity those statements consume is
  saved and compared. Across 9 controlled fixtures × 2 threshold forms × 2 crop modes and on the real 1038×394 image
  fed **MATLAB's own** `seg`/`bk`, all nine output arrays, `t`, `nn_bw`, `nn_k`, the sorted areas, the stable sort
  index, both colour vectors, every `regionprops` centroid and perimeter, the `coverage` struct, the 50-bin FSD and
  both colour-bar tick blocks are **0 px / ≤ 1e-12**.
* **`imfill` is exact in values *and* class** on **21** constructed cases × 2 connectivities including the double,
  single, `uint8`, `int16`, grayscale and `±Inf` branches, and the reason it had to be re-implemented is pinned:
  `scipy.ndimage.binary_fill_holes` matches MATLAB only on the logical conn-4 branch and never on the double input
  ch7 actually passes. The `single` branch is exact **since `5186e9d` only**: the review found `imcomplement`
  promoting `single` to `double`, and the verifier re-measured it against MATLAB on the 5×5 probe — **2 of 25
  elements, max 2.2351742e-08 before; 0 of 25 after** (Deviation 12a). `imcomplement` now has its own row and an L2
  reference for all **11** MATLAB classes, including the `uint32`/`uint64` branches that used to return float64.
* **The 48 printed blocks of Figs. 7.2–7.8 are 47 × 0 px**, and the 48th differs by exactly two pixels that are a
  **book erratum** adjudicated by MATLAB itself (Fig. 7.7(e) `X8`: printed 53, computed 55).
* **The analyst's pre-checks are confirmed**: 215 vs 231 `bwlabel` components, identical Otsu 162/255, 84.13 % of
  samples differing between the two JPEG encodings; and **the porter's two measured claims are confirmed**
  (`>` vs `>=` → 433/290 vs 436/289 from 3 pieces of 40 px and 2 of 1 px; the bounding-box crop 0 px against
  MATLAB on 1211 pieces of which 106 touch a border, 1.6 s vs 243.7 s).

The residual differences are quantified and attributed: the whole of the chapter's inexactness lives in ch06's
Algorithm 3 running on **ch7's re-encoded JPEG** — k-means sorted centres 1.04 gray levels apart, `bk` 0.125 % and
`seg` 0.245 % of pixels, propagating to 1232 vs 1211 pieces and 290 vs 274 brash pieces with the floe count and the
colour-bar ticks unchanged. The three `unverified` rows (the unshipped §7.2/§7.3.1 images, the §7.3.2 sweeps, the
ch10 rectifier) each have an Open item, and **no book number attached to an unshipped image was reproduced from a
substitute**. **Nothing in this chapter is xfailed, skipped or tolerance-relaxed:** the two defects the first pass
and the independent review turned up — `core.histogram.hist` dropping `-Inf` and `imcomplement` promoting `single` to
`double` — were fixed in `9927dfa` / `5186e9d` and are now pinned by passing L2 tests in **both** directions
(Deviation 12), and the three remaining deliberate deviations sit on branches the book never reaches and are listed
as Open item 4. Test suites pass: `tests/test_ch07.py` **275 passed, 0 failed, 0 xfailed, 0 warnings**; full suite
**1849 passed, 2 skipped, 1 xfailed** (the xfail is ch03's), no regression against the ch06 baseline of 1574 / 2 / 1.
