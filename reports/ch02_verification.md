# Chapter 2 verification — Digital Image Processing Preliminaries

Date 2026-09-09 · port commit `9e79f05` (`ch02: port — core primitives ... + 9 scripts`) + porter's uncommitted fixes to
`seaice/core/interp.py` and `seaice/core/histogram.py` (Open items 1–2, re-verified below) · verifier artefacts:
`tests/test_ch02.py`, `reference/ch02/make_refs.py` (+ 8 `.mat`), `reference/ch02/make_compare_figures.py`,
`outputs/ch02/verify/` (compare PNGs, MATLAB `imwrite`/`print` images, `image_diffs.json`), `reports/ch02/figures/`.

## Environment
python 3.11.5, numpy 2.4.6, scipy 1.17.1, scikit-image 0.26.0, matplotlib 3.11.1 |
reference engine: **MATLAB 25.1.0.2833191 (R2025a) Prerelease Update 2** via `tools/run_matlab_ref.py` (`matlab -batch`,
figures invisible) — all 8 reference files (`reference/ch02/refs_log.json`: engine `matlab`, status `ok`).
Octave absent; no fallback used. The original `.m` files were executed verbatim (`run('histogram.m')`,
`run('color_image.m')`, `run('distance_transform.m')`, `run('chain_diff.m')`, `boundaries`/`fchcode`/`bound2im` called
directly from `MATLAB_ROOT/ch2/chain code`); `histogram.m`/`color_image.m` ran with cwd = `outputs/ch02/verify/scratch`
(copy of `rgb.JPG`) so their `saveas` PNGs never touched MATLAB_ROOT.

## pytest
`.venv/Scripts/python.exe -m pytest tests/test_ch02.py -q -p no:cacheprovider` →
**`83 passed in 74.96s (0:01:14)`** (full suite `tests/`: `83 passed in 96.19s`).
History: the first verification round found two port defects (`interp_nearest` mutating its input on scalar queries;
`imhist(logical)` defaulting to 256 bins instead of MATLAB's 2) → `2 failed, 81 passed`. The porter fixed both
(`np.atleast_1d` indices in `interp_nearest`; `nbins: int | None = None` → 2 for bool, 256 otherwise in `imhist` /
`normalized_histogram`); the two tests now pass and were re-measured against the MATLAB references (see table).
All nine `scripts/ch02_*.py` ran headless (`--no-show --out <tmp>`) with exit 0 and wrote files (`test_script_runs[*]`, 9/9 pass).

## Parity table
Levels: L1 synthetic truth · L2 MATLAB reference · L3 figure · L4 quoted number. Errors are max-abs unless stated.

| MATLAB file / function | Python | Evidence | Result (measured) | Parity | Notes |
|---|---|---|---|---|---|
| `color_image.m` (CMY, `imcomplement`) | `ch02_preliminaries.color_components`, `core.color.rgb2cmy`, `core.matlab_compat.imcomplement` | L2 L3 | `I_cmy`, `Ic`, `Im`, `Iy`: 0 differing values (uint8 and `1 - double` planes) | exact | `imcomplement` also checked on uint8/uint16/int8/logical/double vectors (`compat.mat`): identical |
| `color_image.m` (HSI as written, `2*Ig` bug) | `rgb2hsi(matlab_bug=True, scale=255)` | L2 L3 | `Ii` 0.0, `Is` 0.0, `Ih` 0.0; NaN count 0 on both sides (no pixel of `rgb.JPG` has R == G) | exact | `Ih` is flat −π/4 (MATLAB spread 1.2e-14); Fig 2.5(a) shows this flat image |
| — HSI, book Eq. (2.6a–c) (`2*Ib`) | `rgb2hsi()` (default) | L2 (vs corrected MATLAB snippet) | with `scale=255`: H 5.6e-17; S 7.1e-14, I 5.7e-14 (×255). With RGB in [0,1] (text convention): identical except **1 px / 3 145 728** where `R+G == 2B` (V1 = 0 exactly → `atan(V2/0) = ±π/2`, sign set by 1e-16 rounding on both sides); `use_atan2=True` vs `atan2` 2.1e-14 everywhere | reimplemented (exact vs snippet) | The single pixel is a property of Eq. (2.6b) at V1 = 0, not of the port; test asserts that every mismatch satisfies `R+G == 2B` and `|H| = π/2` on both sides |
| — CMYK Eqs. (2.4)–(2.5) (text only) | `core.color.rgb2cmyk(u, b)` | L1 L2 (hand-coded MATLAB Eq. 2.4) | 0.0 for (u,b) = (1,1) and (0.5,0.7); pure-colour truths (red→[0,1,1,0], black→[0,0,0,1]) | reimplemented | no comparable MATLAB builtin |
| `histogram.m` (`rgb2gray`, manual loop, `imhist`) | `gray_histogram`, `channel_histograms`, `core.histogram.imhist`, `normalized_histogram` | L2 L3 L4 | `num`/`cnt`/`y_r`/`y_g`/`y_b`: identical; `GP` 0.0; peaks 47 840@208, 77 513@5, 64 866@212, 71 090@227 identical to MATLAB | exact | |
| — `rgb2gray` | `core.matlab_compat.rgb2gray_matlab` | L1 L2 | **0 of 3 145 728 px differ** (uint8, from MATLAB's decoded `I`); double input 0.0 vs `rgb2gray(im2double(I))`; pure colours → 76/150/29 | exact | NTSC coefficients 0.298936…/0.587043…/0.114021… confirmed |
| — `imread` (JPEG decode) | `core.io.load_book_image` (imageio/Pillow) | L2 | **0 of 9 437 184 samples differ** vs MATLAB `imread('rgb.jpg')` | exact | |
| — `imhist` variants | `imhist(I, n)` | L1 L2 | uint8 64 bins, uint8 256, double 256, double 100, uint16 256, 5-bin edge case: identical counts and centres; logical default → 2 bins `[355, 119645]` at centres `[0, 1]` = MATLAB (after fix, Open item 2); uint8 default still 256 bins, identical | exact | |
| — `round`, `im2uint8`, `im2double` | `core.matlab_compat.*` | L1 L2 | identical on the half-away-from-zero vector, the [−0.1, 1.1] ramp and uint8/uint16/logical inputs | exact | uint8 saturation demonstrated (`200+100 → 255`) in the reference |
| `distance_transform.m` (`bwdist` euclidean/cityblock/chessboard) | `point_distance_maps`, `core.distance.bwdist` | L2 L3 | `De` 7.6e-6 (MATLAB single), `D4` 0, `D8` 0, script's `imgDist` 0; random mask: `Re` 1.9e-7, `R4` 0, `R8` 0; all-zero image → Inf both sides | exact | float32 rounding only |
| — `bwdist(..., 'quasi-euclidean')` | `core.distance.quasi_euclidean_dt` | L1 L2 | port vs analytic `max+(√2−1)·min`: 7.1e-13; MATLAB vs analytic 1.1e-4 (single-precision chamfer accumulation, rel 8e-7); random mask 1.9e-7 | reimplemented (near vs MATLAB) | compared with `rtol=1e-5` (skill default); MATLAB is the less precise side |
| — Eq. (2.9) `distance_transform`, Figs 2.12/2.13 | `core.distance.distance_transform`, `center_distance_map` | L1 L2 L4 | `D12` 3.3e-8 vs MATLAB, 3.3e-8 vs printed Fig 2.12(b) (0/1/1.4142/2/2.2361); `C4`/`C8` 0; `Ce` 1.9e-7; corners 6 / 3 | exact | |
| — `bwlabel` (§2.3, used by `boundaries.m`) | `core.connectivity.label_components` | L1 L2 L4 | Fig 2.11: `L4`/`L8` identical **including label numbers** (5 / 2 objects); 200×300 thresholded `rgb.JPG` crop: `LQ4`/`LQ8` identical (6 objects) | exact | column-major first-occurrence relabel reproduces MATLAB numbering |
| — §2.3 `n4/nd/n8`, `is_adjacent`, `is_m_adjacent`, `find_paths`, `region_boundary_mask` | `core.connectivity` | L1 | sizes 4/4/8 interior, 2/1/3 corner; Fig 2.10: no 4-path, >1 8-paths, exactly 1 m-path; boundary ring 16 px | reimplemented (text only) | no MATLAB code in ch2 |
| `chain code/boundaries.m` | `core.chaincode.boundaries` | L1 L2 | Fig 2.19: 19 pts identical (+1); Fig 2.11: 2 (8-conn) + 5 (4-conn) objects, cw and ccw, all identical; single pixel → 2 identical rows; spur shape (double traversal) identical; hollow square → 1 exterior boundary (25 pts); real mask: 6 objects, 1050 (8-conn) / 1096 (4-conn) points identical; empty image → `[]` | exact (reimplemented) | |
| `chain code/fchcode.m` | `core.chaincode.fchcode`, `first_difference`, `code_reverse`, `min_magnitude` | L1 L2 L4 | `fcc/diff/mm/diffmm/x0y0` identical (18 codes, start (4,2)); 4-conn code (22) + `diff/mm/diffmm` identical; `'reverse'` and ccw codes identical; real mask: 1044 codes and `mm` identical for all 6 objects; `min_magnitude` = lexicographic minimum over rotations (brute force, n = 1…18) | exact | |
| — `minmag` on periodic / length-1 codes | `min_magnitude` tie-break | L2 (negative) L4 | MATLAB `minmag_ref(dif)` on the Fig 2.21 first difference → **error** `Output argument "z" ... not assigned`; MATLAB `fchcode` on a single-pixel boundary → same error; port returns the first candidate = book's normalised first difference `0 6 1 1 6 0 7 1 6 …` (p. 31) / `[0]` | reimplemented | deliberate deviation (documented in `chaincode.py`); ch5 must not rely on MATLAB failing here |
| `chain code/bound2im.m` | `core.chaincode.bound2im` | L1 L2 | `bound2im(b)`, `(b')`, `(b,12,14)`, `(b,M,N,x0,y0)`: identical masks; error paths raise | exact | Python `x0,y0` are 0-based (`MATLAB 2,3` ↔ `Python 1,2`) |
| `chain code/chain_diff.m` | `scripts/ch02_chain_diff.py` → `chain_code_demo` | L2 L3 L4 | `B`, `b`, `d = 19`, `bim` identical; `'cww'` typo reproduced as clockwise | exact | |
| — §2.5 `conv2` | `core.filters.conv2` | L1 L2 | `same` (3×3, 5×5, 4×4, 2×3), `full` (3×3, 5×5), `valid`: ≤ 3.6e-15 | exact | even-kernel centring reproduced |
| — §2.5 `imfilter` | `core.filters.imfilter` | L1 L2 | corr/conv; zeros/replicate/symmetric/circular/constant 2.5; even 4×4 and 2×3 kernels; `'full'` (3×3, 5×5, 4×4); 3-channel input: ≤ 3.6e-15 | exact | |
| — §2.6 set / logical ops, Table 2.1, p. 29 | `core.setops` | L1 L4 | truth tables; `57 & 207 = 9`; De Morgan; `A−B = A∧¬B`; reflect∘reflect = id and `a → −a` about the origin; translate; gray complement/max/min | exact (by construction) | no MATLAB reference needed |
| — §2.8 `interp2` nearest / linear / cubic | `core.interp.interp2`, `interp_nearest`, `interp_bilinear`, `interp_bicubic` | L1 L2 L3 | grid `1:0.4:32`: nearest 0.0, linear 2.8e-14, cubic 6.0e-13 (**borders included**, `pad='quadratic'`); 200 scattered queries incl. `.5` ties and out-of-range → NaN pattern identical, 0 / 2.8e-14 / 6.3e-13; `pad='replicate'` differs by up to 7.9 within 1 px of the border (interior identical); after the Open-item-1 fix scalar out-of-range queries leave the input untouched for all three methods and `Zn`/`Sn`/`R4n` remain 0.0 | exact | tie rule = round half away from zero confirmed |
| — `imresize` | `core.interp.resize` | L2 L3 | nearest ×4/×2/×0.5: 0.0; bilinear ×4/×2 (and ×0.5 with antialiasing off): 0.0; bicubic ×4/×2: interior (≥ 2 source px from the border) 0.0, border up to 8.2 gray levels (976 of 16 384 px > 1e-9) | approx (bicubic border) / exact otherwise | cause: `resize` clips out-of-range query coordinates (u = −0.375 → 0) before the kernel, `imresize` evaluates the kernel at the true coordinate with clamped source indices; `pad='replicate'` alone does not close it (5.2) |
| — `keys_kernel` Eq. (2.41) | `core.interp.keys_kernel` | L1 | rc(0)=1, rc(±1)=rc(±2)=0, rc(0.5)=0.5625, rc(1.5)=−0.0625 (a = −0.5) | exact | |
| — `indexed_to_rgb` (§2.1.3) | `core.color.indexed_to_rgb` | L1 | clip + one-based option | reimplemented (unverified vs `ind2rgb`) | trivial; Fig 2.6 is a diagram |
| — `core.synth` fixtures | `FIG_2_12_A/B`, `FIG_2_11`, `FIG_2_19_OBJECT`, `spur_shape`, `point_image`, `FIG_2_21` | L2 L4 | `B` identical to `chain_diff.m`'s matrix; `D12` from MATLAB matches the printed Fig 2.12(b) to 3.3e-8; Fig 2.21 five sequences identical to p. 31 | exact | |

Every `.m` in `MATLAB_ROOT/ch2` (`color_image.m`, `distance_transform.m`, `histogram.m`, `chain code/boundaries.m`,
`chain code/fchcode.m`, `chain code/bound2im.m`, `chain code/chain_diff.m`) has a row above.

## Figures reproduced
Compare PNGs: `outputs/ch02/verify/<name>_compare.png` (copies in `reports/ch02/figures/`). Layout: Python row |
MATLAB `imwrite`/`print` row (when the `.m` code produces the figure) | book page rendered from `chapters/ch02.pdf`.
`image_diffs.json` holds pixel diffs between Python PNGs and MATLAB PNGs of the same raw image.

| Figure | File | Verdict |
|---|---|---|
| 2.3 (p. 14) | `fig_2_03_compare.png` | Python and MATLAB red/green/blue-colormapped channel images are visually identical and match the book panels. |
| 2.4 (p. 14) | `fig_2_04_compare.png` | Ic/Iy/I_cmy PNGs identical (0 px), Im differs by ±1 level on 0.44 % px (display quantisation of `mat2gray`), all match the book's three panels. |
| 2.5 (p. 15) | `fig_2_05_compare.png` | Is/Ii PNGs identical (0 px); the script's hue is a flat −π/4 whose stretched 1e-14 rounding noise differs between engines (max 2 levels) — both reproduce the featureless Fig 2.5(a). |
| 2.5 corrected | `fig_2_05_corrected_compare.png` | Book-equation hue (2*Ib) shows faint structure on both sides; S and I as above. |
| 2.7 (p. 17) | `fig_2_07_compare.png` | **Substitute image** (`rgb2gray(rgb.JPG)`, PNG identical to MATLAB, 0 px): the Python and MATLAB `imhist` plots agree (peak 47 840 @ 208) but the book's Fig 2.7 uses an image that is not shipped (peak ≈7.2e4 @ ≈195) — `unverified`, see Open item 3. |
| 2.8 (p. 17) | `fig_2_08_compare.png` | R/G/B histogram curves coincide with MATLAB's `plot(x, y_r, ...)` and with the book (red peak at 5, green ≈212, blue ≈227, y-limit 8e4). |
| 2.14 (p. 23) | `fig_2_14_compare.png` | Point image and euclidean map PNGs identical (0 px); cityblock/chessboard ±1 level on 0.8 % / 3.2 % px (float32 `mat2gray` rounding) — circle / diamond / square maps match the book. |
| 2.19/2.20 (p. 30) | `fig_2_19_compare.png` | `bim` masks identical (0 px after 1-bit PNG normalisation); the "starting point" label sits at (4,2) in both the Python and the MATLAB `imshow`+`text` figure; the Python Fig 2.19 panels show the 4- and 8-direction codes of the book. |
| 2.22/2.23 (pp. 33–34) | `fig_2_22_compare.png` | interp2 nearest/linear/cubic upsampled crops are visually identical to MATLAB's (numerically ≤ 6e-13); the book pages are diagrams only. |
| 2.24 / resize (p. 34) | `fig_2_23_compare.png` | ×4 nearest and bilinear PNGs identical (0 px); bicubic differs by ≤ 8 levels on 5.6 % px, all within 2 source pixels of the border. |
| 2.1–2.2 (p. 12) | `fig_2_01_02_compare.png` | Gray-value table and binary pattern are illustrative (the printed 11×11 crop location is unknown); layout follows the book. |
| 2.6 (p. 16) | `fig_2_06_compare.png` | Index matrix + colormap → RGB expansion mirrors the book's structure diagram. |
| 2.9–2.10 (pp. 18–19) | `fig_2_09_10_compare.png` | N4/ND/N8 diagrams and the 4-/8-/m-path examples match the book. |
| 2.11 (p. 20) | `fig_2_11_compare.png` | 5 four-connected vs 2 eight-connected components, same matrix as the book. |
| 2.12 (p. 21) | `fig_2_12_compare.png` | 7×7 matrix and Euclidean DT values (1.4142, 2.2361) match the printed table. |
| 2.13 (p. 22) | `fig_2_13_compare.png` | City-block (corner 6) and chessboard (corner 3) maps match the printed tables. |
| 2.15 (p. 24) | `fig_2_15_compare.png` | Numeric 3×3 convolution example (Eq. 2.15 terms) consistent with the book diagram. |
| 2.16–2.17 (pp. 26–27) | `fig_2_16_compare.png`, `fig_2_17_compare.png` | A, B, Aᶜ, A∪B, A∩B, A−B panels and the reflection/translation of an L-shape match the book's arrangement. |
| 2.18 (p. 29) | `fig_2_18_compare.png` | 4- and 8-direction numbering identical to the book. |
| 2.25 (p. 35) | `fig_2_25_compare.png` | Keys kernel a = −0.5 (MATLAB) vs −0.75 (OpenCV) plotted; book page is the coefficient diagram. |

## Numbers from the text (book vs ours)
| Item | Page | Book | Ours (Python) | MATLAB | Status |
|---|---|---|---|---|---|
| `I(1076,675)` | 14 | R 28, G 76, B 114 | 28, 76, 114 | 28, 76, 114 | match |
| Gray histogram peak (rgb.JPG) | 17 (Fig 2.7 is a different image) | ≈7.2e4 @ ≈195 (other image) | 47 840 @ 208 | 47 840 @ 208 | MATLAB match; book figure not reproducible (Open item 3) |
| R/G/B peaks (Fig 2.8) | 17 | curves, y ≤ 8e4 | 77 513@5, 64 866@212, 71 090@227 | identical | match |
| Fig 2.11 components | 20 | 5 (4-adj), 2 (8-adj) | 5, 2 | 5, 2 | match |
| Fig 2.12(b) values | 21 | 0, 1, 1.4142, 2, 2.2361 | same (3.3e-8) | same | match |
| Fig 2.13 corners | 22 | 6 (city-block), 3 (chessboard) | 6, 3 | 6, 3 | match |
| `57 AND 207` | 29 | 9 | 9 | — | match |
| Chain start / boundary length | 31 | (4,2), 18 codes (19 pts closed) | (4,2), 18 / 19 | (4,2), 18 / 19 | match |
| Original code | 31 | `0 1 2 0 0 7 0 6 6 4 5 6 4 4 3 4 2 2` | same | same | match |
| First difference | 31 | `1 1 6 0 7 1 6 0 6 1 1 6 0 7 1 6 0 6` | same | same | match |
| Normalized code | 31 | `0 0 7 0 6 6 4 5 6 4 4 3 4 2 2 0 1 2` | same | same | match |
| First diff. of normalized | 31 | `0 7 1 6 0 6 1 1 6 0 7 1 6 0 6 1 1 6` | same | same | match |
| Normalized first difference | 31 | `0 6 1 1 6 0 7 1 6 0 6 1 1 6 0 7 1 6` | same | MATLAB `minmag` errors | match (port tie-break) |
| Fig 2.19(b) 4-direction code | 30 | 22 codes | 22 (`0 0 1 1 0 0 3 0 0 3 3 2 2 3 3 2 2 1 2 2 1 1`) | identical | match |

## Deviations & justifications
1. **`rgb2hsi` default follows Eq. (2.6a) (`2*Ib`)**, not `color_image.m` line 21 (`2*Ig`); the script's result is available via `matlab_bug=True` and is exact vs MATLAB. Book text is the authority on *what*.
2. **HSI on [0,1] vs 0–255**: H is scale-free except where V1 = 0 mathematically (1 px here); `scale=255` gives bit-level parity.
3. **`min_magnitude` tie-break** on periodic / length-1 codes where MATLAB's `minmag` errors — required to produce the book's normalised first difference; MATLAB's error is captured in `chain_diff.mat: minmag_err` and `boundaries_multi.mat: errs`.
4. **`boundaries` DIPUM behaviours** (exterior only, double traversal of spurs, single pixel = two identical points) replicated, not fixed — verified against MATLAB.
5. **Quasi-euclidean**: port computed in float64 (exact chamfer); MATLAB returns single with ~1e-4 accumulation error → `near`.
6. **`resize` vs `imresize`**: bicubic border handling differs (see table); no antialiasing when shrinking — `approx` as declared by the porter.
7. **`chain_diff.m` `'cww'` typo** → clockwise trace reproduced with `direction='cw'`.
8. Coordinates: Python 0-based (row, col); all MATLAB comparisons add 1 on the Python side (`b`, `x0y0`, `bound2im` offsets).

## Open items
1. **FIXED, verified — `seaice/core/interp.py: interp_nearest` mutated its input for scalar queries** (0-d `u, v` → `img[i, j]` was a basic-indexing view; `out[out_mask] = fill` wrote `NaN` into the caller's `img[0,0]`). Fix by the porter: indices passed through `np.atleast_1d` (fancy indexing → copy) and reshaped to the query shape. Re-verified: `interp_nearest/interp_bilinear/interp_bicubic(img, -0.1, 0.0)` leave `img` unchanged; `interp2` nearest vs MATLAB `Zn`/`Sn` 0.0, `resize` nearest ×4 vs `imresize` 0.0. Test `TestL1Interp::test_interp_nearest_scalar_query_must_not_mutate_input` passes.
2. **FIXED, verified — `seaice/core/histogram.py: imhist` on logical images** returned 256 bins where MATLAB returns 2. Fix by the porter: `imhist`/`normalized_histogram` take `nbins: int | None = None` → 2 for bool, 256 otherwise. Re-verified vs `compat.mat`: `[355, 119645]` at centres `[0, 1]` (MATLAB `[355; 119645]`, `[0; 1]`); uint8 default still 256 bins and identical to MATLAB. Test `TestL2Compat::test_imhist_logical_default_is_two_bins` passes.
3. **Fig 2.7 `unverified`**: the book's 8-bit grayscale image (histogram peak ≈7.2e4 near 195) is not among the shipped files; the figure is illustrated with `rgb2gray(rgb.JPG)` (peak 47 840 @ 208, MATLAB-identical) and labelled as a substitute. Not a port failure; needs the source image (user / other chapter folder) to close.
4. **`resize` bicubic border (`approx`)**: optional improvement — evaluate the Keys kernel at the unclipped coordinate with clamped source indices (imresize semantics) to reach `exact` at the border for enlargement; interior is already exact. Not required for the ch2 figures.
5. **Behavioural divergence to remember for ch5**: MATLAB `fchcode` raises on single-pixel objects and on periodic codes (`minmag`); the port returns `[0]` / the lexicographic minimum. Any ch5 code that relied on MATLAB's failure (e.g. try/catch around `fchcode`) must be ported with this in mind.

## Verdict: PASS
All seven `.m` files have exact parity against MATLAB R2025a, all nine scripts run headless, all 83 tests pass, all 21
figure comparisons and all quoted numbers agree. The two port defects found in the first round (Open items 1–2) were
fixed by the porter and re-verified against the saved MATLAB references. Remaining open items 3–5 are not port
failures: Fig 2.7 is `unverified` (source image not shipped, substitute labelled), `resize` bicubic border is a
documented `approx`, and the `fchcode`/`minmag` divergence is a documented design decision for ch5 to respect.
