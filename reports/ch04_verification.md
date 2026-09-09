# Chapter 4 verification — Ice Edge Detection

Date 2026-09-09 · port commit `8fc4cf0` (`ch04: port — edges.py (MATLAB edge sobel/prewitt/roberts/log, 74/74 maps 0 px
vs R2025a), morphology.py (strel disk decomposition, imerode/imdilate/open/close/reconstruct, gradients), fspecial, ch04
module + 3 scripts`) · **re-verified 2026-09-09 after the porter's `imclose` fix** (uncommitted working tree on top of
`8fc4cf0`: `seaice/core/morphology.py::imclose` now pre-pads by `ceil(nhood.shape/2)` with MATLAB's pad value and crops;
`imopen` unchanged) · verifier artefacts: `tests/test_ch04.py` (485 tests), `reference/ch04/make_refs.py` (+ 5 `.mat`
incl. the new `imclose_pad.mat`, `inputs.mat`, `refs_log.json`), `reference/ch04/refine_crop.py`,
`reference/ch04/make_compare_figures.py`, `outputs/ch04/verify/` (MATLAB `imwrite` images in `matlab/`, patched script
copies in `scratch/`, `compat_code.m`, `imclose_pad_code.m`, `crop_refine.json`, `image_diffs.json`, `report_numbers.txt`,
pytest logs `pytest_ch04.txt` / `pytest_full.txt`), `reports/ch04/figures/` (18 compare PNGs, git-ignored).

## Environment
python 3.11.5, numpy 2.4.6, scipy 1.17.1, scikit-image 0.26.0, opencv 5.0.0, matplotlib 3.11.1 |
reference engine: **MATLAB 25.1.0.2833191 (R2025a) Prerelease Update 2** via `tools/run_matlab_ref.py` (`matlab -batch`,
figures invisible) — all 4 reference files (`reference/ch04/refs_log.json`: engine `matlab`, status `ok`). Octave absent;
no fallback used.

How the original code was run: `derivative.m` and `morphology.m` hard-code `imread('test.jpg')` and read from the
current folder. Each was copied **verbatim** into `outputs/ch04/verify/scratch/` (`derivative_ref.m`, `morphology_ref.m`)
and run with cwd = scratch holding a copy of `data/book/ch04/test.jpg`; the workspace variables were saved after appending
extra `edge(...)` calls (`derivative.mat`) and `graythresh`/`getnhood`/`decompose` (`morphology.mat`). For the book
figures ("of Figure 4.3(a)") the same two scripts were patched in exactly two tokens — the crop
`im = im(1600:2151, 1979:2552, :)` appended to the `imread` line and `strel('dis', 7)` → `strel('dis', 15)` — giving
`crop_fig4_3.mat`. MATLAB_ROOT was never modified nor put on the MATLAB path. `compat.mat` calls the toolbox functions
(`edge`, `fspecial`, `strel`/`getnhood`/`decompose`, `imerode`, `imdilate`, `imopen`, `imclose`, `imreconstruct`,
`medfilt2`, `bwareaopen`, `conv2`, `imfilter`) on controlled arrays written from Python (`inputs.mat`: seeded random
double/uint8/int16/negative-double fields, linear and diagonal ramps with exact gradient ties, vertical/horizontal/diagonal
steps, integer plateaus, a constant image, two-blob marker/mask pairs, an asymmetric 3×3 SE, an even 2×3 SE, the Fig. 4.8
matrices, the 1-D two-floe profile) and hand-codes Eqs. (4.6)–(4.15) in MATLAB; 816 reference variables. `imclose_pad.mat` (re-verification, 28 variables) calls `imclose` on the signed fixtures `Neg`/`I16` (+ `Rnd8`) with SEs larger than 15 px (`diamond 8`, `disk 10`, `line 31/45°`, `rectangle [3 17]`) and two 15-px controls (`diamond 7`, `disk 8`), plus hand-coded zero-padded / −Inf-padded / unpadded compositions and the `images.UseHalide` setting.

## pytest
Re-verification run (after the porter's `imclose` fix): `.venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider`
(full suite, ch02 + ch03 + ch04) → **`683 passed, 1 xfailed in 4159 s`** (`outputs/ch04/verify/pytest_full.txt`; the wall
time is an artefact of a heavily loaded host — 7 orphaned `MATLABWindow` processes — the same suite took 414 s in the first
run). All 86 ch02 tests and 112 + 1 xfail ch03 tests pass; `tests/test_ch04.py` → **`485 passed in 91 s`** (`pytest_ch04.txt`;
465 original tests + 20 new: 18 `TestL2ImcloseBorderRule::test_imclose_signed_large_se[*]`, `test_decomposed_se_list`,
`test_which_pad_value_matlab_used`). The previously failing 43 `TestL2ErodeDilate::test_imclose[*]` all pass;
`test_imclose_padding_reference_identity` was corrected — its last assertion had encoded the old defect (port == unpadded
composition) and now asserts port == MATLAB `imclose` == padded form ≠ unpadded form (plus `imopen` unpadded). The three
`scripts/ch04_*.py` were re-run headless with `--no-show`: exit 0 each (`outputs/ch04/ch04_run_*.log`); no `.m` file or book
figure uses `imclose`, so the 18 compare PNGs and the 27 MATLAB-vs-Python raw image pairs (0 px, `image_diffs.json`) are
unchanged and were not regenerated.

First run (before the fix, for the record): `43 failed, 620 passed, 1 xfailed in 414 s`, every failure being
`TestL2ErodeDilate::test_imclose[<SE>-<image>]` (border band, Open item 1 — now resolved).

Parity label counts (30 rows below): **exact 22 · near 1 · approx 0 · reimplemented 5 · unverified 1** (+ 1 unlabelled row: Canny, not ported by design).

## Parity table
Levels: L1 synthetic truth · L2 MATLAB reference · L3 figure · L4 quoted number. Errors are max-abs unless stated. Both
`.m` files of `MATLAB_ROOT/ch4` have a row; the remaining rows are the new `seaice/core` primitives and the text-only
algorithms of `analysis/ch04.md` §3.

| MATLAB file / function | Python | Evidence | Result (measured) | Parity | Notes |
|---|---|---|---|---|---|
| `derivative.m` (`double(im)/256`, `edge(im,'sobel',0.05)`) | `ch04_ice_edge_detection.sobel_edges_script`, `scripts/ch04_derivative.py` | L2 L3 | `im` 0 diff (12 252 240 px); `BW` **0 px differ**, 120 224 edge px = MATLAB `n_edge`; `gv`/`gh` ≤ 1e-16; `msk`, `r`, `c` identical; `imshow(BW)` image identical (`image_diffs.json`) | exact | literal `/256` (not `im2double`) reproduced |
| `derivative.m` commented lines 6/8/15 (`medfilt2`, `bwareaopen(BW,20)`, `conv2(double(BW),double(msk))`) | `sobel_edges_script(median=…, min_area=…, smooth=…)`, `bwareaopen` | L2 | `im_med` 0 diff, `BW_med` 0 px (115 436); `BW_ao` 0 px (95 564); `BWc` 2860×4294 `'full'` identical (max 6) | exact | flags off by default as in the script |
| — `edge` variants on `test.jpg` (`derivative.mat`) | `core.edges.edge` | L2 | auto Sobel T 0.04933174586661371 (rel 1e-12) / 121 091 px 0 diff; T = 0.03 162 994 px 0 diff; Prewitt 119 383 px 0 diff, auto T 0.04899426 / 120 596 px 0 diff, `gv_p`/`gh_p` ≤ 1e-16; `'nothinning'` 518 251 px 0 diff; `'horizontal'` 89 644 / `'vertical'` 97 244 px 0 diff; Roberts T = 0.05 150 194 px 0 diff, auto T 0.04546035 / 158 499 px 0 diff | exact | |
| `morphology.m` (binary block: `strel('dis',7)`, `im2bw(im,graythresh(im))`, `imerode`, `imdilate`, `I−J`, `K−I`, `K−J`) | `ch04_ice_edge_detection.morphological_edges`, `scripts/ch04_morphology.py` | L2 L3 | `im` 0 diff; `level` 0.44313725490196076 bit-identical (t* = 113, em 0.96665); `I` (3 796 066 px) / `J` (3 097 387) / `K` (4 607 838) **0 px differ**; `BW1` (698 679) / `BW2` (811 772) / `BW` (1 510 451) identical, class `double` as MATLAB; `getnhood(SE)` = `strel('dis',7)` 13×13/157 px, `decompose` 6 elements | exact | Eq. 4.42 holds (698 679 + 811 772 = 1 510 451) |
| `morphology.m` (grayscale block: `X`, `Y`, `internal`, `external`, `basic`) | same | L2 L3 | all five uint8 arrays **0 levels differ**; internal mean 12.4166 / max 202, external 13.2581 / 202, basic 25.6747; 5 `imshow` images identical | exact | uint8 saturating minus reproduced (`_matlab_minus`) |
| Figs. 4.3/4.6/4.9/4.10/4.15/4.16 configuration (`im(1600:2151,1979:2552)`, r = 15) (`crop_fig4_3.mat`) | `fig_4_3a`, `FIG_4_3A_CROP`, both scripts on the crop | L2 L3 | `crop_gray` 0 diff (552×574); Sobel T = 0.05 **1736 px, 0 diff**; Prewitt 1737 px 0 diff; LoG σ = 2, T = 0.005 1885 px 0 diff; auto Sobel T 0.0468063 / 1760 px 0 diff; T = 0.03 2456 px 0 diff; r = 15: `nh15` 29×29/697 identical, `I` 91 963 / `J` 67 634 / `K` 119 094 / `BW1` 24 329 / `BW2` 27 131 / `BW` 51 460 0 px, `X`/`Y`/`internal`/`external`/`basic` 0 levels; 15 figure images identical | exact | crop rectangle confirmed by the refinement below |
| — Fig. 4.3(a) crop rectangle (analysis risk 4) | `FIG_4_3A_CROP = (1599:2151, 1978:2552)` | L3 | NCC of the area-averaged crop vs the 131×126 PDF bitmap (stored inverted): **0.99886 at offset (0, 0)**, best in ±2 px and in ±8 px; neighbours ±1 px 0.99804 / 0.99798 / 0.99741 / 0.99666; scale search 0.97…1.03 peaks at 1.00 (0.9752 at 0.99, 0.9698 at 1.01) (`crop_refine.json`) | exact (data mapping) | no refinement needed |
| — `edge` sobel / prewitt / roberts with thinning (`computeEdges`), all directions, given and automatic T (`compat.mat`) | `core.edges.edge`, `gradient_sobel_prewitt`, `gradient_roberts`, `thin_gradient` | L1 L2 | 7 fixtures × {sobel, prewitt} × {both, horizontal, vertical} × {thinning, nothinning} × {auto, T} + roberts both × 2 × 2 = **196 maps, all 0 px differ**, incl. the tie fixtures (Ramp: constant \|bx\| → no interior maxima; RampD/Diag: \|bx\| = \|by\|; Step/StepH: 2-px tie ridges keep the pixel whose right/down neighbour is smaller; `'vertical'` on the zero-padded bottom row keeps the tie column too — MATLAB identical); 26 automatic thresholds rel ≤ 1e-12 (the 16 noise-only cases — constant image, or a direction with no gradient — are < 1e-15 on both sides with all-zero maps); `gv`/`gh` on the 7 fixtures ≤ 1.1e-16 (1 ulp on `Rnd` sobel/prewitt `gv`; 0 for roberts) | exact | the reverse-engineered thinning rule holds at every constructed tie |
| — `edge('log')` / `edge('zerocross', H)` | `core.edges.edge`, `log_zero_crossings` | L1 L2 | `test.jpg`: T = 0.005 117 089 px 0 diff, auto T 0.0015959699 / 245 880 px 0 diff; crop 1885 px 0 diff; `Rnd`: auto (t rel 1e-12), T = 0.02, σ = 1.5 auto, σ = 3, `Rnd8/256` σ = 2 — all 0 px; `Ramp` auto 0 px; zerocross with the 4-/8-neighbour Laplacian on integer plateaus (exact zeros, T = 0.5 / 0 / auto) and with `fspecial('log',7,1)` on `Rnd` — 0 px | exact | rule ported from the readable `edge.m` branch incl. the `b == 0` case |
| — `edge('log')` on piecewise-constant images | same | L2 (negative) | `Step` σ = 1, T = 0.001: MATLAB flags col 5 only, port cols 5 **and 7** (7 extra px); cause proven with MATLAB's own response `b_log_Step`: in the flat region the LoG response is rounding noise (MATLAB −9.5e-17…−6.1e-17, Python +2.7e-17; \|b_py − b_ml\| < 1e-12 everywhere) and the strict `b(r,c+1) > 0` test of the `[− +]` pattern turns a +noise neighbour into a crossing; constant image: MATLAB 14 noise px at T = 4.7e-18, port 0 px at T = 1.6e-17 | near | identical away from pixels adjoining a \|b\| < 1e-12 value; cannot be made bit-exact (MATLAB's `imfilter` summation order); irrelevant for real images (Deviation 2) |
| — `fspecial` (sobel, prewitt, laplacian α ∈ {0, 0.2, 1, default}, gaussian 6 forms, log 5 forms, average 3, disk 4, unsharp 2) | `core.filters.fspecial` | L1 L2 L4 | 26 kernels ≤ 6.9e-18 vs MATLAB; Fig. 4.5(a)/(b) all 25 + 25 printed values reproduced to 4 dp (max 5e-5); Gaussian sums to 1, LoG to 0 (< 1e-15) | exact | |
| — `strel` (disk r = 1…20 × n ∈ {0, 4, 6, 8}, diamond 1…10, square 3/4, rectangle [3 5], line 11 cases, octagon 0/3/6/9, periodicline 3, pair 2, arbitrary 2, `'dis'` / `'DISK'` prefix, `'di'` ambiguous) | `core.morphology.strel`, `periodic_line`, `line_strel`, `intline`, `se_origin` | L1 L2 L4 | **116 neighbourhoods identical** to `getnhood`; `strel('dis',7)` = `strel('disk',7)` = MATLAB default n = 4 (13×13/157 px); MATLAB errors on `'di'` (`amb_err = 1`) and the port raises `ValueError`; Fig. 4.7(c) printed 9×9/69-px matrix = `strel('disk',5)`, Fig. 4.7(d) = `strel('diamond',5)` 11×11/61 px; r = 15 → 29×29/697, r = 16 → 31×31/817, r = 5 → 9×9/69 | exact | Adams' periodic-line decomposition, not a Euclidean disk (r = 5, n = 0 would be 11×11/81) |
| — `disk_decomposition` (= `decompose(strel('disk',r,n))`) | `core.morphology.disk_decomposition`, `minkowski_sum` | L1 L2 | 11 cases (r = 3, 5, 7, 10, 15, 16, 20 with n = 4; r = 7, 12 with n = 6, 8): same number of elements (6 / 8 / 10) and **each element identical in the same order** (r = 7: (5,1), (3,3), (1,5), (3,3), (1,5), (5,1)); Minkowski sum of the sequence = `strel('disk', r)` | exact | |
| — `imerode` / `imdilate` (bool, uint8, double, negative double, int16 × disk 7, asymmetric 3×3, even 2×3, line 5, line 7@45°, square 3, diamond 3, disk 2, pair) | `core.morphology.imerode`, `imdilate` | L1 L2 | **108 arrays 0 px / 0 levels differ**, dtype preserved; decomposed-sequence path = MATLAB (`imerode(Bw, decompose)`, `imdilate(Rnd8, decompose)`); MATLAB itself gives `strel` == `getnhood` results; Fig. 4.8(d)/(f) printed matrices reproduced by the port **and** by MATLAB; literal Eq. (4.20)/(4.21) loops (reflection, +Inf/−Inf, 255/0 pads, even-SE origin) agree | exact | reflection convention of Eq. 4.18/4.21 confirmed with the asymmetric SE (non-reflected dilation is a different image) |
| — `imopen` (same 54 image × SE cases + the 1-D profile) | `core.morphology.imopen` | L1 L2 | 54 arrays 0 diff; profile `po`/`pe`/`pd` identical | exact | MATLAB's `imopen.m` applies `imdilate(imerode(A))` without padding |
| — **`imclose`** (same 54 cases + the 1-D profile + 18 signed/large-SE cases in `imclose_pad.mat`) | `core.morphology.imclose` | L1 L2 | **re-verified: 54 of 54 `compat` cases 0 diff** (previously 43 differed in the `ceil(size(nhood)/2)` border band, e.g. Rnd8/disk7 958 px, Rnd/disk7 1139 px, BwB/disk7 185 px); the 1-D profile `pc` identical; **18 of 18 `imclose_pad` cases 0 diff** (`Neg`, `I16`, `Rnd8` × {diamond 7, disk 8, diamond 8, disk 10, line 31/45°, rectangle [3 17]}); the decomposed-list form `imclose(Neg, disk_decomposition(10))` equals MATLAB and the flat-nhood form. Pad-value rule proven from MATLAB's own outputs on `Neg` (double in [−0.5, 0.5)): 17×17 diamond → equals the zero-padded composition (80 px / 0.48 away from the −Inf-padded one), 19×19 disk → zero-padded (3 px / 0.28 from −Inf-padded), 15×15 diamond → −Inf-padded (80 px from zero-padded); all three are 569–746 px away from the unpadded Eq. (4.22) composition. `images.UseHalide.ActiveValue` = 1 during the reference run | exact | R2025a routing (`morphop_fast.m`): nnz < 600 ∧ every side ≤ 15 ∧ not an all-ones rectangle → Halide kernel, border = class minimum (−Inf / intmin / false); otherwise `imclose.m`, `padarray(A, ceil(size(nhood)/2), 'both')` = zeros. The port reproduces both; they coincide for uint8 / logical / non-negative double (0 is the minimum). Depends on the `images.UseHalide` setting (Deviation 1) |
| — `imreconstruct` (binary 8/4-conn, uint8 8/4/cross, double, 1-D with a line conn) | `core.morphology.imreconstruct` | L1 L2 | 7 reconstructions 0 diff; two-blob fixture returns exactly the marked disc; `reconstruct_iterative` = MATLAB; 1-D `rc_prof_d` identical | exact | MATLAB requires a 3×3 conn (a 1×3 array errors) — the port accepts any neighbourhood (Deviation 5) |
| — reconstruction by erosion (Eqs. 4.28–4.30, 4.35–4.38; no MATLAB builtin) | `core.morphology.reconstruct_by_erosion` | L1 L2 | equals `imcomplement(imreconstruct(imcomplement(marker), imcomplement(mask)))` on the 1-D profile (0 diff); hand truth `max(f, floor)` (water basin + h = 80, crack 100, pit 130) reproduced | exact | verified through MATLAB's complement construction |
| — geodesic dilation/erosion, literal iteration (Eqs. 4.25–4.27, 4.32–4.34) | `geodesic_dilation`, `geodesic_erosion`, `reconstruct_iterative` | L1 L2 | `D^(2) = D^(1)[D^(1)]`; two-blob fixture converges in k = 11; iteration = `imreconstruct` = MATLAB on uint8 8-conn; 1-D hand truth `min(f, cap)` (domes cut by h = 40) reproduced, k = 46 / 37 | reimplemented | teaching forms |
| — morphological gradients (Eqs. 4.39–4.42) and MATLAB minus | `core.morphology.morphological_gradient`, `_matlab_minus` | L1 L2 | bool → `double` 0/1 (MATLAB class confirmed), uint8 (saturating), double, int16: 8 gradient arrays 0 diff; `A8 − B8` / `B8 − A8` uint8 saturation identical; disc fixture: internal ring inside, external outside, widths = r, Eq. 4.42 exact | exact | |
| Eqs. (4.1)–(4.8) gradient vector / magnitude (L2, squared, L1) / direction / threshold, Fig. 4.1–4.2 kernels | `gradient_operator`, `gradient_magnitude`, `gradient_direction`, `threshold_gradient` | L1 L2 | vs hand-coded MATLAB `imfilter` with the printed kernels: `Gx_s`/`Gy_s` 8.9e-16, Prewitt/forward ≤ 1e-15, zero-padding variant ≤ 1e-15; magnitudes/directions ≤ 1e-15; `Gx_book = −8·gh`, `Gy_book = −8·gv` (−6 Prewitt) to 1e-14; Eq. 4.6 differences exact | reimplemented | Eq. 4.3 printed as `arctan(Gx/Gy)` → `convention='book'|'math'` both checked vs MATLAB `atan2` |
| Eqs. (4.9)–(4.13) Laplacian, Fig. 4.4 kernels | `laplacian`, `second_difference_forward`, `LAPLACIAN_KERNELS` | L1 L2 | `Lap4` 8.9e-16, `Lap8` 2.7e-15, `d2x`/`d2y` ≤ 1e-15 vs MATLAB; x²+y² gives 4 (4-neighbour) / 12 (8-neighbour) in the interior; `fspecial('laplacian',0)` = Fig. 4.4(a) | reimplemented | |
| p. 64 zero-crossing rule (3×3 max/min difference > T) | `laplacian_zero_crossings` | L1 L2 | identical to the hand-coded MATLAB rule (`zc_text`, `imdilate(max(L,0),ones(3))` / `−imdilate(−min(L,0))`); step: edges hug both sides | reimplemented | different from MATLAB `edge('log')`'s rule (documented) |
| Eqs. (4.14)–(4.15) Gaussian / LoG kernels, Fig. 4.5 | `gaussian_kernel`, `log_kernel` | L1 L2 L4 | analytic samplings 0.0 vs MATLAB (`Gau_an`, `LoG_an`), normalised Gaussian 0.0, `gauss_conv_laplacian` 0.0; `log_kernel('matlab')` = `fspecial('log')` = Fig. 4.5(b); analytic centre −1/π, Fig. 4.5(b) centre −0.3182 = `fspecial` (mean-subtracted), 7×7 unnormalised Gaussian sums to > 0.99 ("3σ covers > 99 %") | reimplemented (analytic) / exact (`matlab` mode) | |
| `bwareaopen` (line 8) | `ch04_ice_edge_detection.bwareaopen` | L1 L2 | P = 20 (8-conn), P = 20 4-conn, P = 5 on a noisy binary: 0 px; `n_ao` 982 identical; L1 3-px diagonal chain 8- vs 4-conn | exact | built on `label_components` (= `bwlabel`) |
| `medfilt2`, `conv2 'full'/'same'/'valid'` (lines 6, 15) | `scipy.ndimage.median_filter(mode='constant')`, `core.filters.conv2` | L2 | `med_Rnd`, `med_Rnd8` (zero-padded) and `med_Rnd_sym` (`'symmetric'` = scipy `reflect`) identical; `cf`/`cs`/`cv` identical | exact | |
| Fig. 4.8 matrices, 1-D profile, two-blob fixture | `core.synth.FIG_4_8_*`, `two_floes_profile`, `two_blobs_with_marker` | L1 L2 | printed (a)/(b)/(d)/(f) transcribed; MATLAB `imerode`/`imdilate` of (a) by (b) equal the printed (d)/(f) (21 → 5 px, 41 px); profile/blob fixtures deterministic, used on both sides | exact (printed truth) / reimplemented (fixtures) | |
| Figs. 4.11–4.14 1-D demos, Fig. 4.7 SE panel, Fig. 4.8 walk-through | `profile_open_close_demo`, `profile_reconstruction_demo`, `fig_4_8_demo`, script demos | L1 L2 L3 | opening removes the 2-px speck, keeps both floes; closing fills the 3-px crack (→ 170) and the pit (→ 200); anti-/extensive and idempotent; all four profile outputs identical to MATLAB `imopen`/`imclose`/`imerode`/`imdilate` | exact | |
| §4.3 experiments (Figs. 4.17–4.20: Sobel T = 0.05/0.03, internal gradient r = 5/16 on a connected-floe crop) | `experiment_sobel_thresholds`, `experiment_internal_gradient`, `scripts/ch04_experiments.py` | L3 (qualitative) | source image not shipped; on `test.jpg`: Sobel 0.05 → 120 224 px / 0.03 → 162 994 px (both = MATLAB), internal gradient r = 5 (9×9/69) 461 937 px, r = 16 (31×31/817) 1 451 506 px, crop 300:1100,50:950: r = 5 leaves 48 interior components, r = 16 28; the components (`edge`, `strel`, `imerode`, gradient) are exact vs MATLAB | unverified (book figures) | Open item 2 |
| `edge('canny')` | not ported (`NotImplementedError` pointing at `skimage.feature.canny`) | L1 | raises as documented | — (not in the book) | Deviation 6 |

## Figures reproduced
Compare PNGs: `outputs/ch04/verify/<name>_compare.png` (copies in `reports/ch04/figures/`). Layout: Python row |
MATLAB `imwrite` row (when the `.m` code produces the figure) | book page rendered from `chapters/ch04.pdf`.
`image_diffs.json`: **all 27 raw image pairs (Fig. 4.3(a)(b)(c), 4.6, 4.9(a–c), 4.10(a)(b), 4.15(a–c), 4.16(a–c) on the crop;
`derivative.m`'s BW and the 11 `morphology.m` images on the full 2856×4290 frame) are identical, 0 px differ each.**

| Figure | File | Verdict |
|---|---|---|
| 4.3 (p. 63) | `fig_4_03_compare.png` | The crop shows the same two floes as the book's (a); Sobel and Prewitt T = 0.05 edge maps are pixel-identical to MATLAB and match the book's (b)/(c) (closed outer contours, the same short spur on the right floe's upper edge). |
| 4.6 (p. 66) | `fig_4_06_compare.png` | LoG 13×13, σ = 2, T = 0.005 map identical to MATLAB and to the book's Fig. 4.6 (thicker, noisier contours with speckle inside the right floe). |
| 4.7 (p. 67) | `fig_4_07_compare.png` | Text-only: the four SEs; (c) is the printed 9×9/69-px disk and (d) the 11×11 diamond (both asserted); the panel also shows `strel('dis',7)` and the Euclidean disk for contrast. |
| 4.8 (p. 70) | `fig_4_08_compare.png` | Text-only: the 11×15 rectangle, cross SE, and the erosion (one row of 5) / dilation (5 rows × 9 cols, corners clipped) results equal the printed matrices. |
| 4.9 (p. 71) | `fig_4_09_compare.png` | Binarised crop, erosion and dilation with the r = 15 disk identical to MATLAB and matching the book's three panels (rounded, shrunken vs. grown floes). |
| 4.10 (p. 71) | `fig_4_10_compare.png` | Grayscale erosion/dilation identical to MATLAB; the dark-blob "bubbles" inside the floes in the book's (a) and the smooth swollen floes of (b) are reproduced. |
| 4.15 (p. 79) | `fig_4_15_compare.png` | Basic / internal / external binary gradients identical to MATLAB and matching the book (the caption's "157" radius is a typo for 15; the r = 15 rings are exactly the book's thickness). |
| 4.16 (p. 79) | `fig_4_16_compare.png` | Grayscale gradients identical to MATLAB; the bright rims with faint interior texture match the book's (a)–(c). |
| 4.11–4.12 (pp. 72–73) | `fig_4_11_4_12_compare.png` | Text-only sketches: the synthetic profile's closing pushes down from above (crack/pit filled) and opening pushes up from below (speck removed), as in the book's curves. |
| 4.13–4.14 (pp. 75–76) | `fig_4_13_4_14_compare.png` | Text-only: reconstruction by dilation cuts the domes by h, by erosion fills the valleys by h; iteration counts 46 / 37. |
| `derivative.m` full frame | `sec_4_1_derivative_full_compare.png` | The shipped script's `imshow(BW)` on the full image: identical to MATLAB (120 224 px). |
| `morphology.m` full frame | `sec_4_2_morphology_full_compare.png`, `sec_4_2_morphology_full_gray_compare.png` | All 11 displayed images identical to MATLAB (r = 7 on the full frame, as the script). |
| 4.17 (p. 80) | `fig_4_17_compare.png` | **Unshipped image**: the procedure on `test.jpg` gives the same qualitative picture (Sobel and internal gradient both outline isolated floes); the book's pack-ice scene itself is not reproducible. |
| 4.18–4.20 (pp. 81–82) | `fig_4_18_compare.png`, `fig_4_19_compare.png`, `fig_4_20_compare.png` | **Unshipped image**: on a touching-floe crop of `test.jpg`, T = 0.03 adds weak edges and noise, r = 16 breaks weak connections but thickens the rims — the behaviour the text describes; pixels not comparable. |

## Numbers from the text (book vs ours)
| Item | Page | Book | Ours (Python) | MATLAB (original .m) | Status |
|---|---|---|---|---|---|
| Fig. 4.5(a) Gaussian 5×5, σ = 1 | 66 | 0.0030 0.0133 0.0219 … 0.1621 (25 values) | all 25 values equal to 4 dp (max dev 5e-5) | `fspecial('gaussian',5,1)` identical to 1e-18 | match |
| Fig. 4.5(b) LoG 5×5, σ = 1 | 66 | 0.0239 0.0460 0.0499 0.0061 −0.0923 −0.3182 | all 25 values equal to 4 dp | `fspecial('log',5,1)` identical | match |
| Fig. 4.7(c) "disk with 5-pixel radius" | 67 | printed 9×9 matrix, 69 ones | `strel('disk',5)` identical (n = 4) | `getnhood(strel('disk',5))` identical | match (a Euclidean disk would be 11×11/81) |
| Fig. 4.7(d) diamond 5 | 67 | 11×11, extreme points at distance 5 | identical, 61 px | identical | match |
| Fig. 4.8(d)/(f) erosion / dilation results | 70 | 5 px row; 41-px rounded rectangle | identical | identical | match |
| T = 0.05 (Sobel/Prewitt, Figs. 4.3, 4.17–4.19) | 63, 80, 81 | 0.05 | 1736 / 1737 edge px on the crop (98.3 % overlap: "similar results") | identical | match |
| 13×13 LoG kernel, σ = 2, T = 0.005 (Fig. 4.6) | 65–66 | 13×13 | `2·ceil(3σ)+1 = 13`; 1885 edge px | identical | match |
| "3σ … covers more than 99 %" | 65 | > 99 % | 7×7 σ = 1 unnormalised Gaussian sum 0.9946 (2-D square) | — | match |
| r = 15 disk (Figs. 4.9, 4.10, 4.15, 4.16) | 69, 71, 78, 79 | 15-pixel radius | 29×29 / 697 px; the r = 15 results reproduce the figures | identical | match |
| Fig. 4.15 caption "157-pixel-radius" | 79 | 157 | typo for 15 (a 157-px radius disk is 313×313, larger than the 552×574 crop allows; 157 is the pixel count of the script's `strel('dis',7)`) | — | typo documented |
| r = 5 / r = 16 disks (Figs. 4.17(c), 4.18(c), 4.20) | 80–82 | 5, 16 | 9×9/69 px, 31×31/817 px | identical nhoods | match (image unverified) |
| T = 0.03 (Fig. 4.19(b)) | 81 | 0.03 | 162 994 px on `test.jpg` (vs 120 224 at 0.05) | identical | match (image unverified) |
| `morphology.m` script values (not printed in the book) | — | — | t* = 113, IC 30.98 %, J 3 097 387, K 4 607 838, BW1 698 679, BW2 811 772, BW 1 510 451 | identical | match |

## Deviations & justifications
1. **`imclose` border rule** (resolved; was Open item 1): the book's Eq. (4.22) states no border convention, and MATLAB's
   `imclose` is *not* the bare `imerode(imdilate(A, se), se)` — it pre-pads by `ceil(size(getnhood(se))/2)` and crops, so
   dark notches that open onto the image border are not closed. The port now does the same, with the pad value chosen by
   R2025a's routing: the Halide kernel (SE nnz < 600, every side ≤ 15, not an all-ones rectangle) uses the class minimum
   (−Inf / intmin / false), `imclose.m` (everything else) uses 0. For every book image class (uint8, logical, [0,1] double)
   the two values coincide; they differ only for signed images with negative values, which the book never closes. The
   Halide route is gated by the MATLAB setting `images.UseHalide` (= 1 in the reference run, saved as `halide_on`); with
   it off MATLAB would zero-pad everything — the port follows the default. Verified: 54 + 18 cases 0 px, the padded /
   −Inf-padded / unpadded discrimination above (`test_which_pad_value_matlab_used`, `test_imclose_padding_reference_identity`).
2. **`edge('log')` on piecewise-constant images** (`near`): MATLAB's rule tests `b > 0` / `b < 0` strictly, so at pixels where
   the LoG response is rounding noise (flat regions next to a step) the result depends on the sign of ~1e-17 values that
   differ between MATLAB's `imfilter` and scipy's correlate. Exact on `test.jpg`, the crop, random fields and integer-Laplacian
   plateaus (whose zeros are exact on both sides); the 7 extra pixels on the synthetic `Step` and the 14-vs-0 noise pixels
   on a constant image are recorded in `test_log_step_noise_sensitivity` / `test_log_constant_image_is_floating_noise`.
3. **Float rounding in reimplemented filters**: `gv` 1 ulp (1.1e-16) on `Rnd` for sobel/prewitt (the builtin filters with the
   integer kernel then divides; the port does the same but scipy's accumulation order differs by one rounding); 8-neighbour
   Laplacian 2.7e-15 on values of magnitude ~3. Within the `exact` vocabulary (≤ 1e-6).
4. **Binary minus → float64**: `I − J` etc. return float64 0/1 (MATLAB: logical − logical = double), documented in
   `morphological_gradient`; the scripts save them as 0/255 PNGs identical to MATLAB's `imwrite`.
5. **`imreconstruct` connectivity argument**: MATLAB requires a 3×3(×3…) `conn`; the port also accepts any neighbourhood
   (the 1-D demo passes `strel('line',3,0)`, a 1×3 array) — a superset, verified equal to MATLAB's `[0 0 0; 1 1 1; 0 0 0]`.
6. **Canny** is not ported (raises `NotImplementedError`); the chapter never uses it.
7. **`'vertical'` / `'horizontal'` thinning on the zero-padded last row/column** keeps a tie pixel that `'both'` drops
   (`edge(Step,'sobel',0.1,'vertical')` → 10 px, not 9): this is MATLAB's `computeEdges` behaviour, confirmed at L2, not a
   deviation.

## Open items
1. **Resolved — `imclose` border rule** (was: 43 failing `TestL2ErodeDilate::test_imclose[*]`, up to 1139 of 6144 px on a
   64×96 double field with `strel('disk',7)`, 185 px on a 20×24 binary object 2 px from the border). After the porter's fix
   all 54 `compat` closings and the 18 new `imclose_pad` closings are **0 px** from MATLAB R2025a, the identity test now
   asserts port == MATLAB == padded form ≠ unpadded form, and `test_which_pad_value_matlab_used` pins the zero-pad /
   class-minimum routing on signed images (80 / 3 / 80 px discriminations). No open action.
2. **Figs. 4.17–4.20 `unverified`**: the §4.3 pack-ice image (Fig. 4.17(a)) and its connected-floe crop (Fig. 4.18(a)) are not
   shipped and have no public source (full-frame correlation with the three shipped JPEGs ≤ 0.24 in the analysis); only the
   procedure is reproduced on `test.jpg`, with every component exact vs MATLAB. Needs the source image from the user; not a
   port failure.
3. **LoG zero-crossing noise sensitivity** (Deviation 2): the port cannot reproduce MATLAB's rounding-noise signs on flat
   regions, so `edge('log')` is `near` on piecewise-constant synthetic inputs (exact on all real images tested). Documented;
   the notebook should say the LoG threshold applies to the *jump* across the crossing and that flat synthetic patches are
   noise-sensitive in MATLAB too (14 spurious pixels on a constant image).
4. **Note for ch5–ch9**: `strel('disk', r)` neighbourhoods and decompositions, `imerode`/`imdilate`/`imopen`/`imclose`,
   `imreconstruct` and `edge` (all methods but Canny) are bit-exact against MATLAB R2025a on 12-Mpx JPEGs and on
   asymmetric / even / signed-image / large-SE edge cases, so later chapters' watershed markers, GVF edge maps and the ch7
   `imopen`/`imclose` shape enhancement can be compared with 0-pixel tolerance.

## Verdict: PASS
Both `.m` files run verbatim in MATLAB R2025a and every output of the two script ports is identical to MATLAB
(`derivative.m` BW and 11 `edge` variants 0 px on 12.25 Mpx; `morphology.m` all 12 arrays 0 px / 0 levels; the r = 15 crop
configuration of Figs. 4.3/4.6/4.9/4.10/4.15/4.16 0 px; all 27 figure images identical; `edge` 196 + 11 gradient maps and
14 LoG/zerocross maps 0 px; 116 `strel` neighbourhoods, 11 decompositions, 108 erosions/dilations, 54 openings, **54 + 18
closings**, 7 reconstructions, 8 gradients identical; every quoted kernel/matrix/parameter reproduced), all three scripts run,
every `.m` has a row, and the full suite passes (683 passed, 1 xfailed; ch04 485 passed). The former Open item 1 (`imclose`
border rule) is resolved with 0-px parity on 72 cases and the pad-value routing pinned against MATLAB's own outputs. The
single `unverified` row (Figs. 4.17–4.20, unshipped source image) is covered by Open item 2 and is not a port failure.
