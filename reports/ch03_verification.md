# Chapter 3 verification — Ice Pixel Detection

Date 2026-09-09 (re-verified the same day after the porter's fixes) · port commit `f24a7a1` (`ch03: port — threshold.py
(graythresh/multithresh/block_otsu/separability), clustering.py (kmeans_gray, kmeans_lloyd), 6 scripts`) + the porter's
uncommitted fix of `seaice/ch03_ice_pixel_detection._num2str`, `seaice/core/threshold._multithresh_pdf` and
`otsu_criterion` (working tree at verification time) · verifier artefacts: `tests/test_ch03.py`,
`reference/ch03/make_refs.py` (+ 13 `.mat`, `refs_log.json`), `reference/ch03/make_compare_figures.py`,
`outputs/ch03/verify/` (MATLAB `imwrite`/`print` images in `matlab/`, patched script copies in `scratch/`, compare PNGs,
`image_diffs.json`, `report_numbers.txt`, pytest logs incl. `pytest_full_reverify.txt`, script logs `run_ch03_*.log`),
`reports/ch03/figures/` (12 compare PNGs, git-ignored, regenerated after the fixes). The `.mat` references were **not**
regenerated (the fixes are Python-side only).

## Environment
python 3.11.5, numpy 2.4.6, scipy 1.17.1, scikit-image 0.26.0, matplotlib 3.11.1 |
reference engine: **MATLAB 25.1.0.2833191 (R2025a) Prerelease Update 2** via `tools/run_matlab_ref.py` (`matlab -batch`,
figures invisible) — all 13 reference files (`reference/ch03/refs_log.json`: engine `matlab`, status `ok`). Octave absent;
no fallback used.

How the original code was run: `Otsu.m`, `kmeans.m`, `local_Otsu.m`, `separability.m` hard-code `imread('test.jpg')` /
`k = 3` / `k = 108` and read from the current folder, and `kmeans.m` shadows the Statistics Toolbox function. Each script
was therefore copied **verbatim** into `outputs/ch03/verify/scratch/` under a non-shadowing name and only the literal
`imread('…')` / `k = …` tokens were patched (`kmeans.m` additionally got an `n_iter` counter); the copies ran with cwd =
scratch, which holds copies of `test.jpg`, `1.jpg`, `2.jpg`, plus the two **substitutes** for the unshipped inputs
(`ch3ice.jpg` = `2.jpg`; `t.jpg` = `data/synthetic/ch03/t_uneven_2_g0.5_b40.jpg`, the ramp written by
`scripts/ch03_local_otsu.py`). MATLAB_ROOT was never modified nor put on the MATLAB path. `compat.mat` calls the toolbox
functions (`graythresh`, `otsuthresh`, `im2bw`, `multithresh`, `imquantize`, `im2uint8`) on controlled arrays saved from
Python (`inputs.mat`) and hand-codes Eqs. (3.3)–(3.22) in MATLAB.

## pytest
`.venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider` (full suite, ch02 + ch03, after the porter's fixes and
the two test updates) → **`194 passed, 1 xfailed, 0 failed in 480 s`** (`outputs/ch03/verify/pytest_full_reverify.txt`,
exit 0): all 86 ch02 tests pass; `tests/test_ch03.py` contributes **108 passed, 1 xfailed** (109 tests). First-pass
result for the record (`pytest_full.txt`): `3 failed, 190 passed, 2 xfailed` — the three failures and one of the xfails
were the port defects fixed since (table below).

Parity label counts (23 rows below): **exact 16 · near 0 · approx 1 · reimplemented 5 · unverified 1**
(first pass: exact 12 · near 3 · approx 2 · reimplemented 5 · unverified 1).

Resolved since the first pass (the three former failing tests now pass; the references were unchanged):

| former failure | now measured |
|---|---|
| `TestL1Chapter::test_local_otsu_titles_num2str`, `TestL2LocalOtsu::test_subplot_titles_num2str` | `_num2str` on the ten MATLAB samples → `73.8472|86.496|3.1416|0.5|107.5|92|20.9869|123.4567|0.12346|1234.5679`, byte-identical to `local_otsu.mat: n2s`; the six Fig. 3.4(c) titles `IC=29.3396% … 26.8086%` / `Threshold=153 … 58` identical to MATLAB's `tstr`/`thstr` |
| `TestL2Otsu::test_multithresh_metric_strict[test]` | Eq. (3.28) metric on test.jpg 0.8316373381 = MATLAB (abs diff 0.0); 1.jpg 0.9903292707 (1e-16), 2.jpg 0.9840568531 (0.0); `_multithresh_pdf` now equals the test-side single-precision emulation bin for bin on all three images (`moved.size == 0`); N = 3 metric on test.jpg 0.9059957850 = MATLAB's 0.9059957850 (was 0.90577) |
| `test_otsu_criterion_empty_class_tail_matches_matlab` (was strict xfail) | NaN pattern of `m1` and `σ_B²` identical to MATLAB's hand-coded loops on 2.jpg (38 NaN tail entries); σ1² of the one-pixel class (t = 215) = 0 exactly (was 6.6e-4); over the finite range σ1² ≤ 5.6e-11, m1 ≤ 1.4e-13; `t_star` 107 / `eta_star` 0.9674474800 unchanged |

Test-side changes for the re-verification: dropped the xfail marker on `test_otsu_criterion_empty_class_tail_matches_matlab`
and changed `test_multithresh_metric_single_precision_hypothesis` to require `moved.size == 0` for every image (it had
encoded the old bins-212/213 discrepancy); no tolerance was loosened.

xfailed (strict, documenting a known divergence): `test_multithresh_int16_known_divergence` (int16 saturation in
`multithresh.m` line 274, not a book case — Open item 2).

All six `scripts/ch03_*.py` ran headless with exit 0 in the pytest run (`test_script_runs[*]`, 6/6) and again after the
fixes into `outputs/ch03/` (`outputs/ch03/verify/run_ch03_*.log`); `reference/ch03/make_compare_figures.py` was rerun —
all 14 raw image pairs are still 0 px different (`image_diffs.json`).

## Parity table
Levels: L1 synthetic truth · L2 MATLAB reference · L3 figure · L4 quoted number. Errors are max-abs unless stated.
Every `.m` in `MATLAB_ROOT/ch3` (`Otsu.m`, `local_Otsu.m`, `separability.m`, `kmeans.m`) has a row; the remaining rows
are the new `seaice/core` primitives and the text-only algorithms of `analysis/ch03.md` §3.

| MATLAB file / function | Python | Evidence | Result (measured) | Parity | Notes |
|---|---|---|---|---|---|
| `Otsu.m` lines 2–14 (`graythresh`, `im2bw`, IC) | `ch03_ice_pixel_detection.otsu_segmentation` | L2 L3 L4 | test/1/2.jpg: `level` bit-identical (t* = 182 / 123 / 107), `em` ≤ 1e-12 (0.527667 / 0.983463 / 0.967447), `bw` **0 px differ** (3 × 12 252 240 px), `ic` ≤ 1e-12 (72.6325 / 15.3616 / 32.0494 %) | exact | JPEG decode + `rgb2gray`: 0 px differ from MATLAB's `I` on all three images |
| `Otsu.m` lines 28–43 (`multithresh(I,2)`, `imquantize`, coverage, mean intensity) | `multi_otsu_segmentation` | L2 L3 L4 | thresholds identical (120/197, 78/156, 61/142 — the analysis pre-check's 79 was wrong, MATLAB says 78); `seg` 0 px differ; `coverage` ≤ 1e-12; `metric` ≤ 1e-12 on all three images (0.8316373381 / 0.9903292707 / 0.9840568531); `average_intensity`: correct means equal MATLAB's on test.jpg (63.8908 / 177.0690 / 218.1751), on 1.jpg/2.jpg MATLAB prints 46.39 / **4759.10** / 418.54 and 27.08 / **611.63** / 219.88 (stale `s` buffer) which `average_intensity_script` reproduces to 1e-9 | exact | the former 1.1e-4 metric gap on test.jpg was the float64 `x·255` product in `grayto8`; fixed (single-precision product), references unchanged |
| — `graythresh` / `otsuthresh` | `core.threshold.graythresh`, `otsuthresh` | L1 L2 | compat: constant, two-valued, tie ({100,110} → 104.5/255), uint16, double, double·255 (clipped), ramp 5..250, 4-valued, bimodal, int16 — `level` bit-identical, `em` ≤ 9e-16; `otsuthresh` on `imhist` and on a 10-bin hand histogram identical; 3 book images identical | exact | tie averaging (`mean(find(...))`) reproduced |
| — `im2bw` | `core.threshold.im2bw` | L1 L2 | `nnz(im2bw(Rnd, i/255))` for all 256 levels identical; uint8 tie level, uint16, double, int16, RGB input, default 0.5: 0 px differ | exact | strict `>` confirmed |
| — `multithresh`, N ≤ 2 | `core.threshold.multithresh` | L1 L2 | thresholds identical for uint8 (ramp 5..250: 127 / 86,167; bimodal 124 / 59,124; 4-valued 104 / 104,194; tie 105), uint16 (32877 / 22096,44428), double (0.49341568 / 0.30942886,0.63825637), degenerate (two-valued N=2 → 0,255; constant → 77 / 1,77 with metric 0); metric ≤ 1e-12 on all of them and on test.jpg/1.jpg/2.jpg; `_multithresh_pdf` equals the single-precision emulation bin for bin on the three JPEGs | exact | dtype of the output matches MATLAB (uint8/uint16/double); `grayto8` emulated with the `x·255` product in single |
| — `multithresh`, N = 3 | `core.threshold.multithresh` (exhaustive Eq. 3.27) | L1 L2 | ramp: identical (66,127,188; metric 0.9379000214); 2.jpg: identical (56,125,179); test.jpg: identical (113,182,211), metric 0.9059957850 identical; 1.jpg: **49,86,160 vs MATLAB 50,101,165** — fminsearch local optimum, port's criterion 0.9940612 > MATLAB's 0.9938168; bimodal image: 59,124,190 vs 59,135,190 on an empty-histogram plateau (identical σ_B²); 4-valued image: MATLAB returns its initial guess 68,125,183 with metric **−Inf**, port 35,104,194 with metric 1 | reimplemented | never `exact` by design (analysis risk 5); the port's σ_B² is ≥ MATLAB's in every case |
| — `multithresh`, int16 input | `core.threshold.multithresh` | L2 (negative) | MATLAB −1157, 31611 vs port −16963, 16705 | approx (int16 only) | `multithresh.m` line 274 `single(A − minA)` saturates in int16 arithmetic; int16 images never occur in the book (xfail test) |
| — `imquantize` | `core.threshold.imquantize` | L1 L2 | ramp / double / 1 and 3 thresholds / with `values`: identical; `seg` and `imquantize(I, thresh, [0 128 255])` on the 3 images: 0 px differ | exact | |
| — `im2uint8` for uint16/int16 (`_im2uint8_any`) | `core.threshold._im2uint8_any` | L2 | all 65 536 uint16 values identical to MATLAB `im2uint8(uint16(0:65535))` (= round(v/257)); int16 sample identical | exact | |
| `local_Otsu.m` (2×3 block Otsu) | `local_otsu`, `core.threshold.block_otsu` | L1 L2 L3 | on the substitute `t.jpg` (2856×4290): `thresh` 153/106/57/158/112/58 identical, `num` identical (599127 … 547442), `IC_local` ≤ 1e-12, `IC` 31.5895 % identical, block-wise `bw` 0 px, global Otsu of the same image (t = 122, IC 27.5232 %) identical; L1 two-half synthetic image exact | exact | Fig. 3.4 book numbers `unverified` (image not shipped) |
| `local_Otsu.m` subplot titles (`num2str`) | `local_otsu()["titles"]`, `_num2str` | L1 L2 | ten `num2str` sample strings (`local_otsu.mat: n2s`) byte-identical; all six `IC=…%` / `Threshold=…` titles identical to MATLAB's `tstr`/`thstr` | exact | MATLAB's `max(floor(log10|x|)+5, 5)` significant-digit rule implemented; integers print as `%d`, NaN/Inf handled |
| `separability.m` (k = 108) | `separability_script` | L2 L3 | on the substitute (2.jpg): `bw` 0 px, `IC` 32.0158 % ≤ 1e-12, `p` ≤ 1e-15, `mg` 80.470193, `sigma2_g` 5613.902205, `sigma2_b` 5431.155541, `eta` 0.9674474800 ≤ 1e-12; second run k = 107: `eta` 0.9674317609, IC 32.0494 % identical | exact | the script's `eta` equals book-convention η(k − 1) (`separability(I, 107)` = 0.9674474800 = MATLAB's `eta` for k = 108) — off-by-one confirmed |
| — Eqs. (3.3)–(3.22) curves (`otsu_criterion`) | `core.threshold.otsu_criterion` | L1 L2 (hand-coded MATLAB loops on 2.jpg) | where both classes are non-empty (t = 12..217): P0/P1 1.3e-15, m 2.8e-14, m0 8.5e-14, σ0² 9.1e-12, σ_W² 5.5e-11, σ_B² (both Eq. 3.16 forms) 4.6e-11, η 8.2e-15; mG and σ_G² identical; `t_star` 107 = `graythresh`, `eta_star` = `em` ≤ 1e-12; identities Eqs. 3.10/3.11/3.18 hold to 1e-9 (L1). C1 moments now from reverse cumulative sums: m1 ≤ 1.4e-13 and σ1² ≤ 5.6e-11 wherever MATLAB is finite, σ1² of the one-pixel class (t = 215) exactly 0, and `P1 == 0` → m1/σ1²/σ_B²/η NaN exactly where MATLAB's direct sums are NaN (38 tail entries, t ≥ 218) | exact | `t_star`/`eta_star` still use the `(mG P0 − m)²/(P0(1 − P0))` form of `otsuthresh` |
| — `separability(gray, t)` | `core.threshold.separability` | L1 L2 | = `graythresh` `em` at t*; η(125) = 0.9650029212 identical to MATLAB's hand-coded value; bounds 0 (constant) / 1 (two-valued) | exact | |
| `kmeans.m` (k = 3, and k = 2) | `kmeans_segmentation`, `core.clustering.kmeans_gray(shift_bug=True)` | L2 L3 L4 | 6 runs (test/1/2 × k = 2/3): `mu` **max diff 0.0**, `n_iter` identical (4/10, 2/7, 4/8), `mask` **0 px differ**, `n`/`ic` identical, `average_intensity` = MATLAB's (stale `ss` emulation) to 1e-9 — e.g. 1.jpg k=2 MATLAB 46.52 / **408.06** (correct 198.39); histogram `h` identical | exact | |
| — `kmeans_gray(shift_bug=False)` (Eqs. 3.36–3.37 in consistent units) | `core.clustering.kmeans_gray` | L1 L2 (negative, by design) | same centroids as MATLAB; mask differs from MATLAB in 1.14 % (test k=2), 24.14 % (test k=3), 0.29 % / 8.58 % (1.jpg), 0.45 % / 1.35 % (2.jpg) of the pixels, all of them at gray levels in `(mid, mid + min − 1]` and moved one class down — exactly the units bug of lines 64–70; correct k=2 IC 95.36 / 15.36 / 32.05 % vs book 96.50 / 15.65 / 32.49 % | reimplemented (documented deviation) | default of the library; book numbers need `shift_bug=True` |
| — Eqs. (3.29)–(3.34) distances | `core.clustering.pairwise_distance` | L1 | (0,0)–(3,4): 5, 25, 4, 7; cosine distance 1 − cos (orthogonal → 1, parallel → 0); Mahalanobis with S = I = Euclidean | exact (closed forms) | cosine = 1 − aᵀb/(|a||b|) (deviation 4) |
| — Eqs. (3.35)–(3.37) Lloyd k-means (Figs. 3.6, 3.8) | `core.clustering.kmeans_lloyd`, `objective_J` | L1 | seeded blobs → centres within 0.6 of (2.5,2.5)/(6.5,6.5), perfect partition, J non-increasing, converged; `'equal'`/`'kmeans++'`/array init; on a 1-D image with `kmeans.m`'s equal-division start it reproduces `kmeans_gray` centroids (1e-9) and mask exactly | reimplemented (text only) | |
| — Eq. (3.1) fixed T, Fig. 3.2(b) histogram marker | `fixed_threshold`, `plot_histogram_with_threshold` | L1 | `bw = I > T`; plot has the T line and 256-bin axis | exact (definition) | |
| — IC bookkeeping | `core.threshold.ice_concentration`, `class_coverage`, `class_mean_intensity`, `stale_mean_intensity` | L1 L2 | hand values; stale emulation on a 2×3 example (column-major fill) and vs MATLAB on all 9 script runs | exact | |
| — Fig. 3.4(a) ramp, Figs. 3.6/3.8 points, Fig. 3.1 image | `core.synth.uneven_illumination`, `two_clusters_2d`, `bimodal_image` | L1 | dtype/shape/monotone ramp (190 → 10), outlier appended, deterministic | reimplemented (synthetic fixtures) | |
| — Figs. 3.6 / 3.8 demo | `kmeans_demo_2d`, `scripts/ch03_kmeans_demo_2d.py` | L1 L3 | converged, six distances, outlier isolated in its own cluster | reimplemented (text only) | |
| Book numbers of Figs. 3.2–3.5, 3.7 (`ch3ice.jpg`, `t.jpg`) | — | — | source image not shipped (OMAE-2012 paper image); substitute values listed below | unverified | Open item 4 |

## Figures reproduced
Compare PNGs: `outputs/ch03/verify/<name>_compare.png` (copies in `reports/ch03/figures/`). Layout: Python row |
MATLAB `imwrite`/`print` row (when the `.m` code produces the figure) | book page rendered from `chapters/ch03.pdf`.
`image_diffs.json`: **all 14 raw image pairs (Otsu masks ×3, multi-Otsu class images ×3, k-means masks ×5, block-Otsu
input and mask, separability mask) are identical, 0 of 12 252 240 px differ each.**

| Figure | File | Verdict |
|---|---|---|
| 3.9 (p. 53) | `fig_3_09_compare.png` | Otsu and k=2 masks of image 1 are pixel-identical between Python and MATLAB and match the book's (b)/(c) panels (long floes, dark water); the book renders the k=2 mask black/white whereas `mask1 = i/k` is gray/white in both engines, and MATLAB's `colormap('default')` shows it in parula. |
| 3.10 (p. 54) | `fig_3_10_compare.png` | Scattered-floe masks identical (0 px) and matching the book's Fig. 3.10(b)/(c). |
| 3.11 (p. 55) | `fig_3_11_compare.png` | Otsu (72.63 %) picks only bright ice, k=2 (96.50 %) leaves only the dark leads on the left — identical to MATLAB and to the book's (b)/(c). |
| 3.12 (p. 57) | `fig_3_12_compare.png` | Three-level multi-Otsu image and k=3 mask identical to MATLAB (0 px) and match Fig. 3.12(a)/(b); the gray-level display (not parula) is what the book shows. |
| 3.3 (p. 43) | `fig_3_03_compare.png` | **Substitute** (2.jpg): `I > 108` mask identical to MATLAB's `separability.m` output; the book's image (pack-ice edge) is a different scene. |
| 3.4 (p. 45) | `fig_3_04_compare.png` | **Substitute** (2.jpg + synthetic ramp): input, global-Otsu mask, block mask identical to MATLAB (0 px); the 2×3 subplot layout with `IC=…%`/`Threshold=…` titles matches `local_Otsu.m`'s figure digit for digit (regenerated after the `num2str` fix); global Otsu visibly loses the dark right side like the book's (b). |
| 3.5 (p. 46) | `fig_3_05_compare.png` | **Substitute**: 3-level image identical to MATLAB; thresholds 61/142 coincide with the book's numbers although the image differs. |
| 3.7 (p. 51) | `fig_3_07_compare.png` | **Substitute**: k=2 and k=3 masks identical to MATLAB (0 px). |
| 3.1 (p. 38) | `fig_3_01_compare.png` | Text-only sketch: synthetic two-mode image and its bimodal histogram with T in the valley. |
| 3.2 (p. 39) | `fig_3_02_compare.png` | Text-only, substitute image: histogram with T = 125 marker and binarised image; layout follows the book's three panels. |
| 3.6 (p. 50) | `fig_3_06_compare.png` | Text-only: the five k-means steps (data, random centroids, assignment, update, convergence) plus J per iteration; structure follows the book's five panels. |
| 3.8 (p. 52) | `fig_3_08_compare.png` | Text-only: with an outlier as an initial centroid the two compact clusters collapse into one and the outlier forms its own cluster, as the book describes. |

## Numbers from the text (book vs ours)
| Item | Page | Book | Ours (Python) | MATLAB (original .m) | Status |
|---|---|---|---|---|---|
| Otsu IC, images 1 / 2 / 3 (Figs. 3.9–3.11(b), Table 3.1) | 53–56 | 15.36 / 32.05 / 72.63 % | 15.3616 / 32.0494 / 72.6325 % (t* = 123 / 107 / 182) | identical | match |
| k-means k=2 IC (Figs. 3.9–3.11(c), Table 3.1) | 53–56 | 15.65 / 32.49 / 96.50 % | 15.6468 / 32.4948 / 96.4977 % (`shift_bug=True`) | identical | match (requires the script's units bug; consistent units give 15.36 / 32.05 / 95.36 %) |
| Multi-Otsu image 3 (Fig. 3.12(a), Table 3.2) | 57–58 | ice 1 54.39, ice 2 42.11, water 3.50 %, IC 96.50 % | 54.3898 / 42.1079 / 3.5023 %, IC 96.4977 %; thresholds 120 / 197 | identical | match |
| k-means k=3 image 3 (Fig. 3.12(b), Table 3.2) | 57–58 | 77.91 / 19.20 / 2.89 %, IC 97.11 % | 77.9121 / 19.2006 / 2.8873 %, IC 97.1127 % | identical | match |
| Table 3.3 multi-Otsu means | 58 | 218.1751 / 177.0690 / 63.8908 | 218.17505778 / 177.06898472 / 63.89075113 | identical | match |
| Table 3.3 k-means means | 58 | 209,0405 [sic] / 161.6657 / 53.8234 | 209.04052943 / 161.66569338 / 53.82336812 | identical | match ("209,0405" = 209.0405) |
| p. 56 remark: both methods minimise within-class variance | 56 | (qualitative) | consistent-units k=3 boundaries 120.5 / 197.6 = multi-Otsu 120 / 197; centroids 63.8908 / 177.0690 / 218.1751 = Table 3.3 multi-Otsu row | — | demonstrated |
| Fig. 3.2(c) T = 125 | 39 | IC 41.47 % | substitute 2.jpg: 31.50 % | — | unverified (image not shipped) |
| Fig. 3.3 Otsu t = 108, η(108), η(125) | 43 | 42.14 %, 0.9643, 0.9620 | substitute: t* = 107, IC(I > 108) 32.02 %, η(107) 0.9674, η(108) 0.9674, η(125) 0.9650 | `eta` 0.9674474800, IC 32.0158 % (identical) | unverified (substitute) |
| Fig. 3.4(b)/(c) global t, block thresholds, block ICs, IC | 45 | 176; 92/132/177/98/126/176; 73.8472/20.9869/25.1055/86.496/21.2756/20.2267 %; 41.32 % | substitute: 122 (27.52 %); 153/106/57/158/112/58; 29.3396/22.5834/15.1796/50.0760/45.5496/26.8086 %; 31.59 % | identical | unverified (substitute) |
| Fig. 3.5 thresholds and coverages | 46 | 61, 142; 54.86 / 4.31 / 40.83 % | substitute: 61, 142; 65.75 / 3.29 / 30.96 % | identical | thresholds coincide; coverages unverified |
| Fig. 3.7 k=2 IC; k=3 coverages | 51 | 42.79 %; 53.52 / 5.16 / 41.32 % | substitute: 32.49 %; 64.95 / 3.75 / 31.30 % | identical | unverified (substitute) |

## Deviations & justifications
1. **`kmeans_gray` default `shift_bug=False`** applies Eqs. (3.36)–(3.37) in consistent units (text is the authority, CUMULATIVE pitfall 17); `shift_bug=True` reproduces `kmeans.m` lines 64–70 and every book k-means number. Both are reported side by side by `scripts/ch03_kmeans.py`; the consistent-units k=2 IC on image 3 is 95.36 %, not the book's 96.50 %.
2. **Empty cluster**: `kmeans.m` loops forever (NaN centroid); `kmeans_gray` raises `ValueError` (L1 test); `max_iter` guard raises `RuntimeError`.
3. **Stale-buffer means** (`Otsu.m` lines 39–42, `kmeans.m` 76–79): library returns correct class means; the scripts' values are available as `average_intensity_script` (`stale_mean_intensity`) and were verified identical to MATLAB on all nine runs — MATLAB prints e.g. 4759.10 for a mean gray level on 1.jpg.
4. **Cosine distance** = 1 − aᵀb/(|a||b|) (Eq. 3.33 prints the similarity); documented in the docstring.
5. **`mask1` displayed in gray**, not parula (`colormap('default')`); the MATLAB parula rendering is included in the compare figures for reference.
6. **`multithresh` N = 3** exhaustive vs `fminsearch` (`reimplemented`); N > 3 raises `NotImplementedError`.
7. **`separability.m` off-by-one** reproduced literally (`eta` = η(k − 1)); `eta_book` and `core.threshold.separability` give the book convention.
8. **`multithresh` int16** input: MATLAB's saturating `single(A − minA)` not emulated (not a book case).
9. Substitutes: `2.jpg` for `ch3ice.jpg`, `2.jpg` + synthetic ramp for `t.jpg` (analysis/ch03.md §5); every number derived from them is labelled as such.

## Open items
1. **Figs. 3.2–3.5 and 3.7 `unverified`**: `ch3ice.jpg` (Fig. 3.2(a), the OMAE-2012 image) and `t.jpg` (its illuminated version) are not shipped and have no public source; all their quoted numbers (T = 125 → 41.47 %; t = 108 → 42.14 %, η 0.9643/0.9620; Fig. 3.4 176, 92/132/177/98/126/176, 41.32 %; Fig. 3.5 54.86/4.31/40.83 %; Fig. 3.7 42.79 %, 53.52/5.16/41.32 %) are reproducible only with that image. The substitutes give MATLAB-identical results (rows above), and multi-Otsu on 2.jpg coincidentally reproduces Fig. 3.5's 61/142. Needs the source image from the user to close; not a port failure.
2. **`multithresh` int16** (`approx`, strict xfail): MATLAB's saturating `single(A − minA)` (`multithresh.m` line 274) is not emulated; only relevant if a later chapter feeds int16 images to `multithresh` (none do in MATLAB_ROOT).
3. **Note for ch4–ch9**: `graythresh`/`im2bw` are bit-exact against MATLAB on 12-Mpx JPEGs, so later chapters' `im2bw(I, graythresh(I))` masks can be compared with 0-pixel tolerance; `multithresh` N ≤ 2 is now exact too (N = 3 stays `reimplemented`: exhaustive vs `fminsearch`); `kmeans.m`'s algorithm is exact, but ch6/7/9 use the Statistics Toolbox `kmeans` instead (different algorithm, to be mapped there).

Closed since the first pass: the `num2str` title digits (former item 1), the single-precision `grayto8` product in
`_multithresh_pdf` (former item 2) and the `otsu_criterion` empty-class tail (former item 3) — all fixed by the porter
and confirmed above against the unchanged MATLAB references.

## Verdict: PASS
All four `.m` files run verbatim in MATLAB R2025a and every output of the port is identical to MATLAB
(`graythresh`/`im2bw`/`multithresh` N ≤ 2/`imquantize`/`block_otsu`/`separability`/`otsu_criterion`/`kmeans_gray`
bit-exact or ≤ 1e-12; all 14 raw image comparisons 0 px; every subplot title string identical; every book number for the
shipped images reproduced), all six scripts run, the full suite (ch02 + ch03) has 0 failures with one strict xfail
(int16, out of scope), and every `.m` has a row. The only `unverified` row is the set of book numbers for the unshipped
`ch3ice.jpg`/`t.jpg` images (Open item 1), which needs data the user does not have and is not a port defect.
