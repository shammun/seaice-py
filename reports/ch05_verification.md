# Chapter 5 verification — Watershed-Based Ice Floe Segmentation

Date 2026-09-09 · port commit `bd6d3a7` (`ch05: port — core.watershed (eml/watershed.m twin), imregionalmin/max,
imimposemin, ch05_watershed module, 9 scripts`) · **review follow-up 2026-09-09** (`reports/ch05_review.md`: M1, S1–S9, V1–V2 — all findings applied; verifier items M1/S3/S4/S6/S7/S8/S9/V1 in this report and `tests/test_ch05.py`, the rest by the porter) · verifier artefacts: `tests/test_ch05.py` (399 tests),
`reference/ch05/make_refs.py` (+ 16 `.mat` incl. `compat.mat` with 557 variables, `inputs.mat` with 82 fixtures,
`refs_log.json`), `reference/ch05/make_compare_figures.py`, `outputs/ch05/verify/` (MATLAB `imwrite`/`print` images
in `matlab/` (158 files), patched script copies in `scratch/`, `compat_code.m`, `image_diffs.json`, `make_refs_log.txt`,
pytest logs `pytest_ch05.txt` / `pytest_full.txt`, `pytest_scripts/<case>/` = the headless script runs),
`reports/ch05/figures/` (13 compare PNGs, git-ignored).

## Environment
python 3.11.5, numpy 2.4.6, scipy 1.17.1, scikit-image 0.26.0, opencv 5.0.0, matplotlib 3.11.1 |
reference engine: **MATLAB 25.1.0.2833191 (R2025a) Prerelease Update 2** via `tools/run_matlab_ref.py` (`matlab -batch`,
figures invisible) — all 16 reference files (`reference/ch05/refs_log.json`: engine `matlab`, status `ok`; no timeout
notes). Octave absent; no fallback used. No stale `MATLABWindow` processes before the runs; the five `MATLABWindow.exe` helpers left behind by the 17 `matlab -batch` launches were ended after the test suites finished (skill hygiene item d).

How the original code was run: every script hard-codes `imread('q.jpg')` and reads from the current folder. Each was
copied **verbatim** into `outputs/ch05/verify/scratch/` under a non-shadowing `*_ref.m` name next to `q.jpg`, the three
DIPUM helpers (`boundaries.m`, `fchcode.m`, `bound2im.m`) and `freeman_concave.m`, and run with cwd = scratch (the ch5
folder was never put on the MATLAB path). Only literals were patched: `'chessboard'` → `'cityblock'` / `'euclidean'` /
`'quasi-euclidean'` for the Fig. 5.9 variants of `distance_watershed.m`; the commented centroid block of
`marker_watershed.m` un-commented **and** used (`imimposemin(imgDist, marker0)`, `dis = marker0 .* bw`) for the
`point_markers` variant; `main.m`'s loop got accumulator statements appended *after* the verbatim
`c = intersect(ep,concave,'rows')` (`EP{i}`, `CONC{i}`, `REG{i}`, `CC{i}`, `NBR{i}`, `CONNECT{i}`, `G2{i}`, `TT(i)`,
`REMOVED(i)`, `SEGB{i}`) and the `imread` literal replaced by a lossless PNG for the synthetic substitute
(`synth.two_touching_floes()`) and for the Fig. 4.3(a) crop of ch04's `test.jpg` (552×574). `nrm_junction_ending.fig`
was opened with `openfig(..., 'invisible')` and its image `CData` and the three line series saved. `compat.mat` calls
the toolbox functions (`watershed`, `imregionalmin`, `imregionalmax`, `imimposemin`, `imreconstruct`, `imfilter`,
`regionprops`, `bitand`, `intersect`) on 82 controlled arrays written from Python (`inputs.mat`): the 12 plateau/tie
fixtures of `synth.plateau_fixtures` plus 25 order-discriminating ones (asymmetric column-vs-row adjacency and its
transpose, equal-priority plateaus of odd/even width, two corridors of different length, an unequal corridor, random
uint8 fields 30×40 with ties and 200×300 with and without ties, a 900-minima grid, int16/int32/uint16/single/double/
logical inputs, ±Inf minima and walls, border/corner minima, a ramp, 1×1 / 2×2 / constant / all-zero / checkerboard
images), 13 regional-extrema fixtures (border plateaus, constant, ±Inf, all-±Inf, row/column, 1×1), 11 `imimposemin`
images × 4 marker images (+ a level-0 plateau holding two markers), the 12 Fig. 5.15 patterns, the Fig. 5.16 matrix,
two touching discs and `regionprops`/`bitand`/`intersect` probes; every call is wrapped so MATLAB's own error text is
recorded when it rejects an input.

## pytest
`.venv/Scripts/python.exe -m pytest tests/test_ch05.py -q -p no:cacheprovider` (after the review fixes) →
**`399 passed in 106 s`**, 0 xfail, 0 warnings (`outputs/ch05/verify/pytest_ch05.txt`). Breakdown: 63 L1 (synthetic
truth: 12 `watershed` incl. the default-connectivity pin of review S3, 18 regional extrema, 7 `imimposemin`, 18 chain
code, 5 merging incl. the peanut fixture of review M1 and the per-line S4 assertions, 3 pipelines/display), 313 L2
(MATLAB parity: 151 `watershed` — 74 fixture cases + 74 `imregionalmin` on the same fixtures + matrix-conn forms,
rejections, order analysis; 40 extrema; 101 `imimposemin`; 3 misc; 15 script-level), 6 L4 (book numbers incl. the
ridge-thickness counts of review S7 and the `'ge3'` rule on `q.jpg`), 9 script runs + 8 non-default CLI flag combinations.

Before the review (verify commit `6ac4cd5`): `394 passed, 1 xfailed in 210 s` with 8 `RuntimeWarning`s (the xfail and
the warnings are the resolved Open items 1 and 5).

Full suite `.venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider` (after the porter's core changes:
strict 3×3 `conn` validation in `core.morphology`, matrix `conn` in `core.watershed`) → **`1145 passed, 2 skipped,
1 xfailed in 527 s`** (`outputs/ch05/verify/pytest_full.txt`; ch02 + ch03 + ch04 unchanged: 746 passed, 2 skipped,
1 xfailed; ch05 399).

Parity label counts (31 rows below, recounted after review S9): **exact 24 · near 0 · approx 1 · reimplemented 3 ·
unverified 1** (+ 1 mixed row `exact (printed truths) / reimplemented (fixtures)` for the synthetic fixtures and 1
display-only row, unlabelled). Every one of the 15 `.m` files (3 byte-identical duplicates included) has a row.

## Parity table
Levels: L1 synthetic truth · L2 MATLAB reference · L3 figure · L4 quoted number. "0 px" = identical arrays; label images
are compared **with their label values** (MATLAB uint8/uint16 cast to int32), not just as partitions.

| MATLAB file / function | Python | Evidence | Result (measured) | Parity | Notes |
|---|---|---|---|---|---|
| `bound2im.m` (top level) | `core.chaincode.bound2im` (ch02) | L2 | `bim` of the 263-point boundary 0 px (`chaincode_corner.mat`) | exact | unchanged ch02 port |
| `boundaries.m` (top level) | `core.chaincode.boundaries(bw, 8, 'cw')` | L2 | `b` (263×2, 1-based) identical incl. start `(35, 11)` and order; Fig. 5.16 matrix: `b516` (8 points, start `(3,2)`) identical for `'cww'` (= clockwise) **and** `'ccw'`; `numel(boundaries(B))` = 1 | exact | `'cww'` typo = anything-but-`'ccw'` = clockwise, confirmed |
| `fchcode.m` (top level) | `core.chaincode.fchcode` | L2 | `c.fcc` (262 codes), `c.x0y0`, `c.diff` identical on `q.jpg`; `f516`/`x0y0_516` identical on Fig. 5.16 | exact | `c.mm` not used by ch5 |
| `watershed_based/bound2im.m` | same target (byte-identical duplicate, md5 `c1ecf6d1…`) | L2 | as above | exact | duplicate |
| `watershed_based/boundaries.m` | same target (duplicate, md5 `cdb551a4…`) | L2 | as above; used by `main.m` through `freeman_concave.m` (per-line boundaries verified via `CONC{i}`) | exact | duplicate |
| `watershed_based/fchcode.m` | same target (duplicate, md5 `a833f9d8…`) | L2 | as above | exact | duplicate |
| `chaincode_corner.m` (Eqs. 5.9–5.18, rule 3 ≤ D ≤ 10, **Fig. 5.17**) | `ch05_watershed.freeman_concave`, `differential_chain_code`, `scripts/ch05_chaincode_corner.py` | L1 L2 L3 L4 | `B` 0 px (RGB `graythresh` = 130/255); `R`, `A`, `Sum`, `D`, `Diff` (262 each) **identical**; `A(l) = −8` (Eq. 5.11), `Sum0 = 0` after normalisation, `Diff ∈ [−5, 7]`; `p` (9 indices) and `concave` (9 points: `(48,27) (49,28) (48,29) (47,30) (62,35) (62,34) (62,33) (63,32) (64,32)`) identical; `k = 1`, `n_boundaries = 1`; L1: digital square → 0 concave points, four `−6` corners, Diff < 0 everywhere; L-shape → concave points only around the inner corner; disc → `Diff < 3` everywhere; `S(N) = −24` (Eq. 5.15) | exact | integer arithmetic; `object='first'` = script's `b{1}` |
| `watershed_based/freeman_concave.m` | `freeman_concave` (same function; called on the double label-valued `im` of `main.m`) | L2 | per-line `CONC{i}` identical for the 3 lines of `q.jpg` (1 / 1 / 9 concave points of the merged regions), the 1 line of the synthetic image (7 points) and the 7 lines of the 552×574 crop (8 / 1 / 1 / 1 / 8 / 14 / 15 points; lines 6–7 are single-pixel lines with `T = 4` and one ending point) | exact | `im2bw(I, graythresh(I))` on a `{0, k}` double image → `I > 0` path reproduced |
| `direct_watershed.m` (§5.1 p. 89, no book figure) | `direct_watershed`, `scripts/ch05_direct_watershed.py` | L2 L3 | `im0` 0 px; `level` = 128/255 (gray Otsu); `imgLabel` **0 px incl. label values** (160 basins, MATLAB class uint8), `bgm` 0 px (2132 ridge px), `im(bgm) = 0` overlay 0 px, `colorimg = bwlabel(im)` 0 px (160 components); 3 `imshow` images identical (`image_diffs.json`) | exact | `label2rgb('shuffle')` colours differ (display only) |
| `distance_propagation.m` (§5.1.2, **Fig. 5.10**) | `inverse_distance_map`, `scripts/ch05_distance_propagation.py` (`--source point|q.jpg`) | L2 L3 | active code (201×201 point, city-block): `img` 0 px, `imgDist` (single) 0.0, `dist0` 0.0, `dist` (uint8, `uint8()` rounding) 0 levels; commented `q.jpg` path: `D_city`/`D_chess` 0.0, `D_euc` ≤ 1e-4 / `D_quasi` ≤ 1e-4 vs MATLAB single, `dist_city`/`dist_euc`/`dist_chess` (uint8) 0 levels, literal `img = ~bw; -bwdist(~img)` 0.0 | exact | contour level count of `imcontour` is display-only (Deviation 7) |
| `distance_watershed.m` (§5.1.2, **Fig. 5.8** chessboard; **Fig. 5.9** Euclidean/city-block rows; quasi-Euclidean extra) | `distance_watershed`, `otsu_mask`, `inverse_distance`, `scripts/ch05_distance_watershed.py --metric …` | L2 L3 L4 | for each of the 4 metrics: `img0` 0 px (RGB Otsu 130/255, 2699 ice px), `imgDist` (single) 0.0 (chess/city) / ≤ 1e-4 (euc/quasi), `Dis_img` 0 px, `dis` 0 px (class double confirmed), `[p, q]` identical in `find` order, `bw0` 0 px, `imgLabel` **0 px with label values** (chess 2 / city 4 / euc 6 / quasi 6 basins, uint8), `bgm` 0 px (143 / 210 / 231 / 231 ridge px, 19 / 63 / 83 / 83 inside ice), `img(bgm) = 0` 0 px, `bwareaopen(…, 5)` 0 px, `colorimg` 0 px, counts (minima 2/4/6/6, minimum px 38/18/6/10, floes 2/4/6/6) equal; 20 raw images identical (1 gray-level display rounding on the city-block `imshow(imgDist, [])`, Deviation 4) | exact | the label `n_floes_pre` (before `bwareaopen`) also equal |
| `gradients_watershed.m` (§5.1.1, **Figs. 5.5, 5.6**) | `sobel_magnitude`, `gradient_watershed`, `scripts/ch05_gradients_watershed.py` | L2 L3 L4 | `I` 0 px, `bw` 0 px, `g` ≤ 1e-12 (unscaled Sobel, replicate), `l` **0 px with label values** (**350** basins, MATLAB class **uint16**), `wr`, `f` 0 px; `g2 = imclose(imopen(g, ones(7)), ones(7))` ≤ 1e-12, `l2` 0 px (**12** basins, uint8), `wr2`, `f2` 0 px; 8 raw images identical (1-level display rounding on `imshow(l2, [])`) | exact | `ones(7,7)` all-ones SE → `imclose.m` zero pre-pad path (ch04) |
| `marker_watershed.m` (§5.1.3, **Fig. 5.12**) | `marker_watershed`, `scripts/ch05_marker_watershed.py` | L2 L3 L4 | `img0` 0 px; `imgDist0` 0.0; `Dis_img` 0 px (**4 minima / 18 px**); `getnhood(strel('disk',5))` = `core.morphology.strel('disk', 5)` (9×9/69); `marker` 0 px (**2 components**); `imimposemin` output **bit-identical in single** (`−Inf` at the 328 marker px, class single confirmed); `imregionalmin(imposed)` 0 px; `bw0` 0 px; `imgLabel` 0 px with labels (2 basins), `bgm` 0 px (140 ridge px), `img` after `bwareaopen` 0 px (**2 floes**), `colorimg` 0 px; 7 raw images identical | exact | |
| `marker_watershed.m` commented centroid block (lines 18–25), un-commented and used as the marker | `component_centroids`, `marker_watershed(point_markers=True)` (`--centroid-markers`) | L2 | `regionprops(..., 'Centroid')` = `[23.3089 68.3591; 46 50]` reproduced (both the in-loop `cat(1, cen.Centroid)` and a direct `regionprops` call), `floor` → `[23 68; 46 50]`, `marker0` 0 px (2 px), imposed map bit-identical, `imgLabel` 0 px (2 basins, 142 ridge px), `bw0`, `img` 0 px, 2 floes; `marker0` image identical | exact | the shipped script only *displays* `marker0`; using it for the imposition is the block's intent (report note) |
| `topological_surface.m` (§5.1 **Fig. 5.1(b)(c)**, **Fig. 5.5(b)**, **Fig. 5.8(c)**) | `topographic_surfaces`, `plotting.surface_plot`, `scripts/ch05_topological_surface.py` | L2 L3 | `I` 0 px, `x = imcomplement(I)` 0 px (uint8), `g` ≤ 1e-12, `img` (gray Otsu 128/255) 0 px, `d = -bwdist(~img, 'chessboard')` 0.0 (single); `topo_gray`/`topo_complement`/`topo_bw` images identical; the three `surf` renderings compared visually (figure table) | exact | surface rendering is display-only (matplotlib vs MATLAB `texturemap`) |
| `watershed_based/main.m` (§5.2 Steps 1–5, **Fig. 5.14(b)–(i)**, Fig. 5.15(b) kernel) | `neighboring_region_merging`, `junction_endpoints`, `ENDPOINT_KERNEL`, `scripts/ch05_main.py` | L2 L3 L4 | `q.jpg`: `bw` 0 px, `D` 0.0, `L` 0 px with labels (4 basins, uint8), `w` 0 px (210), `f = bitand(bw, w)` 0 px (63 px, class logical), `seg0` 0 px, `[label, num] = bwlabel(f, 4)` 0 px / **3**, `wr` identical; per line: `ep` identical (`(64,11),(64,32)` / `(70,12),(70,33)` / `(48,29),(62,33)`, 1-based, `find` order), `T = 3` each, `g2 >= T` masks identical, `neighbor` → `connect = bwlabel` identical, `im = imreconstruct(g, connect)` identical, `concave` identical, `c = intersect` identical (∅, ∅, 2 points), `REMOVED = [1 1 0]`, `seg` after each iteration and final `seg` 0 px, floes **4 → 2**; **synthetic substitute** (`two_touching_floes`): 1 line (31 px), 2 basins, kept, 2 floes — all arrays identical; **Fig. 4.3(a) crop of ch04 `test.jpg`** (552×574): 11 basins, 7 lines (958 px), `T = [3 3 3 3 3 4 4]`, `REMOVED = [0 1 1 1 1 0 0]`, 11 → 7 floes — all arrays and all per-line records identical; 6 raw images identical (1-level display rounding on `-D`) | exact | sequential in-loop update of `seg` reproduced (`SEGB{i}`) |
| `nrm_junction_ending.fig` (authors' Fig. 5.14(f)) | `main` result `f` + per-line endpoints | L2 L3 | `openfig` → image `CData` (96×81 **logical**) = `f` **0 px**; 3 line series with end points `(x, y)` = `(29,48)-(33,62)`, `(12,70)-(33,70)`, `(11,64)-(32,64)` = exactly the port's 6 endpoints (set equality); no text objects re-created by R2025a from the old `hgS_070000` file (`nt = 0`) | exact | the authors' own saved figure confirms the junction lines and ending points |
| MATLAB `watershed(A[, conn])` (compiled `watershed_meyer`; all 6 watershed scripts, `main.m`) | **`core.watershed.watershed`** (line-by-line port of the codegen twin `eml/watershed.m`) | L1 L2 | **74 fixture cases** (37 fixtures × {8, 4}) **0 px with label values** — plateaus, diagonal/anti-diagonal corners, 5-minima plateau, 2-px corridors (equal/unequal, two of different length), asymmetric column-vs-row adjacency and its transpose, equal-priority plateaus of odd and even width, random uint8 30×40 with ties (90 / 196 basins), random uint8 200×300 (**6772 / 12 036 basins, MATLAB class uint16**) and its ternary version (2309 / 7768 basins, uint16), random double 50×60 (338 / 639, uint16), random single, 900-minima grid (uint16), int16 with negatives, int32, uint16, double with `−Inf` minima and a `+Inf` wall, border/corner minima, ramp (1 basin), 1×1, 2×2, constant, logical, all-zero, checkerboard (1 / 50 basins), 1×9 and 9×1 profiles; default conn 0 px; the 3×3 **matrix forms** `[0 1 0;1 1 1;0 1 0]` and `ones(3)` — called as matrices in MATLAB and, after review S6, passed as matrices to the port too — 0 px and equal to the scalar 4 / 8 calls (a matrix that is neither the cross nor all-ones raises); **every script image** (`q.jpg` gray 160 basins, Sobel `g` 350, `g2` 12, city/chess/euc/quasi inverse distance maps, imposed maps ×2, `−D` on the synthetic image and on the 552×574 crop with 11 basins) 0 px; two touching discs (city-block and Euclidean single-precision maps) 0 px. Output class rule confirmed (uint8 ≤ 255 basins, uint16 above); NaN rejected by both (`'Expected input number 1, I, to be non-NaN.'`). **Analysis risk 1 resolved**: (a) the neighbour-scan order of the flooding loop is *provably irrelevant* — a pixel's label depends only on the *set* of labels around it, and the neighbours pushed by one pop are consecutive same-label FIFO entries that no foreign-label pop can interleave at equal priority; measured: 12 random permutations of the offsets × 80 cases (all fixtures + 4 `q.jpg` maps, both connectivities) → 0 differences; (b) the *initial* column-major seed scan **does** matter: the same algorithm on the transposed image (= a row-major scan) changes ridges/partition in **31 of 80 cases** (e.g. `ws_asym_adjacency` 18 ridge px / 56 % partition agreement, `ws_rand_u8_200x300_ties` 3021 ridge px, `q.jpg` gray 92 px, Sobel 55 px), and MATLAB matches the column-major port on every one of them (label values included) — the order is pinned | exact | MATLAB rejects int16/int32 (Deviation 1: references for those 15 fixtures were generated on `double(X)`, identical values); MATLAB is N-D (Deviation 2); port returns int32 |
| `watershed_skimage` (cross-check helper) | `skimage.segmentation.watershed(connectivity, watershed_line=True)` | L1 | same basin count and > 90 % partition agreement on the two-disc fixture; on `q.jpg` the analysis measured 46–122 differing ridge px and inconsistent partitions | approx | never used by the ports; the notebook's "why not the library" cell |
| MATLAB `imregionalmin` / `imregionalmax` (Fig. 5.8(d), 5.12(a); inside `watershed`) | `core.morphology.imregionalmin`, `imregionalmax` (`skimage.local_minima/maxima(allow_borders=True)` + constant-image rule) | L1 L2 | **74 + 52 cases 0 px**: `imregionalmin` on all 37 watershed fixtures × {8, 4} and min + max on the 13 extrema fixtures × {8, 4} (border-touching plateaus, constant uint8/double → all true, ±Inf pixels, all-`+Inf` and all-`−Inf` images → all true, int16, single, uint8 ties, logical, 1×N / N×1 / 1×1); cross `[0 1 0;1 1 1;0 1 0]` = 4, `ones(3)` = 8, default = 8, class logical; script probes: `Dis_img` for 4 metrics, `rm_imp` of the imposed maps 0 px; NaN rejected by both | exact | |
| MATLAB `imimposemin(I, BW[, conn])` (§5.1.3 Steps 1–2, Fig. 5.12(c)) | `core.morphology.imimposemin` (M-code port on `imreconstruct`, arithmetic in the input class) | L1 L2 | **88 cases bit-identical** (`np.equal`, `±Inf` included, dtype preserved): uint8, uint8 saturating at 255, int16, int16 at `intmax`, single, negative single (inverse-distance-like), double, constant single (`h = 0.1`), constant uint8, double with `+Inf` values, double with `−Inf` values × markers {two blobs + a border pixel, all true, all false, adjacent/diagonal markers} × conn {8, 4}; `h` rule (`0.001·range`, `0.1` constant, `1` integer) equal to MATLAB's for all 11 images; two markers inside a level-0 plateau stay two minima (MATLAB `bwlabel` 2 = port), watershed of that map 0 px; double- and uint8-valued marker arrays accepted like MATLAB; `imregionalmin(imposed) == BW` for uint8/single/double as in MATLAB; the script's single `Dimp` (`marker_watershed.m`) bit-identical; logical `I` rejected by both | exact | MATLAB silently accepts NaN, the port raises (Deviation 3) |
| — `imreconstruct` with `±Inf` | `core.morphology.imreconstruct` (ch04) | L2 | marker with a `−Inf` pixel under a mask with a `+Inf` block, 8- and 4-conn: 0 diff | exact | needed by `imimposemin` |
| Eq. (5.2) `M = R^E_I(I + 1) − I` (text-only regional minima) | `ch05_watershed.regional_minima_by_reconstruction` | L1 L2 | equals `imregionalmin` (and MATLAB) on the 12 plateau fixtures × {8, 4}, random fields with `±Inf` pixels, and 12 of the 13 extrema fixtures; **fails on the all-`+Inf` image** (`Inf − Inf`: returns all-False, MATLAB/`imregionalmin` all-True) | reimplemented | Open item 1 (strict xfail) |
| Book Steps 1–2 minima imposition `(g + 1) ∧ f`, `R^E` (text form) | `impose_minima_book` | L1 | markers become the only minima; `watershed(impose_minima_book) == watershed(imimposemin)` on the two-disc map (2 basins) | reimplemented | `marker_value` defaults below `min(g)` for negative maps |
| Eqs. (5.1), (5.3)–(5.8) immersion algorithm with 3×3 dams | `watershed_immersion`, `threshold_set` | L1 | 1-D profile with 3 minima: same dams and same partition as `watershed` (basins numbered in level order instead of `bwlabel` order); plateau-maximum profile: same basin count, partition agreement ≥ 0.75 (dam placed on the plateau, documented) | reimplemented | teaching form only |
| Fig. 5.15 ending-point kernel and 12 patterns; `main.m` rule `abs(imfilter) >= max` vs text `≥ 3` | `ENDPOINT_KERNEL`, `junction_endpoints(rule='max'|'ge3')`, `synth.FIG_5_15_ENDPOINT_PATTERNS` | L1 L2 L4 | MATLAB `imfilter` of the 12 padded patterns: full 5×5 responses identical and centre = **3** for all 12 (both rules find the centre); isolated pixel → **4** (identical response); closed 4×4 loop: MATLAB `max = 2` and `abs(...) >= 2` hits 16 px (12 loop + the 4 enclosed background px with response −2) = port `'max'`; `'ge3'` → none | exact | the literal `abs` also admits background pixels (Deviation 5) |
| commented `regionprops(..., 'Centroid')` block | `component_centroids` | L2 | on a random 20×30 binary fixture: `bwlabel` 0 px and all centroids `allclose` to `cat(1, rp.Centroid)`; on the marker image `[23.3089 68.3591; 46 50]` | exact | MATLAB (x, y), 1-based |
| `bitand(logical, logical)`, `logical .* logical`, `intersect(…, 'rows')` | `&`, `_intersect_rows` | L2 | `class(bitand(true,false))` = logical, `bitand` values identical; `logical .* logical` = double; `intersect` rows (sorted, unique) identical, empty case `0×2` | exact | |
| `synth.two_touching_floes`, `plateau_fixtures`, `FIG_5_15_*`, `FIG_5_16_IMAGE/TRACE_MATLAB` | `core.synth` | L1 L2 | deterministic 96×81×3 uint8; Otsu → 1 component, city-block watershed 2 basins, the junction line's endpoints concave (kept), 2 floes — and the whole `main.m` pipeline on it identical to MATLAB; Fig. 5.16 printed triple `(2,3) → (2,4) → (3,5)` is a contiguous sub-sequence of the DIPUM trace starting at `(3,2)` (MATLAB identical); the 12 patterns each have exactly one 4-neighbour of the centre | exact (printed truths) / reimplemented (fixtures) | |
| `label2rgb(…,'jet','k','shuffle')`, `surf(…,'texturemap')`, `imcontour`, `image(…,'scaled')` | `plotting.label2rgb`, `surface_plot`, `contour_overlay` | L1 L3 | shapes/dtypes/background/seeded determinism; PNGs produced; renderings compared visually (figure table) | — (display only) | MATLAB's private shuffle stream is not reproducible (Deviation 6) |
| §5.3 experiments (Figs. 5.18–5.20, Table 5.1) | `scripts/ch05_experiments.py` (`--chapter ch04 --image test.jpg --crop fig4_3a`, synthetic image) | L2 (crop vs MATLAB) / L3 (book: qualitative) | the §5.3 Ny-Ålesund images are not shipped; on the Fig. 4.3(a) crop the whole pipeline is **exact vs MATLAB** (11 → 7 floes, 4 of 7 lines removed); Table 5.1 is not reproducible | unverified (book figures) | Open item 2 |

## Figures reproduced
Compare PNGs: `outputs/ch05/verify/<name>_compare.png` (copies in `reports/ch05/figures/`). Layout: Python row |
MATLAB `imwrite`/`print` row | book page rendered from `chapters/ch05.pdf`. `image_diffs.json`: **55 raw image pairs,
49 identical**; the 6 remaining pairs differ by **1 gray level** only (city-block `imshow(imgDist, [])` ×4 = 70 px, the
201×201 point map 328 px, `imshow(l2, [])` 767 px) at exact half-integers of the `mat2gray` stretch — a display rounding
artefact (Deviation 4), the underlying arrays being identical at L2.

| Figure | File | Verdict |
|---|---|---|
| 5.1(b)(c) (p. 85) | `fig_5_01_compare.png` | Complemented gray image identical to MATLAB and to the book's (b); both surface plots show the same copper "bath-tub" with the two floe basins and the ridge between them (viewpoints differ, data identical). |
| 5.5 (p. 90) | `fig_5_05_compare.png` | Sobel magnitude, its surface, the 350-basin label image and the black ridge overlay are pixel-identical to MATLAB and match the book's four panels (the dense mesh of spurious lines inside the floes). |
| 5.6 (p. 91) | `fig_5_06_compare.png` | The 7×7 close-opened gradient, its 12-basin watershed and the overlay match MATLAB (labels differ by 1 display level only) and the book: far fewer lines, but the two floes are still cut by extraneous lines. |
| 5.8 (p. 92) | `fig_5_08_compare.png` | All six panels identical to MATLAB and the book: two chessboard minima (a vertical bar and a short diagonal streak), one watershed line through the junction, the two floes separated. |
| 5.9 Euclidean row (p. 93) | `fig_5_09a_euclidean_compare.png` | Identical to MATLAB and the book's top row: 6 point minima, the small "island" box on the ridge and the double horizontal lines cutting the lower floe. |
| 5.9 city-block row (p. 93) | `fig_5_09b_cityblock_compare.png` | Identical to MATLAB and the book's bottom row: 4 minima (18 px with different levels near the centre), 2 horizontal lines plus the junction line → 4 regions. |
| 5.10 (p. 94) | `fig_5_10_compare.png` | Contour renderings differ in style (matplotlib levels vs `imcontour`) but show the same diamond / circular / square iso-distance propagation of the three metrics and the same "island" inside the zoom window; the arrays are exact. |
| `distance_propagation.m` active code | `sec_5_1_2_point_compare.png` | The city-block map of a single point (diamond contours) identical to MATLAB's `imshow` image and equivalent to its `imcontour` figure. |
| 5.12 (p. 97) | `fig_5_12_compare.png` | All five panels identical to MATLAB and the book: 4 minima, the two dilated marker blobs, the imposed map with two flat dark basins, one watershed line, two floes. |
| 5.14(b)–(i) (p. 100) | `fig_5_14_compare.png` | Binary image, `−D`, watershed lines, over-segmented image (4 regions), the three junction lines with their red ending points and the final 2-floe result are identical to MATLAB and match the book's panels; MATLAB's own `figure f + plot` shows the same red dots. |
| 5.14(f) authors' `.fig` | `fig_5_14f_authors_fig_compare.png` | The authors' shipped figure (junction lines + 3 red ending-point pairs) has exactly the port's `f` image (0 px) and the same six ending points. |
| 5.17 (p. 104) | `fig_5_17_compare.png` | The 9 concave points sit at the two junction notches, the same two "gray dots" as the book; MATLAB's figure and `bim` identical. |
| `direct_watershed.m` (no book figure, p. 89) | `sec_5_1_direct_compare.png` | The 160-basin mosaic of `watershed(gray)` and its overlay are identical to MATLAB; only the shuffled `label2rgb` colours differ. |
| 5.11 (p. 95), 5.7, 5.18–5.20 | — | Not reproducible: the images are not shipped (Open item 2). |
| 5.15(a)(b), 5.16 (pp. 101–102) | (tests) | Text-only truths: kernel responses 3 / 4 and the tracing sequence are asserted against MATLAB, no image. |

## Numbers from the text (book vs ours)
| Item | Page | Book | Ours (Python) | MATLAB (original .m) | Status |
|---|---|---|---|---|---|
| city-block DT of Fig. 5.8(a): "four regional minima consisting of 18 local minimum pixels" | 96 | 4 / 18 | 4 / 18 | 4 / 18 (`bwlabel(Dis_img)`, `nnz`) | match |
| "divides the two connected ice floes into four regions" | 96 | 4 | 4 basins / 4 floes | 4 / 4 | match |
| "a marker image containing two connected regions" (5-radius disk) | 96 | 2 | 2 (`strel('disk',5)` 9×9/69) | 2 | match |
| Fig. 5.12(e) "separates the two connected ice floes correctly" | 96 | 2 floes | 2 | 2 | match |
| Fig. 5.15 kernel on the 12 patterns "results in a value 3"; isolated point "larger than 3" | 99 | 3; > 3 | 3 ×12; 4 | 3 ×12; 4 | match |
| Fig. 5.14: junction lines, lines removed, final result | 99–100 | (g)/(h) one line removed, (i) final | 3 lines (22, 22, 19 px), 2 removed, 4 → 2 floes | 3 / 2 / 4 → 2 | match (the book shows one removal step; the script removes two lines) |
| 5.14(f) ending points | 100 | gray dots | `(64,11) (64,32) (70,12) (70,33) (48,29) (62,33)` | identical; authors' `.fig` identical | match |
| concave rule `3 ≤ D(i) ≤ 10`, θ = 15°·D | 104 | 3–10 | `BOOK_PARAMS` 3–10; Fig. 5.17: 9 points, `Diff ∈ [−5, 7]` | 9 points identical | match |
| Eq. (5.11) `A(0) − A(N) = 8`; Eq. (5.15) `S(0) − S(N) = 24` | 103 | 8; 24 | `A(N) = −8`; `S(N) = −24` after normalisation (square, L-shape, disc, `q.jpg`) | `A(l) = −8` | match |
| Fig. 5.16 tracing `b0 (2,3) → b1 (2,4) → b2 (3,5)` | 102 | sequence | contiguous sub-sequence of the DIPUM trace (start `(3,2)`) | identical trace | match (start pixel convention differs, documented) |
| 3×3 Sobel gradient watershed (Fig. 5.5) / 7×7 close-opening (Fig. 5.6) | 89–91 | "far too many" / "fewer" lines | 350 → 12 basins | 350 / 12 | match |
| chessboard: "lowest probability of over-segmentation … tends to cause under-segmentation" | 93 | qualitative | chess 2 / city 4 / euc 6 / quasi 6 basins | identical | match |
| watershed lines are "1-pixel-thick", 4-connected paths | 89, 99 | qualitative | on the four inverse distance maps of `q.jpg` the ridges are 4-connected staircases with **0** 2×2 all-ridge blocks (`bwlabel(f, 4)` = 3 lines) — a diagonal staircase *is* what p. 99 predicts (review S7 corrects analysis risk 9); the real departure is on plateau-rich segmentation functions, where Meyer's flooding leaves thick ridges: **10** 2×2 all-ridge blocks in `watershed(gray)` (`direct_watershed.m`), 23 on the Sobel map, 1 on the close-opened map, **5** on `random_u8_20x25` (`test_p99_ridge_thickness_on_plateaus`) | identical (same arrays) | match on the distance maps; caveat only for plateau functions |
| Table 5.1 (manual 5/12/20/38 floes …) | 107 | table | not reproducible (images unshipped) | — | Open item 2 |

## Deviations & justifications
1. **Input classes of `watershed`**: MATLAB R2025a's `watershed` accepts only `uint8, uint16, single, double, logical`
   (error text recorded for all 15 int16/int32 fixtures: `Expected input number 1, A, to be one of these types: …
   Instead its type was int16.`); the port accepts any real dtype. For those fixtures the reference was generated on
   `double(X)` (identical values → identical flooding) and the port, run on the raw int16/int32 array, is 0 px from it.
   The porter's `plateau_fixtures` are int16 — a MATLAB user would have to cast them. Superset, not a defect.
2. **N-D**: MATLAB's `watershed` accepts `conn = 6/18/26` and 3-D arrays (`ws_c6_ok = ws_3d_ok = 1`); the port is 2-D
   only and raises `ValueError` for `conn = 6` and for 3-D input. The 2-D 3×3 connectivity *matrices* (cross = 4,
   all-ones = 8) are accepted since review S6 and verified against MATLAB's matrix-form calls (`wsc_pf_even_plateau`,
   `wsc8_ws_rand_u8_30x40_ties`, 0 px); any other 3×3 pattern raises (MATLAB likewise requires a valid connectivity).
   The book never uses 3-D.
3. **`imimposemin` with NaN**: MATLAB has no `'nonnan'` check and returns a result for a NaN image (`ii_nan_ok = 1`);
   the port raises like `imregionalmin`/`watershed` do. Stricter, documented.
4. **Display rounding**: `imshow(I, [])` images written by MATLAB (`mat2gray` computes `x·δ − lo·δ`) and by the port
   (`(x − lo)/(hi − lo)`) differ by 1 gray level at exact half-integers `k/12·255`, `k/200·255` (6 of 55 pairs); the
   arrays are identical at L2. Not a port issue.
5. **Ending-point rule**: the script's `abs(imfilter(g, wr)) >= max(...)` (default `'max'`) is reproduced literally,
   including its admission of background pixels with ≥ 2–3 line 4-neighbours (16 hits on a closed loop, MATLAB identical);
   the text's `≥ 3` rule is `'ge3'`. Both give the same 6 endpoints on `q.jpg` (max = 3 for every line).
6. **`label2rgb('shuffle')`** colours come from MATLAB's private fixed-seed stream — not reproducible; display only.
7. **Contour / surface renderings** (`imcontour` level count, `surf` view) are matplotlib approximations of the display;
   the arrays under them are exact.
8. **Output class**: the port returns int32 labels; MATLAB uint8 (≤ 255 regions) / uint16 (measured 6772, 12 036, 2309,
   7768, 900, 338 and 639 regions → uint16). Compared after casting; label *values* are identical.
9. **`imshow` of the imposed map** (`−Inf` markers): MATLAB's `imshow(imgDist, [])` on a `−Inf` image is undefined for
   `imwrite`; both sides map `−Inf` to the finite minimum (my convention in `make_refs`, `_imw_finite`), giving identical PNGs.
10. **Neighbour-scan order** (analysis risk 1) turned out to be a non-issue by construction of Meyer's loop (see the
    `watershed` row); the *initial* scan order is the order-sensitive part and is pinned by 31 sensitive fixture cases.

## Open items
1. **Resolved (review V1)** — `regional_minima_by_reconstruction` on an all-`+Inf` (constant) image: the teaching form
   of Eq. 5.2 is `Inf − Inf` there and returned all-False where MATLAB `imregionalmin` returns all-True. The porter now
   returns all-True for a constant image (as `imregionalmin` does); the former strict `xfail`
   `TestL2RegionalExtrema::test_eq_5_2_identity_vs_matlab[rm_all_inf]` is a normal pass (13 of 13 extrema fixtures).
2. **Figs. 5.7, 5.11, 5.18–5.20 and Table 5.1 `unverified`**: the Ny-Ålesund May 2011 images (brash removed manually)
   are not shipped. The §5.3 procedure is reproduced on the Fig. 4.3(a) crop of ch04's `test.jpg` and on the synthetic
   substitute, both **exact vs MATLAB** (`main_crop.mat`, `main_synth.mat`); the book's counts cannot be checked. Needs the
   source images from the user; not a port failure.
3. **Resolved (review V2, porter)** — docstrings of `core.watershed.watershed` / `synth.plateau_fixtures` /
   `imimposemin` now state that MATLAB rejects int16/int32 `watershed` input (MATLAB cross-check of the int16 fixtures
   on `double(X)`, Deviation 1) and that MATLAB's `imimposemin` accepts NaN while the port raises (Deviation 3).
4. **Resolved (review M1) — former L1 coverage gap**: `test_convex_blob_spurious_lines_are_removed` was vacuous (a
   convex blob's distance transform has a single minimum → 0 junction lines → `all([])`), so the "line removed" branch
   of `neighboring_region_merging` had no synthetic-truth coverage (only the L2 `q.jpg`/crop references exercised it).
   Replaced by `test_peanut_spurious_line_is_removed[euclidean|cityblock]`: two r = 20 discs centred (35, 42)/(35, 48) on
   70×90 → 2 basins, 1 junction line with ending points (16, 45)/(54, 45), 0 concave ending points, line removed,
   floes 2 → 1, `seg == bw`. `test_non_sequential_and_rules` now also asserts the per-line `removed`/`endpoints`/`pixels`
   and the final `seg` of the non-sequential and `'ge3'` runs against the default run on the synthetic image and the
   peanut (review S4); `'ge3'` is now the text's signed `>= 3` rule without `abs` (porter, S2) and still removes lines
   1–2 of `q.jpg` (`test_p99_ge3_rule_on_q_image`).
5. **Resolved (review S8) — `imimposemin` `RuntimeWarning: invalid value encountered in add`** on images holding
   `±Inf` (`im_double_pinf`, `im_double_ninf`): `h = 0.001·(max − min) = Inf`, so `I + h` produces `Inf − Inf = NaN` at
   the `−Inf` pixels before `min(I + h, fm)` — MATLAB computes exactly the same values (the 16 `±Inf` cases are
   bit-identical) and the port now runs that addition under `np.errstate(invalid='ignore')`. The 8 warnings that
   appeared in the first pytest log are gone.

## Verdict: PASS
All 9 scripts run headless (default flags and 8 non-default combinations, exit 0), every one of the 15 `.m` files has a
row and is `exact` against MATLAB R2025a running the original code (`q.jpg`, the synthetic substitute and a 552×574
real-image crop for `main.m`), the new primitives are `exact` on 74 `watershed` / 126 regional-extrema / 88
`imimposemin` constructed cases with label values included (plus the 3×3 matrix-connectivity forms against MATLAB's
matrix calls), the flooding order is pinned (31 order-sensitive cases matched), every quoted number of the text is
reproduced, 49 of 55 raw figure images are identical and the other 6 differ by one display gray level, all review
findings (`reports/ch05_review.md` M1, S1–S9, V1–V2) are applied and evidenced, and the test suites pass
(`tests/test_ch05.py` 399 passed, no xfail, no warnings; full suite 1145 passed, 2 skipped, 1 xfailed). The single
`unverified` row (unshipped §5.3 images) is covered by Open item 2 and is not a port failure.
