# CUMULATIVE knowledge — seaice-py
(Rewritten by the knowledge-keeper after every chapter. Read this first at the start of any chapter. Last rewrite:
after **ch04**, 2026-09-09. Per-chapter detail: `knowledge/chNN.md`; verified call mappings: `knowledge/function_map.md`.)

## Pipeline so far
```
Ch2 primitives (DONE)                Ch3 ice mask (DONE)                  Ch4 edges + morphology (DONE)        Ch5 watershed  Ch6 GVF  Ch7 types  Ch8/9 apps
  io.load_image ──────────────────►  rgb2gray_matlab ─► threshold.graythresh/im2bw ─► binary ice mask ─► every later .m
  matlab_compat.{rgb2gray_matlab,          │            (bit-exact vs MATLAB on 12-Mpx JPEGs → 0-px mask comparisons)
    imcomplement, matlab_round,            ├─► threshold.multithresh(N=2)+imquantize ─► 3 groups (Ch7 ice types, Ch8)
    im2double, im2uint8,                   ├─► threshold.block_otsu(n_r, n_c) ─────────► Ch9 block_threshold.m (same code)
    to_uint8_saturating}                   ├─► clustering.kmeans_gray (authors' kmeans.m) ─► Ch9 movie_kmeans.m
  histogram.{imhist, normalized_histogram} └─► threshold.ice_concentration/class_coverage/class_mean_intensity ─► Ch8/Ch9 IC series
  connectivity.{label_components (= bwlabel), bwareaopen} ─► Ch4 speck removal, Ch5–Ch9 (bwlabel everywhere)
  distance.{bwdist, distance_transform} ─────► Ch5 watershed markers, Ch6 GVF_distance/dist
  filters.{conv2, imfilter, fspecial} ───────► Ch4 edge('sobel'/'log') ─► Ch5 gradient surface, Ch6 GVF edge map (Gaussian blur), Ch7
  edges.edge (MATLAB edge: sobel/prewitt/roberts/log/zerocross, thinning) ─► Ch4 Fig 4.3/4.6, Ch6 GVF edge map, Ch9 outlines
  morphology.{strel (octagon disk), imerode/imdilate, imopen/imclose (pre-pad rule), imreconstruct, morphological_gradient}
      ─► Ch4 Figs 4.9–4.20 ─► Ch5 markers/imimposemin/imfill (build on imreconstruct), Ch7 shape enhancement (imopen/imclose), Ch8/9
  setops.{reflect, translate, gray_union/…} ─► Ch4 §4.2 definitions (dilation reflects the SE), Ch7 mask combinations
  chaincode.{boundaries, fchcode, bound2im} ─► Ch5 concavity / neighbouring-region merging
  interp.{interp2, warp_image, resize} ──────► Ch6 snakedeform (interp2), App. A orthorectification (warp_image)
  clustering.{kmeans_lloyd(init='kmeans++'), pairwise_distance} ─► Ch6/7/9 Statistics-Toolbox kmeans mapping (approx, decide in ch6)
  color.{rgb2cmy, rgb2cmyk, rgb2hsi, indexed_to_rgb}, plotting.*, synth.*, cli.*
Ch5 watershed → Ch6 GVF → Ch7 ice types → Ch8 applications (concentration, floe size) → Ch9 model ice → Ch10 = App. A calibration
```
Conventions fixed in ch02–ch04 and binding for all later chapters:
- Coordinates: book `x` = row, `y` = col (Eq. 2.1). Python is 0-based `(row, col)` everywhere; tests add 1 to compare
  with MATLAB. `ChainCode.x0y0_matlab` is the 1-based pair. Crops quote both forms (`FIG_4_3A_CROP` / `_MATLAB`).
- Numerics stay float64; uint8 conversion only at display/save time via `plotting.*` or `to_uint8_saturating` / `im2uint8`.
  Otsu levels/thresholds stay **float** (tie-averaged half-integers are compared in double, as MATLAB does).
  Morphology (`imerode`/`imdilate`/`imopen`/`imclose`/`imreconstruct`) **preserves the input dtype** (bool → bool);
  `morphological_gradient` follows MATLAB's class rules (logical − logical → double). `core.edges.edge` takes **float only**.
- Every `.m` maps to one Python script (`scripts/chNN_<name>.py`) or one `seaice` function; docstrings cite §/Eq/`.m`.
- References: MATLAB R2025a via `tools/run_matlab_ref.py`, driven by `reference/chNN/make_refs.py`; `.mat` files in
  `reference/chNN/`; tests skip with a clear message if a `.mat` is missing. Run `.m` scripts from a scratch cwd
  (`outputs/chNN/verify/scratch`): some write PNGs (`saveas`), some hard-code `imread('test.jpg')`/`k = 3` (copy the
  script verbatim into scratch under a non-shadowing name and patch only the literals; put substitutes for unshipped
  inputs there under the original file name so both engines decode the same bytes). Never `addpath` a chapter folder.
  Controlled fixtures (`inputs.mat`) are written from Python with **constructed ties / intmax-intmin plants /
  asymmetric and even SEs**; MATLAB type errors are recorded with `try/catch` (`i64_ok = 0`) instead of failing the run.
- Figures: `outputs/chNN/fig_<ch>_<fig>_<slug>.png` only for genuine book figures; `sec_<ch>_<sec>_<slug>.png` otherwise.
- Authors' script bugs: the library default follows the book text/equations; the literal script behaviour is available
  behind an explicit flag (`rgb2hsi(matlab_bug=True)`, `kmeans_gray(shift_bug=True)`, `stale_mean_intensity`) and every
  book number that depends on the bug says so. Literal script idioms that are not bugs (`double(im)/256`) are ported as is.
- Scripts that need book images print `SKIP …` and exit 0 when `data/book/chNN` is absent (public/Colab runs);
  non-default CLI flags get one parametrised test each (`test_script_cli_flags`).

## Available primitives in seaice/core/
| function | module | book § | used by chapters | parity |
|---|---|---|---|---|
| `load_image(chapter, name)` (case-insensitive `imread`; private copy first, public-domain NASA substitute as fallback), `repo_root`, `book_data_dir`, `output_dir`, `fetch` | `io` | — | all | exact (JPEG decode identical to MATLAB on `rgb.JPG` and the four 12-Mpx ch03/ch04 files) |
| `chapter_argparser`, `resolve_dirs` | `cli` | — | all scripts | — |
| `rgb2gray_matlab(rgb)` | `matlab_compat` | §2.2 | ch3–ch10 | exact (0 px differ, 5 images) |
| `imcomplement(img)` (canonical for MATLAB `imcomplement`) | `matlab_compat` | Eq. 2.3 / 2.21 | ch5 `topological_surface.m`, ch7 | exact |
| `matlab_round(x)` | `matlab_compat` | — | ch3 (`multithresh`), ch4 (`intline`), ch6, ch7, ch9 | exact |
| `im2double(img)`, `im2uint8(img)` (float/logical/uint8 only — uint16/int16 rule lives in `threshold._im2uint8_any` for now) | `matlab_compat` | §2.1.1 | ch3, ch7, ch9 | exact |
| `to_uint8_saturating(x)` (canonical for MATLAB `uint8(x)` on 0–255 values) | `matlab_compat` | §2.1.1 | ch3+ | exact |
| `imshow_scale`, `to_display_uint8`, `save_image`, `imshow_matlab`, `show_matrix`, `finish_figure` | `plotting` | `imshow(I)` / `imshow(I,[])` | all | display only |
| `split_rgb`, `rgb2cmy`, `rgb2cmyk(u, b)`, `rgb2hsi(use_atan2, matlab_bug, scale)`, `indexed_to_rgb(one_based)` | `color` | §2.1, Eqs. 2.2–2.6 | ch6/ch7 colour stats, ch8 colormaps | exact / reimplemented (HSI, CMYK) |
| `imhist(img, nbins=None)`, `normalized_histogram` | `histogram` | §2.2, Eqs. 2.7–2.8 | ch3 Otsu/separability, ch6/ch7 | exact (logical → 2 bins) |
| `n4/nd/n8`, `is_adjacent`, `is_m_adjacent`, `find_paths` | `connectivity` | §2.3.1–2.3.3 | teaching only | reimplemented (text) |
| `label_components(bw, conn=8)` (= `bwlabel` incl. numbering; **int32** output), `count_components` | `connectivity` | §2.3.4 | ch5–ch9 | exact |
| `bwareaopen(bw, P, conn=8)` (keep components with ≥ P pixels; built on `label_components`) | `connectivity` | §2.3.4; `derivative.m` line 8 | ch4 flag, ch5 marker cleaning, ch7, ch9 | exact |
| `region_boundary_mask(bw, conn)` | `connectivity` | §2.3.5 | (≈ `bwperim`, unverified vs it) | reimplemented |
| `distance_transform(bw, metric)` (Eq. 2.9 = `bwdist(~f)`) | `distance` | §2.4 | book-equation code | exact |
| `bwdist(bw, metric)` (MATLAB semantics) | `distance` | §2.4.4 | ch5 watershed, ch6 GVF | exact (eucl/city/chess), near (quasi) |
| `quasi_euclidean_dt`, `pixel_distance`, `center_distance_map` | `distance` | Eqs. 2.10–2.12 | ch5 (if `'quasi-euclidean'`) | reimplemented |
| `conv2(f, w, mode='same')` (MATLAB default is `'full'` — pass it explicitly) | `filters` | Eq. 2.14 | ch4 `derivative.m` line 15, ch6 `xconv2` | exact |
| `imfilter(f, w, *matlab_options)` (correlation default; positional `'replicate'` etc. accepted) | `filters` | §2.5 | ch4 `edge`, ch5, ch6 | exact |
| `conv_at(f, w, x, y, correlate)` | `filters` | Eqs. 2.14 / 2.15 | teaching | exact |
| `fspecial(kind, p2, p3, *, hsize, sigma, alpha, radius)` (sobel, prewitt, laplacian α, gaussian, log, average, disk, unsharp; MATLAB defaults; prefix matching) | `filters` | §4.1, Figs. 4.2/4.4/4.5, Eqs. 4.14–4.15 | ch4 `edge`, ch5 Gaussian smoothing, ch6 GVF edge map, ch7 | exact (26 kernels ≤ 7e-18) |
| `edge(a, method, thresh=None, direction='both', thinning=True, sigma=2.0, H=None) -> EdgeResult(bw, thresh, gv, gh)` (sobel/prewitt/roberts/log/zerocross; float input only; canny raises) | `edges` | §4.1, Figs. 4.3/4.6 | ch4, ch6 GVF edge map, ch9 | exact (252 gradient maps + 14 LoG maps 0 px; `'log'` near on flat synthetic patches) |
| `gradient_sobel_prewitt(a, kind, kx, ky)`, `gradient_roberts`, `thin_gradient(b, bx, by, kx, ky, offset, cutoff)`, `log_zero_crossings(b, T)` | `edges` | §4.1 | internal / teaching | exact |
| `strel(shape, *params, n=None)` (bool nhood; disk = n = 4 periodic-line **octagon**, prefix `'dis'`), `disk_decomposition(r, n)`, `minkowski_sum`, `periodic_line`, `line_strel`, `intline`, `se_origin` | `morphology` | §4.2, Fig. 4.7 | ch4, ch5 (`strel('disk')` markers), ch7 (`imopen/imclose` disks), ch8/9 | exact (116 nhoods, 11 decompositions = `getnhood`/`decompose`) |
| `imerode(I, se)`, `imdilate(I, se)` (nhood or decomposed list; dtype preserved; erosion pads 1/intmax/+Inf, dilation pads 0/intmin/−Inf and reflects the SE) | `morphology` | §4.2.1, Eqs. 4.16–4.21 | ch4–ch9 | exact (bool/uint8/double/int16/int32/uint32 vs MATLAB); int64/uint64 L1-only (MATLAB rejects them) |
| `imopen(I, se)` (bare composition), `imclose(I, se)` (MATLAB pre-pad `ceil(size/2)`: 0 via `imclose.m`, class-min via Halide when nnz < 600 ∧ sides ≤ 15 ∧ not all-ones) | `morphology` | §4.2.2, Eqs. 4.22–4.23 | ch5, ch7 shape enhancement | exact (54 + 72 cases) |
| `imreconstruct(marker, mask, conn=8)` (`marker <= mask` enforced; conn 4/8/any nhood), `reconstruct_by_erosion`, `geodesic_dilation/erosion(F, G, se, n)`, `reconstruct_iterative(F, G, se, method) -> (X, k)` | `morphology` | §4.2.3, Eqs. 4.24–4.38 | ch5 (`imimposemin`, `imregionalmin`, `imfill` to be built on it), ch7 | exact (7 cases) / reimplemented (erosion dual, literal loops) |
| `morphological_gradient(I, se, kind='basic'|'internal'|'external', *, eroded=, dilated=)` (MATLAB minus: logical → double, integers saturate) | `morphology` | §4.2.4, Eqs. 4.39–4.42 | ch4 Figs. 4.15–4.20, ch5 watershed surface, ch7 outlines | exact |
| `complement/union/intersection/difference`, `reflect`, `translate`, `gray_complement/union/intersection`, `bitwise_*`, `truth_tables` | `setops` | §2.6, Eqs. 2.16–2.30 | ch4 §4.2, ch7 | exact |
| `boundaries(bw, conn, direction)` (DIPUM, exterior only, closed, `bwlabel` order) | `chaincode` | §2.3.5, §2.7 | ch5 | exact |
| `fchcode(b, conn, direction) -> ChainCode`, `first_difference`, `min_magnitude`, `normalized_first_difference`, `code_reverse`, `chain_to_points` | `chaincode` | §2.7, Eq. 2.31 | ch5 | exact (tie-break where MATLAB errors: reimplemented) |
| `bound2im(b, M, N, x0, y0)` (0-based `x0, y0`) | `chaincode` | §2.7 | ch5 | exact |
| `interp_nearest/interp_bilinear/interp_bicubic`, `interp2(img, u, v, method)`, `keys_kernel` | `interp` | §2.8, Eqs. 2.33–2.41 | ch6 `snakedeform`, ch10 | exact (borders incl.) |
| `warp_image(img, T_inv, out_shape, method)` | `interp` | Eqs. 2.32–2.33 | ch10 App. A | exact by construction |
| `resize(img, scale, method)` | `interp` | §2.8 | (ch9 — see pitfall) | exact nearest/bilinear enlarge; approx bicubic/shrink |
| `graythresh(I) -> (level, em)`, `otsuthresh(counts)` (256-bin Otsu, tie averaging, `em` = η(t*)) | `threshold` | §3.1.1.1, Eqs. 3.20–3.22 | ch4–ch9 (`im2bw(I, graythresh(I))` everywhere) | exact (bit-identical incl. half-integer ties) |
| `im2bw(I, level=0.5)` (strict `>` in double; uint8/uint16/int16/float/RGB/logical) | `threshold` | Eq. 3.1 | ch4–ch9, ch9 `movie_floe.m` (default 0.5) | exact |
| `multithresh(A, N)` (`N ≤ 3`; output in input class) | `threshold` | §3.1.3, Eqs. 3.24–3.28 | ch7 three groups (use `N = 2`), ch8 | exact `N ≤ 2` (uint8/uint16/double); reimplemented `N = 3` (exhaustive vs `fminsearch`); approx int16 |
| `imquantize(A, levels, values=None)` (`1 + Σ(A > level_i)`) | `threshold` | Eq. 3.23 | ch7 | exact |
| `otsu_criterion(counts) -> OtsuCurves`, `separability(gray, t)` (book convention C0 = 0..t) | `threshold` | Eqs. 3.3–3.22 | ch8 (η discussion), teaching | exact vs hand-coded MATLAB loops |
| `block_otsu(gray, n_r, n_c) -> BlockOtsu` (raises on non-divisible sizes like MATLAB) | `threshold` | §3.1.2, Fig. 3.4(c) | ch9 `block_threshold.m` (identical code) | exact |
| `ice_concentration(mask)`, `class_coverage(labels, k)`, `class_mean_intensity(gray, labels, k)` | `threshold` | Tables 3.1–3.3 | ch8/ch9 IC series | exact |
| `_im2uint8_any(I)` (uint16 → `round(v/257)`, int16 offset) | `threshold` | — | ch4+ if uint16 images appear | exact (all 65 536 values) |
| `kmeans_gray(gray, k, shift_bug=False, max_iter) -> KMeansGray` (authors' `kmeans.m`; `shift_bug=True` = script/book numbers) | `clustering` | §3.2.2/§3.3, Eqs. 3.35–3.37 | ch9 `movie_kmeans.m` (check which k-means it calls) | exact (`shift_bug=True`); reimplemented (consistent units) |
| `kmeans_lloyd(X, k, init='random'|'equal'|'kmeans++'|array, seed, max_iter, tol=0) -> KMeansResult`, `objective_J` | `clustering` | Eqs. 3.35–3.37, Figs. 3.6/3.8 | ch6/7/9 Statistics `kmeans` candidate (`approx`) | reimplemented (text) |
| `pairwise_distance(X, Y, metric, cov)` (euclidean/sqeuclidean/chebyshev/cityblock/cosine = 1 − sim/mahalanobis) | `clustering` | Eqs. 3.29–3.34 | teaching, ch6+ distance calls | exact (closed forms, L1 only) |
| `point_image`, `spur_shape`, `set_operation_fixtures`, `book_fixtures`, `FIG_2_*` constants | `synth` | Figs 2.10–2.21 | tests ch2/ch5 | — |
| `uneven_illumination(img, gain, axis, bias, offset, kind)`, `two_clusters_2d(seed, …, outlier)`, `bimodal_image(seed, …)` | `synth` | Figs. 3.4(a), 3.6/3.8, 3.1 | ch9 tank-lighting tests, any thresholding test | synthetic |
| `FIG_4_8_IMAGE/SE/ERODED/DILATED`, `two_floes_profile(n) -> (1, n) uint8`, `two_blobs_with_marker(shape) -> (marker, mask)` | `synth` | Figs. 4.8, 4.11–4.14 | morphology/reconstruction tests (ch5/ch7) | exact (printed) / synthetic |
| `_num2str(x)` (MATLAB scalar `num2str`: `%d` for integers, else `%.{max(floor(log10|x|)+5,5)}g`), `stale_mean_intensity` | `ch03_ice_pixel_detection` (promote to `matlab_compat` when a second chapter needs it) | Fig. 3.4(c) titles | ch9 movie labels | exact |
| `gradient_operator/magnitude/direction`, `threshold_gradient`, `laplacian`, `laplacian_zero_crossings`, `gaussian_kernel`, `log_kernel`, `sobel_edges_script`, `morphological_edges`, `FIG_4_3A_CROP`, `BOOK_PARAMS` | `ch04_ice_edge_detection` (chapter module) | Eqs. 4.1–4.15, scripts | ch5+ baseline edge maps | reimplemented (book forms) / exact (scripts) |

Not yet in core (first needed by): `watershed`, `imimposemin`, `imregionalmin/max`, `imfill`, `regionprops` compat,
`bwperim` verification (ch5); GVF/snake, Canny (if the GVF code uses it), Statistics-Toolbox `kmeans` mapping (ch6);
`bwmorph` LUTs (ch7); `imresize_matlab` with antialiasing (ch9 if needed); DLT / lens distortion (ch10).

## Global pitfalls (MATLAB → Python) confirmed in this project
1. Book `x` = row, `y` = col; MATLAB 1-based; Python 0-based `(row, col)`. `interp2(X, Y, Z, Xq, Yq)` has `X` = column:
   Python `interp2(img, u=Yq−1, v=Xq−1)`.
2. `bwdist(BW)` = distance to nearest **nonzero** = `distance_transform(~BW)`; returns single → `atol≈1e-4`.
3. `conv2` flips the kernel, `imfilter` does not (correlation). Book Eq. (2.15) is printed in correlation form
   (inconsistent with Eq. 2.14); ch4's Fig. 4.2 kernels are applied by correlation (`imfilter`) — check every
   "convolution" claim against the actual MATLAB call. `conv2`'s default mode is `'full'` (output grows).
4. `imfilter(I, h, 'replicate')` positional strings: `core.filters.imfilter` accepts them as-is; never remap to `mode`.
5. `bwlabel` numbers components column-major by first occurrence; `label_components` reproduces it — compare label
   images directly, not only partitions.
6. DIPUM `boundaries.m`: exterior only, closed lists, single pixel → 2 identical points, spurs traversed twice; DIPUM
   `fchcode.m`/`minmag` **errors** on periodic codes and single pixels — the port breaks ties instead (ch5 caveat).
7. `rgb2gray`: NTSC 0.298936/0.587043/0.114021 in double + half-away-from-zero rounding (skimage Rec.709 is wrong).
8. `round` = half away from zero → `matlab_round`; `uint8(x)` = round + saturate on 0–255 values →
   `to_uint8_saturating`; `im2uint8` = ×255 first → `im2uint8`; numpy `astype(uint8)` wraps/truncates — never use it.
   `im2uint8(uint16)` = `round(v/257)` (`threshold._im2uint8_any`).
9. `imhist(logical)` → 2 bins; `imhist(I, n)` bins by `round(v·(n−1)/top)`; floats clipped to [0,1].
10. `imcomplement(double in 0–255)` = `1 − I` (negative values) — MATLAB scripts that `double()` first then complement
    (`color_image.m`) get that; reproduce, do not "fix".
11. `interp2 'cubic'` = Keys `a = −0.5` with quadratic edge extrapolation; `imresize 'bicubic'` differs at borders and
    antialiases when shrinking → `resize` is `approx` there.
12. HSI Eq. (2.6b) uses `atan(V2/V1)` (undefined at `V1 = 0`); `color_image.m` line 21 has `2*Ig` for `2*Ib` (flat hue).
13. MATLAB scripts read `imread('rgb.jpg')` on Windows case-insensitively (`rgb.JPG`): use `load_image`.
14. MATLAB scripts may write files (`saveas`) into the cwd — generate references from a scratch directory.
15. PNG comparisons of float maps: MATLAB's `mat2gray` works in single → ±1 level on a few % of pixels; not a defect.
16. Windows/Anaconda host: `python -m nbconvert --execute ...` (`python -m jupyter nbconvert` may dispatch to
    Anaconda's binary), and the `python3` **kernelspec resolves to Anaconda's python unless `.venv/Scripts` is
    prepended to `PATH`** before the headless run (ch04) — otherwise the notebook executes against the wrong
    site-packages. Set `PYTHONIOENCODING=utf-8` when printing non-ASCII (cp1252 console).
17. Typos in the authors' code are reproduced, not corrected, when the printed figure shows the typo's effect
    (`chain_diff.m` `'cww'` → clockwise); book-equation behaviour is the default when the text is the authority
    (`rgb2hsi` default follows Eq. 2.6a, `matlab_bug=True` reproduces the script).
18. `graythresh` = `im2uint8` → `imhist(256)` → `otsuthresh` with **tie averaging** (`mean(find(σB² == max))`) →
    `level·255` may be a half-integer; `em` = η(t*). `im2bw` compares in double with strict `>` (`uint8 > 104.5`
    confirmed). `skimage.filters.threshold_otsu` has no tie averaging — never use it for parity. `graythresh` does
    **not** convert RGB (histograms all planes); `im2bw` does.
19. `multithresh`'s `getpdf` normalises in **single** and `grayto8` multiplies by 255 in single too — a float64
    product moves a bin (212.4999949 → 212.5 → 213). `N ≥ 3` is `fminsearch` (local; can return its initial guess with
    `metric = −Inf`) → the exhaustive port is `reimplemented`, never `exact`. `t1 = 0` is a candidate for every `N`.
    Degenerate inputs: `getDegenerateThresholds` only when `numel(unique(A)) ≤ N`; int16 saturates (`approx`).
20. **Toolbox-name shadowing**: `ch3/kmeans.m` is the authors' own deterministic histogram k-means, not the Statistics
    Toolbox `kmeans` that ch6/7/9 call. Check each chapter folder for files named like toolbox functions; never
    `addpath` chapter folders when generating references.
21. **Authors' k-means units bug** (`kmeans.m` 64–70: unshifted image vs shifted centroids) — every book k-means number
    needs `kmeans_gray(shift_bug=True)`; an empty cluster loops forever in MATLAB (`NaN == NaN` false) — port raises.
22. **Stale loop buffers** (`s(j) = I(p(j))` never cleared in `Otsu.m`/`kmeans.m`): wrong class means whenever a class
    is smaller than an earlier one (MATLAB prints 4759.10 on 1.jpg). `sum(uint8 vector)` returns **double** (no
    saturation); `find` fills column-major. Emulated by `stale_mean_intensity`; library returns correct means.
23. `num2str(x)` scalar: integers `%d`; else `%.{n}g`, `n = max(floor(log10|x|) + 5, 5)` (`73.8472`, `86.496`,
    `0.12346`). Use `_num2str` for byte-identical titles.
24. 1-based bin index used as intensity (`separability.m` `i = 1..256`): shift-invariant σB²/σG²/η are right but the
    class split is off by one (script `eta` = η(k − 1) while the mask is `I > k`). Port literally, expose the book form.
25. `colormap('default')` after `imshow(double)` → parula rendering in MATLAB; the book prints gray — display only.
26. Analysis pre-checks with "independent" methods are not references (pre-check said 79, MATLAB 78 for multi-Otsu on
    1.jpg; ch4's replicate-padded thinning matched 12 Mpx by luck and failed at constructed ties): every reported
    number comes from the MATLAB run, and `compat` fixtures must contain **constructed edge cases** (ties, borders,
    intmax/intmin), not only random fields.
27. Inventory watchlist (`analysis/_matlab_inventory.md`) misses `multithresh`, `imquantize`, `colormap`, `num2str` —
    grep the `.m` files yourself during analysis.
28. **MATLAB `edge` (sobel/prewitt) ≠ Eq. 4.2 > T**: `fspecial/8` (`/6`) kernels, `'replicate'` padding,
    `b = bx² + by² > T²` (auto `4·mean(b)`, `thresh = sqrt`), then `computeEdges` **thinning** on a **zero-padded** `b`:
    `(|bx| >= kx·|by| − 100eps & b(r,c−1) <= b & b > b(r,c+1)) | (|by| >= ky·|bx| − 100eps & b(r−1,c) <= b & b > b(r+1,c))`
    — multipliers on the *other* component, `<=` left/up, strict `>` right/down; Prewitt filters with the integer
    kernel **then** divides. `skimage.filters.sobel`/`feature.canny` are unrelated → `core.edges.edge`.
29. `edge('log')` thresholds the **jump across the zero crossing** (`|b(p) − b(q)| > T`, `2T` at exact zeros; auto
    `0.75·mean|b|`), negative side, interior only; `near` on piecewise-constant synthetic images (rounding-noise signs).
30. `double(im)/256` (the authors' idiom) ≠ `im2double` (/255): the book's `T = 0.05` and auto thresholds depend on it —
    port literally; `core.edges.edge` refuses integer input so the scaling is always explicit.
31. `fspecial('log')` = unit-sum Gaussian × `(x²+y²−2σ²)/σ⁴`, mean-subtracted (not Eq. 4.15 sampled); `'gaussian'`
    zeroes `< eps·max` entries; default sizes 3×3/0.5 (gaussian), 5×5/0.5 (log), `2·ceil(2σ)+1` when only σ is given
    (`edge('log')` uses `2·ceil(3σ)+1`); `'laplacian'(α)` divides by `α + 1`.
32. **`strel('disk', r)` is an octagon** (n = 4 periodic-line decomposition): r = 5 → 9×9/69, r = 7 → 13×13/157,
    r = 15 → 29×29/697; Euclidean disks (`skimage.morphology.disk`, `cv2.MORPH_ELLIPSE`, `n = 0`) are a different shape.
    `strel` accepts unambiguous prefixes (`'dis'`; `'di'` errors). Origin `floor((size+1)/2)` also for even SEs.
    Decomposed sequences give identical results (Minkowski sum) and are faster on 12-Mpx images.
33. Erosion pads with 1 / intmax / +Inf, dilation with 0 / intmin / −Inf **and reflects the SE** (Eqs. 4.18/4.21) —
    decisive for asymmetric/even SEs (OpenCV's `erode`/`dilate` are non-reflected; flip the kernel and remap the anchor).
34. **`imclose` pre-pads** by `ceil(size(nhood)/2)` — 0 via `imclose.m`, class minimum via the Halide kernel (nnz < 600 ∧
    every side ≤ 15 ∧ not an all-ones rectangle; `images.UseHalide` default on) — then crops; `imopen` does not.
    Identical for uint8/logical/non-negative double; differs for negative values. 43/54 closings failed before this.
35. `scipy.ndimage.minimum_filter/maximum_filter` route `cval` and accumulation through float64 → wrong for int64/uint64
    (intmax pad → intmin; values > 2⁵³) — use numpy pad + shift-and-reduce in the native dtype. MATLAB R2025a
    **rejects int64/uint64** in `imerode`/`imdilate` → those dtypes are L1-only; cast `label_components` output (int64)
    to int32 before any MATLAB comparison.
36. `logical − logical` → **double** 0/1; `uint8 − uint8` saturates at 0 (`_matlab_minus`); `imreconstruct` requires
    `marker <= mask` (errors otherwise) and a 3×3 conn; reconstruction by erosion has no builtin (`reimplemented`).
37. `bwareaopen(BW, P)` keeps `>= P` pixels (8-conn); `skimage.remove_small_objects` changed its threshold parameter in
    0.26 → `connectivity.bwareaopen`. `medfilt2` zero-pads (`'symmetric'` = scipy `reflect`).
38. "Figure X of Figure 4.3(a)" = **crop first, then process** (border effects within r px / 1 px of the crop edge);
    locate unshipped crops by NCC against the PDF bitmap (stored inverted) at full resolution.

## Data inventory
| file | chapter | tier | shows |
|---|---|---|---|
| `data/book/ch02/rgb.JPG` (2048×1536×3, 723 730 B) | ch02 | 1 | sea-ice colour photo; Fig 2.3 pixel (1076,675) = [28,76,114]; gray peak 47 840 @ 208 |
| `data/book/ch03/1.jpg`, `2.jpg`, `test.jpg` (4290×2856×3; 575 592 / 980 482 / 2 342 607 B) | ch03 | 1 | Figs 3.9(a)/3.10(a)/3.11(a) Ny-Ålesund 2011: Otsu t* 123/107/182, IC 15.36/32.05/72.63 %; `2.jpg` doubles as the labelled substitute for the unshipped `ch3ice.jpg` (Fig 3.2(a)) |
| `data/book/ch04/test.jpg` (4290×2856×3, 905 231 B — a different image from ch03's `test.jpg`) | ch04 | 1 | floe field; Otsu t* 113, IC 30.98 %; Fig 4.3(a) = crop `[1599:2151, 1978:2552]` (MATLAB `1600:2151, 1979:2552`), source of Figs 4.3/4.6/4.9/4.10/4.15/4.16 |
| `data/synthetic/ch03/t_uneven_2_g0.5_b40.jpg` (git-ignored, regenerated by `scripts/ch03_local_otsu.py`) | ch03 | 3 | `2.jpg` × illumination ramp — substitute for the unshipped `t.jpg` (Fig 3.4(a)) |
| `seaice/core/synth.py` fixtures | ch02–ch04 | 3 | Fig 2.10/2.11/2.12/2.19/2.21 matrices, 201×201 point image, spur shape, set rectangles; `bimodal_image`, `two_clusters_2d`, `uneven_illumination`; Fig 4.8 matrices, `two_floes_profile`, `two_blobs_with_marker` |
| `reference/ch02/*.mat` (8), `reference/ch03/*.mat` (13), `reference/ch04/*.mat` (6 + `inputs.mat`) | ch02–ch04 | MATLAB refs | ch02: histogram, color_image, distance_transform, chain_diff, boundaries_multi, conv2, interp2, compat; ch03: otsu_{test,1,2}, kmeans_{test,1,2}_k{2,3}, local_otsu, separability_{107,108}, compat; ch04: derivative, morphology, crop_fig4_3, compat (816 vars), imclose_pad, review_followup |
| Public-domain substitutes (`seaice/core/public_images.py`, `data/online/SOURCES.md`) | ch02–ch04 | 2 | NASA Worldview MODIS/Terra 2019-07-25 true colour: ch02 `rgb.jpg` 2048×1536 Beaufort MIZ; ch03 `1.jpg`/`2.jpg`/`test.jpg` at 4290×2856 (sparse ~16 % / ice edge ~30 % / pack ~70 % bright); ch04 `test.jpg` 4290×2856 pack ice 75–77°N 150–147°W (dense floes with leads) — same sizes as the book files so block partitions, crops and pixel indices stay valid |
| Fig 2.7 grayscale floe-field image | ch02 | missing | not shipped; `rgb2gray(rgb.JPG)` used as labelled substitute |
| `ch3ice.jpg` (Fig 3.2(a), OMAE-2012 image), `t.jpg` (Fig 3.4(a)) | ch03 | missing | not shipped, no public source → Figs 3.2–3.5/3.7 numbers unverified |
| §4.3 pack-ice image (Fig 4.17(a)) and its connected-floe crop (Fig 4.18(a)) | ch04 | missing | not shipped, no public source (NCC ≤ 0.24 vs shipped JPEGs) → Figs 4.17–4.20 procedure-only on `test.jpg` |
| Book-shipped images ch5–ch10 (`data/book/chNN/`) | ch05+ | 1 | copied by `/setup-project`; see `analysis/_matlab_inventory.md` |
| `dypic_05100_cam1_top.avi` (ch9 movie scripts), raw JPEG behind ch8 `MCD/*.mat` | ch08/ch09 | missing | see `data/online/SOURCES.md` |

## Parity summary per chapter
| chapter | exact | near | approx | reimplemented | unverified | verdict |
|---|---|---|---|---|---|---|
| ch02 Preliminaries | 23 | 1 (`bwdist` quasi-euclidean) | 1 (`resize` bicubic border / shrink) | 4 (HSI, CMYK, §2.3 text-only, `min_magnitude` tie-break) | 1 (Fig 2.7 source image) | PASS — 86 tests, 7/7 `.m` exact vs MATLAB R2025a |
| ch03 Ice pixel detection | 16 | 0 | 1 (`multithresh` int16) | 5 (`multithresh` N=3, `kmeans_gray` consistent units, Lloyd/distances/2-D demo, synthetic fixtures) | 1 (Figs 3.2–3.5/3.7: `ch3ice.jpg`/`t.jpg` not shipped) | PASS — 112 tests + 1 xfail, 4/4 `.m` exact vs MATLAB R2025a, 14/14 image pairs 0 px, all shipped-image book numbers reproduced |
| ch04 Ice edge detection | 22 | 1 (`edge('log')` on piecewise-constant synthetic input) | 0 | 6 (book-form gradient/Laplacian/zero-crossing/kernels, reconstruction by erosion, geodesic loops) | 1 (Figs 4.17–4.20: §4.3 image not shipped) | PASS — 548 tests + 2 by-design skips (746 total), 2/2 `.m` exact vs MATLAB R2025a (`derivative.m` 0 px on 12.25 Mpx; `morphology.m` 12/12 arrays), 27/27 image pairs 0 px, `edge` 252 maps / `strel` 116 nhoods / erode-dilate 148 / open-close 126 / reconstruct 7 cases 0 px; review 13/13 applied |

## Setup findings (2026-09-09, /setup-project)
- Reference engine: **MATLAB R2025a** (25.1.0.2833191, prerelease) via `tools/run_matlab_ref.py`; Octave absent (fine).
- PDF offset = 31 (pdf 0-based index = printed page − 1 + 31); each `chapters/chNN.txt` starts on its chapter title.
- `MATLAB_ROOT/ch10` = Appendix A (`fisheye_calibration.m` → A.2 lens distortion, `orthoretification.m` → A.1).
- `ch6/Sea_Ice_Floe_Identification` and `ch7/Sea_Ice_Floe_Identification` hold the same 24 `.m` files (byte-identical);
  only `sea_ice_test.jpg` differs. Port once (ch6), reuse in ch7. `ch9/Model_Ice_Floe_Identification` shares the GVF/snake
  files too. ch5 duplicates ch2's `bound2im.m` / `boundaries.m` / `fchcode.m` (byte-identical, md5 verified in ch02).
  ch9 `block_threshold.m` = ch3 `local_Otsu.m` (→ `threshold.block_otsu`).
- Data gaps: ch9 movie scripts need `dypic_05100_cam1_top.avi` (not shipped); ch8 MCD ships only `.mat` results.
  See `data/online/SOURCES.md` (git-ignored, on disk).
- R2025a toolbox sources worth reading before porting a builtin: `edge.m`, `fspecial.m`, `imclose.m`/`imopen.m`,
  `private/morphop_fast.m`, `+images/+internal/+coder/+strel/StructuringElementHelper.m` (ch04); compiled builtins
  (`computeEdges`) must be reverse-engineered and pinned with constructed-tie fixtures.

## Publishing findings (2026-09-09, ch02/ch03 PUBLISH phase)
- Public repo `shammun/seaice-py` (branch `main`), GitHub Pages at `https://shammun.github.io/seaice-py/`. History was
  rewritten with git-filter-repo to purge every book-derived file; `chapters/*.txt`, `data/book/`, `reports/**/figures/`,
  `outputs/` stay local and git-ignored (CLAUDE.md rule 12).
- Data: `seaice.core.io.load_image()` (private copy → NASA public-domain substitute from `seaice/core/public_images.py`);
  substitutes are requested from the Worldview snapshot API at the book image's exact size (2048×1536 for ch02,
  4290×2856 for ch03/ch04 — the API accepted that size).
- Notebook cells 1–3 template lives in `notebooks/build_ch02.py` / `build_ch03.py` / `build_ch04.py`;
  `tools/publish_notebook.py chNN` builds the Colab variant, the styled HTML page (from a run with `data/book` renamed
  away), `index.html` and the README table. Scripts/tests must tolerate a missing `data/book` (print `SKIP`, exit 0;
  `test_script_runs` skips the file count). Crop constants (`FIG_4_3A_CROP`) stay valid on the same-size substitute but
  the notebook must say the crop is only meaningful on the book image.
