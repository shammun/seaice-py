# CUMULATIVE knowledge — seaice-py
(Rewritten by the knowledge-keeper after every chapter. Read this first at the start of any chapter. Last rewrite:
after **ch06**, 2026-09-10. Per-chapter detail: `knowledge/chNN.md`; verified call mappings: `knowledge/function_map.md`.)

## Pipeline so far
```
Ch2 primitives (DONE)   Ch3 ice mask (DONE)   Ch4 edges + morphology (DONE)   Ch5 watershed floes (DONE)   Ch6 GVF snake (DONE)   Ch7 types  Ch8/9 apps
  io.load_image (recursive since ch06) ─► rgb2gray_matlab ─► threshold.graythresh/im2bw ─► binary ice mask ─► every later .m
  matlab_compat.{rgb2gray_matlab,      │   (bit-exact vs MATLAB on 12-Mpx JPEGs; graythresh(RGB) ≠ graythresh(gray) — ch5)
    imcomplement, matlab_round,        ├─► threshold.multithresh(N=2)+imquantize ─► 3 groups (Ch7 ice types, Ch8)
    im2double, im2uint8,               ├─► threshold.block_otsu(n_r, n_c) ─────────► Ch9 block_threshold.m (same code)
    to_uint8_saturating, del2,         ├─► clustering.kmeans_gray (authors' ch3 kmeans.m) ─► Ch9 movie_kmeans.m
    saturate_to_class}                 └─► clustering.kmeans_lloyd('kmeans++') = Statistics-TB kmeans ─► Ch6/7/9
  histogram.{imhist, normalized_histogram} · threshold.{ice_concentration, class_coverage, class_mean_intensity} ─► Ch8/Ch9 IC series
  connectivity.{label_components (= bwlabel), bwareaopen, bwperim} ─► Ch4 speck removal, Ch5 basins/lines, Ch6 regionprops, Ch7–9
  distance.{bwdist (single!), distance_transform} ─► Ch5 −bwdist(~bw) ─► watershed ─► Ch6 §6.3 seed/radius init ─► Ch7–9
  filters.{conv2, imfilter, fspecial, homomorphic_butterworth} ─► Ch4 edge() ─► Ch5 Sobel surface, Ch6 GVF edge map
  edges.edge (sobel/prewitt/roberts/log/zerocross, thinning) ─► Ch4 Figs 4.3/4.6, Ch6 edge overlays, Ch9 outlines
  morphology.{strel (octagon disk), imerode/imdilate, imopen/imclose, imreconstruct, imregionalmin/max, imimposemin,
      regional_maxima_by_reconstruction} ─► Ch4 Figs 4.9–4.20 ─► Ch5 markers ─► Ch6 seed dilation ─► Ch7 shape enhancement
  watershed.watershed (MATLAB Meyer flooding, exact incl. label values) ─► Ch5 all scripts ─► Ch7 floe separation, Ch8 FSD, Ch9
  chaincode.{boundaries, fchcode, bound2im} ─► Ch5 differential chain code ─► neighboring_region_merging ─► Ch7
  interp.{interp2, warp_image, resize} ─► Ch6 snakedeform's interp2, App. A orthorectification
  snake.{bound_mirror_*, gvf, gradient2, xconv2, gaussian_mask/blur, snake_matrix, snakedeform, snakeinterp, snakeindex}
      ─► Ch6 GVF snake ─► **Ch7 (23 byte-identical .m) and Ch9 (13 byte-identical .m) reuse this unchanged**
  regionprops.regionprops (MATLAB moments/hull/perimeter) ─► Ch6 Ra/Rc/Rl criteria ─► Ch7 shape metrics, Ch8 FSD, Ch9
  polygon.{poly2mask, roipoly, convhull, polyarea, polyxpoly, polygeom, minboundrect, clip_polygon_rect}
      ─► Ch6 clips ─► Ch8 §8.2 sea_ice_model + App. B, Ch9 §9.3 rectangularization
  setops.* · color.* · plotting.{…, label2rgb, surface_plot, contour_overlay, snake_plot, quiver_field} · synth.* · cli.*
Ch7 ice types (owns ch6's deferred ice_shape_enhancement.m) → Ch8 applications (owns sea_ice_model.m,
SeaIce_Image_Structure.m, color_hist*.m) → Ch9 model ice (reuses the GVF stack + minboundrect) → Ch10 = App. A calibration
```
Conventions fixed in ch02–ch06 and binding for all later chapters:
- Coordinates: book `x` = row, `y` = col (Eq. 2.1). Python is 0-based `(row, col)` everywhere; tests add 1 to compare
  with MATLAB. `ChainCode.x0y0_matlab` is the 1-based pair. Crops quote both forms (`FIG_4_3A_CROP` / `_MATLAB`).
  `regionprops` `Centroid` / `component_centroids` are MATLAB `(x, y)` 1-based; `find` results are column-major.
- Numerics stay float64 **except distance maps, which stay float32** (MATLAB `bwdist` is single and watershed
  priorities / regional-minima ties depend on it), and **except any value MATLAB computes in another class**: a
  `single op double` promotes to double and rounds to single **once** (ch06, 540/540 radii), and integer-class
  arithmetic rounds+saturates at every step (`matlab_compat.saturate_to_class`). uint8 conversion only at display/save
  time via `plotting.*` or `to_uint8_saturating` / `im2uint8`. Otsu levels stay **float** (tie-averaged half-integers).
  Morphology preserves the input dtype; `core.edges.edge` takes **float only**; label images are int32.
- Every `.m` maps to one Python script (`scripts/chNN_<name>.py`) or one `seaice` function; docstrings cite §/Eq/`.m`
  **and the measured parity label** — re-check every docstring claim against the verification report at chapter end
  (ch06 review found five stale labels and two stale numbers).
- Rule 9 (reuse before re-implementing) is easiest to break when a new core primitive lands: ch06 wrote
  `core.regionprops` and had to rewire ch05's `component_centroids` onto it (review S8). Grep for duplicate logic.
- References: MATLAB R2025a via `tools/run_matlab_ref.py`, driven by `reference/chNN/make_refs.py`; `.mat` files in
  `reference/chNN/`; tests skip with a clear message if a `.mat` is missing. Run `.m` scripts from a scratch cwd
  (`outputs/chNN/verify/scratch`) under **non-shadowing** names (ch06 `dist.m` shadows the Deep Learning Toolbox
  `dist`), patch only literals, never `addpath` a chapter folder. Controlled fixtures (`inputs.mat`) are written from
  Python with **constructed ties / intmax-intmin plants / asymmetric SEs / order-discriminating corridors and
  transposes / integer-class inputs**; MATLAB's own refusals are recorded with `try/catch` as evidence. Per-iteration
  loop variables are captured by accumulator lines appended *after* the verbatim statements. Shipped `.fig` files are
  MAT v5 → `openfig(..., 'invisible')` gives an L3 truth for free.
- Figures: `outputs/chNN/fig_<ch>_<fig>_<slug>.png` only for genuine book figures; `sec_<ch>_<sec>_<slug>.png` otherwise.
- Authors' script bugs: the library default follows the book text/equations; the literal script behaviour is available
  behind an explicit flag (`rgb2hsi(matlab_bug=True)`, `kmeans_gray(shift_bug=True)`, `stale_mean_intensity`,
  `junction_endpoints(rule='max')`, `neighboring_region_merging(sequential=True)`, `gvf_force_field(normalize=True)`,
  `homomorphic_butterworth(matlab_bug=True)`) and every book number that depends on it says so. Literal idioms that are
  not bugs (`double(im)/256`, `'cww'`, the absolute `imread` path of `dist.m`) are ported as is.
- Scripts that need book images print `SKIP …` and exit 0 when `data/book/chNN` is absent (public/Colab runs);
  non-default CLI flags get one parametrised test each (`test_script_cli_flags`).
- Compiled toolbox builtins: look for the readable codegen twin `toolbox/images/images/eml/<name>.m` first (ch5
  `watershed`, ch6 `poly2mask`: 0 px line by line); reverse-engineer only when there is none (ch4 `computeEdges`); when
  even that is impossible (ch6 `polybool` = GPC `gpcmex`), compare **sets and rasterised masks**, never order.

## Available primitives in seaice/core/
| function | module | book § | used by chapters | parity |
|---|---|---|---|---|
| `load_image(chapter, name)` (case-insensitive `imread`, **recursive into sub-folders since ch06**; private copy first, public-domain NASA substitute as fallback, blank-download retry), `repo_root`, `book_data_dir`, `output_dir`, `fetch` | `io` | — | all | exact (JPEG decode identical to MATLAB on every book image tried) |
| `chapter_argparser`, `resolve_dirs` | `cli` | — | all scripts | — |
| `rgb2gray_matlab`, `imcomplement`, `matlab_round`, `im2double`, `im2uint8`, `to_uint8_saturating` | `matlab_compat` | §2.1–2.2, Eq. 2.3 | ch3–ch10 | exact |
| `del2(f, hx=1, hy=None)` (`∇²/(2·ndims)`, borders linearly extrapolated, `n == 3` copy, `n ≤ 2` → 0) | `matlab_compat` | Eq. 6.52c via `GVF.m` | ch6, ch7, ch9 | exact (21 cases 0.0) |
| `saturate_to_class(x, dtype)` (one MATLAB integer-class op: round half away from zero, then saturate) | `matlab_compat` | — | ch6 `GVF`/`gradient2`, ch7+ | exact |
| `imshow_scale`, `to_display_uint8`, `save_image`, `imshow_matlab`, `show_matrix`, `finish_figure`, `label2rgb`, `surface_plot`, `contour_overlay`, `snake_plot`, `quiver_field` | `plotting` | display | all | display only |
| `split_rgb`, `rgb2cmy`, `rgb2cmyk`, `rgb2hsi`, `indexed_to_rgb` | `color` | §2.1, Eqs. 2.2–2.6 | ch6/ch7 colour stats, ch8 | exact / reimplemented (HSI, CMYK) |
| `imhist(img, nbins=None)`, `normalized_histogram` | `histogram` | §2.2, Eqs. 2.7–2.8 | ch3, ch6+ | exact (logical → 2 bins) |
| `n4/nd/n8`, `is_adjacent`, `is_m_adjacent`, `find_paths` | `connectivity` | §2.3.1–2.3.3 | teaching only | reimplemented |
| `label_components(bw, conn=8)` (= `bwlabel` incl. numbering, int32), `count_components` | `connectivity` | §2.3.4 | ch4–ch9 | exact |
| `bwareaopen(bw, P, conn=8)` (keeps ≥ P px) | `connectivity` | §2.3.4 | ch4, ch5, ch6 (`(…,10,4)`), ch7, ch9 | exact |
| `bwperim(bw, conn=4)` (= `bwperim`/`bwmorph('perim8')`; supersedes `region_boundary_mask`) | `connectivity` | §2.3.5 | `regionprops.ConvexHull`, ch7 outlines | exact (9 shapes × {4, 8}) |
| `bwdist(bw, metric)` (**returns float32**), `distance_transform`, `quasi_euclidean_dt`, `pixel_distance`, `center_distance_map` | `distance` | §2.4, §5.1.2 | ch5 inverse distance maps, ch6 seeds/radii | exact (eucl/city/chess), near (quasi) |
| `conv2(f, w, mode='same')`, `imfilter(f, w, *matlab_options)`, `conv_at` | `filters` | §2.5, Eq. 2.14 | ch4–ch6 | exact |
| `fspecial(kind, …)` (sobel/prewitt/laplacian/gaussian/log/average/disk/unsharp, prefixes) | `filters` | §4.1, Eqs. 4.14–4.15 | ch4–ch7 | exact (26 kernels ≤ 7e-18) |
| `homomorphic_butterworth(im, d, n, …, matlab_bug=True)` | `filters` | — (ch6 orphan `homofil.m`) | none known | exact (≤ 1.14e-12) |
| `edge(a, method, thresh, direction, thinning, sigma, H) -> EdgeResult`, `gradient_sobel_prewitt`, `gradient_roberts`, `thin_gradient`, `log_zero_crossings` | `edges` | §4.1 | ch4, ch6, ch9 | exact (252 + 14 maps 0 px) |
| `strel(shape, *params, n=None)` (disk = octagon; `'disk',3` = **5×5/25**, `'disk',5` = 9×9/69), `disk_decomposition`, `minkowski_sum`, `periodic_line`, `line_strel`, `intline`, `se_origin` | `morphology` | §4.2, Fig. 4.7 | ch4–ch9 | exact (116 nhoods) |
| `imerode`, `imdilate`, `imopen`, `imclose` (MATLAB pad/reflect/pre-pad rules; dtype preserved) | `morphology` | §4.2.1–4.2.2, Eqs. 4.16–4.23 | ch4–ch9 | exact (54 + 72 + 148 cases) |
| `imreconstruct(marker, mask, conn=8)`, `reconstruct_by_erosion`, `geodesic_dilation/erosion`, `reconstruct_iterative` | `morphology` | §4.2.3, Eqs. 4.24–4.38 | ch5 `imimposemin`, ch6 Eq. 6.57, ch7 `imfill` | exact / reimplemented (erosion dual) |
| `imregionalmin/imregionalmax(I, conn=8)` (constant image → all True), `conn_to_scalar` | `morphology` | §5.1, Figs. 5.8/5.12 | ch5, ch6 seeds, ch7, ch9 | exact (126 cases) |
| `imimposemin(I, BW, conn=8)` (arithmetic in the input class, ∓Inf markers) | `morphology` | §5.1.3 | ch5, ch7/ch8 markers | exact (88 bit-identical) |
| `regional_maxima_by_reconstruction(I, conn=8, form='6.57'\|'6.58')` | `morphology` | Eqs. 6.57/6.58 | ch6 teaching, ch7 | reimplemented (== `imregionalmax`) |
| `morphological_gradient(I, se, kind)` | `morphology` | §4.2.4, Eqs. 4.39–4.42 | ch4, ch5, ch7 | exact |
| `watershed(A, conn=8) -> int32` (Meyer flooding, `eml/watershed.m` line by line), `watershed_skimage` (approx cross-check), `neighbour_offsets` | `watershed` | §5.1, Eqs. 5.1–5.8 | ch5, ch7–ch9 | exact **with label values** (74 cases × {8,4}) · ≈ 3 s/Mpx |
| `complement/union/intersection/difference`, `reflect`, `translate`, `gray_*`, `bitwise_*`, `truth_tables` | `setops` | §2.6, Eqs. 2.16–2.30 | ch4, ch7 | exact |
| `boundaries`, `fchcode`, `first_difference`, `min_magnitude`, `bound2im`, `chain_to_points` | `chaincode` | §2.7, §5.2.1 | ch5, ch6 | exact (tie-break reimplemented) |
| `interp2(img, u, v, method)` (Keys `a = −0.5` + quadratic edge extrapolation; `fill=` for `'*linear',0`), `warp_image`, `resize`, `keys_kernel` | `interp` | §2.8, Eqs. 2.33–2.41 | ch6 `snakedeform`, ch10 | exact (borders incl.) |
| `bound_mirror_expand/ensure/shrink(A)` | `snake` | Eq. 6.46 | ch6, ch7, ch9 | exact |
| `gvf(f, mu, iters, check_cfl=True)` (**normalises `f` in the input's class**; CFL guard) | `snake` | Eqs. 6.41, 6.50–6.55 | ch6, ch7, ch9 | exact (44 cases 0.0) |
| `gradient2(a, xax, yax)`, `gradient2_complex`, `gradient2_magnitude` (input-class differences) | `snake` | Eq. 6.7 | ch6, ch7, ch9 | exact |
| `xconv2(I, G)`, `gaussian_mask(k, s)`, `gaussian_blur(I, s)` | `snake` | Eqs. 6.10/6.13 | ch6, ch7, ch9 | exact (≤ 9.1e-13 / 1.4e-20 / 6.4e-14) |
| `snake_matrix(N, α, β, book_index=)`, `snake_first_column`, `snakedeform(..., solver='auto'\|'dense'\|'circulant')` (FFT for `N ≥ 32`: 306 s → 2.1 s) | `snake` | Eqs. 6.34–6.40 | ch6, ch7, ch9 | exact (dense) / near (circulant ≤ 4.1e-13) |
| `snakeinterp(x, y, dmax, dmin, max_passes=100)`, `snakeindex(idx)` | `snake` | §6.1.2 | ch6, ch7, ch9 | exact (16/18; the 2 others are the header's own bug where MATLAB errors) |
| **`regionprops(L_or_bw, properties, conn=8) -> list[RegionProps]`**, `region_table` (MATLAB moments +1/12, `perim8` mid-edge hull, Vossepoel–Smeulders perimeter, `roipoly` convex image; both call forms) | `regionprops` | ch9 p. 205 criteria, §7.1/§8.2 | ch6, ch7, ch8, ch9 | exact (≤ 1.07e-14, 40 shapes × 10 props + 344 real components) |
| `poly2mask(x, y, M, N)`, `roipoly(m, n, xi, yi)` (`eml/poly2mask.m` line by line) | `polygon` | §8.2, `regionprops` | ch6, ch8, ch9 | exact (19 masks 0 px) |
| `clip_polygon_rect(x, y, x_range, y_range)` (Sutherland–Hodgman, closed ring = `polybool`'s contract) | `polygon` | §6.5.3 | ch6, ch7, ch9 | reimplemented (sets/masks identical; order not reproducible) |
| `convhull`, `polyarea`, `polyxpoly`, `polygeom`, `minboundrect(x, y, metric='a')` | `polygon` | §8.2, §9.3 | ch6, ch8, ch9 | near / exact (≤ 1.85e-13) / reimplemented / near (`ang1` ±π) / exact (≤ 1.8e-15) |
| `graythresh -> (level, em)`, `otsuthresh`, `im2bw`, `multithresh`, `imquantize`, `otsu_criterion`, `separability`, `block_otsu`, `ice_concentration`, `class_coverage`, `class_mean_intensity`, `_im2uint8_any` | `threshold` | §3.1, Eqs. 3.1–3.28 | ch4–ch9 | exact (multithresh `N = 3` reimplemented, int16 approx) |
| `kmeans_gray(gray, k, shift_bug=False)` (authors' ch3 `kmeans.m`), `kmeans_lloyd(X, k, init, seed, …)` (= Statistics-TB `kmeans`), `objective_J`, `pairwise_distance` | `clustering` | §3.2.2, Eqs. 3.29–3.37 | ch3, ch6 (`approx`), ch9 | exact (`shift_bug=True`) / approx (toolbox mapping) |
| `point_image`, `spur_shape`, `set_operation_fixtures`, `book_fixtures`, `FIG_2_*`, `uneven_illumination`, `two_clusters_2d`, `bimodal_image`, `FIG_4_8_*`, `two_floes_profile`, `two_blobs_with_marker`, `two_touching_floes`, `plateau_fixtures`, `FIG_5_15_*`, `FIG_5_16_*`, `FIG_6_14_IMAGE/_DISTANCE`, `fig_6_16_circles`, `u_shape`, `synthetic_floe_field` | `synth` | Figs 2.10–6.16 | tests ch2–ch6, public fallbacks | exact (printed truths) / synthetic |
| Chapter modules: `ch03_ice_pixel_detection` (`_num2str`, `stale_mean_intensity`, `separability_script`), `ch04_ice_edge_detection` (book-form gradients/Laplacian/kernels, `FIG_4_3A_CROP`), `ch05_watershed` (`otsu_mask`, `inverse_distance`, the 6 watershed pipelines, `freeman_concave`, `neighboring_region_merging`), `ch06_gvf_snake` (`BOOK_PARAMS`, energies, `traditional_snake`, `gvf_force_field`, `initialize_contours`, `component_criteria`, `gvf_distance`, `seaice_kmean_gvf`) | — | per chapter | later chapters call these directly | exact (scripts) / reimplemented (text forms) |

Not yet in core (first needed by): `imfill` on `imreconstruct` with a border seed, `bwmorph` LUTs (ch7);
a fast exact `watershed` engine for 12-Mpx frames (ch7–ch9); `imresize_matlab` with antialiasing (ch9 if needed);
DLT / lens distortion (ch10). Deferred `.m` files owned by later chapters: `ice_shape_enhancement.m` → ch7 §7.1;
`sea_ice_model.m`, `SeaIce_Image_Structure.m`, `color_hist.m`, `color_hist_comparison.m` → ch8 / Appendix B
(`scripts/ch06_sea_ice_demo.py --full` raises `NotImplementedError` until they land).

## Global pitfalls (MATLAB → Python) confirmed in this project
1. Book `x` = row, `y` = col; MATLAB 1-based; Python 0-based `(row, col)`. `interp2(X, Y, Z, Xq, Yq)` has `X` = column:
   Python `interp2(img, u=Yq−1, v=Xq−1)`. `regionprops` `Centroid` is `(x, y)` = (col, row) means, 1-based.
2. `bwdist(BW)` = distance to nearest **nonzero** = `distance_transform(~BW)`; returns **single** → `atol≈1e-4`, and
   keep the float32 array when it feeds `watershed`/`imregionalmin` (float64 shifts values by ~1e-6 and flips ties).
3. **`single op double` promotes to double and rounds to single ONCE** (ch06, `dist.m`'s 540 radii: promote-round-once
   540/540 bit-exact, rounding `sqrt(2)` to single first 360/540 up to 1.91e-6 off, pure double 6/540 up to 7.67e-7).
   A naive all-float32 port is 1 ulp wrong, and a downstream `ceil()`/`round()` turns that into moved pixels.
4. **Integer-class arithmetic happens inside library functions too** (ch06 generalises ch05's `imimposemin` lesson):
   `GVF.m` normalises `(f − fmin)/(fmax − fmin)` in the *input's* class, so a uint8 image gives a **{0, 1}** edge map;
   `gradient2` differences in the input class. Push every step through `matlab_compat.saturate_to_class` — the
   intermediate values are *not* the float64 ones. `uint8` saturates (numpy wraps); integer `/` rounds (numpy truncates).
5. `conv2` flips the kernel, `imfilter` does not (correlation); Eq. (2.15) is printed in correlation form; `conv2`'s
   default mode is `'full'`. `imfilter(I, h, 'replicate')` positional strings are accepted as-is; no option = zero pad.
6. `bwlabel` numbers components column-major by first occurrence; `watershed` basins likewise — compare label *values*.
7. DIPUM `boundaries.m`: exterior only, closed lists, single pixel → 2 identical points, spurs traversed twice, start =
   first object pixel column-major; `fchcode`'s `minmag` **errors** on periodic codes. `'cww'` (typo) = clockwise.
8. `rgb2gray` NTSC in double + half-away-from-zero rounding (skimage Rec.709 is wrong); `round` = half away from zero;
   `uint8(x)` = round + saturate (`to_uint8_saturating`); `im2uint8` = ×255 first; `astype(uint8)` wraps — never use it.
9. `imhist(logical)` → 2 bins; `imhist(I, n)` bins by `round(v·(n−1)/top)`; floats clipped to [0,1].
10. `imcomplement(double in 0–255)` = `1 − I`; `imcomplement(uint8)` = `255 − I`. Reproduce, do not "fix".
11. `interp2 'cubic'` = Keys `a = −0.5` with quadratic edge extrapolation; `imresize 'bicubic'`/shrink differ → `approx`.
    `interp2(..., '*linear', 0)` = `core.interp.interp2(..., fill=0.0)`, exact in and out of range.
12. HSI Eq. (2.6b) uses `atan(V2/V1)`; `color_image.m` line 21 has `2*Ig` for `2*Ib` (flat hue).
13. MATLAB reads `imread('rgb.jpg')` case-insensitively on Windows, and **ch06's images live in sub-folders with a
    space in the name** → always `load_image` (recursive since ch06). A public-domain download can come back blank
    (389-byte constant JPEG once from NASA Worldview) → reject zero variance and retry.
14. MATLAB scripts may write files (`saveas`) into the cwd — generate references from a scratch directory, and give the
    copy a **non-shadowing** name (ch06 `dist.m` shadows the Deep Learning Toolbox `dist`).
15. PNG comparisons of float maps: `mat2gray` works in single → ±1 level at exact half-integers. Not a defect.
16. Windows/Anaconda host: `python -m nbconvert --execute` with `.venv/Scripts` **prepended to `PATH`** (the `python3`
    kernelspec otherwise resolves to Anaconda); `PYTHONIOENCODING=utf-8`; Bash heredocs mangle backslashes → write patch
    scripts to files; kill orphaned `MATLABWindow.exe` after `matlab -batch`.
17. Authors' typos are reproduced when the printed figure shows their effect; book-equation behaviour is the default.
18. `graythresh` = `im2uint8` → `imhist(256)` → `otsuthresh` with **tie averaging** → `level·255` may be a half-integer;
    `im2bw` compares in double with strict `>`. **`graythresh(RGB)` histograms all three planes** (130/255 vs 128/255 on
    `q.jpg`) — port each call literally. ch6 always thresholds the **gray** image.
19. `multithresh`'s `getpdf` normalises in **single**; `N ≥ 3` is `fminsearch` (local) → `reimplemented`; int16 saturates.
20. **Toolbox-name shadowing**: `ch3/kmeans.m` is the authors' own histogram k-means; ch6/ch7/ch9 ship **no** `kmeans.m`,
    so those calls are the Statistics Toolbox `kmeans` → `clustering.kmeans_lloyd(init='kmeans++')` (`approx`: centres
    matched to double precision and `bk` 0 px under `rng(0)`, but 20 restarts show the optimum is not unique).
21. Authors' k-means units bug (`shift_bug=True` for every book number); an empty cluster loops forever in MATLAB.
22. Stale loop buffers (`s(j) = I(p(j))` never cleared) → wrong class means; `sum(uint8 vector)` returns **double**.
23. `num2str(x)` scalar: integers `%d`; else `%.{n}g`, `n = max(floor(log10|x|) + 5, 5)` (`_num2str`).
24. 1-based bin index used as intensity (`separability.m`) → the class split is off by one; port literally.
25. `colormap('default')` parula rendering and `label2rgb(...,'shuffle')`'s private RNG stream are display only.
26. Analysis pre-checks are **hypotheses, not references** (ch3 79 vs 78; ch4 thinning matched 12 Mpx by luck; ch5 needed
    25 order-discriminating fixtures; **ch6's §0.5 probe printed two wrong axis lengths and a wrong `strel('disk',3)`
    size, both corrected from MATLAB itself**). Every reported number comes from the MATLAB run.
27. Inventory watchlist misses `multithresh`, `imquantize`, `colormap`, `num2str`, `del2`, `polybool` — grep yourself.
28. MATLAB `edge` (sobel/prewitt) ≠ Eq. 4.2 > T: `fspecial/8` kernels, `'replicate'` padding, `b > cutoff` then
    `computeEdges` **thinning** on a zero-padded `b` (multipliers on the *other* component, `<=` left/up, `>` right/down).
29. `edge('log')` thresholds the **jump across the zero crossing**; `near` on piecewise-constant synthetic images.
30. `double(im)/256` (the authors' idiom) ≠ `im2double` (/255) — port literally; `core.edges.edge` refuses integer input.
31. `fspecial('log')` = unit-sum Gaussian × `(x²+y²−2σ²)/σ⁴`, mean-subtracted; default sizes 3×3/0.5, 5×5/0.5.
32. **`strel('disk', r)` is an octagon**: r = 3 → **5×5/25**, r = 5 → 9×9/69, r = 7 → 13×13/157 (Euclidean disks are a
    different shape). Prefixes accepted; origin `floor((size+1)/2)`.
33. Erosion pads 1/intmax/+Inf, dilation pads 0/intmin/−Inf **and reflects the SE**; OpenCV does not reflect.
34. **`imclose` pre-pads** by `ceil(size(nhood)/2)` (0 via `imclose.m`, class-min via Halide) then crops; `imopen` does not.
35. `scipy.ndimage.minimum_filter/maximum_filter` route through float64 → wrong for int64/uint64. MATLAB **rejects**
    int64/uint64 in `imerode`/`imdilate`, int16/int32 in `watershed`, and **`uint8 ./ double matrix`** in `gradient2.m`
    (so ch6's `GradientOn = 0` + `GVFOn = 0` combination cannot run in MATLAB at all). Record refusals as evidence.
36. `logical − logical` → double; `uint8 − uint8` saturates; `bitand(logical)` is logical, `logical .* logical` is
    double; `imreconstruct` needs `marker <= mask` and a `{0,1}` marker reconstructs to `{0,1}`.
37. `bwareaopen(BW, P)` keeps `>= P` px; `skimage.remove_small_objects` changed its parameter in 0.26. `medfilt2` zero-pads.
38. "Figure X of Figure 4.3(a)" = **crop first, then process**; locate unshipped crops by NCC against the PDF bitmap
    (stored inverted) — that is how ch06 identified `alg_seg_gray.jpg` as Fig. 6.15(a) (NCC 0.9882).
39. **MATLAB `watershed` is Meyer flooding** (`eml/watershed.m`): `bwlabel(imregionalmin)` seeds, heap on
    `(priority, insertion order)`, **column-major initial scan**, ridge pixels push nothing, `max(A(nb), p)` priorities.
    `skimage.segmentation.watershed(watershed_line=True)` differs in ridges *and* partitions → `approx` only.
40. `imregionalmin` = `skimage.local_minima(..., allow_borders=True)` **except a constant image** (MATLAB all-True).
41. `imimposemin`: genuine `−Inf`/`intmin` markers, `h = 0.001·range` (0.1 constant) / 1 integer, `I + h` **in the input
    class**; ±Inf gives `h = Inf` and NaN → `np.errstate(invalid='ignore')`.
42. Watershed ridges are 4-connected staircases on distance maps but **thick on plateaus** — MATLAB identical.
43. `main.m` literalisms (relative `>= max` ending-point rule, `seg` updated inside the loop, `b{1}`).
44. **Test design**: a convex blob cannot exercise "spurious line removed" (use a peanut); pin the default connectivity
    with a fixture where 4 and 8 differ; an **integer-class** path needs an integer fixture (ch06's M1 defect hid behind
    float64 inputs for a whole verification round); settle an ordering question by *enumerating* candidate rules over
    many cases and by re-parameterising your own output (rotate → 56–82 px, reverse → 0 px).
45. **Look for `eml/<builtin>.m`** before reverse-engineering a compiled builtin (`watershed`, `poly2mask`).
46. **Performance**: the pure-Python watershed heap is ≈ 3 s/Mpx → decide on a faster engine before the first 12-Mpx call.
    `snakedeform`'s dense `inv` is O(N³) per resample (306 s); the symmetric-circulant FFT solve is the same system
    (≤ 8.4e-12) at 2.1 s.
47. **`del2` is not the Laplacian**: `∇²/(2·ndims)`, borders linearly extrapolated, `n == 3` copies, `n ≤ 2` → 0,
    `/ndims` even in the 1-D branch. `GVF.m`'s `mu*4*del2(u)` is precisely Eq. 6.52c's 5-point Laplacian.
48. **`regionprops` must be reimplemented, not delegated to skimage**: skimage's axis lengths (10.2935 vs 10.3581),
    perimeter (+5.5 %) and orientation (+90°) move the `Rc = 0.9` / `Rl = 2` decision boundaries. MATLAB's rules:
    second moments **+ 1/12**, `y` negated, degrees; hull from the **mid-edge points** of `bwmorph('perim8')` pixels via
    `convhull(rr, cc)`; `ConvexImage` = `roipoly(M, N, c, r)` with the `firstRow/firstCol` shift; `Perimeter` =
    Vossepoel–Smeulders `0.980·even + 1.406·odd − 0.091·corners`. `regionprops(label == n, …)` is a second call form.
49. **`polybool` (Mapping TB) is compiled GPC with no readable source and its start vertex is not reproducible**
    (0 of 46 clips follow max-y/min-y/max-x/min-x or lexicographic order; it also reverses the traversal). It returns a
    **closed** ring, and with `Dmin = 0` the duplicate survives `snakeinterp`. Compare vertex *sets* and rasterised
    masks. A rotated start vertex shifts `snakeindex`'s insertion parity → the ±1 point-count instability.
50. **Contour point counts are unstable at the 1e-13 level** where an insertion test is discontinuous (`d > dmax`):
    2266 → 2262 points under a 1e-13 perturbation. Report the curve (Hausdorff) and the mask, not the length.
51. **An eigenvector's sign is arbitrary** (`polygeom`'s `ang1` differs from R2025a's `eig` by ±π in 13 of 15 cases; the
    axes are identical). Pick the convention that reproduces the file's own documented self-test.
52. **Old FEX code may not run in current MATLAB**: `minboundrect.m`'s `convhull(x, y, {'Qt'})` is rejected by R2025a
    (`CONVHULL no longer supports or requires Qhull-specific options`), and modern `convhull` errors on collinear or
    duplicate clouds. Record MATLAB's refusal, patch **one literal**, and say so in the report.
53. **A book figure may need an input the book never prints** (ch6 Fig. 6.16 needs the edge map `|∇I|`, not the binary
    image, or no snake converges), and a caption parameter may hide the very effect the figure claims (at Fig. 6.7(b)'s
    σ = 4 the *traditional* snake also enters the concavity — the failure needs σ ≤ 2). Sweep the parameter and print it.
54. **Verify docstring parity labels at chapter end** — ch06's review found five that contradicted the measurements and
    two stale numbers embedded in comments.

## Data inventory
| file | chapter | tier | shows |
|---|---|---|---|
| `data/book/ch02/rgb.JPG` (2048×1536×3) | ch02 | 1 | sea-ice colour photo; Fig 2.3 pixel (1076,675) = [28,76,114] |
| `data/book/ch03/1.jpg`, `2.jpg`, `test.jpg` (4290×2856×3) | ch03 | 1 | Ny-Ålesund 2011: Otsu t* 123/107/182, IC 15.36/32.05/72.63 %; `2.jpg` doubles as the `ch3ice.jpg` substitute |
| `data/book/ch04/test.jpg` (4290×2856×3) | ch04 | 1 | floe field; Otsu 113, IC 30.98 %; Fig 4.3(a) = crop `[1599:2151, 1978:2552]`, also ch5's real `main.m` test |
| `data/book/ch05/q.jpg` (96×81×3) + `watershed_based/nrm_junction_ending.fig` | ch05 | 1 | two touching floes (Figs 5.1–5.17); the `.fig` is the authors' Fig 5.14(f) L3 truth |
| `data/book/ch06/Sea_Ice_Floe_Identification/sea_ice_test.jpg` (1038×394×3, 180 058 B) | ch06 | 1 | `sea_ice_demo.m` / `dist.m`; **no book figure** (best NCC 0.31); 231 components, 598 seeds, 540 `dist.m` seeds |
| `data/book/ch06/for test/test8.jpg` (108×148×3) | ch06 | 1 | `for_test.m`'s single snake (circle r = 20 at (80, 40)); no book figure |
| `data/book/ch06/for test/alg_seg_gray.jpg` (201×202×3) | ch06 | 1 | **Fig 6.15(a)** (NCC 0.9882 vs the inverted PDF bitmap); read by no `.m` → `scripts/ch06_gvf_distance.py` drives it; 83 components, 46 seeds |
| `data/synthetic/ch03/t_uneven_2_g0.5_b40.jpg` (git-ignored) | ch03 | 3 | `t.jpg` substitute (Fig 3.4(a)) |
| `seaice/core/synth.py` fixtures | ch02–ch06 | 3 | printed matrices (Figs 2.10–2.21, 4.8, 5.15/5.16, **6.14**), `bimodal_image`, `uneven_illumination`, `two_touching_floes`, `plateau_fixtures`, **`fig_6_16_circles` (110×186, 61/9-px diameters), `u_shape`, `synthetic_floe_field`** |
| `reference/ch02..ch06/*.mat` (8/13/6/16/**13** + `inputs*.mat`) | ch02–ch06 | MATLAB refs | ch06: `del2`, snake/GVF, `regionprops`, polygon, `polyarea`, `strel`, `branches`, `dist`, `for_test`, `gvf_distance`, `kmean_stage`, `homofil` from **199 controlled fixtures** |
| Public-domain substitutes (`core/public_images.py`, `data/online/SOURCES.md`) | ch02–ch06 | 2 | NASA Worldview MODIS/Terra 2019-07-25 Beaufort at the book images' **exact sizes** (2048×1536, 4290×2856, 81×96, **394×1038, 148×108, 202×201**) so crops, `xlim`, `r = 15`, `(80, 40)` stay valid |
| Fig 2.7 gray image; `ch3ice.jpg`, `t.jpg`; §4.3 pack ice (Figs 4.17–4.20); §5.3 Ny-Ålesund (Figs 5.7/5.11/5.18–5.20, Table 5.1); **ch6 §6.1/§6.2/§6.5 sources (Figs 6.2, 6.3, 6.5–6.13, 6.18–6.21)** | ch02–ch06 | missing | not shipped, no public source → procedure-only on shipped/synthetic images; every affected figure row is `unverified` |
| `dypic_05100_cam1_top.avi` (ch9), raw JPEG behind ch8 `MCD/*.mat` | ch08/ch09 | missing | see `data/online/SOURCES.md` |

## Parity summary per chapter
| chapter | exact | near | approx | reimplemented | unverified | verdict |
|---|---|---|---|---|---|---|
| ch02 Preliminaries | 23 | 1 (`bwdist` quasi) | 1 (`resize` bicubic/shrink) | 4 (HSI, CMYK, §2.3 text, `min_magnitude`) | 1 (Fig 2.7 image) | PASS — 86 tests, 7/7 `.m` exact |
| ch03 Ice pixel detection | 16 | 0 | 1 (`multithresh` int16) | 5 | 1 (Figs 3.2–3.5/3.7 images) | PASS — 112 tests + 1 xfail, 4/4 `.m` exact, 14/14 image pairs 0 px |
| ch04 Ice edge detection | 22 | 1 (`edge('log')` on flat synthetics) | 0 | 6 | 1 (Figs 4.17–4.20 image) | PASS — 548 tests, 2/2 `.m` exact (0 px on 12.25 Mpx), 27/27 image pairs 0 px |
| ch05 Watershed floe segmentation | 24 | 0 | 1 (`watershed_skimage`, cross-check) | 3 | 1 (§5.3 images) | PASS — 399 tests (1145 total), 15/15 `.m` exact (`watershed` 74 cases with label values; `main.m` per-iteration exact on 3 images), 49/55 image pairs identical |
| ch06 GVF snake | 21 | 6 | 1 (Statistics-TB `kmeans`) | 4 | 0 (+5 deferred to ch7/ch8) | PASS — 429 tests (**1574 total**, 2 skipped, 1 xfailed), 34/34 `.m` rows, 29 measured against MATLAB R2025a: `regionprops` ≤ 1.07e-14 on 40 shapes × 10 props + 344 real components, `GVF`/`del2`/`snakedeform`(dense)/`poly2mask` bit-exact, `dist.m` 540/540 radii bit-exact; residual **0.050 %** of `bw1` traced to `polybool`'s start-vertex rotation; 11/14 image pairs identical |

## Setup findings (2026-09-09, /setup-project)
- Reference engine: **MATLAB R2025a** (25.1.0.2833191, prerelease) via `tools/run_matlab_ref.py`; Octave absent (fine).
  Statistics & ML (`kmeans`) and **Mapping** (`polybool`, `polyxpoly`) are licensed → ch6 needed no fallback.
- PDF offset = 31 (pdf 0-based index = printed page − 1 + 31); each `chapters/chNN.txt` starts on its chapter title.
- `MATLAB_ROOT/ch10` = Appendix A (`fisheye_calibration.m` → A.2, `orthoretification.m` → A.1).
- `ch6/Sea_Ice_Floe_Identification` and `ch7/Sea_Ice_Floe_Identification` hold the same **23 `.m` files, byte-identical**
  (only `sea_ice_test.jpg` differs: ch6 180 058 B, ch7 167 601 B) → **ported once in ch6, reused in ch7**.
  `ch9/Model_Ice_Floe_Identification/` shares 13 of them (the whole GVF/snake stack + `minboundrect`).
  Inside ch6, `for test/` duplicates 9 `Sea_Ice_Floe_Identification/` files byte-identically (md5 in `analysis/ch06.md`).
  ch5 duplicates ch2's `bound2im.m`/`boundaries.m`/`fchcode.m`; ch9 `block_threshold.m` = ch3 `local_Otsu.m`.
- **Attribution obligations** (ch6/ch7/ch9): Xu & Prince GVF toolbox (`iacl.ece.jhu.edu/projects/gvf`; TIP 7(3), 1998),
  MATLAB-4 `gradient` provenance for `gradient2.m` (name it, do not redistribute), John D'Errico (`minboundrect`),
  H. J. Sommer III (`polygeom`), Qin Zhang's academic-use notice + TGRS 53(5):2913–2924, 2015 for everything else.
- Data gaps: ch9 movie scripts need `dypic_05100_cam1_top.avi`; ch8 MCD ships only `.mat` results.
- R2025a toolbox sources worth reading before porting a builtin: `edge.m`, `fspecial.m`, `imclose.m`/`imopen.m`,
  `private/morphop_fast.m`, `StructuringElementHelper.m` (ch04); `eml/watershed.m`, `FifoPriorityQueue.m`,
  `NeighborhoodProcessor.m`, `imimposemin.m`, `imregionalmin.m` (ch05); `del2.m`, `regionprops.m`
  (`ComputeEllipseParams`, `ComputeConvexHull`, `ComputePerimeterCornerPixelList`, `computePerimeterFromBoundary`),
  `eml/poly2mask.m`, `stats/kmeans.m` (ch06). Compiled routines without a twin (`computeEdges`, `gpcmex`) must be
  reverse-engineered or compared set-wise.

## Publishing findings (2026-09-09/10, ch02–ch05 PUBLISH phase)
- Public repo `shammun/seaice-py` (branch `main`), GitHub Pages at `https://shammun.github.io/seaice-py/`. History was
  rewritten with git-filter-repo to purge every book-derived file; `chapters/*.txt`, `data/book/`, `reports/**/figures/`,
  `outputs/` stay local and git-ignored (CLAUDE.md rule 12). Fetch and inspect `origin/main` before every push: a
  Colab "Save" commit once pushed executed outputs rendered from the private images (restored in `53960d5`/`9af4bd4`).
- Data: `seaice.core.io.load_image()` (private copy → NASA public-domain substitute from `seaice/core/public_images.py`);
  substitutes are requested from the Worldview snapshot API at the book image's exact size — and a blank response is
  rejected and retried (ch06).
- Notebook cells 1–3 template lives in `notebooks/build_chNN.py`; `tools/publish_notebook.py chNN` builds the Colab
  variant, the styled HTML page (from a run with `data/book` renamed), `index.html` and the README table. Scripts/tests
  must tolerate a missing `data/book` (print `SKIP`, exit 0). Do not run scripts/tests/notebooks while the publish
  rename is in effect.
