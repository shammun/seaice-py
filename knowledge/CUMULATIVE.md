# CUMULATIVE knowledge — seaice-py
(Rewritten by the knowledge-keeper after every chapter. Read this first at the start of any chapter. Last rewrite:
after **ch07**, 2026-09-10. Per-chapter detail: `knowledge/chNN.md`; verified call mappings: `knowledge/function_map.md`.)

## Pipeline so far
```
Ch2 primitives (DONE)  Ch3 ice mask (DONE)  Ch4 edges+morphology (DONE)  Ch5 watershed (DONE)  Ch6 GVF snake (DONE)  Ch7 ice types (DONE)  Ch8/9/10
  io.load_image (recursive since ch06) ─► rgb2gray_matlab ─► threshold.graythresh/im2bw ─► binary ice mask ─► every later .m
  matlab_compat.{rgb2gray_matlab,      │   (bit-exact vs MATLAB on 12-Mpx JPEGs; graythresh(RGB) ≠ graythresh(gray) — ch5)
    imcomplement (class-preserving      ├─► threshold.multithresh(N=2)+imquantize ─► 3 groups (Ch8)
    since ch07), matlab_round,          ├─► threshold.block_otsu(n_r, n_c) ─────────► Ch9 block_threshold.m (same code)
    im2double, im2uint8,                ├─► clustering.kmeans_gray (authors' ch3 kmeans.m) ─► Ch9 movie_kmeans.m
    to_uint8_saturating, del2,          └─► clustering.kmeans_lloyd('kmeans++') = Statistics-TB kmeans ─► Ch6/7/9
    saturate_to_class}
  histogram.{imhist, normalized_histogram, hist (centres, ch07)} ─► Ch7 FSD ─► Ch8 §8.3 color_hist*.m, App. B
  threshold.{ice_concentration, class_coverage, class_mean_intensity} ─► Ch8/Ch9 IC series
  connectivity.{label_components (= bwlabel), bwareaopen, bwperim} ─► Ch4 specks, Ch5 basins, Ch6 regionprops, Ch7 pieces, Ch8–9
  distance.{bwdist (single!), distance_transform} ─► Ch5 −bwdist(~bw) ─► watershed ─► Ch6 §6.3 seed/radius init ─► Ch7–9
  filters.{conv2, imfilter, fspecial, homomorphic_butterworth} ─► Ch4 edge() ─► Ch5 Sobel surface, Ch6 GVF edge map
  edges.edge (sobel/prewitt/log/zerocross, thinning) ─► Ch4 Figs 4.3/4.6, Ch6 overlays, Ch9 outlines
  morphology.{strel (octagon disk), imerode/imdilate, imopen/imclose, imreconstruct, imregionalmin/max, imimposemin,
      imfill (ch07, class-preserving)} ─► Ch4 Figs 4.9–4.20 ─► Ch5 markers ─► Ch6 seeds ─► **Ch7 Algorithms 2/4** ─► Ch8–9
  watershed.watershed (MATLAB Meyer flooding, exact incl. label values) ─► Ch5 all scripts ─► Ch8 FSD, Ch9
  chaincode.{boundaries, fchcode, bound2im} ─► Ch5 differential chain code ─► neighboring_region_merging
  interp.{interp2, warp_image, resize} ─► Ch6 snakedeform ─► **Ch7 resample_categorical (nearest)** ─► Ch10 App. A
  snake.{bound_mirror_*, gvf, gradient2, xconv2, gaussian_mask/blur, snake_matrix, snakedeform, snakeinterp, snakeindex}
      ─► Ch6 GVF snake ─► **Ch7 (23 byte-identical .m) and Ch9 (13 byte-identical .m) reuse this unchanged**
  regionprops.regionprops (MATLAB moments/hull/perimeter) ─► Ch6 Ra/Rc/Rl ─► Ch7 centroid/perimeter per piece ─► Ch8 FSD, Ch9
  polygon.{poly2mask, roipoly, convhull, polyarea, polyxpoly, polygeom, minboundrect, clip_polygon_rect}
      ─► Ch6 clips ─► Ch8 §8.2 sea_ice_model + App. B, Ch9 §9.3 rectangularization
  plotting.{…, label2rgb (large-index since ch07), size_colorbar (ch07), surface_plot, contour_overlay, snake_plot} · setops · color · synth · cli
Ch7 ice_shape_enhancement (Algorithms 2/4/5, Eqs. 7.5/7.6) ─► **Ch8 applications** (owns sea_ice_model.m,
SeaIce_Image_Structure.m, color_hist*.m — all four consume ch7's IcePiece{Center,Area,Perimeter,PixelsPosition})
                                                            ─► **Ch9 model ice** (GVF stack + minboundrect + hist/imfill)
                                                            ─► **Ch10 = App. A** (owns the rectifier ch7 deliberately did not write)
```
Conventions fixed in ch02–ch07 and binding for all later chapters:
- Coordinates: book `x` = row, `y` = col (Eq. 2.1). Python is 0-based `(row, col)` everywhere; tests add 1 to compare
  with MATLAB. `regionprops` `Centroid` and ch7's `IcePiece.PixelsPosition` (`[c, r]`) are MATLAB `(x, y)` 1-based;
  `find` results are column-major (`ch07_ice_type._find_column_major`).
- Numerics stay float64 **except distance maps, which stay float32**, and **except any value MATLAB computes in another
  class**: `single op double` promotes to double and rounds to single **once** (ch06), integer-class arithmetic
  rounds+saturates at every step (`matlab_compat.saturate_to_class`), and **any function documented "same class as the
  input" must do its arithmetic in that class — including its float branch** (ch07 `imcomplement`). Otsu levels stay
  float; morphology preserves the input dtype; `core.edges.edge` takes float only; label images are int32.
- Every `.m` maps to one Python script (`scripts/chNN_<name>.py`) or one `seaice` function; docstrings cite §/Eq/`.m`
  **and the measured parity label** — re-check every docstring claim (and every line-number citation) against the
  verification report at chapter end (ch06 found five stale labels; ch07 found wrong `hist.m` line numbers).
- Rule 9 (reuse before re-implementing): a new core primitive must replace, not shadow, the old code (ch06
  `component_centroids`); a display helper must be **extended**, not duplicated (ch07 `label2rgb` for ~10⁴ index values).
- References: MATLAB R2025a via `tools/run_matlab_ref.py`, driven by `reference/chNN/make_refs.py`; `.mat` in
  `reference/chNN/`; tests skip with a clear message if a `.mat` is missing. Run `.m` from a scratch cwd
  (`outputs/chNN/verify/scratch`) under **non-shadowing** names, patch only literals (or graphics-only blocks — record
  the patch verbatim in `patches.json`), never `addpath` a chapter folder. Controlled fixtures are written from Python
  with **constructed ties / intmax-intmin plants / asymmetric SEs / order-discriminating corridors / integer- and
  single-class inputs**; MATLAB's own refusals are recorded with `try/catch` as evidence.
- Figures: `outputs/chNN/fig_<ch>_<fig>_<slug>.png` only for genuine book figures; `sec_<ch>_<sec>_<slug>.png` otherwise.
- Authors' script bugs: the library default follows the book text/equations; the literal script behaviour sits behind an
  explicit flag (`rgb2hsi(matlab_bug=True)`, `kmeans_gray(shift_bug=True)`, `stale_mean_intensity`,
  `neighboring_region_merging(sequential=True)`, `gvf_force_field(normalize=True)`, `homomorphic_butterworth(matlab_bug=True)`,
  **ch07 `book_threshold`** for Algorithm 5's `≥` vs the code's `>`) and every book number that depends on it says so.
- Scripts that need book images print `SKIP …` and exit 0 when `data/book/chNN` is absent (public/Colab runs), via
  `load_image(..., allow_fallback=False)`; non-default CLI flags get one parametrised test each.
- Compiled toolbox builtins: look for the readable codegen twin `toolbox/images/images/eml/<name>.m` first (ch5
  `watershed`, ch6 `poly2mask`); read the plain M-file when there is one (ch7 `imfill.m`, `hist.m`, `imcomplement.m`);
  reverse-engineer only when there is none (ch4 `computeEdges`); when even that is impossible (`gpcmex`), compare
  **sets and rasterised masks**, never order.

## Available primitives in seaice/core/
| function | module | book § | used by chapters | parity |
|---|---|---|---|---|
| `load_image(chapter, name, allow_fallback=)` (case-insensitive, **recursive**; private copy → public-domain substitute), `repo_root`, `book_data_dir`, `output_dir`, `fetch` | `io` | — | all | exact (JPEG decode identical to MATLAB on every book image tried) |
| `chapter_argparser`, `resolve_dirs` | `cli` | — | all scripts | — |
| `rgb2gray_matlab`, **`imcomplement` (class-preserving, all 11 MATLAB classes)**, `matlab_round`, `im2double`, `im2uint8`, `to_uint8_saturating` | `matlab_compat` | §2.1–2.2, Eq. 2.3 | ch3–ch10 | exact (`imcomplement` re-verified ch07) |
| `del2(f, hx=1, hy=None)` (`∇²/(2·ndims)`, borders extrapolated) | `matlab_compat` | Eq. 6.52c | ch6, ch7, ch9 | exact (21 cases 0.0) |
| `saturate_to_class(x, dtype)` (one MATLAB integer-class op) | `matlab_compat` | — | ch6 `GVF`, ch7+ | exact |
| `imshow_scale`, `to_display_uint8`, `save_image`, `imshow_matlab`, `show_matrix`, `finish_figure`, `label2rgb` (tolerates ~10⁴ index values + RGB-triple background), **`size_colorbar`** (Eq. 7.6 ticks), `surface_plot`, `contour_overlay`, `snake_plot`, `quiver_field` | `plotting` | display | all | display only |
| `split_rgb`, `rgb2cmy`, `rgb2cmyk`, `rgb2hsi`, `indexed_to_rgb` | `color` | §2.1, Eqs. 2.2–2.6 | ch6/ch7 colour stats, ch8 | exact / reimplemented (HSI, CMYK) |
| `imhist(img, nbins=None)`, `normalized_histogram`, **`hist(y, bins)` → `(counts, centres)`** | `histogram` | §2.2, Eqs. 2.7–2.8; §7.2.4 | ch3, ch6, **ch7**, ch8+ | exact (`hist`: 23 cases 0 px, ch07) |
| `n4/nd/n8`, `is_adjacent`, `is_m_adjacent`, `find_paths` | `connectivity` | §2.3.1–2.3.3 | teaching only | reimplemented |
| `label_components(bw, conn=8)` (= `bwlabel` incl. numbering, int32), `count_components`, `bwareaopen(bw, P)` (keeps ≥ P), `bwperim(bw, conn=4)` | `connectivity` | §2.3.4–2.3.5 | ch4–ch9 | exact |
| `bwdist(bw, metric)` (**float32**), `distance_transform`, `quasi_euclidean_dt`, `pixel_distance`, `center_distance_map` | `distance` | §2.4, §5.1.2 | ch5, ch6 | exact (eucl/city/chess), near (quasi) |
| `conv2`, `imfilter(f, w, *matlab_options)`, `conv_at`, `fspecial(kind, …)`, `homomorphic_butterworth` | `filters` | §2.5, §4.1 | ch4–ch7 | exact (26 kernels ≤ 7e-18) |
| `edge(a, method, …) -> EdgeResult`, `gradient_sobel_prewitt`, `gradient_roberts`, `thin_gradient`, `log_zero_crossings` | `edges` | §4.1 | ch4, ch6, ch9 | exact (252 + 14 maps 0 px) |
| `strel(shape, *params, n=None)` (**disk = octagon**: `'disk',1` = **cross/5 px**, `'disk',2` = 13 px, `'disk',3` = 5×5/25, `'disk',5` = 9×9/69), `disk_decomposition`, `periodic_line`, `line_strel`, `intline`, `se_origin` | `morphology` | §4.2, Fig. 4.7 | ch4–ch9 | exact (116 nhoods) |
| `imerode`, `imdilate`, `imopen`, `imclose` (MATLAB pad/reflect/pre-pad rules; dtype preserved) | `morphology` | §4.2.1–4.2.2 | ch4–ch9 | exact (54 + 72 + 148 cases) |
| `imreconstruct(marker, mask, conn=8)`, `reconstruct_by_erosion`, `geodesic_dilation/erosion`, `reconstruct_iterative` | `morphology` | §4.2.3 | ch5–ch7 | exact / reimplemented (erosion dual) |
| **`imfill(I, 'holes', conn=4)`** — binary **and grayscale**, returns the input class | `morphology` | §7.1.3.2, Eqs. 7.3/7.4 | ch7, ch8, ch9 | exact (21 cases × conn {4,8}, ch07) |
| `imregionalmin/imregionalmax(I, conn=8)`, `imimposemin(I, BW, conn=8)`, `regional_maxima_by_reconstruction`, `morphological_gradient` | `morphology` | §5.1, Eqs. 6.57/6.58, §4.2.4 | ch5–ch9 | exact (126 / 88 cases) |
| `watershed(A, conn=8) -> int32` (Meyer flooding), `watershed_skimage`, `neighbour_offsets` | `watershed` | §5.1 | ch5, ch8–ch9 | exact **with label values** · ≈ 3 s/Mpx |
| `complement/union/intersection/difference`, `reflect`, `translate`, `gray_*`, `bitwise_*`, `truth_tables` | `setops` | §2.6 | ch4, ch7 | exact |
| `boundaries`, `fchcode`, `first_difference`, `min_magnitude`, `bound2im`, `chain_to_points` | `chaincode` | §2.7, §5.2.1 | ch5, ch6 | exact (tie-break reimplemented) |
| `interp2(img, u, v, method)` (Keys `a=−0.5` + quadratic edge extrapolation; `'nearest'` ties half away from zero), `warp_image`, `resize`, `keys_kernel` | `interp` | §2.8 | ch6, **ch7**, ch10 | exact (borders incl.) |
| `bound_mirror_expand/ensure/shrink`, `gvf`, `gradient2(+_complex/_magnitude)`, `xconv2`, `gaussian_mask/blur`, `snake_matrix`, `snakedeform(solver=)`, `snakeinterp`, `snakeindex` | `snake` | §6.1–6.4 | ch6, ch7, ch9 | exact (dense) / near (circulant ≤ 4.1e-13) |
| `regionprops(L_or_bw, properties, conn=8)`, `region_table` (MATLAB moments +1/12, `perim8` mid-edge hull, Vossepoel–Smeulders perimeter; both call forms) | `regionprops` | ch9 p. 205, §7.1/§8.2 | ch6–ch9 | exact (≤ 1.07e-14; ch7 centroid ≤ 1e-12, perimeter ≤ 1e-9 per piece) |
| `poly2mask`, `roipoly`, `clip_polygon_rect`, `convhull`, `polyarea`, `polyxpoly`, `polygeom`, `minboundrect` | `polygon` | §6.5.3, §8.2, §9.3 | ch6, ch8, ch9 | exact / near / reimplemented (see function_map) |
| `graythresh -> (level, em)`, `otsuthresh`, `im2bw`, `multithresh`, `imquantize`, `otsu_criterion`, `separability`, `block_otsu`, `ice_concentration`, `class_coverage`, `class_mean_intensity` | `threshold` | §3.1 | ch4–ch9 | exact (multithresh `N=3` reimplemented) |
| `kmeans_gray(gray, k, shift_bug=False)`, `kmeans_lloyd(X, k, init, seed, …)`, `objective_J`, `pairwise_distance` | `clustering` | §3.2.2 | ch3, ch6, ch7, ch9 | exact (`shift_bug=True`) / approx (toolbox mapping) |
| `point_image`, `spur_shape`, `book_fixtures`, `FIG_2_*`, `FIG_4_8_*`, `FIG_5_15/16_*`, `FIG_6_14_*`, `fig_6_16_circles`, `u_shape`, `synthetic_floe_field`, **`FIG_7_2_*`…`FIG_7_8_*`, `FIG_7_7_STEPS_BOOK`, `FIG_7_7_X8_TYPO`**, `uneven_illumination`, `bimodal_image`, `two_touching_floes`, `plateau_fixtures` | `synth` | Figs 2.10–7.8 | tests ch2–ch7, public fallbacks | exact (printed truths) / synthetic |
| Chapter modules: `ch03_ice_pixel_detection`, `ch04_ice_edge_detection`, `ch05_watershed`, `ch06_gvf_snake` (`BOOK_PARAMS`, `gvf_distance`, `seaice_kmean_gvf`, `initialize_contours`, `component_criteria`), **`ch07_ice_type`** (`morphological_cleaning`, `connected_component_extract`, `hole_fill_dilation`, `border_marker`, `hole_fill_reconstruct`, `adaptive_se_radius`, `size_color`/`color_to_area`/`colorbar_area_ticks`, **`ice_shape_enhancement`**, `sea_ice_edge_detection`, `sea_ice_shape_enhancement`, `ice_types_classification`, `floe_size_distribution`, `tile_grid`, `local_segmentation`, `resample_categorical`, `IcePiece`/`Coverage`/`IceShapeEnhancement`) | — | per chapter | later chapters call these directly | exact (scripts) / reimplemented (text forms) |

Not yet in core (first needed by): `bwmorph` LUTs; a fast exact `watershed` engine for 12-Mpx frames (ch8–ch9);
`imresize_matlab` with antialiasing (ch9 if needed); **DLT / orthorectification / lens distortion (ch10 — ch7
deliberately did not write a rectifier)**. Deferred `.m` owned by later chapters: `sea_ice_model.m`,
`SeaIce_Image_Structure.m`, `color_hist.m`, `color_hist_comparison.m` → **ch8 / Appendix B**
(`scripts/ch06_sea_ice_demo.py --full` still raises `NotImplementedError` for `sea_ice_model` alone;
`ice_shape_enhancement` landed in ch7).

## Global pitfalls (MATLAB → Python) confirmed in this project
1. Book `x` = row, `y` = col; MATLAB 1-based; Python 0-based `(row, col)`. `interp2(X, Y, …)` has `X` = column.
   `regionprops` `Centroid` and ch7 `PixelsPosition` are `(x, y)` = (col, row), 1-based, column-major order.
2. `bwdist(BW)` = distance to nearest **nonzero** = `distance_transform(~BW)`; returns **single** → `atol≈1e-4`, and
   keep the float32 array when it feeds `watershed`/`imregionalmin`.
3. **`single op double` promotes to double and rounds to single ONCE** (ch06, 540/540 radii bit-exact; float32-first
   360/540; float64 6/540). A downstream `ceil()`/`round()` turns 1 ulp into moved pixels.
4. **Class rules live inside library functions too.** `GVF.m` normalises in the *input's* class (uint8 → {0,1} edge map);
   `imimposemin`'s `I + h` likewise; **and a function whose doc says "IM2 has the same class as IM" evaluates its float
   branch in that class** — `imcomplement(single(1e-8))` is exactly `1`, and `imfill` complements twice, so a float64
   port differed from MATLAB on 2 of 25 `single` elements at 2.235e-8 (ch07 M1). Push every step through
   `saturate_to_class` / the input dtype.
5. `conv2` flips the kernel, `imfilter` does not; `conv2`'s default mode is `'full'`; positional option strings.
6. `bwlabel` numbers components column-major by first occurrence; `watershed` basins likewise — compare label *values*.
7. DIPUM `boundaries.m`: exterior only, closed lists, spurs traversed twice; `fchcode`'s `minmag` errors on periodic codes.
8. `rgb2gray` NTSC in double + half-away-from-zero rounding; `uint8(x)` = round + saturate; `astype(uint8)` wraps.
9. `imhist(logical)` → 2 bins; `imhist(I, n)` bins by `round(v·(n−1)/top)`.
10. `imcomplement`: `~I` / `intmax − I` (**all four** unsigned) / `bitcmp` = `−1 − I` (signed) / `1 − I` **in the input
    float class**. `imcomplement(double in 0–255)` = `1 − I`. Reproduce, do not "fix".
11. `interp2 'cubic'` = Keys `a = −0.5` with quadratic edge extrapolation; `imresize 'bicubic'` differs → `approx`.
12. HSI Eq. (2.6b) uses `atan(V2/V1)`; `color_image.m` line 21 has `2*Ig` for `2*Ib`.
13. `imread` is case-insensitive on Windows and ch6/ch7 images live in **sub-folders with a space** → always
    `load_image` (recursive). A public-domain download can come back blank → reject zero variance and retry.
14. MATLAB scripts may write files into the cwd — generate references from a scratch directory under a **non-shadowing** name.
15. PNG comparisons of float maps: `mat2gray` works in single → ±1 level at exact half-integers. Not a defect.
16. Windows/Anaconda host: `python -m nbconvert --execute` with `.venv/Scripts` **prepended to `PATH`**;
    `PYTHONIOENCODING=utf-8`; Bash heredocs mangle backslashes → write patch scripts to files; kill orphaned `MATLABWindow.exe`.
17. Authors' typos are reproduced when the printed figure shows their effect; **book errata are adjudicated by MATLAB**
    (ch07 Fig. 7.7(e) `X8`: printed 53 px, computed 55 — keep both fixtures and say which is which).
18. `graythresh` = `im2uint8` → `imhist(256)` → `otsuthresh` with **tie averaging**; `im2bw` compares in double with
    strict `>`; **`graythresh(RGB)` histograms all three planes**.
19. `multithresh`'s `getpdf` normalises in **single**; `N ≥ 3` is `fminsearch` → `reimplemented`.
20. **Toolbox-name shadowing**: `ch3/kmeans.m` is the authors' own; ch6/ch7/ch9 ship none, so those are Statistics-TB
    `kmeans` → `kmeans_lloyd(init='kmeans++')` (`approx`; never compare labels — compare sorted centres and pixel agreement).
21. Authors' k-means units bug (`shift_bug=True`); an empty cluster loops forever in MATLAB.
22. Stale loop buffers (`s(j) = I(p(j))` never cleared); `sum(uint8 vector)` returns **double**.
23. `num2str(x)` scalar: integers `%d`; else `%.{n}g`, `n = max(floor(log10|x|) + 5, 5)`.
24. 1-based bin index used as intensity (`separability.m`) → port literally.
25. `colormap('default')` and `label2rgb(...,'shuffle')`'s private RNG are display only — compare the **index arrays**
    (ch07 `label2rgb` with ~10⁴ Eq.-7.6 colour values renders a 10 000-entry `jet` LUT; the arrays are 0 px).
26. Analysis pre-checks are **hypotheses, not references** (ch3 79 vs 78; ch5 needed 25 fixtures; ch6's probe printed two
    wrong axis lengths). ch07's four pre-checks (215 vs 231 components, 84.13 % of samples, Otsu 162/255, the Fig. 7.7
    typo) were all **confirmed** by MATLAB — which is what makes them quotable.
27. Inventory watchlist misses `multithresh`, `imquantize`, `colormap`, `num2str`, `del2`, `polybool`, `hist` — grep
    yourself; and the inventory parser mislabels a `function` whose header uses a `...` continuation as a *script*.
28. MATLAB `edge` (sobel/prewitt) ≠ Eq. 4.2 > T: `fspecial/8` kernels, `'replicate'`, then `computeEdges` **thinning**.
29. `edge('log')` thresholds the **jump across the zero crossing**; `near` on piecewise-constant synthetics.
30. `double(im)/256` (the authors' idiom) ≠ `im2double` (/255) — port literally.
31. `fspecial('log')` = unit-sum Gaussian × `(x²+y²−2σ²)/σ⁴`, mean-subtracted.
32. **`strel('disk', r)` is an octagon**: r = 1 → the 4-connected **cross** (5 px, *not* a 3×3 square), r = 2 → 13 px,
    r = 3 → 5×5/25, r = 5 → 9×9/69. So Eq. (7.5)'s opening removes a rectangle's corners: an *identified* area is not
    the labelled area.
33. Erosion pads 1/intmax/+Inf, dilation pads 0/intmin/−Inf **and reflects the SE**; OpenCV does not reflect. An **even**
    SE (ch07's 2×2 square) makes the origin asymmetric — a good reflection fixture.
34. **`imclose` pre-pads** by `ceil(size(nhood)/2)` with 0 then crops; `imopen` does not. Any bounding-box crop of a
    per-piece morphology chain must therefore be proved **against MATLAB** on border-touching pieces (ch07: 1211 pieces,
    106 on a border, plus a `border_notch` fixture; 0 px, 243.7 s → 1.6 s).
35. `scipy.ndimage.minimum_filter/maximum_filter` route through float64 → wrong for int64/uint64. MATLAB **rejects**
    int64/uint64 in `imerode`/`imdilate`, int16/int32 in `watershed`, `uint8 ./ double matrix` in `gradient2.m`.
36. `logical − logical` → double; `uint8 − uint8` saturates; `imreconstruct` needs `marker <= mask` and a `{0,1}` marker
    reconstructs to `{0,1}`.
37. `bwareaopen(BW, P)` keeps `>= P` px; `skimage.remove_small_objects` changed its parameter in 0.26. `medfilt2` zero-pads.
38. "Figure X of Figure 4.3(a)" = **crop first, then process**; locate unshipped crops by NCC against the (inverted) PDF
    bitmap — ch06 found Fig. 6.15(a) at 0.9882 and **ch07 found that `sea_ice_test.jpg` IS Fig. 7.22, printed
    transposed** (0.9787), which ch06 missed by searching only its own pages. Search the *whole* book's bitmaps.
39. **MATLAB `watershed` is Meyer flooding** (`eml/watershed.m`); `skimage.segmentation.watershed` differs in ridges
    *and* partitions → `approx` only.
40. `imregionalmin` = `skimage.local_minima(..., allow_borders=True)` **except a constant image** (MATLAB all-True).
41. `imimposemin`: genuine `−Inf`/`intmin` markers, `h = 0.001·range` / 1 integer, `I + h` **in the input class**.
42. Watershed ridges are 4-connected staircases on distance maps but **thick on plateaus** — MATLAB identical.
43. `main.m` literalisms (relative `>= max` ending rule, `seg` updated inside the loop, `b{1}`).
44. **Test design**: a convex blob cannot exercise "spurious line removed" (use a peanut); an **integer-class** path needs
    an integer fixture and a **single**-class path a single fixture (ch06 M1, ch07 M1 — both hid behind float64 inputs
    for a whole verification round); a fix whose fixtures cannot distinguish it from the bug **is not tested** (ch07's
    `eps(edges)` would have survived a revert — the discriminating case had to be constructed, and the test now computes
    the wrong variant too); pin a fixed defect in **both** directions (monkey-patch the old expression back and assert
    it fails); settle an ordering question by *enumerating* candidate rules over many cases.
45. **Look for `eml/<builtin>.m`** before reverse-engineering a compiled builtin; and read the plain M-file
    (`imfill.m`, `hist.m`, `imcomplement.m`) when the "builtin" turns out to be ordinary M-code.
46. **Performance**: the pure-Python watershed heap is ≈ 3 s/Mpx; `snakedeform`'s dense `inv` is O(N³) per resample
    (306 s → 2.1 s with the circulant FFT solve); a per-piece full-image scratch array is O(pieces × M·N) (pitfall 34).
47. **`del2` is not the Laplacian**: `∇²/(2·ndims)`, borders linearly extrapolated, `/ndims` even in 1-D.
48. **`regionprops` must be reimplemented, not delegated to skimage** (axis lengths, perimeter +5.5 %, orientation +90°
    move the `Rc`/`Rl` decisions). Both call forms; **`regionprops(L == i, …)` may return more than one component** and
    MATLAB's `cat(1, s.Centroid)` appends them all (ch07 S4 — unobserved in 700 fixtures but real).
49. **`polybool` is compiled GPC with no readable source and its start vertex is not reproducible** — compare vertex
    *sets* and rasterised masks. It is still the project's only irreducible residual (0.05 % of ch6's `bw1`, and through
    Algorithm 3 → `seg` 0.245 % on ch7's image).
50. **Contour point counts are unstable at the 1e-13 level** where an insertion test is discontinuous. Report the curve
    (Hausdorff) and the mask, not the length.
51. **An eigenvector's sign is arbitrary** (`polygeom`'s `ang1` ±π); the axes are identical.
52. **Old FEX / old-graphics code may not run in current MATLAB**: `minboundrect.m`'s `convhull(x,y,{'Qt'})` is rejected
    by R2025a, and **HG1 handle graphics** (`get(bar,'Children')` → `'Faces'` → `'FaceVertexCData'`) error there —
    which blocks the reference for ch07's central file `ice_shape_enhancement.m` and will block ch08's `color_hist*.m`.
    Patch **graphics only**, change no numeric literal, record the patch verbatim (`patches.json`), and check that every
    quantity the removed lines consumed is still computed and saved.
53. **A book figure may need an input the book never prints**, and a caption parameter can hide the effect it claims —
    sweep the parameter and print the sweep.
54. **Verify docstring parity labels *and* line-number citations at chapter end** (ch06: five wrong labels; ch07: the
    `hist.m` citations pointed at the complex branch).
55. **MATLAB `hist` uses bin CENTRES, `np.histogram` uses edges.** `hist(y, n)` returns the centres of `n` equal bins
    (widening by `±n/2 − 0.5` when `min == max`); an explicit vector *is* the centres and its two outer bins are
    **unbounded**, so out-of-range values are counted, not dropped. `−Inf` lands in the first bin and `+Inf` in the last
    (the first comparison edge is `−Inf`) — a "drop non-finite values" filter is wrong; the non-finite guard only
    excludes them from `min`/`max`, and an all-non-finite input gives `miny = maxy = 0`.
56. **`eps(x)` is the spacing at `|x|`, not `nextafter`.** `hist.m`'s `edges + eps(edges)` must be
    `edges + abs(np.spacing(edges))`; on a negative edge at an exact power of two `nextafter` steps half as far and
    flips a bin (verified bit-exact against MATLAB's own `eps` on seven values).
57. **`imfill` ≠ `scipy.ndimage.binary_fill_holes`**: the library call matches MATLAB only on the **logical conn-4**
    branch (3 of 13 logical cases differ at conn 8) and never on a numeric input — wrong values for `uint8`, wrong
    values *and class* for grayscale/`int16`/`±Inf`. ch7 passes a **double**, and the returned class is seen by
    `bwlabel`/`imclose`/`out(pp) = t` downstream.
58. **MATLAB `sort` is stable, and where a sort decides an overwrite order it decides the result** — use
    `np.argsort(kind='stable')` and pin it with a tie fixture (ch07 Algorithm 2 superimposes small → large; ties are
    everywhere among 2–5 px brash pieces).
59. **Byte-identical code, different data.** ch7's `Sea_Ice_Floe_Identification/` is 23/23 byte-identical to ch6's, but
    its `sea_ice_test.jpg` is the same photo **re-encoded** (84.13 % of samples differ): `bwlabel(bw,4)` 215 vs 231,
    and a stage ch06 measured at 0 px now differs on 0.125 %. **Diff the data, not just the code, before inheriting a
    previous chapter's numbers**, and make `load_image` resolve to the chapter's own copy.
60. **Beware an agreement a saturating function makes inevitable.** Eq. (7.6) `fix(10⁴(1 − e^{−A/10³}))` is 9997 for
    *every* `A ≥ 8112`, so four of the book's eleven colour-bar tick lists are reproduced by any run with the same
    smallest piece. Before recording an exact match as parity, ask what fraction of the input space would produce it.
61. **A cited measurement must name its input.** ch07's 1232/433/290 came from the port's Algorithm 3, MATLAB's own
    segmentation gives 1211/433/274 — both true, only one is a parity number.

## Data inventory
| file | chapter | tier | shows |
|---|---|---|---|
| `data/book/ch02/rgb.JPG` (2048×1536×3) | ch02 | 1 | sea-ice colour photo; Fig 2.3 pixel (1076,675) = [28,76,114] |
| `data/book/ch03/1.jpg`, `2.jpg`, `test.jpg` (4290×2856×3) | ch03 | 1 | Ny-Ålesund 2011: Otsu 123/107/182, IC 15.36/32.05/72.63 % |
| `data/book/ch04/test.jpg` (4290×2856×3) | ch04 | 1 | floe field; Otsu 113, IC 30.98 %; Fig 4.3(a) = crop `[1599:2151, 1978:2552]` |
| `data/book/ch05/q.jpg` (96×81×3) + `nrm_junction_ending.fig` | ch05 | 1 | two touching floes; the `.fig` is the authors' Fig 5.14(f) L3 truth |
| `data/book/ch06/Sea_Ice_Floe_Identification/sea_ice_test.jpg` (1038×394×3, **180 058 B**, md5 `fca33143…`) | ch06 | 1 | `sea_ice_demo.m`/`dist.m`; 231 components, 598 seeds, 540 radii |
| `data/book/ch06/for test/test8.jpg` (108×148×3), `alg_seg_gray.jpg` (201×202×3 = **Fig 6.15(a)**, NCC 0.9882) | ch06 | 1 | single-snake demo; 83 components, 46 seeds |
| `data/book/ch07/Sea_Ice_Floe_Identification/sea_ice_test.jpg` (1038×394×3, **167 601 B**, md5 `54e56aa8…`) | ch07 | 1 | **Fig 7.22, printed transposed** (NCC 0.9787; 7.23/7.25 are overlays at 0.8886/0.7311). A *re-encoding* of ch6's copy — 84.13 % of samples differ. Otsu 162/255, `bwlabel(·,4)` **215**; MATLAB's segmentation → 1211 pieces, 712 labels, 433 floes / 274 brash, 106 border-touching |
| `seaice/core/synth.py` fixtures | ch02–ch07 | 3 | printed matrices (Figs 2.10–2.21, 4.8, 5.15/5.16, 6.14, **7.2–7.8**), `bimodal_image`, `uneven_illumination`, `two_touching_floes`, `plateau_fixtures`, `fig_6_16_circles`, `u_shape`, `synthetic_floe_field` |
| `reference/ch02..ch07/*.mat` (8/13/6/16/13/**7 + 18 per-fixture**) + `inputs*.mat` | ch02–ch07 | MATLAB refs | ch07: `clf`, `fig767`, `imfill` (21 cases), `imcomplement` (11 classes), `hist` (23 cases), `iceenh` (9 `(bk,seg)` fixtures), `demo` (44.7 min end to end) |
| Public-domain substitutes (`core/public_images.py`, `data/online/SOURCES.md`) | ch02–ch07 | 2 | NASA Worldview MODIS/Terra Beaufort at the book images' **exact sizes** (2048×1536, 4290×2856, 81×96, 394×1038 for **both** ch06 and ch07 keys, 148×108, 202×201) |
| Fig 2.7 gray; `ch3ice.jpg`, `t.jpg`; §4.3 pack ice; §5.3 Ny-Ålesund; ch6 §6.1/§6.2/§6.5 sources; **ch7 Figs 7.1, 7.9, 7.10–7.21 (155×125 crops, the 205×263 §7.2 image, the aerial §7.3.1 scene)** | ch02–ch07 | missing | not shipped, no `.m` produces them → procedure-only; **154/189, 2511/2624 and the eight coverage percentages stay `unverified` and were not fabricated** |
| `dypic_05100_cam1_top.avi` (ch9), raw JPEG behind ch8 `MCD/*.mat` | ch08/ch09 | missing | see `data/online/SOURCES.md` |

## Parity summary per chapter
| chapter | exact | near | approx | reimplemented | unverified | verdict |
|---|---|---|---|---|---|---|
| ch02 Preliminaries | 23 | 1 (`bwdist` quasi) | 1 (`resize` bicubic) | 4 | 1 (Fig 2.7 image) | PASS — 86 tests, 7/7 `.m` exact |
| ch03 Ice pixel detection | 16 | 0 | 1 (`multithresh` int16) | 5 | 1 (Figs 3.2–3.5/3.7 images) | PASS — 112 tests + 1 xfail, 4/4 `.m` exact |
| ch04 Ice edge detection | 22 | 1 (`edge('log')` on flat synthetics) | 0 | 6 | 1 (Figs 4.17–4.20 image) | PASS — 548 tests, 2/2 `.m` exact (0 px on 12.25 Mpx) |
| ch05 Watershed floe segmentation | 24 | 0 | 1 (`watershed_skimage`) | 3 | 1 (§5.3 images) | PASS — 399 tests, 15/15 `.m` exact incl. label values |
| ch06 GVF snake | 21 | 6 | 1 (Statistics-TB `kmeans`) | 4 | 0 (+5 deferred) | PASS — 429 tests (1574 total); `regionprops` ≤ 1.07e-14, `dist.m` 540/540 bit-exact; residual **0.050 %** of `bw1` = `polybool`'s start vertex |
| ch07 Ice type identification | 22 | 5 | 1 (Statistics-TB `kmeans` on ch7's JPEG) | 3 | 3 (unshipped §7.2/§7.3 images, the §7.3.2 sweeps, the ch10 rectifier) | PASS — 275 tests (**1849 total**, 2 skipped, 1 xfailed); 27/27 `.m` rows (6 ported + 17 reused + 4 deferred); `ice_shape_enhancement` **0 px** on 36 fixture runs and on 1211 real pieces; `imfill` exact in values **and class** on 21 cases × 2 conn; 47 of 48 printed Fig. 7.2–7.8 blocks 0 px (the 48th is a **book erratum**) |

## Setup findings (2026-09-09, /setup-project)
- Reference engine: **MATLAB R2025a** via `tools/run_matlab_ref.py`; Octave absent (never needed — Statistics & ML and
  Mapping are licensed).
- PDF offset = 31 (pdf 0-based index = printed page − 1 + 31). `MATLAB_ROOT/ch10` = Appendix A.
- `ch6/` and `ch7/Sea_Ice_Floe_Identification/` hold the same **23 `.m`, byte-identical** (only the JPEG differs — see
  pitfall 59) → ported once in ch6, reused in ch7. `ch9/Model_Ice_Floe_Identification/` shares 13. ch5 duplicates ch2's
  chain-code files; ch9 `block_threshold.m` = ch3 `local_Otsu.m`.
- **Attribution obligations** (ch6/ch7/ch9): Xu & Prince GVF toolbox (`iacl.ece.jhu.edu/projects/gvf`; TIP 7(3), 1998),
  MATLAB-4 `gradient` provenance for `gradient2.m`, John D'Errico (`minboundrect`), H. J. Sommer III (`polygeom`),
  Qin Zhang's academic-use notice + TGRS 53(5):2913–2924, 2015.
- R2025a sources worth reading before porting: `edge.m`, `fspecial.m`, `imclose.m`/`imopen.m`, `morphop_fast.m` (ch04);
  `eml/watershed.m`, `FifoPriorityQueue.m`, `imimposemin.m`, `imregionalmin.m` (ch05); `del2.m`, `regionprops.m`,
  `eml/poly2mask.m`, `stats/kmeans.m` (ch06); **`imfill.m` l. 124–145, `graphics/math/hist.m` l. 81–156,
  `imcomplement.m` l. 39–54, `conndef.m` (ch07)**.

## Publishing findings (2026-09-09/10, ch02–ch06 PUBLISH phase)
- Public repo `shammun/seaice-py` (branch `main`), Pages at `https://shammun.github.io/seaice-py/`. History was rewritten
  with git-filter-repo to purge book-derived files; `chapters/*.txt`, `data/book/`, `reports/**/figures/`, `outputs/`
  stay local and git-ignored (rule 12). **Fetch and inspect `origin/main` before every push** — a Colab "Save" commit
  once pushed executed outputs rendered from the private images.
- Data: `load_image()` (private copy → NASA public-domain substitute from `core/public_images.py`, requested at the book
  image's exact size, blank responses rejected and retried). Each chapter gets its **own** substitute key even when the
  book image is the same photograph (ch06 vs ch07).
- Notebook cells 1–3 template in `notebooks/build_chNN.py`; `tools/publish_notebook.py chNN` builds the Colab variant,
  the styled HTML (from a run with `data/book` renamed), `index.html` and the README table. Scripts/tests must tolerate a
  missing `data/book`. **Do not run scripts/tests/notebooks while the publish rename is in effect.**
