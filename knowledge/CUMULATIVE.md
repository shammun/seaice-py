# CUMULATIVE knowledge — seaice-py
(Rewritten by the knowledge-keeper after every chapter. Read this first at the start of any chapter. Last rewrite:
after **ch09**, 2026-09-11. Per-chapter detail: `knowledge/chNN.md`; verified call mappings: `knowledge/function_map.md`.)

## Pipeline so far
```
Ch2 primitives (DONE) Ch3 ice mask (DONE) Ch4 edges (DONE) Ch5 watershed (DONE) Ch6 GVF snake (DONE) Ch7 ice types (DONE)
Ch8 applications (DONE) Ch9 model ice (DONE)  ───────────────────────────────────────────────────────►  Ch10 = Appendix A
  io.load_image (recursive since ch06) ─► rgb2gray_matlab ─► threshold.graythresh/im2bw ─► binary ice mask ─► every later .m
  matlab_compat.{rgb2gray_matlab,      │   (bit-exact vs MATLAB on 12-Mpx JPEGs; graythresh(RGB) ≠ graythresh(gray) — ch5)
    imcomplement (class-preserving      ├─► threshold.multithresh(N=2)+imquantize ─► 3 groups (Ch8)
    since ch07), matlab_round,          ├─► threshold.block_otsu(n_r, n_c, compare='gt'|'ge')
    im2double, im2uint8,                │      'gt' = ch3 local_Otsu.m · **'ge' = ch9 block_threshold.m — a DIFFERENT file**
    to_uint8_saturating, del2,          ├─► clustering.kmeans_gray (the authors' ch3 kmeans.m) ─► Ch8 §8.1 (text-only k-means)
    saturate_to_class, num2str (ch09),  └─► clustering.kmeans_lloyd('kmeans++') = Statistics-TB kmeans ─► Ch6/7/**Ch9 movie_kmeans.m**
    matlab_colon}
  histogram.{imhist, normalized_histogram, hist (centres, ch07)} ─► Ch7 FSD ─► Ch8 color_hist*.m + App. B FSD ─► Ch9 fsd_error
  threshold.{ice_concentration, class_coverage, class_mean_intensity} ─► Ch8 §8.1 shipborne IC ─► **Ch9 §9.2 tank + video IC**
  connectivity.{label_components (= bwlabel), bwareaopen, bwperim} ─► Ch4 specks, Ch5 basins, Ch6/7 pieces, Ch8, **Ch9 rect/movie_floe**
  distance.{bwdist (single!), distance_transform} ─► Ch5 −bwdist(~bw) ─► watershed ─► Ch6 §6.3 seed/radius init ─► Ch7–9
  filters.{conv2, imfilter, fspecial, homomorphic_butterworth} ─► Ch4 edge() ─► Ch5 Sobel surface, Ch6/9 GVF edge map
  edges.edge (sobel/prewitt/log/zerocross, thinning) ─► Ch4 Figs 4.3/4.6, Ch6 overlays
  morphology.{strel (octagon disk), imerode/imdilate, imopen/imclose, imreconstruct, imregionalmin/max, imimposemin,
      imfill (ch07, class-preserving)} ─► Ch4 ─► Ch5 markers ─► Ch6/9 seeds ─► Ch7 Algorithms 2/4 ─► Ch8
  watershed.watershed (MATLAB Meyer flooding, exact incl. label values) ─► Ch5 all scripts ─► Ch8 FSD
  chaincode.{boundaries, fchcode, bound2im} ─► Ch5 differential chain code ─► neighboring_region_merging
  interp.{interp2, warp_image, resize} ─► Ch6 snakedeform ─► Ch7 resample_categorical (nearest) ─► **Ch10 App. A**
  snake.{bound_mirror_*, gvf, gradient2, xconv2, gaussian_mask/blur, snake_matrix, snakedeform, snakeinterp, snakeindex}
      ─► Ch6 GVF snake ─► **Ch7 (23 byte-identical .m) and Ch9 (14 byte-identical .m incl. minboundrect.m) reuse it unchanged**
  regionprops.regionprops(L, props|'basic') ─► Ch6 Ra/Rc/Rl ─► Ch7 per piece ─► Ch8 FSD ─► **Ch9 movie_floe 'basic'**
  polygon.{poly2mask, roipoly, convhull, polyarea, polyxpoly, polygeom, minboundrect, clip_polygon_rect,
      clip_polygon_convex (ch09, the one Sutherland–Hodgman engine)} ─► Ch6 clips ─► Ch8 §8.2 ─► **Ch9 rect/model_ice_model**
  stats.{mean_caliper_diameter (Eq. 8.1), cumulative_size_distribution (Eq. 8.2)} · fitting.{lsqcurvefit, power_law,
      truncated_power_law, weibull_survival} · icestruct (App. B format + .mat io + overlap_graph)   ─► Ch8 §8.3
  video.{read_video (H,W,3,N), write_video (Uncompressed AVI), video_num_frames, infer_frame_layout}  ─► **Ch9 §9.2.2/§9.3.3**
  plotting.{…, label2rgb, size_colorbar (ch07), matlab_jet + mcd_colorbar (ch08), snake_plot, surface_plot} · setops · color
  synth.{…, model_ice_tank, model_ice_tank_video, segmented_floe_video (ch09 Tier-3 stand-ins)} · cli
Ch7 ice_shape_enhancement (Algorithms 2/4/5) ─► Ch8 §8.2 sea_ice_model ─► App. B IceImage ─► Eqs. 8.1–8.3 (α = 1.3704)
Ch6 gvf_distance (+ ch09's stop='count', component_criteria ratio='minrect') ─► **Ch9 Algorithm 7** ─► rect.m ─► model_ice_model.m
   **[Ch9 DONE: 21/21 .m = 7 PORT + 14 REUSE + 0 DEFER; 3 of its 4 inputs do not ship → 10 book numbers permanently unverified]**

Ch9 ─► **Ch10 = Appendix A (geometric calibration), the LAST chapter. It owns BOTH rectifiers.**
   MATLAB_ROOT/ch10 = `fisheye_calibration.m` (**A.2 radial lens distortion**) + `orthoretification.m` (**A.1
   orthorectification**, note the spelling) + **two shipped images** `IMG_0729.JPG` and `test.jpg` (md5 them against
   ch02–ch09's copies first — pitfall 59). It must implement A.1.1 (deferred by ch07: shooting angle 20°, FOV 46°)
   **and** the linear 4-corner A.1.2 that §8.1 deliberately did not write (ch08 O1: checkerboard 5.90 m above the sea
   plane, camera 25.50 m, a second projection onto the sea plane, ≈ 400 m × 100 m). **There must not be a second
   rectifier**: ch07 owns only `resample_categorical` (a nearest-neighbour resampler for categorical images), ch08 and
   ch09 own none (§9.2.2 points at A.2 for the fish-eye it judges negligible). When `seaice.ch10_calibration` lands,
   wire it in front of `ch08_applications.shipborne_ice_concentration` and into `ch07_ice_type.local_segmentation`
   between Algorithms 3 and 4 (p. 166) — both call sites are documented in their docstrings.
   Ch10 reuses: interp.{interp2 (Keys −0.5 + quadratic edge extrapolation), warp_image, resize}, resample_categorical,
   matlab_compat.{matlab_colon, matlab_round, im2double, to_uint8_saturating, rgb2gray_matlab, num2str},
   polygon.{poly2mask, roipoly, convhull, polyarea, clip_polygon_convex}, io.load_image (+ its own ("ch10", …)
   substitute key), plotting, threshold, fitting.lsqcurvefit (report resnorm/exitflag; SciPy trf is `near`),
   and core.video if a calibration demo runs on a video.
```
Conventions fixed in ch02–ch09 and binding for all later chapters:
- Coordinates: book `x` = row, `y` = col (Eq. 2.1). Python is 0-based `(row, col)` everywhere; tests add 1 to compare
  with MATLAB. `regionprops` `Centroid`, ch7's `IcePiece.PixelsPosition` and ch9's `rect`/`RectFloe` fields are MATLAB
  `(x, y)` = (col, row), 1-based; `find` results are column-major (`ch07_ice_type._find_column_major`).
- Numerics stay float64 **except distance maps, which stay float32**, and **except any value MATLAB computes in another
  class**: `single op double` promotes to double and rounds to single **once** (ch06), integer-class arithmetic
  rounds+saturates at every step (`matlab_compat.saturate_to_class`), and **any function documented "same class as the
  input" must do its arithmetic in that class — including its float branch** (ch07 `imcomplement`). Otsu levels stay
  float; morphology preserves the input dtype; `core.edges.edge` takes float only; label images are int32.
- Every `.m` maps to one Python script (`scripts/chNN_<name>.py`) or one `seaice` function; docstrings cite §/Eq/`.m`
  **and the measured parity label** — re-check every docstring claim (and every line-number citation) against the
  verification report at chapter end (ch06 found five stale labels; ch07 wrong `hist.m` line numbers; ch09 a retracted
  R13 claim still live in `core/synth.py`).
- Rule 9 (reuse before re-implementing): a new core primitive must replace, not shadow, the old code (ch06
  `component_centroids`; ch09 refactored `clip_polygon_rect` onto `clip_polygon_convex` without moving a ch06 number);
  a display helper must be **extended**, not duplicated (ch07 `label2rgb`); a book variant of a shipped routine is an
  **opt-in flag with the shipped default unchanged** (ch09 `block_otsu(compare=)`, `component_criteria(ratio=)`,
  `gvf_distance(stop=)`) — and **validate the flag**, or a typo silently selects the default (ch09 S3).
- References: MATLAB R2025a via `tools/run_matlab_ref.py`, driven by `reference/chNN/make_refs.py`; `.mat` in
  `reference/chNN/`; tests skip with a clear message if a `.mat` is missing. Run `.m` from a scratch cwd
  (`outputs/chNN/verify/scratch`) under **non-shadowing** names, patch only literals (or graphics/IO-only blocks), never
  `addpath` a chapter folder. **The patch record must be GENERATED from the same dictionaries that drive the patching**
  and machine-checked line by line against the read-only originals (ch09 S8: the hand-written record had drifted).
  Controlled fixtures are written from Python with **constructed ties / intmax-intmin plants / asymmetric SEs /
  order-discriminating corridors / integer- and single-class inputs**; MATLAB's own refusals are `try/catch`-recorded.
- Figures: `outputs/chNN/fig_<ch>_<fig>_<slug>.png` only for genuine book figures; `sec_<ch>_<sec>_<slug>.png` otherwise.
- Authors' script bugs: the library default follows the book text/equations; the literal script behaviour sits behind an
  explicit flag (`rgb2hsi(matlab_bug=True)`, `kmeans_gray(shift_bug=True)`, `stale_mean_intensity`,
  `neighboring_region_merging(sequential=True)`, `gvf_force_field(normalize=True)`, `homomorphic_butterworth(matlab_bug=True)`,
  ch07 `book_threshold`, **ch09 `running_max_bug` / `increment_all_bug` / `nan_bug` / `strict_containment` / `empty=`**)
  and every book number that depends on it says so.
- Scripts that need book images print `SKIP …` and exit 0 when `data/book/chNN` is absent (public/Colab runs), via
  `load_image(..., allow_fallback=False)`; non-default CLI flags get one parametrised test each. Where the data does not
  ship at all, the script prints an explicit **"SYNTHETIC (Tier 3) … the book's numbers are NOT reproducible"** banner
  (ch09, all seven scripts).
- Compiled toolbox builtins: look for the readable codegen twin `toolbox/images/images/eml/<name>.m` first (ch5
  `watershed`, ch6 `poly2mask`); read the plain M-file when there is one (ch7 `imfill.m`, `hist.m`, `imcomplement.m`);
  reverse-engineer only when there is none (ch4 `computeEdges`); when even that is impossible (`gpcmex`), compare
  **sets, emptiness and rasterised masks**, never order.

## Available primitives in seaice/core/
| function | module | book § | used by chapters | parity |
|---|---|---|---|---|
| `load_image(chapter, name, allow_fallback=)` (case-insensitive, **recursive**; private copy → public-domain substitute), `repo_root`, `book_data_dir`, `output_dir`, `fetch` | `io` | — | all | exact (JPEG decode identical to MATLAB on every book image tried) |
| `chapter_argparser`, `resolve_dirs` | `cli` | — | all scripts | — |
| `rgb2gray_matlab`, `imcomplement` (class-preserving, all 11 MATLAB classes), `matlab_round`, `im2double`, `im2uint8`, `to_uint8_saturating`, **`num2str`** (promoted ch09) | `matlab_compat` | §2.1–2.2, Eq. 2.3 | ch3–ch10 | exact |
| `del2(f, hx=1, hy=None)` (`∇²/(2·ndims)`, borders extrapolated) | `matlab_compat` | Eq. 6.52c | ch6, ch7, ch9 | exact (21 cases 0.0) |
| `saturate_to_class(x, dtype)`, **`matlab_colon(start, step, stop)`** | `matlab_compat` | — | ch6+, ch8+ | exact / near (≤ 1 ulp on 30 of 126 non-integer angles; counts and endpoints exact) |
| `imshow_scale`, `to_display_uint8`, `save_image`, `imshow_matlab`, `show_matrix`, `finish_figure`, `label2rgb`, `size_colorbar` (Eq. 7.6 ticks), `mcd_colorbar` (clamped `[c₁,c_N]`), `surface_plot`, `contour_overlay`, `snake_plot`, `quiver_field` | `plotting` | display | all | display only |
| `matlab_jet(m)` — R2025a `jet.m` incl. the `mod(m,4)==1` rule | `plotting` | §8.3 | ch8 | exact — **numeric**, its rows are painted into `rgbImage` |
| `split_rgb`, `rgb2cmy`, `rgb2cmyk`, `rgb2hsi`, `indexed_to_rgb` | `color` | §2.1, Eqs. 2.2–2.6 | ch6/ch7/ch8 | exact / reimplemented (HSI, CMYK) |
| `imhist`, `normalized_histogram`, `hist(y, bins)` → `(counts, centres)` | `histogram` | §2.2, §7.2.4 | ch3, ch6–ch9 | exact (`hist`: 23 cases, ch07) |
| `n4/nd/n8`, `is_adjacent`, `is_m_adjacent`, `find_paths` | `connectivity` | §2.3.1–2.3.3 | teaching only | reimplemented |
| `label_components(bw, conn=8)` (= `bwlabel` incl. numbering, int32), `count_components`, `bwareaopen(bw, P)` (keeps ≥ P), `bwperim(bw, conn=4)` | `connectivity` | §2.3.4–2.3.5 | ch4–ch9 | exact |
| `bwdist(bw, metric)` (**float32**), `distance_transform`, `quasi_euclidean_dt`, `pixel_distance`, `center_distance_map` | `distance` | §2.4, §5.1.2 | ch5, ch6, ch9 | exact (eucl/city/chess), near (quasi) |
| `conv2`, `imfilter(f, w, *matlab_options)`, `conv_at`, `fspecial`, `homomorphic_butterworth` | `filters` | §2.5, §4.1 | ch4–ch7 | exact (26 kernels ≤ 7e-18) |
| `edge(a, method, …) -> EdgeResult`, `gradient_sobel_prewitt`, `gradient_roberts`, `thin_gradient`, `log_zero_crossings` | `edges` | §4.1 | ch4, ch6 | exact (252 + 14 maps 0 px) |
| `strel(shape, *params, n=None)` (**disk = octagon**), `disk_decomposition`, `periodic_line`, `line_strel`, `intline`, `se_origin` | `morphology` | §4.2, Fig. 4.7 | ch4–ch9 | exact (116 nhoods) |
| `imerode`, `imdilate`, `imopen`, `imclose` (MATLAB pad/reflect/pre-pad rules; dtype preserved) | `morphology` | §4.2.1–4.2.2 | ch4–ch9 | exact (54 + 72 + 148 cases) |
| `imreconstruct`, `reconstruct_by_erosion`, `geodesic_dilation/erosion`, `reconstruct_iterative` | `morphology` | §4.2.3 | ch5–ch7 | exact / reimplemented (erosion dual) |
| `imfill(I, 'holes', conn=4)` — binary **and** grayscale, returns the input class | `morphology` | §7.1.3.2 | ch7, ch8, ch9 | exact (21 cases × conn {4,8}) |
| `imregionalmin/imregionalmax`, `imimposemin`, `regional_maxima_by_reconstruction`, `morphological_gradient` | `morphology` | §5.1, §4.2.4 | ch5–ch9 | exact (126 / 88 cases) |
| `watershed(A, conn=8) -> int32` (Meyer flooding), `watershed_skimage`, `neighbour_offsets` | `watershed` | §5.1 | ch5, ch8 | exact **with label values** · ≈ 3 s/Mpx |
| `complement/union/intersection/difference`, `reflect`, `translate`, `gray_*`, `bitwise_*`, `truth_tables` | `setops` | §2.6 | ch4, ch7 | exact |
| `boundaries`, `fchcode`, `first_difference`, `min_magnitude`, `bound2im`, `chain_to_points` | `chaincode` | §2.7, §5.2.1 | ch5, ch6 | exact (tie-break reimplemented) |
| `interp2(img, u, v, method)` (Keys `a=−0.5` + quadratic edge extrapolation), `warp_image`, `resize`, `keys_kernel` | `interp` | §2.8 | ch6, ch7, **ch10** | exact (borders incl.) |
| `bound_mirror_*`, `gvf`, `gradient2(+_complex/_magnitude)`, `xconv2`, `gaussian_mask/blur`, `snake_matrix`, `snakedeform(solver=)`, `snakeinterp`, `snakeindex` | `snake` | §6.1–6.4 | ch6, ch7, ch9 | exact (dense) / near (circulant ≤ 4.1e-13) |
| `regionprops(L_or_bw, properties\|'basic', conn=8)`, `region_table`, `BASIC_PROPERTIES` | `regionprops` | ch9 p. 205, §7.1/§8.2 | ch6–ch9 | exact (≤ 1.07e-14; `'basic'` = `{Area, Centroid, BoundingBox}`, ch09) |
| `poly2mask`, `roipoly`, `clip_polygon_rect`, **`clip_polygon_convex`**, `convhull`, `polyarea`, `polyxpoly`, `polygeom`, `minboundrect` | `polygon` | §6.5.3, §8.2, §9.3 | ch6, ch8, ch9 | exact / near / reimplemented (see function_map) |
| `graythresh -> (level, em)`, `otsuthresh`, `im2bw`, `multithresh`, `imquantize`, `otsu_criterion`, `separability`, **`block_otsu(…, compare='gt'\|'ge')`** (+ `ic_mean`/`thresh_mean`), `ice_concentration`, `class_coverage`, `class_mean_intensity` | `threshold` | §3.1, §9.2.1 | ch4–ch9 | exact (multithresh `N=3` reimplemented) |
| `mean_caliper_diameter(area, scale)` (Eq. 8.1), `cumulative_size_distribution(sizes)` (Eq. 8.2) | `stats` | §8.3 | ch8 | exact (0.0 vs the authors' shipped `MCD_results.mat`) |
| `lsqcurvefit(...) -> LsqResult`, `optimset`, `MATLAB_LSQ_DEFAULTS`, `power_law`, `truncated_power_law`, `weibull_survival` | `fitting` | Eq. 8.3, C9/C10 | ch8, ch10? | near (SciPy `trf` = the same Coleman–Li family, different code) / exact (the models) |
| `IceImage/Param/Field/Floe/Polygon/Intersect/Brash/Circle`, `load_iceimage_mat`, `save_iceimage_mat`, `overlap_graph` | `icestruct` | **Appendix B** | ch8 (**not** ch9 — `model_ice_model.m` has its own flat record with an `Intersection` index vector) | exact |
| **`read_video(path) -> (H,W,3,N) uint8`, `write_video(path, frames, fps=12, codec='rawvideo')`, `video_num_frames`, `video_info`, `frames_to_matlab/imageio`, `infer_frame_layout`** | `video` | §9.2.2, §9.3.3 | ch9+ | **exact on Uncompressed AVI** (0 differing bytes both ways; explicitly `near`/`approx` on any lossy container) |
| `kmeans_gray(gray, k, shift_bug=False)`, `kmeans_lloyd(X, k, init, seed, …)`, `objective_J`, `pairwise_distance` | `clustering` | §3.2.2 | ch3, ch6, ch7, ch8, ch9 | exact (`shift_bug=True`) / approx (toolbox mapping) |
| `point_image`, `book_fixtures`, `FIG_2_*`…`FIG_7_8_*`, `uneven_illumination`, `bimodal_image`, `two_touching_floes`, `plateau_fixtures`, `synthetic_floe_field`, **`model_ice_tank`, `model_ice_tank_video`, `segmented_floe_video`** | `synth` | Figs 2.10–7.8, ch9 Tier-3 | tests ch2–ch9, public fallbacks | exact (printed truths) / synthetic |
| Chapter modules: `ch03_ice_pixel_detection`, `ch04_ice_edge_detection`, `ch05_watershed`, `ch06_gvf_snake` (`BOOK_PARAMS`, `gvf_distance(stop=)`, `seaice_kmean_gvf`, `initialize_contours`, `component_criteria(ratio=)`), `ch07_ice_type`, `ch08_applications`, **`ch09_model_ice`** (`block_threshold`, `movie_otsu`, `movie_kmeans`, `movie_floe`, `rect`, `model_ice_model`, `model_ice_demo`, `tank_ice_concentration`, `rect_ice_concentration`, `fsd_error`, `tiled_segmentation`, `segment_video`, `preprocess_frame`, `BOOK_PARAMS_CH9`, `CROP_5100`/`BOX_5100`/`IC_DENOMINATOR_5100 = 77 115`, `TABLE_9_1..9_4`) | — | per chapter | later chapters call these directly | exact (scripts) / reimplemented (text forms) |

Not yet in core (first needed by): `bwmorph` LUTs; a fast exact `watershed` engine for 12-Mpx frames;
`imresize_matlab` with antialiasing; **DLT / orthorectification / radial lens distortion and the linear 4-corner
rectification — ch10, which owns BOTH** (ch7, ch8 and ch9 all deliberately declined to write a rectifier).
**No `.m` is deferred**: ch8 absorbed the last four, ch9 added none, and ch10's two files are the last in the book.

## Global pitfalls (MATLAB → Python) confirmed in this project
1. Book `x` = row, `y` = col; MATLAB 1-based; Python 0-based `(row, col)`. `interp2(X, Y, …)` has `X` = column.
   `regionprops` `Centroid`, ch7 `PixelsPosition` and ch9 `rect`'s `[rectx, recty]` are `(x, y)` = (col, row), 1-based.
2. `bwdist(BW)` = distance to nearest **nonzero** = `distance_transform(~BW)`; returns **single** → `atol≈1e-4`, and
   keep the float32 array when it feeds `watershed`/`imregionalmin`.
3. **`single op double` promotes to double and rounds to single ONCE** (ch06, 540/540 radii bit-exact). A downstream
   `ceil()`/`round()`/`>` turns 1 ulp into moved pixels.
4. **Class rules live inside library functions too** (`GVF.m` normalises in the input's class; `imimposemin`'s `I + h`;
   "IM2 has the same class as IM" is a *computation* contract — ch07 M1).
5. `conv2` flips the kernel, `imfilter` does not; `conv2`'s default mode is `'full'`; positional option strings.
6. `bwlabel` numbers components column-major by first occurrence; `watershed` basins likewise — compare label *values*.
7. DIPUM `boundaries.m`: exterior only, closed lists, spurs traversed twice; `fchcode`'s `minmag` errors on periodic codes.
8. `rgb2gray` NTSC in double + half-away-from-zero rounding; `uint8(x)` = round + saturate; `astype(uint8)` wraps.
9. `imhist(logical)` → 2 bins; `imhist(I, n)` bins by `round(v·(n−1)/top)`.
10. `imcomplement`: `~I` / `intmax − I` / `bitcmp` / `1 − I` **in the input float class**. Reproduce, do not "fix".
11. `interp2 'cubic'` = Keys `a = −0.5` with quadratic edge extrapolation; `imresize 'bicubic'` differs → `approx`.
12. HSI Eq. (2.6b) uses `atan(V2/V1)`; `color_image.m` line 21 has `2*Ig` for `2*Ib`.
13. `imread` is case-insensitive on Windows and ch6/ch7/ch9 images live in **sub-folders with a space** → always
    `load_image` (recursive). A public-domain download can come back blank → reject zero variance and retry.
14. MATLAB scripts may write files into the cwd — generate references from a scratch directory under a **non-shadowing** name.
15. PNG comparisons of float maps: `mat2gray` works in single → ±1 level at exact half-integers. Not a defect.
16. Windows/Anaconda host: `python -m nbconvert --execute` with `.venv/Scripts` **prepended to `PATH`**;
    `PYTHONIOENCODING=utf-8`; Bash heredocs mangle backslashes → write patch scripts to files; kill orphaned `MATLABWindow.exe`.
17. Authors' typos are reproduced when the printed figure shows their effect; **book errata are adjudicated by MATLAB**.
18. `graythresh` = `im2uint8` → `imhist(256)` → `otsuthresh` with **tie averaging**; `im2bw` compares in double with
    strict `>`; **`graythresh(RGB)` histograms all three planes**; `im2bw` with **no level** is 0.5 ⇒ `> 127.5`.
19. `multithresh`'s `getpdf` normalises in **single**; `N ≥ 3` is `fminsearch` → `reimplemented`.
20. **Toolbox-name shadowing**: `ch3/kmeans.m` is the authors' own; ch6/ch7/**ch9 ship none**, so those are Statistics-TB
    `kmeans` → `kmeans_lloyd(init='kmeans++')` (`approx`; compare sorted centres + pixel agreement, never labels —
    ch09 confirmed labels differ on 17 of 24 frames while `IC` is 0.0). Where the *text* only names "the k-means
    method" and no `.m` calls the toolbox, use the authors' deterministic `kmeans_gray` (ch08 S6, ch09 §9.2.1).
21. Authors' k-means units bug (`shift_bug=True`); an empty cluster loops forever in MATLAB.
22. Stale loop buffers (`s(j) = I(p(j))` never cleared); `sum(uint8 vector)` returns **double**. (ch9's `movie_kmeans`
    has **no** stale-buffer bug — check, do not assume.)
23. `num2str(x)` scalar: integers `%d`; else `%.{n}g`, `n = max(floor(log10|x|) + 5, 5)` (`core.matlab_compat.num2str`).
24. 1-based bin index used as intensity (`separability.m`) → port literally.
25. `colormap('default')` and `label2rgb(...,'shuffle')`'s private RNG are display only — compare the **index arrays**.
26. Analysis pre-checks are **hypotheses, not references**, and so are its *negative* conclusions: ch09's analysis
    claimed no reading of Table 9.1's strip arithmetic matches the printed 45/40/15 % — the width(=area) reading matches
    exactly, in the printed order (R8). Re-derive, do not inherit a verdict.
27. Inventory watchlist misses `multithresh`, `imquantize`, `colormap`, `num2str`, `del2`, `polybool`, `hist`,
    `minboundrect`, `mmreader`, `movie2avi`, `getframe`, `strmatch` — grep yourself; the parser also mislabels a
    `function` whose header uses a `...` continuation as a *script*.
28. MATLAB `edge` (sobel/prewitt) ≠ Eq. 4.2 > T: `fspecial/8` kernels, `'replicate'`, then `computeEdges` **thinning**.
29. `edge('log')` thresholds the **jump across the zero crossing**; `near` on piecewise-constant synthetics.
30. `double(im)/256` (the authors' idiom) ≠ `im2double` (/255) — port literally.
31. `fspecial('log')` = unit-sum Gaussian × `(x²+y²−2σ²)/σ⁴`, mean-subtracted.
32. **`strel('disk', r)` is an octagon**: r = 1 → the 4-connected cross (5 px), r = 2 → 13, r = 3 → 5×5/25, r = 5 → 9×9/69.
33. Erosion pads 1/intmax/+Inf, dilation pads 0/intmin/−Inf **and reflects the SE**; OpenCV does not reflect.
34. **`imclose` pre-pads** by `ceil(size(nhood)/2)` with 0 then crops; `imopen` does not — so any bounding-box crop of a
    per-piece morphology chain must be proved against MATLAB on border-touching pieces.
35. `scipy.ndimage.minimum_filter/maximum_filter` route through float64 → wrong for int64/uint64. MATLAB **rejects**
    int64/uint64 in `imerode`/`imdilate`, int16/int32 in `watershed`, `uint8 ./ double matrix` in `gradient2.m`.
36. `logical − logical` → double; `uint8 − uint8` saturates; `imreconstruct` needs `marker <= mask` and a `{0,1}` marker
    reconstructs to `{0,1}`.
37. `bwareaopen(BW, P)` keeps `>= P` px; `skimage.remove_small_objects` changed its parameter in 0.26. `medfilt2` zero-pads.
38. "Figure X of Figure 4.3(a)" = **crop first, then process**; locate unshipped crops by NCC against the (inverted) PDF
    bitmap — and search the **whole** book (ch07 found `sea_ice_test.jpg` IS Fig. 7.22, printed transposed). A negative
    result is also a result: ch09's `model_ice.jpg` matches **nothing** (best NCC 0.167, `matchTemplate` **0.373** vs
    0.9882/0.9787 for genuine matches) — it is a code-only sample crop, so the chapter has **no plate at all**.
39. **MATLAB `watershed` is Meyer flooding** (`eml/watershed.m`); `skimage.segmentation.watershed` differs → `approx`.
40. `imregionalmin` = `skimage.local_minima(..., allow_borders=True)` **except a constant image** (MATLAB all-True).
41. `imimposemin`: genuine `−Inf`/`intmin` markers, `h = 0.001·range` / 1 integer, `I + h` **in the input class**.
42. Watershed ridges are 4-connected staircases on distance maps but **thick on plateaus** — MATLAB identical.
43. `main.m` literalisms (relative `>= max` ending rule, `seg` updated inside the loop, `b{1}`).
44. **Test design**: a convex blob cannot exercise "spurious line removed" (use a peanut); an **integer-class** path needs
    an integer fixture and a **single**-class path a single fixture (ch06 M1, ch07 M1); a fix whose fixtures cannot
    distinguish it from the bug **is not tested** — compute the wrong variant in the test and assert it differs, and pin
    a fixed defect in **both** directions; settle an ordering question by *enumerating* candidate rules; **choose fixture
    *shapes* that can fail** (a square fixture cannot catch a (row,col)↔(x,y) swap — ch09's `rect` swap is invisible in
    `Area` and visible in `Vertices` only on a non-square canvas); **prefer the authors' deterministic routine to a
    seeded library one**; and **a known-answer assertion must be chosen so the wrong implementations fail it** (ch09 S9:
    `29/9` on an axis-aligned rectangle does not discriminate a bounding-box rule — a 45°-rotated rectangle gives
    min-rect **4.0** vs bbox **1.0**).
45. **Look for `eml/<builtin>.m`** before reverse-engineering a compiled builtin; read the plain M-file when the
    "builtin" turns out to be ordinary M-code.
46. **Performance**: the pure-Python watershed heap ≈ 3 s/Mpx; `snakedeform`'s dense `inv` is O(N³) per resample
    (306 s → 2.1 s with the circulant FFT solve); a per-piece full-image scratch array is O(pieces × M·N).
47. **`del2` is not the Laplacian**: `∇²/(2·ndims)`, borders linearly extrapolated, `/ndims` even in 1-D.
48. **`regionprops` must be reimplemented, not delegated to skimage**; both call forms; `'basic'` = `{Area, Centroid,
    BoundingBox}`; `regionprops(L == i, …)` may return more than one component.
49. **`polybool` is compiled GPC with no readable source and its start vertex is not reproducible** — compare vertex
    *sets*, **emptiness** and rasterised masks. Its emptiness rules are non-obvious and must be probed: edge-touching and
    vertex-touching are **empty**, a 1e-9-area overlap is **kept** (ch09; `clip_polygon_convex(drop_degenerate=True)`
    matches 10/10 and is wrong on 3/10 with the flag off).
50. **Contour point counts are unstable at the 1e-13 level** where an insertion test is discontinuous. Report the curve
    (Hausdorff) and the mask, not the length.
51. **An eigenvector's sign is arbitrary** (`polygeom`'s `ang1` ±π); the axes are identical.
52. **Old FEX / old-graphics code may not run in current MATLAB** (`convhull(x,y,{'Qt'})`, HG1 `get(bar,'Children')`),
    and **whole functions get removed**: ch09's `mmreader` and `movie2avi` are gone from R2025a. Patch graphics/IO only,
    record it, and **generate the patch record** rather than writing it by hand (ch09 S8; 38 removed lines machine-checked
    against `MATLAB_ROOT`). A shipped file can even be **un-runnable** (a syntax error) — capture the refusal as evidence.
53. **A book figure may need an input the book never prints**, and a caption parameter can hide the effect it claims.
54. **Verify docstring parity labels *and* line-number citations at chapter end** — including **retracted** claims that
    are still live in a neighbouring module's docstring (ch09 S6).
55. **MATLAB `hist` uses bin CENTRES, `np.histogram` uses edges** — the outer bins are unbounded, `±Inf` are counted.
56. **`eps(x)` is the spacing at `|x|`, not `nextafter`** (`hist.m`'s `edges + eps(edges)`).
57. **`imfill` ≠ `scipy.ndimage.binary_fill_holes`** (only the logical conn-4 branch agrees; never a numeric input).
58. **MATLAB `sort` is stable, and where a sort decides an overwrite order it decides the result.**
59. **Byte-identical code, different data — and byte-identical *names*, different code.** ch7's SIFI folder is 23/23
    identical to ch6's but its JPEG is a re-encoding (84.13 % of samples differ: 215 vs 231 components). ch9 shares
    **14** files with ch6/ch7 (`minboundrect.m` is the 14th) — but its `block_threshold.m` is **NOT** ch3's
    `local_Otsu.m`: 5 diff hunks, and line 26 uses **`>=`** where ch3 uses `>` (plus ch9 drops `num` and the overall
    `IC = Σ num/(r·c)`), which is why `block_otsu` takes `compare='gt'|'ge'`. **md5 both the code and the data before
    inheriting any number**, and assert the equalities in a test.
60. **Beware an agreement a saturating function, a constants-only formula or a degenerate fixture makes inevitable.**
    Eq. (7.6) is 9997 for every `A ≥ 8112`; ch08's nine ticks depend on three literals; **ch09's N4 "proof" was vacuous**
    — for six *equal* blocks `Σ num/(r·c) ≡ mean(ic_b)` identically (measured `0.8673501526073122` both), so the
    arithmetic cannot discriminate the two definitions. Ask what fraction of the input space would produce the match,
    and enumerate the pre-image before quoting an exact tick match as parity (ch09 R7: colour 198 pins the smallest floe
    to exactly 20 px, but {9974…9977} pin the largest only to **[5953, 6119] px**).
61. **A cited measurement must name its input.** ch07's 1232/433/290 vs MATLAB's 1211/433/274; **ch09's `rect` /
    `model_ice_model` `exact` labels hold only on MATLAB's own `bw4`** — end to end from our `gvf_distance` the 28-px
    `bw1` residual moves the overlap flags **36 vs 34** and `rect_ice_concentration` by 0.104 pp.
62. **MATLAB grows an array on an out-of-range assignment instead of erroring**, so a result's *size* is data-dependent
    — never pre-allocate where the original grows. **But a null assignment PAST THE END is not a deletion**: it raises,
    with `MATLAB:subsdeldimmismatch` for the literal `x(k) = []` and `MATLAB:matrix:singleSubscriptNumelMismatch` for a
    call returning `0×0` (ch09 R13; delete-and-shift needs `k <= numel`).
63. **`if x ~= NaN` is `if ~isempty(x)`** — and the *consequence depends on what the wrapped routine returns*: ch08's
    `polyxpoly` reports only boundary crossings so containment is invisible, while **ch09's `polybool` returns the
    intersection region so containment IS detected**. Reproduce the idiom, then ask what it wraps.
64. **`a:step:b` with a non-integer step does not reach `b`** — use `matlab_colon`, pin the count *and* the last value.
65. **A shipped `.mat` of the authors' own results outranks any synthetic fixture** (ch08's 2888 floes + 3452 brash).
66. **An unidentifiable parameter makes "parity" meaningless along that direction** — measure the objective, not the
    parameter; and **capture a toolbox optimiser's defaults from `optimoptions(...)` itself**.
67. **`convhull`'s vertex order is not reproducible either** (105 of 227) — contract on sets and masks.
68. **A comparison against a GROWING vector is not a scalar comparison.** `if scalar >= t` where `t` accumulates means
    `all(...)`, so the effective bound is the **running maximum** (ch09 E4: 17 of 24 frames change, max ΔIC 3.09 pp; a
    monotone fixture cannot reveal it). Likewise **`n = n+1` on a vector increments every element** (E5, closed form
    `n_final[j] = Σ_{m≥j} count_m`) — a value read *inside* the same iteration escapes E5 but not E4. **Read every `if`
    and every `+1` whose operand accumulates.**
69. **A shipped `.m` can contain unreachable validation** — `rect.m`'s `if (nargin<3) || isempty(metric)` in a
    two-argument function makes the whole `elseif` dead: MATLAB accepts `''`, `[]`, `'x'`, `'ap'`, `5` and `"a"`
    identically. Port the dead branch as dead, and honour the `isempty` default.
70. **A library function can return a malformed shape on a degenerate input, and the caller may then error.**
    `minboundrect`'s `repmat(x,1,5)` on a 1×1 gives a **1×5 row** → `Vertices` 1×10 → MATLAB's own `model_ice_model.m`
    raises `MATLAB:badsubscript`; the 2-point branch returns a **2×1** `Perimeter`; the 0-point branch returns `0×0`
    empties. **Probe degenerate inputs against MATLAB** instead of assuming a clean return, and pin its shapes.
71. **`np.hypot` is not `sqrt(dx²+dy²)`** — a different algorithm, ≤ 1 ulp apart, which matters when the result feeds a
    **strict** inequality (ch09's E7 band `k1 < k < k2`). Write the literal expression.
72. **Video parity is a precondition, not a result.** Prove the container decodes **bit-identically** (Uncompressed AVI:
    0 differing bytes both ways) *before* any algorithm claim rests on it, and pin the **axis order** against MATLAB's
    own `size()` — `read(VideoReader)` is `(H, W, 3, N)`, imageio is `(N, H, W, 3)`.
73. **`reimplemented` requires a reference or a known answer.** A careful no-reference wiring is **`unverified`**, must
    carry a `# DEVIATION` marker, must appear in no parity assertion and must get an Open item (ch09 O10).
74. **Where three of four inputs do not ship, say so at the top of every artifact.** ch09 runs §9.2/§9.3.3 on Tier-3
    synthetic stand-ins written once and *read back* so both engines decode identical bytes; a Tier-3 number is **never**
    a book number, and ten printed ch9 values (N2, N3, N5, N6, N9–N12, N18, N19) are permanently `unverified`.

## Data inventory
| file | chapter | tier | shows |
|---|---|---|---|
| `data/book/ch02/rgb.JPG` (2048×1536×3) | ch02 | 1 | sea-ice colour photo; Fig 2.3 pixel (1076,675) = [28,76,114] |
| `data/book/ch03/1.jpg`, `2.jpg`, `test.jpg` (4290×2856×3) | ch03 | 1 | Ny-Ålesund 2011: Otsu 123/107/182, IC 15.36/32.05/72.63 % |
| `data/book/ch04/test.jpg` (4290×2856×3) | ch04 | 1 | floe field; Otsu 113, IC 30.98 %; Fig 4.3(a) = crop `[1599:2151, 1978:2552]` |
| `data/book/ch05/q.jpg` (96×81×3) + `nrm_junction_ending.fig` | ch05 | 1 | two touching floes; the `.fig` is the authors' Fig 5.14(f) L3 truth |
| `data/book/ch06/Sea_Ice_Floe_Identification/sea_ice_test.jpg` (1038×394×3, md5 `fca33143…`) | ch06 | 1 | `sea_ice_demo.m`/`dist.m`; 231 components, 598 seeds, 540 radii |
| `data/book/ch06/for test/test8.jpg`, `alg_seg_gray.jpg` (= **Fig 6.15(a)**, NCC 0.9882) | ch06 | 1 | single-snake demo; 83 components, 46 seeds |
| `data/book/ch07/Sea_Ice_Floe_Identification/sea_ice_test.jpg` (md5 `54e56aa8…`) | ch07 | 1 | **Fig 7.22 printed transposed** and **Fig 8.8** (transposed *and* inverted); a re-encoding of ch6's copy (84.13 % of samples differ) → Otsu 162/255, `bwlabel(·,4)` 215; MATLAB's segmentation 1211 pieces / 433 floes / 274 brash |
| `data/book/ch08/MCD/IceImage_290915_2_jpg.0000179.mat` (877 304 B) + `MCD_results.mat` (`Raw_MCD` 1×2888) | ch08 | 1 | the complete **Appendix-B structure** (2888 floes / 3452 brash, coverages 58.00/4.85/21.21/15.94 %, 51 FSD triplets) and the authors' **own saved output** = the project's gold L3 reference (reproduced at max \|Δ\| **0.0**) |
| `data/book/ch09/Model_Ice_Floe_Identification/model_ice.jpg` (181×76×3, 4130 B, md5 `4d10f303…`, **unique**) + `README.docx` | ch09 | 1 | the **only** ch9 image; bright square model floes on dark water; **not a printed figure** (best `matchTemplate` 0.373). `bwlabel(bw,4)` = 3 components — the §9.3.1 problem itself. The README carries the CRST 111:27–38 (2015) citation obligation |
| `data/synthetic/ch09/04100_analyse.jpg`, `dypic_synth_top.avi`, `05100_synth_segmented.avi` | ch09 | **3** | Tier-3 stand-ins (`core.synth.model_ice_tank` 348×1770, `model_ice_tank_video` 480×640×24, `segmented_floe_video` 240×320×40), written once and **read back**; the AVIs are **Uncompressed** so both engines decode identical bytes |
| `04100_analyse.jpg` (Fig. 9.1), `dypic_05100_cam1_top.avi`, `05100.avi` | ch09 | **missing, unobtainable** | HSVA/DYPIC campaign assets, never published (Tier 2 and Tier 4 both fail) → **N2, N3, N5, N6, N9–N12, N18, N19 permanently `unverified`**; N1, N7, N16, N17, N20 out of scope. Nothing fabricated |
| `MATLAB_ROOT/ch10/IMG_0729.JPG`, `test.jpg` | ch10 | 1 (to copy) | Appendix A's own data — **it ships**; md5 against ch02–ch09's copies before inheriting anything |
| `seaice/core/synth.py` fixtures | ch02–ch09 | 3 | printed matrices (Figs 2.10–7.8), `bimodal_image`, `uneven_illumination`, `two_touching_floes`, `plateau_fixtures`, `fig_6_16_circles`, `u_shape`, `synthetic_floe_field`, the three ch09 model-ice generators |
| `reference/ch02..ch09/*.mat` + `inputs*.mat`, `patches*.json`, `fixtures.py` | ch02–ch09 | MATLAB refs | ch08: 8 sessions + the one-off `optimoptions` probe; **ch09: 12 sessions, 871.3 s** (`video`, `probes`, `r13`, `block`, `rect`, `model`, `polybool`, three `movie_*`, `demo` 427.7 s, `degenerate`), patch record **generated**, 38 removed lines machine-checked |
| Public-domain substitutes (`core/public_images.py`, `data/online/SOURCES.md`) | ch02–ch09 | 2 | NASA Worldview MODIS/Terra Beaufort at the book images' **exact** sizes (2048×1536, 4290×2856, 81×96, 394×1038 for both ch06 and ch07, 148×108, 202×201, **76×181 for ch09**); each chapter gets its own key |
| Fig 2.7 gray; `ch3ice.jpg`, `t.jpg`; §4.3 pack ice; §5.3 Ny-Ålesund; ch6 §6.1/§6.2/§6.5; ch7 Figs 7.1/7.9–7.21; §8.1's six images, Fig. 8.18, Figs. 8.7/8.16/8.17 | ch02–ch08 | missing | procedure-only; every dependent number stays `unverified` and **none was fabricated** |

## Parity summary per chapter
| chapter | exact | near | approx | reimplemented | unverified | verdict |
|---|---|---|---|---|---|---|
| ch02 Preliminaries | 23 | 1 (`bwdist` quasi) | 1 (`resize` bicubic) | 4 | 1 (Fig 2.7 image) | PASS — 86 tests, 7/7 `.m` exact |
| ch03 Ice pixel detection | 16 | 0 | 1 (`multithresh` int16) | 5 | 1 (Figs 3.2–3.5/3.7 images) | PASS — 112 tests + 1 xfail, 4/4 `.m` exact |
| ch04 Ice edge detection | 22 | 1 (`edge('log')` on flat synthetics) | 0 | 6 | 1 (Figs 4.17–4.20 image) | PASS — 548 tests, 2/2 `.m` exact (0 px on 12.25 Mpx) |
| ch05 Watershed floe segmentation | 24 | 0 | 1 (`watershed_skimage`) | 3 | 1 (§5.3 images) | PASS — 399 tests, 15/15 `.m` exact incl. label values |
| ch06 GVF snake | 21 | 6 | 1 (Statistics-TB `kmeans`) | 4 | 0 (+5 deferred) | PASS — 429 tests; `regionprops` ≤ 1.07e-14, `dist.m` 540/540 bit-exact; residual **0.050 %** of `bw1` = `polybool`'s start vertex |
| ch07 Ice type identification | 22 | 5 | 1 | 3 | 3 (unshipped images, the §7.3.2 sweeps, the ch10 rectifier) | PASS — 275 tests; 27/27 `.m` (6 ported + 17 reused + 4 deferred); `ice_shape_enhancement` **0 px** on 36 runs and 1211 real pieces |
| ch08 Applications (+ **Appendix B**) | 10 | 5 | 0 | 0 | 2 (§8.1's results; §8.2's printed 498/201) | PASS — 19 rows (+1 display, +1 mixed), 88 tests (1937 total); 9/9 `.m`, 8 MATLAB sessions, 0 must-fix; the §8.3 chain is **bit-exact** down to the 627×1114×3 painted map; 11 errata |
| ch09 Model sea ice applications | 24 | 3 | 1 (Statistics-TB `kmeans`) | 5 | 2 (`tiled_segmentation`, `segment_video` — gap G2, O10) | PASS — **37 rows / 39 labels** (+1 display, +3 Tier-3 fixture), 53 tests (**1990 total**, 2 skipped, 1 xfailed); **21/21 `.m` = 7 PORT + 14 REUSE + 0 DEFER**, 12 MATLAB sessions (871.3 s), 1 must-fix at review, all 16 findings applied. **Three of four inputs do not ship** → 10 book numbers permanently `unverified`, 5 out of scope, 2 reproduced (N4, N8), 2 adjudicated (R7, R8), 1 transcription (N13). Errata E4/E5/E5b/E6/E7/E8/E9 + R13 all MATLAB-confirmed; video decode parity proven **as a precondition** |

## Setup findings (2026-09-09, /setup-project; corrected through ch09)
- Reference engine: **MATLAB R2025a** via `tools/run_matlab_ref.py`; Octave absent and never needed.
- PDF offset = 31 (pdf 0-based index = printed page − 1 + 31). `MATLAB_ROOT/ch10` = **Appendix A**.
- `ch6/` and `ch7/Sea_Ice_Floe_Identification/` hold the same **23 `.m`, byte-identical** (only the JPEG differs —
  pitfall 59) → ported once in ch6, reused in ch7. `ch9/Model_Ice_Floe_Identification/` shares **14** (the 13 GVF/snake
  files **plus `minboundrect.m`** — corrected in ch09 and asserted by a test against *both* the ch6 and ch7 copies).
  ch5 duplicates ch2's chain-code files. **ch9 `block_threshold.m` is NOT ch3 `local_Otsu.m`** (corrected in ch09: 5
  diff hunks, `>=` vs `>`, no `num`, no overall `IC`) → `block_otsu(compare='gt'|'ge')`.
- **Attribution obligations**: Xu & Prince GVF toolbox (`iacl.ece.jhu.edu/projects/gvf`; TIP 7(3), 1998); MATLAB-4
  `gradient` provenance for `gradient2.m`; John D'Errico (`minboundrect`); H. J. Sommer III (`polygeom`); Qin Zhang's
  academic-non-commercial notice with **TGRS 53(5):2913–2924, 2015** (ch6–ch8) and **CRST 111:27–38, 2015** (ch9).
- R2025a sources worth reading before porting: `edge.m`, `fspecial.m`, `imclose.m`/`imopen.m`, `morphop_fast.m` (ch04);
  `eml/watershed.m`, `FifoPriorityQueue.m`, `imimposemin.m`, `imregionalmin.m` (ch05); `del2.m`, `regionprops.m`,
  `eml/poly2mask.m`, `stats/kmeans.m` (ch06); `imfill.m`, `graphics/math/hist.m`, `imcomplement.m`, `conndef.m` (ch07);
  `jet.m`, `optimoptions('lsqcurvefit')` (ch08); `map/mapobsolete/polybool.m` and the `VideoReader`/`VideoWriter`
  properties (ch09 — and note `mmreader`/`movie2avi` are **gone**).

## Publishing findings (ch02–ch09 PUBLISH phase)
- Public repo `shammun/seaice-py` (branch `main`), Pages at `https://shammun.github.io/seaice-py/`. History was rewritten
  with git-filter-repo to purge book-derived files; `chapters/*.txt`, `data/book/`, `data/synthetic/`, `reports/**/figures/`
  and `outputs/` stay local and git-ignored (rule 12). **Fetch and inspect `origin/main` before every push** — a Colab
  "Save" commit once pushed executed outputs rendered from the private images.
- Data: `load_image()` (private copy → NASA public-domain substitute from `core/public_images.py`, requested at the book
  image's exact size, blank responses rejected and retried). Each chapter gets its **own** substitute key.
- Notebook cells 1–3 template in `notebooks/build_chNN.py`; `tools/publish_notebook.py chNN` builds the Colab variant,
  the styled HTML (from a run with `data/book` renamed), `index.html` and the README table. Scripts/tests must tolerate a
  missing `data/book`. **Do not run scripts/tests/notebooks while the publish rename is in effect.**
- **Prove the no-book path at notebook time, not at publish time** (ch08 precedent, repeated in ch09: the 76×181
  substitute download was exercised with `data/book` renamed away *and* `data/online/ch09` deleted), and **re-execute the
  notebook on the post-review code** (the ch04 lesson; ch09 did, 61 cells / 16 figures / 34 s / 0 errors).
