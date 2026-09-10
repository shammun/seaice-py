# Chapter 6 verification — GVF Snake-Based Ice Floe Boundary Identification and Ice Image Segmentation

Date 2026-09-10 · port commit `8ed4828` (`ch06: port — GVF snake toolbox, MATLAB regionprops, polygon geometry;
8 scripts exit 0`) · verifier artefacts: `tests/test_ch06.py` (388 tests), `reference/ch06/make_refs.py` +
`reference/ch06/fixtures.py` (199 controlled fixtures → `inputs.mat` / `inputs2.mat`, 11 `.mat` references,
`refs_log.json`), `reference/ch06/make_compare_figures.py`, `outputs/ch06/verify/` (MATLAB `imwrite` images in
`matlab/`, patched script copies in `scratch/`, generated MATLAB code in `*_code.m`, `image_diffs.json`,
`make_refs_*.log`, `pytest_ch06.txt`, `pytest_full.txt`, `pytest_scripts/<case>/`), `reports/ch06/figures/`
(7 compare PNGs, git-ignored).

## Environment
python 3.11.5, numpy 2.4.6, scipy 1.17.1, scikit-image 0.26.0, opencv 5.0.0, matplotlib 3.11.1 |
reference engine: **MATLAB 25.1.0.2833191 (R2025a) Prerelease Update 2** via `tools/run_matlab_ref.py`
(`matlab -batch`, figures invisible). All 11 reference files report `engine: matlab`, `status: ok`
(`reference/ch06/refs_log.json`). Octave absent (`progress.json → environment.octave = null`); **no fallback was
used and none was needed** — the Statistics & ML Toolbox (`kmeans`) and the Mapping Toolbox (`polybool`,
`polyxpoly`) are licensed, so every item of §6.1 of the analysis was obtainable at L2.

How the original code was run: the 23 `.m` files of `ch6/Sea_Ice_Floe_Identification/` and the three JPEGs were
copied **verbatim** into `outputs/ch06/verify/scratch/` and MATLAB ran with that folder as cwd (the ch6 folders
were never put on the MATLAB path; nothing under `MATLAB_ROOT` was touched). Five files needed a copy under a
non-shadowing name, patched **only** as listed:

| copy | changes |
|---|---|
| `ch6_dist_script.m` (`for test/dist.m`) | the author's absolute `imread('C:\Users\qinz\…\sea_ice_test.jpg')` → the local file; accumulator statements appended *after* the verbatim statements of the three display loops (`CEN1`, `CEN2`, `R0`, `X0`, `Y0`). A file named `dist.m` would shadow the Deep Learning Toolbox `dist` |
| `for_test_ref.m` (`for test/for_test.m`) | accumulators (`bw_top`, `u_gvf`, `v_gvf`, `XPB`, `XI0`, `XB{i}`, `xx_out`, `bw_burn`); the two `pause` calls removed (display only) |
| `gvf_distance_ref.m` (`GVF_distance.m`) | function renamed, second output `REC`, accumulators appended after verbatim statements; no expression changed |
| `kmean_gvf_ref.m` (`seaice_kmean_GVF_forenhancement.m`) | same treatment, per-pass records |
| `minboundrect_ref.m` (`minboundrect.m`) | **one** literal: `convhull(x,y,{'Qt'})` → `convhull(x,y)` — R2025a *rejects* the 2007 option list (see Deviation 4; the original's refusal is recorded as evidence) |
| `snake_matrix_ref.m` (new) | lines 22–38 of `snakedeform.m` **verbatim**, so the local `A`/`invAI` can be compared |

## pytest
`.venv/Scripts/python.exe -m pytest tests/test_ch06.py -q -p no:cacheprovider` → **`388 passed in 369 s`**,
0 failed, 0 xfail, 1 warning (`outputs/ch06/verify/pytest_ch06.txt`; the warning is Open item 4).
Breakdown by evidence level: **57 L1** (synthetic truth: 10 `del2`, 18 snake/GVF, 6 contour initialisation,
11 polygon, 6 `regionprops`, 2 criteria, 4 misc), **308 L2** (MATLAB parity: 19 `del2`, 8 `BoundMirror*`,
9 `gradient2`, 11 `xconv2`/gaussian, 50 snake unit, 44 GVF, 54 `regionprops`, 60 polygon, 24 extras,
6 `dist.m`, 7 `for_test.m`, 8 `GVF_distance`, 5 `seaice_kmean_GVF`, 3 demo stages), **7 L4** (quoted numbers),
**16 script runs** (8 default + 8 non-default CLI flag combinations, all exit 0).

Full suite `.venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider` → **`1533 passed, 2 skipped,
1 xfailed in 825 s`** (`outputs/ch06/verify/pytest_full.txt`) = the ch02–ch05 baseline (1145 passed, 2 skipped,
1 xfailed) **unchanged** plus the 388 new ch06 tests. **No regression.** Of the 15 warnings, 14 are pre-existing
pytest deprecation warnings in ch03/ch04 (`PytestRemovedIn10Warning`, class-scoped fixture) and 1 is Open item 4.

Parity label counts over the 41 rows below: **exact 19 · near 7 · approx 1 · reimplemented 4 · unverified 0 ·
deferred 5** (+ 4 mixed rows, e.g. `exact` for the mask and `near` for the point order, and 1 display-only row).
Every one of the **34 `.m` files** (the 9 byte-identical `for test/` duplicates included) has a row.

## Parity table
Levels: L1 synthetic truth · L2 MATLAB reference · L3 figure · L4 quoted number. "0 px"/"0.0" = identical arrays.

| MATLAB file / function | Python | Evidence | Result (measured) | Parity | Notes |
|---|---|---|---|---|---|
| `BoundMirrorExpand.m` | `core.snake.bound_mirror_expand` | L1 L2 | 7 fixtures (3×4 header example, 5×7, 2×2, 2×5, 3×3, 4×3, 21×17) **0.0**; equals `np.pad(A, 1, 'reflect')` | exact | |
| `BoundMirrorEnsure.m` | `core.snake.bound_mirror_ensure` | L1 L2 | same 7 fixtures **0.0**; MATLAB's own error text `'either the number of rows or columns is smaller than 3'` reproduced for a 2×2 | exact | idempotent (L1) |
| `BoundMirrorShrink.m` | `core.snake.bound_mirror_shrink` | L1 L2 | 7 fixtures **0.0**; expand→shrink round trip identity | exact | |
| `GVF.m` (Eqs. 6.41, 6.50–6.55) | `core.snake.gvf` | L1 L2 L3 L4 | **44 cases 0.0**: 9 edge maps (U-shape 40×40, random, 3×3, 3×5, 4×4, constant → NaN, `+Inf` plant, disc, Fig. 6.16) × ITER {1, 2, 5, 30} + ITER {100, 250, 500} + μ {0.02, 0.25} + ITER 0; on the book images `u`, `v`, `px`, `py` **0.0** (`test8.jpg` 150 iterations, `alg_seg_gray.jpg` 500 iterations, 201×202); Fig. 6.16 fixture at 5/30/100/250 **0.0**. NaN patterns identical for constant/`±Inf` inputs | exact | ITER = 1 and m/n = 3 (analysis R5's untested cases) covered |
| `gradient2.m` | `core.snake.gradient2`, `gradient2_complex`, `gradient2_magnitude` | L1 L2 | 7 fixtures (9×11, 1×7, 7×1, 2×2, 1×1, 3×5, 1×2) two-output **0.0**, complex form **0.0**, `abs()` ≤ 4.4e-16; scalar spacings `(0.5, 2)` **0.0**; explicit location vectors **0.0**; equals MATLAB's own `gradient` **0.0** | exact | |
| `xconv2.m` | `core.snake.xconv2` | L1 L2 | 6 image×mask pairs (23×19, 16×16 × 3×3, 4×4, 5×3) ≤ **9.1e-13** vs MATLAB `xconv2` and ≤ 6.8e-13 vs `conv2(…,'same')` — the header promises "under 1e-10" | exact | even mask exercises the `floor(n1/2)` crop |
| `gaussianMask.m` (Eq. 6.10) | `core.snake.gaussian_mask` | L1 L2 | σ ∈ {0.5, 1, 2, 4, 5} × k ∈ {1, 2.5} ≤ **1.4e-20**; `(2⌈3σ⌉+1)²` shape | exact | deliberately not `fspecial('gaussian')` |
| `gaussianBlur.m` (Eqs. 6.10/6.13) | `core.snake.gaussian_blur` | L1 L2 | σ ∈ {0.5, 1, 2, 4, 5} ≤ **6.4e-14**; `uint8` input ≤ 7.1e-14 | exact | never executed by the drivers (σ = 0) |
| `snakedeform.m` (Eqs. 6.34–6.40) | `core.snake.snakedeform`, `snake_matrix` | L1 L2 | `A` **0.0** for 12 (N, α, β) combos incl. N = 3, 4, 5, 126; `inv(A + γI)` ≤ 3.3e-16 (γ = 1 and 2.5); deform runs ITER {1, 5, 20} on two contours ≤ **1.8e-13** (`solver='dense'`), ring field 30 iterations ≤ 2.8e-13; `book_index=True` == script indices for constant α, β | exact | `interp2(fx, x, y, '*linear', 0)` **0.0** in and out of range |
| — `snakedeform(solver='circulant')` (FFT fast path) | same | L1 L2 | same MATLAB references ≤ **4.1e-13**; on the full 50-iteration `for_test` evolution the final curve is 2.7e-8 px from MATLAB's and the burnt mask is **0 px** different | near | not optional at ch6 scale (porter measured 306 s → 2.1 s) |
| `snakeinterp.m` | `core.snake.snakeinterp` | L1 L2 | 18 cases (6 contours × 3 `(dmax, dmin)`): **16 exact 0.0**; the 2 remaining are the header's own admitted removal bug — MATLAB raises `Index exceeds array bounds`, the port raises with a diagnostic. On the real pipelines: all 46 `GVF_distance` seeds and all `dist.m`/`for_test.m` contours reproduce MATLAB's point counts and values | exact | `max_passes` cap is an addition (Deviation 6) |
| `snakeindex.m` | `core.snake.snakeindex` | L2 | 12 boolean patterns (all-1, all-0, alternating, length 1/2/3, random) **0.0** | exact | |
| `snakedisp.m` | `core.plotting.snake_plot` | L1 L3 | closes the curve with `[x, x(1)]`; used in every figure | — (display only) | |
| `GVF_distance.m` (§6.3.3 + §6.4 Algorithm 1) | `ch06_gvf_snake.gvf_distance`, `scripts/ch06_gvf_distance.py` | L2 L3 L4 | on `alg_seg_gray.jpg` with the `sea_ice_demo` parameters: `gray`, `level`, `bw`, `f2`, `u`, `v`, `px`, `py` **0.0**; `label`/`num` (83 components) **0 px**; `a`, `rc`, `l`, `w`, `rl` ≤ 1e-9; the selection `k` (32 components) **identical**; `bw2` **0 px**; `img_Dist` **0.0**; `Dis_img` **0 px** (171 maxima px); `dis` after `imdilate` **0 px**; `label1`/`num1` **0 px / 46 seeds**; centroids ≤ 1e-12, radii ≤ 1e-6 (single-precision, Deviation 2); every seed's `snakeinterp` output **identical in length and value**; every clip has MATLAB's vertex set; the 46 final contours are the same **curve** (Hausdorff ≤ **0.479 px**); `bw1` differs by **61 of 31 730 ice px (0.19 %)**, burnt 1221 vs 1220 | near | Deviations 1 + 3; MATLAB 430 s, port 23 s |
| `seaice_kmean_GVF_forenhancement.m` | `ch06_gvf_snake.seaice_kmean_gvf`, `scripts/ch06_sea_ice_demo.py` | L2 L3 | on `alg_seg_gray.jpg` (MATLAB under `rng(0)`): `bw`, `f2`, `px`, `py` **0.0**; k-means sorted centres `[107.872949, 152.264015, 198.773459]` **equal to double precision**; `bk` **0 px** of 40 602; `n_negative` = **0** on both (Risk R7 latent); `bw0` after `bwareaopen(…, 10, 4)` **0 px**; pass 1 `bw1` 61 px (as above); pass 2 `bw0` 11 px; `out` has exactly the levels {0, 0.5, 1} on both and **0.177 %** of the pixels differ. On the full 1038×394 `sea_ice_test.jpg` the k-means stage is **0 px** (`kmean_stage.mat`) | near | k-means row below; MATLAB 419 s |
| `sea_ice_demo.m` | `scripts/ch06_sea_ice_demo.py` | L2 L3 | the parameter block is asserted value-by-value (L4 row); every stage on `sea_ice_test.jpg` verified against MATLAB: `I` **0.0**, `level` 0.0, `bw` **0 px**, `f2` **0.0**, `u`/`v`/`px`/`py` **0.0** (GVF 500 iterations, 29 s in MATLAB), `bwlabel(bw,4)` **0 px / 231 components**, criteria arrays ≤ 1e-9, `k` (107 components) identical, `bw2` **0 px**, `img_Dist` **0.0**, `Dis_img` **0 px**, dilated `dis` **0 px**, 598 seeds, centroids ≤ 1e-12, radii ≤ 1e-6, k-means stage **0 px** | near | the `ice_shape_enhancement` / `sea_ice_model` tail is stubbed (Open item 1) |
| `polygeom.m` | `core.polygon.polygeom` | L1 L2 | 15 polygons (the header's own self-test, a clockwise copy, triangle, square, 10 random convex): `geom = [A, x_cen, y_cen, P]` ≤ **1e-12**, `iner` ≤ 7.3e-12, principal moments `I1`, `I2`, `J` ≤ 1.4e-12. The header's self-test reproduces area 15, centroid (3.415, 6.549), perimeter 16, `Ixx` 659.561, `I1` 11.249, **ang1 = 30°**, ang2 = 120° | near | `ang1`/`ang2` differ from R2025a's `eig` by exactly ±π in 13 of 15 cases (Deviation 5) |
| `minboundrect.m` | `core.polygon.minboundrect` | L1 L2 | 10 point clouds × 2 metrics: `rectx`/`recty` ≤ **1.8e-15**, `area` and `perimeter` ≤ 1e-12 (incl. the `nedges` = 1 and 2 special cases); the author's 50 000-point unit-square example gives area 1.0000 in MATLAB and 0.9999 in the port | exact | the **shipped file does not run in R2025a** (Deviation 4); MATLAB rejects collinear/duplicate clouds where the port returns a degenerate rectangle |
| `homofil.m` | `core.filters.homomorphic_butterworth(…, matlab_bug=True)` | L2 | 6 cases (2 images × `(d, n)` ∈ {(10,1), (30,2), (5,4)}) ≤ **1.14e-12**, `uint8` case likewise | exact | the analysis expected `unverified`; MATLAB *can* run the orphan, so it is now measured. No book section, no caller |
| `for test/dist.m` (§6.3.3) | `scripts/ch06_dist.py` (via `ch06_gvf_snake.initialize_contours`) | L2 L3 | in the script's transposed frame: `bw` **0 px**, `img_Dist` **0.0**, `Dis_img` **0 px**, `dis` **0 px**, `dis1` **0 px**, `[p, q]` (2616 points) identical in MATLAB's column-major `find` order, `label` **0 px**, `num` = **540 seeds**, `CEN1` == `CEN2` and both ≤ 1e-12, `R0` ≤ **4.5e-7**, initial circles ≤ 3.1e-5, `bwlabel(bw,4)` = 231 components; the transpose identity `bw_script == bw_normal.T` holds | near | radii/circles are **single** in MATLAB (Deviation 2); 6 of the 540 seeds take the `r = 2` fallback on both sides |
| `for test/for_test.m` (§6.1.2/§6.2) | `scripts/ch06_for_test.py` | L2 L3 | `bw` **0 px**, `f2` **0.0**, `u`, `v`, `px`, `py` **0.0**, the 64×64 `interp2` quiver grid ≤ 1e-12 (`xSpace`/`ySpace` identical, including the authors' swapped roles); the clip has MATLAB's vertex set; fed MATLAB's own (closed) clip the port reproduces **all 10 five-iteration blocks element-wise ≤ 1.1e-7** (N = 257 … 777 identical); from the port's open ring the final curve is **2.5e-5 px** (Hausdorff) from MATLAB's and the burnt pixel **set** and mask are **identical (0 px, 91 px burnt)** | exact (mask) / near (point order) | Deviation 1; `v` is shadowed by the script's own `[v, h] = size(bw)` (Deviation 7) |
| `for test/{BoundMirrorEnsure, BoundMirrorExpand, BoundMirrorShrink, GVF, gradient2, snakedeform, snakedisp, snakeindex, snakeinterp}.m` (9 files) | same targets | L2 | **byte-identical duplicates** (md5 in `analysis/ch06.md` §0.1) — one port, the rows above apply | exact | re-verified `cmp` before the run |
| `ice_shape_enhancement.m` | → ch7 `ch07_ice_type.ice_shape_enhancement` | — | **not ported in ch06** (book §7.1 has this exact title; byte-identical in `ch7/`) | deferred | Open item 1 |
| `sea_ice_model.m` | → ch8 `ch08_applications.sea_ice_model` | — | not ported in ch06 (book §8.2 / Appendix B); its dependencies `polygeom`, `roipoly`, `polyxpoly`, `convhull` **are** ported and verified here | deferred | Open item 1 |
| `SeaIce_Image_Structure.m` | → ch8 / Appendix B | — | not ported in ch06 | deferred | |
| `color_hist.m` | → ch8 §8.3 / Fig. 8.14 | — | not ported in ch06 | deferred | |
| `color_hist_comparison.m` | → ch8 Fig. 8.15 | — | not ported in ch06 | deferred | |
| **New core primitives** | | | | | |
| MATLAB `del2` (Eq. 6.52c via `GVF.m`) | `core.matlab_compat.del2` | L1 L2 | **21 cases 0.0**: 17 matrices (4×5, 3×3, 3×7, 7×3, 2×3, 3×2, 2×2, 1×5, 5×1, 1×1, 1×3, 3×1, constant, random 9×11, `±Inf` plants, all-`Inf`, 40×33) + `del2(f, h)`, `del2(f, hx, hy)` and explicit location vectors. NaN patterns identical on the all-`Inf` case. Analysis R5's untested list (`ITER = 1`, m/n ∈ {2, 3}, constant `f`, `±Inf`) is covered here and in the `GVF` row | exact | `4·del2` = the 5-point Laplacian on the interior (L1) |
| MATLAB `regionprops` (Area, Centroid, BoundingBox, ConvexArea, Solidity, Major/MinorAxisLength, Eccentricity, Orientation, Perimeter) | **`core.regionprops`** | L1 L2 | **40 constructed shapes × 10 properties ≤ 1.07e-14** (single pixel, diagonal pair, h/v lines, staircase, square, rectangle, ring with a hole, L, disc, ellipses at 0/30/45/90°, peanut, border-touching, 6 random blobs, 4 solidity-boundary shapes around `Rc = 0.9`, 6 axis-ratio shapes around `Rl = 2`, empty, all-foreground); label matrices (3 fixtures × conn {4, 8}) ≤ 1e-14 **and** identical to the shipped `regionprops(label == n, …)` loop form; **every component of the real ch6 masks** (`alg_seg_gray.jpg` 83, `test8.jpg` 30, `sea_ice_test.jpg` 231) ≤ 1e-9 in both call forms; the resulting decision sets `k` are **identical** | exact | risk R2 closed; `Perimeter` is exact, not `near` |
| — `regionprops` `ConvexImage` / `ConvexHull` | `core.regionprops`, `core.polygon.poly2mask` | L2 | `ConvexImage` **0 px** on 6 shapes; the hull **polygon area** identical ≤ 1e-9 | exact (`ConvexImage`, `ConvexArea`, `Solidity`) / near (`ConvexHull` vertex list) | MATLAB's `convhull` keeps collinear hull points (30 vs 8 vertices on `sh_L`); the port's set is a subset and the polygon is the same |
| MATLAB `poly2mask` / `roipoly` | `core.polygon.poly2mask`, `roipoly` | L1 L2 | 8 polygons × 2 image sizes + the `regionprops` hull + a square: **0 px** on all 19 masks; `roipoly(I, x, y)` == `poly2mask` | exact | line-by-line port of `eml/poly2mask.m` |
| `polybool('intersection', rect, poly)` (Mapping TB, compiled GPC) | `core.polygon.clip_polygon_rect` (Sutherland–Hodgman) | L1 L2 | 7 clip cases (inside, over each corner, over an edge, entirely outside, rect inside the circle): **vertex sets identical**, rasterised masks **0 px**; and after `snakeinterp`, as ch6 actually calls it, identical. MATLAB returns **one extra vertex** (it closes the ring) in every non-empty case | reimplemented | Deviation 1 / Open item 2 |
| Mapping TB `polyxpoly` | `core.polygon.polyxpoly` | L1 L2 | crossing squares: the 2 intersection points and their 0-based segment indices identical; the disjoint case empty on both | reimplemented | |
| MATLAB `convhull` | `core.polygon.convhull` | L2 | 3 point clouds: same hull **vertex set**, closed | near | start vertex and collinear points differ (see `ConvexHull`) |
| `bwperim(BW, conn)` | `core.connectivity.bwperim` | L1 L2 | 9 shapes × conn {4, 8} **0 px** | exact | supersedes the ch02 `region_boundary_mask` |
| Eqs. **6.57 / 6.58** regional maxima by reconstruction | `core.morphology.regional_maxima_by_reconstruction` | L1 L2 | both forms == MATLAB `imregionalmax` **0 px** on the peanut's city-block and Euclidean distance maps and on a synthetic floe field | reimplemented | equals `imregionalmax` on every case tried |
| §6.3.3 / Algorithm 1 contour initialisation | `ch06_gvf_snake.initialize_contours` (`form='script'|'book'`) | L1 L2 L3 L4 | script form verified inside the three pipeline rows; on a synthetic floe field `bwf`, `Df`, `Mf`, `disf`, `Lf`, `nf`, centroids and radii all identical to MATLAB; the `book` form gives the **same maxima set** as the script form on Fig. 6.14 and Fig. 6.16 | exact (script) / reimplemented (book form) | |
| Eqs. 6.6–6.13 external energies, Eq. 6.29 traditional snake | `ch06_gvf_snake.{line,edge,termination,external}_energy`, `traditional_snake` | L1 L3 | built from the equations; the U-shape experiment reproduces the §6.1.3 vs §6.2 contrast (`scripts/ch06_ushape.py`) | reimplemented | no `.m` file implements them |
| Statistics TB `kmeans(X, k, 'EmptyAction','singleton')` | `core.clustering.kmeans_lloyd(init='kmeans++', seed=0)` | L2 L4 | sorted centres **equal to double precision** to MATLAB's `rng(0)` run on **both** book images; `bk` **0 px** on 40 602 (alg) and on 408 972 (sea_ice_test) pixels. But 20 MATLAB restarts with different seeds show the optimum is **not unique** (per-centre spread 2.20 / 2.00 / 0.09 gray levels) | approx | fn-map row 61 decided: `kmeans_lloyd`, **not** ch3's shadowing `kmeans_gray` (ch6 ships no `kmeans.m`) |
| `core.synth.fig_6_16_circles`, `FIG_6_14_IMAGE/_DISTANCE`, `u_shape`, `synthetic_floe_field` | `core.synth` | L1 L2 L4 | Fig. 6.14: `bwdist(~A,'cityblock')` == the printed matrix **0 px** and == MATLAB **0.0**; Fig. 6.16 fixture is 110×186 with exactly one 61-px and one 9-px diameter circle; the floe field's whole initialisation chain matches MATLAB | exact (printed truths) / reimplemented (fixtures) | |
| `core.io.resolve_case_insensitive` recursive fallback (R13) | `core.io` | L1 L2 | all three book JPEGs are found under `data/book/ch06/**` (including the folder with a space in its name); MATLAB and `imageio` decode `test8.jpg` and `alg_seg_gray.jpg` and `sea_ice_test.jpg` to **identical** gray arrays (0.0) | exact | |

## Figures reproduced
Compare PNGs: `outputs/ch06/verify/<name>_compare.png`, copies in `reports/ch06/figures/` (git-ignored, CLAUDE.md
rule 12). Layout: Python row | MATLAB `imwrite` row | book page rendered from `chapters/ch06.pdf`.
`outputs/ch06/verify/image_diffs.json`: **14 raw image pairs, 11 identical**; of the other three, two differ by
**1 gray level only** (`imshow(I, [])` display rounding at exact half-integers of the `mat2gray` stretch — the ch05
Deviation 4, arrays identical at L2) and one is the Fig. 6.15(f) segmentation (59 px = the 0.19 % snake
discretisation difference). One pair is not comparable because the Python panel is a matplotlib overlay.

| Figure | File | Verdict |
|---|---|---|
| 6.14(a)/(b) (p. 134) | `fig_6_14_compare.png` | The printed 8×8 blob, its city-block distance transform (values 1/2/3), the three-pixel regional maximum in red, the seed '+' and the green initial circle of radius 3/√2 are reproduced bit-exactly and match the book's panel. |
| 6.15(a)–(f) (p. 136) | `fig_6_15_compare.png` | All six panels agree with MATLAB and with the book: grayscale floe field, connected binary floes, the distance transform's bright floe centres, the regional maxima marked '+', 46 seeds with their initial circles, and a final segmentation in which the connected floes are separated by thin dark boundaries. |
| 6.16(a)–(e) (p. 139) | `fig_6_16_compare.png` | The 110×186 fixture and the four GVF fields show the same behaviour as the book: at 5 iterations the field already fills 98.6 % of the 9-px circle while only 22 % of the 61-px one carries force, and the large circle fills progressively (36 % / 56 % / 77 % at 30 / 100 / 250). The book's panel (a) appears to show circle *outlines* where the port uses filled discs — the edge map `|∇I|` is the same ring either way, and the printed claims reproduce numerically (Numbers table). |
| 6.17(a)/(b) (p. 140) | `fig_6_17_compare.png` | Under 30 GVF iterations the snake stalls near its initial radius (a small ring at the centre, as in the book's (a)); under 250 it sweeps out to the circle boundary (dense rings filling the disc, as in the book's (b)). |
| `for test/dist.m` (no book figure) | `sec_6_3_3_dist_compare.png` | Local maxima, dilated maxima, `imregionalmin(-D)` and the city-block distance transform are pixel-identical to MATLAB (the DT PNG differs by 263 px of 1 gray level, display rounding only); the seed/circle overlay shows the same 540 seeds. |
| `for test/for_test.m` (no book figure) | `sec_6_2_for_test_compare.png` | The 64×64 GVF quiver, the snake evolution from the red initial circle through the yellow 5-iteration curves to the green final contour, and the binary image before/after burning the boundary are all the same as MATLAB's own figures; the two binary images are pixel-identical. |
| `sea_ice_demo.m` stages (no book figure) | `sec_6_4_sea_ice_demo_compare.png` | Otsu mask, k-means mask `bk` and the `bw0` residual are pixel-identical to MATLAB on the full 1038×394 image; the Python row also shows the three-level `out` that the MATLAB stage reference stops short of. |
| 6.2, 6.3, 6.5–6.13, 6.18–6.21 | `outputs/ch06/*.png` (no compare row) | Not reproducible as figures: the images are not shipped and no `.m` file produces them. The *mechanism* is reproduced on synthetic fixtures (`ch06_snake_parameters.py`, `ch06_ushape.py`) — Open item 3. |

## Numbers from the text (book vs ours)
| Item | Page | Book | Ours (Python) | MATLAB (original .m) | Status |
|---|---|---|---|---|---|
| α = 0.05, β = 0.0 "for all snakes in this book" | 121 fn 2 | 0.05 / 0.0 | `BOOK_PARAMS` 0.05 / 0.0 | `sea_ice_demo.m`, `for_test.m` identical | match |
| μ = 0.1 "for all GVF snakes in this book" | 125 fn 3 | 0.1 | 0.1 | identical | match |
| CFL `r = μΔt/(ΔxΔy) ≤ 1/4`, `Δt ≤ ΔxΔy/(4μ)` | Eqs. 6.54/6.55 | ≤ 0.25 | r = 0.1; the port refuses μ > 0.25 at unit steps | `GVF.m` runs at Δt = Δx = Δy = 1 | match |
| city-block radius divisor √2 | 135 fn 4 | ÷√2 | `r = D(seed)/√2`; Fig. 6.14 → 2.1213 | identical (in single) | match |
| Fig. 6.14: "a regional maximum consisting of three local maxima", DT values 1/2/3 | 134 | 1 region / 3 px | 1 component, 3 px, all of value 3 | identical | match |
| Fig. 6.16: 110×186 image, 61-px and 9-px diameters | 139 | 110×186, 61, 9 | exactly those | (synthetic) | match |
| Fig. 6.16(b): "5 iterations are sufficient for the 9-pixel wide diameter … circle" | 139 | qualitative | **98.6 %** of the 9-px circle's interior carries force at 5 iterations (threshold |v| > 0.02) | GVF fields identical (0.0) | match |
| Fig. 6.16(c): "30 iterations are still not enough … 61-pixel wide diameter" | 139 | qualitative | **22.1 %** of the 61-px circle at 5 iterations and **36.0 %** at 30, rising to 56.1 % (100) and 77.3 % (250) | identical | match |
| Fig. 6.17: 30 vs 250 GVF iterations | 140 | qualitative | the snake stalls at an enclosed radius of **5.9 px** (30 iterations, mean distance to the boundary 24.7 px) vs **30.5 px** / mean distance **0.39 px** (250 iterations) | — (no `.m`) | match |
| `sea_ice_demo.m` parameter block (`kms0 3`, `Ra_min 10`, `Ra 2500`, `Rc 0.9`, `Rl 2`, `Num 500`, `iter 100`, `γ 1`, `κ 0.5`, `Dmin 0`, `Dmax 1`, `timer 1`, `σ 0`) | script | — | asserted value-by-value | identical | match |
| `for_test.m` (`Num 150`, `iter 50`, `r 20`, `(x0,y0) = (80,40)`) | script | — | asserted | identical | match |
| `dist.m` (`strel('disk',5)`, fixed `r = 15`) | script | — | asserted | identical | match |
| `t = 0:0.05:6.28` → 126 points, last 6.25 (circle not closed) | all drivers | 126 | 126, last 6.25 | identical | match |
| `strel('disk', 3)` size | analysis §2.2/§6.4 says **7×7 / 37 px** | — | **5×5 / 25 px** | **5×5 / 25 px** (`reference/ch06/strel.mat`) | **analysis wrong, port correct** |
| `strel('disk', 5)` size | analysis says 9×9 / 69 px | — | 9×9 / 69 px | 9×9 / 69 px | match |
| `regionprops` probe: `MajorAxisLength` / `MinorAxisLength` on the 12×14 fixture | analysis §0.5 says **10.35814 / 6.76639** | — | **10.358106 / 6.766408** | `sprintf('%.17g')` → **10.358106483626367 / 6.7664075482213342** | **analysis wrong, port correct** |
| same fixture: Area 53, ConvexArea 56, Solidity 0.94643, Perimeter 25.42200, Centroid (6.92453, 6.09434), BBox [2.5 2.5 9 7], Orientation −26.047 | §0.5 | as listed | identical | identical | match |
| ch9 p. 205 criterion 3 is the min-area **rectangle** ratio; the code uses the **ellipse** axis ratio | ch9 205 | rectangle | both available (`minboundrect` vs `regionprops`); the code's form is the default | — | documented deviation of the book's own code |
| `T_seed` | never printed | — | the code's `strel('disk', 3)` is the only evidence | — | Open item 5 |

## Deviations & justifications
1. **`polybool` closes the ring, `clip_polygon_rect` does not** (measured, the chapter's biggest numerical
   deviation). MATLAB's `polybool('intersection', …)` returns the clipped polygon with `last == first`
   (127 vertices for a 126-point circle); the port's Sutherland–Hodgman returns the open ring. The vertex
   **sets** and the rasterised masks are identical (7/7 clip cases, 0 px), but with `Dmin = 0` the duplicated
   vertex is never removed, so MATLAB's snake carries one extra point through the whole evolution. Measured
   consequences: `for_test.m` — final curve 2.5e-5 px (Hausdorff), burnt mask **0 px**; `GVF_distance` on
   `alg_seg_gray.jpg` — final curves ≤ 0.479 px, `bw1` 61 of 31 730 px. Feeding MATLAB's own closed ring into the
   port reduces the mask difference to **7 px**. → Open item 2 (a one-line fix in `core.polygon`).
2. **Single-precision radii.** `bwdist` returns `single`, so MATLAB evaluates `img_Dist(round(cy), round(cx))/sqrt(2)`
   in single and the initial circle of `dist.m` is a `single` array (the reference is stored as float32); the port
   divides the value promoted to double. Measured: radii ≤ 4.5e-7, circle coordinates ≤ 3.1e-5. The `r == 0 → 2`
   fallback fires on the same 6 of 540 seeds. No effect on any decision.
3. **Long snakes are numerically unstable in their *point count*.** `snakeinterp` inserts where `d > dmax` and on a
   500–2800-point contour hundreds of spacings sit at exactly `dmax`; a **1e-13** perturbation of the starting
   contour already changes the final count (seed 22: 2266 → 2262; seed 1: 533 → 536; MATLAB 2256 / 531). Element-wise
   parity of the long snakes is therefore unobtainable by any float implementation, and the honest metrics are the
   curve (Hausdorff ≤ 0.479 px over 46 seeds) and the mask (0.19 %). Short snakes stay exact (five of the 46 seeds
   agree to ≤ 6e-6, three of them to ≤ 1.5e-13; `for_test.m`'s 777-point snake to 1.1e-7).
4. **`minboundrect.m` does not run in R2025a.** MATLAB's own refusal is recorded:
   `CONVHULL no longer supports or requires Qhull-specific options.` The references come from a copy whose only
   change is dropping `{'Qt'}` (the comment in the file says the option list only silenced warnings). MATLAB's
   modern `convhull` additionally **errors** on collinear or duplicate point clouds
   (`Error computing the convex hull…`) where the port falls back to the two extreme points and returns a
   degenerate rectangle; MATLAB returns `[]` for `nedges = 0` where the port returns NaN, and a **1×2** perimeter
   vector for `nedges = 2` where the port returns the scalar. Superset, documented.
5. **`polygeom`'s principal-axis angle.** An eigenvector's sign is arbitrary; R2025a's `eig` picks the opposite
   one from `numpy.linalg.eigh` in 13 of the 15 test polygons, so `ang1` differs by exactly ±π (the *axes* are
   identical). The port's sign convention is the one that reproduces the M-file's own documented self-test
   (`ang1 = 30°`, `ang2 = 120°`); MATLAB R2025a running the shipped file returns −150° / 120°.
6. **`snakeinterp`'s `max_passes` cap** is an addition (the shipped `while max(d) > dmax` loop is unbounded and the
   header admits a removal bug). On the two fixtures where MATLAB itself dies with `Index exceeds array bounds`
   the port raises a `ValueError` with a diagnostic instead. Stricter, documented.
7. **`for_test.m` shadows its own GVF field**: line 119 `[v, h] = size(bw)` overwrites `v`. The reference keeps a
   copy (`u_gvf`, `v_gvf`) taken immediately before that statement; the port keeps the field.
8. **Display rounding**: `imshow(I, [])` PNGs written by MATLAB (`mat2gray` computes `x·δ − lo·δ`) and by the port
   (`(x − lo)/(hi − lo)`) differ by 1 gray level at exact half-integers — 2 of the 14 raw image pairs (2480 px on
   `fig_6_15_c`, 263 px on the `dist.m` DT). The arrays are identical at L2. Not a port issue (ch05 Deviation 4).
9. **`regionprops` `ConvexHull` vertex list**: MATLAB's `convhull` retains collinear points of a hull edge
   (30 vertices where the port has 8 on `sh_L`); the port's list is a subset and the *polygon* is identical, which
   is what `ConvexImage`/`ConvexArea`/`Solidity` — the quantities the chapter thresholds — depend on (all 0 px).
10. **Unit-normalised GVF force** (`px = u/(mag + 1e-10)`) is the shipped code's convention, not the book's
    Eq. (6.56); the port reproduces the code by default and offers `normalize=False`. Both were run; the
    references use the script form.

## Open items
1. **The chapter's end-to-end demo cannot be shown** (analysis R6, by design): `sea_ice_demo.m` calls
   `ice_shape_enhancement` (book §7.1 → ch7) and `sea_ice_model` (book §8.2 → ch8), which are deliberately not
   ported here; `scripts/ch06_sea_ice_demo.py` stops after `seaice_kmean_gvf` and raises
   `NotImplementedError` behind `--full`. Everything up to and including the three-level `out` is verified against
   MATLAB. Not a port failure — it resolves when ch7/ch8 land. The other three deferred files
   (`SeaIce_Image_Structure.m`, `color_hist.m`, `color_hist_comparison.m`) have no ch6 caller at all.
2. **`core.polygon.clip_polygon_rect` should return a *closed* ring** (porter action, Deviation 1). MATLAB's
   `polybool` appends the first vertex; because `Dmin = 0` that duplicate survives `snakeinterp` and changes the
   snake's discrete system. Measured: with the port's open ring `GVF_distance` on `alg_seg_gray.jpg` differs from
   MATLAB by **61 of 31 730 ice pixels**; fed MATLAB's own closed ring the same code differs by **7 px**. Suggested
   fix: append `(x[0], y[0])` to the result when it is non-empty (and add a `close=` flag if any caller needs the
   open form). The verifier did not change `seaice/`; the tests currently assert the measured `near` bounds and
   will tighten once the fix lands.
3. **Figs. 6.2, 6.3, 6.5–6.13 and 6.18–6.21 are `unverified` as *figures***: the aerial floe-field and model-basin
   images are not shipped with the code and no `.m` file produces them (the analysis' NCC search found no book
   figure for `sea_ice_test.jpg` or `test8.jpg`). The mechanisms are reproduced on synthetic fixtures
   (`scripts/ch06_snake_parameters.py` α/β influence and convergence, `scripts/ch06_ushape.py` the U-shape
   concavity, `scripts/ch06_capture_range.py` the capture range) and every *numeric* claim of §6.5.2 is checked.
   Needs the source images from the user; not a port failure.
4. **`core.matlab_compat.del2` emits `RuntimeWarning: invalid value encountered in subtract`** on an all-`±Inf`
   array (`np.diff` of `Inf − Inf`). MATLAB produces exactly the same NaNs (the reference matches bit for bit,
   NaN pattern included); the port should wrap the differencing in `np.errstate(invalid='ignore')`, as
   `imimposemin` was fixed to do in ch05 (review S8). Cosmetic — 1 warning in the ch06 suite.
5. **`T_seed` is never given a value in the book** (analysis R12). The only evidence is the code's
   `se = strel('disk', 3)`, whose real neighbourhood is **5×5 / 25 px** (not the 7×7 / 37 px the analysis states —
   see the Numbers table). The port takes `se_radius` as a parameter and defaults to the script's 3.
6. **`analysis/ch06.md` needs two corrections** (verifier findings, both settled against MATLAB itself):
   (a) §0.5's `MajorAxisLength 10.35814` / `MinorAxisLength 6.76639` are transcription slips — R2025a returns
   `10.358106483626367` / `6.7664075482213342`, which the port reproduces to 1e-12 (the porter's §8 note 4 was
   right); (b) §2.2 stage 15 and §6.4 state `strel('disk', 3)` = "7×7 / 37 px"; MATLAB returns 5×5 / 25 px.
   `homofil.m` can also be moved from "expect `unverified`" to `exact` (it runs in MATLAB, ≤ 1.14e-12).

## Verdict: PASS
All 8 scripts run headless (8 default + 8 non-default CLI combinations, exit 0), every one of the **34 `.m` files**
has a row — 29 of them ported and measured against MATLAB R2025a running the original code, 5 explicitly deferred
to ch7/ch8 with Open item 1 — and every new `seaice/core` primitive has its own row. The high-risk targets are
closed: `core.regionprops` is **exact** (≤ 1.07e-14) on 40 constructed shapes *and* on all 344 components of the
three real ch6 masks, in both MATLAB call forms, so the `Rc = 0.9` / `Rl = 2` decision sets are identical;
`del2`, `GVF`, `gradient2`, `xconv2`, `gaussianMask/Blur`, `snakeindex`, `snakeinterp`, `snake_matrix`,
`snakedeform` (dense) and `interp2('*linear', 0)` are **bit-exact** on 200+ controlled cases including the
border/`±Inf`/`ITER = 1` cases the analysis listed as untested; `poly2mask`/`roipoly` are 0 px; `clip_polygon_rect`
reproduces `polybool`'s vertex set and mask exactly; and the k-means stage matches MATLAB's `rng(0)` run to double
precision with a 0-pixel mask on both book images. The residual differences are quantified, explained and bounded:
0.19 % of the ice pixels of the Fig. 6.15 segmentation, traced to `polybool`'s closing vertex (Open item 2) plus a
proven 1e-13-level instability of the snake's point count (Deviation 3), with the curves themselves within 0.48 px.
The test suites pass (`tests/test_ch06.py` **388 passed**, 0 failed, 0 xfail; full suite **1533 passed,
2 skipped, 1 xfailed**, no regression against the ch05 baseline). No
`unverified` row remains: the only unverifiable items are the unshipped §6.5 book images, covered by Open item 3.
