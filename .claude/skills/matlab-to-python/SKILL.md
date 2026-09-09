---
name: matlab-to-python
description: How to port MATLAB / Image Processing Toolbox code to Python (numpy, scipy, scikit-image, OpenCV) faithfully — indexing, dtype, column-major, toolbox function equivalents and their known deviations. Load before writing or reviewing any ported code.
---

# matlab-to-python — faithful porting rules

Read `reference/function_map.md` (in this skill folder) for the function-by-function table. This file is the rules.
Also read `knowledge/function_map.md` in the repo root, which accumulates *project-verified* mappings across chapters —
it overrides this skill when they disagree.

## 1. Semantics that silently break ports
| MATLAB | Python | Rule |
|---|---|---|
| 1-based, inclusive `a:b` | 0-based, exclusive | `A(i,j)` → `A[i-1, j-1]`; `A(a:b)` → `A[a-1:b]`; `end` → `-1` / `len` |
| Column-major `A(:)`, `find`, `sub2ind` | row-major | `A.flatten(order="F")`, `np.nonzero` order differs — sort if order matters; `np.ravel_multi_index(..., order="F")` |
| `size(A)` returns rows, cols | `A.shape` | `[m,n]=size(I)` → `m, n = I.shape[:2]` |
| `zeros(n)` is n×n | `np.zeros((n, n))` | never `np.zeros(n)` for `zeros(n)` |
| `length(A)` = max dim | `max(A.shape)` | |
| `uint8` arithmetic **saturates** (200+100=255) | numpy **wraps** (200+100=44) | do math in float (`img_as_float` / `.astype(float)`), convert back with clipping |
| `round` = half away from zero | numpy = banker's | use `np.floor(x + 0.5)` (positive) or `np.sign(x)*np.floor(abs(x)+0.5)` when matching matters |
| `A'` conjugate transpose | `A.T` | `.T` for real arrays |
| `.*`, `./`, `.^` elementwise; `*` matmul | `*`, `/`, `**` elementwise; `@` matmul | |
| `&`/`|` elementwise, `&&`/`||` short-circuit | `&`/`|` arrays, `and`/`or` scalars | |
| `mod(a,b)` sign of b; `rem` sign of a | `np.mod` = MATLAB `mod`; `np.fmod` = `rem` | |
| `im2double` on uint8 = /255 | `skimage.util.img_as_float` | `im2uint8` = `img_as_ubyte` (rounds ×255) |
| `uint8(x)` **cast** (values already 0–255) ≠ `im2uint8(x)` (×255 first) | `np.clip(np.floor(x+0.5), 0, 255).astype(np.uint8)` (project: `core.matlab_compat.to_uint8_saturating`) | the cast rounds half-away-from-zero and saturates; `x.astype(np.uint8)` wraps *and* truncates — never use it for a MATLAB cast (verified ch02) |
| `imhist(BW)` on a **logical** image → 2 bins at [0, 1] | `np.bincount(BW.ravel(), minlength=2)` (project: `core.histogram.imhist`) | `imhist(I, n)` bins by `round(v·(n−1)/top)`, floats clipped to [0,1]; uint8 default 256 (verified ch02) |
| `imshow(I,[])` scales | `plt.imshow(I, cmap="gray")` (auto-scales) | `imshow(I)` uint8 → `vmin=0, vmax=255` |
| `imread` JPEG | `imageio.v3.imread` / `skimage.io.imread` | both give HxWx3 uint8 RGB — same decoder differences possible (libjpeg versions): compare with a tolerance of ±1 gray level for parity tests on JPEGs |
| `rgb2gray` (Rec.601: 0.2989 0.5870 0.1140, uint8 rounded) | `skimage.color.rgb2gray` uses **Rec.709** (0.2125 0.7154 0.0721) | write `core/matlab_compat.rgb2gray_matlab` |
| `imfilter(I,h)` = **correlation**, zero padding, 'same' | `scipy.ndimage.correlate(I, h, mode="constant")` | `imfilter(...,'conv')` or `conv2(...,'same')` → `scipy.signal.convolve2d(I, h, mode="same", boundary="fill")`; `'replicate'` → `mode="nearest"`; `'symmetric'` → `mode="reflect"`. MATLAB passes these as **positional strings in any order** (`imfilter(I,h,'replicate')`): give the Python wrapper `*options` so a mechanical port cannot misassign them to a `mode` argument (project: `core.filters.imfilter`, verified ch02). A book may print a "convolution" equation in correlation form — check which call the `.m` file uses before labelling parity |
| `medfilt2` zero-pads | `scipy.ndimage.median_filter(mode="constant", cval=0)` | scipy default is `reflect` |
| `strel('disk',r)` (default n=4 → *approximate*, decomposed disk) | `skimage.morphology.disk(r)` is an *exact* Euclidean disk | MATLAB's default disk is NOT round. Generate MATLAB's neighborhood via Octave `getnhood(strel('disk',r))` only if exact parity is needed; otherwise use exact disk and label `near`. Octave's disk may also differ — check |
| `bwdist(BW)` distance to nearest **nonzero** | `scipy.ndimage.distance_transform_edt(~BW)` | note the complement! `cityblock` → `distance_transform_cdt(~BW, metric="taxicab")`, `chessboard` → `metric="chessboard"`, `quasi-euclidean` → re-implement (two-pass chamfer 1, √2; exact in float64 while MATLAB's single-precision result is ~1e-4 off). MATLAB returns **single** → compare with `atol≈1e-4` (verified ch02, project `core.distance.bwdist`) |
| `bwlabel(BW)` default 8-conn | `skimage.measure.label(BW, connectivity=2)` | `bwlabel(BW,4)` → `connectivity=1`. MATLAB numbers labels in **column-major order of first occurrence**; relabel skimage's output that way when code indexes cells/`regionprops` by label number (project: `core.connectivity.label_components`, exact incl. numbering, verified ch02) |
| `watershed(A)` 8-conn, 0 on ridge lines | `skimage.segmentation.watershed(A, connectivity=2, watershed_line=True)` | MATLAB watershed uses Fernand Meyer flooding on regional minima; results can differ at plateaus → `near`/`approx` |
| `regionprops` `Centroid` = (x, y) | skimage `centroid` = (row, col) | `Orientation` degrees, sign flipped vs skimage radians; `Perimeter` algorithms differ; `MajorAxisLength` → `axis_major_length`; `ConvexArea` → `area_convex`; `BoundingBox` [x y w h] with 0.5 offsets |
| `imresize` bicubic (Keys a=−0.5, antialias when shrinking) | cv2 bicubic a=−0.75; skimage order=3 = B-spline | for Ch2 §2.8 **implement the kernels from the book's equations**; never claim exact parity with library resize |
| `[level, em] = graythresh(I)` = `im2uint8` → `imhist(256)` → `otsuthresh`, level∈[0,1], `em` = separability η(t*) | port `otsuthresh` (project: `core.threshold.graythresh`, exact, verified ch03); **not** `skimage.filters.threshold_otsu` | MATLAB **averages tied maximisers** (`mean(find(σB² == max))`), so `255·level` can be a half-integer — keep it float. `im2bw(I, level)` = `I > 255·level` (strict >, compared in **double**: `uint8 > 104.5` works); `im2bw` converts RGB with `rgb2gray`, `graythresh` does **not** (it histograms all planes) |
| `[thresh, metric] = multithresh(I, N)` | port R2025a `multithresh.m` (project: `core.threshold.multithresh`, exact N ≤ 2, verified ch03); `skimage.filters.threshold_multiotsu` is `approx` | `getpdf` normalises in **single** and the `grayto8` builtin multiplies by 255 in single too — a float64 product moves a bin edge; output thresholds come back in the **input class**; N ≥ 3 uses `fminsearch` (local; can return its initial guess with `metric = −Inf`) → an exhaustive port is `reimplemented`, never `exact`; degenerate inputs (`numel(unique(A)) ≤ N`) use `getDegenerateThresholds`. `imquantize(A, t)` = `1 + Σ(A > t_i)` |
| `histeq` default 64 bins | `skimage.exposure.equalize_hist` (256) | `approx` unless re-implemented |
| `kmeans` (Statistics Toolbox: k-means++ init, random) | `sklearn.cluster.KMeans(n_clusters=k, n_init=10, random_state=0)` | never expect identical labels; compare cluster *centers* sorted, and pixel agreement ≥ 99%. **First check whether the chapter folder ships its own `kmeans.m`** — a script named like a toolbox function *shadows* it (the book's `ch3/kmeans.m` is a deterministic histogram k-means with equal-division init, ported exactly as `core.clustering.kmeans_gray`, verified ch03); never `addpath` such folders when generating references |
| `num2str(x)` (scalar, default format) | integers → `str(int(x))`; else `f"{x:.{n}g}"` with `n = max(floor(log10|x|) + 5, 5)` (project: `ch03_ice_pixel_detection._num2str`, verified ch03) | 4 significant digits after the leading ones: `73.8472`, `86.496`, `0.12346`, `1234.5679` — needed for byte-identical figure titles |
| `sum(x)` on integer vectors; loop buffers `s(j) = I(p(j))` | `np.sum(x, dtype=np.float64)`; emulate stale entries explicitly | integer `sum` accumulates in **double** (no saturation); an array grown inside a loop is never cleared, so a smaller later class reuses stale entries (verified ch03: the authors' "mean intensity" of 4759.10) — reproduce behind a flag, never silently |
| `edge(I,'sobel')` thresholds & thins | `skimage.filters.sobel` gives magnitude only | re-implement MATLAB's `edge` (threshold = 4·mean(mag) style + thinning) if the book relies on it; document |
| `edge(I,'canny')` sigma=√2, hysteresis thresholds auto | `skimage.feature.canny(sigma=np.sqrt(2))` | `approx` |
| `fspecial('log'/'gaussian'/'laplacian'/'average'/'disk'/'sobel'/'prewitt')` | none exact | implement in `core/filters.py` from MATLAB's documented formulas (gaussian: normalised, hsize; laplacian alpha=0.2; log normalised to sum 0) |
| `imreconstruct(marker, mask)` | `skimage.morphology.reconstruction(seed, mask, method="dilation")` | exact (same algorithm) if `seed <= mask` |
| `imfill(BW,'holes')` | `scipy.ndimage.binary_fill_holes` | exact; grayscale `imfill(I)` → reconstruction by erosion with border seed |
| `imregionalmax/min`, `imhmax/imhmin`, `imextendedmax`, `imimposemin` | `skimage.morphology.local_maxima / h_maxima / h_minima` | `imimposemin` → implement: `fm = -inf at markers`, `g = min(f+1, fm)`… per Gonzalez/Soille; verify vs Octave |
| `bwmorph(BW,'thin'/'skel'/'spur'/'clean'/'bridge'/'remove'/'majority')` | `skimage.morphology.thin`, `skeletonize` | others → LUT re-implementation in `core/morphology.py` (MATLAB uses 3×3 lookup tables) |
| `bwboundaries` / `bwtraceboundary` (Moore tracing, (row,col), 8-conn, closed) | `skimage.measure.find_contours` is **sub-pixel and different** | implement Moore-neighbour tracing in `core/chaincode.py` (needed for Ch2 §2.7 and Ch5 concavity). DIPUM's `boundaries.m`/`fchcode.m` (Gonzalez–Woods–Eddins, often shipped with books) trace **exterior** boundaries only, return closed lists, give two identical points for a single pixel, traverse 1-px spurs twice, and `fchcode`'s `minmag` **errors** on periodic codes — replicate, and decide the tie-break explicitly (verified ch02) |
| `poly2mask` (specific inclusion rule) | `skimage.draw.polygon2mask` | boundary pixels may differ → `near` |
| `gradient(F)` returns `[FX, FY]` (x first) | `np.gradient(F)` returns `(dF/drow, dF/dcol)` | swap; both use central differences with one-sided edges → exact |
| `del2` | none | implement: discrete Laplacian /4 with MATLAB's edge extrapolation |
| `interp2` linear/cubic | `scipy.ndimage.map_coordinates(order=1)` / `RegularGridInterpolator` | scipy cubic = B-spline ≠ MATLAB `'cubic'` = Keys kernel `a = −0.5` with **quadratic edge extrapolation**; implement Keys with that edge rule for exact parity incl. borders (project: `core.interp.interp2`, verified ch02). `interp2(X,Y,...)` has `X` = column, `Y` = row, 1-based → swap and subtract 1. Nearest ties round half away from zero; outside → NaN |
| `rgb2hsv` | `skimage.color.rgb2hsv` | exact (H in [0,1]); book's **HSI** (§2.1) and **CMY** have no builtin → implement from book equations |
| `graycomatrix`/`graycoprops` | `skimage.feature.graycomatrix/graycoprops` | MATLAB default 8 levels, offset [0 1], symmetric=false → set explicitly |
| `VideoReader` | `imageio.v3.imiter` / `cv2.VideoCapture` | |
| `activecontour`, GVF snake (Xu & Prince code) | none | port the authors' `.m` code line by line into `seaice/ch06_gvf_snake.py`; verify with Octave running the original `.m` |

## 2. Porting procedure for one `.m` file
1. Read the whole file; list every toolbox call; decide per call: library (which, which args) / `core/` primitive / re-implement.
2. Write the Python version preserving the *structure* (same variable names where sensible, same step order) so the verifier can line them up.
3. Where MATLAB scripts `imshow` intermediate results, the Python script saves them to `outputs/chNN/` with descriptive names and returns arrays.
4. Put every numeric constant that came from the book in a named variable with a comment citing the equation/figure.
5. Add a `# PARITY:` comment at each non-exact substitution.

## 3. Things that are not bugs
- Different label numbers, different cluster indices, JPEG decoder ±1 level differences, Otsu threshold ±1 level on 256 bins,
  float32 vs float64 noise ~1e-6. Handle with the right comparator (see `tools/compare_arrays.py`), not by hacking the algorithm.
