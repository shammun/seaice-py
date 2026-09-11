# MATLAB → Python function map (starting point; the project's `knowledge/function_map.md` records what was verified)

Legend for "Parity": E exact · N near · A approx (different algorithm) · R re-implement.

## I/O, types, display
| MATLAB | Python | P | Notes |
|---|---|---|---|
| `imread(f)` | `imageio.v3.imread(f)` | E/N | JPEG decoders may differ by ±1 |
| `imwrite(I,f)` | `imageio.v3.imwrite(f, I)` | E | |
| `im2double` / `im2uint8` / `im2single` | `core.matlab_compat.im2double` / `im2uint8` / `img_as_float32` | E | verified ch02 |
| `uint8(x)` cast (0–255 values) | `core.matlab_compat.to_uint8_saturating(x)` | E | round half-away + saturate; not `astype` (wraps) — verified ch02 |
| `round(x)` | `core.matlab_compat.matlab_round(x)` | E | half away from zero — verified ch02 |
| `mat2gray(I)` | `(I-I.min())/(I.max()-I.min())` | E | |
| `imcomplement(I)` | `core.matlab_compat.imcomplement(I)` | E | **returns the INPUT class** (`imcomplement.m` l. 4 "IM2 has the same class as IM"): `~BW`; `intmax(class)−I` for **all four** unsigned classes; `bitcmp(I)` = `−1−I` for signed; `1−I` **evaluated in the input float class** (a float64 port promotes `single` — `imcomplement(single(1e-8))` is exactly `1`). Verified ch02, **corrected ch07** (11 classes) |
| `rgb2gray` | `core.matlab_compat.rgb2gray_matlab` | E | NTSC weights 0.298936/0.587043/0.114021 in double + half-away rounding — verified ch02 (0 px differ) |
| `rgb2hsv`/`hsv2rgb` | `skimage.color.rgb2hsv/hsv2rgb` | E | |
| HSI (book Eq. in §2.1) | `core.color.rgb2hsi` | R | verified ch02 vs corrected MATLAB snippet |
| CMY = 1−RGB | `core.color.rgb2cmy` (= `imcomplement`) | E | verified ch02 |
| CMYK (book Eqs. 2.4–2.5) | `core.color.rgb2cmyk(u, b)` | R | verified ch02 |
| `rgb2ind` | `PIL.Image.quantize` | A | only for §2.1 illustration |
| `ind2rgb(idx, cmap)` | `core.color.indexed_to_rgb(idx, cmap, one_based)` | E | integer idx 0-based, double 1-based, clipped — verified ch02 |
| `imshow(I)`, `imshow(I,[])` | `plt.imshow(I, cmap='gray', vmin=0, vmax=255)` / auto | — | |
| `subplot`, `figure`, `title` | matplotlib | — | |
| `label2rgb(L)`, `label2rgb(L, 'jet', 'k', 'shuffle')` | `core.plotting.label2rgb(L, cmap, background, shuffle, seed)` (or `skimage.color.label2rgb(L, bg_label=0)`) | — (display) | colours differ; MATLAB's `'shuffle'` uses a private fixed-seed stream — never compare (ch05) |
| `montage` | grid with matplotlib | — | |
| `imcrop(I, rect)` | slicing (`rect` = [x y w h], 1-based) | E | |
| `padarray(A, [p q], 'replicate'/'symmetric'/0)` | `np.pad(A, ((p,p),(q,q)), mode='edge'/'symmetric'/'constant')` | E | |
| `VideoReader(f)` + `read(v)` + `get(v,'numberOfFrames')` | `core.video.read_video(path) -> (H, W, 3, N) uint8`, `video_num_frames` (backed by `imageio.v3` / `cv2.VideoCapture`) | E on **Uncompressed AVI** only | `read` returns **`(H, W, 3, N)`** — keep that layout so `vidFrames(:,:,:,k)` ports as `frames[:, :, :, k]`; imageio gives `(N, H, W, 3)`. **Pin the axis order against MATLAB's own `size()`, not just the pixels.** Decode parity is a **precondition**: a lossy container decoded by two codecs can differ by several gray levels and one level moves an Otsu threshold — write every reference uncompressed and measure 0 differing bytes before claiming anything downstream (verified ch09: 0 of 22 118 400 and of 9 216 000). `mmreader` is **removed** from current MATLAB |
| `movie2avi(M, name, 'FPS', f)` (**removed**) / `VideoWriter(name, 'Uncompressed AVI')` + `writeVideo` | `core.video.write_video(path, frames, fps, codec='rawvideo')` | E | bit-identical to `VideoWriter`'s Uncompressed AVI (0 differing bytes, identical file sizes) — verified ch09 |

## Histogram / intensity
| MATLAB | Python | P | Notes |
|---|---|---|---|
| `[z, x] = hist(y, n)` / `hist(y, centres)` | `core.histogram.hist(y, bins)` → `(counts, centers)` | E | **centres, not edges** (`np.histogram` is wrong twice over): scalar `n` returns the centres of `n` equal bins over `[min, max]` (widened by `±n/2 − 0.5` when `min == max`); a **vector** *is* the centres and the two outer bins are **unbounded**, so out-of-range values are counted. `−Inf` falls in the first bin, `+Inf` in the last; an all-non-finite input gives `miny = maxy = 0`. Verified ch07 (23 cases) |
| `hist(...)` bin comparison edges (`hist.m` l. 145 `edges + eps(edges)`) | `edges + abs(np.spacing(edges))` — **not** `np.nextafter(edges, np.inf)` | E | `eps(x)` is the spacing at `\|x\|`: at a negative exact power of two `nextafter` steps half as far and flips a bin. Verified ch07 bit-exact vs MATLAB's `eps` |
| `imhist(I)` (uint8) | `core.histogram.imhist(I)` | E | logical → 2 bins — verified ch02 |
| `imhist(I, n)` | `core.histogram.imhist(I, n)` | E | MATLAB bin rule `round(v·(n−1)/top)` reproduced — verified ch02 |
| `histeq(I)` | `skimage.exposure.equalize_hist` / re-implement 64-bin | A/R | |
| `imadjust(I,[lo hi],[0 1],gamma)` | `skimage.exposure.rescale_intensity` + `adjust_gamma` | N | |
| `stretchlim` | `np.percentile(I, [1, 99])` | N | |
| `[level, em] = graythresh(I)` | `core.threshold.graythresh(I) -> (level, em)` | E | tie averaging → half-integer `255·level`; `em` = η(t*); RGB not converted — verified ch03 (bit-identical). `skimage.filters.threshold_otsu` is only N |
| `[t, em] = otsuthresh(counts)` | `core.threshold.otsuthresh(counts)` | E | verified ch03 |
| `im2bw(I, t)` / `im2bw(I)` | `core.threshold.im2bw(I, level=0.5)` (`I > t*255` in double, strict) | E | uint8/uint16/int16/float/RGB/logical — verified ch03 |
| `multithresh(I, N)`, N ≤ 2 | `core.threshold.multithresh(I, N)` | E | single-precision `getpdf` + `grayto8` product emulated; output in input class — verified ch03 |
| `multithresh(I, N)`, N ≥ 3 | `core.threshold.multithresh(I, 3)` (exhaustive; N > 3 raises) | R | MATLAB `fminsearch` is local → never E — verified ch03 |
| `imquantize(A, levels[, values])` | `core.threshold.imquantize(A, levels, values)` | E | `1 + Σ(A > level_i)` — verified ch03 |
| `local_Otsu.m` / `block_threshold.m` block Otsu | `core.threshold.block_otsu(gray, n_r, n_c)` | E | verified ch03 |
| `num2str(x)` scalar | `ch03_ice_pixel_detection._num2str(x)` | E | `%d` / `%.{max(floor(log10|x|)+5,5)}g` — verified ch03 |
| `imbinarize(I,'adaptive','Sensitivity',s)` | `skimage.filters.threshold_local` | A | different local statistic; consider re-implementing MATLAB's (local mean × (1−s) style) |
| `adaptthresh` | same as above | A | |
| authors' `kmeans.m` (histogram k-means, shadows the toolbox) | `core.clustering.kmeans_gray(gray, k, shift_bug=True)` | E | deterministic equal-division init; `shift_bug=True` = script's units bug — verified ch03 |
| Statistics Toolbox `kmeans(X,k)` | `core.clustering.kmeans_lloyd(X, k, init='kmeans++', seed=0)` | A | R2025a defaults `sqeuclidean`/`Start 'plus'`/1 replicate/100 iterations = the same algorithm, different RNG. Verified ch06: sorted centres equal to double precision and the mask 0 px vs MATLAB under `rng(0)` on two images — **but 20 restarts show the optimum is not unique**, so compare sorted centres + pixel agreement, never labels |
| `pdist2(a, b, metric)` / distance formulas | `core.clustering.pairwise_distance(X, Y, metric)` | E | closed forms; L1 only — ch03 |

## Neighbourhoods, connectivity, distance (Ch2 §2.3–2.4, Ch5)
| MATLAB | Python | P | Notes |
|---|---|---|---|
| `bwlabel(BW, conn)` | `core.connectivity.label_components(BW, conn)` | E (incl. label numbers) | column-major relabel of skimage output — verified ch02 |
| `bwconncomp` | `label` + `regionprops` | E | |
| `bwareaopen(BW, p[, conn])` | `core.connectivity.bwareaopen(BW, p, conn)` (keep components with ≥ p pixels via `bwlabel` port) | E | verified ch04; `remove_small_objects` changed its threshold argument in skimage 0.26 |
| `bwselect` | `label` + seed lookup | E | |
| `bwareafilt` | filter by `regionprops` area | E | |
| `bwdist(BW, 'euclidean')` | `core.distance.bwdist(BW, 'euclidean')` (`distance_transform_edt(~BW)`) | E | MATLAB single → atol 1e-4 — verified ch02 |
| `bwdist(BW, 'cityblock')` | `core.distance.bwdist(BW, 'cityblock')` (`distance_transform_cdt(~BW, metric='taxicab')`) | E | verified ch02 |
| `bwdist(BW, 'chessboard')` | `core.distance.bwdist(BW, 'chessboard')` (`distance_transform_cdt(~BW, metric='chessboard')`) | E | verified ch02 |
| `bwdist(BW, 'quasi-euclidean')` | `core.distance.bwdist(BW, 'quasi-euclidean')` (→ `quasi_euclidean_dt`) | R/N | two-pass chamfer (1, √2); MATLAB single is the less precise side — verified ch02 |
| `[D, IDX] = bwdist(...)` | `distance_transform_edt(~BW, return_indices=True)` | E | IDX is linear column-major in MATLAB |
| `bwperim(BW)` | `BW & ~binary_erosion(BW, footprint=cross)` | E | MATLAB uses 4-conn for perim |
| `bwboundaries(BW)` / DIPUM `boundaries(BW, conn, dir)` | `core.chaincode.boundaries(BW, conn, direction)` (Moore, exterior only, closed, `bwlabel` order) | R (E vs DIPUM `boundaries.m`) | verified ch02; `bwboundaries` itself (holes, `'noholes'`) not yet compared |
| DIPUM `fchcode(b, conn, dir)` / `bound2im` | `core.chaincode.fchcode` / `bound2im` | E | `minmag` tie-break where MATLAB errors — verified ch02 |
| `bwtraceboundary(BW, P, dir)` | `core.chaincode.trace_boundary` | R | |
| `regionprops(L, props)` | **`core.regionprops.regionprops(L, props, conn=8)`** (MATLAB algorithms) | E | verified ch06 (≤ 1.07e-14, 40 shapes × 10 props + 344 real components, both call forms). `skimage.measure.regionprops` is `A`: different axis lengths, perimeter (+5.5 %) and orientation (+90°) — never for parity |
| `regionprops(L, 'Centroid')`, `cat(1, s.Centroid)` | `core.regionprops.regionprops(L, 'Centroid')` (1-based `(x, y)`); `ch05_watershed.component_centroids` is a thin wrapper | E | verified ch05/ch06 |
| `bwperim(BW)`, `bwmorph(BW, 'perim8')` | `core.connectivity.bwperim(bw, conn)` | E | verified ch06 (9 shapes × conn {4, 8} 0 px); image-border pixels are perimeter pixels |

## Filtering, gradients (Ch2 §2.5, Ch4 §4.1, Ch6)
| MATLAB | Python | P | Notes |
|---|---|---|---|
| `conv2(A, K, 'same')` | `core.filters.conv2(A, K, 'same')` | E | verified ch02 (even kernels too) |
| `conv2(A, K, 'valid'/'full')` | `core.filters.conv2(A, K, mode)` | E | verified ch02 |
| `imfilter(I, h)` | `core.filters.imfilter(I, h)` (correlation) | E | uint8 output: clip + round like MATLAB — verified ch02 |
| `imfilter(I, h, 'replicate')` | `core.filters.imfilter(I, h, 'replicate')` (positional options accepted) | E | verified ch02 |
| `imfilter(I, h, 'symmetric'/'circular'/X)` | `core.filters.imfilter(I, h, 'symmetric')` etc. | E | verified ch02 |
| `imfilter(I, h, 'conv')`, `'full'` | `core.filters.imfilter(I, h, 'conv')`, `'full'` | E | verified ch02 |
| `fspecial(...)` (sobel/prewitt/laplacian/gaussian/log/average/disk/unsharp) | `core.filters.fspecial(kind, p2, p3)` | E | verified ch04 (26 kernels ≤ 7e-18); `'log'` is mean-subtracted, not the sampled ∇²G |
| `imgaussfilt(I, s)` | `scipy.ndimage.gaussian_filter(I, s, truncate=2)` ⚠ | N | MATLAB kernel size 2·ceil(2s)+1 → `truncate=2.0`; padding 'replicate' → `mode='nearest'` |
| `medfilt2(I, [m n])` | `scipy.ndimage.median_filter(I, size=(m,n), mode='constant')` | E | verified ch04; `'symmetric'` → `mode='reflect'` |
| `ordfilt2(I, k, dom)` | `scipy.ndimage.rank_filter(I, k-1, footprint=dom)` | E | |
| `stdfilt`, `entropyfilt`, `rangefilt` | `generic_filter(np.std)`, `skimage.filters.rank.entropy`, `maximum-minimum` | N/A | |
| `[gx, gy] = gradient(F)` | `gy, gx = np.gradient(F)` | E | |
| `imgradient(I)` (Sobel default) | `np.hypot(sobel_h, sobel_v)` with MATLAB Sobel kernel | E | skimage `sobel` normalises by 4 → scale |
| `[BW, t, gv, gh] = edge(I, 'sobel'/'prewitt'/'roberts', T, dir, 'thinning')` | `core.edges.edge(I, method, thresh, direction, thinning) -> EdgeResult` | E | verified ch04 (252 maps 0 px vs R2025a): `/8` (`/6`) kernels, replicate padding, `b > T²` or `4·mean(b)`, zero-padded thinning rule (see SKILL.md); float input only |
| `edge(I, 'log', t, sigma)`, `edge(I, 'zerocross', t, H)` | `core.edges.edge(I, 'log', t, sigma=)` / `'zerocross'` (→ `log_zero_crossings`) | E (N on flat synthetic patches) | verified ch04; threshold is on the jump across the crossing |
| `edge(I, 'canny')` | `skimage.feature.canny(I, sigma=sqrt(2))` | A | |
| `del2(F)`, `del2(F, h)`, `del2(F, hx, hy)` | `core.matlab_compat.del2(f, hx=1.0, hy=None)` | E | verified ch06 (21 cases 0.0) — `∇²/(2·ndims)` with **linearly extrapolated borders**, `n == 3` copy, `n ≤ 2` → 0; `4·del2` is the 5-point Laplacian |
| `[u, v] = GVF(f, mu, ITER)`, `snakedeform`, `snakeinterp`, `snakeindex`, `BoundMirror*`, `gradient2`, `xconv2`, `gaussianMask/Blur` (Xu & Prince) | `core.snake.*` | E | verified ch06 on 200+ MATLAB cases; `gvf` normalises `f` **in the input's class**; `snakedeform(solver='circulant')` = the same system by FFT (306 s → 2.1 s) |
| `hough`, `houghpeaks`, `houghlines` | `skimage.transform.hough_line(_peaks)`, `probabilistic_hough_line` | A | |

## Morphology (Ch4 §4.2, Ch5, Ch7)
| MATLAB | Python | P | Notes |
|---|---|---|---|
| `strel('disk', r[, n])` (n=4 default → octagon), `'dis'` prefix | `core.morphology.strel('disk', r[, n])`; `disk_decomposition(r, n)` = `decompose` | E | verified ch04 (116 nhoods = `getnhood`); `skimage.morphology.disk(r)` is only the `n = 0` Euclidean case |
| `strel('square', n)` / `'rectangle'` / `'line'` / `'diamond'` / `'octagon'` / `'periodicline'` / `'pair'` / `strel(nhood)` | `core.morphology.strel(shape, *params)` | E | verified ch04 |
| `imerode(I, se)` / `imdilate` | `core.morphology.imerode/imdilate(I, se)` (OpenCV `BORDER_CONSTANT` + explicit pad, flipped kernel for dilation; numpy fallback for int32/uint32/int64/uint64) | E | verified ch04 (148 arrays); erosion pads +Inf/intmax/1, dilation −Inf/intmin/0 with the reflected SE; MATLAB rejects int64/uint64 |
| `imopen` | `core.morphology.imopen` (bare composition) | E | verified ch04 |
| `imclose` | `core.morphology.imclose` (MATLAB pre-pad `ceil(size/2)`: 0 via `imclose.m`, class-min via Halide) | E | verified ch04 (72 cases); `skimage.morphology.closing` is only N at the border |
| `K − J` on logical / uint8 (morphological gradients) | `core.morphology.morphological_gradient(I, se, kind)` | E | verified ch04; logical − logical → double, uint8 saturates |
| `imtophat`/`imbothat` | `white_tophat`/`black_tophat` | E | |
| `imreconstruct(marker, mask[, conn])` | `core.morphology.imreconstruct(marker, mask, conn)` (skimage `reconstruction`, `marker <= mask` enforced) | E | verified ch04 (7 cases); erosion dual `reconstruct_by_erosion` is R (no builtin) |
| `imfill(I, 'holes')` / `imfill(BW, 'hole')` / `imfill(I, conn, 'holes')` | `core.morphology.imfill(I, 'holes', conn=4)` (port of `imfill.m` l. 124–145; returns the **input class**) | E | verified ch07 (21 cases × conn {4,8}, values **and** class). Default conn = `conndef(2,'minimal')` = **4**; `'hole'`/`'h'` are `validatestring` prefixes |
| `imfill(BW, 'holes')` via SciPy | `scipy.ndimage.binary_fill_holes(BW)` | E/A | correct **only** for a *logical* image at conn 4; 3 of 13 logical cases differ at conn 8 and it never reproduces a numeric input (wrong values for uint8, wrong values *and class* for grayscale/int16/±Inf) — measured ch07 |
| `imfill(I)` grayscale | `core.morphology.imfill(I, 'holes')` (same M-code path; the grayscale branch is the default one) | E | verified ch07 on three grayscale surfaces + an `±Inf` plant |
| `imclearborder` | `skimage.segmentation.clear_border` | E | |
| `imregionalmax/min(I[, conn])` | `core.morphology.imregionalmax/imregionalmin(I, conn)` (`local_maxima/local_minima(allow_borders=True)` + constant image → all True; NaN raises) | E | verified ch05 (126 cases); bare skimage differs on constant images |
| `imhmax/imhmin(I, h)` | `h_maxima/h_minima(I, h)` | E | |
| `imextendedmax/min` | `local_maxima(h_maxima(...))` | E | |
| `imimposemin(I, markers[, conn])` | `core.morphology.imimposemin(I, BW, conn)` (M-code port: `∓Inf` markers, `h` rule, arithmetic in the input class) | E | verified ch05 (88 cases bit-identical incl. single) |
| `watershed(I)`, `watershed(I, conn)` | `core.watershed.watershed(I, conn=8)` (line-by-line port of `eml/watershed.m`; int32 labels, 0 = line) | E (label values) | verified ch05 (74 fixture cases + script images); MATLAB rejects int16/int32 input |
| `watershed` via skimage | `core.watershed.watershed_skimage(I, conn)` = `skimage.segmentation.watershed(I, connectivity=2\|1, watershed_line=True)` | A | different queue rules → ridges and partitions differ; cross-check only (ch05) |
| `bwmorph(BW, 'thin', Inf)` | `skimage.morphology.thin(BW)` | N | |
| `bwmorph(BW, 'skel', Inf)` | `skeletonize(BW)` | A | different algorithm |
| `bwmorph(BW, 'spur'/'clean'/'bridge'/'remove'/'majority'/'fill')` | `core.morphology.bwmorph` (3×3 LUT) | R | |
| `bwulterode` | `local_maxima(distance_transform_edt(BW))` | N | |

## Geometry, interpolation, calibration (Ch2 §2.8, Ch9, App. A)
| MATLAB | Python | P | Notes |
|---|---|---|---|
| `imresize(I, s, 'nearest'/'bilinear')` enlarge, antialias off | `core.interp.resize(I, s, method)` | E | verified ch02 (0.0 full array) |
| `imresize(I, s, 'bicubic')`, or any shrink | `core.interp.resize` (A) → write `core.interp.imresize_matlab` if needed numerically | A | border rule + antialiasing differ — verified ch02 |
| `interp2(X,Y,V,Xq,Yq,'nearest'/'linear'/'cubic')` | `core.interp.interp2(V, u=Yq−1, v=Xq−1, method)` | E | all three exact incl. borders (Keys a=−0.5, quadratic edge) — verified ch02 |
| `imrotate`, `imwarp`, `fitgeotrans`, `projective2d` | `skimage.transform.rotate/warp/ProjectiveTransform/estimate` | N | App. A orthorectification (DLT) → implement DLT from book Eqs. **One rectifier only** — ch7 and ch8 both deliberately declined to write theirs and left both App. A forms (A.1.1 analytic, A.1.2 linear 4-corner) to the calibration chapter |
| `a:step:b` (colon), `jet(m)` used numerically | `core.matlab_compat.matlab_colon`, `core.plotting.matlab_jet` | N / E | verified ch08; see SKILL §1 |
| `polyfit`/`polyval` | `np.polyfit`/`np.polyval` | E | |
| `fminsearch` | `scipy.optimize.minimize(method='Nelder-Mead')` | N | |
| `lsqnonlin`/`lsqcurvefit` | `core.fitting.lsqcurvefit` (wraps `scipy.optimize.least_squares(method='trf')`, returns all six MATLAB outputs) | N | verified ch08: same Coleman–Li family, different code; MATLAB's defaults captured from `optimoptions` (`trust-region-reflective`, tol 1e-6, `MaxIter` 400, `MaxFunEvals` 100·nvars); report `resnorm`/`exitflag` — a gap > ~1e-5 rel. is a different local minimum |
| `optimset('TolFun',…,'MaxIter',…,'MaxFunEvals',…)` | `core.fitting.optimset(...) -> LsqOptions` | E | verified ch08; `TolFun→ftol`, `TolX→xtol`, `MaxFunEvals→max_nfev`, `MaxIter` recorded only |
| `poly2mask(x, y, m, n)` | `core.polygon.poly2mask(x, y, M, N)` (port of `eml/poly2mask.m`) | E | verified ch06 (0 px on 19 masks); `skimage.draw.polygon2mask` is `N` |
| `roipoly(I, xi, yi)` / `roipoly(m, n, xi, yi)` (non-interactive forms) | `core.polygon.roipoly(m, n=None, xi=None, yi=None)` | E | verified ch06 (== `poly2mask`); only the 1-argument call is interactive |
| `polybool('intersection', ...)` (Mapping TB, compiled GPC) | `core.polygon.clip_polygon_rect(x, y, x_range, y_range)` (Sutherland–Hodgman, closed ring) | R | verified ch06: vertex **sets**/masks identical, **ordering not reproducible** (start vertex rotated, traversal reversed) |
| `polyxpoly(x1,y1,x2,y2)`, `polyarea(x,y)`, `convhull(x,y)` | `core.polygon.polyxpoly` / `polyarea` / `convhull` | R / E / N | verified ch06; `convhull` keeps collinear hull points and rejects the old `{'Qt'}` option list. **Its vertex *order* is not reproducible either** (105 of 227 hulls, ch08) — compare vertex sets and rasterised masks |
| `polygeom(x,y)` (Sommer, FEX), `minboundrect(x,y)` (D'Errico, FEX) | `core.polygon.polygeom`, `core.polygon.minboundrect` | N / E | verified ch06; `polygeom`'s principal angle differs by ±π (eigenvector sign); `minboundrect` = the minimum-area bounding rectangle used for length-to-width criteria |
| `regionprops(... 'Orientation','MajorAxisLength',...)` | `core.regionprops` (see above) | E | verified ch06 |
| `graycomatrix(I, 'Offset', [0 1], 'NumLevels', 8)` | `graycomatrix(I_quantised, [1], [0], levels=8, symmetric=False)` | E | quantise the same way |
| `graycoprops` | `skimage.feature.graycoprops` | E | |
