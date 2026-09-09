# MATLAB → Python function map (starting point; the project's `knowledge/function_map.md` records what was verified)

Legend for "Parity": E exact · N near · A approx (different algorithm) · R re-implement.

## I/O, types, display
| MATLAB | Python | P | Notes |
|---|---|---|---|
| `imread(f)` | `imageio.v3.imread(f)` | E/N | JPEG decoders may differ by ±1 |
| `imwrite(I,f)` | `imageio.v3.imwrite(f, I)` | E | |
| `im2double` / `im2uint8` / `im2single` | `img_as_float` / `img_as_ubyte` / `img_as_float32` | E | |
| `mat2gray(I)` | `(I-I.min())/(I.max()-I.min())` | E | |
| `imcomplement(I)` | `255-I` (uint8) / `1-I` (float) / `~BW` | E | |
| `rgb2gray` | `core.matlab_compat.rgb2gray_matlab` | E | Rec.601 weights + MATLAB rounding |
| `rgb2hsv`/`hsv2rgb` | `skimage.color.rgb2hsv/hsv2rgb` | E | |
| HSI (book Eq. in §2.1) | implement | R | |
| CMY = 1−RGB | implement | R | |
| `rgb2ind`/`ind2rgb` | `PIL.Image.quantize` / index lookup | A | only for §2.1 illustration |
| `imshow(I)`, `imshow(I,[])` | `plt.imshow(I, cmap='gray', vmin=0, vmax=255)` / auto | — | |
| `subplot`, `figure`, `title` | matplotlib | — | |
| `label2rgb(L)` | `skimage.color.label2rgb(L, bg_label=0)` | A | colours differ |
| `montage` | grid with matplotlib | — | |
| `imcrop(I, rect)` | slicing (`rect` = [x y w h], 1-based) | E | |
| `padarray(A, [p q], 'replicate'/'symmetric'/0)` | `np.pad(A, ((p,p),(q,q)), mode='edge'/'symmetric'/'constant')` | E | |
| `VideoReader`/`read` | `imageio.v3.imiter`, `cv2.VideoCapture` | E | |

## Histogram / intensity
| MATLAB | Python | P | Notes |
|---|---|---|---|
| `imhist(I)` (uint8) | `np.bincount(I.ravel(), minlength=256)` | E | |
| `imhist(I, n)` | `np.histogram(I, bins=n, range=(0,255))` | N | bin edges differ slightly |
| `histeq(I)` | `skimage.exposure.equalize_hist` / re-implement 64-bin | A/R | |
| `imadjust(I,[lo hi],[0 1],gamma)` | `skimage.exposure.rescale_intensity` + `adjust_gamma` | N | |
| `stretchlim` | `np.percentile(I, [1, 99])` | N | |
| `graythresh(I)` | `skimage.filters.threshold_otsu(I, nbins=256)/255` | N | |
| `im2bw(I, t)` | `I > t*255` | E | |
| `imbinarize(I,'adaptive','Sensitivity',s)` | `skimage.filters.threshold_local` | A | different local statistic; consider re-implementing MATLAB's (local mean × (1−s) style) |
| `adaptthresh` | same as above | A | |
| `multithresh` | `skimage.filters.threshold_multiotsu` | N | |
| `kmeans(X,k)` | `sklearn.cluster.KMeans(k, n_init=10, random_state=0)` | A | random init; compare centres |

## Neighbourhoods, connectivity, distance (Ch2 §2.3–2.4, Ch5)
| MATLAB | Python | P | Notes |
|---|---|---|---|
| `bwlabel(BW, conn)` | `skimage.measure.label(BW, connectivity=2 if conn==8 else 1)` | E (partition) | |
| `bwconncomp` | `label` + `regionprops` | E | |
| `bwselect`, `bwareaopen(BW, p)` | `remove_small_objects(BW, p)` | E | 8-conn default in both |
| `bwareafilt` | filter by `regionprops` area | E | |
| `bwdist(BW, 'euclidean')` | `distance_transform_edt(~BW)` | E | |
| `bwdist(BW, 'cityblock')` | `distance_transform_cdt(~BW, metric='taxicab')` | E | |
| `bwdist(BW, 'chessboard')` | `distance_transform_cdt(~BW, metric='chessboard')` | E | |
| `bwdist(BW, 'quasi-euclidean')` | `core.distance.quasi_euclidean` | R | two-pass chamfer (1, √2) |
| `[D, IDX] = bwdist(...)` | `distance_transform_edt(~BW, return_indices=True)` | E | IDX is linear column-major in MATLAB |
| `bwperim(BW)` | `BW & ~binary_erosion(BW, footprint=cross)` | E | MATLAB uses 4-conn for perim |
| `bwboundaries(BW)` | `core.chaincode.trace_boundaries` (Moore) | R | |
| `bwtraceboundary(BW, P, dir)` | `core.chaincode.trace_boundary` | R | |
| `regionprops(L, props)` | `skimage.measure.regionprops(L)` | N | see property renames in SKILL.md |

## Filtering, gradients (Ch2 §2.5, Ch4 §4.1, Ch6)
| MATLAB | Python | P | Notes |
|---|---|---|---|
| `conv2(A, K, 'same')` | `scipy.signal.convolve2d(A, K, mode='same', boundary='fill')` | E | |
| `conv2(A, K, 'valid'/'full')` | same with mode | E | |
| `imfilter(I, h)` | `scipy.ndimage.correlate(I.astype(float), h, mode='constant')` | E | uint8 output: clip + round like MATLAB |
| `imfilter(I, h, 'replicate')` | `mode='nearest'` | E | |
| `imfilter(I, h, 'symmetric')` | `mode='reflect'` | E | |
| `imfilter(I, h, 'conv')` | `scipy.ndimage.convolve` | E | |
| `fspecial(...)` | `core.filters.fspecial` | R | implement all kernels used in the book |
| `imgaussfilt(I, s)` | `scipy.ndimage.gaussian_filter(I, s, truncate=2)` ⚠ | N | MATLAB kernel size 2·ceil(2s)+1 → `truncate=2.0`; padding 'replicate' → `mode='nearest'` |
| `medfilt2(I, [m n])` | `scipy.ndimage.median_filter(I, size=(m,n), mode='constant')` | E | |
| `ordfilt2(I, k, dom)` | `scipy.ndimage.rank_filter(I, k-1, footprint=dom)` | E | |
| `stdfilt`, `entropyfilt`, `rangefilt` | `generic_filter(np.std)`, `skimage.filters.rank.entropy`, `maximum-minimum` | N/A | |
| `[gx, gy] = gradient(F)` | `gy, gx = np.gradient(F)` | E | |
| `imgradient(I)` (Sobel default) | `np.hypot(sobel_h, sobel_v)` with MATLAB Sobel kernel | E | skimage `sobel` normalises by 4 → scale |
| `edge(I, 'sobel'/'prewitt'/'roberts')` | `core.edges.edge_matlab(I, method)` | R | threshold = 4·mean(mag²)^½-style + non-max thinning; read MATLAB doc & Ch4 |
| `edge(I, 'log', t, sigma)` | `core.edges.log_zero_crossings` | R | zero-crossing with threshold |
| `edge(I, 'canny')` | `skimage.feature.canny(I, sigma=sqrt(2))` | A | |
| `del2(F)` | `core.filters.del2` | R | |
| `hough`, `houghpeaks`, `houghlines` | `skimage.transform.hough_line(_peaks)`, `probabilistic_hough_line` | A | |

## Morphology (Ch4 §4.2, Ch5, Ch7)
| MATLAB | Python | P | Notes |
|---|---|---|---|
| `strel('disk', r)` (n=4 default) | `skimage.morphology.disk(r)` | N | shape differs (see SKILL.md) |
| `strel('disk', r, 0)` | `disk(r)` | E | |
| `strel('square', n)` / `'rectangle'` / `'line'` / `'diamond'` / `'octagon'` | `square`, `rectangle`, `line` (implement), `diamond`, `octagon` | E/N | |
| `imerode(I, se)` / `imdilate` | `skimage.morphology.erosion/dilation` (grayscale) or `binary_erosion/binary_dilation` | E | MATLAB reflects SE for dilation; irrelevant for symmetric SEs |
| `imopen`/`imclose` | `opening`/`closing` | E | |
| `imtophat`/`imbothat` | `white_tophat`/`black_tophat` | E | |
| `imreconstruct(marker, mask)` | `reconstruction(marker, mask, method='dilation')` | E | |
| `imfill(BW, 'holes')` | `scipy.ndimage.binary_fill_holes(BW)` | E | |
| `imfill(I)` grayscale | `core.morphology.fill_holes_gray` (reconstruction by erosion) | R | |
| `imclearborder` | `skimage.segmentation.clear_border` | E | |
| `imregionalmax/min` | `local_maxima/local_minima` (connectivity=2) | E | |
| `imhmax/imhmin(I, h)` | `h_maxima/h_minima(I, h)` | E | |
| `imextendedmax/min` | `local_maxima(h_maxima(...))` | E | |
| `imimposemin(I, markers)` | `core.morphology.impose_minima` | R | |
| `watershed(I)` | `skimage.segmentation.watershed(I, connectivity=2, watershed_line=True)` | N/A | |
| `watershed(I, conn)` | `connectivity=1` for 4 | N | |
| `bwmorph(BW, 'thin', Inf)` | `skimage.morphology.thin(BW)` | N | |
| `bwmorph(BW, 'skel', Inf)` | `skeletonize(BW)` | A | different algorithm |
| `bwmorph(BW, 'spur'/'clean'/'bridge'/'remove'/'majority'/'fill')` | `core.morphology.bwmorph` (3×3 LUT) | R | |
| `bwulterode` | `local_maxima(distance_transform_edt(BW))` | N | |

## Geometry, interpolation, calibration (Ch2 §2.8, Ch9, App. A)
| MATLAB | Python | P | Notes |
|---|---|---|---|
| `imresize(I, s, 'nearest'/'bilinear'/'bicubic')` | `core.interp.imresize_matlab` | R | implement from §2.8 equations; antialias when shrinking |
| `interp2(X,Y,V,Xq,Yq,'linear')` | `scipy.ndimage.map_coordinates(V, [rows, cols], order=1)` | E | |
| `imrotate`, `imwarp`, `fitgeotrans`, `projective2d` | `skimage.transform.rotate/warp/ProjectiveTransform/estimate` | N | App. A orthorectification (DLT) → implement DLT from book Eqs |
| `polyfit`/`polyval` | `np.polyfit`/`np.polyval` | E | |
| `fminsearch` | `scipy.optimize.minimize(method='Nelder-Mead')` | N | |
| `lsqnonlin`/`lsqcurvefit` | `scipy.optimize.least_squares` | N | |
| `poly2mask(x, y, m, n)` | `skimage.draw.polygon2mask((m,n), np.c_[y,x])` | N | |
| `roipoly` | interactive → replace with stored polygon coords | — | |
| `regionprops(... 'Orientation','MajorAxisLength',...)` | see above | N | |
| `graycomatrix(I, 'Offset', [0 1], 'NumLevels', 8)` | `graycomatrix(I_quantised, [1], [0], levels=8, symmetric=False)` | E | quantise the same way |
| `graycoprops` | `skimage.feature.graycoprops` | E | |
