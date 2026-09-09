# Verified MATLAB → Python function map (project-specific; overrides the skill's reference table when they disagree)

Exactly **one** canonical Python target per MATLAB call. Parity labels: `exact` / `near` / `approx` / `reimplemented` /
`unverified` (see seaice-book skill). "Verified in" = chapter whose `reports/chNN_verification.md` holds the evidence.
Never remove a verified row; if a later chapter finds a better mapping, edit the row and note "superseded chNN".

| MATLAB | Python we use | Parity | Verified in | Note |
|---|---|---|---|---|
| `imread('x.jpg')` (JPEG) | `seaice.core.io.load_image(chapter, name)[0]` | exact | ch02 | imageio/Pillow decode identical to MATLAB on `rgb.JPG` (0 / 9 437 184 samples); resolves case-insensitively (`rgb.jpg` → `rgb.JPG`) |
| `rgb2gray(I)` | `seaice.core.matlab_compat.rgb2gray_matlab(I)` | exact | ch02 | NTSC 0.298936/0.587043/0.114021 in double, round half away from zero; 0 of 3 145 728 px differ; double input also exact. Not `skimage.color.rgb2gray` (Rec. 709) |
| `imcomplement(I)` | `seaice.core.matlab_compat.imcomplement(I)` | exact | ch02 | **canonical** (review Should-fix 7); uint8/uint16/int8/logical/double checked. `setops.gray_complement` / `bitwise_not` are equation-named forms — do not use them for ports. `imcomplement(double 0–255)` = `1 − I` |
| `uint8(x)` (cast of doubles in 0–255 units) | `seaice.core.matlab_compat.to_uint8_saturating(x)` | exact | ch02 | **canonical** for the cast: MATLAB round (half away from zero) + saturate; never `astype(np.uint8)` (wraps/truncates) |
| `im2uint8(I)` | `seaice.core.matlab_compat.im2uint8(I)` | exact | ch02 | float [0,1] → `round(255x)` clipped; logical → 0/255; uint8 unchanged. Distinct from `uint8()` above |
| `im2double(I)` | `seaice.core.matlab_compat.im2double(I)` | exact | ch02 | uint8 /255, uint16 /65535, logical 0/1 |
| `double(I)` (no scaling) | `I.astype(np.float64)` | exact | ch02 | `color_image.m` idiom; keep 0–255 units |
| `round(x)` | `seaice.core.matlab_compat.matlab_round(x)` | exact | ch02 | half away from zero (numpy is half-to-even) |
| `imhist(I)`, `[c, x] = imhist(I, n)` | `seaice.core.histogram.imhist(I, nbins=None)` | exact | ch02 | logical → 2 bins at `[0, 1]`; uint8 → 256; uint16/double/n-bin variants identical incl. centres |
| manual `num(k+1)=#(I==k); GP=num/(m*n)` (Eqs. 2.7–2.8) | `seaice.core.histogram.normalized_histogram(I)` | exact | ch02 | `histogram.m` lines 25–28 |
| `bwdist(BW, 'euclidean'/'cityblock'/'chessboard')` | `seaice.core.distance.bwdist(BW, metric)` | exact | ch02 | distance to nearest **nonzero**; MATLAB returns single → compare `atol=1e-4`; all-zero input → Inf both sides |
| `bwdist(BW, 'quasi-euclidean')` | `seaice.core.distance.bwdist(BW, 'quasi-euclidean')` (→ `quasi_euclidean_dt`) | near | ch02 | port float64 chamfer exact to 7e-13 vs analytic; MATLAB single accumulates 1e-4 (rel 8e-7); compare `rtol=1e-5` |
| Book Eq. (2.9) DT (object → nearest background) = `bwdist(~f)` | `seaice.core.distance.distance_transform(f, metric)` | exact | ch02 | use for book-equation code; use `bwdist` for MATLAB scripts |
| `bwlabel(BW, conn)` | `seaice.core.connectivity.label_components(BW, conn=8)` | exact | ch02 | label **numbers** reproduced (column-major first occurrence); `conn` 4 or 8; int32 output |
| `max(L(:))` component count | `seaice.core.connectivity.count_components(BW, conn)` | exact | ch02 | Fig 2.11: 5 / 2 |
| `bwperim(BW)` (not called in ch2; §2.3.5 boundary definition) | `seaice.core.connectivity.region_boundary_mask(BW, conn=8)` | reimplemented | ch02 | text-only; image-border pixels count as boundary. **Unverified vs `bwperim`** (whose default is 4-conn) — verify before relabelling when a chapter calls it |
| `conv2(f, w, 'same'/'full'/'valid')` | `seaice.core.filters.conv2(f, w, mode)` | exact | ch02 | true convolution, zero padding; even-kernel centring reproduced (≤ 3.6e-15) |
| `imfilter(f, w)`, `imfilter(f, w, 'replicate'/'symmetric'/'circular'/X, 'conv'/'corr', 'same'/'full')` | `seaice.core.filters.imfilter(f, w, *options)` | exact | ch02 | correlation by default; MATLAB positional string options accepted in any order (or keywords `mode=`, `padding=`, `output=`); 3-channel input ok; ≤ 5.3e-15 |
| `boundaries(BW, conn, dir)` (DIPUM, `ch2/chain code/`, copies in `ch5/`) | `seaice.core.chaincode.boundaries(BW, conn=8, direction='cw')` | exact | ch02 | list index = `bwlabel` label − 1; arrays `(Q, 2)` 0-based (row, col), closed; exterior only; `'ccw'` reverses; `'cww'` typo in `chain_diff.m` = `'cw'` |
| `fchcode(b, conn, dir)` (DIPUM) | `seaice.core.chaincode.fchcode(b, conn=8, direction='same')` → `ChainCode(fcc, diff, mm, diffmm, x0y0, conn, x0y0_matlab)` | exact | ch02 | `x0y0` 0-based, `x0y0_matlab` 1-based; `'reverse'` = `code_reverse`; conn validated first |
| `fchcode` local `minmag` on periodic / length-1 codes | `seaice.core.chaincode.min_magnitude(code)` | reimplemented | ch02 | MATLAB **errors** (`Output argument "z" not assigned`); port returns the lexicographic minimum rotation (= book Fig 2.21 last line). ch5 `try/catch` caveat |
| `fchcode` local `codediff` (Eq. 2.31) | `seaice.core.chaincode.first_difference(code, conn)` | exact | ch02 | `(C(i+1) − C(i)) mod conn`, circular |
| `bound2im(b)`, `bound2im(b, M, N)`, `bound2im(b, M, N, x0, y0)` (DIPUM) | `seaice.core.chaincode.bound2im(b, M=None, N=None, x0=None, y0=None)` | exact | ch02 | `x0, y0` 0-based (MATLAB `2,3` ↔ Python `1,2`); `b` may be `(Q,2)` or `(2,Q)` |
| `cellfun('length', b)` on boundary cells | `[len(x) for x in b]` | exact | ch02 | `chain_diff.m` |
| `interp2(X, Y, V, Xq, Yq, 'nearest'/'linear'/'cubic')` (unit grid) | `seaice.core.interp.interp2(V, u=Yq−1, v=Xq−1, method)` | exact | ch02 | 0-based (row, col); nearest tie = half away from zero; cubic = Keys `a=−0.5` + quadratic edge extrapolation (`pad='quadratic'`, default) exact incl. borders (6e-13); outside → NaN |
| `imresize(I, s, 'nearest'/'bilinear', 'Antialiasing', false)` (enlarge) | `seaice.core.interp.resize(I, s, method)` | exact | ch02 | 0.0 on full arrays for ×2, ×4 and ×0.5 (antialias off); output size `ceil(s·M)`, pixel-centre convention |
| `imresize(I, s, 'bicubic')`, any shrink with antialiasing | `seaice.core.interp.resize(I, s, 'bicubic')` | approx | ch02 | border differs (≤ 8.2 levels, 5.6 % px within 1.25 source px of the edge; interior 0.0); no antialiasing. Implement `imresize_matlab` if a chapter needs it numerically (ch9?) |
| `ind2rgb(idx, cmap)` | `seaice.core.color.indexed_to_rgb(idx, cmap, one_based=False)` | exact | ch02 | integer classes 0-based, double 1-based (`one_based=True`), out-of-range clipped like MATLAB; non-integer doubles truncated where MATLAB raises |
| `imcomplement(I)` used as CMY (Eq. 2.3) | `seaice.core.color.rgb2cmy(I)` | exact | ch02 | thin wrapper; same result as `imcomplement` |
| HSI block of `color_image.m` (Eqs. 2.6a–c) | `seaice.core.color.rgb2hsi(I, use_atan2=False, matlab_bug=False, scale=None)` | reimplemented (exact vs corrected MATLAB; `matlab_bug=True, scale=255` exact vs script) | ch02 | default = book (`2*Ib`); script bug `2*Ig` reproduced on request; `atan` undefined at `V1 = 0` (1 px of `rgb.JPG` on [0,1] input) |
| CMYK Eqs. (2.4)–(2.5) (no MATLAB builtin; `rgb2cmyk` is ICC-based) | `seaice.core.color.rgb2cmyk(I, u=1.0, b=1.0)` | reimplemented | ch02 | 0.0 vs hand-coded MATLAB Eq. 2.4 |
| `I(:,:,1)`, `I(:,:,2)`, `I(:,:,3)` | `seaice.core.color.split_rgb(I)` | exact | ch02 | dtype preserved |
| `zeros(201,201); img(101,101)=1` (Fig 2.14a) | `seaice.core.synth.point_image(201)` | exact | ch02 | 0-based centre (100,100) |
| `imshow(I)` / `imshow(I, [])` / `mat2gray` | `seaice.core.plotting.imshow_matlab(ax, I, autoscale=False/True)`, `imshow_scale`, `save_image` | display (exact scaling; PNG ±1 level from MATLAB float32 `mat2gray`) | ch02 | numerics never pass through here |
| `saveas(gcf, 'x.png')`, `figure`, `subplot` | `seaice.core.plotting.finish_figure(fig, path, show)` + matplotlib | — | ch02 | scripts write to `outputs/chNN/` |
| `padarray(A, [1 1], 0)` (inside `boundaries.m`) | `np.pad(A, 1, mode='constant')` | exact | ch02 | internal to `chaincode.boundaries` |
| `sub2ind` / `ind2sub` (column-major) | `np.ravel_multi_index(..., order='F')` / `np.unravel_index(..., order='F')` | exact | ch02 | used inside `bound2im`, `boundaries` |
