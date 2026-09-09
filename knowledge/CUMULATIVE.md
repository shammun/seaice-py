# CUMULATIVE knowledge — seaice-py
(Rewritten by the knowledge-keeper after every chapter. Read this first at the start of any chapter. Last rewrite:
after **ch02**, 2026-09-09. Per-chapter detail: `knowledge/chNN.md`; verified call mappings: `knowledge/function_map.md`.)

## Pipeline so far
```
Ch2 primitives (DONE)                       Ch3 ice mask   Ch4 edges   Ch5 watershed floes   Ch6 GVF boundaries
  io.load_image ──────────────────────►  rgb2gray_matlab → imhist → (graythresh/im2bw, kmeans: to write)
  matlab_compat.{rgb2gray_matlab, imcomplement, matlab_round, im2double, im2uint8, to_uint8_saturating}
  histogram.{imhist, normalized_histogram} ──► Ch3 Otsu/separability, Ch6/7 color_hist
  connectivity.label_components (= bwlabel) ─► Ch5–Ch9 (bwlabel everywhere)
  distance.{bwdist, distance_transform} ─────► Ch5 watershed markers, Ch6 GVF_distance/dist
  filters.{conv2, imfilter, conv_at} ────────► Ch4 Sobel/Prewitt/LoG (imfilter = correlation!), Ch5, Ch6 xconv2
  setops.{reflect, translate, gray_union/…} ─► Ch4 §4.2 morphology definitions, Ch7 mask combinations
  chaincode.{boundaries, fchcode, bound2im} ─► Ch5 concavity / neighbouring-region merging
  interp.{interp2, warp_image, resize} ──────► Ch6 snakedeform (interp2), App. A orthorectification (warp_image)
  color.{rgb2cmy, rgb2cmyk, rgb2hsi, indexed_to_rgb}, plotting.*, synth.*, cli.*
Ch7 ice types → Ch8 applications (concentration, floe size) → Ch9 model ice → Ch10 = App. A calibration
```
Conventions fixed in ch02 and binding for all later chapters:
- Coordinates: book `x` = row, `y` = col (Eq. 2.1). Python is 0-based `(row, col)` everywhere; tests add 1 to compare
  with MATLAB. `ChainCode.x0y0_matlab` is the 1-based pair.
- Numerics stay float64; uint8 conversion only at display/save time via `plotting.*` or `to_uint8_saturating` / `im2uint8`.
- Every `.m` maps to one Python script (`scripts/chNN_<name>.py`) or one `seaice` function; docstrings cite §/Eq/`.m`.
- References: MATLAB R2025a via `tools/run_matlab_ref.py`, driven by `reference/chNN/make_refs.py`; `.mat` files in
  `reference/chNN/`; tests skip with a clear message if a `.mat` is missing. Run `.m` scripts from a scratch cwd
  (`outputs/chNN/verify/scratch`) because some write PNGs (`saveas`).
- Figures: `outputs/chNN/fig_<ch>_<fig>_<slug>.png` only for genuine book figures; `sec_<ch>_<sec>_<slug>.png` otherwise.

## Available primitives in seaice/core/
| function | module | book § | used by chapters | parity |
|---|---|---|---|---|
| `load_image(chapter, name)` (case-insensitive `imread`; private copy first, public-domain NASA substitute as fallback), `repo_root`, `book_data_dir`, `output_dir`, `fetch` | `io` | — | all | exact (JPEG decode identical to MATLAB on `rgb.JPG`) |
| `chapter_argparser`, `resolve_dirs` | `cli` | — | all scripts | — |
| `rgb2gray_matlab(rgb)` | `matlab_compat` | §2.2 | ch3–ch10 | exact (0 px differ) |
| `imcomplement(img)` (canonical for MATLAB `imcomplement`) | `matlab_compat` | Eq. 2.3 / 2.21 | ch5 `topological_surface.m`, ch7 | exact |
| `matlab_round(x)` | `matlab_compat` | — | ch6, ch7, ch9 | exact |
| `im2double(img)`, `im2uint8(img)` | `matlab_compat` | §2.1.1 | ch7, ch9 | exact |
| `to_uint8_saturating(x)` (canonical for MATLAB `uint8(x)` on 0–255 values) | `matlab_compat` | §2.1.1 | ch3+ | exact |
| `imshow_scale`, `to_display_uint8`, `save_image`, `imshow_matlab`, `show_matrix`, `finish_figure` | `plotting` | `imshow(I)` / `imshow(I,[])` | all | display only |
| `split_rgb`, `rgb2cmy`, `rgb2cmyk(u, b)`, `rgb2hsi(use_atan2, matlab_bug, scale)`, `indexed_to_rgb(one_based)` | `color` | §2.1, Eqs. 2.2–2.6 | ch6/ch7 colour stats, ch8 colormaps | exact / reimplemented (HSI, CMYK) |
| `imhist(img, nbins=None)`, `normalized_histogram` | `histogram` | §2.2, Eqs. 2.7–2.8 | ch3 Otsu/separability, ch6/ch7 | exact (logical → 2 bins) |
| `n4/nd/n8`, `is_adjacent`, `is_m_adjacent`, `find_paths` | `connectivity` | §2.3.1–2.3.3 | teaching only | reimplemented (text) |
| `label_components(bw, conn=8)` (= `bwlabel` incl. numbering), `count_components` | `connectivity` | §2.3.4 | ch5–ch9 | exact |
| `region_boundary_mask(bw, conn)` | `connectivity` | §2.3.5 | (≈ `bwperim`, unverified vs it) | reimplemented |
| `distance_transform(bw, metric)` (Eq. 2.9 = `bwdist(~f)`) | `distance` | §2.4 | book-equation code | exact |
| `bwdist(bw, metric)` (MATLAB semantics) | `distance` | §2.4.4 | ch5 watershed, ch6 GVF | exact (eucl/city/chess), near (quasi) |
| `quasi_euclidean_dt`, `pixel_distance`, `center_distance_map` | `distance` | Eqs. 2.10–2.12 | ch5 (if `'quasi-euclidean'`) | reimplemented |
| `conv2(f, w, mode)` | `filters` | Eq. 2.14 | ch4, ch6 `xconv2` | exact |
| `imfilter(f, w, *matlab_options)` (correlation default; positional `'replicate'` etc. accepted) | `filters` | §2.5 | ch4, ch5, ch6 | exact |
| `conv_at(f, w, x, y, correlate)` | `filters` | Eqs. 2.14 / 2.15 | teaching | exact |
| `complement/union/intersection/difference`, `reflect`, `translate`, `gray_complement/union/intersection`, `bitwise_*`, `truth_tables` | `setops` | §2.6, Eqs. 2.16–2.30 | ch4 §4.2, ch7 | exact |
| `boundaries(bw, conn, direction)` (DIPUM, exterior only, closed, `bwlabel` order) | `chaincode` | §2.3.5, §2.7 | ch5 | exact |
| `fchcode(b, conn, direction) -> ChainCode`, `first_difference`, `min_magnitude`, `normalized_first_difference`, `code_reverse`, `chain_to_points` | `chaincode` | §2.7, Eq. 2.31 | ch5 | exact (tie-break where MATLAB errors: reimplemented) |
| `bound2im(b, M, N, x0, y0)` (0-based `x0, y0`) | `chaincode` | §2.7 | ch5 | exact |
| `interp_nearest/interp_bilinear/interp_bicubic`, `interp2(img, u, v, method)`, `keys_kernel` | `interp` | §2.8, Eqs. 2.33–2.41 | ch6 `snakedeform`, ch10 | exact (borders incl.) |
| `warp_image(img, T_inv, out_shape, method)` | `interp` | Eqs. 2.32–2.33 | ch10 App. A | exact by construction |
| `resize(img, scale, method)` | `interp` | §2.8 | (ch9 — see pitfall) | exact nearest/bilinear enlarge; approx bicubic/shrink |
| `point_image`, `spur_shape`, `set_operation_fixtures`, `book_fixtures`, `FIG_2_*` constants | `synth` | Figs 2.10–2.21 | tests ch2/ch5 | — |

Not yet in core (first needed by): `graythresh`/`im2bw`/`kmeans` wrappers (ch3), `fspecial`, `edge` re-implementation,
morphology (`strel`, `imerode/imdilate`, `bwmorph`) (ch4), `watershed`, `imimposemin`, `regionprops` compat (ch5),
GVF/snake (ch6), `imresize_matlab` with antialiasing (ch9 if needed), DLT / lens distortion (ch10).

## Global pitfalls (MATLAB → Python) confirmed in this project
1. Book `x` = row, `y` = col; MATLAB 1-based; Python 0-based `(row, col)`. `interp2(X, Y, Z, Xq, Yq)` has `X` = column:
   Python `interp2(img, u=Yq−1, v=Xq−1)`.
2. `bwdist(BW)` = distance to nearest **nonzero** = `distance_transform(~BW)`; returns single → `atol≈1e-4`.
3. `conv2` flips the kernel, `imfilter` does not (correlation). Book Eq. (2.15) is printed in correlation form
   (inconsistent with Eq. 2.14) — check every "convolution" claim in ch4 against the actual MATLAB call.
4. `imfilter(I, h, 'replicate')` positional strings: `core.filters.imfilter` accepts them as-is; never remap to `mode`.
5. `bwlabel` numbers components column-major by first occurrence; `label_components` reproduces it — compare label
   images directly, not only partitions.
6. DIPUM `boundaries.m`: exterior only, closed lists, single pixel → 2 identical points, spurs traversed twice; DIPUM
   `fchcode.m`/`minmag` **errors** on periodic codes and single pixels — the port breaks ties instead (ch5 caveat).
7. `rgb2gray`: NTSC 0.298936/0.587043/0.114021 in double + half-away-from-zero rounding (skimage Rec.709 is wrong).
8. `round` = half away from zero → `matlab_round`; `uint8(x)` = round + saturate on 0–255 values →
   `to_uint8_saturating`; `im2uint8` = ×255 first → `im2uint8`; numpy `astype(uint8)` wraps/truncates — never use it.
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
    Anaconda's binary); set `PYTHONIOENCODING=utf-8` when printing non-ASCII (cp1252 console).
17. Typos in the authors' code are reproduced, not corrected, when the printed figure shows the typo's effect
    (`chain_diff.m` `'cww'` → clockwise); book-equation behaviour is the default when the text is the authority
    (`rgb2hsi` default follows Eq. 2.6a, `matlab_bug=True` reproduces the script).

## Data inventory
| file | chapter | tier | shows |
|---|---|---|---|
| `data/book/ch02/rgb.JPG` (2048×1536×3, 723 730 B) | ch02 | 1 | sea-ice colour photo; Fig 2.3 pixel (1076,675) = [28,76,114]; gray peak 47 840 @ 208 |
| `seaice/core/synth.py` fixtures | ch02 | 3 | Fig 2.10/2.11/2.12/2.19/2.21 matrices & sequences, 201×201 point image, spur shape, 16×16 set rectangles |
| `reference/ch02/*.mat` (8) | ch02 | MATLAB refs | histogram, color_image, distance_transform, chain_diff, boundaries_multi, conv2, interp2, compat |
| Fig 2.7 grayscale floe-field image | ch02 | missing | not shipped; `rgb2gray(rgb.JPG)` used as labelled substitute |
| Book-shipped images ch3–ch10 (`data/book/chNN/`) | ch03+ | 1 | copied by `/setup-project`; see `analysis/_matlab_inventory.md` |
| `dypic_05100_cam1_top.avi` (ch9 movie scripts), raw JPEG behind ch8 `MCD/*.mat` | ch08/ch09 | missing | see `data/online/SOURCES.md` |

## Parity summary per chapter
| chapter | exact | near | approx | reimplemented | unverified | verdict |
|---|---|---|---|---|---|---|
| ch02 Preliminaries | 23 | 1 (`bwdist` quasi-euclidean) | 1 (`resize` bicubic border / shrink) | 4 (HSI, CMYK, §2.3 text-only, `min_magnitude` tie-break) | 1 (Fig 2.7 source image) | PASS — 86 tests, 7/7 `.m` exact vs MATLAB R2025a |

## Setup findings (2026-09-09, /setup-project)
- Reference engine: **MATLAB R2025a** (25.1.0.2833191, prerelease) via `tools/run_matlab_ref.py`; Octave absent (fine).
- PDF offset = 31 (pdf 0-based index = printed page − 1 + 31); each `chapters/chNN.txt` starts on its chapter title.
- `MATLAB_ROOT/ch10` = Appendix A (`fisheye_calibration.m` → A.2 lens distortion, `orthoretification.m` → A.1).
- `ch6/Sea_Ice_Floe_Identification` and `ch7/Sea_Ice_Floe_Identification` hold the same 24 `.m` files (byte-identical);
  only `sea_ice_test.jpg` differs. Port once (ch6), reuse in ch7. `ch9/Model_Ice_Floe_Identification` shares the GVF/snake
  files too. ch5 duplicates ch2's `bound2im.m` / `boundaries.m` / `fchcode.m` (byte-identical, md5 verified in ch02).
- Data gaps: ch9 movie scripts need `dypic_05100_cam1_top.avi` (not shipped); ch8 MCD ships only `.mat` results.
  See `data/online/SOURCES.md` (git-ignored, on disk).

## Publishing findings (2026-09-09, ch02 PUBLISH phase)
- Public repo `shammun/seaice-py` (branch `main`), GitHub Pages at `https://shammun.github.io/seaice-py/`. History was
  rewritten with git-filter-repo to purge every book-derived file; `chapters/*.txt`, `data/book/`, `reports/**/figures/`,
  `outputs/` stay local and git-ignored (CLAUDE.md rule 12).
- Data: `seaice.core.io.load_image()` (private copy → NASA public-domain substitute from `seaice/core/public_images.py`);
  ch02's substitute is a MODIS Terra Beaufort MIZ scene of 2019-07-25 at the book image's 2048×1536 size.
- Notebook cells 1–3 template lives in `notebooks/build_ch02.py`; `tools/publish_notebook.py chNN` builds the Colab
  variant, the styled HTML page (from a run with `data/book` renamed away), `index.html` and the README table.
