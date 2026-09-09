# ch02 review

Independent fresh-context review (port-reviewer agent, 2026-09-09). Reviewer reran `tests/test_ch02.py` -> **83 passed in 74.69 s**. All 8 `reference/ch02/*.mat` present. Every fixture matrix in `seaice/core/synth.py` (Figs 2.10a/bc, 2.11, 2.12a/b, 2.19a, 2.20 coordinates, 2.21 sequences) was re-checked value-by-value against `chapters/ch02.txt`; all correct.

## Verdict: PASS-WITH-FIXES

No numerical defect was found in any reusable primitive. The must-fix items are traceability/labelling errors (a docstring and a figure that claim to implement an equation they do not), plus a wrong number in the report.

## Must-fix

1. **`seaice/core/filters.py:101-119` (`conv_at`) and `seaice/ch02_preliminaries.py:193-214` (`convolution_example`), `scripts/ch02_convolution.py:43,48-53` — Eq. (2.15) is not implemented as printed, but the code, figure title and console output claim it is.**
   Book evidence (`chapters/ch02.txt:1510-1514`): Eq. (2.15) reads `h(x,y) = ω(−1,−1) f(x−1,y−1) + ω(−1,0) f(x−1,y) + … + ω(1,1) f(x+1,y+1)`, i.e. `Σ ω(s,t) f(x+s, y+t)` — *correlation* form, which contradicts the book's own Eq. (2.14) (`f(x−s, y−t)`, line 1493). `conv_at` computes `w[s+hm, t+hn] * f[x-s, y-t]` (Eq. 2.14 form) while its docstring says "written out as Book Eq. (2.15)". On the demo kernel (antisymmetric Sobel-type) the two differ in sign: `conv_at` = **+1.0**, Eq. (2.15) as printed = **−1.0** (= `imfilter` correlation). So `fig_2_15_convolution.png` ("Eq. (2.15) at (3,4) gives h = 1") and the printed term list are wrong relative to the printed equation.
   Fix: (a) in `conv_at` docstring and `convolution_example` say explicitly: *implements Eq. (2.14); note the book's Eq. (2.15) as printed uses f(x+s, y+t) (correlation, = MATLAB `imfilter`), inconsistent with Eq. (2.14); we follow (2.14)/`conv2`*; (b) either add `correlate: bool = False` to `conv_at` and show both values in the figure, or retitle the figure/output "Eq. (2.14)". Add the inconsistency to `analysis/ch02.md` §"book text issues" and to `reports/ch02_verification.md` "Deviations". `conv2`/`imfilter` themselves are correct and verified vs MATLAB (`h3…h23`, `g3…g4r` at 1e-12) — untouched.

2. **`reports/ch02_verification.md:54` — wrong number in the `imresize` row.** It states bicubic ×4 differs on "976 of 16 384 px > 1e-9". Measured against `reference/ch02/interp2.mat`: **2 896 px > 1e-9** (max 8.21), 742 px > 0.5, **921 px differ after uint8 rounding (5.62 %)** — the "5.6 %" in the Figures table (line 78) is the uint8 count, so 976 matches nothing. Replace with the measured values and say which threshold each is. Also "all within 2 source pixels of the border" is correct (max distance of a differing output pixel from the border = 5 output px = 1.25 source px).

## Should-fix

1. **`tests/test_ch02.py:759-771` — bilinear/nearest `resize` parity is asserted only on the interior (`[4:-4]`, `[2:-2]`, `[1:-1]`) while the report (line 54) claims 0.0 on the full image.** Measured full-image error 0 for `R4l`, `R2l`, `Rhl`, `R4n`, `Rhn`, so the claim is true but untested. Change those asserts to the full arrays (`atol=0`) so the `exact` label for nearest/bilinear enlargement has an L2 test behind it; keep the interior crop only for bicubic.

2. **`reports/ch02_verification.md:56` + Open items — `indexed_to_rgb` is labelled "reimplemented (**unverified** vs `ind2rgb`)" but is not listed in Open items.** Cheapest closure: add `ind2rgb(uint8 idx, cmap)` and `ind2rgb(double idx, cmap)` (incl. an out-of-range index) to `ref_compat()` in `reference/ch02/make_refs.py:266` and assert equality — MATLAB clips to `[1, m]` and treats integer classes as 0-based, exactly what `color.py:117-130` does — then relabel `exact`. Otherwise add the Open item.

3. **`tests/test_ch02.py:192-198` (`test_region_boundary_mask`) cannot detect a conn-4/8 swap** — for a solid square both give the same 16-pixel ring. Add a notched case: `bw = ones((5,5)); bw[0,0] = False` -> `region_boundary_mask(bw,4).sum() == 15`, `region_boundary_mask(bw,8).sum() == 16`, and `(1,1)` is boundary only under 8-conn.

4. **`seaice/core/filters.py:41-98` — `imfilter` 'full' output is verified only with zero padding** (`g3full/g5full/g4full`; 'replicate'/'symmetric'/'circular' + 'full' are not in `conv2.mat`) yet the row is labelled `exact` without qualification. Either add `imfilter(f,w3,'replicate','full')` and `'symmetric','full'` to `ref_conv2()` or note the untested combination in the report.

5. **`seaice/core/filters.py:41` — positional-argument trap for later chapters.** MATLAB code in ch4–ch6 is written `imfilter(I, h, 'replicate')`; the Python signature's third positional is `mode` ('corr'/'conv'), so a mechanical port raises `ValueError`. Accepting MATLAB-style `*options` (any of 'replicate'/'symmetric'/'circular'/number/'conv'/'corr'/'same'/'full' in any order) would remove a recurring porting hazard. Record in `knowledge/function_map.md` either way.

6. **Figure numbering of interpolation outputs.** `scripts/ch02_interpolation.py:36-38` writes `fig_2_22_interpolation_comparison.png` and `fig_2_23_resize_comparison.png`; in the book Fig 2.22 is the coordinate-transform diagram, Fig 2.23 the nearest-neighbour diagram and Fig 2.24 the bilinear diagram — a three-method resize panel is none of them. Rename to `sec_2_8_interp2_grid.png` / `sec_2_8_resize_x4.png` (book figure numbers only for reproductions), and fix the report row. Similarly `fig_2_14_point_dt_quasi_euclidean.png` (`scripts/ch02_distance_transform.py:36`) carries the Fig 2.14 number although quasi-euclidean is not in Fig 2.14(a–d) — drop the fig prefix for that file.

7. **Duplicated primitives (minor):** `setops.gray_complement` ≡ `matlab_compat.imcomplement`, `setops.bitwise_not(nbits=8)` ≡ both, `matlab_compat.to_uint8_saturating` overlaps `im2uint8`. Acceptable, but `knowledge/function_map.md` should name **one** canonical function per MATLAB call (`imcomplement`, `uint8()`) so ch4+ do not pick different ones.

8. **`seaice/core/chaincode.py:311-312`** — `fchcode(b, conn=5)` computes everything and only then raises. Move the `conn not in (4, 8)` check to the top of `fchcode`. Cosmetic.

## Confirmed correct

- **Eq. (2.3) CMY** (`color.py:30-38` = `imcomplement`): uint8 `255−I`, float `1−I`; script's `imcomplement(double 0–255)` giving negative values reproduced; L2 0 diff.
- **Eqs. (2.4)–(2.5) CMYK** (`color.py:41-59`): `[C;M;Y;K] = [1;1;1;0] − [R;G;B;0] − Kb[u;u;u;−b]`, `K = b·Kb`; matches `ch02.txt:420-454`.
- **Eq. (2.6a–c) HSI** (`color.py:97-113`): matrix rows `[1/3,1/3,1/3]`, `[−1/√6,−1/√6,2/√6]`, `[1/√6,−1/√6,0]`; `H = atan(V2/V1)` (default, as printed), `atan2` opt-in; `matlab_bug=True` reproduces `color_image.m:21` (`2*Ig`) — L2 0 diff on `Ii/Is/Ih`. The single-pixel `±π/2` difference at `R+G == 2B` with normalised input is a property of Eq. (2.6b) at V1 = 0.
- **Eqs. (2.7)–(2.8)** (`histogram.py`): `imhist` bin rule `round(v·(n−1)/top)` reproduced (hand-checked 3-bin case), logical default 2 bins, float clipping to [0,1]; `GP = num/(m·n)`; L2 identical for 64/100/5/256-bin, uint16, logical.
- **`rgb2gray`** (`matlab_compat.py:70-98`): NTSC coefficients, double then half-away-from-zero rounding; 0 of 3 145 728 px differ; Rec.709 would give 54/182/18 instead of 76/150/29 on pure colours (test discriminates).
- **Eq. (2.9)–(2.12)** (`distance.py`): `distance_transform(f)` = `edt(obj)`; `bwdist(BW)` = `distance_transform(~BW)`; tests would catch a flip. All-zero → `Inf`, all-one → 0. Quasi-euclidean chamfer exact to 1e-13 vs `max + (√2−1)·min`; MATLAB (single) is the less precise side.
- **§2.3 connectivity** (`connectivity.py`): N4/ND/N8 orderings match `ch02.txt:717,722`; m-adjacency implements (i)/(ii) of `ch02.txt:759-762`; `find_paths` on Fig 2.10 gives 0 four-paths, >1 eight-paths, exactly 1 m-path. `label_components` numbering = column-major first occurrence (L2 identical incl. numbers on Fig 2.11 and a 6-object real mask).
- **Eq. (2.14) / `conv2`, `imfilter`** (`filters.py`): true convolution via `convolve2d`; even-kernel 'same' crop at `ceil((m−1)/2)` and `imfilter` centre `(n−1)//2` verified vs MATLAB and by hand against MATLAB's `floor((size+1)/2)` rule; 'full' + zero padding equivalent to MATLAB's single `padarray(size(h)−1)`.
- **Eqs. (2.16)–(2.30), Table 2.1, `57 AND 207 = 9`** (`setops.py`): correct; `reflect` maps the origin to `(M−1−r0, N−1−c0)`; `translate` drops out-of-canvas elements.
- **Eq. (2.31)** (`chaincode.first_difference`): `(C(i+1)−C(i)) mod conn`, wrap on the last element; matches `codediff` and Fig 2.21 line 2.
- **`boundaries.m`** (`chaincode.py:91-195`): offsets `[−1, M−1, M, M+1, 1, −M+1, −M, −M−1]`, both LUTs, `START/BOUNDARY` marking, column-major candidate scan, `rr+1`, Jacob's stopping rule, single-pixel → two identical points, `ind2sub`, `'ccw'` = reverse — all replicated, none "fixed". Hand-traced a 6-pixel asymmetric shape: identical. L2: Fig 2.19 (19 pts), Fig 2.11 (2 + 5 objects, cw/ccw), spur, hollow square (exterior only), 200×300 real mask (1050/1096 points) identical.
- **`fchcode.m`**: code table `z = 4(dx+2)+(dy+2)` incl. `z = 10 → 0` for a zero step; `coderev` flip ±4; 4-conn halving with warning; `minmag` = lexicographic minimum over rotations (brute-force test n = 1…18) with the documented tie-break where MATLAB errors; `x0y0` 0-based with `x0y0_matlab` +1. Fig 2.20/2.21 five sequences and start (4,2) reproduced; direction numbering 0 = E, 1 = NE … matches `fchcode.m` and Fig 2.20.
- **`bound2im.m`**: 1-based `x − min(x) + 1` ↔ 0-based `x − min(x)`; nargin 1/3/5 branches; `NR = round((M−C)/2)` with `matlab_round`; 5-arg check `C + x0 > M` ≡ MATLAB `C + x0_m − 1 > M`; L2 identical for all four calling forms.
- **`chain_diff.m`**: `'cww'` → clockwise, `b = b{1}` preserved, `bound2im(b,M,N,min,min)` with 0-based mins.
- **Eqs. (2.33)–(2.41)** (`interp.py`): nearest tie = half away from zero (L2 incl. `.5` ties and NaN outside); bilinear Eqs. (2.35)–(2.37) literally; bicubic uses lattice `f(i+m, j+n)` (book's `f(u+m, v+n)` typo noted); Keys kernel Eq. (2.41) with `a = −0.5`, MATLAB quadratic edge extrapolation → cubic exact incl. borders (6e-13). `interp_nearest` no longer mutates a float64 input on scalar queries.
- **Coordinate convention**: Eq. (2.1) has `f(M,y)` in the last row → `x` = row, `y` = col, as every docstring states; `interp2` documents the `u = Yq−1, v = Xq−1` swap.
- **Scripts**: all nine run headless; `ch02_chain_diff.py` exits non-zero if any book check fails; Fig 2.7 substitute is labelled in the title, docstring and report; pixel `(1076,675)` plotted at `(674,1075)`.
- **Report honesty**: every `exact` row has an L2 test with `atol ≤ 1e-12` (or 0); `near`/`approx`/`reimplemented` used per vocabulary; MATLAB engine/version recorded; all seven `.m` files have rows.

## Notes for later chapters

- `imfilter(f, w, ...)` Python positional order is `(f, w, mode, padding, output)` — MATLAB's `imfilter(I, h, 'replicate')` must become `imfilter(I, h, padding="replicate")` unless Should-fix 5 is applied. `conv2` = true convolution; `imfilter` default = correlation.
- `bwdist(BW)` (MATLAB semantics) vs `distance_transform(f)` (Eq. 2.9 semantics) are complements of each other; ch5 `distance_watershed.m` / ch6 `GVF_distance.m` call MATLAB `bwdist`, so port them with `core.distance.bwdist`, not `distance_transform`.
- `boundaries()` traces **exterior** boundaries only and returns closed 0-based `(row, col)` lists in `bwlabel` order; `fchcode()` never raises on periodic/single-pixel codes where MATLAB's `minmag` errors — any ch5 `try/catch` around `fchcode` must be ported knowingly.
- `label_components` reproduces MATLAB label numbers, so ch5/ch7 code that indexes `regionprops` by label can rely on numbering, not just partition.
- `interp2(img, u, v)` takes 0-based `(row, col)`; ch6 `snakedeform.m` (`interp2(x, y, px, ...)`) and App. A warps must swap `(Xq, Yq) → (v, u)` and subtract 1. `resize` is `approx` vs `imresize` (bicubic border, no antialiasing) — do not use it where ch9 relies on `imresize` numerically.
- Book Eq. (2.15) is in correlation form and inconsistent with Eq. (2.14); when ch4 cites "convolution" for Sobel/Prewitt, check which form the MATLAB code actually uses (`imfilter` = correlation) before labelling.

## Disposition (orchestrator)
- Must-fix 1, Should-fix 5, 6, 8 → matlab-porter (code). Must-fix 2, Should-fix 1–4 → port-verifier (tests/references/report). Should-fix 7 → knowledge-keeper (canonical names in function_map.md).
