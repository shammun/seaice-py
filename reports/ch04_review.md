# Chapter 4 review — Ice Edge Detection (fresh-context, read-only)

Date 2026-09-09 · reviewed at verify commit `3ca6870` · scope: `chapters/ch04.txt` (Eqs. 4.1–4.42, Figs. 4.1–4.20),
`derivative.m`, `morphology.m`, `seaice/core/edges.py`, `seaice/core/morphology.py`, `seaice/core/filters.py::fspecial`,
`seaice/core/synth.py`, `seaice/ch04_ice_edge_detection.py`, `scripts/ch04_*.py`, `tests/test_ch04.py` (485 passed, 80 s),
`reference/ch04/make_refs.py`, `reports/ch04_verification.md`, plus the R2025a sources `imclose.m`, `imopen.m`,
`private/morphop_fast.m`, `edge.m` (routing claims checked independently).

## Must fix

1. **must-fix** — `seaice/core/morphology.py:264` (`_min_max_filter`, scipy fallback): `imerode` on **int64 / uint64**
   images is wrong at every border-reaching pixel for any non-full SE. `cval = np.iinfo(int64).max` passed to
   `ndimage.minimum_filter` is routed through float64 and cast back → overflows to `intmin`, so the erosion pad becomes the
   minimum instead of +∞ (Eq. 4.20 border convention). Reproduced: 9×11 random int64, `strel('disk',2)` → 64 px = intmin vs
   the literal Eq. 4.20 loop; asym SE 19 px; diamond 1 36 px; uint64 55 px. Nothing in the suite notices because every L2
   fixture is bool/uint8/double/int16 (OpenCV path). Correction: pad explicitly with numpy in the scipy branch and crop.
   Matters for ch5: label/marker images (`skimage.measure.label` default int64) are the first non-cv2 dtype the morphology meets.

## Should fix

2. **should-fix** — `analysis/ch04.md:22`, `:115–118` state the thinning rule with **replicate-padded** `b`; the verified code
   (`edges.py:116`, zero padding) and the L2 evidence back zero padding. Fix the analysis text.
3. **should-fix** — `reports/ch04_verification.md:74` labels `reconstruct_by_erosion` `exact`; the code marker is
   `# PARITY: reimplemented` and there is no MATLAB builtin. Relabel `reimplemented (verified via the complement identity,
   0 diff)`; counts become 21 exact / 6 reimplemented.
4. **should-fix** — `reports/ch04_verification.md:169–172` (Open item 4) promises bit-exact morphology on signed images for
   ch5–ch9; true only for the OpenCV dtypes. Add an Open item for the scipy-fallback dtypes (int32/uint32/int64/uint64).
5. **should-fix** — `edges.py:238–240` accepts `direction='horizontal'|'vertical'` for **Roberts** but
   `reference/ch04/make_refs.py:325` only generates `'both'` and `tests/test_ch04.py:896` excludes the other two. Add the
   references + tests or narrow the docstring's "all three directions" claim.
6. **should-fix** — `seaice/ch04_ice_edge_detection.py:247` `bwareaopen` is a general toolbox function reused by ch5/ch7;
   move it to `seaice/core/connectivity.py` (next to `label_components`) and re-export from the chapter module.
7. **should-fix** — `tests/test_ch04.py:1207–1219` `test_script_runs` only exercises the default CLI; none of `--crop fig4_3a`,
   `--method log|roberts|prewitt`, `--thresh -1`, `--median`, `--min-area`, `--smooth`, `--radius 15 --crop fig4_3a`,
   `--demo none`, `experiments --crop r0:r1,c0:c1` are run. Add one parametrised run per script on the crop.
8. **should-fix** (would have caught item 1) — parametrise
   `TestL1ErodeDilate.test_reflection_with_asymmetric_se_literal_eqs_4_20_4_21` over
   `dtype ∈ {uint8, int16, float64, int32, uint32, int64, uint64}` against the literal loops; add `imclose`/`imopen`
   cross-backend equality (`int16` vs `int32` on the same values).

## Nits

9. **nit** — `edges.py:205–207`: float32 input is silently promoted to float64 (MATLAB computes in single). Document it.
10. **nit** — `ch04_ice_edge_detection.py:171–239`: `laplacian`, `second_difference_forward`, `gaussian_kernel`,
    `log_kernel('analytic')` lack the inline `# PARITY: reimplemented` marker.
11. **nit** — `ch04_ice_edge_detection.py:335–344` `morphological_edges` recomputes 6 redundant 12-Mpx erosions/dilations
    inside `morphological_gradient`; pass the precomputed J/K/X/Y.
12. **nit** — `morphology.py:180–185` `strel('disk', r, n_pos, n=kw)`: positional N silently wins over keyword `n`; raise on conflict.
13. **nit** — `analysis/ch04.md:243` says the binary gradients return `uint8 0/1`; the code returns float64 0/1 (MATLAB double).

## Fine / verified

- Eqs. 4.1–4.8, Figs. 4.1–4.2: kernels transcribed with x = row as printed; `Gx_book = −8·gh`, `Gy_book = −8·gv` correct and
  L2-checked to 1e-14; Eq. 4.3 `arctan(Gx/Gy)` → `atan2(Gx, Gy)` with the `'math'` alternative and a PARITY note.
- Eqs. 4.9–4.13, Fig. 4.4: 4-/8-neighbour kernels as printed; Eq. 4.10 forward second difference centred correctly; p. 64
  zero-crossing rule read as "both signs present in the 3×3 window", marked reimplemented.
- Eqs. 4.14–4.15, Fig. 4.5: Eq. 4.15 denominator `2πσ⁶` correct; `fspecial('log')` matches R2025a `fspecial.m` and the 26 L2
  kernels; Fig. 4.5 values reproduced to 4 dp; `'unsharp'` centre `(α+5)/(α+1)` equals `I − laplacian(α)`.
- MATLAB `edge`: `/8`, `/6`, replicate padding, `cutoff = T²` vs `4·mean(b)` (Roberts 6), kx/ky per direction, LoG
  `fsize = 2·ceil(3σ)+1`, `op −= mean`, `0.75·mean|b|`, the four sign patterns and the `b == 0` case with `2T` all line up
  with `edge.m:353–442`; non-float rule matches the docstring's reason for raising `TypeError`. 196 + 11 + 14 maps 0 px.
- `strel`: Adams periodic-line decomposition, `intline` round-half-up, `'dis'` prefix matching with the ambiguous `'di'`
  error — 116 nhoods and 11 decomposition sequences identical to `getnhood`/`decompose`; Fig. 4.7(c) 9×9/69 px, Fig. 4.7(d)
  diamond 11×11/61 reproduced.
- Eqs. 4.16–4.21 borders/reflection: erosion pads 1/intmax/+Inf, dilation 0/intmin/−Inf; dilation uses the reflected SE
  with the anchor mapped to `(M−1−r0, N−1−c0)`; Fig. 4.8 matrices re-counted from the text (13×17) — transcription exact.
- Eqs. 4.22–4.23: `morphop_fast.m` thresholds 600 / 15, `is2DFull → useAlternate`, `imclose.m:76–78` zero `padarray`,
  Halide path = class min, `imopen.m:44` bare composition — the port's `uses_imclose_m` rule is exactly that; 54 + 18 L2 cases 0 px.
- Eqs. 4.24–4.38: `imreconstruct` = skimage fixed point with PARITY note, `marker ≤ mask` enforced, conn 4/8/arbitrary
  tested vs MATLAB; geodesic steps and literal `X_k` loops match; 1-D hand truths hold.
- Eqs. 4.39–4.42 / `_matlab_minus`: logical−logical → double, uint8/int16 saturation, Eq. 4.42 identity — all L2-exact;
  `morphology.m` 12 arrays and the r = 15 crop configuration 0 px / 0 levels.
- Semantics: literal `/256`, Fig. 4.3(a) crop `im(1600:2151,1979:2552)` → 0-based slices correct (NCC 0.99886),
  `medfilt2` zero-pad, `bwareaopen` ≥ P 8-conn, `conv2` `'full'` (4294×2860) in the script order.
- Scripts: `if __name__` guards, `load_image(..., allow_fallback=False)` + SKIP, book figure numbers in file names, no
  duplicated primitives; the §4.3 driver states Figs. 4.17–4.20 are procedure-only.

**Overall:** faithful to the book and to R2025a on everything the chapter exercises (bool/uint8/double), with unusually
strong L2 evidence; the one real bug is the untested scipy fallback in `_min_max_filter` (int64/uint64 erosion pads with
intmin), which must be fixed and covered before ch5 hands label images to `imerode`.
