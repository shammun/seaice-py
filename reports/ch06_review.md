# Chapter 6 — independent port review

**Chapter:** ch06, "GVF Snake-Based Ice Floe Boundary Identification and Ice Image Segmentation" (printed pp. 109–144)
**Reviewed:** 2026-09-10 · fresh-context `port-reviewer`, read-only
**Scope read:** `chapters/ch06.txt` §6.1–6.4 and Eqs. 6.34–6.58 in full · all 34 `.m` files in `matlab/ch6/` ·
R2025a's own `del2.m` and `regionprops.m` (`ComputeConvexHull`, `ComputePerimeterCornerPixelList`,
`computePerimeterFromBoundary`) · the whole Python port · `tests/test_ch06.py` · `reports/ch06_verification.md` ·
plus targeted probes run by the reviewer.

**Outcome:** 1 must-fix · 9 should-fix · 6 nits · 10 untested paths.
Written by the orchestrator from the reviewer's reply (the reviewer is read-only). Dispositions are recorded in the
last column and in `progress.json`.

---

## Must-fix

### M1 — `GradientOn = 0` computes a different field than MATLAB (integer-class arithmetic dropped)
`seaice/ch06_gvf_snake.py:207`

```python
f  = gaussian_blur(gray, sigma) if sigma else np.asarray(gray, dtype=np.float64)
f2 = gradient2_magnitude(...)   if gradient_on else np.asarray(f, dtype=np.float64)
```

`GVF_distance.m:50-61` sets `f = I` (**uint8**) and, with `GradientOn = 0`, `f2 = f` — still uint8. `GVF.m:21-23`
then evaluates `f = (f - fmin) / (fmax - fmin)` **in uint8**: the subtraction saturates and the division rounds to
an integer, so MATLAB's edge map collapses to a 0/1 image thresholded at the mid-range. `BoundMirrorExpand`'s
`B = zeros(...)` only promotes to double *after* that.

Reviewer's probe on a 0–200 gray image: the port feeds `gvf` the full-range float image (`f2 ∈ [0, 200]`), MATLAB
feeds it a binarised one — completely different `u, v`. The `gvf_force_field` docstring quotes the M-code verbatim
and claims "exact to the script", so this is an **over-claim**, not merely an untested branch.

**Fix:** keep the input class (`f = gray` when `sigma == 0`, `f2 = f` when `not gradient_on`) and reproduce MATLAB's
input-class normalisation inside `core.snake.gvf` — the ch05 `imimposemin` precedent, arithmetic in the input class.
If that is not wanted, make `gvf_force_field`/`gvf_distance` raise for `gradient_on=False` on an integer image and
delete the "exact to the script" claim. Add an L2 fixture: `GVF(uint8(rgb2gray(test8)), 0.1, 5)` with `GradientOn = 0`.

---

## Should-fix

| # | Location | What is wrong | Fix |
|---|---|---|---|
| S2 | `seaice/ch06_gvf_snake.py:527` | Docstring says `strel('disk',3)` is "7×7/37 px". MATLAB and `core.morphology.strel` both give **5×5 / 25 px** (`reference/ch06/strel.mat`; the report's Numbers table; `notebooks/build_ch06.py:1013` already says 5×5). This is the only printed evidence for `T_seed`, so a wrong number propagates into ch7/ch9. | Replace with "5×5 / 25 px octagon". |
| S3 | `seaice/ch06_gvf_snake.py:280` | Comment `# float32, as MATLAB's single-precision bwdist` is false — `core.distance.bwdist` returns **float64** and its own module docstring says so. Harmless for `'cityblock'` (integer values) but contradicts `analysis/ch06.md` §2.2 stage 12 and the ch05 rule about float32 distance maps feeding `imregionalmin`. | Drop the comment, or actually cast to float32 and say why. |
| S4 | `seaice/ch06_gvf_snake.py:306`, `_run_snake_passes:468` | `GVF_distance.m:125,129-130` computes the radii and circles in **single**: `img_Dist` is single, so `r = img_Dist(...)/sqrt(2)` and `cen(1) + r*cos(t)` are single, and only then `double(...)`. The port uses float64 throughout. Report Deviation 2 measures the effect (radii ≤ 4.5e-7, circle coords ≤ 3.1e-5), but `ceil()` downstream is discontinuous, so this is one of the two sources of the residual 61/31 730 px. | Three casts: `r = np.float32(np.float32(D) / np.sqrt(2.0))`, `x = np.float64(np.float32(cx + r*np.cos(CIRCLE_T)))`. Then tighten `test_initial_circles` from `1e-4` to `0.0`. |
| S5 | `seaice/core/snake.py:436` | `snakedeform(solver='circulant')` **silently ignores `book_index`** — `_circulant_eigs(N, alpha, beta, gamma)` never receives it and `snake_first_column` has no such parameter. Today it happens not to matter (constant α, β make the two matrices identical; probed 8.5e-14), but the API accepts the flag and quietly drops it. | Raise `ValueError` when `book_index and solver == 'circulant'`, or route the flag through. |
| S6 | five docstrings | **Parity labels contradict the measured report.** `core/filters.py:357` calls `homofil` `unverified` though the report and `test_homofil` measure ≤ 1.14e-12 → `exact`. `core/polygon.py:308` `polygeom` claims "exact"; the report says **near** (Deviation 5) and `test_polygeom` only checks `ang1/ang2` mod π. `core/polygon.py:282` `polyarea` claims "exact" with **no L2 reference test and no row in the report's parity table**, though the Verdict says every new core primitive has one. `ch06_gvf_snake.py:540,603` claim "exact"/"exact except k-means" for `gvf_distance`/`seaice_kmean_gvf`, measured **near** (61/31 730 px, 0.177 %). `core/regionprops.py:227` still calls `Perimeter` `near`, now measured **exact**. | Make each docstring say what the report measured. For `polyarea`, either add a reference (5 polygons incl. a clockwise and a self-intersecting one) or relabel `reimplemented` — verifier's call. |
| S7 | `scripts/ch06_for_test.py:24-27` | Prose states the opposite of what happens: "both ranges stay inside the image, so the field is merely sampled on a skewed grid". Measured: `ySpace = 1:148/64:148` indexes **rows** of a 108-row field, so **1088 of 4096** quiver samples are NaN. MATLAB's `interp2` returns NaN there too — the port is right, the prose is wrong, and `test_quiver_grid` already pins the NaN pattern. | Rewrite: ~27 % of the grid falls off the bottom of the image and comes back NaN in both. |
| S8 | `seaice/ch05_watershed.py:341` | **Rule 9 violation** — `component_centroids` duplicates `core.regionprops.regionprops(L, "Centroid")`. `analysis/ch06.md` §4.2 row 108 said "promote into `core.regionprops`"; ch06 wrote the new one and left ch05's in place. | Make `component_centroids` a thin wrapper over `core.regionprops`, keeping its signature and the ch05 tests, so there is one code path. |
| S9 | `seaice/ch06_gvf_snake.py:489-490`, `:473` | The port silently swallows what MATLAB crashes on: it adds `ok &= (xx >= 1) & (yy >= 1)`, but `GVF_distance.m:150` has no lower guard, so a contour point with `ceil(x) <= 0` makes MATLAB error. Snake points are unconstrained after `snakedeform`, so this is reachable on other images. Same for `if xc.size < 3: continue`. | Keep the guard but count dropped points into `SeedRecord` (e.g. `n_out_of_range`) and print it, so a future divergence from MATLAB is visible rather than silent. |
| S10 | `reports/ch06_verification.md:186-217` | Open items list 6 entries, but several **branches of the two ported functions are exercised nowhere** while the parity rows read as if the whole `.m` file was covered: `GradientOn = 0` (M1), `GVFOn = 0`, `sigma != 0` end-to-end, `timer > 1`, `normalize=False` (the book's Eq. 6.56 form — Deviation 10 says "both were run" but no test does), `strict_bwareaopen=True` / `n_negative > 0` (Risk R7 is latent at 0 on both book images, so the quirk never executes), `kmeans_impl='sklearn'` and the callable form, and S9's guard. | Add one Open item enumerating them. (Open item 4 is already fixed; open item 2 is being applied.) |

---

## Nits

- `core/regionprops.py:343` clamps `MinorAxisLength` with `max(uxx + uyy - common, 0)`; MATLAB does not clamp and
  would return a complex number for a never-observed −1e-17. Fine, but document it rather than diverge silently.
- `core/regionprops.py:113` — `_as_label_image` raises `ValueError: zero-size array to reduction` for a 0×0 float
  input, while `regionprops` on an empty bool array works. One-line guard.
- `core/snake.py:64` — `bound_mirror_expand` raises for `m < 2 or n < 2`; MATLAB's `BoundMirrorExpand` runs happily
  on a 1×N, so `GVF.m` on a 1×N image works in MATLAB and raises here. Pathological, but the docstring says "exact".
- `core/matlab_compat.py:185` — the column-vector branch of `del2` ignores `hy`; MATLAB's `parse_inputs` would use
  `v{2}` as `loc{1}` for `del2(colvec, hx, hy)`. Unreachable from ch06.
- `scripts/ch06_gvf_distance.py` panels (c)/(d) show the distance transform and maxima of `bw2` (only components
  that failed `Ra`/`Rc`/`Rl`), whereas book Fig. 6.15(c)/(d) show them for the whole binary image (Algorithm 1
  step 3: "D ← distance map of SEGMENTATION"). The script is faithful to the shipped code — add one caption
  sentence so a reader comparing with p. 136 is not confused.
- `core/polygon.py:261` — `convhull` claims "exact for the hull vertex set", contradicted by `tests/test_ch06.py:796`
  (`mine <= theirs`; MATLAB keeps collinear hull points). The report already labels it `near`.

---

## Untested paths, ranked by likelihood of being wrong

1. **`gvf_force_field(gradient_on=False)`** — proven wrong (M1). Test: `f2 = rgb2gray(imread('test8.jpg')); [u,v] = GVF(f2,0.1,5)`.
2. **`gvf_force_field(sigma != 0)` end-to-end** — `gaussianBlur` alone is exact at L2, but MATLAB passes it a uint8 `I`
   (`xconv2` promotes via `fft2`) and the chained `abs(gradient2(double(f)))` was never compared. Cheap L2 fixture, `sigma = 2`.
3. **`strict_bwareaopen` / `n_negative > 0`** (Risk R7) — measured 0 on both book images, so `bwareaopen` on a signed
   double with −1 pixels, the whole point of the quirk, has never executed.
4. **`timer > 1`** — the re-segmentation loop and its `break`. The `k.size == 0` break was probed and works, but no test asserts it.
5. **`GVFOn = 0`** (`[u,v] = gradient2(f2)` instead of GVF) — one cheap L2 fixture.
6. **`normalize=False`** — the book's own Eq. (6.56) form. Deviation 10 claims it was run; no test does.
7. **`kmeans_impl='sklearn'` and the callable form** — no test; the callable branch does `np.asarray(...).ravel()` with no validation.
8. **`initialize_contours(form='book')` beyond two toy fixtures** — never run on `alg_seg_gray.jpg`, where `-inf`
   background plus `imreconstruct` could differ.
9. **The out-of-range burn guard** (S9) — never triggered by any fixture.
10. **`regionprops` on 0-size input** and **`snakeinterp` reducing below 3 points** — both raise; MATLAB errors differently.

---

## Verified correct (spot-checked against the book and the `.m` source, not taken from the report)

- **Eqs. 6.34/6.36** — book pp. 119–120 confirmed line by line. `snake_matrix(book_index=True)` matches Eq. 6.36,
  `book_index=False` matches `snakedeform.m:29-33` with the authors' mirrored `m1`/`p1` names; the five wrap-around
  `diag(...)` terms (`core/snake.py:322-326`) transcribe `snakedeform.m:36-40` exactly; `snake_first_column` is the
  true first column of `A + γI` (checked for N = 3, 4).
- **Eq. 6.40** — `(A+λI)^{-1}(λx − f_x)` with `−f_x → +κ·v` justified by Eq. 6.56a's `−u`; the circulant FFT solve is
  the same linear system (probed 8.5e-14).
- **Eq. 6.53a** — `u + μ∇²₅u − |∇f|²(u − f_x)` is algebraically identical to the book's printed form at
  `Δt = Δx = Δy = 1`; `4·del2` restores Eq. 6.52c.
- **`del2`** — compared statement by statement against R2025a's `del2.m`: interior stencil, the `n > 3`
  extrapolation, the `n == 3` copy, the `n ≤ 2` zero, `v/ndims(f)` (÷2 even in the 1-D branch), the `del2(f,hx,hy)`
  dimension swap, and the row-vector transpose flag. Probe `del2([1 4 7.5 2])` → `[2.5, 0.125, −2.25, −4.625]`.
- **`regionprops`** — `ComputeEllipseParams` (the `+1/12`, the `y` negation, the `uyy > uxx` branch),
  `ComputePerimeterCornerPixelList`, `ComputeConvexHull` (**`convhull(rr, cc)` argument order is right**, checked at
  `regionprops.m:1129-1132`), `ComputeConvexImage`'s `roipoly(M, N, c, r)` with the `firstRow/firstCol` shift, and
  `computePerimeterFromBoundary`'s `isEven`/`isCorner` guard all match. `_perimeter_pixels` is provably equivalent to
  `bwmorph(...,'perim8')`, including where two 4-connected components touch diagonally.
- **Eqs. 6.57/6.58** — exactly the book's p. 133 wording, plus the constant-image and `+Inf` cases `imregionalmax` needs.
- **§6.3.3 / Algorithm 1** — `bwdist(~bw2,'cityblock')`, `imgDist(~bw2) = −inf`, `imregionalmin`, `Dis_img .* bw2` as
  double, `imdilate(strel('disk',3))`, `bwlabel(dis,8)`, 1-based `round(cen(2)), round(cen(1))`, `/√2`, `r == 0 → 2`,
  `t = 0:0.05:6.28` → 126 open points. Fig. 6.14's printed 8×8 matrices in `core/synth.py:470-483` are
  character-for-character what p. 135 prints (re-extracted).
- **Call order** — the port clips *before* `snakeinterp` in `scripts/ch06_for_test.py:109-110` and *after* it in
  `_run_snake_passes:470-472`, matching `for_test.m:84,92` and `GVF_distance.m:131-132`.
- **`ceil(iter/5)`/`floor(iter/5)` block split**, `dist.m`'s transpose and column-major `[p,q] = find`, `bw_kmeans`
  column-major reshape, stable `ind0 = argsort(s0)`, `out = bw1 + 0.5·bw0`, and `bwareaopen`'s default 8-conn vs the
  `,4` in the k-means line — all match the `.m` files.
- **`minboundrect`** and **`polygeom`** are D'Errico's and Sommer's algorithms verbatim, including the
  `nedges ∈ {0,1,2}` special cases and the centroidal→global inertia transfer.
- **Attribution complete and correct** — Xu & Prince + the IACL URL + the 1998 TIP reference in `core/snake.py:12-25`;
  D'Errico and H. J. Sommer III in `core/polygon.py:14-21`; the MathWorks-4 `gradient` provenance named without
  redistributing the source; Qin Zhang's academic-use notice and the 2015 TGRS citation in `ch06_gvf_snake.py:18-25`.

---

## Reviewer's evidence on the in-flight closed-ring question

The `.m` source is **decisive that the ring must be closed**. `for_test.m` calls `polybool` *before* `snakeinterp` on
the open 126-point circle, and the MATLAB reference records `XI0` at 252 points against the port's open-ring 251 —
the duplicated vertex survives `snakeinterp.m:27-34` there too (`IDX = (d < dmin)` with `dmin = 0` is false for a
zero-length segment), independently of `GVF_distance.m`. With `analysis/ch06.md` §0.5's probe (126 → 127, last =
first), closing is the right contract.

The residual 778-vs-777 is **not** evidence against it: `polybool` also rotates the start vertex and reverses the
traversal, so MATLAB's duplicate sits on a different point of the circle, and with `Dmax = 1` that shifts
`snakeindex`'s insertion parity by one — the ±1 instability already quantified in report Deviation 3 (2266 → 2262
under a 1e-13 perturbation). The honest assertions are the ones used elsewhere: identical burnt-pixel *set* and
mask, Hausdorff ≤ 1e-3, and `abs(len(py) - len(ml)) <= 2`.
