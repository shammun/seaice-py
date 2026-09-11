# Chapter 9 verification — Model Sea Ice Image Processing Applications (book pp. 195–212)

Date 2026-09-11 · verify phase against the port at commit `918864e`
(`ch09: port — 7 scripts exit 0, core/video.py, 1937 suite unchanged`).

> ## Read this before any number below
>
> **Three of chapter 9's four external inputs do not ship and are unobtainable.**  `04100_analyse.jpg`
> (Fig. 9.1, the overall tank image), `dypic_05100_cam1_top.avi` (§9.2.2) and `05100.avi` (§9.3.3) are
> HSVA/DYPIC campaign assets that were never published (risk R1).  Only
> `Model_Ice_Floe_Identification/model_ice.jpg` ships — and it is **not a printed figure** of the book.
>
> Consequently every row of the parity table below carries **two** independent judgements:
>
> * **as parity against MATLAB R2025a** — measured, on the *same* input bytes, and usually `exact`;
> * **as a book number** — `unverified` for everything that depends on the three missing files.
>
> `seaice.core.synth` generates Tier-3 stand-ins so the original `.m` files can be *run* and compared.  A number
> computed on a synthetic stand-in is **never** a book number and is never presented as one.  The book values
> N2, N3, N5, N6, N9, N10, N11, N12 and N19 are **not reproduced** and each has a one-line reason in
> "Open items".  Nothing was fabricated and no parameter search was performed to approach a printed value.

## Environment

python 3.11.5 · numpy 2.4.6 · scipy 1.17.1 · scikit-image 0.26.0 · opencv 5.0.0 · matplotlib 3.11.1 ·
imageio 2.37.4 (+ imageio-ffmpeg)
**Reference engine: MATLAB 25.1.0.2833191 (R2025a) Prerelease Update 2** via `tools/run_matlab_ref.py`
(`matlab -batch`, `set(0,'DefaultFigureVisible','off')`, `save(..., '-v7')`), driven by
`reference/ch09/make_refs.py`.  **Octave was not used anywhere** (`progress.json → environment.octave` is `null`
and no fallback was taken).  Toolboxes exercised: Image Processing, **Statistics** (`kmeans`),
**Mapping** (`polybool`, `toolbox/map/mapobsolete/polybool.m`).

### MATLAB sessions — 11, all `status: ok` (`outputs/ch09/verify/refs_log.json`)

| session | `.mat` written | seconds | what it pins |
|---|---|---|---|
| `video` | `video.mat` | 29.3 | **precondition** — `read(VideoReader)` on both Tier-3 AVIs, incl. `size()` |
| `probes` | `probes.mat` | 24.6 | errata E4/E5/E8, `>`/`>=`, `regionprops 'basic'`, `exist` of the removed functions |
| `r13` | `r13.mat` | 100.5 | risk R13 — the two ways MATLAB rejects `floe(k) = max([])` |
| `block` | `block.mat` + `ch09_block.mat` | 43.9 | `block_threshold.m` verbatim + the `gt`/`ge` tie fixture + global Otsu |
| `rect` | `rect.mat` | 23.4 | `rect.m` on two non-square masks, the swapped call, the collinear refusal |
| `model` | `model.mat` | 43.9 | `model_ice_model.m` on 11 rectangles + the strict `k` band |
| `polybool` | `polybool.mat` | 23.7 | `polybool('intersection', …)` on 10 pair geometries |
| `movie_otsu` | `ch09_movie_otsu.mat` | 41.1 | `movie_otsu.m` verbatim (both errata live) |
| `movie_kmeans` | `ch09_movie_kmeans.mat` | 36.4 | `movie_kmeans.m` verbatim under `rng(0)` |
| `movie_floe` | `ch09_movie_floe.mat` | 32.4 | `movie_floe.m` verbatim |
| `demo` | `ch09_demo.mat` | 427.7 | `model_ice_demo.m` end to end with lines 53–66 un-commented |

Total MATLAB wall time **826.9 s**.  Input provenance (md5 of every file both engines read) is in
`outputs/ch09/verify/input_md5.json`.

### Patch audit — the porter's `reference/ch09/patches.json`, edit by edit

`reference/ch09/patches.json` has **9 entries**.  I re-derived an independent patch set
(`reference/ch09/patches_verified.json`, produced by `reference/ch09/make_refs.py`) from the originals and
checked every one of the porter's edits against the shipped `.m` line it claims to replace.

| entry | class | edits | verdict |
|---|---|---|---|
| `block_threshold.m` → `ch09_block_threshold_ref.m` | rename + appended `save` | 1 | **OK**. The `imread('04100_analyse.jpg')` line is genuinely unchanged — the *file name* in the shipped script is the one the Tier-3 stand-in is written under, so the cwd does the redirection. No statement edited. |
| `movie_otsu.m` → `ch09_movie_otsu_ref.m` | IO + graphics | 3 | **OK**. L7 input name; L50–53 `figure/plot/xlabel/ylabel` removed *while keeping* `k = 1:numFrames;`; L59–65 `figure`/`imshow`/`getframe`/`movie2avi` → `VideoWriter('Uncompressed AVI')` writing the identical `bw(k).cadata` at the identical 12 fps. **Lines 16–48 — the whole numeric loop, both errata included — are byte-identical to the original.** Every quantity the removed lines consumed (`M` was a `getframe` buffer of `bw`) is still computed and saved. |
| `movie_kmeans.m` → `ch09_movie_kmeans_ref.m` | IO + graphics | 3 | **OK**, same shape; `map1`, `s`, `a`, `IC`, `out` all saved. |
| `movie_floe.m` → `ch09_movie_floe_ref.m` | IO | 1 + appended `save` | **OK**. `mmreader` → `VideoReader` is forced (`exist('mmreader') == 0`, proven in `probes.mat`). No other line touched; the file has no graphics at all. |
| `minboundrect.m` | API | 1 | **OK**, and it is the *same* one-literal patch ch06's reference already carries (`convhull(x,y,{'Qt'})` → `convhull(x,y)`; R2025a: *"CONVHULL no longer supports or requires Qhull-specific options"*). **Nit:** the recorded "removed" text is `edges = convhull(x,y,{'Qt'});` while the file's line 102 reads `  edges = convhull(x,y,{'Qt'});  % 'Pp' will silence the warnings` — the leading indentation and the trailing comment were dropped from the *record*, not from the patch. |
| `model_ice_demo.m` → `ch09_model_ice_demo_ref.m` | un-comment + `save` | 1 block | **OK**. The three executable lines of the shipped comment block (`S = rect(bw4);`, `k1 = 0.4; k2 = 2.5;`, `s_model = model_ice_model(S, bw4, k1, k2);`) are the authors' own, reproduced verbatim. |
| `model_ice_model.m` | none | 0 | **OK** — confirmed runnable unpatched in R2025a; `polybool` warns about the ring orientation and still returns the right answer. |
| `rect.m` | none | 0 | **OK** — `strmatch` still exists (`exist == 2`), though the branch that calls it is unreachable (E8). |
| the 13 GVF/snake files | none | 0 | **OK** — byte-identical to ch6/ch7 (asserted by a test). |

**Conclusion: all 9 patch entries are IO/graphics/output-plumbing only; no numeric literal and no numeric
expression is altered anywhere.**  The independent patch set produced identical results in every session, which
is the strongest form of this audit.

---

## The precondition: video decode parity (risk R2) — closed, in both directions

`test_uncompressed_avi_decodes_bit_identically_in_matlab_and_python`

| claim | measurement |
|---|---|
| MATLAB `read(VideoReader('dypic_synth_top.avi'))` vs `core.video.read_video` | **0 differing bytes** of 480·640·3·24 = 22 118 400 |
| MATLAB `read(VideoReader('05100_synth_segmented.avi'))` vs `read_video` | **0 differing bytes** of 240·320·3·40 = 9 216 000 |
| axis order pinned against MATLAB's own `size()` | `sz1 = [480 640 3 24]`, `sz2 = [240 320 3 40]` — i.e. **`(H, W, 3, N)`**, not imageio's `(N, H, W, 3)` |
| `get(v,'numberOfFrames')` | 24 / 40, equal to `core.video.video_num_frames` |
| **write** side: MATLAB `VideoWriter('otsu.avi','Uncompressed AVI')` vs `core.video.write_video(codec='rawvideo')` on the same 24 masks | **bit-identical** (`0` differing bytes, 6 389 000-byte files); same for `05400_kmeans.avi` |

This is stated first on purpose: **every §9.2.2 and §9.3.3 number below is an algorithm comparison only because
this is true.**  If a reader supplies a real (lossy) DYPIC video, the label for that run drops to `near`/`approx`.

---

## Parity table

One row per `analysis/ch09.md` §6 port-plan row and per new/extended core primitive.
Evidence levels: **L1** synthetic truth · **L2** MATLAB R2025a parity · **L3** figure reproduction ·
**L4** numbers printed in the book.

### The 21 `.m` files (7 PORT + 14 REUSE)

| # | MATLAB file | Python | Evidence | Result (measured) | Parity vs MATLAB | As a book number |
|---|---|---|---|---|---|---|
| 1 | `block_threshold.m` (47 l) | `ch09_model_ice.block_threshold` → `core.threshold.block_otsu(compare='ge')`, `scripts/ch09_block_threshold.py` | L1, L2 `ch09_block.mat`+`block.mat`, L3 | `thresh` **0.0** and `IC` **0.0** on all six blocks; the six `im2bw` tiles **0 px** of 615 960; `r,c = 348,1770` | **exact** | `unverified` (N3/N5 — image missing) |
| 2 | `movie_otsu.m` (83 l) | `ch09_model_ice.movie_otsu`, `scripts/ch09_movie_otsu.py` | L1, L2 `ch09_movie_otsu.mat`, L3 | `t` **0.0** (24 levels), the **E5-corrupted `n` vector equal element-for-element**, `IC` **0.0**, `bw` **0 px** of 2 105 688, `gray` **0 px** of 2 105 688, `I1`/`I2` identical, denominator 77 115 | **exact** | `unverified` (N9/N10/N11 — video missing) |
| 3 | `movie_kmeans.m` (85 l) | `ch09_model_ice.movie_kmeans`, `scripts/ch09_movie_kmeans.py` | L1, L2 `ch09_movie_kmeans.mat`, L3 | `IC` **0.0** on 24 frames; sorted cluster means ≤ **1.44e-13**; `out` masks **100.00 %** agreement; `si`, denominator, column-major `reshape` exact. `a(k)` (the label number) differs on 17 of 24 frames — not reproducible by construction | **approx** (Statistics-TB `kmeans` has its own RNG); the wrapper is **exact** | `unverified` (N9) |
| 4 | `…/model_ice_demo.m` (70 l) | `ch09_model_ice.model_ice_demo`, `scripts/ch09_model_ice_demo.py` | L1, L2 `ch09_demo.mat`, L3 | `rgb2gray` **identical**, `graythresh` level **0.0**, `im2bw` **0 px**; then `bw1` **28 px of 13 756 = 0.204 %**, `bw2`/`bw3`/`bw4` **28 px**; 30 rectangles and 23 accepted on both sides | **near** (inherited ch06 GVF/`polybool` residual, re-measured on ch9's own image) | n/a — `model_ice.jpg` is not a printed figure |
| 5 | `…/model_ice_model.m` (72 l) | `ch09_model_ice.model_ice_model` | L1, L2 `model.mat` + `ch09_demo.mat` | On **MATLAB's own `bw4`**: `Vertices` ≤ **2.84e-14**, `Center` ≤ **2.84e-14**, `Area` ≤ **1.14e-12**, `Perimeter` ≤ **5.68e-14**, 23/23 accepted, **all 23 `Intersection` sets identical (34 entries)**. On the 11-rectangle fixture: `bw` raster **0 px**, `ratios` ≤ 1e-12, adjacency matrix identical | **exact** for `Area`/`Center`/`Perimeter`/`Intersection`/`bw`; `near` for `Vertices` order (GPC) | n/a |
| 6 | `…/rect.m` (62 l) | `ch09_model_ice.rect`, `scripts/ch09_rect.py` | L1, L2 `rect.mat` + `ch09_demo.mat` | On two non-square masks (6 + 4 components): `Vertices`/`Center`/`Area`/`Perimeter` all ≤ **1e-12**, `bwlabel` label matrices **identical**. On MATLAB's own `bw4` (30 components): ≤ **5.68e-14** / ≤ **1.14e-12** | **exact** | n/a |
| 7 | `…/movie_floe.m` (31 l) | `ch09_model_ice.movie_floe`, `scripts/ch09_movie_floe.py` | L1, L2 `ch09_movie_floe.mat`, L3 | `floe(1..40)` **identical integers**; `im2bw` masks **0 px** of 3 072 000; `bwareaopen(·,20,4)` masks **0 px** of 3 072 000; per-frame 4-connected component counts identical; on frame 5 the `bwlabel` **label values** identical and `regionprops('basic')` `Area`/`Centroid`/`BoundingBox` **0.0** | **exact** | `unverified` (N19 — video missing) |
| 8 | `…/GVF_distance.m` (166 l) | `ch06_gvf_snake.gvf_distance` (**reused**, md5 `39cca98a…` = ch6 = ch7) | L2 `ch09_demo.mat` | `bw1` **28 px of 13 756 = 0.204 %** with the ch9 parameter set (ch06 measured 16 px of 31 730 on its own image) | **near** | n/a |
| 9 | `…/minboundrect.m` (204 l) | `core.polygon.minboundrect` (**reused**, md5 `3e179fb6…`) | L2 `probes.mat`, `rect.mat` | ring `x`,`y` ≤ **1e-12**, `a` and `p` ≤ 1e-12; 5 points, closed, positive cross product | **exact** | n/a |
| 10 | `…/GVF.m` (44 l) | `core.snake.gvf` (**reused**) | L2 (ch06, re-exercised here through `gvf_distance`) | class trap asserted: `GVF_distance.m` passes `abs(gradient2(double(f)))`, so the ch06 M1 uint8 defect is not reachable in ch9 | **exact** | n/a |
| 11 | `…/gradient2.m` (58 l) | `core.snake.gradient2` (**reused**) | L2 (ch06) | — | **exact** | n/a |
| 12 | `…/xconv2.m` (26 l) | `core.snake.xconv2` (**reused**) | L2 (ch06) | not reached by any ch9 driver (`sigma = 0`) | **exact** | n/a |
| 13 | `…/gaussianMask.m` (11 l) | `core.snake.gaussian_mask` (**reused**) | L2 (ch06) | not reached (`sigma = 0`) | **exact** | n/a |
| 14 | `…/gaussianBlur.m` (15 l) | `core.snake.gaussian_blur` (**reused**) | L2 (ch06) | not reached (`sigma = 0`) | **exact** | n/a |
| 15 | `…/BoundMirrorExpand.m` (31 l) | `core.snake.bound_mirror_expand` (**reused**) | L2 (ch06) | — | **exact** | n/a |
| 16 | `…/BoundMirrorEnsure.m` (41 l) | `core.snake.bound_mirror_ensure` (**reused**) | L2 (ch06) | — | **exact** | n/a |
| 17 | `…/BoundMirrorShrink.m` (27 l) | `core.snake.bound_mirror_shrink` (**reused**) | L2 (ch06) | — | **exact** | n/a |
| 18 | `…/snakedeform.m` (51 l) | `core.snake.snakedeform` (**reused**) | L2 (ch06) | ch9's contours are short (`N < CIRCULANT_MIN_N = 32`) so the **dense** path runs, which ch06 measured `exact` | **exact** (dense path) | n/a |
| 19 | `…/snakeinterp.m` (63 l) | `core.snake.snakeinterp` (**reused**) | L2 (ch06) | — | **exact** | n/a |
| 20 | `…/snakeindex.m` (9 l) | `core.snake.snakeindex` (**reused**) | L2 (ch06) | — | **exact** | n/a |
| 21 | `…/snakedisp.m` (21 l) | `core.plotting.snake_plot` (**reused**) | — | never called by a ch9 driver | **display** | n/a |

The 14 REUSE rows are backed by the md5 test `test_fourteen_ch9_files_are_byte_identical_to_ch6_and_ch7`,
which checks each file against **both** the ch6 and the ch7 copy, so "verified in ch06" is a checkable statement
and not a hope.  (`analysis/ch09.md` §0.2 is right and `knowledge/CUMULATIVE.md`'s "13" is wrong:
`minboundrect.m` is the 14th.)

### New and extended primitives

| symbol | Evidence | Result (measured) | Parity |
|---|---|---|---|
| **new** `core.video.read_video` / `write_video` / `video_num_frames` (**P2**) | L1, L2 `video.mat` | decode **0 differing bytes** on both fixtures, axis order `(H,W,3,N)` pinned against MATLAB's `size()`; encode bit-identical to `VideoWriter('Uncompressed AVI')` | **exact** on Uncompressed AVI (explicitly `near`/`approx` on any lossy container) |
| **new** `core.polygon.clip_polygon_convex` (**P1**) | L1, L2 `polybool.mat` | emptiness verdict matches `polybool` on **10 of 10** geometries (contained, disjoint, partial, edge-touching, vertex-touching, **partial-edge-overlap**, rotated/rotated, identical, 1e-9 gap, 1e-9 overlap); vertex **sets** identical on all 5 non-empty cases; with `drop_degenerate=False` the port is **wrong on 3 of 10** | **reimplemented** (only emptiness is consumed and emptiness *is* reproducible) |
| **new** `core.synth.model_ice_tank` (**P3**) | L1 | deterministic; the pre-noise blur populates the gray levels around the block threshold: **35–59 px sit exactly on each of the six thresholds** (measured in MATLAB, `block.mat` `at_th`), so the `gt`/`ge` difference is real and not a degenerate-fixture artefact. With `blur=0` the ±20-level window holds ≤ 3 occupied levels and ≤ 3 distinct masks; with the shipped default, 23+ of each | fixture (Tier 3) |
| **new** `core.synth.model_ice_tank_video` (**P4**) | L1, L2 | thresholds rise 122 → 143 then fall to 100, so E4 is observable: **17 of 24 frames change** when `running_max_bug` is switched off | fixture (Tier 3) |
| **new** `core.synth.segmented_floe_video` (**P5**) | L1, L2 | contains a 19-px and a 20-px component (pins `bwareaopen`'s `>= P`), a diagonally touching pair (pins `bwlabel(·,4)`: after `bwareaopen(·,20,4)` the frame has **15** 4-connected components but only **14** 8-connected ones) and an optional blank frame | fixture (Tier 3) |
| **ext** `core.threshold.block_otsu(compare=…)` (**X1**) | L1, L2 `block.mat` | default is still `'gt'` (ch3, unchanged); on the constructed tie fixture MATLAB's own counts are `ge − gt = [0, 1, 0, 0, 2, 1]` and the port reproduces both; one block has a **half-integer** level 104.5 | **exact** |
| **ext** `BlockOtsu.ic_mean` / `thresh_mean` | L1, L4 | `mean(83.93, 84.57, 82.65, 80.69, 82.97, 84.07) = 83.1467 → "83.14"`, `mean(83,82,85,81,83,91) = 84.17 → "84"` — book number **N4** reproduced. Through `block_otsu` `ic_mean ≡ ic` by construction (equal blocks), so the distinction is exercised on an **explicitly unequal-block record** (0.500000 vs 0.989011) | **exact** (arithmetic) |
| **ext** `core.regionprops.regionprops(L,'basic')` (**X2**) | L2 `probes.mat`, `ch09_movie_floe.mat` | MATLAB's `fieldnames` = `{Area, Centroid, BoundingBox}`, identical to `BASIC_PROPERTIES`; the 15 components of frame 5 match at **0.0** on all three | **exact** |
| **ext** `ch06_gvf_snake.component_criteria(ratio=…)` (**R6**) | L1 | default `'ellipse'` unchanged; on a 3-component fixture the two rules give a **different failing set** (`k = [0]` vs `k = [0, 2]`) and different `rl` (3.00/1.00/2.00 vs 3.22/1.00/2.20) | `'ellipse'` **exact**; `'minrect'` **reimplemented** (book text, no `.m`) |
| **ext** `ch06_gvf_snake.gvf_distance(stop=…)` (**G1**) | L1 | default `'criteria'` unchanged and never sets the flag; `stop='count'` fires on the first repeated count — on `model_ice.jpg` at `timer = 4` it runs **55 seeds instead of 67** and stops the loop, while `bw1` is unchanged (11 407 px) | **reimplemented** (Algorithm 7 line 7, absent from the shipped code) |
| `core.polygon.clip_polygon_rect` refactored onto the shared engine | L1 | 199 non-empty clips: **identical vertex sets**, identical vertex counts; `clip_polygon_rect` still uses the exact axis-aligned intersector, so ch06's numbers are untouched (whole ch06 suite green). The general convex engine rotates the ring on 134 of 199, which moves a 12×12 raster by **≤ 4 px** (9 cases, mean 0.08 px) — ch06's documented `poly2mask` start-vertex sensitivity | **exact** for `clip_polygon_rect`; the rotation residual is bounded and measured |
| `ch09_model_ice.tank_ice_concentration` (§9.2.1 text only) | L2 `block.mat` | Otsu level and IC **0.0** vs MATLAB `graythresh`/`im2bw`; the k=2 branch uses ch03's own `kmeans.m` port (`impl='authors'`, deterministic — the ch08-review rule) | **exact** as code; the §9.2.1 *results* `unverified` |
| `ch09_model_ice.rect_ice_concentration` (§9.3.2.1 text only) | L1 | `Σ Area / (M·N)` with overlaps **counted twice by construction** (the book's point); on a 2-rectangle overlap fixture 200/1600 > the union | **reimplemented** |
| `ch09_model_ice.fsd_error` (Fig. 9.16(b), text only) | L1 | bin-wise difference on shared `hist` **centres**; raises on mismatched centres | **reimplemented** |
| `ch09_model_ice.tiled_segmentation` (§9.3.2, text only) | L1 | grid/overlap/stitch rule are **ours**; only `TABLE_9_4` is the book's, and it is asserted as a transcription | **reimplemented** — no parity claim |
| `ch09_model_ice.segment_video` (gap **G2**) | L1 smoke | **ours**, not the book's; carries an explicit `# DEVIATION` and appears in no parity assertion | **reimplemented** — no parity claim |

### Parity counts

**36 rows with a label — vs MATLAB: `exact` 24 · `near` 3 · `approx` 1 · `reimplemented` 6 · `display` 1 ·
`fixture (Tier-3)` 3 (the three `core.synth` generators, which are inputs, not ports) · `unverified` 0.**

Itemised: `exact` = `.m` rows 1, 2, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 (16) + `core.video`,
`block_otsu(compare)`, `ic_mean/thresh_mean`, `regionprops('basic')`, `component_criteria(ratio='ellipse')`,
`clip_polygon_rect`, `tank_ice_concentration`, and row 5's `Area/Center/Perimeter/Intersection/bw` (8) = **24**;
`near` = rows 4, 8 and row 5's `Vertices` ordering = **3**; `approx` = row 3 (`movie_kmeans`, Statistics-TB RNG)
= **1**; `reimplemented` = `clip_polygon_convex`, `component_criteria(ratio='minrect')`, `gvf_distance(stop='count')`,
`rect_ice_concentration`, `fsd_error`, `tiled_segmentation`/`segment_video` = **6**; `display` = row 21 = **1**.

**As book numbers the counts are very different: 2 reproduced (N4, N8) + 2 adjudicated (N21/R8, N14+N15/R7) +
1 transcription-only (N13) — and 9 permanently `unverified` (N2, N3, N5, N6, N9, N10, N11, N12, N19).**

---

## Figures reproduced

`model_ice.jpg` is **not** a printed plate (best whole-book NCC 0.167, best `matchTemplate` 0.373, confirmed by
the analyst and re-asserted by a md5/shape test), and the tank image and the two videos do not ship at all.
**There is therefore no book plate anywhere in chapter 9 to put side by side.**  All L3 evidence below is
*Python vs MATLAB R2025a output on the same input*, i.e. L2 rendered visually plus the procedural shape of the
book's figure.  Files are in `reports/ch09/figures/` (git-ignored, rule 12).

| book figure | compare PNG | verdict (one sentence, after looking at it) |
|---|---|---|
| Fig. 9.4 (2×3 local Otsu) | `fig_9_04_compare.png` | The Python and MATLAB six-block masks of the Tier-3 tank are visually indistinguishable and their XOR panel is uniformly black — 0 differing pixels of 615 960 — with the six IC values printed identically. |
| Figs. 9.3 / 9.5 (global Otsu, k-means) | `fig_9_03_05_compare.png` | Python's global-Otsu mask and MATLAB's `graythresh`/`im2bw` mask are identical at level 138.0 (IC 86.71 %), and the k = 2 mask from the authors' own `kmeans.m` is visibly the same field. |
| Figs. 9.6 / 9.7 (crop, blank, per-frame masks) | `fig_9_06_07_compare.png` | The blanked vessel box, the crop and both per-frame masks match MATLAB panel for panel, with the black 94×113 rectangle sitting exactly where lines 22–30 put it. |
| Figs. 9.8 / 9.9 / 9.10 (IC(t), threshold(t)) | `fig_9_08_10_compare.png` | Python and MATLAB IC(t) curves lie on top of each other for both methods, and the threshold panel shows the running-maximum plateau of erratum E4 visibly departing from `t(k)` after frame 7 — the dotted "corrected" IC curve is the measurable consequence. |
| Fig. 9.11 (contour initialization, two passes) | `fig_9_11_compare.png` | The two-pass segmentation separates the aligned squares exactly as §9.3.1 describes, and the XOR panel shows the whole Python/MATLAB disagreement is 28 isolated single pixels scattered along floe boundaries. |
| Fig. 9.15 (rectangularization + model) | `fig_9_15_compare.png` | The 30 minimum-area rectangles hug the segmented floes with their red centre crosses, and the 23-rectangle model raster reproduces MATLAB's accepted set and all 34 overlap flags. |
| Fig. 9.16 (FSD + error) | `fig_9_16_compare.png` | The FSD/error pair has the book's form (identified vs rectangularized, then the bin-wise difference) but on 30 floes instead of hundreds — the book's 0–6000 px / 0–300 axes are not reachable without the tank image. |
| Figs. 9.17 / 9.18 (segmented frame, max floe area) | `fig_9_17_18_compare.png` | The segmented frame is pixel-identical to MATLAB's and the max-floe-area time series overlays exactly, showing the intended non-monotone shape that a `max`-vs-`last` confusion would break. |
| Figs. 9.12 / 9.13 / 9.16 colour ticks (R7) | `fig_9_12_13_ticks.png` | Both printed tick lists fall on one curve: the same colour range read at `n = 6` and at `nn = 8`, which is the whole of R7. |

---

## Numbers from the text

| # | book value | page | ours | verdict |
|---|---|---|---|---|
| **N4** | Fig. 9.4 caption "average IC = **83.14 %**, average threshold = **84**" | 198 | `mean(FIG_9_4_IC) = 83.1467 %` → truncated 83.14; `mean(FIG_9_4_THRESH) = 84.1667` → 84 | **reproduced** — and it proves the caption is the *unweighted mean of the six blocks*, not ch3's pixel-weighted total (ch9 deleted that line) |
| **N8** | crop `x∈[125,521]`, `y∈[180,400]`; vessel box `x∈[268,380]`, `y∈[307,400]`; IC denominator | 200 (code) | 221 × 397; 94 × 113 = 10 622; **77 115** = `IC_DENOMINATOR_5100` | **reproduced** (pure arithmetic on the script's literals), and MATLAB's own `r, c, r2, c2` confirm 221/397/94/113 |
| **N13** | Table 9.4, 20 GVF iteration counts | 207 | transcription asserted (150, 65, 65, 65, 160, 60, 110, 130, 90, 130, 170, 160, 100, 90, 90, 100, 90, 80, 90, 80) | **transcription only** — they are inputs; the 20 sub-images are not shipped |
| **N21 / R8** | "one strip of 1.50 m, four of 1.00 m and nine of 0.50 m coincide with the percentages recorded in Table 9.1" (45 / 40 / 15 %) | 196 | by **strip width** = by ice **area** (all strips span the same tank length): 4.5 : 4.0 : 1.5 of 10.0 m = **45 / 40 / 15 %** ✔ ; by **strip count** 9:4:1 = 64.3 / 28.6 / 7.1 % ; by **floe count** after the cross-cut (a strip of width *w* yields *L/w* squares) 18 : 4 : 0.667 = 79.4 / 17.6 / 2.9 % | **R8 resolved in the book's favour** — all three readings reported, and the width/area reading matches the printed columns **exactly and in the printed order**. See "Deviations" D1: `analysis/ch09.md` and the port's `TABLE_9_1` comment state the opposite, and are wrong |
| **N14 + N15 / R7** | Fig. 9.12 ticks **20, 202, 424, 710, 1113, 1798, 5878** (7); Figs. 9.13/9.16(a)/(b) ticks **20, 153, 307, 488, 710, 996, 1398, 2081, 5952** (9) | 207, 208, 210 | **all 16 printed integers reproduced exactly** from a *single* Eq.-(7.6) colour range `[198, 9974]` read with ch07's two `ice_shape_enhancement.m` conventions: `n = 6` on the map (line 196) and `nn = 8` on the histogram (line 227). `ysh = min : fix((max−min)/n) : max` does **not reach** `max`, and the two step sizes stop at 9972 and 9974 | **R7 adjudicated: neither an erratum nor different data.** Pre-image bands (CUMULATIVE pitfall 60): colour 198 pins the smallest floe to **exactly 20 px** (= `Ra_min`); the four colour maxima {9974, 9975, 9976, 9977} that fit both lists pin the largest floe only to **[5953, 6119] px**, so 5878 and 5952 are tick *labels*, not floe areas |
| N2 | Fig. 9.3 global Otsu IC **83.17 %**, threshold **84** | 198 | — | **unverified** (see Open items) |
| N3 | Fig. 9.4 six block ICs **83.93 / 84.57 / 82.65 / 80.69 / 82.97 / 84.07 %**, thresholds **83 / 82 / 85 / 81 / 83 / 91** | 198 | — | **unverified** |
| N5 / N6 | Table 9.2 run 5100 **83.17 / 83.14 / 82.86 %**; run 5200 **62.50 / 62.51 / 62.00 %** | 199 | — | **unverified** |
| N9 | Fig. 9.7 at t = 816 s: Otsu IC **87.26 %**, threshold **100**; k-means IC **86.91 %** | 201 | — | **unverified** |
| N10 / N12 | run 5100 saturates at ≈ **89 %** after ≈ **200 s**; band **80–89 %** | 200–201 | — | **unverified** |
| N11 | Table 9.3 start times **200/300/600/300 s**, average IC **88.93/80.39/81.69/84.83 %** | 203 | — | **unverified** |
| N18 | IC of Fig. 9.12 **76.96 %**, threshold **83.17 %**, Σ rectangles **87.75 %** | 208 | not reproduced; the *ordering* on `model_ice.jpg` is reported in the port's own output but is **not** the book's field | **unverified** as numbers; the identities are implemented and unit-tested |
| N19 | Fig. 9.18 time **0–1000 s**, max floe area **0–3×10⁴ px** | 211 | — | **unverified** |

Numbers measured on the **Tier-3** stand-ins, recorded here so the report can be audited but explicitly **not**
book numbers: Tier-3 tank block ICs `84.12 / 87.84 / 89.24 / 86.04 / 87.64 / 85.53 %` (mean 86.735 %), global
Otsu level 138.0 / IC 86.708 %; Tier-3 video Otsu IC(t) 46.66–62.50 %, k-means 47.92–64.03 %; Tier-3 segmented
video max floe area 987–1961 px over 40 frames.  On MATLAB's own `bw4` from `model_ice.jpg`: 30 rectangles,
23 accepted, 34 overlap flags.

---

## Deviations and justifications

**D1 — `analysis/ch09.md` R8/N21 and the port's `TABLE_9_1` comment are wrong (documentation defect).**
The analysis computes the area shares as `1·1.5² : 4·1.0² : 9·0.5² = 26.5 / 47.1 / 26.5 %`, which treats each
strip as producing **one** square; the text says the strips were "cut off such that the length was equal to the
width of the strip", i.e. each strip is cross-cut into many squares, so the area share equals the **width**
share.  It then lists the width reading as "15 / 40 / 45 %" and concludes "none of which is 45 / 40 / 15 % in
the printed order" — but the printed order is (0.50, 1.00, 1.50) m at (45, 40, 15) %, and 0.50 m *is* the 45 %
class.  The book is self-consistent.  `seaice/ch09_model_ice.py` reproduces the wrong conclusion verbatim in
the `TABLE_9_1` comment; **the code is unaffected, only the comment.**  Reported, not edited (the porter owns
`seaice/`).

**D2 — `rect(img, metric)` validates where MATLAB does not.**  `rect.m`'s `if (nargin<3)` in a two-argument
function is always true (erratum E8), so MATLAB silently ignores `metric` and **accepts a bogus value**
(`rect(bw,'not-a-metric')` returns normally; `probes.mat` `e8_ok == 1`, `e8_id == ''`).  The port raises
`ValueError`.  This is a deliberate, documented divergence (the docstring says the parameter "is validated the
way the unreachable branch would have validated it"), it cannot change any result, and it is pinned by a test —
but it is a divergence and belongs here.

**D3 — MATLAB's `rect.m` cannot run on a collinear component; the port can.**  `minboundrect.m` line 102 calls
`convhull` whenever the component has more than 3 pixels, and R2025a raises
`MATLAB:convhull:EmptyConvhull2DErrId` *"Error computing the convex hull. The points may be collinear."* on a
1-pixel line (a **single** pixel is fine — `minboundrect` takes the `n <= 1` branch).  This was found by this
verification: the first `rect` fixture contained a 1-px line and a 1-px blob and MATLAB refused the whole call.
The port returns a degenerate rectangle instead.  Neither shipped driver feeds `rect` a collinear component
(`bw4` comes from `bwareaopen(·, 20)`), so no result moves; the refusal is recorded as evidence and the
numeric fixtures were rebuilt without collinear components.

**D4 — `movie_kmeans.m` calls the Statistics Toolbox `kmeans`, so its labels are not reproducible.**
ch9 ships no `kmeans.m`, so — unlike ch08 §8.1, where the *text* names "the k-means method" and ch03's own
deterministic `kmeans.m` applies — line 44 really is the RNG routine, and `ima` is `im2double` output (not
integer), which `kmeans_gray` cannot consume.  The stand-in is `core.clustering.kmeans_lloyd(k=2,
init='kmeans++', seed=0)`.  Measured on the Tier-3 video the two agree completely on the *partition*
(sorted centres ≤ 1.44e-13, masks 100 %, `IC` 0.0) while the label **numbers** differ on 17 of 24 frames —
exactly the expected signature.  Label: `approx`.

**D5 — `polybool`'s vertex order is not reproducible, only its emptiness.**  Confirmed again here: the port's
`clip_polygon_convex` matches MATLAB's emptiness on 10/10 geometries and MATLAB's vertex **sets** on 5/5
non-empty ones, but no ordering rule was sought.  `model_ice_model.m` consumes only `isempty`, so this costs
nothing.  Newly pinned relations MATLAB drops as empty: **partial-edge overlap** and **vertex touching** (the
analysis had not probed either); a **1e-9-area** overlap is kept.

**D6 — the `gvf_distance` residual on ch9's own image is 28 px (0.204 %), not ch06's 16 px (0.050 %).**
CUMULATIVE pitfall 59 (identical code, different data) applies; the number is now recorded and asserted with a
ceiling of 35 px so a regression cannot hide.  Its cause is ch06's documented `polybool` start-vertex rotation
feeding `snakeindex`'s insertion parity — nothing new.

**D7 — the shared Sutherland–Hodgman engine rotates the ring relative to `clip_polygon_rect`.**
`clip_polygon_rect` applies the half-planes in the order (xmin, xmax, ymin, ymax) that ch06 verified;
`clip_polygon_convex` walks the clip ring's edges.  On 199 random clips the **vertex sets are identical** but
the sequence is rotated on 134 of them, which moves a rasterised 12×12 mask by ≤ 4 px in 9 cases (mean 0.08 px).
`clip_polygon_rect` itself still uses the exact axis-aligned intersector, so **no ch06 number moved** (whole
ch06 suite green).

**D8 — Tier-3 data.** Everything in §9.2 and §9.3.3 runs on seeded synthetic stand-ins.  They are written once
as files and *read back* by both engines (the ch03 `t.jpg` precedent for the JPEG; **Uncompressed AVI** for the
videos), so both sides decode identical bytes.  Determinism is asserted by a test.

---

## Errata (all confirmed against MATLAB R2025a in this verification)

| id | where | what | evidence |
|---|---|---|---|
| **E4** | `movie_otsu.m:39` | `if I(k).cdata(i,j) >= t*255` compares the pixel with the **whole growing `t` vector**, so `if` means `all(...)` and the effective threshold is the running maximum `max_{j<=k} t(j)` | `probes.mat`: `uint8(50) >= [25 51]` → `[1 0]`, the `if` is **not** taken; `>= [25 40]` → taken; `if <empty>` → not taken. Consequence measured: on the Tier-3 video **17 of 24 frames** change when the bug is switched off, max ΔIC 3.09 pp |
| **E5** | `movie_otsu.m:40` | the body is `n = n+1`, not `n(k) = n(k)+1`, so **every** element of the growing vector is incremented; closed form `n_final[j] = Σ_{m≥j} count_m` | `probes.mat`: counts (7, 6, 4) → MATLAB's `n = [17 10 4]` and `ic = [7 6 4]`. **`IC` escapes E5** (it reads `n(k)` inside the same iteration) but **not E4** — both halves confirmed, and the full `n` vector matches MATLAB element for element over 24 frames |
| **E5b** | `movie_otsu.m` | the written `otsu.avi` and the plotted `IC` **legitimately disagree**: the count uses `>=` at the *running maximum*, `im2bw(I(k).cdata, t(k))` uses strict `>` at the *per-frame* level | asserted: the AVI's per-frame mask counts equal `movie_otsu(..., running_max_bug=False, count_rule='gt')` and differ from `IC`'s numerator |
| **E6** | `movie_kmeans.m:79` | the input is run **5100** and the output AVI is named `05400_kmeans.avi` | source, unchanged |
| **E7** | §9.3.2.1 p. 207 vs `model_ice_model.m:34` | the text says ratios *less than* a threshold are removed; the code is `if k < k2 && k > k1` with `k` **not** normalised to ≥ 1 — a symmetric band with **strict** inequalities | `model.mat`: on rectangles with `k = 0.39, 0.40, 0.50, 1.00, 2.40, **2.50**`, MATLAB keeps exactly `{0.50, 1.00, 2.40}` — both boundary values rejected |
| **E8** | `rect.m:32` | `if (nargin<3)` in a **two**-argument function is always true, so `metric` is dead and `strmatch` is unreachable | `probes.mat`: `rect(bw,'not-a-metric')` returns normally; `rect(bw,'p')` is `isequal` to `rect(bw)` |
| **E9** | `model_ice_model.m:54` | `if xx ~= NaN` is `if ~isempty(xx)` | `polybool.mat`: `pb_taken == ~pb_empty` on all 10 pairs. Here the consequence is the **opposite** of ch08's `polyxpoly` case — `polybool` returns the intersection *region*, so containment **is** detected (5 vertices, area 36) |
| **R13 (corrected twice)** | `movie_floe.m:25` | `floe(k) = max(ice_areas)` on a blank frame is a **null assignment past the end** — MATLAB raises, it does **not** delete-and-shift | `r13.mat`: the identifier depends on the RHS form. `x(k) = []` (a **literal**) is parsed as a deletion → `MATLAB:subsdeldimmismatch` "Matrix index is out of range for deletion."; `x(k) = max(ice_areas)` — a **function call** returning `0×0`, i.e. the shipped line — is an assignment → `MATLAB:matrix:singleSubscriptNumelMismatch` "Unable to perform assignment because the left and right sides have a different number of elements". **The porter's docstring quotes the second and is correct for the shipped line**; `v = [1 2 3]; v(2) = []` does delete-and-shift but the loop never reaches that case |
| **not an erratum — R7** | Figs. 9.12 vs 9.13/9.16 | 7 ticks ending 5878 vs 9 ticks ending 5952 for "the same" data | both lists reproduced exactly from **one** colour range `[198, 9974]` with ch07's `n = 6` (map) and `nn = 8` (histogram); the difference is the `min : fix((max−min)/n) : max` truncation, which stops at 9972 and 9974 |
| **not an erratum — R8** | Table 9.1 / p. 196 | "one 1.50 m strip, four 1.00 m, nine 0.50 m coincide with the percentages" | the strip-width (= area) reading gives exactly 45 / 40 / 15 % in the printed order; the analysis's contrary claim is the defect (D1) |

---

## Gaps in the shipped code (confirmed, and confirmed to carry no parity claim)

* **G1 — Algorithm 7's `N0 ≠ N1` convergence test is not implemented.** `GVF_distance.m` line 80 computes `num`
  and never compares it.  The port exposes it as `gvf_distance(stop='count')`, **default unchanged**; the branch
  is executed by a test (55 vs 67 seeds on `model_ice.jpg` at `timer = 4`) and labelled `reimplemented`.
* **G2 — §9.3.3's per-frame Algorithm-7 segmentation has no shipped code at all.** `movie_floe.m` reads an AVI
  that already holds segmented frames (its `im2bw` has no level ⇒ 0.5 ⇒ `> 127.5`; MATLAB's own behaviour
  confirmed in `probes.mat`).  `ch09_model_ice.segment_video` is **ours**; it carries a `# DEVIATION` marker,
  says so in its docstring, and appears in **no** parity assertion — asserted by a test.
* **Text vs code, criterion 3** (p. 205 says the minimum-area bounding **rectangle** ratio; `GVF_distance.m`
  computes the **ellipse** axis ratio).  `ratio='ellipse'` remains the default everywhere, so no ch06/ch07
  number moves; `ratio='minrect'` is opt-in, has no MATLAB reference and is labelled `reimplemented`.

---

## Tests

`tests/test_ch09.py` — **48 tests, all passing** (`.venv/Scripts/python.exe -m pytest tests/test_ch09.py -q
-p no:cacheprovider`).  Composition: 24 L1 (provenance, transcription, errata in both directions, the additive
edits' defaults **and** their opt-in branches, fixture non-degeneracy) and 24 L2/L3.

Discriminating power was *demonstrated*, not assumed:

* the E4 test builds a video whose Otsu threshold **falls** and asserts the corrected variant gives a different
  count — a monotone fixture would pass vacuously;
* the E5 test computes the buggy closed form **and** the clean vector and asserts they differ;
* the `gt`/`ge` test uses a fixture with pixels **exactly on** the threshold (MATLAB's own counts differ by
  `[0,1,0,0,2,1]`) and a **half-integer** level, and the real Tier-3 tank is shown to have 35–59 such pixels per
  block, so the ch08 "empty histogram band" trap is excluded;
* the `ic_mean`/`ic` test shows they are **identical by construction** through `block_otsu` (equal blocks only)
  and exercises the distinction on an explicitly unequal-block record;
* the `rect.m` `(c, r)` swap test shows the rectangle **Areas are unchanged** by the swap (so an Area-only test
  proves nothing) and catches it on `Vertices`/`Center` thanks to the non-square canvas;
* the `drop_degenerate` test shows the flag is **wrong on 3 of 10** geometries when switched off;
* the `ratio='minrect'` and `stop='count'` tests each show the opt-in branch **changes the result**;
* the R7 test computes the *other* tick convention and asserts it does **not** reproduce the printed list.

Full suite: see the Verdict line.

---

## Open items

1. **O1 — `04100_analyse.jpg` (Fig. 9.1) does not ship and is unobtainable**, so book numbers **N2** (83.17 %,
   threshold 84), **N3** (the six block ICs/thresholds) and **N5/N6** (Table 9.2, both runs) are permanently
   `unverified`.  No public-domain top-view photograph of a *cut, rectangular* model-ice field exists (Tier 2
   fails) and the DYPIC/HSVA campaign data are unpublished (Tier 4 fails).  A Tier-3 stand-in gives MATLAB
   parity only.
2. **O2 — `dypic_05100_cam1_top.avi` does not ship**, so **N9** (t = 816 s: 87.26 % / threshold 100 / 86.91 %),
   **N10** (~89 % after ~200 s), **N11** (Table 9.3) and **N12** (the 80–89 % saturation band) are permanently
   `unverified`.
3. **O3 — `05100.avi` does not ship**, so **N19** (Fig. 9.18's 0–1000 s × 0–3×10⁴ px axes) is permanently
   `unverified`.
4. **O4 — N18's three ice concentrations (76.96 / 83.17 / 87.75 %) cannot be reproduced.**  All three identities
   are implemented and unit-tested, but they need the tank image; the ordering measured on `model_ice.jpg` is
   the port's own and is not the book's field.  Nothing was tuned to approach the printed values.
5. **O5 — Table 9.4's 20 GVF iteration counts (N13) are inputs with no shipped driver.**  The 20 overlapping
   sub-images are not defined anywhere in the book (no grid shape, no overlap, no stitch rule), so
   `tiled_segmentation` is ours and `reimplemented`; only the transcription is asserted.
6. **O6 — `analysis/ch09.md` R8/N21 and the `TABLE_9_1` comment in `seaice/ch09_model_ice.py` state a false
   conclusion** (D1).  The code is unaffected; the comment and the analysis text should be corrected by their
   owners.  The correct reading is in the "Numbers from the text" table and is asserted by
   `test_R8_table_9_1_strip_arithmetic_all_three_readings`.
7. **O7 — `rect` raises on a bogus `metric` where MATLAB accepts it silently** (D2), and **`rect` accepts a
   collinear component where MATLAB's `convhull` refuses the whole call** (D3).  Neither can move a shipped
   result; both are pinned by tests and listed here because they are divergences from the original.
8. **O8 — the `movie_kmeans` label numbers are not reproducible** (D4).  The partition, the cluster means and
   `IC` are; the labels are not, and the row is `approx` for that reason.  If a future reader needs reproducible
   labels, `impl='sklearn'` exists but is equally seed-dependent.
9. **O9 — R7's colour maximum is a band, not a point.**  Four colour maxima {9974…9977} reproduce both printed
   tick lists, which pins the largest floe of Figs. 9.12/9.13/9.16 only to **[5953, 6119] px**.  The smallest is
   pinned exactly (20 px).  No floe *count* is recoverable from the tick lists.

---

## Verdict: **PASS**

All seven `scripts/ch09_*.py` run end to end and exit 0 (37 files in `outputs/ch09/`).  All eleven MATLAB
R2025a sessions completed with `status: ok`.  All 21 `.m` files have a parity row, each with a MATLAB reference
or an explicit "verified in ch06, byte-identity asserted here" justification.  `tests/test_ch09.py` passes 48/48
and the full suite is **1985 passed / 2 skipped / 1 xfailed** = the 1937-test ch02–ch08 baseline + 48, with **no
ch02–ch08 regression and no tolerance loosened anywhere**.  Every `unverified` item is a missing-data item and
has an explicit entry in "Open items" (O1–O5, O9), and the two behavioural divergences found during this
verification (O7) are documented and pinned.
