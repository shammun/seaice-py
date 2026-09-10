# Chapter 7 — independent port review

**Reviewer:** `port-reviewer` agent, fresh context, read-only.
**Date:** 2026-09-10. **Reviewed at commit:** `9927dfa` (after the verification fixes).
**Inputs:** `chapters/ch07.txt`, `analysis/ch07.md`, `reports/ch07_verification.md`, `seaice/ch07_ice_type.py`,
`seaice/core/{morphology,histogram,plotting,synth,matlab_compat}.py`, `scripts/ch07_*.py`,
`K30735_.../matlab/ch7/`, R2025a's own `hist.m` / `imfill.m` / `imcomplement.m`, plus three fresh
`matlab -batch` adjudication probes.

**Outcome: 1 must-fix, 7 should-fix, 7 nits.** Everything else checked out — see "Verified" below.

---

## Must fix

### M1 — `imcomplement` promotes `single` to `double`, so `imfill` is not exact on the `single` branch

`seaice/core/matlab_compat.py:145-151`. The float fallback is `1.0 - img.astype(np.float64)`, but R2025a
`toolbox/images/images/imcomplement.m` l. 39-54 computes `1 - im` **in the input class** (`IM2 has the same
class`, l. 4). `imfill.m` l. 128-140 calls `imcomplement` twice around the reconstruction, so MATLAB
round-trips a `single` image through `1 - (1 - x)` in **single** while the port does it in **float64**;
`morphology.py:839` casts back only at the end, hiding the class but not the values.

Adjudicated against MATLAB (arrays exchanged via `.mat` so both sides see identical bits) on a 5x5 `single`
image with `I(1,1)=1e-8`, `I(1,3)=1+1e-8`, `I(5,5)=0.1`, rest 2:

| pixel | MATLAB `imfill(I)` | port |
|---|---|---|
| (1,1) | `0` | `1e-08` |
| (5,5) | `0.100000024` | `0.1` |

**2 of 25 elements differ, max 2.235e-8.** MATLAB's `imcomplement(single(1e-8))` returns `1`, confirming the
mechanism. The 20 committed `imfill` fixtures miss it because `single_hole` is 0/1 and `gray_single`'s values
happen to round-trip exactly.

`uint32`/`uint64` are wrong too — they fall through to the float branch, so `imcomplement(np.uint32([0,1,2]))`
returns `float64 [1., 0., -1.]` instead of `uint32 [4294967295, 4294967294, 4294967293]`.

**Fix:** return `1 - img` in the input dtype for floats; add the `uint32`/`uint64` branch
(`np.iinfo(dtype).max - img`). Latent for ch07 (which passes `double`), but `bwdist` returns `single` by this
project's own rule, so **ch08/ch09 will hit it**. The report's `imfill` row must also stop claiming the
`single` branch is exact.

*Code finding (porter) + one doc line (verifier).*

---

## Should fix

### S1 — the verification report is stale; it advertises a defect that no longer exists

`reports/ch07_verification.md:59-61, 68, 112, 247-255, 282-286, 325-328`. Commit `9927dfa` fixed the `-Inf`
filter and removed the `xfail`. `pytest tests/test_ch07.py -k "not test_script"` → **228 passed, 0 xfailed**;
no strict xfail remains. The `hist` parity label must become **exact** (line 112 still says `near` / "1 case
fails"), Open items 1 and 7(a)(b) must be struck, and the counts updated from `242 passed, 1 xfailed` /
`1816 passed, 2 xfailed` to **243** / **1817 passed, 1 xfailed**. *Doc finding (verifier).*

### S2 — the `eps(edges)` fix has no committed test; the discriminating case is missing

`seaice/core/histogram.py:143-147`. None of the 16 `hist_cases()` fixtures has an internal edge at a negative
power of two, so reverting to `np.nextafter` would keep the suite green. Adjudicated in MATLAB:

```
MATLAB: y = -2 + eps(-2)/2;  [z,n] = hist(y, [-3 -1 1])   ->  z = [1 0 0]
port (current, abs(np.spacing))    -> [1 0 0]   OK
port with np.nextafter(edges, inf) -> [0 1 0]   WRONG
```

Add `"eps_edge_neg_pow2": (np.array([-2.0 + abs(np.spacing(-2.0))/2]), np.array([-3., -1., 1.]))` to
`hist_cases()`, re-run `make_refs.py hist`, and assert directly that `abs(np.spacing(e)) == eps(e)` for the
seven quoted values (all seven confirmed). *Test finding (verifier).*

### S3 — `hist`'s "six non-finite probes" are cited but not in the repo

`seaice/core/histogram.py:114-115` claims agreement on "16 cases **plus six all-/mixed-non-finite probes**",
but `hist.mat` holds only the 16 + 2 empty keys and `fixtures.py` has one non-finite case. The
`finite.size == 0` branch (`histogram.py:125`, MATLAB `miny = maxy = 0`) is untested. The reviewer ran all six
against MATLAB and **the port matches every one** — so commit them:

| y | bins | MATLAB `z` | port |
|---|---|---|---|
| `[Inf -Inf NaN]` | 4 | `[1 0 0 1]` | same |
| `[NaN NaN]` | 3 | `[0 0 0]` | same |
| `[Inf Inf]` | 3 | `[0 0 2]` | same |
| `[-Inf -Inf]` | 3 | `[2 0 0]` | same |
| `[1 -Inf]` | 2 | `[1 1]` | same |
| `[-Inf 0 Inf]` | 5 | `[1 0 1 0 1]` | same |

*Test finding (verifier); the docstring sentence is a doc finding until they land.*

### S4 — `ice_shape_enhancement` takes `props[0]` where MATLAB concatenates every component

`seaice/ch07_ice_type.py:779-782`. `ice_shape_enhancement.m` l. 129-133 writes
`cen = regionprops(out==i,'centroid'); cen = cat(1, cen.Centroid)` and l. 142/149 appends, so if a later,
larger piece overwrites the middle of an earlier label MATLAB appends **k** centroids and the port appends one.
The reviewer searched for the case (400 rectangle + 300 smoothed-noise fixtures, 8-connectivity on every
surviving label) and found **zero** splits — a later piece can only grow into its own concavities/holes, which
a disjoint 4-connected neighbour cannot straddle. Latent, not observed; but handle `len(props) > 1` the way
the M-file does, or assert, so the assumption is explicit. *Code finding (porter).*

### S5 — `core.histogram.hist`'s line citations point at the wrong lines of `hist.m`

`seaice/core/histogram.py:85, 90, 93-96, 105, 107, 154`. R2025a `toolbox/matlab/graphics/math/hist.m` actually
has: the block at **118-156**, `min==max` widening at **119-122**, `linspace` at **123**, centres at **129**,
the explicit-centres branch at **133-137**, `edgesc = edges + eps(edges)` at **145**, the fold at **150-154**,
and the non-finite guard at **91-105** (the "107-115" cited on line 112 is the *complex* branch). The newer
inline comments are right, so the docstring contradicts itself. Rule 3. *Doc finding (porter — inside `seaice/`).*

### S6 — three deviations are in the code but in neither the report's Deviations nor its Open items

(a) `ch07_ice_type.py:455-456` — when `d == 0`, MATLAB's `min:0:max` is **empty** and its `YT` loop never runs;
the port returns one tick. `tests/test_ch07.py:586-587` tolerates it (`vals.size <= 1`) while the docstring
still says "Parity: **exact**".
(b) `ch07_ice_type.py:822` — `fsd = None` on empty `floe_area`, where MATLAB `hist([], 50)` returns
`zeros(1,50)` / `1:50` (`hist.m` l. 81-89). The MATLAB ref wrapper's `if ~isempty(...)` guard meant this was
never compared.
(c) `histogram.py:129-130` raises for `n < 1`, where MATLAB `hist(y,0)` takes the `binwidth = Inf` branch.

None is reachable from the book's parameters, but "Open items" is meant to be complete.
*Doc finding (verifier) + one docstring softening (porter).*

### S7 — `progress.json` carries counts the report itself corrected

`chapters.ch07.notes` says "19 REUSE from ch06 (all 25 SIFI .m byte-identical)". Re-run `cmp`:
`ch7/Sea_Ice_Floe_Identification/` holds **23** `.m`, **23/23 byte-identical** to ch6's, and the reuse count is
**17** (6 ported here + 17 reused + 4 deferred = 27). *Doc finding (orchestrator).*

---

## Nits

| # | Where | What |
|---|---|---|
| N1 | `core/synth.py:1240-1241` | `FIG_7_8_MARKER` docstring says "(48 pixels)"; it is **27** — the 48 is `H`. Already corrected in `analysis/ch07.md` (`4d65711`) but not here. |
| N2 | `core/synth.py:556` | "All **44** printed blocks" — it is **48** (4+8+7+8+4+10+7), as `reports/ch07_verification.md:115` says. |
| N3 | `ch07_ice_type.py:16-22` | The module port table lists 5 `.m`; `sea_ice_demo.m` → `scripts/ch07_sea_ice_demo.py` is missing. Rule 3 wants the home recorded at the source too. |
| N4 | `analysis/ch07.md:166` | "the ch7 parameter block **adds** `se_th`/`min_floe`/`min_brash`" — ch6's `sea_ice_demo.m` is byte-identical, so nothing is added; ch06 simply did not consume them. `reports/ch07_verification.md:87` repeats it. |
| N5 | `analysis/ch07.md:278` vs `core/synth.py:874` | Fig. 7.5(e)'s 8th block written `X7 ∪ A` in one place, `X6 ∪ A` in the other. Same array; pick one. |
| N6 | `ch07_ice_type.py:457` | `int(np.floor((hi-lo)/d + 1e-12)) + 1` — `lo`, `hi`, `d` are integral, so the ratio is exact and the absolute `1e-12` fudge is dead weight (and the wrong scale if `C1` rose). Prefer `(int(hi)-int(lo))//int(d) + 1`. |
| N7 | `core/synth.py:1217-1219` | `FIG_7_7_STEPS` is built by mutating copies of `FIG_7_7_STEPS_BOOK`'s arrays and both tuples hold **writable** module-level arrays; a test that writes into a block would corrupt every later test. `arr.setflags(write=False)`. |

---

## Verified (no action)

- **Every printed matrix of Figs. 7.2-7.8 independently re-extracted from `chapters/ch07.txt`** (digit runs at
  text lines 50/350/650/950, 1329-2066, 2180-2913, 2946-3841, 3881-4455, 4496-5556, 5578-6575) and compared
  row-major against `core.synth`: **48/48 step blocks plus all 12 image/complement/seed/marker panels 0 px**,
  except Fig. 7.7(e) `X8`, where the book's 53 px is correctly kept in `FIG_7_7_STEPS_BOOK`, the recursion's
  55 px in `FIG_7_7_STEPS`, and the two typo cells `(7,8),(8,8)` 0-based exactly as `FIG_7_7_X8_TYPO` records.
  `FIG_7_6_IMAGE` (25 px) matches the printed Fig. 7.6(b). **The transcriptions are from the book, not from
  the code's own output.**
- `ice_shape_enhancement.m` l. 39-181 lines up statement for statement: `k = k - k.*bw`, light-then-dark label
  numbering, stable `sort` (pinned by MATLAB's own `ind` on the `ties` fixture and on 1211 real pieces),
  `for i = 1:max(max(l))`, the **double** `imfill` around `imclose`/`imopen`, `r = length(p)` on the **raw**
  pre-fill area, `fill` from the *first* fill only, the running `t` with 4-connected `bwlabel` order per crop
  (cropping provably preserves the column-major first-occurrence order), `if area0 ~= 0`, `fix` not `round` in
  Eq. (7.6), `pixels = [c, r]` 1-based in column-major `find` order, and `coverage` from `sum(floe_area)`.
- **Eqs. 7.1-7.6 all check out** against the printed text (pp. 158-160 read verbatim). Eq. 7.5's `<`/`≥`
  matches the code's `if r < se_th` (the README's "≤" is the outlier, correctly ignored); Eq. 7.6's
  `C1=10000, C2=1000` and its inverse; Algorithm 5's printed `≥` vs the shipped `>` is exposed as
  `book_threshold` with both forms referenced.
- `core.morphology.imfill` is a faithful line-by-line port of `imfill.m` l. 124-145 including the
  class-dependent `-Inf` pad and the `validatestring` prefix rule — apart from M1's intermediate class.
- `core.histogram.hist` matches `hist.m` structurally in both branches; the `edges + eps(edges)` shift is now
  genuinely `eps` (`abs(np.spacing(e)) == eps(e)` confirmed in MATLAB for all seven values); and
  `searchsorted(..., 'right')-1` + folding the overflow bin reproduces `histc` including `±Inf`.
- `tile_grid`'s midpoint cut is an **exact partition**: 5 heights × 4 widths × 5 tile sizes × 7 overlaps =
  **700 configurations**, every pixel claimed exactly once, every core inside its bounds, full coverage, no
  exceptions, including `overlap > tile/2`.
- **Parity labels are honest.** `exact` labels have MATLAB L2 behind them; `reimplemented` is used for
  Eqs. 7.1/7.2, Algorithm 5 (book form) and Algorithm 6; `local_segmentation`'s docstring states plainly that
  the book fixes neither tile size, overlap, nor merge rule.
- **Rule 12** holds: all four book-image scripts use `load_image(..., allow_fallback=False)` and print `SKIP`
  on `FileNotFoundError`; `tests/test_ch07.py:1048-1052` enforces it.
- **Nothing is fabricated.** `154/189`, `2511/2624` and the eight coverage percentages are printed as "not
  reproducible here" by `scripts/ch07_ice_shape_enhancement.py:173-174` and marked `unverified` in the report.
- **23/23** of `ch7/Sea_Ice_Floe_Identification/*.m` are byte-identical to ch6's (`cmp -s`, re-verified),
  confirming the 17-reuse claim.
- The suite would catch the plausible bugs: complement flips, `round` for `fix` (`size_color(2) == 19`),
  unstable sort, row-major `find`, `strel('disk',1)` mistaken for a 3x3 square, and `>` vs `≥`.
