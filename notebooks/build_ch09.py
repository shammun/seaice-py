"""Build ``notebooks/ch09_model_sea_ice_applications.ipynb`` with nbformat (re-runnable; never edited by hand).

Chapter 9 — Model Sea Ice Image Processing Applications (Zhang & Skjetne 2018, pp. 195-212).
Follows the ``colab-notebook`` skill: title -> cell 1 (Drive mount / cwd) -> cell 2 (clone or pull the public
repo) -> cell 3 (``load_image``: private copy or public-domain substitute) -> one section per book section
(9.1 ... 9.3.3) -> parameter play -> summary / feeds forward.  All algorithms are imported from ``seaice``.

THE DEFINING CONSTRAINT OF THIS CHAPTER: three of its four inputs do not ship and are unobtainable
(``04100_analyse.jpg``, ``dypic_05100_cam1_top.avi``, ``05100.avi``).  Every cell that runs on a Tier-3
synthetic stand-in says so **in its own output**, and no book number is ever printed as if it were reproduced:
the ``not_reproduced()`` helper prints them only under an explicit banner, and the ``book()`` / ``tank_book()``
guards can only decorate a value that was actually computed from the book's own input.

Run:  ``.venv/Scripts/python.exe notebooks/build_ch09.py``
Verify: ``.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900
        --output executed_ch09.ipynb notebooks/ch09_model_sea_ice_applications.ipynb``  (then delete the copy).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

NB_NAME = "ch09_model_sea_ice_applications.ipynb"
HERE = Path(__file__).resolve().parent

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nb.metadata["colab"] = {"name": NB_NAME, "provenance": []}
C = nb.cells


def md(text: str) -> None:
    C.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text: str, tags: list[str] | None = None) -> None:
    cell = nbf.v4.new_code_cell(textwrap.dedent(text).strip())
    if tags:
        cell.metadata["tags"] = tags
    C.append(cell)


# =====================================================================================================================
# 1. Title + book mapping
# =====================================================================================================================
md(r"""
# Chapter 9 — Model Sea Ice Image Processing Applications

*Sea Ice Image Processing with MATLAB* (Q. Zhang & R. Skjetne, CRC Press 2018), Chapter 9, pp. 195–212 —
Python port `seaice-py`, notebook `notebooks/ch09_model_sea_ice_applications.ipynb`.

**Sections covered** (one notebook section each, in book order):
9.1 Experimental setup and model sea ice image data · 9.2 Ice concentration (9.2.1 overall tank image,
9.2.2 model sea ice video) · 9.3 Ice floe identification (9.3.1 contour initialization for crowded rectangular
floes, 9.3.2 overall tank image, 9.3.2.1 floe modeling, 9.3.3 video — monitoring maximum floe size).

**MATLAB files this notebook replaces** (`MATLAB_ROOT/ch9/`, 21 `.m` files — 7 ported, 14 reused verbatim from
Chapter 6):

| MATLAB file | Python (imported below) | Parity vs MATLAB R2025a |
|---|---|---|
| `block_threshold.m` | `ch09_model_ice.block_threshold` → `core.threshold.block_otsu(compare='ge')` | exact |
| `movie_otsu.m` | `ch09_model_ice.movie_otsu` | exact (both errata reproduced) |
| `movie_kmeans.m` | `ch09_model_ice.movie_kmeans` | approx (Statistics-Toolbox `kmeans` RNG; the partition is exact) |
| `Model_Ice_Floe_Identification/model_ice_demo.m` | `ch09_model_ice.model_ice_demo` | near (inherited ch06 GVF residual: 28 px of 13 756) |
| `…/rect.m` | `ch09_model_ice.rect` | exact |
| `…/model_ice_model.m` | `ch09_model_ice.model_ice_model` | exact for `Area`/`Center`/`Perimeter`/`Intersection`/`bw`; near for vertex order |
| `…/movie_floe.m` | `ch09_model_ice.movie_floe` | exact |
| `…/GVF_distance.m`, `GVF.m`, `gradient2.m`, `xconv2.m`, `gaussianMask.m`, `gaussianBlur.m`, `BoundMirror*.m`, `snakedeform.m`, `snakeinterp.m`, `snakeindex.m`, `snakedisp.m`, `minboundrect.m` | **reused** — `seaice.ch06_gvf_snake`, `seaice.core.snake`, `core.polygon` | verified in ch06; these 14 files are **byte-identical** to their ch6 and ch7 copies (md5-asserted by a test) |

Text-only material with no shipped `.m` (written from the book): global Otsu and *k*-means on the tank image
(Figs. 9.3/9.5) → `tank_ice_concentration`; the 20 overlapping sub-images of Table 9.4 → `tiled_segmentation`;
Σ-rectangle ice concentration (p. 208) → `rect_ice_concentration`; the Fig. 9.16(b) error → `fsd_error`;
Algorithm 7's missing convergence test → `gvf_distance(stop='count')`; §9.3.3's per-frame segmentation →
`segment_video`. **Everything in that list is ours, not the book's, and is labelled as such where it is used.**

New reusable primitives this chapter contributed: `seaice.core.video` (`read_video` / `write_video` /
`video_num_frames`, in MATLAB's `(H, W, 3, N)` axis order), `core.polygon.clip_polygon_convex` (MATLAB
`polybool`'s emptiness verdict) and three `core.synth` generators. Learned knowledge: `knowledge/ch09.md` and
`knowledge/CUMULATIVE.md`; evidence: `reports/ch09_verification.md`.

> ## ⚠️ Read this before any number below — chapter 9 is the chapter whose data does not exist
>
> Three of the chapter's four inputs are HSVA/DYPIC campaign assets that were **never published** and are
> **not** in the book's MATLAB archive: `04100_analyse.jpg` (the Fig. 9.1 overall tank image, §9.2.1),
> `dypic_05100_cam1_top.avi` (§9.2.2) and `05100.avi` (§9.3.3). No public-domain substitute exists for a *cut
> rectangular model-ice field* in a towing tank, and no login-based portal offers the campaign data.
>
> The only file that ships is `Model_Ice_Floe_Identification/model_ice.jpg` (181 × 76 RGB) — and it is **not a
> printed figure**: a whole-book search over ~1800 embedded bitmaps peaked at NCC 0.167 and a multi-scale
> template match inside this chapter's pages peaked at 0.373 (for comparison ch06's Fig. 6.15(a) scored 0.988).
> So §9.3's figures below are **procedural** reproductions, never side-by-side plate comparisons.
>
> Everything else runs on **Tier-3 synthetic stand-ins** from `seaice.core.synth`, written to disk and read
> back so that MATLAB and Python decode the same bytes. Each such cell prints a `TIER-3 SYNTHETIC` line in its
> own output. **A number computed on a stand-in is never a book number.** The book values
> **N2** (83.17 %, threshold 84), **N3** (the six block ICs and thresholds), **N5/N6** (Table 9.2),
> **N9** (87.26 % / 100 / 86.91 % at *t* = 816 s), **N10** (≈89 % after ≈200 s), **N11** (Table 9.3),
> **N12** (the 80–89 % saturation band) and **N19** (the Fig. 9.18 axes) are **permanently unverified**; where
> this notebook shows them at all it shows them under an explicit `NOT reproduced` banner printed by the
> `not_reproduced()` helper.
>
> What *is* genuinely reproduced, and is shown as such: **N4** (the Fig. 9.4 caption's averages),
> **N8** (the IC denominator 221·397 − 94·113 = 77 115), **N21/R8** (Table 9.1's 45/40/15 % from the strip
> widths) and **N14+N15/R7** (both printed colour-bar tick lists from one colour range).
""")

# =====================================================================================================================
# 2. Setup cells (Colab / Drive / local) — copied verbatim from build_ch02.py
# =====================================================================================================================
md(r"""
## Setup (Google Colab or local Jupyter)

Run the two cells below first. On **Colab** the first cell mounts your Google Drive and moves into
`MyDrive/Sea_Ice_Colab` — the folder where your private copy of the book's data lives
(`data/book/ch09/Model_Ice_Floe_Identification/model_ice.jpg`) and where downloads and the generated Tier-3
fixtures are cached between sessions; readers without Drive get a temporary `/content/Sea_Ice_Colab`. The second
cell clones (or updates) the public repository `seaice-py` there and installs its requirements. **Locally** both
cells are no-ops that move to the repository root. No GPU is needed.

> ⚠️ **If you run this with your own copy of the book's data, do not use *File → Save a copy in GitHub*.** That
> saves the cell outputs — figures rendered from copyrighted book material — into the public repository. Save to
> Drive instead (*File → Save a copy in Drive*). The repository's `_colab.ipynb` is always regenerated with
> outputs stripped.
""")
code(r'''
import os
try:
    from google.colab import drive; drive.mount('/content/drive')
    os.makedirs('/content/drive/MyDrive/Sea_Ice_Colab', exist_ok=True)
    DRIVE_OK = True
except Exception:
    DRIVE_OK = False
    if os.path.isdir('/content'):                                   # Colab without a Google Drive
        print("No Google Drive — using /content/Sea_Ice_Colab")
        os.makedirs('/content/Sea_Ice_Colab', exist_ok=True); os.chdir('/content/Sea_Ice_Colab')
    else:                                                           # local Jupyter: go to the repository root
        import pathlib
        here = pathlib.Path.cwd().resolve()
        for cand in (here, *here.parents):
            if (cand / "seaice").is_dir() and (cand / "data").is_dir():
                os.chdir(cand); break
    print(os.getcwd())
if DRIVE_OK:                                                        # the chdir to Drive only runs after a successful mount
    os.chdir('/content/drive/MyDrive/Sea_Ice_Colab')
    print(os.getcwd())
''')
code(r'''
import os, sys
IN_COLAB = os.path.isdir('/content') and ('google.colab' in sys.modules or os.getcwd().startswith('/content'))
if IN_COLAB:
    if os.path.exists('seaice-py'):
        !git -C seaice-py pull
    else:
        !git clone --depth 1 https://github.com/shammun/seaice-py.git
    !pip install -q -r seaice-py/requirements-colab.txt
    sys.path.insert(0, "seaice-py")
else:
    sys.path.insert(0, os.getcwd())                                 # local: the repository root from the cell above
print("seaice imported from:", "seaice-py/ (Colab clone)" if IN_COLAB else os.getcwd())
''')

# =====================================================================================================================
# 3. Data cells
# =====================================================================================================================
md(r"""
## Data

`seaice.core.io.load_image("ch09", "model_ice.jpg")` replaces MATLAB's `imread('model_ice.jpg')` in
`model_ice_demo.m`. It looks for **your private copy of the book image** first —
`data/book/ch09/Model_Ice_Floe_Identification/model_ice.jpg` in the current folder (on Colab:
`MyDrive/Sea_Ice_Colab`), in the repository, or in Drive — and otherwise downloads a **public-domain NASA
substitute** of the same 76 × 181 size (MODIS/Terra true colour, Beaufort Sea pack ice, 25 July 2019) into
`data/online/ch09/`. The banner it prints tells you which one you got.

The book image is a *laboratory* photograph (white model ice cut into rectangles, floating on the dark water of
the HSVA towing tank); the substitute is a real sea-ice scene with the same **structure** — bright distinct
floes on dark water in a tall narrow strip — not the same subject. Since `model_ice.jpg` is not a printed
figure, no book number is attached to it either way.

The second cell looks for the three inputs that do **not** ship. If you own them, drop them anywhere under
`data/book/ch09/` and every §9.2 / §9.3.3 cell will use them automatically; otherwise the notebook generates the
seeded Tier-3 stand-ins and marks every result they produce.
""")
code(r'''
from seaice.core.io import data_roots, load_image

I, SOURCE = load_image("ch09", "model_ice.jpg")        # = imread('model_ice.jpg'); uint8 (181, 76, 3)
FROM_BOOK = SOURCE.startswith("book")
book = (lambda s: f"  (book: {s})") if FROM_BOOK else (lambda s: "")   # book values only for the book's own input
if not FROM_BOOK:
    print("Running on a public-domain substitute image; figures show the same operations, "
          "but values quoted in the book only hold for the book's own image.")
print(f"model_ice.jpg: {I.shape} {I.dtype}   [{SOURCE}]")
print("NOTE: model_ice.jpg is NOT a printed figure of the book (best template match against every chapter-9 "
      "page bitmap: 0.373, vs 0.988 for ch06's Fig. 6.15(a)).  Every section-9.3 figure below is therefore a "
      "PROCEDURAL reproduction, not a plate comparison.")
if I.max() == I.min():
    print("WARNING: the image is constant — the download most likely failed; re-run this cell.")
''')
code(r'''
from pathlib import Path

def find_book_file(name: str):
    """First match of `name` under any `data/book/ch09/` root, or None (the notebook then uses a Tier-3 stand-in).

    The roots are exactly the ones `load_image` searches: the current folder, the repository, and
    `MyDrive/Sea_Ice_Colab` on Colab.  Sub-folders are walked, so the book's own layout works verbatim.
    """
    for root in data_roots():
        folder = root / "data" / "book" / "ch09"
        if folder.is_dir():
            for hit in sorted(folder.rglob(name)):
                return hit
    return None

TANK_IMAGE = find_book_file("04100_analyse.jpg")          # block_threshold.m line 2       (section 9.2.1, Fig. 9.1)
TANK_VIDEO = find_book_file("dypic_05100_cam1_top.avi")   # movie_otsu.m / movie_kmeans.m  (section 9.2.2)
FLOE_VIDEO = find_book_file("05100.avi")                  # movie_floe.m line 5            (section 9.3.3)

HAVE_TANK_IMAGE = TANK_IMAGE is not None
HAVE_TANK_VIDEO = TANK_VIDEO is not None
HAVE_FLOE_VIDEO = FLOE_VIDEO is not None

# The same guard idea as `book()`, one per missing input: a book value can only ever be printed NEXT TO a value
# computed from the book's own input.  With the input absent these lambdas return "" and nothing is quoted.
tank_book  = (lambda s: f"  (book: {s})") if HAVE_TANK_IMAGE else (lambda s: "")
video_book = (lambda s: f"  (book: {s})") if HAVE_TANK_VIDEO else (lambda s: "")
floe_book  = (lambda s: f"  (book: {s})") if HAVE_FLOE_VIDEO else (lambda s: "")

def not_reproduced(what: str, value: str, why: str) -> None:
    """Print a book number **under an explicit banner** saying it was not reproduced, and why.

    This is the only way a chapter-9 book number reaches the screen when its input is missing.  It exists so a
    reader can still see what the book claims without ever mistaking it for something this notebook computed.
    """
    print(f"NOT reproduced — {why}\n    the book prints: {what} = {value}")

def tier3(what: str) -> None:
    """One line, in the cell's own output, marking a figure/number as a synthetic stand-in."""
    print(f"TIER-3 SYNTHETIC: {what}.  The corresponding BOOK NUMBER IS NOT REPRODUCED (the input does not ship).")

if HAVE_TANK_IMAGE or HAVE_TANK_VIDEO or HAVE_FLOE_VIDEO:
    print("book inputs found:", [p.name for p in (TANK_IMAGE, TANK_VIDEO, FLOE_VIDEO) if p is not None])
else:
    print("SKIP (book data absent): none of the three chapter-9 inputs below was found.\n"
          "    04100_analyse.jpg         the overall tank image of Fig. 9.1        (section 9.2.1)\n"
          "    dypic_05100_cam1_top.avi  the top-view carriage video               (section 9.2.2)\n"
          "    05100.avi                 the already-segmented floe video          (section 9.3.3)\n"
          "  They are NOT in the book's MATLAB archive: they are HSVA/DYPIC campaign assets that were never\n"
          "  published, there is no public-domain photograph of a cut rectangular model-ice field, and no\n"
          "  login-based portal offers them.  If you obtain them, put them in\n"
          "    <this folder>/data/book/ch09/   (on Colab: MyDrive/Sea_Ice_Colab/data/book/ch09/)\n"
          "  and re-run this cell.  Otherwise the notebook runs the identical MATLAB algorithms on seeded\n"
          "  Tier-3 stand-ins from seaice.core.synth, and every number they produce is OURS, not the book's.")
''')

md(r"""
## Imports

Everything algorithmic comes from the `seaice` package; the notebook only calls it and draws figures. The
expensive step is the GVF snake (`gvf_distance`), which takes a few seconds on the small 181 × 76
`model_ice.jpg`; the two cells that apply it to a *tile grid* (§9.3.2) and to *video frames* (§9.3.3) are
deliberately downscaled and seed-capped through the `SCALE` / `MAX_SEEDS` constants defined there, and say so.
""")
code(r'''
import time

import numpy as np
import seaice.core                                        # imports seaice.core.plotting (sets the Agg backend)
from seaice.ch06_gvf_snake import component_criteria, gvf_distance
from seaice.ch07_ice_type import colorbar_area_ticks
from seaice.ch09_model_ice import (BOOK_PARAMS_CH9, BOX_5100, CROP_5100, FIG_9_4_IC, FIG_9_4_THRESH,
                                   IC_DENOMINATOR_5100, TABLE_9_1, TABLE_9_2, TABLE_9_3, TABLE_9_4,
                                   block_threshold, ensure_synthetic_segmented_video, ensure_synthetic_tank,
                                   ensure_synthetic_tank_video, floe_size_histogram, fsd_error, model_ice_demo,
                                   model_ice_model, movie_floe, movie_kmeans, movie_otsu, preprocess_frame,
                                   rect, rect_ice_concentration, segment_video, tank_ice_concentration,
                                   tiled_segmentation)
from seaice.core.connectivity import bwareaopen, label_components
from seaice.core.histogram import imhist
from seaice.core.io import read_image
from seaice.core.matlab_compat import rgb2gray_matlab
from seaice.core.plotting import imshow_matlab, label2rgb
from seaice.core.regionprops import regionprops
from seaice.core.threshold import block_otsu, graythresh, ice_concentration, im2bw
from seaice.core.video import read_video, video_num_frames, write_video

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100

print("chapter-9 parameter block (model_ice_demo.m lines 11–38), transcribed value by value:")
print("   ", ", ".join(f"{k}={v}" for k, v in BOOK_PARAMS_CH9.items()))
print("compare with ch06's sea_ice_demo.m: Ra_min 20 (was 10), Ra 1000 (2500), Num 150 (500), iter 150 (100), "
      "kappa 0.6 (0.5), timer 2 (1) — and no k-means stage.")
''')

# =====================================================================================================================
# 9.1 Experimental setup and model sea ice image data
# =====================================================================================================================
md(r"""
## 9.1 Experimental setup and model sea ice image data (pp. 195–196)

The data come from the **DYPIC** project's May-2011 campaign in the HSVA ice basin: a model Arctic drillship is
towed obliquely (heading 180°) through *managed* ice while a top-view still camera takes 28 photographs that are
stitched into one overall tank image (Fig. 9.1) and a top-view video camera on the carriage films the vessel's
neighbourhood. The managed ice is made by cutting a **54 m** level-ice sheet into strips — one of 1.50 m, four
of 1.00 m and nine of 0.50 m — cross-cutting the strips into **squares**, and spreading them over **64 m** of
tank. Four runs are analysed (5100, 5200, 5300, 5400) with the target conditions of **Table 9.1**.

Because every strip spans the same tank length and is then cut into squares, the **area share equals the width
share**: $9(0.50) + 4(1.00) + 1(1.50) = 10.0$ m, i.e. $4.5 : 4.0 : 1.5$ of $10.0$ m = **45 / 40 / 15 %** against
the printed size labels 0.50 / 1.00 / 1.50 m. That is Table 9.1 verbatim — the table is self-consistent and
this is book number **N21** *reproduced* (it needs no data at all). The next cell prints all three readings one
could take of the same sentence, so you can see why the other two do not match.
""")
code(r'''
sizes = np.array([0.50, 1.00, 1.50])            # p. 196, the printed size labels (in the printed order)
n_strips = np.array([9, 4, 1])                  # "nine 0.50 m strips, four 1.00 m strips, one 1.50 m strip"

width = n_strips * sizes                        # metres of sheet width in each size class
by_width = 100 * width / width.sum()            # == by AREA (every strip spans the same tank length)
by_strip_count = 100 * n_strips / n_strips.sum()
n_floes = (10.0 / sizes) * n_strips             # a strip of width w cut into squares yields L/w of them
by_floe_count = 100 * n_floes / n_floes.sum()

print(f"total sheet width                 : {width.sum():.1f} m of a 54 m sheet, spread over 64 m of tank")
print(f"by strip WIDTH (= by ice AREA)    : {by_width.round(1).tolist()} %   <-- Table 9.1, exactly and in order")
print(f"by strip COUNT                    : {by_strip_count.round(1).tolist()} %")
print(f"by FLOE COUNT after the cross-cut : {by_floe_count.round(1).tolist()} %")
print("book number N21 REPRODUCED: 45 / 40 / 15 % — pure arithmetic on the printed strip counts, no data needed.\n")

print("Table 9.1 (p. 196) — target ice conditions, transcribed:")
print(f"{'run':>6} {'target IC':>10}   floe edge lengths [m] at 45 / 40 / 15 %")
for run, row in TABLE_9_1.items():
    print(f"{run:>6} {100 * row['target_ic']:>9.0f}%   {row['sizes_m']}")
print("\nRun 5200 removes floes (lower IC); 5300 halves every floe diagonally; 5400 re-inserts them.")
''')

md(r"""
### The Fig. 9.1 tank image: what we use instead, and why

`04100_analyse.jpg` — the stitched overall tank image of run 5100 that `block_threshold.m` reads on its line 2 —
is **not in the book's archive**. The printed Fig. 9.1 bitmap is 442 × 87 px (aspect 5.08 : 1) and shows the
bright out-of-tank triangle in the upper-right corner that p. 199 blames for the ice-concentration deficit.
`seaice.core.synth.model_ice_tank` generates a **Tier-3 stand-in** with the same geometry (348 × 1770, aspect
5.09 : 1, rows divisible by 2 and columns by 3 so the 2 × 3 block grid of §9.2.1 slices evenly), the Table-9.1
size mix at the run-5100 target concentration, uneven illumination, JPEG-like blur, noise and the corner
triangle. It is written **once** as a JPEG and read back, so Python and MATLAB decode the same bytes.
""")
code(r'''
if HAVE_TANK_IMAGE:
    TANK = read_image(TANK_IMAGE)
    TANK_LABEL = f"the book's own 04100_analyse.jpg ({TANK_IMAGE.name})"
    print(f"using the book's overall tank image: {TANK.shape} from {TANK_IMAGE.as_posix()}")
else:
    tank_path, TANK = ensure_synthetic_tank()          # seeded; written once as JPEG q95 4:4:4 and read back
    TANK_LABEL = "Tier-3 synthetic stand-in for 04100_analyse.jpg"
    tier3("the overall tank image of Fig. 9.1 is generated by seaice.core.synth.model_ice_tank")
    print(f"    file: {tank_path.as_posix()}   shape {TANK.shape}   aspect {TANK.shape[1] / TANK.shape[0]:.2f}:1"
          f"   (the printed Fig. 9.1 bitmap is 442x87, aspect 5.08:1)")

fig, ax = plt.subplots(figsize=(13, 3.2))
imshow_matlab(ax, TANK)
ax.set_title(f"Figure 9.1 (procedure) — overall tank image, run 5100\n{TANK_LABEL}")
plt.show()
''')

# =====================================================================================================================
# 9.2 Ice concentration
# =====================================================================================================================
md(r"""
## 9.2 Ice concentration (p. 196)

Nothing new is derived here: the Chapter-3 routines are applied unchanged to model-ice data. Ice concentration
is the fraction of ice pixels,

$$\mathrm{IC} = \frac{\#\{(x,y) : f(x,y) \in \text{ice}\}}{M \cdot N},$$

and the ice/water decision comes either from a threshold $t$ (global Otsu, or Otsu inside blocks) or from
*k*-means with $k = 2$, taking the **brighter** cluster as ice.

### 9.2.1 Ice concentration from the overall tank image (pp. 197–199)

The book's argument for why all three methods agree here is worth keeping: the grayscale histogram is **clearly
bimodal** (Fig. 9.2), the illumination is nearly uniform, and there is only one ice type — so Otsu's two
assumptions hold and global ≈ local ≈ *k*-means. All three land 3–8 % below the target, which the authors blame
on imperfect sheet preparation plus the bright out-of-tank triangle.
""")
code(r'''
TANK_GRAY = rgb2gray_matlab(TANK)                          # block_threshold.m line 3: rgb2gray
counts, levels = imhist(TANK_GRAY)

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(levels, counts, width=1.0, color="0.25")
ax.set_xlabel("gray level"); ax.set_ylabel("number of pixels")
ax.set_title(f"Figure 9.2 (procedure) — grayscale histogram of the tank image\n{TANK_LABEL}")
ax.set_xlim(0, 300)                                        # the book's own axes (gray 0-300, p. 197)
plt.show()

peak_lo = int(levels[np.argmax(counts[:128])])
peak_hi = int(128 + np.argmax(counts[128:]))
print(f"two modes at gray {peak_lo} (water) and {peak_hi} (ice) — bimodal, which is Otsu's assumption"
      + tank_book("Fig. 9.2, gray 0-300, frequency 0-2.5e5"))
if not HAVE_TANK_IMAGE:
    print("(the histogram above is the stand-in's, not the book's; its shape is bimodal by construction)")
''')

md(r"""
**Global Otsu (Fig. 9.3) and *k*-means with k = 2 (Fig. 9.5).** Neither has a shipped `.m` file — the book
prints the two figures and their ice concentrations in the text only — so `tank_ice_concentration(I, method=…)`
composes the verified Chapter-3 primitives: `graythresh` + `im2bw` + `ice_concentration`, or the authors' own
deterministic `ch3/kmeans.m` (`core.clustering.kmeans_gray`) taking the brighter cluster as ice.
""")
code(r'''
ic_otsu, mask_otsu, th_otsu = tank_ice_concentration(TANK, "otsu")
ic_km,   mask_km,   _       = tank_ice_concentration(TANK, "kmeans")      # impl='authors' = ch3/kmeans.m

fig, axes = plt.subplots(3, 1, figsize=(13, 8))
imshow_matlab(axes[0], TANK);      axes[0].set_title(f"input — {TANK_LABEL}")
imshow_matlab(axes[1], mask_otsu); axes[1].set_title(f"Figure 9.3 (procedure) — global Otsu, "
                                                    f"threshold {th_otsu:.0f}, IC = {100 * ic_otsu:.2f} %")
imshow_matlab(axes[2], mask_km);   axes[2].set_title(f"Figure 9.5 (procedure) — k-means, k = 2, "
                                                    f"IC = {100 * ic_km:.2f} %")
plt.tight_layout(); plt.show()

print(f"global Otsu : threshold {th_otsu:>6.1f}   IC {100 * ic_otsu:6.2f} %" + tank_book("83.17 %, threshold 84"))
print(f"k-means k=2 :                  IC {100 * ic_km:6.2f} %" + tank_book("82.86 %"))
if not HAVE_TANK_IMAGE:
    tier3("the two ice concentrations above are the stand-in's")
    not_reproduced("Fig. 9.3 global Otsu IC / threshold (N2)", "83.17 %, threshold 84",
                   "04100_analyse.jpg does not ship and is unobtainable")
    not_reproduced("Table 9.2 run 5100 (N5)", "target 86.00 %, global Otsu 83.17 %, local Otsu 83.14 %, "
                   "k-means 82.86 %", "same missing image")
''')

md(r"""
**Local (block) Otsu — `block_threshold.m`, Fig. 9.4.** The script splits the image into `n_r = 2` by
`n_c = 3` blocks, runs `graythresh`/`im2bw` inside each one and prints the block's own IC and threshold. It is
Chapter 3's `local_Otsu.m` with **five** differences, of which exactly one is numeric: line 26 compares
`temp(r1,c1) >= th` where ch3 has `>`. Our `core.threshold.block_otsu` therefore takes a `compare=` parameter
(`'gt'` = ch3, `'ge'` = ch9) instead of being copied; `block_threshold()` is the thin ch9 wrapper.

The caption's "average IC" is the **unweighted mean of the six blocks**, not Chapter 3's pixel-weighted total —
ch9 deleted that line. Book number **N4** proves it and is reproduced two cells below.
""")
code(r'''
B = block_threshold(TANK)                                   # = block_otsu(rgb2gray(TANK), 2, 3, compare='ge')

fig, axes = plt.subplots(2, 3, figsize=(13, 4.2))
for b, ax in enumerate(axes.ravel()):
    rs, cs = B.slices[b]
    imshow_matlab(ax, B.bw[rs, cs])
    ax.set_title(f"IC = {100 * B.ic_local[b]:.2f} %\nthreshold = {B.thresholds[b]:.0f}", fontsize=9)
fig.suptitle(f"Figure 9.4 (procedure) — 2 x 3 local Otsu, `block_threshold.m`\n{TANK_LABEL}")
plt.tight_layout(); plt.show()

print("block ICs   :", (100 * B.ic_local).round(2).tolist(), tank_book(str(list(FIG_9_4_IC))))
print("thresholds  :", B.thresholds.round(0).astype(int).tolist(), tank_book(str(list(FIG_9_4_THRESH))))
print(f"average IC  : {100 * B.ic_mean:.2f} %   (unweighted mean of the six blocks — the caption's quantity)")
print(f"pixel-weighted IC (ch3's deleted line 50): {100 * B.ic:.2f} %  — identical here BY CONSTRUCTION, because")
print("    block_otsu cuts six equal blocks; the two quantities only separate when the blocks differ in size.")
if not HAVE_TANK_IMAGE:
    tier3("the six block values above are the stand-in's")
    not_reproduced("Fig. 9.4 six block ICs (N3)", " / ".join(f"{v:.2f}" for v in FIG_9_4_IC) + " %",
                   "04100_analyse.jpg does not ship")
    not_reproduced("Fig. 9.4 six thresholds (N3)", " / ".join(str(v) for v in FIG_9_4_THRESH),
                   "04100_analyse.jpg does not ship")
''')

md(r"""
### Trap — `>=` is not `>`: ch9's `block_threshold.m` is *not* ch3's `local_Otsu.m`

`knowledge/CUMULATIVE.md` used to record these two scripts as the same algorithm. They are not: every pixel
whose gray level **equals** the block threshold is ice in chapter 9 and water in chapter 3. MATLAB R2025a:
`uint8([100 101 102]) > 101` → `[F F F]`, `>= 101` → `[F T T]`, while `im2bw(·, 101/255)` → `[F F T]` (strict
`>`). So in **both** chapters the displayed tile (`im2bw`) and the counted IC already use different rules, and
chapter 9 widens the gap by one gray level. The cell below measures the difference on this image — and, because
a fixture whose histogram is empty at the threshold would make the test pass vacuously, it first counts how many
pixels actually sit *on* each threshold.
""")
code(r'''
gt = block_otsu(TANK_GRAY, 2, 3, compare="gt")              # ch3/local_Otsu.m line 27
ge = block_otsu(TANK_GRAY, 2, 3, compare="ge")              # ch9/block_threshold.m line 26

print(f"{'block':>5} {'threshold':>10} {'px == th':>9} {'count >':>9} {'count >=':>9} {'IC(>) %':>9} {'IC(>=) %':>9}")
for b, (rs, cs) in enumerate(ge.slices):
    tile = TANK_GRAY[rs, cs]
    on_th = int(np.count_nonzero(tile == ge.thresholds[b]))
    print(f"{b:>5} {ge.thresholds[b]:>10.1f} {on_th:>9} {int(gt.counts[b]):>9} {int(ge.counts[b]):>9} "
          f"{100 * gt.ic_local[b]:>9.3f} {100 * ge.ic_local[b]:>9.3f}")
print(f"\nthe two rules differ on {int((ge.counts - gt.counts).sum())} pixels in total "
      f"({100 * (ge.ic_mean - gt.ic_mean):+.4f} pp of average IC).")
print("The pixels-on-the-threshold column is what makes this a real test rather than a vacuous one.")
''')

md(r"""
### Book number N4 — reproduced, with no data at all

The Fig. 9.4 caption reads "average IC = 83.14 %, average threshold = 84". Those two numbers are the
**unweighted means of the six printed block values**, which is pure arithmetic on the printed figure and
therefore *is* reproducible even though the image is gone. It also settles what the caption means, because
chapter 9 deleted chapter 3's pixel-weighted `IC = sum(num(:))/(r*c)`.
""")
code(r'''
ic_mean = float(np.mean(FIG_9_4_IC))
th_mean = float(np.mean(FIG_9_4_THRESH))
print(f"mean({', '.join(f'{v}' for v in FIG_9_4_IC)}) = {sum(FIG_9_4_IC):.2f}/6 = {ic_mean:.4f} %"
      f"  -> printed '83.14 %'   {'OK' if f'{ic_mean:.4f}'.startswith('83.14') else 'MISMATCH'}")
print(f"mean({', '.join(str(v) for v in FIG_9_4_THRESH)}) = {sum(FIG_9_4_THRESH)}/6 = {th_mean:.4f}"
      f"  -> printed '84'        {'OK' if round(th_mean) == 84 else 'MISMATCH'}")
print("book number N4 REPRODUCED — and it proves the caption's 'average IC' is the unweighted block mean.")
print(f"\nTable 9.2 as printed (p. 199) — transcribed only, NOT reproduced: {TABLE_9_2}")
not_reproduced("Table 9.2 run 5200 (N6)", "target 70.00 %, 62.50 / 62.51 / 62.00 %",
               "the run-5200 tank image does not ship either")
''')

# =====================================================================================================================
# 9.2.2 Ice concentration from model sea ice video
# =====================================================================================================================
md(r"""
### 9.2.2 Ice concentration from model sea ice video (pp. 199–201)

Four HSVA videos longer than 24 min at 25 fps are **decimated to one frame per second** and fed to
`movie_otsu.m` / `movie_kmeans.m`, which run the same global Otsu and *k*-means per frame and plot IC against
time (Figs. 9.8/9.9) and the Otsu threshold against time (Fig. 9.10). Two pre-processing steps come first
(Fig. 9.6): the tank impediments are **cropped away** and the vessel is **blanked with a black rectangle**.
Both happen *before* `rgb2gray`, so the histogram Otsu sees contains 10 622 forced-black pixels — the book
admits this on p. 200 and compensates by removing that area from the IC **denominator** (but not from the
numerator).

`core.video.read_video` replaces `VideoReader` + `read` and returns MATLAB's `(H, W, 3, N)` array, so
`vidFrames(:,:,:,k)` ports as `frames[:, :, :, k]`; `core.video.write_video` replaces `movie2avi`, which R2025a
removed. On an **Uncompressed AVI** both directions are byte-identical to MATLAB (0 differing bytes of
22 118 400) — which is the only reason the numbers below are an algorithm comparison rather than a codec one.
""")
code(r'''
N_FRAMES = 24                                              # the book decimates > 24 min at 25 fps to 1 fps

if HAVE_TANK_VIDEO:
    FRAMES = read_video(TANK_VIDEO)
    VIDEO_LABEL = f"the book's own {TANK_VIDEO.name}"
    print(f"using the book's video: {FRAMES.shape} (H, W, 3, N), {video_num_frames(TANK_VIDEO)} frames")
else:
    video_path, FRAMES = ensure_synthetic_tank_video(n_frames=N_FRAMES)   # Uncompressed AVI, written once
    VIDEO_LABEL = "Tier-3 synthetic stand-in for dypic_05100_cam1_top.avi"
    tier3("the tank video is generated by seaice.core.synth.model_ice_tank_video")
    print(f"    file: {video_path.as_posix()}   frames {FRAMES.shape} (H, W, 3, N), uncompressed AVI")
    print("    it is built to make the chapter's two errata observable: the Otsu threshold RISES and then FALLS,")
    print("    the ice concentration changes over time, and a lamp highlight sits on the water.")

frame1 = FRAMES[:, :, :, 0]
cropped, I1, I2, r2, c2 = preprocess_frame(frame1, CROP_5100, BOX_5100)   # lines 19-30, verbatim
blanked = frame1.copy(); blanked[BOX_5100[0] - 1:BOX_5100[1], BOX_5100[2] - 1:BOX_5100[3], :] = 0

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
imshow_matlab(axes[0], frame1);  axes[0].set_title("(a) raw frame")
imshow_matlab(axes[1], blanked); axes[1].set_title(f"(b) vessel blanked, {r2} x {c2} px")
imshow_matlab(axes[2], cropped); axes[2].set_title(f"(c) cropped, {cropped.shape[0]} x {cropped.shape[1]} px")
fig.suptitle(f"Figure 9.6 (procedure) — pre-processing, `movie_otsu.m` lines 19–30\n{VIDEO_LABEL}")
plt.tight_layout(); plt.show()
''')

md(r"""
#### Book number N8 — reproduced: the ice-concentration denominator

The crop and the vessel box are numeric literals *inside the scripts*, so their arithmetic is reproducible even
without the video. `x ∈ [125, 521]`, `y ∈ [180, 400]` gives a 221 × 397 crop; the box `x ∈ [268, 380]`,
`y ∈ [307, 400]` covers 94 × 113 = 10 622 px; the denominator the scripts divide by is
$221 \cdot 397 - 94 \cdot 113 = 77\,115$. (Supporting evidence that the printed "domain image" really is this
crop: Fig. 9.6(b) is printed at aspect 1.807 and Fig. 9.7 at 1.789, against the crop's 397/221 = 1.796.)
""")
code(r'''
y1, y2, x1, x2 = CROP_5100
y3, _, x3, x4 = BOX_5100
r, c = y2 - y1 + 1, x2 - x1 + 1
br, bc = y2 - y3 + 1, x4 - x3 + 1
print(f"crop      x in [{x1}, {x2}], y in [{y1}, {y2}]  ->  {r} x {c} = {r * c} px")
print(f"vessel box x in [{x3}, {x4}], y in [{y3}, {y2}]  ->  {br} x {bc} = {br * bc} px")
print(f"denominator = {r}*{c} - {br}*{bc} = {r * c - br * bc}   (IC_DENOMINATOR_5100 = {IC_DENOMINATOR_5100})")
assert r * c - br * bc == IC_DENOMINATOR_5100 == 77115
print("book number N8 REPRODUCED (pure arithmetic on the scripts' literals; MATLAB's own r, c, r2, c2 agree).")
''')

md(r"""
#### Per-frame Otsu and *k*-means, and the two curves the book plots

`movie_otsu(frames)` returns every array the M-file leaves in the workspace — the per-frame level `t`, the
corrupted counter `n`, the clean `counts`, the plotted `IC`, the `im2bw` masks that go into the AVI, and the
*effective* level the count really used. `movie_kmeans(frames)` does the same for the clustering version; note
that its output file is named `05400_kmeans.avi` while its input is run **5100** (erratum **E6**).
""")
code(r'''
t0 = time.time()
MO = movie_otsu(FRAMES)                       # both errata live, exactly as shipped
MK = movie_kmeans(FRAMES, seed=0)             # Statistics-Toolbox kmeans -> kmeans_lloyd(init='kmeans++')
print(f"{len(MO.IC)} frames in {time.time() - t0:.1f} s;  denominator {MO.denominator}")

k_axis = np.arange(1, len(MO.IC) + 1)         # the M-file's `k`, labelled "Time" because the video is 1 fps
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
axes[0].plot(k_axis, 100 * MO.IC, "b.-", label="global Otsu (as shipped)")
axes[0].plot(k_axis, 100 * MK.IC, "g.-", label="k-means, k = 2")
axes[0].set_xlabel("Time [s]  (frame index; the video is decimated to 1 fps)")
axes[0].set_ylabel("ice concentration [%]"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_title("Figures 9.8 / 9.9 (procedure) — IC(t)")
axes[1].plot(k_axis, 255 * MO.t, "r.-", label="t(k) — the per-frame Otsu level")
axes[1].plot(k_axis, 255 * MO.effective_level, "k.--", label="the level the count really uses (E4)")
axes[1].set_xlabel("Time [s]"); axes[1].set_ylabel("threshold [gray level]")
axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_title("Figure 9.10 (procedure) — threshold(t)")
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
mid = len(MO.IC) // 2
imshow_matlab(axes[0], MO.gray[mid]); axes[0].set_title("(a) cropped gray frame")
imshow_matlab(axes[1], MO.bw[mid]);   axes[1].set_title(f"(b) Otsu mask, level {255 * MO.t[mid]:.0f}")
imshow_matlab(axes[2], MK.out[mid]);  axes[2].set_title(f"(c) k-means mask, IC = {100 * MK.IC[mid]:.2f} %")
fig.suptitle(f"Figure 9.7 (procedure) — frame {mid + 1}\n{VIDEO_LABEL}")
plt.tight_layout(); plt.show()

print(f"Otsu   IC range {100 * MO.IC.min():.2f} .. {100 * MO.IC.max():.2f} %")
print(f"kmeans IC range {100 * MK.IC.min():.2f} .. {100 * MK.IC.max():.2f} %")
if not HAVE_TANK_VIDEO:
    tier3("both curves above are the stand-in's")
    not_reproduced("Fig. 9.7 at t = 816 s (N9)", "Otsu IC 87.26 %, threshold 100; k-means IC 86.91 %",
                   "dypic_05100_cam1_top.avi does not ship and is unobtainable")
    not_reproduced("run 5100 saturation (N10 / N12)", "approx. 89 % after approx. 200 s; band 80-89 % across runs",
                   "same missing video")
    not_reproduced("Table 9.3 (N11)", f"start times / average IC {TABLE_9_3}", "same missing video")
''')

md(r"""
### Trap E4 — `if I(i,j) >= t*255` compares the pixel with the whole *growing* threshold vector

`movie_otsu.m` line 33 writes `t(k) = graythresh(...)`, so after *k* frames `t` is a **1 × k vector**. Line 39
then says

```matlab
if I(k).cdata(i,j) >= t*255            % t is the WHOLE vector, not t(k)
```

A MATLAB `if` on an array means `all(...)`, so the pixel must exceed **every threshold seen so far**: the
effective threshold is the **running maximum** $\max_{j \le k} t(j)$, not $t(k)$. R2025a confirms it directly:
`uint8(50) >= [25 51]` → `[1 0]` and the `if` is *not* taken. The consequence is real and measurable — the
plotted IC is not the IC of the mask the script writes into `otsu.avi`.
""")
code(r'''
CORRECTED = movie_otsu(FRAMES, running_max_bug=False)       # the only change: `t(k)` instead of `max(t(1:k))`
d = 100 * (CORRECTED.IC - MO.IC)
changed = int(np.count_nonzero(np.abs(d) > 1e-12))
print(f"frames whose IC changes when E4 is switched off: {changed} of {len(d)}")
print(f"largest difference: {np.abs(d).max():.3f} percentage points at frame {int(np.argmax(np.abs(d))) + 1}")

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
axes[0].plot(k_axis, 255 * MO.t, "r.-", label="t(k)")
axes[0].plot(k_axis, 255 * MO.effective_level, "k.--", label="running max = what E4 uses")
axes[0].set_title("E4 — the effective threshold is a running maximum")
axes[0].set_xlabel("frame"); axes[0].set_ylabel("gray level"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(k_axis, 100 * MO.IC, "b.-", label="as shipped (E4 live)")
axes[1].plot(k_axis, 100 * CORRECTED.IC, "c.--", label="running_max_bug=False")
axes[1].set_title(f"consequence: up to {np.abs(d).max():.2f} pp of ice concentration")
axes[1].set_xlabel("frame"); axes[1].set_ylabel("IC [%]"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("\nOnce the threshold never rises again the two agree; the damage is done while the maximum is stale.")
print("A video whose threshold only increases would hide this entirely — which is why the stand-in is built "
      "with a threshold that rises and then falls.")
''')

md(r"""
### Trap E5 — `n = n+1` increments *every* element of the counter

Line 40 of the same loop is `n = n+1`, not `n(k) = n(k)+1`. `n` is a growing vector, so each frame increments
**all** of its entries. The closed form is

$$n_{\text{final}}[j] = \sum_{m \ge j} \text{count}_m ,$$

i.e. entry *j* accumulates every later frame's count as well. MATLAB R2025a on three frames with counts
(7, 6, 4) returns `n = [17 10 4]`. Crucially, `IC(k)` **escapes E5** — it reads `n(k)` *inside* the same
iteration, before the later frames inflate it — but it does **not** escape E4. And separately, the AVI the
script writes uses `im2bw(I(k).cdata, t(k))`, i.e. strict `>` at the *per-frame* level, so the written video and
the plotted curve legitimately disagree in both rule and value.
""")
code(r'''
print("frame :", k_axis[:8].tolist(), "...")
print("counts:", MO.counts[:8].astype(int).tolist(), "...   <- what the loop counted in that frame (E5-free)")
print("n     :", MO.n[:8].astype(int).tolist(), "...   <- what the script's `n` actually holds (E5 live)")

closed_form = np.array([MO.counts[j:].sum() for j in range(len(MO.counts))])
print(f"\nclosed form  n[j] = sum_(m>=j) count_m  reproduces it exactly: {np.array_equal(closed_form, MO.n)}")
print(f"n[0] is inflated from {int(MO.counts[0])} to {int(MO.n[0])}; only the LAST entry is undamaged.")

ic_from_n = MO.n / MO.denominator
print(f"\nIC computed from the corrupted n would be wrong by up to "
      f"{100 * np.abs(ic_from_n - MO.IC).max():.2f} pp — but the script reads n(k) inside the loop, so the "
      f"plotted IC escapes E5: {np.allclose(MO.IC, MO.counts / MO.denominator)}")

avi_counts = np.array([int(m.sum()) for m in MO.bw])        # the masks that go into otsu.avi
print(f"\nAVI vs curve: the written mask uses im2bw(., t(k)) = strict '>' at the per-frame level, the plotted IC "
      f"uses '>=' at the running maximum.\n    mask counts {avi_counts[:6].tolist()} ...\n"
      f"    IC numerator {MO.counts[:6].astype(int).tolist()} ...   equal on "
      f"{int(np.count_nonzero(avi_counts == MO.counts))} of {len(avi_counts)} frames")
''')

md(r"""
`movie2avi` was removed in R2025a; `core.video.write_video(path, frames, fps)` replaces it and writes an
Uncompressed AVI that is **bit-identical** to MATLAB's `VideoWriter(..., 'Uncompressed AVI')`. The cell below
writes `otsu.avi` exactly as line 65 does (12 fps) and reads it back to prove the round trip, then deletes it.
""")
code(r'''
import tempfile

masks_rgb = np.stack([np.repeat((255 * m.astype(np.uint8))[:, :, None], 3, axis=2) for m in MO.bw])
with tempfile.TemporaryDirectory() as tmp:
    out = write_video(Path(tmp) / "otsu.avi", masks_rgb, 12.0)     # = movie2avi(M, 'otsu.avi', 'FPS', 12)
    back = read_video(out)
    print(f"wrote {out.name}: {out.stat().st_size / 1e6:.2f} MB, {video_num_frames(out)} frames at 12 fps")
    print(f"read back as {back.shape} (H, W, 3, N); round trip differs on "
          f"{int(np.count_nonzero(back != np.moveaxis(masks_rgb, 0, -1)))} bytes")
''')

# =====================================================================================================================
# 9.3 Ice floe identification
# =====================================================================================================================
md(r"""
## 9.3 Ice floe identification (pp. 201–211)

Chapter 6's GVF-snake pipeline is applied to model ice. The driver is `model_ice_demo.m`, whose parameter block
differs from ch06's `sea_ice_demo.m` in six values (printed by the imports cell) — most importantly
**`timer = 2`**.

### 9.3.1 Contour initialization for crowded rectangular floes (pp. 204–205)

When aligned square floes touch, binarization leaves **no hole** between them: the distance transform's minima
merge into one seed and the snake cannot split the pair (Fig. 9.11(a)–(c)). The book's remedy is to run contour
initialization **and** segmentation **again** on the result (Fig. 9.11(d)(e)) — that is all `timer = 2` means.
A floe counts as well segmented when

1. its area is **less** than a threshold ($R_a = 1000$),
2. its convexity — floe area ÷ minimum-area bounding polygon area — is **larger** than a threshold ($R_c = 0.9$),
3. the **length/width ratio** of its minimum-area bounding rectangle is **less** than a threshold ($R_l = 2$);

and the components that fail are the ones re-seeded on the next pass. This is **Algorithm 7** (p. 206), the
chapter's only numbered algorithm.
""")
code(r'''
t0 = time.time()
D = model_ice_demo(I, full=True)                  # lines 9-51 + the commented lines 53-66 (`full=True`)
print(f"model_ice_demo on {I.shape[0]} x {I.shape[1]} px: {time.time() - t0:.1f} s, "
      f"{D.gvf.n_seeds} seeds over {len(D.gvf.passes)} passes (timer = {BOOK_PARAMS_CH9['timer']})")
for rec in D.gvf.passes:
    print(f"    pass {rec.time}: {rec.num} components, {len(rec.k)} failed the criteria, "
          f"{len(rec.seeds)} seeds deformed")

L_before = label_components(D.gvf.bw, 4)          # the binarization, before any snake
L_after = label_components(D.bw4, 4)
fig, axes = plt.subplots(1, 4, figsize=(12, 6))
imshow_matlab(axes[0], I);           axes[0].set_title("(a) input\nmodel_ice.jpg", fontsize=9)
imshow_matlab(axes[1], D.gvf.bw);    axes[1].set_title(f"(b) im2bw, level {D.gvf.level:.3f}\n"
                                                      f"{int(L_before.max())} components", fontsize=9)
axes[2].imshow(label2rgb(L_before)); axes[2].set_axis_off()
axes[2].set_title("(c) the crowding problem:\ntouching squares = one blob", fontsize=9)
axes[3].imshow(label2rgb(L_after));  axes[3].set_axis_off()
axes[3].set_title(f"(d)(e) after {BOOK_PARAMS_CH9['timer']} passes\n{int(L_after.max())} floes", fontsize=9)
fig.suptitle("Figure 9.11 (procedure) — contour initialization for crowded rectangular floes")
plt.tight_layout(); plt.show()

print(f"\n{int(L_before.max())} connected components before segmentation, {int(L_after.max())} floes after — "
      "the aligned squares really are one blob until the snakes cut them apart.")
print("This is a PROCEDURAL reproduction: model_ice.jpg is not the printed Fig. 9.11(a), which is a landscape "
      "crop of a different part of the same tank.")
''')

md(r"""
### Gap G1 — Algorithm 7's convergence test is never implemented

Lines 2, 6, 7 and 19 of the printed Algorithm 7 keep a floe count $N_0$, compare it with the new count $N_1$ and
stop when they are equal. `GVF_distance.m` line 80 *computes* `num` — and never compares it: the real stop rule
in the shipped code is "no component fails the criteria", plus the `timer` countdown. We expose the book's rule
as an opt-in `gvf_distance(stop='count')` (default unchanged, so no ch06 number moves); it is **ours**, labelled
`reimplemented`, and it is shown here only to make the gap visible.
""")
code(r'''
lab, num, area, solidity, major, minor, rl, k = component_criteria(
    D.bw4, BOOK_PARAMS_CH9["Ra"], BOOK_PARAMS_CH9["Rc"], BOOK_PARAMS_CH9["Rl"], 4)
print(f"{num} floes; the criteria of p. 205 with Ra={BOOK_PARAMS_CH9['Ra']}, Rc={BOOK_PARAMS_CH9['Rc']}, "
      f"Rl={BOOK_PARAMS_CH9['Rl']} flag {len(k)} of them as not-well-segmented")
print(f"    area     : {area.min():.0f} .. {area.max():.0f} px")
print(f"    solidity : {solidity.min():.3f} .. {solidity.max():.3f}")
print(f"    l/w      : {rl.min():.2f} .. {rl.max():.2f}")
print("\nNOTE (text vs code): p. 205 defines criterion 3 on the minimum-area bounding RECTANGLE, but "
      "GVF_distance.m lines 90-95 use the ELLIPSE axis ratio (MajorAxisLength/MinorAxisLength).")
_, _, _, _, mj2, mn2, rl2, k2 = component_criteria(D.bw4, BOOK_PARAMS_CH9["Ra"], BOOK_PARAMS_CH9["Rc"],
                                                   BOOK_PARAMS_CH9["Rl"], 4, ratio="minrect")
print(f"    ellipse ratio flags {len(k)} floes, the book's min-rectangle ratio flags {len(k2)}; they disagree on "
      f"{len(set(k.tolist()) ^ set(k2.tolist()))} floe(s), and the ratios themselves differ by up to "
      f"{np.nanmax(np.abs(rl - rl2)):.2f}.")
print("    `ratio='minrect'` is OURS (the book's text, no MATLAB reference); 'ellipse' stays the default "
      "everywhere, so no ch06/ch07 number moves.")
''')

md(r"""
### 9.3.2 Ice floe identification for the overall tank image (pp. 205–207)

One GVF parameter set cannot serve the whole 5 : 1 tank strip, so the book splits it into **20 overlapping
sub-images**, runs Algorithm 7 in each with its own GVF iteration count (**Table 9.4**), stitches the results
and finishes with Chapter 7's ice-shape enhancement to fill noise holes (Figs. 9.12, 9.13).

**No `.m` file ships for this.** The book gives no grid shape, no overlap width and no stitch rule, so
`tiled_segmentation(I, grid, overlap, gvf_iters=TABLE_9_4)` is **ours** — only the 20 iteration counts are the
book's, and they are asserted as a transcription, nothing more. The cell below therefore runs a deliberately
small version (a downscaled 1 × 2 grid, capped seeds) purely to show the mechanism; nothing it prints may be
compared with Fig. 9.12.
""")
code(r'''
print("Table 9.4 (p. 207), the 20 per-sub-image GVF iteration counts — transcription only:")
print("   ", list(TABLE_9_4))

SCALE = 2                       # downscale factor, purely a runtime guard (see the ch06 lesson)
MAX_SEEDS = 30                  # cap on contours per pass, likewise
demo_tile = TANK[:TANK.shape[0] // 2, :TANK.shape[1] // 3][::SCALE, ::SCALE]
t0 = time.time()
bw_tiles, tile_res = tiled_segmentation(demo_tile, grid=(1, 2), overlap=8, gvf_iters=TABLE_9_4[:2],
                                        max_seeds=MAX_SEEDS)
print(f"\n1 x 2 grid on a {demo_tile.shape[0]} x {demo_tile.shape[1]} crop (SCALE={SCALE}, MAX_SEEDS={MAX_SEEDS}): "
      f"{time.time() - t0:.1f} s, Num = {TABLE_9_4[0]} and {TABLE_9_4[1]} per tile")

Lt = label_components(bwareaopen(bw_tiles, 20), 4)
fig, axes = plt.subplots(3, 1, figsize=(13, 5))
imshow_matlab(axes[0], demo_tile);  axes[0].set_title("input crop of the tank image", fontsize=9)
imshow_matlab(axes[1], bw_tiles);   axes[1].set_title("stitched segmentation (OR over overlapping tiles)", fontsize=9)
axes[2].imshow(label2rgb(Lt)); axes[2].set_axis_off()
axes[2].set_title(f"{int(Lt.max())} identified floes", fontsize=9)
fig.suptitle("Figure 9.12 (mechanism only) — tiled Algorithm 7; the grid, the overlap and the OR stitch are OURS")
plt.tight_layout(); plt.show()

print("\nDEVIATION: the grid shape, the overlap width and the OR stitch are not in the book. This figure "
      "demonstrates the idea of Table 9.4; it is NOT a reproduction of Fig. 9.12.")
if not HAVE_TANK_IMAGE:
    tier3("the crop above is the stand-in tank image")
''')

md(r"""
#### R7 — the two printed colour-bar tick lists come from **one** colour range

Fig. 9.12's colour bar prints **7** ticks ending 5878; Figs. 9.13 / 9.16(a) / 9.16(b) print **9** ticks ending
5952 — for data the text calls the same. That looked like an erratum and is not. Chapter 7's
`ice_shape_enhancement.m` builds ticks as `ysh = min : fix((max-min)/n) : max`, with `n = 6` on the map (line
196) and `nn = 8` on the histogram (line 227). Because the step is truncated, the colon **does not reach** the
maximum, and it stops short by a different amount in each case. All 16 printed integers come out of the single
Eq. (7.6) colour range $[198, 9974]$ — reproduced exactly below (book numbers **N14 + N15**).

The honest part: the tick *labels* are not floe areas. Colour 198 pins the smallest floe to exactly 20 px
(= `Ra_min`), but four colour maxima {9974, 9975, 9976, 9977} reproduce both printed lists, which pins the
largest floe only to the band **[5953, 6119] px**.
""")
code(r'''
colours = np.array([198.0, 9974.0])                      # one Eq. (7.6) colour range
v6, lab6 = colorbar_area_ticks(colours, 6)               # ice_shape_enhancement.m line 196 (the map)
v8, lab8 = colorbar_area_ticks(colours, 8)               # line 227 (the histogram)
print("n = 6  ->", list(map(int, lab6)), "   book Fig. 9.12       :", [20, 202, 424, 710, 1113, 1798, 5878])
print("nn = 8 ->", list(map(int, lab8)), "  book Figs. 9.13/9.16 :",
      [20, 153, 307, 488, 710, 996, 1398, 2081, 5952])
print(f"both REPRODUCED exactly (N14 + N15).  With n = 6 the colon stops at {int(v6[-1])}, short of the 9974 "
      f"maximum; with nn = 8 it lands on {int(v8[-1])} exactly — that difference in truncation is the whole of "
      "the apparent contradiction.")

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.plot(v6, lab6, "o-", label="n = 6  (Fig. 9.12 colour bar)")
ax.plot(v8, lab8, "s--", label="nn = 8 (Figs. 9.13 / 9.16)")
ax.set_xlabel("Eq. (7.6) colour value"); ax.set_ylabel("printed tick label [px]")
ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("R7 — both printed tick lists lie on ONE colour curve")
plt.show()
print("Pre-image caution (CUMULATIVE pitfall 60): 5878 and 5952 are tick LABELS. The colour maximum is only "
      "pinned to {9974...9977}, i.e. the largest floe to [5953, 6119] px.")
''')

md(r"""
### 9.3.2.1 Model sea ice floe modeling (pp. 207–208)

For ice–structure simulators every identified floe is replaced by its **minimum-area bounding rectangle**
(`rect.m` → `minboundrect`), and the resulting rectangles are rasterised and checked pairwise for overlap
(`model_ice_model.m`). Two details of `rect.m` must not be "tidied up": it calls `minboundrect(c, r, 'a')` with
**x = column, y = row**, and its `Area` / `Perimeter` are the **rectangle's**, not the component's pixel count —
a different meaning from ch07's `IcePiece.Area` and ch08's `Floe.Area`.

Rectangularization makes some floes overlap (Fig. 9.15(b)), so each record carries `Vertices`, `Center`, `Area`,
`Perimeter` **and** an `Intersection` index list.
""")
code(r'''
S = D.S                                                   # = rect(bw4), the commented line 54 of the demo
M = D.model                                               # = model_ice_model(S, bw4, 0.4, 2.5), line 66
print(f"{len(S)} minimum-area rectangles; areas {min(f.Area for f in S):.1f} .. {max(f.Area for f in S):.1f} px")
print(f"aspect ratios k = |v1-v2| / |v3-v2| : {M.ratios.min():.3f} .. {M.ratios.max():.3f}")
print(f"accepted by the band k1 < k < k2 = ({BOOK_PARAMS_CH9['k1']}, {BOOK_PARAMS_CH9['k2']}): "
      f"{len(M.accepted)} of {len(S)}")
n_overlap = sum(len(f.Intersection) for f in M.s_model)
print(f"overlap flags: {n_overlap} entries over {len(M.s_model)} modelled floes")

fig, axes = plt.subplots(1, 3, figsize=(11, 6))
imshow_matlab(axes[0], D.bw4); axes[0].set_title("identified floes (bw4)", fontsize=9)
imshow_matlab(axes[1], D.bw4)
for f in S:
    axes[1].plot(f.Vertices[:, 0] - 1, f.Vertices[:, 1] - 1, "y-", linewidth=0.8)
    axes[1].plot(f.Center[0] - 1, f.Center[1] - 1, "r+", markersize=4)
axes[1].set_title(f"(a) {len(S)} rectangles\n`rect.m` + `minboundrect`", fontsize=9)
imshow_matlab(axes[2], M.bw); axes[2].set_title(f"(b) model raster\n{len(M.s_model)} accepted", fontsize=9)
fig.suptitle("Figure 9.15 (procedure) — ice floe rectangularization and the floe model")
plt.tight_layout(); plt.show()
''')

md(r"""
#### Three ice concentrations of the same scene (p. 208)

The book compares **76.96 %** (the segmented image, whose boundary pixels became water), **83.17 %** (global
Otsu) and **87.75 %** (Σ rectangle areas ÷ image domain). The third is the largest *because overlapping
rectangles are counted twice*, which the authors say "compensates" the boundary loss — so the double counting is
deliberate and `rect_ice_concentration` does not de-duplicate. The identities are implemented and unit-tested;
the numbers below are this image's, not the book's.
""")
code(r'''
ic_seg = ice_concentration(D.bw4)
ic_thr = ice_concentration(D.gvf.bw)
ic_rect = rect_ice_concentration(M, D.bw4.shape)
ic_rect_all = rect_ice_concentration(S, D.bw4.shape)     # every rectangle, accepted or not
print(f"segmentation IC (boundaries are water) : {100 * ic_seg:6.2f} %")
print(f"global Otsu IC                         : {100 * ic_thr:6.2f} %")
print(f"sum of ACCEPTED rectangle areas        : {100 * ic_rect:6.2f} %")
print(f"sum of ALL {len(S)} rectangle areas          : {100 * ic_rect_all:6.2f} %   (overlaps counted twice)")
order_ok = ic_seg < ic_thr < ic_rect_all
print(f"\nThe ORDERING the book describes on p. 208 — segmentation < thresholding < rectangles — holds here too "
      f"when every rectangle is counted: {order_ok}.")
print(f"After the aspect band removes {len(S) - len(M.accepted)} of the {len(S)} rectangles the sum can fall below "
      "the segmentation figure: it is the FILTER, not the rectangularization, that decides.")
not_reproduced("the three ice concentrations of p. 208 (N18)", "76.96 % / 83.17 % / 87.75 %",
               "they belong to the overall tank image, which does not ship; nothing was tuned to approach them")
''')

md(r"""
#### Floe size distribution and the Fig. 9.16(b) error

Fig. 9.16(a) is the FSD of the rectangles and Fig. 9.16(b) the bin-wise difference against the FSD of the
identified floes (Fig. 9.13). `fsd_error` refuses to subtract two histograms that were not computed on the same
bin **centres** — MATLAB's `hist` takes centres, `numpy.histogram` takes edges, and mixing them silently is the
classic way to get a plausible wrong picture.
""")
code(r'''
areas_identified = np.array([s.Area for s in regionprops(label_components(D.bw4, 4), "basic")])
areas_rect = np.array([f.Area for f in M.s_model])
centres = np.linspace(0, max(areas_identified.max(), areas_rect.max()), 10)
z_id, x_id = floe_size_histogram(areas_identified, centres)
z_rc, _ = floe_size_histogram(areas_rect, centres)
err = fsd_error(z_rc, z_id)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
w = 0.4 * (centres[1] - centres[0])
axes[0].bar(x_id - w / 2, z_id, width=w, label="identified floes (pixel area)")
axes[0].bar(x_id + w / 2, z_rc, width=w, label="rectangles (rectangle area)")
axes[0].set_xlabel("floe size [px]"); axes[0].set_ylabel("frequency"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_title("Figures 9.13 / 9.16(a) (procedure) — FSD")
axes[1].bar(x_id, err, width=2 * w, color="firebrick")
axes[1].axhline(0, color="k", linewidth=0.8)
axes[1].set_xlabel("floe size [px]"); axes[1].set_ylabel("rectangles − identified")
axes[1].set_title("Figure 9.16(b) (procedure) — bin-wise error"); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"{len(areas_identified)} floes vs {len(areas_rect)} rectangles; error per bin: {err.astype(int).tolist()}")
print("The book's axes (0-6000 px, frequency 0-300, error -60..+60) need the tank image's hundreds of floes; "
      "this scene has tens.  Shape reproduced, numbers not.")
''')

md(r"""
### Four traps in the §9.3.2.1 code, each runnable

* **E7** — p. 207 says rectangles with a length-to-width ratio *less than* a threshold are removed. The code is
  `if k < k2 && k > k1` with `k` **not** normalised to ≥ 1, i.e. a symmetric band $(0.4, 2.5)$ with **strict**
  inequalities: both extremes are removed and both boundary values are rejected.
* **E8** — `rect.m`'s `if (nargin < 3)` sits in a **two**-argument function, so it is always true: `metric` is
  dead code and line 52 hard-codes `'a'`. MATLAB therefore accepts a bogus metric silently; our port validates
  it and raises (deviation **D2**), which cannot change any result.
* **E9** — `if xx ~= NaN` is `if ~isempty(xx)`. Chapter 8 met the same idiom with `polyxpoly`, where it made
  containment *undetectable*; here the call is `polybool('intersection', …)`, which returns the intersection
  **region**, so containment **is** detected. The one rule that must be copied exactly is that
  **edge-touching counts as no overlap** — a naive clipper returns a degenerate zero-area polygon and gets it
  wrong (our `clip_polygon_convex(drop_degenerate=True)` matched MATLAB on 10 of 10 geometries; with the flag
  off it is wrong on 3 of 10).
* **D3** — MATLAB's `rect.m` **cannot run** on a collinear component: `minboundrect` calls `convhull`, and
  R2025a raises `MATLAB:convhull:EmptyConvhull2DErrId` ("the points may be collinear"), failing the whole call.
  Our port returns a degenerate rectangle instead. No shipped driver can hit it (`bw4` has been through
  `bwareaopen(·, 20)`), but a reader feeding `rect` a thin line should know.
""")
code(r'''
from seaice.core.polygon import clip_polygon_convex

# --- E7: the band is symmetric and strict
band = np.array([0.39, 0.40, 0.50, 1.00, 2.40, 2.50, 3.00])
keep = (band > BOOK_PARAMS_CH9["k1"]) & (band < BOOK_PARAMS_CH9["k2"])
print("E7  ratios       :", band.tolist())
print("    kept (k1<k<k2):", band[keep].tolist(), " <- both boundary values 0.40 and 2.50 are REJECTED "
      "(MATLAB agrees, measured)")

# --- E8: `metric` is dead in MATLAB; we validate it
try:
    rect(D.bw4, "not-a-metric")
    print("E8  our port accepted a bogus metric (unexpected)")
except ValueError as e:
    print(f"E8  MATLAB ignores `metric` silently (nargin<3 is always true); our port raises: {e}")
print(f"    rect(bw,'p') == rect(bw): "
      f"{np.allclose(rect(D.bw4, 'p')[0].Vertices, rect(D.bw4)[0].Vertices)}  (both are forced to 'a')")

# --- E9: containment vs edge-touching, the two cases `polybool` decides
outer = (np.array([0.0, 10.0, 10.0, 0.0]), np.array([0.0, 0.0, 10.0, 10.0]))
inner = (np.array([2.0, 8.0, 8.0, 2.0]), np.array([2.0, 2.0, 8.0, 8.0]))
touch = (np.array([10.0, 20.0, 20.0, 10.0]), np.array([0.0, 0.0, 10.0, 10.0]))
for name, other in (("contained", inner), ("edge-touching", touch)):
    x, y = clip_polygon_convex(other[0], other[1], outer[0], outer[1], drop_degenerate=True)
    print(f"E9  {name:>14}: intersection has {len(x)} vertices -> overlap flagged: {len(x) > 0}")
print("    (ch08's polyxpoly version would say 'no overlap' for the contained case — same idiom, opposite effect)")

# --- D3: a collinear component
line = np.zeros((12, 12), dtype=bool); line[5, 2:9] = True
rs = rect(line)
print(f"D3  a 1-px line: our rect returns {len(rs)} rectangle(s), area {rs[0].Area:.1f} (degenerate); "
      "MATLAB raises MATLAB:convhull:EmptyConvhull2DErrId and the whole call fails")
''')

# =====================================================================================================================
# 9.3.3 maximum floe size from video
# =====================================================================================================================
md(r"""
### 9.3.3 Ice floe identification for model sea ice video: monitoring maximum floe size (pp. 208–211)

The use case is risk management: warn when the **largest floe** approaching the protected vessel becomes too
big. `movie_floe.m` is 31 lines — for every frame, `im2bw` → `bwareaopen(·, 20, 4)` → `bwlabel(·, 4)` →
`regionprops(·, 'basic')` → `max(ice_areas)` — and that per-frame maximum, in **pixels**, is the *y* value of
Fig. 9.18. There is no plotting code at all; the printed figure was made interactively.

Note what line 18 implies: `im2bw` is called with **no level**, so the default 0.5 (i.e. `> 127.5` after
`rgb2gray`) is used — which only makes sense if the AVI already holds **segmented** frames. The per-frame
Algorithm-7 segmentation the section describes has **no shipped code** at all (gap **G2**).
""")
code(r'''
if HAVE_FLOE_VIDEO:
    SEG = read_video(FLOE_VIDEO)
    SEG_LABEL = f"the book's own {FLOE_VIDEO.name}"
else:
    seg_path, SEG = ensure_synthetic_segmented_video(n_frames=40)
    SEG_LABEL = "Tier-3 synthetic stand-in for 05100.avi (already-segmented frames)"
    tier3("the segmented floe video is generated by seaice.core.synth.segmented_floe_video")
    print(f"    file: {seg_path.as_posix()}   frames {SEG.shape}")

t0 = time.time()
MF = movie_floe(SEG, keep_labels=True)                 # = movie_floe.m lines 15-27
print(f"{len(MF.floe)} frames in {time.time() - t0:.1f} s;  max floe area "
      f"{MF.floe.min():.0f} .. {MF.floe.max():.0f} px")

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
axes[0].imshow(label2rgb(MF.labels[4])); axes[0].set_axis_off()
axes[0].set_title(f"Figure 9.17(b) (procedure) — frame 5, {int(MF.labels[4].max())} 4-connected components")
axes[1].plot(np.arange(1, len(MF.floe) + 1), MF.floe, "b.-")
axes[1].set_xlabel("Time [s]  (frame index at 1 fps)"); axes[1].set_ylabel("maximum floe area [px]")
axes[1].grid(alpha=0.3)
axes[1].set_title("Figure 9.18 (procedure) — maximum floe size against time")
plt.tight_layout(); plt.show()

print(f"per-frame component counts: {[len(a) for a in MF.areas[:8]]} ...")
if not HAVE_FLOE_VIDEO:
    tier3("the time series above is the stand-in's")
    not_reproduced("Fig. 9.18 axes (N19)", "time 0-1000 s, maximum floe area 0-3e4 px",
                   "05100.avi does not ship and is unobtainable")
''')

md(r"""
#### R13 — a blank frame does **not** delete an element; MATLAB raises

`max([])` returns `[]`, so `floe(k) = max(ice_areas)` on a frame with no surviving component assigns an empty
right-hand side. The analysis predicted MATLAB's delete-and-shift semantics; the verification measured
something else. Deletion is what `x(k) = []` does **for `k ≤ numel(x)`**, and this loop appends one element per
frame, so `k` is always one *past* the end: R2025a raises
`MATLAB:matrix:singleSubscriptNumelMismatch`. Our `movie_floe(..., empty=…)` makes the choice explicit —
`'raise'` (the default, = MATLAB), `'delete'` (the semantics people expect, which shifts every later index) or
`'nan'` (keeps the frame axis aligned).
""")
code(r'''
_, SEG_BLANK = ensure_synthetic_segmented_video(Path(tempfile.gettempdir()) / "ch09_blank_demo.avi",
                                                n_frames=8, blank_frame=3, regenerate=True)
for mode in ("raise", "nan", "delete"):
    try:
        res = movie_floe(SEG_BLANK, empty=mode)
        print(f"empty={mode:<7}: {len(res.floe)} values for 8 frames -> {np.round(res.floe, 0).tolist()}")
    except ValueError as e:
        print(f"empty={mode:<7}: raises — {str(e).splitlines()[0]}")
print("\n'delete' shortens the vector, so every later frame index silently shifts left; that is why the port "
      "refuses by default instead of guessing.")
''')

md(r"""
#### Gap G2 — the per-frame segmentation §9.3.3 describes has no code

`movie_floe.m` starts from an AVI that is already binary. To close the loop, `segment_video` runs Algorithm 7
on each frame and hands the result to `movie_floe` — but **that wiring is ours, not the book's**: it carries a
`# DEVIATION` marker in the source, appears in no parity assertion, and must never be compared with Fig. 9.17
or Fig. 9.18. It is also slow (one GVF field plus one snake per seed per frame), so the demonstration below uses
two frames at a quarter scale with capped seeds.
""")
code(r'''
N_DEMO, DOWNSCALE, SEEDS = 2, 4, 20                       # runtime guards, not book parameters
t0 = time.time()
seg_frames = segment_video([FRAMES[:, :, :, k] for k in range(N_DEMO)], downscale=DOWNSCALE, max_seeds=SEEDS)
print(f"segment_video on {N_DEMO} frames at 1/{DOWNSCALE} scale, max {SEEDS} seeds: {time.time() - t0:.1f} s "
      f"-> {seg_frames.shape}")

rendered = np.stack([np.repeat((255 * f.astype(np.uint8))[:, :, None], 3, axis=2) for f in seg_frames])
MF2 = movie_floe(rendered, empty="nan")
fig, axes = plt.subplots(1, N_DEMO + 1, figsize=(12, 3.4))
imshow_matlab(axes[0], FRAMES[:, :, :, 0][::DOWNSCALE, ::DOWNSCALE]); axes[0].set_title("raw frame 1", fontsize=9)
for j in range(N_DEMO):
    imshow_matlab(axes[j + 1], seg_frames[j]); axes[j + 1].set_title(f"Algorithm 7, frame {j + 1}", fontsize=9)
fig.suptitle("Figure 9.17 (mechanism only) — per-frame segmentation; THIS WIRING IS OURS (gap G2)")
plt.tight_layout(); plt.show()
print(f"maximum floe area per segmented frame: {np.round(MF2.floe, 0).tolist()} px  — ours, not the book's")
''')

# =====================================================================================================================
# Parameter play
# =====================================================================================================================
md(r"""
## Parameter play (optional)

The aspect band $(k_1, k_2)$ of `model_ice_model.m` is the chapter's most consequential free parameter: it
decides which rectangles enter the simulator model, and therefore the Σ-rectangle ice concentration of p. 208.
The book's own values are $k_1 = 0.4$, $k_2 = 2.5$ (line 65 of `model_ice_demo.m`, in the commented block —
they appear nowhere in the text). Re-running is cheap because the rectangles are already computed. Requires
`ipywidgets`; without it only the static preview runs.
""")
code(r'''
def play(k1=0.4, k2=2.5):
    m = model_ice_model(S, D.bw4, float(k1), float(k2))
    print(f"k1 = {k1:.2f}, k2 = {k2:.2f}: {len(m.accepted)} of {len(S)} rectangles accepted, "
          f"sum of areas / image = {100 * rect_ice_concentration(m, D.bw4.shape):.2f} %, "
          f"{sum(len(f.Intersection) for f in m.s_model)} overlap flags")
    fig, ax = plt.subplots(figsize=(3.2, 6))
    imshow_matlab(ax, m.bw)
    ax.set_title(f"model raster, k in ({k1:.2f}, {k2:.2f})", fontsize=9)
    plt.show()

play()                                                    # the script's own setting
try:
    from ipywidgets import FloatSlider, interact
    interact(play, k1=FloatSlider(0.4, min=0.1, max=1.0, step=0.05),
             k2=FloatSlider(2.5, min=1.0, max=5.0, step=0.1))
except ImportError:
    print("ipywidgets not installed — only the static preview above is shown")
''')

# =====================================================================================================================
# MATLAB <-> Python map + summary
# =====================================================================================================================
md(r"""
## MATLAB ↔ Python mapping used in this chapter

| MATLAB | Python (`seaice`) | Parity | Note |
|---|---|---|---|
| `VideoReader` + `read(v)` | `core.video.read_video` | exact on Uncompressed AVI | returns MATLAB's `(H, W, 3, N)`; 0 differing bytes of 22 118 400 |
| `get(v, 'numberOfFrames')` | `core.video.video_num_frames` | exact | still a valid alias in R2025a |
| `movie2avi(M, f, 'FPS', 12)` | `core.video.write_video(..., codec='rawvideo')` | exact | `movie2avi` was **removed**; bit-identical to `VideoWriter('Uncompressed AVI')` |
| `mmreader` | `core.video.read_video` | — | **removed** in R2025a; the single reference patch in `movie_floe.m` |
| `graythresh` / `im2bw` | `core.threshold.graythresh` / `im2bw` | exact | inherited from ch03 |
| `im2bw(RGB)` with no level | `core.threshold.im2bw(frame)` | exact | default 0.5 ⇒ `> 127.5` after `rgb2gray` |
| block loop of `local_Otsu.m` / `block_threshold.m` | `core.threshold.block_otsu(compare='gt' \| 'ge')` | exact | `'ge'` is ch9's line 26 — the one numeric difference from ch3 |
| `kmeans(X, 2, 'EmptyAction', 'singleton')` (Statistics TB) | `core.clustering.kmeans_lloyd(init='kmeans++')` | approx | partition, centres and IC agree; the **label numbers** do not (RNG) |
| the book's own `ch3/kmeans.m` | `core.clustering.kmeans_gray` | exact | used where the *text* names k-means with no `.m` behind it |
| `bwlabel(·, 4)` / `bwareaopen(·, P, conn)` | `core.connectivity.label_components` / `bwareaopen` | exact | `bwareaopen` keeps components with area **≥ P** |
| `regionprops(L, 'basic')` | `core.regionprops.regionprops(L, 'basic')` | exact | `'basic'` = `Area`, `Centroid`, `BoundingBox` — MATLAB's own `fieldnames` |
| `minboundrect(c, r, 'a')` | `core.polygon.minboundrect` | exact | x = column, y = row; its `convhull(x,y,{'Qt'})` needs the ch06 one-literal patch in R2025a |
| `roipoly(double(bw), vx, vy)` | `core.polygon.roipoly` | exact | 0 px against MATLAB's own raster |
| `polybool('intersection', …)` (Mapping TB) | `core.polygon.clip_polygon_convex(drop_degenerate=True)` | reimplemented | **emptiness** matches on 10/10 geometries; vertex order is not reproducible and is not consumed |
| `GVF_distance.m` (Algorithm 7) | `ch06_gvf_snake.gvf_distance` | near | byte-identical `.m`; only ch9's parameters are new; residual 28 px of 13 756 |
| `imfill(bw, 8, 'holes')` | `core.morphology.imfill(bw, 'holes', conn=8)` | exact | inherited from ch02 |
| `hist(y, centres)` | `core.histogram.hist` | exact | bin **centres**, not edges |
| `max(ice_areas)` on an empty frame | `movie_floe(..., empty='raise')` | exact | MATLAB raises `MATLAB:matrix:singleSubscriptNumelMismatch` |
""")
md(r"""
## Summary — what is verified, what is not, and what Chapter 10 needs

**What Chapter 9 adds.** No new image processing: it *applies* Chapters 3, 6 and 7 to model-basin ice, and in
doing so contributes three reusable primitives — `seaice.core.video` (`read_video` / `write_video` /
`video_num_frames`, in MATLAB's `(H, W, 3, N)` order), `core.polygon.clip_polygon_convex` (`polybool`'s
emptiness verdict, now shared with `clip_polygon_rect`) and three `core.synth` tank/video generators — plus the
`compare=` switch on `block_otsu`, `regionprops(·, 'basic')`, `component_criteria(ratio=…)` and
`gvf_distance(stop=…)`.

**Parity against MATLAB R2025a** (36 labelled rows, `reports/ch09_verification.md`): **24 exact · 3 near ·
1 approx · 6 reimplemented** (+ 1 display-only and 3 Tier-3 fixtures). Eleven MATLAB sessions, 827 s of
reference time, all `status: ok`. The whole `movie_otsu.m` numeric loop — **both errata included** — matches
element for element; `movie_floe.m` matches on identical integers; `rect.m` and `model_ice_model.m` match to
1e-12 on MATLAB's own `bw4`, including all 34 overlap flags. `movie_kmeans.m` is `approx` only because the
Statistics Toolbox's `kmeans` has its own RNG: the partition, the cluster means (≤ 1.44e-13) and IC agree, the
label *numbers* do not. `tests/test_ch09.py` passes **48/48** and the full suite is **1985 passed / 2 skipped /
1 xfailed**, with no earlier chapter regressed and no tolerance loosened.

**What is reproduced as a book number.** **N4** (the Fig. 9.4 caption's 83.14 % / 84 as the unweighted block
means), **N8** (the IC denominator 221·397 − 94·113 = 77 115), **N21/R8** (Table 9.1's 45/40/15 % from the strip
widths — the table is correct and its columns are not reversed) and **N14 + N15/R7** (all 16 printed colour-bar
tick integers from a single Eq.-(7.6) colour range with `n = 6` and `nn = 8`). **N13** (Table 9.4) is a
transcription: its 20 values are inputs.

**What is permanently unverified, and why nothing was tuned to close it.** `04100_analyse.jpg`,
`dypic_05100_cam1_top.avi` and `05100.avi` were never published, so **N2** (83.17 %, threshold 84), **N3** (the
six block ICs and thresholds), **N5/N6** (Table 9.2), **N9** (87.26 % / 100 / 86.91 % at *t* = 816 s), **N10**
(≈89 % after ≈200 s), **N11** (Table 9.3), **N12** (the 80–89 % band), **N18** (76.96 / 83.17 / 87.75 %) and
**N19** (the Fig. 9.18 axes) cannot be computed by anyone, here or elsewhere. The Tier-3 stand-ins give MATLAB
parity — which is a real and useful result — and nothing more. `model_ice.jpg` is not a printed figure, so §9.3
has no plate to compare against either.

**Errata and gaps confirmed in MATLAB** (full list in the verification report): **E4** the running-maximum
threshold; **E5** the all-elements increment; **E5b** the AVI and the plotted curve disagreeing by construction;
**E6** the run-5400 output name on run-5100 input; **E7** the symmetric strict aspect band the text describes as
one-sided; **E8** the dead `metric`; **E9** `if xx ~= NaN`; **R13** the blank frame raising rather than
deleting. **G1** Algorithm 7's `N0 ≠ N1` stop is never implemented, **G2** §9.3.3's per-frame segmentation has
no shipped code, and the text/code disagreement on criterion 3 (bounding rectangle vs ellipse axes) remains.
Everything we wired into those gaps is opt-in, labelled `reimplemented`, and carries no parity claim.

**Feeds forward — Chapter 10 (Appendix A, geometric calibration).** It owns **both** rectifiers: the analytic
orthorectification of A.1.1 and the **linear four-corner method of A.1.2**, plus the radial lens-distortion
correction of A.2 that §9.2.2 explicitly deferred ("fish-eye distortion is judged negligible here; a rectifier
is in Appendix A"). It will reuse `core.interp` (ch02), `core.polygon`, `core.io` and — for any rectified video
demonstration — this chapter's `core.video`. When it lands, §8.1's shipborne procedure and §9.2.2's fish-eye
frames both become runnable end to end on any oblique frame a reader supplies.

The learned knowledge for this chapter is in `knowledge/ch09.md` and `knowledge/CUMULATIVE.md`; the evidence is
in `reports/ch09_verification.md`.
""")

out = HERE / NB_NAME
nbf.write(nb, out)
print(f"wrote {out} ({len(nb.cells)} cells)")
