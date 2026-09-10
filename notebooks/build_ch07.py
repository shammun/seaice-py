"""Build ``notebooks/ch07_ice_type_identification.ipynb`` with nbformat (re-runnable; never edited by hand).

Chapter 7 — Sea Ice Type Identification (Zhang & Skjetne 2018, pp. 145–174).  Follows the ``colab-notebook``
skill: title → cell 1 (Drive mount / cwd) → cell 2 (clone or pull the public repo) → cell 3 (``load_image``:
private copy or public-domain substitute) → one section per book section (7.1.1, 7.1.2, 7.1.3.1, 7.1.3.2, 7.1.4,
7.2.1–7.2.4, 7.3.1.1, 7.3.1.2, 7.3.2) → parameter play → MATLAB ↔ Python map → summary / what is and is not
verified / feeds forward.  All algorithms are imported from ``seaice``; the notebook only calls them and draws.

Run:  ``.venv/Scripts/python.exe notebooks/build_ch07.py``
Verify: ``.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900
        --output executed_ch07.ipynb notebooks/ch07_ice_type_identification.ipynb``  (then delete the copy).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

NB_NAME = "ch07_ice_type_identification.ipynb"
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
# Chapter 7 — Sea Ice Type Identification

*Sea Ice Image Processing with MATLAB* (Q. Zhang & R. Skjetne, CRC Press 2018), Chapter 7, pp. 145–174 —
Python port `seaice-py`, notebook `notebooks/ch07_ice_type_identification.ipynb`.

**Sections covered** (one notebook section each, in book order):
7.1.1 Morphological cleaning (Fig. 7.2) · 7.1.2 Connected-component extraction and labelling (**Eq. 7.1**,
Figs. 7.3/7.4) · 7.1.3.1 Hole filling by iterative dilation (**Eq. 7.2**, Figs. 7.5–7.7) · 7.1.3.2 Hole filling by
reconstruction (**Eqs. 7.3–7.4**, Fig. 7.8) · 7.1.4 The ice shape enhancement algorithm (**Algorithm 2**,
**Eqs. 7.5/7.6**) · 7.2.1 Sea ice pixel extraction · 7.2.2 Sea ice edge detection (**Algorithm 3**) ·
7.2.3 Sea ice shape enhancement (**Algorithm 4**) · 7.2.4 Ice type classification and the FSD (**Algorithm 5**) ·
7.3.1.1 Local (tiled) processing (**Algorithm 6**) · 7.3.1.2 Geometric calibration (deferred to Appendix A) ·
7.3.2 A preliminary sensitivity study.

Chapter 6 ended with a **binary segmentation** in which connected floes are cut apart by GVF snakes. Chapter 7
turns that segmentation into *identified objects*: it cleans and fills each piece (§7.1), classifies the pieces
into **ice floe / brash ice / slush / water** and measures the floe size distribution (§7.2), and then discusses
what to do when one image is too large or too distorted to be processed as a whole (§7.3). The whole chapter is a
**binary** operation — "since the output of the GVF snake segmentation method (Algorithm 1) is the segmented
binary image, the ice shape enhancement algorithm will be a binary operation" (p. 145).

**MATLAB files this notebook replaces** (`MATLAB_ROOT/ch7/`, 27 `.m` files; parity from
`reports/ch07_verification.md`, MATLAB R2025a running the original code as the reference):

| MATLAB file | Python (imported below) | Parity |
|---|---|---|
| `cleaning & labeling & filling/morphology_cleaning.m` (Fig. 7.2) | `seaice.ch07_ice_type.morphological_cleaning` | exact (every block 0 px vs MATLAB and vs the printed matrices) |
| `cleaning & labeling & filling/labeling.m` (Eq. 7.1, Figs. 7.3/7.4) | `ch07_ice_type.connected_component_extract` | exact vs the file; the equation itself is `reimplemented` |
| `cleaning & labeling & filling/filling.m` (Eq. 7.2, Fig. 7.5) | `ch07_ice_type.hole_fill_dilation` | exact vs the file; `reimplemented` from the equation |
| `cleaning & labeling & filling/filling_reconstruct.m` (Eqs. 7.3/7.4, Fig. 7.8) | `ch07_ice_type.border_marker`, `hole_fill_reconstruct`, `core.morphology.imfill` | exact |
| `Sea_Ice_Floe_Identification/ice_shape_enhancement.m` (**Algorithms 2/4/5**, Eqs. 7.5/7.6) | **`ch07_ice_type.ice_shape_enhancement`** | exact vs the M-file — 36 controlled fixture runs and the real image, all nine output arrays 0 px |
| `Sea_Ice_Floe_Identification/sea_ice_demo.m` (the driver) | `scripts/ch07_sea_ice_demo.py` | near (the residual is Chapter 6's k-means/snake, not this chapter's) |
| `Sea_Ice_Floe_Identification/seaice_kmean_GVF_forenhancement.m` (**Algorithm 3**) | `ch06_gvf_snake.seaice_kmean_gvf` via `ch07_ice_type.sea_ice_edge_detection` | near (byte-identical to Chapter 6's copy — reused, not re-ported) |
| `GVF.m`, `GVF_distance.m`, `snakedeform.m`, `snakeinterp.m`, `BoundMirror*.m`, `gaussian*.m`, `xconv2.m`, `gradient2.m`, `snakedisp.m`, `minboundrect.m`, `polygeom.m`, `homofil.m` | `core.snake.*`, `core.polygon.*`, `ch06_gvf_snake.*` | exact / near — **all 23 `.m` in `ch7/Sea_Ice_Floe_Identification/` are byte-identical to `ch6/`'s** |
| MATLAB `imfill(·,'holes')` on a **double** image, MATLAB `hist(y, n)` | **`core.morphology.imfill`**, **`core.histogram.hist`** (new in this chapter) | exact (40 comparisons) / near (16 of 17; only `-Inf` differs) |
| **Algorithm 6** (p. 168) + §7.3.1.1 tiling and stitching | `ch07_ice_type.tile_grid`, `local_segmentation` | reimplemented — **no MATLAB file exists** and the book fixes no tile size, overlap or stitching rule |
| §7.3.1.2 orthorectification of a *labelled* image | `ch07_ice_type.resample_categorical` (+ Appendix A / ch10) | resampling verified; the camera model is deferred |
| §7.3.2 sensitivity sweeps (Figs. 7.24/7.27) | `scripts/ch07_sensitivity.py` | **unverified by construction** — the book prints no counts |
| `sea_ice_model.m`, `color_hist*.m`, `SeaIce_Image_Structure.m` | deferred to Ch. 8 / Appendix B | — |

> **Data and honesty.** §7.1's Figs. 7.2–7.8 are **printed matrices**: they are transcribed in
> `seaice.core.synth` (`FIG_7_2_*` … `FIG_7_8_*`) and reproduce bit-for-bit for every reader, with or without the
> book's images. §7.2 and §7.3 need photographs. The book's own §7.1/§7.2/§7.3.1 source images
> (Figs. 7.1, 7.9, 7.10–7.21) are **not shipped anywhere in the book's code archive**, so the numbers attached to
> them — **154/189** and **2511/2624** pieces, the eight coverage percentages, the printed colour-bar tick lists —
> **cannot be reproduced and are not reproduced here**; they are labelled `unverified` in
> `reports/ch07_verification.md` (Open item 6) and nothing below pretends otherwise. What the book *does* ship is
> `sea_ice_test.jpg` (394 × 1038), which the analysis identified as **Fig. 7.22**, printed transposed. Every §7.2 /
> §7.3 figure below is that image, and every number printed from it is **our** number, not the book's.
> `seaice.core.io.load_image()` finds your private copy first and otherwise downloads a public-domain NASA MODIS
> scene of exactly the same size, so the notebook runs either way.
""")

# =====================================================================================================================
# 2. Setup cells (Colab / Drive / local) — copied verbatim from build_ch02.py
# =====================================================================================================================
md(r"""
## Setup (Google Colab or local Jupyter)

Run the two cells below first. On **Colab** the first cell mounts your Google Drive and moves into
`MyDrive/Sea_Ice_Colab` — the folder where your private copy of the book image lives
(`data/book/ch07/Sea_Ice_Floe_Identification/sea_ice_test.jpg`) and where downloads are cached between sessions;
readers without Drive get a temporary `/content/Sea_Ice_Colab`. The second cell clones (or updates) the public
repository `seaice-py` there and installs its requirements. **Locally** both cells are no-ops that move to the
repository root. No GPU is needed.

> ⚠️ **If you run this with your own copy of the book image, do not use *File → Save a copy in GitHub*.** That
> saves the cell outputs — figures rendered from the copyrighted book image — into the public repository. Save to
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
# 3. Data cell
# =====================================================================================================================
md(r"""
## Data

`seaice.core.io.load_image("ch07", "sea_ice_test.jpg")` replaces `imread('sea_ice_test.jpg')` at the top of
`sea_ice_demo.m`. It looks for **your private copy of the book image** first — `data/book/ch07/…` in the current
folder (on Colab: `MyDrive/Sea_Ice_Colab`), in the repository, or in Drive, searching the book's own
`Sea_Ice_Floe_Identification/` sub-folder — and otherwise downloads a **public-domain NASA substitute** of the
same 394 × 1038 size into `data/online/ch07/`. It prints which source it used and returns the label, so the cells
below quote image-dependent reference values only when the book's own image is loaded (the `book(…)` and
`matlab_ref(…)` helpers).

Chapter 7's copy of `sea_ice_test.jpg` is **not** Chapter 6's: it is the same photograph re-encoded (84.1 % of the
samples differ, mean |Δ| 2.93), so no Chapter 6 number measured on that file transfers, and none is quoted here.
Everything in §7.1 runs on the **printed matrices** of Figs. 7.2–7.8 instead of a photograph, so §7.1 is identical
for every reader.
""")
code(r'''
from seaice.core.io import load_image

I, SOURCE = load_image("ch07", "sea_ice_test.jpg")        # sea_ice_demo.m line 9; the book prints it as Fig. 7.22
FROM_BOOK = SOURCE.startswith("book")
book = (lambda s: f"  (book: {s})") if FROM_BOOK else (lambda s: "")        # book values only for the book image
matlab_ref = (lambda s: f"  [MATLAB R2025a on the book image: {s}]") if FROM_BOOK else (lambda s: "")
if not FROM_BOOK:
    print("Running on a public-domain substitute image; figures show the same operations, "
          "but values quoted in the book only hold for the book's own image.")
print(f"sea_ice_test.jpg: {I.shape} {I.dtype}   [{SOURCE}]")
print(f"MATLAB's size() order is rows x cols x 3; displayed the way the book prints Fig. 7.22 it is "
      f"{I.shape[1]} x {I.shape[0]}" + book("394 x 1038, p. 168"))
if I.max() == I.min():
    print("WARNING: the image is constant — the download most likely failed; re-run this cell.")
''')

md(r"""
## Imports

Everything algorithmic comes from the `seaice` package; the notebook only calls it and draws figures. Three
constants control the runtime: §7.2 runs the full pipeline at the image's **own resolution** (`SCALE = 1`), while
§7.3.1's tiled run and §7.3.2's parameter sweeps process every 3rd and every 4th row/column (`SCALE_LOCAL = 3`,
`SCALE_SWEEP = 4`) — they need 8 and 7 pipeline runs respectively, and the book prints no number for either that a
full-resolution run could be compared against. Each cell says which scale it used.
""")
code(r'''
import time

import numpy as np
import seaice.core                                      # imports seaice.core.plotting (sets the Agg backend)
from seaice.core import synth                           # the printed matrices of Figs. 7.2-7.8
from seaice.core.histogram import hist as matlab_hist   # MATLAB hist() semantics (bin CENTRES) - new in ch7
from seaice.core.morphology import imfill, strel        # MATLAB imfill() incl. the grayscale branch - new in ch7
from seaice.core.plotting import imshow_matlab, label2rgb, show_matrix, size_colorbar
from seaice.ch06_gvf_snake import BOOK_PARAMS
from seaice.ch07_ice_type import (adaptive_se_radius, border_marker, colorbar_area_ticks, color_to_area,
                                  connected_component_extract, hole_fill_dilation, hole_fill_reconstruct,
                                  ice_shape_enhancement, ice_types_classification, local_segmentation,
                                  morphological_cleaning, resample_categorical, sea_ice_edge_detection,
                                  size_color, tile_grid)

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams["figure.dpi"] = 100

P = BOOK_PARAMS["sea_ice_demo"]        # sea_ice_demo.m lines 9-46, incl. ch7's se_th = 50, min_floe = 40, min_brash = 1
SCALE, SCALE_LOCAL, SCALE_SWEEP = 1, 3, 4
ALG3 = dict(kms0=P["kms0"], sigma=P["sigma"], GradientOn=P["GradientOn"], GVFOn=P["GVFOn"], mu=P["mu"],
            alpha=P["alpha"], beta=P["beta"], gamma=P["gamma"], kappa=P["kappa"], Dmin=P["Dmin"], Dmax=P["Dmax"],
            Ra_min=P["Ra_min"], Ra=P["Ra"], Rc=P["Rc"], Rl=P["Rl"], se_radius=P["se_radius"], timer=P["timer"],
            keep_history=False)

def T(img):
    """Display transpose only: the book prints this scene as Fig. 7.22, i.e. 394 rows x 1038 columns."""
    return np.swapaxes(img, 0, 1) if img.ndim == 3 else img.T

print("ch7 parameter block:", ", ".join(f"{k} = {v}" for k, v in P.items()))
''')

# =====================================================================================================================
# 7.1 Ice shape enhancement
# =====================================================================================================================
md(r"""
## 7.1 Ice shape enhancement (pp. 145–153)

The segmentation that comes out of Chapter 6 is not yet a set of *objects*: floes contain holes where meltponds or
speckle were darker than the ice, and smaller pieces sit inside larger ones. §7.1 fixes both with four binary
tools, each of which the book demonstrates on a **printed matrix**. Those matrices are transcribed in
`seaice.core.synth`, so the whole of §7.1 reproduces exactly for every reader — no photograph is involved.
""")

md(r"""
### 7.1.1 Morphological cleaning (pp. 145–146, Fig. 7.2) — replaces `ch7/cleaning & labeling & filling/morphology_cleaning.m`

Cleaning is a **closing followed by an opening**, with the *same* structuring element. The closing
$A \bullet B = (A \oplus B) \ominus B$ "closes narrow cracks, fills long thin channels, and eliminates the holes
that are smaller than the structuring element"; the opening $A \circ B = (A \ominus B) \oplus B$ "breaks thin
connections between objects, removes small protrusions, and eliminates complete regions of an object that cannot
contain the structuring element" (p. 145). Fig. 7.2 runs both on a printed 13 × 23 matrix with a **2 × 2 square**
SE — an *even* element, whose origin is therefore off-centre, which is exactly why the result shifts the way the
book prints it. The M-file also spells the same result out as an explicit dilate → erode → erode → dilate chain
(`c1`…`c4`), which is in the code but not in the text.
""")
code(r'''
A72 = synth.FIG_7_2_IMAGE                                      # Fig. 7.2(a): the printed 13x23 matrix
c = morphological_cleaning(A72.astype(float), strel("square", 2))
print(f"Fig. 7.2(a) object pixels {int(A72.sum())}  ->  (b) closing {int(c.f1.sum())}, "
      f"(c) opening {int(c.f2.sum())}, (d) cleaning {int(c.f0.sum())}   (book: 111 -> 115 / 104 / 108)")
for tag, got, want in (("(b) closing", c.f1, synth.FIG_7_2_CLOSED), ("(c) opening", c.f2, synth.FIG_7_2_OPENED),
                       ("(d) cleaning", c.f0, synth.FIG_7_2_CLEANED)):
    print(f"  {tag:14s} vs the printed matrix: {int((np.asarray(got, bool) != want).sum())} px differ")
print(f"  the M-file's explicit chain: c2 == closing {np.array_equal(c.c2 != 0, c.f1 != 0)}, "
      f"c4 == cleaning {np.array_equal(c.c4 != 0, c.f0 != 0)}")

fig, axes = plt.subplots(4, 1, figsize=(11, 12))
for ax, m, t in ((axes[0], A72, "(a) A — the printed 13x23 image (111 px)"),
                 (axes[1], c.f1, "(b) closing A . B (115 px) — the bridge and the isolated pixel survive"),
                 (axes[2], c.f2, "(c) opening A o B (104 px) — the bridge is broken, the isolated pixel is gone"),
                 (axes[3], c.f0, "(d) cleaning = opening of the closing (108 px)")):
    show_matrix(ax, np.asarray(m, int), fontsize=5, title=t)
fig.suptitle("Figure 7.2  Morphological cleaning with a 2x2 square structuring element", y=0.92)
plt.show()
''')
md(r"""
Closing first, then opening: the closing repairs the ice (it fuses the two blocks through the thin bridge and
fills the one-pixel hole), and the opening then removes what is too thin to be a real object. Applying them the
other way round would delete the small features *before* the closing could repair them.
""")

md(r"""
### 7.1.2 Connected-component extraction and labelling (pp. 146–149, Figs. 7.3/7.4) — replaces `labeling.m`

Labelling starts from "an arbitrary object pixel that is not yet assigned" $p$, and grows it with a **constrained
(geodesic) dilation** — **Eq. (7.1)**:

$$X_k = (X_{k-1} \oplus B) \cap A, \qquad k = 1, 2, 3, \ldots, \qquad X_0 = \{p\},$$

stopping "when $X_k = X_{k-1}$". The structuring element chooses the connectivity: a **3 × 3 square** gives
8-connectivity (Fig. 7.3, which "completes at the 8th iteration") and a **cross**, `strel('diamond', 1)`, gives
4-connectivity (Fig. 7.4, 7 printed blocks — the shipped `labeling.m` has it as a commented line). The two
answers differ: the 4-connected component stops at the diagonal step that 8-connectivity walks straight through.
""")
code(r'''
A73, SEED = synth.FIG_7_3_IMAGE, synth.FIG_7_3_SEED             # Fig. 7.3(a)/(c): the 9x9 image and the seed p
r8 = connected_component_extract(A73, SEED, "square3")           # Fig. 7.3 — 8-connectivity
r4 = connected_component_extract(A73, SEED, "diamond1")          # Fig. 7.4 — 4-connectivity (the commented line 24)
for tag, r, printed, n_book in (("Fig. 7.3 (3x3 square, 8-conn)", r8, synth.FIG_7_3_STEPS, 8),
                                ("Fig. 7.4 (cross, 4-conn)", r4, synth.FIG_7_4_STEPS, 7)):
    bad = sum(int((b != p).sum()) for b, p in zip(r.blocks, printed))
    print(f"{tag}: converged at k = {r.n_iter}, {len(printed)} printed blocks, sums "
          f"{[int(b.sum()) for b in r.blocks[:len(printed)]]}, {bad} px differ from the book"
          f"  (book: {n_book} blocks)")
print(f"8-connectivity recovers the whole object (X7 == A: {np.array_equal(r8.component, A73)}); "
      f"4-connectivity keeps {int(r4.component.sum())} of {int(A73.sum())} px — the diagonal tail is a "
      "separate 4-connected component")

fig, axes = plt.subplots(2, 8, figsize=(19, 5.4))
for j in range(8):
    show_matrix(axes[0, j], synth.FIG_7_3_STEPS[j].astype(int), fontsize=4.5,
                title=("X0 (+) B" if j == 0 else f"X{j}") + f"  ({int(synth.FIG_7_3_STEPS[j].sum())})")
for j in range(8):
    if j < 7:
        show_matrix(axes[1, j], synth.FIG_7_4_STEPS[j].astype(int), fontsize=4.5,
                    title=("X0 (+) B" if j == 0 else f"X{j}") + f"  ({int(synth.FIG_7_4_STEPS[j].sum())})")
    else:
        axes[1, j].axis("off")
axes[0, 0].set_ylabel("Fig. 7.3  8-conn"); axes[1, 0].set_ylabel("Fig. 7.4  4-conn")
fig.suptitle("Figures 7.3(d) and 7.4(d)  Eq. (7.1) constrained dilation, 3x3 square vs cross SE", y=1.0)
plt.show()
''')

md(r"""
### 7.1.3.1 Hole filling by iterative dilation (pp. 149–152, Figs. 7.5–7.7) — replaces `filling.m`

The same recursion fills a hole if the mask is replaced by the **complement** — **Eq. (7.2)**:

$$X_k = (X_{k-1} \oplus B) \cap A^c, \qquad X_0 = \{p\},\ p \in \text{hole},$$

and the filled image is $X_n \cup A$. It has two failure modes, and the book prints both. With the cross SE a
second, *separate* 4-connected hole is simply not reached (Fig. 7.6). With the 3 × 3 square SE the hole of that
same image turns out to be **8-connected to the background**, so the flood escapes and paints the whole
complement (Fig. 7.7) — nothing is filled at all. That is the algorithm's real weakness: it must be *told* whether
the seed lies in a hole or in the background, "which is difficult to automatically fulfil" (p. 152) — hence
§7.1.3.2.
""")
code(r'''
f75 = hole_fill_dilation(A73, synth.FIG_7_5_SEED, "diamond1")      # Fig. 7.5: seed inside the hole, cross SE
printed = synth.FIG_7_5_STEPS                                       # [X0 (+) B, X1..X6, X6 | A]
bad = sum(int((b != p).sum()) for b, p in zip(f75.blocks, printed[:-1]))
bad += int((f75.filled != printed[-1]).sum())
print(f"Fig. 7.5(e): converged at k = {f75.n_iter}, block sums {[int(b.sum()) for b in f75.blocks[:7]]} "
      f"-> filled {int(f75.filled.sum())} px; {bad} px differ from the 8 printed blocks")
print("  (the text says 'finishes at the 7th iteration'; the printed blocks stop at X6 before the union, and "
      "X7 == X6, so both readings give the same 32-pixel image)")

fig, axes = plt.subplots(1, 8, figsize=(19, 3.0))
for j, blk in enumerate(printed):
    lbl = "X0 (+) B" if j == 0 else (f"X{j}" if j < 7 else "X6 U A")
    show_matrix(axes[j], blk.astype(int), fontsize=4.5, title=f"{lbl}  ({int(blk.sum())})")
fig.suptitle("Figure 7.5(e)  Eq. (7.2) hole filling by constrained dilation (cross SE)")
plt.show()
''')
code(r'''
A76 = synth.FIG_7_6_IMAGE                       # Fig. 7.5's image + one pixel at 1-based (5, 5): two 4-conn holes
f76 = hole_fill_dilation(A76, synth.FIG_7_5_SEED, "diamond1")     # Fig. 7.6 — cross SE: only one hole is reached
f77 = hole_fill_dilation(A76, synth.FIG_7_5_SEED, "square3")      # Fig. 7.7 — 3x3 square SE: the flood escapes
print(f"Fig. 7.6: {int(A76.sum())} object px, the recursion saturates after {f76.n_iter} iterations at "
      f"{int(f76.component.sum())} px and the union has {int(f76.filled.sum())} px — the second 4-connected hole "
      "survives, unfilled")
print(f"Fig. 7.7: with the 3x3 square SE the same seed floods {int(f77.component.sum())} px = the whole "
      f"complement ({int((~A76).sum())} px); the 'filled' image is the whole 9x9 frame "
      f"({int(f77.filled.sum())} px) — nothing was filled")
print("  block sums:", [int(b.sum()) for b in f77.blocks])

# --- the book erratum in Fig. 7.7(e) X8 (verification report, Deviation 1) --------------------------------------
diff = np.asarray(synth.FIG_7_7_STEPS[8]) != np.asarray(synth.FIG_7_7_STEPS_BOOK[8])
print(f"\nFig. 7.7(e) block X8: the recursion gives {int(synth.FIG_7_7_STEPS[8].sum())} px, the book prints "
      f"{int(synth.FIG_7_7_STEPS_BOOK[8].sum())} px — they differ at "
      f"{[(int(r) + 1, int(cc) + 1) for r, cc in zip(*np.nonzero(diff))]} (1-based).")
print("  X7 already has 1-based (7, 9) and (8, 8) set, so a 3x3 dilation MUST set them, and the printed "
      "X9 = A^c has them set too. MATLAB R2025a running the original code gives 55 px: the printed 53 is a "
      "book erratum, and the port reproduces the equation, not the misprint.")
print(f"  our X8 == the corrected fixture: {np.array_equal(f77.blocks[8], synth.FIG_7_7_STEPS[8])}")

fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
show_matrix(axes[0], A76.astype(int), fontsize=5, title=f"(b) variant image ({int(A76.sum())} px)")
show_matrix(axes[1], f76.filled.astype(int), fontsize=5,
            title=f"7.6: one hole filled ({int(f76.filled.sum())} px)")
show_matrix(axes[2], synth.FIG_7_7_STEPS[8].astype(int), fontsize=5, highlight=diff.astype(float) + 0.6,
            title="7.7 X8 computed (55 px)")
show_matrix(axes[3], synth.FIG_7_7_STEPS_BOOK[8].astype(int), fontsize=5, highlight=diff.astype(float) + 0.6,
            title="7.7 X8 as printed (53 px)")
fig.suptitle("Figures 7.6/7.7  The two failure cases of Eq. (7.2) — and a two-pixel book erratum (shaded cells)")
plt.show()
''')

md(r"""
### 7.1.3.2 Hole filling by reconstruction (pp. 152–153, Fig. 7.8) — replaces `filling_reconstruct.m`

The automatic version seeds the recursion from the **image border** instead of from inside a hole. The marker is
**Eq. (7.3)**

$$F_m(x, y) = \begin{cases} 1 - F(x, y) & (x, y) \text{ on the border of } F\\ 0 & \text{otherwise,}\end{cases}$$

and the filled image is **Eq. (7.4)**, the complement of the reconstruction by dilation of that marker under the
complement mask:

$$H = \left[R^{D}_{F^{c}}(F_m)\right]^{c}.$$

Whatever the border can reach is background; everything else is a hole. "This algorithm is fully automatic since
it does not need any information of the holes" (p. 152) — and it is exactly what MATLAB's `imfill(F,'holes')`
does. Chapter 7 needed `imfill` on a **double** image (the M-file calls it on one), where
`scipy.ndimage.binary_fill_holes` returns the wrong class *and*, for 8-connectivity, sometimes the wrong values —
so `seaice.core.morphology.imfill` is a line-by-line port of MATLAB's own `imfill.m` (40 comparisons, all 0 px).
""")
code(r'''
F = synth.FIG_7_8_IMAGE                                            # Fig. 7.8(b): the printed 9x11 image F
Fm = border_marker(F)                                              # Eq. (7.3)
H = hole_fill_reconstruct(F, conn=4)                               # Eq. (7.4)
print(f"F {int(F.sum())} px; Eq. (7.3) marker {int(Fm.sum())} px "
      f"(== the M-file's hand-written x0: {np.array_equal(Fm, synth.FIG_7_8_MARKER)}); "
      f"H {int(H.sum())} px; filled holes H n F^c {int((H & ~F).sum())} px")
print(f"Eq. (7.4) == imfill(F, 'holes'): {np.array_equal(H, imfill(F, 'holes', conn=4))}"
      f"; the 7 printed Fig. 7.8(e) blocks differ by "
      f"{int((np.asarray(synth.FIG_7_8_STEPS[-2]) != H).sum() + (np.asarray(synth.FIG_7_8_STEPS[-1]) != (H & ~F)).sum())} px")
print(f"imfill keeps the input's class, which is what ice_shape_enhancement.m needs: "
      f"float64 in -> {imfill(F.astype(float), 'hole').dtype} out, bool in -> {imfill(F, 'holes').dtype} out")

labels = ["Fm (+) B", "D(1)", "D(2)", "D(3)", "D(4)", "H = [R]^c", "H n F^c"]
fig, axes = plt.subplots(1, 9, figsize=(19, 2.8))
show_matrix(axes[0], F.astype(int), fontsize=4.5, title=f"(b) F  ({int(F.sum())})")
show_matrix(axes[1], Fm.astype(int), fontsize=4.5, title=f"(d) Fm — Eq. (7.3)  ({int(Fm.sum())})")
for j, (blk, lbl) in enumerate(zip(synth.FIG_7_8_STEPS, labels)):
    show_matrix(axes[j + 2], np.asarray(blk, int), fontsize=4.5, title=f"{lbl}  ({int(np.asarray(blk).sum())})")
fig.suptitle("Figure 7.8  Automatic hole filling by reconstruction — Eqs. (7.3) and (7.4), cross SE")
plt.show()
''')

md(r"""
### 7.1.4 The ice shape enhancement algorithm (pp. 151–153, **Algorithm 2**) — part of `ice_shape_enhancement.m`

**Algorithm 2** puts the three tools together:

```
1: PIECES <- labeled regions in SEGMENTATION arranged from small to large
2: BW     <- empty black image
3: for each labeled region piece in PIECES (from small to large) do
4:     piece <- morphological clean and fill hole
5:     BW    <- BW with piece superimposed and labeled
6: end for
7: IDENTIFICATION <- labeled regions in BW
```

The **order matters**: "the arrangement of ice pieces in order of increasing size is required […] Otherwise, the
smaller ice piece contained in a larger ice floe may not be removed" (p. 151). Superimposing small pieces first
lets the filled larger floe overwrite them; the other way round the small piece would punch a hole in its host.
The cleaning SE grows with the piece — **Eq. (7.5)**:

$$r = \begin{cases} r_1 & \text{if } size_{ice} < size_{th}\\ r_2 & \text{if } size_{ice} \ge size_{th}\end{cases}
\qquad (r_1 = 1,\ r_2 = 2,\ size_{th} = 50)$$

and every identified piece is finally *coloured by its size* — **Eq. (7.6)**:

$$Color(p) = \begin{cases} 0 & p \notin ICE_{SEA}\\ C_1\left(1 - e^{-area(i)/C_2}\right) & p \in ice_{sea}(i)
\end{cases} \qquad (C_1 = 10000,\ C_2 = 1000),$$

so "smaller ice pieces are blue and larger ice pieces are red". The colour bar is labelled with Eq. (7.6)
*inverted*, $-\mathrm{round}\!\left(C_2\ln(1 - c/C_1)\right)$, which turns the colour back into an area.
""")
code(r'''
# A 40x60 demonstration of why Algorithm 2 sorts small -> large: a small piece inside a larger floe's hole.
seg_demo = np.zeros((40, 60)); seg_demo[6:34, 8:34] = 1.0        # one large floe...
seg_demo[15:25, 16:26] = 0.0                                      # ...with a hole in it...
seg_demo[18:22, 19:23] = 1.0                                      # ...containing a small separate piece
seg_demo[10:14, 44:48] = 1.0                                      # and one small piece out in the water
d = ice_shape_enhancement(seg_demo != 0, seg_demo, min_floe=P["min_floe"], min_brash=P["min_brash"],
                          se_th=P["se_th"], nbins=0)
print(f"input: {d.ice_area.size} labelled pieces, areas {sorted(d.ice_area.tolist())} px "
      f"(sorted small -> large by a STABLE sort; order = {d.order.tolist()})")
alive = int((np.bincount(d.out.astype(np.int64).ravel(), minlength=d.t + 1)[1:] > 0).sum())
print(f"after Algorithm 2: {d.t} labels superimposed, {alive} survive -> {len(d.ice_floe)} floe + "
      f"{len(d.brash_ice)} brash; the piece inside the big floe's hole is gone — the filled floe was "
      "superimposed on top of it")
print(f"Eq. (7.5): area 49 -> disk radius {adaptive_se_radius(49)}, area 50 -> {adaptive_se_radius(50)} "
      f"(se_th = {P['se_th']})")
areas = np.array([3, 40, 131, 788, 1273, 8112])
print(f"Eq. (7.6): areas {areas.tolist()} -> colours {size_color(areas).tolist()} -> inverted back to "
      f"{color_to_area(size_color(areas)).tolist()}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
imshow_matlab(axes[0], seg_demo, title="SEGMENTATION: 3 labelled pieces")
imshow_matlab(axes[1], d.fill != 0, title="after hole filling only (step 4)")
axes[2].imshow(label2rgb(d.out, cmap="jet", background=(1, 1, 1), shuffle=True))
axes[2].set_title(f"IDENTIFICATION: {alive} surviving pieces (step 7)"); axes[2].axis("off")
fig.suptitle("Algorithm 2 — cleaning + filling piece by piece, small to large")
plt.show()
''')
md(r"""
The interior piece disappears because the larger floe is superimposed *after* it, already hole-filled. Reversing
the order would leave the small piece painted over the floe — the book's warning, made visible.

The colour-bar arithmetic is worth pinning down separately, because the *tick integers* are the only thing about
Figs. 7.13/7.15/7.19–7.21/7.26/7.28 that can be checked at all: they are Eq. (7.6) inverted, applied to the
`min : fix((max−min)/n) : max` colour values MATLAB puts on the bar.
""")
code(r'''
# The printed Fig. 7.13 tick labels follow from the (min colour, step) pair the figure implies - pure arithmetic.
values = 29.0 + 1195.0 * np.arange(7)                     # the unique (lo, d) that MATLAB's ysh would have here
print("Eq. (7.6) inverted on those colour values:", color_to_area(values).tolist())
print("book Fig. 7.13 printed tick labels        : [3, 131, 277, 448, 656, 917, 1273]   (p. 159)")
print("=> the ARITHMETIC of the colour bar is reproduced. The AREAS behind it are not: Fig. 7.13's 205x263 "
      "source image is not shipped with the book, so this notebook cannot and does not reproduce the pieces "
      "themselves (verification report, Open item 6).")
''')

# =====================================================================================================================
# 7.2 General sea ice image processing
# =====================================================================================================================
md(r"""
## 7.2 General sea ice image processing (pp. 151–162) — replaces `sea_ice_demo.m` + `ice_shape_enhancement.m`

§7.2 is the chapter's pipeline: **Algorithm 3** (ice edge detection) → **Algorithm 4** (shape enhancement) →
**Algorithm 5** (type classification + FSD). The book demonstrates it on a 205 × 263 grayscale image
(Figs. 7.10–7.16) and reports "a total of **154** ice floes and **189** brash ice pieces … 60.52 % ice floe,
3.34 % brash ice, 16.03 % slush, and 20.11 % water" (p. 160).

> **That image is not shipped with the book's code, so those numbers are `unverified` and are not reproduced
> here.** Everything below runs the same three algorithms on the image loaded in the data cell; every count and
> percentage printed is **ours**, measured on **our** input, and is labelled as such. Comparing them with the
> book's would be meaningless — different photograph, different scale, different scene.

The first cell runs Algorithm 3, which is the expensive one (GVF field + one snake per initial contour, twice).
At full resolution it takes roughly half a minute to two minutes depending on the machine; the result is kept in
memory and reused by every §7.2 cell.
""")
code(r'''
RGB = I[::SCALE, ::SCALE]
print(f"Algorithm 3 on {RGB.shape[0]} x {RGB.shape[1]} px (SCALE = {SCALE}), Num = {P['Num']} GVF iterations, "
      f"iter = {P['iter']} snake iterations, kms0 = {P['kms0']} clusters ...")
t0 = time.time()
G = sea_ice_edge_detection(RGB, Num=P["Num"], iter=P["iter"], **ALG3)      # = seaice_kmean_GVF_forenhancement.m
print(f"  {time.time() - t0:.1f} s")
SEG, BK = G.out, G.bk
print(f"SEGMENTATION_seaice levels {np.unique(SEG).tolist()}: light ice {int((SEG == 1).sum())} px, "
      f"dark ice {int((SEG == 0.5).sum())} px, water {int((SEG == 0).sum())} px; "
      f"ICE (k-means) {int((BK != 0).sum())} px")
''')

md(r"""
### 7.2.1 Sea ice pixel extraction (pp. 153–156, Fig. 7.10)

Two detectors from Chapter 3 are used **together**, on purpose. Otsu thresholding returns only the *"light ice"*
(Fig. 7.10(b)); k-means with three or more clusters returns all the ice, the darkest cluster being water
(Fig. 7.10(c)); and the *"dark ice"* is the difference, $DARK = ICE - LIGHT$ (Fig. 7.10(d)). Why not just use
k-means? Because Otsu **under-detects**, and the holes it leaves are precisely where §6.3.3's distance-transform
initialisation can put snake contours; a k-means mask of a crowded ice field has too few holes to seed anything.
Otsu multi-thresholding for the dark ice would cost more time (p. 155).
""")
code(r'''
LIGHT, DARK = G.bw, G.bw0                                          # Otsu 'light ice' and ICE - LIGHT
print(f"ICE (k-means, {P['kms0']} clusters) {int((BK != 0).sum())} px = "
      f"{100 * (BK != 0).mean():.1f} % of the image")
print(f"LIGHT (Otsu)  {int((LIGHT != 0).sum())} px; DARK = ICE - LIGHT {int((DARK != 0).sum())} px; "
      f"the k-means mask has {100 * (BK != 0).mean() - 100 * (LIGHT != 0).mean():+.1f} pp more ice than Otsu")
fig, axes = plt.subplots(4, 1, figsize=(13, 9))
imshow_matlab(axes[0], T(RGB), title="(a) input image")
imshow_matlab(axes[1], T(LIGHT != 0), title="(b) 'light ice' — Otsu thresholding")
imshow_matlab(axes[2], T(BK != 0), title="(c) all ice — k-means, darkest cluster = water")
imshow_matlab(axes[3], T(DARK != 0), title="(d) 'dark ice' = (c) - (b)")
fig.suptitle("Section 7.2.1 (procedure of Figure 7.10)  Two detectors: Otsu leaves the holes the snakes need")
plt.show()
''')

md(r"""
### 7.2.2 Sea ice edge detection (p. 157, **Algorithm 3**, Fig. 7.11)

```
1: GVF   <- GVF derived from grayscale of input image
2: ICE   <- binary ice image by the k-means clustering method
3: LIGHT <- binary "light" ice image by the thresholding method
4: DARK  <- ICE - LIGHT
5: SEG_L <- ice floe segmentation (Algorithm 1) on LIGHT
6: SEG_D <- ice floe segmentation (Algorithm 1) on DARK
7: SEGMENTATION_seaice <- SEG_L + SEG_D (labeled differently)
```

Chapter 6's Algorithm 1 is run **twice**, and the two results are merged into a *three-level* image — light ice
1, dark ice 0.5, water 0. The different labels are not cosmetic: without them, a light floe touching a dark floe
would merge into one connected component and could never be separated again. This is the reused Chapter 6 code
(`seaice_kmean_GVF_forenhancement.m` is byte-identical in both chapters); its parity is `near`, and on this image
the port's `seg` differs from MATLAB's on 0.245 % of pixels.
""")
code(r'''
SEG_L, SEG_D = G.pass1.bw1, G.pass2.bw1
print(f"SEG_L (Algorithm 1 on LIGHT): {int((SEG_L != 0).sum())} px; "
      f"SEG_D (Algorithm 1 on DARK): {int((SEG_D != 0).sum())} px")
print(f"SEGMENTATION = SEG_L + 0.5 * SEG_D, three levels {np.unique(SEG).tolist()}"
      + matlab_ref("the port's seg differs from MATLAB's on 0.245 % of pixels, bk on 0.125 % — Chapter 6's "
                   "k-means/snake residual, not this chapter's"))
fig, axes = plt.subplots(3, 1, figsize=(13, 7))
imshow_matlab(axes[0], T(SEG_L != 0), title="SEG_L — Algorithm 1 on the 'light ice'")
imshow_matlab(axes[1], T(SEG_D != 0), title="SEG_D — Algorithm 1 on the 'dark ice'")
imshow_matlab(axes[2], T(SEG), autoscale=True, title="SEGMENTATION_seaice: white = light ice (1), gray = dark ice (0.5)")
fig.suptitle("Section 7.2.2 Algorithm 3 (procedure of Figure 7.11)  Light and dark ice must be labelled differently")
plt.show()
''')

md(r"""
### 7.2.3 Sea ice shape enhancement (p. 158, **Algorithm 4**, Figs. 7.12/7.13)

```
1: PIECES_seaice         <- labeled regions in SEGMENTATION_seaice
2: IDENTIFICATION_seaice <- ice shape enhancement (Algorithm 2)
```

Algorithm 4 is Algorithm 2 applied to the light **and** dark pieces *together* — "if the ice shape enhancement
algorithm is performed to the two layers independently, overlapping identifications would be produced" — with the
Eq. (7.5) adaptive disk. `ice_shape_enhancement` does exactly that: it builds one labelling out of both layers
(a dark pixel that is also light belongs to the light layer), sorts the pieces by area with a **stable** sort,
and superimposes them one at a time. It is `exact` against the M-file: on MATLAB's own segmentation of this image
all nine output arrays are identical, pixel for pixel.
""")
code(r'''
t0 = time.time()
R = ice_shape_enhancement(BK, SEG, min_floe=P["min_floe"], min_brash=P["min_brash"], se_th=P["se_th"], nbins=50)
print(f"Algorithms 4+5 (ice_shape_enhancement.m): {time.time() - t0:.1f} s")
print(f"  labelled pieces: {R.nn_bw} light + {R.nn_k} dark = {R.ice_area.size}; areas "
      f"{int(R.ice_area.min())}..{int(R.ice_area.max())} px, cleaned and filled small -> large")
present = np.bincount(R.out.astype(np.int64).ravel(), minlength=R.t + 1)[1:]
print(f"  superimposed pieces t = max(out) = {R.t}; {int((present == 0).sum())} label(s) completely overwritten "
      "by a larger piece")
print(f"  {len(R.ice_floe)} ice floes (area > {int(P['min_floe'])} px) and {len(R.brash_ice)} brash pieces"
      + matlab_ref("MATLAB's own segmentation of this image gives 1211 pieces (982 light + 229 dark), 712 "
                   "labels and 433 floes / 274 brash; fed the port's segmentation the same code gives 433 / 290"))
fig, axes = plt.subplots(3, 1, figsize=(13, 7))
imshow_matlab(axes[0], T(SEG), autoscale=True, title="(a) segmentation (Algorithm 3)")
imshow_matlab(axes[1], T(R.fill != 0), title="(b) after hole filling only")
imshow_matlab(axes[2], T(R.out != 0), title=f"(c) after shape enhancement — {R.t} identified pieces")
fig.suptitle("Section 7.2.3 Algorithm 4 (procedure of Figure 7.12)")
plt.show()
''')
code(r'''
# Eq. (7.6): colour every identified piece by its size, and label the colour bar with the inverted equation.
colours = np.concatenate([R.color_floe, R.color_brash]) if (R.color_floe.size or R.color_brash.size) else np.zeros(0)
tick_values, tick_labels = colorbar_area_ticks(colours, 6)
print(f"Eq. (7.6) colour values {int(colours.min())}..{int(colours.max())}; colour-bar tick labels (areas) "
      f"{tick_labels.tolist()}")
print("book Fig. 7.13 prints 3, 131, 277, 448, 656, 917, 1273 — for a different (unshipped) image; not comparable.")
cmap = plt.get_cmap("jet").copy(); cmap.set_bad("white")
fig, ax = plt.subplots(figsize=(13, 5.6))
im = ax.imshow(T(np.ma.masked_where(R.index == 0, R.index)), cmap=cmap, interpolation="nearest")
if R.floe_cen.size:
    ax.plot(R.floe_cen[:, 1] - 1, R.floe_cen[:, 0] - 1, "k*", markersize=3)
if R.brash_cen.size:
    ax.plot(R.brash_cen[:, 1] - 1, R.brash_cen[:, 0] - 1, "k.", markersize=1.5)
ax.set_axis_off()
size_colorbar(fig, im, colours, 6, ax=ax, label="piece area (px), Eq. (7.6) inverted")
ax.set_title("Section 7.2.3 (procedure of Figure 7.13)  Eq. (7.6) size-coded pieces; black dots = piece positions")
plt.show()
''')

md(r"""
### 7.2.4 Sea ice types classification and the FSD (pp. 160–162, **Algorithm 5**, Figs. 7.14–7.16)

```
1: FLOE  <- labeled regions in IDENTIFICATION_seaice with the sizes equal to or larger than T_floe
2: BRASH <- labeled regions in IDENTIFICATION_seaice with the sizes smaller than T_floe
3: PIXEL_seaice <- IDENTIFICATION_seaice U ICE
4: SLUSH <- PIXEL_seaice n (IDENTIFICATION_seaice)^c
5: WATER <- (PIXEL_seaice)^c
```

Brash ice is "an accumulation of floating ice made up of fragments not more than 2 m across", so the split is a
**tunable threshold** `T_floe` — a pixel count, an area or a characteristic length; the shipped script uses
`min_floe = 40` pixels. Step 3 is easy to miss and matters: the ice-pixel set has to be **updated after** the
enhancement, because cleaning turns some water pixels into ice. What is ice but belongs to no identified piece is
**slush**, and the edge pixels left between two connected floes are the **residue** of Fig. 7.16 (counted as
slush here, but reported separately).

One deliberate difference between the book and its own code: Algorithm 5 says "**equal to or larger than**
`T_floe`", the M-file writes `if area0 > min_floe`. With `min_brash = 1` the strict form also **drops every
1-pixel piece** from both ice layers. `ice_shape_enhancement(..., book_threshold=True)` gives the printed form;
the default reproduces the code.
""")
code(r'''
cov = R.coverage.as_percent()
print(f"coverage of OUR image: {cov['IceFloe']:.2f} % ice floe, {cov['BrashIce']:.2f} % brash ice, "
      f"{cov['Slush']:.2f} % slush, {cov['Water']:.2f} % water  (sum {sum(cov.values()):.2f} %)")
print(f"residue (Fig. 7.16): {int((R.index_residue != 0).sum())} px of slush that no identified piece covers")
print("The book's 60.52 / 3.34 / 16.03 / 20.11 % and its 154 floes / 189 brash belong to Fig. 7.10(a), a 205x263 "
      "image that is NOT shipped with the code: `unverified`, not reproduced (report Open item 6).")

fl, br, sl, wa = ice_types_classification(R.out, BK, T_floe=P["min_floe"])   # the pseudocode, written out alone
print(f"Algorithm 5 as printed: the four layers tile the image exactly once: "
      f"{bool(((fl + br + sl + wa) == 1).all())}")

Rb = ice_shape_enhancement(BK, SEG, min_floe=P["min_floe"], min_brash=P["min_brash"], se_th=P["se_th"],
                           nbins=0, book_threshold=True)            # the printed '>=' instead of the code's '>'
print(f"\nthreshold form: code '>' gives {len(R.ice_floe)} floes / {len(R.brash_ice)} brash; "
      f"book '>=' gives {len(Rb.ice_floe)} / {len(Rb.brash_ice)} "
      f"(pieces of exactly {int(P['min_floe'])} px move up; 1-pixel pieces reappear as brash)")

fig, axes = plt.subplots(5, 1, figsize=(13, 11))
for ax, layer, t in ((axes[0], R.index_floe != 0, f"(a) ice floes — {len(R.ice_floe)}"),
                     (axes[1], R.index_brash != 0, f"(b) brash ice — {len(R.brash_ice)}"),
                     (axes[2], R.index_slush != 0, "(c) slush"),
                     (axes[3], R.index_water != 0, "(d) water"),
                     (axes[4], R.index_residue != 0, "residue between connected floes (Fig. 7.16)")):
    imshow_matlab(ax, T(layer), title=t)
fig.suptitle("Section 7.2.4 Algorithm 5 (procedure of Figures 7.14 and 7.16)  The four ice-type layers")
plt.show()
''')
code(r'''
# The floe size distribution: hist(floe_area, 50) with MATLAB's bin-CENTRE semantics, bars coloured by Eq. (7.6).
fsd = R.fsd
print(f"FSD: {int(fsd.counts.sum())} floes in {fsd.nbins} bins, centres {fsd.centers[0]:.1f}..."
      f"{fsd.centers[-1]:.1f} px, modal bin {fsd.centers[int(fsd.counts.argmax())]:.1f} px "
      f"({int(fsd.counts.max())} floes)")
counts, centers = matlab_hist(np.array([f.Area for f in R.ice_floe], dtype=float), fsd.nbins)
print(f"MATLAB's hist() takes bin CENTRES, np.histogram takes edges — core.histogram.hist is the line-by-line "
      f"port, so the bars sit where the book's do (same centres: {np.allclose(centers, fsd.centers)}, "
      f"same counts: {np.array_equal(counts, fsd.counts)})")
width = float(fsd.centers[1] - fsd.centers[0]) if fsd.centers.size > 1 else 1.0
norm = (fsd.colors - fsd.colors.min()) / max(float(fsd.colors.max() - fsd.colors.min()), 1.0)
fig, ax = plt.subplots(figsize=(11, 4.5))
bars = ax.bar(fsd.centers, fsd.counts, width=width, color=plt.get_cmap("jet")(norm), edgecolor="k", linewidth=0.2)
ax.set_xlabel("floe area (pixels)"); ax.set_ylabel("number of ice floes")
ax.set_title("Section 7.2.4 (procedure of Figure 7.15)  Floe size distribution, bars coloured by Eq. (7.6)")
sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(float(fsd.colors.min()), float(fsd.colors.max())))
size_colorbar(fig, sm, fsd.colors, 8, ax=ax, label="floe area (px), Eq. (7.6) inverted")
plt.show()
''')

# =====================================================================================================================
# 7.3 Case studies
# =====================================================================================================================
md(r"""
## 7.3 Case studies and discussion (pp. 163–174)

### 7.3.1.1 Local processing (pp. 163–164, **Algorithm 6**, Fig. 7.18)

A large, perspective-distorted image (Fig. 7.17) cannot be processed with one set of parameters: "the GVF capture
range derived by a uniform parameter sometimes cannot represent an overall ice image and should be adjusted
according to each sub-image […] but at the expense of more processing time and possibly manual intervention"
(p. 163). So the image is divided into **overlapping** sub-images — overlapping because of the §6.5.3 border
effect, where a floe cut by a sub-image edge segments badly — Algorithm 3 is run on each, "the overlapping parts
are removed and the sub-segmented images are merged by an image stitching method", and Algorithms 4–5 then run on
the stitched result.

> **There is no MATLAB file for this section**, and the book fixes **neither the tile size, nor the overlap size,
> nor the stitching rule**. `tile_grid` / `local_segmentation` are `reimplemented` with all three as parameters
> (the defaults below are this port's choice); what *can* be proved is structural — the tiles cover the image, and
> the cores that survive the overlap removal partition it exactly once. The book's own Fig. 7.17 scene is not
> shipped, so its 2511/2624 pieces and 65.98/5.03/17.52/11.47 % are `unverified` and are not reproduced.

To keep the notebook to a few minutes, this section processes **every 3rd row and column** (`SCALE_LOCAL = 3`) —
eight sub-images instead of a full-resolution run.
""")
code(r'''
SMALL = I[::SCALE_LOCAL, ::SCALE_LOCAL]
tiles, n_rows, n_cols = tile_grid(SMALL.shape[:2], 128, 24)
claimed = np.zeros(SMALL.shape[:2], dtype=int)
for rec in tiles:
    k0, k1, k2, k3 = rec.core
    claimed[k0:k1, k2:k3] += 1
print(f"{SMALL.shape[0]} x {SMALL.shape[1]} px (SCALE_LOCAL = {SCALE_LOCAL}) -> {n_rows} x {n_cols} = "
      f"{len(tiles)} sub-images of 128 x 128 px with 24 px overlap")
print(f"every pixel is claimed by exactly one core after the overlap is removed: {bool((claimed == 1).all())}")
fig, ax = plt.subplots(figsize=(13, 5.6))
imshow_matlab(ax, T(SMALL))
for rec in tiles:
    r0, r1, c0, c1 = rec.bounds
    k0, k1, k2, k3 = rec.core
    ax.add_patch(mpatches.Rectangle((r0 - 0.5, c0 - 0.5), r1 - r0, c1 - c0, fill=False, ec="r", lw=1.0))
    ax.add_patch(mpatches.Rectangle((k0 - 0.5, k2 - 0.5), k1 - k0, k3 - k2, fill=False, ec="y", lw=1.2, ls="--"))
ax.set_title("Section 7.3.1.1  red = overlapping sub-image, yellow dashed = the core kept after stitching")
plt.show()
''')
code(r'''
print(f"Algorithm 6 steps 1-5: Algorithm 3 on each of the {len(tiles)} sub-images ...")
t0 = time.time()
LS = local_segmentation(SMALL, tile=128, overlap=24, merge="crop", Num=P["Num"], iter=P["iter"], **ALG3)
t_local = time.time() - t0
bad = [r.index for r in LS.tiles if r.error]
print(f"  {t_local:.1f} s; stitched SEG: light {int((LS.seg == 1).sum())} px, dark {int((LS.seg == 0.5).sum())} px"
      + (f"; degenerate sub-images: {bad}" if bad else "; no degenerate sub-image"))
E_LOCAL = ice_shape_enhancement(LS.bk, LS.seg, min_floe=P["min_floe"], min_brash=P["min_brash"], se_th=P["se_th"],
                                nbins=0)
print(f"  steps 7-10 (Algorithms 4+5): {E_LOCAL.t} pieces -> {len(E_LOCAL.ice_floe)} floes, "
      f"{len(E_LOCAL.brash_ice)} brash")

t0 = time.time()
GW = sea_ice_edge_detection(SMALL, Num=P["Num"], iter=P["iter"], **ALG3)        # the whole-image run to compare
t_global = time.time() - t0
E_GLOBAL = ice_shape_enhancement(GW.bk, GW.out, min_floe=P["min_floe"], min_brash=P["min_brash"],
                                 se_th=P["se_th"], nbins=0)
print(f"whole-image Algorithm 3 on the same {SMALL.shape[0]} x {SMALL.shape[1]} px for comparison: "
      f"{t_global:.1f} s -> {E_GLOBAL.t} pieces, {len(E_GLOBAL.ice_floe)} floes, {len(E_GLOBAL.brash_ice)} brash")
print(f"the two segmentations differ on {int((LS.seg != GW.out).sum())} of {LS.seg.size} px "
      f"({100 * (LS.seg != GW.out).mean():.2f} %) — mostly along the stitching seams. Which is *better* cannot "
      "be decided here: the book ships no ground truth for this image.")

demo = LS.tiles[len(LS.tiles) // 2]
dr0, dr1, dc0, dc1 = demo.bounds
fig, axes = plt.subplots(4, 1, figsize=(13, 9))
imshow_matlab(axes[0], T(SMALL), title="overall sea ice image (every 2nd row/column)")
imshow_matlab(axes[1], T(SMALL[dr0:dr1, dc0:dc1]), title=f"one overlapping sub-image {demo.index}")
imshow_matlab(axes[2], T(LS.seg[dr0:dr1, dc0:dc1]), autoscale=True, title="its sub-segmentation (Algorithm 3)")
imshow_matlab(axes[3], T(LS.seg), autoscale=True, title="overall segmentation, stitched ('the overlapping parts are removed')")
fig.suptitle("Section 7.3.1.1 (procedure of Figure 7.18)  divide -> ice edge detection -> stitch")
plt.show()
''')

md(r"""
### 7.3.1.2 Geometric calibration (pp. 164–167) — deferred to Appendix A

The oblique camera of Fig. 7.17 makes far-range floes look small, so the segmentation has to be
**orthorectified** before the sizes mean anything. Two things in §7.3.1.2 belong to Chapter 7 and one does not.
The one that does not is the camera model — the authors could not obtain the camera parameters and *estimated* a
shooting angle of 20° and a field of view of 46° from the statistical similarity of near-range and far-range size
distributions; the rectification itself is Appendix A.1.1 and is ported in Chapter 10. This port deliberately
does **not** contain a second rectifier.

What Chapter 7 does own is (i) *where* the calibration belongs in the pipeline — after Algorithm 3 and **before**
Algorithm 4, because calibrating the raw image blurs the boundaries and costs GVF iterations, while calibrating
after Algorithms 4–5 turns small far-range floes into brash — and (ii) the fact that a segmented image must be
resampled with **nearest-neighbour** interpolation, "since the pixel values of the segmented image represent the
ice categories". Any averaging interpolant invents categories that do not exist, such as a 0.75 half-way between
"dark ice" (0.5) and "light ice" (1).
""")
code(r'''
# Nearest-neighbour resampling of a categorical image preserves the label set; bilinear invents categories.
from seaice.core.interp import interp2
M7, N7 = LS.seg.shape
u, v = np.meshgrid(np.linspace(0, M7 - 1, M7 // 2), np.linspace(0, N7 - 1, N7 // 2),   # 0-based (row, col) grid
                   indexing="ij")                                                       # a plain 2x resampling
near = resample_categorical(LS.seg, u, v)
lin = interp2(LS.seg, u, v, method="linear")
print(f"labels in SEG                          : {np.unique(LS.seg).tolist()}")
print(f"labels after nearest-neighbour resample: {np.unique(near).tolist()}")
print(f"labels after bilinear resample         : {len(np.unique(lin))} distinct values, e.g. "
      f"{np.unique(lin)[:6].round(3).tolist()} ... — categories that do not exist")
fig, axes = plt.subplots(2, 1, figsize=(13, 4.6))
imshow_matlab(axes[0], T(near), autoscale=True, title="nearest neighbour — still 3 categories")
imshow_matlab(axes[1], T(lin), autoscale=True, title="bilinear — invented intermediate categories")
fig.suptitle("Section 7.3.1.2  Why a labelled image must be resampled with nearest-neighbour interpolation")
plt.show()
''')

md(r"""
### 7.3.2 A preliminary sensitivity study (pp. 168–174, Figs. 7.24/7.27)

The last section varies one parameter at a time on the 394 × 1038 marginal-ice-zone image (Fig. 7.22 — the image
loaded above) and counts the floes and brash pieces that come out:

* **§7.3.2.1** the snake's iteration limit is swept **1 → 122** with the GVF field fixed at **500** iterations
  (Fig. 7.24). With one iteration the initial contours have barely moved, the image is heavily over-segmented,
  and Algorithm 4 then *merges* those fragments into under-segmentation (Figs. 7.25/7.26(a)). The counts rise,
  peak, and fall to a steady value once every snake has converged.
* **§7.3.2.2** the GVF iteration count is swept **1 → 1500 in steps of 20** with the snake limit fixed at **100**
  (Fig. 7.27). Too few GVF iterations means a small capture range, noise dominates, and the result is
  over-segmented (Fig. 7.28(a)); the curves settle as the field diffuses. The brash curves oscillate more than
  the floe curves throughout, "since the boundaries of brash ice are weaker".

> **This section is `unverified` by construction.** The book prints **no counts at all** for Figs. 7.24 and
> 7.27 — only the axis ranges (y 200…1600 and 200…2000) and the tick positions. There is nothing to compare a
> number against, and the numbers below are not book values: only the *shape* of the curves is comparable. The
> full sweeps are 25 + 75 pipeline runs; the cell below runs **seven** settings — the ones the book prints a
> panel for — on every 4th row and column (`SCALE_SWEEP = 4`), which keeps the qualitative behaviour and the
> runtime in the same minute.
""")
code(r'''
TINY = I[::SCALE_SWEEP, ::SCALE_SWEEP]
SNAKE_VALUES, GVF_VALUES = (1, 10, 70, 100), (1, 61, 181, 500)      # the book's own panels 1/10/70 and 1/61/181
runs = {}                                                            # (Num, iter) -> result, so 500/100 runs once

def one_run(num, iters):
    """Algorithms 3 -> 4 -> 5 once (cached); returns (floes, brash, seconds, result)."""
    if (num, iters) not in runs:
        t0 = time.time()
        g = sea_ice_edge_detection(TINY, Num=num, iter=iters, **ALG3)
        e = ice_shape_enhancement(g.bk, g.out, min_floe=P["min_floe"], min_brash=P["min_brash"],
                                  se_th=P["se_th"], nbins=0)
        runs[(num, iters)] = (len(e.ice_floe), len(e.brash_ice), time.time() - t0, e)
    return runs[(num, iters)]

print(f"sweeping on {TINY.shape[0]} x {TINY.shape[1]} px (SCALE_SWEEP = {SCALE_SWEEP}); "
      "NONE of these counts is a book value")
sweeps = {}
for name, values in (("snake", SNAKE_VALUES), ("gvf", GVF_VALUES)):
    rows = []
    for val in values:
        num, iters = (P["Num"], val) if name == "snake" else (val, P["iter"])
        nf, nb, dt, e = one_run(num, iters)
        rows.append((val, nf, nb))
        print(f"  {name:5s} = {val:4d} (Num {num:4d}, iter {iters:4d}): {nf:4d} floes, {nb:4d} brash, "
              f"{nf + nb:4d} total   ({dt:4.1f} s)")
    sweeps[name] = rows
print(f"{len(runs)} distinct pipeline runs (Num 500 / iter 100 is shared by both sweeps)")

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
for ax, (name, rows) in zip(axes, sweeps.items()):
    x = [r[0] for r in rows]
    ax.plot(x, [r[1] for r in rows], "o-", label="Floe")
    ax.plot(x, [r[2] for r in rows], "s-", label="Brash")
    ax.plot(x, [r[1] + r[2] for r in rows], "^-", label="Floe+Brash")
    ax.set_xlabel("Snake iterations" if name == "snake" else "GVF iterations")
    ax.set_ylabel("number of ice floes / brash pieces")
    ax.set_xticks(x); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"procedure of Figure {'7.24' if name == 'snake' else '7.27'} — "
                 f"{'GVF' if name == 'snake' else 'snake'} iterations fixed at "
                 f"{P['Num'] if name == 'snake' else P['iter']}")
fig.suptitle(f"Section 7.3.2  Sensitivity of the identified piece counts (our counts, at 1/{SCALE_SWEEP} "
             "resolution — the book prints none)")
plt.show()
''')
code(r'''
# The three panels the book prints for the snake sweep, side by side (Figs. 7.25/7.26 show the same idea).
fig, axes = plt.subplots(3, 1, figsize=(13, 7))
for ax, val in zip(axes, (1, 10, 70)):
    e = runs[(P["Num"], val)][3]
    ax.imshow(T(label2rgb(e.index, cmap="jet", background=(1, 1, 1), shuffle=False)))
    ax.set_axis_off()
    ax.set_title(f"snake iterations = {val}: {len(e.ice_floe)} floes + {len(e.brash_ice)} brash")
fig.suptitle("Section 7.3.2.1 (procedure of Figures 7.26(a)-(c))  Too few snake iterations => under-segmentation")
plt.show()

# --- why the printed colour-bar ticks of Fig. 7.26(a) must NOT be read as a parity check --------------------------
print("Fig. 7.26(a) prints the colour-bar ticks 2, 184, 407, 695, 1100, 1792, 8112 (p. 171). Eq. (7.6) saturates:")
for area in (8000, 8112, 12000, 100000):
    print(f"  a piece of {area:6d} px gets colour value {int(size_color(area))} (the ceiling is 10000)")
for largest in (8112, 12000, 30000, 100000):
    _, lab = colorbar_area_ticks(np.array([size_color(2), size_color(largest)]), 6)
    print(f"  smallest piece 2 px, largest {largest:6d} px -> ticks {lab.tolist()}")
print("=> ANY run whose smallest piece is 2 px and whose largest is >= 8112 px prints that exact list. The "
      "agreement is a saturation coincidence, not evidence that the segmentation matches the book's "
      "(verification report, Deviation 5 / Open item 5).")
''')

# =====================================================================================================================
# Parameter play
# =====================================================================================================================
md(r"""
## Parameter play (optional)

Two knobs of Algorithms 4 and 5 on the segmentation already computed in §7.2, so each move only re-runs
`ice_shape_enhancement` (a second or so, no snakes): **`T_floe`** (`min_floe`), the pixel area that separates ice
floes from brash ice, and **`se_th`**, the Eq. (7.5) area at which the cleaning disk grows from radius 1 to
radius 2. `T_floe` slides pieces between the two ice classes without changing the identification at all;
`se_th` changes the identification itself, because a larger disk erodes thin necks and can split or delete a
piece. Requires `ipywidgets`; without it only the static preview runs.
""")
code(r'''
def play(min_floe=40, se_th=50):
    r = ice_shape_enhancement(BK, SEG, min_floe=min_floe, min_brash=P["min_brash"], se_th=se_th, nbins=0)
    c = r.coverage.as_percent()
    print(f"T_floe = {min_floe} px, se_th = {se_th} px -> {r.t} identified pieces, {len(r.ice_floe)} floes, "
          f"{len(r.brash_ice)} brash; coverage {c['IceFloe']:.1f} / {c['BrashIce']:.1f} / {c['Slush']:.1f} / "
          f"{c['Water']:.1f} %")
    fig, axes = plt.subplots(2, 1, figsize=(13, 4.6))
    imshow_matlab(axes[0], T(r.index_floe != 0), title=f"ice floes ({len(r.ice_floe)})")
    imshow_matlab(axes[1], T(r.index_brash != 0), title=f"brash ice ({len(r.brash_ice)})")
    plt.show()

play()                                                # static preview with the script's own settings
try:
    from ipywidgets import interact, IntSlider
    interact(play, min_floe=IntSlider(40, 5, 400, 5), se_th=IntSlider(50, 5, 400, 5))
except ImportError:
    print("ipywidgets not installed - only the static preview above is shown")
''')

# =====================================================================================================================
# MATLAB <-> Python map + summary
# =====================================================================================================================
md(r"""
## MATLAB ↔ Python mapping used in this chapter

| MATLAB | Python (`seaice`) | Parity | Note |
|---|---|---|---|
| `imclose(I, se)` / `imopen(I, se)` with `strel('square', 2)` | `core.morphology.imclose` / `imopen` via `ch07_ice_type.morphological_cleaning` | exact | the 2 × 2 SE is *even*, so its origin is off-centre — this is what makes Fig. 7.2 shift |
| the `xx = imdilate(x0, se); x1 = xx & I; …` chain (Eq. 7.1) | `ch07_ice_type.connected_component_extract` | exact vs the file | stops at the fixed point instead of unrolling to `x9`; equals the `bwlabel` component containing the seed |
| the same chain on `~I0` (Eq. 7.2) | `ch07_ice_type.hole_fill_dilation` | exact vs the file | `blocks`, `component`, `filled` = the printed blocks |
| the hand-written border marker `x0` (Eq. 7.3) | `ch07_ice_type.border_marker` | exact | verified equal to the M-file's literal |
| `imfill(BW, 'holes')` (Eq. 7.4) | `ch07_ice_type.hole_fill_reconstruct`, `core.morphology.imfill` | exact | `imfill` is a line-by-line port of MATLAB's `imfill.m` — it keeps the **input's class** and handles the grayscale branch; `scipy.ndimage.binary_fill_holes` does neither |
| `bwlabel(BW, 4)`, `regionprops(·,'Centroid'/'Perimeter')` | `core.connectivity.label_components`, `core.regionprops.regionprops` | exact | inherited from ch. 4–6 |
| `hist(y, n)` | `core.histogram.hist` | near | MATLAB's `hist` uses bin **centres** and unbounded outer bins; only `-Inf` differs (report Open item 1) |
| `fix((1 - exp(-area/1000)) * 10000)` (Eq. 7.6) | `ch07_ice_type.size_color` | exact | `fix` = truncation, not rounding |
| `-round(1000*log(1 - ysh/10000))` (colour-bar labels) | `ch07_ice_type.color_to_area`, `colorbar_area_ticks`, `core.plotting.size_colorbar` | exact | all eleven printed tick lists follow from the arithmetic; four of them **saturate** and prove nothing |
| `label2rgb(index, @jet, [1 1 1])` | `core.plotting.label2rgb` | display only | survives ~10 000 distinct colour values; colour PNGs are never compared numerically |
| `ice_shape_enhancement(bk, seg, min_floe, min_brash, se_th)` | `ch07_ice_type.ice_shape_enhancement` | exact | 36 fixture runs + the real image, all nine arrays 0 px; the M-file's HG1 bar colouring (lines 214–224) **errors in R2025a** and is replaced by `FloeSizeDistribution.colors` |
| `seaice_kmean_GVF_forenhancement(...)` (Algorithm 3) | `ch07_ice_type.sea_ice_edge_detection` → `ch06_gvf_snake.seaice_kmean_gvf` | near | reused, not re-ported: the 23 `.m` in `ch7/Sea_Ice_Floe_Identification/` are byte-identical to ch6's |
| *(no file — Algorithm 6, §7.3.1.1)* | `ch07_ice_type.tile_grid`, `local_segmentation` | reimplemented | the book states no tile size, overlap or stitching rule |
| *(no file — §7.3.1.2 resampling)* | `ch07_ice_type.resample_categorical` | verified (label set preserved) | the camera model is Appendix A / ch. 10 |
""")
md(r"""
## Summary — what this chapter gives the next one

**What Chapter 7 adds.** Four binary primitives (cleaning, Eq. 7.1 labelling, Eq. 7.2 / Eqs. 7.3–7.4 hole
filling), one substantial algorithm — `ice_shape_enhancement`, which is Algorithms 2, 4 **and** 5 in a single
252-line M-file — and two new reusable core functions, `core.morphology.imfill` (MATLAB's own, class-preserving)
and `core.histogram.hist` (bin centres, not edges). The pipeline is now: Chapter 3 ice mask → Chapter 4/5 edges
and floe separation → Chapter 6 GVF snakes → **Chapter 7 identified pieces classified into floe / brash / slush /
water, with a floe size distribution**.

**What is verified, and how.** Every printed matrix of Figs. 7.2–7.8 (48 blocks) was compared against MATLAB
R2025a running the original `.m` files: 47 are identical and the 48th is a **book erratum** (Fig. 7.7(e) `X_8`,
55 px not 53). `ice_shape_enhancement` is `exact` against the M-file on 36 controlled fixtures and on the real
image — fed MATLAB's own segmentation, all nine output arrays are 0 px. `imfill` matches MATLAB on 40
comparisons *and* keeps the class; `hist` on 16 of 17 cases. Full detail: `reports/ch07_verification.md`.

**What is not verified, and why.** The source images for Figs. 7.1, 7.9 and 7.10–7.21 are not shipped anywhere in
the book's code archive, so **154/189 pieces, 2511/2624 pieces and the eight coverage percentages cannot be
reproduced** — they are `unverified` (Open item 6) and this notebook does not compute look-alike numbers and
label them as the book's. §7.3.2 is `unverified` by construction: the book prints no counts for Figs. 7.24/7.27.
Algorithm 6's tiling is `reimplemented` because no code and no parameters exist for it, and §7.3.1.2's
rectification is deferred to Appendix A.

**Feeds forward.** Chapter 8 (applications) consumes exactly what this chapter returns: `ice_floe` and
`brash_ice` — lists whose fields are named `Center`, `Area`, `Perimeter`, `PixelsPosition` because
`sea_ice_model.m` (§8.2, the polygon fit per floe and disk fit per brash piece) and `SeaIce_Image_Structure.m`
(Appendix B) read those names — plus `index_floe`, `coverage` and the floe size distribution that `color_hist.m`
re-draws in §8.3. Chapter 10 (Appendix A) supplies the orthorectification that step 6 of Algorithm 6 needs, and
will call `resample_categorical` for the segmented image. The learned knowledge for this chapter is in
`knowledge/ch07.md` and `knowledge/CUMULATIVE.md`.
""")

out = HERE / NB_NAME
nbf.write(nb, out)
print(f"wrote {out} ({len(nb.cells)} cells)")
