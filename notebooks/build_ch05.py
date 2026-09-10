"""Build ``notebooks/ch05_watershed_segmentation.ipynb`` with nbformat (re-runnable; the notebook is never edited by hand).

Chapter 5 — Watershed-Based Ice Floe Segmentation (Zhang & Skjetne 2018, pp. 83–108).
Follows the ``colab-notebook`` skill: title → cell 1 (Drive mount / cwd) → cell 2 (clone or pull the public repo)
→ cell 3 (``load_image``: private copy or public-domain substitute) → one section per book section
(5.1, 5.1.1, 5.1.2, 5.1.3, 5.2, 5.2.1.1, 5.2.1.2, 5.3) → optional parameter play → MATLAB ↔ Python map →
summary / feeds forward.  All algorithms are imported from ``seaice``; the notebook only calls them and draws figures.

Run:  ``.venv/Scripts/python.exe notebooks/build_ch05.py``
Verify: ``.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900
        --output executed_ch05.ipynb notebooks/ch05_watershed_segmentation.ipynb``  (then delete the executed copy).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

NB_NAME = "ch05_watershed_segmentation.ipynb"
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
# Chapter 5 — Watershed-Based Ice Floe Segmentation

*Sea Ice Image Processing with MATLAB* (Q. Zhang & R. Skjetne, CRC Press 2018), Chapter 5, pp. 83–108 —
Python port `seaice-py`, notebook `notebooks/ch05_watershed_segmentation.ipynb`.

**Sections covered** (one notebook section each, in book order):
5.1 Watershed segmentation (Eqs. 5.1–5.8, Figs. 5.1–5.4) · 5.1.1 Gradients as segmentation function (Figs. 5.5–5.7) ·
5.1.2 The distance transform (Figs. 5.8–5.11) · 5.1.3 Marker-controlled watershed (Fig. 5.12) · 5.2 Watershed +
neighbouring-region merging (Figs. 5.13–5.15) · 5.2.1.1 Boundary tracing (Fig. 5.16) · 5.2.1.2 Differential chain code
(Eqs. 5.9–5.19, Fig. 5.17) · 5.3 Experimental results (Figs. 5.18–5.20, Table 5.1).

The output of this chapter is a **binary floe mask in which touching floes are separated**: the Chapter 3 ice mask is
turned into an inverse distance map, flooded by the watershed transform, and the spurious watershed lines are removed
by the authors' neighbouring-region merging (a line is kept only if one of its ending points is a *concave* boundary
point). The primitives introduced here — `watershed`, `imregionalmin` / `imregionalmax`, `imimposemin` — are reused by
Chapters 7–9.

**MATLAB files this notebook replaces** (`MATLAB_ROOT/ch5/`):

| MATLAB file | Python (imported below) | Parity (from `reports/ch05_verification.md`, MATLAB R2025a as reference) |
|---|---|---|
| `topological_surface.m` (complement, Sobel surface, inverse chessboard DT, `surf(…, 'texturemap')`) | `seaice.ch05_watershed.topographic_surfaces` → `seaice.core.plotting.surface_plot` | exact (arrays 0 px / 0.0; the surface rendering is display-only) |
| `direct_watershed.m` (`watershed(gray)`, ridges superimposed, `bwlabel`, `label2rgb`) | `direct_watershed` → `seaice.core.watershed.watershed` | exact (160 basins, labels and 2132 ridge px 0 px) |
| `gradients_watershed.m` (unscaled Sobel, `watershed(g)`, `imclose(imopen(g, ones(7)))`) | `sobel_magnitude`, `gradient_watershed` | exact (350 / 12 basins, all arrays 0 px) |
| `distance_propagation.m` (201×201 point, `-bwdist`, `uint8` rescale, `imcontour`) | `inverse_distance_map` → `seaice.core.distance.bwdist`, `plotting.contour_overlay` | exact (contour rendering display-only) |
| `distance_watershed.m` (chessboard `-bwdist`, `imregionalmin`, `watershed`, `bwareaopen(…, 5)`) | `distance_watershed`, `otsu_mask`, `inverse_distance` | exact for all four metrics (minima, labels, floes 0 px) |
| `marker_watershed.m` (`imregionalmin` → `imdilate(strel('disk', 5))` → `imimposemin` → `watershed`) | `marker_watershed` → `seaice.core.morphology.imregionalmin / imimposemin / strel / imdilate` | exact (imposed map bit-identical in single, 2 floes) |
| `chaincode_corner.m` / `watershed_based/freeman_concave.m` (Eqs. 5.9–5.18, rule 3 ≤ D ≤ 10) | `freeman_concave`, `differential_chain_code` | exact (`R`, `A`, `Sum`, `D`, `Diff`, 9 concave points identical) |
| `watershed_based/main.m` (junction lines, Fig. 5.15 kernel, `imreconstruct`, merging loop) | `neighboring_region_merging`, `junction_endpoints`, `ENDPOINT_KERNEL` | exact (3 lines, endpoints, per-line decisions, 4 → 2 floes; also on a 552×574 crop and a synthetic image) |
| `boundaries.m`, `fchcode.m`, `bound2im.m` (both copies; byte-identical to `ch2/chain code/`) | `seaice.core.chaincode` (ported in Chapter 2) | exact |
| MATLAB `watershed` itself (compiled Meyer flooding) | **`seaice.core.watershed.watershed`** — line-by-line port of the readable `eml/watershed.m` twin | exact on 74 constructed cases (label values included; flooding order pinned by 31 order-sensitive fixtures) |
| *(text only: Eq. 5.2, §5.1.3 Steps 1–2, Eqs. 5.1 / 5.3–5.8)* | `regional_minima_by_reconstruction`, `impose_minima_book`, `watershed_immersion`, `threshold_set` | reimplemented (identities vs the exact primitives) |

The chapter's learned knowledge (concepts, primitives, pitfalls, what later chapters need) is in
`knowledge/ch05.md` (written by the knowledge phase) and `knowledge/CUMULATIVE.md`.

> **Data.** Every figure of §5.1–5.2 comes from one tiny image, `q.jpg` (96×81 RGB: two bright floes touching on
> dark water — Figs. 5.1(a), 5.8(a), 5.14(a)). The book's images are copyrighted and are **not** in this public
> repository: `seaice.core.io.load_image()` first looks for your own private copy (`data/book/ch05/q.jpg` next to the
> repository, or `MyDrive/Sea_Ice_Colab/data/book/ch05/` on Colab) and otherwise downloads a public-domain NASA MODIS
> window of the same 81×96 size (a few large floes touching at the bottom of the frame), registered in
> `seaice/core/public_images.py`. The §5.3 images (Ny-Ålesund, May 2011, Figs. 5.7, 5.11, 5.18–5.20, Table 5.1) are
> **not shipped at all**; §5.3 below runs the same pipeline on the Fig. 4.3(a) crop of Chapter 4's `test.jpg` (or its
> substitute) and on a synthetic two-floe image. Figs. 5.15 and 5.16 are printed matrices, reproduced from
> `seaice.core.synth` fixtures.
""")

# =====================================================================================================================
# 2. Setup cells (Colab / Drive / local) — copied verbatim from build_ch02.py / build_ch04.py
# =====================================================================================================================
md(r"""
## Setup (Google Colab or local Jupyter)

Run the two cells below first. On **Colab** the first cell mounts your Google Drive and moves into
`MyDrive/Sea_Ice_Colab` — the folder where your private copy of the book image lives (`data/book/ch05/q.jpg`)
and where downloads are cached between sessions; readers without Drive get a temporary `/content/Sea_Ice_Colab`.
The second cell clones (or updates) the public repository `seaice-py` there and installs its requirements.
**Locally** both cells are no-ops that move to the repository root. No GPU is needed.

> ⚠️ **If you run this with your own copy of the book image, do not use *File → Save a copy in GitHub*.** That saves
> the cell outputs — figures rendered from the copyrighted book image — into the public repository. Save to Drive
> instead (*File → Save a copy in Drive*). The repository's `_colab.ipynb` is always regenerated with outputs stripped.
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

`seaice.core.io.load_image("ch05", "q.jpg")` replaces the `imread('q.jpg')` at the top of every Chapter 5 script. It
looks for **your private copy of the book image** first — `data/book/ch05/q.jpg` in the current folder (on Colab:
`MyDrive/Sea_Ice_Colab`), in the repository, or explicitly in Drive — and otherwise downloads a **public-domain NASA
substitute** (MODIS/Terra true-colour window of Beaufort Sea pack ice, 25 July 2019, requested at the same 81×96 size
so that the book's `xlim([0 85])` and the Fig. 5.10 zoom windows stay meaningful) into `data/online/ch05/`. It prints
which source it used and returns the label, so the cells below quote the book's image-dependent numbers — 4 regional
minima / 18 pixels, 4 regions, 2 markers, 3 junction lines, 9 concave points — **only when the book's own image is
loaded** (`book(...)` / `matlab_ref(...)` helpers). The chapter's parameters (7×7 square, 5-pixel disk, `bwareaopen`
5, concavity band 3–10) hold for any image.
""")
code(r'''
from seaice.core.io import load_image

I, SOURCE = load_image("ch05", "q.jpg")             # = imread('q.jpg') of every ch5 script; uint8 (96, 81, 3)
FROM_BOOK = SOURCE.startswith("book")
book = (lambda s: f"  (book: {s})") if FROM_BOOK else (lambda s: "")           # book-quoted values only for the book image
matlab_ref = (lambda s: f"  [MATLAB R2025a on the book image: {s}]") if FROM_BOOK else (lambda s: "")   # verified reference values
if not FROM_BOOK:
    print("Running on a public-domain substitute image; figures show the same operations, "
          "but values quoted in the book only hold for the book's own image.")
print("image:", I.shape, I.dtype, f"  [{SOURCE}]")
''')

md(r"""
## Imports

Everything algorithmic comes from the `seaice` package; the notebook only calls it and draws figures.
(`seaice.core.plotting` selects the non-interactive Agg backend for the command-line scripts, so `%matplotlib inline`
is issued *after* the `seaice` imports to switch back to inline display.) Two binarisations are prepared once,
because the scripts differ: `distance_watershed.m`, `marker_watershed.m`, `main.m` and `chaincode_corner.m` call
`im2bw(img, graythresh(img))` on the **RGB** image (`graythresh` histograms all three planes), while
`direct_watershed.m`, `gradients_watershed.m` and `topological_surface.m` threshold the **gray** image; on `q.jpg` the
two Otsu levels are 130/255 and 128/255 and the masks differ slightly. `otsu_mask` reproduces both.
""")
code(r'''
import time
import numpy as np
import seaice.core                                    # imports seaice.core.plotting (sets the Agg backend)
from seaice.core import synth
from seaice.core.plotting import imshow_matlab, show_matrix, label2rgb, surface_plot   # display helpers only
from seaice.core.matlab_compat import rgb2gray_matlab
from seaice.core.connectivity import label_components
from seaice.ch05_watershed import BOOK_PARAMS, otsu_mask, inverse_distance

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100

G = rgb2gray_matlab(I)                                # I = rgb2gray(I) of the gray-based scripts (MATLAB weights + rounding)
BW = otsu_mask(I)                                     # img = im2bw(img, graythresh(img)) on the RGB (Fig. 5.8(a), 5.14(b))
BW_GRAY = otsu_mask(G)                                # the gray-image variant (topological_surface.m, gradients_watershed.m)
M, N = G.shape
IMG_LABEL = "q.jpg" if FROM_BOOK else "substitute image"
print(f"gray image {M}x{N}, levels {int(G.min())} .. {int(G.max())}")
print(f"Otsu mask of the RGB: {int(BW.sum())} ice px ({100 * BW.mean():.1f} %), {int(label_components(BW).max())} connected ice region(s)"
      f"{matlab_ref('level 130/255, 2699 ice px, one component = the two touching floes')}")
print(f"Otsu mask of the gray image: {int(BW_GRAY.sum())} ice px; the two masks differ in {int((BW != BW_GRAY).sum())} px"
      f"{matlab_ref('level 128/255')}")
print("book parameters (BOOK_PARAMS):", BOOK_PARAMS)
''')

# =====================================================================================================================
# 5.1 Watershed segmentation
# =====================================================================================================================
md(r"""
## 5.1 Watershed segmentation (pp. 84–89) — Eqs. (5.1)–(5.8), Figs. 5.1–5.4; `ch5/topological_surface.m`, `ch5/direct_watershed.m`

Read a gray image as a **topographic surface** — height = intensity — and imagine rain falling on it: every drop runs
down to a **regional minimum** (a connected set of equal-valued pixels whose neighbours are all strictly higher; a
*local* minimum is merely a pixel not higher than its neighbours), the pixels draining to the same minimum form its
**catchment basin**, and the ridges separating basins are the **watershed lines** (Figs. 5.1–5.2). Because
segmentation looks for basins, the floe image is *complemented* first, so that bright floes become basins and the
dark water between them a ridge. The book then describes Vincent–Soille's immersion: with the threshold set (cross-section
below level $h$)

$$T_h(I) = \{(u, v) \mid I(u, v) < h\} \quad (5.1), \qquad M = R^E_I(I + 1) - I \quad (5.2)$$

(Eq. 5.2 extracts all regional minima at once by grayscale reconstruction *by erosion* of $I+1$ under $I$,
Eqs. 4.35–4.38), the surface is flooded from $h_{min}$ upward: the basin of minimum $M_i$ flooded to level $h$ is
$C_h(M_i) = C(M_i) \cap T_h(I)$ (5.3), their union is $C[h] = \bigcup_i C_h(M_i)$ (5.4), and the complete
segmentation is $C[h_{max}+1] = \bigcup_i C(M_i)$ (5.5). The nested sets $C[h-1] \subseteq C[h]$ (5.6),
$C[h] \subseteq T_h(I)$ (5.7) and $C[h-1] \subseteq T_h(I)$ (5.8) mean that every component $S$ of $T_h(I)$ contains
either no flooded basin (a new minimum appears), exactly one (the basin grows to $S$) or several (a one-pixel **dam**
is built by dilating the basins inside $S$ with a $3\times3$ element until they meet) — the three cases of Fig. 5.4.
With 8-connected basins the lines come out 4-connected, one pixel thick and unbroken (pp. 88–89).

MATLAB's `watershed` (compiled) implements the same idea as Meyer's flooding from a priority queue; the port
`seaice.core.watershed.watershed` follows the readable `eml/watershed.m` twin line by line and is pixel-identical to
MATLAB (label values included), while `watershed_immersion` is the book's level-by-level recursion written from
Eqs. (5.1)–(5.8) for teaching.
""")
code(r'''
from seaice.ch05_watershed import topographic_surfaces

surf = topographic_surfaces(I)                        # topological_surface.m: gray, 255 - gray, Sobel g, gray Otsu mask, -bwdist chessboard
fig, axes = plt.subplots(1, 2, figsize=(9, 5.2))
imshow_matlab(axes[0], surf["gray"], title=f"(a) {IMG_LABEL} (rgb2gray)")
imshow_matlab(axes[1], surf["complement"], title="(b) imcomplement: 255 − I  (floes become basins)")
fig.suptitle("Fig. 5.1 (a)(b)  A gray floe image and its complement (topological_surface.m)")
plt.show()
fig = surface_plot(surf["complement"], cmap="copper", xlim=(0, 85),
                   title="(c) surf(imcomplement(I)), copper, xlim [0 85]: two basins and the ridge between them")
plt.show()
print(f"complement range {int(surf['complement'].min())} .. {int(surf['complement'].max())} = 255 − gray{matlab_ref('x = imcomplement(I) 0 px')}")
''')
code(r'''
from seaice.core.morphology import imregionalmin
from seaice.core.watershed import watershed
from seaice.ch05_watershed import regional_minima_by_reconstruction, threshold_set, watershed_immersion

X = surf["complement"]                                # the topographic surface of Fig. 5.1(c)
rm = imregionalmin(X)                                 # MATLAB imregionalmin (8-connected)
rm_eq = regional_minima_by_reconstruction(X)          # Eq. 5.2: M = R^E_I(I + 1) − I > 0
print(f"regional minima of the complemented gray image: {int(label_components(rm).max())} components / {int(rm.sum())} px; "
      f"Eq. 5.2 by reconstruction equals imregionalmin: {np.array_equal(rm, rm_eq)}")
levels = [int(v) for v in np.percentile(X, [15, 35, 55, 75])]
fig, axes = plt.subplots(1, 5, figsize=(18, 4.4))
imshow_matlab(axes[0], rm, title=f"regional minima M (Eq. 5.2): {int(rm.sum())} px")
for ax, h in zip(axes[1:], levels):
    imshow_matlab(ax, threshold_set(X, h), title=f"T_h(I) = {{I < {h}}}  (Eq. 5.1)")
fig.suptitle("Section 5.1  Regional minima and the threshold sets (cross-sections) that the immersion floods, level by level")
plt.show()

# --- Figs. 5.2-5.3 in 1-D: one row of the surface through both floes, flooded by MATLAB's watershed and by the book's recursion
D_row = inverse_distance(BW, "cityblock")            # -bwdist(~BW, 'cityblock'): the §5.1.2 surface (one minimum per floe)
row = max(range(M), key=lambda r: int(watershed(D_row[r][None, :]).max()))   # the row whose -D profile has the most basins
profiles = [("complement 255 − I (the gray image: many small minima and plateaus)", X[row, :].astype(np.float64)),
            ("−bwdist(~BW, 'cityblock') (one basin per floe, a dam at the neck)", D_row[row, :].astype(np.float64))]
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, (name, prof) in zip(axes, profiles):
    Lm = watershed(prof[None, :])[0]                  # MATLAB flooding of a 1×N row
    Li = watershed_immersion(prof[None, :])[0]        # Eqs. 5.1, 5.3-5.8 level by level
    x = np.arange(N)
    ax.plot(x, prof, "k", lw=1.5, label=f"row {row + 1} of the surface")
    for k in range(1, int(Lm.max()) + 1):
        ax.fill_between(x, prof.min() - 2, prof, where=Lm == k, alpha=0.3)
    dams = np.flatnonzero(Lm == 0)
    ax.vlines(dams, prof.min() - 2, prof.max() + 2, colors="r", lw=1.5, label=f"watershed lines: {dams.size} dams, {int(Lm.max())} basins")
    ax.set_title(f"{name}: the book's immersion recursion places the same dams: {np.array_equal(Lm == 0, Li == 0)} (they can differ only on plateaus)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
axes[1].set_xlabel("column")
fig.suptitle("Figs. 5.2-5.3 (1-D)  Catchment basins (shaded) and dams (red) of two segmentation functions")
plt.show()
''')
code(r'''
from seaice.ch05_watershed import direct_watershed

dw = direct_watershed(G)                              # direct_watershed.m: watershed of the uint8 gray image itself
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
imshow_matlab(axes[0], dw.ridge, title=f"bgm = (watershed(im) == 0): {dw.n_basins} basins")
imshow_matlab(axes[1], dw.overlay, title="im(bgm) = 0: watershed lines in black on the gray image")
imshow_matlab(axes[2], label2rgb(dw.labels), title=f"label2rgb(bwlabel(im), 'jet', 'k', 'shuffle'): {dw.n_components} pieces")
fig.suptitle("Section 5.1 (p. 89)  Watershed of the gray image itself — direct_watershed.m (no book figure): severe over-segmentation")
plt.show()
print(f"watershed(gray): {dw.n_basins} basins, {int(dw.ridge.sum())} ridge px{matlab_ref('160 basins, 2132 ridge px, labels 0 px')}; "
      "every texture minimum inside a floe becomes a basin, so the raw gray image is not a usable segmentation function")
''')
md(r"""
### Which `watershed`? — the port vs the book's recursion vs scikit-image

Three implementations of the same definition give different lines on plateaus and ties. `seaice.core.watershed.watershed`
reproduces MATLAB exactly (Meyer's flooding with a FIFO priority queue, seeded by a column-major scan of the regional
minima, no propagation from ridge pixels); `watershed_immersion` is the level-by-level recursion of Eqs. (5.1)–(5.8)
with dams built by constrained dilation; `skimage.segmentation.watershed(..., watershed_line=True)` seeds the queue
with the minima themselves, re-pushes pixels, keeps propagating from line pixels and scans neighbours in a different
order — on the book image its ridge differs from MATLAB in 46–122 pixels per distance map and even the basin
partition changes, which is why the ports never use it (parity label **approx**; `watershed_skimage` is kept only for
this comparison). The neighbour *scan* order of the flooding loop turned out to be irrelevant by construction; the
*initial* column-major seed scan is what pins MATLAB's result.
""")
code(r'''
from seaice.core.watershed import watershed_skimage, neighbour_offsets

print("MATLAB neighbour order (linear offsets on the padded frame, 8-conn):", neighbour_offsets(8, M + 2))
D_city = inverse_distance(BW, "cityblock")            # -bwdist(~BW, 'cityblock') in single precision, as the scripts
for name, A in [("gray image", G.astype(np.float64)), ("-D city-block", D_city)]:
    t0 = time.perf_counter(); Lm = watershed(A); t_port = time.perf_counter() - t0
    t0 = time.perf_counter(); Li = watershed_immersion(A); t_imm = time.perf_counter() - t0
    Ls = watershed_skimage(A)
    ice = BW if name.startswith("-D") else np.ones_like(BW)
    print(f"{name:14s}: port {int(Lm.max())} basins / {int((Lm == 0).sum())} ridge px ({t_port:.2f} s); "
          f"book recursion {int(Li.max())} basins, ridge differs from the port in {int(((Li == 0) != (Lm == 0)).sum())} px ({t_imm:.2f} s); "
          f"skimage {int(Ls.max())} basins, ridge differs in {int(((Ls == 0) != (Lm == 0)).sum())} px "
          f"({int((((Ls == 0) != (Lm == 0)) & ice).sum())} inside the ice)")
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
imshow_matlab(axes[0], watershed(D_city) == 0, title="port of MATLAB watershed(-D): ridge")
imshow_matlab(axes[1], watershed_immersion(D_city) == 0, title="watershed_immersion (Eqs. 5.1-5.8): dams")
imshow_matlab(axes[2], watershed_skimage(D_city) == 0, title="skimage watershed_line=True (approx)")
fig.suptitle("Three watersheds of the same inverse city-block distance map")
plt.show()
''')

# =====================================================================================================================
# 5.1.1 Gradients
# =====================================================================================================================
md(r"""
## 5.1.1 Watershed segmentation using gradients (pp. 89–90) — Figs. 5.5–5.7; `ch5/gradients_watershed.m`

The natural segmentation function is a **gradient magnitude** (Chapter 4): objects have low gradient inside and
high gradient at their contours, so basins should be objects and ridges their boundaries. Fig. 5.5 applies the
$3\times3$ Sobel operator to Fig. 5.1(a) and floods it — the result is a dense mesh of lines, because every noise
ripple of the gradient is a regional minimum. Fig. 5.6 smooths the gradient with a **$7\times7$ square close-opening**,
$g_2 = (g \circ B) \bullet B$, which removes most small minima but still leaves extraneous lines through the floes;
and Fig. 5.7 (image not shipped) shows the real limitation: where the edge between two touching floes is weak or
invisible there is no ridge at all, so gradient watersheds cannot separate them — which motivates the distance
transform of §5.1.2.

**What the script does.** `gradients_watershed.m` filters `double(I)` (0–255) with the *unscaled* `fspecial('sobel')`
kernels and replicate padding, takes $g = \sqrt{I_x^2 + I_y^2}$ (units of 0–255·8, no `/8`), floods `g` and
`imclose(imopen(g, ones(7,7)), ones(7,7))` and superimposes the lines in black (`f(wr) = 0`). Both label images are
pixel-identical to MATLAB (350 and 12 basins on the book image).
""")
code(r'''
from seaice.ch05_watershed import gradient_watershed

gw = gradient_watershed(G, smooth=BOOK_PARAMS["close_open_size"])       # g, watershed(g), g2 = close-open 7x7, watershed(g2)
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
imshow_matlab(axes[0], gw.g, autoscale=True, title="(a) Sobel gradient magnitude g  (imshow(g, []))")
imshow_matlab(axes[1], gw.L, autoscale=True, title=f"(c) watershed(g): imshow(l, []) — {gw.n_basins} basins")
imshow_matlab(axes[2], gw.overlay, title="(d) f(wr) = 0: watershed lines in black on the gray image")
fig.suptitle("Fig. 5.5  Watershed segmentation of the 3×3 Sobel gradient of Fig. 5.1(a): far too many lines")
plt.show()
fig = surface_plot(gw.g, cmap="copper", xlim=(0, 85), title="Fig. 5.5(b)  surface of the gradient magnitude")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
imshow_matlab(axes[0], gw.g2, autoscale=True, title="(a) g2 = imclose(imopen(g, ones(7)), ones(7))")
imshow_matlab(axes[1], gw.L2, autoscale=True, title=f"(b) watershed(g2): {gw.n_basins2} basins")
imshow_matlab(axes[2], gw.overlay2, title="(c) lines of the smoothed gradient on the gray image")
fig.suptitle("Fig. 5.6  The same after a 7×7 square close-opening: fewer, but still extraneous, lines")
plt.show()
print(f"basins: raw gradient {gw.n_basins}, after the 7×7 close-opening {gw.n_basins2}{matlab_ref('350 / 12, label images 0 px')}; "
      f"g range 0 .. {gw.g.max():.0f} (0-255·8 units: unscaled Sobel kernels), g2 range {gw.g2.min():.0f} .. {gw.g2.max():.0f}")
''')
md(r"""
Smoothing cuts the basin count by an order of magnitude, yet the lines that remain still cross the floes rather than
follow the invisible junction between them: the gradient is high at texture and low at the junction, the opposite
of what a floe segmentation needs.
""")

# =====================================================================================================================
# 5.1.2 Distance transform
# =====================================================================================================================
md(r"""
## 5.1.2 Watershed segmentation using the distance transform (pp. 90–95) — Figs. 5.8–5.11; `ch5/distance_propagation.m`, `ch5/distance_watershed.m`

For a *binary* ice mask the distance transform (§2.4) gives each ice pixel its distance to the nearest water pixel;
floe centres are far from water, the neck between two touching floes is close to it. Negating the map,
$-D = -\mathrm{bwdist}(\lnot BW)$ ("inverse distance map"), turns floe centres into regional minima and the neck into
a ridge — the regional minima of $-D$ are the regional maxima of $D$, and its watershed lines ("talweg lines") cut
the mask exactly where it is thinnest (Fig. 5.8: mask, inverse map, surface, minima, watershed line, result). The
choice of metric decides how the distance propagates from the water (Fig. 5.10): **city-block** in diamonds,
**Euclidean** in circles, **chessboard** in squares. Euclidean maps have the most regional minima and produce small
"islands" between floes (Fig. 5.9 top), city-block is a moderate compromise (Fig. 5.9 bottom; several minima with
different levels near a floe centre), and chessboard over-segments least but **under-segments** when a large part of
two boundaries touches (Fig. 5.11).

**What the scripts do.** `distance_propagation.m` shows the propagation shape on a 201×201 single-point image
(`imgDist = -bwdist(~img, 'cityblock')`, rescaled to `uint8` and drawn with `imcontour`); its commented lines produce
Fig. 5.10 from the mask, with the zoom windows columns 24–40 × rows 44–64 and columns 18–30 × rows 45–90.
`distance_watershed.m` binarises the RGB with Otsu, takes `-bwdist(~img, 'chessboard')` (single precision), marks
`imregionalmin` in black, floods with `watershed`, deletes the lines from the mask and drops pieces below 5 pixels
(`bwareaopen(img, 5)`). All arrays are identical to MATLAB for all four metrics.
""")
code(r'''
from seaice.ch05_watershed import inverse_distance_map

METRICS = ["cityblock", "euclidean", "chessboard"]
pt = synth.point_image(201)                           # img = zeros(201); img(101, 101) = 1
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
for ax, m in zip(axes, METRICS):
    r = inverse_distance_map(~pt, m)                  # the script: img = ~img; imgDist = -bwdist(~img, metric); dist = uint8(dist0*255/max)
    ax.imshow(r.dist, cmap="gray", extent=(0.5, 201.5, 201.5, 0.5), interpolation="nearest")   # image(dist, 'CDataMapping', 'scaled')
    Xg, Yg = np.meshgrid(np.arange(1, 202), np.arange(1, 202))
    ax.contour(Xg, Yg, r.imgDist, levels=12, cmap="viridis", linewidths=0.8)                     # imcontour(imgDist)
    ax.set_title(f"-bwdist(~img, '{m}'): min {float(r.imgDist.min()):g}, uint8 centre {int(r.dist[100, 100])}")
fig.suptitle("distance_propagation.m  Inverse distance map of a single point: diamond / circular / square propagation")
plt.show()

maps = {m: inverse_distance_map(BW, m) for m in METRICS}          # the commented lines: the mask of Fig. 5.8(a)
ZOOMS = [("(a) full image", None, None), ("(b) zoom cols 24–40 × rows 44–64", (24, 40), (44, 64)),
         ("(c) zoom cols 18–30 × rows 45–90", (18, 30), (45, 90))]            # 1-based windows as printed
Xg, Yg = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))
fig, axes = plt.subplots(3, 3, figsize=(15, 16))
for i, (sub, xl, yl) in enumerate(ZOOMS):
    for ax, m in zip(axes[i], METRICS):
        ax.imshow(maps[m].dist, cmap="gray", extent=(0.5, N + 0.5, M + 0.5, 0.5), interpolation="nearest")
        ax.contour(Xg, Yg, maps[m].imgDist, levels=14, cmap="viridis", linewidths=0.8)
        if xl is not None:
            ax.set_xlim(xl[0], xl[1]); ax.set_ylim(yl[1], yl[0])
        ax.set_title(f"{sub}: {m}", fontsize=10)
fig.suptitle("Fig. 5.10  Contours of the three inverse distance maps of the ice mask (city-block / Euclidean / chessboard) and two zooms")
plt.show()
for m in METRICS:
    print(f"{m:11s}: -bwdist min {float(maps[m].imgDist.min()):7.3f} (single), uint8 rescale max {int(maps[m].dist.max())}")
print(f"the 201x201 point map and the mask maps are the script's arrays{matlab_ref('imgDist 0.0 for city-block / chessboard, <= 1e-4 for Euclidean (single); uint8 rescale 0 levels for all three')}")
''')
code(r'''
from seaice.ch05_watershed import distance_watershed

dc = distance_watershed(BW, BOOK_PARAMS["metric_fig_5_8"], BOOK_PARAMS["min_area"])   # distance_watershed.m as shipped (chessboard)
fig, axes = plt.subplots(2, 3, figsize=(14, 11))
imshow_matlab(axes[0, 0], dc.bw, title="(a) img = im2bw(img, graythresh(img))")
imshow_matlab(axes[0, 1], dc.imgDist, autoscale=True, title="(b) -bwdist(~img, 'chessboard')  (imshow(., []))")
imshow_matlab(axes[0, 2], dc.minima_overlay, title=f"(d) regional minima in black: {dc.n_minima} minima / {dc.n_minimum_px} px")
imshow_matlab(axes[1, 0], dc.ridge, title=f"(e) watershed line: {dc.n_basins} basins")
imshow_matlab(axes[1, 1], dc.seg_ao, title=f"(f) img(bgm) = 0, bwareaopen(5): {dc.n_floes} floes")
imshow_matlab(axes[1, 2], label2rgb(dc.labels), title="(f) label2rgb(bwlabel(img))")
fig.suptitle("Fig. 5.8  Watershed segmentation of the inverse chessboard distance map (distance_watershed.m)")
plt.show()
fig = surface_plot(dc.imgDist, cmap="copper", xlim=(0, 85), title="Fig. 5.8(c)  surface of the inverse chessboard distance map")
plt.show()
print(f"chessboard: {dc.n_minima} regional minima ({dc.n_minimum_px} px) -> {dc.n_basins} basins -> {dc.n_floes} floes after bwareaopen({BOOK_PARAMS['min_area']})"
      f"{matlab_ref('2 minima / 38 px, 2 basins, 143 ridge px (19 inside the ice), 2 floes; all arrays 0 px')}")
''')
code(r'''
rows = {m: distance_watershed(BW, m, BOOK_PARAMS["min_area"]) for m in ["euclidean", "cityblock", "quasi-euclidean"]}
fig, axes = plt.subplots(3, 3, figsize=(13, 15))
for axr, (m, d) in zip(axes, rows.items()):
    imshow_matlab(axr[0], d.minima_overlay, title=f"{m}: {d.n_minima} regional minima / {d.n_minimum_px} px")
    imshow_matlab(axr[1], d.ridge, title=f"watershed lines: {d.n_basins} basins")
    imshow_matlab(axr[2], label2rgb(d.labels), title=f"after bwareaopen({BOOK_PARAMS['min_area']}): {d.n_floes} floes")
fig.suptitle("Fig. 5.9 (Euclidean / city-block rows) + quasi-Euclidean  Distance watershed with other metrics")
plt.show()
print(f"{'metric':16s} {'minima':>7s} {'min px':>7s} {'basins':>7s} {'floes':>6s}")
for m, d in [("chessboard", dc)] + list(rows.items()):
    extra = book("four regional minima consisting of 18 local minimum pixels; divides the two connected floes into four regions") if m == "cityblock" else ""
    print(f"{m:16s} {d.n_minima:7d} {d.n_minimum_px:7d} {d.n_basins:7d} {d.n_floes:6d}{extra}")
print(matlab_ref("chess 2/38/2/2, city 4/18/4/4, euc 6/6/6/6, quasi 6/10/6/6 — identical").strip() or
      "(book image: chess 2 / city 4 / Euclidean 6 / quasi-Euclidean 6 basins — the ordering the text describes)")
''')
md(r"""
The metric ordering the book describes shows up directly in the minima counts: Euclidean circles create the most
regional minima (and the "island" pieces on the ridge), city-block is in between, chessboard has the fewest — and on
a mask where two floes share a long straight junction the chessboard map may not produce a ridge at all
(under-segmentation, Fig. 5.11). City-block is the compromise §5.2 adopts.
""")

# =====================================================================================================================
# 5.1.3 Marker-controlled watershed
# =====================================================================================================================
md(r"""
## 5.1.3 Marker-controlled watershed segmentation (pp. 95–96) — Fig. 5.12; `ch5/marker_watershed.m`

Over-segmentation comes from having more regional minima than objects, so the cure is to **impose** one *marker*
(connected component) per object as the only minima of the segmentation function $g$. With a marker image $f$ that
is $0$ on the markers and $+\infty$ elsewhere, minima imposition is a reconstruction by erosion (Eqs. 4.35–4.38):

$$\text{Step 1: } (g + 1) \wedge f, \qquad \text{Step 2: } g' = R^E_{(g+1)\wedge f}(f),$$

where $\wedge$ is the pointwise minimum and the "+1" keeps distinct markers apart inside a level-0 minimum. Every
regional minimum of $g'$ is now a marker and the watershed of $g'$ has exactly one basin per marker. The worked
example (Fig. 5.12) takes the city-block inverse distance map of Fig. 5.8(a), whose **four regional minima (18 pixels)**
split the two floes into four regions, dilates the minima with a **5-pixel disk** into a marker image with **two
connected regions**, imposes them and obtains the two floes correctly. Choosing markers automatically (size, shape,
location, texture) is the hard part in general.

**What the script does.** `imregionalmin(-D)` → `imdilate(..., strel('disk', 5))` (MATLAB's 9×9 / 69-pixel octagonal
disk) → `imimposemin(-D, marker)` → `watershed`; MATLAB's `imimposemin` uses $-\infty$ / $+\infty$ marker images and
$h = 0.001\,(\max g - \min g)$ instead of the literal $+1$ for floating-point maps — `seaice.core.morphology.imimposemin`
ports that M-code and is bit-identical in single precision; `impose_minima_book` is the literal Steps 1–2. The
commented block of the script (one-pixel markers at the component centroids) is `point_markers=True`.
""")
code(r'''
from seaice.ch05_watershed import marker_watershed, impose_minima_book
from seaice.core.morphology import imimposemin, strel

R = BOOK_PARAMS["marker_disk_radius"]                 # 5 (pp. 96-97)
mk = marker_watershed(BW, BOOK_PARAMS["metric_merging"], R, min_area=BOOK_PARAMS["min_area"])   # marker_watershed.m
imp = mk.imposed
imp_disp = np.where(np.isfinite(imp), imp, imp[np.isfinite(imp)].min())            # -Inf markers drawn at the finite minimum
fig, axes = plt.subplots(2, 3, figsize=(14, 11))
imshow_matlab(axes[0, 0], mk.bw & ~mk.minima, title=f"(a) regional minima of -D city-block: {mk.n_minima} / {mk.n_minimum_px} px")
imshow_matlab(axes[0, 1], mk.marker_overlay, title=f"(b) marker = imdilate(minima, strel('disk', {R})): {mk.n_markers} components")
imshow_matlab(axes[0, 2], imp_disp, autoscale=True, title="(c) imimposemin(-D, marker)  (markers at -Inf)")
imshow_matlab(axes[1, 0], mk.ridge, title=f"(d) watershed line of the imposed map: {mk.n_basins} basins")
imshow_matlab(axes[1, 1], mk.seg_ao, title=f"(e) segmented: {mk.n_floes} floes")
imshow_matlab(axes[1, 2], label2rgb(mk.labels), title="(e) label2rgb(bwlabel(img))")
fig.suptitle("Fig. 5.12  Marker-controlled watershed of the inverse city-block distance map (marker_watershed.m)")
plt.show()
se = strel("disk", R)
print(f"(a) {mk.n_minima} regional minima / {mk.n_minimum_px} px{book('four regional minima consisting of 18 local minimum pixels')}")
print(f"(b) strel('disk', {R}) = {se.shape[0]}x{se.shape[1]}, {int(se.sum())} px octagon -> {mk.n_markers} marker component(s){book('a marker image containing two connected regions')}")
print(f"(c) imimposemin: class {imp.dtype}, {int(np.isneginf(imp).sum())} px at -Inf, imregionalmin(imposed) == marker: {np.array_equal(imregionalmin(imp), mk.marker)}"
      f"{matlab_ref('bit-identical in single, 328 marker px')}")
print(f"(d)(e) watershed: {mk.n_basins} basins, {int(mk.ridge.sum())} ridge px -> {mk.n_floes} floes{book('separates the two connected ice floes correctly')}"
      f"{matlab_ref('2 basins, 140 ridge px, 2 floes')}")

gb = impose_minima_book(mk.imgDist0, mk.marker)       # the text's Steps 1-2 with the literal (g + 1) ∧ f and R^E
print(f"book Steps 1-2 vs MATLAB imimposemin: same regional minima {np.array_equal(imregionalmin(gb), mk.marker)}, "
      f"same watershed {np.array_equal(watershed(gb), mk.L)} (the uniform offset h vs +1 does not change the flooding order)")
mk0 = marker_watershed(BW, BOOK_PARAMS["metric_merging"], R, point_markers=True)   # the commented centroid block
print(f"one-pixel markers at floor(regionprops Centroid) = {np.floor(mk0.centroids).astype(int).tolist()} (x, y; 1-based) -> "
      f"{mk0.n_basins} basins, {mk0.n_floes} floes{matlab_ref('[23 68; 46 50], 2 basins, 2 floes')}")
''')
md(r"""
Dilating the minima merges the ones that belong to the same floe (they are within 5 pixels of each other) without
bridging the two floes, so the imposed map has exactly one flat basin per floe and the watershed reduces to the single
junction line. A one-pixel marker at each component centroid does the same job here; the text prefers large markers
because they are more robust.
""")

# =====================================================================================================================
# 5.2 Neighbouring-region merging
# =====================================================================================================================
md(r"""
## 5.2 Combination of the watershed and neighbouring-region merging (pp. 96–99) — Figs. 5.13–5.15; `ch5/watershed_based/main.m`

Markers need to be chosen; the authors' alternative removes spurious lines *after* the watershed by a shape argument.
Ice floes are approximately **convex**, so where two floes really touch, the junction line ends in **concave**
corners of the merged outline; a watershed line whose two ending points are both convex must be spurious. The
pipeline (flow chart Fig. 5.13; Zhang, Skjetne & Su 2013) is: **Step 1** binarise (Otsu, Chapter 3); **Step 2**
city-block inverse distance map → 8-connected watershed → over-segmented mask (Fig. 5.14(b)–(e)); **Step 3** junction
lines = watershed lines AND ice mask (Fig. 5.14(f)); **Step 4** ending points: the lines are 4-connected and one pixel
thick, so an ending point has exactly one 4-neighbour on the line (the 12 patterns of Fig. 5.15(a)) and filtering with
the kernel

$$w = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix} \quad (\text{Fig. 5.15(b)})$$

gives exactly 3 there (an isolated pixel gives 4); **Step 5** for each line, if *neither* ending point is a concave
point of the two neighbouring regions taken together, delete the line and merge the regions (Fig. 5.14(g)–(h));
iterate over all lines (Fig. 5.14(i)). Concavity is decided by the differential chain code of §5.2.1.

**What the script does.** `main.m` follows the steps literally: `D = bwdist(~bw, 'cityblock'); L = watershed(-D);
f = bitand(bw, L == 0); [label, num] = bwlabel(f, 4)`; per line `abs(imfilter(double(g), wr)) >= max(...)` for the
ending points (the text says ≥ 3; both give the same points on proper lines), `imreconstruct(g, bwlabel(seg + line))`
for the merged neighbouring region, `freeman_concave` for its concave points, `intersect(ep, concave, 'rows')` for the
decision — and it updates `seg` **inside** the loop, so later lines see regions already merged. The port reproduces
every per-line record.
""")
code(r'''
from seaice.ch05_watershed import ENDPOINT_KERNEL, junction_endpoints
from seaice.core.filters import imfilter

pats = synth.FIG_5_15_ENDPOINT_PATTERNS               # the 12 printed 3x3 patterns (centre = ending point)
iso = np.zeros((3, 3), dtype=np.int64); iso[1, 1] = 1  # an isolated pixel
fig, axes = plt.subplots(2, 7, figsize=(17, 5.2))
show_matrix(axes[0, 0], ENDPOINT_KERNEL.astype(int), title="(b) kernel w", fontsize=10)
axes[0, 0].set_title("(b) kernel w", color="C3")
resp = []
for k, ax in enumerate(axes.ravel()[1:13]):
    r = imfilter(pats[k].astype(np.float64), ENDPOINT_KERNEL)[1, 1]              # correlation, zero padding (main.m)
    resp.append(r)
    show_matrix(ax, pats[k], title=f"pattern {k + 1}: w → {r:g}", fontsize=10)
r_iso = imfilter(iso.astype(np.float64), ENDPOINT_KERNEL)[1, 1]
show_matrix(axes[1, 6], iso, title=f"isolated pixel: w → {r_iso:g}", fontsize=10, cmap="Oranges")
fig.suptitle("Fig. 5.15  (a) the 12 ending-point patterns of a 4-connected 1-px line and (b) the detection kernel: response 3 at an ending point")
plt.show()
found = [(1, 1) in {tuple(q) for q in junction_endpoints(p, "ge3").tolist()} for p in pats]   # a 2-px line has two ending points
print(f"kernel responses at the 12 centres: {[int(v) for v in resp]} (book: 3); isolated pixel: {r_iso:g} (book: 'larger than 3'); "
      f"junction_endpoints(rule='ge3') finds the centre in all 12: {all(found)}"
      f"{matlab_ref('imfilter responses identical for the 12 patterns and the isolated pixel')}")
''')
code(r'''
from seaice.ch05_watershed import neighboring_region_merging

t0 = time.perf_counter()
mr = neighboring_region_merging(BW, BOOK_PARAMS["metric_merging"], endpoint_rule="max", sequential=True)   # main.m
print(f"main.m pipeline on the {M}x{N} mask: {time.perf_counter() - t0:.2f} s")
fig, axes = plt.subplots(2, 3, figsize=(14, 11))
imshow_matlab(axes[0, 0], mr.bw, title="(b) Step 1: Otsu mask")
imshow_matlab(axes[0, 1], -mr.D, autoscale=True, title="(c) Step 2: -bwdist(~bw, 'cityblock')")
imshow_matlab(axes[0, 2], mr.w, title=f"(d) watershed lines w = (L == 0): {int(mr.L.max())} basins")
imshow_matlab(axes[1, 0], label2rgb(label_components(mr.seg0)), title=f"(e) over-segmented seg0: {mr.n_floes_before} regions")
imshow_matlab(axes[1, 1], mr.f, title=f"(f) Step 3-4: junction lines f = bw & w ({mr.num}) + ending points")
for ln in mr.lines:
    axes[1, 1].plot(ln.endpoints[:, 1], ln.endpoints[:, 0], "r.", ms=9)
    axes[1, 1].text(ln.pixels[:, 1].mean() + 2, ln.pixels[:, 0].mean(), str(ln.label), color="y", fontsize=9)
imshow_matlab(axes[1, 2], label2rgb(label_components(mr.seg)), title=f"(i) Step 5: after merging: {mr.n_floes_after} floes")
fig.suptitle("Fig. 5.14 (b)-(f), (i)  Watershed + neighbouring-region merging (watershed_based/main.m)")
plt.show()

for ln in mr.lines:
    ep = "; ".join(f"({r + 1}, {c + 1})" for r, c in ln.endpoints.tolist())
    print(f"line {ln.label}: {ln.n_pixels} px, ending points {ep} (1-based), merged region {int((ln.region > 0).sum())} px with "
          f"{ln.concave.shape[0]} concave boundary points, concave ending points {ln.concave_endpoints.shape[0]} -> "
          f"{'REMOVED (regions merged)' if ln.removed else 'kept'}")
print(f"{mr.n_removed} of {mr.num} lines removed; floes {mr.n_floes_before} -> {mr.n_floes_after}"
      f"{book('3 junction lines (22, 22, 19 px); the two horizontal lines are removed, the junction line is kept: 4 -> 2 floes')}"
      f"{matlab_ref('ending points (64,11) (64,32) / (70,12) (70,33) / (48,29) (62,33); REMOVED = [1 1 0]; every array 0 px')}")

# --- Fig. 5.14 (g)(h): the decision for the first lines (region with its concave points, and the segmentation after it)
show = mr.lines[:3]
seg_running = mr.seg0.copy()
fig, axes = plt.subplots(len(show), 2, figsize=(11, 5.4 * len(show)))
axes = np.atleast_2d(axes)
for ln in mr.lines:
    if ln.removed:
        seg_running[ln.pixels[:, 0], ln.pixels[:, 1]] = True
    if ln in show:
        ax0, ax1 = axes[show.index(ln)]
        imshow_matlab(ax0, ln.region > 0, title=f"(g) line {ln.label}: reconstructed neighbouring region")
        ax0.plot(ln.concave[:, 1], ln.concave[:, 0], "c.", ms=7, label="concave boundary points")
        ax0.plot(ln.endpoints[:, 1], ln.endpoints[:, 0], "r+", ms=10, mew=1.5, label="ending points")
        ax0.plot(ln.pixels[:, 1], ln.pixels[:, 0], "y.", ms=3, label="junction line")
        ax0.legend(loc="lower right", fontsize=7)
        imshow_matlab(ax1, label2rgb(label_components(seg_running)),
                      title=f"(h) after line {ln.label}: {'removed (merged)' if ln.removed else 'kept'} -> {int(label_components(seg_running).max())} regions")
fig.suptitle(f"Fig. 5.14 (g)(h)  Step 5 line by line (first {len(show)} of {mr.num} lines; seg is updated inside the loop as in main.m)")
plt.show()
''')
md(r"""
The lines that cut straight across a floe end in convex boundary points of the merged region and are deleted; the
line at the true junction ends in the two notches, which are concave, and survives. Because `seg` is updated inside
the loop, a later line is judged against regions that earlier decisions already merged.
""")

# =====================================================================================================================
# 5.2.1.1 Boundary tracing
# =====================================================================================================================
md(r"""
## 5.2.1.1 Boundary tracing (pp. 99–102) — Fig. 5.16; `boundaries.m` (DIPUM, = `ch2/chain code/boundaries.m`)

Concave detection works on the object's closed outer boundary, traced as an ordered list of pixels by the
Moore-neighbour "bug" algorithm: start at the uppermost-leftmost object pixel $b_0$ with its west neighbour as the
backtracking pixel $p_0$; from $p$, scan the 8 neighbours of $b$ clockwise until the first object pixel $b_1$ is found
and remember the last background pixel visited as the new $p$; repeat from $b_1$; stop (Jacob's rule) when the
current pixel is $b_0$ again *and* the next boundary pixel would be $b_1$ again. Fig. 5.16 walks through the 6×6
example: $b_0 = (2,3) \to b_1 = (2,4) \to b_2 = (3,5) \to \dots$ Only the exterior boundary is traced; holes must be
filled first (§7.1.3). The DIPUM `boundaries.m` used by the scripts starts at the first object pixel in **column-major**
order — (3,2) for Fig. 5.16 — so its closed list is a rotation of the book's sequence.
""")
code(r'''
from seaice.core.chaincode import boundaries, fchcode

img16 = synth.FIG_5_16_IMAGE
b16 = boundaries(img16, 8, "cw")[0]                   # closed: last point == first point
order = np.zeros(img16.shape, dtype=int)
for k, (r, c) in enumerate(b16[:-1].tolist()):
    order[r, c] = k + 1
fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
show_matrix(axes[0], img16.astype(int), title="Fig. 5.16  the 6×6 object", fontsize=10)
show_matrix(axes[1], order, title="visit order of boundaries(B, 8, 'cw') (0 = not on the boundary)", fontsize=10, highlight=order > 0)
plt.show()
seq = [(int(r) + 1, int(c) + 1) for r, c in b16[:-1].tolist()]
trip = [tuple(t) for t in synth.FIG_5_16_TRACE_MATLAB]
i0 = seq.index(trip[0])
print("traced sequence (1-based row, col):", seq)
print(f"book's b0 → b1 → b2 = {trip} is a contiguous sub-sequence: {seq[i0:i0 + 3] == trip} (DIPUM starts at column-major first pixel {seq[0]})")

b_all = boundaries(BW, 8, "cw")                       # chaincode_corner.m: b = boundaries(B, 8, 'cww') on the Otsu mask
b = b_all[0]
c = fchcode(b)
print(f"{IMG_LABEL}: {len(b_all)} object boundary(ies); b{{1}} has {b.shape[0]} points (closed), {c.fcc.size} Freeman codes, start x0y0 = {c.x0y0_matlab} (1-based)"
      f"{matlab_ref('1 object, 263 points, 262 codes, start (35, 11) — identical')}")
''')

# =====================================================================================================================
# 5.2.1.2 Differential chain code
# =====================================================================================================================
md(r"""
## 5.2.1.2 Differential chain code (pp. 103–104) — Eqs. (5.9)–(5.19), Fig. 5.17; `ch5/chaincode_corner.m`, `watershed_based/freeman_concave.m`

An 8-direction Freeman code $C(i)$ (§2.7) is too coarse to measure curvature, so the book builds up a smoothed
turning measure along the clockwise boundary. The **relative code** is the rotation between successive codes in units
of 45°,

$$R(i) = [C(i) - C(i-1) + 8] \bmod 8, \qquad R(i) > 4 \Rightarrow R(i) := R(i) - 8 \quad (5.9),$$

the **absolute code** accumulates it, $A(0) = 0,\ A(i) = A(i-1) + R(i)$ (5.10), and one full clockwise turn gives
$A(0) - A(N) = 8$ (5.11). Summing three consecutive absolute codes, $S(i) = A(i) + A(i-1) + A(i-2)$ (5.12), with the
wrap-around corrections $S(0) = A(0) + A(N-1) + A(N-2) + 16$ (5.13), $S(1) = A(1) + A(0) + A(N-1) + 8$ (5.14) and
$S(0) - S(N) = 24$ (5.15), gives a 24-direction "absolute chain code sum"; its difference three nodes apart is the
**differential chain code**

$$D(i) = S(i+3) - S(i) \quad (5.16), \qquad D(N-j) = S(3-j) - 24 - S(N-j),\ j = 1, 2, 3 \quad (5.17\text{–}5.18),$$

proportional to the turning angle $\theta = 15° \cdot D(i)$ (5.19). Tracing clockwise, a positive $D$ is a concave turn;
the robust rule of the book declares node $i$ **concave if $3 \le D(i) \le 10$** (45°–150°). Fig. 5.17 marks the
concave points of Fig. 5.8(a): two clusters, one at each notch of the junction. `differential_chain_code` is the
literal loop of the script (integer arithmetic, identical to MATLAB), `freeman_concave` wraps Otsu → `boundaries` →
`fchcode` → rule.
""")
code(r'''
from seaice.ch05_watershed import freeman_concave, differential_chain_code

fc = freeman_concave(I)                               # chaincode_corner.m on the RGB (Otsu on all planes, b{1}, rule 3..10)
lo, hi = BOOK_PARAMS["concave_min"], BOOK_PARAMS["concave_max"]
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
imshow_matlab(axes[0], fc.bim, title="bim = bound2im(b, M, N, min(b(:,1)), min(b(:,2))) + concave points (red)")
axes[0].plot(fc.points[:, 1], fc.points[:, 0], "r.", ms=9)
axes[0].plot(fc.code.x0y0[1], fc.code.x0y0[0], "g+", ms=11, mew=1.5, label="starting point b0")
axes[0].legend(loc="lower right", fontsize=8)
axes[1].plot(np.arange(1, fc.Diff.size + 1), fc.Diff, "k-", lw=1)
axes[1].axhspan(lo, hi, color="r", alpha=0.15, label=f"concave band {lo} ≤ D ≤ {hi}  ({lo * 15}°–{hi * 15}°)")
axes[1].plot(fc.index + 1, fc.Diff[fc.index], "r.", ms=8)
axes[1].set_xlabel("boundary node i"); axes[1].set_ylabel("differential chain code D(i)  (θ = 15°·D, Eq. 5.19)")
axes[1].legend(loc="upper right", fontsize=8)
fig.suptitle("Fig. 5.17  Concave points of the floe boundary by the differential chain code (chaincode_corner.m)")
plt.show()
print(f"boundary of {fc.boundary.shape[0]} points, {fc.code.fcc.size} codes; Eq. 5.11: A(N) = {int(fc.A[-1])} (one clockwise turn = −8); "
      f"Diff in [{int(fc.Diff.min())}, {int(fc.Diff.max())}]; {fc.points.shape[0]} concave points"
      f"{book('two gray dots at the junction notches')}{matlab_ref('9 points (48,27) (49,28) (48,29) (47,30) (62,35) (62,34) (62,33) (63,32) (64,32), all arrays identical')}")
print("concave points (1-based row, col):", [tuple(p) for p in fc.points_matlab.tolist()])
k = min(10, fc.code.fcc.size)
print(f"first {k} nodes:  C = {fc.code.fcc[:k].tolist()}\n                 R = {fc.R[:k].tolist()}  (Eq. 5.9)\n                 A = {fc.A[:k].tolist()}  (Eq. 5.10)"
      f"\n                 S = {fc.S[:k].tolist()}  (Eqs. 5.12-5.14, S(0) normalised to 0)\n              Diff = {fc.Diff[:k].tolist()}  (Eqs. 5.16-5.18)")

# --- sanity on synthetic shapes: a digital square has no concave point, an L-shape has them only at its inner corner
sq = np.zeros((30, 30), dtype=bool); sq[5:25, 5:25] = True
Lsh = sq.copy(); Lsh[5:15, 15:25] = False
for name, m in [("square", sq), ("L-shape", Lsh)]:
    r = freeman_concave(m)
    print(f"{name}: {r.points.shape[0]} concave points at {[tuple(p) for p in r.points_matlab.tolist()]}, Diff range [{int(r.Diff.min())}, {int(r.Diff.max())}] (convex corners give D = −6)")
''')

# =====================================================================================================================
# 5.3 Experiments
# =====================================================================================================================
md(r"""
## 5.3 Experimental results and discussion (pp. 104–108) — Figs. 5.18–5.20, Table 5.1

The authors apply the §5.2 pipeline to four Ny-Ålesund images (May 2011; the same expedition as Chapters 3–4) after
removing brash ice **manually**, and count floes against a manual segmentation (Table 5.1): the watershed alone gives
12 / 16 / 25 / 60 floes for 5 / 12 / 20 / 38 true ones (over-segmentation 7 / 5 / 7 / 26), merging brings them to
5 / 11 / 17 / 37 with 0 / 0 / 0 / 4 over-segmented and 0 / 1 / 3 / 6 under-segmented floes. The remaining errors
(Fig. 5.19) are of two kinds: a concave floe corner that is not a junction keeps a spurious line, and a line with a
genuinely concave ending point can still be misplaced; under-segmentation (a missing line) cannot be repaired by
merging at all. Higher resolution helps (Fig. 5.20). The method suits images whose junctions are invisible after
binarisation and not crowded marginal-ice-zone scenes with few holes, because the distance transform sees only the mask.

> **The §5.3 images are not shipped** (`unverified` in the report). The cells below run the identical pipeline on the
> Fig. 4.3(a) crop of Chapter 4's `test.jpg` (or its NASA substitute, loaded through `load_image("ch04", "test.jpg")`)
> and on a **synthetic** two-floe image (`synth.two_touching_floes`, stated as synthetic). On the book's crop the port
> is exact vs MATLAB (11 basins, 7 lines, 4 removed, 11 → 7 floes); Table 5.1 itself cannot be reproduced. If a raw
> Otsu mask contains a 1–2-pixel piece of ice that is entirely a watershed line, `main.m` has no boundary to trace
> there (MATLAB indexes `A(l-2)` on an empty code and errors); the cell then reports it and repeats the run after the
> "brash removal" step the authors did by hand — an opening with a 3-pixel disk and `bwareaopen(50)` (Chapter 4
> primitives).
""")
code(r'''
from seaice.ch04_ice_edge_detection import fig_4_3a, FIG_4_3A_CROP_MATLAB
from seaice.core.morphology import imopen
from seaice.core.connectivity import bwareaopen

def merge_and_show(rgb, tag):
    """Run neighboring_region_merging on the Otsu mask of ``rgb`` and draw the Fig. 5.18-style panel (display glue)."""
    bw = otsu_mask(rgb)
    note = "raw Otsu mask"
    t0 = time.perf_counter()
    try:
        res = neighboring_region_merging(bw, BOOK_PARAMS["metric_merging"])
    except ValueError as exc:
        print(f"{tag}: raw Otsu mask rejected — {exc}\n   -> MATLAB's main.m fails the same way on this mask; "
              "repeating after the §5.3 'brash removal' step: imopen(bw, strel('disk', 3)) + bwareaopen(50)")
        bw = bwareaopen(imopen(bw, strel("disk", 3)), 50)
        note = "opened (disk 3) + bwareaopen(50) mask"
        res = neighboring_region_merging(bw, BOOK_PARAMS["metric_merging"])
    dt = time.perf_counter() - t0
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.6))
    imshow_matlab(axes[0], rgb, title=f"{tag}\n{bw.shape[0]}x{bw.shape[1]}, {note}")
    imshow_matlab(axes[1], label2rgb(label_components(res.seg0)), title=f"watershed of -D city-block: {res.n_floes_before} regions")
    imshow_matlab(axes[2], res.f, title=f"{res.num} junction lines (red = removed)")
    for ln in res.lines:
        if ln.removed:
            axes[2].plot(ln.pixels[:, 1], ln.pixels[:, 0], "r.", ms=2)
    imshow_matlab(axes[3], label2rgb(label_components(res.seg)), title=f"after merging: {res.n_floes_after} floes")
    fig.suptitle("Fig. 5.18-style  Watershed + neighbouring-region merging (procedure of §5.3 on a stand-in image)")
    plt.show()
    print(f"{tag}: {int(label_components(bw).max())} connected ice regions -> {int(res.L.max())} basins, {res.num} junction lines, "
          f"{res.n_removed} removed -> floes {res.n_floes_before} -> {res.n_floes_after}  ({dt:.1f} s)")
    return res

try:
    I4, SOURCE4 = load_image("ch04", "test.jpg")     # Chapter 4's photograph (same expedition) or its NASA substitute
    (r0, r1), (c0, c1) = FIG_4_3A_CROP_MATLAB
    crop4 = fig_4_3a(I4)                              # Fig. 4.3(a): im(1600:2151, 1979:2552)
    tag4 = f"ch04 test.jpg crop im({r0}:{r1}, {c0}:{c1})" if SOURCE4.startswith("book") else f"ch04 substitute, window im({r0}:{r1}, {c0}:{c1})"
    res4 = merge_and_show(crop4, tag4)
    if SOURCE4.startswith("book"):
        print("  [MATLAB R2025a on this crop: 11 basins, 7 lines (958 px), REMOVED = [0 1 1 1 1 0 0], 11 -> 7 floes; all arrays identical]")
except Exception as exc:                              # no private copy and the substitute could not be fetched
    print(f"ch04 test.jpg unavailable ({type(exc).__name__}: {exc}) — put data/book/ch04/test.jpg next to the repository "
          "(or in MyDrive/Sea_Ice_Colab/data/book/ch04/) or allow the NASA download; continuing with the synthetic image")

S = synth.two_touching_floes()                        # SYNTHETIC: two convex bright floes touching along a short neck (96x81 RGB, seed 0)
resS = merge_and_show(S, "synthetic two_touching_floes (synthetic image)")
print("  [MATLAB R2025a on the synthetic image: 2 basins, 1 line (31 px) kept, 2 floes; all arrays identical]")
''')
md(r"""
On a real crop the pipeline behaves as Table 5.1 describes: the city-block watershed over-segments every large
floe into several regions, merging deletes the lines that end in convex points and keeps the ones ending in notches.
The counts are for the image loaded here, not the book's; the synthetic image is the minimal case — one true junction
with concave ending points, so its single line is kept.
""")

# =====================================================================================================================
# Parameter play (optional)
# =====================================================================================================================
md(r"""
## Parameter play (optional)

The two knobs of the marker-controlled watershed on the loaded mask: the **distance metric** of the inverse map and
the **disk radius** used to dilate the regional minima into markers (Fig. 5.12(b)). A radius that is too small leaves
several markers per floe (over-segmentation); one that is too large bridges the two floes (under-segmentation).
Requires `ipywidgets`; the cell prints a message and falls back to the static preview if it is not installed.
""")
code(r'''
def play(metric="cityblock", r=5):
    mk = marker_watershed(BW, metric, r, min_area=BOOK_PARAMS["min_area"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    imshow_matlab(axes[0], mk.marker_overlay, title=f"{metric}: {mk.n_minima} minima -> {mk.n_markers} markers (disk r = {r})")
    imshow_matlab(axes[1], mk.ridge, title=f"watershed of the imposed map: {mk.n_basins} basins")
    imshow_matlab(axes[2], label2rgb(mk.labels), title=f"{mk.n_floes} floes after bwareaopen({BOOK_PARAMS['min_area']})")
    plt.show()

play()                                                # static preview (book values) — visible without widgets
try:
    from ipywidgets import interact, IntSlider, Dropdown
    interact(play, metric=Dropdown(options=["cityblock", "chessboard", "euclidean", "quasi-euclidean"], value="cityblock"),
             r=IntSlider(5, 1, 15, 1))
except ImportError:
    print("ipywidgets not installed - only the static preview above is shown")
''')

# =====================================================================================================================
# MATLAB <-> Python map + summary / feeds forward
# =====================================================================================================================
md(r"""
## MATLAB ↔ Python mapping used in this chapter

| MATLAB | Python (`seaice`) | Parity | Note |
|---|---|---|---|
| `watershed(A[, conn])` | `core.watershed.watershed(A, conn=8)` → int32 labels, 0 = line | exact (74 fixture cases + every script image, label values included) | Meyer flooding seeded by a column-major scan of `imregionalmin`; no propagation from line pixels; MATLAB rejects int16/int32 input, the port accepts any real dtype; 2-D only |
| — | `core.watershed.watershed_skimage` | approx | `skimage.segmentation.watershed(watershed_line=True)`: different queue rules → 46–122 ridge px differ; comparison only |
| `imregionalmin`, `imregionalmax` | `core.morphology.imregionalmin / imregionalmax` | exact (126 cases) | `skimage.morphology.local_minima/maxima(allow_borders=True)`; constant image → all true; NaN rejected |
| `imimposemin(I, BW[, conn])` | `core.morphology.imimposemin` | exact (88 cases, bit-identical incl. single) | M-code port: `±Inf` markers, `h = 0.001·range` (float) / `1` (integer), arithmetic in the input class; NaN rejected (MATLAB does not) |
| `bwdist(~bw, metric)` (single) | `core.distance.bwdist`; `ch05_watershed.inverse_distance` keeps float32 | exact (city/chess), ≤ 1e-4 in single (Euclidean, quasi) | float32 keeps the flooding priorities identical to MATLAB |
| `im2bw(rgb, graythresh(rgb))` | `ch05_watershed.otsu_mask` → `core.threshold.graythresh / im2bw` | exact | RGB `graythresh` histograms all planes (130/255) ≠ gray (128/255) |
| `fspecial('sobel')` + `imfilter(double(I), h, 'replicate')` | `ch05_watershed.sobel_magnitude` → `core.filters` | exact (≤ 1e-12) | unscaled kernels, 0–255 input (unlike `edge`) |
| `imopen / imclose(g, ones(7,7))` | `core.morphology.imopen / imclose` with a bool all-ones SE | exact | `imclose` zero pre-pad path (Chapter 4) |
| `strel('disk', 5)`, `imdilate` | `core.morphology.strel('disk', 5)` (9×9, 69 px octagon), `imdilate` | exact | never `skimage.morphology.disk` |
| `bwareaopen(img, 5)`, `bwlabel(f, 4)`, `bwlabel` | `core.connectivity.bwareaopen`, `label_components(f, 4)` | exact | numbering = MATLAB's column-major first pixel |
| `bitand(logical, logical)`, `logical .* logical`, `find`, `intersect(…, 'rows')` | `&`, `&`, `_find_rows_cols` (column-major order), `_intersect_rows` (sorted unique rows) | exact | `bitand` of logicals is logical; `.*` gives double |
| `imfilter(double(g), [0 -1 0; -1 4 -1; 0 -1 0])`, `abs(…) >= max` | `junction_endpoints(mask, rule='max' \| 'ge3')`, `ENDPOINT_KERNEL` | exact | the literal `abs` also admits background pixels with ≥ 3 line neighbours; the text's rule is `'ge3'` |
| `imreconstruct(g, bwlabel(neighbor))` | `core.morphology.imreconstruct` (Chapter 4) | exact | grayscale reconstruction of the 0/1 line under the label image = the merged region |
| `boundaries(B, 8, 'cww')`, `fchcode`, `bound2im` | `core.chaincode.boundaries(B, 8, 'cw')`, `fchcode`, `bound2im` (Chapter 2) | exact | `'cww'` is a typo = anything but `'ccw'` = clockwise |
| chain-code loops of `chaincode_corner.m` / `freeman_concave.m` | `differential_chain_code`, `freeman_concave(I, object='first')` | exact | `b{1}` = first object (the script computes and ignores the longest) |
| `regionprops(…, 'Centroid')`, `floor` | `component_centroids` (1-based (x, y)) | exact | commented block of `marker_watershed.m` |
| `label2rgb(L, 'jet', 'k', 'shuffle')`, `surf(…, 'texturemap')`, `imcontour`, `image(…, 'scaled')` | `core.plotting.label2rgb / surface_plot / contour_overlay` | display only | MATLAB's shuffle stream is not reproducible |
""")
md(r"""
## Summary — and what the next chapters need from this one

**What chapter 5 established.** A gray image read as a topographic surface is partitioned by the watershed
transform into catchment basins of its regional minima (Eqs. 5.1–5.8); the gray image and its gradient are poor
segmentation functions for floes (hundreds of basins, and no ridge where the junction is invisible), the **inverse
distance map of the ice mask** is a good one (one minimum per floe, a ridge at the neck), and its residual
over-segmentation is removed either by **marker imposition** (dilated minima → `imimposemin` → one basin per marker,
Fig. 5.12) or by the authors' **neighbouring-region merging** (watershed lines whose two ending points are convex —
differential chain code $D(i)$ outside 3…10 — are deleted, Figs. 5.13–5.17). City-block is the recommended metric;
the method assumes convex floes and junction-free binarisation, and cannot fix under-segmentation.

**Parity (from `reports/ch05_verification.md`, MATLAB R2025a as reference).** All 15 `.m` files are `exact`: every
script image (labels, ridges, minima, imposed maps, per-line merging records) is 0 px from MATLAB on `q.jpg`, on a
synthetic image and on a 552×574 real crop; `core.watershed.watershed` is exact on 74 constructed plateau / tie /
corridor / random cases with label values included and its flooding order is pinned by 31 order-sensitive fixtures;
`imregionalmin/max` exact on 126 cases, `imimposemin` bit-identical on 88 (single included). `approx`:
`watershed_skimage` (comparison only). `reimplemented`: Eq. 5.2 `regional_minima_by_reconstruction` (constant and
all-`±Inf` images special-cased like `imregionalmin`), `impose_minima_book`, `watershed_immersion`. `unverified`:
Figs. 5.7, 5.11, 5.18–5.20 and Table 5.1 (images not shipped). 399 chapter tests pass (no xfail); full suite 1145
passed (2 skipped, 1 xfailed — all from earlier chapters). Review: 1 must-fix + 9 should-fix + 2 verifier items applied.

**Feeds forward (`seaice/` functions the later chapters import):**

| Needed by | Primitive |
|---|---|
| Ch6 (GVF snake, §6.3 automatic contour initialisation) | the ch5 segmentation seeds the snake: separated floe masks (`neighboring_region_merging(...).seg`, `marker_watershed(...).seg_ao`, `distance_watershed(...).seg_ao`) and/or the distance-transform minima (`imregionalmin(inverse_distance(bw))`, dilated with `strel('disk', r)` as in Fig. 5.12(b)); `core.chaincode.boundaries` of each `label_components` floe as the initial contour; `component_centroids` for seed points; `imdilate` of contour masks. Keep distance maps float32 |
| Ch7 (ice types, shape enhancement) | `core.watershed.watershed`, `imregionalmax` of distance maps for floe centres, `imimposemin`, `neighboring_region_merging` as the floe-separation step before classification (convex floes, brash removed first); `label2rgb` for label displays; hole filling (`imfill`) still to be built on `imreconstruct` |
| Ch8–Ch9 (applications, model ice) | separated floe masks → floe size distributions and ice concentration per floe; `watershed` + `bwdist` on the rectangular model-ice floes; `component_centroids` (other `regionprops` properties still to be ported with MATLAB's algorithms) |

**Open thread — speed.** `core.watershed.watershed` is a pure-Python heap flood (≈ 3 s per Mpx): instant on this
chapter's 96×81 image and the 552×574 crop, but a 12-Mpx frame of Chapters 7–9 would take ~40 s per call. Later
chapters will either add a faster engine that reproduces the same 74 reference cases (label values included) or use
`watershed_skimage` for large images, explicitly labelled `approx`, keeping the exact engine for parity tests.

**Pitfalls to carry forward.** `graythresh` on an RGB image histograms all three planes (its level differs from the
gray image's); `bwdist` is *single* and the flooding priorities depend on it — keep float32; MATLAB's `watershed`
seeds its queue in column-major order and never propagates from line pixels (skimage does both differently, so it is
only `approx`); `imimposemin` uses `−Inf` markers and `h = 0.001·range`, not the book's literal +1 (same watershed);
`strel('disk', r)` is an octagon; `bwlabel(f, 4)` is needed for 4-connected watershed lines; `main.m` updates `seg`
inside its loop and its ending-point rule is `abs(imfilter) >= max`, not the text's `≥ 3`; `freeman_concave` uses the
*first* object of the mask and errors on boundaries shorter than 3 codes (so isolated 1–2-px ice pieces on a line must
be cleaned away first, as the authors did manually).

*Optional: to regenerate the MATLAB references yourself, run `reference/ch05/make_refs.py` (uses
`tools/run_matlab_ref.py`, MATLAB `-batch`); the notebook does not need MATLAB or Octave.*
""")

out = HERE / NB_NAME
nbf.write(nb, out)
print(f"wrote {out.as_posix()} ({len(C)} cells)")
