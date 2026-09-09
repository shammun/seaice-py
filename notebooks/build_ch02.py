"""Build ``notebooks/ch02_preliminaries.ipynb`` with nbformat (re-runnable; the notebook is never edited by hand).

Chapter 2 — Digital Image Processing Preliminaries (Zhang & Skjetne 2018, pp. 11–36).
Follows the ``colab-notebook`` skill: title → cell 1 (Drive mount / cwd) → cell 2 (clone or pull the public repo)
→ cell 3 (``load_image``: private copy or public-domain substitute) → one section per book section
(2.1 … 2.8) → optional parameter play → summary / feeds forward.  All algorithms are imported from ``seaice``.

Run:  ``.venv/Scripts/python.exe notebooks/build_ch02.py``
Verify: ``.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900
        --output executed_ch02.ipynb notebooks/ch02_preliminaries.ipynb``  (then delete the executed copy).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

NB_NAME = "ch02_preliminaries.ipynb"
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
# Chapter 2 — Digital Image Processing Preliminaries

*Sea Ice Image Processing with MATLAB* (Q. Zhang & R. Skjetne, CRC Press 2018), Chapter 2, pp. 11–36 —
Python port `seaice-py`, notebook `notebooks/ch02_preliminaries.ipynb`.

**Sections covered** (one notebook section each, in book order):
2.1 Image types · 2.2 Image histogram · 2.3 Basic relationships between pixels · 2.4 Distance transform ·
2.5 Convolution · 2.6 Set and logical operations · 2.7 Chain code · 2.8 Image interpolation.

**MATLAB files this notebook replaces** (`MATLAB_ROOT/ch2/`):

| MATLAB file | Python (imported below) | Parity (from `reports/ch02_verification.md`) |
|---|---|---|
| `color_image.m` | `seaice.ch02_preliminaries.color_components` → `seaice.core.color.rgb2cmy / rgb2hsi / rgb2cmyk` | exact (CMY, I, S); HSI hue **reimplemented** from Eq. (2.6a) — the script has a bug, see §2.1 |
| `histogram.m` | `gray_histogram`, `channel_histograms` → `seaice.core.histogram.imhist` | exact |
| `distance_transform.m` | `point_distance_maps` → `seaice.core.distance.bwdist` | exact (quasi-euclidean: reimplemented / near) |
| `chain code/boundaries.m`, `fchcode.m`, `bound2im.m`, `chain_diff.m` | `chain_code_demo` → `seaice.core.chaincode.boundaries / fchcode / bound2im` | exact (reimplemented line by line); `min_magnitude` tie-break on periodic codes is a documented deviation |
| *(text only: §2.3, §2.5, §2.6, §2.8, Figs 2.1–2.2, 2.6)* | `seaice.core.connectivity`, `filters`, `setops`, `interp`, `synth` | exact vs MATLAB `bwlabel`, `conv2`, `imfilter`, `interp2`; `resize` vs `imresize` is **approx** |

The chapter's learned knowledge (concepts, primitives, pitfalls, what later chapters need) is in
`knowledge/ch02.md` (written by the knowledge phase) and `knowledge/CUMULATIVE.md`.

> **Data.** Chapter 2 uses one photograph, the book's `rgb.JPG` (2048×1536 RGB, Figs. 2.3 / 2.8). The book's
> images are copyrighted and are **not** in this public repository: `seaice.core.io.load_image()` first looks for
> your own private copy (`data/book/ch02/rgb.JPG` next to the repository, or `MyDrive/Sea_Ice_Colab/data/book/ch02/`
> on Colab) and otherwise downloads a public-domain NASA MODIS scene of the Beaufort Sea marginal ice zone with the
> same size. Every other input (7×7 and 10×13 matrices, the 8×9 chain-code object, the 201×201 point image, the
> 16×16 sets) is a transcribed book fixture or a synthetic array from `seaice.core.synth`. The grayscale image of the
> printed Fig. 2.7 is not shipped with the book's code either.
""")

# =====================================================================================================================
# 2. Setup cell (Colab / Drive / local)
# =====================================================================================================================
md(r"""
## Setup (Google Colab or local Jupyter)

Run the two cells below first. On **Colab** the first cell mounts your Google Drive and moves into
`MyDrive/Sea_Ice_Colab` — the folder where your private copy of the book images lives (`data/book/ch02/rgb.JPG`)
and where downloads are cached between sessions; readers without Drive get a temporary `/content/Sea_Ice_Colab`.
The second cell clones (or updates) the public repository `seaice-py` there and installs its requirements.
**Locally** both cells are no-ops that move to the repository root. No GPU is needed.
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

`seaice.core.io.load_image("ch02", "rgb.JPG")` replaces MATLAB's `imread('rgb.jpg')`. It looks for **your private
copy of the book image** first — `data/book/ch02/rgb.JPG` in the current folder (on Colab: `MyDrive/Sea_Ice_Colab`),
in the repository, or explicitly in Drive — and otherwise downloads a **public-domain NASA substitute** (MODIS/Terra
true-colour scene of the Beaufort Sea marginal ice zone, 25 July 2019, same 2048×1536 size) into
`data/online/ch02/`. It prints which source it used and returns the label, so the cells below only quote the
book's numbers when the book's own image is loaded. Synthetic / transcribed fixtures come from `seaice.core.synth`.
""")
code(r'''
from seaice.core.io import load_image

I, SOURCE = load_image("ch02", "rgb.JPG")            # = imread('rgb.jpg'); uint8 (M, N, 3)
FROM_BOOK = SOURCE.startswith("book")
book = (lambda s: f"  (book: {s})") if FROM_BOOK else (lambda s: "")   # book-quoted values only for the book image
if not FROM_BOOK:
    print("Running on a public-domain substitute image; figures show the same operations, "
          "but values quoted in the book only hold for the book's own image.")
print("image:", I.shape, I.dtype)
print(f"Fig. 2.3 pixel I(1076, 675) [1-based] = {I[1075, 674].tolist()}{book('28, 76, 114')}")
''')

md(r"""
## Imports

Everything algorithmic comes from the `seaice` package; the notebook only calls it and draws figures.
(`seaice.core.plotting` selects the non-interactive Agg backend for the command-line scripts, so `%matplotlib inline`
is issued *after* the `seaice` imports to switch back to inline display.)
""")
code(r'''
import numpy as np
import seaice.core                                    # imports seaice.core.plotting (sets the Agg backend)
from seaice.core import synth
from seaice.core.plotting import imshow_matlab, show_matrix   # MATLAB imshow scaling / value grids (display only)
from seaice.core.matlab_compat import rgb2gray_matlab

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100

G = rgb2gray_matlab(I)     # rgb2gray with MATLAB's Rec.601 weights and rounding (verified: 0 of 3 145 728 px differ)
print("gray image", G.shape, G.dtype, "levels", int(G.min()), "..", int(G.max()))
''')

# =====================================================================================================================
# 2.1 Image types
# =====================================================================================================================
md(r"""
## 2.1 Image types (pp. 11–16) — replaces `ch2/color_image.m` (Figs. 2.4–2.5) and the channel part of `ch2/histogram.m` (Fig. 2.3)

A digital image is an $M \times N$ matrix $f(x, y)$, $1 \le x \le M$ (row), $1 \le y \le N$ (column) — Eq. (2.1) —
which is exactly MATLAB's `I(x, y)` indexing. A **grayscale** image stores $B$-bit integers in $[0, 2^B - 1]$
(Fig. 2.1, 8 bits → 256 levels), $B = 1$ gives a **binary** (logical) image (Fig. 2.2), and an **indexed** image is an
integer matrix whose entries point into an $m \times 3$ colormap with rows in $[0, 1]$ (Fig. 2.6).
An **RGB** image is $M \times N \times 3$ with $\mathbf{C}(x,y) = [R(x,y),\, G(x,y),\, B(x,y)]^T$ (Eq. 2.2, Fig. 2.3).
The CMY components are the complements of the normalised RGB values, Eq. (2.3)
$[C, M, Y]^T = [1, 1, 1]^T - [R, G, B]^T$ (on `uint8` this is `imcomplement`, i.e. $255 - I$), and CMYK adds a black
component through Eqs. (2.4)–(2.5), $K_b = \min(1-R,\, 1-G,\, 1-B)$, with an under-colour-removal factor $u$ and a
darkness factor $b$.
HSI uses the linear transform of Eq. (2.6a)

$$\begin{bmatrix} I \\ V_1 \\ V_2 \end{bmatrix} =
\begin{bmatrix} 1/3 & 1/3 & 1/3 \\ -1/\sqrt6 & -1/\sqrt6 & 2/\sqrt6 \\ 1/\sqrt6 & -1/\sqrt6 & 0 \end{bmatrix}
\begin{bmatrix} R \\ G \\ B \end{bmatrix}, \qquad
H = \arctan\frac{V_2}{V_1} \;\;(2.6\text{b}), \qquad S = \sqrt{V_1^2 + V_2^2} \;\;(2.6\text{c}).$$

> **`matlab_bug` switch.** Line 21 of `color_image.m` computes `V1 = -Ir/√6 - Ig/√6 + 2*Ig/√6` — `Ig` where
> Eq. (2.6a) has `Ib` — so $V_1 = -V_2$ and the hue is the constant $\arctan(-1) = -\pi/4$ everywhere. That is why the
> printed Fig. 2.5(a) is featureless. `seaice.core.color.rgb2hsi` follows the **book equation** by default (the text is
> the authority on *what* the algorithm does) and offers `matlab_bug=True` only to reproduce the printed panel; both
> variants were checked against MATLAB R2025a (exact).
""")
code(r'''
from seaice.ch02_preliminaries import image_type_examples

ex = image_type_examples(I)      # gray / binary / indexed examples (Figs. 2.1, 2.2, 2.6)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
imshow_matlab(axes[0], ex["gray"], title="Grayscale image rgb2gray(I): 8 bits, 256 levels")
axes[0].add_patch(plt.Rectangle((899.5, 199.5), 11, 11, fill=False, edgecolor="r", lw=1.5))
shade = lambda m: (m.astype(float) - 150) / 105       # display shading only
show_matrix(axes[1], ex["gray_crop"], fmt="{:d}", cmap="Greys", highlight=shade(ex["gray_crop"]),
            title="11x11 window (rows 201-211, cols 901-911, 1-based)")
show_matrix(axes[2], ex["fig_2_1_printed"], fmt="{:d}", cmap="Greys", highlight=shade(ex["fig_2_1_printed"]),
            title="Values printed in Fig. 2.1 (location unknown)")
fig.suptitle("Fig. 2.1  Pixel values in a grayscale image")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
show_matrix(axes[0], ex["pattern"].astype(int), cmap="Greys", title="Fig. 2.2  synthetic 16x16 binary pattern (B = 1)")
imshow_matlab(axes[1], ex["binary"], title="rgb2gray(I) > 128  (logical image)")
axes[2].imshow(ex["indexed_rgb"], interpolation="nearest")
axes[2].set_axis_off()
axes[2].set_title("Fig. 2.6  indexed image -> RGB through its colormap")
plt.show()
print("index matrix:\n", ex["indexed"], "\ncolormap rows:\n", ex["cmap"])
''')
md(r"""
The 11×11 window shows what "gray value" means numerically (bright ice is close to 255, water close to 0); the printed
Fig. 2.1 values come from an unknown location of a different image, so only the *form* is compared. Thresholding the gray image at 128 gives a
first, crude ice/water binary image — Chapter 3 will do this properly.
""")
code(r'''
from matplotlib.colors import ListedColormap
from seaice.ch02_preliminaries import channel_histograms, color_components

ch = channel_histograms(I)       # R, G, B planes + histograms (histogram.m lines 1-20, 50-58)
res = color_components(I)        # CMY / CMYK / HSI (color_image.m)

def ramp(k):                     # colormap([0:1/255:1]', zeros(256,1), zeros(256,1)) of histogram.m (display glue)
    cm = np.zeros((256, 3)); cm[:, k] = np.arange(256) / 255.0
    return ListedColormap(cm)

# --- Fig. 2.3: RGB image and its three channel images ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
imshow_matlab(axes[0, 0], I, title="Color image")
for ax, key, k, title in zip(axes.flat[1:], ("R", "G", "B"), (0, 1, 2),
                             ("Red component image", "Green component image", "Blue component image")):
    ax.imshow(ch[key], cmap=ramp(k), vmin=0, vmax=255); ax.set_axis_off(); ax.set_title(title)
px = ch["pixel_1076_675"]
for ax in axes.flat:
    ax.plot(674, 1075, ".k", markersize=8)
fig.suptitle(f"Fig. 2.3  Pixel values in an RGB image: R(1076,675)={px[0]}, G={px[1]}, B={px[2]}{book('28, 76, 114')}")
plt.show()

# --- Fig. 2.4: C, M, Y components (Eq. 2.3) -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, key, title in zip(axes, ("Ic", "Im", "Iy"), ("(a) Cyan", "(b) Magenta", "(c) Yellow")):
    imshow_matlab(ax, res[key], autoscale=True, title=f"{title} component image")   # imshow(x, [])
fig.suptitle("Fig. 2.4  CMY components, Eq. (2.3):  [C; M; Y] = [1; 1; 1] - [R; G; B]")
plt.show()
r, c = 1075, 674
print(f"pixel (1076,675): RGB = {I[r, c].tolist()},  CMY (uint8 imcomplement) = {res['I_cmy'][r, c].tolist()}")
''')
code(r'''
# --- Fig. 2.5: HSI components — book equation (default) vs the script's buggy line 21 -------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
for row, keys, note in ((0, ("H", "S", "I"), "Eq. (2.6a-c) as printed  (V1 uses 2B/sqrt(6))"),
                        (1, ("H_matlab", "S_matlab", "I_matlab"),
                         "color_image.m line 21 (2*Ig instead of 2*Ib): H = atan(-1) = -pi/4 everywhere")):
    for ax, key, title in zip(axes[row], keys, ("Hue", "Saturation", "Intensity")):
        imshow_matlab(ax, res[key], autoscale=True, title=f"{title}  [{key}]")
    axes[row, 1].text(0.5, 1.18, note, transform=axes[row, 1].transAxes, ha="center", fontsize=11)
fig.suptitle("Fig. 2.5  HSI components: top = book Eq. (2.6), bottom = what color_image.m computes (printed figure)", y=1.02)
plt.show()

Hb = res["H_matlab"]
finite = np.isfinite(Hb)
print("script hue: unique finite values =", np.unique(np.round(Hb[finite], 6)).tolist(),
      f"(= -pi/4 = {-np.pi/4:.6f}), NaN fraction = {(~finite).mean():.6f}")
print(f"book hue at (1076,675): H = {res['H'][r, c]:.6f} rad, S = {res['S'][r, c]:.6f}, I = {res['I'][r, c]:.6f}")

# --- CMYK, Eqs. (2.4)-(2.5), text only (no MATLAB code) -------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
for k, (ax, title) in enumerate(zip(axes, ("C", "M", "Y", "K"))):
    imshow_matlab(ax, res["CMYK"][..., k], title=f"{title}  (u = 1, b = 1)")
fig.suptitle("CMYK components, Eqs. (2.4)-(2.5):  Kb = min(1 - R, 1 - G, 1 - B)")
plt.show()
print("CMYK at (1076,675):", np.round(res["CMYK"][r, c], 4).tolist())
''')
md(r"""
With the book's Eq. (2.6a) the hue carries faint structure (open water vs. ice), while the script's version is a flat
$-\pi/4$ — the panel that was printed. Saturation and intensity are identical in both variants (exact vs MATLAB).
""")

# =====================================================================================================================
# 2.2 Histogram
# =====================================================================================================================
md(r"""
## 2.2 Image histogram (pp. 16–17) — replaces `ch2/histogram.m`

The histogram of an image with $L = 2^B$ levels counts the pixels at every level $r_k$, Eq. (2.7)
$h(r_k) = n_k$, $k = 0, \dots, L-1$; dividing by the number of pixels $n = MN$ gives the normalised histogram,
Eq. (2.8) $p(r_k) = h(r_k)/n = n_k/(MN)$, an estimate of the probability of level $r_k$ (it sums to 1).
`histogram.m` computes Eq. (2.7)/(2.8) with an explicit loop (`length(find(I == k))`) and again with `imhist`;
`seaice.core.histogram.imhist` reproduces MATLAB's binning (256 exact levels for `uint8`, 2 bins for logical images).
Histograms are the basis of the thresholding methods of Chapter 3 (Otsu picks the valley between the water and ice modes).

> **Fig. 2.7 is not reproducible.** The printed Fig. 2.7 shows a dense floe field whose histogram peaks at ≈7.2·10⁴
> pixels near level 195; that image is not shipped with the book's code (and matches none of the ch3–ch5 images).
> The left panel below therefore shows the grayscale of the image loaded above (for the book's `rgb.JPG`: peak
> 47 840 at level 208, identical to MATLAB) and is labelled as an illustration, not a reproduction. Fig. 2.8 uses the
> loaded colour image directly.
""")
code(r'''
from seaice.ch02_preliminaries import gray_histogram

gh = gray_histogram(G)           # num (loop, Eq. 2.7), GP (Eq. 2.8), counts/x (imhist)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
imshow_matlab(axes[0], G, title="(a) Grayscale image  [not the printed one: rgb2gray of the loaded image]")
axes[1].bar(gh["x"], gh["counts"], width=1.0, color="k")
axes[1].set(xlim=(-0.5, 255.5), xlabel="Intensity value", ylabel="Number of pixels", title="(b) imhist(I), Eq. (2.7)")
fig.suptitle("Fig. 2.7 (illustrated with the loaded image; the printed figure uses an image that is not shipped)")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
imshow_matlab(axes[0], I, title="(a) Color image")
x = ch["x"]
axes[1].plot(x, ch["y_r"], "r-", x, ch["y_g"], "g--", x, ch["y_b"], "b:", linewidth=2.5)
axes[1].legend(["Red", "Green", "Blue"], loc="best", frameon=False)
axes[1].set(xlim=(0, 255), ylim=(0, 80000), xlabel="Intensity value", ylabel="Number of pixels",
            title="(b) Channel histograms  ([y_r, x] = imhist(Ir) ...)")
fig.suptitle("Fig. 2.8  RGB image and its channel histograms")
plt.show()

print(f"loop count == imhist count: {np.array_equal(gh['num'], gh['counts'])};  sum p(r_k) = {gh['GP'].sum():.12f}")
print(f"gray peak {gh['counts'].max()} @ level {int(gh['counts'].argmax())}"
      f"{book('rgb.JPG 47 840 @ 208; the printed Fig. 2.7 image ~72 000 @ ~195')}")
for key, name, quoted in (("y_r", "R", "77 513 @ 5"), ("y_g", "G", "64 866 @ 212"), ("y_b", "B", "71 090 @ 227")):
    print(f"{name} peak {int(ch[key].max())} @ level {int(ch[key].argmax())}{book(quoted)}")
''')
md(r"""
The channel histograms of a sea-ice photograph are bimodal — dark water at low levels and bright ice near the top of
the range (for the book's `rgb.JPG`: red peak at level 5, green/blue peaks at 212/227) — which is what makes global
thresholding work in Chapter 3. Compare with the peaks printed above for the image you loaded.
""")

# =====================================================================================================================
# 2.3 Basic relationships between pixels
# =====================================================================================================================
md(r"""
## 2.3 Basic relationships between pixels (pp. 17–20) — text only; primitives in `seaice.core.connectivity`

A pixel $p = (x, y)$ has four horizontal/vertical neighbours $N_4(p) = \{(x \pm 1, y), (x, y \pm 1)\}$ at distance 1,
four diagonal neighbours $N_D(p)$ at distance $\sqrt2$, and $N_8(p) = N_4(p) \cup N_D(p)$ (Fig. 2.9); border pixels
have neighbours outside the image. Two pixels with values in a set $V$ are **4-adjacent** if $q \in N_4(p)$,
**8-adjacent** if $q \in N_8(p)$, and **m-adjacent** if $q \in N_4(p)$, or $q \in N_D(p)$ and $N_4(p) \cap N_4(q)$
contains no pixel of $V$ — m-adjacency removes the ambiguity of multiple 8-paths (Fig. 2.10).
A **path** is a sequence of pairwise adjacent pixels, and pixels joined by a path are **connected**; a maximal
connected set is a **connected component** and a **region** is a connected set whose **boundary** is the pixels with at
least one neighbour outside it (§2.3.5).
The same binary image has **5** components under 4-adjacency but **2** under 8-adjacency (Fig. 2.11) — the choice of
connectivity changes how many floes you count. `label_components` reproduces MATLAB `bwlabel`, including its label
numbering (verified exact), because Chapters 5–9 index objects by that number.
""")
code(r'''
from seaice.ch02_preliminaries import pixel_relationship_examples
from seaice.core.connectivity import n4, n8, nd

ex = pixel_relationship_examples()

def neigh_panel(ax, cells, label, title):      # 3x3 diagram of a neighbourhood (display glue)
    grid = np.full((3, 3), "", dtype=object); grid[1, 1] = "p"
    hl = np.zeros((3, 3)); hl[1, 1] = 0.5
    for r_, c_ in cells:
        grid[r_, c_] = label; hl[r_, c_] = 1.0
    show_matrix(ax, grid, fmt="{}", highlight=hl, title=title, fontsize=11)

p = (1, 1)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
neigh_panel(axes[0], n4(p), "N4", "(a) 4-neighbors N4(p)")
neigh_panel(axes[1], nd(p), "ND", "(b) D-neighbors ND(p)")
neigh_panel(axes[2], n8(p), "N8", "(c) 8-neighbors N8(p)")
fig.suptitle("Fig. 2.9  Neighborhoods of a pixel")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
show_matrix(axes[0], ex["fig_2_10_a"].astype(int), title=f"(a) 4-path  ({len(ex['paths_4'])} path)")
for ax, key, title in ((axes[1], "paths_8", "(b) 8-paths"), (axes[2], "paths_m", "(c) m-path")):
    show_matrix(ax, ex["fig_2_10_bc"].astype(int), title=f"{title}: {len(ex[key])} path(s) from (1,3) to (3,3)")
    for k, path in enumerate(ex[key]):
        pr = np.array(path)
        ax.plot(pr[:, 1] + 0.12 * k, pr[:, 0] + 0.12 * k, "-o", lw=1.5, ms=4, label=f"path {k + 1}")
    ax.legend(fontsize=7, loc="lower left")
fig.suptitle("Fig. 2.10  4-, 8- and m-paths (m-adjacency removes the 8-path ambiguity)")
plt.show()
print(f"|N4|,|ND|,|N8| at an interior pixel: {len(ex['N4_interior'])},{len(ex['ND_interior'])},{len(ex['N8_interior'])};"
      f"  at a corner: {len(ex['N4_corner'])},{len(ex['ND_corner'])},{len(ex['N8_corner'])}")
''')
code(r'''
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, key, n, title in ((axes[0], "labels_4", ex["n_components_4"], "(a) 4-connected components"),
                          (axes[1], "labels_8", ex["n_components_8"], "(b) 8-connected components")):
    L = ex[key]
    ax.imshow(np.where(L > 0, L, np.nan), cmap="tab10", vmin=0.5, vmax=10.5, interpolation="nearest")
    for (r_, c_), v in np.ndenumerate(ex["fig_2_11"].astype(int)):
        ax.text(c_, r_, str(v), ha="center", va="center", fontsize=8)
    ax.set_xticks(np.arange(-0.5, L.shape[1], 1), minor=True); ax.set_yticks(np.arange(-0.5, L.shape[0], 1), minor=True)
    ax.grid(which="minor", color="k", lw=0.4)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(f"{title}: {n} components (colour = bwlabel number)")
fig.suptitle("Fig. 2.11  Connected components of the same binary image (book: 5 vs 2)")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
show_matrix(axes[0], ex["boundary_4"].astype(int), title="Boundary pixels (4-neighbour test)")
show_matrix(axes[1], ex["boundary_8"].astype(int), title="Boundary pixels (8-neighbour test)")
fig.suptitle("Section 2.3.5  Boundary of a region: pixels with at least one neighbour outside R")
plt.show()
print("labels with 4-adjacency (bwlabel numbering, column-major first occurrence):\n", ex["labels_4"])
''')
md(r"""
The diagonal touch between the big block and the small pieces is enough for 8-adjacency to merge them (2 objects) but
not for 4-adjacency (5 objects). Chapter 5 will exploit exactly this when it separates touching floes.
""")

# =====================================================================================================================
# 2.4 Distance transform
# =====================================================================================================================
md(r"""
## 2.4 Distance transform (pp. 20–23) — replaces `ch2/distance_transform.m` (Fig. 2.14) + Figs. 2.12–2.13

The distance transform assigns every object pixel its distance to the nearest background pixel, Eq. (2.9)
$D(p) = 0$ if $p \in B$ (background), $D(p) = \min_{q \in B} d(p, q)$ if $p \in O$ (object), for any metric $d$ that
is positive definite, symmetric and satisfies the triangle inequality. Three metrics are used:
Euclidean, Eq. (2.10) $d_e = \sqrt{(x-u)^2 + (y-v)^2}$ (iso-distance set = circle);
city-block, Eq. (2.11) $d_4 = |x-u| + |y-v|$ (diamond; $r = 1$ gives $N_4$);
chessboard, Eq. (2.12) $d_8 = \max(|x-u|, |y-v|)$ (square; $r = 1$ gives $N_8$).
Euclidean is the most accurate but needs square roots; $d_4$ and $d_8$ are integer and fast.
MATLAB's `bwdist(BW)` measures the distance to the nearest **nonzero** pixel, so Eq. (2.9) is `bwdist(~f)`;
`distance_transform.m` deliberately computes `-bwdist(point)` from one centre pixel and autoscales it with
`imshow(..., [])`, which is what Fig. 2.14 shows. `seaice.core.distance` keeps both semantics
(`distance_transform` = Eq. 2.9, `bwdist` = MATLAB) and is exact against MATLAB for all three metrics.
""")
code(r'''
from seaice.ch02_preliminaries import distance_fixture_examples, point_distance_maps

fx = distance_fixture_examples()          # Fig. 2.12 (7x7 EDT) and Fig. 2.13 (centre distances)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
show_matrix(axes[0], fx["A12"].astype(int), title="(a) Binary image")
show_matrix(axes[1], np.round(fx["D12"], 4), fmt="{:.4g}", highlight=fx["A12"],
            title="(b) Euclidean distance transform, Eq. (2.9)")
fig.suptitle("Fig. 2.12  Distance transform of a small binary matrix")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
show_matrix(axes[0], fx["C4"].astype(int), highlight=fx["C4"] <= 2, title="(a) City-block distance d4, Eq. (2.11)")
show_matrix(axes[1], fx["C8"].astype(int), highlight=fx["C8"] <= 2, title="(b) Chessboard distance d8, Eq. (2.12)")
fig.suptitle("Fig. 2.13  Distances from the centre of a 7x7 grid (shaded: r <= 2)")
plt.show()
print("Fig. 2.12(b) reproduced:", np.allclose(fx["D12"], fx["D12_book"], atol=1e-4),
      "| distinct values:", np.unique(np.round(fx["D12"], 4)).tolist())
print(f"Fig. 2.13 corner values: city-block {fx['C4'][0, 0]:.0f} (book 6), chessboard {fx['C8'][0, 0]:.0f} (book 3)")
''')
code(r'''
maps = point_distance_maps(201)           # distance_transform.m: img(101,101)=1; -bwdist(img, metric)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
imshow_matlab(axes[0], maps["point"], title="(a) 201x201 image, one pixel at (101,101)")
for ax, metric, title in zip(axes[1:], ("euclidean", "cityblock", "chessboard"),
                             ("(b) Euclidean, Eq. (2.10)", "(c) City-block, Eq. (2.11)", "(d) Chessboard, Eq. (2.12)")):
    imshow_matlab(ax, maps[metric], autoscale=True, title=title)          # imshow(imgDist, [])
fig.suptitle("Fig. 2.14  -bwdist(point, metric) displayed with imshow(..., [])")
plt.show()
for metric in ("euclidean", "cityblock", "chessboard", "quasi-euclidean"):
    d = -maps[metric]
    print(f"{metric:16s}: 0 at the centre, max {d.max():.4f} at the corners")
''')
md(r"""
The iso-distance contours are circles, diamonds and squares as Eqs. (2.10)–(2.12) predict. Chapter 5 uses the
Euclidean transform of the ice mask to place watershed markers at floe centres.
""")

# =====================================================================================================================
# 2.5 Convolution
# =====================================================================================================================
md(r"""
## 2.5 Convolution (pp. 22–24) — text only; primitives in `seaice.core.filters`

Linear filtering is the convolution of the image $f$ with a kernel $\omega$, Eq. (2.13)
$h = \omega * f = \iint \omega(u, v)\, f(x-u, y-v)\, du\, dv$; on a lattice with an $m \times n$ kernel it becomes
Eq. (2.14)

$$h(x, y) = \sum_{s=-m/2}^{m/2} \sum_{t=-n/2}^{n/2} \omega(s, t)\, f(x - s,\, y - t),$$

which for $3 \times 3$ expands to nine explicit terms, Eq. (2.15) (Fig. 2.15). Odd-sized kernels are preferred so that
the centre is unique. Eq. (2.14) is *true* convolution (the kernel is flipped, $f(x-s, y-t)$ = MATLAB `conv2`), whereas
MATLAB `imfilter` defaults to **correlation** (no flip, $f(x+s, y+t)$), which matters for the asymmetric derivative
kernels of Chapter 4 and the `xconv2` of Chapter 6 — `seaice.core.filters.conv2` and `imfilter` reproduce both exactly,
and `imfilter(I, h, "replicate")` accepts MATLAB's positional option strings (`'conv'`, `'replicate'`, `'symmetric'`, …).

> **Book inconsistency.** The nine-term expansion printed as Eq. (2.15) on p. 24 reads $\sum \omega(s,t)\, f(x+s, y+t)$,
> i.e. the *correlation* form, which contradicts the sign convention of Eq. (2.14); the package follows Eq. (2.14) /
> `conv2` as the definition of convolution and exposes the printed form through `conv_at(..., correlate=True)`. On the
> antisymmetric Sobel-type demo kernel below the two readings differ in sign (+1 vs −1 at the chosen pixel).
""")
code(r'''
from seaice.ch02_preliminaries import convolution_example

cx = convolution_example(seed=0)          # 6x6 integer image, 3x3 Sobel-type kernel (synthetic)
x, y = cx["x"], cx["y"]

fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
hl = np.zeros_like(cx["f"]); hl[x - 1:x + 2, y - 1:y + 2] = 0.6; hl[x, y] = 1.0
norm = lambda h: (h - h.min()) / (np.ptp(h) + 1e-9)
show_matrix(axes[0], cx["f"].astype(int), fmt="{:d}", highlight=hl, title=f"Image f (3x3 neighbourhood of ({x + 1},{y + 1}))")
show_matrix(axes[1], cx["w"], fmt="{:g}", highlight=np.abs(cx["w"]) / 2, title="Kernel w(s,t), centre = w(0,0)")
show_matrix(axes[2], cx["h_conv2"], fmt="{:g}", highlight=norm(cx["h_conv2"]), title="h = conv2(f, w, 'same')  (Eq. 2.14)")
show_matrix(axes[3], cx["h_imfilter_corr"], fmt="{:g}", highlight=norm(cx["h_imfilter_corr"]),
            title="imfilter(f, w) = correlation = Eq. (2.15) as printed (no flip)")
fig.suptitle(f"Fig. 2.15  3x3 kernel response at ({x + 1},{y + 1}): Eq. (2.14) convolution h = {cx['h_xy_conv']:g}, "
             f"Eq. (2.15) as printed (correlation) h = {cx['h_xy_corr']:g}")
plt.show()

print(f"Eq. (2.14) convolution at (x, y) = ({x + 1}, {y + 1}) [1-based]:  sum w(s,t) * f(x-s, y-t)")
total = 0.0
for s, t, ws, fv in cx["terms_conv"]:
    print(f"  w({s:+d},{t:+d}) * f(x{-s:+d}, y{-t:+d}) = {ws:g} * {fv:g} = {ws * fv:g}")
    total += ws * fv
print(f"  sum = {total:g}   (conv2 gives {cx['h_conv2'][x, y]:g}, conv_at gives {cx['h_xy_conv']:g})")

print(f"\nEq. (2.15) as printed (correlation) at the same pixel:  sum w(s,t) * f(x+s, y+t)")
total = 0.0
for s, t, ws, fv in cx["terms_corr"]:
    print(f"  w({s:+d},{t:+d}) * f(x{s:+d}, y{t:+d}) = {ws:g} * {fv:g} = {ws * fv:g}")
    total += ws * fv
print(f"  sum = {total:g}   (imfilter(f, w) gives {cx['h_imfilter_corr'][x, y]:g}, conv_at(..., correlate=True) gives {cx['h_xy_corr']:g})")
''')
md(r"""
The two right-hand grids differ only in sign for this antisymmetric kernel (+1 under Eq. 2.14, −1 under Eq. 2.15 as
printed at the marked pixel) — the visible fingerprint of the flip that separates convolution from correlation; for
symmetric kernels (Gaussian, mean) the two readings coincide.
""")

# =====================================================================================================================
# 2.6 Set and logical operations
# =====================================================================================================================
md(r"""
## 2.6 Set and logical operations (pp. 24–29) — text only; primitives in `seaice.core.setops`

A binary image is the set of its foreground coordinates, Eq. (2.16) $G = \{(x, y) \mid g(x, y) = 1\}$, and set
notation follows: membership $\omega \in A$ (2.17) / $\omega \notin A$ (2.18), subset $A \subseteq B$ (2.19), set
builder $B = \{\omega \mid \text{condition}\}$ (2.20). With the image rectangle as universe $U$:
complement $A^c = \{\omega \mid \omega \notin A\}$ (2.21), union $A \cup B$ (2.22), intersection $A \cap B$ (2.23) and
difference $A - B = A \cap B^c$ (2.24), shown on 16×16 grids in Fig. 2.16.
Morphology (Chapter 4) also needs the **reflection** $\hat A = \{\omega \mid \omega = -a,\ a \in A\}$ (2.25) and the
**translation** $(A)_z = \{c \mid c = a + z,\ a \in A\}$ (2.26) of Fig. 2.17.
A grayscale image is a 3-D set $G = \{(x, y, z) \mid z = g(x, y)\}$ (2.27) whose complement is $L - g$ with
$L = 2^k - 1$ (2.28), whose union is the pixel-wise **max** (2.29) and whose intersection is the pixel-wise **min** (2.30).
Logical NOT/OR/AND on binary images follow Table 2.1; on integers they act bit by bit, e.g. $57 \wedge 207 =
00111001_2 \wedge 11001111_2 = 00001001_2 = 9$ (p. 29).
""")
code(r'''
from seaice.ch02_preliminaries import set_operation_examples
from seaice.core.setops import truth_tables

so = set_operation_examples(G)            # synthetic A, B (Fig. 2.16), L-shape (Fig. 2.17), gray ops on G

fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
panels = (("A", "A"), ("B", "B"), ("A_c", "Complement  A^c  (Eq. 2.21)"), ("A_or_B", "Union  A ∪ B  (Eq. 2.22)"),
          ("A_and_B", "Intersection  A ∩ B  (Eq. 2.23)"), ("A_minus_B", "Difference  A − B  (Eq. 2.24)"))
for ax, (key, title) in zip(axes.flat, panels):
    show_matrix(ax, so[key].astype(int), fmt="{:d}", cmap="Greys", fontsize=6, title=title)
fig.suptitle("Fig. 2.16  Basic set operations on binary images (synthetic 16x16 sets)")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, key, origin, title in ((axes[0], "L", so["L_origin"], "A (origin = black dot)"),
                               (axes[1], "L_hat", so["L_hat_origin"], "Reflection  Â  (Eq. 2.25)"),
                               (axes[2], "L_z", tuple(np.add(so["L_origin"], so["z"])),
                                f"Translation  (A)_z, z = {so['z']}  (Eq. 2.26)")):
    show_matrix(ax, so[key].astype(int), fmt="", cmap="Greys", title=title)
    ax.plot(origin[1], origin[0], "ko", ms=7)
fig.suptitle("Fig. 2.17  Reflection and translation of a set A")
plt.show()
print(f"|A| = {so['A'].sum()}, |B| = {so['B'].sum()}, |A ∪ B| = {so['A_or_B'].sum()}, |A ∩ B| = {so['A_and_B'].sum()}, "
      f"|A − B| = {so['A_minus_B'].sum()}, |A^c| = {so['A_c'].sum()}")
''')
code(r'''
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
imshow_matlab(axes[0], so["gray"], title="A = rgb2gray(I)  (gray image loaded above)")
imshow_matlab(axes[1], so["gray_c"], title="A^c = L − A, L = 255  (Eq. 2.28)")
imshow_matlab(axes[2], so["gray_union"], title="A ∪ A^c = max  (Eq. 2.29)")
imshow_matlab(axes[3], so["gray_intersection"], title="A ∩ A^c = min  (Eq. 2.30)")
fig.suptitle("Section 2.6.2  Set operations on grayscale images")
plt.show()

print("Table 2.1 truth tables:")
for name, rows in truth_tables().items():
    print(f"  {name}: {rows}")
print(f"57 AND 207 = {so['bitwise_57_and_207']}   (book: 9;  {57:08b} & {207:08b} = {57 & 207:08b})")
''')
md(r"""
The gray-level union/intersection are the max/min images (bright everywhere / dark everywhere here because $A$ and
$A^c$ are complementary); Chapter 7 combines masks this way. The bitwise example confirms `np.bitwise_and` is the
right tool for integer images, `&` on booleans for binary ones.
""")

# =====================================================================================================================
# 2.7 Chain code
# =====================================================================================================================
md(r"""
## 2.7 Chain code (pp. 29–31) — replaces `ch2/chain code/chain_diff.m` (+ `boundaries.m`, `fchcode.m`, `bound2im.m`)

A Freeman chain code describes a boundary as a starting point plus a sequence of direction numbers — 4 directions
$(0 \ldots 3)$ or 8 directions $(0 \ldots 7)$ in steps of $45^\circ$, numbered as in Fig. 2.18 (0 = east, 2 = north,
4 = west, 6 = south; odd = diagonal). Fig. 2.19 shows the 8×9 object of `chain_diff.m` with its 22-code 4-direction and
18-code 8-direction boundaries; Fig. 2.20 lists the coordinates from the starting point $(4, 2)$ with the code
`0 1 2 0 0 7 0 6 6 4 5 6 4 4 3 4 2 2`.
The code depends on the starting point, so it is treated as circular and rotated to the **integer of minimum
magnitude** (normalised code); rotation invariance comes from the **first difference**, Eq. (2.31)

$$D_1(i) = \big(C(i+1) - C(i)\big) \bmod 8,\quad i = 0, \dots, N-2, \qquad D_1(N-1) = \big(C(0) - C(N-1)\big) \bmod 8,$$

which can itself be normalised (Fig. 2.21 lists all five sequences). Chain codes are not scale invariant and are
noise-sensitive; Chapter 5 uses them to find concave corners between touching floes.

> **Two MATLAB quirks reproduced deliberately.** (i) `chain_diff.m` calls `boundaries(B, 8, 'cww')` — a typo for
> `'ccw'` that MATLAB does not recognise, so the trace stays **clockwise**, which is what Figs. 2.20–2.21 print;
> the port uses `direction="cw"`. (ii) `fchcode.m`'s `minmag` never assigns its output when the code is periodic, so
> MATLAB *errors* on the first-difference sequence (period 9); `seaice.core.chaincode.min_magnitude` breaks the tie by
> returning the lexicographically smallest rotation, which yields the book's "normalized first difference" — labelled
> `reimplemented`, verified against the printed sequence.
""")
code(r'''
from seaice.ch02_preliminaries import chain_code_demo
from seaice.core.chaincode import STEPS_4, STEPS_8

cd = chain_code_demo(synth.FIG_2_19_OBJECT)      # boundaries(B, 8, 'cw') -> bound2im -> fchcode (+ 4-direction extras)
c, c4 = cd["c"], cd["c4"]

def direction_diagram(ax, steps, title):          # Fig. 2.18 arrows (display glue; row axis points down)
    for k, (dr, dc) in enumerate(steps):
        ax.annotate("", xy=(dc, dr), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.text(1.25 * dc, 1.25 * dr, str(k), ha="center", va="center", fontsize=12)
    ax.set(xlim=(-1.6, 1.6), ylim=(1.6, -1.6), aspect="equal", title=title); ax.set_axis_off()

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
direction_diagram(axes[0], STEPS_4, "(a) 4-direction")
direction_diagram(axes[1], STEPS_8, "(b) 8-direction")
fig.suptitle("Fig. 2.18  Numbering scheme of the chain code")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
show_matrix(axes[0], cd["B"].astype(int), title="(a) Binary image with an object (chain_diff.m matrix B)")
for ax, b, cc, bim, title in ((axes[1], cd["b4"], c4, cd["bim4"], "(b) 4-direction code"),
                              (axes[2], cd["b"], c, cd["bim"], "(c) 8-direction code")):
    show_matrix(ax, np.zeros_like(cd["B"], dtype=int), fmt="", highlight=bim, cmap="Greys", title=title)
    for (r_, col), code_ in zip(b[:-1], cc.fcc):
        ax.text(col, r_, str(int(code_)), ha="center", va="center", fontsize=9, color="b")
    ax.plot(b[:, 1], b[:, 0], "r-", lw=0.8)
    ax.plot(cc.x0y0[1], cc.x0y0[0], "ro", ms=6)
fig.suptitle("Fig. 2.19  Chain codes of an object boundary (red dot = starting point; code at the departing pixel)")
plt.show()
print(f"boundary: {cd['d'][0]} points (closed, first == last), starting point (row, col) = {cd['x0y0_matlab']} [1-based]")
''')
code(r'''
print("Fig. 2.20  Representation for a boundary (1-based (row, col); starting point first)")
print("  row  col  8-dir code")
for r_, c_, code_ in cd["table"]:
    print(f"  {r_:3d}  {c_:3d}  {code_}")

print("\nFig. 2.21  8-directional chain codes and first differences")
book = synth.FIG_2_21
expect = {"Original chain code": book["original"], "First difference": book["first_difference"],
          "Normalized chain code": book["normalized"],
          "First difference (of normalized)": book["normalized_first_difference_of_normalized"],
          "Normalized first difference": book["normalized_first_difference"]}
for name, seq in cd["sequences"].items():
    ok = np.array_equal(seq, expect[name])
    print(f"  {name:34s}: {' '.join(map(str, seq.tolist()))}   {'== book' if ok else '!= book'}")
print(f"\nFig. 2.19(b) 4-direction code ({len(c4.fcc)} codes): {' '.join(map(str, c4.fcc.tolist()))}")
''')
md(r"""
All five sequences of Fig. 2.21 and the (4, 2) starting point of Fig. 2.20 are reproduced (and, except for the last
line that MATLAB cannot compute, they are identical to `chain_diff.m` run in MATLAB R2025a).
""")

# =====================================================================================================================
# 2.8 Image interpolation
# =====================================================================================================================
md(r"""
## 2.8 Image interpolation (pp. 32–36) — text only; primitives in `seaice.core.interp`

Geometric operations map lattice coordinates through a transform $(\eta, \xi) = T\{(x, y)\}$ (Eq. 2.32, Fig. 2.22);
in practice each output pixel is mapped **back**, $(u, v) = T^{-1}\{(m, n)\}$ (Eq. 2.33), and the image value at the
non-integer point $(u, v)$ must be interpolated.
**Nearest neighbour** takes the value of the closest lattice point (Fig. 2.23): fast, but blocky.
**Bilinear** fits $f(u, v) = \sum_{m,n=0}^{1} a_{mn} u^m v^n$ (Eq. 2.34) to the four neighbours $P_1 = (i, j)$,
$P_2 = (i, j+1)$, $P_3 = (i+1, j)$, $P_4 = (i+1, j+1)$: two interpolations along the columns, Eqs. (2.35)–(2.36),
then one along the rows, Eq. (2.37), or in matrix form Eq. (2.38)
$f(P) = [\,i+1-u,\; u-i\,] \begin{bmatrix} f(P_1) & f(P_2) \\ f(P_3) & f(P_4) \end{bmatrix} \begin{bmatrix} j+1-v \\ v-j \end{bmatrix}$ (Fig. 2.24).
**Bicubic** fits 16 coefficients $\sum_{m,n=0}^{3} a_{mn} u^m v^n$ (Eq. 2.39, Fig. 2.25); its practical form is the
cubic convolution of Eq. (2.40), $f(u, v) = \sum_{m=-1}^{2}\sum_{n=-1}^{2} f(i+m, j+n)\, r_c(m+i-u)\, r_c(n+j-v)$,
with the Keys kernel of Eq. (2.41)

$$r_c(x) = \begin{cases} (a+2)|x|^3 - (a+3)|x|^2 + 1, & |x| \le 1 \\ a|x|^3 - 5a|x|^2 + 8a|x| - 4a, & 1 < |x| \le 2 \\ 0, & \text{otherwise} \end{cases}$$

where MATLAB uses $a = -0.5$ and OpenCV $a = -0.75$.
Parity: `interp2` nearest/bilinear/bicubic are **exact** vs MATLAB `interp2`; `resize` vs `imresize` is only
**approx** (identical interior, but `imresize` evaluates the kernel at the unclipped coordinate near the border and
anti-aliases when shrinking). Chapter 6 needs bilinear `interp2` on the GVF field and Appendix A needs `warp_image`.
""")
code(r'''
from seaice.ch02_preliminaries import interpolation_demo

ip = interpolation_demo(I)                # 32x32 crop G(700:731, 900:931); grid 1:0.4:32 and resize x4
titles = {"nearest": "Nearest neighbour, Eq. (2.33)", "bilinear": "Bilinear, Eqs. (2.35)-(2.38)",
          "bicubic": f"Bicubic (Keys a = {ip['a']}), Eqs. (2.40)-(2.41)"}
for prefix, label in (("grid_", "interp2 grid 1:0.4:32  (exact vs MATLAB interp2)"),
                      ("resize_", "resize x4, imresize pixel-centre convention  (approx vs imresize at the border)")):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    axes[0].imshow(ip["P"], cmap="gray", vmin=0, vmax=255, interpolation="nearest"); axes[0].set_title("Input crop P (32x32)")
    for ax, m in zip(axes[1:], ("nearest", "bilinear", "bicubic")):
        ax.imshow(ip[prefix + m], cmap="gray", vmin=0, vmax=255, interpolation="nearest"); ax.set_title(titles[m])
    for ax in axes:
        ax.set_axis_off()
    fig.suptitle(f"Section 2.8  Image interpolation — {label}")
    plt.show()
''')
code(r'''
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ip["kernel_x"], ip["kernel_keys"], label=f"rc(x), a = {ip['a']} (MATLAB interp2 / imresize)")
ax.plot(ip["kernel_x"], ip["kernel_keys_opencv"], "--", label="rc(x), a = -0.75 (OpenCV)")
ax.axhline(0, color="k", lw=0.5)
ax.set(xlabel="x", ylabel="rc(x)", title="Eq. (2.41)  Cubic convolution (Keys) kernel")
ax.legend()
plt.show()

P = ip["P"]
for m in ("nearest", "bilinear", "bicubic"):
    g = ip["grid_" + m]
    sub = g[::5, ::5]                     # step 0.4 -> every 5th query is a lattice point
    lattice_ok = np.allclose(sub, P[::2, ::2][:sub.shape[0], :sub.shape[1]], atol=1e-9)
    print(f"{m:9s}: grid {g.shape}, resize {ip['resize_' + m].shape}; lattice values reproduced exactly: {lattice_ok}")
print("keys_kernel at x = 0, ±1, ±2:", np.round(ip["kernel_keys"][[250, 150, 350, 50, 450]], 6).tolist())
''')
md(r"""
Nearest neighbour shows the saw-tooth blocks, bilinear is smooth but slightly blurred, bicubic keeps edges sharper (and
can overshoot, since $r_c$ goes negative for $1 < |x| < 2$). All three reproduce the lattice values exactly, as an
interpolating kernel must.
""")

# =====================================================================================================================
# Parameter play (optional)
# =====================================================================================================================
md(r"""
## Parameter play (optional)

Two knobs from this chapter on a downscaled copy of the gray image (`SCALE` = subsampling step, keeps it fast):
the binary threshold (§2.1.1) and the connectivity used to count components (§2.3.4). Requires `ipywidgets`; the cell
prints a message and falls back to a static run if it is not installed.
""")
code(r'''
from seaice.core.connectivity import label_components

SCALE = 8                                 # subsample step: 2048x1536 -> 256x192 (fast enough for a slider)
G_small = G[::SCALE, ::SCALE]

def play(threshold=128, conn=8):
    bw = G_small > threshold              # 1-line glue: B = 1 image
    n = int(label_components(bw, conn).max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    imshow_matlab(axes[0], G_small, title=f"gray (every {SCALE}th pixel)")
    imshow_matlab(axes[1], bw, title=f"gray > {threshold}: {n} components ({conn}-connectivity)")
    plt.show()

play()                                    # static preview (threshold 128, 8-connectivity) — visible without widgets
try:
    from ipywidgets import interact, IntSlider, Dropdown
    interact(play, threshold=IntSlider(128, 0, 255, 1), conn=Dropdown(options=[4, 8], value=8))
except ImportError:
    print("ipywidgets not installed - only the static preview above is shown")
''')

# =====================================================================================================================
# Summary + feeds forward
# =====================================================================================================================
md(r"""
## Summary — and what the next chapters need from this one

**What chapter 2 established.** Images are matrices indexed `(row, col)` like MATLAB `I(x, y)`; colour models
(CMY/CMYK/HSI) are linear transforms of RGB; histograms (Eqs. 2.7–2.8) expose the water/ice modes; connectivity
(4 vs 8 vs m) decides how many objects an image contains; distance transforms (Eqs. 2.9–2.12) measure "how deep inside
an object" a pixel is; convolution vs correlation differ by a kernel flip; binary images are sets with complement,
union, intersection, difference, reflection and translation; chain codes plus first differences describe boundaries
invariantly; nearest / bilinear / bicubic interpolation resample images for geometric transforms.

**Parity (from `reports/ch02_verification.md`, MATLAB R2025a as reference).** All seven `.m` files are `exact`
(`imread`, `rgb2gray`, `imhist`, `imcomplement`, `bwdist`, `bwlabel`, `conv2`, `imfilter`, `interp2`, `boundaries`,
`fchcode`, `bound2im` all 0-difference or ≤1e-12); `reimplemented`: HSI hue from Eq. (2.6a), CMYK, quasi-euclidean
`bwdist` (`near`, MATLAB is single precision), `min_magnitude` tie-break; `approx`: `resize` vs `imresize` (bicubic
border, antialiasing); `unverified`: Fig. 2.7 (source image not shipped — substitute shown above).

**Feeds forward (`seaice/core/` primitives the later chapters import):**

| Needed by | Primitive |
|---|---|
| Ch3 (thresholding, k-means) | `matlab_compat.rgb2gray_matlab`, `histogram.imhist` / `normalized_histogram`, `io.load_image` (private copy → public-domain substitute) |
| Ch4 (edges, morphology) | `filters.conv2` / `imfilter` (correlation vs convolution!), `setops.reflect` / `translate` (structuring elements), `connectivity.label_components` |
| Ch5 (watershed, floe splitting) | `distance.bwdist` / `distance_transform` (markers), `chaincode.boundaries` / `fchcode` / `bound2im` (concavity analysis; ch5 ships byte-identical copies of these `.m` files), `label_components` in `bwlabel` order |
| Ch6 (GVF snake) | `interp.interp2` (bilinear on the GVF field), `distance.bwdist`, `filters.imfilter` (`xconv2`, `gaussianBlur`) |
| Ch7–Ch9 | `setops.gray_union` / `gray_intersection` (mask combination), `matlab_compat.matlab_round` / `im2uint8` / `im2double` |
| App. A (calibration) | `interp.warp_image` (orthorectification, lens-distortion resampling) |

**Pitfalls to carry forward.** Python is 0-based `(row, col)` everywhere (add 1 to compare with MATLAB `b`, `x0y0`,
`bound2im` offsets); `bwdist` ≠ Eq. (2.9) (`distance_transform(f) == bwdist(~f)`); `imfilter` correlates unless
`'conv'`; MATLAB `fchcode`/`minmag` *error* on single-pixel objects and periodic codes where the port returns a value —
ch5 code must not rely on that failure; `rgb2hsi` default ≠ `color_image.m` (use `matlab_bug=True` only for the figure).

*Optional: to regenerate the MATLAB references yourself, run `reference/ch02/make_refs.py` (uses
`tools/run_matlab_ref.py`, MATLAB `-batch`); the notebook does not need MATLAB or Octave.*
""")

out = HERE / NB_NAME
nbf.write(nb, out)
print(f"wrote {out.as_posix()} ({len(C)} cells)")
