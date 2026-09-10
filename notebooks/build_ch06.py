"""Build ``notebooks/ch06_gvf_snake.ipynb`` with nbformat (re-runnable; the notebook is never edited by hand).

Chapter 6 — GVF Snake-Based Ice Floe Boundary Identification and Ice Image Segmentation
(Zhang & Skjetne 2018, pp. 109–144).  Follows the ``colab-notebook`` skill: title → cell 1 (Drive mount / cwd)
→ cell 2 (clone or pull the public repo) → cell 3 (``load_image``: private copies or public-domain substitutes)
→ one section per book section (6.1, 6.1.1, 6.1.2, 6.1.3, 6.2, 6.3, 6.4, 6.5) → parameter play →
MATLAB ↔ Python map → summary / what is and is not verified / feeds forward.  All algorithms are imported from
``seaice``; the notebook only calls them and draws figures.

Run:  ``.venv/Scripts/python.exe notebooks/build_ch06.py``
Verify: ``.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900
        --output executed_ch06.ipynb notebooks/ch06_gvf_snake.ipynb``  (then delete the executed copy).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

NB_NAME = "ch06_gvf_snake.ipynb"
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
# Chapter 6 — GVF Snake-Based Ice Floe Boundary Identification and Ice Image Segmentation

*Sea Ice Image Processing with MATLAB* (Q. Zhang & R. Skjetne, CRC Press 2018), Chapter 6, pp. 109–144 —
Python port `seaice-py`, notebook `notebooks/ch06_gvf_snake.ipynb`.

**Sections covered** (one notebook section each, in book order):
6.1 Traditional parametric snake model (Eqs. 6.1–6.40, Figs. 6.1–6.7) · 6.2 Gradient vector flow (GVF) snake
(Eqs. 6.41–6.56, Figs. 6.8–6.9) · 6.3 Contours initialization (Eqs. 6.57–6.58, Figs. 6.11–6.14) ·
6.4 Ice image segmentation — **Algorithm 1** (Fig. 6.15) · 6.5 Discussion (Figs. 6.16–6.21).

Chapters 3–5 produced a *binary* ice mask and cut touching floes along the thinnest neck (watershed). Chapter 6
replaces that cut by a **deformable curve**: a closed contour is placed inside each floe and evolves in a force
field derived from the image until it sits on the floe's boundary. The two classical failures of such a snake —
a tiny **capture range** and no force into boundary **concavities** — are removed by the **gradient vector flow**
field of Xu & Prince, and the contours are initialised automatically from the **distance transform** of the mask
(the same primitive Chapter 5 used for its markers). The output is a segmentation in which connected floes are
separated by snake boundaries; Chapters 7–9 consume it.

**MATLAB files this notebook replaces** (`MATLAB_ROOT/ch6/`, 34 `.m` files in two folders; parity from
`reports/ch06_verification.md`, MATLAB R2025a running the original code as the reference):

| MATLAB file | Python (imported below) | Parity |
|---|---|---|
| `GVF.m` (Eqs. 6.41, 6.50–6.55) | `seaice.core.snake.gvf` | exact (44 cases 0.0; `u`, `v`, `px`, `py` on both book images 0.0) |
| `BoundMirrorExpand/Ensure/Shrink.m` (natural BC, Eq. 6.46) | `core.snake.bound_mirror_expand / _ensure / _shrink` | exact |
| `gradient2.m` (edge map $f = \lvert\nabla I\rvert$) | `core.snake.gradient2`, `gradient2_magnitude` | exact |
| `snakedeform.m` (Eqs. 6.34–6.40) | `core.snake.snakedeform`, `snake_matrix` | exact (dense solver); `near` for the FFT fast path |
| `snakeinterp.m`, `snakeindex.m` (point respacing) | `core.snake.snakeinterp`, `snakeindex` | exact (16/18 fixtures 0.0; the other 2 are the header's own admitted bug) |
| `gaussianMask.m`, `gaussianBlur.m`, `xconv2.m` (Eq. 6.10) | `core.snake.gaussian_mask / gaussian_blur / xconv2` | exact |
| `snakedisp.m` | `core.plotting.snake_plot` | display only |
| `GVF_distance.m` (§6.3.3 + §6.4 Algorithm 1) | `seaice.ch06_gvf_snake.gvf_distance` | near (every stage 0 px / 0.0; only the snake discretisation differs — see the honest close) |
| `seaice_kmean_GVF_forenhancement.m` (Algorithm 1 twice + k-means) | `ch06_gvf_snake.seaice_kmean_gvf` | near (k-means stage `bk` 0 px vs MATLAB's `rng(0)` run) |
| `sea_ice_demo.m` (the driver) | `scripts/ch06_sea_ice_demo.py` | near (the `ice_shape_enhancement`/`sea_ice_model` tail belongs to Ch. 7/8) |
| `for test/dist.m` (§6.3.3 seeds and circles) | `ch06_gvf_snake.initialize_contours`, `scripts/ch06_dist.py` | near (radii are `single` in MATLAB) |
| `for test/for_test.m` (§6.1.2/§6.2 single contour) | `scripts/ch06_for_test.py` | exact (mask 0 px) |
| MATLAB `del2` (Eq. 6.52c inside `GVF.m`) | `core.matlab_compat.del2` | exact (21 cases 0.0) |
| MATLAB `regionprops` (Area, Solidity, axes, Perimeter…) | **`core.regionprops`** | exact (≤ 1.07e-14 on 40 shapes and on all 344 real components) |
| `polybool('intersection', …)` (Mapping Toolbox) | `core.polygon.clip_polygon_rect` | reimplemented (vertex set and mask identical) |
| `polygeom.m`, `minboundrect.m`, `roipoly`/`poly2mask`, `polyxpoly`, `convhull` | `core.polygon.*` | near / exact |
| `homofil.m` (orphan: no caller, no book section) | `core.filters.homomorphic_butterworth` | exact (≤ 1.14e-12) |
| `ice_shape_enhancement.m`, `sea_ice_model.m`, `SeaIce_Image_Structure.m`, `color_hist*.m` | deferred to Ch. 7 / Ch. 8 / Appendix B | — |
| *(text only: Eqs. 6.6–6.13, 6.29, 6.57/6.58)* | `ch06_gvf_snake.{line,edge,termination,external}_energy`, `traditional_snake`, `core.morphology.regional_maxima_by_reconstruction` | reimplemented |

> **Credit.** The snake/GVF toolbox (`GVF.m`, `snakedeform.m`, `snakeinterp.m`, `snakeindex.m`, `snakedisp.m`,
> `BoundMirror*.m`, `gaussianMask/Blur.m`, `xconv2.m`) is **Chenyang Xu and Jerry L. Prince**'s freely distributed
> academic code (Image Analysis and Communications Lab, Johns Hopkins University,
> <http://iacl.ece.jhu.edu/projects/gvf>; C. Xu & J. L. Prince, *Snakes, Shapes, and Gradient Vector Flow*,
> IEEE Trans. Image Processing 7(3):359–369, 1998), which the book's own footnote 1 (p. 110) points at; every
> ported function names them. `minboundrect.m` is **John D'Errico**'s and `polygeom.m` is **H. J. Sommer III**'s
> MATLAB File Exchange geometry code. The chapter's own code (`GVF_distance.m`,
> `seaice_kmean_GVF_forenhancement.m`, `dist.m`, `for_test.m`, …) is by **Qin Zhang** (NTNU, project *Arctic DP*)
> and is marked "available for academic non-commercial use"; this port is a re-implementation for study.

> **Data.** Chapter 6 ships three images. `alg_seg_gray.jpg` (202×201) is **Figure 6.15(a)** (identified at
> NCC 0.9882 against the printed bitmap); `test8.jpg` (148×108) drives `for test/for_test.m`; `sea_ice_test.jpg`
> (394×1038, a tall shipborne view) drives `sea_ice_demo.m` and `dist.m`. They are copyrighted and are **not** in
> this public repository: `seaice.core.io.load_image()` first looks for your own private copies
> (`data/book/ch06/**`, keeping the book's `Sea_Ice_Floe_Identification/` and `for test/` sub-folders, or the same
> tree in `MyDrive/Sea_Ice_Colab`) and otherwise downloads **public-domain NASA MODIS windows of exactly the same
> pixel sizes**, registered in `seaice/core/public_images.py`. Every number below is computed from the image that
> was actually loaded; book-quoted values are printed **only** when the book's own images are present.
> Figs. 6.2–6.9 and 6.16–6.21 use images the book never shipped, so they run on seeded **synthetic** fixtures from
> `seaice.core.synth` (said so in each cell); Fig. 6.14 is a printed 8×8 truth and reproduces bit-exactly.
""")

# =====================================================================================================================
# 2. Setup cells (Colab / Drive / local) — copied verbatim from build_ch02.py
# =====================================================================================================================
md(r"""
## Setup (Google Colab or local Jupyter)

Run the two cells below first. On **Colab** the first cell mounts your Google Drive and moves into
`MyDrive/Sea_Ice_Colab` — the folder where your private copies of the book images live
(`data/book/ch06/Sea_Ice_Floe_Identification/sea_ice_test.jpg`, `data/book/ch06/for test/test8.jpg`,
`data/book/ch06/for test/alg_seg_gray.jpg`) and where downloads are cached between sessions; readers without
Drive get a temporary `/content/Sea_Ice_Colab`. The second cell clones (or updates) the public repository
`seaice-py` there and installs its requirements. **Locally** both cells are no-ops that move to the repository
root. No GPU is needed.

> ⚠️ **If you run this with your own copies of the book images, do not use *File → Save a copy in GitHub*.** That
> saves the cell outputs — figures rendered from the copyrighted book images — into the public repository. Save to
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

`seaice.core.io.load_image("ch06", …)` replaces the `imread(...)` at the top of every Chapter 6 script (including
`dist.m`'s hard-coded absolute Windows path). It looks for **your private copies of the book images** first — in
`data/book/ch06/` in the current folder (on Colab: `MyDrive/Sea_Ice_Colab`), in the repository, or in Drive, and
it searches the book's own sub-folders recursively, so a verbatim copy of the MATLAB tree works unchanged — and
otherwise downloads **public-domain NASA substitutes** of the same pixel sizes into `data/online/ch06/`. It prints
which source it used and returns the label, so the cells below quote the book's image-dependent numbers only when
the book's own images are loaded (the `book(...)` / `matlab_ref(...)` helpers). The chapter's *parameters*
(α = 0.05, β = 0, μ = 0.1, the ÷√2 radius rule, `strel('disk', 3)`) hold for any image.
""")
code(r'''
from seaice.core.io import load_image

ALG,   SRC_ALG   = load_image("ch06", "alg_seg_gray.jpg")     # Figure 6.15(a): 202x201 RGB (visually gray)
T8,    SRC_T8    = load_image("ch06", "test8.jpg")            # for test/for_test.m: 148x108 RGB
STRIP, SRC_STRIP = load_image("ch06", "sea_ice_test.jpg")     # sea_ice_demo.m / for test/dist.m: 394x1038 RGB
SOURCE = SRC_ALG
FROM_BOOK = all(s.startswith("book") for s in (SRC_ALG, SRC_T8, SRC_STRIP))
book = (lambda s: f"  (book: {s})") if FROM_BOOK else (lambda s: "")          # book-quoted values only for the book images
matlab_ref = (lambda s: f"  [MATLAB R2025a on the book image: {s}]") if FROM_BOOK else (lambda s: "")
if not FROM_BOOK:
    print("Running on a public-domain substitute image; figures show the same operations, "
          "but values quoted in the book only hold for the book's own image.")
LOADED = (("alg_seg_gray.jpg", ALG, SRC_ALG), ("test8.jpg", T8, SRC_T8), ("sea_ice_test.jpg", STRIP, SRC_STRIP))
for name, img, src in LOADED:
    print(f"{name:18s} {img.shape} {img.dtype}   [{src}]")
flat = [name for name, img, _ in LOADED if img.max() == img.min()]      # a failed download returns a blank JPEG
if flat:
    print("WARNING: " + ", ".join(flat) + " is a constant (blank) image — the download most likely failed. "
          "Delete the cached file under data/online/ch06/ and re-run this cell before going on.")
''')

md(r"""
## Imports

Everything algorithmic comes from the `seaice` package; the notebook only calls it and draws figures.
(`seaice.core.plotting` selects the non-interactive Agg backend for the command-line scripts, so
`%matplotlib inline` is issued *after* the `seaice` imports to switch back to inline display.) Unlike four
Chapter 5 scripts, **every** Chapter 6 script thresholds the *gray* image: `bw = im2bw(I, graythresh(I))` with
`I = rgb2gray(...)`.
""")
code(r'''
import time
import numpy as np
import seaice.core                                     # imports seaice.core.plotting (sets the Agg backend)
from seaice.core import synth
from seaice.core.plotting import imshow_matlab, show_matrix, label2rgb, snake_plot, quiver_field
from seaice.core.matlab_compat import rgb2gray_matlab, del2
from seaice.core.threshold import graythresh, im2bw
from seaice.core.connectivity import label_components
from seaice.core.distance import bwdist
from seaice.ch06_gvf_snake import BOOK_PARAMS, CIRCLE_T, CITYBLOCK_RADIUS_DIVISOR

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100

P = BOOK_PARAMS["sea_ice_demo"]                        # the canonical parameter block (sea_ice_demo.m lines 9-46)
GRAY = rgb2gray_matlab(ALG)                            # I = rgb2gray(I)
LEVEL, _ = graythresh(GRAY)                            # graythresh on the GRAY image (all ch6 scripts)
BW = im2bw(GRAY, LEVEL)                                # Figure 6.15(b)
M, N = GRAY.shape
IMG_LABEL = "alg_seg_gray.jpg" if FROM_BOOK else "substitute image"
print(f"{IMG_LABEL}: gray {M}x{N}, levels {int(GRAY.min())} .. {int(GRAY.max())}")
print(f"graythresh = {LEVEL:.6f} (255*level = {255 * LEVEL:g}); ice pixels {int(BW.sum())} ({100 * BW.mean():.2f} %), "
      f"{int(label_components(BW, 4).max())} 4-connected ice components"
      f"{matlab_ref('gray, level, bw and the 83-component labelling are 0 px / 0.0 against MATLAB')}")
print("sea_ice_demo.m parameters:", P)
print("t = 0:0.05:6.28 ->", CIRCLE_T.size, "points, last", f"{CIRCLE_T[-1]:g}",
      "(the initial circle is NOT closed; snakeinterp closes it implicitly)")
''')

# =====================================================================================================================
# 6.1 Traditional parametric snake model
# =====================================================================================================================
md(r"""
## 6.1 Traditional parametric snake model (pp. 110–122) — Eqs. (6.1)–(6.40), Figs. 6.1–6.6

A **snake** is a closed parametric curve $c(s) = (x(s), y(s))$, $s \in [0, 1]$, $c(0) = c(1)$ (Fig. 6.1) that
minimises

$$E = \int_0^1 E_{int}(c(s)) + E_{ext}(c(s))\, ds \quad (6.2), \qquad
E_{int} = \tfrac{1}{2}\alpha\left|\frac{dc}{ds}\right|^2 + \tfrac{1}{2}\beta\left|\frac{d^2c}{ds^2}\right|^2 \quad (6.3).$$

The **elasticity** $\alpha$ penalises stretching, so a large $\alpha$ keeps the snake points evenly spaced and the
curve short and ripple-free (Fig. 6.2); the **rigidity** $\beta$ penalises bending, so a large $\beta$ smooths
corners away (Fig. 6.3). Footnote 2 (p. 121) fixes **α = 0.05 and β = 0 for every snake in this book** — the
curve is elastic but free to form corners. Setting the variational derivative of (6.2) to zero gives the
Euler–Lagrange equations (6.24)/(6.25), i.e. the force balance $F_{int} + F_{ext} = 0$ (6.27)–(6.29), and
discretising them with $h = 1$ (Eqs. 6.31–6.36) turns the internal force into a **pentadiagonal circulant**
matrix $A$ (Eq. 6.37, printed on p. 120).
""")
code(r'''
from seaice.core.snake import snake_matrix, snake_first_column

A8 = snake_matrix(8, P["alpha"], P["beta"])            # snakedeform.m lines 22-38, N = 8
A8b = snake_matrix(8, 0.05, 0.1)                       # with a non-zero rigidity: all five diagonals appear
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
show_matrix(axes[0], np.round(A8, 3), fmt="{:g}", cmap="Blues", fontsize=8,
            title=f"A for N = 8, alpha = {P['alpha']}, beta = {P['beta']} (the book's default)")
show_matrix(axes[1], np.round(A8b, 3), fmt="{:g}", cmap="Blues", fontsize=7,
            title="A for alpha = 0.05, beta = 0.1: pentadiagonal + wrap-around")
fig.suptitle("Eq. (6.37)  The circulant snake matrix A built from the Eq. (6.36) coefficients a, b, c, d, e")
plt.show()
print(f"A is symmetric: {np.allclose(A8b, A8b.T)}; circulant (every row is a shift of the first): "
      f"{np.allclose(A8b, np.array([np.roll(snake_first_column(8, 0.05, 0.1), k) for k in range(8)]))}")
lam = np.linalg.eigvalsh(A8)
print(f"beta = 0 makes A = alpha*(2I - S - S^T), the circulant 1-D Laplacian: eigenvalues "
      f"{np.round(lam, 4).tolist()} = alpha*(2 - 2cos(2*pi*k/N)) -> {np.allclose(np.sort(lam), np.sort(P['alpha'] * (2 - 2 * np.cos(2 * np.pi * np.arange(8) / 8))))}")
print("Eq. (6.40): x^{t+1} = (A + gamma*I)^{-1} (gamma*x^t + kappa*v_x) — one factorisation for the whole "
      "evolution because alpha, beta, gamma are constant (book p. 121).")
''')

md(r"""
### 6.1.1.2 External energy (pp. 113–115) and 6.1.2 implementation

The external energy carries the image information. Dropping the line and termination terms of Eq. (6.5), the
object-boundary case is

$$E_{ext} = \gamma E_{edge} = -\gamma\left|\nabla I(x, y)\right|^2 \quad (6.9), \qquad
E_{ext} = -\gamma\left|\nabla (G_\sigma * I)\right|^2 \quad (6.11),$$

with the Gaussian $G_\sigma = \frac{1}{2\pi\sigma^2}e^{-(x^2+y^2)/2\sigma^2}$ (6.10); for binary images
$E_{ext} = -\gamma I$ (6.12) or $-\gamma\,G_\sigma * I$ (6.13). The force is $F_{ext} = -\nabla E_{ext}$ (6.29)
and the snake is marched with the semi-implicit step

$$x^{t+1} = (A + \lambda I)^{-1}(\lambda x^t - f_x) \quad (6.40a),$$

evaluating $f_x$ at the snake's non-integer positions by **bilinear interpolation** (Eq. 6.33 —
MATLAB `interp2(fx, x, y, '*linear', 0)`). The cell below reproduces Figs. 6.2 and 6.3 on a **synthetic** floe
(the book's own image is not shipped): it deliberately omits `snakeinterp`, because respacing the points would
hide exactly the effect $\alpha$ has.
""")
code(r'''
from seaice.core.snake import gvf, gradient2, gradient2_magnitude, gaussian_blur, snakedeform, snakeinterp

FLOE = synth.synthetic_floe_field(shape=(160, 160), n_floes=1, seed=4, radius=(38, 39))   # SYNTHETIC (seeded)
MF, NF = FLOE.shape
f2_floe = gradient2_magnitude(FLOE.astype(np.float64))          # edge map, Eq. (6.7)
u_f, v_f = gvf(f2_floe, P["mu"], 200)                           # a GVF field to deform in (section 6.2)
mag = np.hypot(u_f, v_f)
px_f, py_f = u_f / (mag + 1e-10), v_f / (mag + 1e-10)           # the drivers' unit-normalised force (px = u/|v|)
cy, cx = [c.mean() + 1.0 for c in np.nonzero(FLOE > 0)]
bwF = FLOE > 0
edgeF = bwF & ~(np.pad(bwF, 1)[2:, 1:-1] & np.pad(bwF, 1)[:-2, 1:-1] &
                np.pad(bwF, 1)[1:-1, 2:] & np.pad(bwF, 1)[1:-1, :-2])   # the floe's own boundary ring
d_edge = bwdist(edgeF, "euclidean")                             # distance of any pixel to that boundary
x0 = cx + 12.0 * np.cos(CIRCLE_T)
y0 = cy + 12.0 * np.sin(CIRCLE_T)

def spacing(x, y):
    d = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    return float(d.sum()), float(d.mean()), float(d.std())

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for row, (tag, values, fixed, other) in enumerate((("alpha", [0.0, 0.05, 0.5], P["beta"], "beta"),
                                                   ("beta", [0.0, 0.05, 0.5], P["alpha"], "alpha"))):
    print(f"Fig. 6.{2 + row}: influence of {tag} ({other} = {fixed}, 200 iterations, no snakeinterp)")
    print(f"{tag:>7} | {'points':>6} | {'length':>8} | {'mean spacing':>12} | {'std spacing':>11}")
    for ax, val in zip(axes[row], values):
        a, b = (val, fixed) if tag == "alpha" else (fixed, val)
        xs, ys = snakedeform(x0, y0, a, b, P["gamma"], P["kappa"], px_f, py_f, 200)
        L, m, sd = spacing(xs, ys)
        print(f"{val:>7g} | {len(xs):>6d} | {L:8.2f} | {m:12.4f} | {sd:11.4f}")
        imshow_matlab(ax, FLOE, title=f"{tag} = {val:g}  ({other} = {fixed:g}), length {L:.1f}")
        snake_plot(ax, x0, y0, "r--", lw=0.8)
        snake_plot(ax, xs, ys, "g-", lw=1.2)
        ax.plot(xs - 1, ys - 1, "y.", ms=2)
fig.suptitle("Figs. 6.2 / 6.3 (synthetic floe)  Elasticity alpha (top) and rigidity beta (bottom); yellow dots = snake points")
plt.show()
''')
md(r"""
A larger $\alpha$ shortens the curve and equalises the point spacing (the standard deviation drops), exactly the
behaviour the book describes for Fig. 6.2; $\beta$ smooths the contour but, with the book's $\beta = 0$, corners
are allowed — which is what an irregular floe boundary needs.
""")

md(r"""
### 6.1.3 The two limitations (pp. 122–123) — Figs. 6.6, 6.7

The gradient of an edge map has three properties: the vectors point at the edges, they are large only *near* the
edges, and they are ≈ 0 in homogeneous regions. Hence (i) a contour started outside the narrow band around the
boundary feels no force at all and simply shrinks under its own internal energy — the **capture range** problem
(Fig. 6.6, and p. 115: the snake "will evolve under its own energy … and form a circle that keeps shrinking"); and
(ii) inside a boundary **concavity** the two opposite walls pull the curve sideways with no component pushing it
down the channel, so the snake stops at the mouth of a U-shape (Fig. 6.7). Both cells below use **synthetic**
fixtures: `synth.synthetic_floe_field` and Xu & Prince's classic U (`synth.u_shape`), because the book's images
are not shipped. Note that the Gaussian of Eq. (6.10) is itself what buys capture range: with $\sigma = 4$
(the Fig. 6.7(b) caption value) the blurred field already reaches into the channel, so the sweep prints
$\sigma = 1, 2, 4$ and lets the numbers make the point.
""")
code(r'''
# --- Fig. 6.6: the capture range, measured (synthetic floe) ----------------------------------------------------
from seaice.ch06_gvf_snake import external_energy

E = external_energy(FLOE.astype(np.float64), kind="edge", gamma=1.0, sigma=2.0)   # Eq. (6.11), gamma = 1
ex, ey = gradient2(E)

def unit_max(a, b):
    """Scale a field by its GLOBAL maximum magnitude, so two fields can be compared like for like."""
    m = float(np.hypot(a, b).max()) or 1.0
    return a / m, b / m

tpx, tpy = unit_max(-ex, -ey)                                    # Eq. (6.29) F_ext = -grad E_ext
gpx, gpy = unit_max(u_f, v_f)                                    # the raw GVF field, same scaling

def evolve(fx, fy, x, y, blocks=30, alpha=None, beta=None, kappa=None):
    """ceil(iter/5) blocks of snakedeform(..., 5) + snakeinterp — the drivers' inner loop (glue only)."""
    xs, ys = snakeinterp(x, y, P["Dmax"], P["Dmin"])
    hist = [(xs.copy(), ys.copy())]
    for _ in range(blocks):
        xs, ys = snakedeform(xs, ys, P["alpha"] if alpha is None else alpha, P["beta"] if beta is None else beta,
                             P["gamma"], P["kappa"] if kappa is None else kappa, fx, fy, 5)
        xs, ys = snakeinterp(xs, ys, P["Dmax"], P["Dmin"])
        hist.append((xs.copy(), ys.copy()))
    return xs, ys, hist

def sample(field_x, field_y, x, y):
    rows = np.clip(np.round(y).astype(int) - 1, 0, MF - 1)
    cols = np.clip(np.round(x).astype(int) - 1, 0, NF - 1)
    return np.hypot(field_x, field_y)[rows, cols].mean(), d_edge[rows, cols].mean()

R_FAR, R_NEAR = 70.0, 44.0
xf, yf = cx + R_FAR * np.cos(CIRCLE_T), cy + R_FAR * np.sin(CIRCLE_T)
print(f"mean |F| along the starting circle at r = {R_FAR:g} (the floe boundary is at r ~ 38.5):")
print(f"  traditional -grad E_ext (sigma = 2): {sample(tpx, tpy, xf, yf)[0]:.4f}")
print(f"  GVF field (200 iterations, same scaling): {sample(gpx, gpy, xf, yf)[0]:.4f}  <- the enlarged capture range")
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
runs = [("traditional force, start r = 70 (outside the capture range)", tpx, tpy, R_FAR, 40),
        ("traditional force, start r = 44 (inside it)", tpx, tpy, R_NEAR, 40),
        ("GVF force, normalised per pixel as the drivers do, start r = 70", px_f, py_f, R_FAR, 80)]
for ax, (tag, fx, fy, r_start, blocks) in zip(axes, runs):
    x0f, y0f = cx + r_start * np.cos(CIRCLE_T), cy + r_start * np.sin(CIRCLE_T)
    xs, ys, hist = evolve(fx, fy, x0f, y0f, blocks, kappa=1.0)
    _, d_end = sample(fx, fy, xs, ys)
    print(f"{tag:62s}: {5 * blocks:>3d} iterations -> final half-width {0.5 * (xs.max() - xs.min()):5.1f} px, "
          f"mean distance to the floe boundary {d_end:5.2f} px")
    imshow_matlab(ax, FLOE, title=f"{tag}\nfinal mean distance to the boundary {d_end:.1f} px")
    for h in hist[1:-1]:
        snake_plot(ax, *h, "y-", lw=0.4)
    snake_plot(ax, *hist[0], "r-", lw=1.0)
    snake_plot(ax, xs, ys, "g-", lw=1.4)
fig.suptitle("Fig. 6.6 (synthetic)  Outside the capture range the traditional force is ~0 and the snake never finds the floe")
plt.show()
''')
code(r'''
# --- Fig. 6.7: the boundary-concavity problem on the classic U shape (synthetic) --------------------------------
U = synth.u_shape(shape=(96, 96), thickness=3, gap=28, margin=14)                  # SYNTHETIC
MU_, NU_ = U.shape
has_left  = np.cumsum(U, axis=1) > 0
has_right = np.cumsum(U[:, ::-1], axis=1)[:, ::-1] > 0
has_below = np.cumsum(U[::-1, :], axis=0)[::-1, :] > 0
channel = (~U) & has_left & has_right & has_below                                  # the concavity
mouth_row = int(np.nonzero(channel.any(axis=1))[0].min())
base_row = int(np.nonzero(U.sum(axis=1) == U.sum(axis=1).max())[0].min())

def unit_max(a, b):
    m = float(np.hypot(a, b).max()) or 1.0
    return a / m, b / m

def penetration(xs, ys):
    rows = np.clip(np.round(ys).astype(int) - 1, 0, MU_ - 1)
    cols = np.clip(np.round(xs).astype(int) - 1, 0, NU_ - 1)
    inch = channel[rows, cols]
    if not inch.any():
        return 0.0
    return 100.0 * (int(rows[inch].max()) + 1 - mouth_row) / max(base_row - mouth_row, 1)

xU = NU_ / 2.0 + 0.42 * NU_ * np.cos(CIRCLE_T)
yU = MU_ / 2.0 + 0.42 * MU_ * np.sin(CIRCLE_T)
print(f"U shape {MU_}x{NU_}: concavity = {int(channel.sum())} px between rows {mouth_row + 1} (mouth) and {base_row + 1} (base)")
print(f"{'sigma':>6} | {'channel px with > 1 % of the peak force':>38} | {'snake penetration':>18}")
trad = {}
for sg in (1.0, 2.0, 4.0):                                                          # Eq. (6.13) E_ext = -(G_sigma * I)
    blur = gaussian_blur(U.astype(np.float64), sg)
    bx, by = unit_max(*gradient2(blur))
    xs, ys, hist = evolve(bx, by, xU, yU, 40, kappa=1.0)
    trad[sg] = (bx, by, xs, ys, hist)
    print(f"{sg:>6g} | {100.0 * (np.hypot(bx, by)[channel] > 0.01).mean():37.1f}% | {penetration(xs, ys):17.0f} %")
SIGMA_CMP = 2.0                                                                     # the field section 6.2 compares against
bx, by, xs, ys, hist = trad[SIGMA_CMP]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
imshow_matlab(axes[0], U, title="(a) U-shaped object (synthetic)")
imshow_matlab(axes[1], -gaussian_blur(U.astype(np.float64), 4.0), autoscale=True,
              title="(b) E_ext = -G_sigma * I, sigma = 4 (Eq. 6.13)")
quiver_field(axes[2], bx, by, step=2)
axes[2].set_title(f"(c) F_ext = -grad E_ext, sigma = {SIGMA_CMP:g}: {penetration(xs, ys):.0f} % penetration")
for h in hist[1:-1]:
    snake_plot(axes[2], *h, "y-", lw=0.4)
snake_plot(axes[2], xs, ys, "g-", lw=1.4)
fig.suptitle("Fig. 6.7 (synthetic U)  The traditional external force has no component that pulls the snake into the concavity")
plt.show()
''')
md(r"""
The sweep is the quantitative version of §6.1.1.2 vs §6.1.3: with a small $\sigma$ almost no force reaches the
channel and the snake stalls at its mouth; blurring more buys capture range but also blurs the boundary the snake
is supposed to find. GVF gets both at once.
""")

# =====================================================================================================================
# 6.2 GVF snake
# =====================================================================================================================
md(r"""
## 6.2 Gradient vector flow (GVF) snake (pp. 123–128) — Eqs. (6.41)–(6.56), Figs. 6.8–6.9; `GVF.m`

Instead of using $\nabla f$ of an edge map $f$ directly, GVF **diffuses** it into a field $v = (u, w)$ that
minimises

$$E = \iint \mu\left(u_x^2 + u_y^2 + w_x^2 + w_y^2\right) + \left|\nabla f\right|^2\left|v - \nabla f\right|^2 dx\,dy \quad (6.41).$$

Near an edge the second term dominates and $v \approx \nabla f$; in homogeneous regions the first dominates and
the Euler equations

$$\mu\nabla^2 u - (u - f_x)(f_x^2 + f_y^2) = 0 \quad (6.50a)$$

reduce to **Laplace's equation**, so the field is smoothly interpolated from the boundaries — which is exactly
how it acquires a large capture range *and* vectors that point into concavities. Solving (6.50) as the
generalized diffusion (6.51) with the explicit scheme (6.52)–(6.53) gives `GVF.m`'s one-line update, stable while
$r = \mu\Delta t/(\Delta x\Delta y) \le 1/4$ (Eqs. 6.54/6.55); with $\Delta t = \Delta x = \Delta y = 1$ that is
$\mu \le 1/4$, and footnote 3 (p. 125) fixes **μ = 0.1 for every GVF snake in this book**. The snake equations are
unchanged apart from replacing $\partial E_{ext}/\partial x$ by $-u$ (Eq. 6.56), i.e. `snakedeform`'s $+\kappa v$.
The mirror boundary condition of `BoundMirrorExpand/Ensure` is the natural condition $\nabla u \cdot d\sigma = 0$
of Eq. (6.46).
""")
code(r'''
# MATLAB writes the 5-point Laplacian of Eq. (6.52c) as 4*del2 — check the identity, and the CFL number
R = np.random.default_rng(0).normal(size=(9, 11))
lap5 = (R[2:, 1:-1] + R[:-2, 1:-1] + R[1:-1, 2:] + R[1:-1, :-2] - 4 * R[1:-1, 1:-1])
print(f"4*del2(u) == the 5-point Laplacian on the interior: "
      f"{np.allclose(4 * del2(R)[1:-1, 1:-1], lap5)} (max diff {np.abs(4 * del2(R)[1:-1, 1:-1] - lap5).max():.2e})")
print(f"CFL (Eqs. 6.54/6.55): r = mu*dt/(dx*dy) = {P['mu']} <= 0.25 -> stable; the port refuses mu > 0.25 at unit steps.")

# --- Fig. 6.8/6.9: the plain gradient field vs the GVF field of the same U shape --------------------------------
f_edge = gaussian_blur(U.astype(np.float64), SIGMA_CMP)           # the same edge map as the sigma = 2 row above
gu, gw = gvf(f_edge / (f_edge.max() or 1.0), P["mu"], 200)        # Eqs. (6.41)/(6.53), mu = 0.1
gpx, gpy = unit_max(gu, gw)
gxs, gys, ghist = evolve(gpx, gpy, xU, yU, 40, kappa=1.0)
print(f"\nplain gradient (sigma = {SIGMA_CMP:g}): {100.0 * (np.hypot(bx, by)[channel] > 0.01).mean():5.1f} % of the channel "
      f"carries > 1 % of the peak force, mean downward component {by[channel].mean():+.4f}")
print(f"GVF (200 iterations, mu = {P['mu']}): {100.0 * (np.hypot(gpx, gpy)[channel] > 0.01).mean():5.1f} % of the channel, "
      f"mean downward component {gpy[channel].mean():+.4f}")
print(f"snake penetration into the concavity: traditional {penetration(xs, ys):.0f} %, GVF {penetration(gxs, gys):.0f} %")

fig, axes = plt.subplots(2, 2, figsize=(12, 11))
quiver_field(axes[0, 0], bx, by, step=2)
axes[0, 0].set_title(f"(a) grad(G_sigma * I), sigma = {SIGMA_CMP:g}: force only near the edges")
quiver_field(axes[0, 1], gpx, gpy, step=2)
axes[0, 1].set_title(f"(b) GVF field v = (u, w), {200} iterations, mu = {P['mu']}")
imshow_matlab(axes[1, 0], U, title=f"traditional snake: {penetration(xs, ys):.0f} % penetration")
for h in hist[1:-1]:
    snake_plot(axes[1, 0], *h, "y-", lw=0.4)
snake_plot(axes[1, 0], *hist[0], "r-", lw=1.0); snake_plot(axes[1, 0], xs, ys, "g-", lw=1.4)
imshow_matlab(axes[1, 1], U, title=f"GVF snake: {penetration(gxs, gys):.0f} % penetration")
for h in ghist[1:-1]:
    snake_plot(axes[1, 1], *h, "y-", lw=0.4)
snake_plot(axes[1, 1], *ghist[0], "r-", lw=1.0); snake_plot(axes[1, 1], gxs, gys, "g-", lw=1.4)
fig.suptitle("Figs. 6.8 / 6.9 (synthetic U)  The GVF field fills the whole image and points into the concavity")
plt.show()
''')
md(r"""
Left: the raw gradient field is empty except within a couple of pixels of the boundary. Right: after 200
diffusion iterations the same information has spread over the whole image and, inside the channel, the vectors
have acquired a downward component — which is precisely the property Eq. (6.50)'s Laplace behaviour predicts.
""")

md(r"""
### `for test/for_test.m` — one GVF snake on the book's own `test8.jpg`

The chapter's own single-contour demo: `Num = 150` GVF iterations on the gradient edge map of the gray image,
one circle of radius **20** at $(x_0, y_0) = (80, 40)$ clipped to the image rectangle (`polybool`), then
`iter = 50` snake iterations in blocks of 5 with a `snakeinterp` respacing after each block, and finally the
contour is *burnt* into the Otsu mask (`bw(ceil(y), ceil(x)) = 0`) — the boundary superposition of Algorithm 1
step 10. `replaces ch6/for test/for_test.m`.
""")
code(r'''
from seaice.ch06_gvf_snake import gvf_force_field
from seaice.core.polygon import clip_polygon_rect

FT = BOOK_PARAMS["for_test"]
g8 = rgb2gray_matlab(T8)
lvl8, _ = graythresh(g8)
bw8 = im2bw(g8, lvl8)
s1, s2 = bw8.shape
t0 = time.perf_counter()
f2_8, u8, v8, px8, py8 = gvf_force_field(g8, sigma=FT["sigma"], gradient_on=bool(FT["GradientOn"]),
                                         gvf_on=bool(FT["GVFOn"]), num=FT["Num"], mu=FT["mu"])
x = FT["x0"] + FT["r"] * np.cos(CIRCLE_T)
y = FT["y0"] + FT["r"] * np.sin(CIRCLE_T)
xc, yc = clip_polygon_rect(x, y, (0.0, float(s2)), (0.0, float(s1)))      # polybool('intersection', s_2, s_1, x, y)
xi, yi = snakeinterp(xc, yc, FT["Dmax"], FT["Dmin"])
hist8 = [(xi.copy(), yi.copy())]
xs8, ys8 = xi, yi
for i in range(int(np.ceil(FT["iter"] / 5))):
    xs8, ys8 = snakedeform(xs8, ys8, FT["alpha"], FT["beta"], FT["gamma"], FT["kappa"], px8, py8, 5)
    xs8, ys8 = snakeinterp(xs8, ys8, FT["Dmax"], FT["Dmin"])
    hist8.append((xs8.copy(), ys8.copy()))
bw8_out = bw8.copy()
xx, yy = np.ceil(xs8).astype(np.int64), np.ceil(ys8).astype(np.int64)
ok = (xx >= 1) & (yy >= 1) & (xx <= s2) & (yy <= s1)
bw8_out[yy[ok] - 1, xx[ok] - 1] = False
print(f"for_test.m on {'test8.jpg' if FROM_BOOK else 'the substitute'} ({s1}x{s2}): graythresh {lvl8:.6f}, "
      f"{int(bw8.sum())} ice px; GVF {FT['Num']} iterations, |v| max {np.hypot(u8, v8).max():.4f}")
print(f"initial circle {CIRCLE_T.size} points -> after the rectangle clip {xc.size} -> after snakeinterp {xi.size}; "
      f"{len(hist8) - 1} blocks of 5 iterations, final contour {len(xs8)} points")
print(f"burnt {int(bw8.sum()) - int(bw8_out.sum())} boundary pixels into the mask  ({time.perf_counter() - t0:.1f} s)"
      f"{matlab_ref('bw, f2, u, v, px, py all 0.0; the burnt pixel set identical, 91 px')}")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
imshow_matlab(axes[0, 0], T8, title=f"(a) {'test8.jpg' if FROM_BOOK else 'substitute'} ({s1}x{s2})")
imshow_matlab(axes[0, 1], f2_8, autoscale=True, title="(b) edge map f2 = |grad(double(I))|  (Eq. 6.7)")
quiver_field(axes[1, 0], px8, py8, step=3, scale=60)
axes[1, 0].set_title(f"(c) normalised GVF force (px, py), {FT['Num']} iterations")
imshow_matlab(axes[1, 1], T8, title="(d) snake: red = initial circle r = 20 at (80, 40), yellow = every 5 iterations, green = final")
snake_plot(axes[1, 1], *hist8[0], "r-", lw=1.2)
for h in hist8[1:-1]:
    snake_plot(axes[1, 1], *h, "y-", lw=0.6)
snake_plot(axes[1, 1], xs8, ys8, "g-", lw=1.5)
fig.suptitle("for test/for_test.m  A single GVF snake, from the manual circle to the burnt boundary")
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
imshow_matlab(axes[0], bw8, title="im2bw(I, graythresh(I)) before")
imshow_matlab(axes[1], bw8_out, title="after burning the contour: bw(ceil(y), ceil(x)) = 0")
fig.suptitle("Algorithm 1 step 10 in miniature — the snake boundary is written into the mask")
plt.show()
''')

# =====================================================================================================================
# 6.3 Contours initialization
# =====================================================================================================================
md(r"""
## 6.3 Contours initialization (pp. 128–136) — Eqs. (6.57)/(6.58), Figs. 6.11–6.14; `for test/dist.m`

A parametric snake cannot split (Fig. 6.11), so it can only segment **one** object at a time: the pipeline needs
one contour per floe, and the contour must be placed well. §6.3.1 (Fig. 6.12): a contour started in the water
finds the *water* boundary, one on a weak connection finds the connection, one near the boundary recovers only
part of it — so initialise **inside the floe, as close to its centre as possible**. §6.3.2 (Fig. 6.13): the shape
is a **circle**, because it deforms most uniformly towards an unknown irregular outline, and it should start as
close to the true boundary as possible (a tiny circle needs more iterations and can be trapped by reflections or
speckle). §6.3.3 makes this automatic with the distance transform of Chapter 5: all regional maxima of $D$ come
from grayscale reconstruction by dilation,

$$M_{max} = I - R^D_I(I - 1) \quad (6.57), \qquad M_{max} = I + 1 - R^D_{I+1}(I) \quad (6.58),$$

maxima closer than $T_{seed}$ are merged by a dilation, the **centre of each merged region is a seed** and the
**radius of its circle is the distance value at the seed** — divided by $\sqrt 2$ because the **city-block**
metric is used (footnote 4, p. 135), so the circle stays strictly inside the floe. Fig. 6.14 is the printed 8×8
worked example, and it reproduces bit-exactly.
""")
code(r'''
from seaice.core.morphology import imregionalmin, regional_maxima_by_reconstruction
from seaice.ch06_gvf_snake import initialize_contours

A14 = synth.FIG_6_14_IMAGE                                  # the printed Fig. 6.14(a) matrix
D14 = bwdist(~A14, "cityblock")                             # Fig. 6.14(b)
print(f"bwdist(~A, 'cityblock') == the printed Fig. 6.14(b) matrix: {np.array_equal(D14, synth.FIG_6_14_DISTANCE)}"
      f"  (bit-exact, no image needed)")
imgd = -D14.astype(np.float64); imgd[~A14] = -np.inf        # the script's recipe
maxima = imregionalmin(imgd) & A14
d_book = D14.astype(np.float64).copy(); d_book[~A14] = -np.inf
m657 = regional_maxima_by_reconstruction(d_book, form="6.57") & A14
m658 = regional_maxima_by_reconstruction(d_book, form="6.58") & A14
pts = np.argwhere(maxima)
print(f"regional maximum: {int(label_components(maxima, 8).max())} component of {int(maxima.sum())} local maxima at "
      f"{[(int(r) + 1, int(c) + 1) for r, c in pts]}, all of value {sorted(set(D14[maxima].astype(int).tolist()))}"
      f"{book('a regional maximum consisting of three local maxima')}")
print(f"Eq. (6.57) and Eq. (6.58) give the same set: {np.array_equal(m657, maxima)} / {np.array_equal(m658, maxima)}")
init14 = initialize_contours(A14, se_radius=3)
for n in range(init14.num):
    cxx, cyy = init14.centroids[n]
    print(f"seed {n + 1}: centroid (x, y) = ({cxx:.4f}, {cyy:.4f}), D(seed) = {int(D14[round(cyy) - 1, round(cxx) - 1])}, "
          f"r = D/sqrt(2) = {init14.radii[n]:.4f}{book('3/sqrt(2) = 2.1213')}")

fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
show_matrix(axes[0], A14.astype(int), fmt="{:d}", title="(a) binary image matrix")
show_matrix(axes[1], D14.astype(int), fmt="{:d}", title="(b) city-block distance transform, seed and initial contour")
axes[1].plot(pts[:, 1], pts[:, 0], "rs", ms=14, mfc="none", mew=1.5, label="regional maximum")
for n in range(init14.num):
    cxx, cyy = init14.centroids[n]
    r14 = init14.radii[n]
    axes[1].plot(cxx - 1, cyy - 1, "g+", ms=14, mew=2, label="seed" if n == 0 else None)
    axes[1].plot(cxx - 1 + r14 * np.cos(CIRCLE_T), cyy - 1 + r14 * np.sin(CIRCLE_T), "g-", lw=1.5,
                 label=f"initial contour r = {r14:.4f}" if n == 0 else None)
axes[1].legend(loc="upper right", fontsize=7)
fig.suptitle("Figure 6.14  Contour initialization based on the distance transform (printed 8x8 truth, reproduced bit-exactly)")
plt.show()
''')
code(r'''
# --- for test/dist.m on sea_ice_test.jpg: the same recipe on a real image ---------------------------------------
DP = BOOK_PARAMS["dist"]
gs = rgb2gray_matlab(STRIP).T                                # dist.m: bw = im2bw(I', graythresh(I')) -- it transposes
lvl_s, _ = graythresh(gs)
bw_s = im2bw(gs, lvl_s)
t0 = time.perf_counter()
init_s = initialize_contours(bw_s, se_radius=DP["se_radius"])       # se = strel('disk', 5)
print(f"dist.m on {'sea_ice_test.jpg' if FROM_BOOK else 'the substitute'} (transposed to {bw_s.shape[0]}x{bw_s.shape[1]}): "
      f"{int(bw_s.sum())} ice px, {int(init_s.dis.sum())} local maxima -> imdilate(strel('disk', {DP['se_radius']})) -> "
      f"{init_s.num} seeds  ({time.perf_counter() - t0:.1f} s)"
      f"{matlab_ref('bw, img_Dist, Dis_img, dis all 0 px; 540 seeds; radii <= 4.5e-7')}")
if init_s.num:
    print(f"radii r = D(seed)/sqrt(2): min {init_s.radii.min():.2f}, median {np.median(init_s.radii):.2f}, "
          f"max {init_s.radii.max():.2f} px; {int((init_s.radii == 2.0).sum())} seeds take the 'r == 0 -> 2' fallback")
fig, axes = plt.subplots(1, 3, figsize=(17, 6.4))
imshow_matlab(axes[0], init_s.img_dist, autoscale=True, title="city-block distance transform of the mask")
imshow_matlab(axes[1], bw_s, title=f"fixed radius r = {DP['r']} at every seed (dist.m's first figure)")
imshow_matlab(axes[2], bw_s, title=f"r = D(seed)/sqrt(2) (footnote 4): circles stay inside the floes")
for n in range(init_s.num):
    cxx, cyy = init_s.centroids[n]
    for ax, rr in ((axes[1], float(DP["r"])), (axes[2], init_s.radii[n])):
        ax.plot(cxx - 1, cyy - 1, "r+", ms=4, mew=0.7)
        ax.plot(cxx - 1 + rr * np.cos(CIRCLE_T), cyy - 1 + rr * np.sin(CIRCLE_T), "b-", lw=0.5)
fig.suptitle("for test/dist.m (section 6.3.3)  Seeds from the regional maxima of the distance transform and their initial circles")
plt.show()
''')
md(r"""
A fixed radius is either too small (slow, and vulnerable to reflections — Fig. 6.13) or crosses the floe
boundary; the distance value at the seed adapts it to each floe, and the $\sqrt2$ divisor compensates for the
city-block metric over-estimating the true Euclidean clearance. Several seeds inside one floe are harmless — only
slower — because each snake converges to the same boundary.
""")
code(r'''
# --- sections 6.3.1/6.3.2 (Figs. 6.12/6.13): where and how big the initial circle should be -------------------
# The book's own images for these two figures are not shipped, so the rules are measured on the SYNTHETIC floe.
STARTS = [("centre, r = D(centre)/sqrt(2) (the section 6.3.3 rule)", cx, cy, float(bwdist(~bwF, "cityblock").max()) / CITYBLOCK_RADIUS_DIVISOR),
          ("centre, but a tiny circle r = 3 (section 6.3.2)", cx, cy, 3.0),
          ("in the water, off the floe (section 6.3.1)", cx + 70.0, cy + 55.0, 6.0)]
fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
for ax, (tag, sx, sy, sr) in zip(axes, STARTS):
    xs_i, ys_i = snakeinterp(sx + sr * np.cos(CIRCLE_T), sy + sr * np.sin(CIRCLE_T), P["Dmax"], P["Dmin"])
    hist_i, dists = [(xs_i.copy(), ys_i.copy())], []
    for _ in range(30):                                  # 30 blocks of 5 iterations
        xs_i, ys_i = snakedeform(xs_i, ys_i, P["alpha"], P["beta"], P["gamma"], P["kappa"], px_f, py_f, 5)
        xs_i, ys_i = snakeinterp(xs_i, ys_i, P["Dmax"], P["Dmin"])
        hist_i.append((xs_i.copy(), ys_i.copy()))
        dists.append(sample(px_f, py_f, xs_i, ys_i)[1])
    dists = np.array(dists)
    hit = int(np.argmax(dists < 1.0)) + 1 if (dists < 1.0).any() else 0
    reached = f"reaches the boundary (mean distance < 1 px) after {5 * hit} iterations" if hit else "never reaches the boundary"
    print(f"{tag:52s} r0 = {sr:5.1f}: {reached}; final mean distance {dists[-1]:.2f} px, {len(xs_i)} points")
    imshow_matlab(ax, FLOE, title=f"{tag}\n{reached}")
    for h in hist_i[1:-1]:
        snake_plot(ax, *h, "y-", lw=0.4)
    snake_plot(ax, *hist_i[0], "r-", lw=1.2)
    snake_plot(ax, xs_i, ys_i, "g-", lw=1.4)
fig.suptitle("Figs. 6.12 / 6.13 (mechanism, synthetic)  Start inside the floe and as close to its boundary as possible")
plt.show()

# The GVF field of the loaded ice image, computed once here and reused by Algorithm 1 in section 6.4.
t0 = time.perf_counter()
FIELD = gvf_force_field(GRAY, sigma=P["sigma"], gradient_on=bool(P["GradientOn"]), gvf_on=bool(P["GVFOn"]),
                        num=P["Num"], mu=P["mu"])            # (f2, u, v, px, py)
print(f"\nGVF field of the {M}x{N} ice image, {P['Num']} iterations, mu = {P['mu']}: {time.perf_counter() - t0:.1f} s")
''')

# =====================================================================================================================
# 6.4 Ice image segmentation
# =====================================================================================================================
md(r"""
## 6.4 Ice image segmentation (pp. 135–137) — **Algorithm 1**, Fig. 6.15; `GVF_distance.m`, `seaice_kmean_GVF_forenhancement.m`

Everything above is assembled into Algorithm 1: compute the GVF field of the gray image; binarise; take the
city-block distance map of the mask; extract its regional maxima; merge them into seeds; and for each seed run a
GVF snake from the circle of radius $D(\text{seed})/\sqrt2$ and **superimpose** the resulting boundary on the
segmentation ($\texttt{bw1}(\lceil y \rceil, \lceil x \rceil) = 0$), so connected floes end up separated. The
boundary pixels are labelled "**residue ice**" for later handling (p. 135). The shipped code implements a
superset: an outer `for time = 1:timer` loop that only re-segments the components failing three criteria —
`Area > Ra`, `Solidity < Rc`, `MajorAxisLength/MinorAxisLength > Rl` — which are documented not here but on
p. 205 of Chapter 9; every Chapter 6 driver sets `timer = 1`, so one pass runs. Fig. 6.15(a)–(f) is the whole
procedure on `alg_seg_gray.jpg`. `replaces ch6/Sea_Ice_Floe_Identification/GVF_distance.m`.
""")
code(r'''
from seaice.ch06_gvf_snake import gvf_distance

t0 = time.perf_counter()
res = gvf_distance(ALG, sigma=P["sigma"], GradientOn=P["GradientOn"], GVFOn=P["GVFOn"], Num=P["Num"], mu=P["mu"],
                   iter=P["iter"], alpha=P["alpha"], beta=P["beta"], gamma=P["gamma"], kappa=P["kappa"],
                   Dmin=P["Dmin"], Dmax=P["Dmax"], Ra_min=P["Ra_min"], Ra=P["Ra"], Rc=P["Rc"], Rl=P["Rl"],
                   se_radius=P["se_radius"], timer=P["timer"], field_cache=FIELD)     # the field computed above
dt = time.perf_counter() - t0
rec = res.passes[0]
init = rec.init
n_before = int(label_components(res.bw, 4).max())
n_after = int(label_components(res.bw1, 4).max())
print(f"Algorithm 1 on {IMG_LABEL} ({M}x{N}) in {dt:.1f} s")
print(f"  bwlabel(bw, 4): {rec.num} components; failing the Ch. 9 p. 205 criteria: {rec.k.size} "
      f"(Area > {P['Ra']}: {int((rec.area > P['Ra']).sum())}, Solidity < {P['Rc']}: {int((rec.solidity < P['Rc']).sum())}, "
      f"axis ratio > {P['Rl']}: {int((rec.rl > P['Rl']).sum())})"
      f"{matlab_ref('83 components, 32 selected — identical')}")
if init is not None:
    print(f"  {int(init.dis.sum())} local maxima -> {init.num} seeds; radii min {init.radii.min():.2f}, "
          f"median {np.median(init.radii):.2f}, max {init.radii.max():.2f}"
          f"{matlab_ref('171 maxima px, 46 seeds, centroids <= 1e-12, radii <= 1e-6')}")
print(f"  {res.n_seeds_run} snakes run; boundaries burnt: {int(res.bw.sum()) - int(res.bw1.sum())} px "
      f"('residue ice'); 4-connected floes {n_before} -> {n_after}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
imshow_matlab(axes[0, 0], res.gray, title="(a) grayscale ice image")
imshow_matlab(axes[0, 1], res.bw, title=f"(b) im2bw(I, graythresh(I)): {n_before} connected components")
if init is not None:
    imshow_matlab(axes[0, 2], init.img_dist, autoscale=True, title="(c) city-block distance transform")
    mx = np.argwhere(init.dis != 0)
    imshow_matlab(axes[1, 0], rec.bw2, title=f"(d) regional maxima of D ({mx.shape[0]} px, '+')")
    axes[1, 0].plot(mx[:, 1], mx[:, 0], "r+", ms=3, mew=0.5)
    imshow_matlab(axes[1, 1], res.bw, title=f"(e) {init.num} seeds and their initial circles")
    for n in range(init.num):
        cxx, cyy = init.centroids[n]
        rr_ = init.radii[n]
        axes[1, 1].plot(cxx - 1, cyy - 1, "r+", ms=4, mew=0.7)
        axes[1, 1].plot(cxx - 1 + rr_ * np.cos(CIRCLE_T), cyy - 1 + rr_ * np.sin(CIRCLE_T), "b-", lw=0.5)
imshow_matlab(axes[1, 2], label2rgb(label_components(res.bw1, 4)),
              title=f"(f) segmentation: {n_before} -> {n_after} floes")
fig.suptitle("Figure 6.15  GVF snake-based ice image segmentation (Algorithm 1) — GVF_distance.m")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
imshow_matlab(axes[0], res.bw, title=f"(f) the {len(rec.seeds)} final snake contours on the binary image")
for s in rec.seeds:
    snake_plot(axes[0], s.x_final, s.y_final, "r-", lw=0.7)
imshow_matlab(axes[1], res.bw & ~res.bw1, title="the burnt boundary pixels alone = 'residue ice' (p. 135)")
fig.suptitle("Algorithm 1 step 10  Each converged snake is written into the mask, cutting the connected floes apart")
plt.show()
''')
md(r"""
Every seed produced its own contour and each converged onto the boundary of the floe it started in; where two
floes were merged into one binary component, the two contours meet and the burnt pixels split them. Seeds that
share a floe simply retrace the same boundary.
""")

md(r"""
### `seaice_kmean_GVF_forenhancement.m` — Algorithm 1 twice, for a three-level segmentation

`sea_ice_demo.m` does not call `GVF_distance` directly: it calls a variant that runs the same body **twice** on
one shared GVF field — once on the Otsu mask (bright ice) and once on the residual `bw0 = bk − bw` of a
**k-means** mask `bk` with `kms0 = 3` clusters, in which only the darkest cluster is water (§3.2). The return is
`out = bw1 + 0.5·bw0`, three levels: **1 = bright ice floes, 0.5 = dark / slush ice, 0 = water**, which
Chapter 7's `ice_shape_enhancement` consumes as `seg == 1` and `seg == 0.5`. Chapter 6 ships no `kmeans.m`, so
this call is the Statistics Toolbox routine (k-means++ on the raw gray values), ported as
`core.clustering.kmeans_lloyd`; its labels are `approx` by nature, but the derived mask matched MATLAB's
`rng(0)` run pixel-for-pixel on both book images.
""")
code(r'''
from seaice.ch06_gvf_snake import seaice_kmean_gvf

t0 = time.perf_counter()
km = seaice_kmean_gvf(ALG, kms0=P["kms0"], Num=P["Num"], mu=P["mu"], iter=P["iter"], alpha=P["alpha"],
                      beta=P["beta"], gamma=P["gamma"], kappa=P["kappa"], Dmin=P["Dmin"], Dmax=P["Dmax"],
                      Ra_min=P["Ra_min"], Ra=P["Ra"], Rc=P["Rc"], Rl=P["Rl"], se_radius=P["se_radius"],
                      timer=P["timer"], kmeans_seed=0)
print(f"seaice_kmean_GVF_forenhancement on {IMG_LABEL}: {time.perf_counter() - t0:.1f} s")
print(f"  k-means (kms0 = {P['kms0']}) sorted cluster centres: {np.round(np.sort(km.s0), 4).tolist()}"
      f"{matlab_ref('[107.872949, 152.264015, 198.773459]; bk 0 px vs MATLAB rng(0)')}")
print(f"  bk (only the darkest cluster is water): {int(km.bk.sum())} ice px vs the Otsu mask's {int(km.bw.sum())}; "
      f"bw0 = bk - bw has {km.n_negative} px at -1 (the bwareaopen quirk is latent here) and "
      f"{int(km.bw0.sum())} px after bwareaopen({P['Ra_min']}, 4)")
lv, ct = np.unique(km.out, return_counts=True)
print(f"  out levels {lv.tolist()} with {ct.tolist()} pixels  (1 = bright ice, 0.5 = dark/slush ice, 0 = water)")
fig, axes = plt.subplots(1, 4, figsize=(19, 5.4))
imshow_matlab(axes[0], km.bw, title="pass 1 input: Otsu mask")
imshow_matlab(axes[1], km.bk, title=f"k-means mask bk ({P['kms0']} clusters, darkest = water)")
imshow_matlab(axes[2], km.bw0, title="pass 2 input: bw0 = bwareaopen(bk - bw)")
axes[3].imshow(km.out, cmap="viridis", interpolation="nearest"); axes[3].set_axis_off()
axes[3].set_title("out = bw1 + 0.5*bw0: three levels")
fig.suptitle("seaice_kmean_GVF_forenhancement.m  The same Algorithm 1 run on bright ice and on the k-means residual")
plt.show()
''')

# =====================================================================================================================
# 6.5 Discussion
# =====================================================================================================================
md(r"""
## 6.5 Discussion (pp. 137–144) — Figs. 6.16–6.21

**6.5.1 Stopping criterion.** Neither Kass et al. nor Xu & Prince gives one; the code simply caps the iteration
count (`iter = 100`, in blocks of 5 with a `snakeinterp` between blocks). Stopping when the energy decrease falls
below a threshold is risky, because a snake can crawl slowly along a channel and then accelerate again.

**6.5.2 Capture range.** The capture range is set by the **number of GVF iterations**: the field diffuses one
pixel further per iteration or so, and the rule of thumb is that it should scale with object size. Fig. 6.16 is a
$110\times186$ binary image with a **61-pixel** and a **9-pixel** diameter circle: 5 iterations already fill the
small circle, 30 are still not enough for the large one. Too large a capture range lets strong edges dominate
weak ones (under-segmentation, Fig. 6.18: 800 vs 80 iterations); too small a one lets noise block the snake
(over-segmentation, Fig. 6.19: 600 vs 60). Fig. 6.17 shows the same snake reaching or not reaching the large
circle under 250 vs 30 GVF iterations. The image is fully specified in the text, so `synth.fig_6_16_circles()`
reproduces it (the circle *positions* are not printed and are chosen to match the printed layout); the edge map
is the chapter's own $f = |\nabla I|$.
""")
code(r'''
BW16 = synth.fig_6_16_circles()                                # 110x186, 61-px and 9-px diameters (printed spec)
lab16 = label_components(BW16, 8)
sizes = [int((lab16 == k).sum()) for k in range(1, int(lab16.max()) + 1)]
big16, small16 = lab16 == int(np.argmax(sizes)) + 1, lab16 == int(np.argmin(sizes)) + 1
f16 = gradient2_magnitude(BW16.astype(np.float64))             # Eq. (6.7), as GVF_distance.m line 58
ITERS = list(BOOK_PARAMS["figures"]["gvf_iters_fig_6_16"])     # 5, 30, 100, 250
print(f"Fig. 6.16(a): {BW16.shape[0]}x{BW16.shape[1]} binary image with the printed 61-px and 9-px diameters "
      f"({int(big16.sum())} and {int(small16.sum())} pixels; the discrete 2*max(bwdist) measure gives "
      f"{2 * float(bwdist(~big16, 'euclidean').max()):.0f} and {2 * float(bwdist(~small16, 'euclidean').max()):.0f} px)")
print(f"{'GVF iterations':>14} | {'|v| max':>8} | {'61-px circle filled':>19} | {'9-px circle filled':>18}")
fields = {}
for it in ITERS:
    uu, vv = gvf(f16, P["mu"], it)
    m16 = np.hypot(uu, vv)
    fields[it] = (uu, vv)
    print(f"{it:>14} | {m16.max():8.4f} | {100 * (m16[big16] > 0.02).mean():18.1f}% | {100 * (m16[small16] > 0.02).mean():17.1f}%")
fig, axes = plt.subplots(len(ITERS) + 1, 1, figsize=(11, 3.6 * (len(ITERS) + 1)))
imshow_matlab(axes[0], BW16, title="(a) 110x186 binary image: 61-px and 9-px diameter circles (synthetic, printed spec)")
for ax, it in zip(axes[1:], ITERS):
    quiver_field(ax, *fields[it], step=2, scale=25)
    ax.set_title(f"GVF field via {it} iterations")
fig.suptitle("Figure 6.16  The capture range grows with the number of GVF iterations")
plt.show()
print(book("5 iterations are sufficient for the 9-pixel circle; 30 are still not enough for the 61-pixel circle").strip()
      or "(the two printed claims are the two columns above)")
''')
code(r'''
# --- Fig. 6.17: the same snake under 30 vs 250 GVF iterations ---------------------------------------------------
rr16, cc16 = np.nonzero(big16)
cy16, cx16 = rr16.mean() + 1.0, cc16.mean() + 1.0
d_out = bwdist(~big16, "euclidean")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, it in zip(axes, BOOK_PARAMS["figures"]["gvf_iters_fig_6_17"]):
    uu, vv = gvf(f16, P["mu"], it)
    m16 = np.hypot(uu, vv)
    xs16, ys16, h16 = evolve(uu / (m16 + 1e-10), vv / (m16 + 1e-10),
                             cx16 + 6.0 * np.cos(CIRCLE_T), cy16 + 6.0 * np.sin(CIRCLE_T), 40)
    rows = np.clip(np.round(ys16).astype(int) - 1, 0, BW16.shape[0] - 1)
    cols = np.clip(np.round(xs16).astype(int) - 1, 0, BW16.shape[1] - 1)
    print(f"{it:>3} GVF iterations: final contour {len(xs16)} points, enclosed half-width "
          f"{0.5 * (xs16.max() - xs16.min()):.1f} px (the circle's radius is 30.5), mean distance to the boundary "
          f"{d_out[rows, cols].mean():.2f} px")
    imshow_matlab(ax, BW16, title=f"evolution under {it} GVF iterations")
    for h in h16[:-1]:
        snake_plot(ax, *h, "y-", lw=0.5)
    snake_plot(ax, *h16[0], "r-", lw=1.2)
    snake_plot(ax, xs16, ys16, "g-", lw=1.4)
    ax.set_xlim(cx16 - 60, cx16 + 60); ax.set_ylim(cy16 + 55, cy16 - 55)
fig.suptitle("Figure 6.17  A small starting circle reaches the boundary only when the capture range is large enough")
plt.show()
''')
md(r"""
**6.5.3 Border effects.** Large images have to be processed in sub-images, but a floe cut by a sub-image border
segments badly for two reasons: the snake cannot split and stalls against the border (Fig. 6.20), and the GVF
field of an *incomplete* floe radiates from the **image border** rather than from the floe centre
(Fig. 6.21) — so the contour cannot even be initialised sensibly near the cut. The book's own aerial and
model-basin images for Figs. 6.18–6.21 are not shipped, so the cell below shows only the *mechanism*, on the
loaded ice mask: cutting it in two and re-running the §6.3.3 initialisation moves seeds towards the cut, inflates
their radii (the distance transform of a truncated floe measures the distance to the image edge, not to water)
and produces initial circles that have to be clipped away.
""")
code(r'''
# The mechanism, measured on the loaded ice mask: cut it in two and re-run the section 6.3.3 initialisation.
CUT = N // 2
init_full = initialize_contours(BW, se_radius=P["se_radius"])          # seeds of the whole mask
init_cut = initialize_contours(BW[:, :CUT], se_radius=P["se_radius"])  # seeds of the left sub-image
Df, Dc = init_full.img_dist, init_cut.img_dist
near_full = [(cxx, cyy, rr) for (cxx, cyy), rr in zip(init_full.centroids, init_full.radii) if CUT - 15 < cxx <= CUT]
near_cut = [(cxx, cyy, rr) for (cxx, cyy), rr in zip(init_cut.centroids, init_cut.radii) if cxx > CUT - 15]
outside = sum(1 for cxx, _, rr in near_cut if cxx + rr > CUT)
print(f"whole mask: {init_full.num} seeds; left sub-image (cut at column {CUT}): {init_cut.num} seeds")
print(f"mean distance value in the 5 columns next to the cut: whole {Df[:, CUT - 5:CUT].mean():.2f}, "
      f"sub-image {Dc[:, -5:].mean():.2f} — the cut has no water beyond it, so the distance transform "
      "over-estimates the clearance of every floe it truncates")
print(f"seeds within 15 px of the cut: {len(near_full)} in the whole mask, {len(near_cut)} in the sub-image; "
      f"{outside} of the latter have circles that stick out of the sub-image and must be clipped "
      "(the polybool step of Algorithm 1)")
fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
for ax, (tag, bwx, ini) in zip(axes, [("whole mask", BW, init_full), (f"left sub-image (cut at column {CUT})", BW[:, :CUT], init_cut)]):
    imshow_matlab(ax, bwx, title=f"{tag}: {ini.num} seeds")
    for n in range(ini.num):
        cxx, cyy = ini.centroids[n]
        rr_ = ini.radii[n]
        ax.plot(cxx - 1, cyy - 1, "r+", ms=4, mew=0.7)
        ax.plot(cxx - 1 + rr_ * np.cos(CIRCLE_T), cyy - 1 + rr_ * np.sin(CIRCLE_T), "b-", lw=0.5)
    ax.axvline(CUT - 1, color="c", lw=1.0)
fig.suptitle("Section 6.5.3 (mechanism)  A floe cut by a sub-image border gets seeds pushed towards the cut and circles that leave the image")
plt.show()
''')

# =====================================================================================================================
# Parameter play
# =====================================================================================================================
md(r"""
## Parameter play (optional)

Two knobs of §6.3.3 on the loaded image: the **distance metric** (city-block is the book's choice, footnote 4)
and the **radius of the disk that merges the regional maxima into seeds** (`strel('disk', 3)` in the shipped
code — MATLAB's disk of radius 3 is a 5×5, 25-pixel octagon). A small disk leaves several seeds per floe (more
snakes, same boundary, only slower); a large one merges maxima belonging to *different* floes and the circle can
then straddle a junction. Requires `ipywidgets`; without it only the static preview runs.
""")
code(r'''
def play(metric="cityblock", se_radius=3, divisor=1.414):
    ini = initialize_contours(BW, metric=metric, se_radius=se_radius, radius_divisor=divisor)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    imshow_matlab(axes[0], ini.img_dist, autoscale=True, title=f"-bwdist(~bw, '{metric}')")
    imshow_matlab(axes[1], BW, title=f"{int(ini.dis.sum())} local maxima -> {ini.num} seeds (disk r = {se_radius})")
    for n in range(ini.num):
        cxx, cyy = ini.centroids[n]
        rr_ = ini.radii[n]
        axes[1].plot(cxx - 1, cyy - 1, "r+", ms=5, mew=0.9)
        axes[1].plot(cxx - 1 + rr_ * np.cos(CIRCLE_T), cyy - 1 + rr_ * np.sin(CIRCLE_T), "b-", lw=0.6)
    plt.show()

play()                                               # static preview with the book's settings
try:
    from ipywidgets import interact, IntSlider, FloatSlider, Dropdown
    interact(play, metric=Dropdown(options=["cityblock", "euclidean", "chessboard", "quasi-euclidean"], value="cityblock"),
             se_radius=IntSlider(3, 1, 9, 1), divisor=FloatSlider(1.414, min=1.0, max=2.0, step=0.1))
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
| `GVF(f, mu, ITER)` | `core.snake.gvf(f, mu, iters)` | exact | normalises `f` to [0, 1] internally; `mu*4*del2(u)` = μ·(5-point Laplacian); refuses μ > 0.25 (CFL, Eq. 6.55) |
| `del2(u)` | `core.matlab_compat.del2` | exact | MATLAB's `del2` is $\nabla^2/4$ with **linearly extrapolated borders** — never `scipy.ndimage.laplace` |
| `BoundMirrorExpand/Ensure/Shrink` | `core.snake.bound_mirror_*` | exact | `expand` == `np.pad(A, 1, 'reflect')`; the natural BC $\nabla u\cdot d\sigma = 0$ (Eq. 6.46) |
| `gradient2(a)` / `abs(gradient2(a))` | `core.snake.gradient2` / `gradient2_magnitude` | exact | MATLAB-4 `gradient` semantics: `np.gradient(a)[::-1]`; the 1-output form is complex |
| `snakedeform(x, y, α, β, γ, κ, fx, fy, ITER)` | `core.snake.snakedeform(..., solver='auto')` | exact (dense) / near (FFT) | `A + γI` is a symmetric circulant → solved by FFT in O(N log N) for N ≥ 32 (306 s → 2 s at chapter scale); `solver='dense'` is MATLAB's literal `inv` |
| `snakeinterp(x, y, Dmax, Dmin)`, `snakeindex` | `core.snake.snakeinterp / snakeindex` | exact | city-block spacing; unbounded `while` loop capped by `max_passes` |
| `interp2(fx, x, y, '*linear', 0)` | `core.interp.interp2(fx, u=y−1, v=x−1, 'bilinear', fill=0.0)` | exact | 1-based `(x = column, y = row)`; the trailing 0 is the *extrapolation value* |
| `gaussianMask(k, s)`, `gaussianBlur`, `xconv2` | `core.snake.gaussian_mask / gaussian_blur / xconv2` | exact | `R = ceil(3σ)` — **not** `fspecial('gaussian')` |
| `imregionalmin(-D)` with `-Inf` background | `core.morphology.imregionalmin` (ch05) | exact | = the regional maxima of `D` inside the mask |
| Eqs. (6.57)/(6.58) | `core.morphology.regional_maxima_by_reconstruction(I, form=…)` | reimplemented | equals `imregionalmax` on every case tried |
| `bwdist(~bw, 'cityblock')` | `core.distance.bwdist` (float32) | exact | keep `single`: the radii are computed from it |
| `strel('disk', 3)` / `('disk', 5)`, `imdilate` | `core.morphology.strel / imdilate` (ch04) | exact | disk 3 = **5×5, 25 px** octagon; disk 5 = 9×9, 69 px |
| `regionprops(BW, 'Area' \| 'Solidity' \| 'MajorAxisLength' \| 'MinorAxisLength' \| 'Centroid' \| 'Perimeter')` | **`core.regionprops`** | exact (≤ 1.07e-14) | scikit-image differs (perimeter +5.5 %, axes, orientation +90°) — never use it for these |
| `polybool('intersection', rect, poly)` (Mapping TB) | `core.polygon.clip_polygon_rect` | reimplemented | Sutherland–Hodgman; same vertex set and mask; MATLAB also **closes** the ring |
| `roipoly(I, x, y)` / `poly2mask`, `polyxpoly`, `convhull` | `core.polygon.roipoly / poly2mask / polyxpoly / convhull` | exact / near | line-by-line port of `eml/poly2mask.m` |
| `polygeom`, `minboundrect` | `core.polygon.polygeom / minboundrect` | near / exact | Sommer and D'Errico File Exchange code; needed by Ch. 8/9 |
| `kmeans(double(I(:)), 3, 'EmptyAction','singleton')` (Statistics TB) | `core.clustering.kmeans_lloyd(init='kmeans++')` | approx (labels) | ch6 ships no `kmeans.m`, so this is **not** ch3's histogram k-means; compare sorted centres, never labels |
| `graythresh`, `im2bw`, `bwlabel`, `bwareaopen`, `rgb2gray` | `core.threshold`, `core.connectivity`, `core.matlab_compat` (ch02–ch05) | exact | in ch6 `graythresh` always sees the **gray** image |
| `snakedisp(x, y, style)`, `quiver`, `label2rgb` | `core.plotting.snake_plot / quiver_field / label2rgb` | display only | `snake_plot` closes the curve and converts 1-based → 0-based axes |
""")
md(r"""
## Summary — what is (and is not) verified, and what the next chapters need

**What chapter 6 established.** A parametric snake minimises Eq. (6.2); its internal energy (6.3) is controlled
by $\alpha$ (spacing/tension) and $\beta$ (rigidity) — the book uses α = 0.05, β = 0 throughout — and its
discretisation (6.34)–(6.40) is a pentadiagonal circulant solve per iteration. The traditional external force
$-\nabla E_{ext}$ has a tiny capture range and cannot enter concavities; **GVF** (6.41)–(6.55), a diffusion of
the edge-map gradient with μ = 0.1, fixes both, and its capture range is tuned by the number of diffusion
iterations. Contours are initialised automatically from the **city-block distance transform** of the ice mask:
regional maxima (6.57/6.58) → dilation-merged **seeds** → circles of radius $D(\text{seed})/\sqrt2$ → one GVF
snake per seed, whose boundary is burnt into the mask (**Algorithm 1**), leaving connected floes separated and
the boundary pixels labelled *residue ice*.

**Verification status (from `reports/ch06_verification.md`, MATLAB R2025a running the original `.m` code).**
`exact` on 200+ controlled cases: `del2`, `GVF` (44 cases, plus `u`, `v`, `px`, `py` on both book images),
`gradient2`, `xconv2`, `gaussianMask/Blur`, `snake_matrix`, `snakedeform` (dense solver), `snakeinterp`,
`snakeindex`, `interp2('*linear', 0)`, `poly2mask`/`roipoly`, `bwperim`, and — the chapter's highest risk —
**`core.regionprops`** (≤ 1.07e-14 on 40 constructed shapes and on all 344 components of the three real masks, so
the `Rc = 0.9` / `Rl = 2` decision sets are identical to MATLAB's). `near` for the two pipelines: every stage
(mask, edge map, GVF field, criteria, `bw2`, distance map, maxima, seeds, centroids, radii) is 0 px / 0.0 against
MATLAB, and the differences are confined to the snake's *discretisation* — MATLAB's `polybool` closes the clipped
ring while Sutherland–Hodgman did not, and on a 500–2800-point contour hundreds of point spacings sit exactly at
`Dmax`, so a 1e-13 perturbation changes the point count. The final curves stay well inside a pixel of MATLAB's
and only a fraction of a percent of the burnt mask pixels differ; the FFT snake solver (needed to make the
chapter run in seconds instead of minutes) is itself part of that bound. `reimplemented`: `clip_polygon_rect`
(vertex set and rasterised mask identical to `polybool`), the Eq. (6.6)–(6.13) energies and the traditional
snake, and Eqs. (6.57)/(6.58). `approx`: the k-means *labels* (any k-means is initialisation dependent — but the
derived mask `bk` matched MATLAB's `rng(0)` run pixel-for-pixel on both book images). **Not verified as
figures**: Figs. 6.2, 6.3, 6.5–6.13 and 6.18–6.21 — the aerial floe-field and model-basin images were never
shipped and no `.m` file produces them, so the notebook shows the *mechanism* on seeded synthetic fixtures and
says so; only Fig. 6.14 (printed truth, bit-exact), Fig. 6.15 (`alg_seg_gray.jpg`) and Figs. 6.16/6.17 (fully
specified in the text) are figure-level reproductions. `sea_ice_demo.m` also cannot be shown end to end here: its
last two stages are `ice_shape_enhancement` (book §7.1) and `sea_ice_model` (§8.2), deliberately deferred.

**Feeds forward:**

| Needed by | Primitive |
|---|---|
| Ch. 7 (ice types, shape enhancement) | the three-level `seaice_kmean_gvf(...).out` (1 = floe, 0.5 = slush, 0 = water) is exactly what `ice_shape_enhancement` consumes; `core.regionprops` for every shape metric; `core.snake.*` (the 23 `.m` files of `ch7/Sea_Ice_Floe_Identification/` are byte-identical to ch6's) |
| Ch. 8 (applications, floe size distribution, Appendix B) | `core.polygon.polygeom`, `convhull`, `roipoly`/`poly2mask`, `polyxpoly` for `sea_ice_model`'s polygon/disk fits; `regionprops` areas and centroids for the FSD |
| Ch. 9 (model ice) | `GVF_distance` itself is called by `model_ice_demo.m`; `core.polygon.minboundrect` for the §9.3 rectangularization; the p. 205 criteria (`Ra`, `Rc`, `Rl`) implemented in `component_criteria` |

**Pitfalls to carry forward.** MATLAB's `del2` is $\nabla^2/4$ with extrapolated borders; `strel('disk', 3)` is a
5×5 / 25-px octagon (not 7×7); `regionprops`'s axes, `Solidity` and `Perimeter` must come from `core.regionprops`,
never from scikit-image; `bwdist` is *single* and the seed radii inherit that; `t = 0:0.05:6.28` gives 126 points
and an **unclosed** circle; the shipped code normalises the GVF force per pixel (`px = u/(|v| + 1e-10)`), which is
Xu & Prince's demo convention, not the book's Eq. (6.56); and a dense `inv(A + γI)` per `snakeinterp` is
unusable at chapter scale — the circulant FFT solve is what makes the notebook run in seconds.

*Optional: to regenerate the MATLAB references yourself, run `reference/ch06/make_refs.py` (uses
`tools/run_matlab_ref.py`, MATLAB `-batch`); the notebook needs neither MATLAB nor Octave.*
""")

out = HERE / NB_NAME
nbf.write(nb, out)
print(f"wrote {out.as_posix()} ({len(C)} cells)")
