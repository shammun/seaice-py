"""Build ``notebooks/ch03_ice_pixel_detection.ipynb`` with nbformat (re-runnable; the notebook is never edited by hand).

Chapter 3 — Ice Pixel Detection (Zhang & Skjetne 2018, pp. 37–58).
Follows the ``colab-notebook`` skill: title → cell 1 (Drive mount / cwd) → cell 2 (clone or pull the public repo)
→ cell 3 (``load_image``: private copy or public-domain substitute) → one section per book section
(3.1, 3.1.1.1, 3.1.2, 3.1.3, 3.2, 3.2.2, 3.3) → optional parameter play → summary / feeds forward.
All algorithms are imported from ``seaice``; the notebook only calls them and draws figures.

Run:  ``.venv/Scripts/python.exe notebooks/build_ch03.py``
Verify: ``.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800
        --output executed_ch03.ipynb notebooks/ch03_ice_pixel_detection.ipynb``  (then delete the executed copy).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

NB_NAME = "ch03_ice_pixel_detection.ipynb"
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
# Chapter 3 — Ice Pixel Detection

*Sea Ice Image Processing with MATLAB* (Q. Zhang & R. Skjetne, CRC Press 2018), Chapter 3, pp. 37–58 —
Python port `seaice-py`, notebook `notebooks/ch03_ice_pixel_detection.ipynb`.

**Sections covered** (one notebook section each, in book order):
3.1 Thresholding (global threshold, Eq. 3.1) · 3.1.1.1 Otsu thresholding (Eqs. 3.2–3.22) · 3.1.2 Local thresholding ·
3.1.3 Multithresholding (Eqs. 3.23–3.28) · 3.2 Clustering (distances, Eqs. 3.29–3.34) · 3.2.2 K-means clustering
(Eqs. 3.35–3.37) · 3.3 Experiment results and discussion (Figs. 3.9–3.12, Tables 3.1–3.3).

The output of this chapter is the **ice mask** (binary or 3-level) and the **ice concentration**
$IC = \#\text{ice pixels} / (MN)$ that Chapters 4–9 build on.

**MATLAB files this notebook replaces** (`MATLAB_ROOT/ch3/`):

| MATLAB file | Python (imported below) | Parity (from `reports/ch03_verification.md`, MATLAB R2025a as reference) |
|---|---|---|
| `Otsu.m` lines 2–14 | `seaice.ch03_ice_pixel_detection.otsu_segmentation` → `seaice.core.threshold.graythresh / im2bw` | exact (thresholds bit-identical, masks 0 px differ on all three images) |
| `Otsu.m` lines 28–43 | `multi_otsu_segmentation` → `seaice.core.threshold.multithresh / imquantize` | exact for N ≤ 2 (N = 3 is **reimplemented**: exhaustive Eq. 3.27 instead of MATLAB's `fminsearch`) |
| `local_Otsu.m` | `local_otsu` → `seaice.core.threshold.block_otsu` | exact vs MATLAB on the substitute input; the book's Fig. 3.4 numbers are **unverified** (image not shipped) |
| `separability.m` | `separability_script` → `seaice.core.threshold.otsu_criterion / separability` | exact on the substitute (the script's off-by-one is reproduced, see §3.1.1.1) |
| `kmeans.m` (the authors' own histogram k-means, not the toolbox function it shadows) | `kmeans_segmentation` → `seaice.core.clustering.kmeans_gray` | exact with `shift_bug=True` (all book k-means numbers); the default applies Eqs. 3.36–3.37 consistently (**reimplemented**, documented deviation) |
| *(text only: Eq. 3.1 / Fig. 3.2, Figs. 3.6 and 3.8, Eqs. 3.29–3.34)* | `fixed_threshold`, `plot_histogram_with_threshold`, `kmeans_demo_2d` → `seaice.core.clustering.kmeans_lloyd / pairwise_distance`, `seaice.core.synth` | reimplemented from the text (property-tested) |

The chapter's learned knowledge (concepts, primitives, pitfalls, what later chapters need) is in
`knowledge/ch03.md` (written by the knowledge phase) and `knowledge/CUMULATIVE.md`.

> **Data.** Chapter 3 ships three photographs from the Ny-Ålesund campaign of May 2011, all 4290×2856 RGB:
> `1.jpg` (Fig. 3.9(a) "sea ice image 1"), `2.jpg` (Fig. 3.10(a) "sea ice image 2") and `test.jpg` (Fig. 3.11(a)
> "sea ice image 3", the input hard-coded in `Otsu.m` / `kmeans.m`). The book's images are copyrighted and are
> **not** in this public repository: `seaice.core.io.load_image()` first looks for your own private copies
> (`data/book/ch03/` next to the repository, or `MyDrive/Sea_Ice_Colab/data/book/ch03/` on Colab) and otherwise
> downloads public-domain NASA MODIS scenes registered in `seaice/core/public_images.py`.
> The image of Figs. 3.2–3.5 and 3.7 (`ch3ice.jpg` / `t.jpg`, from the authors' OMAE-2012 paper) is **not shipped
> with the book's code at all**; this notebook uses `2.jpg` in its place (its "histogram twin": same size, Otsu
> threshold 107 vs 108, identical multi-Otsu thresholds 61/142) and, for the uneven-illumination Fig. 3.4, a
> synthetic illumination ramp from `seaice.core.synth.uneven_illumination`. Numbers printed for those figures are
> therefore *ours for the substitute*, never the book's, and are labelled as such.
""")

# =====================================================================================================================
# 2. Setup cell (Colab / Drive / local) — copied verbatim from build_ch02.py (only the chapter id / image names differ)
# =====================================================================================================================
md(r"""
## Setup (Google Colab or local Jupyter)

Run the two cells below first. On **Colab** the first cell mounts your Google Drive and moves into
`MyDrive/Sea_Ice_Colab` — the folder where your private copy of the book images lives (`data/book/ch03/1.jpg`,
`2.jpg`, `test.jpg`) and where downloads are cached between sessions; readers without Drive get a temporary
`/content/Sea_Ice_Colab`. The second cell clones (or updates) the public repository `seaice-py` there and installs its
requirements. **Locally** both cells are no-ops that move to the repository root. No GPU is needed.

> ⚠️ **If you run this with your own copy of the book images, do not use *File → Save a copy in GitHub*.** That saves
> the cell outputs — figures rendered from the copyrighted book images — into the public repository. Save to Drive
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

`seaice.core.io.load_image("ch03", "test.jpg")` replaces MATLAB's `imread('test.jpg')`. It looks for **your private
copy of the book image** first — `data/book/ch03/test.jpg` in the current folder (on Colab: `MyDrive/Sea_Ice_Colab`),
in the repository, or explicitly in Drive — and otherwise downloads a **public-domain NASA substitute** (MODIS/Terra
true-colour scenes of the Arctic marginal ice zone, registered per image name in `seaice/core/public_images.py`) into
`data/online/ch03/`. It prints which source it used and returns the label, so the cells below only quote the
book's numbers when the book's own images are loaded (`book(...)` helper). The same is done for `1.jpg` and `2.jpg`.
""")
code(r'''
from seaice.core.io import load_image

I, SOURCE = load_image("ch03", "test.jpg")           # = imread('test.jpg'): "sea ice image 3", Fig. 3.11(a); uint8 (M, N, 3)
IMAGES, SOURCES = {"test.jpg": I}, {"test.jpg": SOURCE}
for name in ("1.jpg", "2.jpg"):                      # Fig. 3.9(a) "sea ice image 1", Fig. 3.10(a) "sea ice image 2"
    IMAGES[name], SOURCES[name] = load_image("ch03", name)
FROM_BOOK = all(s.startswith("book") for s in SOURCES.values())
book = (lambda s: f"  (book: {s})") if FROM_BOOK else (lambda s: "")   # book-quoted values only for the book images
if not FROM_BOOK:
    print("Running on a public-domain substitute image; figures show the same operations, "
          "but values quoted in the book only hold for the book's own image.")
for name, img in IMAGES.items():
    print(f"{name}: {img.shape} {img.dtype}   [{SOURCES[name]}]")
''')

md(r"""
## Imports

Everything algorithmic comes from the `seaice` package; the notebook only calls it and draws figures.
(`seaice.core.plotting` selects the non-interactive Agg backend for the command-line scripts, so `%matplotlib inline`
is issued *after* the `seaice` imports to switch back to inline display.) The three grayscale images are computed
once here with MATLAB's `rgb2gray` weights and rounding (exact, Chapter 2) and reused by every section — the
12-megapixel images are the most expensive part of this notebook.
""")
code(r'''
import numpy as np
import seaice.core                                    # imports seaice.core.plotting (sets the Agg backend)
from seaice.core import synth
from seaice.core.plotting import imshow_matlab        # MATLAB imshow scaling (display only)
from seaice.core.matlab_compat import rgb2gray_matlab
from seaice.core.histogram import imhist

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100

G = {name: rgb2gray_matlab(img) for name, img in IMAGES.items()}   # I = rgb2gray(imread(...)) of Otsu.m / kmeans.m
G1, G2, G3 = G["1.jpg"], G["2.jpg"], G["test.jpg"]
for name, g in G.items():
    print(f"{name}: gray {g.shape} {g.dtype}, levels {int(g.min())} .. {int(g.max())}")
''')

# =====================================================================================================================
# 3.1 Thresholding
# =====================================================================================================================
md(r"""
## 3.1 Thresholding (pp. 37–39) — text only (Eq. 3.1, Figs. 3.1–3.2); `separability.m` lines 11–14 are its code form

Ice is brighter than water, so the most direct ice/water segmentation compares every gray value $f(x, y)$ ($x$ = row,
$y$ = column, as in Chapter 2) with a threshold $T$ and produces a binary image, Eq. (3.1):

$$g(x, y) = \begin{cases} 1 & \text{if } f(x, y) > T \\ 0 & \text{if } f(x, y) \le T \end{cases}$$

A single $T$ for the whole image is **global** thresholding; a $T$ that changes across the image is **variable
(local)** thresholding (§3.1.2). When the histogram is bimodal (Fig. 3.1) a good $T$ sits in the valley between the
water and ice modes — the book reads $T = 125$ off the histogram of its Fig. 3.2(a) and obtains an ice concentration
$IC = \#\{g = 1\} / (MN) = 41.47\,\%$. Picking the valley by eye is unreliable when the histogram is noisy or the modes
overlap, and it must be redone for every image, which motivates the automatic Otsu threshold of §3.1.1.1.

> **Substitute.** Fig. 3.2(a) is not shipped, so panel (a) below is the grayscale of `2.jpg`; its IC is *ours*, not
> the book's 41.47 %. Fig. 3.1 is a sketch drawn from a synthetic two-mode image (`seaice.core.synth.bimodal_image`).
""")
code(r'''
from seaice.ch03_ice_pixel_detection import fixed_threshold, plot_histogram_with_threshold

# --- Fig. 3.1: a bimodal histogram (synthetic image) --------------------------------------------------------------
syn = synth.bimodal_image(seed=0)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
imshow_matlab(axes[0], syn, title="synthetic two-mode image (synth.bimodal_image)")
plot_histogram_with_threshold(axes[1], syn, T=125.0)
axes[1].set_title("bimodal histogram with T in the valley")
fig.suptitle("Fig. 3.1 (sketch)  A bimodal histogram — synthetic data")
plt.show()

# --- Fig. 3.2: hand-picked T = 125 on the substitute image, Eq. (3.1) -------------------------------------------
T = 125.0
bw_T, ic_T = fixed_threshold(G2, T)                   # g = f > T ; IC = #ice / (M N)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
imshow_matlab(axes[0], G2, title="(a) grayscale of 2.jpg  [substitute for the unshipped Fig. 3.2(a)]")
plot_histogram_with_threshold(axes[1], G2, T)
axes[1].set_title(f"(b) histogram of (a), threshold T = {T:g}")
imshow_matlab(axes[2], bw_T, title=f"(c) binarised, Eq. (3.1): IC = {100 * ic_T:.2f} %")
fig.suptitle("Fig. 3.2  Selecting a threshold in the valley of the histogram  [substitute image; the book's 41.47 % is for another image]")
plt.show()
counts2, x2 = imhist(G2)
print(f"2.jpg histogram: peak {int(counts2.max())} pixels @ level {int(counts2.argmax())}; "
      f"IC(T = {T:g}) = {100 * ic_T:.2f} %  (this image, not the book's Fig. 3.2(a))")
''')
md(r"""
Water forms the dark mode and ice the bright one; anything above $T$ becomes ice. The result is only as good as the
chosen $T$ — move it by a few levels and the IC changes — which is why the next section lets the histogram decide.
""")

# =====================================================================================================================
# 3.1.1.1 Otsu
# =====================================================================================================================
md(r"""
## 3.1.1.1 Otsu thresholding (pp. 40–43) — replaces `ch3/separability.m` (Fig. 3.3) and the `graythresh` part of `ch3/Otsu.m`

Otsu treats the normalised histogram $p_i = n_i / n$ (Eq. 3.3, with $n = MN = \sum_i n_i$, Eq. 3.2) as a probability
distribution over the levels $i = 0, \dots, L-1$ and asks, for each candidate $t$, how well the two classes
$C_0 = \{0, \dots, t\}$ and $C_1 = \{t+1, \dots, L-1\}$ are separated. Their probabilities and means are
$P_0(t) = \sum_{i \le t} p_i$, $P_1(t) = 1 - P_0(t)$ (Eqs. 3.4–3.5), $m_0(t) = \frac{1}{P_0}\sum_{i \le t} i\,p_i$,
$m_1(t) = \frac{1}{P_1}\sum_{i > t} i\,p_i$ (Eqs. 3.6–3.7), with the cumulative mean $m(t) = \sum_{i \le t} i\,p_i$
(Eq. 3.8) and the global mean $m_G = \sum_i i\,p_i$ (Eq. 3.9), so that $P_0 m_0 + P_1 m_1 = m_G$ (Eq. 3.10).
The between-class variance has three equivalent forms (Eq. 3.16), the last one needing only first-order statistics:

$$\sigma_B^2(t) = P_0 (m_0 - m_G)^2 + P_1 (m_1 - m_G)^2 = P_0 P_1 (m_0 - m_1)^2 = \frac{(m_G P_0(t) - m(t))^2}{P_0(t)\,(1 - P_0(t))},$$

and because $\sigma_W^2(t) + \sigma_B^2(t) = \sigma_G^2$ (Eq. 3.18, with $\sigma_W^2 = P_0\sigma_0^2 + P_1\sigma_1^2$,
Eq. 3.15, and the constant global variance $\sigma_G^2 = \sum_i (i - m_G)^2 p_i$, Eq. 3.17) the three criteria
$\lambda = \sigma_B^2/\sigma_W^2$, $\kappa = \sigma_G^2/\sigma_W^2$, $\eta = \sigma_B^2/\sigma_G^2$ (Eq. 3.14) are
monotone in one another (Eq. 3.19), so maximising the cheapest, $\eta(t)$ (Eq. 3.20), is the same as minimising the
within-class variance. The threshold is found by exhaustive search over the integer $t$,
$\sigma_B^2(t^*) = \max_{0 < t < L-1} \sigma_B^2(t)$ (Eq. 3.21, ties averaged), and $0 \le \eta(t^*) \le 1$ (Eq. 3.22)
measures the separability: 0 for a constant image, 1 for a two-valued one. For the book's Fig. 3.2(a) this gives
$t^* = 108$, $IC = 42.14\,\%$, $\eta(108) = 0.9643$ against $\eta(125) = 0.9620$ for the manual choice.

MATLAB's `[level, em] = graythresh(I)` is exactly this search on the 256-bin histogram (`em` = $\eta(t^*)$), and
`im2bw(I, level)` is Eq. (3.1) with $T = 255 \cdot$ `level`; `seaice.core.threshold.graythresh / im2bw` are ported
line by line from the R2025a sources and are bit-identical to MATLAB on all three 12-Mpx images.
`otsu_criterion(counts)` evaluates every quantity of Eqs. (3.3)–(3.22) for all $t$ at once.

> **`separability.m` quirk (reproduced, not corrected).** The script uses the 1-based bin index as the intensity and
> puts bins `1..k` (levels $0 \dots k-1$) into $C_0$, so its `eta` is $\eta(k - 1)$ while its mask is `I > k`; the
> shift leaves $\sigma_B^2$ and $\sigma_G^2$ unchanged, so the value is right but evaluated one level low.
> `separability_script` follows the script; `separability(gray, t)` uses the book convention.
""")
code(r'''
from seaice.core.threshold import otsu_criterion, separability, graythresh, im2bw

cv = otsu_criterion(counts2)                          # Eqs. (3.3)-(3.22) for every t = 0..255 on the 2.jpg histogram
t = np.arange(256)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
axes[0].bar(t, cv.p, width=1.0, color="k")
axes[0].set(xlim=(0, 255), xlabel="level i", ylabel="p_i", title="normalised histogram p_i  (Eq. 3.3)")
axes[1].plot(t, cv.sigma_B2, "r-", label=r"$\sigma_B^2(t)$  (Eq. 3.16)")
axes[1].plot(t, cv.sigma_W2, "b-", label=r"$\sigma_W^2(t)$  (Eq. 3.15)")
axes[1].axhline(cv.sigma_G2, color="k", ls="--", label=r"$\sigma_G^2$  (Eq. 3.17)")
axes[1].set(xlim=(0, 255), xlabel="threshold t", title=r"$\sigma_W^2 + \sigma_B^2 = \sigma_G^2$  (Eq. 3.18)")
axes[1].legend(frameon=False)
axes[2].plot(t, cv.eta, "k-", lw=1.5, label=r"$\eta(t) = \sigma_B^2(t)/\sigma_G^2$  (Eq. 3.20)")
axes[2].axvline(cv.t_star, color="r", ls="--", label=f"Otsu t* = {cv.t_star:g}, eta* = {cv.eta_star:.4f}  (Eq. 3.21)")
axes[2].axvline(125, color="b", ls=":", label=f"manual T = 125, eta = {separability(G2, 125):.4f}")
axes[2].set(xlim=(0, 255), ylim=(0, 1), xlabel="threshold t", ylabel=r"$\eta$", title="separability measure  (Eqs. 3.20-3.22)")
axes[2].legend(loc="lower center", frameon=False, fontsize=8)
fig.suptitle("Section 3.1.1.1  Otsu's discriminant criterion on the histogram of 2.jpg")
plt.show()

level, em = graythresh(G2)                            # MATLAB [level, em] = graythresh(I)
print(f"graythresh: level = {level:.6f} -> t* = {255 * level:g}, em = {em:.4f}   |   otsu_criterion: t* = {cv.t_star:g}, eta* = {cv.eta_star:.4f}")
print(f"identities over all t with both classes non-empty: max|Eq.3.10| = {np.nanmax(np.abs(cv.P0 * cv.m0 + cv.P1 * cv.m1 - cv.mG)):.1e}, "
      f"max|Eq.3.11| = {np.max(np.abs(cv.P0 + cv.P1 - 1)):.1e}, max|Eq.3.18| = {np.nanmax(np.abs(cv.sigma_W2 + cv.sigma_B2 - cv.sigma_G2)):.1e}")
print(f"eta(108) = {separability(G2, 108):.4f}, eta(125) = {separability(G2, 125):.4f} for 2.jpg"
      "   (book p. 43, for its own unshipped image: 0.9643 / 0.9620)")
two_valued = np.where(G2 > 255 * level, 255, 0).astype(np.uint8)   # 1-line glue: the two-valued image of Eq. (3.22)'s upper bound
print(f"Eq. (3.22) bounds: constant image eta = {graythresh(np.full((8, 8), 77, np.uint8))[1]:.1f}, "
      f"two-valued {{0, 255}} image eta = {graythresh(two_valued)[1]:.1f}")
''')
code(r'''
from seaice.ch03_ice_pixel_detection import otsu_segmentation, separability_script

sp = separability_script(G2, k=108)                   # separability.m verbatim: bw = I > 108, eta from Eqs. 3.3-3.20
ot2 = otsu_segmentation(G2)                           # Otsu.m lines 2-14: t = graythresh(I); bw = im2bw(I, t); ic
fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
imshow_matlab(axes[0], G2, title="(a) 2.jpg  [substitute for the unshipped Fig. 3.2(a)]")
imshow_matlab(axes[1], sp["bw"], autoscale=True, title=f"(b) separability.m: I > 108, IC = {100 * sp['IC']:.2f} %, eta = {sp['eta']:.4f}")
imshow_matlab(axes[2], ot2["bw"], title=f"(c) Otsu.m: im2bw(I, graythresh(I)), t* = {ot2['threshold']:g}, IC = {100 * ot2['ic']:.2f} %")
fig.suptitle("Fig. 3.3  Global Otsu threshold  [substitute image; book: t = 108, IC = 42.14 %, eta(108) = 0.9643 for its own image]")
plt.show()
print(f"separability.m variables: mg = {sp['mg']:.4f} (1-based: mG + 1), sigma2_g = {sp['sigma2_g']:.4f}, sigma2_b = {sp['sigma2_b']:.4f}, eta = {sp['eta']:.6f}")
print(f"  -> the script's eta is eta(t = {sp['t_eval']}) in the book convention = {separability(G2, sp['t_eval']):.6f};  eta(t = 108) = {sp['eta_book']:.6f}")
print(f"Otsu on 2.jpg: t* = {ot2['threshold']:g} (Fig. 3.10(b) value, see Sec. 3.3), IC = {100 * ot2['ic']:.2f} %{book('32.05 %')}")
''')
md(r"""
The $\eta(t)$ curve has a single broad maximum: the automatic $t^*$ lies in the histogram valley and beats the manual
$T = 125$ by a small margin in $\eta$, as the book reports for its own image. The identities of Eqs. (3.10), (3.11)
and (3.18) hold to round-off, and `graythresh` returns exactly the maximiser of the curve.
""")

# =====================================================================================================================
# 3.1.2 Local thresholding
# =====================================================================================================================
md(r"""
## 3.1.2 Local thresholding (pp. 43–45) — replaces `ch3/local_Otsu.m` (Fig. 3.4)

A global threshold assumes the same ice and water intensities everywhere; when one side of the scene is lit and the
other in shadow, Otsu's single $t$ under-detects ice on the dark side and over-detects it on the bright side
(Fig. 3.4(b): $t = 176$ on the book's artificially illuminated Fig. 3.2(a)). The remedy in the book is the simplest
form of variable thresholding: split the image into sub-images and run Otsu in each one — `local_Otsu.m` uses a
$2 \times 3$ grid (`n_r = 2`, `n_c = 3`), computes `graythresh` per block, counts the ice pixels above each block's
threshold and sums them, $IC = \sum_b n_b / (rc)$; Fig. 3.4(c) shows the six block masks with their thresholds
92/132/177/98/126/176 and block ICs, giving 41.32 % overall against 42.14 % for the clean image. The block grid must
tile the image exactly (MATLAB errors on non-integer indices); the price of the method is choosing the subdivision,
and blocks that contain only ice or only water get a meaningless threshold.
`seaice.core.threshold.block_otsu` (the same code as `ch9/block_threshold.m`) is exact against MATLAB.

> **Substitute.** The illuminated image `t.jpg` is not shipped, so the ramp of Fig. 3.4(a) is synthesised here:
> `synth.uneven_illumination(G2, gain=0.5, bias=40)` multiplies by 1.5 and adds 40 levels at the left edge, fading
> linearly to ×0.5 − 40 at the right edge. All numbers below are for this synthetic input.
""")
code(r'''
from seaice.ch03_ice_pixel_detection import local_otsu

Gu = synth.uneven_illumination(G2, gain=0.5, axis=1, bias=40.0)      # Fig. 3.4(a)-like: x1.5 + 40 (left) -> x0.5 - 40 (right)
r, c = Gu.shape
Gu = Gu[:r - r % 2, :c - c % 3]                       # 1-line glue: 2x3 blocks must tile exactly (no-op for 2856x4290)
lo = local_otsu(Gu, n_r=2, n_c=3)                     # local_Otsu.m: graythresh per block, IC = sum(num) / (r c)

fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
imshow_matlab(axes[0], Gu, title="(a) 2.jpg with a synthetic illumination ramp  [substitute for t.jpg]")
imshow_matlab(axes[1], lo["global_bw"], title=f"(b) global Otsu of (a): t = {lo['global_threshold']:g}, IC = {100 * lo['global_ic']:.2f} %")
imshow_matlab(axes[2], lo["bw"], title=f"(c) local Otsu, 2x3 blocks: IC = {100 * lo['ic']:.2f} %")
fig.suptitle(f"Fig. 3.4  Global vs local Otsu under uneven illumination  [synthetic ramp; clean-image Otsu IC = {100 * ot2['ic']:.2f} %]")
plt.show()

fig, axes = plt.subplots(2, 3, figsize=(13, 6.6))
for ax, sl, title in zip(axes.ravel(), lo["slices"], lo["titles"]):
    imshow_matlab(ax, lo["bw"][sl], title=title)      # imshow(im2bw(temp, t)); title({'IC=..%'; 'Threshold=..'})
fig.suptitle(f"Fig. 3.4(c)  Local Otsu thresholding, block by block (titles as local_Otsu.m prints them), IC = {100 * lo['ic']:.2f} %")
plt.show()
print("block thresholds:", np.round(lo["thresholds"], 1).tolist(), "  block ICs (%):", np.round(100 * lo["ic_local"], 2).tolist())
print(f"global Otsu on the ramped image: t = {lo['global_threshold']:g}, IC = {100 * lo['global_ic']:.2f} %;  local 2x3: IC = {100 * lo['ic']:.2f} %;  "
      f"clean 2.jpg Otsu: t* = {ot2['threshold']:g}, IC = {100 * ot2['ic']:.2f} %")
''')
md(r"""
The global threshold of the ramped image is pulled up by the bright left side, so the dark right side loses its
floes; the six local thresholds fall from left to right with the illumination and the block-wise IC returns close to
the clean-image value — the behaviour the book describes for Fig. 3.4.
""")

# =====================================================================================================================
# 3.1.3 Multithresholding
# =====================================================================================================================
md(r"""
## 3.1.3 Multithresholding (pp. 44–46) — replaces `ch3/Otsu.m` lines 28–43 (Fig. 3.5; Fig. 3.12(a) in §3.3)

With $k - 1$ thresholds $T_1 < \dots < T_{k-1}$ the image is quantised into $k$ classes, Eq. (3.23):
$g = g_k$ if $f > T_{k-1}$, $g = g_j$ if $T_{j-1} < f \le T_j$, $g = g_1$ if $f \le T_1$ — which is MATLAB's
`imquantize`, `index = 1 + \sum_i (A > T_i)`. Otsu's criterion generalises directly: for classes
$C_1 = [0, t_1], C_2 = [t_1 + 1, t_2], \dots, C_k = [t_{k-1} + 1, L-1]$ the between-class variance is

$$\sigma_B^2 = \sum_{j=1}^{k} P_j\,(m_j - m_G)^2, \qquad P_j = \sum_{i \in C_j} p_i, \qquad m_j = \frac{1}{P_j}\sum_{i \in C_j} i\,p_i$$

(Eqs. 3.24–3.26), maximised over all $0 < t_1 < \dots < t_{k-1} < L-1$ (Eq. 3.27), with the separability
$\eta = \sigma_B^2(t^*_1, \dots) / \sigma_G^2$ (Eq. 3.28). For the book's Fig. 3.2(a) the two thresholds are 61 and 142
and the classes cover 54.86 % (water, black), 4.31 % (darker ice, gray) and 40.83 % (bright ice, white) — Fig. 3.5.
The exhaustive search is cheap for $k = 2, 3$ but grows combinatorially, which is why MATLAB's `multithresh` only
searches exhaustively for $N \le 2$ thresholds and falls back to `fminsearch` beyond; `seaice.core.threshold.multithresh`
is exact for $N \le 2$ and maximises Eq. (3.27) exhaustively for $N = 3$ (`reimplemented`, can beat MATLAB's local
search). By coincidence multi-Otsu on `2.jpg` gives the same 61/142 as the book's image — the coverages differ.
""")
code(r'''
from seaice.ch03_ice_pixel_detection import multi_otsu_segmentation
from seaice.core.threshold import multithresh, imquantize

mo2 = multi_otsu_segmentation(G2, N=2)                # Otsu.m: thresh = multithresh(I, 2); seg = imquantize(I, thresh); coverage; means
th = ", ".join(str(int(v)) for v in mo2["thresh"])
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
imshow_matlab(axes[0], G2, title="(a) 2.jpg  [substitute for the unshipped Fig. 3.2(a)]")
imshow_matlab(axes[1], mo2["seg"], autoscale=True,
              title=f"(b) 3-level Otsu: thresholds {th}; black {100 * mo2['coverage'][0]:.2f} %, gray {100 * mo2['coverage'][1]:.2f} %, white {100 * mo2['coverage'][2]:.2f} %")
fig.suptitle("Fig. 3.5  Multithresholding by Otsu's method (N = 2 thresholds, 3 classes)  [substitute image; book: 61 / 142, 54.86 / 4.31 / 40.83 %]")
plt.show()
print(f"multithresh(I, 2) on 2.jpg: thresholds [{th}], metric eta = {mo2['metric']:.4f} (Eq. 3.28); coverages "
      f"{' / '.join(f'{100 * v:.2f}' for v in mo2['coverage'])} %; class means {' / '.join(f'{v:.2f}' for v in mo2['average_intensity'])}")
th1, metric1 = multithresh(G2, 1)
print(f"multithresh(I, 1) = {int(th1[0])} with metric {metric1:.4f}  ==  graythresh t* = {ot2['threshold']:g}, em = {ot2['em']:.4f}  (same criterion)")
A = np.array([[0, 60, 61], [62, 142, 143], [200, 255, 61]], dtype=np.uint8)
print("imquantize(A, [61, 142]) on a 3x3 example (Eq. 3.23, classes 1..3):\n", imquantize(A, mo2["thresh"]))
''')
md(r"""
The middle (gray) class collects the darker ice — brash, slush, shadowed floe edges — that a single threshold would
assign to water or to ice arbitrarily; §3.3 uses exactly this three-way split on image 3.
""")

# =====================================================================================================================
# 3.2 Clustering
# =====================================================================================================================
md(r"""
## 3.2 Clustering (pp. 46–48) — text only; primitives in `seaice.core.clustering`

Clustering groups unlabeled observations so that members of a group are closer to each other than to other groups,
which requires a **distance**; the book lists six for vectors $a, b$: Euclidean $\|a - b\|_2 = \sqrt{\sum_i (a_i - b_i)^2}$
(Eq. 3.29), squared Euclidean $\|a - b\|_2^2$ (3.30), maximum norm $\|a - b\|_\infty = \max_i |a_i - b_i|$ (3.31),
city-block $\|a - b\|_1 = \sum_i |a_i - b_i|$ (3.32), cosine $\cos(a, b) = a^T b / (|a||b|)$ (3.33) and Mahalanobis
$d_M = \sqrt{(a-b)^T S^{-1} (a-b)}$ with the covariance $S$ (3.34). Methods are **hierarchical** (agglomerative
bottom-up or divisive top-down, producing a tree of nested clusters) or **partition** clustering, which fixes the number
of clusters, starts from an initial partition and reassigns objects until convergence; image segmentation needs one
partition of millions of pixels, so partition methods — k-means in particular — are the practical choice.
`pairwise_distance` wraps the closed forms (exact); note that Eq. (3.33) prints the cosine *similarity*, and the
function returns the usual cosine *distance* $1 - \cos(a, b)$ so that, like the other five, smaller means closer.
""")
code(r'''
from seaice.core.clustering import pairwise_distance

a, b = np.array([[1.0, 1.0]]), np.array([[4.0, 5.0]])           # a - b = (3, 4): the classic 3-4-5 triangle
S = np.array([[4.0, 0.0], [0.0, 1.0]])                # a covariance with unequal variances for Eq. (3.34)
print("distances between a = (1, 1) and b = (4, 5):")
for metric, eq in (("euclidean", "3.29"), ("sqeuclidean", "3.30"), ("chebyshev", "3.31 (maximum norm)"),
                   ("cityblock", "3.32"), ("cosine", "3.33 (as 1 - cos)"), ("mahalanobis", "3.34, S = diag(4, 1)")):
    d = pairwise_distance(a, b, metric, cov=S)[0, 0]
    print(f"  {metric:12s} Eq. {eq:24s} = {d:.4f}")
print("cosine distance of parallel vectors (1,1)-(2,2):", pairwise_distance([[1.0, 1.0]], [[2.0, 2.0]], "cosine")[0, 0])
''')

# =====================================================================================================================
# 3.2.2 K-means
# =====================================================================================================================
md(r"""
## 3.2.2 K-means clustering (pp. 48–52) — text only (Figs. 3.6, 3.8); `ch3/kmeans.m` for the image version (Fig. 3.7)

K-means partitions $x_1, \dots, x_n$ into $k$ clusters $S_1, \dots, S_k$ by minimising the within-cluster sum of
squared distances to the centroids $c_i$, Eq. (3.35):

$$J = \sum_{i=1}^{k} \sum_{j=1}^{n_i} \big\| x_j^{(i)} - c_i \big\|^2 .$$

Starting from $k$ initial centroids (Step 1: random data points, or user-specified), it alternates the **assignment**
step, $S_i = \{x_p : \|x_p - c_i\|^2 \le \|x_p - c_j\|^2\ \forall j\}$ (Eq. 3.36), and the **update** step,
$c_i = \frac{1}{n_i}\sum_j x_j^{(i)}$ (Eq. 3.37), until no centroid moves (Step 4). Each step can only lower $J$, which
is bounded below by 0, so the iteration converges in finitely many steps — but to a *local* minimum that depends on the
initial centroids and on $k$ (Fig. 3.6), and a single far-away outlier can drag a centroid so that two compact
clusters end up merged (Fig. 3.8). Unlike Otsu, no variance is ever computed, which makes k-means fast on large data.
`seaice.core.clustering.kmeans_lloyd` is this algorithm written from the text (reimplemented; $J$ is checked to be
non-increasing); the points of Figs. 3.6/3.8 are synthetic and seeded.

**The authors' `kmeans.m`** applies the same two steps to the *gray-level histogram* rather than to the pixels: it shifts
the levels so the minimum becomes 1, starts from the "equal division points" $\mu_i = i\,m/(k+1)$ (§3.3), assigns each
present level to the nearest $\mu_i$ by $|v - \mu_i|$ and updates $\mu_i$ as the histogram-weighted mean, stopping when
the $\mu_i$ no longer change — deterministic, no randomness. On the book's Fig. 3.2(a) it gives $IC = 42.79\,\%$ with
$k = 2$ and 53.52 / 5.16 / 41.32 % with $k = 3$ (Fig. 3.7).

> **`shift_bug`.** Lines 64–70 of `kmeans.m` build the final mask by comparing the *unshifted* image with the
> centroids in *shifted* units, so every class boundary sits $\min(I) - 1$ levels too low (12 levels on `2.jpg`, 20 on
> `test.jpg`) and a little more of the image is called ice. `kmeans_gray(..., shift_bug=True)` reproduces the script
> and therefore **every k-means number printed in the book**; the default `shift_bug=False` applies Eqs. (3.36)–(3.37)
> in consistent units (the text is the authority) — the centroids are identical, only the mask differs. Both are shown.
""")
code(r'''
from seaice.ch03_ice_pixel_detection import kmeans_demo_2d

demo = kmeans_demo_2d(seed=0)                         # 30 synthetic points, k = 2, random initial centroids (Fig. 3.6) + outlier (Fig. 3.8)
X, res = demo["X"], demo["result"]
hist = res.history                                    # (centres, labels, J) after every iteration
COLORS = ("tab:blue", "tab:red", "tab:green")

def panel(ax, P, centers=None, labels=None, title=""):      # display glue for the book's panel layout
    if labels is None:
        ax.plot(P[:, 0], P[:, 1], "o", color="0.4", ms=5, label="Data")
    else:
        for i in range(int(labels.max()) + 1):
            sel = labels == i
            ax.plot(P[sel, 0], P[sel, 1], "o", color=COLORS[i % 3], ms=5, label=f"Cluster {i + 1}")
    if centers is not None:
        ax.plot(centers[:, 0], centers[:, 1], "k*", ms=14, label="Centroids")
    ax.set(xlim=(0, 10), ylim=(0, 10), aspect="equal", title=title)
    ax.legend(loc="upper left", fontsize=7, frameon=False)

c0, l0, _ = hist[0]
c1, l1, _ = hist[1] if len(hist) > 1 else hist[0]
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
panel(axes[0, 0], X, title="(a) Data set to be partitioned into two clusters")
panel(axes[0, 1], X, centers=c0, title="(b) Select initial centroids at random (Step 1)")
panel(axes[0, 2], X, centers=c0, labels=l0, title="(c) Assign each point to the nearest centroid (Eq. 3.36)")
panel(axes[1, 0], X, centers=c1, labels=l0, title="(d) Recalculate the centroids (Eq. 3.37)")
panel(axes[1, 1], X, centers=res.centers, labels=res.labels, title=f"(e) Repeat until nothing changes ({res.n_iter} iterations)")
axes[1, 2].plot([h[2] for h in hist], "ko-")
axes[1, 2].set(xlabel="iteration", ylabel="J  (Eq. 3.35)", title="objective J per iteration (non-increasing)")
fig.suptitle("Fig. 3.6  The k-means clustering process (synthetic data, seed 0)")
plt.show()

for it, (cc, lab, J) in enumerate(hist):
    print(f"iteration {it}: centroids {np.round(cc, 3).tolist()}, J = {J:.4f}, cluster sizes {[int((lab == i).sum()) for i in range(2)]}")
print("converged:", res.converged, "| J non-increasing:", all(hist[i + 1][2] <= hist[i][2] + 1e-12 for i in range(len(hist) - 1)))

rc, ro = demo["result_clean"], demo["result_outlier"]
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
panel(axes[0], X, centers=rc.centers, labels=rc.labels, title="(a) Two compact and well-separated clusters")
panel(axes[1], demo["X_out"], centers=ro.centers, labels=ro.labels, title=f"(b) Effect of an outlier at {demo['outlier']} on the clusters")
fig.suptitle("Fig. 3.8  An outlier (used as one initial centroid) forces the two compact clusters into one cluster")
plt.show()
print(f"cluster sizes without / with the outlier: {[int((rc.labels == i).sum()) for i in range(2)]} / {[int((ro.labels == i).sum()) for i in range(2)]}")
''')
code(r'''
from seaice.ch03_ice_pixel_detection import kmeans_segmentation

km2 = kmeans_segmentation(G2, k=2, shift_bug=True)    # kmeans.m with k = 2 (the book's numbers come from this variant)
km2c = kmeans_segmentation(G2, k=2, shift_bug=False)  # Eqs. (3.36)-(3.37) in consistent units
km3 = kmeans_segmentation(G2, k=3, shift_bug=True)    # kmeans.m as shipped (k = 3)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
imshow_matlab(axes[0], km2["mask1"], title=f"(a) kmeans.m, k = 2: IC = {100 * km2['ic']:.2f} %  (script mask, shift bug)")
imshow_matlab(axes[1], km2c["mask1"], title=f"    k = 2 in consistent units: IC = {100 * km2c['ic']:.2f} %")
imshow_matlab(axes[2], km3["mask1"], title=f"(b) kmeans.m, k = 3: {' / '.join(f'{100 * v:.2f}' for v in km3['coverage'])} %  (display i/k)")
fig.suptitle("Fig. 3.7  K-means on the gray histogram of 2.jpg  [substitute image; book: k = 2 IC 42.79 %, k = 3 53.52 / 5.16 / 41.32 %]")
plt.show()
for name, kres in (("k = 2 (script)", km2), ("k = 2 (consistent)", km2c), ("k = 3 (script)", km3)):
    print(f"{name:20s}: {kres['n_iter']} iterations; centroids (gray) {np.round(kres['centroids'], 2).tolist()}; "
          f"class boundaries {np.round(kres['boundaries'], 1).tolist()}; coverage {np.round(100 * kres['coverage'], 2).tolist()} %")
print(f"Otsu on the same image: t* = {ot2['threshold']:g} — the k = 2 boundary of the consistent run ({km2c['boundaries'][0]:.1f}) "
      "is Otsu's threshold up to rounding: both minimise the within-class variance (book p. 56)")
''')
md(r"""
With two clusters the histogram k-means lands on (almost) the Otsu threshold, as the book's remark on p. 56 predicts;
the script's units bug moves the boundary a dozen levels down, which is what makes its IC slightly larger. The third
cluster of `kmeans.m` again isolates the darker ice.
""")

# =====================================================================================================================
# 3.3 Results
# =====================================================================================================================
md(r"""
## 3.3 Experiment results and discussion (pp. 52–58) — `ch3/Otsu.m` and `ch3/kmeans.m` on the three shipped images (Figs. 3.9–3.12, Tables 3.1–3.3)

Both methods are applied to three Ny-Ålesund images (May 2011), first with two classes: Otsu (`Otsu.m`) and k-means
with $k = 2$ (`kmeans.m`, equal-division initialisation) give Figs. 3.9–3.11 and Table 3.1 — 15.36 vs 15.65 % on
image 1, 32.05 vs 32.49 % on image 2, but 72.63 vs 96.50 % on image 3. Where all ice is uniformly bright the two agree,
because both minimise the within-class variance; on image 3 Otsu keeps only the "light ice" and drops the brash ice,
slush and submerged ice whose gray levels are close to water, although the IC definition (Chapter 1) includes them.
Dividing image 3 into **three** groups — "ice group 1" (floes, brightest), "ice group 2" (brash and slush, gray) and
water — by multi-Otsu with two thresholds and by k-means with $k = 3$ recovers them (Fig. 3.12, Table 3.2:
54.39 / 42.11 / 3.50 % → $IC = 96.50\,\%$, and 77.91 / 19.20 / 2.89 % → 97.11 %), and the lower group means of k-means
(Table 3.3) show that it claims more pixels as ice; the authors take 96–97 % as the correct range and prefer k-means
for high-concentration scenes with much brash ice, also because it is faster. Both methods *force* at least two
classes, so an image with 0 % or 100 % ice must be caught as a special case.

Every value below is computed from the loaded images; the book's numbers are appended by `book(...)` only when the
book's own images are loaded. Multi-Otsu N = 2 and k-means with `shift_bug=True` are exact against MATLAB here.
""")
code(r'''
from seaice.ch03_ice_pixel_detection import BOOK_IMAGES, BOOK_VALUES

OT = {img_id: otsu_segmentation(G[fname]) for img_id, (fname, _, _) in BOOK_IMAGES.items()}
KM2 = {img_id: kmeans_segmentation(G[fname], k=2, shift_bug=True) for img_id, (fname, _, _) in BOOK_IMAGES.items()}
KM2c = {img_id: kmeans_segmentation(G[fname], k=2, shift_bug=False) for img_id, (fname, _, _) in BOOK_IMAGES.items()}

for img_id, (fname, label, fig_no) in BOOK_IMAGES.items():
    ot, km = OT[img_id], KM2[img_id]
    b_ot, b_km = BOOK_VALUES["otsu_ic"][img_id], BOOK_VALUES["kmeans2_ic"][img_id]   # Table 3.1 values (shown only if FROM_BOOK)
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.4))
    imshow_matlab(axes[0], IMAGES[fname], title=f"(a) {label}  ({fname})")
    imshow_matlab(axes[1], ot["bw"], title=f"(b) Otsu, t* = {ot['threshold']:g}: IC = {100 * ot['ic']:.2f} %{book(f'{b_ot:.2f} %')}")
    imshow_matlab(axes[2], km["mask1"], title=f"(c) k-means, 2 clusters: IC = {100 * km['ic']:.2f} %{book(f'{b_km:.2f} %')}")
    fig.suptitle(f"{fig_no}  {label} and its ice pixel detection")
    plt.show()

print("Table 3.1  Ice concentrations (%)")
print(f"{'Image no.':22s} {'1':>10s} {'2':>10s} {'3':>10s}")
print(f"{'Otsu':22s} " + " ".join(f"{100 * OT[i]['ic']:10.2f}" for i in ("1", "2", "test")) + book(" / ".join(f"{BOOK_VALUES['otsu_ic'][i]:.2f}" for i in ("1", "2", "test"))))
print(f"{'K-means (kmeans.m)':22s} " + " ".join(f"{100 * KM2[i]['ic']:10.2f}" for i in ("1", "2", "test")) + book(" / ".join(f"{BOOK_VALUES['kmeans2_ic'][i]:.2f}" for i in ("1", "2", "test"))))
print(f"{'K-means (consistent)':22s} " + " ".join(f"{100 * KM2c[i]['ic']:10.2f}" for i in ("1", "2", "test")) + "   <- Eqs. 3.36-3.37 without the units bug")
print("Otsu thresholds t*:", {i: OT[i]["threshold"] for i in ("1", "2", "test")}, " k-means boundaries (script):",
      {i: round(float(KM2[i]["boundaries"][0]), 1) for i in ("1", "2", "test")})
''')
md(r"""
Images 1 and 2 (bright floes on dark water) give nearly the same IC by both methods; on image 3 Otsu's mask drops the
gray brash-ice/slush field that k-means keeps — the 24-point gap of Table 3.1 that motivates the three-group split.
""")
code(r'''
MO3 = multi_otsu_segmentation(G3, N=2)                # Otsu.m lines 28-43 on test.jpg
KM3 = kmeans_segmentation(G3, k=3, shift_bug=True)    # kmeans.m as shipped (k = 3) on test.jpg
KM3c = kmeans_segmentation(G3, k=3, shift_bug=False)
th3 = ", ".join(str(int(v)) for v in MO3["thresh"])

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
imshow_matlab(axes[0], MO3["seg"], autoscale=True,
              title=f"(a) multi Otsu, thresholds {th3}: IC = {100 * MO3['ic']:.2f} %{book('96.50 %')}")
imshow_matlab(axes[1], KM3["mask1"], title=f"(b) k-means, 3 clusters: IC = {100 * KM3['ic']:.2f} %{book('97.11 %')}")
fig.suptitle("Fig. 3.12  Ice pixel detection of sea ice image 3 by dividing it into 3 groups (white = ice group 1, gray = ice group 2, black = water)")
plt.show()

cov_mo, cov_km = 100 * MO3["coverage"][::-1], 100 * KM3["coverage"][::-1]      # order: ice group 1, ice group 2, water
mean_mo, mean_km = MO3["average_intensity"][::-1], KM3["average_intensity"][::-1]
bc_mo, bc_km = BOOK_VALUES["multi_otsu_coverage_test"][::-1], BOOK_VALUES["kmeans3_coverage_test"][::-1]
bm_mo, bm_km = BOOK_VALUES["multi_otsu_means_test"][::-1], BOOK_VALUES["kmeans3_means_test"][::-1]
print("Table 3.2  Group coverages and ice concentration for sea ice image 3 (%)")
print(f"{'':12s} {'Ice group 1':>12s} {'Ice group 2':>12s} {'Water':>8s} {'IC':>8s}")
print(f"{'Multi Otsu':12s} {cov_mo[0]:12.2f} {cov_mo[1]:12.2f} {cov_mo[2]:8.2f} {100 * MO3['ic']:8.2f}{book(' / '.join(f'{v:.2f}' for v in bc_mo) + ', IC 96.50')}")
print(f"{'K-means':12s} {cov_km[0]:12.2f} {cov_km[1]:12.2f} {cov_km[2]:8.2f} {100 * KM3['ic']:8.2f}{book(' / '.join(f'{v:.2f}' for v in bc_km) + ', IC 97.11')}")
print("\nTable 3.3  Average intensity of each group")
print(f"{'Multi Otsu':12s} {mean_mo[0]:12.4f} {mean_mo[1]:12.4f} {mean_mo[2]:8.4f}{book(' / '.join(f'{v:.4f}' for v in bm_mo))}")
print(f"{'K-means':12s} {mean_km[0]:12.4f} {mean_km[1]:12.4f} {mean_km[2]:8.4f}{book(' / '.join(f'{v:.4f}' for v in bm_km) + '  (printed 209,0405)')}")
print(f"\nk-means converged in {KM3['n_iter']} iterations; centroids (gray units) {np.round(KM3['centroids'], 4).tolist()}")
print(f"p. 56 remark, both minimise the within-class variance: consistent-units k = 3 boundaries {np.round(KM3c['boundaries'], 1).tolist()} "
      f"vs multi-Otsu thresholds [{th3}]; the k-means centroids are the multi-Otsu group means {np.round(MO3['average_intensity'], 4).tolist()}")
''')
md(r"""
The three-level images agree on the floes and on the open water; they differ in how much of the gray brash/slush field
each assigns to "ice group 1", which is why k-means' group means are lower and its IC slightly higher. In consistent
units the $k = 3$ centroids *are* the multi-Otsu class means and the boundaries coincide with 120/197 — the equivalence
of the two criteria made visible.
""")

# =====================================================================================================================
# Parameter play (optional)
# =====================================================================================================================
md(r"""
## Parameter play (optional)

Two knobs from this chapter on a downscaled copy of image 3 (`SCALE` = subsampling step, keeps it fast): the global
threshold $T$ of Eq. (3.1) and the number of clusters $k$ of the histogram k-means. Requires `ipywidgets`; the cell
prints a message and falls back to the static preview if it is not installed.
""")
code(r'''
from seaice.core.clustering import kmeans_gray
from seaice.core.threshold import ice_concentration

SCALE = 8                                             # subsample step: 4290x2856 -> ~536x357 (fast enough for a slider)
G_small = G3[::SCALE, ::SCALE]
T0 = int(round(OT["test"]["threshold"]))

def play(T=T0, k=2):
    bw = G_small > T                                  # 1-line glue: Eq. (3.1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    imshow_matlab(axes[0], G_small, title=f"image 3, every {SCALE}th pixel")
    imshow_matlab(axes[1], bw, title=f"gray > {T}: IC = {100 * ice_concentration(bw):.2f} %")
    try:
        km = kmeans_gray(G_small, k)                  # consistent units (shift_bug=False)
        imshow_matlab(axes[2], km.mask1, title=f"k-means k = {k}: IC = {100 * km.ic:.2f} % (darkest cluster = water)")
    except ValueError as exc:                         # an empty cluster (kmeans.m would loop forever)
        axes[2].set_axis_off(); axes[2].set_title(f"k = {k}: {exc}")
    plt.show()

play()                                                # static preview (Otsu threshold, k = 2) — visible without widgets
try:
    from ipywidgets import interact, IntSlider, Dropdown
    interact(play, T=IntSlider(T0, 0, 255, 1), k=Dropdown(options=[2, 3, 4], value=2))
except ImportError:
    print("ipywidgets not installed - only the static preview above is shown")
''')

# =====================================================================================================================
# Summary + feeds forward
# =====================================================================================================================
md(r"""
## Summary — and what the next chapters need from this one

**What chapter 3 established.** An ice mask is a thresholded gray image (Eq. 3.1) and the ice concentration is the
fraction of mask pixels. Otsu picks the threshold that maximises the between-class variance of the histogram
(Eqs. 3.3–3.22; `graythresh` / `im2bw`), block-wise Otsu copes with uneven illumination (`block_otsu`), and multi-Otsu
(Eqs. 3.23–3.28; `multithresh` / `imquantize`) or the authors' histogram k-means (Eqs. 3.35–3.37; `kmeans_gray`)
split the image into water / dark ice / bright ice so that brash ice and slush count as ice. Otsu and 2-means agree
when the ice is uniformly bright; with much dark ice, three groups are needed and k-means claims slightly more ice.

**Parity (from `reports/ch03_verification.md`, MATLAB R2025a as reference).** All four `.m` files are `exact`:
`graythresh`, `im2bw`, `multithresh` (N ≤ 2), `imquantize`, `block_otsu`, `separability` and `kmeans_gray(shift_bug=True)`
are bit-identical or ≤ 1e-12 vs MATLAB, and all 14 mask images differ in 0 pixels; every book number for the three
shipped images (Tables 3.1–3.3, Figs. 3.9–3.12) is reproduced. `reimplemented`: `multithresh` N = 3 (exhaustive vs
`fminsearch`), `kmeans_gray` default (`shift_bug=False`, consistent units), `kmeans_lloyd` / distances / synthetic
fixtures (text only). `unverified`: the numbers of Figs. 3.2–3.5 and 3.7, whose source image is not shipped.

**Feeds forward (`seaice/core/` primitives the later chapters import):**

| Needed by | Primitive |
|---|---|
| Ch4 (edges, morphology) | `threshold.graythresh` + `threshold.im2bw` — `im2bw(I, graythresh(I))` is the starting mask of `ch4/morphology.m` and of every ch5–ch9 script (bit-exact, so masks can be compared with 0-pixel tolerance) |
| Ch5 (watershed, floe splitting) | the same Otsu mask feeds `distance.bwdist` markers; `threshold.ice_concentration` for the floe statistics |
| Ch6 (GVF snake) | `graythresh` / `im2bw` in `GVF_distance.m`; `clustering.kmeans_lloyd(init="kmeans++")` leaves room for the Statistics-Toolbox `kmeans` used there (different algorithm — to be mapped in ch6 as `approx`) |
| Ch7 (ice types) | the three-group split of §3.3 (`multithresh(I, 2)` / `kmeans_gray(I, 3)`) is the floe / brash-slush / water grouping; `threshold.class_coverage`, `class_mean_intensity` |
| Ch8–Ch9 (concentration, model ice) | `threshold.ice_concentration` (IC time series), `threshold.block_otsu` (= `ch9/block_threshold.m`), `kmeans_gray` (`ch9/movie_kmeans.m`), `synth.uneven_illumination` for tank-lighting tests |

**Pitfalls to carry forward.** `im2bw` is a strict `>` at `255 * level` (pixels equal to $t^*$ are water); tie-averaged
Otsu thresholds are half-integers and must stay float; `kmeans.m`'s mask is built in the wrong units (`shift_bug=True`
only when reproducing the book) and its `s`/`ss` mean-intensity buffers are never cleared (`average_intensity_script`
reproduces the stale values); `multithresh` normalises in single precision before binning; an empty cluster makes
`kmeans.m` loop forever (the port raises); both methods force ≥ 2 classes, so 0 % / 100 % ice must be special-cased.

*Optional: to regenerate the MATLAB references yourself, run `reference/ch03/make_refs.py` (uses
`tools/run_matlab_ref.py`, MATLAB `-batch`); the notebook does not need MATLAB or Octave.*
""")

out = HERE / NB_NAME
nbf.write(nb, out)
print(f"wrote {out.as_posix()} ({len(C)} cells)")
