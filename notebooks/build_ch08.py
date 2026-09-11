"""Build ``notebooks/ch08_image_processing_applications.ipynb`` with nbformat (re-runnable; never edited by hand).

Chapter 8 — Sea Ice Image Processing Applications (Zhang & Skjetne 2018, pp. 175–194).  Follows the
``colab-notebook`` skill: title → cell 1 (Drive mount / cwd) → cell 2 (clone or pull the public repo) → cell 3
(``load_image``: private copy or public-domain substitute, plus the two shipped ``.mat`` files of §8.3) → one
section per book section (8.1, 8.2, 8.2.1, 8.3, Appendix B) → the MATLAB traps this chapter turns up →
parameter play → MATLAB ↔ Python map → summary / what is and is not verified / feeds forward.  All algorithms
are imported from ``seaice``; the notebook only calls them and draws.

Run:  ``.venv/Scripts/python.exe notebooks/build_ch08.py``
Verify: ``.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900
        --output executed_ch08.ipynb notebooks/ch08_image_processing_applications.ipynb``  (then delete the copy).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

NB_NAME = "ch08_image_processing_applications.ipynb"
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
# Chapter 8 — Sea Ice Image Processing Applications

*Sea Ice Image Processing with MATLAB* (Q. Zhang & R. Skjetne, CRC Press 2018), Chapter 8, pp. 175–194 —
Python port `seaice-py`, notebook `notebooks/ch08_image_processing_applications.ipynb`.

Chapters 2–7 built a pipeline; Chapter 8 is what the pipeline is **for**. Three field campaigns, three
platforms, three questions:

| section | platform | question | can we reproduce it? |
|---|---|---|---|
| **8.1** (pp. 176–181) | a 4-lens camera on the icebreaker **IB *Frej*** (OATRC'15), 1 frame / 10 s | *ice concentration* in real time: Otsu vs k-means | **method only** — no code and no images ship (open item **O1**) |
| **8.2** (pp. 181–189) | a **CryoWing UAV** over Ny-Ålesund | *characterise* a real ice field so that a DEM simulator can be initialised from it | **pipeline yes, printed counts no** (open item **O2**) |
| **8.3** (pp. 189–194) | a **helicopter** off IB *Oden* (OATRC'15) | the *floe size distribution* and its power-law exponent | **yes, bit-exactly** — the authors ship their own data |
| **App. B** (pp. 221–225) | — | the MATLAB data structure all of it is stored in | **yes** — reproduced field by field |

**Sections covered** (one notebook section each, in book order): 8.1.2 Methods (Otsu / k-means, Figs. 8.4–8.6) ·
8.2 Ice-field characterization (Figs. 8.8–8.11) · 8.2.1 Sea ice numerical modeling (Figs. 8.12–8.15) ·
8.3 Floe size statistic (**Eq. 8.1** MCD, **Eqs. 8.2/8.3** cumulative FSD and power law, Figs. 8.19–8.21) ·
Appendix B — the `IceImage` data structure · the **MATLAB traps** this chapter turns up · parameter play ·
summary.

**MATLAB files this notebook replaces** (9 `.m` files: 5 in `MATLAB_ROOT/ch8/MCD/`, 4 deferred here from
`ch{6,7}/Sea_Ice_Floe_Identification/` — those four are **byte-identical** in ch6 and ch7). Parity from
`reports/ch08_verification.md`; the reference engine is MATLAB R2025a running the original code:

| MATLAB file | Python (imported below) | Parity |
|---|---|---|
| `MCD/main_WL_new.m` (the §8.3 driver) | `seaice.ch08_applications.mcd_analysis` | **exact** — every array, max \|Δ\| = **0.0** |
| `MCD/plot_color_bar_and_floe.m` (Figs. 8.19/8.20) | `ch08_applications.plot_color_bar_and_floe` | **exact** (arrays): the 627×1114×3 painted map is **byte-identical**; the figures are display |
| `MCD/fitting_iceFloes_distribution.m` (Fig. 8.21) | `ch08_applications.cumulative_fsd_powerlaw` | **near** — α agrees with MATLAB to 10 s.f.; the file **as shipped is a syntax error** (erratum E1) |
| `MCD/PowerLaw_fitting_method_and_plotting.m` (orphan — no caller anywhere) | `ch08_applications.power_law_fit` | near |
| `MCD/three_fitting_method_and_plotting.m` (call site commented out) | `ch08_applications.three_distribution_fits` | near — the truncated power law's 3rd parameter is *unidentifiable* (deviation D2) |
| `SIFI/sea_ice_model.m` (§8.2.1, Figs. 8.12/8.13) | `ch08_applications.sea_ice_model` | `Area`/`Center`/`Perimeter`/`Radius`/`Intersect` **exact**; `Vertices` **near** (hull vertex *order*) |
| `SIFI/SeaIce_Image_Structure.m` (Appendix B) | `core.icestruct.*`, `ch08_applications.sea_ice_image_structure` | **exact** — all 12 numeric `Field` members and all 51 `FSD` triplets |
| `SIFI/color_hist.m` (Figs. 8.11/8.14) | `ch08_applications.color_hist` | **exact** |
| `SIFI/color_hist_comparison.m` (Fig. 8.15) | `ch08_applications.color_hist_comparison` | **exact** |
| *(no `.m` — §8.1's two methods are text only)* | `ch08_applications.shipborne_ice_concentration` | method **exact** (ch03's Otsu / the authors' own k-means); **results `unverified`** |
| *(no `.m` — §8.2's "Algorithms 3, 4 and 5 are carried out directly")* | `ch08_applications.sea_ice_field` | **unverified** (it *is* ch07's chain, verified there) |

> **What this notebook will and will not claim.** §8.3 is the most solidly verified chain in the book: it is
> reproduced **bit-exactly** against MATLAB *and* against the authors' own shipped `MCD_results.mat`, down to
> α = **1.3704**. §8.2 is honest but incomplete — our pipeline finds **433 floes / 290 brash**, the book prints
> **498 / 201**, the authors' parameter set appears nowhere in the book, and **no parameter search was
> performed** to close the gap (open item O2). §8.1 has neither code nor data: it is demonstrated on a
> *synthetic* frame, and **every number the book prints for §8.1 stays unverified** (open item O1). The
> rectification that precedes §8.1 is Appendix A.1.2 and belongs to **Chapter 10** — no second rectifier is
> written here.
""")

# =====================================================================================================================
# 2. Setup cells (Colab / Drive / local) — copied verbatim from build_ch02.py
# =====================================================================================================================
md(r"""
## Setup (Google Colab or local Jupyter)

Run the two cells below first. On **Colab** the first cell mounts your Google Drive and moves into
`MyDrive/Sea_Ice_Colab` — the folder where your private copy of the book's data lives
(`data/book/ch08/MCD/*.mat`, and `data/book/ch07/.../sea_ice_test.jpg`) and where downloads are cached between
sessions; readers without Drive get a temporary `/content/Sea_Ice_Colab`. The second cell clones (or updates)
the public repository `seaice-py` there and installs its requirements. **Locally** both cells are no-ops that
move to the repository root. No GPU is needed.

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

Chapter 8 needs two things, and they behave differently.

1. **The §8.2 image.** Figure 8.8 is *the same photograph* as Chapter 6/7's `sea_ice_test.jpg`, printed
   transposed and inverted (the analysis correlated the printed bitmap at **NCC +0.982** with the inverted
   transpose of the shipped file, and +0.974 with the Fig. 7.22 bitmap). `seaice.core.io.load_image` finds
   **your private copy** first — `data/book/ch08/…`, then `data/book/ch07/…`, in the current folder, in the
   repository or in Drive — and otherwise downloads a **public-domain NASA MODIS scene of the same
   394 × 1038 size** into `data/online/ch08/`, so the notebook runs either way.
2. **The §8.3 data.** `ch8/MCD/` ships two `.mat` files: `IceImage_290915_2_jpg.0000179.mat` (the Appendix-B
   structure of the helicopter frame — **2888 floes, 3452 brash pieces**) and `MCD_results.mat` (the authors'
   own saved `Raw_MCD`, 1 × 2888 — a **gold** reference). They come with the book's MATLAB archive, are **not**
   in the public repository, and there is no public-domain substitute for a *data structure*. When they are
   missing §8.3 below does **not** fail: it runs the identical chain on the field **we** identify in §8.2,
   prints `SKIP (book data absent)` wherever a book number would go, and says so on every figure.
""")
code(r'''
from seaice.core.io import data_roots, load_image

# Figure 8.8 = ch6/ch7's `sea_ice_test.jpg`.  Private copy under ch08, then under ch07 (where the book's own
# archive puts it), then the registered public-domain substitute.  No image path is hard-coded anywhere else.
I = SOURCE = None
for _chapter in ("ch08", "ch07"):
    try:
        I, SOURCE = load_image(_chapter, "sea_ice_test.jpg", allow_fallback=False)
        break
    except FileNotFoundError:
        pass
if I is None:
    I, SOURCE = load_image("ch08", "sea_ice_test.jpg")               # public-domain substitute

FROM_BOOK = SOURCE.startswith("book")
book = (lambda s: f"  (book: {s})") if FROM_BOOK else (lambda s: "")        # book values only for the book image
if not FROM_BOOK:
    print("Running on a public-domain substitute image; figures show the same operations, "
          "but values quoted in the book only hold for the book's own image.")
print(f"Figure 8.8 (= sea_ice_test.jpg): {I.shape} {I.dtype}   [{SOURCE}]")
print(f"the book prints it transposed, i.e. {I.shape[1]} x {I.shape[0]}" + book("Fig. 8.8, p. 184"))
if I.max() == I.min():
    print("WARNING: the image is constant — the download most likely failed; re-run this cell.")
''')
code(r'''
from pathlib import Path

def find_book_mat(name: str):
    """First match of `name` under any `data/book/ch08/` root, or None (the notebook then degrades gracefully).

    The roots are exactly the ones `load_image` searches: the current folder, the repository, and
    `MyDrive/Sea_Ice_Colab` on Colab.  Sub-folders are walked, so the book's own `MCD/` layout works verbatim.
    """
    for root in data_roots():
        folder = root / "data" / "book" / "ch08"
        if folder.is_dir():
            for hit in sorted(folder.rglob(name)):
                return hit
    return None

MAT_ICEIMAGE = find_book_mat("IceImage_290915_2_jpg.0000179.mat")     # main_WL_new.m line 7
MAT_MCD      = find_book_mat("MCD_results.mat")                       # fitting_iceFloes_distribution.m line 7
HAVE_BOOK_MAT = MAT_ICEIMAGE is not None
mcd_book = (lambda s: f"  (book: {s})") if HAVE_BOOK_MAT else (lambda s: "")

if HAVE_BOOK_MAT:
    print(f"section 8.3 data: {MAT_ICEIMAGE.as_posix()}")
    print(f"                  {MAT_MCD.as_posix() if MAT_MCD else 'MCD_results.mat NOT found (optional)'}")
else:
    print("SKIP (book data absent): the two section 8.3 .mat files were not found.\n"
          "  They ship inside the book's MATLAB archive as ch8/MCD/IceImage_290915_2_jpg.0000179.mat and\n"
          "  ch8/MCD/MCD_results.mat; they are not redistributable, so the public repository does not carry\n"
          "  them, and no public-domain substitute for a data STRUCTURE exists.\n"
          "  If you own the book: copy that folder to  <this folder>/data/book/ch08/MCD/  (on Colab:\n"
          "  MyDrive/Sea_Ice_Colab/data/book/ch08/MCD/) and re-run this cell.\n"
          "  Otherwise section 8.3 runs the SAME chain on the ice field identified in section 8.2, and every\n"
          "  number it prints is ours, not the book's.")
''')

md(r"""
## Imports

Everything algorithmic comes from the `seaice` package; the notebook only calls it and draws figures. §8.2 runs
the Chapter 6/7 pipeline at the image's **own resolution** (`SCALE = 1`) — about half a minute to a minute, one
GVF field plus one snake per initial contour, twice. §8.3 is essentially free: the expensive work was done by
the authors and is inside the `.mat`.
""")
code(r'''
import time

import numpy as np
import seaice.core                                        # imports seaice.core.plotting (sets the Agg backend)
from seaice.ch06_gvf_snake import BOOK_PARAMS
from seaice.ch07_ice_type import IcePiece, colorbar_area_ticks, size_color
from seaice.ch08_applications import (CIRCLE_ANGLES, COLOR_LIMIT_N, LENGTH_OVER_PIXEL, MCD_X_BIN,
                                      color_hist, color_hist_comparison, cumulative_fsd_powerlaw,
                                      iceimage_to_pieces, matlab_colon, mcd_analysis,
                                      plot_color_bar_and_floe, sea_ice_field, sea_ice_image_structure,
                                      sea_ice_model, shipborne_ice_concentration, three_distribution_fits)
from seaice.ch08_applications import _matlab_c1c2          # internal: the c(1), c(2) read of deviation D4
from seaice.core.histogram import hist as matlab_hist
from seaice.core.icestruct import load_iceimage_mat, overlap_graph, save_iceimage_mat
from seaice.core.plotting import imshow_matlab, label2rgb, matlab_jet, mcd_colorbar, size_colorbar
from seaice.core.stats import cumulative_size_distribution, mean_caliper_diameter
from seaice.core.synth import synthetic_floe_field

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 100

P = BOOK_PARAMS["sea_ice_demo"]          # ch6/ch7's shipped parameter block — section 8.2 uses it unchanged
SCALE = 1

def T(img):
    """Display transpose only: the book prints this scene as Fig. 8.8 (= Fig. 7.22), 394 rows x 1038 columns."""
    return np.swapaxes(img, 0, 1) if img.ndim == 3 else img.T

print(f"length_over_Pixel = {LENGTH_OVER_PIXEL} m/px, color_limit_N = {COLOR_LIMIT_N}, "
      f"x_bin = 1:100 ({MCD_X_BIN.size} centres)")
print("ch6/ch7 parameter block:", ", ".join(f"{k} = {v}" for k, v in P.items()))
''')

# =====================================================================================================================
# 8.1 — shipborne ice concentration
# =====================================================================================================================
md(r"""
## 8.1 A shipborne camera system to acquire sea ice concentration at engineering scale (pp. 176–181)

During expedition **OATRC'15** (NTNU + the Swedish Polar Research Secretariat) a four-lens "Ice Concentration"
camera covering roughly 180° was mounted on IB *Frej* and triggered **once every 10 s**. The camera looks
obliquely at the sea, so a raw frame is perspective-distorted — many pixels per square metre near the ship, few
far away — and a pixel count would be meaningless. The frame is therefore **rectified** first, with the linear
four-corner method of **Appendix A.1.2**: the image coordinates of the four corners of a rectangle of known
physical size (a checkerboard) give a transformation matrix. Because the checkerboard sits **5.90 m** above the
sea plane and the camera **25.50 m**, the rectified plane has to be projected a second time onto the sea plane;
the two white triangles outside the lens footprint are cropped, leaving a region of about **400 m × 100 m**.

Ice concentration is then simply *ice pixels / total pixels*, and §8.1.2 extracts the ice pixels **two ways at
once**: the global **Otsu** threshold of §3.1 (white = ice), and **k-means with k = 3** of §3.2, whose clusters
are read as water (black) / **wet ice** (grey: rubble, young ice, melt ponds) / dry ice (white) — with the grey
cluster counted as ice. That is why the k-means series of Fig. 8.5 sits systematically *above* the Otsu series
over the six hours of 26 September 2015, and it explains both discrepancy events of Fig. 8.6: **Event #1**, dark
young ice that Otsu calls water, and **Event #2**, adverse illumination that pushes ice below Otsu's threshold.
"If we choose two clusters for the k-means method (k = 2), then this method is approximately reduced to the Otsu
thresholding method" (p. 179).

> **No MATLAB file and no images ship for §8.1.** Figures 8.1–8.6 are credited to Lu, Zhang, Lubbad, Løset &
> Skjetne (OTC Arctic Technology Conference 2016), and neither the photographs nor the ~2160-frame time series
> are anywhere in the book's code archive. So this section demonstrates the **method** on a *synthetic* frame,
> and **every number the book prints for §8.1 is `unverified`** (open item O1) — none of them is reproduced
> here. The rectification is **not** implemented here either: it is Appendix A.1.2 and belongs to Chapter 10,
> which owns the rectifier. Writing a second one in Chapter 8 would be exactly the kind of silent duplication
> the project forbids.
>
> The two *methods*, on the other hand, are already verified: `graythresh`/`im2bw` are `exact` from ch03, and
> `shipborne_ice_concentration(..., impl='authors')` — the **default** — routes k-means through
> `core.clustering.kmeans_gray`, a line-by-line port of the authors' own `MATLAB_ROOT/ch3/kmeans.m`, which is
> **deterministic** (equal-division initialisation, histogram-weighted centroids, exact `mu == oldmu` stop) and
> `exact`. The alternative `impl='lloyd'` is a generic seeded k-means++/Lloyd routine and is only `approx`.
""")
code(r'''
# A synthetic three-level frame, the way section 8.1.2 reads the k-means clusters: water / "wet ice" (grey
# rubble, young ice, melt ponds) / dry ice.  Tier 3 — synthetic and seeded.  NOT the book's imagery.
rng = np.random.default_rng(0)
dry = synthetic_floe_field((300, 400), n_floes=14, seed=0, radius=(18, 45)) > 0
wet = (synthetic_floe_field((300, 400), n_floes=10, seed=1, radius=(10, 26)) > 0) & ~dry
frame = np.full((300, 400), 12.0)
frame[wet] = 75.0                       # deliberately BELOW the Otsu threshold: the Event #1 situation, p. 181
frame[dry] = 210.0
frame = np.clip(np.floor(frame + rng.normal(0.0, 6.0, frame.shape) + 0.5), 0, 255).astype(np.uint8)
print(f"synthetic layers (ground truth): dry ice {100 * dry.mean():.2f} %, wet ice {100 * wet.mean():.2f} %, "
      f"water {100 * (~dry & ~wet).mean():.2f} %")

o  = shipborne_ice_concentration(frame, "otsu")                                   # section 3.1
k3 = shipborne_ice_concentration(frame, "kmeans", k=3)                            # section 3.2, ice = grey + white
print(f"Otsu          : level = {o.level:.6f} ({255 * o.level:.1f}/255), IC = {100 * o.concentration:.2f} % "
      f"-> it finds the dry ice only ({100 * dry.mean():.2f} % is the truth for dry ice alone)")
print(f"k-means (k = 3): centres {np.round(k3.centers, 2).tolist()}, ice = the 2 brightest clusters, "
      f"IC = {100 * k3.concentration:.2f} %")
print(f"k-means - Otsu = {100 * (k3.concentration - o.concentration):+.2f} pp — the SIGN the book reports for "
      "Figure 8.5 (p. 180): the grey 'wet ice' cluster counts as ice, so k-means always reads higher.")
print("These numbers are OURS, measured on a synthetic frame. The book's 6-hour series, Event #1 and Event #2 "
      "are `unverified` (open item O1): their imagery does not ship.")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, im, title in ((axes[0], frame, "synthetic rectified frame (Tier 3)"),
                      (axes[1], o.mask, f"Otsu: IC = {100 * o.concentration:.1f} %"),
                      (axes[2], k3.labels, "k-means, k = 3: water / wet ice / dry ice"),
                      (axes[3], k3.mask, f"ice = grey + white: IC = {100 * k3.concentration:.1f} %")):
    ax.imshow(im, cmap="gray"); ax.set_axis_off(); ax.set_title(title)
fig.suptitle("Section 8.1.2 (procedure of Figures 8.4-8.6) — two ice-pixel detectors on ONE frame "
             "(synthetic data, not the book's)")
plt.show()
''')
md(r"""
The grey layer is the whole story of §8.1: Otsu can only split the histogram once, so wet ice falls on the water
side of the threshold; k-means with three clusters gives it a class of its own, and the analyst decides that the
class is ice. Which answer is "right" depends on what an ice-load formula counts as ice — the book reports both
series and lets the onboard ice observer arbitrate.

The claim "k = 2 ≈ Otsu" deserves a careful test rather than a nod, and the test needs a **discriminating**
fixture. On a two-tone frame the claim is true for *every* threshold in the empty band between the two levels,
so any agreement is an artefact. The frame below instead has pixels at **every** grey level between water and
ice, so the two thresholds have to agree pixel by pixel to agree at all.
""")
code(r'''
# A discriminating fixture: water 30 | a grey ramp covering EVERY level 55...205 | dry ice 225, +/-5 noise.
rng = np.random.default_rng(1)
ramp = np.zeros((80, 120), dtype=np.uint8)
ramp[:, :50] = 30
ramp[:, 50:100] = np.linspace(55, 205, 50).astype(np.uint8)[None, :]
ramp[:, 100:] = 225
ramp = np.clip(ramp.astype(np.int16) + rng.integers(-5, 6, ramp.shape), 0, 255).astype(np.uint8)

r_otsu = shipborne_ice_concentration(ramp, "otsu")
r_k2   = shipborne_ice_concentration(ramp, "kmeans", k=2, ice_clusters=1)     # impl='authors' (the default)
r_k3   = shipborne_ice_concentration(ramp, "kmeans", k=3)
mid = float(r_k2.centers.mean())
print(f"Otsu threshold           : {255 * r_otsu.level:.4f}/255")
print(f"k = 2 centres            : {np.round(r_k2.centers, 6).tolist()}  ->  midpoint {mid:.4f}")
print(f"=> two DIFFERENT thresholds; the masks nevertheless agree on "
      f"{100 * (r_k2.mask == r_otsu.mask).mean():.2f} % of pixels "
      f"({100 * r_k2.concentration:.4f} % vs {100 * r_otsu.concentration:.4f} % ice)")
lo, hi = int(np.floor(min(mid, 255 * r_otsu.level))), int(np.ceil(max(mid, 255 * r_otsu.level)))
print(f"   the fixture is not degenerate: {int((ramp == lo).sum())} pixels sit at level {lo} and "
      f"{int((ramp == hi).sum())} at {hi}, so a one-level move of either threshold would break the agreement")
print(f"k = 3, ice = the two brightest clusters: {100 * r_k3.concentration:.4f} % "
      f"({100 * (r_k3.concentration - r_otsu.concentration):+.2f} pp) — the Figure 8.5 mechanism again")
print("\np. 179's claim is reproduced AS A RELATION on this fixture. It is still not the book's number: "
      "section 8.1's own frames do not exist here (open item O1).")
''')

# =====================================================================================================================
# 8.2 — ice field characterization
# =====================================================================================================================
md(r"""
## 8.2 Numerical characterization of a real ice field (pp. 181–189)

The §8.2 data came from a NORUT remote-sensing mission at **78°55′N 11°56′E** (Hamnerabben, Ny-Ålesund,
6–8 May 2011): a **CryoWing** UAV (Table 8.1(a)) carrying a Canon EOS 450D with an EF 28 mm lens, 4290 × 2856 px
(Table 8.1(b)). The purpose is different from Chapter 7's: not to *look* at an ice field but to hand it to a
numerical simulator. Because the camera points straight down, "**since the image is undistorted, Algorithms 3, 4
and 5 are carried out directly**" (p. 182) — the entire Chapter 6/7 chain with no rectification and no tiling.
That is what `sea_ice_field` does, and there is no `.m` file for it in `ch8/`: it is `sea_ice_demo.m`'s first two
calls, already ported.

> **Open item O2 — read this before the numbers.** The book reports **498 ice floes and 201 brash pieces** and
> coverages **76.73 / 0.46 / 9.05 / 13.76 %** for this image. Running Algorithms 3/4/5 on the very same
> photograph with the **shipped** `sea_ice_demo.m` parameter block gives **433 floes / 290 brash** and
> **76.23 / 0.99 / 9.21 / 13.57 %**. The coverages are close; the *counts* are not, and the book never prints
> the parameters it used. A second, independent piece of evidence says the same: Fig. 8.10's colour-bar tick
> list has a unique pre-image, and it pins the book's **smallest** piece at exactly **1 px** and its largest in
> **[3484, 3503] px**, against our 2 px and 2871 px — a different segmentation, not a different colour bar.
> **No parameter search was performed to close the gap**, deliberately: tuning until a number matches would
> destroy the evidence value of every other number in the chapter. So §8.2's figures below are reproduced *as a
> pipeline* and compared qualitatively; the counts are `unverified`.
""")
code(r'''
RGB = I[::SCALE, ::SCALE]
print(f"Algorithms 3+4+5 on {RGB.shape[0]} x {RGB.shape[1]} px ('carried out directly', p. 182): "
      f"Num = {P['Num']} GVF iterations, iter = {P['iter']} snake iterations, kms0 = {P['kms0']} clusters ...")
t0 = time.time()
ENH = sea_ice_field(RGB, verbose=False)                 # = sea_ice_edge_detection -> ice_shape_enhancement
print(f"  {time.time() - t0:.1f} s")

COV = ENH.coverage.as_percent()
print(f"  {len(ENH.ice_floe)} ice floes, {len(ENH.brash_ice)} brash pieces"
      + book("498 ice floes, 201 brash pieces, p. 182 — NOT reproduced, open item O2"))
print(f"  coverage {COV['IceFloe']:.2f} % floe / {COV['BrashIce']:.2f} % brash / {COV['Slush']:.2f} % slush / "
      f"{COV['Water']:.2f} % water" + book("76.73 / 0.46 / 9.05 / 13.76 %, p. 182"))
print(f"  floe areas {int(min(f.Area for f in ENH.ice_floe))} .. "
      f"{int(max(f.Area for f in ENH.ice_floe))} px"
      + book("the Fig. 8.10 tick list implies the book's were 1 px .. [3484, 3503] px"))

fig, axes = plt.subplots(4, 1, figsize=(13, 9))
for ax, layer, t in ((axes[0], ENH.index_floe != 0, f"(a) ice floes — {len(ENH.ice_floe)}"),
                     (axes[1], ENH.index_brash != 0, f"(b) brash ice — {len(ENH.brash_ice)}"),
                     (axes[2], ENH.index_slush != 0, f"(c) slush — {COV['Slush']:.2f} %"),
                     (axes[3], ENH.index_water != 0, f"(d) water — {COV['Water']:.2f} %")):
    imshow_matlab(ax, T(layer), title=t)
fig.suptitle("Section 8.2 (procedure of Figure 8.9)  The four ice-type layers of the field to be characterized")
plt.show()
''')
code(r'''
# Figure 8.10: every identified piece coloured by its size with ch07's Eq. (7.6),
#     Color(p) = C1 * (1 - exp(-area/C2)),   C1 = 10000, C2 = 1000,
# and the colour bar labelled with that equation INVERTED, -round(C2*ln(1 - c/C1)).
cmap = plt.get_cmap("jet").copy(); cmap.set_bad("white")
fig, ax = plt.subplots(figsize=(13, 5.6))
im = ax.imshow(T(np.ma.masked_where(ENH.index == 0, ENH.index)), cmap=cmap, interpolation="nearest")
if ENH.floe_cen.size:
    ax.plot(ENH.floe_cen[:, 0] - 1, ENH.floe_cen[:, 1] - 1, "k*", markersize=3)
if ENH.brash_cen.size:
    ax.plot(ENH.brash_cen[:, 0] - 1, ENH.brash_cen[:, 1] - 1, "k.", markersize=1.5)
ax.set_axis_off()
colours = np.concatenate([ENH.color_floe, ENH.color_brash])
size_colorbar(fig, im, colours, 6, ax=ax, label="piece area (px), Eq. (7.6) inverted")
ax.set_title("Section 8.2 (procedure of Figure 8.10)  size-coded pieces; * = floe centres, . = brash centres")
plt.show()

_, ticks = colorbar_area_ticks(colours, 6)
print(f"our colour-bar tick labels: {ticks.tolist()}")
print("book Figure 8.10 (p. 186) : [1, 177, 391, 664, 1040, 1650, 3483]")
print("  Those seven integers have a unique pre-image (C_min = 9, C_max in 9693..9698), which is how we know "
      "the book's smallest piece was 1 px and its largest ~3.5k px. Ours differ — open item O2, again.")
''')

md(r"""
### 8.2.1 Sea ice numerical modeling (pp. 184–189) — replaces `sea_ice_model.m`

A discrete-element simulator cannot ingest a bitmap; it needs *objects*. So every identified piece is replaced
by a simple geometric primitive: **each ice floe by its bounding minimum-area (convex) polygon** — in the code
`convhull(x, y, 'simplify', true)` on the floe's own pixel coordinates — and **each brash piece by an
area-equivalent disk**, $r_i = \sqrt{A_i/\pi}$, $p_i = 2\pi r_i$. Because a hull is bigger than the shape it
wraps, "the polygonized floes … may overlap other floes and brash pieces", so each modelled piece also carries
an **overlap flag**: the serial numbers of the floes and brash pieces whose boundary it crosses. A DEM simulator
uses that list to resolve the initial interpenetration before the first time step (Fig. 8.16).

Three details of the shipped code matter more than the geometry:

* the areas, centres and perimeters come from `polygeom` (H. J. Sommer III), not from `regionprops` — the
  `regionprops` + `polyarea` alternative is in the file, commented out;
* floe–floe and floe–brash overlap is `polyxpoly`, a genuine polyline crossing test, but **brash–brash** is the
  pure circle test $\lVert c_i - c_j \rVert < r_i + r_j$;
* the crossing test is guarded by `if xx ~= NaN`, which does **not** mean what it looks like — see the *MATLAB
  traps* section below.

Parity: `Area`, `Center`, `Perimeter`, `Radius` and all four `Intersect` lists are **exact** against MATLAB and
against the authors' own shipped structure (all 3992 overlap entries, as sets); only the hull's vertex *order*
differs, so the contract is on the vertex **set** and on the rasterised mask.
""")
code(r'''
t0 = time.time()
MODEL = sea_ice_model(ENH.ice_floe, ENH.brash_ice, ENH.index_floe)
print(f"sea_ice_model.m: {time.time() - t0:.1f} s for {len(MODEL.floe)} floes + {len(MODEL.brash)} brash pieces")
print(f"  crossing tests actually run: {MODEL.n_pairs_tested} of the M-file's {MODEL.n_pairs_total} "
      "(an axis-aligned bounding-box prefilter — provably inert: a common point of two polylines lies in BOTH "
      "bounding boxes, so disjoint boxes cannot cross)")

poly_area = np.array([f.Area for f in MODEL.floe])
raw_area = np.array([float(p.Area) for p in ENH.ice_floe])
print(f"  polygon areas {poly_area.min():.1f} .. {poly_area.max():.1f} px^2 "
      f"(pixel counts {raw_area.min():.0f} .. {raw_area.max():.0f})")
print(f"  p. 184 says 'the polygonized floes will not be smaller than the actual identified floes' — yet "
      f"{int((poly_area < raw_area).sum())} of {poly_area.size} continuous hull areas are SMALLER than the "
      "pixel count they enclose, because polygeom measures the hull through pixel CENTRES (erratum E6). "
      "The rasterised hull is still a superset, which is the claim that matters for a simulator.")
print(f"  brash radii {min(b.Radius for b in MODEL.brash):.3f} .. "
      f"{max(b.Radius for b in MODEL.brash):.3f} px")
print(f"  overlap flags: floe-floe {sum(f.Intersect.floe.size for f in MODEL.floe)}, "
      f"floe-brash {sum(f.Intersect.brash.size for f in MODEL.floe)}, "
      f"brash-brash {sum(b.Intersect.brash.size for b in MODEL.brash)}, "
      f"brash-floe {sum(b.Intersect.floe.size for b in MODEL.brash)}")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
axes[0].imshow(MODEL.bw_floe.T, cmap="gray", vmin=0, vmax=1)
for f in MODEL.floe:
    axes[0].plot(f.Vertices[:, 1] - 1, f.Vertices[:, 0] - 1, linewidth=0.4)
    axes[0].plot(f.Center[1] - 1, f.Center[0] - 1, "r+", markersize=2)
axes[0].set_axis_off(); axes[0].set_title("(a) polygonized ice floes (convex hulls) and their centres")
axes[1].imshow(MODEL.bw_brash.T, cmap="gray", vmin=0, vmax=1)
for b in MODEL.brash:
    axes[1].plot(b.Center[1] - 1, b.Center[0] - 1, "r.", markersize=1)
axes[1].set_axis_off(); axes[1].set_title("(b) circularized brash ice (area-equivalent disks)")
fig.suptitle("Section 8.2.1 (procedure of Figure 8.12)  the sea ice numerical model")
plt.show()
''')
code(r'''
# Figure 8.13 — a close-up, so that the polygon boundaries and the two kinds of centre are visible.
# The book does not say which region it shows, so the densest 200x200 box is used and labelled as our choice.
H, W = ENH.index_floe.shape[:2]
cx = np.array([f.Center[0] for f in MODEL.floe]); cy = np.array([f.Center[1] for f in MODEL.floe])
w = min(200, W); h = min(200, H)
x0 = int(np.clip(np.median(cx) - w / 2, 0, max(W - w, 0)))
y0 = int(np.clip(np.median(cy) - h / 2, 0, max(H - h, 0)))
t_circ = np.linspace(0, 2 * np.pi, 60)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(x0, x0 + w); ax.set_ylim(y0 + h, y0); ax.set_aspect("equal")
for i, f in enumerate(MODEL.floe):
    if x0 - 50 <= f.Center[0] <= x0 + w + 50 and y0 - 50 <= f.Center[1] <= y0 + h + 50:
        ax.plot(f.Vertices[:, 0] - 1, f.Vertices[:, 1] - 1, "b-", linewidth=1)
        ax.plot(ENH.ice_floe[i].Center.ravel()[0] - 1, ENH.ice_floe[i].Center.ravel()[1] - 1, "k*", markersize=6)
        ax.plot(f.Center[0] - 1, f.Center[1] - 1, "r+", markersize=8)
for b in MODEL.brash:
    if x0 - 50 <= b.Center[0] <= x0 + w + 50 and y0 - 50 <= b.Center[1] <= y0 + h + 50:
        ax.plot(b.Center[0] - 1 + b.Radius * np.cos(t_circ), b.Center[1] - 1 + b.Radius * np.sin(t_circ),
                "g-", linewidth=0.8)
        ax.plot(b.Center[0] - 1, b.Center[1] - 1, "r.", markersize=4)
ax.set_title(f"Section 8.2.1 (procedure of Figure 8.13) — close-up, rows {y0}..{y0 + h}, cols {x0}..{x0 + w}\n"
             "blue = hull, green = disk, black * = identified centre, red + = modelled centre")
plt.show()
print("The identified centre and the modelled centre nearly coincide — the convex hull is a good area-preserving "
      "stand-in for a rounded MIZ floe, which is the assumption the whole section rests on.")
''')

md(r"""
### The floe size distributions of Figs. 8.11, 8.14 and 8.15 — replaces `color_hist.m` and `color_hist_comparison.m`

Both files are the same 50-bin histogram over `min_x : inter : max_x` = `20 : 70 : 3500`, drawn with each bar
jet-coloured by ch07's **Eq. (7.6)** applied to its own bin centre, and with the colour bar's ticks labelled by
that equation *inverted*. Fig. 8.11 histograms the **identified** floes (pixel areas), Fig. 8.14 the
**polygonized** floes (`sea_ice_model` areas), and Fig. 8.15 the difference — the "floe size distribution error
due to the shape simplification".

Two things about those figures are worth knowing before you read any agreement as evidence:

* the nine printed tick labels **20, 149, 297, 471, 682, 950, 1317, 1902, 3487** are a function of the three
  literals `min_x = 20`, `max_x = 3500`, `nn = 8` **and of nothing else** — a single 1-pixel floe and 500 copies
  of a 3499-pixel floe give the identical list. Reproducing them proves nothing about the floes (that is why the
  verification report explicitly disqualifies them as parity evidence);
* those labels come from `color_hist.m`'s `max_x = 3500`, but the shipped `color_hist_comparison.m` sets
  `max_x = 6000`, which would print **20, 153, 307, 488, 710, 996, 1398, 2081, 5952**. The figure in the book was
  made with the *other* file's constant — erratum **E3**. The port keeps the shipped 6000 as the default and
  takes `max_x=3500` as an argument to reproduce the plate.
""")
code(r'''
def draw_color_hist(ax, ch, title, ylabel="number of ice floes"):
    """`color_hist.m` lines 18-33 redrawn: bar(n, z), each bar jet-coloured at Eq. (7.6) of its own centre.

    (The M-file colours the bars through an HG1 `FaceVertexCData` block that *errors* in MATLAB R2025a; the
    numbers behind it — `z`, `n`, `color`, `ysh`, `YT` — are reproduced exactly, and only the drawing differs.)
    """
    cmap = plt.get_cmap("jet")
    norm = plt.Normalize(ch.color_min, ch.color_max)
    ax.bar(ch.n, ch.z, width=ch.params["inter"], color=cmap(norm(ch.color)), edgecolor="k", linewidth=0.2)
    ax.set_xlabel("floe size [pixels]"); ax.set_ylabel(ylabel); ax.set_title(title)
    cb = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cb.set_ticks(ch.tick_values); cb.set_ticklabels([str(int(v)) for v in ch.tick_labels])
    return cb

CH_RAW  = color_hist(raw_area)                          # Figure 8.11 — the identified floes
CH_POLY = color_hist(poly_area)                         # Figure 8.14 — the polygonized floes
print(f"{CH_RAW.nbins} bins, centres {CH_RAW.centers[0]:g} .. {CH_RAW.centers[-1]:g} px; the bars are drawn at "
      f"n + inter/2 = {CH_RAW.n[0]:g} .. {CH_RAW.n[-1]:g} (the M-file shifts them)")
print(f"z sums to {int(CH_RAW.z.sum())} = every floe: MATLAB's hist() takes bin CENTRES and its two outer bins "
      "are UNBOUNDED, so nothing falls off the ends (core.histogram.hist, not np.histogram)")
print(f"tick labels: {CH_RAW.tick_labels.tolist()}")
print("book Figs. 8.11/8.14/8.15 (pp. 187-188): [20, 149, 297, 471, 682, 950, 1317, 1902, 3487] — the same, and "
      "that agreement is a property of three constants, not of the data.")

fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
draw_color_hist(axes[0], CH_RAW, "Figure 8.11 — identified ice floes (pixel areas)")
draw_color_hist(axes[1], CH_POLY, "Figure 8.14 — polygonized ice floes (convex-hull areas)")
plt.show()
''')
code(r'''
# Figure 8.15 — the difference.  `color_hist_comparison.m` line 29 computes `z_d = z - z0` with
# z  = hist(polygonized areas)  (Fig. 8.14)  and  z0 = hist(identified areas)  (Fig. 8.11),
# i.e. POLYGONIZED minus IDENTIFIED.
CMP = color_hist_comparison(poly_area, raw_area, max_x=3500)      # max_x = 3500 reproduces the printed bar (E3)
print(f"z_d[:6] = {CMP.z_d[:6].astype(int).tolist()}, range {int(CMP.z_d.min())} .. {int(CMP.z_d.max())}, "
      f"sum {int(CMP.z_d.sum())} (= 0: polygonization moves floes between bins, it does not create them)")
print(f"first bin z_d[0] = {int(CMP.z_d[0]):+d}")
print("\nERRATUM E11 — the book states this subtraction the wrong way round. p. 188 reads 'by subtracting the "
      "histograms of Figure 8.12(a) from Figure 8.10', i.e. identified - polygonized; the code computes "
      "polygonized - identified. The PRINTED PLATE follows the code: its leftmost bar rises ABOVE the zero "
      "line, the same sign as our z_d[0] above. Fig. 8.15's own caption ('Error between the floe size "
      "distributions of Figure 8.11 and Figure 8.14') is order-neutral and correct.")
print("The code and the plate are the authority; the body sentence is the erratum. A global sign flip in the "
      "port would now be caught by the sign of the first bin, not merely by the shape of the bars.")

fig, ax = plt.subplots(figsize=(10, 4.6))
draw_color_hist(ax, CMP, "Figure 8.15 — floe size distribution error, $z - z_0$ (polygonized - identified)",
                ylabel="difference in number of ice floes")
ax.axhline(0, color="k", linewidth=0.8)
plt.show()
''')
md(r"""
The error is largest at the small-size end and oscillates about zero: hull-fitting nudges most floes up by one
or two bins, so a bin gains what its left neighbour loses. That is exactly what the book means by "floe size
distribution error … due to the shape simplification" — the *shape* of the distribution survives
polygonization, its fine structure does not.
""")

# =====================================================================================================================
# 8.3 — floe size statistic
# =====================================================================================================================
md(r"""
## 8.3 Sea ice floe size statistic (pp. 189–194)

This is the part of the chapter the reader can reproduce completely. During OATRC'15 a helicopter (AS-335NP off
IB *Oden*) flew at about **1314 m** at 40 m/s with a gyro-stabilised **Red Dragon** camera (5568 × 3132, 25 fps,
sampled at 0.1 Hz) and covered 3 × 3 nautical miles. One frame (Fig. 8.18) was processed by the Chapter 7
pipeline into **2888 ice floes and 3452 brash pieces**, covering 58.00 / 4.85 / 21.21 / 15.94 % — and that
processed frame is the Appendix-B structure that ships with the book. Pixels become metres through a single
literal, `length_over_Pixel = 1.1794` m/px, obtained from IB *Oden*'s known length in frames taken at an
equivalent height (the book never prints the number; only the code has it).

Each floe is then sized by **Eq. (8.1)**, the "mean clipper diameter" (MCD):

$$L_i = \sqrt{\frac{4 A_i}{\pi}}$$

which is the diameter of the circle of the same area. Two remarks the book does not make: "clipper" is a typo
for **caliper** (the standard term, Rothrock & Thorndike 1984), and Eq. (8.1) is not a caliper diameter at all —
it is the **area-equivalent circle diameter**. The acronym is kept because the code, the folder and the figures
all use it (erratum **E5**).

> **Without the shipped `.mat` files** the cells below run the identical chain on the field identified in §8.2
> and say so on every figure. Nothing is faked: the book's own numbers are printed only when its own data is
> present.
""")
code(r'''
if HAVE_BOOK_MAT:
    ICE = load_iceimage_mat(MAT_ICEIMAGE)                    # main_WL_new.m line 7
    ICE_LABEL = "the authors' shipped section 8.3 structure"
else:
    # Graceful degradation: the same Appendix-B structure, built from OUR section 8.2 field (SeaIce_Image_Structure.m).
    ICE = sea_ice_image_structure(ENH.ice_floe, ENH.brash_ice, MODEL.floe, MODEL.brash, ENH.coverage,
                                  ENH.index_floe, ENH.index_residue)
    ICE_LABEL = "OUR section 8.2 field (SKIP: the book's section 8.3 .mat is absent)"
print(f"{ICE_LABEL}\n  {ICE.summary()}")
print(f"  Field: NumFloes = {ICE.Field.NumFloes}, NumBrash = {ICE.Field.NumBrash}"
      + mcd_book("2888 floes, 3452 brash pieces, p. 190"))
cov = [ICE.Field.CovFloes, ICE.Field.CovBrash, ICE.Field.CovSlush, ICE.Field.CovWater]
print("  coverage " + " / ".join(f"{100 * c:.5f}" for c in cov) + " %"
      + mcd_book("58.00 / 4.85 / 21.21 / 15.94 %, p. 190"))
if HAVE_BOOK_MAT:
    print("  ERRATUM E4: the water figure is TRUNCATED, not rounded — 15.945957 % is printed as 15.94 %, "
          "where correct rounding gives 15.95 %. The other three are the same either way.")

# Eq. (8.1), on the pixel areas (main_WL_new.m lines 30-31).
areas_px = np.array([f.Area for f in ICE.Floe], dtype=float)
mcd = mean_caliper_diameter(areas_px, LENGTH_OVER_PIXEL)
print(f"\nEq. (8.1)  L = sqrt(4A/pi):  {mcd.size} floes, MCD {mcd.min():.4f} .. {mcd.max():.4f} m, "
      f"mean {mcd.mean():.4f} m" + mcd_book("Fig. 8.21 spans 10^0.8 .. 10^2.1 m"))
print(f"  check by inverting it: pi*(L/2)^2 / lop^2 - A = {np.abs(np.pi * (mcd / 2) ** 2 / LENGTH_OVER_PIXEL ** 2 - areas_px).max():.3e} px^2")
''')
code(r'''
# `main_WL_new.m` in one call: Eq. (8.1) on the polygon areas AND on the pixel areas, then
# plot_color_bar_and_floe for each.  The second call is Figure 8.19 and its histogram is Figure 8.20.
t0 = time.time()
R = mcd_analysis(ICE, length_over_pixel=LENGTH_OVER_PIXEL, color_limit_n=COLOR_LIMIT_N)
print(f"main_WL_new.m: {time.time() - t0:.2f} s")
print(f"  Raw  (pixel areas -> Figs. 8.19/8.20): branch = {R.raw.kind}, hist(MCD, 1:100) sums to "
      f"{int(R.raw.counts.sum())}, peak {int(R.raw.counts.max())} at {R.raw.centers[int(R.raw.counts.argmax())]:g} m"
      + mcd_book("Fig. 8.20 axes stop at N = 450, so the 454-floe peak at 7 m is clipped by the plate"))
print(f"  Poly (polygon areas, computed but never printed in the book): branch = {R.poly.kind}, peak "
      f"{int(R.poly.counts.max())} at {R.poly.centers[int(R.poly.counts.argmax())]:g} m")
print(f"  colour index = the NEAREST histogram centre (not the hist bin): {R.raw.color_index.min()}"
      f"..{R.raw.color_index.max()}, of which {int((R.raw.color_index > COLOR_LIMIT_N).sum())} floes are clamped "
      f"to the {COLOR_LIMIT_N}th colour — that is what the book's '>=30 [m]' top tick means")
print(f"  rgbImage: {R.raw.rgb_image.shape}  (MATLAB never pre-allocates it — see the MATLAB traps below)")

if MAT_MCD is not None:
    import scipy.io as sio
    gold = np.asarray(sio.loadmat(MAT_MCD)["Raw_MCD"], dtype=float).ravel()
    print(f"\n  vs the authors' own saved Raw_MCD in MCD_results.mat ({gold.size} values): "
          f"max |delta| = {np.abs(gold - R.raw_mcd).max():.1f}")
    print("  That 0.0 is the strongest single piece of evidence in this chapter: the port reproduces the "
          "authors' own published intermediate result bit for bit, without ever seeing their code run.")
''')
code(r'''
# Figure 8.19 — every floe painted in the jet colour of its MCD, with a white dot at its centre.
IMG19 = R.raw.rgb_image if HAVE_BOOK_MAT else T(R.raw.rgb_image)
dot_x = R.raw.centres_xy[:, 0] - 1 if HAVE_BOOK_MAT else R.raw.centres_xy[:, 1] - 1
dot_y = R.raw.centres_xy[:, 1] - 1 if HAVE_BOOK_MAT else R.raw.centres_xy[:, 0] - 1

fig, ax = plt.subplots(figsize=(14, 8))
ax.imshow(IMG19)
ax.plot(dot_x, dot_y, "w.", markersize=1.5)
ax.set_axis_off()
mcd_colorbar(fig, ax, R.raw.color_m, (R.raw.centers[0], R.raw.centers[COLOR_LIMIT_N - 1]))
ax.set_title(("Figure 8.19 — ice floes coloured by MCD (white dots = floe centres)" if HAVE_BOOK_MAT else
              "Procedure of Figure 8.19 on OUR section 8.2 field (the book's section 8.3 data is absent)"))
plt.show()

# Figure 8.20 — the MCD histogram, bar i coloured color_M(min(i, 30), :).
idx = np.minimum(np.arange(1, R.raw.centers.size + 1), COLOR_LIMIT_N) - 1
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(R.raw.centers, R.raw.counts, width=1.0, color=R.raw.color_m[idx], edgecolor="none")
ax.set_xlabel("MCD $L_i$ [m]"); ax.set_ylabel("Number of ice floes $N$"); ax.grid(alpha=0.3); ax.margins(x=0)
mcd_colorbar(fig, ax, R.raw.color_m, (R.raw.centers[0], R.raw.centers[COLOR_LIMIT_N - 1]))
ax.set_title(("Figure 8.20 — MCD histogram of the floes of Figure 8.19" if HAVE_BOOK_MAT else
              "Procedure of Figure 8.20 on OUR section 8.2 field"))
plt.show()
nz = np.flatnonzero(R.raw.counts)
print(f"non-zero bins: centres {R.raw.centers[nz][:8].astype(int).tolist()} ... "
      f"{R.raw.centers[nz][-3:].astype(int).tolist()} m"
      + mcd_book("6..58 m plus two isolated bars at 76 and 84 m"))
print(f"colour bar: caxis = [{R.raw.centers[0]:g}, {R.raw.centers[COLOR_LIMIT_N - 1]:g}] -> MATLAB's default "
      "ticks 5, 10, 15, 20, 25, 30, printed in the book as '5 10 15 20 25 >=30 [m]'")
''')

md(r"""
### Eqs. (8.2) and (8.3): the cumulative floe size distribution and its power law (Fig. 8.21)

The **cumulative** distribution is evaluated at every observed size — **Eq. (8.2)**:

$$N_c(L) = \frac{N(\ge L)}{N_{total}},$$

so it starts at 1 for the smallest floe and ends at $1/N_{total}$ for the largest (the book calls it "the number
of ice floes per unit area with size no smaller than $L$", but the formula it prints, and the code it runs, is a
plain fraction with no area normalisation). The model is **Eq. (8.3)**:

$$N_c(L) \propto L^{-\alpha},$$

fitted in the code as $\hat N_c(L) = \varepsilon_1 L^{-\varepsilon_2}$ with $\alpha = \varepsilon_2$. The book
reports **α = 1.3704**.

Three things the code says and the text does not:

* **erratum E1 — `fitting_iceFloes_distribution.m` does not run as shipped.** Line 57 is
  `legend(gca,{'Observed data','Power law fitting''},'Box','on')`; the doubled quote escapes into the string and
  MATLAB R2025a refuses to parse the file ("*Invalid expression …*"). The script cannot have produced Fig. 8.21
  in the form it ships. Everything before line 57 — the whole fit — runs unchanged.
* **erratum E2 — the book describes an estimator it does not use.** p. 192 says α "is the slope of the power law
  curve on log-log plot (which is a straight line)". The code runs `lsqcurvefit` on the **untransformed**
  $(L, N_c)$ pairs. A log-log ordinary-least-squares line on the same data gives **1.87**, not 1.3704. Both are
  computed below; they are different estimators of the same exponent, and the log-log line weights the rare
  large floes far more heavily.
* the chapter computes **no goodness-of-fit statistic** for any of its three candidate distributions:
  `resnorm`, `residual`, `exitflag`, `output` and `jacobian` are all requested from `lsqcurvefit` and discarded.
""")
code(r'''
L, Nc = cumulative_size_distribution(R.raw_mcd)                     # Eq. (8.2), the M-file's literal O(N^2) loop
print(f"Eq. (8.2): N_c from {Nc.max():.6f} down to {Nc.min():.4e} = 1/{R.raw_mcd.size}"
      + mcd_book("Fig. 8.21 spans 10^-3.5 .. 10^0"))

FIT = cumulative_fsd_powerlaw(R.raw_mcd)                            # Eq. (8.3) via lsqcurvefit
print(f"Eq. (8.3): eps = [{FIT.eta[0]:.8f}, {FIT.eta[1]:.8f}]  ->  alpha = {FIT.alpha:.6f}"
      + mcd_book("alpha = 1.3704, p. 192"))
print(f"           resnorm = {FIT.resnorm:.10f}, exitflag = {FIT.exitflag}, "
      f"funcCount = {FIT.output['funcCount']}")
if HAVE_BOOK_MAT:
    print("           MATLAB R2025a's own lsqcurvefit on this very data returns [11.85785670, 1.37035854]:")
    print("           the measured relative differences are 6.6e-10 (eps1) and 2.4e-10 (alpha) — ten "
          "significant figures — with resnorm agreeing to 5.9e-14 and the same exitflag 3 "
          "(reports/ch08_verification.md).")

slope, intercept = np.polyfit(np.log10(L), np.log10(Nc), 1)         # the estimator the TEXT describes (E2)
print(f"log-log OLS line (the text's description): alpha = {-slope:.4f} — a different number from a different "
      "estimator, on the same data")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].loglog(L, Nc, "ro", markersize=4, markerfacecolor="none", label="Observed data")
axes[0].loglog(FIT.x_plot, FIT.y_plot, "k", linewidth=2, label="Power law fitting")
axes[0].set_xlabel("MCD [m]"); axes[0].set_ylabel("Cumulative Frequency")
axes[0].grid(True, which="both", alpha=0.3); axes[0].legend(loc="lower left")
axes[0].set_title(("Figure 8.21 — cumulative FSD and Eq. (8.3)" if HAVE_BOOK_MAT else
                   "Procedure of Figure 8.21 on OUR field") + f"\n$\\alpha$ = {FIT.alpha:.4f}")
axes[1].loglog(L, Nc, "ro", markersize=4, markerfacecolor="none", label="Observed data")
axes[1].loglog(FIT.x_plot, FIT.y_plot, "k", linewidth=2, label=f"lsqcurvefit (the code): $\\alpha$ = {FIT.alpha:.4f}")
axes[1].loglog(FIT.x_plot, 10 ** intercept * FIT.x_plot ** slope, "b--", linewidth=2,
               label=f"log-log OLS (the text): $\\alpha$ = {-slope:.4f}")
axes[1].set_xlabel("MCD [m]"); axes[1].set_ylabel("Cumulative Frequency")
axes[1].grid(True, which="both", alpha=0.3); axes[1].legend(loc="lower left")
axes[1].set_title("Erratum E2 — the estimator the text describes is not the one the code runs")
plt.show()
print("The book's own verdict, 'showing a good fit for the smaller floe sizes', is visible in the left panel: "
      "the line tracks the data up to ~20 m and then lies above it — the large floes are rarer than a pure "
      "power law predicts, which is the classical motivation for a TRUNCATED power law.")
''')
code(r'''
# `three_fitting_method_and_plotting.m` — the two distributions the book fits but never prints.  Its only call
# site, `fitting_iceFloes_distribution.m:37`, is commented out in the shipped file.
T3 = three_distribution_fits(FIT.x, FIT.y)
print(f"upper-truncated power law  eps1*(L^-eps2 - eps3^-eps2): eps = {np.round(T3.truncated_power.eta, 6).tolist()}, "
      f"resnorm = {T3.truncated_power.resnorm:.6e}")
print(f"Weibull survival  exp(-(L/eps2)^eps1)                 : eps = {np.round(T3.weibull.eta, 6).tolist()}, "
      f"resnorm = {T3.weibull.resnorm:.6e}")
print(f"plain power law (the one the book uses)               : eps = {np.round(T3.power.eta, 6).tolist()}, "
      f"resnorm = {T3.power.resnorm:.6e}")
print("\nDEVIATION D2 — the truncated power law's third parameter is UNIDENTIFIABLE. As eps3 -> infinity the "
      "model degenerates to the plain power law, so the objective is FLAT along eps3, and two solvers can stop "
      "at wildly different eps3 while fitting equally well. Measured on the verification fixtures, MATLAB "
      "stopped at eps3 = 766 where SciPy stopped at 9.94e5, and refitting with eps3 pinned anywhere in "
      "[766, 1e6] moves the residual sum by < 1e-6. (On the real data above, eps3 instead lands at a small, "
      "well-determined value and the truncated model fits BETTER than the plain power law — which is why the "
      "authors tried it.) The disagreement is a property of the model, not of the port, so it is reported "
      "rather than tuned away.")

fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(FIT.x, FIT.y, "ro", markersize=3, markerfacecolor="none", label="Observed data")
for fit_i, name, style in ((T3.power, "power law", "k-"),
                           (T3.truncated_power, "upper-truncated power law", "g--"),
                           (T3.weibull, "Weibull survival", "m-.")):
    ax.loglog(fit_i.x_plot, fit_i.y_plot, style, linewidth=1.8, label=name)
ax.set_xlabel("MCD [m]"); ax.set_ylabel("Cumulative Frequency"); ax.set_ylim(FIT.y.min() / 3, 2)
ax.grid(True, which="both", alpha=0.3); ax.legend(loc="lower left")
ax.set_title("three_fitting_method_and_plotting.m — the three candidate distributions (never printed in the book)")
plt.show()
''')

# =====================================================================================================================
# Appendix B
# =====================================================================================================================
md(r"""
## Appendix B — the `IceImage` data structure (pp. 221–225)

"The processed results are stored as a MATLAB data structure … presented in Appendix B" (p. 184). The appendix's
Figs. B.1–B.3 are IDE screenshots of exactly this structure, and `SeaIce_Image_Structure.m` is what builds it.
It landed in Chapter 8 because Chapter 8 is the first chapter that *consumes* it. Four blocks:

* **`Param`** (17 fields) — everything known at capture time: `NumPix_x`/`NumPix_y`, `TiltAngle`, `PanAngle`,
  `Location`, `Creator`, and eleven fields the script leaves empty.
* **`Field`** (15 fields) — the ice field as a whole: physical lengths and pixel scales, `NumFloes`/`NumBrash`,
  the four coverages plus `CovOther` (the residue between connected floes, never printed in the book), and the
  **`FSD`** cell array of `[int_min, int_max, num]` triplets.
* **`Floe(i)`** — `Center`, `Area` (in pixels), `Perimeter`, the `Polygon` sub-structure (`Vertices`, `Center`,
  `Area`, `Perimeter`, `Intersect`) and `Pixels`.
* **`Brash(i)`** — `Center`, `Area`, the `Circle` sub-structure (`Radius`, `Perimeter`, `Intersect`), `Pixels`.

The rings stored in `Polygon.Vertices` are **open**: lines 15–16 of the M-file delete the duplicated closing
vertex, because "there shall be no duplicated vertices" (p. 222). `core.icestruct` mirrors all of this as
dataclasses plus `.mat` I/O, and is `exact` against the shipped structure: all 12 numeric `Field` members, all
51 `FSD` triplets, every open ring including vertex order, and all **3992** overlap-list entries.

One field-level contradiction worth recording: the script hard-codes `Location = 'Ny-Alesund'`, `Creator = 'UAV'`,
`LengthSI_x = 50`, `LengthSI_y = 18` — the §8.2 UAV field — while the shipped `.mat` holds
`Creator = 'Helicopter'`, `PrjName = 'OATRC 2015'` and empty lengths. The `.mat` was produced by a **§8.3 variant**
of the §8.2 script (erratum **E7**), so the port makes them arguments with the script's values as defaults.
""")
code(r'''
print(f"Param : NumPix_x = {ICE.Param.NumPix_x}, NumPix_y = {ICE.Param.NumPix_y}, "
      f"TiltAngle = {ICE.Param.TiltAngle}, PanAngle = {ICE.Param.PanAngle}, "
      f"Location = {ICE.Param.Location!r}, Creator = {ICE.Param.Creator!r}, PrjName = {ICE.Param.PrjName!r}")
print(f"Field : LengthSI = {ICE.Field.LengthSI_x} x {ICE.Field.LengthSI_y} m, PixScale = "
      f"{ICE.Field.PixScale_x} x {ICE.Field.PixScale_y} m/px, CovOther = {ICE.Field.CovOther:.10f}")
f0 = ICE.Floe[0]
print(f"Floe(1): Center = {np.round(np.atleast_1d(f0.Center).ravel(), 4).tolist()}, Area = {f0.Area} px, "
      f"Perimeter = {float(np.atleast_1d(f0.Perimeter)[0]):.4f}")
print(f"         Polygon: {f0.Polygon.Vertices.shape[0]} vertices (OPEN ring: first {np.round(f0.Polygon.Vertices[0], 1).tolist()}"
      f" != last {np.round(f0.Polygon.Vertices[-1], 1).tolist()}), Area = {f0.Polygon.Area:.4f}, "
      f"Intersect.floe = {np.atleast_1d(f0.Polygon.Intersect.floe).tolist()[:8]}")
b0 = ICE.Brash[0]
print(f"Brash(1): Center = {np.round(np.atleast_1d(b0.Center).ravel(), 4).tolist()}, Area = {b0.Area} px, "
      f"Circle.Radius = {b0.Circle.Radius:.6f}, Circle.Perimeter = {b0.Circle.Perimeter:.6f} (= 2*pi*r, "
      "recomputed here rather than copied from brash(i).Perimeter)")

G = overlap_graph(ICE)
print(f"\noverlap lists: floe-floe {G['n_floe_floe']}, floe-brash {G['n_floe_brash']}, "
      f"brash-brash {G['n_brash_brash']}, brash-floe {G['n_brash_floe']} entries"
      + mcd_book("1106 / 1171 / 1171 / 544 on the shipped structure — all reproduced"))
print(f"  consistency: floe-floe symmetric {G['floe_floe_symmetric']}, brash-brash symmetric "
      f"{G['brash_brash_symmetric']}, floe-brash consistent with brash-floe {G['floe_brash_consistent']}")

# .mat round-trip: save_iceimage_mat / load_iceimage_mat are the Appendix-B I/O pair.
import tempfile
tmp = Path(tempfile.gettempdir()) / "ch08_iceimage_roundtrip.mat"
save_iceimage_mat(tmp, ICE)
BACK = load_iceimage_mat(tmp)
print(f"\n.mat round-trip through save_iceimage_mat/load_iceimage_mat: {BACK.summary()}; "
      f"Floe(1).Polygon.Area identical: {BACK.Floe[0].Polygon.Area == ICE.Floe[0].Polygon.Area}")
''')
code(r'''
# The FSD block, and the trap in it (erratum E8).  `SeaIce_Image_Structure.m` line 98 is
#     inter = fix((max_x - min_x)/nbins);  [z, n] = hist(floe_area, min_x:inter:max_x);
# so `n` are bin CENTRES; the true edges are their midpoints and both outer bins are UNBOUNDED.  Lines 100-101
# then label the triplets [int_min, int_max] = [n, [n(2:end)-1, max_x]] — labels that describe a DIFFERENT rule.
fsd = np.array(ICE.Field.FSD)
print(f"Field.FSD: {len(fsd)} triplets, first three {fsd[:3].tolist()}, counts sum to {int(fsd[:, 2].sum())} "
      f"= NumFloes ({ICE.Field.NumFloes})")

recount = np.array([int(((areas_px >= lo) & (areas_px <= hi)).sum()) for lo, hi, _ in fsd])
bad = int((recount != fsd[:, 2]).sum())
print(f"\nre-counting the floe areas from the PRINTED interval labels [int_min, int_max]:")
print(f"  first interval {fsd[0, 0]}..{fsd[0, 1]} -> {recount[0]} floes, but the stored count is {fsd[0, 2]}")
print(f"  {bad} of the {len(fsd)} intervals disagree with their own labels")
print(f"  the true first bin is everything <= {(fsd[0, 0] + fsd[1, 0]) / 2:.1f} (the midpoint of the first two "
      "centres), and it is unbounded below")
print("VERDICT (erratum E8): the shipped LABELS are wrong, the shipped COUNTS are right. Reproducing the counts "
      "requires MATLAB's hist() semantics (core.histogram.hist, bin centres), not the labels — which is exactly "
      "how the port reproduces all 51 shipped triplets at 0.")

z_hist, n_hist = matlab_hist(areas_px, matlab_colon(float(areas_px.min()),
                                                    float(np.trunc((areas_px.max() - areas_px.min()) / 50)),
                                                    float(areas_px.max())))
print(f"  hist() with those centres reproduces the stored counts: {np.array_equal(z_hist, fsd[:, 2])}")

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.bar(fsd[:, 0], fsd[:, 2], width=max(int(fsd[1, 0] - fsd[0, 0]), 1), color="steelblue", edgecolor="k",
       linewidth=0.2, label="stored Field.FSD counts")
ax.step(fsd[:, 0], recount, where="mid", color="crimson", linewidth=1.4,
        label="re-counted from the printed labels")
ax.set_xlabel("floe area [pixels] (interval minimum)"); ax.set_ylabel("number of floes"); ax.legend()
ax.set_title(f"Appendix B Field.FSD — stored counts vs the labels' own reading ({bad}/{len(fsd)} intervals differ)")
plt.show()
''')

# =====================================================================================================================
# MATLAB traps
# =====================================================================================================================
md(r"""
## The MATLAB traps this chapter turns up

Chapter 8's code is short, but it contains more "reads one way, runs another" constructs than any earlier
chapter. Each of these was confirmed by running the original file in MATLAB R2025a, and each is **reproduced
faithfully** rather than corrected — a port whose output differs from the authors' is not a better port.

| # | what the code says | what MATLAB actually does | consequence |
|---|---|---|---|
| **E9** | `[xx,yy] = polyxpoly(...); if xx ~= NaN` | `x ~= NaN` is `true` for *every* element (even for `NaN` itself), so the `if` reduces to **`if ~isempty(xx)`** | overlap means the boundaries **cross**; a floe entirely **inside** another is reported as *not* overlapping |
| **E10** | `t = 0:0.05:6.28` — "a circle" | **126** samples ending at **6.25 rad**, because 6.28 is not a multiple of 0.05 (and 6.28 < 2π anyway) | every brash "circle" in the book's results is an **open 125-gon** with a 0.033 rad gap |
| **R17** | `rgbImage(y(j),x(j),:) = color` with no `zeros(...)` before it | MATLAB **grows** the array to the largest index ever assigned | `size(rgbImage)` is `(max y, max x, 3)` over the *painted* pixels — not the image size, unless both maxima happen to be attained |
| **D4** | `c = cat(1, brash_ice(i).Center); ... c(1), c(2)` | linear indexing is **column-major**: for a 2 × 2 `Center` (4 of the shipped 3452 pieces have one) `c(1), c(2)` is `[x1, x2]`, **not** `[x1, y1]` | those four disks are centred at a point that is not any pixel's centroid — and that is what produced the shipped structure |
| **E8** | `FSD{i} = [int_min(i), int_max(i), num(i)]` | `num` comes from `hist` with **bin centres**; `int_min`/`int_max` describe edges | the shipped labels are wrong, the shipped counts are right (demonstrated above) |
| **E1** | `legend(gca,{'Observed data','Power law fitting''},'Box','on')` | the doubled quote escapes into the string: **syntax error**, the file does not parse | `fitting_iceFloes_distribution.m` as shipped cannot have drawn Fig. 8.21 |

The first four are demonstrated below. The lesson behind all of them is the same one Chapter 2 started with:
**MATLAB's semantics are the specification, not MATLAB's appearance.**
""")
code(r'''
# --- E10: the "circle" that is not closed -------------------------------------------------------------------
print(f"t = 0:0.05:6.28  ->  {CIRCLE_ANGLES.size} angles, first {CIRCLE_ANGLES[0]:g}, "
      f"last {CIRCLE_ANGLES[-1]:g} rad   (2*pi = {2 * np.pi:.6f})")
print(f"  the gap between the last sample and the first is {2 * np.pi - CIRCLE_ANGLES[-1]:.4f} rad = "
      f"{100 * (2 * np.pi - CIRCLE_ANGLES[-1]) / (2 * np.pi):.2f} % of the circumference: every modelled brash "
      "piece is an OPEN 125-gon")
arange = np.arange(0, 6.28, 0.05)
print(f"  `np.arange(0, 6.28, 0.05)` happens to land on the same {arange.size} values here (max difference "
      f"{np.abs(arange - CIRCLE_ANGLES).max():.1e}) — but it is NOT the same rule: MATLAB's colon "
      "fixes the element COUNT as fix((stop-start)/step)+1 and rebuilds the values, while np.arange accumulates "
      "and excludes the stop. Where the stop IS reached the two disagree outright:")
print(f"    matlab_colon(0, 0.1, 1) -> {matlab_colon(0.0, 0.1, 1.0).size} values ending "
      f"{matlab_colon(0.0, 0.1, 1.0)[-1]:g};   np.arange(0, 1, 0.1) -> {np.arange(0, 1, 0.1).size} values "
      f"ending {np.arange(0, 1, 0.1)[-1]:g}")
print("  core.matlab_compat.matlab_colon implements MATLAB's rule: it agrees with MATLAB on the length and the "
      "last value exactly, and to <= 1 ulp on 30 of these 126 angles (which changes the brash raster by 0 px)")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, ang, ttl in ((axes[0], CIRCLE_ANGLES, "t = 0:0.05:6.28 (the shipped code)"),
                     (axes[1], np.linspace(0, 2 * np.pi, 126), "a genuinely closed circle")):
    ax.plot(np.cos(ang), np.sin(ang), "b-", linewidth=1.2)
    ax.plot([np.cos(ang[-1]), np.cos(ang[0])], [np.sin(ang[-1]), np.sin(ang[0])], "r:", linewidth=1.2)
    ax.set_aspect("equal"); ax.set_title(ttl); ax.grid(alpha=0.3)
axes[0].annotate("the gap", xy=(1.0, -0.02), xytext=(1.15, -0.45),
                 arrowprops=dict(arrowstyle="->", color="r"), color="r")
fig.suptitle("Trap E10 — 126 points ending at 6.25 rad, not a closed circle")
plt.show()
''')
code(r'''
# --- D4: `c(1), c(2)` is a COLUMN-MAJOR read -----------------------------------------------------------------
centre_1x2 = np.array([[100.0, 300.0]])                      # the normal case: one connected component
centre_2x2 = np.array([[100.0, 300.0], [200.0, 400.0]])      # cat(1, cen.Centroid) when the label split in two
print(f"Center = {centre_1x2.tolist()}  ->  c(1), c(2) = {_matlab_c1c2(centre_1x2)}   (= x, y, as intended)")
print(f"Center = {centre_2x2.tolist()}  ->  c(1), c(2) = {_matlab_c1c2(centre_2x2)}   (= x1, x2 — NOT x1, y1)")
print("MATLAB stores matrices column by column, so element 2 of a 2x2 is the SECOND ROW of the FIRST COLUMN. "
      "4 of the 3452 shipped brash pieces have such a Center, and the shipped Circle sub-structures prove that "
      "MATLAB read them this way — so the port does too (deviation D4). Functionally inert here: every later "
      "USE of the field re-reads c(1), c(2) the same way.")

# --- R17: MATLAB grows `rgbImage` instead of pre-allocating it ------------------------------------------------
sub = ICE.Floe[:50]
sub_mcd = R.raw_mcd[:50]
grown = plot_color_bar_and_floe(COLOR_LIMIT_N, sub_mcd, ICE.Param.NumPix_y, len(sub), sub, LENGTH_OVER_PIXEL)
fixed = plot_color_bar_and_floe(COLOR_LIMIT_N, sub_mcd, ICE.Param.NumPix_y, len(sub), sub, LENGTH_OVER_PIXEL,
                                image_shape=(ICE.Param.NumPix_y, ICE.Param.NumPix_x))
print(f"\nthe image is {ICE.Param.NumPix_y} x {ICE.Param.NumPix_x} px")
print(f"  painting only the first {len(sub)} floes: MATLAB's grown rgbImage is "
      f"{grown.rgb_image.shape} — (max y, max x) over the PAINTED pixels")
print(f"  pre-allocating it instead would give {fixed.rgb_image.shape}")
print(f"  the painted content is identical where both exist: "
      f"{np.array_equal(grown.rgb_image, fixed.rgb_image[:grown.rgb_image.shape[0], :grown.rgb_image.shape[1]])}")
print("On the book's own section 8.3 field both maxima happen to be attained (the grown array is 627 x 1114 = "
      "the image size), so a full run there cannot tell 'grown' from 'pre-allocated' — which is why the "
      "verification used two deliberately smaller subsets instead, for which MATLAB returned 627x1096x3 and "
      "489x865x3. Both were reproduced byte for byte, 0 differing bytes of 2 061 576 and 1 268 955.")
''')
code(r'''
# --- E9: `if xx ~= NaN` never sees containment ----------------------------------------------------------------
def square_pixels(x0, y0, w, h):
    """1-based [x, y] of every pixel of an axis-aligned block, in MATLAB `find` (column-major) order."""
    xs, ys = np.meshgrid(np.arange(x0, x0 + w), np.arange(y0, y0 + h))
    return np.column_stack([xs.ravel(order="F"), ys.ravel(order="F")]).astype(float)

big = square_pixels(10, 10, 60, 60)
small = square_pixels(30, 30, 10, 10)                      # strictly INSIDE the convex hull of `big`
nested = [IcePiece(Center=np.array([39.5, 39.5]), Area=big.shape[0], Perimeter=236.0, PixelsPosition=big),
          IcePiece(Center=np.array([34.5, 34.5]), Area=small.shape[0], Perimeter=36.0, PixelsPosition=small)]
blank = np.zeros((90, 90))

literal = sea_ice_model(nested, [], blank)                              # what the shipped code does
strict = sea_ice_model(nested, [], blank, strict_containment=True)      # the geometrically correct answer
print(f"the small floe's hull lies entirely inside the big one's.")
print(f"  the shipped test (`if xx ~= NaN` = `if ~isempty(xx)`): floe 1 overlaps "
      f"{literal.floe[0].Intersect.floe.tolist()}, floe 2 overlaps {literal.floe[1].Intersect.floe.tolist()} "
      "-> NOTHING is reported")
print(f"  with strict_containment=True                        : floe 1 overlaps "
      f"{strict.floe[0].Intersect.floe.tolist()}, floe 2 overlaps {strict.floe[1].Intersect.floe.tolist()}")
print("The boundaries do not CROSS, so polyxpoly returns empty and the published algorithm sees no overlap. "
      "That is a genuine limitation of the method as published, not a coding slip in the port; the opt-in flag "
      "exists so the difference can be measured, and the default reproduces the book.")

fig, ax = plt.subplots(figsize=(5, 5))
for f, colr in zip(literal.floe, ("b", "g")):
    ax.plot(f.Vertices[:, 0], f.Vertices[:, 1], colr + "-", linewidth=1.5)
ax.set_aspect("equal"); ax.invert_yaxis(); ax.grid(alpha=0.3)
ax.set_title("Trap E9 — one hull inside another:\nthe boundaries never cross, so the overlap flag stays empty")
plt.show()
''')

# =====================================================================================================================
# Parameter play
# =====================================================================================================================
md(r"""
## Parameter play (optional)

`color_limit_N` is the only free parameter of §8.3's figures: it fixes both the number of colours
(`jet(color_limit_N)`) and the MCD above which every floe shares the last colour — the `≥30 [m]` of the book's
colour bar. Moving it does not change a single measured value, only what the map *shows*, which makes it a clean
demonstration of where display ends and data begins. Re-running is cheap (a fraction of a second), so this cell
is interactive. Requires `ipywidgets`; without it only the static preview runs.
""")
code(r'''
def play(color_limit_n=30):
    r = mcd_analysis(ICE, length_over_pixel=LENGTH_OVER_PIXEL, color_limit_n=int(color_limit_n))
    clamped = int((r.raw.color_index > color_limit_n).sum())
    print(f"color_limit_N = {color_limit_n}: {clamped} of {r.raw.color_index.size} floes "
          f"({100 * clamped / r.raw.color_index.size:.1f} %) share the last colour, i.e. the bar reads "
          f"'>= {r.raw.centers[int(color_limit_n) - 1]:g} m'.  Raw_MCD is unchanged: max |delta| = "
          f"{np.abs(r.raw_mcd - R.raw_mcd).max():.1f}")
    img = r.raw.rgb_image if HAVE_BOOK_MAT else T(r.raw.rgb_image)
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(img); ax.set_axis_off()
    mcd_colorbar(fig, ax, r.raw.color_m, (r.raw.centers[0], r.raw.centers[int(color_limit_n) - 1]))
    ax.set_title(f"Figure 8.19 with color_limit_N = {color_limit_n}")
    plt.show()

play()                                                    # the script's own setting
try:
    from ipywidgets import interact, IntSlider
    interact(play, color_limit_n=IntSlider(30, 5, 90, 5))
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
| `sqrt(A*4/pi)` (Eq. 8.1) | `core.stats.mean_caliper_diameter` | exact | reproduces the shipped `MCD_results.mat` at **0.0** on all 2888 values |
| `N_L(i) = size(find(MCD>=MCD(i)),2)/N_total` (Eq. 8.2) | `core.stats.cumulative_size_distribution` | exact | the M-file's literal `O(N²)` loop, ties repeated |
| `lsqcurvefit(F, p0, x, y)` | `core.fitting.lsqcurvefit` (SciPy `trf`) | near | MATLAB's own `optimoptions` defaults are transcribed; 5–10 s.f. agreement, `exitflag` identical in 4 of 4 |
| `jet(m)` | `core.plotting.matlab_jet` | exact | R2025a's `ceil(n/2) - (mod(m,4)==1)` rule; `jet(255)` matches bit for bit |
| `colormap/caxis/colorbar` of Figs. 8.19/8.20 | `core.plotting.mcd_colorbar` | display | the numbers behind it (`caxis = [1, 30]`) are exact |
| `hist(y, centres)` | `core.histogram.hist` | near/exact | bin **centres**, unbounded outer bins — `np.histogram` takes edges and would drop the tails |
| `start:step:stop` | `core.matlab_compat.matlab_colon` | near | MATLAB's element-count rule; `0:0.05:6.28` → 126 values ending 6.25 (≤ 1 ulp on 30 of them) |
| `convhull(x, y, 'simplify', true)` | `core.polygon.convhull` | near | the vertex **set** matches; MATLAB's start vertex and traversal direction are not reproducible |
| `polygeom(x, y)` | `core.polygon.polygeom` | exact | `Area` ≤ 1.8e-12 on all 2888 shipped floes |
| `roipoly`, `poly2mask` | `core.polygon.roipoly` | exact | 0 px against MATLAB's own raster |
| `polyxpoly(x1,y1,x2,y2)` | `core.polygon.polyxpoly` (and the faster `_segments_cross`) | exact | 4000/4000 randomised agreement, incl. collinear and touch cases |
| `struct` / `cat(1, …)` / `isfield`, and the whole Appendix-B layout | `core.icestruct` (+ `save_iceimage_mat` / `load_iceimage_mat`) | exact | every field of the shipped structure, incl. all 3992 overlap entries |
| `graythresh`/`im2bw`, `kmeans.m` | `core.threshold`, `core.clustering.kmeans_gray` | exact | inherited from ch03; §8.1's *results* remain `unverified` |
| the ch7 chain (`Algorithms 3/4/5`) | `ch07_ice_type.sea_ice_edge_detection` → `ice_shape_enhancement` | near/exact | verified in ch07; reused, not re-ported |
""")
md(r"""
## Summary — what is verified, what is not, and what the next chapters need

**What Chapter 8 adds.** Nothing new in the way of image processing: it *uses* Chapters 2–7 and adds
**measurement**. Five reusable primitives came out of it — `core.stats.mean_caliper_diameter` (Eq. 8.1),
`core.stats.cumulative_size_distribution` (Eq. 8.2), `core.fitting` (`lsqcurvefit` + the three model functions),
`core.icestruct` (the Appendix-B structure and its `.mat` I/O) and `core.matlab_compat.matlab_colon` — plus
`core.plotting.matlab_jet` / `mcd_colorbar`.

**What is `exact`.** The whole §8.3 chain: `main_WL_new.m` → `plot_color_bar_and_floe.m` → Eq. (8.1) →
Eq. (8.2) is **bit-exact** against MATLAB R2025a — every array at max |Δ| = **0.0**, including the
627 × 1114 × 3 painted map, byte for byte — and `Raw_MCD` also matches the authors' own shipped
`MCD_results.mat` at 0.0. Re-running `sea_ice_model.m` on their 2888 floes and 3452 brash pieces reproduces
their Appendix-B structure field by field, including all 3992 overlap-list entries as sets. `color_hist.m`,
`color_hist_comparison.m` and `SeaIce_Image_Structure.m` are `exact` on every input tested.

**What is `near`, and why.** The power-law fits: SciPy's `trf` is the same Coleman–Li family as MATLAB's
trust-region-reflective algorithm but not the same code, so α agrees to 10 significant figures rather than
exactly (1.3703585384 vs 1.3703585381 — both print as **1.3704**). `matlab_colon` is within **1 ulp** on 30 of
the 126 non-integer angles, which changes the brash raster by **0 px**. `convhull`'s vertex *order* is not
reproducible, so the contract is the vertex set plus the raster. The upper-truncated power law's third
parameter is unidentifiable (D2), so Python and MATLAB stop at different points of the same flat valley.

**What is `unverified`, and why no attempt was made to fix it.** §8.1 (open item **O1**): no code, no images;
the method is exercised on synthetic frames and *every* number the book prints for §8.1 is left alone. §8.2
(open item **O2**): our 433 floes / 290 brash against the book's 498 / 201, with the authors' parameter set
unprinted and **no parameter search performed** — a fit obtained by tuning would be worth nothing. Fig. 8.18
itself (the raw §8.3 photograph) does not ship (**O3**), so §8.3 starts from the derived structure and the
Fig. 8.19 comparison uses a reconstruction from `Floe.Pixels`, labelled as such.

**Errata confirmed in MATLAB** (full list in `reports/ch08_verification.md`): E1 the unrunnable
`fitting_iceFloes_distribution.m`; E2 the text describing a log-log slope its code never computes; E3 Fig. 8.15's
colour bar belonging to the *other* file; E4 the truncated 15.94 %; E5 "clipper" for caliper; E6 the
"not smaller" claim that is false in both readings; E7 the `.mat` contradicting its own script; E8 the FSD's
wrong labels and right counts; E9 `if xx ~= NaN`; E10 the open 125-gon; E11 Fig. 8.15's subtraction stated
backwards.

**Feeds forward.**
*Chapter 9 (model-ice applications)* works on tank ice — rectangular floes, concentration from video, maximum
floe size monitoring — and will reuse `core.matlab_compat.matlab_colon`, `core.stats`, `core.icestruct` and the
whole ch6/ch7 identification chain, on synthetic and model-basin data rather than aerial photographs.
*Chapter 10 (Appendix A, geometric calibration)* owns the rectifier this chapter deliberately did **not**
write: the analytic orthorectification of A.1.1 and the linear four-corner method of A.1.2 that §8.1's shipborne
frames and §7.3.1.2's oblique images both need. When it lands, §8.1's procedure becomes runnable end to end on
any oblique frame a reader supplies.

The learned knowledge for this chapter is in `knowledge/ch08.md` and `knowledge/CUMULATIVE.md`; the evidence is
in `reports/ch08_verification.md`.
""")

out = HERE / NB_NAME
nbf.write(nb, out)
print(f"wrote {out} ({len(nb.cells)} cells)")
