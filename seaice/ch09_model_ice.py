"""Chapter 9 — Model sea ice image processing applications (ice-basin / model-basin imagery).

Book: Chapter 9, printed pages 195–212.  §9.1 experimental setup and model sea ice image data (Table 9.1);
§9.2 ice concentration — §9.2.1 from the overall tank image (Figs. 9.1–9.5, Table 9.2), §9.2.2 from model sea
ice video (Figs. 9.6–9.10, Table 9.3); §9.3 ice floe identification — §9.3.1 contour initialization for crowded
rectangular floes (Fig. 9.11, **Algorithm 7** p. 206), §9.3.2 the overall tank image (Figs. 9.12–9.14,
Table 9.4), §9.3.2.1 model sea ice floe modeling (Figs. 9.15–9.16), §9.3.3 video: monitoring maximum floe size
(Figs. 9.17–9.18).

**Chapter 9 contains no numbered equations.**  Its formal content is Algorithm 7, the three re-segmentation
criteria of p. 205 and four tables; every formula it uses comes from Ch. 3 (Otsu Eqs. 3.3–3.22, k-means
Eqs. 3.35–3.37), Ch. 1 (``IC = #ice/(MN)``) and Ch. 6 (the GVF-snake system, Eqs. 6.34–6.55).

MATLAB sources (``MATLAB_ROOT/ch9/``) — 21 ``.m`` files, **14 of them byte-identical to their ch6/ch7 namesakes**
and therefore *reused*, not re-ported (project rule 9):

======================================================= ==========================================================
``block_threshold.m``                                   :func:`block_threshold` → ``core.threshold.block_otsu(..., compare='ge')``
``movie_otsu.m``                                        :func:`movie_otsu`
``movie_kmeans.m``                                      :func:`movie_kmeans`
``Model_Ice_Floe_Identification/model_ice_demo.m``      :func:`model_ice_demo` (driver ``scripts/ch09_model_ice_demo.py``)
``Model_Ice_Floe_Identification/rect.m``                :func:`rect`
``Model_Ice_Floe_Identification/model_ice_model.m``     :func:`model_ice_model`
``Model_Ice_Floe_Identification/movie_floe.m``          :func:`movie_floe`
``…/GVF_distance.m`` (md5 ``39cca98a…``)                :func:`seaice.ch06_gvf_snake.gvf_distance` (reused)
``…/minboundrect.m`` (md5 ``3e179fb6…``)                :func:`seaice.core.polygon.minboundrect` (reused)
``…/{GVF, gradient2, xconv2, gaussianMask, gaussianBlur, BoundMirror{Expand,Ensure,Shrink}, snakedeform,
snakeinterp, snakeindex, snakedisp}.m``                 :mod:`seaice.core.snake` / ``core.plotting`` (reused)
======================================================= ==========================================================

Errata reproduced literally (the corrected form is always an opt-in flag)
-------------------------------------------------------------------------
* **E4** ``movie_otsu.m`` line 39 — ``if I(k).cdata(i,j) >= t*255`` compares the pixel with the **whole growing
  vector** ``t``, so ``if`` means ``all(...)`` and the effective threshold is the **running maximum**
  ``max_{j<=k} t(j)``.  Confirmed in R2025a: ``if uint8(50) >= [25 51]`` is **not** taken.
  Flag: ``running_max_bug=False``.
* **E5** ``movie_otsu.m`` line 40 — the body is ``n = n+1``, not ``n(k) = n(k)+1``, so **every** element of the
  growing vector ``n`` is incremented.  R2025a probe: three frames with counts 7, 6, 4 end at ``n = [17 10 4]``
  (``n_final[j] = Σ_{m>=j} count_m``).  ``IC(k)`` is read *inside* the same iteration, so ``IC`` escapes E5 but
  **not** E4.  Flag: ``increment_all_bug=False``.
* **E6** ``movie_kmeans.m`` line 79 — the input is run **5100** and the output AVI is named ``05400_kmeans.avi``.
* **E7** §9.3.2.1 p. 207 says rectangles with a length-to-width ratio *less than* a threshold are removed; the
  code (``model_ice_demo.m`` line 65 + ``model_ice_model.m`` line 34) removes ratios **outside** ``(k1, k2)``
  and ``k`` is not normalised to ``>= 1``, so ``(0.4, 2.5)`` is a **symmetric band**.
* **E8** ``rect.m`` line 32 — ``if (nargin<3)`` in a **2-argument** function is always true, so ``metric`` is
  dead code and only ``'a'`` is reachable (line 52 hard-codes ``'a'`` anyway).
* **E9 (ch08's)** ``model_ice_model.m`` line 54 ``if xx ~= NaN`` is ``if ~isempty(xx)`` — a NaN comparison is
  false for every element and ``if`` on an empty array is false.  Here the consequence is the **opposite** of
  ch08's: ``polybool`` returns the intersection *region*, so containment **is** detected.  Flag:
  ``strict_containment=True``.
* **R13 (correction to `analysis/ch09.md`)** ``movie_floe.m`` line 25 ``floe(k) = max(ice_areas)`` on a blank
  frame assigns ``[]`` at index ``k`` where ``numel(floe) == k-1``.  The analysis predicted MATLAB would *delete*
  element ``k``; an R2025a probe shows it **errors** — ``MATLAB:matrix:singleSubscriptNumelMismatch``, "Unable to
  perform assignment because the left and right sides have a different number of elements" — because a null
  assignment past the end of the vector is not a deletion.  Deletion only happens for ``k <= numel(floe)``, which
  this loop never reaches.  :func:`movie_floe`'s default ``empty='raise'`` therefore *is* the literal behaviour;
  ``empty='delete'`` offers the deletion semantics for the reachable case.

Gaps in the shipped code
------------------------
* **G1** — Algorithm 7's convergence test (lines 2/6/7/19, "stop if the total number of floes after steps N and
  N+1 is equal") is **not implemented**: ``GVF_distance.m`` line 80 computes ``num`` and never compares it.
  Opt-in as :func:`seaice.ch06_gvf_snake.gvf_distance` ``(stop='count')``.
* **G2** — §9.3.3's *per-frame Algorithm-7 segmentation* has no shipped code at all; ``movie_floe.m`` consumes an
  AVI that already holds segmented (binary) frames.  :func:`segment_video` wires it, and is **ours, not the
  book's** — it must never appear in a parity claim.
* **Text-vs-code** — p. 205 criterion 3 is the minimum-area bounding **rectangle** ratio; ``GVF_distance.m``
  computes the **ellipse** axis ratio.  ch06's verified default is unchanged (``ratio='ellipse'``); the book's
  variant is :func:`seaice.ch06_gvf_snake.component_criteria` ``(ratio='minrect')``.

Data
----
Only ``Model_Ice_Floe_Identification/model_ice.jpg`` (181 × 76 × 3 uint8) is shipped.  ``04100_analyse.jpg``,
``dypic_05100_cam1_top.avi`` and ``05100.avi`` are HSVA/DYPIC campaign assets that were never published (risk R1);
the Tier-3 stand-ins are :func:`seaice.core.synth.model_ice_tank`,
:func:`seaice.core.synth.model_ice_tank_video` and :func:`seaice.core.synth.segmented_floe_video`.  **Every book
number of §9.2 and §9.3.3 (N2, N3, N5, N6, N9–N12, N19) is permanently `unverified`** — the scripts say so and
skip; nothing is invented.

Licence note for the authors' own files
---------------------------------------
``GVF_distance.m``, ``model_ice_demo.m``, ``rect.m`` and ``model_ice_model.m`` are by **Qin Zhang**, NTNU
Department of Marine Technology, project "Arctic DP" (RCN no. 199567).  Their headers and the shipped
``README.docx`` state that the code is *"only available for academic non-commercial use"* and ask for a citation
of Q. Zhang, R. Skjetne, I. Metrikin and S. Løset, "Image Processing for Ice Floe Analyses in Broken-ice Model
Testing", *Cold Regions Science and Technology* **111**:27–38, 2015.  The GVF snake toolbox is by **C. Xu and
J. L. Prince** (http://www.iacl.ece.jhu.edu/static/gvf/, IEEE TIP 7(3):359–369, 1998) and ``minboundrect.m`` by
**John D'Errico** (MATLAB File Exchange).  The functions below are a re-implementation written for study of the
book; the original M-code is not redistributed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .ch06_gvf_snake import gvf_distance
from .core.clustering import kmeans_gray, kmeans_lloyd
from .core.connectivity import bwareaopen, label_components
from .core.histogram import hist
from .core.matlab_compat import im2double, rgb2gray_matlab, to_uint8_saturating
from .core.morphology import imfill
from .core.polygon import clip_polygon_convex, minboundrect, roipoly
from .core.regionprops import regionprops
from .core.threshold import BlockOtsu, block_otsu, graythresh, ice_concentration, im2bw

__all__ = [
    "BOOK_PARAMS_CH9", "CROP_5100", "BOX_5100", "IC_DENOMINATOR_5100",
    "TABLE_9_1", "TABLE_9_2", "TABLE_9_3", "TABLE_9_4", "FIG_9_4_IC", "FIG_9_4_THRESH",
    "RectFloe", "ModelFloe", "ModelIceModel", "MovieOtsu", "MovieKmeans", "MovieFloe", "ModelIceDemo",
    "block_threshold", "movie_otsu", "movie_kmeans", "movie_floe", "rect", "model_ice_model",
    "model_ice_demo", "tank_ice_concentration", "rect_ice_concentration", "fsd_error",
    "tiled_segmentation", "segment_video", "preprocess_frame", "floe_size_histogram",
    "ensure_synthetic_tank", "ensure_synthetic_tank_video", "ensure_synthetic_segmented_video",
]

# ===============================================================================================================
# The chapter's constants — every one of them transcribed from a printed table or from an .m literal
# ===============================================================================================================

#: ``model_ice_demo.m`` lines 11–38, value by value.  Compare with ``ch06_gvf_snake.BOOK_PARAMS['sea_ice_demo']``:
#: ``Ra_min`` 20 (was 10), ``Ra`` 1000 (2500), ``Num`` 150 (500), ``iter`` 150 (100), ``kappa`` 0.6 (0.5),
#: ``timer`` **2** (1); ``Rc`` 0.9, ``Rl`` 2 and ``se = strel('disk', 3)`` unchanged; and **no k-means stage**.
BOOK_PARAMS_CH9: dict[str, Any] = dict(
    Ra_min=20,        # line 11  minimum floe area
    Ra=1000,          # line 12  maximum floe area
    Rc=0.9,           # line 13  convexity threshold
    Rl=2,             # line 14  threshold ratio between length and width
    timer=2,          # line 15  outer re-segmentation loop count
    se_radius=3,      # line 17  se = strel('disk', 3)
    Num=150,          # line 21  number of GVF iterations
    iter=150,         # line 22  number of snake iterations
    sigma=0,          # line 27  gaussianBlur (0 => gaussianBlur/xconv2 are never executed)
    GradientOn=1,     # line 28
    GVFOn=1,          # line 30
    mu=0.1,           # line 31
    alpha=0.05,       # line 33
    beta=0.0,         # line 34
    gamma=1.0,        # line 35
    kappa=0.6,        # line 36
    Dmin=0.0,         # line 37
    Dmax=1.0,         # line 38
    k1=0.4,           # line 65 (commented block) — model_ice_model's minimum L/W ratio
    k2=2.5,           # line 65 (commented block) — model_ice_model's maximum L/W ratio
)

#: ``movie_otsu.m`` lines 19–20 / ``movie_kmeans.m`` lines 19–20 — the keep window, 1-based inclusive MATLAB
#: indices ``(y1, y2, x1, x2)`` = ``(180, 400, 125, 521)`` → a **221 × 397** crop.
CROP_5100: tuple[int, int, int, int] = (180, 400, 125, 521)

#: ``movie_otsu.m`` line 22 — the vessel box blanked to black *before* the crop, ``(y3, y2, x3, x4)``
#: = ``(307, 400, 268, 380)`` → **94 × 113 = 10 622 px** (Fig. 9.6(b)).
BOX_5100: tuple[int, int, int, int] = (307, 400, 268, 380)

#: ``movie_otsu.m`` line 46 / ``movie_kmeans.m`` line 55 — ``r*c - r2*c2`` = 221·397 − 94·113 = **77 115**
#: (book number N8, pure arithmetic on the scripts' literals).
IC_DENOMINATOR_5100: int = (400 - 180 + 1) * (521 - 125 + 1) - (400 - 307 + 1) * (380 - 268 + 1)

#: Table 9.1 p. 196 — target ice conditions per run: ``(target IC, floe edge lengths in m, their shares)``.
#: See risk **R8**: "one 1.50 m strip, four 1.00 m and nine 0.50 m coincide with the percentages" gives
#: 26.5/47.1/26.5 % by **area**, 7/29/64 % by **count** and 15/40/45 % by **strip width** — none of which is
#: 45/40/15 % in the printed order.  All three readings are recorded; none is silently chosen.
TABLE_9_1: dict[int, dict[str, Any]] = {
    5100: dict(target_ic=0.86, sizes_m=(0.50, 1.00, 1.50), shares=(0.45, 0.40, 0.15)),
    5200: dict(target_ic=0.70, sizes_m=(0.50, 1.00, 1.50), shares=(0.45, 0.40, 0.15)),
    5300: dict(target_ic=0.70, sizes_m=(0.25, 0.50, 0.75), shares=(0.45, 0.40, 0.15)),
    5400: dict(target_ic=0.86, sizes_m=(0.25, 0.50, 0.75), shares=(0.45, 0.40, 0.15)),
}

#: Table 9.2 p. 199 — ice concentration by method, in percent (**book numbers, not computable: the image
#: `04100_analyse.jpg` is missing**; risk R1, targets N5/N6).
TABLE_9_2: dict[int, dict[str, float]] = {
    5100: dict(target=86.00, global_otsu=83.17, local_otsu=83.14, kmeans=82.86),
    5200: dict(target=70.00, global_otsu=62.50, local_otsu=62.51, kmeans=62.00),
}

#: Table 9.3 p. 203 — saturation start time (s) and average IC (%) after it, per run (**book numbers, N11**).
TABLE_9_3: dict[int, tuple[int, float]] = {5100: (200, 88.93), 5200: (300, 80.39),
                                           5300: (600, 81.69), 5400: (300, 84.83)}

#: Table 9.4 p. 207 — the GVF iteration count ``Num`` for each of the **20 overlapping sub-images** of the
#: overall tank image (book number N13; these are *inputs*, so a port can only assert the transcription).
TABLE_9_4: tuple[int, ...] = (150, 65, 65, 65, 160, 60, 110, 130, 90, 130,
                              170, 160, 100, 90, 90, 100, 90, 80, 90, 80)

#: Fig. 9.4 p. 198 — the six printed block ice concentrations (%) in reading order (book number N3).
FIG_9_4_IC: tuple[float, ...] = (83.93, 84.57, 82.65, 80.69, 82.97, 84.07)

#: Fig. 9.4 p. 198 — the six printed block thresholds (book number N3).
FIG_9_4_THRESH: tuple[int, ...] = (83, 82, 85, 81, 83, 91)


# ===============================================================================================================
# §9.2.1 — ice concentration from the overall tank image
# ===============================================================================================================

def block_threshold(I: np.ndarray, n_r: int = 2, n_c: int = 3) -> BlockOtsu:
    """``block_threshold.m`` — 2 × 3 local Otsu on the overall tank image (Book §9.2.1, Fig. 9.4, p. 198).

    MATLAB source: ``MATLAB_ROOT/ch9/block_threshold.m`` lines 2–45.  A thin wrapper over
    :func:`seaice.core.threshold.block_otsu` with ``compare='ge'`` — the *one numeric* difference from ch3's
    ``local_Otsu.m`` (line 26 ``>=`` instead of line 27 ``>``; see that function's docstring for the full
    five-hunk diff).  The script also drops ch3's ``num`` bookkeeping and its overall
    ``IC = sum(num(:))/(r*c)``, so the Fig. 9.4 caption's "average IC" is the **unweighted mean of the six
    block values** — :attr:`~seaice.core.threshold.BlockOtsu.ic_mean`, proved by book number N4:
    mean(83.93, 84.57, 82.65, 80.69, 82.97, 84.07) = 498.88/6 = 83.1467 % → the printed "83.14 %".

    Parameters
    ----------
    I : ndarray
        The tank image; RGB is converted with ``rgb2gray`` (line 3) exactly as the script does.
    n_r, n_c : int
        The script's ``n_r = 2``, ``n_c = 3`` (lines 7–8).

    Returns :class:`~seaice.core.threshold.BlockOtsu`.  Parity: exact.
    """
    gray = rgb2gray_matlab(I) if np.asarray(I).ndim == 3 else np.asarray(I)
    return block_otsu(gray, n_r, n_c, compare="ge")


def tank_ice_concentration(I: np.ndarray, method: str = "otsu", *, k: int = 2, impl: str = "authors",
                           seed: int = 0) -> tuple[float, np.ndarray, float]:
    """Ice concentration of the overall tank image by global Otsu or by k-means — Book §9.2.1, Figs. 9.3/9.5.

    **Text only**: the book prints Fig. 9.3 (global Otsu, IC = 83.17 %, threshold = 84) and Fig. 9.5 (k-means,
    k = 2, IC = 82.86 %) but ships no ``.m`` for either; ``block_threshold.m`` covers the local-Otsu column of
    Table 9.2 alone.  Both are the Chapter-3 routines applied unchanged (p. 196), so this is a two-line
    composition of verified primitives.

    Parameters
    ----------
    method : {'otsu', 'kmeans'}
    k : int
        Number of clusters for ``method='kmeans'`` (the book uses 2).
    impl : {'authors', 'lloyd++'}
        ``'authors'`` (**default**) = :func:`seaice.core.clustering.kmeans_gray`, the book's own deterministic
        ``ch3/kmeans.m`` — the ch08-review rule: where the *text* names "the k-means method" with no ``.m``
        behind it, use the authors' routine, not a seeded Lloyd stand-in.  ``'lloyd++'`` is opt-in and
        seed-dependent (`approx`).

    Returns ``(ic, mask, threshold_or_nan)`` — ``ic`` a fraction in [0, 1], ``threshold`` = ``255 * level`` for
    Otsu and NaN for k-means.  Parity: exact as code; the **book's numbers stay `unverified`** (no image).
    """
    gray = rgb2gray_matlab(I) if np.asarray(I).ndim == 3 else np.asarray(I)
    if method == "otsu":
        level, _ = graythresh(gray)
        mask = im2bw(gray, level)
        return ice_concentration(mask), mask, float(level * 255.0)
    if method != "kmeans":
        raise ValueError("method must be 'otsu' or 'kmeans'")
    if impl == "authors":
        res = kmeans_gray(gray, k)                     # the authors' own ch3/kmeans.m (deterministic, no seed)
        mask = res.mask == (int(np.argmax(res.centroids)) + 1)   # the brightest cluster is ice
    elif impl == "lloyd++":
        X = gray.astype(np.float64).ravel(order="F")[:, None]
        res = kmeans_lloyd(X, k, init="kmeans++", seed=seed)
        bright = int(np.argmax(res.centers.ravel()))
        mask = (res.labels == bright).reshape(gray.shape, order="F")
    else:
        raise ValueError("impl must be 'authors' (ch3/kmeans.m) or 'lloyd++' (the toolbox stand-in)")
    return ice_concentration(mask), mask, float("nan")


# ===============================================================================================================
# §9.2.2 — ice concentration from model sea ice video
# ===============================================================================================================

def preprocess_frame(frame: np.ndarray, crop: tuple[int, int, int, int] = CROP_5100,
                     box: tuple[int, int, int, int] = BOX_5100) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """``movie_otsu.m`` / ``movie_kmeans.m`` lines 19–30 — blank the vessel, then crop.

    ::

        I1(k).cadata = mov(k).cdata(y1:y2, x1:x2, :);   % stored, NEVER used again
        I2(k).cadata = mov(k).cdata(y3:y2, x3:x4, :);   % stored, NEVER used again
        r2 = y2-y3+1;  c2 = x4-x3+1;
        mov(k).cdata(y3:y2, x3:x4, :) = 0;             % blank the vessel (Fig. 9.6(b))
        mov(k).cdata = mov(k).cdata(y1:y2, x1:x2, :);  % crop

    The blanking happens **before** the crop and **before** ``rgb2gray``, so the Otsu histogram / the k-means
    data contain ``r2*c2`` forced-black pixels — the book admits this on p. 200 and compensates by removing them
    from the IC denominator (but not from the numerator).

    Returns ``(cropped, I1, I2, r2, c2)``; ``I1``/``I2`` are the two arrays the scripts store and never read,
    kept so a reference comparison can check them.  Parity: exact.
    """
    y1, y2, x1, x2 = crop
    y3, _y2b, x3, x4 = box
    f = np.asarray(frame)
    I1 = f[y1 - 1:y2, x1 - 1:x2, ...].copy()
    I2 = f[y3 - 1:y2, x3 - 1:x4, ...].copy()
    r2, c2 = y2 - y3 + 1, x4 - x3 + 1
    g = f.copy()
    g[y3 - 1:y2, x3 - 1:x4, ...] = 0
    return g[y1 - 1:y2, x1 - 1:x2, ...], I1, I2, r2, c2


@dataclass
class MovieOtsu:
    """Every array ``movie_otsu.m`` leaves in the workspace.

    Attributes
    ----------
    t : (N,) ``graythresh`` level per frame (line 33)
    n : (N,) the script's ``n`` **after** erratum E5 corrupted it (``n_final[j] = Σ_{m>=j} count_m``)
    counts : (N,) the per-frame count actually read as ``n(k)`` inside the loop (E5-free)
    IC : (N,) ``n(k) / (r*c - r2*c2)`` — the quantity plotted in Figs. 9.8/9.10
    effective_level : (N,) the level the count really used: the **running maximum** with E4, ``t(k)`` without
    bw : (N, r, c) bool, ``im2bw(I(k).cdata, t(k))`` — strict ``>`` at the *per-frame* level, so the written AVI
        and the plotted IC legitimately disagree (different rule **and** different value)
    gray : (N, r, c) uint8 cropped gray frames
    denominator : ``r*c - r2*c2``
    """

    t: np.ndarray
    n: np.ndarray
    counts: np.ndarray
    IC: np.ndarray
    effective_level: np.ndarray
    bw: np.ndarray
    gray: np.ndarray
    denominator: int
    running_max_bug: bool = True
    increment_all_bug: bool = True
    count_rule: str = "ge"


def movie_otsu(frames: np.ndarray, *, crop: tuple[int, int, int, int] = CROP_5100,
               box: tuple[int, int, int, int] = BOX_5100, running_max_bug: bool = True,
               increment_all_bug: bool = True, count_rule: str = "ge",
               keep_frames: bool = True) -> MovieOtsu:
    """``movie_otsu.m`` — per-frame global Otsu ice concentration of a tank video (Book §9.2.2, Figs. 9.6–9.8, 9.10).

    MATLAB source: ``MATLAB_ROOT/ch9/movie_otsu.m`` lines 16–48 (the per-frame loop) and 50–53 (``plot(k, IC(k))``
    with ``xlabel('Time')`` — the x axis is the **frame index**, called "Time" because the video is decimated to
    1 fps, p. 199).  Lines 59–65 replay ``bw(k).cadata`` and write ``otsu.avi`` at 12 fps
    (``core.video.write_video``; ``movie2avi`` is removed in R2025a).

    The exact plotted quantity, literally::

        IC(k) = #{(i,j) : gray_k(i,j) >= 255 * max_{j<=k} t(j)} / (221*397 - 94*113)

    Parameters
    ----------
    frames : ndarray
        ``(H, W, 3, N)`` (MATLAB's ``read(VideoReader)`` order, :func:`seaice.core.video.read_video`) or
        ``(N, H, W, 3)``.
    running_max_bug : bool
        ``True`` (**default = the shipped code**) reproduces erratum **E4**; ``False`` uses the per-frame
        ``t(k)``, which is what Figs. 9.8/9.10 claim to show.
    increment_all_bug : bool
        ``True`` (**default = the shipped code**) reproduces erratum **E5** in the returned ``n``.  ``IC`` is
        unaffected either way (``n(k)`` is read inside the same iteration).
    count_rule : {'ge', 'gt'}
        ``'ge'`` = line 39's ``>=``.  ``'gt'`` makes the count agree with ``im2bw``'s strict ``>``.
    keep_frames : bool
        Keep the per-frame ``gray``/``bw`` stacks (memory).

    Returns :class:`MovieOtsu`.  Parity: **exact** on an Uncompressed AVI (risk R2).
    """
    if count_rule not in ("ge", "gt"):
        raise ValueError("count_rule must be 'ge' (movie_otsu.m line 39) or 'gt' (im2bw's rule)")
    stack = _as_frame_list(frames)
    N = len(stack)
    t = np.zeros(N)
    counts = np.zeros(N, dtype=np.int64)
    effective = np.zeros(N)
    grays: list[np.ndarray] = []
    bws: list[np.ndarray] = []
    r = c = r2 = c2 = 0
    for k in range(N):
        cropped, _I1, _I2, r2, c2 = preprocess_frame(stack[k], crop, box)
        gray = rgb2gray_matlab(cropped) if cropped.ndim == 3 else cropped
        r, c = gray.shape
        level, _ = graythresh(gray)
        t[k] = level
        # E4: `if I(i,j) >= t*255` with `t` the whole 1..k vector -> `all()` -> the running maximum.
        lv = float(np.max(t[: k + 1])) if running_max_bug else float(level)
        effective[k] = lv
        th = lv * 255.0
        v = gray.astype(np.float64)
        counts[k] = int(np.count_nonzero(v >= th if count_rule == "ge" else v > th))
        if keep_frames:
            grays.append(gray)
            bws.append(im2bw(gray, level))          # line 45 — the *per-frame* level, strict `>`
    # E5: `n = n+1` increments every element, so the saved vector is the reverse cumulative sum.
    n = np.cumsum(counts[::-1])[::-1].copy() if increment_all_bug else counts.copy()
    denom = r * c - r2 * c2
    IC = counts.astype(np.float64) / denom if denom else np.full(N, np.nan)
    return MovieOtsu(t=t, n=n, counts=counts, IC=IC, effective_level=effective,
                     bw=np.array(bws) if bws else np.zeros((0, 0, 0), bool),
                     gray=np.array(grays) if grays else np.zeros((0, 0, 0), np.uint8),
                     denominator=int(denom), running_max_bug=running_max_bug,
                     increment_all_bug=increment_all_bug, count_rule=count_rule)


@dataclass
class MovieKmeans:
    """Every array ``movie_kmeans.m`` leaves in the workspace.

    Attributes
    ----------
    map1 : (N, M*N_px) 1-based cluster index per pixel, **column-major** like MATLAB's ``(:)``
    s : (N, k) per-frame cluster mean intensities (line 48)
    a : (N,) 1-based index of the **brighter** cluster = ice (line 51)
    IC : (N,) ``length(p) / (si(1)*si(2) - r2*c2)``
    out : (N, r, c) uint8 0/255 masks (line 58 ``reshape(mask1, si(1), si(2))``, column-major)
    """

    map1: np.ndarray
    s: np.ndarray
    a: np.ndarray
    IC: np.ndarray
    out: np.ndarray
    denominator: int
    impl: str = "lloyd++"
    seed: int = 0


def movie_kmeans(frames: np.ndarray, *, k: int = 2, crop: tuple[int, int, int, int] = CROP_5100,
                 box: tuple[int, int, int, int] = BOX_5100, impl: str = "lloyd++", seed: int = 0,
                 keep_maps: bool = True) -> MovieKmeans:
    """``movie_kmeans.m`` — per-frame k-means (k = 2) ice concentration of a tank video (§9.2.2, Figs. 9.7(c)/9.8).

    MATLAB source: ``MATLAB_ROOT/ch9/movie_kmeans.m`` lines 16–61, then ``plot(k, IC(k))`` and
    ``movie2avi(M, '05400_kmeans.avi', 'FPS', 12)`` (erratum **E6**: the input is run 5100).  Per frame::

        mov = rgb2gray(cropped);  im = im2double(mov);  ima = double(im);  si = size(ima);
        ima = double(im(:));                                    % column-major, 87 737 x 1
        map1 = kmeans(ima, 2, 'EmptyAction', 'singleton');       % Statistics Toolbox
        s(i) = sum(ima .* (map1 == i)) / sum(map1 == i);         % cluster mean intensity
        a = find(s == max(s));  p = find(map1 == a);  mask(p) = 1;
        IC = length(p) / (si(1)*si(2) - r2*c2);
        out = reshape(uint8(mask*255), si(1), si(2));

    ``s`` persists between frames in MATLAB but ``kms = 2`` is constant, so both entries are always overwritten
    — unlike ch3's ``Otsu.m``/``kmeans.m`` there is **no stale-buffer bug** here.  A two-way tie in ``max(s)``
    would make ``find`` return two elements and MATLAB would error on the scalar assignment ``a(k) = …``; this
    port raises the same way.

    # DEVIATION: `approx` — ch9 ships **no** ``kmeans.m``, so line 44 resolves to the **Statistics Toolbox**
    # routine (k-means++ start, RNG-dependent), not to the authors' deterministic ``ch3/kmeans.m``.  The stand-in
    # is :func:`seaice.core.clustering.kmeans_lloyd(X, 2, init='kmeans++', seed=seed)`; compare sorted cluster
    # centres and pixel agreement (>= 99 %), never label numbers.  Note ``ima`` is ``im2double`` output, i.e.
    # non-integer, which ``kmeans_gray`` (a histogram method) cannot take even if it applied.
    # Everything *around* the clustering — crop, blanking, ``si``, the denominator, the column-major
    # ``reshape`` — is **exact** and is asserted separately.

    Returns :class:`MovieKmeans`.
    """
    stack = _as_frame_list(frames)
    N = len(stack)
    maps: list[np.ndarray] = []
    s_all = np.zeros((N, k))
    a = np.zeros(N, dtype=np.int64)
    IC = np.zeros(N)
    outs: list[np.ndarray] = []
    denom = 0
    for idx in range(N):
        cropped, _I1, _I2, r2, c2 = preprocess_frame(stack[idx], crop, box)
        gray = rgb2gray_matlab(cropped) if cropped.ndim == 3 else cropped
        im = im2double(gray)                         # line 34
        si = im.shape                                # line 40 — taken BEFORE flattening
        ima = im.ravel(order="F").astype(np.float64)  # line 42 — MATLAB's (:) is column-major
        if impl == "lloyd++":
            res = kmeans_lloyd(ima[:, None], k, init="kmeans++", seed=seed)
            map1 = res.labels + 1                    # MATLAB's labels are 1-based
        elif impl == "sklearn":                       # pragma: no cover - optional cross-check
            from sklearn.cluster import KMeans

            map1 = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(ima[:, None]) + 1
        else:
            raise ValueError("impl must be 'lloyd++' or 'sklearn'")
        sv = np.array([ima[map1 == i].sum() / np.count_nonzero(map1 == i)
                       if np.any(map1 == i) else np.nan for i in range(1, k + 1)])
        s_all[idx] = sv
        winners = np.nonzero(sv == np.nanmax(sv))[0]
        if winners.size != 1:
            raise ValueError(f"movie_kmeans: frame {idx + 1} has {winners.size} clusters tied at max(s); "
                             "MATLAB errors on the scalar assignment `a(k) = find(s == max(s))`")
        a[idx] = winners[0] + 1
        p = np.nonzero(map1 == a[idx])[0]
        denom = si[0] * si[1] - r2 * c2
        IC[idx] = p.size / denom if denom else np.nan
        mask = np.zeros(ima.size)
        mask[p] = 1.0
        mask1 = to_uint8_saturating(mask * 255.0)     # line 57 — uint8() cast, not im2uint8
        outs.append(mask1.reshape(si, order="F"))     # line 58 — column-major reshape
        if keep_maps:
            maps.append(map1)
    return MovieKmeans(map1=np.array(maps) if maps else np.zeros((0, 0), np.int64), s=s_all, a=a, IC=IC,
                       out=np.array(outs), denominator=int(denom), impl=impl, seed=seed)


# ===============================================================================================================
# §9.3.2.1 — rectangularization and the model-ice floe model
# ===============================================================================================================

@dataclass
class RectFloe:
    """One rectangularized model floe — the ``struct('Vertices','Center','Area','Perimeter')`` of ``rect.m``.

    **``Area`` and ``Perimeter`` are the RECTANGLE's**, from ``minboundrect``, *not* the component's pixel count
    — a different meaning from ch7's ``IcePiece.Area`` and ch8's ``Floe.Area``.  Do not mix them in a
    floe-size distribution without saying which one is plotted.

    Attributes
    ----------
    Vertices : (5, 2) float — ``[rectx, recty]``, a **closed** ring (``minboundrect`` returns 5 points),
        column (x) first, row (y) second
    Center : (2,) float — ``[mean(rectx(1:4)), mean(recty(1:4))]``
    Area : float — the rectangle's area
    Perimeter : float — the rectangle's perimeter
    """

    Vertices: np.ndarray
    Center: np.ndarray
    Area: float
    Perimeter: float


def rect(img: np.ndarray, metric: str = "a", *, conn: int = 4) -> list[RectFloe]:
    """``rect.m`` — the minimum-area bounding rectangle of every object (Book §9.3.2.1, Fig. 9.15(a), p. 207).

    MATLAB source: ``MATLAB_ROOT/ch9/Model_Ice_Floe_Identification/rect.m`` lines 48–61::

        [label, num] = bwlabel(img, 4);
        for i = 1 : num
            [r, c] = find(label == i);                          % r = ROWS, c = COLUMNS, 1-based
            [rectx, recty, a, p] = minboundrect(c, r, 'a');     % x = COLUMN, y = ROW  <-- the swap
            cenx = sum(rectx(1:4))/4;  ceny = sum(recty(1:4))/4;
            c = [cenx, ceny];                                   % `c` reused: was the column vector
            v = [rectx, recty];
            s0 = struct('Vertices', v, 'Center', c, 'Area', a, 'Perimeter', p);

    Two things must not be "cleaned up": the ``(c, r)`` argument order (a transposition here silently swaps
    every rectangle's length and width — a *square* fixture cannot catch it, ch08 review lesson), and the reuse
    of the name ``c`` for the centre.

    Erratum **E8**: ``metric`` is **dead code** — ``if (nargin<3)`` in a two-argument function is always true,
    so ``metric`` is forced to ``'a'`` and the ``strmatch`` validation branch is unreachable; line 52 hard-codes
    ``'a'`` regardless.  The parameter is kept here for signature fidelity, is validated the way the unreachable
    branch would have validated it, and is then **ignored** exactly as MATLAB ignores it.

    Parameters
    ----------
    img : ndarray
        Binary (or nonzero-is-object) image — ``model_ice_demo.m`` passes ``bw4``.
    metric : str
        Accepted and validated, never used (E8).  ``'a'``/``'p'`` or any unambiguous contraction.
    conn : int
        ``bwlabel``'s connectivity; the M-file hard-codes **4**.

    Returns a list of :class:`RectFloe` in ``bwlabel`` label order.  Parity: exact.
    """
    m = str(metric).lower()
    if not (m and ("area".startswith(m) or "perimeter".startswith(m))):
        raise ValueError("metric does not match either 'area' or 'perimeter'")
    label = label_components(np.asarray(img) != 0, conn)
    num = int(label.max())
    out: list[RectFloe] = []
    for i in range(1, num + 1):
        r, c = np.nonzero(label == i)                       # 0-based; +1 for MATLAB's 1-based `find`
        rectx, recty, a, p = minboundrect(c.astype(np.float64) + 1.0, r.astype(np.float64) + 1.0, "a")
        cen = np.array([rectx[:4].sum() / 4.0, recty[:4].sum() / 4.0])
        out.append(RectFloe(Vertices=np.column_stack([rectx, recty]), Center=cen,
                            Area=float(a), Perimeter=float(p)))
    return out


@dataclass
class ModelFloe:
    """One entry of ``model_ice_model.m``'s ``s_model`` struct array.

    Note the field name: ch9 calls it **``Intersection``** (a flat index vector), where ch8's ``sea_ice_model.m``
    has ``Intersect`` (a nested ``struct('floe', 'brash')``).  The indices are **1-based serial numbers into the
    accepted list ``ss``**, not into the input ``S`` — the M-file's inner loop runs over ``ss``.
    """

    Vertices: np.ndarray
    Center: np.ndarray
    Area: float
    Perimeter: float
    Intersection: np.ndarray


@dataclass
class ModelIceModel:
    """Return value of :func:`model_ice_model`."""

    s_model: list[ModelFloe]
    bw: np.ndarray
    accepted: np.ndarray          # 0-based indices into the input `S` that passed the k1 < k < k2 filter
    ratios: np.ndarray            # the `k` of line 32 for every input floe (NOT normalised to >= 1)


def model_ice_model(S: Sequence[RectFloe], img: np.ndarray, k1: float = 0.4, k2: float = 2.5, *,
                    nan_bug: bool = True, strict_containment: bool = False,
                    drop_degenerate: bool = True) -> ModelIceModel:
    """``model_ice_model.m`` — aspect filter, raster union and overlap flags (Book §9.3.2.1, Fig. 9.15, p. 207).

    MATLAB source: ``MATLAB_ROOT/ch9/Model_Ice_Floe_Identification/model_ice_model.m`` lines 23–67::

        bw = zeros(size(img));  s_model = [];  ss = [];
        for i = 1 : length(S)
            v = cat(1, S(i).Vertices);
            k = sqrt((v(1,1)-v(2,1))^2 + (v(1,2)-v(2,2))^2) / ...
                sqrt((v(3,1)-v(2,1))^2 + (v(3,2)-v(2,2))^2);      % length/width ratio
            if k < k2 && k > k1
                bb = roipoly(double(bw), v(:,1), v(:,2));
                bw(bb == 1) = 1;  ss = [ss, S(i)];
            end
        end
        for i = 1 : length(ss)
            for j = 1 : length(ss), if j ~= i
                [xx, yy] = polybool('intersection', v(:,1), v(:,2), vj(:,1), vj(:,2));
                if xx ~= NaN,  inter = [inter; j];  end
            end, end
            s_model = [s_model; struct('Vertices', v, 'Center', c, 'Area', a, 'Perimeter', p, 'Intersection', inter)];
        end

    Erratum **E7 / risk R9**: ``k`` is **not** normalised to ``>= 1`` (it is side(v1,v2) over side(v2,v3), in the
    order ``minboundrect`` emitted them), so the shipped ``(k1, k2) = (0.4, 2.5)`` is a **symmetric band**, not
    the one-sided "remove ratios less than a threshold" the text describes on p. 207.  Ported literally.

    Erratum **E9** (ch08's): ``if xx ~= NaN`` is ``if ~isempty(xx)``.  Here the consequence is the **opposite**
    of ch08's ``polyxpoly`` case: ``polybool`` returns the intersection *region*, so a rectangle strictly inside
    another **is** flagged.  An R2025a probe pins all four cases — contained 5 vertices, partial 5, disjoint
    empty, **edge-touching (shared full edge, zero area) empty** — and
    :func:`seaice.core.polygon.clip_polygon_convex` with ``drop_degenerate=True`` matches all four.

    Parameters
    ----------
    S : sequence of :class:`RectFloe`
        Output of :func:`rect`.
    img : ndarray
        The segmented binary image; only its **shape** is used (``bw = zeros(size(img))``).
    k1, k2 : float
        The band; ``model_ice_demo.m`` line 65 sets ``0.4`` and ``2.5``.
    nan_bug : bool
        ``True`` (**default = the shipped code**): the overlap flag is ``~isempty(polybool(...))``.  ``False``
        raises, because there is no other sensible reading of ``xx ~= NaN`` — the flag exists so a reader can
        see the erratum is deliberate.
    strict_containment : bool
        ``True`` additionally flags a pair whose rectangles are nested without their boundaries crossing.  With
        ``polybool`` semantics this is **already** detected, so the flag is a no-op here and is kept only for
        API symmetry with :func:`seaice.ch08_applications.sea_ice_model` (where it is not a no-op).
    drop_degenerate : bool
        Passed to :func:`~seaice.core.polygon.clip_polygon_convex`; ``False`` would flag edge-touching
        rectangles as overlapping, which ``polybool`` does not.

    Returns :class:`ModelIceModel`.  Parity: `near` (``polybool``'s vertex order is not reproducible; only
    emptiness is consumed and that *is* reproducible).
    """
    if not nan_bug:
        raise NotImplementedError(
            "model_ice_model(nan_bug=False): `if xx ~= NaN` has no corrected reading other than `~isempty(xx)`; "
            "use strict_containment= for the geometry question instead")
    shape = np.asarray(img).shape[:2]
    bw = np.zeros(shape, dtype=np.float64)
    ratios = np.zeros(len(S))
    accepted: list[int] = []
    ss: list[RectFloe] = []
    for i, s in enumerate(S):
        v = np.asarray(s.Vertices, dtype=np.float64)
        # line 32-33: k = |v1 - v2| / |v3 - v2|  (1-based vertices 1, 2, 3 of the closed ring)
        num = np.hypot(v[0, 0] - v[1, 0], v[0, 1] - v[1, 1])
        den = np.hypot(v[2, 0] - v[1, 0], v[2, 1] - v[1, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            k = num / den
        ratios[i] = k
        if k < k2 and k > k1:                                    # line 34 — NOT normalised to >= 1
            bb = roipoly(bw, v[:, 0], v[:, 1])                   # line 35
            bw[bb] = 1.0                                         # line 36
            accepted.append(i)
            ss.append(s)
    s_model: list[ModelFloe] = []
    verts = [np.asarray(s.Vertices, dtype=np.float64) for s in ss]
    for i, s in enumerate(ss):
        v = verts[i]
        inter: list[int] = []
        for j in range(len(ss)):
            if j == i:
                continue                                          # line 51 `if j ~= i`
            vj = verts[j]
            xx, _yy = clip_polygon_convex(v[:, 0], v[:, 1], vj[:, 0], vj[:, 1],
                                          drop_degenerate=drop_degenerate)
            if xx.size > 0:                                       # line 54 `if xx ~= NaN` == `~isempty(xx)`
                inter.append(j + 1)                               # 1-based serial number into `ss`
            elif strict_containment and _nested(v, vj):
                inter.append(j + 1)
        s_model.append(ModelFloe(Vertices=v, Center=np.asarray(s.Center, dtype=np.float64),
                                 Area=float(s.Area), Perimeter=float(s.Perimeter),
                                 Intersection=np.array(inter, dtype=np.int64)))
    return ModelIceModel(s_model=s_model, bw=bw != 0, accepted=np.array(accepted, dtype=np.int64),
                         ratios=ratios)


def _nested(v1: np.ndarray, v2: np.ndarray) -> bool:
    """Is either rectangle entirely inside the other?  Only used by ``strict_containment=True`` (a no-op with
    ``polybool`` semantics, which already detect containment — see :func:`model_ice_model`)."""
    return _point_in_polygon(v1[0, 0], v1[0, 1], v2[:, 0], v2[:, 1]) or \
        _point_in_polygon(v2[0, 0], v2[0, 1], v1[:, 0], v1[:, 1])


def _point_in_polygon(px: float, py: float, x: np.ndarray, y: np.ndarray) -> bool:
    """Ray-crossing test (even–odd rule)."""
    inside = False
    n = x.size
    for i in range(n):
        j = (i - 1) % n
        if ((y[i] > py) != (y[j] > py)) and \
                (px < (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i] + 1e-300) + x[i]):
            inside = not inside
    return inside


def rect_ice_concentration(s_model: Sequence[ModelFloe] | ModelIceModel, shape: tuple[int, int]) -> float:
    """``Σ rectangle area / (M·N)`` — the third ice concentration compared on p. 208 (the book's 87.75 %).

    **Text only** (§9.3.2.1, no ``.m``).  The book's point is that this figure is **higher** than both the
    segmentation IC (76.96 %) and the thresholding IC (83.17 %) because **overlapping rectangles are counted
    twice**, which "compensates" the boundary pixels the segmentation turned into water.  The double counting is
    therefore deliberate — do not de-duplicate against :attr:`ModelIceModel.bw`.

    Parity: `reimplemented` (the book prints the identity, not code).
    """
    floes = s_model.s_model if isinstance(s_model, ModelIceModel) else list(s_model)
    total = float(sum(f.Area for f in floes))
    return total / float(shape[0] * shape[1])


def fsd_error(counts_rect: np.ndarray, counts_identified: np.ndarray) -> np.ndarray:
    """Bin-wise difference of two floe-size distributions — Book Fig. 9.16(b) (p. 210), error axis −60 … +60.

    **Text only.**  Both histograms must have been computed on the **same bin centres**
    (:func:`seaice.core.histogram.hist`); the error is ``counts_rect − counts_identified``.
    Parity: `reimplemented`.
    """
    a = np.asarray(counts_rect, dtype=np.float64)
    b = np.asarray(counts_identified, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("fsd_error: the two histograms must share the same bin centres")
    return a - b


# ===============================================================================================================
# §9.3.1/§9.3.2 — the demo driver, and the text-only tiling of the overall tank image
# ===============================================================================================================

@dataclass
class ModelIceDemo:
    """Return value of :func:`model_ice_demo` — every array ``model_ice_demo.m`` leaves in the workspace."""

    gvf: Any                       # seaice.ch06_gvf_snake.GVFDistance
    bw1: np.ndarray
    bw2: np.ndarray
    bw3: np.ndarray
    bw4: np.ndarray
    S: list[RectFloe] | None = None
    model: ModelIceModel | None = None


def model_ice_demo(I: np.ndarray, *, full: bool = False, params: dict | None = None,
                   max_seeds: int | None = None, keep_history: bool = False, progress=None,
                   solver: str = "auto", stop: str = "criteria") -> ModelIceDemo:
    """``model_ice_demo.m`` — the chapter's only runnable driver (Book §9.3.1/§9.3.2, Fig. 9.11).

    MATLAB source: ``MATLAB_ROOT/ch9/Model_Ice_Floe_Identification/model_ice_demo.m`` lines 9–66::

        bw1 = GVF_distance(I, sigma, GradientOn, GVFOn, Num, mu, iter, alpha, beta, gamma, kappa,
                           Dmin, Dmax, Ra_min, Ra, Rc, Rl, se, timer);
        bw2 = bwareaopen(bw1, Ra_min);          % 8-connected (the default)
        bw3 = bw2;
        % bw3 = imclearborder(bw3, 4);          % commented out in the shipped file
        bw3 = bwareaopen(bw3, Ra_min, 4);       % 4-connected
        bw4 = imfill(bw3, 8, 'holes');
        % S = rect(bw4);                        % lines 53-62, commented out
        % k1 = 0.4; k2 = 2.5;                   % line 65
        % s_model = model_ice_model(S, bw4, k1, k2);

    Lines 53–66 are **commented out in the shipped file**; ``full=True`` runs them (the ch06 ``--full``
    precedent).  The parameter block is :data:`BOOK_PARAMS_CH9`, transcribed value by value; ``timer = 2``
    means the outer re-segmentation loop of §9.3.1 runs twice — the remedy p. 205 prescribes for crowded
    aligned squares whose binarization leaves no hole between them.

    Parameters beyond the M-file: ``max_seeds``/``progress``/``solver``/``keep_history`` are the ch06 runtime
    guards, and ``stop='count'`` opts into Algorithm 7's own convergence test (gap G1).

    Returns :class:`ModelIceDemo`.  Parity: `near` — inherited from ch06's ``gvf_distance`` (``polybool``
    start-vertex residual; ch9's contours are short, ``N < 32``, so ``snakedeform`` takes the **dense** path,
    which ch06 measured as exact).
    """
    P = dict(BOOK_PARAMS_CH9)
    if params:
        P.update(params)
    res = gvf_distance(I, sigma=P["sigma"], GradientOn=P["GradientOn"], GVFOn=P["GVFOn"], Num=P["Num"],
                       mu=P["mu"], iter=P["iter"], alpha=P["alpha"], beta=P["beta"], gamma=P["gamma"],
                       kappa=P["kappa"], Dmin=P["Dmin"], Dmax=P["Dmax"], Ra_min=P["Ra_min"], Ra=P["Ra"],
                       Rc=P["Rc"], Rl=P["Rl"], se_radius=P["se_radius"], timer=P["timer"],
                       max_seeds=max_seeds, keep_history=keep_history, progress=progress, solver=solver,
                       stop=stop)
    bw1 = res.bw1
    bw2 = bwareaopen(bw1, int(P["Ra_min"]))                 # line 45 — 8-connected default
    bw3 = bwareaopen(bw2, int(P["Ra_min"]), 4)              # line 50 — 4-connected
    bw4 = imfill(bw3, "holes", conn=8)                      # line 51
    out = ModelIceDemo(gvf=res, bw1=bw1, bw2=bw2, bw3=bw3, bw4=bw4)
    if full:
        out.S = rect(bw4)                                   # line 54
        out.model = model_ice_model(out.S, bw4, P["k1"], P["k2"])   # line 66
    return out


def tiled_segmentation(I: np.ndarray, grid: tuple[int, int] = (2, 10), overlap: int = 16,
                       gvf_iters: Sequence[int] = TABLE_9_4, *, params: dict | None = None,
                       max_seeds: int | None = None, progress=None) -> tuple[np.ndarray, list[Any]]:
    """The **20 overlapping sub-images** of §9.3.2 (p. 205–207, Table 9.4) — stitched by OR.

    **Text only**: "a single GVF parameter set cannot serve the whole tank image, so it is split into 20
    overlapping sub-images (the overlap avoids border effects), Algorithm 7 is run locally with the per-sub-image
    GVF iteration count of Table 9.4, and the results are stitched" — but **no ``.m`` file drives the 20
    sub-images**, and the book gives no stitch rule and no grid shape.

    # DEVIATION: `reimplemented` — the ``grid``, the ``overlap`` in pixels and the OR stitch are **ours**.  The
    # only thing taken from the book is ``gvf_iters`` = :data:`TABLE_9_4`, whose 20 values are asserted in the
    # tests as a transcription.  Nothing computed here may be quoted against a book figure.

    Returns ``(bw1, per_tile_results)``.
    """
    gvf_iters = tuple(int(v) for v in gvf_iters)
    n_r, n_c = grid
    if len(gvf_iters) != n_r * n_c:
        raise ValueError(f"tiled_segmentation: {len(gvf_iters)} iteration counts for a {n_r}x{n_c} grid")
    img = np.asarray(I)
    M, N = img.shape[:2]
    hs = int(np.ceil(M / n_r))
    ws = int(np.ceil(N / n_c))
    out = np.zeros((M, N), dtype=bool)
    results: list[Any] = []
    P = dict(BOOK_PARAMS_CH9)
    if params:
        P.update(params)
    for i in range(n_r):
        for j in range(n_c):
            b = i * n_c + j
            r0, r1 = max(0, i * hs - overlap), min(M, (i + 1) * hs + overlap)
            c0, c1 = max(0, j * ws - overlap), min(N, (j + 1) * ws + overlap)
            tile = img[r0:r1, c0:c1]
            if progress is not None:
                progress(b, n_r * n_c)
            res = gvf_distance(tile, sigma=P["sigma"], GradientOn=P["GradientOn"], GVFOn=P["GVFOn"],
                               Num=gvf_iters[b], mu=P["mu"], iter=P["iter"], alpha=P["alpha"],
                               beta=P["beta"], gamma=P["gamma"], kappa=P["kappa"], Dmin=P["Dmin"],
                               Dmax=P["Dmax"], Ra_min=P["Ra_min"], Ra=P["Ra"], Rc=P["Rc"], Rl=P["Rl"],
                               se_radius=P["se_radius"], timer=P["timer"], max_seeds=max_seeds,
                               keep_history=False)
            out[r0:r1, c0:c1] |= res.bw1
            results.append(res)
    return out, results


# ===============================================================================================================
# §9.3.3 — maximum floe size from video
# ===============================================================================================================

@dataclass
class MovieFloe:
    """Return value of :func:`movie_floe` — the Fig. 9.18 time series and its inputs.

    Attributes
    ----------
    floe : (N,) the script's ``floe(k) = max(ice_areas)`` in **pixels** — the y value of Fig. 9.18
    labels : list of (r, c) int32 ``bwlabel(out(k).cdata, 4)`` label matrices (empty when ``keep_labels=False``)
    areas : list of (n_k,) per-frame component areas (``[icedata.Area]``)
    empty_frames : (m,) indices of frames whose ``ice_areas`` was empty
    """

    floe: np.ndarray
    labels: list[np.ndarray]
    areas: list[np.ndarray]
    empty_frames: np.ndarray


def movie_floe(frames: np.ndarray, *, min_area: int = 20, conn: int = 4, empty: str = "raise",
               keep_labels: bool = False) -> MovieFloe:
    """``movie_floe.m`` — the maximum floe area per frame (Book §9.3.3, Fig. 9.18, pp. 208–211).

    MATLAB source: ``MATLAB_ROOT/ch9/Model_Ice_Floe_Identification/movie_floe.m`` lines 15–27::

        im(k).cdata  = im2bw(mov(k).cdata);            % NO level -> 0.5 -> rgb2gray then > 127.5
        out(k).cdata = bwareaopen(im(k).cdata, 20, 4);
        cc           = bwlabel(out(k).cdata, 4);
        icedata      = regionprops(cc, 'basic');       % Area, Centroid, BoundingBox
        ice_areas    = [icedata.Area];
        floe(k)      = max(ice_areas);                 % <- Fig. 9.18's y value, in PIXELS

    Because line 18 calls ``im2bw`` with **no level**, the AVI must already hold the **segmented (binary)
    frames**: the per-frame Algorithm-7 segmentation the section describes has no shipped code (gap **G2** — see
    :func:`segment_video`, which is *ours*).  ``mmreader`` on line 5 is removed in R2025a and is patched to
    ``VideoReader`` for the reference run (IO only, ``reference/ch09/patches.json``).  There is **no plotting
    code**: Fig. 9.18 was made interactively; ``scripts/ch09_movie_floe.py`` plots it.

    Parameters
    ----------
    empty : {'raise', 'delete', 'nan'}
        What to do when a frame has no component left after ``bwareaopen`` (``max([])`` returns ``[]``).

        ``'raise'`` (**default**) is what **MATLAB R2025a actually does**: ``floe(k) = []`` with
        ``numel(floe) == k-1`` is a null assignment *past the end*, which errors with
        ``MATLAB:matrix:singleSubscriptNumelMismatch`` ("Unable to perform assignment because the left and right
        sides have a different number of elements").  This **corrects `analysis/ch09.md` risk R13**, which
        predicted a silent deletion: deletion is what ``x(k) = []`` does for ``k <= numel(x)``, and this loop,
        which appends one element per frame, never reaches that case.

        ``'delete'`` reproduces the deletion semantics anyway (the vector is shortened and every later index
        shifts), for the reachable case and for readers of the original claim.  ``'nan'`` keeps the frame
        index aligned with a NaN.

    Returns :class:`MovieFloe`.  Parity: **exact** on an Uncompressed AVI.
    """
    if empty not in ("raise", "delete", "nan"):
        raise ValueError("empty must be 'raise', 'delete' or 'nan'")
    stack = _as_frame_list(frames)
    floe: list[float] = []
    labels: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    empties: list[int] = []
    for k, frame in enumerate(stack):
        bw = im2bw(frame)                                   # line 18 — default level 0.5
        cleaned = bwareaopen(bw, int(min_area), conn)       # line 20
        cc = label_components(cleaned, conn)                # line 22
        stats = regionprops(cc, "basic")                    # line 23
        a = np.array([s.Area for s in stats], dtype=np.float64)
        areas.append(a)
        if keep_labels:
            labels.append(cc)
        if a.size == 0:
            empties.append(k)
            if empty == "raise":
                raise ValueError(
                    f"movie_floe: frame {k + 1} has no component >= {min_area} px, so `max(ice_areas)` is [] and "
                    "`floe(k) = []` is a null assignment past the end of the vector — MATLAB R2025a errors with "
                    "MATLAB:matrix:singleSubscriptNumelMismatch.  Pass empty='delete' or empty='nan'.")
            if empty == "nan":
                floe.append(np.nan)
            # 'delete': append nothing -- the vector shortens and every later index shifts (MATLAB's `x(k)=[]`
            # semantics for the reachable case k <= numel(x)).
            continue
        floe.append(float(a.max()))                         # line 25
    return MovieFloe(floe=np.array(floe), labels=labels, areas=areas,
                     empty_frames=np.array(empties, dtype=np.int64))


def segment_video(frames: np.ndarray, *, params: dict | None = None, max_seeds: int | None = None,
                  downscale: int = 1, progress=None) -> np.ndarray:
    """Algorithm 7 applied **per frame** to a tank video — Book §9.3.3, Fig. 9.17(b).

    # DEVIATION: **this wiring is OURS, not the book's** (gap **G2**).  §9.3.3 describes running the §9.3.2
    # pipeline on every decimated frame and feeding the result to ``movie_floe.m``, but **no ``.m`` file ships
    # it**: ``movie_floe.m`` reads an AVI that already contains segmented frames.  Nothing produced by this
    # function may appear in a parity claim or be compared with Fig. 9.17/9.18.

    Runtime warning: :func:`seaice.ch06_gvf_snake.gvf_distance` takes ~3 s on ch9's 181 × 76 crop; a 480 × 640
    frame is ~20× larger, so a 60-frame video is measured in hours.  Use ``downscale`` and ``max_seeds``.

    Returns an ``(N, H, W)`` bool stack of segmented frames, ready for :func:`movie_floe` after rendering to
    0/255 RGB.
    """
    P = dict(BOOK_PARAMS_CH9)
    if params:
        P.update(params)
    stack = _as_frame_list(frames)
    out = []
    for k, frame in enumerate(stack):
        f = frame[::downscale, ::downscale] if downscale > 1 else frame
        if progress is not None:
            progress(k, len(stack))
        res = gvf_distance(f, sigma=P["sigma"], GradientOn=P["GradientOn"], GVFOn=P["GVFOn"], Num=P["Num"],
                           mu=P["mu"], iter=P["iter"], alpha=P["alpha"], beta=P["beta"], gamma=P["gamma"],
                           kappa=P["kappa"], Dmin=P["Dmin"], Dmax=P["Dmax"], Ra_min=P["Ra_min"], Ra=P["Ra"],
                           Rc=P["Rc"], Rl=P["Rl"], se_radius=P["se_radius"], timer=P["timer"],
                           max_seeds=max_seeds, keep_history=False)
        bw = bwareaopen(res.bw1, int(P["Ra_min"]))
        bw = bwareaopen(bw, int(P["Ra_min"]), 4)
        out.append(imfill(bw, "holes", conn=8))
    return np.array(out)


# ===============================================================================================================
# helpers
# ===============================================================================================================

def _as_frame_list(frames: np.ndarray) -> list[np.ndarray]:
    """Accept ``(H, W, 3, N)`` (MATLAB ``read(VideoReader)`` order), ``(N, H, W, 3)`` or a list of frames.

    The layout is inferred from the shape: ``(·, ·, 3, N≠3)`` is MATLAB's ``HWCN``, ``(·, ·, ·, 3)`` is imageio's
    ``NHWC``.  A stack that is **3 frames wide and 3 channels deep at once** (``(H, W, 3, 3)`` vs ``(3, H, W, 3)``)
    cannot be told apart and is read as ``NHWC``; pass a list of frames if that ever matters.
    """
    if isinstance(frames, (list, tuple)):
        return [np.asarray(f) for f in frames]
    arr = np.asarray(frames)
    if arr.ndim == 3:
        return [arr]
    if arr.ndim != 4:
        raise ValueError(f"expected a 4-D frame stack, got shape {arr.shape}")
    if arr.shape[2] == 3 and arr.shape[3] != 3:
        return [arr[:, :, :, k] for k in range(arr.shape[3])]      # (H, W, 3, N)
    if arr.shape[3] == 3:
        return [arr[k] for k in range(arr.shape[0])]               # (N, H, W, 3)
    raise ValueError(f"cannot tell the frame layout of shape {arr.shape}; "
                     "pass (H, W, 3, N) or (N, H, W, 3)")


def floe_size_histogram(areas, bins: int | np.ndarray = 10) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB ``[z, x] = hist(areas, bins)`` on a floe-area list — Book Figs. 9.13 / 9.16(a) (0–6000 px,
    frequency 0–300).  A thin wrapper over :func:`seaice.core.histogram.hist` so both FSDs of :func:`fsd_error`
    are computed on the same bin **centres**.  Parity: exact (the histogram); the plotted range is display."""
    return hist(np.asarray(areas, dtype=np.float64), bins)


# ===============================================================================================================
# Tier-3 fixture FILES — written once, then read back, so MATLAB and Python decode identical bytes
#
# The ch03 `t.jpg` precedent: a JPEG that both engines *decode* is the only way an `imread` chain can be compared
# without a decoder difference (+-1 gray level moves an Otsu threshold).  The same logic gives the videos an
# **Uncompressed AVI** container (risk R2).  Everything under `data/synthetic/` is git-ignored and regenerated on
# demand, so nothing book-derived and nothing large is committed.
# ===============================================================================================================

SYNTHETIC_DIR_NAME = "ch09"


def _synthetic_dir(base: "Path | str | None" = None) -> "Path":
    from .core.io import REPO_ROOT

    d = Path(base) if base is not None else REPO_ROOT / "data" / "synthetic" / SYNTHETIC_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_synthetic_tank(path: "Path | str | None" = None, *, regenerate: bool = False,
                          **kwargs) -> tuple["Path", np.ndarray]:
    """Write (once) and read back the Tier-3 stand-in for ``04100_analyse.jpg`` — Book Fig. 9.1 / §9.2.1.

    The image comes from :func:`seaice.core.synth.model_ice_tank`; it is saved as **JPEG quality 95 with 4:4:4
    chroma** (``subsampling=0``) and then **read back**, so every number downstream is computed on the decoded
    bytes — the same bytes MATLAB's ``imread`` would see.  This is the ch03 ``t.jpg`` precedent: comparing a
    freshly generated float array with MATLAB's JPEG decode would measure the codec, not the algorithm.

    Returns ``(path, rgb)``.  **Tier 3 — synthetic.**  No number computed from it is a book number.
    """
    from .core.io import read_image
    from .core.synth import model_ice_tank

    p = Path(path) if path is not None else _synthetic_dir() / "04100_analyse.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    if regenerate or not p.exists():
        from PIL import Image

        Image.fromarray(model_ice_tank(**kwargs)).save(p, format="JPEG", quality=95, subsampling=0)
    return p, read_image(p)


def ensure_synthetic_tank_video(path: "Path | str | None" = None, *, n_frames: int = 24,
                                regenerate: bool = False, fps: float = 12.0,
                                **kwargs) -> tuple["Path", np.ndarray]:
    """Write (once) and read back the Tier-3 stand-in for ``dypic_05100_cam1_top.avi`` — §9.2.2.

    :func:`seaice.core.synth.model_ice_tank_video` rendered as an **Uncompressed AVI**
    (:func:`seaice.core.video.write_video`, ``codec='rawvideo'``), verified to round-trip bit-exactly through
    MATLAB's ``VideoReader`` — the mitigation for risk R2 (a lossy container would turn every ice-concentration
    number into a decoder comparison).

    Returns ``(path, frames)`` with ``frames`` in **MATLAB's ``(H, W, 3, N)`` order**, exactly what
    ``read(VideoReader(...))`` returns.
    """
    from .core.synth import model_ice_tank_video
    from .core.video import read_video, write_video

    p = Path(path) if path is not None else _synthetic_dir() / "dypic_synth_top.avi"
    if regenerate or not p.exists():
        write_video(p, model_ice_tank_video(n_frames, **kwargs), fps)
    return p, read_video(p)


def ensure_synthetic_segmented_video(path: "Path | str | None" = None, *, n_frames: int = 40,
                                     regenerate: bool = False, fps: float = 12.0,
                                     **kwargs) -> tuple["Path", np.ndarray]:
    """Write (once) and read back the Tier-3 stand-in for ``05100.avi`` — §9.3.3, Fig. 9.18.

    :func:`seaice.core.synth.segmented_floe_video` rendered as an **Uncompressed AVI**.  Its content is binary
    (0/255), so ``movie_floe.m``'s level-free ``im2bw`` (threshold 127.5) recovers the masks exactly, which is
    what the script's own logic requires (gap G2 — the AVI the authors read was already segmented).

    Returns ``(path, frames)`` in MATLAB's ``(H, W, 3, N)`` order.
    """
    from .core.synth import segmented_floe_video
    from .core.video import read_video, write_video

    p = Path(path) if path is not None else _synthetic_dir() / "05100_synth_segmented.avi"
    if regenerate or not p.exists():
        write_video(p, segmented_floe_video(n_frames, **kwargs), fps)
    return p, read_video(p)
