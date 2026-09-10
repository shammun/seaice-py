"""Chapter 7 — Sea Ice Type Identification (Zhang & Skjetne, CRC Press 2018, pp. 145–174).

The chapter turns the GVF-snake segmentation of ch6 (Algorithm 1) into *identified* individual ice pieces and
classifies them into four layers:

* **§7.1 Ice shape enhancement** — morphological cleaning (§7.1.1, Fig. 7.2), connected-component extraction by
  constrained dilation (§7.1.2, **Eq. 7.1**, Figs. 7.3/7.4), hole filling by constrained dilation (§7.1.3.1,
  **Eq. 7.2**, Figs. 7.5–7.7) and by reconstruction (§7.1.3.2, **Eqs. 7.3–7.4**, Fig. 7.8), and the shape
  enhancement algorithm itself (§7.1.4, **Algorithm 2**, Fig. 7.9).
* **§7.2 General sea ice image processing** — pixel extraction (§7.2.1), edge detection (§7.2.2, **Algorithm 3**),
  shape enhancement (§7.2.3, **Algorithm 4**, **Eq. 7.5**), type classification and the FSD (§7.2.4,
  **Algorithm 5**, **Eq. 7.6**, Figs. 7.13–7.16).

MATLAB sources ported here (one Python home each, `analysis/ch07.md` §3):

===================================================================  ====================================
``ch7/cleaning & labeling & filling/morphology_cleaning.m`` (42 l)    :func:`morphological_cleaning`
``ch7/cleaning & labeling & filling/labeling.m`` (53 l)               :func:`connected_component_extract`
``ch7/cleaning & labeling & filling/filling.m`` (55 l)                :func:`hole_fill_dilation`
``ch7/cleaning & labeling & filling/filling_reconstruct.m`` (71 l)    :func:`border_marker`, :func:`hole_fill_reconstruct`
``ch7/Sea_Ice_Floe_Identification/ice_shape_enhancement.m`` (252 l)   :func:`ice_shape_enhancement`
``ch7/Sea_Ice_Floe_Identification/sea_ice_demo.m`` (driver)           ``scripts/ch07_sea_ice_demo.py``
===================================================================  ====================================

All 23 ``.m`` files of ``ch7/Sea_Ice_Floe_Identification/`` are **byte-identical** to ``ch6/``'s copies
(``cmp`` 23/23).  The 21 not listed above are already ported (`knowledge/function_map.md`): the whole
:mod:`seaice.core.snake` stack,
:mod:`seaice.core.polygon`, :func:`seaice.ch06_gvf_snake.gvf_distance` and
:func:`seaice.ch06_gvf_snake.seaice_kmean_gvf` (= Algorithm 3).  ``sea_ice_model.m``, ``color_hist*.m`` and
``SeaIce_Image_Structure.m`` belong to ch8 / Appendix B and are deliberately **not** ported here.

This code is a port of material that is "only available for academic non-commercial use"; the authors ask that
users cite Q. Zhang and R. Skjetne, *Image processing for identification of sea-ice floes and the floe size
distributions*, IEEE TGRS 53(5):2913–2924, 2015, and *Image techniques for identifying sea-ice parameters*,
Modeling, Identification and Control 35(4):293–301, 2014.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core.connectivity import label_components
from .core.histogram import hist as matlab_hist
from .core.morphology import imclose, imdilate, imfill, imopen, imreconstruct, strel
from .core.regionprops import regionprops

__all__ = [
    "SE_SQUARE_2", "C1_COLOR", "C2_AREA", "SE_TH", "MIN_FLOE", "MIN_BRASH",
    "MorphologicalCleaning", "ConstrainedDilation", "HoleFillDilation", "IceShapeEnhancement", "IcePiece",
    "Coverage", "FloeSizeDistribution",
    "morphological_cleaning", "connected_component_extract", "hole_fill_dilation", "border_marker",
    "hole_fill_reconstruct", "adaptive_se_radius", "size_color", "color_to_area", "colorbar_area_ticks",
    "ice_shape_enhancement", "sea_ice_edge_detection", "sea_ice_shape_enhancement", "ice_types_classification",
    "floe_size_distribution", "TileRecord", "LocalSegmentation", "tile_grid", "local_segmentation",
    "resample_categorical",
]

# --------------------------------------------------------------------------------------------------------------
# Book constants (every one of them cited where it is printed)
# --------------------------------------------------------------------------------------------------------------

#: Fig. 7.2 uses a **2×2 square** structuring element (``morphology_cleaning.m`` line 17
#: ``se = strel('square', 2)``).  It is *even*, so its origin is ``floor((2+1)/2) = 1`` (1-based) — that
#: asymmetry is what makes the closing/opening of Fig. 7.2 shift the way the book prints it.
SE_SQUARE_2 = "square", 2

#: Eq. (7.6) constants, p. 159: ``Color(p) = C1 (1 - exp(-area(i)/C2))`` with ``C1 = 10000``, ``C2 = 1000``.
C1_COLOR = 10000.0
C2_AREA = 1000.0

#: ``sea_ice_demo.m`` lines 27–29 — the §7.3.2 parameter block for the shape enhancement.
SE_TH = 50      #: Eq. (7.5) ``size_th``: pieces below this area are cleaned with ``r1 = 1``, the rest with ``r2 = 2``.
MIN_FLOE = 40   #: Algorithm 5 ``T_floe`` (pixel area) separating ice floes from brash ice.
MIN_BRASH = 1   #: smallest brash-ice area; with the script's strict ``>`` this **drops** 1-pixel pieces.


# ==============================================================================================================
# §7.1.1 Morphological cleaning — Fig. 7.2 (morphology_cleaning.m)
# ==============================================================================================================

@dataclass
class MorphologicalCleaning:
    """Every array ``morphology_cleaning.m`` displays.

    ``f1``/``f2``/``f0`` are the first figure (Fig. 7.2(b)/(c)/(d)); ``c1``–``c4`` are the second figure, the
    explicit dilate → erode → erode → dilate chain that spells the same result out step by step (in the code but
    not in the book text).
    """

    I: np.ndarray
    f1: np.ndarray   #: Fig. 7.2(b) ``imclose(I, se)``
    f2: np.ndarray   #: Fig. 7.2(c) ``imopen(I, se)``
    f0: np.ndarray   #: Fig. 7.2(d) ``imopen(f1, se)`` = the morphological cleaning
    c1: np.ndarray   #: ``imdilate(I, se)``
    c2: np.ndarray   #: ``imerode(c1, se)``  (= ``f1``)
    c3: np.ndarray   #: ``imerode(c2, se)``
    c4: np.ndarray   #: ``imdilate(c3, se)`` (= ``f0``)


def morphological_cleaning(I: np.ndarray, se: np.ndarray | None = None) -> MorphologicalCleaning:
    """Morphological cleaning = **closing first, then opening** — Book §7.1.1, Fig. 7.2 (p. 145–146).

    MATLAB source: ``MATLAB_ROOT/ch7/cleaning & labeling & filling/morphology_cleaning.m`` lines 17–27::

        se = strel('square', 2);
        f1 = imclose(I, se);   f2 = imopen(I, se);   f0 = imopen(f1, se);
        c1 = imdilate(I, se);  c2 = imerode(c1, se); c3 = imerode(c2, se);  c4 = imdilate(c3, se);

    "Both binary closing and opening operations can smooth the contours of objects […] The closing operation is
    able to close narrow cracks, fill long thin channels, and eliminate the holes that are smaller than the
    structuring element.  The opening operation is able to break thin connections between objects, remove small
    protrusions, and eliminate complete regions of an object that cannot contain the structuring element." (p. 145)

    Parameters
    ----------
    I : ndarray
        Binary image.  The M-file's literal is a **double** 0/1 matrix, so the default keeps ``I``'s dtype
        (:data:`seaice.core.synth.FIG_7_2_IMAGE` is bool; pass ``I.astype(float)`` for the M-file's class).
    se : ndarray, optional
        Structuring-element neighbourhood; default ``strel('square', 2)``.

    Returns
    -------
    :class:`MorphologicalCleaning`

    Parity: **exact** — this is a thin composition of :func:`seaice.core.morphology.imclose` / ``imopen`` /
    ``imerode`` / ``imdilate``, all verified bit-exact against R2025a in ch4 (including ``imclose``'s zero
    pre-pad and ``imdilate``'s SE reflection, which an *even* 2×2 SE actually exercises).  ``f1``, ``f2`` and
    ``f0`` reproduce the printed Fig. 7.2(b)/(c)/(d) 0 px (sums 111 → 115 / 104 / 108).
    """
    from .core.morphology import imerode  # local import keeps the module's public surface tidy

    if se is None:
        se = strel(*SE_SQUARE_2)
    I = np.asarray(I)
    f1 = imclose(I, se)
    f2 = imopen(I, se)
    f0 = imopen(f1, se)
    c1 = imdilate(I, se)
    c2 = imerode(c1, se)
    c3 = imerode(c2, se)
    c4 = imdilate(c3, se)
    return MorphologicalCleaning(I=I, f1=f1, f2=f2, f0=f0, c1=c1, c2=c2, c3=c3, c4=c4)


# ==============================================================================================================
# §7.1.2 Connected-component extraction — Eq. (7.1), Figs. 7.3/7.4 (labeling.m)
# ==============================================================================================================

@dataclass
class ConstrainedDilation:
    """Result of an Eq. (7.1) / Eq. (7.2) constrained-dilation recursion.

    ``blocks[0]`` is ``X_0 ⊕ B`` (the first thing the book prints), ``blocks[k]`` for ``k ≥ 1`` is ``X_k``.
    ``n_iter`` is the smallest ``k`` with ``X_k == X_{k-1}``; ``component`` is ``X_n``.
    """

    blocks: list[np.ndarray]
    component: np.ndarray
    n_iter: int
    converged: bool


def _constrained_dilation(mask: np.ndarray, seed: np.ndarray, se: np.ndarray,
                          max_iter: int) -> ConstrainedDilation:
    """``X_k = (X_{k-1} ⊕ B) ∩ mask``, recording ``[X_0 ⊕ B, X_1, X_2, …]`` exactly as the book prints them."""
    mask = np.asarray(mask).astype(bool)
    x = np.asarray(seed).astype(bool)
    blocks = [imdilate(x, se)]          # the M-files' `xx = imdilate(x0, se)` — printed before X_1
    x = blocks[0] & mask                # x1 = xx & I
    blocks.append(x)
    k, converged = 1, False
    while k < int(max_iter):
        nxt = imdilate(x, se) & mask    # X_{k+1}
        k += 1
        blocks.append(nxt)
        converged = bool(np.array_equal(nxt, x))
        x = nxt
        if converged:
            break
    return ConstrainedDilation(blocks=blocks, component=x, n_iter=k, converged=converged)


def connected_component_extract(A: np.ndarray, seed, se: np.ndarray | str = "square3",
                                max_iter: int = 9) -> ConstrainedDilation:
    """Extract the connected component of ``A`` containing ``p`` — Book §7.1.2, **Eq. (7.1)**, Figs. 7.3/7.4.

    .. math::  X_k = (X_{k-1} \\oplus B) \\cap A,\\qquad k = 1, 2, 3, \\ldots,\\qquad X_0 = \\{p\\}

    "the procedure terminates when :math:`X_k = X_{k-1}`" (p. 147).  ``B`` selects the connectivity: the **3×3
    square** for 8-connectivity (Fig. 7.3(a), completes at the 8th printed block) and the **cross**
    ``strel('diamond', 1)`` for 4-connectivity (Fig. 7.4(a), 7 printed blocks).

    MATLAB source: ``ch7/cleaning & labeling & filling/labeling.m`` lines 23–35::

        se = strel('square', 3);      % Fig. 7.3
        % se = strel('diamond', 1);   % Fig. 7.4 (the commented line 24)
        xx = imdilate(x0, se);   x1 = xx & I;   x2 = imdilate(x1, se) & I;   ...   x9 = ...

    Parameters
    ----------
    A : ndarray
        Binary image.
    seed : ndarray or (row, col)
        ``X_0``: either a binary image with the seed pixels set, or a **0-based** ``(row, col)`` pair.
    se : ndarray or {'square3', 'diamond1'}
        The structuring element ``B``; the two strings are the book's two cases.
    max_iter : int
        Cap on the recursion (``labeling.m`` unrolls it to ``x9``).

    Returns
    -------
    :class:`ConstrainedDilation` with ``blocks = [X_0 ⊕ B, X_1, …]``, ``component`` and ``n_iter``.

    Parity: **reimplemented** (the equation, not a toolbox call) on top of the exact
    :func:`seaice.core.morphology.imdilate`.  Cross-check: ``component`` equals the
    :func:`seaice.core.connectivity.label_components` (= ``bwlabel``) component that contains the seed, for both
    connectivities.  All 8 printed blocks of Fig. 7.3(d) and all 7 of Fig. 7.4(d) are reproduced 0 px.
    """
    A = np.asarray(A)
    se = _resolve_se(se)
    seed = _resolve_seed(seed, A.shape)
    return _constrained_dilation(A.astype(bool), seed, se, max_iter)


def _resolve_se(se) -> np.ndarray:
    """``'square3'`` → ``strel('square', 3)`` (8-conn), ``'diamond1'`` → ``strel('diamond', 1)`` (4-conn)."""
    if isinstance(se, str):
        key = se.lower().replace("_", "").replace("-", "")
        if key in ("square3", "square", "sq3", "8"):
            return strel("square", 3)
        if key in ("diamond1", "diamond", "cross", "4"):
            return strel("diamond", 1)
        raise ValueError(f"unknown structuring element {se!r} (use 'square3' or 'diamond1')")
    return np.asarray(se).astype(bool)


def _resolve_seed(seed, shape: tuple[int, int]) -> np.ndarray:
    """Accept either a binary ``X_0`` image or a 0-based ``(row, col)`` pixel."""
    seed = np.asarray(seed)
    if seed.shape == shape and seed.ndim == 2:
        return seed.astype(bool)
    if seed.shape == (2,):
        x0 = np.zeros(shape, dtype=bool)
        x0[int(seed[0]), int(seed[1])] = True
        return x0
    raise ValueError("seed must be a binary image of the same shape or a 0-based (row, col) pair")


# ==============================================================================================================
# §7.1.3.1 Hole filling by constrained dilation — Eq. (7.2), Figs. 7.5/7.6/7.7 (filling.m)
# ==============================================================================================================

@dataclass
class HoleFillDilation(ConstrainedDilation):
    """:class:`ConstrainedDilation` plus ``filled`` = ``X_n ∪ A`` (``filling.m`` line 36 ``I1 = x9 | I0``)."""

    filled: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    complement: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))


def hole_fill_dilation(A: np.ndarray, seed, se: np.ndarray | str = "diamond1",
                       max_iter: int = 9) -> HoleFillDilation:
    """Fill a hole by constrained dilation of a seed **inside** it — Book §7.1.3.1, **Eq. (7.2)**, Fig. 7.5.

    .. math::  X_k = (X_{k-1} \\oplus B) \\cap A^c,\\qquad X_0 = \\{p\\},\\ p \\in \\text{hole}

    and the filled image is :math:`X_n \\cup A` (p. 149–150).  The recursion is Eq. (7.1)'s with the mask
    replaced by the **complement**, so this function shares :func:`_constrained_dilation` with it.

    MATLAB source: ``ch7/cleaning & labeling & filling/filling.m`` lines 23–36::

        se = strel('diamond', 1);
        I  = ~I0;
        xx = imdilate(x0, se);  x1 = xx & I;  x2 = imdilate(x1, se) & I;  ...  x9 = ...
        I1 = x9 | I0;

    The two failure cases of §7.1.3.1 are reached by changing the arguments, not the code:

    * **Fig. 7.6** — :data:`seaice.core.synth.FIG_7_6_IMAGE` (the Fig. 7.5 image plus one pixel at 1-based
      (5, 5)) has *two* 4-connected holes, and the cross SE fills only the one containing the seed;
    * **Fig. 7.7** — the same image with ``se='square3'``: the hole is 8-connected to the background, so the
      recursion floods the whole background and "fills" nothing.

    That is the algorithm's weakness the book highlights: it must be told whether the seed is in a hole or in the
    background, which is "difficult to automatically fulfil" — hence §7.1.3.2 / :func:`hole_fill_reconstruct`.

    The text says Fig. 7.5 "finishes at the 7th iteration"; the **fixed point is** ``X_6`` (``X_7 == X_6``), and
    the printed blocks stop at ``X_6`` before the union, so the 7th printed block is ``X_6`` and both readings
    give the same final image.  :attr:`ConstrainedDilation.n_iter` reports the ``k`` at which ``X_k == X_{k-1}``
    (7 here), so the fixed point is ``blocks[n_iter - 1]``.

    Parity: **reimplemented** (from the equation).  All 8 printed blocks of Fig. 7.5(e), all 4 of Fig. 7.6(e) and
    9 of the 10 of Fig. 7.7(e) are reproduced 0 px; the 10th (``X_8``) differs by the 2 pixels of the book typo
    recorded in :data:`seaice.core.synth.FIG_7_7_X8_TYPO`.
    """
    A = np.asarray(A).astype(bool)
    se = _resolve_se(se)
    x0 = _resolve_seed(seed, A.shape)
    comp = ~A
    r = _constrained_dilation(comp, x0, se, max_iter)
    return HoleFillDilation(blocks=r.blocks, component=r.component, n_iter=r.n_iter, converged=r.converged,
                            filled=r.component | A, complement=comp)


# ==============================================================================================================
# §7.1.3.2 Hole filling by reconstruction — Eqs. (7.3)/(7.4), Fig. 7.8 (filling_reconstruct.m)
# ==============================================================================================================

def border_marker(F: np.ndarray) -> np.ndarray:
    """The marker image of **Eq. (7.3)**, Book §7.1.3.2, p. 152, Fig. 7.8(d).

    .. math::

        F_m(x, y) = \\begin{cases} 1 - F(x, y) & \\text{if } (x, y) \\text{ is on the border of } F \\\\
                                   0           & \\text{otherwise} \\end{cases}

    MATLAB source: ``filling_reconstruct.m`` lines 23–31 — the ``x0`` literal, which was checked here to be
    exactly Eq. (7.3) applied to that file's ``I0`` (0 px vs the printed Fig. 7.8(d), **27 pixels**; the 48 of
    Fig. 7.8's later blocks is ``H``, not the marker).  The M-file writes the marker out by hand; this is the
    general form.

    Parity: **reimplemented** (the equation; the M-file has no code for it).
    """
    F = np.asarray(F).astype(bool)
    Fm = np.zeros(F.shape, dtype=bool)
    Fm[0, :] = ~F[0, :]
    Fm[-1, :] = ~F[-1, :]
    Fm[:, 0] = ~F[:, 0]
    Fm[:, -1] = ~F[:, -1]
    return Fm


def hole_fill_reconstruct(F: np.ndarray, conn: int = 4) -> np.ndarray:
    """Automatic hole filling by morphological reconstruction — Book §7.1.3.2, **Eq. (7.4)**, Fig. 7.8.

    .. math::  H = \\left[ R^{D}_{F^c}(F_m) \\right]^{c}

    with :math:`F_m` from Eq. (7.3) (:func:`border_marker`).  "this algorithm is fully automatic since it does
    not need any information of the holes" (p. 152).

    MATLAB source: ``filling_reconstruct.m`` lines 36–48, which unrolls the reconstruction as nine constrained
    dilations and then complements: ``x10 = ~x9``; ``I1 = x10 | I0``; ``I2 = x10 & I``.  The library form is
    ``imfill(F, 'holes')`` — :func:`seaice.core.morphology.imfill`, which is the same equation.

    Parameters
    ----------
    F : ndarray
        Binary image.
    conn : int
        Connectivity of the reconstruction; 4 (the cross of Fig. 7.8, and MATLAB ``imfill``'s default) or 8 (the
        commented ``strel('square', 3)`` of ``filling_reconstruct.m`` line 34).

    Returns
    -------
    bool ndarray ``H`` — ``F`` with every hole filled.

    Parity: **exact** — built on :func:`seaice.core.morphology.imreconstruct`; equal to
    ``imfill(F, 'holes')`` and to ``scipy.ndimage.binary_fill_holes`` for logical input, and it reproduces
    Fig. 7.8's printed ``H`` (48 px) and ``H ∩ F^c`` (9 px) 0 px.
    """
    F = np.asarray(F).astype(bool)
    Fm = border_marker(F)
    return ~imreconstruct(Fm, ~F, conn)


# ==============================================================================================================
# §7.1.4 / §7.2.3 / §7.2.4 — Algorithms 2, 4 and 5, Eqs. (7.5) and (7.6)
# ==============================================================================================================

def adaptive_se_radius(size_ice: float, size_th: float = SE_TH, r1: int = 1, r2: int = 2) -> int:
    """**Eq. (7.5)**, Book §7.2.3, p. 158 — the disk radius adapted to the piece size.

    .. math::

        r = \\begin{cases} r_1 & \\text{if } size_{ice} < size_{th} \\\\
                           r_2 & \\text{if } size_{ice} \\ge size_{th} \\end{cases}

    MATLAB source: ``ice_shape_enhancement.m`` lines 83–89 ``r = length(p); if r < se_th, r = 1; else r = 2;
    end; se = strel('disk', r)``, i.e. ``r1 = 1``, ``r2 = 2``, ``size_th = se_th = 50`` (``sea_ice_demo.m``
    line 27).  Note that ``size_ice`` is the area of the piece **before** filling and cleaning (``length(p)``,
    the raw label pixel count), not the filled area.

    The shipped ``README.docx`` states "radius = 1, area ≤ se_th"; the equation and the code both use a strict
    ``<``, so the README is the odd one out and is ignored here.

    Parity: exact vs the M-file (and vs the printed equation, which agrees with it).
    """
    return int(r1) if float(size_ice) < float(size_th) else int(r2)


def size_color(area, C1: float = C1_COLOR, C2: float = C2_AREA):
    """**Eq. (7.6)**, Book §7.2.3, p. 159 — the size-coded colour of an identified ice piece.

    .. math::

        Color(p) = \\begin{cases} 0 & \\text{if } p \\notin ICE_{SEA} \\\\
        C_1 \\left(1 - e^{-area(i)/C_2}\\right) & \\text{if } p \\in ice_{sea}(i) \\end{cases}

    with :math:`C_1 = 10000`, :math:`C_2 = 1000`.  MATLAB source: ``ice_shape_enhancement.m`` line 127
    ``color_label = fix((1 - exp(-area0/1000)) * 10000)`` — note ``fix`` (**truncation towards zero**), not
    ``round``.  "smaller ice pieces are blue and larger ice pieces are red" (p. 159).

    Parity: exact (``np.trunc`` == MATLAB ``fix`` for the non-negative values this produces).
    """
    return np.trunc((1.0 - np.exp(-np.asarray(area, dtype=np.float64) / C2)) * C1).astype(np.int64)


def color_to_area(color, C1: float = C1_COLOR, C2: float = C2_AREA):
    """Eq. (7.6) inverted — the area that produced a given colour value, ``-round(C2 · log(1 - color/C1))``.

    MATLAB source: ``ice_shape_enhancement.m`` line 204 / 235
    ``YT{1,i} = -round(1000 * log(1 - ysh(i)/10000))``; the resulting integers are the colour-bar tick labels
    printed in Figs. 7.13, 7.15, 7.19, 7.20, 7.21, 7.26 and 7.28.  MATLAB ``round`` is half **away from zero**;
    the values here are positive so :func:`numpy.floor` of ``x + 0.5`` is used.

    Parity: exact (the printed tick integers are the reference).
    """
    from .core.matlab_compat import matlab_round

    c = np.asarray(color, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return -matlab_round(C2 * np.log(1.0 - c / C1)).astype(np.int64)


def colorbar_area_ticks(colors, n: int = 6, *, C1: float = C1_COLOR,
                        C2: float = C2_AREA) -> tuple[np.ndarray, np.ndarray]:
    """The colour-bar tick values and their Eq. (7.6)-inverted labels — ``ice_shape_enhancement.m`` l. 196–206.

    ::

        n   = 6;                                        % 8 for the FSD histogram (line 227)
        d   = fix((max(area_ice) - min(area_ice)) / n);
        ysh = min(area_ice) : d : max(area_ice);
        YT{1,i} = -round(1000 * log(1 - ysh(i)/10000));

    ``colors`` are Eq. (7.6) **colour values** (``[color_floe, color_brash]`` for the map, ``color_floe`` for the
    histogram), not areas.  Because ``d`` is truncated, ``length(ysh)`` is usually ``n + 1`` — which is why the
    book prints **seven** ticks for ``n = 6`` (Fig. 7.13: 3, 131, 277, 448, 656, 917, 1273) and **nine** for
    ``nn = 8`` (Fig. 7.21).

    Returns
    -------
    (values, labels)
        ``values`` = ``ysh`` (the colour values the ticks sit at), ``labels`` = the printed integers (areas).
        Both are empty when ``colors`` is empty.

    Parity: **near** — exact for every non-degenerate input (the output integers are the book's printed truths),
    with one documented deviation (ch07 review S6(a)): when ``d = fix((max - min)/n)`` is ``0`` (all colours
    equal, or a spread smaller than ``n``) MATLAB's ``min : 0 : max`` is the **empty** vector, so its
    ``for i = 1:length(ysh)`` loop never runs and the figure gets *no* ticks at all.  This function returns the
    single tick ``min`` instead, because an axis with zero ticks is useless and every caller here is a plot.
    """
    c = np.asarray(colors, dtype=np.float64).ravel()
    if c.size == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    lo, hi = float(c.min()), float(c.max())
    d = float(np.trunc((hi - lo) / int(n)))      # MATLAB fix()
    if d <= 0:
        # DEVIATION (ch07 review S6(a)): MATLAB `lo : 0 : hi` is empty and its YT loop never runs; we keep one
        # usable tick so the colour bar is still labelled.
        return np.array([lo]), color_to_area(np.array([lo]), C1, C2)
    # `lo : d : hi`.  `colors` are Eq. (7.6) integers and `d = fix(...)`, so lo, hi and d are all integral and
    # the count is exact integer arithmetic (ch07 review N6 — no floating-point fudge needed).
    count = (int(hi) - int(lo)) // int(d) + 1
    values = lo + d * np.arange(count, dtype=np.float64)
    return values, color_to_area(values, C1, C2)


# --------------------------------------------------------------------------------------------------------------
# Algorithm 2 / 4 / 5 — ice_shape_enhancement.m
# --------------------------------------------------------------------------------------------------------------

@dataclass
class IcePiece:
    """One element of the M-file's ``s0`` struct array (``ice_shape_enhancement.m`` lines 135–136).

    The field names are the ones ``SeaIce_Image_Structure.m`` (Appendix B) and ``sea_ice_model.m`` (§8.2) read,
    so they are kept verbatim: ``Center``, ``Area``, ``Perimeter``, ``PixelsPosition``.

    ``Center``/``Perimeter`` follow MATLAB's ``cat(1, cen.Centroid)`` (lines 130/133): one row **per
    connected component** of ``out == i``.  In every observed run each label is a single component, so
    they are a ``(2,)`` vector and a scalar; the ``k > 1`` shapes are reproduced rather than assumed away
    (ch07 review S4).
    """

    Center: np.ndarray          #: ``cat(1, cen.Centroid)`` — 1-based ``[x, y]``; ``(2,)`` for one component, ``(k, 2)`` if the label ever splits
    Area: int                   #: ``length(find(out == i))``
    Perimeter: float | np.ndarray  #: ``cat(1, per.Perimeter)`` — a float for one component, ``(k,)`` if the label ever splits
    PixelsPosition: np.ndarray  #: ``[c, r]`` — 1-based ``(x, y)`` of every pixel, in MATLAB ``find`` (column-major) order
    label: int = 0              #: the ``out`` label ``i`` this piece came from (not in the M-file; for traceability)


@dataclass
class Coverage:
    """The M-file's ``coverage`` struct (line 181) — fractions of the whole image, in ``[0, 1]``."""

    IceFloe: float
    BrashIce: float
    Slush: float
    Water: float

    def as_percent(self) -> dict[str, float]:
        """The four numbers as percentages, the way the book quotes them (e.g. 60.52 / 3.34 / 16.03 / 20.11)."""
        return {k: 100.0 * v for k, v in self.__dict__.items()}


@dataclass
class FloeSizeDistribution:
    """``[z, n] = hist(floe_area, nbins)`` plus the Eq. (7.6) bar colours (``ice_shape_enhancement.m`` l. 209–223)."""

    counts: np.ndarray
    centers: np.ndarray
    colors: np.ndarray
    nbins: int


@dataclass
class IceShapeEnhancement:
    """Everything ``ice_shape_enhancement.m`` computes (its 9 outputs plus the intermediates a test needs)."""

    out: np.ndarray            #: the M-file's ``out`` — cleaned, filled, **relabelled** ice pieces (1 … ``t``)
    index_floe: np.ndarray     #: Fig. 7.14(a) floe layer, valued with the Eq. (7.6) colour
    ice_floe: list[IcePiece]   #: per-floe ``Center`` / ``Area`` / ``Perimeter`` / ``PixelsPosition``
    index_brash: np.ndarray    #: Fig. 7.14(b) brash layer, valued with the Eq. (7.6) colour
    brash_ice: list[IcePiece]
    index_slush: np.ndarray    #: Fig. 7.14(c) slush layer (0/1)
    index_water: np.ndarray    #: Fig. 7.14(d) water layer (0/1)
    index_residue: np.ndarray  #: Fig. 7.16 residue layer (0/1)
    coverage: Coverage
    # ---- intermediates (not returned by the M-file, but assigned inside it) -----------------------------------
    l: np.ndarray              #: the labelling of light (1 … ``nn_bw``) and dark (``nn_bw`` + 1 …) pieces
    fill: np.ndarray           #: union of every piece **after hole filling only** (before the morphology)
    index: np.ndarray          #: ``index_floe + index_brash``
    ice_area: np.ndarray       #: raw piece areas, in label order (the argument of ``sort``)
    order: np.ndarray          #: ``ind`` of ``[A, ind] = sort(ice_area)`` — **0-based** here
    color_floe: np.ndarray
    color_brash: np.ndarray
    floe_cen: np.ndarray       #: (n, 2) 1-based ``[x, y]``
    brash_cen: np.ndarray
    t: int                     #: number of superimposed pieces = ``max(out)``
    nn_bw: int                 #: number of light ("SEG_L") pieces
    nn_k: int                  #: number of dark ("SEG_D") pieces after removing the overlap with the light ones
    #: The §7.2.4 FSD; ``None`` when ``nbins`` is falsy or no floe survived, where MATLAB's
    #: ``hist([], 50)`` would instead give ``zeros(1, 50)`` at centres ``1:50`` (ch07 review S6(b)).
    fsd: FloeSizeDistribution | None = None


def _find_column_major(mask: np.ndarray) -> np.ndarray:
    """MATLAB ``find(mask)`` — linear indices in **column-major** order (the port keeps them 0-based)."""
    return np.flatnonzero(np.asarray(mask).ravel(order="F"))


def _linear_to_rc(idx: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Split column-major linear indices into 0-based ``(row, col)``."""
    m = shape[0]
    return idx % m, idx // m


def _set_linear(arr: np.ndarray, idx: np.ndarray, value) -> None:
    """``arr(idx) = value`` for MATLAB (column-major) linear indices."""
    r, c = _linear_to_rc(idx, arr.shape)
    arr[r, c] = value


def _index_lists(labels: np.ndarray, n: int) -> list[np.ndarray]:
    """Column-major pixel index list of each label ``1 … n`` of ``labels``, computed in one sort."""
    flat = np.asarray(labels).ravel(order="F")
    order = np.argsort(flat, kind="stable")
    starts = np.searchsorted(flat[order], np.arange(1, n + 2))
    return [order[starts[i]:starts[i + 1]] for i in range(n)]


def _label_lists(bw: np.ndarray, conn: int = 4) -> tuple[np.ndarray, int, list[np.ndarray]]:
    """``[L, n] = bwlabel(bw, conn)`` plus the column-major pixel index list of every label (one pass)."""
    L = label_components(np.asarray(bw) != 0, conn)
    n = int(L.max())
    return L, n, _index_lists(L, n)


def _crop_margin(r: int) -> int:
    """Margin around a piece's bounding box for the per-piece morphology.

    It must exceed the reach of ``imclose``'s pre-pad + dilate + erode (``ceil(size(nhood)/2) + 2r``) so that the
    crop's own boundary can never influence the piece; ``4r + 6`` is comfortably above that for the book's
    ``r ∈ {1, 2}``.
    """
    return 4 * int(r) + 6


def _piece_morphology(shape: tuple[int, int], rows: np.ndarray, cols: np.ndarray, r: int,
                      crop: bool) -> tuple[int, int, np.ndarray, np.ndarray]:
    """``imfill`` → ``imclose`` → ``imopen`` → ``imfill`` of one piece (``ice_shape_enhancement.m`` l. 70–93).

    The M-file runs every piece on a **full-image** scratch array ``b = zeros(size(out)); b(p) = 1``.  Cropping is
    only legitimate because (a) the crop window is clipped to the image, so a piece touching the image border
    keeps the real border (``imerode`` pads with 1 *there* in both cases and ``imclose`` pre-pads with 0 in both
    cases), and (b) everywhere else the margin is wider than the operators' reach, so the crop's own boundary
    cannot reach the piece.  ``crop=False`` reproduces the M-file literally.  The equivalence is measured
    **against MATLAB's own** ``out``/``fill`` on the real 1038x394 ``sea_ice_test.jpg`` run — **1211 pieces, of
    which 106 touch an image border** (that run is fed MATLAB's own ``seg``/``bk``) — where both modes are 0 px
    (1.6 s vs 243.7 s), and on the ``border_notch`` fixtures.  Fed instead the *port's own* Algorithm-3 output the
    same image yields 1232 pieces, 84 of them touching a border, and the two modes agree there as well; the piece
    counts differ because the **input** segmentation does (ch6's ``near`` label), not because of the crop.

    Returns ``(row_offset, col_offset, filled, cleaned)`` — the two results as arrays covering the window
    ``[row_offset:, col_offset:]``, so the caller never allocates a full-image scratch array per piece.
    """
    M, N = shape
    se = strel("disk", int(r))
    if crop:
        m = _crop_margin(r)
        r0, r1 = max(0, int(rows.min()) - m), min(M, int(rows.max()) + m + 1)
        c0, c1 = max(0, int(cols.min()) - m), min(N, int(cols.max()) + m + 1)
    else:
        r0, r1, c0, c1 = 0, M, 0, N
    b = np.zeros((r1 - r0, c1 - c0), dtype=np.float64)
    b[rows - r0, cols - c0] = 1.0
    filled = imfill(b, "hole", conn=4)          # line 73
    b = imclose(filled, se)                     # line 91
    b = imopen(b, se)                           # line 92
    b = imfill(b, "hole", conn=4)               # line 93
    return r0, c0, filled, b


def ice_shape_enhancement(bk: np.ndarray, seg: np.ndarray, min_floe: float = MIN_FLOE,
                          min_brash: float = MIN_BRASH, se_th: float = SE_TH, *,
                          nbins: int = 50, crop: bool = True,
                          book_threshold: bool = False) -> IceShapeEnhancement:
    """Sea ice shape enhancement and type classification — **Algorithms 2, 4 and 5**, **Eqs. (7.5)/(7.6)**.

    Book: §7.1.4 (Algorithm 2, p. 153), §7.2.3 (Algorithm 4, p. 158), §7.2.4 (Algorithm 5, p. 160);
    Figs. 7.9, 7.12(c), 7.13, 7.14(a)–(d), 7.15, 7.16.
    MATLAB source: ``MATLAB_ROOT/ch7/Sea_Ice_Floe_Identification/ice_shape_enhancement.m`` lines 39–237 (the
    whole file; lines 183–237 are its two figures).

    Algorithm 2 (lines 39–103)::

        1: PIECES <- labeled regions in SEGMENTATION arranged from small to large
        2: BW     <- empty black image
        3: for each labeled region piece in PIECES (from small to large) do
        4:     piece <- morphological clean and fill hole
        5:     BW    <- BW with piece superimposed and labeled
        6: end for
        7: IDENTIFICATION <- labeled regions in BW

    "the arrangement of ice pieces in order of increasing size is required […] Otherwise, the smaller ice piece
    contained in a larger ice floe may not be removed." (p. 151)  Algorithm 4 is Algorithm 2 applied to the
    **light and dark pieces together** ("light ice" ``seg == 1``, "dark ice" ``seg == 0.5`` minus the overlap,
    lines 39–61) with the Eq. (7.5) adaptive disk.  Algorithm 5 (lines 106–181) splits the identified pieces into
    floe / brash on ``T_floe``, then ``PIXEL_seaice = IDENTIFICATION ∪ ICE`` (line 160 ``bk0(p) = 1``),
    ``SLUSH = PIXEL ∩ IDENTIFICATION^c`` and ``WATER = PIXEL^c``.

    Parameters
    ----------
    bk : ndarray
        ``ICE`` — the binary k-means ice image of Algorithm 3 (``KmeanGVF.bk``).
    seg : ndarray
        ``SEGMENTATION_seaice`` with the three levels 1 (light ice), 0.5 (dark ice), 0 (water) —
        ``KmeanGVF.out``.
    min_floe, min_brash : float
        ``T_floe`` and the smallest brash area (``sea_ice_demo.m``: 40 and 1).
    se_th : float
        Eq. (7.5) ``size_th`` (``sea_ice_demo.m``: 50).
    nbins : int
        FSD histogram bins (line 211 ``nbins = 50``).  ``0`` skips the histogram.
    crop : bool
        Run the per-piece morphology on a padded bounding-box crop instead of a full-image scratch array
        (see :func:`_piece_morphology`).  Mathematically identical, ~100× faster on the book image.
    book_threshold : bool
        ``False`` (default) = the **code**'s strict ``area0 > min_floe`` / ``area0 > min_brash``;
        ``True`` = the **book**'s Algorithm 5 wording "sizes equal to or larger than ``T_floe``" (``>=``).
        The two differ for pieces of exactly ``min_floe`` or ``min_brash`` pixels — with ``min_brash = 1`` the
        script's form **drops every 1-pixel piece** from all three ice layers (it still counts as slush).
        Measured on ``sea_ice_test.jpg`` **fed the port's own Algorithm-3 output**: 433 floes / 290 brash with the
        code's ``>`` against 436 / 289 with the book's ``>=`` (3 pieces of exactly 40 px move up, 2 pieces of
        exactly 1 px reappear).  Those two counts are input-dependent — on **MATLAB's** segmentation of the same
        image the code's form gives 433 / 274 — but the *difference* between the two forms is the point.
        See ``analysis/ch07.md`` risk R3.

    Returns
    -------
    :class:`IceShapeEnhancement`

    Deviations
    ----------
    * **HG1 bar colouring.**  Lines 214–224 (``ch = get(h, 'Children'); fvd = get(ch, 'Faces');
      fvcd = get(ch, 'FaceVertexCData'); …``) are HG1-only code and **error in R2025a** (``get`` on a bar handle
      returns an empty ``GraphicsPlaceholder``).  Only the *display* is affected: the numbers they colour are
      ``color(i) = fix((1 - exp(-n(i)/1000)) * 10000)``, which this port returns as
      :attr:`FloeSizeDistribution.colors` so the script can colour its bars directly.
    * **Bounding-box crop** (``crop=True``): a deliberate, proven-equivalent optimisation of the M-file's
      per-piece full-image scratch arrays, not an algorithmic change.  ``crop=False`` is the literal form.
    * ``regionprops(out == i, …)`` is evaluated with :mod:`seaice.core.regionprops` (MATLAB's own algorithms,
      exact to 1.07e-14) rather than ``skimage.measure.regionprops`` (a different perimeter and axis definition).

    Parity: **exact vs the M-file** for the arrays it computes.  Fed **MATLAB's own** ``seg``/``bk`` for the
    1038x394 ``sea_ice_test.jpg`` — **1211 pieces (982 light + 229 dark), 712 labels, 433 floes / 274 brash** —
    all nine output arrays are 0 px and ``t``/``nn_bw``/``nn_k``, the sorted areas and stable sort index, both
    colour vectors, every centroid/perimeter, the ``coverage`` struct, the 50-bin FSD and both colour-bar tick
    blocks agree to <= 1e-12 (36 controlled-fixture runs likewise 0 px).  The only inexactness the chapter carries
    comes from its **inputs**: run on the port's own Algorithm-3 output the very same code reports 1232 pieces and
    433 / 290 floe/brash instead, because ``seg``/``bk`` inherit ch6's ``near``/``approx`` labels (0.245 % / 0.125 %
    of pixels).  Those are port-input numbers, not parity numbers.
    """
    seg = np.asarray(seg, dtype=np.float64)
    bk = np.asarray(bk, dtype=np.float64)
    if bk.shape != seg.shape:
        raise ValueError("bk and seg must have the same shape")
    M, N = seg.shape

    # ---- lines 39-61: label the light pieces, then the dark pieces minus their overlap with the light ones ----
    bw = (seg == 1).astype(np.float64)          # light ice segmentation
    k = (seg == 0.5).astype(np.float64)         # dark ice segmentation
    k = k - k * bw                              # line 54 — a dark pixel that is also light belongs to the light layer

    l = np.zeros((M, N), dtype=np.float64)
    ice_area: list[int] = []
    pixel_lists: list[np.ndarray] = []

    _, nn_bw, lists_bw = _label_lists(bw, 4)
    for i, p in enumerate(lists_bw, start=1):
        _set_linear(l, p, float(i))
        ice_area.append(p.size)
        pixel_lists.append(p)

    _, nn_k, lists_k = _label_lists(k, 4)
    for i, p in enumerate(lists_k, start=1):
        _set_linear(l, p, float(i + nn_bw))
        ice_area.append(p.size)
        pixel_lists.append(p)

    ice_area_arr = np.asarray(ice_area, dtype=np.int64)
    # line 63 `[A, ind] = sort(ice_area)` — MATLAB's sort is **stable**, and the tie order decides which piece
    # is superimposed first and therefore which label survives in `out` (analysis/ch07.md risk R8).
    order = np.argsort(ice_area_arr, kind="stable")

    # ---- lines 65-103: Algorithm 2's loop, small -> large ------------------------------------------------------
    out = np.zeros((M, N), dtype=np.float64)
    fill = np.zeros((M, N), dtype=np.float64)
    t = 0
    n_pieces = min(int(l.max()), order.size)     # `for i = 1 : max(max(l))`
    for i in range(n_pieces):
        p = pixel_lists[order[i]]
        rows, cols = _linear_to_rc(p, (M, N))

        r = adaptive_se_radius(p.size, se_th)    # Eq. (7.5), lines 83-89 (uses the **raw** piece area)
        r0, c0, filled, b = _piece_morphology((M, N), rows, cols, r, crop)

        # lines 75-81: every component of the merely-filled piece contributes to `fill` (the Fig. 7.16 reference)
        fr, fc = np.nonzero(filled)
        fill[fr + r0, fc + c0] = 1.0

        # lines 95-102: superimpose with a running label; a later (larger) piece **overwrites** an earlier one
        _, kk, lists_b = _label_lists(b, 4)
        for pp in lists_b:
            if pp.size > 0:
                t += 1
                br, bc = _linear_to_rc(pp, b.shape)
                out[br + r0, bc + c0] = float(t)

    # ---- lines 106-153: Algorithm 5's floe/brash split, Eq. (7.6) colours -------------------------------------
    index_floe = np.zeros((M, N), dtype=np.float64)
    index_brash = np.zeros((M, N), dtype=np.float64)
    floe_area: list[int] = []
    brash_area: list[int] = []
    color_floe: list[int] = []
    color_brash: list[int] = []
    floe_cen: list[np.ndarray] = []
    brash_cen: list[np.ndarray] = []
    ice_floe: list[IcePiece] = []
    brash_ice: list[IcePiece] = []

    # `for i = 1 : max(max(out))` with `p = find(out == i)`.  `out` is **not** re-labelled: a later (larger)
    # piece may have overwritten an earlier label completely, which is what line 125's `if area0 ~= 0` catches.
    out_lists = _index_lists(out, t)

    for i in range(1, t + 1):
        p = out_lists[i - 1]
        area0 = int(p.size)                      # `area0 = length(p)`
        if area0 == 0:                           # line 125 — a piece completely overwritten by a larger one
            continue
        rows, cols = _linear_to_rc(p, (M, N))    # `[r, c] = find(out == i)`; `pixels = [c, r]` (1-based x, y)
        pixels = np.column_stack([cols + 1, rows + 1])

        color_label = int(size_color(area0))     # Eq. (7.6), line 127
        # `regionprops(out == i, 'centroid'/'perimeter')` (lines 129-133).  Evaluated on the bounding box padded
        # by 1 px and clipped to the image: `bwperim` and the boundary tracer only look one pixel out, and a
        # piece that touches the image border keeps that border in the crop too, so the values are identical.
        br0, br1 = max(0, int(rows.min()) - 1), min(M, int(rows.max()) + 2)
        bc0, bc1 = max(0, int(cols.min()) - 1), min(N, int(cols.max()) + 2)
        region = np.zeros((br1 - br0, bc1 - bc0), dtype=bool)
        region[rows - br0, cols - bc0] = True
        props = regionprops(region, ("Centroid", "Perimeter"))
        # `cen = cat(1, cen.Centroid)` / `per = cat(1, per.Perimeter)` (lines 130/133): a label whose pixels form
        # k > 1 connected components contributes **k** rows, and `floe_cen = [floe_cen; cen]` appends all of them.
        # (`struct('Center', cen, ...)` does not expand a numeric matrix, so `s0` stays 1x1 with a k x 2 field.)
        # k > 1 is unreachable for the shipped pipeline — a later, larger piece can only overwrite an earlier
        # label inside its own concavities/holes, which a disjoint 4-connected neighbour cannot straddle; the
        # ch07 review searched 700 fixtures and found 0 splits — but MATLAB's shape is reproduced rather than
        # assumed away, so a future caller cannot get a silently different answer.
        if props:
            offset = np.array([bc0, br0], dtype=np.float64)
            cen_all = np.array([pr.Centroid for pr in props], dtype=np.float64) + offset  # k x 2
            per_all = np.array([pr.Perimeter for pr in props], dtype=np.float64)          # k
        else:                                     # unreachable: `p` is non-empty here
            cen_all = np.array([[np.nan, np.nan]])
            per_all = np.array([np.nan])
        cen = cen_all[0] if cen_all.shape[0] == 1 else cen_all      # MATLAB 1x2 row -> (2,) for the usual k == 1
        per = float(per_all[0]) if per_all.size == 1 else per_all

        piece = IcePiece(Center=np.asarray(cen, dtype=np.float64), Area=area0, Perimeter=per,
                         PixelsPosition=pixels, label=i)

        is_floe = (area0 >= min_floe) if book_threshold else (area0 > min_floe)
        is_brash = (area0 >= min_brash) if book_threshold else (area0 > min_brash)
        if is_floe:
            index_floe[rows, cols] = color_label
            floe_area.append(area0)
            color_floe.append(color_label)
            floe_cen.append(cen_all)      # `floe_cen = [floe_cen; cen]` — k rows, not one
            ice_floe.append(piece)
        elif is_brash:
            index_brash[rows, cols] = color_label
            brash_area.append(area0)
            color_brash.append(color_label)
            brash_cen.append(cen_all)     # `brash_cen = [brash_cen; cen]` — k rows, not one
            brash_ice.append(piece)

    # ---- lines 155-169: the slush / water / residue layers ----------------------------------------------------
    index = index_floe + index_brash
    ice_mask = index != 0                        # `p = find(index ~= 0)`

    bk0 = bk.copy()
    bk0[ice_mask] = 1.0                          # PIXEL_seaice = IDENTIFICATION U ICE (Algorithm 5 step 3)
    index_slush = bk0.copy()
    index_slush[ice_mask] = 0.0                  # SLUSH = PIXEL n IDENTIFICATION^c (step 4)
    index_water = 1.0 - bk0                      # WATER = PIXEL^c (step 5)
    index_residue = np.ones((M, N), dtype=np.float64)
    index_residue[fill != 0] = 0.0
    index_residue = index_residue * index_slush  # Fig. 7.16 — slush that no piece ever covered

    # ---- lines 171-181: coverage -------------------------------------------------------------------------------
    total = float(M * N)
    coverage = Coverage(IceFloe=float(np.sum(floe_area)) / total,
                        BrashIce=float(np.sum(brash_area)) / total,
                        Slush=float(np.count_nonzero(index_slush == 1)) / total,
                        Water=float(np.count_nonzero(index_water == 1)) / total)

    # DEVIATION (ch07 review S6(b)): with no floes at all MATLAB's `hist([], 50)` (hist.m lines 81-89) returns
    # `zeros(1, 50)` counts at centres `1:50`; we return `fsd = None` so callers cannot plot a meaningless bar
    # chart of an empty ice field.  Unreachable for the book's images; `core.histogram.hist` itself does
    # reproduce the MATLAB values if called directly.
    fsd = floe_size_distribution(floe_area, nbins) if (nbins and floe_area) else None

    return IceShapeEnhancement(
        out=out, index_floe=index_floe, ice_floe=ice_floe, index_brash=index_brash, brash_ice=brash_ice,
        index_slush=index_slush, index_water=index_water, index_residue=index_residue, coverage=coverage,
        l=l, fill=fill, index=index, ice_area=ice_area_arr, order=order,
        color_floe=np.asarray(color_floe, dtype=np.int64), color_brash=np.asarray(color_brash, dtype=np.int64),
        floe_cen=(np.vstack(floe_cen) if floe_cen else np.zeros((0, 2))).astype(np.float64),
        brash_cen=(np.vstack(brash_cen) if brash_cen else np.zeros((0, 2))).astype(np.float64),
        t=t, nn_bw=nn_bw, nn_k=nn_k, fsd=fsd)


def floe_size_distribution(floe_area, nbins: int = 50) -> FloeSizeDistribution:
    """The §7.2.4 floe size distribution — ``ice_shape_enhancement.m`` lines 210–223, Figs. 7.15 / 7.21.

    ``[z, n] = hist(floe_area, nbins)`` with ``nbins = 50``, drawn as a bar chart whose bars are coloured by
    Eq. (7.6) applied to the **bin centres**: ``color(i) = fix((1 - exp(-n(i)/1000)) * 10000)`` (line 221).
    MATLAB's ``hist`` uses bin **centres**; :func:`seaice.core.histogram.hist` is the line-by-line port.

    Parity: exact (given the same ``floe_area``).
    """
    counts, centers = matlab_hist(np.asarray(floe_area, dtype=np.float64), int(nbins))
    return FloeSizeDistribution(counts=counts, centers=centers, colors=size_color(centers), nbins=int(nbins))


# ==============================================================================================================
# §7.2.2 / §7.2.4 — the two text-only algorithms that wrap the pieces above
# ==============================================================================================================

def sea_ice_edge_detection(I: np.ndarray, **params):
    """**Algorithm 3** (Book §7.2.2, p. 157) — sea ice edge detection, a thin wrapper over ch6's Algorithm 1.

    ::

        1: GVF   <- GVF derived from grayscale of input image
        2: ICE   <- binary ice image by the k-means clustering method
        3: LIGHT <- binary "light" ice image by the thresholding method
        4: DARK  <- ICE - LIGHT
        5: SEG_L <- ice floe segmentation (Algorithm 1) on LIGHT
        6: SEG_D <- ice floe segmentation (Algorithm 1) on DARK
        7: SEGMENTATION_seaice <- SEG_L + SEG_D  (labeled differently)

    Steps 1–7 are literally lines 51 / 189–208 / 297 of
    ``ch7/Sea_Ice_Floe_Identification/seaice_kmean_GVF_forenhancement.m``, which is **byte-identical** to ch6's
    copy and already ported as :func:`seaice.ch06_gvf_snake.seaice_kmean_gvf` (rule 9: reuse, do not re-port).
    The "labeled differently" of step 7 is the three-level ``out = bw1 + 0.5 * bw0``.

    Returns the :class:`seaice.ch06_gvf_snake.KmeanGVF` record; ``.out`` is ``SEGMENTATION_seaice``, ``.bk`` is
    ``ICE``, ``.bw`` is ``LIGHT``, ``.bw0`` is ``DARK``, ``.pass1.bw1`` is ``SEG_L`` and ``.pass2.bw1`` ``SEG_D``.

    Parity: **near** — inherited from :func:`seaice.ch06_gvf_snake.seaice_kmean_gvf` (the k-means *labels* are
    ``approx`` by construction).  The residual depends on the input: 0.049 % of ``out`` on ch6's
    ``alg_seg_gray.jpg`` (ch06's measurement), and **0.245 % of ``seg`` / 0.125 % of ``bk``** when it is called,
    as ch7 calls it, on ch7's own re-encoded ``sea_ice_test.jpg`` (re-measured against MATLAB here).
    """
    from .ch06_gvf_snake import seaice_kmean_gvf

    return seaice_kmean_gvf(I, **params)


def sea_ice_shape_enhancement(bk: np.ndarray, seg: np.ndarray, **kwargs) -> IceShapeEnhancement:
    """**Algorithm 4** (Book §7.2.3, p. 158) — Algorithm 2 applied to the *merged* light + dark labelling.

    ::

        1: PIECES_seaice         <- labeled regions in SEGMENTATION_seaice
        2: IDENTIFICATION_seaice <- ice shape enhancement (Algorithm 2)

    "the ice shape enhancement algorithm should be performed to all the detected ice pieces, both the 'light ice'
    and the 'dark ice' […] if the ice shape enhancement algorithm is performed to the two layers independently,
    overlapping identifications would be produced."  That is exactly what
    :func:`ice_shape_enhancement` already does (its lines 39–61 build one labelling ``l`` out of both layers), so
    this is an alias kept for traceability of the algorithm number.

    Parity: exact vs the M-file.
    """
    return ice_shape_enhancement(bk, seg, **kwargs)


def ice_types_classification(identification: np.ndarray, ice: np.ndarray, T_floe: float = MIN_FLOE, *,
                             min_brash: float = MIN_BRASH,
                             book_threshold: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """**Algorithm 5** (Book §7.2.4, p. 160) written out on its own, from the pseudocode.

    ::

        1: FLOE  <- labeled regions in IDENTIFICATION_seaice with the sizes equal to or larger than T_floe
        2: BRASH <- labeled regions in IDENTIFICATION_seaice with the sizes smaller than T_floe
        3: PIXEL_seaice <- IDENTIFICATION_seaice U ICE
        4: SLUSH <- PIXEL_seaice n (IDENTIFICATION_seaice)^c
        5: WATER <- (PIXEL_seaice)^c

    "brash ice is 'accumulation of floating ice made up of fragments not more than 2 m across'" (p. 160);
    ``T_floe`` may be given as a pixel number, an area or a characteristic length — here it is the pixel area,
    as in ``ice_shape_enhancement.m``.

    ``book_threshold=True`` (the default **here**, because this function *is* the book's pseudocode) uses the
    printed ``>=``; the shipped code writes ``if area0 > min_floe`` / ``elseif area0 > min_brash``, which is what
    :func:`ice_shape_enhancement` reproduces by default.  With ``min_brash = 1`` the two differ: the code's form
    drops every 1-pixel piece from FLOE **and** BRASH (it survives only as slush).

    Parameters
    ----------
    identification : ndarray
        The label image ``out`` of Algorithm 4.
    ice : ndarray
        ``ICE`` — the binary k-means ice image of Algorithm 3.

    Returns
    -------
    (floe, brash, slush, water) : four 0/1 float layers.

    Parity: **reimplemented** (book form).  The script form is :func:`ice_shape_enhancement`, verified against
    the ``.m`` file.
    """
    L = np.asarray(identification)
    ice = np.asarray(ice, dtype=np.float64)
    n = int(L.max()) if L.size else 0
    floe = np.zeros(L.shape, dtype=np.float64)
    brash = np.zeros(L.shape, dtype=np.float64)
    for i in range(1, n + 1):
        m = L == i
        area = int(m.sum())
        if area == 0:
            continue
        if (area >= T_floe) if book_threshold else (area > T_floe):
            floe[m] = 1.0
        elif (area >= min_brash) if book_threshold else (area > min_brash):
            brash[m] = 1.0
    ident = (floe + brash) != 0
    pixel = np.maximum(ice, ident.astype(np.float64))   # step 3: PIXEL = IDENTIFICATION U ICE
    slush = pixel * (~ident).astype(np.float64)         # step 4
    water = 1.0 - pixel                                 # step 5
    return floe, brash, slush, water


# ==============================================================================================================
# §7.3.1 Distorted overall sea ice image processing — Algorithm 6 (p. 168)
# ==============================================================================================================

@dataclass
class TileRecord:
    """One sub-image of :func:`local_segmentation`.

    ``bounds`` is the sub-image itself ``(r0, r1, c0, c1)`` (half-open, 0-based) and ``core`` the part that
    survives after "the overlapping parts are removed"; the two coincide only for a single-tile grid.
    """

    index: tuple[int, int]
    bounds: tuple[int, int, int, int]
    core: tuple[int, int, int, int]
    n_light: int = 0
    n_dark: int = 0
    error: str | None = None


@dataclass
class LocalSegmentation:
    """Result of :func:`local_segmentation` — the stitched ``SEG`` of Algorithm 6 step 5."""

    seg: np.ndarray             #: stitched three-level segmentation (1 light / 0.5 dark / 0 water)
    bk: np.ndarray              #: stitched k-means ice mask ``ICE`` (Algorithm 5 needs it)
    tiles: list[TileRecord]
    tile: tuple[int, int]
    overlap: tuple[int, int]
    merge: str
    rows: int                   #: number of tile rows
    cols: int                   #: number of tile columns


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Start offsets of overlapping windows of size ``tile`` covering ``[0, length)`` with ``overlap`` shared px.

    The last window is pulled back so that it ends exactly at ``length`` (its overlap with the previous one is
    therefore >= ``overlap``); a single window is used when ``tile >= length``.
    """
    tile = min(int(tile), int(length))
    step = max(1, tile - int(overlap))
    starts = list(range(0, max(1, length - tile + 1), step))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def tile_grid(shape: tuple[int, int], tile, overlap) -> tuple[list[TileRecord], int, int]:
    """The overlapping sub-image grid of §7.3.1.1 and the disjoint cores that survive the overlap removal.

    Book: §7.3.1.1, p. 163–164, Fig. 7.18 — "The image is first divided into smaller regions such that each
    region can be analyzed individually.  To avoid image border effects as discussed in Section 6.5.3, it is
    recommended that the neighboring subregions overlap sufficiently. […] After that, the overlapping parts are
    removed and the subsegmented images are merged by an image stitching method."

    The book fixes **no** tile size and **no** overlap size, so both are parameters.  The cores are cut at the
    **midpoint of each overlap**, which makes them an exact partition of the image (every pixel is claimed by
    exactly one sub-image) and keeps every kept pixel as far as possible from a sub-image border — the point of
    the overlap in the first place.

    Parameters
    ----------
    shape : (M, N)
    tile, overlap : int or (int, int)
        Sub-image size and overlap in pixels; a scalar applies to both axes.

    Returns
    -------
    (tiles, n_tile_rows, n_tile_cols)

    Parity: **reimplemented** — there is no MATLAB file for §7.3.1 and the book states none of these sizes.
    """
    M, N = int(shape[0]), int(shape[1])
    th, tw = (tile, tile) if np.isscalar(tile) else (int(tile[0]), int(tile[1]))
    oh, ow = (overlap, overlap) if np.isscalar(overlap) else (int(overlap[0]), int(overlap[1]))
    rs, cs = _tile_starts(M, th, oh), _tile_starts(N, tw, ow)
    th, tw = min(th, M), min(tw, N)

    def cores(starts: list[int], size: int, total: int) -> list[tuple[int, int]]:
        out = []
        for i, st in enumerate(starts):
            lo = 0 if i == 0 else (st + starts[i - 1] + size) // 2
            hi = total if i == len(starts) - 1 else (starts[i + 1] + st + size) // 2
            out.append((lo, hi))
        return out

    rc, cc = cores(rs, th, M), cores(cs, tw, N)
    tiles = [TileRecord(index=(i, j), bounds=(r, r + th, c, c + tw),
                        core=(rc[i][0], rc[i][1], cc[j][0], cc[j][1]))
             for i, r in enumerate(rs) for j, c in enumerate(cs)]
    return tiles, len(rs), len(cs)


def local_segmentation(I: np.ndarray, tile=256, overlap=64, *, merge: str = "crop",
                       progress=None, **params) -> LocalSegmentation:
    """**Algorithm 6** steps 1–5 / §7.3.1.1 — local (tiled) sea ice edge detection with stitching.

    Book: §7.3.1.1 "Local processing" (pp. 163–164) and **Algorithm 6** "Overall sea ice floe and brash
    identification algorithm" (p. 168), illustrated by **Fig. 7.18**::

        1: SUB <- sub-images divided from the input image
        2: for each sub-image sub in SUB do
        3:     seg <- ice edge detection (Algorithm 3) on sub
        4: end for
        5: SEG <- overall segmentation image with all seg stitched
        6: SEG <- geometric calibrated SEG                       <- section 7.3.1.2 / Appendix A, ch10
        7: ID  <- sea ice shape enhancement (Algorithm 4) on SEG <- ice_shape_enhancement()
        8-10: FLOE / BRASH / DISTRIBUTION                        <- ice_shape_enhancement() / Algorithm 5

    Why local at all: "the GVF capture range derived by a uniform parameter sometimes cannot represent an
    overall ice image and should be adjusted according to each sub-image […] (but at the expense of more
    processing time and possibly manual intervention)" (p. 163).  The overlap exists to avoid the §6.5.3 image
    border effect.

    **There is no MATLAB file for this section** and the book specifies **neither the tile size, nor the overlap
    size, nor the stitching rule** — all three are parameters here, and the defaults (256 px tiles, 64 px
    overlap, midpoint crop) are this port's choice, not the book's.

    Parameters
    ----------
    I : ndarray
        RGB or grayscale sea ice image.
    tile, overlap : int or (int, int)
        Sub-image size and overlap in pixels (see :func:`tile_grid`).
    merge : {'crop', 'max', 'first'}
        How step 5 stitches the sub-segmentations.

        * ``'crop'`` (default) — the prose: "the overlapping parts are removed", each sub-image contributing only
          its core (:func:`tile_grid`), which is an exact partition;
        * ``'max'`` — the word printed on Fig. 7.18's last arrow, "Superimpose": every sub-image writes its whole
          extent and the **larger** level wins, so light ice (1) beats dark ice (0.5) beats water (0).  This keeps
          the border-effect pixels of every tile and is therefore *not* what the prose describes;
        * ``'first'`` — the first sub-image to claim a pixel keeps it (a raster-order stitch).
    progress : callable, optional
        Called as ``progress(k, n_tiles, TileRecord)`` after each sub-image.
    **params
        Passed to :func:`sea_ice_edge_detection` (= :func:`seaice.ch06_gvf_snake.seaice_kmean_gvf`), e.g.
        ``Num``, ``iter``, ``kms0``.  Per-sub-image tuning of these — which is what p. 163 recommends — is left
        to the caller: run this function once per parameter set, or call it per tile.

    Returns
    -------
    :class:`LocalSegmentation`

    Notes
    -----
    A degenerate sub-image (constant intensity, or one on which the k-means of Algorithm 3 cannot form ``kms0``
    non-empty clusters) is recorded with its exception text in :attr:`TileRecord.error` and contributes zeros;
    nothing is silently swallowed — the caller is expected to report those tiles.

    Parity: **reimplemented** (`analysis/ch07.md` risk R13).  Step 6, the geometric calibration of §7.3.1.2,
    is **not** performed here: it depends on Appendix A.1.1's camera model (shooting angle 20 degrees, FOV 46
    degrees) and lands in ch10; :func:`resample_categorical` is the only piece of it ch07 owns.
    """
    if merge not in ("crop", "max", "first"):
        raise ValueError("merge must be 'crop', 'max' or 'first'")
    I = np.asarray(I)
    M, N = I.shape[:2]
    tiles, n_rows, n_cols = tile_grid((M, N), tile, overlap)

    seg = np.zeros((M, N), dtype=np.float64)
    bk = np.zeros((M, N), dtype=np.float64)
    claimed = np.zeros((M, N), dtype=bool)

    for kk, rec in enumerate(tiles):
        r0, r1, c0, c1 = rec.bounds
        sub = I[r0:r1, c0:c1]
        try:
            res = sea_ice_edge_detection(sub, **params)
            sub_seg, sub_bk = res.out, res.bk
            rec.n_light = int(label_components(sub_seg == 1, 4).max())
            rec.n_dark = int(label_components(sub_seg == 0.5, 4).max())
        except Exception as exc:                       # degenerate sub-image — recorded, never hidden
            rec.error = f"{type(exc).__name__}: {exc}"
            sub_seg = np.zeros((r1 - r0, c1 - c0), dtype=np.float64)
            sub_bk = np.zeros_like(sub_seg)

        if merge == "crop":                            # step 5, the prose: drop the overlapping parts
            k0, k1, k2, k3 = rec.core
            seg[k0:k1, k2:k3] = sub_seg[k0 - r0:k1 - r0, k2 - c0:k3 - c0]
            bk[k0:k1, k2:k3] = sub_bk[k0 - r0:k1 - r0, k2 - c0:k3 - c0]
        elif merge == "max":                           # step 5, Fig. 7.18's "Superimpose"
            np.maximum(seg[r0:r1, c0:c1], sub_seg, out=seg[r0:r1, c0:c1])
            np.maximum(bk[r0:r1, c0:c1], sub_bk, out=bk[r0:r1, c0:c1])
        else:                                          # 'first'
            free = ~claimed[r0:r1, c0:c1]
            seg[r0:r1, c0:c1][free] = sub_seg[free]
            bk[r0:r1, c0:c1][free] = sub_bk[free]
            claimed[r0:r1, c0:c1] = True
        if progress is not None:
            progress(kk + 1, len(tiles), rec)

    th = tiles[0].bounds[1] - tiles[0].bounds[0]
    tw = tiles[0].bounds[3] - tiles[0].bounds[2]
    oh, ow = (overlap, overlap) if np.isscalar(overlap) else (int(overlap[0]), int(overlap[1]))
    return LocalSegmentation(seg=seg, bk=bk, tiles=tiles, tile=(th, tw), overlap=(oh, ow), merge=merge,
                             rows=n_rows, cols=n_cols)


def resample_categorical(L: np.ndarray, u: np.ndarray, v: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Nearest-neighbour resampling of a **categorical** image — the only piece of §7.3.1.2 that lives in ch07.

    Book: §7.3.1.2 "Geometric calibration" (pp. 164–167).  "the interpolation method for the image resampling
    should be the **nearest** neighbour interpolation, since the pixel values of the segmented image represent
    the ice categories" — an averaging interpolant would invent categories that do not exist (a 0.75 between
    "dark ice" 0.5 and "light ice" 1).  The book also fixes *where* in the pipeline it belongs: after Algorithm 3
    and **before** Algorithm 4, because calibrating the raw image blurs the boundaries and costs GVF iterations,
    while calibrating after Algorithms 4–5 mis-classifies the far-range floes as brash.

    The camera model itself (shooting angle 20 degrees, FOV 46 degrees, Appendix A.1.1) is **not** implemented
    here — ch07 must not contain a second rectifier (`analysis/ch07.md` risk R14); it lands in ch10.  This
    function only performs the resampling step, given the sampling grid the rectifier produces.

    Parameters
    ----------
    L : ndarray
        Categorical image (e.g. the three-level ``SEG``).
    u, v : ndarray
        0-based (row, column) sample coordinates, as :func:`seaice.core.interp.interp2` takes them.
    fill : float
        Value for samples that fall outside ``L`` (``interp2`` returns NaN there).

    Parity: **exact** for the resampling (it is :func:`seaice.core.interp.interp2` with ``'nearest'``, verified
    against MATLAB in ch02, ties rounded half away from zero); the rectification itself is **deferred to ch10**.
    """
    from .core.interp import interp2

    out = interp2(np.asarray(L, dtype=np.float64), u, v, "nearest")
    return np.where(np.isnan(out), fill, out)
