"""Chapter 6 — GVF snake-based ice floe boundary identification and ice image segmentation.

Book: Chapter 6, printed pages 109–144.  §6.1 traditional parametric snake (Eqs. 6.1–6.40), §6.2 gradient
vector flow (Eqs. 6.41–6.56), §6.3 contour initialization (§6.3.3 Eqs. 6.57/6.58), §6.4 ice image segmentation
(**Algorithm 1**, p. 136), §6.5 discussion (stopping criterion, capture range, border effects).

MATLAB sources (``MATLAB_ROOT/ch6/``):

* ``Sea_Ice_Floe_Identification/GVF_distance.m``            → :func:`gvf_distance`
* ``Sea_Ice_Floe_Identification/seaice_kmean_GVF_forenhancement.m`` → :func:`seaice_kmean_gvf`
* ``Sea_Ice_Floe_Identification/sea_ice_demo.m``            → ``scripts/ch06_sea_ice_demo.py``
* ``for test/dist.m``                                       → ``scripts/ch06_dist.py``
* ``for test/for_test.m``                                   → ``scripts/ch06_for_test.py``

The Xu & Prince toolbox itself (``GVF.m``, ``snakedeform.m``, ``snakeinterp.m``, ``BoundMirror*.m``,
``gradient2.m``, ``gaussian*.m``, ``xconv2.m``) lives in :mod:`seaice.core.snake`, which carries the attribution.

Licence note for the authors' own files
---------------------------------------
``GVF_distance.m``, ``seaice_kmean_GVF_forenhancement.m``, ``sea_ice_demo.m``, ``dist.m`` and ``for_test.m`` are
by **Qin Zhang**, NTNU Department of Marine Technology, project "Arctic DP" (RCN no. 199567); their headers state
that the code is *"only available for academic non-commercial use"* and ask for a citation of Q. Zhang and
R. Skjetne, "Image Processing for Identification of Sea-Ice Floes and the Floe Size Distributions", *IEEE
Transactions on Geoscience and Remote Sensing* **53**(5):2913–2924, 2015.  The functions below are a
re-implementation written for study of the book.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core.clustering import kmeans_lloyd
from .core.connectivity import bwareaopen, label_components
from .core.distance import bwdist
from .core.filters import fspecial, imfilter
from .core.matlab_compat import matlab_round, rgb2gray_matlab
from .core.morphology import imdilate, imregionalmin, strel
from .core.polygon import clip_polygon_rect
from .core.regionprops import regionprops
from .core.snake import gaussian_blur, gradient2, gradient2_magnitude, gvf, snakedeform, snakeinterp
from .core.threshold import graythresh, im2bw

__all__ = [
    "BOOK_PARAMS", "SeedRecord", "PassRecord", "GVFDistance", "KmeanGVF", "ContourInit",
    "external_energy", "edge_energy", "line_energy", "termination_energy",
    "traditional_snake", "initialize_contours", "gvf_distance", "seaice_kmean_gvf",
    "gvf_force_field", "component_criteria",
]

#: The parameter blocks printed in the three drivers (``sea_ice_demo.m`` lines 9–46, ``for_test.m`` lines 4–34,
#: ``dist.m`` lines 3–4) plus the two book footnotes that fix them (footnote 2 p. 121: alpha = 0.05, beta = 0;
#: footnote 3 p. 125: mu = 0.1; footnote 4 p. 135: the city-block radius divisor sqrt(2)).
BOOK_PARAMS: dict[str, dict] = {
    "sea_ice_demo": dict(kms0=3, Ra_min=10, Ra=2500, Rc=0.9, Rl=2, se_radius=3, Num=500, iter=100,
                         se_th=50, min_floe=40, min_brash=1, sigma=0, GradientOn=1, GVFOn=1, mu=0.1,
                         alpha=0.05, beta=0.0, gamma=1.0, kappa=0.5, Dmin=0.0, Dmax=1.0, timer=1),
    "for_test": dict(sigma=0, GradientOn=1, GVFOn=1, Num=150, mu=0.1, iter=50, alpha=0.05, beta=0.0,
                     gamma=1.0, kappa=0.5, Dmin=0.0, Dmax=1.0, Ra_min=20, Ra=1000, Rc=0.9, Rl=2,
                     se_radius=3, r=20, x0=80, y0=40, timer=1, d=10, order=2),
    "dist": dict(se_radius=5, r=15),
    "figures": dict(sigma_fig_6_4=5.0, sigma_fig_6_7=4.0, gvf_iters_fig_6_16=(5, 30, 100, 250),
                    gvf_iters_fig_6_17=(30, 250), gvf_iters_fig_6_18=(800, 80), gvf_iters_fig_6_19=(600, 60)),
}

#: ``t = 0:0.05:6.28`` — 126 points, last one 6.25, so the initial circle is **not** closed (gap 0.033 rad).
#: ``snakeinterp`` closes it implicitly through ``[x; x(1)]``.
CIRCLE_T = np.arange(0.0, 6.28 + 1e-12, 0.05)

#: Radius divisor for the city-block distance transform (book footnote 4, p. 135).
CITYBLOCK_RADIUS_DIVISOR = np.sqrt(2.0)


# ==============================================================================================================
# §6.1.1.2 External energies — Eqs. (6.6)-(6.13)
# ==============================================================================================================

def line_energy(I: np.ndarray) -> np.ndarray:
    """Book Eq. (6.6) ``E_line = I(x, y)``.

    "Depending on the sign of ``γ_line``, the snake will be attracted either to light lines or dark lines"
    (§6.1.1.2, p. 113).  No MATLAB file — implemented from the equation.  Parity: reimplemented.
    """
    return np.asarray(I, dtype=np.float64)


def edge_energy(I: np.ndarray, sigma: float = 0.0) -> np.ndarray:
    """Book Eq. (6.7) ``E_edge = −|∇I(x, y)|²`` (Eq. 6.11 with ``sigma > 0``: ``−|∇(G_σ * I)|²``).

    The minus sign turns "maximise the gradient magnitude" into "minimise the energy"; smoothing by the Gaussian
    of Eq. (6.10) enlarges the capture range at the cost of blurring the boundary (Fig. 6.4(b) uses σ = 5,
    Fig. 6.7(b) σ = 4).  No MATLAB file — implemented from the equations, using
    :func:`seaice.core.snake.gaussian_blur` (Xu & Prince's ``gaussianBlur.m``) for ``G_σ * I``.
    Parity: reimplemented.
    """
    f = np.asarray(I, dtype=np.float64)
    if sigma:
        f = gaussian_blur(f, sigma)
    gx, gy = gradient2(f)
    return -(gx * gx + gy * gy)


def termination_energy(I: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Book Eq. (6.8) ``E_term = ∂θ/∂n⊥ = (C_yy C_x² − 2 C_xy C_x C_y + C_xx C_y²)/(C_x² + C_y²)^{3/2}``.

    The curvature of the level contours of the smoothed image ``C = G_σ * I`` (Eq. 6.10); it detects line
    terminations and corners and is "rarely used" ([111], p. 114).  No MATLAB file — implemented from the
    equation; derivatives come from :func:`seaice.core.snake.gradient2`.  Parity: reimplemented.
    """
    C = gaussian_blur(np.asarray(I, dtype=np.float64), sigma) if sigma else np.asarray(I, dtype=np.float64)
    Cx, Cy = gradient2(C)
    Cxx, Cxy = gradient2(Cx)
    Cyx, Cyy = gradient2(Cy)
    denom = (Cx * Cx + Cy * Cy) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        E = (Cyy * Cx * Cx - 2.0 * Cxy * Cx * Cy + Cxx * Cy * Cy) / denom
    return np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)


def external_energy(I: np.ndarray, kind: str = "edge", gamma: float = 1.0, sigma: float = 0.0,
                    gamma_line: float = 0.0, gamma_edge: float = 1.0, gamma_term: float = 0.0) -> np.ndarray:
    """Book Eqs. (6.4)–(6.5), (6.9), (6.11)–(6.13) — the external (image) energy of a snake.

    ``kind``:

    ``'edge'``      Eq. (6.9) ``E_ext = γ E_edge = −γ|∇I|²`` (Eq. 6.11 when ``sigma > 0``)
    ``'gray'``      Eq. (6.12) ``E_ext = −γ I`` (binary images)
    ``'gray_blur'`` Eq. (6.13) ``E_ext = −γ G_σ * I``
    ``'image'``     Eq. (6.5) ``E_image = γ_line E_line + γ_edge E_edge + γ_term E_term``

    ``γ`` must not be 0 — "otherwise the snake will shrink to a point" (p. 115).  No MATLAB file — implemented
    from the equations.  Parity: reimplemented.
    """
    I = np.asarray(I, dtype=np.float64)
    if gamma == 0:
        raise ValueError("gamma must not be 0 (p. 115: the snake would shrink to a point)")
    if kind == "edge":
        return gamma * edge_energy(I, sigma)
    if kind == "gray":
        return -gamma * I
    if kind == "gray_blur":
        return -gamma * (gaussian_blur(I, sigma) if sigma else I)
    if kind == "image":
        return gamma * (gamma_line * line_energy(I) + gamma_edge * edge_energy(I, sigma)
                        + gamma_term * termination_energy(I, sigma if sigma else 1.0))
    raise ValueError("kind must be 'edge', 'gray', 'gray_blur' or 'image'")


# ==============================================================================================================
# §6.1.2 The traditional (non-GVF) snake — Eqs. (6.29), (6.37)-(6.40)
# ==============================================================================================================

def traditional_snake(I: np.ndarray, x, y, *, alpha: float = 0.05, beta: float = 0.0, gamma: float = 1.0,
                      kappa: float = 0.5, sigma: float = 0.0, iters: int = 50, blocks: int = 1,
                      dmax: float = 1.0, dmin: float = 0.0, energy_kind: str = "edge",
                      solver: str = "auto") -> tuple[np.ndarray, np.ndarray, list]:
    """The traditional parametric snake of §6.1: external force ``F_ext = −∇E_ext`` (Eq. 6.29) instead of the
    GVF field.

    Book: §6.1.1.2 (Eqs. 6.9/6.11), §6.1.2 (Eqs. 6.37–6.40), Figs. 6.4–6.7.  With ``E_ext = −γ|∇(G_σ * I)|²``
    the force ``−∇E_ext`` points towards the edges but has a very small capture range and no component that
    pulls the contour into a boundary concavity (§6.1.3, Fig. 6.7) — exactly the two limitations GVF removes.

    No MATLAB file implements this (the shipped code always uses the GVF field or, with ``GVFOn = 0``, the raw
    ``gradient2(f2)``); it is built from the equations on top of :func:`seaice.core.snake.snakedeform`.

    Returns ``(x, y, history)`` with one ``(x, y)`` pair per deformation block.  Parity: reimplemented.
    """
    E = external_energy(np.asarray(I, dtype=np.float64), kind=energy_kind, sigma=sigma)
    ex, ey = gradient2(E)
    fx, fy = -ex, -ey  # Eq. (6.29) F_ext = -grad E_ext
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    x, y = snakeinterp(x, y, dmax, dmin)
    history = [(x.copy(), y.copy())]
    for _ in range(int(blocks)):
        x, y = snakedeform(x, y, alpha, beta, gamma, kappa, fx, fy, iters, solver=solver)
        x, y = snakeinterp(x, y, dmax, dmin)
        history.append((x.copy(), y.copy()))
    return x, y, history


def gvf_force_field(gray: np.ndarray, *, sigma: float = 0.0, gradient_on: bool = True, gvf_on: bool = True,
                    num: int = 500, mu: float = 0.1,
                    normalize: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stages 4–7 shared by every ch6 driver: edge map → GVF field → unit-normalised force ``(px, py)``.

    MATLAB source: ``GVF_distance.m`` lines 50–76 (identical in ``seaice_kmean_GVF_forenhancement.m`` lines
    69–95 and ``for_test.m`` lines 44–70)::

        if sigma ~= 0, f = gaussianBlur(I, sigma); else f = I; end
        if GradientOn, f2 = abs(gradient2(double(f))); else f2 = f; end
        if GVFOn, [u, v] = GVF(f2, mu, Num); else [u, v] = gradient2(f2); end
        mag = sqrt(u.*u + v.*v); px = u ./ (mag + 1e-10); py = v ./ (mag + 1e-10);

    Book: Eq. (6.7)/(6.11) for the edge map ``f = |∇I|`` and Eqs. (6.41)/(6.53) for the GVF diffusion.

    # DEVIATION: `exact` to the script, but the **book's Eq. (6.56) uses the GVF field v itself**, not
    # ``v/|v|``.  The unit normalisation is Xu & Prince's demo convention: it makes the effective external force
    # weight ``kappa`` the same everywhere and keeps a non-zero force even in flat regions.  ``normalize=False``
    # gives the book form.

    Returns
    -------
    (f2, u, v, px, py) : the edge map, the raw GVF field and the (normalised) external force field.
    """
    f = gaussian_blur(gray, sigma) if sigma else np.asarray(gray, dtype=np.float64)
    f2 = gradient2_magnitude(np.asarray(f, dtype=np.float64)) if gradient_on else np.asarray(f, dtype=np.float64)
    if gvf_on:
        u, v = gvf(f2, mu, num)
    else:
        u, v = gradient2(f2)
    if normalize:
        mag = np.sqrt(u * u + v * v)
        px = u / (mag + 1e-10)
        py = v / (mag + 1e-10)
    else:
        px, py = u.copy(), v.copy()
    return f2, u, v, px, py


# ==============================================================================================================
# §6.3.3 / Algorithm 1 — contour initialization from the distance transform
# ==============================================================================================================

@dataclass
class ContourInit:
    """Everything §6.3.3 computes on a binary mask: distance map, regional maxima, merged seeds and radii."""

    bw: np.ndarray
    img_dist: np.ndarray
    minima_map: np.ndarray          # imregionalmin(-D) with -Inf background  (= regional maxima of D)
    dis: np.ndarray                 # Dis_img .* bw  -- the local maxima inside the mask (double, 0/1)
    dis_dilated: np.ndarray         # after imdilate(dis, se): the merged "seed" regions
    label: np.ndarray
    num: int
    centroids: np.ndarray           # (num, 2) MATLAB (x, y), 1-based
    radii: np.ndarray               # (num,) initial circle radii
    contours: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)


def initialize_contours(bw: np.ndarray, *, metric: str = "cityblock", se_radius: int = 3,
                        radius_divisor: float = CITYBLOCK_RADIUS_DIVISOR, min_radius: float = 2.0,
                        form: str = "script", conn_seed: int = 8, abs_radius: bool = True) -> ContourInit:
    """§6.3.3 / Algorithm 1 steps 3–8: distance transform → regional maxima → seeds → initial circles.

    Book: §6.3.3 pp. 133–136 and Algorithm 1 p. 136.  Fig. 6.14 is the worked 8×8 example: the city-block
    distance transform of an ice blob has **one regional maximum made of three local maxima** of value 3; the
    maxima are dilated to merge those within ``T_seed``, the centre of each dilated region is a *seed*, and the
    radius of the initial circle is the distance value at the seed — divided by ``sqrt(2)`` because the
    city-block metric is used (**footnote 4, p. 135**), so the circle stays strictly inside the floe.

    MATLAB source: ``GVF_distance.m`` lines 113–130 (identical in ``seaice_kmean_GVF_forenhancement.m``)::

        img_Dist = bwdist(~bw2, 'cityblock');
        imgDist  = -img_Dist;  imgDist(~bw2) = -inf;
        Dis_img  = imregionalmin(imgDist);
        dis      = Dis_img .* bw2;
        dis      = imdilate(dis, se);            % se = strel('disk', 3)
        [label1, num1] = bwlabel(dis, 8);
        cen = regionprops(label1 == n1, 'centroid');
        r   = img_Dist(round(cen(2)), round(cen(1))) / sqrt(2);   if r == 0, r = 2; end
        x = cen(1) + r*cos(t);  y = cen(2) + r*sin(t);            % t = 0:0.05:6.28

    Parameters
    ----------
    form : {'script', 'book'}
        ``'script'`` (default) is the code above.  ``'book'`` evaluates the text's own wording — the regional
        **maxima** of ``D`` through Eq. (6.57) ``M_max = I − R^D_I(I − 1)`` — and is provably the same set:
        the regional maxima of ``D`` inside the mask are the regional minima of ``−D`` with a ``−Inf``
        background (verified on Fig. 6.14).
    abs_radius : bool
        ``True`` reproduces ``seaice_kmean_GVF_forenhancement.m`` line 146 / ``dist.m`` line 41
        (``abs(img_Dist(...)/sqrt(2))``); ``GVF_distance.m`` line 125 omits the ``abs``.  Harmless — ``bwdist``
        is never negative.

    Parity: exact (script form).
    """
    bw = np.asarray(bw) != 0
    # Review S3: `core.distance.bwdist` returns float64, not float32 -- MATLAB returns `single` and we keep
    # float64 deliberately (see its module docstring).  Harmless for 'cityblock', whose values are integers.
    img_dist = bwdist(~bw, metric)
    imgd = -img_dist.astype(np.float64)
    imgd[~bw] = -np.inf
    if form == "script":
        minima_map = imregionalmin(imgd)
    elif form == "book":
        from .core.morphology import regional_maxima_by_reconstruction
        d = img_dist.astype(np.float64).copy()
        d[~bw] = -np.inf  # the background must not join the maxima plateau
        minima_map = regional_maxima_by_reconstruction(d)
    else:
        raise ValueError("form must be 'script' or 'book'")
    dis = (minima_map & bw).astype(np.float64)  # logical .* logical -> double in MATLAB
    se = strel("disk", se_radius)
    dis_dilated = imdilate(dis, se)
    label = label_components(dis_dilated != 0, conn_seed)
    num = int(label.max())
    stats = regionprops(label, ("Centroid",))
    centroids = np.array([s.Centroid for s in stats], dtype=np.float64).reshape(-1, 2)
    radii = np.zeros(num, dtype=np.float64)
    contours: list[tuple[np.ndarray, np.ndarray]] = []
    M, N = bw.shape
    for n in range(num):
        cx, cy = centroids[n]
        rr = int(matlab_round(cy)) - 1
        cc = int(matlab_round(cx)) - 1
        val = float(img_dist[min(max(rr, 0), M - 1), min(max(cc, 0), N - 1)]) / radius_divisor
        r = abs(val) if abs_radius else val
        if r == 0:
            r = min_radius
        radii[n] = r
        contours.append((cx + r * np.cos(CIRCLE_T), cy + r * np.sin(CIRCLE_T)))
    return ContourInit(bw=bw, img_dist=img_dist, minima_map=minima_map, dis=dis, dis_dilated=dis_dilated,
                       label=label, num=num, centroids=centroids, radii=radii, contours=contours)


# ==============================================================================================================
# The two pipelines
# ==============================================================================================================

@dataclass
class SeedRecord:
    """One contour of Algorithm 1 (steps 6–11), with every intermediate the verifier compares."""

    index: int
    centroid: np.ndarray
    radius: float
    x_circle: np.ndarray
    y_circle: np.ndarray
    x_interp: np.ndarray
    y_interp: np.ndarray
    x_clip: np.ndarray
    y_clip: np.ndarray
    blocks: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    x_final: np.ndarray | None = None
    y_final: np.ndarray | None = None
    burnt_rc: np.ndarray | None = None  # 0-based (row, col) pairs written to 0


@dataclass
class PassRecord:
    """One iteration of the ``for time = 1:timer`` loop (the outer re-segmentation loop of ch9 p. 205)."""

    time: int
    label: np.ndarray
    num: int
    area: np.ndarray
    solidity: np.ndarray
    major: np.ndarray
    minor: np.ndarray
    rl: np.ndarray
    k: np.ndarray                    # 0-based indices of the components that fail a criterion
    bw2: np.ndarray | None = None
    init: ContourInit | None = None
    seeds: list[SeedRecord] = field(default_factory=list)
    stopped: bool = False            # True when `length(k) == 0` broke the loop


@dataclass
class GVFDistance:
    """Return value of :func:`gvf_distance` — every array of ``GVF_distance.m``."""

    gray: np.ndarray
    level: float
    bw: np.ndarray
    f2: np.ndarray
    u: np.ndarray
    v: np.ndarray
    px: np.ndarray
    py: np.ndarray
    passes: list[PassRecord]
    bw1: np.ndarray
    n_seeds: int = 0
    n_seeds_run: int = 0


@dataclass
class KmeanGVF:
    """Return value of :func:`seaice_kmean_gvf` — the two passes of ``seaice_kmean_GVF_forenhancement.m``."""

    gray: np.ndarray
    bw: np.ndarray
    f2: np.ndarray
    u: np.ndarray
    v: np.ndarray
    px: np.ndarray
    py: np.ndarray
    pass1: GVFDistance
    map0: np.ndarray
    s0: np.ndarray
    ind0: np.ndarray
    bk: np.ndarray
    bw0_raw: np.ndarray
    bw0: np.ndarray
    n_negative: int
    pass2: GVFDistance
    out: np.ndarray


def component_criteria(bw: np.ndarray, Ra: float, Rc: float, Rl: float,
                       conn: int = 4) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray, np.ndarray]:
    """The three re-segmentation criteria of book Ch. 9 p. 205, as ``GVF_distance.m`` lines 80–102 implement them.

    ``[label, num] = bwlabel(bw1, 4)``; per component ``Area``, ``Solidity``, ``MajorAxisLength``,
    ``MinorAxisLength``; ``rl = l ./ w``; ``k = unique([find(a > Ra); find(rc < Rc); find(rl > Rl)])``.

    Criterion 1 = the floe area exceeds the threshold; criterion 2 = ``Solidity`` (the convex hull *is* the
    minimum-area bounding polygon); criterion 3 is written in the book as the length-to-width ratio of the
    minimum-area bounding **rectangle** but implemented with the **ellipse** axis ratio — a documented deviation
    of the shipped code from the text (:func:`seaice.core.polygon.minboundrect` computes the book's version).

    Returns ``(label, num, area, solidity, major, minor, rl, k)`` with ``k`` 0-based and sorted (MATLAB's
    ``unique``).  Parity: exact.
    """
    label = label_components(np.asarray(bw) != 0, conn)
    num = int(label.max())
    if num == 0:
        z = np.zeros(0)
        return label, 0, z, z, z, z, z, np.zeros(0, dtype=np.int64)
    stats = regionprops(label, ("Area", "Solidity", "MajorAxisLength", "MinorAxisLength"))
    area = np.array([s.Area for s in stats])
    solidity = np.array([s.Solidity for s in stats])
    major = np.array([s.MajorAxisLength for s in stats])
    minor = np.array([s.MinorAxisLength for s in stats])
    with np.errstate(divide="ignore", invalid="ignore"):
        rl = major / minor
    k = np.unique(np.concatenate([np.nonzero(area > Ra)[0], np.nonzero(solidity < Rc)[0],
                                  np.nonzero(rl > Rl)[0]]))
    return label, num, area, solidity, major, minor, rl, k.astype(np.int64)


def _run_snake_passes(bw1: np.ndarray, px: np.ndarray, py: np.ndarray, *, iter: int, alpha: float, beta: float,
                      gamma: float, kappa: float, Dmin: float, Dmax: float, Ra_min: float, Ra: float,
                      Rc: float, Rl: float, se_radius: int, timer: int, max_seeds: int | None,
                      keep_history: bool, progress,
                      solver: str = "auto") -> tuple[np.ndarray, list[PassRecord], int, int]:
    """The ``for time = 1:timer`` body shared verbatim by ``GVF_distance.m`` and both passes of
    ``seaice_kmean_GVF_forenhancement.m`` (lines 79–161 / 100–183 / 212–295)."""
    bw1 = np.asarray(bw1).astype(np.float64).copy() if not np.asarray(bw1).dtype == np.bool_ \
        else np.asarray(bw1).copy()
    s1, s2 = bw1.shape
    passes: list[PassRecord] = []
    n_seeds_total = 0
    n_seeds_run = 0
    for time in range(1, int(timer) + 1):
        label, num, area, solidity, major, minor, rl, k = component_criteria(bw1 != 0, Ra, Rc, Rl, conn=4)
        rec = PassRecord(time=time, label=label, num=num, area=area, solidity=solidity, major=major,
                         minor=minor, rl=rl, k=k)
        if k.size == 0:
            rec.stopped = True
            passes.append(rec)
            break
        bw2 = np.zeros((s1, s2), dtype=np.float64)
        for m in k:
            bw2[label == (m + 1)] = 1.0
        bw2 = bwareaopen(bw2 != 0, int(Ra_min))  # bwareaopen(bw2, Ra_min) -- 8-conn default
        rec.bw2 = bw2
        init = initialize_contours(bw2, se_radius=se_radius)
        rec.init = init
        n_seeds_total += init.num
        run = init.num if max_seeds is None else min(init.num, int(max_seeds))
        n_seeds_run += run
        for n1 in range(run):
            if progress is not None:
                progress(n1, run)
            cx, cy = init.centroids[n1]
            r = float(init.radii[n1])
            x = cx + r * np.cos(CIRCLE_T)
            y = cy + r * np.sin(CIRCLE_T)
            xi, yi = snakeinterp(x, y, Dmax, Dmin)
            # polybool('intersection', s_2, s_1, x, y): clip to the image rectangle (see core.polygon)
            xc, yc = clip_polygon_rect(xi, yi, (0.0, float(s2)), (0.0, float(s1)))
            if xc.size < 3:
                continue
            seed = SeedRecord(index=n1, centroid=init.centroids[n1].copy(), radius=r, x_circle=x, y_circle=y,
                              x_interp=xi, y_interp=yi, x_clip=xc, y_clip=yc)
            xs, ys = xc, yc
            n_blocks = int(np.ceil(iter / 5))
            floor5 = int(np.floor(iter / 5))
            for i in range(1, n_blocks + 1):
                steps = 5 if i <= floor5 else iter - floor5 * 5
                xs, ys = snakedeform(xs, ys, alpha, beta, gamma, kappa, px, py, steps, solver=solver)
                xs, ys = snakeinterp(xs, ys, Dmax, Dmin)
                if keep_history:
                    seed.blocks.append((xs.copy(), ys.copy()))
            seed.x_final, seed.y_final = xs, ys
            xx = np.ceil(xs).astype(np.int64)
            yy = np.ceil(ys).astype(np.int64)
            ok = (xx <= s2) & (yy <= s1)  # the script has no lower guard; the polybool clip supplies it
            ok &= (xx >= 1) & (yy >= 1)
            bw1[yy[ok] - 1, xx[ok] - 1] = 0
            seed.burnt_rc = np.column_stack([yy[ok] - 1, xx[ok] - 1])
            rec.seeds.append(seed)
        passes.append(rec)
    return bw1, passes, n_seeds_total, n_seeds_run


def gvf_distance(I: np.ndarray, *, sigma: float = 0.0, GradientOn: bool = True, GVFOn: bool = True,
                 Num: int = 500, mu: float = 0.1, iter: int = 100, alpha: float = 0.05, beta: float = 0.0,
                 gamma: float = 1.0, kappa: float = 0.5, Dmin: float = 0.0, Dmax: float = 1.0,
                 Ra_min: float = 10, Ra: float = 2500, Rc: float = 0.9, Rl: float = 2, se_radius: int = 3,
                 timer: int = 1, normalize: bool = True, max_seeds: int | None = None,
                 keep_history: bool = True, progress=None, solver: str = "auto",
                 field_cache: tuple[np.ndarray, ...] | None = None) -> GVFDistance:
    """Port of ``MATLAB_ROOT/ch6/Sea_Ice_Floe_Identification/GVF_distance.m`` — **Algorithm 1** (§6.4, p. 136).

    ::

        1: GVF          <- GVF derived from the grayscale input image
        2: SEGMENTATION <- binary ice image
        3: D            <- distance map of SEGMENTATION
        4: M            <- regional maxima of D
        5: S            <- seeds: merge the regional maxima of M within T_seed
        6: for each seed s in S:
        7:     r <- D(s)                      (/sqrt(2) for city-block, footnote 4)
        8:     c <- initial contour = circle at s of radius r
        9:     B <- boundary from the GVF snake on c
        10:    SEGMENTATION <- SEGMENTATION with B superimposed
        11: end for
        12: return SEGMENTATION

    The shipped code implements a **superset** of Algorithm 1: an outer ``for time = 1:timer`` loop that only
    re-segments the components failing the three criteria of book Ch. 9 p. 205 (see :func:`component_criteria`).
    Every ch6 driver sets ``timer = 1``, so one pass is run.

    Parameters follow the M-file's argument list; ``se_radius`` replaces the ``se = strel('disk', 3)`` object
    (MATLAB's octagonal disk, **5×5 / 25 px** — :func:`seaice.core.morphology.strel`; corrected 2026-09-10 from
    ``reference/ch06/strel.mat``, review finding S2.  This is the only printed evidence for ``T_seed``, so the
    number matters to ch7/ch9).

    Extra keyword arguments not in the M-file
    -----------------------------------------
    ``normalize`` — see :func:`gvf_force_field`; ``max_seeds`` — stop after N contours (runtime cap, reported in
    the result); ``keep_history`` — record the contour after each 5-iteration block; ``progress`` — callback
    ``(i, n)``; ``solver`` — passed to :func:`seaice.core.snake.snakedeform` (``'dense'`` = MATLAB's literal
    ``inv``, ``'auto'`` = the FFT solve for long contours; see the DEVIATION note there — with ``'auto'`` a
    handful of contour points can land on the other side of a ``ceil`` boundary, which is why the burnt-pixel
    count may differ by ~1 in 1000 from the dense run); ``field_cache`` — reuse a computed ``(f2, u, v, px, py)``.

    Returns
    -------
    :class:`GVFDistance` with every intermediate array of the M-file.  Parity target: exact.
    """
    I = np.asarray(I)
    gray = rgb2gray_matlab(I) if I.ndim == 3 else I
    level, _ = graythresh(gray)
    bw = im2bw(gray, level)
    if field_cache is None:
        f2, u, v, px, py = gvf_force_field(gray, sigma=sigma, gradient_on=bool(GradientOn),
                                           gvf_on=bool(GVFOn), num=Num, mu=mu, normalize=normalize)
    else:
        f2, u, v, px, py = field_cache
    bw1, passes, n_seeds, n_run = _run_snake_passes(
        bw, px, py, iter=iter, alpha=alpha, beta=beta, gamma=gamma, kappa=kappa, Dmin=Dmin, Dmax=Dmax,
        Ra_min=Ra_min, Ra=Ra, Rc=Rc, Rl=Rl, se_radius=se_radius, timer=timer, max_seeds=max_seeds,
        keep_history=keep_history, progress=progress, solver=solver)
    return GVFDistance(gray=gray, level=float(level), bw=bw, f2=f2, u=u, v=v, px=px, py=py, passes=passes,
                       bw1=bw1 != 0, n_seeds=n_seeds, n_seeds_run=n_run)


def seaice_kmean_gvf(I: np.ndarray, *, kms0: int = 3, sigma: float = 0.0, GradientOn: bool = True,
                     GVFOn: bool = True, Num: int = 500, mu: float = 0.1, iter: int = 100,
                     alpha: float = 0.05, beta: float = 0.0, gamma: float = 1.0, kappa: float = 0.5,
                     Dmin: float = 0.0, Dmax: float = 1.0, Ra_min: float = 10, Ra: float = 2500,
                     Rc: float = 0.9, Rl: float = 2, se_radius: int = 3, timer: int = 1,
                     normalize: bool = True, max_seeds: int | None = None, keep_history: bool = True,
                     kmeans_impl: str = "lloyd++", kmeans_seed: int = 0, strict_bwareaopen: bool = False,
                     progress=None, solver: str = "auto") -> KmeanGVF:
    """Port of ``seaice_kmean_GVF_forenhancement.m`` — :func:`gvf_distance` run **twice**, once on the Otsu mask
    and once on the k-means residual, giving a three-level segmentation.

    Book: §6.3/§6.4 combined with the k-means ice detection of §3.2.  The GVF field is computed **once** (lines
    69–95) and shared by both passes.  Pass 1 (lines 99–183) is exactly ``GVF_distance`` on
    ``im2bw(I, graythresh(I))``.  Then (lines 189–208)::

        map0 = kmeans(double(I(:)), kms0, 'EmptyAction', 'singleton');
        s0(i) = sum(ima .* (map0 == i)) / sum(map0 == i);   [A0, ind0] = sort(s0);
        bw_kmeans = ones(...);  bw_kmeans(map0 == ind0(1)) = 0;   % only the darkest cluster is water
        bk = reshape(bw_kmeans, si);   bw0 = bw_kmeans - bw;   bw0 = bwareaopen(bw0, Ra_min, 4);

    Pass 2 (lines 212–295) runs the same body on ``bw0``, and line 297 returns
    ``out = bw1 + bw0 * 0.5`` — **three levels**: 1 = bright ice, 0.5 = dark/slush ice, 0 = water.

    Parameters
    ----------
    kmeans_impl : {'lloyd++', 'sklearn'} or callable
        The Statistics Toolbox ``kmeans`` (defaults verified in R2025a ``stats/stats/kmeans.m``:
        ``Distance 'sqeuclidean'``, ``Start 'plus'`` = k-means++, ``Replicates 1``, ``MaxIter 100``) is random.
        ``'lloyd++'`` uses :func:`seaice.core.clustering.kmeans_lloyd` with k-means++ initialisation — the same
        algorithm, a different RNG stream.  ch6 ships **no** ``kmeans.m`` shadow (unlike ch3), so the authors'
        histogram k-means is *not* what this call resolves to.

        # DEVIATION: `approx` for the cluster **labels** (any k-means is initialisation dependent); the data are
        # 1-D with at most 256 distinct values and k = 3, so the optimum is essentially unique and the derived
        # mask ``bk`` is expected to agree with MATLAB's to within a handful of pixels.  Compare *sorted cluster
        # centres*, never label numbers.
    strict_bwareaopen : bool
        ``bw0 = bw_kmeans − bw`` is a **double** and is ``−1`` wherever Otsu calls a pixel ice but k-means calls
        it water.  MATLAB's ``bwareaopen`` binarises with ``~= 0``, so those pixels become *foreground* of
        ``bw0``.  ``False`` (default) reproduces that literally; ``True`` keeps only the ``+1`` pixels.  The
        number of ``−1`` pixels is reported as ``KmeanGVF.n_negative``.

    Returns
    -------
    :class:`KmeanGVF`.  Parity: **near** -- k-means cluster centres and the ``bk``/``bw0`` masks match MATLAB
    exactly (0 px), while the three-level ``out`` differs on 0.177 % of pixels through the same ``ceil`` and
    single-precision effects as :func:`gvf_distance` (`reports/ch06_verification.md` Deviations 1-3; corrected
    2026-09-10, review finding S6).
    """
    I = np.asarray(I)
    gray = rgb2gray_matlab(I) if I.ndim == 3 else I
    level, _ = graythresh(gray)
    bw = im2bw(gray, level)
    s1, s2 = bw.shape
    f2, u, v, px, py = gvf_force_field(gray, sigma=sigma, gradient_on=bool(GradientOn), gvf_on=bool(GVFOn),
                                       num=Num, mu=mu, normalize=normalize)
    common = dict(sigma=sigma, GradientOn=GradientOn, GVFOn=GVFOn, Num=Num, mu=mu, iter=iter, alpha=alpha,
                  beta=beta, gamma=gamma, kappa=kappa, Dmin=Dmin, Dmax=Dmax, Ra_min=Ra_min, Ra=Ra, Rc=Rc,
                  Rl=Rl, se_radius=se_radius, timer=timer, normalize=normalize, max_seeds=max_seeds,
                  keep_history=keep_history, progress=progress, solver=solver,
                  field_cache=(f2, u, v, px, py))
    pass1 = gvf_distance(gray, **common)

    # ---- k-means on the raw gray values (column-major, like MATLAB's I(:)) --------------------------------
    ima = gray.astype(np.float64).ravel(order="F")
    if callable(kmeans_impl):
        map0 = np.asarray(kmeans_impl(ima, kms0)).ravel()
    elif kmeans_impl == "sklearn":
        from sklearn.cluster import KMeans
        map0 = KMeans(n_clusters=kms0, n_init=10, random_state=kmeans_seed).fit_predict(ima[:, None])
    elif kmeans_impl == "lloyd++":
        map0 = kmeans_lloyd(ima[:, None], kms0, init="kmeans++", seed=kmeans_seed, max_iter=100).labels
    else:
        raise ValueError("kmeans_impl must be 'lloyd++', 'sklearn' or a callable")
    s0 = np.array([ima[map0 == i].mean() if np.any(map0 == i) else np.nan for i in range(kms0)])
    ind0 = np.argsort(s0, kind="stable")
    bw_kmeans = np.ones(ima.shape, dtype=np.float64)
    bw_kmeans[map0 == ind0[0]] = 0.0  # only the darkest cluster is water
    bk = bw_kmeans.reshape((s1, s2), order="F")
    bw0_raw = bk - bw.astype(np.float64)
    n_negative = int((bw0_raw < 0).sum())
    src = (bw0_raw > 0) if strict_bwareaopen else (bw0_raw != 0)
    bw0 = bwareaopen(src, int(Ra_min), 4)

    pass2_bw1, passes2, n_seeds2, n_run2 = _run_snake_passes(
        bw0, px, py, iter=iter, alpha=alpha, beta=beta, gamma=gamma, kappa=kappa, Dmin=Dmin, Dmax=Dmax,
        Ra_min=Ra_min, Ra=Ra, Rc=Rc, Rl=Rl, se_radius=se_radius, timer=timer, max_seeds=max_seeds,
        keep_history=keep_history, progress=progress, solver=solver)
    pass2 = GVFDistance(gray=gray, level=float(level), bw=bw0, f2=f2, u=u, v=v, px=px, py=py, passes=passes2,
                        bw1=pass2_bw1 != 0, n_seeds=n_seeds2, n_seeds_run=n_run2)

    out = pass1.bw1.astype(np.float64) + pass2.bw1.astype(np.float64) * 0.5
    return KmeanGVF(gray=gray, bw=bw, f2=f2, u=u, v=v, px=px, py=py, pass1=pass1, map0=map0, s0=s0, ind0=ind0,
                    bk=bk, bw0_raw=bw0_raw, bw0=bw0, n_negative=n_negative, pass2=pass2, out=out)


def sobel_edge_overlay(gray: np.ndarray) -> np.ndarray:
    """Convenience for the figures: the Sobel gradient magnitude of a gray image in 0–255 units
    (``imfilter(double(I), fspecial('sobel'))``), used only for side-by-side displays of the GVF edge map."""
    g = np.asarray(gray, dtype=np.float64)
    w = fspecial("sobel")
    gy = imfilter(g, w)
    gx = imfilter(g, w.T)
    return np.hypot(gx, gy)
