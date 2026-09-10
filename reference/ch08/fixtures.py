"""Deterministic fixtures shared by ``reference/ch08/make_refs.py`` (MATLAB side) and ``tests/test_ch08.py``.

Every fixture is built here **once** so that both sides consume the identical numbers: the Python tests call
these functions, and ``make_refs.py`` writes the same objects into ``reference/ch08/*_inputs.mat`` for MATLAB
to ``load``.

Two families:

* ``model_fixtures()`` — inputs for ``sea_ice_model.m`` / ``SeaIce_Image_Structure.m``.  Besides a real
  227-floe / 240-brash **window of the shipped §8.3 field**, they contain constructed cases that discriminate
  behaviours no real datum exercises:

  - ``nested``   — a small square strictly **inside** a big one, plus a third square that genuinely crosses it.
    The M-file's ``if xx ~= NaN`` (= ``if ~isempty(polyxpoly(...))``) must report the nested pair as
    **non**-overlapping and the crossing pair as overlapping.  Both directions in one fixture.
  - ``center2x2`` — a brash piece whose ``Center`` is a **2×2** matrix (4 of the shipped 3452 are), with one
    partner reachable only under MATLAB's column-major ``c(1), c(2)`` = ``(x1, x2)`` reading and one partner
    reachable only under the naive ``(x1, y1)`` reading.  Whichever list MATLAB returns settles it.
  - ``touching`` — two squares sharing exactly one boundary point and two whose bounding boxes touch but whose
    boundaries do not, for the AABB prefilter.

* ``hist_fixtures()`` / ``fit_fixtures()`` — areas for ``color_hist*.m`` and ``(x, y)`` pairs for the two
  orphan fitting files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "data/book/ch08/MCD/IceImage_290915_2_jpg.0000179.mat"

#: The window of the shipped §8.3 field used as the ``sea_ice_model`` L2 fixture (1-based, inclusive).
WINDOW = (1, 200, 1, 200)


def have_book_data() -> bool:
    return SHIPPED.exists()


def _square(x0: float, y0: float, w: float, h: float) -> np.ndarray:
    """Every integer pixel of an axis-aligned rectangle, as 1-based ``[x, y]`` rows (column-major order)."""
    xs = np.arange(x0, x0 + w)
    ys = np.arange(y0, y0 + h)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    # MATLAB `find` order = column-major: all rows of column 1, then column 2, ...
    return np.column_stack([X.ravel(order="F"), Y.ravel(order="F")]).astype(np.float64)


def _piece(pixels: np.ndarray) -> dict:
    pixels = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
    # NOTE: `Area` must be a **double** - ch7 produces it as `length(find(out == i))`, and MATLAB integer
    # division would silently round `sqrt(aa/pi)` if it arrived as int64 from scipy.io.savemat.
    return {"Center": pixels.mean(axis=0), "Area": float(pixels.shape[0]),
            "Perimeter": float(pixels.shape[0]), "PixelsPosition": pixels}


def _brash(center, area: float, pixels=None) -> dict:
    c = np.atleast_2d(np.asarray(center, dtype=np.float64))
    if pixels is None:
        pixels = np.array([[float(c.ravel(order="F")[0]), float(c.ravel(order="F")[1])]])
    return {"Center": c if c.shape[0] > 1 else c.ravel(), "Area": float(area),
            "Perimeter": float(area), "PixelsPosition": np.atleast_2d(np.asarray(pixels, dtype=np.float64))}


def model_fixtures() -> dict[str, dict]:
    """``{name: {'ice_floe': [...], 'brash_ice': [...], 'shape': (rows, cols)}}`` — all constructed cases."""
    out: dict[str, dict] = {}

    # --- nested: B is strictly inside A (containment WITHOUT a boundary crossing); C genuinely crosses A ------
    A = _square(10, 10, 60, 60)          # x 10..69, y 10..69
    B = _square(30, 30, 10, 10)          # x 30..39, y 30..39  -> strictly inside A's hull
    C = _square(60, 60, 30, 30)          # x 60..89, y 60..89  -> hull crosses A's hull
    out["nested"] = {"ice_floe": [_piece(A), _piece(B), _piece(C)],
                     "brash_ice": [_brash([35.0, 35.0], 4.0),        # 1: disk inside A, no crossing
                                   _brash([69.0, 40.0], 200.0)],     # 2: disk straddling A's right edge
                     "shape": (110, 110)}

    # --- center2x2: MATLAB reads c(1), c(2) column-major out of a 2x2 Center ---------------------------------
    # Center = [[100, 300], [200, 400]]  ->  column-major flat = [100, 200, 300, 400]
    #   MATLAB           c(1), c(2) = (100, 200)
    #   naive row read   (x1, y1)   = (100, 300)
    # Piece 2 sits at (100, 200) and piece 3 at (100, 300); each is reachable under exactly one reading.
    out["center2x2"] = {
        # two floes of *different* area: `SeaIce_Image_Structure.m` line 97 computes
        # `inter = fix((max-min)/50)`, and an all-equal-area fixture would make `min:0:max` empty.
        "ice_floe": [_piece(_square(150, 150, 6, 6)), _piece(_square(300, 300, 10, 10))],
        "brash_ice": [_brash([[100.0, 300.0], [200.0, 400.0]], 100.0, pixels=[[100.0, 300.0]]),
                      _brash([100.0, 200.0], 100.0),
                      _brash([100.0, 300.0], 100.0)],
        "shape": (450, 450)}

    # --- touching: boundaries sharing a single point vs bounding boxes that touch without a crossing ---------
    out["touching"] = {
        "ice_floe": [_piece(_square(10, 10, 20, 20)),      # 1: x 10..29
                     _piece(_square(29, 29, 20, 20)),      # 2: shares the corner pixel (29,29) with 1
                     _piece(_square(30, 10, 30, 20))],     # 3: AABB disjoint from 1 in x (30 > 29)
        "brash_ice": [_brash([29.0, 29.0], 4.0)],
        "shape": (80, 80)}
    return out


def window_fixture(window: tuple[int, int, int, int] = WINDOW) -> dict | None:
    """The real §8.3 pieces whose pixels lie entirely inside ``window`` (returns ``None`` without the book data).

    Selecting by pixel window keeps the fixture self-consistent (a piece is either wholly in or wholly out) and
    is *not* expected to reproduce the shipped ``Intersect`` lists — pieces outside the window are simply not
    there.  It is the L2 fixture: MATLAB and Python are run on exactly these pieces.
    """
    if not have_book_data():
        return None
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from seaice.core.icestruct import load_iceimage_mat

    ice = load_iceimage_mat(SHIPPED)
    x0, x1, y0, y1 = window

    def inside(p) -> bool:
        px = np.atleast_2d(np.asarray(p))
        return bool(px[:, 0].min() >= x0 and px[:, 0].max() <= x1
                    and px[:, 1].min() >= y0 and px[:, 1].max() <= y1)

    floes, fidx = [], []
    for i, f in enumerate(ice.Floe):
        if inside(f.Pixels):
            fidx.append(i)
            floes.append({"Center": np.asarray(f.Center, dtype=np.float64), "Area": float(f.Area),
                          "Perimeter": float(f.Perimeter),
                          "PixelsPosition": np.asarray(f.Pixels, dtype=np.float64)})
    brash, bidx = [], []
    for i, b in enumerate(ice.Brash):
        if inside(b.Pixels):
            bidx.append(i)
            brash.append({"Center": np.asarray(b.Center, dtype=np.float64), "Area": float(b.Area),
                          "Perimeter": float(b.Area),
                          "PixelsPosition": np.asarray(b.Pixels, dtype=np.float64)})
    return {"ice_floe": floes, "brash_ice": brash, "shape": (y1, x1),
            "floe_indices": np.array(fidx, dtype=np.int64), "brash_indices": np.array(bidx, dtype=np.int64)}


def hist_fixtures() -> dict[str, np.ndarray]:
    """Area vectors for ``color_hist.m`` / ``color_hist_comparison.m``.

    ``edges`` deliberately puts values **below** ``min_x`` and **above** ``max_x`` and exactly on the internal
    bin midpoints, because ``hist``'s outer bins are unbounded and its edges are shifted by ``eps(edge)``.
    """
    return {
        "edges": np.array([1.0, 19.0, 20.0, 54.0, 55.0, 55.5, 56.0, 90.0, 3479.0, 3480.0, 3481.0,
                           3515.0, 3516.0, 6000.0, 12000.0]),
        "small": np.array([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0, 233.0]),
        "one": np.array([700.0]),
    }


def fit_fixtures() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``(x, y)`` pairs for the two orphan fitting files.

    ``exact`` is drawn *exactly* from ``y = 3 x^-1.5`` (the fit must recover the parameters), ``noisy`` adds a
    deterministic perturbation, and ``survival`` is a genuine cumulative distribution so the Weibull branch of
    ``three_fitting_method_and_plotting.m`` has something to bite on.
    """
    x = np.arange(1.0, 51.0)
    exact = (x, 3.0 * x ** (-1.5))
    pert = 1.0 + 0.05 * np.cos(np.arange(x.size))
    noisy = (x, 3.0 * x ** (-1.5) * pert)
    n = 200
    s = np.sort(2.0 + 30.0 * (np.arange(1, n + 1) / n) ** 2.5)
    nc = (n - np.searchsorted(s, s, side="left")) / n
    survival = (s, nc)
    return {"exact": exact, "noisy": noisy, "survival": survival}
