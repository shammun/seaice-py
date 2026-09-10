"""Controlled fixtures for the chapter-7 MATLAB references.

Everything Python and MATLAB must both see bit-for-bit is built here and written to
``reference/ch07/inputs.mat`` by ``make_refs.py`` (there is no ``push()`` into ``matlab -batch``).

Three families:

* ``imfill_cases``    — the ``core.morphology.imfill`` branch matrix (logical / double / uint8 / single / int16 /
  grayscale, conn 4 vs 8, border holes, nested holes, all-foreground, all-background, ``±Inf`` plants).
* ``hist_cases``      — MATLAB ``hist`` semantics (scalar ``n``, explicit centres, ``min == max``, values outside
  the centre range, the ``edges + eps(edges)`` boundary, ``n = 1``, empty, non-finite).
* ``seg_fixtures``    — controlled ``(bk, seg)`` pairs for ``ice_shape_enhancement.m``, one per branch listed in
  ``analysis/ch07.md`` §6 item 3.
"""
from __future__ import annotations

import numpy as np


# ==============================================================================================================
# imfill
# ==============================================================================================================
def _square_with_hole(n: int = 9) -> np.ndarray:
    a = np.zeros((n, n), bool)
    a[1:-1, 1:-1] = True
    a[3:6, 3:6] = False
    return a


def _nested() -> np.ndarray:
    a = np.zeros((15, 15), bool)
    a[1:14, 1:14] = True
    a[3:12, 3:12] = False
    a[5:10, 5:10] = True
    a[6:9, 6:9] = False
    return a


def _border_hole() -> np.ndarray:
    """A 'hole' that is open to the image border — must NOT be filled."""
    a = np.zeros((9, 9), bool)
    a[0:8, 1:8] = True
    a[0:4, 3:5] = False          # channel running to the top border
    return a


def _diagonal_hole() -> np.ndarray:
    """A hole whose background pixels reach the outside only diagonally: conn 4 fills it, conn 8 does not."""
    a = np.zeros((7, 7), bool)
    a[1:6, 1:6] = True
    a[2, 2] = a[3, 3] = a[4, 4] = False
    a[2, 4] = a[4, 2] = False
    return a


def _ring8() -> np.ndarray:
    """An 8-connected (but not 4-connected) ring: its interior is a hole for a 4-connected background flood."""
    a = np.zeros((9, 9), bool)
    rr = [2, 2, 3, 4, 5, 6, 6, 6, 5, 4, 3, 2]
    cc = [3, 4, 5, 6, 6, 5, 4, 3, 2, 2, 2, 3]
    a[rr, cc] = True
    return a


def imfill_cases() -> dict[str, np.ndarray]:
    """``{name: array}`` — every case is run through MATLAB ``imfill`` with conn 4 **and** 8."""
    rng = np.random.default_rng(7)
    sq = _square_with_hole()
    cases: dict[str, np.ndarray] = {
        "logical_hole": sq,
        "double_hole": sq.astype(np.float64),                 # the ch7 call: imfill(double, 'hole')
        "uint8_hole01": sq.astype(np.uint8),
        "uint8_hole255": (sq.astype(np.uint8) * 255),
        "single_hole": sq.astype(np.float32),
        "nested": _nested(),
        "nested_double": _nested().astype(np.float64),
        "border_hole": _border_hole(),
        "diagonal_hole": _diagonal_hole(),
        "ring8": _ring8(),
        "all_true": np.ones((6, 7), bool),
        "all_false": np.zeros((6, 7), bool),
        "single_pixel": (lambda a: (a.__setitem__((2, 2), True), a)[1])(np.zeros((5, 5), bool)),
        "random_binary": rng.random((30, 40)) > 0.45,
        "checker": (np.indices((10, 10)).sum(axis=0) % 2).astype(bool),
        # --- genuinely grayscale (the 'grayscale branch' of imfill.m) ---
        "gray_wells": _gray_wells(),
        "gray_uint8": (_gray_wells() * 40 + 20).astype(np.uint8),
        "gray_single": _gray_wells().astype(np.float32),
        "int16_signed": (_gray_wells() * 1000 - 2000).astype(np.int16),
        "inf_plant": _inf_plant(),
    }
    return cases


def _gray_wells() -> np.ndarray:
    """A smooth grayscale surface with two closed basins (the classic grayscale ``imfill`` case)."""
    y, x = np.mgrid[0:20, 0:24].astype(np.float64)
    g = 3.0 + 0.5 * np.sin(x / 3.0) + 0.5 * np.cos(y / 4.0)
    g[5:9, 5:9] = 0.2      # basin 1
    g[12:16, 14:20] = 0.5  # basin 2
    g[0, :] = 4.0
    return g


def _inf_plant() -> np.ndarray:
    g = _gray_wells().copy()
    g[2, 2] = np.inf
    g[17, 20] = -np.inf
    return g


# ==============================================================================================================
# hist
# ==============================================================================================================
def hist_cases() -> dict[str, tuple[np.ndarray, object]]:
    """``{name: (y, bins)}``; ``bins`` is a scalar ``n`` or an explicit **centre** vector."""
    rng = np.random.default_rng(11)
    return {
        "small_n3": (np.array([1.0, 2, 3, 4, 5]), 3),
        "small_n10": (np.array([1.0, 2, 3, 4, 5]), 10),
        "edge_ties": (np.array([0.0, 1, 2, 3, 4]), 4),          # values sit exactly on the internal edges
        "constant": (np.array([5.0, 5, 5, 5]), 4),               # min == max branch
        "constant_n1": (np.array([5.0, 5, 5, 5]), 1),
        "single_value": (np.array([7.0]), 5),
        "n1": (np.array([1.0, 2, 3, 9]), 1),
        "negative": (np.array([-5.0, -2.5, 0.0, 2.5, 5.0]), 4),
        "random50": (rng.normal(500, 200, 137), 50),
        "areas_like_floe": (np.concatenate([rng.integers(41, 3000, 400).astype(float),
                                            np.array([41.0, 41.0, 2999.0])]), 50),
        "centres_uniform": (rng.integers(1, 6000, 300).astype(float), np.arange(20.0, 3501.0, 70.0)),
        "centres_outside": (np.array([-100.0, 0.0, 25.0, 3600.0, 9000.0]), np.arange(20.0, 3501.0, 70.0)),
        "centres_nonuniform": (np.array([0.0, 1, 2, 5, 9, 20]), np.array([0.0, 3.0, 10.0, 25.0])),
        "centres_two": (np.array([0.0, 1, 2, 3]), np.array([1.0, 2.0])),
        "with_nan_inf": (np.array([1.0, 2.0, np.nan, 3.0, np.inf, -np.inf, 4.0]), 4),
        "big_ints": (np.arange(1.0, 101.0), 7),
    }


# ==============================================================================================================
# ice_shape_enhancement (bk, seg) fixtures
# ==============================================================================================================
def _blank(shape=(40, 60)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)


def seg_fixtures() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{name: (bk, seg)}`` — ``seg`` has the three levels {0, 0.5, 1}, ``bk`` is the k-means ice mask."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # --- A. every branch in one image ---------------------------------------------------------------------
    seg = _blank((40, 60))
    seg[2:20, 2:24] = 1.0            # big light floe (396 px >> se_th)
    seg[8:14, 8:16] = 0.0            # ... with a hole ...
    seg[10:12, 10:13] = 1.0          # ... containing a small piece (the reason Algorithm 2 sorts small->large)
    seg[0:5, 40:47] = 1.0            # touches the top border (35 px < se_th)
    seg[35:40, 0:6] = 1.0            # touches the bottom/left border
    seg[24:31, 30:37] = 0.5          # a dark piece, 49 px = se_th - 1
    seg[24:32, 44:52] = 0.5          # a dark piece, 64 px > se_th
    seg[30:32, 44:46] = 1.0          # ... overlapping a light one (line 54 `k = k - k.*bw`)
    seg[37, 30] = 1.0                # a 1-pixel piece  (dropped by the strict `> min_brash`)
    seg[37, 40] = 0.5                # a 1-pixel dark piece
    seg[20, 55] = 1.0                # another 1-px piece
    seg[16:18, 55:57] = 1.0          # a 4-px piece that imopen(disk 1) may delete
    bk = (seg > 0).astype(np.float64)
    bk[33:36, 20:30] = 1.0           # slush: k-means ice that no piece covers
    out["mixed"] = (bk, seg)

    # --- B. exact-threshold areas: se_th, se_th +/- 1, min_floe, min_brash --------------------------------
    seg = _blank((30, 90))
    # rows of rectangles with exactly 49, 50, 51, 39, 40, 41, 1, 2 pixels
    specs = [(1, 1, 7, 7, 49), (1, 12, 5, 10, 50), (1, 26, 3, 17, 51),
             (10, 1, 3, 13, 39), (10, 16, 4, 10, 40), (10, 28, 41, 1, 41),
             (20, 1, 1, 1, 1), (20, 4, 1, 2, 2)]
    for r0, c0, h, w, want in specs:
        if h * w != want:
            h, w = 1, want
        seg[r0:r0 + h, c0:c0 + w] = 1.0
    out["thresholds"] = ((seg > 0).astype(np.float64), seg)

    # --- C. sort ties: many pieces of exactly the same area ------------------------------------------------
    seg = _blank((24, 40))
    for i in range(6):
        seg[2 + 3 * i, 2:6] = 1.0                     # six 4-px pieces (a 6-way tie)
    for i in range(4):
        seg[2 + 3 * i, 20:26] = 0.5                   # four 6-px dark pieces
    seg[20:23, 30:36] = 1.0                           # one bigger piece
    out["ties"] = ((seg > 0).astype(np.float64), seg)

    # --- D. no dark ice at all (nn_k = 0) -----------------------------------------------------------------
    seg = _blank((20, 20))
    seg[4:12, 4:14] = 1.0
    out["light_only"] = ((seg > 0).astype(np.float64), seg)

    # --- E. empty seg (no ice at all) ---------------------------------------------------------------------
    seg = _blank((12, 15))
    out["empty"] = (np.zeros((12, 15)), seg)

    # --- F. a small piece completely swallowed by a larger one (out label overwritten -> area0 == 0) -------
    seg = _blank((30, 30))
    seg[3:26, 3:26] = 1.0
    seg[10:18, 10:18] = 0.0
    seg[12:16, 12:16] = 0.5          # a dark piece sitting inside the big light floe's hole
    out["swallowed"] = ((seg > 0).astype(np.float64), seg)

    # --- G. a ring whose hole `imopen` reopens, plus a piece imopen deletes entirely -----------------------
    seg = _blank((26, 40))
    yy, xx = np.mgrid[0:26, 0:40]
    ring = ((yy - 12) ** 2 + (xx - 12) ** 2 <= 81) & ((yy - 12) ** 2 + (xx - 12) ** 2 >= 25)
    seg[ring] = 1.0
    seg[5, 30] = seg[5, 31] = 1.0    # a 2-px piece that imopen(disk 1) deletes
    seg[20:22, 30:32] = 0.5
    bk = (seg > 0).astype(np.float64)
    bk[24, 0:40] = 1.0
    out["ring"] = (bk, seg)

    # --- H. bk NOT covering the identification (the coverage identity check) ------------------------------
    bk, seg = out["mixed"]
    out["bk_partial"] = (np.zeros_like(bk), seg)

    # --- I. the adversarial case for the bounding-box crop (analysis R11) ---------------------------------
    # ch04 pinned that MATLAB's `imclose` **pre-pads with 0**, so a dark notch open to the *image* border is
    # not closed.  A crop window that does not contain that border would close it.  Every piece here touches a
    # border and carries a notch cut in from it; one piece spans the whole image.
    seg = _blank((26, 44))
    seg[0:9, 1:12] = 1.0                       # touches the top border ...
    seg[0:5, 5:7] = 0.0                        # ... with a 2-px notch cut in from the top edge
    seg[18:26, 30:43] = 1.0                    # touches the bottom border ...
    seg[22:26, 35:37] = 0.0                    # ... with a notch cut in from the bottom edge
    seg[10:20, 0:8] = 0.5                      # a dark piece on the left border ...
    seg[13:16, 0:3] = 0.0                      # ... with a notch cut in from the left edge
    seg[0:26, 43] = 1.0                        # a 1-px column filling the whole right border
    seg[6:8, 43] = 0.0                         # ... broken by a 2-px gap (closing reconnects it)
    seg[24, 0] = 1.0                           # a 1-px piece in the very corner
    bk = (seg > 0).astype(np.float64)
    out["border_notch"] = (bk, seg)

    return out
