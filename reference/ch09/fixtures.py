"""Deterministic fixtures shared by ``reference/ch09/make_refs.py`` and ``tests/test_ch09.py``.

Nothing here depends on MATLAB: every array is built from literals so that the Python side and the MATLAB side
see byte-identical inputs (the masks go out as ``.mat``, the tank image as the JPEG both engines decode).

Design notes — each fixture exists to make a *specific* wrong port fail (verify-port "test-design lessons"):

* ``rect_masks`` are **non-square** canvases containing a 1-pixel blob, a 1-pixel straight line (``minboundrect``'s
  degenerate hull branch), an exact square, a 5 : 1 sliver, a 45°-rotated square, a non-convex plus and a
  border-touching blob.  A square canvas cannot catch ``rect.m``'s ``minboundrect(c, r, 'a')`` ``(x = column,
  y = row)`` swap.
* ``model_rect_sets`` contain a **contained**, a **disjoint**, a **partially overlapping**, an **edge-touching**
  (full shared edge), a **vertex-touching** and a **partial-edge-overlap** pair — the six cases that decide
  ``clip_polygon_convex(drop_degenerate=True)`` against ``polybool``.
* ``ratio_rect_set`` has ``k`` exactly at ``k1 = 0.4`` and exactly at ``k2 = 2.5`` (both must be *rejected*, the
  tests are strict) and on both sides of 1, because ``model_ice_model.m``'s ``k`` is **not** normalised to ``>= 1``.
* ``tie_block_image`` is a 4 × 6 gray image whose Otsu threshold is attained by pixels **exactly**, so ``>`` and
  ``>=`` give different counts (the ch3 vs ch9 difference of `analysis/ch09.md` §0.3).
"""
from __future__ import annotations

import numpy as np

__all__ = ["rect_mask_a", "rect_mask_b", "rect_mask_degenerate", "model_rect_set", "ratio_rect_set", "polybool_pairs",
           "tie_block_image", "ring"]


def ring(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """A closed 5 × 2 axis-aligned ring ``[x y]`` in ``minboundrect``'s vertex order.

    ``v1 = (x0, y0)``, ``v2 = (x1, y0)``, ``v3 = (x1, y1)`` so ``model_ice_model.m``'s
    ``k = |v1 - v2| / |v3 - v2|`` is the **width / height** ratio.
    """
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64)


def rect_mask_a() -> np.ndarray:
    """40 x 70 (deliberately **not** square) with six 4-connected, **non-degenerate** components.

    1. an 8 x 8 exact square, 2. a 4 x 20 sliver (5 : 1), 3. a 45deg-rotated square (diamond),
    4. a blob touching the top and right borders, 5. a 2 x 5 rectangle, 6. a non-convex L.

    Every component has a 2-D hull: MATLAB's ``convhull`` -- which ``minboundrect.m`` line 102 calls -- **errors**
    on a collinear point cloud, so a 1-pixel blob or a 1-pixel line makes ``rect.m`` itself unrunnable.  Those two
    shapes live in :func:`rect_mask_degenerate` and are probed with ``try/catch`` instead (verify-port: "a rejected
    input is evidence, not a failed run").
    """
    bw = np.zeros((40, 70), dtype=bool)
    bw[14:22, 4:12] = True                            # 8 x 8 square
    bw[24:28, 19:39] = True                           # 4 x 20 sliver (5 : 1)
    rr, cc = np.mgrid[0:40, 0:70]
    bw |= (np.abs(rr - 32) + np.abs(cc - 55)) <= 5    # diamond (45deg-rotated square)
    bw[0:5, 65:70] = True                             # touches the top and right borders
    bw[4:6, 20:25] = True                             # 2 x 5 rectangle
    bw[10:20, 45:48] = True                           # L: long arm
    bw[17:20, 45:58] = True                           # L: foot
    return bw


def rect_mask_degenerate() -> np.ndarray:
    """12 x 20 holding only a **single pixel** and a **1-pixel straight line** — the two shapes on which MATLAB's
    ``convhull`` (and therefore ``minboundrect.m`` and ``rect.m``) raises *"Error computing the convex hull. The
    points may be collinear."*  Used for the ``try/catch`` probe, never for a numeric comparison."""
    bw = np.zeros((12, 20), dtype=bool)
    bw[3, 4] = True                                   # single pixel
    bw[8, 6:13] = True                                # 1 x 7 line (collinear)
    return bw


def rect_mask_b() -> np.ndarray:
    """25 x 60 with a non-convex plus, an L-shape and two 2 x 2 blocks that touch **only diagonally**.

    The diagonal pair is one 8-connected component but **two** 4-connected ones, which pins ``rect.m``'s
    hard-coded ``bwlabel(img, 4)``.
    """
    bw = np.zeros((25, 60), dtype=bool)
    bw[4:15, 8:11] = True                             # plus: vertical bar
    bw[8:11, 2:17] = True                             # plus: horizontal bar
    bw[3:18, 25:28] = True                            # L: long arm
    bw[15:18, 25:40] = True                           # L: foot
    bw[4:6, 45:47] = True
    bw[6:8, 47:49] = True                             # 8-connected to the previous, NOT 4-connected
    return bw


def model_rect_set() -> tuple[np.ndarray, tuple[int, int], dict[str, tuple[int, int]]]:
    """Eleven rectangles in a 100 × 100 frame realising the six overlap relations.

    Returns ``(V, shape, relations)`` with ``V`` of shape ``(5, 2, 11)`` (MATLAB's page order) and ``relations``
    naming the 1-based index pair for each case.
    """
    rects = [
        ring(5, 5, 25, 25),        # 1  outer
        ring(10, 10, 20, 20),      # 2  contained in 1
        ring(30, 5, 50, 25),       # 3
        ring(40, 15, 60, 35),      # 4  partial overlap with 3
        ring(5, 35, 25, 55),       # 5
        ring(25, 35, 45, 55),      # 6  shares the FULL edge x = 25 with 5 (zero-area intersection)
        ring(55, 45, 75, 65),      # 7
        ring(75, 65, 95, 85),      # 8  touches 7 at the single vertex (75, 65)
        ring(5, 65, 25, 85),       # 9
        ring(25, 75, 45, 95),      # 10 shares PART of the edge x = 25 (y in [75, 85]) with 9
        ring(75, 5, 95, 25),       # 11 disjoint from everything
    ]
    V = np.stack(rects, axis=2)
    relations = dict(contained=(1, 2), partial=(3, 4), edge_touching=(5, 6),
                     vertex_touching=(7, 8), partial_edge=(9, 10), disjoint=(11, 1))
    return V, (100, 100), relations


def ratio_rect_set() -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """Six disjoint rectangles whose ``k = width / height`` is 0.39, **0.40**, 0.50, 1.00, 2.40, **2.50**.

    ``model_ice_model.m`` line 34 is ``if k < k2 && k > k1`` — both *strict*, so the two boundary rectangles must
    be **rejected**.  Returns ``(V, shape, expected_k)``.
    """
    specs = [(3.9, 10.0), (4.0, 10.0), (5.0, 10.0), (10.0, 10.0), (24.0, 10.0), (25.0, 10.0)]
    rects = []
    x = 5.0
    for w, h in specs:
        rects.append(ring(x, 5.0, x + w, 5.0 + h))
        x += w + 8.0
    V = np.stack(rects, axis=2)
    k = np.array([w / h for w, h in specs])
    return V, (40, int(x) + 10), k


def polybool_pairs() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Ten ``(subject, clip)`` ring pairs for the direct ``polybool('intersection', …)`` probe.

    Beyond the six relations of :func:`model_rect_set` this adds a rotated/rotated partial overlap, an identical
    pair, a hair-gap near-miss (1e-9) and a hair-overlap (1e-9) — the floating-point edge of risk R4.
    """
    def rot(cx, cy, w, h, ang):
        c, s = np.cos(ang), np.sin(ang)
        pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        R = np.array([[c, -s], [s, c]])
        p = pts @ R.T + np.array([cx, cy])
        return np.vstack([p, p[0]])

    pairs = [
        ("contained", ring(0, 0, 10, 10), ring(2, 2, 8, 8)),
        ("disjoint", ring(0, 0, 4, 4), ring(10, 10, 14, 14)),
        ("partial", ring(0, 0, 10, 10), ring(5, 5, 15, 15)),
        ("edge_touching", ring(0, 0, 10, 10), ring(10, 0, 20, 10)),
        ("vertex_touching", ring(0, 0, 10, 10), ring(10, 10, 20, 20)),
        ("partial_edge", ring(0, 0, 10, 10), ring(10, 5, 20, 15)),
        ("rotated_partial", rot(0, 0, 10, 6, 0.4), rot(4, 2, 8, 8, -0.9)),
        ("identical", ring(0, 0, 10, 10), ring(0, 0, 10, 10)),
        ("hair_gap", ring(0, 0, 10, 10), ring(10 + 1e-9, 0, 20, 10)),
        ("hair_overlap", ring(0, 0, 10, 10), ring(10 - 1e-9, 0, 20, 10)),
    ]
    names = [p[0] for p in pairs]
    A = np.stack([p[1] for p in pairs], axis=2)
    B = np.stack([p[2] for p in pairs], axis=2)
    return A, B, names


def tie_block_image() -> np.ndarray:
    """A 4 × 6 uint8 image whose per-block Otsu threshold **is attained** by some pixels.

    ``block_otsu(..., n_r=2, n_c=3)`` slices it into six 2 × 2 blocks; every block holds the value ``th`` itself,
    so ``compare='gt'`` (ch3 ``local_Otsu.m``) and ``compare='ge'`` (ch9 ``block_threshold.m``) must disagree.
    """
    blocks = [
        np.array([[10, 10], [200, 200]]),      # th = 104.5  -- a HALF-INTEGER level (graythresh tie averaging)
        np.array([[60, 61], [62, 63]]),        # th =  61    -- 1 pixel exactly at th  -> gt/ge differ by 1
        np.array([[5, 5], [5, 250]]),          # th = 127    -- no pixel at th          -> gt/ge agree
        np.array([[40, 41], [42, 200]]),       # th = 120.5  -- half-integer again
        np.array([[100, 100], [101, 102]]),    # th = 100    -- 2 pixels exactly at th -> gt/ge differ by 2
        np.array([[8, 9], [10, 11]]),          # th =   9    -- 1 pixel exactly at th  -> gt/ge differ by 1
    ]
    img = np.zeros((4, 6), dtype=np.uint8)
    for b, blk in enumerate(blocks):
        i, j = divmod(b, 3)
        img[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = blk
    return img
