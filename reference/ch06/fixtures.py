"""Constructed fixtures for the chapter 6 MATLAB references.

Written to ``reference/ch06/inputs.mat`` by :mod:`make_refs` so that MATLAB and Python operate on **the same
arrays** (no JPEG decoding, no re-derivation on either side).  Everything here is deterministic.

Groups
------
``del2_*``    matrices for ``del2`` (Risk R5: ``m``/``n`` in {1, 2, 3}, constant, ``+-Inf``, non-unit spacing)
``bm_*``      matrices for ``BoundMirrorExpand/Ensure/Shrink``
``g2_*``      matrices for ``gradient2`` (1-D rows/columns included)
``gvf_*``     edge maps for ``GVF`` (small, so 500 iterations stay cheap)
``xc_*``      images and masks for ``xconv2`` vs ``conv2(..., 'same')``
``sidx_*``    boolean patterns for ``snakeindex``
``snk_*``     contours for ``snakeinterp`` / ``snakedeform``
``fld_*``     force fields for ``snakedeform`` / ``interp2('*linear', 0)``
``sh_*``      binary shapes for ``regionprops``
``pg_*``      polygons for ``polygeom`` / ``poly2mask`` / ``minboundrect`` / ``polyxpoly`` / ``polybool``
"""
from __future__ import annotations

import numpy as np


def _circle(cx: float, cy: float, r: float) -> tuple[np.ndarray, np.ndarray]:
    """The chapter's own initial contour: ``t = 0:0.05:6.28`` (126 points, not closed)."""
    t = np.arange(0.0, 6.28 + 1e-12, 0.05)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def del2_fixtures() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(601)
    fx: dict[str, np.ndarray] = {}
    fx["del2_4x5"] = np.arange(1, 21, dtype=np.float64).reshape(4, 5) ** 2
    fx["del2_3x3"] = np.array([[1.0, 2, 3], [4, 5, 6], [7, 8, 10]])
    fx["del2_3x7"] = rng.normal(size=(3, 7))
    fx["del2_7x3"] = rng.normal(size=(7, 3))
    fx["del2_2x3"] = np.array([[1.0, 2, 3], [4, 5, 6]])
    fx["del2_3x2"] = np.array([[1.0, 2], [4, 5], [7, 9]])
    fx["del2_2x2"] = np.array([[1.0, 2], [4, 5]])
    fx["del2_1x5"] = np.array([[1.0, 4, 9, 16, 25]])
    fx["del2_5x1"] = np.array([[1.0], [4], [9], [16], [25]])
    fx["del2_1x1"] = np.array([[7.0]])
    fx["del2_1x3"] = np.array([[1.0, 5, 2]])
    fx["del2_3x1"] = np.array([[1.0], [5], [2]])
    fx["del2_const"] = np.full((5, 6), 3.25)
    fx["del2_rand"] = rng.normal(size=(9, 11))
    t = rng.normal(size=(7, 8))
    t[2, 3] = np.inf
    t[5, 1] = -np.inf
    fx["del2_inf"] = t
    fx["del2_all_inf"] = np.full((4, 5), np.inf)
    fx["del2_big"] = rng.normal(size=(40, 33))
    return fx


def boundmirror_fixtures() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(602)
    fx: dict[str, np.ndarray] = {}
    fx["bm_3x4"] = np.array([[1.0, 2, 3, 11], [4, 5, 6, 12], [7, 8, 9, 13]])  # the header's own example
    fx["bm_5x7"] = rng.normal(size=(5, 7))
    fx["bm_2x2"] = np.array([[1.0, 2], [3, 4]])
    fx["bm_2x5"] = rng.normal(size=(2, 5))
    fx["bm_3x3"] = rng.normal(size=(3, 3))
    fx["bm_4x3"] = rng.normal(size=(4, 3))
    fx["bm_big"] = rng.normal(size=(21, 17))
    return fx


def gradient2_fixtures() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(603)
    fx: dict[str, np.ndarray] = {}
    fx["g2_9x11"] = rng.normal(size=(9, 11))
    fx["g2_1x7"] = np.arange(1.0, 8.0).reshape(1, 7) ** 1.5
    fx["g2_7x1"] = np.arange(1.0, 8.0).reshape(7, 1) ** 1.5
    fx["g2_2x2"] = np.array([[1.0, 3], [7, 2]])
    fx["g2_1x1"] = np.array([[4.0]])
    fx["g2_3x5"] = rng.normal(size=(3, 5))
    fx["g2_1x2"] = np.array([[2.0, 5.0]])
    return fx


def gvf_fixtures() -> dict[str, np.ndarray]:
    """Small edge maps: GVF is O(iters x pixels), so keep them tiny for the 500-iteration cases."""
    rng = np.random.default_rng(604)
    fx: dict[str, np.ndarray] = {}
    # a U-shape edge map (Xu & Prince's classic), 40x40
    u = np.zeros((40, 40))
    u[8:32, 8:14] = 1.0
    u[8:32, 26:32] = 1.0
    u[26:32, 8:32] = 1.0
    fx["gvf_u"] = u
    fx["gvf_rand"] = rng.random((25, 30))
    fx["gvf_3x3"] = np.array([[0.0, 1, 0], [1, 2, 1], [0, 1, 0]])
    fx["gvf_3x5"] = rng.random((3, 5))
    fx["gvf_4x4"] = rng.random((4, 4))
    fx["gvf_const"] = np.full((6, 7), 2.0)
    t = rng.random((6, 7))
    t[2, 2] = np.inf
    fx["gvf_inf"] = t
    # a disc edge map: |grad| of a filled circle
    yy, xx = np.mgrid[0:30, 0:30]
    disc = ((yy - 15.0) ** 2 + (xx - 15.0) ** 2 <= 81).astype(np.float64)
    gy, gx = np.gradient(disc)
    fx["gvf_disc"] = np.hypot(gx, gy)
    return fx


def xconv_fixtures() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(605)
    fx: dict[str, np.ndarray] = {}
    fx["xc_I"] = rng.random((23, 19)) * 100.0
    fx["xc_I2"] = rng.random((16, 16)) * 255.0
    fx["xc_G3"] = rng.random((3, 3))
    fx["xc_G4"] = rng.random((4, 4))  # even mask: the floor(n1/2) crop matters
    fx["xc_G5x3"] = rng.random((5, 3))
    return fx


def snakeindex_fixtures() -> dict[str, np.ndarray]:
    fx: dict[str, np.ndarray] = {}
    pats = {
        "sidx_all1": [1, 1, 1, 1, 1, 1],
        "sidx_all0": [0, 0, 0, 0, 0, 0],
        "sidx_alt": [1, 0, 1, 0, 1, 0],
        "sidx_alt2": [0, 1, 0, 1, 0, 1],
        "sidx_one": [0, 0, 1, 0, 0],
        "sidx_last": [0, 0, 0, 0, 1],
        "sidx_first": [1, 0, 0, 0, 0],
        "sidx_len1": [1],
        "sidx_len1z": [0],
        "sidx_len2": [1, 0],
        "sidx_len3": [0, 1, 1],
        "sidx_rand": [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
    }
    for k, v in pats.items():
        fx[k] = np.asarray(v, dtype=np.float64).reshape(1, -1)
    return fx


def snake_fixtures() -> dict[str, np.ndarray]:
    """Contours (row vectors) and force fields for ``snakeinterp`` / ``snakedeform``."""
    rng = np.random.default_rng(606)
    fx: dict[str, np.ndarray] = {}
    x, y = _circle(50.0, 40.0, 20.0)
    fx["snk_circle_x"] = x.reshape(1, -1)
    fx["snk_circle_y"] = y.reshape(1, -1)
    x2, y2 = _circle(30.0, 30.0, 3.0)
    fx["snk_small_x"] = x2.reshape(1, -1)
    fx["snk_small_y"] = y2.reshape(1, -1)
    # a square with 4 corners only (spacing >> dmax -> many insertion passes)
    sq_x = np.array([10.0, 40.0, 40.0, 10.0])
    sq_y = np.array([10.0, 10.0, 35.0, 35.0])
    fx["snk_square_x"] = sq_x.reshape(1, -1)
    fx["snk_square_y"] = sq_y.reshape(1, -1)
    # a polyline with duplicate vertices (Risk R10: the removal branch / degenerate spacing)
    dx = np.array([5.0, 5.0, 9.0, 14.0, 14.0, 9.0, 5.0])
    dy = np.array([5.0, 5.0, 5.0, 9.0, 14.0, 14.0, 9.0])
    fx["snk_dup_x"] = dx.reshape(1, -1)
    fx["snk_dup_y"] = dy.reshape(1, -1)
    # a self-touching polyline
    fx["snk_star_x"] = (30 + 12 * np.cos(np.arange(8) * 2 * np.pi * 3 / 8)).reshape(1, -1)
    fx["snk_star_y"] = (30 + 12 * np.sin(np.arange(8) * 2 * np.pi * 3 / 8)).reshape(1, -1)
    # short contours for the dense solver (N = 8, 12)
    x8, y8 = 20 + 6 * np.cos(np.arange(8) * np.pi / 4), 20 + 6 * np.sin(np.arange(8) * np.pi / 4)
    fx["snk_n8_x"] = x8.reshape(1, -1)
    fx["snk_n8_y"] = y8.reshape(1, -1)
    # force fields
    fx["fld_px"] = rng.normal(size=(60, 70))
    fx["fld_py"] = rng.normal(size=(60, 70))
    yy, xx = np.mgrid[0:60, 0:70]
    fx["fld_ring_px"] = ((35.0 - xx) / (np.hypot(xx - 35.0, yy - 30.0) + 1e-10))
    fx["fld_ring_py"] = ((30.0 - yy) / (np.hypot(xx - 35.0, yy - 30.0) + 1e-10))
    return fx


def shape_fixtures() -> dict[str, np.ndarray]:
    """Binary shapes for ``regionprops`` (uint8 0/1; MATLAB casts them with ``logical``)."""
    rng = np.random.default_rng(607)
    fx: dict[str, np.ndarray] = {}

    def blank(m, n):
        return np.zeros((m, n), dtype=np.uint8)

    # --- the analysis' 12x14 probe fixture (BoundingBox [2.5 2.5 9 7], Area 53) --------------------------
    a = blank(12, 14)
    a[2:9, 2:11] = 1              # rows 3-9, cols 3-11 (1-based)
    a[2:4, 8:11] = 0              # 2x3 bite, rows 3-4, cols 9-11
    a[7:9, 2:4] = 0               # 2x2 bite, rows 8-9, cols 3-4
    fx["sh_probe"] = a

    fx["sh_single"] = blank(7, 7)
    fx["sh_single"][3, 3] = 1
    fx["sh_two_px"] = blank(7, 7)
    fx["sh_two_px"][2, 2] = 1
    fx["sh_two_px"][3, 3] = 1     # diagonal pair (one 8-connected component)
    fx["sh_hline"] = blank(7, 13)
    fx["sh_hline"][3, 2:11] = 1
    fx["sh_vline"] = blank(13, 7)
    fx["sh_vline"][2:11, 3] = 1
    fx["sh_diag"] = blank(12, 12)
    for i in range(9):
        fx["sh_diag"][1 + i, 1 + i] = 1
    fx["sh_stair"] = blank(12, 12)
    for i in range(5):
        fx["sh_stair"][1 + i, 1 + 2 * i] = 1
        fx["sh_stair"][1 + i, 2 + 2 * i] = 1
    fx["sh_square"] = blank(12, 12)
    fx["sh_square"][2:10, 2:10] = 1
    fx["sh_rect"] = blank(12, 20)
    fx["sh_rect"][3:9, 2:18] = 1
    ring = blank(15, 15)
    ring[2:13, 2:13] = 1
    ring[5:10, 5:10] = 0
    fx["sh_ring"] = ring
    L = blank(14, 14)
    L[2:12, 2:6] = 1
    L[8:12, 6:12] = 1
    fx["sh_L"] = L
    yy, xx = np.mgrid[0:25, 0:25]
    fx["sh_disc"] = (((yy - 12.0) ** 2 + (xx - 12.0) ** 2) <= 81).astype(np.uint8)
    # ellipses at 0 / 30 / 45 / 90 degrees
    for ang in (0, 30, 45, 90):
        th = np.deg2rad(ang)
        Y, X = np.mgrid[0:41, 0:41]
        xr = (X - 20) * np.cos(th) + (Y - 20) * np.sin(th)
        yr = -(X - 20) * np.sin(th) + (Y - 20) * np.cos(th)
        fx[f"sh_ell{ang}"] = (((xr / 15.0) ** 2 + (yr / 6.0) ** 2) <= 1).astype(np.uint8)
    # peanut (concave, low solidity)
    Y, X = np.mgrid[0:40, 0:60]
    fx["sh_peanut"] = ((((X - 20) ** 2 + (Y - 20) ** 2) <= 144)
                       | (((X - 38) ** 2 + (Y - 20) ** 2) <= 144)).astype(np.uint8)
    # a shape touching the image border on all four sides
    b = blank(15, 15)
    b[0:15, 0:5] = 1
    b[6:9, 0:15] = 1
    fx["sh_border"] = b
    # random blobs
    for k in range(6):
        img = (rng.random((22, 26)) > 0.55).astype(np.uint8)
        from scipy.ndimage import binary_closing
        img = binary_closing(img, np.ones((3, 3))).astype(np.uint8)
        fx[f"sh_rand{k}"] = img
    # multi-component label sources
    m = blank(20, 30)
    m[2:8, 2:8] = 1
    m[2:8, 12:26] = 1
    m[12:18, 4:20] = 1
    m[15, 25] = 1
    fx["sh_multi"] = m
    # near the Rc = 0.9 / Rl = 2 decision boundaries -------------------------------------------------
    # solidity just above / below 0.9: a square with a triangular bite of tuned size
    for k, bite in enumerate((2, 3, 4, 5)):
        s = blank(24, 24)
        s[2:22, 2:22] = 1
        for i in range(bite * 2):
            s[2 + i, 2:2 + max(0, bite * 2 - i)] = 0
        fx[f"sh_solid{k}"] = s
    # axis ratio near 2: rectangles 20x10, 20x11, 21x10, and ellipses
    fx["sh_ratio0"] = blank(16, 26)
    fx["sh_ratio0"][3:13, 3:23] = 1                       # 10 x 20
    fx["sh_ratio1"] = blank(17, 26)
    fx["sh_ratio1"][3:14, 3:23] = 1                       # 11 x 20
    fx["sh_ratio2"] = blank(16, 27)
    fx["sh_ratio2"][3:13, 3:24] = 1                       # 10 x 21
    for k, (aa, bb) in enumerate(((14.0, 7.0), (14.0, 6.9), (14.0, 7.1))):
        Y, X = np.mgrid[0:40, 0:40]
        fx[f"sh_ellr{k}"] = ((((X - 20) / aa) ** 2 + ((Y - 20) / bb) ** 2) <= 1).astype(np.uint8)
    # an empty image and an all-foreground image
    fx["sh_empty"] = blank(8, 9)
    fx["sh_full"] = np.ones((8, 9), dtype=np.uint8)
    return fx


def polygon_fixtures() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(608)
    fx: dict[str, np.ndarray] = {}
    # polygeom's own header self-test (3x5 rectangle rotated 30 degrees)
    fx["pg_test_x"] = np.array([[2.0, 0.5, 4.83, 6.33]])
    fx["pg_test_y"] = np.array([[4.0, 6.598, 9.098, 6.5]])
    # random simple polygons: convex hulls of random point clouds (guaranteed simple)
    for k in range(10):
        pts = rng.normal(size=(9, 2)) * np.array([5.0, 3.0]) + np.array([20.0, 15.0])
        ang = np.arctan2(pts[:, 1] - pts[:, 1].mean(), pts[:, 0] - pts[:, 0].mean())
        pts = pts[np.argsort(ang)]
        from scipy.spatial import ConvexHull
        h = ConvexHull(pts)
        p = pts[h.vertices]
        fx[f"pg_rand{k}_x"] = p[:, 0].reshape(1, -1)
        fx[f"pg_rand{k}_y"] = p[:, 1].reshape(1, -1)
    # clockwise version of one of them (polygeom's A < 0 branch)
    fx["pg_cw_x"] = fx["pg_rand0_x"][:, ::-1].copy()
    fx["pg_cw_y"] = fx["pg_rand0_y"][:, ::-1].copy()
    # a triangle, a square, a self-intersecting bowtie
    fx["pg_tri_x"] = np.array([[1.0, 6.0, 3.0]])
    fx["pg_tri_y"] = np.array([[1.0, 2.0, 7.0]])
    fx["pg_sq_x"] = np.array([[2.0, 9.0, 9.0, 2.0]])
    fx["pg_sq_y"] = np.array([[2.0, 2.0, 8.0, 8.0]])
    fx["pg_bow_x"] = np.array([[1.0, 5.0, 1.0, 5.0]])
    fx["pg_bow_y"] = np.array([[1.0, 5.0, 5.0, 1.0]])
    # point clouds for minboundrect, including the nedges = 0/1/2 special cases
    fx["pc_one_x"] = np.array([[3.0]])
    fx["pc_one_y"] = np.array([[4.0]])
    fx["pc_two_x"] = np.array([[1.0, 5.0]])
    fx["pc_two_y"] = np.array([[2.0, 7.0]])
    fx["pc_three_x"] = np.array([[0.0, 4.0, 2.0]])
    fx["pc_three_y"] = np.array([[0.0, 1.0, 5.0]])
    fx["pc_collinear_x"] = np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])
    fx["pc_collinear_y"] = np.array([[0.0, 2.0, 4.0, 6.0, 8.0]])
    fx["pc_dup_x"] = np.array([[2.0, 2.0, 2.0, 2.0]])
    fx["pc_dup_y"] = np.array([[3.0, 3.0, 3.0, 3.0]])
    for k in range(6):
        p = rng.normal(size=(40, 2)) @ np.array([[2.0, 0.7], [0.0, 1.0]]) + np.array([10.0, 5.0])
        fx[f"pc_rand{k}_x"] = p[:, 0].reshape(1, -1)
        fx[f"pc_rand{k}_y"] = p[:, 1].reshape(1, -1)
    fx["pc_square_x"] = np.array([[0.0, 1.0, 1.0, 0.0, 0.5]])
    fx["pc_square_y"] = np.array([[0.0, 0.0, 1.0, 1.0, 0.5]])
    # poly2mask / roipoly cases (1-based image coordinates)
    for k, (cx, cy, r) in enumerate(((10.5, 10.5, 6.0), (10.0, 10.0, 6.5), (3.2, 4.7, 2.3),
                                     (10.5, 10.5, 9.9), (10.5, 10.5, 0.4))):
        x, y = _circle(cx, cy, r)
        fx[f"pm{k}_x"] = x.reshape(1, -1)
        fx[f"pm{k}_y"] = y.reshape(1, -1)
    fx["pm_tri_x"] = np.array([[1.0, 15.0, 8.0]])
    fx["pm_tri_y"] = np.array([[1.0, 3.0, 18.0]])
    fx["pm_int_x"] = np.array([[2.0, 12.0, 12.0, 2.0]])   # integer-coordinate square
    fx["pm_int_y"] = np.array([[2.0, 2.0, 9.0, 9.0]])
    fx["pm_half_x"] = np.array([[2.5, 12.5, 12.5, 2.5]])  # half-integer square
    fx["pm_half_y"] = np.array([[2.5, 2.5, 9.5, 9.5]])
    # polybool clipping cases: circle vs the image rectangle of a 30 x 40 image
    for k, (cx, cy, r) in enumerate(((20.0, 15.0, 8.0),     # fully inside
                                     (3.0, 4.0, 8.0),       # over the (0,0) corner
                                     (38.0, 27.0, 9.0),     # over the far corner
                                     (20.0, 2.0, 7.0),      # over the top edge
                                     (-15.0, 15.0, 5.0),    # entirely outside
                                     (0.0, 15.0, 6.0),      # centred on the left edge
                                     (20.0, 15.0, 30.0))):  # rectangle entirely inside the circle
        x, y = _circle(cx, cy, r)
        fx[f"pb{k}_x"] = x.reshape(1, -1)
        fx[f"pb{k}_y"] = y.reshape(1, -1)
    # polyxpoly
    fx["px1_x"] = np.array([[0.0, 10.0, 10.0, 0.0, 0.0]])
    fx["px1_y"] = np.array([[0.0, 0.0, 10.0, 10.0, 0.0]])
    fx["px2_x"] = np.array([[5.0, 15.0, 15.0, 5.0, 5.0]])
    fx["px2_y"] = np.array([[5.0, 5.0, 15.0, 15.0, 5.0]])
    fx["px3_x"] = np.array([[20.0, 30.0]])
    fx["px3_y"] = np.array([[20.0, 30.0]])
    return fx


def all_fixtures() -> dict[str, np.ndarray]:
    fx: dict[str, np.ndarray] = {}
    for f in (del2_fixtures, boundmirror_fixtures, gradient2_fixtures, gvf_fixtures, xconv_fixtures,
              snakeindex_fixtures, snake_fixtures, shape_fixtures, polygon_fixtures):
        fx.update(f())
    return fx
