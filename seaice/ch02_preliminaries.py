"""Chapter 2 — Digital Image Processing Preliminaries: demo-level functions.

Each function reproduces one MATLAB script of ``MATLAB_ROOT/ch2/`` or one text-only example of the book, using
the primitives in :mod:`seaice.core`.  The ``scripts/ch02_*.py`` drivers only load data, call these functions and
write figures to ``outputs/ch02/``.

Book: Zhang & Skjetne (2018), Chapter 2, pp. 11–36.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .core import synth
from .core.chaincode import ChainCode, bound2im, boundaries, chain_to_points, fchcode, normalized_first_difference
from .core.color import indexed_to_rgb, rgb2cmy, rgb2cmyk, rgb2hsi, split_rgb
from .core.connectivity import find_paths, label_components, n4, n8, nd, region_boundary_mask
from .core.distance import bwdist, center_distance_map, distance_transform
from .core.filters import conv2, conv_at, imfilter
from .core.histogram import imhist, normalized_histogram
from .core.interp import interp2, keys_kernel, resize
from .core.matlab_compat import imcomplement, rgb2gray_matlab
from .core.setops import bitwise_and, complement, difference, gray_complement, gray_intersection, gray_union, \
    intersection, reflect, translate, union

# ---------------------------------------------------------------------------------------------------------------
# §2.1  color_image.m
# ---------------------------------------------------------------------------------------------------------------


def color_components(rgb: np.ndarray, u: float = 1.0, b: float = 1.0) -> dict[str, Any]:
    """CMY, CMYK and HSI components of an RGB image — port of ``MATLAB_ROOT/ch2/color_image.m``.

    Book: §2.1.2.2 Eq. (2.3) → Fig. 2.4; Eqs. (2.4)–(2.5) (CMYK, text only); §2.1.2.3 Eqs. (2.6a–c) → Fig. 2.5.

    Returns a dict with

    * ``R, G, B`` uint8 planes (``Ir, Ig, Ib`` before the ``double`` cast);
    * ``I_cmy`` (uint8, ``imcomplement(I)``) and ``Ic, Im, Iy`` — the script complements the *double* planes
      (0–255 values), giving ``1 - Ir`` etc.; stored here exactly like that (float, mostly negative) because
      ``imshow(Ic, [])`` autoscales them, so the displayed image equals ``255 - Ir``;
    * ``H, S, I`` — Eq. (2.6) on RGB normalised to ``[0, 1]`` (the text's convention);
    * ``H_matlab, S_matlab, I_matlab`` — the script's values: unnormalised 0–255 input **and** the ``2*Ig`` bug of
      line 21 in ``H`` (see :func:`seaice.core.color.rgb2hsi`); ``S_matlab = 255 * S`` only if the bug is absent,
      so it is computed separately;
    * ``CMYK`` — Eq. (2.4) with factors ``u``, ``b`` (float ``(M, N, 4)``).
    """
    rgb = np.asarray(rgb)
    R, G, B = split_rgb(rgb)
    Ir, Ig, Ib = (x.astype(np.float64) for x in (R, G, B))
    out: dict[str, Any] = {"R": R, "G": G, "B": B}
    out["I_cmy"] = rgb2cmy(rgb)  # imcomplement(I) on uint8 → 255 - I
    out["Ic"], out["Im"], out["Iy"] = imcomplement(Ir), imcomplement(Ig), imcomplement(Ib)  # 1 - double(...)
    H, S, I = rgb2hsi(rgb)  # book Eq. (2.6), RGB in [0, 1]
    out.update(H=H, S=S, I=I)
    Hb, Sb, Ib_ = rgb2hsi(rgb, matlab_bug=True, scale=255.0)  # exactly what color_image.m computes
    out.update(H_matlab=Hb, S_matlab=Sb, I_matlab=Ib_)
    out["CMYK"] = rgb2cmyk(rgb, u=u, b=b)
    return out


# ---------------------------------------------------------------------------------------------------------------
# §2.2  histogram.m
# ---------------------------------------------------------------------------------------------------------------


def gray_histogram(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Histogram of a uint8 grayscale image in the three forms of ``histogram.m`` lines 22–46.

    Book: §2.2, Eq. (2.7) (``num``), Eq. (2.8) (``GP``), Fig. 2.7 (``imhist``).

    Returns ``num`` (manual count per level), ``GP`` (normalised), ``counts``/``x`` (``imhist``).  ``num`` and
    ``counts`` are identical by construction (both exact level counts).
    """
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = rgb2gray_matlab(gray)
    m, n = gray.shape
    counts, x = imhist(gray)
    num = np.array([int(np.count_nonzero(gray == k)) for k in range(256)])  # length(find(I == k))
    GP, _ = normalized_histogram(gray)
    assert np.array_equal(num, counts)
    assert abs(GP.sum() - 1.0) < 1e-12 and m * n == gray.size
    return {"num": num, "GP": GP, "counts": counts, "x": x, "gray": gray}


def channel_histograms(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """``[y_r, x] = imhist(Ir)`` etc. — ``histogram.m`` lines 50–52 (Fig. 2.8).

    Also returns the channel planes for the Fig. 2.3 panels and the quoted pixel ``I(1076, 675)`` (1-based).
    """
    R, G, B = split_rgb(np.asarray(rgb))
    y_r, x = imhist(R)
    y_g, _ = imhist(G)
    y_b, _ = imhist(B)
    pix = np.asarray(rgb)[1075, 674] if rgb.shape[0] > 1075 and rgb.shape[1] > 674 else None
    return {"x": x, "y_r": y_r, "y_g": y_g, "y_b": y_b, "R": R, "G": G, "B": B, "pixel_1076_675": pix}


# ---------------------------------------------------------------------------------------------------------------
# §2.1 text-only: image types (Figs. 2.1, 2.2, 2.6)
# ---------------------------------------------------------------------------------------------------------------


def image_type_examples(rgb: np.ndarray, crop_origin: tuple[int, int] = (200, 900)) -> dict[str, Any]:
    """Grayscale / binary / indexed image examples of §2.1.1 and §2.1.3 (text only).

    * ``gray`` = ``rgb2gray_matlab(rgb)``; ``gray_crop`` = an 11×11 window at ``crop_origin`` (0-based) in the
      spirit of Fig. 2.1 (the printed values are from an unknown location; ``fig_2_1_printed`` holds them);
    * ``binary`` = ``gray > 128`` (B = 1 image) and ``pattern`` = the synthetic 16×16 pattern for Fig. 2.2;
    * ``indexed`` / ``cmap`` / ``indexed_rgb`` = a tiny indexed image expanded through its colormap (Fig. 2.6).
    """
    gray = rgb2gray_matlab(np.asarray(rgb))
    r0, c0 = crop_origin
    crop = gray[r0:r0 + 11, c0:c0 + 11]
    binary = gray > 128
    idx = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 3, 3]])
    cmap = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 1.0]])
    return {
        "gray": gray, "gray_crop": crop, "fig_2_1_printed": synth.FIG_2_1_GRAY, "binary": binary,
        "pattern": synth.binary_pattern_16(), "indexed": idx, "cmap": cmap, "indexed_rgb": indexed_to_rgb(idx, cmap),
        "bits": 8, "levels": 256,
    }


# ---------------------------------------------------------------------------------------------------------------
# §2.3 text-only: neighbourhoods, paths, components (Figs. 2.9–2.11)
# ---------------------------------------------------------------------------------------------------------------


def pixel_relationship_examples() -> dict[str, Any]:
    """Worked examples of §2.3: N4/ND/N8 at an interior and a corner pixel (Fig. 2.9), the 4-/8-/m-paths of
    Fig. 2.10, the 5-vs-2 connected components of Fig. 2.11, and the region boundary of §2.3.5."""
    shape = (5, 5)
    p_int, p_corner = (2, 2), (0, 0)
    fig10a, fig10bc = synth.FIG_2_10_A, synth.FIG_2_10_BC
    src, dst = (0, 2), (2, 2)  # top-right pixel to bottom-right pixel of the 3×3 example
    paths4_a = find_paths(fig10a, (0, 2), (1, 1), 4)
    paths8 = find_paths(fig10bc, src, dst, 8)
    pathsm = find_paths(fig10bc, src, dst, "m")
    bw11 = synth.FIG_2_11_COMPONENTS
    L4, L8 = label_components(bw11, 4), label_components(bw11, 8)
    return {
        "shape": shape, "p_interior": p_int, "p_corner": p_corner,
        "N4_interior": n4(p_int, shape), "ND_interior": nd(p_int, shape), "N8_interior": n8(p_int, shape),
        "N4_corner": n4(p_corner, shape), "ND_corner": nd(p_corner, shape), "N8_corner": n8(p_corner, shape),
        "fig_2_10_a": fig10a, "fig_2_10_bc": fig10bc, "paths_4": paths4_a, "paths_8": paths8, "paths_m": pathsm,
        "fig_2_11": bw11, "labels_4": L4, "labels_8": L8, "n_components_4": int(L4.max()),
        "n_components_8": int(L8.max()), "boundary_8": region_boundary_mask(bw11, 8),
        "boundary_4": region_boundary_mask(bw11, 4),
    }


# ---------------------------------------------------------------------------------------------------------------
# §2.4  distance_transform.m (+ Figs. 2.12, 2.13)
# ---------------------------------------------------------------------------------------------------------------


def point_distance_maps(size: int = 201) -> dict[str, np.ndarray]:
    """Negative distance maps from the centre pixel — port of ``MATLAB_ROOT/ch2/distance_transform.m``.

    Book: §2.4.4, Fig. 2.14.  The script does ``img = zeros(201); img(101,101) = 1; img = ~img;
    imgDist = -bwdist(~img, metric)`` for ``'euclidean'``, ``'cityblock'``, ``'chessboard'`` and shows each with
    ``imshow(imgDist, [])`` (autoscaled).  ``'quasi-euclidean'`` is added for completeness (not in the script).

    Returns ``point`` (the binary input, Fig. 2.14a) and ``<metric>`` → ``-bwdist(point, metric)``.
    """
    point = synth.point_image(size)
    img = ~point  # img = ~img
    out = {"point": point}
    for metric in ("euclidean", "cityblock", "chessboard", "quasi-euclidean"):
        out[metric] = -bwdist(~img, metric)  # -bwdist(~img, metric)
    return out


def distance_fixture_examples() -> dict[str, np.ndarray]:
    """Figs. 2.12–2.13: Euclidean DT of the 7×7 matrix (Eq. 2.9 with Eq. 2.10) and the city-block / chessboard
    distances from the centre of a 7×7 grid (Eqs. 2.11–2.12)."""
    A = synth.FIG_2_12_A
    return {
        "A12": A, "D12": distance_transform(A, "euclidean"), "D12_book": synth.FIG_2_12_B,
        "C4": center_distance_map(7, "cityblock"), "C8": center_distance_map(7, "chessboard"),
        "Ce": center_distance_map(7, "euclidean"),
    }


# ---------------------------------------------------------------------------------------------------------------
# §2.5 text-only: convolution (Fig. 2.15)
# ---------------------------------------------------------------------------------------------------------------


def convolution_example(seed: int = 0) -> dict[str, Any]:
    """Numeric illustration of Book §2.5, Eqs. (2.14)–(2.15), Fig. 2.15: a 6×6 integer image, a 3×3 kernel, the
    full response by :func:`~seaice.core.filters.conv2` and the nine explicit terms at the centre pixel.

    Two readings of the nine-term sum are computed with :func:`~seaice.core.filters.conv_at`, because the book is
    inconsistent: Eq. (2.14) defines convolution with ``f(x−s, y−t)`` (kernel flipped, = ``conv2``), whereas Eq. (2.15)
    as printed on p. 24 expands to ``ω(s,t) f(x+s, y+t)`` (kernel *not* flipped = correlation = MATLAB ``imfilter``).
    On the antisymmetric Sobel-type demo kernel the two differ in sign (``+1`` vs ``−1`` at the chosen pixel).

    Returns
    -------
    dict with ``f``, ``w``, ``h_conv2`` (Eq. 2.14 / ``conv2``), ``h_imfilter_corr`` (correlation / ``imfilter``),
    ``x``, ``y`` (0-based pixel), ``terms_conv`` (Eq. 2.14 terms ``(s, t, ω(s,t), f(x−s,y−t))``), ``terms_corr``
    (Eq. 2.15-as-printed terms ``(s, t, ω(s,t), f(x+s,y+t))``), ``h_xy_conv`` (Eq. 2.14 value), ``h_xy_corr``
    (Eq. 2.15-as-printed value) and, for backward compatibility, ``terms`` = ``terms_conv`` and ``h_xy`` =
    ``h_xy_conv``.
    """
    rng = np.random.default_rng(seed)
    f = rng.integers(0, 10, size=(6, 6)).astype(np.float64)
    w = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])  # vertical Sobel-type kernel
    h_conv = conv2(f, w, "same")
    h_corr = imfilter(f, w, "corr")
    h_conv_imfilter = imfilter(f, w, "conv")
    x, y = 2, 3
    terms_conv = []  # Eq. (2.14): ω(s,t) · f(x−s, y−t)
    terms_corr = []  # Eq. (2.15) as printed: ω(s,t) · f(x+s, y+t)  (correlation form)
    for s in (-1, 0, 1):
        for t in (-1, 0, 1):
            terms_conv.append((s, t, w[s + 1, t + 1], f[x - s, y - t]))
            terms_corr.append((s, t, w[s + 1, t + 1], f[x + s, y + t]))
    h_xy_conv = conv_at(f, w, x, y)                   # Eq. (2.14)
    h_xy_corr = conv_at(f, w, x, y, correlate=True)   # Eq. (2.15) as printed
    assert abs(h_xy_conv - h_conv[x, y]) < 1e-12 and np.allclose(h_conv, h_conv_imfilter)
    assert abs(h_xy_corr - h_corr[x, y]) < 1e-12
    assert np.allclose(h_corr, conv2(f, w[::-1, ::-1], "same"))
    return {"f": f, "w": w, "h_conv2": h_conv, "h_imfilter_corr": h_corr, "x": x, "y": y,
            "terms_conv": terms_conv, "terms_corr": terms_corr, "h_xy_conv": h_xy_conv, "h_xy_corr": h_xy_corr,
            # backward-compatible aliases (Eq. 2.14 reading)
            "terms": terms_conv, "h_xy": h_xy_conv}


# ---------------------------------------------------------------------------------------------------------------
# §2.6 text-only: set and logical operations (Figs. 2.16, 2.17, Table 2.1)
# ---------------------------------------------------------------------------------------------------------------


def set_operation_examples(gray: np.ndarray | None = None) -> dict[str, Any]:
    """All operations of §2.6 on the synthetic sets of Fig. 2.16 (A, B) and Fig. 2.17 (L-shape), plus the
    grayscale set operations (Eqs. 2.28–2.30) on ``gray`` if given and the bitwise example ``57 AND 207 = 9``."""
    fx = synth.set_operation_fixtures()
    A, B, L, origin = fx["A"], fx["B"], fx["L"], fx["L_origin"]
    L_hat, origin_hat = reflect(L, origin)
    z = (1, 3)
    out: dict[str, Any] = {
        "A": A, "B": B, "A_c": complement(A), "A_or_B": union(A, B), "A_and_B": intersection(A, B),
        "A_minus_B": difference(A, B), "L": L, "L_origin": origin, "L_hat": L_hat, "L_hat_origin": origin_hat,
        "z": z, "L_z": translate(L, z), "bitwise_57_and_207": int(bitwise_and(np.uint8(57), np.uint8(207))),
    }
    if gray is not None:
        g = np.asarray(gray)
        gc = gray_complement(g)
        out.update(gray=g, gray_c=gc, gray_union=gray_union(g, gc), gray_intersection=gray_intersection(g, gc))
    return out


# ---------------------------------------------------------------------------------------------------------------
# §2.7  chain code/chain_diff.m
# ---------------------------------------------------------------------------------------------------------------


def chain_code_demo(B: np.ndarray | None = None) -> dict[str, Any]:
    """Boundary tracing and chain codes of the Fig. 2.19 object — port of ``MATLAB_ROOT/ch2/chain code/chain_diff.m``.

    Book: §2.7, Figs. 2.19(b, c), 2.20, 2.21.

    Steps as in the script: ``b = boundaries(B, 8, 'cww')`` (the ``'cww'`` typo is not ``'ccw'``, so MATLAB
    traces **clockwise** — reproduced with ``direction='cw'``), keep the longest boundary, ``bim = bound2im(b, M,
    N, min(b(:,1)), min(b(:,2)))``, ``c = fchcode(b)``.  Extras for the book figures: the 4-direction code
    (``boundaries(B, 4)`` + ``fchcode(b4, 4)``, Fig. 2.19b) and the normalised first difference (Fig. 2.21 last
    line, not computed by ``fchcode.m``).

    Returns ``B, b (0-based), b_matlab (1-based), bim, c (ChainCode), d (lengths), b4, c4, bim4,
    sequences (dict of the five Fig. 2.21 rows), table (Fig. 2.20 rows: 1-based coordinates + code)``.
    """
    B = synth.FIG_2_19_OBJECT if B is None else (np.asarray(B) != 0)
    M, N = B.shape
    b_all = boundaries(B, 8, "cw")  # 'cww' in chain_diff.m → clockwise
    d = np.array([len(x) for x in b_all])  # cellfun('length', b)
    k = int(np.argmax(d))
    b = b_all[0]  # b = b{1} (the script takes the first cell, not b{k}); identical here (one object)
    bim = bound2im(b, M, N, int(b[:, 0].min()), int(b[:, 1].min()))
    c: ChainCode = fchcode(b)
    b4_all = boundaries(B, 4, "cw")
    b4 = b4_all[0]
    c4 = fchcode(b4, 4)
    bim4 = bound2im(b4, M, N, int(b4[:, 0].min()), int(b4[:, 1].min()))
    nfd = normalized_first_difference(c.fcc, 8)
    pts = chain_to_points(c.x0y0, c.fcc, 8)
    assert np.array_equal(pts[:-1], b[:-1]) and np.array_equal(pts[-1], b[0])
    sequences = {
        "Original chain code": c.fcc, "First difference": c.diff, "Normalized chain code": c.mm,
        "First difference (of normalized)": c.diffmm, "Normalized first difference": nfd,
    }
    table = [(int(r) + 1, int(cc) + 1, int(code)) for (r, cc), code in zip(b[:-1], c.fcc)]
    return {"B": B, "b": b, "b_matlab": b + 1, "bim": bim, "c": c, "d": d, "k_max": k, "b4": b4, "c4": c4,
            "bim4": bim4, "sequences": sequences, "table": table, "x0y0_matlab": c.x0y0_matlab}


# ---------------------------------------------------------------------------------------------------------------
# §2.8 text-only: interpolation (Figs. 2.22–2.25)
# ---------------------------------------------------------------------------------------------------------------


def interpolation_demo(img: np.ndarray, crop: tuple[int, int, int, int] = (699, 731, 899, 931),
                       step: float = 0.4, scale: float = 4.0, a: float = -0.5) -> dict[str, Any]:
    """Upsample a crop of a grayscale image with nearest / bilinear / bicubic interpolation (§2.8).

    ``crop = (r0, r1, c0, c1)`` in 0-based half-open Python slices (default = MATLAB ``G(700:731, 900:931)``).
    Two query grids are used: (i) ``interp2``-style ``0:step:31`` (MATLAB ``meshgrid(1:0.4:32)``) and (ii)
    :func:`~seaice.core.interp.resize` by ``scale`` with the ``imresize`` pixel-centre convention.
    """
    img = np.asarray(img)
    G = rgb2gray_matlab(img).astype(np.float64) if img.ndim == 3 else img.astype(np.float64)
    r0, r1, c0, c1 = crop
    P = G[r0:r1, c0:c1]
    q = np.arange(0.0, P.shape[0] - 1 + 1e-9, step)
    U, V = np.meshgrid(q, q, indexing="ij")
    out: dict[str, Any] = {"P": P, "U": U, "V": V, "a": a}
    for name in ("nearest", "bilinear", "bicubic"):
        kw = {"a": a} if name == "bicubic" else {}
        out["grid_" + name] = interp2(P, U, V, name, **kw)
        out["resize_" + name] = resize(P, scale, name, **kw)
    xs = np.linspace(-2.5, 2.5, 501)
    out["kernel_x"] = xs
    out["kernel_keys"] = keys_kernel(xs, a)
    out["kernel_keys_opencv"] = keys_kernel(xs, -0.75)
    return out
