"""Pixel neighbourhoods, adjacency, paths, connected components and region boundaries — Book §2.3.

Text-only section (no MATLAB code in ch2), but ``bwlabel`` is used by ``chain code/boundaries.m`` and by almost
every later chapter, so :func:`label_components` reproduces MATLAB ``bwlabel`` including its *label numbering*
(components numbered in column-major order of first occurrence).

Coordinates are 0-based ``(row, col)`` tuples; the book writes ``(x, y)`` with ``x`` = row, ``y`` = column
(Eq. 2.1).
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from skimage.measure import label as _sk_label

Pixel = tuple[int, int]


def _in_bounds(p: Pixel, shape: tuple[int, ...]) -> bool:
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def n4(p: Pixel, shape: tuple[int, ...] | None = None) -> list[Pixel]:
    """4-neighbours ``N4(p) = {(x+1,y), (x-1,y), (x,y+1), (x,y-1)}`` (Book §2.3.1, Fig. 2.9a), each at distance 1.

    ``shape`` (rows, cols) drops neighbours that fall outside the image (border pixels have fewer neighbours).
    """
    x, y = p
    cand = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    return [q for q in cand if shape is None or _in_bounds(q, shape)]


def nd(p: Pixel, shape: tuple[int, ...] | None = None) -> list[Pixel]:
    """Diagonal neighbours ``ND(p) = {(x+1,y+1), (x+1,y-1), (x-1,y+1), (x-1,y-1)}`` (§2.3.1, Fig. 2.9b), distance √2."""
    x, y = p
    cand = [(x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)]
    return [q for q in cand if shape is None or _in_bounds(q, shape)]


def n8(p: Pixel, shape: tuple[int, ...] | None = None) -> list[Pixel]:
    """8-neighbours ``N8(p) = N4(p) ∪ ND(p)`` (§2.3.1, Fig. 2.9c)."""
    return n4(p, shape) + nd(p, shape)


def _in_V(img: np.ndarray | None, p: Pixel, V: Iterable | None) -> bool:
    """True if pixel ``p`` has a value in the adjacency set ``V`` (default: nonzero / True)."""
    if img is None:
        return True
    if not _in_bounds(p, img.shape):
        return False
    v = img[p[0], p[1]]
    if V is None:
        return bool(v)
    return v in set(V)


def is_adjacent(p: Pixel, q: Pixel, conn: int | str = 4, img: np.ndarray | None = None,
                V: Iterable | None = None) -> bool:
    """Adjacency test of Book §2.3.2 for ``conn`` in ``{4, 8, 'm'}``.

    * 4-adjacent: ``q ∈ N4(p)``;  8-adjacent: ``q ∈ N8(p)``;
    * m-adjacent: ``q ∈ N4(p)``, or ``q ∈ ND(p)`` and ``N4(p) ∩ N4(q)`` has no pixel with a value in ``V``.

    Both ``p`` and ``q`` must have values in ``V`` (default ``V`` = nonzero) when ``img`` is given.
    """
    if img is not None and not (_in_V(img, p, V) and _in_V(img, q, V)):
        return False
    shape = None if img is None else img.shape
    if conn == 4:
        return q in n4(p, shape)
    if conn == 8:
        return q in n8(p, shape)
    if conn in ("m", "M"):
        return is_m_adjacent(p, q, img, V)
    raise ValueError("conn must be 4, 8 or 'm'")


def is_m_adjacent(p: Pixel, q: Pixel, img: np.ndarray | None = None, V: Iterable | None = None) -> bool:
    """Mixed adjacency (Book §2.3.2 (c)): removes the ambiguity of 8-paths (Fig. 2.10c)."""
    if img is not None and not (_in_V(img, p, V) and _in_V(img, q, V)):
        return False
    shape = None if img is None else img.shape
    if q in n4(p, shape):
        return True
    if q in nd(p, shape):
        common = set(n4(p, shape)) & set(n4(q, shape))
        return not any(_in_V(img, r, V) for r in common)
    return False


def find_paths(img: np.ndarray, p: Pixel, q: Pixel, conn: int | str = 4, V: Iterable | None = None,
               max_paths: int = 1000) -> list[list[Pixel]]:
    """Enumerate all simple paths ``p1, ..., pn`` from ``p`` to ``q`` through pixels with values in ``V``.

    Book: §2.3.3 (path = sequence of adjacent pixels, length ``n``; Fig. 2.10 shows that 8-adjacency admits
    several paths while m-adjacency leaves exactly one).  Depth-first search; intended for tiny examples.
    """
    img = np.asarray(img)
    paths: list[list[Pixel]] = []

    def neighbours(a: Pixel) -> list[Pixel]:
        cands = n4(a, img.shape) if conn == 4 else n8(a, img.shape)
        return [b for b in cands if is_adjacent(a, b, conn, img, V)]

    def dfs(a: Pixel, path: list[Pixel]) -> None:
        if len(paths) >= max_paths:
            return
        if a == q:
            paths.append(list(path))
            return
        for b in neighbours(a):
            if b not in path:
                path.append(b)
                dfs(b, path)
                path.pop()

    if _in_V(img, p, V):
        dfs(p, [p])
    return paths


def label_components(bw: np.ndarray, conn: int = 8) -> np.ndarray:
    """Connected-component labelling with MATLAB ``bwlabel`` numbering.

    Book: §2.3.4–2.3.5 (connected components / regions; Fig. 2.11: 5 components with 4-adjacency, 2 with
    8-adjacency).  MATLAB source: ``chain code/boundaries.m`` line ``L = bwlabel(BW, conn)``; used by ch5–ch9.

    Parameters
    ----------
    bw : ndarray (M, N)
        Binary image (nonzero = object).
    conn : {4, 8}, default 8
        MATLAB ``bwlabel`` default is 8.

    Returns
    -------
    labels : int32 array (M, N)
        0 = background; components numbered 1..n **in column-major order of their first pixel** (``find``-order),
        which is how ``bwlabel`` numbers runs.  Partition = ``skimage.measure.label`` (exact); numbering
        relabelled to match MATLAB.
    """
    bw = np.asarray(bw) != 0
    if conn not in (4, 8):
        raise ValueError("conn must be 4 or 8")
    lab = _sk_label(bw, connectivity=1 if conn == 4 else 2, background=0)
    flat = lab.flatten(order="F")  # column-major scan like MATLAB
    n = int(flat.max()) if flat.size else 0
    if n == 0:
        return lab.astype(np.int32)
    # first occurrence (in column-major order) of each label 1..n
    first = np.full(n + 1, flat.size, dtype=np.int64)
    idx = np.nonzero(flat)[0]
    # np.minimum.at handles repeated labels; positions are ascending so the first hit is the minimum
    np.minimum.at(first, flat[idx], idx)
    order = np.argsort(first[1:], kind="stable") + 1  # labels sorted by first appearance
    remap = np.zeros(n + 1, dtype=np.int32)
    remap[order] = np.arange(1, n + 1, dtype=np.int32)
    return remap[lab]


def count_components(bw: np.ndarray, conn: int = 8) -> int:
    """Number of connected components (``max(bwlabel(BW, conn)(:))``), Book §2.3.4 / Fig. 2.11."""
    return int(label_components(bw, conn).max())


def region_boundary_mask(bw: np.ndarray, conn: int = 8) -> np.ndarray:
    """Boundary of a region: object pixels with at least one ``conn``-neighbour outside the region.

    Book: §2.3.5 ("the set of pixels in the region for which one or more neighbours are not in R"; the boundary
    of a finite region is a closed path).  Pixels touching the image border count as boundary pixels (their
    missing neighbours are treated as background, as MATLAB ``bwperim`` does).  Note MATLAB ``bwperim`` defaults
    to ``conn = 4``; pass ``conn=4`` for that behaviour.
    """
    bw = np.asarray(bw) != 0
    padded = np.pad(bw, 1, constant_values=False)
    shifts = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if conn == 8:
        shifts += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    elif conn != 4:
        raise ValueError("conn must be 4 or 8")
    has_bg_neighbour = np.zeros_like(bw)
    M, N = bw.shape
    for dr, dc in shifts:
        nb = padded[1 + dr:1 + dr + M, 1 + dc:1 + dc + N]
        has_bg_neighbour |= ~nb
    return bw & has_bg_neighbour
