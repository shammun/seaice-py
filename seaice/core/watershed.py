"""MATLAB ``watershed`` — Meyer's flooding from the regional minima, ported line by line from the readable codegen
twin ``toolbox/images/images/eml/watershed.m`` (R2025a; the interpreted ``watershed.m`` calls the compiled builtin
``images.internal.builtins.watershed_meyer``, which the analyst's pre-check matched 0 px on 27 cases).

Book: Chapter 5 §5.1 (pp. 84–89), Eqs. (5.1)–(5.8) (immersion view: threshold sets ``T_h``, catchment basins
``C(M_i)``, dams), Fig. 5.2–5.4; the connectivity duality of pp. 88–89 (8-connected basins ↔ 4-connected lines).
MATLAB sources that call it: every ``MATLAB_ROOT/ch5/*watershed*.m`` script and ``ch5/watershed_based/main.m``
(``L = watershed(-D)``), later ch7–ch9.

What the M-code does (and this port reproduces, in this order):

1. ``[L, n] = bwlabel(imregionalmin(A, conn), conn)`` — basins are numbered like ``bwlabel`` (column-major first
   pixel; :func:`seaice.core.connectivity.label_components`).
2. A FIFO priority queue (min-heap on ``(priority, insertion order)``; ``FifoPriorityQueue.m``) and a visited flag
   ``S`` per pixel.  Initial scan: every pixel ``i`` in **linear (column-major) order**; if ``L(i) ~= 0`` mark
   ``S(i)`` and push each neighbour that is unvisited and unlabelled with priority ``A(nb)``.
3. Flooding: pop ``(d, p)``; scan the neighbours of ``d`` in the ``find``-order of the 3×3 connectivity matrix
   (column-major offsets ``(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)`` = ascending linear
   index; out-of-image neighbours dropped, ``Padding = NONE``).  If two *different* labels are seen the pixel is a
   watershed pixel: it keeps label 0 **and pushes nothing**.  Otherwise ``L(d) = label`` and every unvisited
   neighbour is pushed with priority ``max(A(nb), p)``.  A pixel is pushed at most once.

The priority queue is ``heapq`` on ``(priority, order, index)``; ``FifoPriorityQueue.push/pop`` (sift rules on
priority first, insertion order second) implement exactly that lexicographic order.  The image is handled on a
1-pixel pad whose cells are pre-marked *visited* with label 0, which removes the bounds checks without changing
which neighbours are seen or in which order.

Differences from ``skimage.segmentation.watershed(..., watershed_line=True)`` (why it cannot be used for parity):
skimage seeds the queue with the marker pixels themselves, re-pushes pixels, tags every queue entry with a source
label and marks a line pixel when a *neighbour's* label differs from that source, keeps propagating from line
pixels, and visits neighbours sorted by distance then row-major — 46 of 63 ice-interior ridge pixels differ on the
city-block case of ``q.jpg`` (analysis/ch05.md).  :func:`watershed_skimage` is kept only as a labelled cross-check.

Parity: exact (target; label values included).  Output int32 (MATLAB picks uint8/uint16/… by region count).
"""
from __future__ import annotations

import heapq

import numpy as np

from .connectivity import label_components
from .morphology import conn_to_scalar, imregionalmin

__all__ = ["watershed", "watershed_skimage", "neighbour_offsets"]


def neighbour_offsets(conn: int, n_rows_padded: int) -> list[int]:
    """Linear (column-major) neighbour offsets in MATLAB's ``NeighborhoodProcessor`` order.

    ``computeParameters`` walks the 3×3 connectivity matrix with ``for pind = 1:numel(nhConn)`` (column-major), so
    the neighbours of a pixel are visited as ``(-1,-1), (0,-1), (1,-1), (-1,0), (1,0), (-1,1), (0,1), (1,1)`` for
    ``conn = 8`` and ``(0,-1), (-1,0), (1,0), (0,1)`` for ``conn = 4`` — i.e. by ascending linear index.
    ``n_rows_padded`` is the row count of the (padded) column-major frame.
    """
    Mp = int(n_rows_padded)
    if conn == 8:
        return [-Mp - 1, -Mp, -Mp + 1, -1, 1, Mp - 1, Mp, Mp + 1]
    if conn == 4:
        return [-Mp, -1, 1, Mp]
    raise ValueError("watershed: conn must be 4 or 8 (MATLAB images:watershed:limitedConn)")


def watershed(A: np.ndarray, conn: int | np.ndarray = 8) -> np.ndarray:
    """MATLAB ``L = watershed(A[, conn])`` — Meyer flooding from the regional minima; ``0`` on the watershed lines.

    Book: §5.1 (pp. 84–89; the immersion formulation Eqs. 5.1–5.8 is the teaching form,
    :func:`seaice.ch05_watershed.watershed_immersion`), §5.1.1–5.1.3 (segmentation functions), §5.2 Step 2
    ("8-connected watershed ... each watershed line forms a 4-connected path", p. 99).  MATLAB source: R2025a
    ``toolbox/images/images/eml/watershed.m`` (see module docstring); called by ``direct_watershed.m``
    (``watershed(im)`` on uint8), ``gradients_watershed.m`` (``watershed(g)``), ``distance_watershed.m`` /
    ``marker_watershed.m`` (``watershed(imgDist)`` on single) and ``watershed_based/main.m`` (``watershed(-D)``).

    Parameters
    ----------
    A : ndarray (M, N)
        Any real numeric or bool image (``±Inf`` allowed, NaN rejected like MATLAB's ``imregionalmin``).  The
        values are used as flooding priorities in double precision (``FifoPriorityQueue.push`` casts to double).
        Note that MATLAB's ``watershed`` itself **rejects int16/int32 input** (R2025a ``watershed.m`` line 155:
        ``validateattributes(A, {'uint8','uint16','single','double','logical'}, ...)``); the port accepts every
        real numeric width, and the MATLAB cross-check of the int16 :func:`seaice.core.synth.plateau_fixtures`
        is done on ``double(X)``, which floods identically because only the value order matters.
    conn : {8, 4} or 3×3 array
        8 (MATLAB default): 8-connected basins, 4-connected lines; 4: 4-connected basins, 8-connected lines.
        The matrix forms ``ones(3)`` (= 8) and the cross ``[0 1 0; 1 1 1; 0 1 0]`` (= 4) are accepted like
        MATLAB; any other matrix raises ``ValueError`` (MATLAB ``images:watershed:limitedConn``), see
        :func:`seaice.core.morphology.conn_to_scalar`.

    Returns
    -------
    ndarray int32 (M, N)
        ``0`` = watershed (ridge) pixels; basins ``1..n`` numbered like ``bwlabel(imregionalmin(A, conn), conn)``
        (MATLAB returns uint8/uint16/… depending on ``n``; compare after casting).

    Parity: exact (label values included) — verified against the compiled builtin on the chapter's images and
    constructed plateau/tie fixtures (chapter report).
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("watershed: 2-D images only (images:validate:twoDimensionalImageSupport)")
    # 4 / 8 / cross(3) / ones(3) → scalar; anything else raises (images:watershed:limitedConn)
    conn = conn_to_scalar(conn)
    if A.dtype == np.bool_:
        A = A.astype(np.uint8)  # imregionalmin strips the logical flag (+I)
    if np.issubdtype(A.dtype, np.floating) and np.isnan(A).any():
        raise ValueError("watershed: NaN values are not allowed (imregionalmin 'nonnan')")
    if A.size == 0:
        return np.zeros(A.shape, dtype=np.int32)
    M, N = A.shape
    offsets = neighbour_offsets(conn, M + 2)

    # [Lin, numConns] = bwlabel(imregionalmin(A, conn), conn)
    L0 = label_components(imregionalmin(A, conn), conn)

    # Column-major linear frame with a 1-pixel pad (pad cells: visited, label 0 → never pushed, never counted).
    Ap = np.pad(A.astype(np.float64), 1, mode="constant", constant_values=0.0)
    Lp = np.pad(L0.astype(np.int64), 1, mode="constant", constant_values=0)
    Af = Ap.T.ravel().tolist()  # linear index i = c * (M + 2) + r  (MATLAB sub2ind order)
    Lf = Lp.T.ravel().tolist()
    Sp = np.ones(Ap.shape, dtype=np.uint8)
    Sp[1:-1, 1:-1] = 0
    S = bytearray(Sp.T.ravel().tobytes())

    heap: list[tuple[float, int, int]] = []
    push, pop = heapq.heappush, heapq.heappop
    order = 0
    WSHED = 0

    # Initial scan: for i = 1:numelA (column-major over the *original* pixels)
    Mp = M + 2
    for c in range(1, N + 1):
        base = c * Mp
        for i in range(base + 1, base + M + 1):
            if Lf[i] != WSHED:
                S[i] = 1
                for off in offsets:
                    nb = i + off
                    if not S[nb] and Lf[nb] == WSHED:
                        S[nb] = 1
                        order += 1
                        push(heap, (Af[nb], order, nb))

    # Flooding: while ~queue.isempty()
    while heap:
        p, _, d = pop(heap)
        watershed_state = False
        label = WSHED
        for off in offsets:
            ln = Lf[d + off]
            if not watershed_state and ln != WSHED:
                if label != WSHED and ln != label:
                    watershed_state = True
                else:
                    label = ln
        if not watershed_state:
            Lf[d] = label
            for off in offsets:
                nb = d + off
                if not S[nb]:
                    S[nb] = 1
                    order += 1
                    a = Af[nb]
                    push(heap, (a if a > p else p, order, nb))

    Lout = np.asarray(Lf, dtype=np.int64).reshape(N + 2, Mp).T[1:-1, 1:-1]
    return np.ascontiguousarray(Lout, dtype=np.int32)


def watershed_skimage(A: np.ndarray, conn: int = 8) -> np.ndarray:
    """Cross-check only: ``skimage.segmentation.watershed(A, connectivity=..., watershed_line=True)``.

    Same idea (flooding from the regional minima, 0 on lines) but a different algorithm — see the module docstring
    for the four rules that differ — so ridges and even the basin partition disagree with MATLAB on plateaus.
    Provided for the notebook's "why not the library?" cell; never used by the ported pipelines.
    """
    from skimage.segmentation import watershed as _sk_watershed

    A = np.asarray(A)
    if A.dtype == np.bool_:
        A = A.astype(np.uint8)
    # PARITY: approx — skimage's flooding seeds with the minima pixels, re-pushes pixels, propagates from line
    # pixels and scans neighbours in a different order; 46–122 ridge pixels differ from MATLAB on q.jpg.
    connectivity = 2 if conn_to_scalar(conn) == 8 else 1
    return _sk_watershed(A.astype(np.float64), connectivity=connectivity, watershed_line=True).astype(np.int32)
