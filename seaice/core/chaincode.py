"""Boundary tracing and Freeman chain codes — Book §2.7, Eq. (2.31), Figs. 2.18–2.21.

MATLAB sources (DIPUM functions shipped in ``MATLAB_ROOT/ch2/chain code/`` and byte-identical copies in ch5):

* ``boundaries.m``  → :func:`boundaries`   (Moore-neighbour tracing with Jacob's stopping rule)
* ``fchcode.m``     → :func:`fchcode` (+ :func:`code_reverse`, :func:`min_magnitude`, :func:`first_difference`)
* ``bound2im.m``    → :func:`bound2im`

Coordinate convention
---------------------
Everything here is **0-based ``(row, col)``**: boundary arrays are ``(Q, 2)`` with ``b[:, 0]`` = row,
``b[:, 1]`` = column; ``ChainCode.x0y0`` and the ``x0, y0`` arguments of :func:`bound2im` are 0-based too.
MATLAB's outputs are 1-based, so tests add 1 (``b_matlab == b_python + 1``).  The book writes ``(x, y)`` with
``x`` = row and ``y`` = column, so ``x0y0 = (row, col)``.

Direction numbering (Fig. 2.18, 8-connected): ``0`` = east (+col), ``1`` = north-east, ``2`` = north (−row),
``3`` = north-west, ``4`` = west, ``5`` = south-west, ``6`` = south (+row), ``7`` = south-east.  The 4-connected
code is ``0`` = east, ``1`` = north, ``2`` = west, ``3`` = south (= 8-code / 2).

Deliberate replication of DIPUM behaviour (ch5 depends on it — do not "fix"):

* only the *exterior* boundary of each labelled object is traced (holes are never traced);
* objects are traced in ``bwlabel`` label order, starting at the first object pixel (column-major scan) whose
  north neighbour is background, initial search direction north-east, clockwise;
* the output is closed (first point repeated at the end); a single-pixel object gives two identical points;
* shapes with one-pixel-wide spurs are traversed twice along the spur.

Deviation from ``fchcode.m``: ``minmag`` never returns when the code is periodic (all remaining candidates are
identical) and MATLAB raises "Output argument z not assigned".  :func:`min_magnitude` breaks the tie
deterministically by returning the first remaining candidate (the one with the smallest starting index), which
reproduces the book's Fig. 2.21 "normalised first difference".
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .connectivity import label_components
from .matlab_compat import matlab_round

# fchcode.m code table: z = 4*(dx + 2) + (dy + 2) → Freeman code (dx = Δrow, dy = Δcol).  MATLAB's C is a 1×15
# vector with zeros at unset indices, so a zero step (dx = dy = 0 → z = 10) is silently coded 0; replicated.
_CODE_TABLE = np.zeros(16, dtype=np.int64)
for _z, _code in ((11, 0), (7, 1), (6, 2), (5, 3), (9, 4), (13, 5), (14, 6), (15, 7)):
    _CODE_TABLE[_z] = _code

#: (Δrow, Δcol) step for each 8-direction code (Fig. 2.18b).
STEPS_8 = np.array([(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)], dtype=np.int64)
#: (Δrow, Δcol) step for each 4-direction code (Fig. 2.18a).
STEPS_4 = STEPS_8[::2].copy()


@dataclass
class ChainCode:
    """Output of :func:`fchcode` (mirrors the MATLAB struct ``c``).

    Attributes
    ----------
    fcc : ndarray of int
        Freeman chain code (Fig. 2.20).
    diff : ndarray of int
        First difference of ``fcc`` (Eq. 2.31a–b), ``c.diff``.
    mm : ndarray of int
        ``fcc`` rotated to the integer of minimum magnitude (start-point normalisation), ``c.mm``.
    diffmm : ndarray of int
        First difference of ``mm``, ``c.diffmm`` (this is what ``fchcode.m`` computes; the book's "normalised first
        difference" is :func:`normalized_first_difference` instead).
    x0y0 : tuple (row, col)
        0-based starting point (``c.x0y0 - 1``).
    conn : int
        4 or 8.
    """

    fcc: np.ndarray
    diff: np.ndarray
    mm: np.ndarray
    diffmm: np.ndarray
    x0y0: tuple[int, int]
    conn: int = 8
    x0y0_matlab: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.x0y0_matlab = (int(self.x0y0[0]) + 1, int(self.x0y0[1]) + 1)


# --------------------------------------------------------------------------------------------------------------
# boundaries.m
# --------------------------------------------------------------------------------------------------------------
def boundaries(bw: np.ndarray, conn: int = 8, direction: str = "cw") -> list[np.ndarray]:
    """Trace the exterior boundary of every object in a binary image — port of DIPUM ``boundaries.m``.

    Book: §2.3.5 (boundary = closed path), §2.7 (chain-code input), Figs. 2.19–2.21.  MATLAB source:
    ``MATLAB_ROOT/ch2/chain code/boundaries.m`` (also ``ch5/boundaries.m``).

    Parameters
    ----------
    bw : ndarray (M, N)
        Binary image (nonzero = object).
    conn : {8, 4}, default 8
        Connectivity used both for labelling (``bwlabel``) and for tracing.
    direction : {'cw', 'ccw'}, default 'cw'
        ``'ccw'`` simply reverses each traced list.  (``chain_diff.m`` passes the typo ``'cww'``, which MATLAB
        treats as anything-but-``'ccw'`` → clockwise; pass ``'cw'`` to reproduce it.)

    Returns
    -------
    list of int64 arrays (Q, 2)
        ``B[k]`` = 0-based ``(row, col)`` boundary of object ``k + 1`` in ``bwlabel`` order; closed (first row is
        repeated at the end).  Objects are traced exactly as the MATLAB code does (see module docstring).

    Parity: reimplemented line by line; target exact vs MATLAB (after ``+1``).
    """
    if conn not in (4, 8):
        raise ValueError("conn must be 4 or 8")
    if direction not in ("cw", "ccw"):
        raise ValueError("direction must be 'cw' or 'ccw'")
    reverse = direction == "ccw"  # strcmp(dir, 'ccw')
    bw = np.asarray(bw) != 0
    L = label_components(bw, conn)
    num_objects = int(L.max())
    B: list[np.ndarray] = [np.zeros((0, 2), dtype=np.int64) for _ in range(num_objects)]
    if num_objects == 0:
        return B

    # Pad label matrix with zeros; work on a column-major flat copy so that MATLAB's linear-index offsets apply.
    Lp2 = np.pad(L.astype(np.int64), 1, constant_values=0)
    M = Lp2.shape[0]  # rows of the padded matrix (= size(Lp, 1))
    Lp = Lp2.flatten(order="F")  # linear index idx = c * M + r  (0-based column-major)

    if conn == 8:
        offsets = np.array([-1, M - 1, M, M + 1, 1, -M + 1, -M, -M - 1])  # N NE E SE S SW W NW
        next_search_direction_lut = np.array([8, 8, 2, 2, 4, 4, 6, 6])
        next_direction_lut = np.array([2, 3, 4, 5, 6, 7, 8, 1])
    else:
        offsets = np.array([-1, M, 1, -M])  # N E S W
        next_search_direction_lut = np.array([4, 1, 2, 3])
        next_direction_lut = np.array([2, 3, 4, 1])
    START, BOUNDARY = -1, -2

    # Candidate starting locations: object pixels whose north neighbour is background, in column-major order
    # ([rr, cc] = find(Lp(2:end-1, :) > 0 & Lp(1:end-2, :) == 0); rr = rr + 1).
    cand = (Lp2[1:-1, :] > 0) & (Lp2[:-2, :] == 0)
    cc_idx, rr_idx = np.nonzero(cand.T)  # transpose → column-major enumeration
    rr_idx = rr_idx + 1

    for r, c in zip(rr_idx.tolist(), cc_idx.tolist()):
        idx = c * M + r
        lab = int(Lp[idx])
        if not (lab > 0 and Lp[idx - 1] == 0 and B[lab - 1].shape[0] == 0):
            continue
        # Found the start of the next boundary.
        which = lab
        scratch = [idx]
        Lp[idx] = START
        currentpixel = idx
        initial_departure_direction: int | None = None
        done = False
        next_search_direction = 2
        while not done:
            direction = next_search_direction
            found_next_pixel = False
            for _ in range(len(offsets)):
                neighbor = currentpixel + int(offsets[direction - 1])
                if Lp[neighbor] != 0:
                    if Lp[currentpixel] == START and initial_departure_direction is None:
                        # initial departure from the starting pixel
                        initial_departure_direction = direction
                    elif Lp[currentpixel] == START and initial_departure_direction == direction:
                        # about to retrace our path → done
                        done = True
                        found_next_pixel = True
                        break
                    # take the next step along the boundary
                    next_search_direction = int(next_search_direction_lut[direction - 1])
                    found_next_pixel = True
                    scratch.append(neighbor)
                    if Lp[neighbor] != START:
                        Lp[neighbor] = BOUNDARY
                    currentpixel = neighbor
                    break
                direction = int(next_direction_lut[direction - 1])
            if not found_next_pixel:
                # no neighbour at all: single-pixel object
                scratch = [scratch[0], scratch[0]]
                done = True
        lin = np.asarray(scratch, dtype=np.int64)
        rows = lin % M - 1  # ind2sub of the padded matrix, then remove the pad (MATLAB: row - 1 → 1-based)
        cols = lin // M - 1
        B[which - 1] = np.stack([rows, cols], axis=1)

    if reverse:
        B = [b[::-1].copy() for b in B]
    return B


# --------------------------------------------------------------------------------------------------------------
# fchcode.m and its local functions
# --------------------------------------------------------------------------------------------------------------
def code_reverse(fcc: np.ndarray) -> np.ndarray:
    """Traverse an 8-connected code in the opposite direction (``fchcode.m`` local ``coderev``): flip, then ±4."""
    cr = np.asarray(fcc, dtype=np.int64)[::-1].copy()
    lo = (cr >= 0) & (cr <= 3)
    hi = (cr >= 4) & (cr <= 7)
    cr[lo] += 4
    cr[hi] -= 4
    return cr


def min_magnitude(code: np.ndarray) -> np.ndarray:
    """Rotate a circular chain code to the integer of minimum magnitude (``fchcode.m`` local ``minmag``).

    Book: §2.7 (start-point normalisation, Fig. 2.21 "Normalized chain code").  Candidates are all rotations
    that start with ``min(code)``; they are eliminated column by column keeping the minima.

    Deviation: when the code is periodic, several candidates stay tied through the last column and MATLAB's
    ``minmag`` ends without assigning its output (runtime error).  Here the **first** remaining candidate (the
    rotation with the smallest starting index) is returned, which yields the book's normalised first
    difference.  Equivalent to the lexicographically smallest rotation.
    """
    c = np.asarray(code, dtype=np.int64).ravel()
    n = c.size
    if n == 0:
        return c.copy()
    starts = np.nonzero(c == c.min())[0]
    A = np.stack([np.roll(c, -int(k)) for k in starts], axis=0)
    J = np.arange(A.shape[0])
    for k in range(1, n):
        col = A[J, k]
        J = J[col == col.min()]
        if J.size == 1:
            return A[J[0]].copy()
    # PARITY: minmag tie-break — periodic code; MATLAB errors here, we return the first candidate.
    return A[J[0]].copy()


def first_difference(code: np.ndarray, conn: int = 8) -> np.ndarray:
    """First difference of a circular chain code, Book Eq. (2.31a–b) (``fchcode.m`` local ``codediff``).

    ``D1(i) = (C(i+1) - C(i)) mod conn`` for ``i = 0..N-2`` and ``D1(N-1) = (C(0) - C(N-1)) mod conn``;
    counts counter-clockwise direction changes between consecutive segments (rotation invariant).
    """
    if conn not in (4, 8):
        raise ValueError("conn must be 4 or 8")
    c = np.asarray(code, dtype=np.int64).ravel()
    d = np.roll(c, -1) - c
    d[d < 0] += conn
    return d


def normalized_first_difference(code: np.ndarray, conn: int = 8) -> np.ndarray:
    """Min-magnitude rotation of the first difference (Book Fig. 2.21, last line: "Normalized first difference").

    Not computed by ``fchcode.m`` (whose ``diffmm`` is the difference of the normalised *code*); provided because
    the text describes it (§2.7, "the first difference can also be normalized ...").
    """
    return min_magnitude(first_difference(code, conn))


def fchcode(b: np.ndarray, conn: int = 8, direction: str = "same") -> ChainCode:
    """Freeman chain code of a closed boundary — port of DIPUM ``fchcode.m``.

    Book: §2.7, Figs. 2.19–2.21, Eq. (2.31).  MATLAB source: ``MATLAB_ROOT/ch2/chain code/fchcode.m``.

    Parameters
    ----------
    b : ndarray (np, 2)
        0-based ``(row, col)`` boundary points, 1-pixel-thick, fully connected, closed.  If the first and last
        points coincide (as :func:`boundaries` returns) the last one is dropped.
    conn : {8, 4}
        With ``conn=4`` all 8-codes must be even (they are halved); otherwise a warning is issued and the
        8-code is kept (MATLAB behaviour).
    direction : {'same', 'reverse'}
        ``'reverse'`` outputs the code traversed the other way (:func:`code_reverse`), same starting point.

    Returns
    -------
    ChainCode
        ``fcc``, ``diff`` (Eq. 2.31), ``mm`` (min-magnitude rotation), ``diffmm`` (difference of ``mm``),
        ``x0y0`` (0-based start), plus ``x0y0_matlab`` (1-based) for comparisons.

    Raises
    ------
    ValueError
        if consecutive points are more than one pixel apart ("curve is broken or points are out of order").
    """
    b = np.asarray(b, dtype=np.int64)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("b must be of size np-by-2")
    if b.shape[0] < b.shape[1]:
        raise ValueError("B must be of size np-by-2.")
    if direction not in ("same", "reverse"):
        raise ValueError("direction must be 'same' or 'reverse'")
    if b.shape[0] > 1 and np.array_equal(b[0], b[-1]):
        b = b[:-1]
    x0y0 = (int(b[0, 0]), int(b[0, 1]))
    a = np.roll(b, -1, axis=0)  # circshift(b, [-1, 0]): last row becomes the first row of b
    DEL = a - b
    if np.any(np.abs(DEL) > 1):
        raise ValueError("The input curve is broken or points are out of order.")
    z = 4 * (DEL[:, 0] + 2) + (DEL[:, 1] + 2)
    fcc = _CODE_TABLE[z]
    if direction == "reverse":
        fcc = code_reverse(fcc)
    if conn == 4:
        if np.any(fcc % 2 == 1):
            warnings.warn("The specified 4-connected code cannot be satisfied.", stacklevel=2)
        else:
            fcc = fcc // 2
    elif conn != 8:
        raise ValueError("conn must be 4 or 8")
    diff = first_difference(fcc, conn)
    mm = min_magnitude(fcc)
    diffmm = first_difference(mm, conn)
    return ChainCode(fcc=fcc, diff=diff, mm=mm, diffmm=diffmm, x0y0=x0y0, conn=conn)


def chain_to_points(x0y0: tuple[int, int], fcc: np.ndarray, conn: int = 8) -> np.ndarray:
    """Rebuild the boundary points from a start point and a chain code (Book Fig. 2.20, "Representations").

    Returns ``(len(fcc) + 1, 2)`` 0-based ``(row, col)`` points; for a closed boundary the last equals the first.
    """
    steps = STEPS_8 if conn == 8 else STEPS_4
    fcc = np.asarray(fcc, dtype=np.int64)
    pts = np.zeros((fcc.size + 1, 2), dtype=np.int64)
    pts[0] = x0y0
    pts[1:] = np.asarray(x0y0) + np.cumsum(steps[fcc], axis=0)
    return pts


# --------------------------------------------------------------------------------------------------------------
# bound2im.m
# --------------------------------------------------------------------------------------------------------------
def bound2im(b: np.ndarray, M: int | None = None, N: int | None = None, x0: int | None = None,
             y0: int | None = None) -> np.ndarray:
    """Convert a boundary to a binary image — port of DIPUM ``bound2im.m``.

    Book: §2.7 (display of Fig. 2.19/2.20 boundaries).  MATLAB source: ``MATLAB_ROOT/ch2/chain code/bound2im.m``
    (called in ``chain_diff.m`` as ``bound2im(b, M, N, min(b(:,1)), min(b(:,2)))``).

    Three calling conventions, as in MATLAB:

    * ``bound2im(b)`` — tight image of size ``max(x) × max(y)`` after shifting the boundary so its minimum is 0;
    * ``bound2im(b, M, N)`` — boundary approximately centred in an ``M × N`` image;
    * ``bound2im(b, M, N, x0, y0)`` — top-most point placed at row ``x0``, left-most at column ``y0``
      (**0-based** here; MATLAB's ``x0, y0`` are 1-based, so ``x0_python = x0_matlab - 1``).

    ``b`` may be ``(np, 2)`` or ``(2, np)`` (transposed if it has more columns than rows); coordinates are
    rounded MATLAB-style.  Raises ``ValueError`` if the boundary does not fit.  Parity: exact.
    """
    b = np.asarray(b, dtype=np.float64)
    if b.ndim != 2:
        raise ValueError("b must be 2-D")
    if b.shape[0] < b.shape[1]:
        b = b.T
    x = matlab_round(b[:, 0]).astype(np.int64)
    y = matlab_round(b[:, 1]).astype(np.int64)
    x = x - x.min()  # MATLAB: x - min(x) + 1  (1-based) → 0-based here
    y = y - y.min()
    C = x.max() - x.min() + 1
    D = y.max() - y.min() + 1
    nargs = sum(v is not None for v in (M, N, x0, y0))
    if nargs == 0:
        shape = (int(x.max()) + 1, int(y.max()) + 1)
    elif nargs == 2:
        if C > M or D > N:
            raise ValueError("The boundary is outside the M-by-N region.")
        NR = int(matlab_round((M - C) / 2))
        NC = int(matlab_round((N - D) / 2))
        x = x + NR
        y = y + NC
        shape = (int(M), int(N))
    elif nargs == 4:
        if x0 < 0 or y0 < 0:
            raise ValueError("x0 and y0 must be non-negative integers.")
        x = x + int(matlab_round(x0))
        y = y + int(matlab_round(y0))
        if C + x0 > M or D + y0 > N:
            raise ValueError("The shifted boundary is outside the M-by-N region.")
        shape = (int(M), int(N))
    else:
        raise ValueError("Incorrect number of inputs: use (b), (b, M, N) or (b, M, N, x0, y0).")
    B = np.zeros(shape, dtype=bool)
    B[x, y] = True
    return B
