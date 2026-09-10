"""The book's ``IceImage`` data structure — **Appendix B** (pp. 221–225) as Python dataclasses.

Appendix B is a *format*, not an algorithm, so it lives in :mod:`seaice.core` and not in a chapter module: ch8
§8.2.1 fills it from ``sea_ice_model.m``'s output and ch9's model-ice work will fill the same structure from its
rectangle fits.  The field names, the nesting and the units are copied verbatim from the book's own listing
(pp. 221–223), and the constructor that fills it is
:func:`seaice.ch08_applications.sea_ice_image_structure` (a port of
``MATLAB_ROOT/ch7/Sea_Ice_Floe_Identification/SeaIce_Image_Structure.m``, byte-identical to ch6's copy).

The shipped ``data/book/ch08/MCD/IceImage_290915_2_jpg.0000179.mat`` is exactly this structure, derived from the
§8.3 helicopter frame (Fig. 8.18): 2888 floes, 3452 brash pieces, a 1114×627 image.  Figures B.1–B.3 are MATLAB
IDE screenshots of it.

Conventions kept from MATLAB (do **not** "fix" them):

* every coordinate is **1-based** and written ``[index_x, index_y]`` = ``[column, row]`` (Appendix B, p. 222,
  and ch7's ``IcePiece.PixelsPosition``);
* ``Intersect.floe`` / ``Intersect.brash`` hold **1-based serial numbers** of the intersecting pieces;
* ``Polygon.Vertices`` is an **open** ring — ``SeaIce_Image_Structure.m`` lines 15–16 delete the duplicated
  closing vertex that ``sea_ice_model.m`` returns ("There shall be no duplicated vertices, assuming the last
  vertex connects to the first vertex", p. 222);
* an unset parameter is MATLAB ``[]`` and is represented here by ``None``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = [
    "Intersect", "Polygon", "Floe", "Circle", "Brash", "Param", "Field", "IceImage",
    "load_iceimage_mat", "save_iceimage_mat", "overlap_graph",
]


# ==================================================================================================================
# The dataclasses (Appendix B pp. 221-223, field by field)
# ==================================================================================================================

@dataclass
class Intersect:
    """``.Intersect`` — "Substructure corresponding to intersection between ice floes" (p. 222).

    Both members are 1-based index vectors, e.g. ``Intersect.floe = [6, 13, 34]``.
    """

    floe: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    brash: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


@dataclass
class Polygon:
    """``.Floe(i).Polygon`` — "data corresponding to polygon (convex hull) fit to the true ice floe" (p. 222)."""

    Vertices: np.ndarray                #: ``(n, 2)`` 1-based ``[x, y]``, **open** ring (no duplicated last vertex)
    Center: np.ndarray                  #: ``[x, y]`` centroid of the polygon (``polygeom`` ``geom(2:3)``)
    Area: float                         #: polygon area (continuous; **may be smaller** than ``Floe.Area``)
    Perimeter: float                    #: polygon perimeter in pixels
    Intersect: Intersect = field(default_factory=Intersect)


@dataclass
class Floe:
    """``.Floe(i)`` — one identified ice floe (p. 222)."""

    Center: np.ndarray                  #: ``[index_x, index_y]`` of the *real* floe (ch7 ``regionprops`` centroid);
    #: ``(k, 2)`` when MATLAB's ``cat(1, cen.Centroid)`` appended more than one component (see :func:`_center`)
    Area: int                           #: area **in number of pixels**
    Perimeter: float                    #: perimeter of the real floe, in pixels
    Polygon: Polygon | None = None
    Pixels: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    #: ``(Area, 2)`` 1-based ``[x, y]`` of every pixel of the floe, in MATLAB ``find`` (column-major) order


@dataclass
class Circle:
    """``.Brash(i).Circle`` — "circular fit to the true brash piece" (p. 222)."""

    Radius: float                       #: ``sqrt(Area/π)`` in pixels
    Perimeter: float                    #: ``2πr`` in pixels
    Intersect: Intersect = field(default_factory=Intersect)


@dataclass
class Brash:
    """``.Brash(i)`` — one identified brash-ice piece (p. 222)."""

    Center: np.ndarray                  #: ``[index_x, index_y]`` (the **pixel** centroid, not the circle centre —
    #: they coincide because the circle is centred on the centroid)
    Area: int
    Circle: Circle | None = None
    Pixels: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))


@dataclass
class Param:
    """``.Param`` — "Global image parameters all stored with image at image capture" (p. 221), 17 fields.

    ``SeaIce_Image_Structure.m`` lines 49–72 sets only ``NumPix_x``/``NumPix_y`` (from the image size),
    ``TiltAngle = 90``, ``PanAngle = 0``, ``Location = 'Ny-Alesund'`` and ``Creator = 'UAV'``; everything else is
    ``[]``.  The shipped ``.mat`` contradicts the last two (``Creator = 'Helicopter'``, ``PrjName = 'OATRC 2015'``,
    ``Location = []``) because it was produced by a §8.3 variant of the same script — analysis/ch08.md risk R12.
    """

    NumPix_x: int | None = None         #: number of pixels in x (columns)
    NumPix_y: int | None = None         #: number of pixels in y (rows)
    MAMSL: Any = None                   #: metres above mean sea level of the camera
    FOV: Any = None                     #: field-of-view angle [deg]
    TiltAngle: Any = None               #: vertical shooting angle [deg]
    PanAngle: Any = None                #: horizontal shooting angle [deg]
    CamRot: Any = None
    LeverArm: Any = None
    VesselRot: Any = None
    VesselPos: Any = None
    FileName: Any = None
    Time: Any = None
    Location: Any = None
    Creator: Any = None
    Caption: Any = None
    PrjName: Any = None
    PrjNum: Any = None


@dataclass
class Field:
    """``.Field`` — "global information about the ice field" (pp. 221–222), 15 fields."""

    Georef: Any = None
    Rot: Any = None
    LengthSI_x: Any = None              #: ice-field domain length in x [m]
    LengthSI_y: Any = None
    PixScale_x: Any = None              #: ``LengthSI_x / NumPix_x`` [m/px]
    PixScale_y: Any = None
    PixArea: Any = None                 #: ``PixScale_x * PixScale_y`` [m²]
    NumFloes: int | None = None
    NumBrash: int | None = None
    CovFloes: float | None = None       #: coverage fractions in ``[0, 1]``
    CovBrash: float | None = None
    CovSlush: float | None = None
    CovWater: float | None = None
    CovOther: float | None = None       #: residual pixels (never printed in the book)
    FSD: list[np.ndarray] = field(default_factory=list)
    #: "cell array ``{[int_min, int_max, num], ...}``" (p. 222).  **The printed interval labels do not describe
    #: the counting rule**: ``num`` comes from ``hist(area, min:inter:max)``, whose arguments are bin *centres*,
    #: so the true edges are the midpoints and the two outer bins are unbounded (analysis/ch08.md C5, risk R11).
    #: Re-counting the areas from ``[int_min, int_max]`` reproduces **none** of the 51 shipped triplets.


@dataclass
class IceImage:
    """``IceImage`` — the whole structure (``SeaIce_Image_Structure.m`` line 115)."""

    Param: Param = field(default_factory=Param)
    Field: Field = field(default_factory=Field)
    Floe: list[Floe] = field(default_factory=list)
    Brash: list[Brash] = field(default_factory=list)

    def summary(self) -> str:
        """One-line description, the way ``main_WL_new.m`` would let MATLAB echo it."""
        return (f"IceImage: {self.Param.NumPix_x}x{self.Param.NumPix_y} px, "
                f"{len(self.Floe)} floes, {len(self.Brash)} brash pieces, "
                f"FSD {len(self.Field.FSD)} intervals")


# ==================================================================================================================
# MAT-file I/O
# ==================================================================================================================

def _scalar_or_none(v: Any) -> Any:
    """MATLAB ``[]`` (loaded by scipy as an empty array) → ``None``; a 1×1 value → a Python scalar."""
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return None
        if v.size == 1 and v.ndim <= 1:
            return v.reshape(()).item()
        return v
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


def _index_vector(v: Any) -> np.ndarray:
    """An ``Intersect`` member: MATLAB ``[]`` or an ``(n, 1)`` column of 1-based indices → ``(n,)`` int64."""
    if v is None:
        return np.zeros(0, dtype=np.int64)
    a = np.atleast_1d(np.asarray(v))
    if a.size == 0:
        return np.zeros(0, dtype=np.int64)
    return a.ravel().astype(np.int64)


def _center(v: Any) -> np.ndarray:
    """``.Center``: kept in its **stored shape**.

    It is normally ``(2,)`` = ``[index_x, index_y]``, but ch7's ``cat(1, cen.Centroid)`` appends one row per
    connected component of ``out == i``, so a label that splits stores a ``(k, 2)`` matrix (4 of the shipped
    3452 brash pieces do — ch07 review S4 in the wild).  ``sea_ice_model.m`` then reads ``c(1)``/``c(2)`` as
    **column-major linear indices** into that matrix, so the shape must not be flattened here.
    """
    return np.atleast_1d(np.asarray(v, dtype=np.float64))


def _pixels(v: Any) -> np.ndarray:
    """``.Pixels`` / ``.Vertices``: always an ``(n, 2)`` array, even for a single row."""
    a = np.asarray(v)
    if a.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.atleast_2d(a)


def load_iceimage_mat(path: str | Path, variable: str = "IceImage") -> IceImage:
    """Read a shipped ``IceImage`` MAT-file (Appendix B) into :class:`IceImage`.

    Book: Appendix B, pp. 221–225; the file this is written for is
    ``data/book/ch08/MCD/IceImage_290915_2_jpg.0000179.mat`` (the §8.3 field behind Figs. 8.18–8.21).

    Uses ``scipy.io.loadmat(struct_as_record=False, squeeze_me=True)`` and walks the nested ``mat_struct``
    objects.  Every MATLAB ``[]`` becomes ``None`` (or an empty index vector inside ``Intersect``), every 1×1
    value becomes a Python scalar, and every ``(n, 2)`` coordinate block keeps its 1-based MATLAB values.

    Parameters
    ----------
    path : path-like
        The ``.mat`` file.
    variable : str
        Name of the variable to read (the shipped file holds exactly one, ``IceImage``).

    Returns
    -------
    IceImage
    """
    import scipy.io as sio

    raw = sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    if variable not in raw:
        names = [k for k in raw if not k.startswith("__")]
        raise KeyError(f"{Path(path).name} has no variable {variable!r} (it holds {names})")
    s = raw[variable]

    p = Param(**{f.name: _scalar_or_none(getattr(s.Param, f.name, None)) for f in fields(Param)})

    fld_kwargs: dict[str, Any] = {}
    for f in fields(Field):
        if f.name == "FSD":
            continue
        fld_kwargs[f.name] = _scalar_or_none(getattr(s.Field, f.name, None))
    fsd_raw = getattr(s.Field, "FSD", None)
    fsd: list[np.ndarray] = []
    if fsd_raw is not None and np.size(fsd_raw) > 0:
        for triplet in np.atleast_1d(fsd_raw):
            fsd.append(np.asarray(triplet).ravel().astype(np.int64))
    fld = Field(FSD=fsd, **fld_kwargs)

    floes: list[Floe] = []
    for f in np.atleast_1d(s.Floe):
        pg = getattr(f, "Polygon", None)
        polygon = None
        if pg is not None:
            it = getattr(pg, "Intersect", None)
            polygon = Polygon(
                Vertices=_pixels(pg.Vertices),
                Center=_center(pg.Center),
                Area=_scalar_or_none(pg.Area),
                Perimeter=_scalar_or_none(pg.Perimeter),
                Intersect=Intersect(floe=_index_vector(getattr(it, "floe", None)),
                                    brash=_index_vector(getattr(it, "brash", None))),
            )
        floes.append(Floe(Center=_center(f.Center),
                          Area=int(_scalar_or_none(f.Area)),
                          Perimeter=float(_scalar_or_none(f.Perimeter)),
                          Polygon=polygon,
                          Pixels=_pixels(f.Pixels)))

    brash: list[Brash] = []
    for b in np.atleast_1d(s.Brash):
        cr = getattr(b, "Circle", None)
        circle = None
        if cr is not None:
            it = getattr(cr, "Intersect", None)
            circle = Circle(Radius=float(_scalar_or_none(cr.Radius)),
                            Perimeter=float(_scalar_or_none(cr.Perimeter)),
                            Intersect=Intersect(floe=_index_vector(getattr(it, "floe", None)),
                                                brash=_index_vector(getattr(it, "brash", None))))
        brash.append(Brash(Center=_center(b.Center),
                           Area=int(_scalar_or_none(b.Area)),
                           Circle=circle,
                           Pixels=_pixels(b.Pixels)))

    return IceImage(Param=p, Field=fld, Floe=floes, Brash=brash)


def _to_mat(value: Any) -> Any:
    """``None`` → MATLAB ``[]`` (a ``0×0`` array); everything else is passed through."""
    if value is None:
        return np.zeros((0, 0))
    return value


def _intersect_dict(it: Intersect | None) -> dict[str, Any]:
    it = it or Intersect()
    return {"floe": np.asarray(it.floe, dtype=np.float64).reshape(-1, 1) if np.size(it.floe) else np.zeros((0, 0)),
            "brash": np.asarray(it.brash, dtype=np.float64).reshape(-1, 1) if np.size(it.brash) else np.zeros((0, 0))}


def save_iceimage_mat(path: str | Path, ice: IceImage, variable: str = "IceImage") -> Path:
    """Write an :class:`IceImage` back as a MATLAB v5 struct, in the shipped file's layout.

    ``Floe`` and ``Brash`` become ``(N, 1)`` struct arrays, ``Field.FSD`` a ``1×K`` **cell** array of triplets,
    and every ``None`` an empty ``[]`` — so that ``load('...')`` in MATLAB gives a structure indistinguishable
    (in shape and field order) from the one ``SeaIce_Image_Structure.m`` builds.  Round-trips through
    :func:`load_iceimage_mat`.
    """
    import scipy.io as sio

    def struct_array(items: Sequence[dict[str, Any]]) -> np.ndarray:
        arr = np.empty((len(items), 1), dtype=object)
        for i, d in enumerate(items):
            arr[i, 0] = d
        return arr

    floe_items = []
    for f in ice.Floe:
        pg = f.Polygon
        poly = (np.zeros((0, 0)) if pg is None else
                {"Vertices": np.asarray(pg.Vertices), "Center": np.atleast_2d(np.asarray(pg.Center, dtype=np.float64)),
                 "Area": _to_mat(pg.Area), "Perimeter": _to_mat(pg.Perimeter),
                 "Intersect": _intersect_dict(pg.Intersect)})
        floe_items.append({"Center": np.atleast_2d(np.asarray(f.Center, dtype=np.float64)),
                           "Area": f.Area, "Perimeter": f.Perimeter,
                           "Polygon": poly, "Pixels": np.asarray(f.Pixels)})

    brash_items = []
    for b in ice.Brash:
        cr = b.Circle
        circ = (np.zeros((0, 0)) if cr is None else
                {"Radius": cr.Radius, "Perimeter": cr.Perimeter, "Intersect": _intersect_dict(cr.Intersect)})
        brash_items.append({"Center": np.atleast_2d(np.asarray(b.Center, dtype=np.float64)),
                            "Area": b.Area, "Circle": circ, "Pixels": np.asarray(b.Pixels)})

    fsd = np.empty((1, len(ice.Field.FSD)), dtype=object)
    for i, t in enumerate(ice.Field.FSD):
        fsd[0, i] = np.asarray(t).reshape(1, -1)

    param = {f.name: _to_mat(getattr(ice.Param, f.name)) for f in fields(Param)}
    fld = {f.name: _to_mat(getattr(ice.Field, f.name)) for f in fields(Field) if f.name != "FSD"}
    fld["FSD"] = fsd

    out = {variable: {"Param": param, "Field": fld,
                      "Floe": struct_array(floe_items), "Brash": struct_array(brash_items)}}
    sio.savemat(str(path), out, do_compression=True)
    return Path(path)


# ==================================================================================================================
# The overlap graph (§8.2.1 "overlap flag")
# ==================================================================================================================

def overlap_graph(ice: IceImage) -> dict[str, Any]:
    """The §8.2.1 "overlap flag" lists as a graph, with the symmetry checks Appendix B implies.

    "To support 3-D DEM simulators, an overlap flag is stored on every polygonized floe listing the serial
    numbers of the floes and brash pieces it intersects" (p. 184).

    Returns
    -------
    dict
        ``floe_floe`` / ``floe_brash`` / ``brash_brash`` / ``brash_floe``: lists of 1-based index arrays, one
        per piece; ``n_floe_floe`` … : the total number of stored entries; ``floe_floe_symmetric`` and
        ``brash_brash_symmetric``: whether ``j ∈ L[i] ⇔ i ∈ L[j]``; ``floe_brash_consistent``: whether the
        floe→brash list is the transpose of the brash→floe list.

    On the shipped §8.3 structure these are 1106 floe–floe, 1171 floe–brash, 1171 brash–floe and 544
    brash–brash entries, all three checks ``True``.
    """
    ff = [np.asarray(f.Polygon.Intersect.floe if f.Polygon else [], dtype=np.int64) for f in ice.Floe]
    fb = [np.asarray(f.Polygon.Intersect.brash if f.Polygon else [], dtype=np.int64) for f in ice.Floe]
    bb = [np.asarray(b.Circle.Intersect.brash if b.Circle else [], dtype=np.int64) for b in ice.Brash]
    bf = [np.asarray(b.Circle.Intersect.floe if b.Circle else [], dtype=np.int64) for b in ice.Brash]

    def symmetric(lists: list[np.ndarray]) -> bool:
        pairs = {(i + 1, int(j)) for i, l in enumerate(lists) for j in l}
        return all((j, i) in pairs for (i, j) in pairs)

    fb_pairs = {(i + 1, int(j)) for i, l in enumerate(fb) for j in l}
    bf_pairs = {(int(j), i + 1) for i, l in enumerate(bf) for j in l}

    return {
        "floe_floe": ff, "floe_brash": fb, "brash_brash": bb, "brash_floe": bf,
        "n_floe_floe": int(sum(l.size for l in ff)),
        "n_floe_brash": int(sum(l.size for l in fb)),
        "n_brash_brash": int(sum(l.size for l in bb)),
        "n_brash_floe": int(sum(l.size for l in bf)),
        "floe_floe_symmetric": symmetric(ff),
        "brash_brash_symmetric": symmetric(bb),
        "floe_brash_consistent": fb_pairs == bf_pairs,
    }
