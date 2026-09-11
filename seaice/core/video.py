"""Video IO in MATLAB's array layout — the Python stand-in for ``VideoReader``/``read``/``movie2avi``.

Book: Ch. 9 §9.2.2 (ice concentration from model sea ice video, Figs. 9.6–9.10) and §9.3.3 (maximum floe size,
Figs. 9.17–9.18).  MATLAB sources: ``MATLAB_ROOT/ch9/movie_otsu.m`` lines 7–13/59–65,
``MATLAB_ROOT/ch9/movie_kmeans.m`` lines 7–13/73–79 and
``MATLAB_ROOT/ch9/Model_Ice_Floe_Identification/movie_floe.m`` lines 5–11.

Why the odd axis order
----------------------
``vidFrames = read(readerobj)`` returns an ``(H, W, 3, N)`` uint8 array and every script then writes
``mov(k).cdata = vidFrames(:,:,:,k)``.  :func:`read_video` keeps that exact layout so the index arithmetic of the
M-files ports one-to-one (``frames[:, :, :, k]``), instead of the ``(N, H, W, 3)`` that imageio hands back.

R2025a notes (probed, `analysis/ch09.md` §0.6)
----------------------------------------------
* ``mmreader`` (``movie_floe.m`` line 5) and ``movie2avi`` (``movie_otsu.m`` line 65, ``movie_kmeans.m`` line 79)
  are **removed** from R2025a; the reference runs patch them to ``VideoReader`` and ``VideoWriter`` + ``writeVideo``
  (IO only, recorded in ``reference/ch09/patches.json``).
* ``get(v, 'numberOfFrames')`` still works as a case-insensitive alias of ``NumFrames``.

Decode parity (risk R2)
-----------------------
A **lossy** AVI decoded by MATLAB's codec and by ffmpeg can differ by several gray levels, and one gray level moves
an Otsu threshold, which moves every ice concentration.  Everything used as a parity reference in this project is
therefore written as **Uncompressed AVI** (MATLAB ``VideoWriter(..., 'Uncompressed AVI')``; here
``codec='rawvideo'`` in an AVI container), where both sides see byte-identical frames.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["VideoInfo", "read_video", "write_video", "video_num_frames", "video_info",
           "frames_to_matlab", "frames_to_imageio", "infer_frame_layout"]


@dataclass(frozen=True)
class VideoInfo:
    """The ``VideoReader`` properties the ch9 scripts read (``Height``, ``Width``, ``NumFrames``, ``FrameRate``)."""

    path: Path
    height: int
    width: int
    num_frames: int
    frame_rate: float


def frames_to_matlab(frames: np.ndarray) -> np.ndarray:
    """``(N, H, W, 3)`` (imageio order) → ``(H, W, 3, N)`` (MATLAB ``read(VideoReader)`` order)."""
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"expected a 4-D (N, H, W, 3) stack, got shape {frames.shape}")
    return np.transpose(frames, (1, 2, 3, 0))


def frames_to_imageio(frames: np.ndarray) -> np.ndarray:
    """``(H, W, 3, N)`` (MATLAB order) → ``(N, H, W, 3)`` (imageio order)."""
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"expected a 4-D (H, W, 3, N) stack, got shape {frames.shape}")
    return np.transpose(frames, (3, 0, 1, 2))


def infer_frame_layout(shape) -> str | None:
    """The project's **single** rule for telling a 4-D frame stack's axis order from its shape.

    Returns ``'nhwc'`` for imageio's ``(N, H, W, 3)``, ``'hwcn'`` for MATLAB's ``read(VideoReader)`` order
    ``(H, W, 3, N)``, or ``None`` when neither applies (no axis of length 3 in either place).

    The last axis is tested **first**, so the genuinely ambiguous ``(3, H, W, 3)`` / ``(H, W, 3, 3)`` case — a
    3-frame, 3-channel stack — resolves to ``'nhwc'``.  Pass the layout explicitly (or a list of frames) when
    that matters.  Shared by :func:`_normalise_layout` here and by
    :func:`seaice.ch09_model_ice._as_frame_list`, which previously spelled the same rule two different ways
    (review nit N-b); the two are behaviourally identical on every shape either accepted before.
    """
    shape = tuple(shape)
    if len(shape) != 4:
        return None
    if shape[3] == 3:
        return "nhwc"
    if shape[2] == 3:
        return "hwcn"
    return None


def _normalise_layout(frames: np.ndarray, layout: str) -> np.ndarray:
    """Return the stack in imageio's ``(N, H, W, 3)`` order.

    ``layout='auto'`` decides with :func:`infer_frame_layout`: ``(..., 3)`` in the last axis is ``NHWC``,
    otherwise ``(·, ·, 3, ·)`` is MATLAB's ``HWCN``.  When both are 3 (a 3-frame 3-channel stack) ``NHWC`` wins —
    pass ``layout`` explicitly to remove the ambiguity.  A stack that matches neither is written as ``HWCN`` and
    then fails the channel check below, as before.
    """
    frames = np.asarray(frames)
    if frames.ndim == 3:                       # a single frame, or a stack of gray frames
        if frames.shape[-1] == 3:
            frames = frames[None, ...]         # one RGB frame
        else:
            frames = np.repeat(frames[..., None], 3, axis=-1)   # (N, H, W) gray -> RGB
    if frames.ndim != 4:
        raise ValueError(f"expected 3-D or 4-D frames, got shape {frames.shape}")
    lay = layout.lower()
    if lay == "auto":
        lay = infer_frame_layout(frames.shape) or "hwcn"
    if lay == "hwcn":
        frames = frames_to_imageio(frames)
    elif lay != "nhwc":
        raise ValueError("layout must be 'auto', 'NHWC' or 'HWCN'")
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    if frames.shape[-1] != 3:
        raise ValueError(f"frames must have 3 colour channels, got {frames.shape[-1]}")
    return frames


def read_video(path: str | Path) -> np.ndarray:
    """``vidFrames = read(VideoReader(path))`` — every frame as one ``(H, W, 3, N)`` uint8 array.

    MATLAB source: ``movie_otsu.m`` lines 7–10 / ``movie_kmeans.m`` lines 7–10 / ``movie_floe.m`` lines 5–8.
    Backed by ``imageio.v3`` (the ``imageio-ffmpeg`` plugin) with ``cv2.VideoCapture`` as a fallback.

    Parity: **exact** on an Uncompressed AVI (identical bytes on both sides); ``near``/``approx`` on any lossy
    container — see the module docstring (risk R2).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")
    stack = _read_imageio(path)
    if stack is None:
        stack = _read_cv2(path)
    if stack is None or stack.size == 0:
        raise OSError(f"could not decode any frame from {path}")
    return frames_to_matlab(stack)


def _read_imageio(path: Path) -> np.ndarray | None:
    try:
        import imageio.v3 as iio

        frames = [np.asarray(f) for f in iio.imiter(path, plugin="FFMPEG")]
    except Exception:                                            # pragma: no cover - plugin/codec dependent
        return None
    if not frames:
        return None
    out = np.stack(frames, axis=0)
    if out.ndim == 3:                                            # gray decode
        out = np.repeat(out[..., None], 3, axis=-1)
    return out[..., :3].astype(np.uint8, copy=False)


def _read_cv2(path: Path) -> np.ndarray | None:                  # pragma: no cover - fallback path
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr[..., ::-1].copy())                     # BGR -> RGB
    cap.release()
    return np.stack(frames, axis=0).astype(np.uint8) if frames else None


def video_info(path: str | Path) -> VideoInfo:
    """The ``VideoReader`` properties used by the ch9 scripts, read without decoding every frame when possible."""
    path = Path(path)
    try:                                                          # pragma: no cover - metadata plugin dependent
        import imageio.v3 as iio

        props = iio.improps(path, plugin="FFMPEG")
        meta = iio.immeta(path, plugin="FFMPEG")
        n, h, w = int(props.shape[0]), int(props.shape[1]), int(props.shape[2])
        fps = float(meta.get("fps", 0.0))
        if n > 0:
            return VideoInfo(path=path, height=h, width=w, num_frames=n, frame_rate=fps)
    except Exception:
        pass
    stack = read_video(path)
    h, w, _, n = stack.shape
    return VideoInfo(path=path, height=h, width=w, num_frames=n, frame_rate=0.0)


def video_num_frames(path: str | Path) -> int:
    """``get(readerobj, 'numberOfFrames')`` (``movie_otsu.m`` line 13; still a valid alias in R2025a)."""
    return video_info(path).num_frames


def write_video(path: str | Path, frames: np.ndarray, fps: float = 12.0, *, layout: str = "auto",
                codec: str = "rawvideo") -> Path:
    """``movie2avi(M, path, 'FPS', fps)`` — write a frame stack as an **Uncompressed AVI** by default.

    MATLAB source: ``movie_otsu.m`` line 65 (``'otsu.avi'``, 12 fps) and ``movie_kmeans.m`` line 79
    (``'05400_kmeans.avi'``, 12 fps).  ``movie2avi`` was removed in R2025a; the reference run patches it to
    ``VideoWriter(name, 'Uncompressed AVI')`` + ``writeVideo`` (IO only — see ``reference/ch09/patches.json``).

    Parameters
    ----------
    frames : ndarray
        ``(N, H, W, 3)``, ``(H, W, 3, N)`` (MATLAB order) or a gray stack ``(N, H, W)``; uint8, or logical/float
        which is scaled the way ``imshow`` renders it (False/0 → 0, True/1 → 255).
    fps : float
        Frame rate (``movie2avi``'s ``'FPS'``).
    layout : {'auto', 'NHWC', 'HWCN'}
    codec : str
        ``'rawvideo'`` = Uncompressed AVI (the only codec used for parity fixtures, risk R2).

    Returns the path written.
    """
    import imageio.v3 as iio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = _normalise_layout(frames, layout)
    if arr.dtype == np.bool_:
        arr = (arr.astype(np.uint8) * 255)
    elif arr.dtype != np.uint8:
        a = np.asarray(arr, dtype=np.float64)
        if np.nanmax(a, initial=0.0) <= 1.0:
            a = a * 255.0
        arr = np.clip(np.floor(a + 0.5), 0, 255).astype(np.uint8)
    kwargs = dict(plugin="FFMPEG", fps=float(fps), codec=codec, macro_block_size=1)
    if codec == "rawvideo":
        # 4:4:4 8-bit RGB so nothing is subsampled: MATLAB's 'Uncompressed AVI' is 24-bit RGB too.
        kwargs["pixelformat"] = "bgr24"
    iio.imwrite(path, arr, **kwargs)
    return path
