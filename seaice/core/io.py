"""Data / output path helpers shared by every chapter.

Book: no equation; replaces MATLAB ``imread('rgb.jpg')`` (relative path, case-insensitive on Windows)
and ``saveas`` / ``figure`` bookkeeping in the chapter scripts.  MATLAB sources: every ``MATLAB_ROOT/chN/*.m``
that calls ``imread``.

The repository root is located from this file (``seaice/core/io.py`` → two levels up), so scripts work from
any current directory.  All paths use :class:`pathlib.Path`.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

#: Repository root (the folder that contains ``seaice/``, ``data/``, ``outputs/``).
REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


def book_data_dir(chapter: str) -> Path:
    """``data/book/<chapter>/`` for a chapter id such as ``"ch02"``."""
    return REPO_ROOT / "data" / "book" / chapter


def output_dir(chapter: str, base: str | Path | None = None) -> Path:
    """``outputs/<chapter>/`` (created if missing).  ``base`` overrides the default ``outputs/`` root."""
    out = (Path(base) if base is not None else REPO_ROOT / "outputs" / chapter)
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_case_insensitive(folder: str | Path, name: str) -> Path:
    """Return ``folder/name`` matching ``name`` case-insensitively (MATLAB on Windows opens ``rgb.jpg`` for ``rgb.JPG``).

    Raises ``FileNotFoundError`` with a helpful message if nothing matches.
    """
    folder = Path(folder)
    direct = folder / name
    if direct.exists():
        return direct
    if folder.is_dir():
        lname = name.lower()
        for cand in folder.iterdir():
            if cand.name.lower() == lname:
                return cand
    raise FileNotFoundError(
        f"{name!r} not found in {folder} (case-insensitive). Copy the book image there (see data-sources skill)."
    )


def load_book_image(chapter: str, name: str, data_dir: str | Path | None = None) -> np.ndarray:
    """Read a book-shipped image as a numpy array (uint8, RGB ``(M, N, 3)`` for JPEG colour images).

    Parameters
    ----------
    chapter : str
        Chapter id, e.g. ``"ch02"`` → ``data/book/ch02/``.
    name : str
        File name, matched case-insensitively (``"rgb.jpg"`` finds ``rgb.JPG``).
    data_dir : path-like, optional
        Explicit folder overriding ``data/book/<chapter>``.

    Notes
    -----
    Equivalent of MATLAB ``imread``.  JPEG decoding uses imageio/Pillow (libjpeg); MATLAB's decoder may differ by
    ±1 gray level on a few pixels.  The Fig. 2.3 pixel ``I(1076, 675) = (28, 76, 114)`` matches exactly.
    """
    folder = Path(data_dir) if data_dir is not None else book_data_dir(chapter)
    path = resolve_case_insensitive(folder, name)
    img = iio.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:  # drop alpha if a PNG has one
        img = img[:, :, :3]
    return np.asarray(img)


def fetch(url: str, dest: str | Path, timeout: int = 60) -> Path:
    """Download ``url`` to ``dest`` once (cached); return the path.  Tier-2 data helper (see data-sources skill)."""
    import requests  # local import: optional dependency at runtime

    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network
        raise RuntimeError(f"Could not download {url} → {dest}: {exc}") from exc
    dest.write_bytes(resp.content)
    return dest
