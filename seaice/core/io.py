"""Data / output path helpers shared by every chapter, including the private-Drive / public-fallback image loader.

Book: no equation; replaces MATLAB ``imread('rgb.jpg')`` (relative path, case-insensitive on Windows) and the
``saveas`` / ``figure`` bookkeeping in the chapter scripts.  MATLAB sources: every ``MATLAB_ROOT/chN/*.m`` that
calls ``imread``.

Public-repo data scheme (CLAUDE.md rule 12)
-------------------------------------------
The book's images are copyrighted and are **not** in the repository.  :func:`load_image` therefore looks for the
reader's own private copy first and otherwise falls back to a registered public-domain substitute
(:mod:`seaice.core.public_images`), downloaded once into ``data/online/<chapter>/``:

1. ``<cwd>/data/book/<chapter>/<name>`` — on Colab the cwd is ``/content/drive/MyDrive/Sea_Ice_Colab``
   → label ``"book (private Drive)"`` when the cwd is under ``/content/drive``, else ``"book (local)"``;
2. ``<repo root>/data/book/<chapter>/<name>`` → ``"book (local)"``;
3. ``/content/drive/MyDrive/Sea_Ice_Colab/data/book/<chapter>/<name>`` explicitly → ``"book (private Drive)"``;
4. the registered substitute, fetched into the first writable root → ``"public-domain substitute (NASA)"``.

Every call prints one line ``Data source: <label> — <path>`` and returns the label so notebooks can branch on it
(book-quoted numbers are only meaningful when the label starts with ``"book"``).
"""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v3 as iio
import numpy as np

#: Repository root (the folder that contains ``seaice/``, ``data/``, ``outputs/``).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Folder the Colab notebooks ``chdir`` into after mounting Google Drive (private data lives under it).
DRIVE_DIR = Path("/content/drive/MyDrive/Sea_Ice_Colab")

#: Label returned by :func:`load_image` when a public-domain substitute is used.
PUBLIC_LABEL = "public-domain substitute (NASA)"


def repo_root() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


def book_data_dir(chapter: str) -> Path:
    """``<repo root>/data/book/<chapter>/`` for a chapter id such as ``"ch02"`` (the private, git-ignored copy)."""
    return REPO_ROOT / "data" / "book" / chapter


def output_dir(chapter: str, base: str | Path | None = None) -> Path:
    """``outputs/<chapter>/`` (created if missing).  ``base`` overrides the default ``outputs/`` root."""
    out = (Path(base) if base is not None else REPO_ROOT / "outputs" / chapter)
    out.mkdir(parents=True, exist_ok=True)
    return out


def data_roots() -> list[Path]:
    """Roots searched for ``data/...`` in priority order: current directory, repository root, the Colab Drive folder.

    Duplicates are removed (locally the cwd usually *is* the repo root); the Drive folder is included only when it
    exists (i.e. on Colab with Drive mounted), which implements lookup step 3 of :func:`load_image` even when the
    notebook's cwd is somewhere else.
    """
    roots: list[Path] = []
    for cand in (Path.cwd(), REPO_ROOT, DRIVE_DIR):
        try:
            cand = cand.resolve()
        except OSError:
            continue
        if cand.is_dir() and cand not in roots:
            roots.append(cand)
    return roots


def _source_label(root: Path) -> str:
    """``"book (private Drive)"`` for a root under ``/content/drive``, otherwise ``"book (local)"``."""
    return "book (private Drive)" if root.as_posix().startswith("/content/drive") else "book (local)"


def resolve_case_insensitive(folder: str | Path, name: str) -> Path:
    """Return ``folder/name`` matching ``name`` case-insensitively (MATLAB on Windows opens ``rgb.jpg`` for ``rgb.JPG``).

    The **top level** of ``folder`` is searched first (unchanged behaviour: a top-level file always wins).  Only if
    that misses are sub-directories walked, again case-insensitively, and the first match in sorted order is
    returned (sorted so the result is deterministic across filesystems).

    Raises ``FileNotFoundError`` if nothing matches.
    """
    # NOTE (porter): the recursive fallback was added in ch06.  ch06 is the first chapter whose book images live in
    # sub-folders of the MATLAB archive (``ch6/Sea_Ice_Floe_Identification/sea_ice_test.jpg`` and
    # ``ch6/for test/{test8,alg_seg_gray}.jpg`` -- note the space in the folder name).  A reader on Colab copies the
    # book's MATLAB folder into Drive verbatim, sub-folders and all, so the published notebook has to find
    # ``sea_ice_test.jpg`` under ``data/book/ch06/**`` without anybody flattening the tree by hand.
    folder = Path(folder)
    direct = folder / name
    if direct.is_file():
        return direct
    if folder.is_dir():
        lname = name.lower()
        for cand in sorted(folder.iterdir()):
            if cand.is_file() and cand.name.lower() == lname:
                return cand
        for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
            try:
                return resolve_case_insensitive(sub, name)
            except FileNotFoundError:
                continue
    raise FileNotFoundError(f"{name!r} not found in {folder} (case-insensitive, sub-folders included).")


def read_image(path: str | Path) -> np.ndarray:
    """Decode an image file to a uint8 array (RGB ``(M, N, 3)`` for colour images; alpha dropped).

    Equivalent of MATLAB ``imread``.  JPEG decoding uses imageio/Pillow (libjpeg); on the book's ``rgb.JPG`` the
    result is identical to MATLAB R2025a (0 of 9 437 184 samples differ, see ``reports/ch02_verification.md``).
    """
    img = np.asarray(iio.imread(Path(path)))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def fetch(url: str, dest: str | Path, timeout: int = 60) -> Path:
    """Download ``url`` to ``dest`` once (cached) and return the path.

    A non-empty existing ``dest`` is returned without any network access.  Failures raise ``RuntimeError`` with the
    URL, the destination and the underlying cause so a notebook reader can act on it.
    """
    import requests  # local import: keeps import time low for the scripts

    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if not resp.content or ctype.startswith("text/html"):
            raise ValueError(f"unexpected response (content-type {ctype!r}, {len(resp.content)} bytes)")
        tmp.write_bytes(resp.content)
        os.replace(tmp, dest)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {url}\n  -> {dest}\n  cause: {type(exc).__name__}: {exc}\n"
            "  Check the network connection, or place your own copy of the image at the destination path."
        ) from exc
    return dest


def load_image(
    chapter: str,
    name: str,
    *,
    allow_fallback: bool = True,
    data_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, str]:
    """Load a book image from the reader's private copy, or a registered public-domain substitute.

    Parameters
    ----------
    chapter : str
        Chapter id, e.g. ``"ch02"``.
    name : str
        The book's file name, matched case-insensitively (``"rgb.jpg"`` finds ``rgb.JPG``).
    allow_fallback : bool
        ``True`` (notebooks): use the public-domain substitute from :mod:`seaice.core.public_images` when no private
        copy is found.  ``False`` (scripts, tests): raise ``FileNotFoundError`` instead, so callers can skip.
    data_dir : path-like, optional
        Explicit folder holding the private copy (the scripts' ``--data`` option); searched *instead of* the roots.
    verbose : bool
        Print ``Data source: <label> — <path>``.

    Returns
    -------
    (image, source_label)
        ``image`` is a uint8 array; ``source_label`` is ``"book (local)"``, ``"book (private Drive)"`` or
        ``"public-domain substitute (NASA)"``.  Book-quoted values only hold when the label starts with ``"book"``.
    """
    searched: list[Path] = []
    if data_dir is not None:  # explicit folder (scripts' --data): searched exclusively
        candidates = [(Path(data_dir), "book (local)")]
    else:
        candidates = [(root / "data" / "book" / chapter, _source_label(root)) for root in data_roots()]

    for folder, label in candidates:
        try:
            path = resolve_case_insensitive(folder, name)
        except FileNotFoundError:
            searched.append(folder)
            continue
        if verbose:
            print(f"Data source: {label} — {path.as_posix()}")
        return read_image(path), label

    where = "\n".join(f"  - {p.as_posix()}" for p in searched)
    if not allow_fallback:
        raise FileNotFoundError(
            f"Book image {chapter}/{name} not found (private copy required; it is not in the public repository).\n"
            f"Searched:\n{where}\n  Copy it from the book's MATLAB archive to data/book/{chapter}/ "
            f"(or MyDrive/Sea_Ice_Colab/data/book/{chapter}/ on Colab)."
        )

    from .public_images import lookup  # local import: avoids a cycle at package import time

    entry = lookup(chapter, name)
    if entry is None:
        raise FileNotFoundError(
            f"Book image {chapter}/{name} not found and no public-domain substitute is registered for it.\n"
            f"Searched:\n{where}"
        )
    last_error: Exception | None = None
    for root in data_roots():
        dest = root / "data" / "online" / chapter / entry["filename"]
        try:
            path = fetch(entry["url"], dest, timeout=120)
            break
        except (OSError, RuntimeError) as exc:  # read-only root, or download failure: try the next root
            last_error = exc
    else:
        raise RuntimeError(f"Could not obtain the public-domain substitute for {chapter}/{name}: {last_error}")
    if verbose:
        print(f"Data source: {PUBLIC_LABEL} — {path.as_posix()}")
        print(f"  ({entry['credit']}; {entry['licence']})")
    return read_image(path), PUBLIC_LABEL
