"""Shared ``argparse`` front-end for the ``scripts/chNN_*.py`` drivers (``--data``, ``--out``, ``--show/--no-show``)."""
from __future__ import annotations

import argparse
from pathlib import Path

from .io import REPO_ROOT


def chapter_argparser(chapter: str, description: str) -> argparse.ArgumentParser:
    """Parser with the project-wide options.  Relative paths are resolved against the repository root."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", default=f"data/book/{chapter}", help="folder with the book images (default: %(default)s)")
    p.add_argument("--out", default=f"outputs/{chapter}", help="folder for figures (default: %(default)s)")
    p.add_argument("--show", action=argparse.BooleanOptionalAction, default=False,
                   help="also display figures (non-blocking; default --no-show)")
    return p


def resolve_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return ``(data_dir, out_dir)`` as absolute paths; ``out_dir`` is created."""
    data = Path(args.data)
    out = Path(args.out)
    if not data.is_absolute():
        data = REPO_ROOT / data
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    return data, out
