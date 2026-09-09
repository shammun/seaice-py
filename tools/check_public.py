"""Fail if anything book-derived is about to be (or has been) committed to the PUBLIC repository (CLAUDE.md rule 12).

Checks every tracked file (or the files given on the command line):

* ``*.ipynb`` must have **no outputs** (Colab's "Save a copy in GitHub" writes outputs, which embed figures rendered
  from the reader's private book image);
* no tracked path under ``chapters/*.txt``, ``data/book/``, ``data/manual/``, ``reports/**/figures/``, ``outputs/``,
  ``notebooks/executed_*``;
* no text file contains the loader labels ``book (local)`` / ``book (private Drive)`` inside notebook outputs or HTML
  output areas (the published HTML must come from the public-domain run).

Usage::

    .venv/Scripts/python.exe tools/check_public.py            # all tracked files
    .venv/Scripts/python.exe tools/check_public.py a.ipynb b.html
    git config core.hooksPath .githooks                         # installs the pre-push hook that runs this

Exit code 0 = clean, 1 = violations (listed).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"^(chapters/ch\d+\.txt|data/book/|data/manual/|reports/.*/figures/|outputs/|notebooks/executed_)")
LABELS = re.compile(r"book \((local|private Drive)\)")


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def check(paths: list[str]) -> list[str]:
    problems: list[str] = []
    for rel in paths:
        if FORBIDDEN.match(rel):
            problems.append(f"{rel}: book-derived path must not be tracked")
            continue
        p = ROOT / rel
        if not p.is_file():
            continue
        if p.suffix == ".ipynb":
            try:
                nb = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{rel}: not valid JSON ({exc})")
                continue
            n_out = sum(len(c.get("outputs", [])) for c in nb.get("cells", []) if c.get("cell_type") == "code")
            if n_out:
                problems.append(f"{rel}: {n_out} cell output(s) — strip outputs (Colab 'Save to GitHub' writes them; "
                                "figures from a private book image would be published)")
            text = json.dumps(nb)
            if LABELS.search(text) and n_out:
                problems.append(f"{rel}: outputs mention a book data source")
        elif p.suffix == ".html":
            html = p.read_text(encoding="utf-8", errors="replace")
            areas = re.findall(r'<div class="jp-OutputArea-output[^"]*"[^>]*>(.*?)</div>', html, flags=re.S)
            if any(LABELS.search(a) for a in areas):
                problems.append(f"{rel}: HTML output area says the book image was used — rebuild from the public-domain run")
    return problems


def main(argv: list[str]) -> int:
    paths = argv or tracked_files()
    problems = check(paths)
    if problems:
        print("PUBLIC-REPO CHECK FAILED (CLAUDE.md rule 12):")
        for msg in problems:
            print("  -", msg)
        return 1
    print(f"public-repo check OK ({len(paths)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
