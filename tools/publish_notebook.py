"""Publish a chapter notebook: a Colab-ready variant, a styled standalone HTML page, and the site index.

For ``notebooks/<name>.ipynb`` this produces

* ``notebooks/<name>_colab.ipynb`` — the same notebook with a "Google Colab Setup" block (clone the public repo,
  install ``requirements-colab.txt``, optional Drive mount for manual data) in place of the local ``setup`` cell,
  outputs stripped, Colab metadata set.
* ``notebooks/<name>.html`` — the notebook executed headlessly and rendered with nbconvert's JupyterLab template,
  wrapped in the same "clean educational" card used by the author's other note collections: breadcrumb to the
  index, *Download .ipynb* and *Open in Colab* buttons, stylesheet ``assets/clean-educational.css``.
* ``index.html`` + the index table in ``README.md`` (between the ``INDEX_TABLE`` markers), regenerated from
  ``book.yaml`` and whatever notebooks exist, so every chapter appears once it is published.

Usage (from the repo root)::

    .venv/Scripts/python.exe tools/publish_notebook.py ch02          # one chapter
    .venv/Scripts/python.exe tools/publish_notebook.py --all         # every notebooks/chNN_*.ipynb
    .venv/Scripts/python.exe tools/publish_notebook.py ch02 --no-execute   # reuse notebooks/executed_<name>.ipynb
    .venv/Scripts/python.exe tools/publish_notebook.py --index-only

The GitHub repository / Pages site are fixed below; change ``REPO`` if you publish from a fork.
"""
from __future__ import annotations

import argparse
import copy
import html
import re
import sys
import time
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
ASSETS = ROOT / "assets"

REPO = "shammun/seaice-py"
BRANCH = "main"
REPO_URL = f"https://github.com/{REPO}"
PAGES_URL = f"https://{REPO.split('/')[0]}.github.io/{REPO.split('/')[1]}"
SITE_TITLE = "seaice-py — Sea Ice Image Processing with MATLAB, ported to Python"
INTRO = (
    "Every MATLAB script of <em>Sea Ice Image Processing with MATLAB</em> (Qin Zhang &amp; Roger Skjetne, CRC Press "
    "2018) converted to verified, documented Python — one chapter at a time, each chapter building on the previous "
    "ones. Each chapter ships a Python module, tests that compare against MATLAB R2025a reference outputs, "
    "a verification report, and a teaching notebook you can read here or open in Google Colab."
)


def colab_url(name: str) -> str:
    return f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/notebooks/{name}_colab.ipynb"


# ---------------------------------------------------------------------------------------------------------------------
# 1. Colab variant
# ---------------------------------------------------------------------------------------------------------------------
COLAB_MD = f"""## Google Colab Setup

Run the cells below **once** at the start of each Colab session. They clone the public repository
[`{REPO}`]({REPO_URL}) (the `seaice` package, the chapter data and the notebooks), install the packages from
`requirements-colab.txt`, and put the repository on `sys.path`.

* **No GPU needed** — everything in this notebook runs in well under a minute on the Colab CPU runtime.
* **Google Drive** is only needed for images you have to download manually (none for this chapter). The optional
  second cell mounts Drive and links `MyDrive/seaice-py/data/manual` into the cloned repository.
* Running this notebook **locally** (Jupyter / VS Code) works too: the first cell then just moves to the
  repository root instead of cloning.
"""

COLAB_SETUP = f"""# --- Google Colab setup: clone the repository and install its requirements (run once per session) ---
import os, sys, pathlib

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    if not pathlib.Path("seaice-py").exists():
        !git clone --depth 1 {REPO_URL}.git seaice-py
    os.chdir("seaice-py")
    !pip install -q -r requirements-colab.txt
else:
    # Local Jupyter: find the repository root = the folder that contains seaice/ and data/.
    here = pathlib.Path.cwd().resolve()
    for cand in (here, *here.parents):
        if (cand / "seaice").is_dir() and (cand / "data").is_dir():
            os.chdir(cand)
            break
    else:
        raise FileNotFoundError("Start this notebook from inside the seaice-py repository (e.g. its notebooks/ folder).")

ROOT = pathlib.Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("IN_COLAB:", IN_COLAB, "| cwd:", ROOT.as_posix())
"""

COLAB_DRIVE = """# --- OPTIONAL: mount Google Drive for manually downloaded data (not needed for this chapter) -----------
# Uncomment and run if a chapter asks you to put images in MyDrive/seaice-py/data/manual/chNN/.
# from google.colab import drive
# drive.mount("/content/drive")
# manual = pathlib.Path("/content/drive/MyDrive/seaice-py/data/manual")
# manual.mkdir(parents=True, exist_ok=True)
# link = ROOT / "data" / "manual"
# if not link.exists():
#     link.symlink_to(manual, target_is_directory=True)
# print("manual data folder:", link, "->", manual)
"""


def make_colab_variant(nb_path: Path) -> Path:
    nb = nbformat.read(nb_path, as_version=4)
    out = copy.deepcopy(nb)
    cells = out.cells

    # Find the cell tagged `setup` (the local/Colab/Drive switch of build_chNN.py); fall back to "after the title".
    idx = next((i for i, c in enumerate(cells) if "setup" in c.metadata.get("tags", [])), None)
    new_cells = [
        nbformat.v4.new_markdown_cell(COLAB_MD),
        nbformat.v4.new_code_cell(COLAB_SETUP),
        nbformat.v4.new_code_cell(COLAB_DRIVE),
    ]
    if idx is None:
        cells[1:1] = new_cells
    else:
        # Drop the "## Setup" markdown that precedes the setup cell, then replace the cell itself.
        start = idx
        if idx > 0 and cells[idx - 1].cell_type == "markdown" and cells[idx - 1].source.lstrip().startswith("## Setup"):
            start = idx - 1
        cells[start : idx + 1] = new_cells

    for c in cells:
        if c.cell_type == "code":
            c.outputs = []
            c.execution_count = None
        c.metadata.pop("execution", None)

    out.metadata["colab"] = {"provenance": [], "name": f"{nb_path.stem}_colab.ipynb"}
    out.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    out.metadata["language_info"] = {"name": "python"}
    nbformat.validate(out)
    dest = nb_path.with_name(f"{nb_path.stem}_colab.ipynb")
    nbformat.write(out, dest)
    return dest


# ---------------------------------------------------------------------------------------------------------------------
# 2. Executed HTML page
# ---------------------------------------------------------------------------------------------------------------------
def execute_notebook(nb_path: Path, executed: Path, timeout: int = 900) -> float:
    from nbconvert.preprocessors import ExecutePreprocessor

    nb = nbformat.read(nb_path, as_version=4)
    t0 = time.perf_counter()
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})
    nbformat.write(nb, executed)
    return time.perf_counter() - t0


def notebook_title(nb) -> str:
    for c in nb.cells:
        if c.cell_type == "markdown":
            for line in c.source.splitlines():
                if line.startswith("#"):
                    return line.lstrip("#").strip()
    return "notebook"


def render_html(executed: Path, name: str, title: str) -> str:
    from nbconvert import HTMLExporter

    exporter = HTMLExporter(template_name="lab")
    body, _ = exporter.from_filename(str(executed))

    # Stylesheet right after the charset meta (same as the reference pages).
    link = (
        f'\n<link rel="stylesheet" href="../assets/clean-educational.css?v={int(time.time())}">'
        # long code spans inside markdown tables must wrap instead of overflowing into the next column
        "\n<style>.ce-container table code { white-space: pre-wrap; overflow-wrap: anywhere; }</style>"
    )
    body = re.sub(r'(<meta charset="utf-8"\s*/?>)', lambda m: m.group(1) + link, body, count=1)
    body = re.sub(r"<title>.*?</title>", f"<title>{html.escape(title)}</title>", body, count=1, flags=re.S)

    header = f"""
<div class="ce-container">
<header class="ce-breadcrumb">
  <a href="../index.html">&larr; All chapters</a>
  <span class="ce-source">Source: <code>{name}.ipynb</code></span>
</header>
<div class="ce-actions">
  <span class="ce-actions-label">Get the notebook</span>
  <a class="ce-action-btn" href="{name}.ipynb" download>
    <span class="ce-action-icon">&darr;</span> Download .ipynb
  </a>
  <a class="ce-action-btn ce-action-colab" href="{colab_url(name)}" target="_blank" rel="noopener">
    <span class="ce-action-icon">&#9654;</span> Open in Colab
  </a>
</div>
"""
    body = re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + header, body, count=1)
    body = body.replace("</body>", "</div>\n</body>", 1)
    return body


# ---------------------------------------------------------------------------------------------------------------------
# 3. Site index (index.html + README table)
# ---------------------------------------------------------------------------------------------------------------------
def chapter_rows() -> list[dict]:
    book = yaml.safe_load((ROOT / "book.yaml").read_text(encoding="utf-8"))
    rows = []
    for ch in book["chapters"]:
        if not ch.get("do_port"):
            continue
        cid = ch["id"]
        nb = sorted(p for p in NOTEBOOKS.glob(f"{cid}_*.ipynb") if not p.stem.endswith("_colab") and not p.stem.startswith("executed_"))
        name = nb[0].stem if nb else None
        rows.append({
            "id": cid,
            "number": ch["number"],
            "title": ch["title"],
            "name": name,
            "html": name and (NOTEBOOKS / f"{name}.html").exists(),
            "colab": name and (NOTEBOOKS / f"{name}_colab.ipynb").exists(),
        })
    return rows


def write_index(rows: list[dict]) -> None:
    trs = []
    for r in rows:
        label = f"Ch. {r['number']}" if r["number"] < 10 else "App. A"
        if r["name"]:
            view = f'<a href="notebooks/{r["name"]}.html">View</a>' if r["html"] else "—"
            ipynb = f'<a href="notebooks/{r["name"]}.ipynb">.ipynb</a>'
            colab = f'<a href="{colab_url(r["name"])}" target="_blank" rel="noopener">Open</a>' if r["colab"] else "—"
        else:
            view = ipynb = colab = '<span class="ce-pending">— (pending)</span>'
        trs.append(f"<tr><td>{label}</td><td>{html.escape(r['title'])}</td><td>{view}</td><td>{ipynb}</td><td>{colab}</td></tr>")
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<link rel="stylesheet" href="assets/clean-educational.css?v={int(time.time())}">
<title>seaice-py</title>
<style>.ce-pending {{ color: var(--ce-text-faint); }} table {{ width: 100%; }}</style>
</head>
<body>
<div class="ce-container">
<h1>{SITE_TITLE}</h1>
<p>{INTRO}</p>
<p>Source code, tests and verification reports: <a href="{REPO_URL}">{REPO}</a> on GitHub.
The book PDF and the authors' original MATLAB archive are <strong>not</strong> part of the repository.</p>
<h2>Chapters</h2>
<table>
<thead><tr><th>Chapter</th><th>Title</th><th>HTML</th><th>Notebook</th><th>Colab</th></tr></thead>
<tbody>
{chr(10).join(trs)}
</tbody>
</table>
<p class="ce-source">Reference engine for parity tests: MATLAB R2025a. Parity labels (exact / near / approx /
reimplemented / unverified) are documented per chapter in <code>reports/chNN_verification.md</code>.</p>
</div>
</body>
</html>
"""
    (ROOT / "index.html").write_text(page, encoding="utf-8")
    (ROOT / ".nojekyll").touch()


def write_readme(rows: list[dict]) -> None:
    lines = ["| Chapter | HTML | Notebook | Colab |", "|---|---|---|---|"]
    for r in rows:
        label = f"Ch. {r['number']}: {r['title']}" if r["number"] < 10 else f"App. A: {r['title']}"
        if r["name"]:
            view = f"[View]({PAGES_URL}/notebooks/{r['name']}.html)" if r["html"] else "—"
            ipynb = f"[.ipynb](notebooks/{r['name']}.ipynb)"
            colab = f"[Open]({colab_url(r['name'])})" if r["colab"] else "—"
        else:
            view = ipynb = colab = "— (pending)"
        lines.append(f"| {label} | {view} | {ipynb} | {colab} |")
    table = "\n".join(lines)

    readme = ROOT / "README.md"
    start, end = "<!-- INDEX_TABLE_START -->", "<!-- INDEX_TABLE_END -->"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        if start in text and end in text:
            pre, rest = text.split(start, 1)
            _, post = rest.split(end, 1)
            readme.write_text(f"{pre}{start}\n{table}\n{end}{post}", encoding="utf-8")
            return
    intro_md = re.sub(r"<[^>]+>", "", INTRO).replace("&amp;", "&")
    readme.write_text(
        f"""# seaice-py — *Sea Ice Image Processing with MATLAB*, ported to Python

{intro_md}

Read the chapters online at **{PAGES_URL}/** or open any notebook in Google Colab from the table below.

## Chapters

{start}
{table}
{end}

## Run locally

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows;  .venv/bin/pip on Linux/macOS
.venv/Scripts/python -m pytest -q                     # parity tests against the saved MATLAB references
.venv/Scripts/python scripts/ch02_histogram.py       # any chapter script; figures go to outputs/chNN/
jupyter notebook notebooks/ch02_preliminaries.ipynb  # the teaching notebook (runs locally without Colab)
```

## Layout

| Path | What |
|---|---|
| `seaice/core/` | reusable primitives (histogram, distance transforms, chain codes, interpolation, …) |
| `seaice/chNN_*.py`, `scripts/chNN_*.py` | chapter modules and one script per original MATLAB demo |
| `tests/test_chNN.py`, `reference/chNN/` | pytest parity tests and the MATLAB R2025a reference generator |
| `reports/chNN_verification.md` | parity table, figure comparisons and verdict per chapter |
| `notebooks/chNN_*.ipynb` (+ `_colab`, `.html`) | teaching notebooks; `tools/publish_notebook.py` builds the Colab and HTML versions |
| `knowledge/` | what each chapter learned and what later chapters reuse |

## About the source material

This port follows Qin Zhang and Roger Skjetne, *Sea Ice Image Processing with MATLAB* (CRC Press, 2018). The book
text, its figures and its image data are **not** included here and must be obtained from the publisher; the
authors' MATLAB code archive is not redistributed either. Every Python function cites the book section, equation and
source `.m` file it implements so you can follow along with your own copy. The notebooks run on public-domain NASA
imagery unless you supply the book images privately (put them in `data/book/chNN/` locally, or in
`MyDrive/seaice-py/data/book/chNN/` for Colab). The `.mat` reference files used by the parity tests are regenerated
with `reference/chNN/make_refs.py` on a machine with MATLAB.

## License

The Python code, tests and documentation are released under the MIT License (see `LICENSE`). The license does not
cover the book or the authors' MATLAB code.
""",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------------------------------------------------
def publish(nb_path: Path, execute: bool) -> None:
    name = nb_path.stem
    print(f"== {name}")
    colab = make_colab_variant(nb_path)
    print(f"   colab variant : {colab.relative_to(ROOT).as_posix()}  ({colab.stat().st_size/1e3:.0f} kB)")

    executed = NOTEBOOKS / f"executed_{name}.ipynb"
    if execute or not executed.exists():
        dt = execute_notebook(nb_path, executed)
        print(f"   executed      : {dt:.0f} s")
    nb = nbformat.read(executed, as_version=4)
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    if errors:
        raise RuntimeError(f"{len(errors)} cell(s) raised during execution — fix the notebook before publishing")
    page = render_html(executed, name, notebook_title(nb))
    out = NOTEBOOKS / f"{name}.html"
    out.write_text(page, encoding="utf-8")
    n_img = page.count("<img")
    print(f"   html          : {out.relative_to(ROOT).as_posix()}  ({out.stat().st_size/1e6:.1f} MB, {len(nb.cells)} cells, {n_img} images)")
    print(f"   colab url     : {colab_url(name)}")
    executed.unlink()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("targets", nargs="*", help="chapter ids (ch02) or notebook paths")
    ap.add_argument("--all", action="store_true", help="publish every notebooks/chNN_*.ipynb")
    ap.add_argument("--no-execute", action="store_true", help="reuse notebooks/executed_<name>.ipynb if present")
    ap.add_argument("--index-only", action="store_true", help="only regenerate index.html and the README table")
    a = ap.parse_args(argv)

    if not a.index_only:
        targets: list[Path] = []
        if a.all:
            targets = sorted(p for p in NOTEBOOKS.glob("ch??_*.ipynb") if not p.stem.endswith("_colab") and not p.stem.startswith("executed_"))
        for t in a.targets:
            p = Path(t)
            if p.suffix == ".ipynb":
                targets.append(p if p.is_absolute() else ROOT / p)
            else:
                found = sorted(q for q in NOTEBOOKS.glob(f"{t}_*.ipynb") if not q.stem.endswith("_colab"))
                if not found:
                    ap.error(f"no notebook found for {t!r} in notebooks/")
                targets.append(found[0])
        if not targets:
            ap.error("give a chapter id, a notebook path, --all or --index-only")
        for nb_path in targets:
            publish(nb_path, execute=not a.no_execute)

    rows = chapter_rows()
    write_index(rows)
    write_readme(rows)
    print(f"== index.html + README.md updated ({sum(1 for r in rows if r['name'])}/{len(rows)} chapters published)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
