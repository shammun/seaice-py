---
name: colab-notebook
description: How to build, structure, and headlessly execute the per-chapter Google Colab notebooks for seaice-py (Drive/GitHub setup cells, data cells, section-by-section teaching layout, nbformat generation, nbconvert execution). Load when creating or fixing any .ipynb.
---

# colab-notebook — one notebook per chapter

## Build with code, not by hand
Generate the notebook with `nbformat` from a Python builder script `notebooks/build_chNN.py` (commit the builder).
That keeps it reproducible and lets you re-run after code changes. Skeleton:

```python
import nbformat as nbf
nb = nbf.v4.new_notebook(); nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
nb.metadata["colab"] = {"name": "ch02_preliminaries.ipynb", "provenance": []}
C = nb.cells
C.append(nbf.v4.new_markdown_cell(...)); C.append(nbf.v4.new_code_cell(...))
nbf.write(nb, "notebooks/ch02_preliminaries.ipynb")
```

## Mandatory cell order
1. **Title + book mapping** (markdown): chapter, sections covered, which MATLAB files this replaces, link to `knowledge/chNN.md`.
2. **Cell 1 — Drive mount + working directory** (first code cell, copy verbatim from `notebooks/build_ch02.py`).
   It mounts Google Drive and `chdir`s to `/content/drive/MyDrive/Sea_Ice_Colab` (the reader's private data lives
   under it: `data/book/chNN/`); readers without Drive get `/content/Sea_Ice_Colab`; outside Colab it is a no-op
   that `chdir`s to the repository root. The three lines
   ```python
   import os
   os.chdir('/content/drive/MyDrive/Sea_Ice_Colab')
   print(os.getcwd())
   ```
   are kept unchanged, guarded so the `chdir` only runs when the mount succeeded.
   **Cell 2 — repository**: on Colab `git -C seaice-py pull` if `./seaice-py` exists, else
   `git clone --depth 1 https://github.com/shammun/seaice-py.git`; then `pip install -q -r seaice-py/requirements-colab.txt`
   and `sys.path.insert(0, "seaice-py")`. Outside Colab: no-op (`sys.path.insert(0, os.getcwd())`).
   `requirements-colab.txt` = requirements.txt minus oct2py/jupyter/pdf tools.
3. **Cell 3 — data cell**: `I, SOURCE = seaice.core.io.load_image("chNN", "<book file name>")` — private copy first
   (`<cwd>/data/book`, repo root, Drive), then the registered public-domain substitute (`seaice/core/public_images.py`,
   downloaded into `data/online/chNN/`). `FROM_BOOK = SOURCE.startswith("book")`; when it is false the cell prints
   the one-line banner *"Running on a public-domain substitute image; figures show the same operations, but values
   quoted in the book only hold for the book's own image."* Every later cell that quotes a book value computes it
   from the loaded image and appends `(book: …)` **only if `FROM_BOOK`** (use the `book(...)` helper defined here).
   No image path is hard-coded anywhere else; all later cells call `seaice` functions. Never silently fail: a
   missing image with no registered substitute raises with instructions.
4. **One section per book section** (2.1, 2.2, …): markdown with the concept (2–6 sentences, key equation in LaTeX,
   cite the book figure numbers) → code that calls `seaice.chNN_*` functions (never re-implement algorithms inside
   the notebook) → figure reproduced with the same layout as the book figure → 1–2 sentence interpretation.
   Where a MATLAB script exists for the section, name it in the markdown ("replaces `ch2/histogram.m`").
5. **Parameter play** (optional, 1 cell): `ipywidgets` sliders for 1–2 key parameters (threshold, SE radius).
6. **Summary + what the next chapter needs from this one** (markdown) — copied from `knowledge/chNN.md` "Feeds forward".

## Rules
- Notebooks import from `seaice/`; the only algorithm code allowed inline is 1–3 line glue.
- Every figure cell ends with `plt.show()`; set `plt.rcParams["figure.dpi"] = 100`.
- No `oct2py`/Octave in notebooks by default (Colab can `!apt-get install -y octave octave-image` but it is slow); an
  optional final cell may show how to run a MATLAB reference if the user wants.
- Keep total runtime under ~5 minutes on Colab CPU; downsample big images with a clearly labelled `SCALE` constant.
- Windows paths never appear in notebooks; use `pathlib` and forward slashes.

## Headless execution = verification
```
python -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=900 \
    --output executed_chNN.ipynb notebooks/chNN_<slug>.ipynb
```
The build is only done when this succeeds locally (Windows, data in `data/`). Delete `executed_*.ipynb` (git-ignored).
Then also open the executed copy and check no cell output contains `Error` or an empty figure.

## Publishing (PUBLISH phase of /do-chapter — the repo is public)
Colab gets the code by cloning `https://github.com/shammun/seaice-py.git` (cell 2); the reader's private book images
live in their own Drive (`MyDrive/Sea_Ice_Colab/data/book/chNN/`), never in the repo. After the notebook passes:
1. run it twice headlessly — with `data/book` present and with `data/book` renamed to `data/book_private` (public-domain
   substitute; delete the cached download first to prove the fetch) — both 0 errors, all figures;
2. with `data/book` still renamed, `.venv/Scripts/python.exe tools/publish_notebook.py chNN` → `chNN_<slug>_colab.ipynb`
   (outputs stripped, Colab metadata), `chNN_<slug>.html` (styled page with Download / Open-in-Colab buttons), `index.html`,
   README table; rename `data/book` back; grep the HTML for `book (local)` (must be 0);
3. commit `chNN: publish — …`, push, check `https://shammun.github.io/seaice-py/notebooks/chNN_<slug>.html`.
Use `python -m nbconvert` (not `python -m jupyter nbconvert`, which can dispatch to Anaconda's binary on this host).

## Public-repo rule
The repo is public. Never commit book text, book-shipped images, PDF page crops, or executed notebooks that contain them. Published notebooks load data through seaice.core.io.load_image(), which prefers the reader's private Drive copy and falls back to public-domain imagery. Book-figure comparisons stay local in reports/**/figures/ (git-ignored).
