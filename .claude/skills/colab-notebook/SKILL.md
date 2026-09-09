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
2. **Setup cell** (code, tagged `setup`) — detects Colab and prepares the package:
   ```python
   import os, sys, subprocess, pathlib
   IN_COLAB = "google.colab" in sys.modules
   SOURCE = "github"          # "github" or "drive"  ← user picks
   REPO_URL = "https://github.com/<user>/seaice-py.git"
   DRIVE_DIR = "/content/drive/MyDrive/seaice-py"
   if IN_COLAB:
       if SOURCE == "drive":
           from google.colab import drive; drive.mount("/content/drive")
           os.chdir(DRIVE_DIR)
       else:
           if not pathlib.Path("seaice-py").exists():
               subprocess.run(["git", "clone", "--depth", "1", REPO_URL], check=True)
           os.chdir("seaice-py")
       subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
   else:
       os.chdir(pathlib.Path(__file__).resolve().parents[1] if "__file__" in globals() else pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd())
   sys.path.insert(0, os.getcwd()); print("cwd:", os.getcwd())
   ```
   `requirements-colab.txt` = requirements.txt minus oct2py/jupyter/pdf tools (create it in /setup-project).
3. **Data cell** — obtains the chapter's images: first `data/book/chNN/` (already in the repo/Drive), else the download
   function from `seaice/core/io.py` (`fetch(url, dest)` with caching), else a printed instruction block telling the user
   what to download and where to put it in Drive (`MyDrive/seaice-py/data/manual/chNN/`). Never silently fail: if data is
   missing, raise with the instruction text.
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

## Delivering to Google Drive (see GUIDE.md §7)
Either commit and let Colab `git clone`, or mirror `notebooks/ seaice/ data/ knowledge/ requirements-colab.txt`
into `G:\My Drive\seaice-py\` with `tools/sync_to_drive.ps1` (robocopy). The notebook's `SOURCE` variable selects which.
