---
name: setup-project
description: One-time bootstrap for seaice-py — verify source files, Python venv, MATLAB (Octave fallback), split the book PDF into chapters, inventory the MATLAB code, copy book data, seed knowledge files, initial git commit. Run once via /setup-project before any chapter work.
disable-model-invocation: true
---

# /setup-project

Work through every step; print a checklist at the end with ✅/❌ per step. Do not start chapter work here.

## 1. Verify layout (stop and tell the user if anything is missing)
The repo root is `C:\Users\sislam27\Work\Climate Dynamics PHD\Sea Ice Image Processing with MATLAB` (the folder Claude Code
was started in — confirm with `pwd`/`cd`). Nothing is copied into a `source/` folder; the assets are used in place:
- **BOOK_PDF**: exactly one `*.pdf` directly in the root (`Sea ice image processing with MATLAB® by Skjetne, Roger Zhang, Qin (z-lib.org).pdf`, ~26 MB).
  Check it is a real PDF with ~260 pages: `.venv/Scripts/python.exe -c "import fitz,glob; f=max(glob.glob('*.pdf'),key=lambda p:__import__('os').path.getsize(p)); print(f, fitz.open(f).page_count)"`.
  If there are several PDFs, set `book.pdf` in `book.yaml` to the right file name.
- **MATLAB_ROOT**: `K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab/` must contain `ch2` … `ch10`.
  If the folder is elsewhere (e.g. `matlab/` directly in the root), fix `book.matlab_root` in `book.yaml`
  (`tools/inventory_matlab.py` also auto-detects a `matlab` folder that contains `ch2`, and prints a NOTE when it does).
- The zip `K30735_..._matlab codes.zip` is ignored; never unzip or read it.
- Read every file in `MATLAB_ROOT/ch10` and decide what it corresponds to (expected: Appendix A calibration; could be
  Appendix B or misc). Update `book.yaml` (`ch10.matlab_dir`, title) and CLAUDE.md mapping if different. Also check
  whether any folder contains code for *two* book chapters or none.

## 2. Environment
- `python --version` (need ≥3.11). Create venv if absent: `python -m venv .venv`; then `.venv/Scripts/python.exe -m pip install -U pip -r requirements.txt`.
- Create `requirements-colab.txt` = requirements.txt without oct2py, jupyter, ipykernel, nbconvert, nbclient, pypdf, pymupdf, pytest-xdist.
- MATLAB (primary reference engine — detect this FIRST): `.venv/Scripts/python.exe tools/run_matlab_ref.py --check`
  looks for `C:\Program Files\MATLAB\R2025a\bin\matlab.exe` (then other releases, `MATLAB_EXE`, `PATH`) and verifies it by
  running `matlab -batch "disp(version)"` (allow a few minutes on first launch; a first-launch failure can be retried once).
  Record the version string (e.g. `25.1.0.2833191 (R2025a)`) in `progress.json → environment.matlab` (null if absent) and
  the exe path in `environment.matlab_exe`.
- Octave (fallback only): `octave-cli --version` (or `octave --version`). **If MATLAB is present, a missing Octave is fine —
  record `octave: null` and move on; do not ask the user to install it.** Only if MATLAB is absent, tell the user:
  `winget install GNU.Octave`, then in Octave run `pkg install -forge image` once; add
  `C:\Program Files\GNU Octave\Octave-<ver>\mingw64\bin` to PATH; verify with
  `.venv/Scripts/python.exe -c "from oct2py import octave; octave.eval('pkg load image'); print(octave.eval('version'))"`.
- Write results into `progress.json → environment` with today's date, e.g.
  `{"matlab": "25.1.0.2833191 (R2025a)", "matlab_exe": "C:\\Program Files\\MATLAB\\R2025a\\bin\\matlab.exe", "octave": null, "python": "3.11.5", "checked_on": "YYYY-MM-DD"}`.
- `git --version`; `git init` if no repo; ensure `.gitignore` exists (it excludes the PDF, the zip and the MATLAB folder from git — they stay on disk, read-only).

## 3. Split the PDF into one file per chapter (this is where the single book PDF becomes chapters/)
`.venv/Scripts/python.exe tools/split_pdf.py --dry-run` → the tool auto-locates BOOK_PDF in the root, prints the chapter → PDF-page
mapping and the first lines of each chapter start → inspect → run again without `--dry-run`. This writes
`chapters/ch01.pdf … ch11.pdf`, `chapters/ch01.txt … ch11.txt` (ch10 = Appendix A, ch11 = Appendix B) and `chapters/_page_map.json`.
Open `chapters/ch02.txt` and confirm it starts at "Digital Image Processing Preliminaries" (printed page 11) and
`chapters/ch03.txt` at "Ice Pixel Detection". If off by k pages, rerun with `--offset`. Record the offset in `book.yaml`
(`book.pdf_offset`).

## 4. Inventory MATLAB and copy data
- `.venv/Scripts/python.exe tools/inventory_matlab.py` → `analysis/_matlab_inventory.md`.
- Copy every non-`.m` file from `MATLAB_ROOT/chN/` to `data/book/chNN/` (preserve subfolders; two-digit chapter ids). Print sizes.
- Check the last section of the book front matter and Appendix B (`chapters/ch11.txt`) for any URL to online supplementary
  material; if found, try fetching it and note in `data/online/SOURCES.md`.

## 5. Seed files
- `seaice/__init__.py`, `seaice/core/__init__.py` (empty), `tests/__init__.py`, `tests/conftest.py` adding repo root to `sys.path`.
- `knowledge/CUMULATIVE.md` and `knowledge/function_map.md` from `templates/` if they don't exist.
- `tools/sync_to_drive.ps1` (robocopy mirror of notebooks, seaice, data, knowledge, requirements-colab.txt to a
  `$DriveDir` parameter defaulting to `G:\My Drive\seaice-py`) — create it.

## 6. Smoke test
`.venv/Scripts/python.exe -c "import numpy, scipy, skimage, cv2, sklearn, matplotlib; print('ok')"` and
`.venv/Scripts/python.exe -m pytest -q` (0 tests is fine).

## 7. Commit
`git add -A && git commit -m "setup: environment, split PDF, MATLAB inventory, seed knowledge"`.
Then print the checklist and say: "Next: run `/do-chapter 2`".
