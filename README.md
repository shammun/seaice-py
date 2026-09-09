# seaice-py — *Sea Ice Image Processing with MATLAB*, ported to Python

Every MATLAB script of Sea Ice Image Processing with MATLAB (Qin Zhang & Roger Skjetne, CRC Press 2018) converted to verified, documented Python — one chapter at a time, each chapter building on the previous ones. Each chapter ships a Python module, tests that compare against MATLAB R2025a reference outputs, a verification report, and a teaching notebook you can read here or open in Google Colab.

Read the chapters online at **https://shammun.github.io/seaice-py/** or open any notebook in Google Colab from the table below.

## Chapters

<!-- INDEX_TABLE_START -->
| Chapter | HTML | Notebook | Colab |
|---|---|---|---|
| Ch. 2: Digital Image Processing Preliminaries | [View](https://shammun.github.io/seaice-py/notebooks/ch02_preliminaries.html) | [.ipynb](notebooks/ch02_preliminaries.ipynb) | [Open](https://colab.research.google.com/github/shammun/seaice-py/blob/main/notebooks/ch02_preliminaries_colab.ipynb) |
| Ch. 3: Ice Pixel Detection | — (pending) | — (pending) | — (pending) |
| Ch. 4: Ice Edge Detection | — (pending) | — (pending) | — (pending) |
| Ch. 5: Watershed-Based Ice Floe Segmentation | — (pending) | — (pending) | — (pending) |
| Ch. 6: GVF Snake-Based Ice Floe Boundary Identification and Ice Image Segmentation | — (pending) | — (pending) | — (pending) |
| Ch. 7: Sea Ice Type Identification | — (pending) | — (pending) | — (pending) |
| Ch. 8: Sea Ice Image Processing Applications | — (pending) | — (pending) | — (pending) |
| Ch. 9: Model Sea Ice Image Processing Applications | — (pending) | — (pending) | — (pending) |
| App. A: Appendix A — Geometric Calibration (orthorectification, lens distortion) | — (pending) | — (pending) | — (pending) |
<!-- INDEX_TABLE_END -->

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

## Not in the repository

The book PDF and the authors' original MATLAB code archive are copyrighted and stay out of git; the port cites the
book section, equation and source `.m` file in every docstring instead. The `.mat` reference files are regenerated
with `reference/chNN/make_refs.py` on a machine with MATLAB.
