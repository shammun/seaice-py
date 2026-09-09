# seaice-py — Porting "Sea Ice Image Processing with MATLAB" (Zhang & Skjetne, CRC Press 2018) to Python

## Mission
Convert every MATLAB script in the book's code folders into verified, documented Python, **one chapter at a time, in
order**, and produce one Google Colab notebook per chapter. Each chapter must build on the knowledge and code of the
previous ones. The book text is the authority on *what* an algorithm does; the MATLAB code is the authority on *how*
the authors did it.

## Project root and the two source assets (this is where Claude Code runs)
Repo root (absolute): `C:\Users\sislam27\Work\Climate Dynamics PHD\Sea Ice Image Processing with MATLAB`
All paths below are relative to this root. The root path contains spaces — always quote paths in shell commands and use
`pathlib` in Python.

| Name used in skills/agents | Actual location (relative to root) | Notes |
|---|---|---|
| **BOOK_PDF** | `Sea ice image processing with MATLAB® by Skjetne, Roger Zhang, Qin (z-lib.org).pdf` (the only `*.pdf` in the root, ~26 MB) | Read-only. `tools/split_pdf.py` finds it automatically (`book.yaml → book.pdf: auto`) |
| **MATLAB_ROOT** | `K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab/` containing `ch2` … `ch10` | Read-only reference; never edit. `book.yaml → book.matlab_root` |
| (ignored) | `K30735_Sea Ice Image Processing with MATLAB_matlab codes.zip` | The original zip; ignored by git and never read |

## Where everything else lives (created by the pipeline)
| Path | What |
|---|---|
| `chapters/chNN.pdf`, `chapters/chNN.txt` | One split PDF + extracted text per chapter — **made by `/setup-project` from BOOK_PDF**; `chapters/_page_map.json` records the page mapping |
| `book.yaml` | Chapter map: titles, printed page ranges, MATLAB folder ↔ book chapter mapping |
| `analysis/chNN.md` | Chapter analysis (concepts, equations, every .m file, port plan); `analysis/_matlab_inventory.md` = auto inventory |
| `seaice/` | The Python package. `seaice/core/` = reusable primitives; `seaice/chNN_*.py` = chapter modules |
| `scripts/chNN_*.py` | One Python script per original MATLAB demo script (same name, `.py`) |
| `tests/test_chNN.py` | pytest tests for the chapter |
| `reference/chNN/` | Octave/MATLAB-generated reference outputs (`.mat`) used for parity tests (+ `make_refs.py`) |
| `reports/chNN_verification.md` | Parity table + evidence that the chapter works |
| `notebooks/chNN_<slug>.ipynb` (+ `build_chNN.py`) | Colab notebook for the chapter |
| `knowledge/chNN.md`, `knowledge/CUMULATIVE.md`, `knowledge/function_map.md` | Learned knowledge; CUMULATIVE is read at the start of every chapter |
| `data/book/chNN/` | Images shipped inside MATLAB_ROOT/chN (copied, never modified); `data/online/` downloaded; `data/synthetic/` generated; `data/manual/` user-provided |
| `outputs/chNN/` | Figures produced by scripts |
| `progress.json` | Machine-readable status per chapter and phase |

## Chapter ↔ MATLAB folder mapping (confirmed in /setup-project)
Book: Ch1 Intro (no code) · Ch2 Preliminaries · Ch3 Ice pixel detection · Ch4 Ice edge detection · Ch5 Watershed floe
segmentation · Ch6 GVF snake · Ch7 Ice type identification · Ch8 Applications · Ch9 Model-ice applications ·
App. A Geometric calibration · App. B Data structure. MATLAB folders `ch2`…`ch9` ↔ book chapters 2…9; `ch10` ↔ Appendix A
(verify by reading its files). Work order: 2,3,4,5,6,7,8,9,10(App A).

## Non-negotiable rules
1. **Sequential.** Never start chapter N+1 until `progress.json` shows chapter N with `verify: pass` and `knowledge: done`.
2. **Read before writing.** Before porting a chapter: read `knowledge/CUMULATIVE.md`, `analysis/chNN.md`, the chapter text
   in `chapters/chNN.txt`, and every `.m` file in `MATLAB_ROOT/chN/`. Never port from memory of "what the book probably does".
3. **Every port is traceable.** Each Python function docstring cites the book section / equation number and the source `.m`
   file. Each `.m` file must have exactly one corresponding Python script or function (recorded in `analysis/chNN.md`).
4. **No silent approximations.** If a MATLAB toolbox function has no exact Python equivalent, either (a) re-implement it
   from the book's equations in `seaice/core/` or (b) use the closest library call *and* record the deviation in the
   verification report with a parity label (`exact` / `near` / `approx` / `reimplemented`).
5. **Verification is evidence, not opinion.** "Works correctly" means: tests pass, parity against MATLAB reference
   outputs where obtainable, book figures reproduced side-by-side, and the report says so with numbers.
   **MATLAB R2025a is installed** (`C:\Program Files\MATLAB\R2025a\bin\matlab.exe`) and is the primary reference engine:
   `tools/run_matlab_ref.py` runs the original `.m` code via `matlab -batch` and saves `.mat` references. Because the
   references come from real MATLAB, the parity labels `exact` and `near` are measured **against MATLAB itself**, not
   against Octave's approximation of it — and MATLAB-only functions (`imbinarize`, `adaptthresh`, `activecontour`) are now
   directly testable at L2 instead of falling back to `approx`. oct2py/Octave is only a fallback when MATLAB is absent.
6. **Subagents do the heavy lifting; the main session orchestrates.** Use the agents in `.claude/agents/` so the main
   context stays small. Everything important is written to disk so a fresh session can resume from `progress.json`.
7. **Windows host.** Use forward slashes in Python, `pathlib` everywhere, quote paths (the root has spaces). Long commands
   go in `tools/*.py`, not in one-liners. The venv is `.venv`; run Python as `.venv/Scripts/python.exe`.
8. **Never modify BOOK_PDF, MATLAB_ROOT, or the zip.** Read them; write everything else into the folders above.
9. **Reuse before re-implementing.** Check `seaice/core/` and `knowledge/function_map.md` first; a primitive written for
   chapter 2 must be reused in chapter 5, not duplicated.
10. **Data policy** (see skill `data-sources`): book-shipped images first, then free online data, then synthetic, then
    "manual download → Google Drive" instructions. Never fabricate results for data you do not have.
11. **Commit after every phase** with message `chNN: <phase> — <one line>`.
12. **The repo is public.** Never commit book text, book-shipped images, PDF page crops, or executed notebooks that
    contain them. Published notebooks load data through `seaice.core.io.load_image()`, which prefers the reader's
    private Drive copy and falls back to public-domain imagery. Book-figure comparisons stay local in
    `reports/**/figures/` (git-ignored).

## Python stack (pinned in requirements.txt)
numpy · scipy · scikit-image · opencv-python-headless · matplotlib · scikit-learn · imageio · pillow · pandas · pytest ·
oct2py (Octave bridge) · nbformat/nbconvert/jupyter (notebooks) · pypdf/pymupdf (PDF) · requests

## Skills (invoke with /name) and what they are for
- `/setup-project` — one-time: check tools, **split BOOK_PDF into chapters/**, inventory MATLAB_ROOT, copy book data, seed knowledge.
- `/do-chapter N` — the full pipeline for one chapter (analyze → port → verify → review → notebook → knowledge).
- `/verify-chapter N`, `/notebook-chapter N` — rerun a single phase.
- `/status` — print progress and what to do next.
- Background skills loaded by agents: `seaice-book`, `matlab-to-python`, `verify-port`, `colab-notebook`, `chapter-knowledge`, `data-sources`.

## Style
Python ≥3.11, type hints, numpy docstrings, functions over scripts, `if __name__ == "__main__":` guards, figures saved to
`outputs/chNN/` with the book's figure number in the filename (e.g. `fig_2_07_histogram.png`). No notebooks as source of
truth — notebooks import from `seaice/`.
