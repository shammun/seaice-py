---
name: do-chapter
description: Run the full seaice-py pipeline for one book chapter (analyze → port → verify → review → notebook → knowledge) using the project subagents, with gates and progress tracking. Usage - /do-chapter 2  (or /do-chapter 2 --from verify  to resume at a phase).
disable-model-invocation: true
---

# /do-chapter N [--from PHASE] [--skip-review]

You are the **orchestrator**. Keep your own context small: read only `progress.json`, `book.yaml`, the phase artifacts'
*headers*, and delegate real work to subagents (`.claude/agents/`). Pass each subagent a complete, self-contained brief.

## 0. Preflight
1. Parse N → id `chNN` (N=2 → `ch02`), MATLAB dir from `book.yaml`.
2. Read `progress.json`. Refuse if the previous chapter (`ch<N-1>`, N>2) is not `verify: pass` and `knowledge: done`
   — unless the user explicitly says to override. Set `current_chapter`.
3. Confirm `chapters/chNN.txt` and `MATLAB_ROOT/<dir>/` (MATLAB_ROOT = `K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab`, see book.yaml) exist; if not, tell the user to run `/setup-project`.
4. Print the plan: the seven phases (analyze, port, verify, review, notebook, knowledge, publish) and which are being
   skipped/resumed.

## 1. ANALYZE → `analysis/chNN.md`   (agent: `chapter-analyst`)
Brief must include: chapter id/title/sections, paths to `chapters/chNN.txt`, `chapters/chNN.pdf`, the MATLAB folder, the
inventory rows for this folder, `knowledge/CUMULATIVE.md`, and the required output format (see agent). When it returns:
skim the "Port plan" table; every `.m` file must appear. If not, send it back once. Set `analyze: pass`, commit.

**Gate:** print the port plan table to the user with the risk list. Continue automatically unless the user asked to be consulted.

## 2. PORT → `seaice/chNN_*.py`, `seaice/core/*`, `scripts/chNN_*.py`   (agent: `matlab-porter`)
Brief: the port plan from analysis, the MATLAB folder, `knowledge/function_map.md`, `knowledge/CUMULATIVE.md`, the data
location `data/book/chNN/`, and the rule "every script must run end-to-end and save its figures to outputs/chNN/".
When it returns: run `.venv/Scripts/python.exe scripts/chNN_<each>.py` yourself (or let the agent show the logs) — all
must exit 0. Set `port: pass`, commit.

## 3. VERIFY → `tests/test_chNN.py`, `reference/chNN/`, `reports/chNN_verification.md`   (agent: `port-verifier`)
Brief: the port plan, list of new Python functions, Octave availability from `progress.json`, the verify-port skill
rules, and "return the Verdict line and the Open items". If verdict = FAIL: send the failing items to `matlab-porter`
with the report path, then re-run `port-verifier` (max 3 loops). If still failing → set `verify: fail`, write the
notes, **stop and report to the user**. Else `verify: pass`, commit.

## 4. REVIEW (agent: `port-reviewer`, fresh eyes; skip with --skip-review)
Brief: chapter text path, analysis, the list of Python files, the verification report. It must check equations vs
code and flag anything the verifier could not test (`unverified`/`approx`). Fix "must-fix" items via `matlab-porter`
and re-run affected tests. Set `review: pass`, commit.

## 5. NOTEBOOK → `notebooks/chNN_<slug>.ipynb` + `notebooks/build_chNN.py`   (agent: `notebook-builder`)
Brief: chapter sections, the seaice functions available (from analysis "Port plan" + knowledge), data locations, the
colab-notebook skill. It must execute the notebook headlessly and report the runtime. Set `notebook: pass`, commit.

## 6. KNOWLEDGE → `knowledge/chNN.md`, `CUMULATIVE.md`, `function_map.md`, `progress.json`   (agent: `knowledge-keeper`)
Brief: paths to analysis, report, review, notebook; the chapter-knowledge skill format. Set `knowledge: done`, commit.

## 7. PUBLISH → `notebooks/chNN_<slug>_colab.ipynb`, `notebooks/chNN_<slug>.html`, `index.html`, `README.md` table
(orchestrator does this itself; established for ch02 on 2026-09-09 — the repo is PUBLIC, see CLAUDE.md rule 12)
1. **Substitute images.** For every book-shipped image the chapter's notebook loads, register one public-domain
   substitute in `seaice/core/public_images.py` (`(chapter, name.lower())` → url/filename/credit/licence) and add a
   row to `data/online/SOURCES.md`. Prefer a NASA Worldview Snapshots MODIS Terra true-colour scene (Beaufort Sea
   marginal ice zone, July) requested at the **book image's own pixel size** so every crop/pixel index in the
   notebook stays valid; fetch it with a real request, view the JPEG (reject cloudy/featureless scenes), record the
   verification date. Nothing under `data/online/` except `SOURCES.md` is committed.
2. **Notebook cells 1–3** must follow `notebooks/build_ch02.py` verbatim (colab-notebook skill): Drive mount +
   `chdir MyDrive/Sea_Ice_Colab`; clone/pull `https://github.com/shammun/seaice-py.git`; `I, SOURCE = load_image(...)`
   + banner + `book(...)` helper. Book-quoted values appear only when `SOURCE.startswith("book")`. Scripts/tests use
   `load_image(..., allow_fallback=False)` and skip when the private copy is absent.
3. **Two headless runs**: (a) `data/book` present → `Data source: book (local)`; (b) `mv data/book data/book_private`
   and delete the cached substitute → `Data source: public-domain substitute (NASA)` (proves the download). Both: 0
   errors, all figures rendered.
4. **Publish from the PUBLIC run** (with `data/book` still renamed): `.venv/Scripts/python.exe tools/publish_notebook.py chNN`
   (writes the `_colab.ipynb` with outputs stripped, the styled `.html`, `index.html`, the README table), then
   `mv data/book_private data/book`. Grep the HTML: `book (local)` count must be 0, no book-image bytes, figure count
   as expected.
5. Set `publish: pass` in `progress.json`; commit `chNN: publish — …`; push (`git push`; if the permission classifier
   blocks it, ask the user to run it with `!`). Confirm `https://shammun.github.io/seaice-py/notebooks/chNN_<slug>.html`
   returns 200 and the Colab badge URL `https://colab.research.google.com/github/shammun/seaice-py/blob/main/notebooks/chNN_<slug>_colab.ipynb`
   resolves (raw GitHub fetch of the `_colab.ipynb`).

## 8. Report to the user (≤ 15 lines)
- Verdict, parity summary (counts of exact/near/approx/reimplemented/unverified), figures reproduced.
- Data the user must download manually (if any) with the Drive path.
- The two `Data source:` lines (book run / public-domain run), the Pages URL and the Colab URL.
- "Next: `/clear` then `/do-chapter N+1`".

## Resuming
`--from verify` skips earlier phases (they must already be `pass`). A fresh session can always resume: everything is on disk.
