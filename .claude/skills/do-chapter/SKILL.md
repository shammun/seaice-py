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
4. Print the plan: the six phases and which are being skipped/resumed.

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

## 7. Report to the user (≤ 15 lines)
- Verdict, parity summary (counts of exact/near/approx/reimplemented/unverified), figures reproduced.
- Data the user must download manually (if any) with the Drive path.
- Notebook path + how to open it in Colab.
- "Next: `/clear` then `/do-chapter N+1`".

## Resuming
`--from verify` skips earlier phases (they must already be `pass`). A fresh session can always resume: everything is on disk.
