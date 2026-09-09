# Step-by-Step Guide: Converting "Sea Ice Image Processing with MATLAB" to Python + Colab with Claude Code

**Project folder (Claude Code runs here):** `C:\Users\sislam27\Work\Climate Dynamics PHD\Sea Ice Image Processing with MATLAB`

This guide has four parts. Part 1 is the one-time setup (about 30 minutes). Part 2 is the prompt playbook: every prompt you will ever type, with *when* to use it. Part 3 walks through the chapter loop you repeat nine times. Part 4 covers Google Drive and Colab, troubleshooting, and a reference of what gets produced.

## 0. How the system works

You already have the two source assets in the project folder: the single book PDF, and the folder `K30735_Sea Ice Image Processing with MATLAB_matlab codes\matlab\` with `ch2` … `ch10`. Nothing is moved or copied. The kit you unzip into that same folder adds the "brain":

| Kit component | What it does |
|---|---|
| `CLAUDE.md` | Project rules Claude Code reads every session: sequential chapters, every `.m` file gets one Python counterpart, no silent approximations, never touch the PDF or MATLAB folder |
| `book.yaml` | Chapter map (titles, printed page ranges, which MATLAB folder belongs to which chapter) |
| `tools/split_pdf.py` | **Splits the single book PDF into `chapters/ch01…ch11.pdf` + `.txt`** (auto-finds the PDF in the root, auto-detects the page offset) |
| `tools/inventory_matlab.py` | Lists every `.m` file, the toolbox functions it calls, the images it reads |
| `tools/compare_arrays.py` | Compares Python output with MATLAB/Octave output (tolerances, label renumbering) |
| `.claude/skills/` (11 skills) | Slash commands `/setup-project`, `/do-chapter`, `/verify-chapter`, `/notebook-chapter`, `/status`, plus background knowledge the agents load (`matlab-to-python` pitfalls + function map, `verify-port`, `colab-notebook`, `data-sources`, `chapter-knowledge`, `seaice-book`) |
| `.claude/agents/` (6 subagents) | `chapter-analyst`, `matlab-porter`, `port-verifier`, `port-reviewer`, `notebook-builder`, `knowledge-keeper` — each works in its own context window |
| `knowledge/`, `templates/`, `progress.json` | The growing knowledge base and the progress tracker |

The pipeline per chapter:

```
/do-chapter N
  1 chapter-analyst  : reads chapters/chNN.txt + every .m in matlab/chN  -> analysis/chNN.md (port plan)
  2 matlab-porter    : writes seaice/chNN_*.py, seaice/core/*, scripts/chNN_*.py; runs every script
  3 port-verifier    : tests + Octave reference outputs + book-figure comparisons -> reports/chNN_verification.md (PASS/FAIL)
  4 port-reviewer    : fresh-context check of code vs book equations
  5 notebook-builder : notebooks/chNN_<slug>.ipynb built with nbformat, executed headlessly
  6 knowledge-keeper : knowledge/chNN.md + CUMULATIVE.md + function_map.md + progress.json + git commit
/clear  ->  /do-chapter N+1   (reads knowledge/CUMULATIVE.md first, so it builds on everything learned)
```

Why Octave: you probably do not have MATLAB. GNU Octave runs most of the book's `.m` files unchanged, which gives ground-truth reference outputs to compare the Python against numerically. That is what makes "verified" mean something rather than "it ran without error".

Chapter ↔ folder mapping: book chapters 2–9 ↔ `ch2`–`ch9`. The book has no chapter 10; the `ch10` folder is almost certainly Appendix A (geometric calibration). `/setup-project` reads its files and confirms. Chapter 1 has no code.

---

## PART 1 — ONE-TIME SETUP (do once, in this order)

### Step 1.1 — Install Python 3.11 or 3.12
1. Download from python.org (Windows installer). Tick **"Add python.exe to PATH"**.
2. Open a new PowerShell window and check:
```powershell
python --version
```
Expected: `Python 3.11.x` or `3.12.x`. (3.13 works too but some wheels lag; prefer 3.11/3.12.)

### Step 1.2 — Install Git
```powershell
winget install Git.Git
```
Close and reopen PowerShell, then check `git --version`. Also set your identity once:
```powershell
git config --global user.name "Shammunul Islam"
git config --global user.email "your@email"
```

### Step 1.3 — Install GNU Octave (gives MATLAB reference outputs for verification)
```powershell
winget install GNU.Octave
```
Then:
1. Start **Octave (GUI)** from the Start menu once. In its command window run:
```
pkg install -forge image
pkg load image
imhist(uint8(magic(8)))
```
   (the last line should pop up a histogram — that proves the image package works).
2. Add Octave to PATH so Claude Code can call it: find the folder containing `octave-cli.exe`, typically
   `C:\Program Files\GNU Octave\Octave-<version>\mingw64\bin`. Add it to your **User** PATH (Settings → System → About → Advanced system settings → Environment Variables).
3. Open a new PowerShell and check:
```powershell
octave-cli --version
```
If you cannot install Octave, skip this step: everything still works, but parity tests are skipped and the verifier relies on synthetic-truth tests and figure reproduction, labelling those items honestly as `unverified`/`approx`.

### Step 1.4 — (Optional, recommended) Google Drive for Desktop
Install from google.com/drive/download and sign in. Afterwards a drive letter like `G:\My Drive\` exists. This is how notebooks get to Colab in Part 4.

### Step 1.5 — Confirm the project folder contents
Open PowerShell in the project folder:
```powershell
cd "C:\Users\sislam27\Work\Climate Dynamics PHD\Sea Ice Image Processing with MATLAB"
dir
dir ".\K30735_Sea Ice Image Processing with MATLAB_matlab codes\matlab"
```
You must see: one `.pdf` (the book, ~26 MB), the folder `K30735_Sea Ice Image Processing with MATLAB_matlab codes` (and optionally the `.zip` of the same name, which is ignored), and inside `...\matlab\` the folders `ch2` … `ch10`.

If the `matlab` folder is somewhere else (for example directly in the project root), that is fine — after Step 1.6 open `book.yaml` and set `matlab_root:` to the correct relative path (the tools also auto-detect a folder named `matlab` that contains `ch2`).

### Step 1.6 — Unzip the kit into the project folder
Download `seaice-py-kit.zip`, then:
```powershell
cd "C:\Users\sislam27\Work\Climate Dynamics PHD\Sea Ice Image Processing with MATLAB"
Expand-Archive -Path "$env:USERPROFILE\Downloads\seaice-py-kit.zip" -DestinationPath . -Force
Move-Item -Path ".\seaice-py-kit\*" -Destination . -Force
Move-Item -Path ".\seaice-py-kit\.claude" -Destination . -Force
Remove-Item ".\seaice-py-kit" -Recurse -Force
dir -Force
```
`dir -Force` must now show `CLAUDE.md`, `book.yaml`, `progress.json`, `requirements.txt`, `GUIDE.md`, `.gitignore`, the folders `.claude`, `tools`, `knowledge`, `templates`, plus your original PDF and `K30735...` folder. Check the hidden folder specifically:
```powershell
dir .claude\skills
dir .claude\agents
```
(11 skill folders, 6 agent files.) If `.claude` is missing, the second `Move-Item` did not run — repeat it.

### Step 1.7 — Start Claude Code inside the project folder
```powershell
cd "C:\Users\sislam27\Work\Climate Dynamics PHD\Sea Ice Image Processing with MATLAB"
claude
```
Claude Code must be started **in this folder** — that is how it finds `CLAUDE.md`, `.claude\skills` and `.claude\agents`.

### Step 1.8 — Confirm the skills are visible
Inside Claude Code type:
```
/skills
```
(or `/help`). You should see `setup-project`, `do-chapter`, `verify-chapter`, `notebook-chapter`, `status` and the background skills. If not, see Troubleshooting T1.

### Step 1.9 — First status check
```
/status
```
Expected: a table with every chapter `todo`, environment unknown, and the message "Next: run /setup-project". Setup is complete. Continue to Part 2/3.

---

## PART 2 — THE PROMPT PLAYBOOK (which prompt, when)

### 2.1 Decision table

| Situation | Prompt to type | Details in |
|---|---|---|
| Fresh install, nothing run yet | `/setup-project` | §3.1 |
| The PDF split landed on the wrong pages | "The split is off…" prompt | §3.1 follow-ups |
| Octave was not found | Install it (Step 1.3) then "Re-check the environment…" | §3.1 follow-ups |
| Ready to start / continue the book | `/clear` then `/do-chapter N` | §3.2 |
| A chapter stopped with verify FAIL | "Read reports/chNN… Open items…" root-cause prompt | §3.4 |
| Session was interrupted mid-chapter | `/do-chapter N --from <phase>` | §3.4 |
| You edited code and want the notebook rebuilt | `/notebook-chapter N` | §3.4 |
| You doubt an `approx`/`unverified` label | "Generate an Octave reference…" prompt | §3.4 |
| A chapter needs data that is not shipped | "List exactly which data files…" prompt | §3.4 |
| You want to *learn* the chapter | Study-sheet / explain prompts | §3.5 |
| Ready to open a notebook in Colab | "Sync the Colab artifacts to Google Drive…" | §4.1 |
| You don't know where you are | `/status` | any time |

### 2.2 Rules of thumb
- One chapter per session. Always type `/clear` before `/do-chapter`. Everything needed is on disk, and a clean context is faster, cheaper and more accurate.
- Never paste the book text or MATLAB code into the chat. The agents read `chapters/chNN.txt` and the `.m` files from disk.
- When Claude Code asks a permission question, answer `y` — the kit's `.claude/settings.json` already pre-approves python, pytest, git, octave and jupyter, and *denies* edits to the PDF and the MATLAB folder.
- If Claude Code proposes to "skip verification for now" or "loosen the tolerance", say no: `Do not loosen tolerances. Report the discrepancy in Open items with your best hypothesis and stop.`

---

## PART 3 — THE WORK, PHASE BY PHASE

### 3.1 Phase A — `/setup-project` (once)

**Type:**
```
/setup-project
```

**What Claude Code does (about 10–15 minutes):**
1. Confirms it is in the project root, that exactly one book PDF sits there, and that `K30735_...\matlab\ch2…ch10` exist.
2. Creates `.venv` and installs `requirements.txt`; creates `requirements-colab.txt`.
3. Detects Octave (and MATLAB, if ever present) and records the result in `progress.json`.
4. **Splits the single book PDF** with `tools/split_pdf.py` into `chapters/ch01.pdf … ch11.pdf` and `chapters/ch01.txt … ch11.txt` (ch10 = Appendix A, ch11 = Appendix B), auto-detecting the front-matter page offset and writing `chapters/_page_map.json`.
5. Runs `tools/inventory_matlab.py` → `analysis/_matlab_inventory.md` (every `.m` file with its toolbox calls and image files).
6. Reads the files in `ch10` and decides which appendix/chapter they belong to; fixes `book.yaml` if needed.
7. Copies the images shipped with the MATLAB code (e.g. `ch2/rgb.jpg`) into `data/book/chNN/`.
8. Seeds `seaice/`, `tests/`, `knowledge/CUMULATIVE.md`, `knowledge/function_map.md`, `tools/sync_to_drive.ps1`, runs a smoke test, `git init` + first commit.
9. Prints a ✅/❌ checklist and says "Next: run /do-chapter 2".

**What you check (2 minutes):**
- Open `chapters\ch02.txt` — it must start with "Digital Image Processing Preliminaries" (printed page 11). Open `chapters\ch05.txt` — "Watershed-Based Ice Floe Segmentation" (printed page 83).
- The checklist line for Octave says ✅ (if you installed it).
- `analysis\_matlab_inventory.md` lists ch2 … ch10 with file counts.

**Follow-up prompts for this phase**

If the chapter text starts on the wrong page:
```
The split is off: chapters/ch02.txt starts on printed page 10 instead of 11. Rerun tools/split_pdf.py with the corrected --offset, then re-check that ch02, ch05 and ch09 each begin with their chapter title, and record the offset in book.yaml.
```
If Octave was installed after setup:
```
Re-check the environment: run octave-cli --version and the oct2py check from the setup-project skill, update progress.json → environment, and confirm that pkg load image works.
```
If the ch10 folder turned out to be something unexpected:
```
Explain what the files in K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab/ch10 implement, which book section each corresponds to, and update book.yaml and CLAUDE.md so the mapping is correct.
```
If the MATLAB folder is in a different place:
```
The MATLAB code folders ch2…ch10 are actually at <relative path>. Update book.yaml → book.matlab_root, confirm tools/inventory_matlab.py finds them, and rerun the inventory.
```

### 3.2 Phase B — the chapter loop (repeat for N = 2 … 10)

**Type:**
```
/clear
/do-chapter 2
```

**What Claude Code does (30–90 minutes per chapter depending on size):** the six-agent pipeline from §0. It prints a plan first, then a short line as each phase finishes. It pauses for you only when: (a) verification fails three times in a row, (b) data is missing and needs a manual download, or (c) chapter N−1 is not marked complete.

**Final report you will get:** verdict, parity counts (exact/near/approx/reimplemented/unverified), figures reproduced, any manual-data instructions with the Drive path, the notebook path, and "Next: /clear then /do-chapter 3".

**Then do the 5-minute human check (§3.3), sync to Drive (§4.1) if you want the notebook in Colab now, and continue:**
```
/clear
/do-chapter 3
```
…and so on through `/do-chapter 10`.

**Chapter-specific notes (so you know what to expect):**

| Chapter | Expect |
|---|---|
| 2 Preliminaries | Longest setup chapter: builds `seaice/core/` primitives that everything else reuses — MATLAB-compatible `rgb2gray`, `fspecial` kernels, distance transforms (incl. quasi-Euclidean), Moore boundary tracing + chain codes, MATLAB-style `imresize`. Be present for this one. |
| 3 Ice pixel detection | Otsu / local thresholding, k-means. k-means is random-init in both languages: parity is by cluster centres, not labels. |
| 4 Edge detection | MATLAB's `edge()` and `strel('disk')` are not what scikit-image does; expect `reimplemented`/`near` labels with explanations. |
| 5 Watershed + merging | Heavy. Watershed plateaus can differ from MATLAB; neighboring-region merging uses the chain-code concavity test from ch2. Expect 1–2 verify loops. |
| 6 GVF snake | The authors' snake code is plain MATLAB → runs in Octave → strong references. Slow to run; scripts use a `SCALE` constant. |
| 7 Ice type | Shape enhancement + texture; several book figures to reproduce. |
| 8 Applications | Some inputs (shipborne camera sequences) may not be shipped → Tier 2 online imagery or Tier 4 manual download; the report tells you exactly what and where. |
| 9 Model ice | Tank images usually not shipped → synthetic rectangular-floe images are generated and clearly labelled. |
| 10 (Appendix A) | Orthorectification (DLT) and lens-distortion polynomials; verified on synthetic calibration grids. |

### 3.3 After each chapter — the 5-minute check
1. Open `reports\chNN_verification.md`: read the **Verdict**, the parity counts and **Open items**. Any `unverified` item must have an open item you are willing to accept.
2. Open two images in `reports\chNN\figures\` — Python on the left, Octave/MATLAB output or the book figure on the right.
3. Open `knowledge\chNN.md`, section 7 "Feeds forward" — that is what the next chapter will assume exists.
4. `git log --oneline -8` in PowerShell — one commit per phase.
5. Optional now, or later in batch: run the notebook in Colab (§4).

### 3.4 Follow-up prompts during the chapter loop (use as needed)

Resume after an interruption (phases already `pass` are skipped):
```
/do-chapter 5 --from verify
```
Verification stopped with FAIL:
```
Read reports/ch06_verification.md "Open items". For each item, find the most likely root cause with file:line evidence in seaice/ and in the original .m file (check: complement of masks, 1-based indexing, uint8 wrap-around, 4- vs 8-connectivity, structuring-element shape, JPEG decode, random init). Fix the port with the matlab-porter agent — never the tolerance — then run /verify-chapter 6.
```
You doubt an `approx` or `unverified` label:
```
In reports/ch04_verification.md the Sobel edge result is labelled "approx". Generate an Octave reference by running the original edge() call from the ch4 MATLAB code on data/book/ch04/<image>, compare pixel-wise with our implementation using tools/compare_arrays.py, and either upgrade the label with evidence or explain the exact algorithmic difference in the report.
```
Rebuild only the notebook after you (or Claude) changed code:
```
/notebook-chapter 5
```
A chapter needs data that is not in the MATLAB folders:
```
List exactly which input files chapter 8 needs that are not in data/book/ch08, the best free online source for each (NASA Worldview snapshot, NASA Earth Observatory, public-domain IceBridge photos), and the exact Google Drive path MyDrive/seaice-py/data/manual/ch08/ where I should save anything that needs a login. Make the scripts and the notebook look in data/online/ch08 and data/manual/ch08 with a helpful error message, then rerun the affected scripts.
```
Skip the independent review to save time (not recommended for ch5/ch6):
```
/do-chapter 7 --skip-review
```
Override the sequential gate (only if you deliberately want to jump ahead):
```
I understand chapter 6 is not complete; run /do-chapter 7 anyway and note the missing dependency in analysis/ch07.md.
```
Something looks wrong in a figure:
```
Figure fig_5_18_compare.png: our segmentation shows over-segmented floes compared with the book. Check the marker generation (distance-transform threshold and h-minima value) against the values stated in chapters/ch05.txt and the ch5 .m files, fix, and regenerate the comparison.
```

### 3.5 Learning prompts (after a chapter is done — this is where the knowledge files pay off)

Study sheet:
```
/clear
Using knowledge/ch03.md, analysis/ch03.md and chapters/ch03.txt, build a study sheet in reports/ch03_study.md: the 5 key ideas, each equation with a 2-line intuition and a 3x3 worked numerical example, the MATLAB→Python differences that bit us, and 5 quiz questions with answers at the bottom.
```
Deep explanation of one algorithm:
```
Using knowledge/ch05.md and seaice/ch05_watershed.py, explain step by step with a small worked example on a 6x6 image how the marker-controlled watershed and the neighboring-region merging decide whether to merge two floes. Then show me the exact lines in the ch5 .m files that do the concavity test and the matching Python lines.
```
Cross-chapter overview (after ch6 or later):
```
Using knowledge/CUMULATIVE.md, draw the full processing pipeline from raw image to identified floes as a text diagram, naming the seaice function used at each step and the chapter that introduced it.
```

---

## PART 4 — GOOGLE DRIVE, COLAB, TROUBLESHOOTING, REFERENCE

### 4.1 Route A — Google Drive (recommended, matches your workflow)
1. Google Drive for Desktop is installed (Step 1.4) and `G:\My Drive\` exists.
2. After a chapter (or after several), either type in Claude Code:
```
Sync the Colab artifacts to Google Drive with tools/sync_to_drive.ps1 (DriveDir "G:\My Drive\seaice-py") and list what was copied.
```
   or run it yourself in PowerShell:
```powershell
cd "C:\Users\sislam27\Work\Climate Dynamics PHD\Sea Ice Image Processing with MATLAB"
powershell -ExecutionPolicy Bypass -File tools\sync_to_drive.ps1 -DriveDir "G:\My Drive\seaice-py"
```
   This mirrors `notebooks\`, `seaice\`, `knowledge\`, `data\book|online|synthetic\`, `requirements-colab.txt`, `book.yaml` into `MyDrive/seaice-py/`. Wait for the Drive tray icon to finish uploading.
3. Open https://drive.google.com → `seaice-py/notebooks/` → right-click `ch02_preliminaries.ipynb` → **Open with → Google Colaboratory**.
4. In the first code cell make sure `SOURCE = "drive"`, run it: it mounts Drive, changes directory to `MyDrive/seaice-py`, installs `requirements-colab.txt`, and adds the folder to `sys.path`. All later cells import from `seaice/` in Drive.
5. Data: book images are already there. Online (Tier 2) data is re-downloaded by the notebook's data cell into `data/online/`. Manual (Tier 4) data goes to `MyDrive/seaice-py/data/manual/chNN/` — upload via the Drive web UI, or from Colab after mounting:
```
!wget -O /content/drive/MyDrive/seaice-py/data/manual/ch08/<file> "<url>"
```

### 4.2 Route B — GitHub (fallback / for sharing)
1. In Claude Code: `Create a private GitHub repo named seaice-py and push the current branch.` (or do it manually: `git remote add origin https://github.com/<you>/seaice-py.git && git push -u origin main`). The PDF, the zip and the MATLAB folder are git-ignored; `data/book/` images are small and are committed.
2. In the notebook setup cell set `SOURCE = "github"` and `REPO_URL`. For a private repo store a token in Colab **Secrets** (key icon) named `GH_TOKEN` and ask: `Update the notebook setup cells to clone the private repo using the GH_TOKEN Colab secret.`
3. Colab → File → Open notebook → GitHub tab → paste the repo URL → pick the chapter notebook.

### 4.3 Data policy the notebooks follow
1. **Book images** shipped with the MATLAB code (`ch2/rgb.jpg`, …) — reproduce the book figures exactly; default everywhere.
2. **Free online** sea-ice imagery — NASA Worldview snapshots (MODIS/VIIRS true colour, any date/box, no login), NASA Earth Observatory photos, public-domain NOAA/USCG/IceBridge photos. Downloaded by a cached `fetch()` helper; sources recorded in `data/online/SOURCES.md`.
3. **Synthetic** — Ch9 model floes and unit tests; always labelled synthetic.
4. **Manual → Drive** — IceBridge DMS (NASA Earthdata login) or Sentinel-2 when a chapter really needs them; the notebook prints exact instructions and the Drive path.

### 4.4 Troubleshooting

| # | Symptom | Fix |
|---|---|---|
| T1 | `/do-chapter` not listed in `/skills` | You started `claude` outside the project folder, or `.claude` did not move in Step 1.6 (`dir -Force` to see hidden folders). |
| T2 | `oct2py` cannot find Octave | Add Octave's `mingw64\bin` to PATH (Step 1.3) or set the environment variable `OCTAVE_EXECUTABLE` to the full path of `octave-cli.exe`; restart the terminal and Claude Code. |
| T3 | Octave: `'imbinarize' undefined` | Expected: MATLAB-only function. The verifier falls back and labels the item; nothing to do. |
| T4 | Split chapters misaligned | Use the "The split is off…" prompt in §3.1 (or `python tools/split_pdf.py --offset <n>`). |
| T5 | "no book PDF found" | The PDF must sit directly in the project root; if there are several PDFs, set `book.pdf` in `book.yaml` to the right file name. |
| T6 | Verification loops 3× and stops | Use the FAIL prompt in §3.4; most causes are complement / 1-based / connectivity / SE shape. |
| T7 | Notebook fails only on Colab | Usually a path or a package missing from `requirements-colab.txt`; paste the Colab error into Claude Code. |
| T8 | Context-window warnings | `/clear` between chapters; never paste book text; let the agents read from disk. |
| T9 | Permission prompt on every command | `/permissions` in Claude Code → add the tool pattern; the kit already allows python/pytest/git/octave/jupyter. |
| T10 | Path errors with spaces | The project root has spaces; the kit uses `pathlib` and quoted paths. If a script breaks, tell Claude Code: `Quote all paths; the repo root contains spaces.` |

### 4.5 Reference — what each chapter produces

| Artifact | Purpose |
|---|---|
| `analysis/chNN.md` | Concepts, equations, every `.m` file mapped to a Python target, data + verification plan, risks |
| `seaice/chNN_<topic>.py` + `seaice/core/*.py` | Library code (reused by later chapters and by the notebooks) |
| `scripts/chNN_<matlab_name>.py` | One runnable script per original MATLAB script; saves figures to `outputs/chNN/` |
| `tests/test_chNN.py`, `reference/chNN/make_refs.py` | Synthetic-truth tests, Octave reference generation, parity tests |
| `reports/chNN_verification.md` + `figures/` | Evidence: parity table, figure comparisons, numbers vs the book, open items, verdict |
| `notebooks/chNN_<slug>.ipynb` + `build_chNN.py` | The Colab notebook, executed headlessly before it is accepted |
| `knowledge/chNN.md`, `CUMULATIVE.md`, `function_map.md` | The growing knowledge base each next chapter reads first |

### 4.6 Reference — the agents and skills
| Agent | Job | Writes |
|---|---|---|
| chapter-analyst | Reads chapter text + all `.m` files; port plan | `analysis/chNN.md` |
| matlab-porter | Writes the Python; runs every script | `seaice/`, `scripts/` |
| port-verifier | Tests, Octave references, figure comparison, report | `tests/`, `reference/`, `reports/` |
| port-reviewer | Independent line-by-line check vs equations | (reply only) |
| notebook-builder | Builds + executes the Colab notebook | `notebooks/` |
| knowledge-keeper | Captures knowledge, updates progress, commits | `knowledge/`, `progress.json` |

| Skill | Type | Purpose |
|---|---|---|
| `/setup-project` | command | One-time bootstrap incl. PDF split |
| `/do-chapter N [--from phase] [--skip-review]` | command | Full chapter pipeline |
| `/verify-chapter N`, `/notebook-chapter N` | command | Rerun one phase |
| `/status` | command | Progress + next command |
| `seaice-book` | background | Naming, docstring contract, parity vocabulary, definition of done |
| `matlab-to-python` | background | Semantics pitfalls + 90-row MATLAB→Python function map |
| `verify-port` | background | The four evidence levels, tolerances, report format |
| `colab-notebook` | background | Mandatory notebook layout, Drive/GitHub setup cell, headless execution |
| `chapter-knowledge` | background | Knowledge file format; how chapters build on each other |
| `data-sources` | background | Data tiers and Drive/Colab access |

### 4.7 Reusing the kit for another MATLAB book
Replace `book.yaml`, put the new PDF and code folder in the new project root, rewrite the "book in one paragraph" in `.claude/skills/seaice-book/SKILL.md`, and keep everything else. The `matlab-to-python` skill keeps improving as the knowledge-keeper promotes lessons into it.
