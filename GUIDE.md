# Step-by-Step Guide: Converting "Sea Ice Image Processing with MATLAB" to Python + Colab with Claude Code

**Project folder (Claude Code runs here):** `C:\Users\sislam27\Work\Climate Dynamics PHD\Sea Ice Image Processing with MATLAB`

**Public results:** repository `https://github.com/shammun/seaice-py` · site `https://shammun.github.io/seaice-py/` · chapter 2 notebook in Colab via the "Open in Colab" button on each chapter page.

This guide was rewritten on 9 September 2026 after chapter 2 was completed and published, so it describes the workflow **as it actually ran**, not as it was planned. Part 1 is the one-time setup (what was really needed on this machine). Part 2 is the prompt playbook: every prompt that was typed, with *when* to use it. Part 3 walks through the chapter loop you repeat for chapters 2–10, including the new **publish** phase. Part 4 covers GitHub Pages, Colab, the private-image / public-fallback data scheme, troubleshooting, and a reference of what gets produced.

## 0. How the system works

The two source assets stay untouched in the project folder: the single book PDF, and the folder `K30735_Sea Ice Image Processing with MATLAB_matlab codes\matlab\` with `ch2` … `ch10`. The kit adds the "brain":

| Kit component | What it does |
|---|---|
| `CLAUDE.md` | Project rules Claude Code reads every session: sequential chapters, every `.m` file gets one Python counterpart, no silent approximations, never touch the PDF or MATLAB folder, and (rule 12, added before going public) **nothing derived from the copyrighted book is ever committed** |
| `book.yaml` | Chapter map (titles, printed page ranges, which MATLAB folder belongs to which chapter, PDF page offset = 31) |
| `tools/split_pdf.py` | Splits the book PDF into `chapters/ch01…ch11.pdf` + `.txt` (local only, git-ignored) |
| `tools/inventory_matlab.py` | Lists every `.m` file, the toolbox functions it calls, the images it reads |
| `tools/run_matlab_ref.py` | Runs the original `.m` code in **MATLAB R2025a** (`matlab -batch`) and saves `.mat` reference outputs — the parity engine |
| `tools/compare_arrays.py` | Compares Python output with the MATLAB references (tolerances, label renumbering) |
| `tools/publish_notebook.py` | **Publish phase:** builds the Colab variant, the styled HTML page, `index.html` and the README table for a chapter |
| `tools/check_public.py` + `.githooks/pre-push` | Refuses to push notebooks with outputs, HTML built from the book image, or any book-derived path |
| `seaice/core/public_images.py`, `data/online/SOURCES.md` | Registry of public-domain substitute images (one per book image) with provenance |
| `.claude/skills/` (11 skills) | Slash commands `/setup-project`, `/do-chapter`, `/verify-chapter`, `/notebook-chapter`, `/status`, plus background knowledge the agents load (`matlab-to-python`, `verify-port`, `colab-notebook`, `data-sources`, `chapter-knowledge`, `seaice-book`) |
| `.claude/agents/` (6 subagents) | `chapter-analyst`, `matlab-porter`, `port-verifier`, `port-reviewer`, `notebook-builder`, `knowledge-keeper` — each works in its own context window |
| `knowledge/`, `progress.json` | The growing knowledge base and the progress tracker (seven phases per chapter) |

The pipeline per chapter:

```text
/do-chapter N
  1 chapter-analyst  : reads chapters/chNN.txt + every .m in matlab/chN  -> analysis/chNN.md (port plan)
  2 matlab-porter    : writes seaice/chNN_*.py, seaice/core/*, scripts/chNN_*.py; runs every script
  3 port-verifier    : tests + MATLAB reference outputs + book-figure comparisons -> reports/chNN_verification.md (PASS/FAIL)
  4 port-reviewer    : fresh-context check of code vs book equations (PASS / PASS-WITH-FIXES / FAIL)
  5 notebook-builder : notebooks/chNN_<slug>.ipynb built with nbformat, executed headlessly
  6 knowledge-keeper : knowledge/chNN.md + CUMULATIVE.md + function_map.md + progress.json + git commit
  7 publish (orchestrator): substitute image registered, notebook run twice (private image / public fallback),
                            tools/publish_notebook.py -> _colab.ipynb + .html + index, push, GitHub Pages check
/clear  ->  /do-chapter N+1   (reads knowledge/CUMULATIVE.md first, so it builds on everything learned)
```

**Why MATLAB, not Octave.** The original plan assumed GNU Octave. MATLAB R2025a is installed on this machine, so `/setup-project` detected it and the verifier runs the authors' `.m` code in real MATLAB. That makes the parity labels `exact` / `near` measurements against MATLAB itself, and MATLAB-only functions (`imbinarize`, `adaptthresh`, `activecontour`) are directly testable. Octave was never installed and is only a fallback the tools support if MATLAB is absent.

**Chapter ↔ folder mapping (confirmed by `/setup-project`):** book chapters 2–9 ↔ `ch2`–`ch9`; the `ch10` folder is Appendix A (`fisheye_calibration.m` → A.2 lens distortion, `orthoretification.m` → A.1). Chapter 1 has no code. `ch6` and `ch7` ship byte-identical `Sea_Ice_Floe_Identification` folders (port once, reuse); `ch5` duplicates the chain-code files of `ch2`.

---

## PART 1 — ONE-TIME SETUP (what was actually needed on this machine)

### Step 1.1 — Python 3.11
Already present (Python 3.11.5). Check with:
```powershell
python --version
```
`/setup-project` creates the project's own `.venv` and installs `requirements.txt`; every later command uses `.venv\Scripts\python.exe`, never the Anaconda Python that is also on this machine (see T7).

### Step 1.2 — Git and the GitHub CLI
Git was already installed and configured (`Shammunul Islam`). The GitHub CLI `gh` is installed and logged in as `shammun` (needed once, in Step 4.1, to create the public repository and enable GitHub Pages). Check:
```powershell
git --version
gh auth status
```

### Step 1.3 — MATLAB R2025a (the reference engine)
Nothing to install: `C:\Program Files\MATLAB\R2025a\bin\matlab.exe` exists and `/setup-project` recorded it in `progress.json → environment`. The verifier calls it non-interactively (`matlab -batch`) through `tools/run_matlab_ref.py`; each chapter's `reference/chNN/make_refs.py` regenerates the `.mat` files. If you ever work on a machine without MATLAB, install GNU Octave (`winget install GNU.Octave`, then `pkg install -forge image`) and the same tools fall back to it with weaker labels.

### Step 1.4 — Google Drive: **not** needed on this PC
The original plan mirrored the repo into `G:\My Drive\seaice-py` with `tools/sync_to_drive.ps1`. On this machine `G:` is an external disk ("Elements"), Google Drive for Desktop is not mounted, and the sync was never run. Instead, Colab gets the code from GitHub and the notebooks mount **your Google Drive inside Colab** (folder `MyDrive/Sea_Ice_Colab`) only for your private copies of the book images (Part 4.2). `tools/sync_to_drive.ps1` remains in the kit but is unused.

### Step 1.5 — Confirm the project folder contents
```powershell
cd "C:\Users\sislam27\Work\Climate Dynamics PHD\Sea Ice Image Processing with MATLAB"
dir
dir ".\K30735_Sea Ice Image Processing with MATLAB_matlab codes\matlab"
```
You must see: one `.pdf` (the book, ~26 MB), the folder `K30735_Sea Ice Image Processing with MATLAB_matlab codes` (the `.zip` of the same name is ignored), and inside `...\matlab\` the folders `ch2` … `ch10`. All three are git-ignored forever.

### Step 1.6 — Unzip the kit into the project folder
```powershell
cd "C:\Users\sislam27\Work\Climate Dynamics PHD\Sea Ice Image Processing with MATLAB"
Expand-Archive -Path "$env:USERPROFILE\Downloads\seaice-py-kit.zip" -DestinationPath . -Force
Move-Item -Path ".\seaice-py-kit\*" -Destination . -Force
Move-Item -Path ".\seaice-py-kit\.claude" -Destination . -Force
Remove-Item ".\seaice-py-kit" -Recurse -Force
dir -Force
```
`dir -Force` must show `CLAUDE.md`, `book.yaml`, `progress.json`, `requirements.txt`, `GUIDE.md`, `.gitignore`, the folders `.claude`, `tools`, `knowledge`, plus your original PDF and `K30735...` folder. Check the hidden folder:
```powershell
dir .claude\skills
dir .claude\agents
```
(11 skill folders, 6 agent files.)

### Step 1.7 — Start Claude Code inside the project folder
```powershell
cd "C:\Users\sislam27\Work\Climate Dynamics PHD\Sea Ice Image Processing with MATLAB"
claude
```
Claude Code must be started **in this folder** — that is how it finds `CLAUDE.md`, `.claude\skills` and `.claude\agents`. Claude Code also keeps a small persistent memory for this folder (host quirks, the publish checklist), so a new session already knows the workflow.

### Step 1.8 — Confirm the skills and the status
```prompt
/skills
```
You should see `setup-project`, `do-chapter`, `verify-chapter`, `notebook-chapter`, `status` and the background skills. Then:
```prompt
/status
```
Expected on a fresh install: every chapter `todo` and "Next: run /setup-project".

### Step 1.9 — Optional: Chrome extension for screenshots
The Claude-in-Chrome extension was not connected on this machine. It is not needed: when a page must be checked visually, Claude Code serves the repo with `python -m http.server` and screenshots it with headless Chrome (`chrome.exe --headless=new --screenshot=...`).

---

## PART 2 — THE PROMPT PLAYBOOK (which prompt, when)

### 2.1 Decision table

| Situation | Prompt to type | Details in |
|---|---|---|
| Fresh install, nothing run yet | `/setup-project` | §3.1 |
| The PDF split landed on the wrong pages | "The split is off…" prompt | §3.1 follow-ups |
| Ready to start / continue the book | `/clear` then `/do-chapter N` | §3.2 |
| A chapter stopped with verify FAIL | "Read reports/chNN… Open items…" root-cause prompt | §3.4 |
| Session was interrupted mid-chapter | `/do-chapter N --from <phase>` | §3.4 |
| You edited code and want the notebook rebuilt | `/notebook-chapter N` | §3.4 |
| You doubt an `approx`/`unverified` label | "Generate a MATLAB reference…" prompt | §3.4 |
| A chapter needs data that is not shipped | "List exactly which data files…" prompt | §3.4 |
| The chapter is done but not yet on the website | "Publish chapter N…" prompt | §3.6 |
| A Colab "Save to GitHub" commit appeared on main | "Check origin/main for a Colab commit…" prompt | §4.2 |
| You want to *learn* the chapter | Study-sheet / explain prompts | §3.5 |
| Claude Code says a command was blocked by the permission classifier | Run it yourself with the `!` prefix | §2.2 |
| You changed the workflow and want it remembered | "Save these findings in your memory…" prompt | §2.3 |
| You don't know where you are | `/status` | any time |

### 2.2 Rules of thumb (learned in chapter 2)
- One chapter per session. Always type `/clear` before `/do-chapter`. Everything needed is on disk.
- Never paste the book text or MATLAB code into the chat. The agents read `chapters/chNN.txt` and the `.m` files from disk.
- **Outward-facing commands may be blocked.** Claude Code's permission classifier refused `gh repo create --public --push`. When Claude Code reports such a block, run the exact command yourself by typing it with a leading `!` in the Claude Code prompt — the output lands in the conversation and Claude Code continues from there:
```prompt
! gh repo create shammun/seaice-py --public --source . --remote origin --push --description "Sea Ice Image Processing with MATLAB (Zhang & Skjetne 2018) ported to verified Python, one Colab notebook per chapter"
```
- If Claude Code proposes to "skip verification for now" or "loosen the tolerance", say no:
```prompt
Do not loosen tolerances. Report the discrepancy in Open items with your best hypothesis and stop.
```
- **Never use *File → Save a copy in GitHub* from Colab** when the notebook ran on your private book image: it writes the cell outputs (figures of the copyrighted image) into the public repository. Save to Drive instead. This happened once on 9 September 2026 and had to be force-pushed away (§4.2).
- The repository is public. Book text, book-shipped images, PDF page crops, figure comparisons against book pages and executed notebooks stay local (git-ignored). `tools/check_public.py` runs before every push.

### 2.3 Keeping the workflow itself up to date
After changing how things are done, ask Claude Code to write it down — both into its memory and into the kit files:
```prompt
Ensure that these new findings, and the way we got to this point, are saved in your memory so that next time I ask for the next chapter you know what to do: make the Colab file, the HTML file, put things in the right files and folders, and arrange the code the way we did for chapter 2.
```
In chapter 2 this produced: the PUBLISH phase in the `do-chapter` skill, the `publish` column in `progress.json` and `/status`, the updated `data-sources` and `colab-notebook` skills, and the "Publishing findings" section of `knowledge/CUMULATIVE.md`.

---

## PART 3 — THE WORK, PHASE BY PHASE

### 3.1 Phase A — `/setup-project` (once) — as it actually ran

**Type:**
```prompt
/setup-project
```

**What Claude Code did (about 15 minutes):**
1. Confirmed the project root, the single book PDF, and `K30735_...\matlab\ch2…ch10`.
2. Created `.venv`, installed `requirements.txt`, created `requirements-colab.txt`.
3. Detected **MATLAB R2025a** (Octave absent) and recorded it in `progress.json`.
4. Split the book PDF into `chapters/ch01…ch11.pdf` + `.txt` and wrote `chapters/_page_map.json`. The automatic offset detection was **wrong** (it gave −3); the correct offset is 31 and was fixed with the follow-up prompt below and recorded in `book.yaml`.
5. Ran `tools/inventory_matlab.py` → `analysis/_matlab_inventory.md`.
6. Confirmed `ch10` = Appendix A and updated `book.yaml`.
7. Copied the images shipped with the MATLAB code into `data/book/chNN/` (all chapters) — these are **your private copies**; they were later removed from git history and are git-ignored.
8. Seeded `seaice/`, `tests/`, `knowledge/CUMULATIVE.md`, `knowledge/function_map.md`, ran a smoke test, `git init` + first commit. Two small follow-up commits fixed a shell-mangled section in `CUMULATIVE.md`, NUL characters, and added `.gitattributes` (Windows CRLF handling).

**What you check (2 minutes):** `chapters\ch02.txt` starts with "Digital Image Processing Preliminaries"; `chapters\ch05.txt` with "Watershed-Based Ice Floe Segmentation"; the checklist says MATLAB ✅; `analysis\_matlab_inventory.md` lists ch2 … ch10.

**Follow-up prompts used or available for this phase**

The split was off (this one was needed):
```prompt
The split is off: chapters/ch02.txt starts on the wrong page. Rerun tools/split_pdf.py with the corrected --offset, then re-check that ch02, ch05 and ch09 each begin with their chapter title, and record the offset in book.yaml.
```
If the ch10 folder needs explaining:
```prompt
Explain what the files in K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab/ch10 implement, which book section each corresponds to, and update book.yaml and CLAUDE.md so the mapping is correct.
```
If the MATLAB folder is in a different place:
```prompt
The MATLAB code folders ch2…ch10 are actually at <relative path>. Update book.yaml → book.matlab_root, confirm tools/inventory_matlab.py finds them, and rerun the inventory.
```

### 3.2 Phase B — the chapter loop (repeat for N = 2 … 10)

**Type:**
```prompt
/clear
/do-chapter 2
```

**What Claude Code does (chapter 2 took about 2½ hours of agent time):** the pipeline from §0. It prints a plan, then a short update as each phase finishes, and commits after every phase (`ch02: analyze — …`, `ch02: port — …`, …). It pauses for you only when verification fails three times, when data needs a manual download, or when chapter N−1 is not complete.

**How chapter 2 actually went (so you know what "normal" looks like):**

| Phase | Duration | Outcome |
|---|---|---|
| analyze | 10 min | `analysis/ch02.md`: 7 `.m` files mapped, 16 risks, two bugs found in the book's own code (`2*Ig` typo in `color_image.m`, `'cww'` typo in `chain_diff.m`) |
| port | 22 min | 15 `seaice/core` modules, 9 scripts, all exit 0, 50 figures |
| verify | 27 min + 5 min fix loop | 83 tests vs MATLAB references; FAIL on two one-line defects (`interp_nearest` aliasing, `imhist` on logical images) → porter fixed → PASS |
| review | 9 min | PASS-WITH-FIXES: no numerical defect; Eq. (2.15) as printed is in correlation form (book inconsistency), MATLAB-style `imfilter` options, figure-file names → applied, tests now 86 |
| notebook | 11 min | 42 cells, 25 figures, executes in 43 s |
| knowledge | 8 min | `knowledge/ch02.md`, CUMULATIVE, 37 rows in `function_map.md` |
| publish | (added afterwards, §3.6) | Colab variant + HTML page + index, public repo, GitHub Pages |

Final parity for chapter 2: 23 exact · 1 near · 1 approx · 4 reimplemented · 1 unverified (the image behind Fig 2.7 is not shipped).

**Then do the 5-minute human check (§3.3), publish (§3.6), and continue:**
```prompt
/clear
/do-chapter 3
```
…and so on through `/do-chapter 10`.

**Chapter-specific notes:**

| Chapter | Expect |
|---|---|
| 2 Preliminaries | Done. Built the `seaice/core/` primitives everything else reuses: MATLAB-compatible `rgb2gray`, `imhist`, `bwlabel` numbering, `bwdist`, `conv2`/`imfilter`, DIPUM `boundaries`/`fchcode`/`bound2im`, `interp2`. Coordinates are 0-based `(row, col)` everywhere. |
| 3 Ice pixel detection | Otsu / local thresholding, k-means (`graythresh`/`im2bw` still to be written). k-means is random-init in both languages: parity is by cluster centres, not labels. Needs a public-domain substitute registered for each of `1.jpg`, `2.jpg`, `test.jpg`, `ch3ice.jpg`. |
| 4 Edge detection | MATLAB's `edge()` and `strel('disk')` are not what scikit-image does; expect `reimplemented`/`near` labels. `imfilter` = correlation — check each `.m` before labelling. |
| 5 Watershed + merging | Heavy. Reuses ch2's chain codes (`fchcode` never raises where MATLAB's `minmag` errors — port `try/catch` knowingly). Expect 1–2 verify loops. |
| 6 GVF snake | Plain MATLAB → strong references. `interp2` needs the `(Xq, Yq) → (v, u)` swap and −1. Slow; scripts use a `SCALE` constant. |
| 7 Ice type | Same floe-identification code as ch6 (byte-identical folder); several book figures to reproduce. |
| 8 Applications | Shipborne camera inputs partly missing (only `.mat` results shipped) → substitutes / manual download; the report says exactly what and where. |
| 9 Model ice | The basin video `dypic_05100_cam1_top.avi` is not shipped → synthetic rectangular-floe images, clearly labelled. |
| 10 (Appendix A) | Orthorectification and lens-distortion polynomials; verified on synthetic calibration grids; reuses `interp.warp_image`. |

### 3.3 After each chapter — the 5-minute check
1. Open `reports\chNN_verification.md`: read the **Verdict**, the parity counts and **Open items**.
2. Open two images in `reports\chNN\figures\` (local only) — Python on the left, the book page on the right.
3. Open `knowledge\chNN.md`, section "Feeds forward" — what the next chapter will assume exists.
4. `git log --oneline -8` — one commit per phase.
5. After publishing: open `https://shammun.github.io/seaice-py/notebooks/chNN_<slug>.html` and click **Open in Colab**.

### 3.4 Follow-up prompts during the chapter loop (use as needed)

Resume after an interruption (phases already `pass` are skipped):
```prompt
/do-chapter 5 --from verify
```
Verification stopped with FAIL:
```prompt
Read reports/ch06_verification.md "Open items". For each item, find the most likely root cause with file:line evidence in seaice/ and in the original .m file (check: complement of masks, 1-based indexing, uint8 wrap-around, 4- vs 8-connectivity, structuring-element shape, JPEG decode, random init). Fix the port with the matlab-porter agent — never the tolerance — then run /verify-chapter 6.
```
You doubt an `approx` or `unverified` label:
```prompt
In reports/ch04_verification.md the Sobel edge result is labelled "approx". Generate a MATLAB reference with tools/run_matlab_ref.py by running the original edge() call from the ch4 MATLAB code on data/book/ch04/<image>, compare pixel-wise with our implementation using tools/compare_arrays.py, and either upgrade the label with evidence or explain the exact algorithmic difference in the report.
```
Rebuild only the notebook after you (or Claude) changed code:
```prompt
/notebook-chapter 5
```
A chapter needs data that is not in the MATLAB folders:
```prompt
List exactly which input files chapter 8 needs that are not in data/book/ch08, the best public-domain source for each (NASA Worldview snapshot, NASA Earth Observatory, IceBridge photos), register a substitute for each book image name in seaice/core/public_images.py with provenance in data/online/SOURCES.md, and give the exact Google Drive path MyDrive/Sea_Ice_Colab/data/manual/ch08/ for anything that needs a login. Then rerun the affected scripts.
```
Skip the independent review to save time (not recommended for ch5/ch6):
```prompt
/do-chapter 7 --skip-review
```
Something looks wrong in a figure:
```prompt
Figure fig_5_18_compare.png: our segmentation shows over-segmented floes compared with the book. Check the marker generation (distance-transform threshold and h-minima value) against the values stated in chapters/ch05.txt and the ch5 .m files, fix, and regenerate the comparison.
```

### 3.5 Learning prompts (after a chapter is done)

Study sheet:
```prompt
/clear
Using knowledge/ch03.md, analysis/ch03.md and chapters/ch03.txt, build a study sheet in reports/ch03_study.md: the 5 key ideas, each equation with a 2-line intuition and a 3x3 worked numerical example, the MATLAB→Python differences that bit us, and 5 quiz questions with answers at the bottom.
```
Deep explanation of one algorithm:
```prompt
Using knowledge/ch05.md and seaice/ch05_watershed.py, explain step by step with a small worked example on a 6x6 image how the marker-controlled watershed and the neighboring-region merging decide whether to merge two floes. Then show me the exact lines in the ch5 .m files that do the concavity test and the matching Python lines.
```
Cross-chapter overview (after ch6 or later):
```prompt
Using knowledge/CUMULATIVE.md, draw the full processing pipeline from raw image to identified floes as a text diagram, naming the seaice function used at each step and the chapter that introduced it.
```

### 3.6 Phase C — PUBLISH (the step added after chapter 2)

From chapter 3 on, `/do-chapter N` runs this automatically as phase 7. For chapter 2 it was done with the three prompts below, which are kept here because they define what "published" means.

**1. Make the HTML page and the Colab variant, like the fast.ai notes:**
```prompt
Convert the notebook to an HTML file like "<path to an existing notes page>.html" and give it an Open-in-Colab button as that one has. Also add Colab setup cells at the beginning of the Colab file like "<path to an existing _colab.ipynb>". Have you already pushed the files to GitHub? I think that is needed.
```
Result: `tools/publish_notebook.py`, `assets/clean-educational.css`, `index.html`, the README index table, and the `_colab.ipynb` + `.html` for the chapter.

**2. Make the repository safe to publish** (history rewrite — the repo had no remote yet, so this was safe):
```prompt
This repo is about to become PUBLIC, so nothing derived from the copyrighted book may be in it. Add chapters/*.txt, data/book/, data/manual/, reports/**/figures/, outputs/, notebooks/executed_*.ipynb to .gitignore (keep chapters/_page_map.json). Untrack those paths and purge them from ALL history with git-filter-repo. The files stay on disk as my private working copies. Add an MIT LICENSE and a README section "About the source material". Add rule 12 to CLAUDE.md and one line to the data-sources, colab-notebook and verify-port skills: the repo is public; never commit book text, book-shipped images, PDF page crops, or executed notebooks that contain them. Commit as "public release prep".
```
Result: 12 commits rewritten, pack size 309 kB, `LICENSE`, README section, rule 12.

**3. Implement the private-Drive / public-fallback data scheme** (the decisions that now live in the `data-sources` and `colab-notebook` skills):
```prompt
Implement load_image(chapter, name): look for the reader's private copy first (<cwd>/data/book, the repo, /content/drive/MyDrive/Sea_Ice_Colab/data/book), else download the registered public-domain substitute (one NASA image per book image name, verified with a real request, provenance in data/online/SOURCES.md). Print "Data source: <label> — <path>". Notebook cell 1 mounts Drive and chdirs to MyDrive/Sea_Ice_Colab, cell 2 clones or pulls the repo, cell 3 loads the image and prints a banner when the fallback is used; book-quoted values are only printed for the book image. Execute the notebook twice (with and without data/book), regenerate the HTML from the public-domain run, and show me the two "Data source:" lines.
```
Result: `seaice/core/io.py: load_image()`, `seaice/core/public_images.py` (ch02 → NASA MODIS Terra, Beaufort Sea marginal ice zone, 25 July 2019, requested at the book image's own 2048×1536 size so every pixel index in the notebook stays valid), the two runs:

```text
Data source: book (local) — .../data/book/ch02/rgb.JPG
Data source: public-domain substitute (NASA) — .../data/online/ch02/nasa_modis_terra_beaufort_miz_2019-07-25.jpg
Running on a public-domain substitute image; figures show the same operations, but values quoted in the book only hold for the book's own image.
```

**What the publish phase does for every later chapter** (see the `do-chapter` skill §7):
1. Registers one public-domain substitute per new book image (`seaice/core/public_images.py` + `data/online/SOURCES.md`), at the book image's pixel size, after fetching and *looking at* it (chapter 2 needed four tries to find a cloud-free floe field).
2. Builds the notebook with cells 1–3 copied from `notebooks/build_ch02.py`.
3. Runs it twice headlessly: with `data/book` present → `Data source: book (local)`; with `data/book` renamed to `data/book_private` and the cache deleted → `Data source: public-domain substitute (NASA)`. Both must have 0 errors and all figures.
4. With `data/book` still renamed, runs `tools/publish_notebook.py chNN` → `notebooks/chNN_<slug>_colab.ipynb` (outputs stripped), `notebooks/chNN_<slug>.html`, `index.html`, README table. Renames `data/book` back. Greps the HTML: `book (local)` must be 0.
5. Sets `publish: pass`, commits `chNN: publish — …`, pushes (the pre-push hook runs `tools/check_public.py`), and confirms the Pages URL and the Colab URL respond.

If a chapter was finished before the publish phase existed, or you want to redo it:
```prompt
Publish chapter N: register a public-domain substitute for each book image it loads, rebuild the notebook with cells 1–3 from build_ch02.py, execute it with and without data/book, run tools/publish_notebook.py chNN from the public-domain run, verify the HTML contains no "book (local)" label, commit "chNN: publish — …", push, and confirm the GitHub Pages and Colab URLs.
```

---

## PART 4 — GITHUB PAGES, COLAB, DATA, TROUBLESHOOTING, REFERENCE

### 4.1 The public repository and GitHub Pages (done once, after chapter 2)
1. Local branch renamed from `master` to `main` (Colab and Pages links use `main`).
2. The repository was created and pushed by **you** (the classifier blocks Claude Code from doing it):
```prompt
! gh repo create shammun/seaice-py --public --source . --remote origin --push --description "Sea Ice Image Processing with MATLAB (Zhang & Skjetne 2018) ported to verified Python, one Colab notebook per chapter"
```
3. Claude Code then enabled GitHub Pages (legacy build from `main`, path `/`) with the GitHub API and verified `https://shammun.github.io/seaice-py/` and the chapter page return 200. The first build takes about a minute.
4. Later chapters only need `git push` (Claude Code does it; if blocked, type `! git push`).

What lives where on the site: `index.html` (chapter table, regenerated by the publish tool), `notebooks/chNN_<slug>.html` (the executed notebook with a **Download .ipynb** and an **Open in Colab** button), `assets/clean-educational.css` (shared with your fast.ai notes).

### 4.2 Using the notebooks in Colab
1. Open the chapter page on the site and click **Open in Colab** (URL pattern `https://colab.research.google.com/github/shammun/seaice-py/blob/main/notebooks/chNN_<slug>_colab.ipynb`).
2. Run cell 1: it mounts your Google Drive and moves into `MyDrive/Sea_Ice_Colab` (created if missing). Readers without Drive get a temporary `/content/Sea_Ice_Colab`.
3. Run cell 2: clones `seaice-py` into that folder (or `git pull` on later sessions) and installs `requirements-colab.txt`.
4. Run cell 3: `load_image("chNN", "<book image>")`. **To use the book's own images, put your private copies in `MyDrive/Sea_Ice_Colab/data/book/chNN/`** (upload via the Drive web UI) — the label becomes `book (private Drive)` and the book-quoted values are printed. Without them the public-domain substitute is downloaded into `MyDrive/Sea_Ice_Colab/data/online/chNN/` and a banner says so.
5. **Saving:** *File → Save a copy in Drive*. **Never** *Save a copy in GitHub* after running with the book image — it commits the outputs (book-derived figures) to the public repo. If it happens anyway:
```prompt
Check origin/main for a "Created using Colab" commit. If the saved notebook contains outputs from a book image, replace the remote tip with the clean local main using git push --force-with-lease, regenerate the _colab.ipynb with outputs stripped, and tell me whether the orphaned commit still needs a GitHub support purge.
```
(On 9 September 2026 this exact situation occurred; the tip was replaced within minutes. GitHub keeps the orphaned commit fetchable by SHA until it garbage-collects; support can purge it on request.)

Verified on Colab (9 Sep 2026): Drive mounted, repo cloned, `Data source: book (private Drive) — /content/drive/MyDrive/Sea_Ice_Colab/data/book/ch02/rgb.JPG`, 25 figures, 0 errors.

### 4.3 Data policy (private copies vs public fallback)
| Tier | What | Where | Committed? |
|---|---|---|---|
| 1 Book-shipped | images inside the MATLAB archive (`ch2/rgb.jpg`, …) | `data/book/chNN/` locally; `MyDrive/Sea_Ice_Colab/data/book/chNN/` for Colab | **never** (purged from history; git-ignored) |
| 2 Public-domain substitute | one NASA image per book image name, same pixel size | downloaded on demand to `data/online/chNN/` | only the registry (`seaice/core/public_images.py`) and `data/online/SOURCES.md` |
| 3 Synthetic | model-basin floes, unit-test fixtures | `seaice/core/synth.py`, `data/synthetic/` | code only |
| 4 Manual (login) | IceBridge DMS, Sentinel-2 | `data/manual/chNN/`, `MyDrive/Sea_Ice_Colab/data/manual/chNN/` | never |

`seaice.core.io.load_image(chapter, name)` implements the lookup order (cwd → repo → Drive → substitute) and returns `(image, source_label)`. Scripts and tests use `allow_fallback=False` and skip with a clear message when your private copy is absent; the parity tests therefore only run on machines with the book images and MATLAB.

Also local-only (git-ignored): `chapters/*.txt` and `*.pdf` (the split book), `outputs/`, `reports/**/figures/` (side-by-side comparisons with rendered book pages), `reference/**/*.mat`, `notebooks/executed_*.ipynb`.

### 4.4 Troubleshooting

| # | Symptom | Fix |
|---|---|---|
| T1 | `/do-chapter` not listed in `/skills` | You started `claude` outside the project folder, or `.claude` did not move in Step 1.6 (`dir -Force` to see hidden folders). |
| T2 | Verifier cannot run MATLAB | Check `C:\Program Files\MATLAB\R2025a\bin\matlab.exe` and `progress.json → environment`; `tools/run_matlab_ref.py --help` shows the CLI. Original scripts that `imread('rgb.jpg')` relatively must be run from a scratch cwd (`histogram.m` also writes PNGs there). |
| T3 | Claude Code says a command was "blocked by the classifier" | Outward-facing actions (creating a public repo, pushing) need you: type the command with a leading `!` in the prompt (§2.2). |
| T4 | Split chapters misaligned | The offset is 31 (`book.yaml → pdf_offset`); rerun `python tools/split_pdf.py --offset 31`. |
| T5 | "no book PDF found" | The PDF must sit directly in the project root; if there are several PDFs, set `book.pdf` in `book.yaml`. |
| T6 | Verification loops 3× and stops | Use the FAIL prompt in §3.4; most causes are complement / 1-based / connectivity / SE shape / correlation-vs-convolution. |
| T7 | `python -m jupyter nbconvert` fails with a scikit-image binary error | It dispatched to Anaconda's `jupyter-nbconvert.exe` on PATH. Use `.venv\Scripts\python.exe -m nbconvert …` (the publish tool already does). |
| T8 | `G:\My Drive` not found | `G:` is an external disk on this PC; Google Drive for Desktop is not installed. Not needed — see Step 1.4. |
| T9 | Chrome extension "not connected" | Not needed; screenshots are taken with headless Chrome against a local `http.server`. |
| T10 | `git push` rejected (non-fast-forward) after using Colab | A Colab "Save to GitHub" commit landed on `main`. Inspect it (§4.2); never merge outputs from a book image. |
| T11 | Notebook works locally, fails on Colab | Usually a package missing from `requirements-colab.txt` or a path assumption; paste the Colab error into Claude Code. |
| T12 | Context-window warnings | `/clear` between chapters; never paste book text; let the agents read from disk. |
| T13 | Path errors with spaces | The project root has spaces; the kit uses `pathlib` and quoted paths. Tell Claude Code: `Quote all paths; the repo root contains spaces.` |

### 4.5 Reference — what each chapter produces

| Artifact | Purpose | Public? |
|---|---|---|
| `analysis/chNN.md` | Concepts, equations, every `.m` file mapped to a Python target, data + verification plan, risks, book-text issues | yes |
| `seaice/chNN_<topic>.py` + `seaice/core/*.py` | Library code (reused by later chapters and by the notebooks) | yes |
| `scripts/chNN_<matlab_name>.py` | One runnable script per original MATLAB script; saves figures to `outputs/chNN/` | yes (figures no) |
| `tests/test_chNN.py`, `reference/chNN/make_refs.py` | Synthetic-truth tests, MATLAB reference generation, parity tests | yes (`.mat` no) |
| `reports/chNN_verification.md`, `reports/chNN_review.md` | Evidence: parity table, numbers vs the book, deviations, open items, verdict; the independent review | yes |
| `reports/chNN/figures/` | Side-by-side comparisons with the book pages | **no** (local) |
| `notebooks/chNN_<slug>.ipynb` + `build_chNN.py` | The teaching notebook, executed headlessly before it is accepted | yes (no outputs) |
| `notebooks/chNN_<slug>_colab.ipynb`, `notebooks/chNN_<slug>.html` | Colab variant and the published page (built from the public-domain run) | yes |
| `knowledge/chNN.md`, `CUMULATIVE.md`, `function_map.md` | The growing knowledge base each next chapter reads first | yes |
| `seaice/core/public_images.py`, `data/online/SOURCES.md` | Substitute-image registry and provenance | yes |

### 4.6 Reference — the agents, skills and tools
| Agent | Job | Writes |
|---|---|---|
| chapter-analyst | Reads chapter text + all `.m` files; port plan | `analysis/chNN.md` |
| matlab-porter | Writes the Python; runs every script; applies verifier/reviewer fixes | `seaice/`, `scripts/` |
| port-verifier | Tests, MATLAB references, figure comparison, report | `tests/`, `reference/`, `reports/` |
| port-reviewer | Independent line-by-line check vs equations | (reply, saved to `reports/chNN_review.md`) |
| notebook-builder | Builds + executes the notebook | `notebooks/` |
| knowledge-keeper | Captures knowledge, updates progress, commits | `knowledge/`, `progress.json` |

| Skill / tool | Type | Purpose |
|---|---|---|
| `/setup-project` | command | One-time bootstrap incl. PDF split |
| `/do-chapter N [--from phase] [--skip-review]` | command | Full chapter pipeline, seven phases |
| `/verify-chapter N`, `/notebook-chapter N` | command | Rerun one phase |
| `/status` | command | Progress (incl. `publish`) + next command |
| `seaice-book` | background | Naming, docstring contract, parity vocabulary, definition of done |
| `matlab-to-python` | background | Semantics pitfalls + MATLAB→Python function map (grows each chapter) |
| `verify-port` | background | Evidence levels, tolerances, report format, MATLAB reference generation |
| `colab-notebook` | background | Cells 1–3 template, headless execution, publishing, the Colab-save hazard |
| `chapter-knowledge` | background | Knowledge file format; how chapters build on each other |
| `data-sources` | background | Private copies vs public substitutes; registering a substitute |
| `tools/publish_notebook.py chNN` | tool | Colab variant + HTML page + index + README table |
| `tools/check_public.py` | tool | Public-repo safety check (also the pre-push hook) |
| `tools/build_guide_docx.py` | tool | Regenerates `GUIDE.docx` from `GUIDE.md` |

### 4.7 Reusing the kit for another MATLAB book
Replace `book.yaml`, put the new PDF and code folder in the new project root, rewrite the "book in one paragraph" in `.claude/skills/seaice-book/SKILL.md`, clear `seaice/core/public_images.py` and `progress.json`, and keep everything else. The `matlab-to-python` skill keeps improving as the knowledge-keeper promotes lessons into it.
