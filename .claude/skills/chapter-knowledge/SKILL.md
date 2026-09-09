---
name: chapter-knowledge
description: How knowledge is captured and carried forward between chapters in seaice-py — the knowledge/chNN.md format, CUMULATIVE.md, function_map.md updates, and progress.json. Load at the start of every chapter (to read) and at the end (to write).
---

# chapter-knowledge — make chapter N+1 smarter than chapter N

## Read at chapter start (in this order)
1. `knowledge/CUMULATIVE.md` — the running digest (concepts, primitives available in `seaice/core/`, pitfalls).
2. `knowledge/function_map.md` — verified MATLAB→Python mappings with parity labels from earlier chapters.
3. `knowledge/ch<N-1>.md` "Feeds forward" section.

## Write at chapter end: `knowledge/chNN.md`
```
# Chapter N — <title>   (ported <date>, commit <hash>)
## 1. Concepts (what the chapter teaches, in our words, ≤ 15 bullets, with equation numbers)
## 2. Algorithms implemented   (name → seaice function → book §/Eq → MATLAB file → parity label)
## 3. New reusable primitives added to seaice/core/  (signature + one line each)
## 4. MATLAB→Python lessons learned (only NEW pitfalls; each becomes a row in function_map.md)
## 5. Data used (file, source tier, size, what it shows) and data still missing
## 6. Results vs book (the numbers/figures table, summarised)
## 7. Feeds forward — what later chapters will need from this one (e.g. "Ch5 needs distance_transform(quasi) and chaincode.first_difference")
## 8. Open questions / things to revisit
```

## Update `knowledge/CUMULATIVE.md` (rewrite, don't append blindly; keep ≤ ~400 lines)
Sections: *Pipeline so far* (a diagram in text: Ch2 primitives → Ch3 ice mask → …) · *Available primitives* (table of
`seaice/core/` functions) · *Global pitfalls* (dedup) · *Data inventory* · *Parity summary per chapter*.

## Update `knowledge/function_map.md`
One row per MATLAB function actually used: `| MATLAB | Python we use | Parity | Verified in | Note |`. Never remove a
verified row; if a later chapter finds a better mapping, edit the row and note "superseded chNN".

## Optional: promote a lesson to a skill
If the same lesson would apply to any MATLAB→Python port (not just this book), also add it to
`.claude/skills/matlab-to-python/SKILL.md` §1 or `reference/function_map.md` — that is how the *skill itself* improves.

## progress.json
Set the phase key to `pass`/`done` with a one-line `notes`, set `current_chapter` to the next id, and commit:
`git add -A && git commit -m "chNN: knowledge — <title> complete"`.
