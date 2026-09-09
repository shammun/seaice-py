---
name: status
description: Print seaice-py progress (per chapter, per phase), open items from the latest verification reports, environment status, and the exact next command to run. Usage - /status
disable-model-invocation: true
---

# /status
1. Read `progress.json`; print a table `chapter | analyze | port | verify | review | notebook | knowledge | notes`.
2. For the current chapter, print the "Open items" section of `reports/chNN_verification.md` if it exists.
3. Print environment (python/octave/matlab presence, checked_on).
4. Print `git log --oneline -5`.
5. Print the next command: `/setup-project` if never run, `/do-chapter N --from <first non-pass phase>` otherwise, or
   "All chapters done" — and remind: run `/clear` before starting a new chapter to keep context small.
