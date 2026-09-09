---
name: verify-chapter
description: Re-run only the verification phase for a chapter (tests, Octave references, figure comparison, report). Usage - /verify-chapter 3
disable-model-invocation: true
---

# /verify-chapter N
Same as phase 3 of `/do-chapter` (see that skill): brief the `port-verifier` agent, loop with `matlab-porter` on failures
(max 3), update `progress.json` (`verify`), commit, and print the Verdict + Open items.
