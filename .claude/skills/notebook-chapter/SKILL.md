---
name: notebook-chapter
description: Re-build and headlessly execute the Colab notebook for one chapter. Usage - /notebook-chapter 3
disable-model-invocation: true
---

# /notebook-chapter N
Same as phase 5 of `/do-chapter`: brief the `notebook-builder` agent with the colab-notebook skill, require a successful
`nbconvert --execute`, update `progress.json` (`notebook`), commit, and print the notebook path and runtime.
