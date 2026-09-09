---
name: seaice-book
description: Conventions and context for the seaice-py project (porting "Sea Ice Image Processing with MATLAB" to Python). Load whenever working on any chapter, MATLAB file, Python module, test, notebook, or knowledge file in this repo.
---

# seaice-book — project conventions

## The book in one paragraph
Zhang & Skjetne (2018) build a pipeline for optical (camera) sea-ice images: preliminaries (Ch2: image types, histograms,
pixel connectivity, distance transforms, convolution, set/morphological ops, chain codes, interpolation) → ice pixel
detection by thresholding (Otsu, local) and clustering (k-means) (Ch3) → ice edge detection by derivative (Sobel/Prewitt/
LoG/Canny) and morphological (erosion/dilation/gradient/reconstruction) methods (Ch4) → watershed floe segmentation with
distance-transform markers and neighboring-region merging via boundary/chain-code concavity analysis (Ch5) → GVF-snake
floe boundary refinement with automatic contour initialization (Ch6) → ice type identification (shape enhancement,
brash/slush/floe classification, general processing pipeline) (Ch7) → applications: shipborne camera ice concentration,
ice-field characterization for simulators, floe size distribution (Ch8) → model-basin (tank) ice: rectangular floes,
concentration from video, maximum floe size monitoring (Ch9) → Appendix A: orthorectification and radial lens distortion.

## File naming
- Chapter ids are two-digit: `ch02` … `ch10`. MATLAB folders are `ch2` … `ch10`.
- Python chapter modules: `seaice/ch02_preliminaries.py`, `seaice/ch03_ice_pixel_detection.py`, `seaice/ch04_edge_detection.py`,
  `seaice/ch05_watershed.py`, `seaice/ch06_gvf_snake.py`, `seaice/ch07_ice_type.py`, `seaice/ch08_applications.py`,
  `seaice/ch09_model_ice.py`, `seaice/ch10_calibration.py`.
- Reusable primitives live in `seaice/core/` (e.g. `core/matlab_compat.py`, `core/filters.py`, `core/morphology.py`,
  `core/distance.py`, `core/chaincode.py`, `core/interp.py`, `core/io.py`, `core/plotting.py`). Anything used by ≥2 chapters goes here.
- Scripts mirror MATLAB scripts: `MATLAB_ROOT/ch2/histogram.m` (MATLAB_ROOT = `K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab`) → `scripts/ch02_histogram.py`.
- Figures: `outputs/chNN/fig_<chapter>_<fig>_<slug>.png`, with the book figure number when reproducing a book figure.

## Docstring contract (every ported function)
```python
def distance_transform(bw: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Distance from each background pixel to the nearest ice (foreground) pixel.

    Book: §2.4, Eqs. (2.5)-(2.8), Fig. 2.12-2.14.  MATLAB source: MATLAB_ROOT/ch2/distance_transform.m (bwdist).
    Parity: exact for euclidean/cityblock/chessboard (vs Octave bwdist); quasi-euclidean re-implemented (see core/distance.py).
    """
```

## Chapter pipeline (owned by /do-chapter)
analyze → port → verify → review → notebook → knowledge. Each phase writes its artifact to disk and updates
`progress.json`. A phase that fails leaves a `blocked`/`fail` status with a note; nothing downstream runs.

## Parity vocabulary (use exactly these words in reports)
- `exact`   — bit/near-bit identical to MATLAB/Octave reference (binary/label images identical up to relabeling; floats ≤1e-6).
- `near`    — same algorithm, tiny numeric differences (e.g. float rounding, boundary handling) — ≤1e-3 relative or ≤0.5% pixels.
- `approx`  — a library routine with a *different* algorithm was used (e.g. skimage `canny` vs MATLAB `edge('canny')`); documented.
- `reimplemented` — no library equivalent; written from the book's equations and verified against Octave running the original `.m` code, or against synthetic cases with known answers.
- `unverified` — could not obtain a reference; only smoke-tested. Must be listed in the report's "Open items".

## What "done" means for a chapter
1. `analysis/chNN.md` exists with a complete .m ↔ .py map.
2. Every `.m` has a Python counterpart; `scripts/chNN_*.py` all run end-to-end without error and save figures.
3. `tests/test_chNN.py` passes; `reports/chNN_verification.md` has the parity table and figure comparisons.
4. `notebooks/chNN_*.ipynb` executes headlessly (`jupyter nbconvert --execute`) and has Colab setup cells.
5. `knowledge/chNN.md` written, `knowledge/CUMULATIVE.md` and `knowledge/function_map.md` updated, `progress.json` updated, git committed.
