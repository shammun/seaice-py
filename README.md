# seaice-py — *Sea Ice Image Processing with MATLAB*, ported to Python

Every MATLAB script of Sea Ice Image Processing with MATLAB (Qin Zhang & Roger Skjetne, CRC Press 2018) converted to verified, documented Python — one chapter at a time, each chapter building on the previous ones. Each chapter ships a Python module, tests that compare against MATLAB R2025a reference outputs, a verification report, and a teaching notebook you can read here or open in Google Colab.

Read the chapters online at **https://shammun.github.io/seaice-py/** or open any notebook in Google Colab from the table below.

## Chapters

<!-- INDEX_TABLE_START -->
| Chapter | HTML | Notebook | Colab |
|---|---|---|---|
| Ch. 2: Digital Image Processing Preliminaries | [View](https://shammun.github.io/seaice-py/notebooks/ch02_preliminaries.html) | [.ipynb](notebooks/ch02_preliminaries.ipynb) | [Open](https://colab.research.google.com/github/shammun/seaice-py/blob/main/notebooks/ch02_preliminaries_colab.ipynb) |
| Ch. 3: Ice Pixel Detection | [View](https://shammun.github.io/seaice-py/notebooks/ch03_ice_pixel_detection.html) | [.ipynb](notebooks/ch03_ice_pixel_detection.ipynb) | [Open](https://colab.research.google.com/github/shammun/seaice-py/blob/main/notebooks/ch03_ice_pixel_detection_colab.ipynb) |
| Ch. 4: Ice Edge Detection | [View](https://shammun.github.io/seaice-py/notebooks/ch04_ice_edge_detection.html) | [.ipynb](notebooks/ch04_ice_edge_detection.ipynb) | [Open](https://colab.research.google.com/github/shammun/seaice-py/blob/main/notebooks/ch04_ice_edge_detection_colab.ipynb) |
| Ch. 5: Watershed-Based Ice Floe Segmentation | [View](https://shammun.github.io/seaice-py/notebooks/ch05_watershed_segmentation.html) | [.ipynb](notebooks/ch05_watershed_segmentation.ipynb) | [Open](https://colab.research.google.com/github/shammun/seaice-py/blob/main/notebooks/ch05_watershed_segmentation_colab.ipynb) |
| Ch. 6: GVF Snake-Based Ice Floe Boundary Identification and Ice Image Segmentation | [View](https://shammun.github.io/seaice-py/notebooks/ch06_gvf_snake.html) | [.ipynb](notebooks/ch06_gvf_snake.ipynb) | [Open](https://colab.research.google.com/github/shammun/seaice-py/blob/main/notebooks/ch06_gvf_snake_colab.ipynb) |
| Ch. 7: Sea Ice Type Identification | [View](https://shammun.github.io/seaice-py/notebooks/ch07_ice_type_identification.html) | [.ipynb](notebooks/ch07_ice_type_identification.ipynb) | [Open](https://colab.research.google.com/github/shammun/seaice-py/blob/main/notebooks/ch07_ice_type_identification_colab.ipynb) |
| Ch. 8: Sea Ice Image Processing Applications | [View](https://shammun.github.io/seaice-py/notebooks/ch08_image_processing_applications.html) | [.ipynb](notebooks/ch08_image_processing_applications.ipynb) | [Open](https://colab.research.google.com/github/shammun/seaice-py/blob/main/notebooks/ch08_image_processing_applications_colab.ipynb) |
| Ch. 9: Model Sea Ice Image Processing Applications | [View](https://shammun.github.io/seaice-py/notebooks/ch09_model_sea_ice_applications.html) | [.ipynb](notebooks/ch09_model_sea_ice_applications.ipynb) | [Open](https://colab.research.google.com/github/shammun/seaice-py/blob/main/notebooks/ch09_model_sea_ice_applications_colab.ipynb) |
| App. A: Appendix A — Geometric Calibration (orthorectification, lens distortion) | — (pending) | — (pending) | — (pending) |
<!-- INDEX_TABLE_END -->

## Run locally

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows;  .venv/bin/pip on Linux/macOS
.venv/Scripts/python -m pytest -q                     # parity tests against the saved MATLAB references
.venv/Scripts/python scripts/ch02_histogram.py       # any chapter script; figures go to outputs/chNN/
jupyter notebook notebooks/ch02_preliminaries.ipynb  # the teaching notebook (runs locally without Colab)
```

## Layout

| Path | What |
|---|---|
| `seaice/core/` | reusable primitives (histogram, distance transforms, chain codes, interpolation, …) |
| `seaice/chNN_*.py`, `scripts/chNN_*.py` | chapter modules and one script per original MATLAB demo |
| `tests/test_chNN.py`, `reference/chNN/` | pytest parity tests and the MATLAB R2025a reference generator |
| `reports/chNN_verification.md` | parity table, figure comparisons and verdict per chapter |
| `notebooks/chNN_*.ipynb` (+ `_colab`, `.html`) | teaching notebooks; `tools/publish_notebook.py` builds the Colab and HTML versions |
| `knowledge/` | what each chapter learned and what later chapters reuse |

## About the source material

This port follows Qin Zhang and Roger Skjetne, *Sea Ice Image Processing with MATLAB* (CRC Press, 2018). The book
text, its figures and its image data are **not** included here and must be obtained from the publisher; the
authors' MATLAB code archive is not redistributed either. Every Python function cites the book section, equation and
source `.m` file it implements so you can follow along with your own copy. The notebooks run on public-domain NASA
imagery unless you supply the book images privately (put them in `data/book/chNN/` locally, or in
`MyDrive/seaice-py/data/book/chNN/` for Colab). The `.mat` reference files used by the parity tests are regenerated
with `reference/chNN/make_refs.py` on a machine with MATLAB.

## License

The Python code, tests and documentation are released under the MIT License (see `LICENSE`). The license does not
cover the book or the authors' MATLAB code.
