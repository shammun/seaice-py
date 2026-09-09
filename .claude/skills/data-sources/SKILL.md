---
name: data-sources
description: Data policy for seaice-py — which images to use for each chapter (book-shipped, free online sea-ice imagery, synthetic model-ice, or manual download into Google Drive), how to fetch and cache them, and how notebooks access them on Colab. Load whenever a script or notebook needs input data.
---

# data-sources — where the images come from

## Tier order (stop at the first tier that satisfies the need)
**Tier 1 — Book-shipped.** Any image/video/`.mat` inside `MATLAB_ROOT/chN/` = `K30735_Sea Ice Image Processing with MATLAB_matlab codes/matlab/chN/` (e.g. `ch2/rgb.jpg`). In `/setup-project`
copy every non-`.m` file to `data/book/chNN/` (keep names). These reproduce the book figures exactly and are the
default for every chapter. They are for the user's own study of the book he owns; do not redistribute publicly.

**Tier 2 — Free online sea-ice imagery (no login).** Use for "try it on another image" cells and where a chapter's
own data is absent. Add a fetcher in `seaice/core/io.py`:
```python
def fetch(url: str, dest: Path, timeout: int = 60) -> Path:  # cached download with a clear error message
```
Candidate sources (verify each URL with a real request before relying on it; record working ones in `data/online/SOURCES.md`):
- NASA Worldview Snapshots API (MODIS/VIIRS true-colour, any date/bbox, JPEG/PNG, no login):
  `https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&CRS=EPSG:4326&TIME=2023-07-15&WRAP=DAY&BBOX=70,-160,72,-155&FORMAT=image/jpeg&WIDTH=1024&HEIGHT=1024`
  (Beaufort Sea marginal ice zone in summer gives good floe fields at 250 m).
- NASA Earth Observatory "Image of the Day" pages (public-domain JPEGs; search "sea ice floes" / "marginal ice zone").
- Wikimedia Commons public-domain aerial sea-ice photos (NOAA / USCG / NASA Operation IceBridge DMS photos).
- NSIDC / NASA GIBS WMTS tiles (`https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/...`) for programmatic access.
Prefer NASA/NOAA (public domain). Record licence in `SOURCES.md`.

**Tier 3 — Synthetic.** Needed mainly for Ch9 (model-basin rectangular floes) and for unit tests: generate with
`seaice/core/synth.py` (random rectangles/discs with rotation, overlap, noise, uneven illumination, JPEG-like blur),
seeded. Always state in the notebook that the image is synthetic.

**Tier 4 — Manual download (login required) → Google Drive.** E.g. NASA IceBridge DMS L1B images (Earthdata login),
Sentinel-2 L2A scenes (Copernicus Data Space login). Emit a *precise* instruction block:
```
MANUAL DATA NEEDED for chapter N:
  1. Go to <url>; log in (free account).
  2. Search: <exact parameters>; download <file pattern>.
  3. Save to Google Drive: MyDrive/seaice-py/data/manual/chNN/<filename>
  4. In Colab, mount Drive and this notebook will find it at /content/drive/MyDrive/seaice-py/data/manual/chNN/.
  Locally: put it in data/manual/chNN/ (git-ignored).
```
and make the code look in `data/manual/chNN/` with a helpful error.

## Locations & Colab access
- Local: `data/book/`, `data/online/`, `data/synthetic/`, `data/manual/` (last three git-ignored; `SOURCES.md` files committed).
- Colab via GitHub: the repo includes `data/book/` (small), online data is re-fetched by the notebook, synthetic is regenerated.
- Colab via Drive: `tools/sync_to_drive.ps1` mirrors `data/` to `G:\My Drive\seaice-py\data\`; the notebook mounts Drive.
- To *save* online data into Drive from Colab: `!wget -O /content/drive/MyDrive/seaice-py/data/online/<name>.jpg "<url>"` after mounting.

## Never
Fabricate an image's provenance, silently substitute a different image for a book figure, or embed base64 images in notebooks.

## Public-repo rule
The repo is public. Never commit book text, book-shipped images, PDF page crops, or executed notebooks that contain them. Published notebooks load data through seaice.core.io.load_image(), which prefers the reader's private Drive copy and falls back to public-domain imagery. Book-figure comparisons stay local in reports/**/figures/ (git-ignored).
