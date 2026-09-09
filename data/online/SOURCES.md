# Online data sources for seaice-py

## Book supplementary material — none published
Checked 2026-09-09 (/setup-project): the front matter (PDF pages 1–32), Appendix B (`chapters/ch11.txt`) and all chapter
texts contain **no URL to supplementary code or data**. The only links are the publisher's (crcpress.com,
taylorandfrancis.com, copyright.com) and www.mathworks.com. The MATLAB code archive (`K30735_…_matlab codes.zip`) shipped
with the book is the complete authors' code; the images it contains are copied to `data/book/chNN/`.

## Known gaps in the shipped data (fill per the data-sources policy: online → synthetic → manual)
| chapter | file referenced by the .m code | shipped? | note |
|---|---|---|---|
| ch9 | `dypic_05100_cam1_top.avi` (movie_kmeans.m, movie_otsu.m) | no | model-ice basin video; needs synthetic/manual substitute |
| ch8 | source image behind `MCD/IceImage_290915_2_jpg.0000179.mat` | .mat only | the processed ice-image struct is shipped, the raw JPEG is not |

## Downloaded datasets
(none yet — add one row per download: URL, licence, date, local path)

## Public-domain substitutes for book images (registry: `seaice/core/public_images.py`)
Used automatically by `seaice.core.io.load_image()` when the reader has no private copy of the book image.
Downloaded on demand into `data/online/<chapter>/` (git-ignored); only this file and the registry are committed.

| chapter | book image | substitute file | URL | credit | licence | verified |
|---|---|---|---|---|---|---|
| ch02 | `rgb.JPG` | `nasa_modis_terra_beaufort_miz_2019-07-25.jpg` (2048×1536 RGB JPEG, ~495 kB) | `https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&LAYERS=MODIS_Terra_CorrectedReflectance_TrueColor&CRS=EPSG:4326&FORMAT=image/jpeg&WIDTH=2048&HEIGHT=1536&WRAP=DAY&TIME=2019-07-25&BBOX=71.5,-142.0,74.0,-132.0` | NASA Worldview Snapshots, MODIS/Terra corrected reflectance (true colour), Beaufort Sea marginal ice zone, 25 July 2019 | NASA imagery is in the public domain (NASA Earthdata data and information policy) | 2026-09-09, HTTP 200 image/jpeg, 494 846 bytes; floes and open water clearly visible |
