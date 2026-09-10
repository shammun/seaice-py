"""L3 evidence for chapter 5: side-by-side figure comparisons (Python | MATLAB | book page).

For every book figure of chapter 5 that the MATLAB code produces this writes ``outputs/ch05/verify/fig_X_Y_compare.png``
(copied to ``reports/ch05/figures/``, git-ignored) with

* top row    — the Python output PNGs from ``outputs/ch05/`` (produced by ``scripts/ch05_*.py --no-show``),
* bottom row — the images MATLAB wrote with ``imwrite`` / ``print`` while ``reference/ch05/make_refs.py`` ran the
  original scripts (``outputs/ch05/verify/matlab/``),
* right column — the book page rendered from ``chapters/ch05.pdf`` with pymupdf at 110 dpi.

Where both sides are *raw* images of the same size (not rendered figures) the pixel difference between the Python
PNG and the MATLAB PNG is measured and written to ``outputs/ch05/verify/image_diffs.json``.

Usage: .venv/Scripts/python.exe reference/ch05/make_compare_figures.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pymupdf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/ch05"
ML = OUT / "verify/matlab"
VERIFY = OUT / "verify"
REPORT_FIG = ROOT / "reports/ch05/figures"
PDF = ROOT / "chapters/ch05.pdf"
FIRST_PRINTED_PAGE = 83  # chapters/ch05.pdf page index 0 = printed page 83


def page_image(printed_page: int, dpi: int = 110) -> np.ndarray:
    doc = pymupdf.open(PDF)
    pix = doc[printed_page - FIRST_PRINTED_PAGE].get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img[..., :3]


def _read(p: Path) -> np.ndarray | None:
    """Read a PNG; 1-bit PNGs written by MATLAB ``imwrite(logical)`` come back as 0/1 and are scaled to 0/255."""
    if not p.exists():
        return None
    img = iio.imread(p)
    if img.dtype == np.bool_ or (img.dtype == np.uint8 and img.max() <= 1):
        img = img.astype(np.uint8) * 255
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    return img


def _thumb(img: np.ndarray, max_side: int = 900) -> np.ndarray:
    step = int(np.ceil(max(img.shape[:2]) / max_side))
    return img[::step, ::step] if step > 1 else img


def compose(name: str, py: list[tuple[str, Path]], ml: list[tuple[str, Path]], page: int, caption: str) -> Path:
    ncol = max(len(py), len(ml), 1)
    nrow = 2 if ml else 1
    fig = plt.figure(figsize=(3.2 * ncol + 4.5, 3.0 * nrow + 0.9))
    gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[1] * ncol + [1.4])
    rows = [("Python (seaice)", py)] + ([("MATLAB R2025a (original .m)", ml)] if ml else [])
    for r, (row_label, items) in enumerate(rows):
        for c in range(ncol):
            ax = fig.add_subplot(gs[r, c])
            ax.axis("off")
            if c < len(items):
                title, path = items[c]
                img = _read(path)
                if img is None:
                    ax.text(0.5, 0.5, f"missing:\n{path.name}", ha="center", va="center", fontsize=8)
                else:
                    img = _thumb(img)
                    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0 if img.ndim == 2 else None,
                              vmax=255 if img.ndim == 2 else None, interpolation="nearest")
                ax.set_title(f"{row_label}\n{title}" if c == 0 else title, fontsize=8)
    ax = fig.add_subplot(gs[:, ncol])
    ax.imshow(page_image(page))
    ax.axis("off")
    ax.set_title(f"book p. {page}", fontsize=9)
    fig.suptitle(caption, fontsize=10)
    fig.tight_layout()
    dest = VERIFY / f"{name}_compare.png"
    fig.savefig(dest, dpi=110)
    plt.close(fig)
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    shutil.copy(dest, REPORT_FIG / dest.name)
    return dest


def image_diff(py_path: Path, ml_path: Path) -> dict | None:
    a, b = _read(py_path), _read(ml_path)
    if a is None or b is None:
        return {"python": py_path.name, "matlab": ml_path.name, "comparable": False, "missing": True}
    if a.ndim == 3 and b.ndim == 2:
        a = a[..., 0] if np.array_equal(a[..., 0], a[..., 1]) else a
    if b.ndim == 3 and a.ndim == 2:
        b = b[..., 0] if np.array_equal(b[..., 0], b[..., 1]) else b
    if a.shape != b.shape:
        return {"python": py_path.name, "matlab": ml_path.name, "shape_python": list(a.shape),
                "shape_matlab": list(b.shape), "comparable": False}
    d = np.abs(a.astype(int) - b.astype(int))
    return {"python": py_path.name, "matlab": ml_path.name, "shape": list(a.shape), "comparable": True,
            "max_abs_diff": int(d.max()), "n_diff": int((d > 0).sum()), "frac_diff": float((d > 0).mean())}


def main() -> int:
    VERIFY.mkdir(parents=True, exist_ok=True)
    figs = [
        ("fig_5_01", [("(b) imcomplement(I)", OUT / "fig_5_01b_complement.png"),
                      ("(c) surf(imcomplement(I)) copper", OUT / "fig_5_01c_surface_complement.png")],
         [("x = imcomplement(I)", ML / "topo_complement.png"), ("surf(double(x)) texturemap", ML / "topo_fig1.png")], 85,
         "Fig. 5.1 - complemented gray image and its topographic surface (topological_surface.m)"),
        ("fig_5_05", [("(a) Sobel |g|", OUT / "fig_5_05a_sobel_gradient.png"), ("(b) surf(g)", OUT / "fig_5_05b_surface_gradient.png"),
                      ("(c) watershed(g) labels", OUT / "fig_5_05c_watershed_labels.png"), ("(d) ridges on I", OUT / "fig_5_05d_overlay.png")],
         [("imshow(g, [])", ML / "grad_a_g.png"), ("surf(double(g))", ML / "topo_fig2.png"), ("imshow(l, [])", ML / "grad_c_l.png"),
          ("f(wr) = 0", ML / "grad_d_f.png")], 90,
         "Fig. 5.5 - watershed of the Sobel gradient magnitude (gradients_watershed.m / topological_surface.m)"),
        ("fig_5_06", [("(a) close-opening 7x7", OUT / "fig_5_06a_smoothed_gradient.png"), ("(b) watershed(g2)", OUT / "fig_5_06b_watershed_labels.png"),
                      ("(c) ridges on I", OUT / "fig_5_06c_overlay.png")],
         [("imshow(g2, [])", ML / "grad_6a_g2.png"), ("imshow(l2, [])", ML / "grad_6b_l2.png"), ("f2(wr2) = 0", ML / "grad_6c_f2.png")], 91,
         "Fig. 5.6 - watershed of the smoothed gradient (imclose(imopen(g, ones(7)), ones(7)))"),
        ("fig_5_08", [("(a) im2bw", OUT / "fig_5_08a_binary.png"), ("(b) -bwdist chessboard", OUT / "fig_5_08b_inverse_chessboard_dt.png"),
                      ("(c) surface", OUT / "fig_5_08c_surface_chessboard_dt.png"), ("(d) minima in black", OUT / "fig_5_08d_regional_minima.png"),
                      ("(e) watershed line", OUT / "fig_5_08e_watershed_line.png"), ("(f) segmented", OUT / "fig_5_08f_segmented.png")],
         [("img", ML / "dw_chess_a_binary.png"), ("imshow(imgDist, [])", ML / "dw_chess_b_imgDist.png"), ("surf(double(d))", ML / "topo_fig3.png"),
          ("bw0", ML / "dw_chess_d_minima_overlay.png"), ("bgm", ML / "dw_chess_e_ridge.png"), ("bwareaopen(img, 5)", ML / "dw_chess_f_segmented.png")], 92,
         "Fig. 5.8 - chessboard distance-transform watershed (distance_watershed.m)"),
        ("fig_5_09a_euclidean", [("minima (black)", OUT / "fig_5_09_euclidean_regional_minima.png"), ("watershed line", OUT / "fig_5_09_euclidean_watershed_line.png"),
                                 ("segmented", OUT / "fig_5_09_euclidean_segmented.png"), ("-bwdist", OUT / "fig_5_09_euclidean_inverse_dt.png")],
         [("bw0", ML / "dw_euc_d_minima_overlay.png"), ("bgm", ML / "dw_euc_e_ridge.png"), ("img", ML / "dw_euc_f_segmented.png"),
          ("imshow(imgDist, [])", ML / "dw_euc_b_imgDist.png")], 93,
         "Fig. 5.9 (Euclidean row) - distance_watershed.m with 'euclidean'"),
        ("fig_5_09b_cityblock", [("minima (black)", OUT / "fig_5_09_cityblock_regional_minima.png"), ("watershed line", OUT / "fig_5_09_cityblock_watershed_line.png"),
                                 ("segmented", OUT / "fig_5_09_cityblock_segmented.png"), ("-bwdist", OUT / "fig_5_09_cityblock_inverse_dt.png")],
         [("bw0", ML / "dw_city_d_minima_overlay.png"), ("bgm", ML / "dw_city_e_ridge.png"), ("img", ML / "dw_city_f_segmented.png"),
          ("imshow(imgDist, [])", ML / "dw_city_b_imgDist.png")], 93,
         "Fig. 5.9 (city-block row) - distance_watershed.m with 'cityblock'"),
        ("fig_5_10", [("(a) contours euc | city | chess", OUT / "fig_5_10a_contours.png"), ("(b) zoom cols 24-40, rows 44-64", OUT / "fig_5_10b_zoom_cols24_40_rows44_64.png"),
                      ("(c) zoom cols 18-30, rows 45-90", OUT / "fig_5_10c_zoom_cols18_30_rows45_90.png")],
         [("imcontour(-bwdist cityblock)", ML / "dprop_q_fig1.png"), ("imcontour(-bwdist euclidean)", ML / "dprop_q_fig2.png"),
          ("imcontour(-bwdist chessboard)", ML / "dprop_q_fig3.png")], 94,
         "Fig. 5.10 - iso-distance contours of the three transforms of Fig. 5.8(a) (distance_propagation.m, commented q.jpg lines)"),
        ("sec_5_1_2_point", [("-bwdist of a point (cityblock)", OUT / "sec_5_1_2_point_inverse_distance_cityblock.png"),
                             ("image(dist) + imcontour", OUT / "sec_5_1_2_point_propagation_cityblock.png")],
         [("imshow(imgDist, [])", ML / "dprop_imgDist.png"), ("image(dist,'scaled'); imcontour", ML / "dprop_fig3.png")], 93,
         "distance_propagation.m (active code): 201x201 single point, city-block propagation (diamond)"),
        ("fig_5_12", [("(a) minima", OUT / "fig_5_12a_regional_minima.png"), ("(b) marker = dilate r=5", OUT / "fig_5_12b_marker.png"),
                      ("(c) imimposemin", OUT / "fig_5_12c_imposed_distance.png"), ("(d) watershed line", OUT / "fig_5_12d_watershed_line.png"),
                      ("(e) segmented", OUT / "fig_5_12e_segmented.png")],
         [("~Dis_img", ML / "marker_a_minima.png"), ("marker", ML / "marker_b_marker.png"), ("imshow(imgDist, [])", ML / "marker_c_imposed.png"),
          ("bgm", ML / "marker_d_ridge.png"), ("bwareaopen(img, 5)", ML / "marker_e_segmented.png")], 97,
         "Fig. 5.12 - marker-controlled watershed (marker_watershed.m)"),
        ("fig_5_14", [("(b) binary", OUT / "fig_5_14b_binary.png"), ("(c) -D cityblock", OUT / "fig_5_14c_inverse_cityblock_dt.png"),
                      ("(d) watershed lines", OUT / "fig_5_14d_watershed_lines.png"), ("(e) over-segmented", OUT / "fig_5_14e_oversegmented.png"),
                      ("(f) junction lines + endpoints", OUT / "fig_5_14f_junction_lines_endpoints.png"), ("(i) final", OUT / "fig_5_14i_final.png")],
         [("bw", ML / "main_b_binary.png"), ("-D", ML / "main_c_negD.png"), ("w = L == 0", ML / "main_d_ridges.png"),
          ("seg0", ML / "main_e_seg0.png"), ("figure f + plot(y, x, 'r.')", ML / "main_fig2.png"), ("seg", ML / "main_i_seg.png")], 100,
         "Fig. 5.14 - watershed + neighbouring-region merging (watershed_based/main.m)"),
        ("fig_5_14f_authors_fig", [("(f) junction lines + endpoints", OUT / "fig_5_14f_junction_lines_endpoints.png"),
                                   ("f = bw & w", OUT / "fig_5_14f_junction_lines.png")],
         [("authors' nrm_junction_ending.fig", ML / "fig_5_14f_authors.png"), ("its image CData", ML / "fig_5_14f_cdata.png")], 100,
         "Fig. 5.14(f) - the authors' shipped .fig file (openfig) vs the port"),
        ("fig_5_17", [("concave points (red)", OUT / "fig_5_17_concave_points.png"), ("bound2im(b)", OUT / "sec_5_2_1_boundary_image.png")],
         [("figure bim + plot(concave, 'r.')", ML / "cc_fig1.png"), ("bim", ML / "cc_bim.png")], 104,
         "Fig. 5.17 - concave detection by the differential chain code (chaincode_corner.m)"),
        ("sec_5_1_direct", [("ridges of watershed(gray)", OUT / "sec_5_1_direct_watershed_ridges.png"), ("overlay", OUT / "sec_5_1_direct_watershed_overlay.png"),
                            ("label2rgb (colours differ)", OUT / "sec_5_1_direct_watershed_labels.png")],
         [("bgm", ML / "direct_bgm.png"), ("im(bgm) = 0", ML / "direct_overlay.png"), ("label2rgb(...,'shuffle')", ML / "direct_color.png")], 89,
         "direct_watershed.m (no book figure): watershed of the gray image itself"),
    ]
    diffs = []
    raw_pairs = [
        ("fig_5_01b_complement.png", "topo_complement.png"), ("fig_5_05a_sobel_gradient.png", "grad_a_g.png"),
        ("fig_5_05c_watershed_labels.png", "grad_c_l.png"), ("fig_5_05d_overlay.png", "grad_d_f.png"),
        ("sec_5_1_1_ridges.png", "grad_wr.png"), ("fig_5_06a_smoothed_gradient.png", "grad_6a_g2.png"),
        ("fig_5_06b_watershed_labels.png", "grad_6b_l2.png"), ("fig_5_06c_overlay.png", "grad_6c_f2.png"),
        ("sec_5_1_1_ridges_smoothed.png", "grad_wr2.png"),
        ("fig_5_08a_binary.png", "dw_chess_a_binary.png"), ("fig_5_08b_inverse_chessboard_dt.png", "dw_chess_b_imgDist.png"),
        ("fig_5_08_minima_mask.png", "dw_chess_minima_mask.png"), ("fig_5_08d_regional_minima.png", "dw_chess_d_minima_overlay.png"),
        ("fig_5_08e_watershed_line.png", "dw_chess_e_ridge.png"), ("fig_5_08f_segmented.png", "dw_chess_f_segmented.png"),
        ("fig_5_09_euclidean_binary.png", "dw_euc_a_binary.png"), ("fig_5_09_euclidean_inverse_dt.png", "dw_euc_b_imgDist.png"),
        ("fig_5_09_euclidean_regional_minima.png", "dw_euc_d_minima_overlay.png"), ("fig_5_09_euclidean_watershed_line.png", "dw_euc_e_ridge.png"),
        ("fig_5_09_euclidean_segmented.png", "dw_euc_f_segmented.png"),
        ("fig_5_09_cityblock_inverse_dt.png", "dw_city_b_imgDist.png"), ("fig_5_09_cityblock_regional_minima.png", "dw_city_d_minima_overlay.png"),
        ("fig_5_09_cityblock_watershed_line.png", "dw_city_e_ridge.png"), ("fig_5_09_cityblock_segmented.png", "dw_city_f_segmented.png"),
        ("sec_5_1_2_quasi_euclidean_inverse_dt.png", "dw_quasi_b_imgDist.png"), ("sec_5_1_2_quasi_euclidean_regional_minima.png", "dw_quasi_d_minima_overlay.png"),
        ("sec_5_1_2_quasi_euclidean_watershed_line.png", "dw_quasi_e_ridge.png"), ("sec_5_1_2_quasi_euclidean_segmented.png", "dw_quasi_f_segmented.png"),
        ("sec_5_1_2_point_inverse_distance_cityblock.png", "dprop_imgDist.png"),
        ("sec_5_1_2_q_inverse_distance_cityblock.png", "dw_city_b_imgDist.png"), ("sec_5_1_2_q_inverse_distance_euclidean.png", "dw_euc_b_imgDist.png"),
        ("sec_5_1_2_q_inverse_distance_chessboard.png", "dw_chess_b_imgDist.png"),
        ("sec_5_1_3_inverse_cityblock_dt.png", "marker_imgDist0.png"), ("fig_5_12a_regional_minima.png", "marker_a_minima.png"),
        ("fig_5_12b_marker.png", "marker_b_marker.png"), ("fig_5_12c_imposed_distance.png", "marker_c_imposed.png"),
        ("sec_5_1_3_marker_overlay.png", "marker_marker_overlay.png"), ("fig_5_12d_watershed_line.png", "marker_d_ridge.png"),
        ("fig_5_12e_segmented.png", "marker_e_segmented.png"),
        ("sec_5_1_3_cityblock_r5_centroid_marker0.png", "marker_cen_marker0.png"),
        ("sec_5_1_3_cityblock_r5_centroid_d_watershed_line.png", "marker_cen_d_ridge.png"),
        ("sec_5_1_3_cityblock_r5_centroid_e_segmented.png", "marker_cen_e_segmented.png"),
        ("fig_5_14b_binary.png", "main_b_binary.png"), ("fig_5_14c_inverse_cityblock_dt.png", "main_c_negD.png"),
        ("fig_5_14d_watershed_lines.png", "main_d_ridges.png"), ("fig_5_14e_oversegmented.png", "main_e_seg0.png"),
        ("fig_5_14f_junction_lines.png", "main_f_lines.png"), ("fig_5_14i_final.png", "main_i_seg.png"),
        ("fig_5_14f_junction_lines.png", "fig_5_14f_cdata.png"),
        ("sec_5_2_1_otsu_mask.png", "cc_B.png"), ("sec_5_2_1_boundary_image.png", "cc_bim.png"),
        ("sec_5_1_direct_watershed_ridges.png", "direct_bgm.png"), ("sec_5_1_direct_watershed_overlay.png", "direct_overlay.png"),
        ("fig_5_01a_gray.png", "topo_gray.png"), ("sec_5_1_gray_otsu_mask.png", "topo_bw.png"),
    ]
    for name, py, ml, page, caption in figs:
        dest = compose(name, py, ml, page, caption)
        print("wrote", dest.name)
    for a, b in raw_pairs:
        d = image_diff(OUT / a, ML / b)
        if d is not None:
            diffs.append(d)
    (VERIFY / "image_diffs.json").write_text(json.dumps(diffs, indent=2))
    comparable = [d for d in diffs if d.get("comparable")]
    ident = sum(1 for d in comparable if d["n_diff"] == 0)
    print(f"raw image pairs: {len(diffs)} listed, {len(comparable)} comparable, {ident} identical")
    for d in diffs:
        if not d.get("comparable") or d["n_diff"]:
            print("  ", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
