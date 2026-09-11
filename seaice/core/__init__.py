"""Reusable primitives shared by all chapters (see ``knowledge/function_map.md`` for the MATLAB ↔ Python map).

Chapter 2 primitives:

* :mod:`~seaice.core.io` — data paths, private-Drive / public-fallback loader (``load_image``, ``read_image``, ``data_roots``, ``output_dir``, ``fetch``)
* :mod:`~seaice.core.matlab_compat` — ``rgb2gray_matlab``, ``imcomplement``, ``matlab_round``, ``im2double``, ``im2uint8``
* :mod:`~seaice.core.plotting` — MATLAB-style ``imshow`` scaling and PNG saving
* :mod:`~seaice.core.color` — ``rgb2cmy``, ``rgb2cmyk``, ``rgb2hsi``, ``indexed_to_rgb`` (§2.1)
* :mod:`~seaice.core.histogram` — ``imhist``, ``normalized_histogram`` (§2.2)
* :mod:`~seaice.core.connectivity` — neighbourhoods, adjacency, ``label_components`` (= ``bwlabel``),
  ``bwareaopen`` (§2.3; ``bwareaopen`` first used in ch4 ``derivative.m``)
* :mod:`~seaice.core.distance` — ``distance_transform``, ``bwdist``, ``pixel_distance`` (§2.4)
* :mod:`~seaice.core.filters` — ``conv2``, ``imfilter`` (§2.5)
* :mod:`~seaice.core.setops` — set / logical operations, ``reflect``, ``translate`` (§2.6)
* :mod:`~seaice.core.chaincode` — ``boundaries``, ``fchcode``, ``bound2im`` (§2.7)
* :mod:`~seaice.core.interp` — nearest / bilinear / bicubic interpolation, ``warp_image``, ``resize`` (§2.8)
* :mod:`~seaice.core.synth` — synthetic inputs and transcribed book fixtures

Chapter 3 primitives:

* :mod:`~seaice.core.threshold` — ``graythresh``, ``im2bw``, ``multithresh``, ``imquantize``, ``block_otsu``,
  ``otsu_criterion``, ``separability``, ``ice_concentration``, ``class_coverage`` (§3.1)
* :mod:`~seaice.core.clustering` — ``kmeans_gray`` (the authors' ``kmeans.m``), ``kmeans_lloyd``,
  ``pairwise_distance`` (§3.2)

Chapter 4 primitives:

* :mod:`~seaice.core.filters` — ``fspecial`` (MATLAB kernels: sobel, prewitt, laplacian, gaussian, log, average, disk)
* :mod:`~seaice.core.edges` — ``edge`` (MATLAB ``edge`` for sobel/prewitt/roberts/log/zerocross, with thinning),
  ``thin_gradient``, ``log_zero_crossings`` (§4.1)
* :mod:`~seaice.core.morphology` — ``strel`` (incl. MATLAB's approximate disk), ``imerode``, ``imdilate``, ``imopen``,
  ``imclose``, ``imreconstruct``, ``reconstruct_by_erosion``, ``geodesic_dilation/erosion``, ``reconstruct_iterative``,
  ``morphological_gradient`` (§4.2)

Chapter 5 primitives:

* :mod:`~seaice.core.watershed` — ``watershed`` (MATLAB Meyer flooding, line-by-line port of ``eml/watershed.m``),
  ``watershed_skimage`` (cross-check only)
* :mod:`~seaice.core.morphology` — ``imregionalmin``, ``imregionalmax``, ``imimposemin`` (§5.1, §5.1.3)
* :mod:`~seaice.core.plotting` — ``label2rgb``, ``surface_plot``, ``contour_overlay`` (display only)

Chapter 6 primitives:

* :mod:`~seaice.core.snake` — the Xu & Prince GVF snake toolbox: ``gvf``, ``snakedeform``, ``snakeinterp``,
  ``snakeindex``, ``snake_matrix``, ``bound_mirror_*``, ``gradient2``, ``xconv2``, ``gaussian_mask/blur``
  (§6.1.2, §6.2)
* :mod:`~seaice.core.regionprops` — MATLAB's ``regionprops`` algorithms (Area, Centroid, BoundingBox,
  ConvexHull/Image/Area, Solidity, Major/MinorAxisLength, Eccentricity, Orientation, Perimeter)
* :mod:`~seaice.core.polygon` — ``clip_polygon_rect`` (for ``polybool``), ``poly2mask``/``roipoly``,
  ``polygeom``, ``minboundrect``, ``polyxpoly``, ``convhull``, ``polyarea``
* :mod:`~seaice.core.matlab_compat` — ``del2`` (MATLAB's Laplacian/4, used by ``GVF.m``)
* :mod:`~seaice.core.connectivity` — ``bwperim``; :mod:`~seaice.core.morphology` —
  ``regional_maxima_by_reconstruction`` (Eqs. 6.57/6.58); :mod:`~seaice.core.filters` —
  ``homomorphic_butterworth`` (``homofil.m``, an orphan utility); :mod:`~seaice.core.plotting` —
  ``snake_plot``, ``quiver_field``; :mod:`~seaice.core.synth` — ``u_shape``, ``fig_6_16_circles``,
  ``FIG_6_14_IMAGE/_DISTANCE``, ``synthetic_floe_field``

Chapter 7 primitives:

* :mod:`~seaice.core.morphology` — ``imfill`` (MATLAB's binary **and** grayscale hole filling, Eqs. 7.3/7.4)
* :mod:`~seaice.core.histogram` — ``hist`` (MATLAB's bin-**centre** histogram, used by the §7.2.4 FSD and ch8)
* :mod:`~seaice.core.plotting` — ``size_colorbar`` (the Eq. 7.6 colour bar); ``label2rgb`` extended to the
  ``label2rgb(index, @jet, [1 1 1])`` form of ``ice_shape_enhancement.m``
* :mod:`~seaice.core.synth` — ``FIG_7_2_*``, ``FIG_7_3_*``, ``FIG_7_4_STEPS``, ``FIG_7_5_*``, ``FIG_7_6_*``,
  ``FIG_7_7_STEPS``/``_BOOK``/``FIG_7_7_X8_TYPO``, ``FIG_7_8_*`` (the printed matrices of §7.1)

Chapter 9 primitives:

* :mod:`~seaice.core.video` — ``read_video`` / ``write_video`` / ``video_num_frames`` in MATLAB's
  ``(H, W, 3, N)`` layout (``VideoReader``/``read``/``movie2avi``); Uncompressed AVI for every parity fixture
* :mod:`~seaice.core.polygon` — ``clip_polygon_convex`` (``polybool('intersection', ...)`` for two convex rings;
  ``clip_polygon_rect`` is now a special case of it)
* :mod:`~seaice.core.synth` — ``model_ice_tank``, ``model_ice_tank_video``, ``segmented_floe_video`` (the
  Tier-3 stand-ins for the three unpublished HSVA/DYPIC inputs)
* :mod:`~seaice.core.threshold` — ``block_otsu(..., compare='gt'|'ge')``; :mod:`~seaice.core.regionprops` —
  the ``'basic'`` property shorthand
"""
from __future__ import annotations

from . import chaincode, clustering, color, connectivity, distance, edges, filters, histogram, interp, io, \
    matlab_compat, morphology, plotting, polygon, regionprops as regionprops_module, setops, snake, synth, \
    threshold, watershed
from .chaincode import ChainCode, bound2im, boundaries, code_reverse, fchcode, first_difference, min_magnitude, \
    normalized_first_difference
from .clustering import KMeansGray, KMeansResult, kmeans_gray, kmeans_lloyd, objective_J, pairwise_distance
from .color import indexed_to_rgb, rgb2cmy, rgb2cmyk, rgb2hsi, split_rgb
from .connectivity import bwareaopen, bwperim, count_components, find_paths, is_adjacent, is_m_adjacent, \
    label_components, n4, n8, nd, region_boundary_mask
from .distance import bwdist, center_distance_map, distance_transform, pixel_distance, quasi_euclidean_dt
from .edges import EdgeResult, edge, gradient_roberts, gradient_sobel_prewitt, log_zero_crossings, thin_gradient
from .filters import conv2, conv_at, fspecial, homomorphic_butterworth, imfilter
from .histogram import hist, imhist, normalized_histogram
from .morphology import disk_decomposition, geodesic_dilation, geodesic_erosion, imclose, imdilate, imerode, \
    imfill, imimposemin, imopen, imreconstruct, imregionalmax, imregionalmin, intline, line_strel, minkowski_sum, \
    morphological_gradient, periodic_line, reconstruct_by_erosion, reconstruct_iterative, \
    regional_maxima_by_reconstruction, se_origin, strel
from .polygon import (clip_polygon_convex, clip_polygon_rect, convhull, minboundrect, poly2mask, polyarea,
                      polygeom, polyxpoly, roipoly)
from .regionprops import BASIC_PROPERTIES, RegionProps, region_table, regionprops
from .snake import CIRCULANT_MIN_N, bound_mirror_ensure, bound_mirror_expand, bound_mirror_shrink, gaussian_blur, \
    gaussian_mask, gradient2, gradient2_complex, gradient2_magnitude, gvf, snake_first_column, snake_matrix, \
    snakedeform, snakeindex, snakeinterp, xconv2
from .video import (VideoInfo, frames_to_imageio, frames_to_matlab, read_video, video_info,
                    video_num_frames, write_video)
from .watershed import watershed as watershed_transform, watershed_skimage
from .interp import interp2, interp_bicubic, interp_bilinear, interp_nearest, keys_kernel, resize, warp_image
from .io import REPO_ROOT, data_roots, load_image, output_dir, read_image, repo_root
from .matlab_compat import (del2, im2double, im2uint8, imcomplement, matlab_round, num2str,
                            rgb2gray_matlab)
from .plotting import finish_figure, imshow_matlab, imshow_scale, label2rgb, quiver_field, save_image, \
    show_matrix, size_colorbar, snake_plot, to_display_uint8
from .threshold import BlockOtsu, OtsuCurves, block_otsu, class_coverage, class_mean_intensity, graythresh, \
    ice_concentration, im2bw, imquantize, multithresh, otsu_criterion, otsuthresh, separability
from .setops import complement, difference, gray_complement, gray_intersection, gray_union, intersection, reflect, \
    translate, union

__all__ = [
    "chaincode", "clustering", "color", "connectivity", "distance", "edges", "filters", "histogram", "interp", "io",
    "matlab_compat", "morphology", "plotting", "polygon", "setops", "snake", "synth", "threshold", "video",
    "watershed",
    "VideoInfo", "read_video", "write_video", "video_info", "video_num_frames", "frames_to_matlab",
    "frames_to_imageio", "clip_polygon_convex", "BASIC_PROPERTIES",
    "CIRCULANT_MIN_N", "bound_mirror_ensure", "bound_mirror_expand", "bound_mirror_shrink", "gaussian_blur",
    "gaussian_mask", "gradient2", "gradient2_complex", "gradient2_magnitude", "gvf", "snake_first_column",
    "snake_matrix", "snakedeform", "snakeindex", "snakeinterp", "xconv2",
    "RegionProps", "region_table", "regionprops",
    "clip_polygon_rect", "convhull", "minboundrect", "poly2mask", "polyarea", "polygeom", "polyxpoly", "roipoly",
    "bwperim", "del2", "homomorphic_butterworth", "regional_maxima_by_reconstruction", "quiver_field", "snake_plot",
    "watershed_transform", "watershed_skimage", "imregionalmin", "imregionalmax", "imimposemin",
    "EdgeResult", "edge", "gradient_roberts", "gradient_sobel_prewitt", "log_zero_crossings", "thin_gradient",
    "fspecial",
    "disk_decomposition", "geodesic_dilation", "geodesic_erosion", "imclose", "imdilate", "imerode", "imopen",
    "imreconstruct", "intline", "line_strel", "minkowski_sum", "morphological_gradient", "periodic_line",
    "reconstruct_by_erosion", "reconstruct_iterative", "se_origin", "strel",
    "KMeansGray", "KMeansResult", "kmeans_gray", "kmeans_lloyd", "objective_J", "pairwise_distance",
    "BlockOtsu", "OtsuCurves", "block_otsu", "class_coverage", "class_mean_intensity", "graythresh",
    "ice_concentration", "im2bw", "imquantize", "multithresh", "otsu_criterion", "otsuthresh", "separability",
    "ChainCode", "bound2im", "boundaries", "code_reverse", "fchcode", "first_difference", "min_magnitude",
    "normalized_first_difference",
    "indexed_to_rgb", "rgb2cmy", "rgb2cmyk", "rgb2hsi", "split_rgb",
    "bwareaopen", "count_components", "find_paths", "is_adjacent", "is_m_adjacent", "label_components", "n4", "n8",
    "nd", "region_boundary_mask",
    "bwdist", "center_distance_map", "distance_transform", "pixel_distance", "quasi_euclidean_dt",
    "conv2", "conv_at", "imfilter",
    "hist", "imhist", "normalized_histogram",
    "imfill", "label2rgb", "size_colorbar",
    "interp2", "interp_bicubic", "interp_bilinear", "interp_nearest", "keys_kernel", "resize", "warp_image",
    "REPO_ROOT", "data_roots", "load_image", "output_dir", "read_image", "repo_root",
    "im2double", "im2uint8", "imcomplement", "matlab_round", "num2str", "rgb2gray_matlab",
    "finish_figure", "imshow_matlab", "imshow_scale", "save_image", "show_matrix", "to_display_uint8",
    "complement", "difference", "gray_complement", "gray_intersection", "gray_union", "intersection", "reflect",
    "translate", "union",
]
