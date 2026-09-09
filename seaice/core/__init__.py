"""Reusable primitives shared by all chapters (see ``knowledge/function_map.md`` for the MATLAB ↔ Python map).

Chapter 2 primitives:

* :mod:`~seaice.core.io` — data paths, private-Drive / public-fallback loader (``load_image``, ``read_image``, ``data_roots``, ``output_dir``, ``fetch``)
* :mod:`~seaice.core.matlab_compat` — ``rgb2gray_matlab``, ``imcomplement``, ``matlab_round``, ``im2double``, ``im2uint8``
* :mod:`~seaice.core.plotting` — MATLAB-style ``imshow`` scaling and PNG saving
* :mod:`~seaice.core.color` — ``rgb2cmy``, ``rgb2cmyk``, ``rgb2hsi``, ``indexed_to_rgb`` (§2.1)
* :mod:`~seaice.core.histogram` — ``imhist``, ``normalized_histogram`` (§2.2)
* :mod:`~seaice.core.connectivity` — neighbourhoods, adjacency, ``label_components`` (= ``bwlabel``) (§2.3)
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
"""
from __future__ import annotations

from . import chaincode, clustering, color, connectivity, distance, filters, histogram, interp, io, \
    matlab_compat, plotting, setops, synth, threshold
from .chaincode import ChainCode, bound2im, boundaries, code_reverse, fchcode, first_difference, min_magnitude, \
    normalized_first_difference
from .clustering import KMeansGray, KMeansResult, kmeans_gray, kmeans_lloyd, objective_J, pairwise_distance
from .color import indexed_to_rgb, rgb2cmy, rgb2cmyk, rgb2hsi, split_rgb
from .connectivity import count_components, find_paths, is_adjacent, is_m_adjacent, label_components, n4, n8, nd, \
    region_boundary_mask
from .distance import bwdist, center_distance_map, distance_transform, pixel_distance, quasi_euclidean_dt
from .filters import conv2, conv_at, imfilter
from .histogram import imhist, normalized_histogram
from .interp import interp2, interp_bicubic, interp_bilinear, interp_nearest, keys_kernel, resize, warp_image
from .io import REPO_ROOT, data_roots, load_image, output_dir, read_image, repo_root
from .matlab_compat import im2double, im2uint8, imcomplement, matlab_round, rgb2gray_matlab
from .plotting import finish_figure, imshow_matlab, imshow_scale, save_image, show_matrix, to_display_uint8
from .threshold import BlockOtsu, OtsuCurves, block_otsu, class_coverage, class_mean_intensity, graythresh, \
    ice_concentration, im2bw, imquantize, multithresh, otsu_criterion, otsuthresh, separability
from .setops import complement, difference, gray_complement, gray_intersection, gray_union, intersection, reflect, \
    translate, union

__all__ = [
    "chaincode", "clustering", "color", "connectivity", "distance", "filters", "histogram", "interp", "io",
    "matlab_compat", "plotting", "setops", "synth", "threshold",
    "KMeansGray", "KMeansResult", "kmeans_gray", "kmeans_lloyd", "objective_J", "pairwise_distance",
    "BlockOtsu", "OtsuCurves", "block_otsu", "class_coverage", "class_mean_intensity", "graythresh",
    "ice_concentration", "im2bw", "imquantize", "multithresh", "otsu_criterion", "otsuthresh", "separability",
    "ChainCode", "bound2im", "boundaries", "code_reverse", "fchcode", "first_difference", "min_magnitude",
    "normalized_first_difference",
    "indexed_to_rgb", "rgb2cmy", "rgb2cmyk", "rgb2hsi", "split_rgb",
    "count_components", "find_paths", "is_adjacent", "is_m_adjacent", "label_components", "n4", "n8", "nd",
    "region_boundary_mask",
    "bwdist", "center_distance_map", "distance_transform", "pixel_distance", "quasi_euclidean_dt",
    "conv2", "conv_at", "imfilter",
    "imhist", "normalized_histogram",
    "interp2", "interp_bicubic", "interp_bilinear", "interp_nearest", "keys_kernel", "resize", "warp_image",
    "REPO_ROOT", "data_roots", "load_image", "output_dir", "read_image", "repo_root",
    "im2double", "im2uint8", "imcomplement", "matlab_round", "rgb2gray_matlab",
    "finish_figure", "imshow_matlab", "imshow_scale", "save_image", "show_matrix", "to_display_uint8",
    "complement", "difference", "gray_complement", "gray_intersection", "gray_union", "intersection", "reflect",
    "translate", "union",
]
