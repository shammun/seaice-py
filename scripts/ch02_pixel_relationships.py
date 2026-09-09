"""Text-only demo of Book §2.3 — neighbourhoods (Fig. 2.9), paths (Fig. 2.10), connected components (Fig. 2.11).

No MATLAB script exists; the primitives are :mod:`seaice.core.connectivity` (``label_components`` = ``bwlabel``).

Usage: ``python scripts/ch02_pixel_relationships.py [--out outputs/ch02] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch02_preliminaries import pixel_relationship_examples  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.plotting import finish_figure, show_matrix  # noqa: E402

CH = "ch02"


def _neigh_panel(ax: plt.Axes, cells: list[tuple[int, int]], label: str, title: str) -> None:
    grid = np.full((3, 3), "", dtype=object)
    grid[1, 1] = "p"
    hl = np.zeros((3, 3))
    for r, c in cells:
        grid[r, c] = label
        hl[r, c] = 1.0
    hl[1, 1] = 0.5
    ax.imshow(hl, cmap="Blues", vmin=0, vmax=1.6, interpolation="nearest")
    for (r, c), v in np.ndenumerate(grid):
        ax.text(c, r, v, ha="center", va="center", fontsize=11)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="k", lw=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title)


def main(argv: list[str] | None = None) -> int:
    args = chapter_argparser(CH, __doc__.splitlines()[0]).parse_args(argv)
    _, out = resolve_dirs(args)
    ex = pixel_relationship_examples()
    written: list[Path] = []

    # --- Fig. 2.9 --------------------------------------------------------------------------------------------
    p = (1, 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    from seaice.core.connectivity import n4, n8, nd  # local: only for the 3x3 diagram

    _neigh_panel(axes[0], n4(p), "N4", "(a) 4-neighbors N4(p)")
    _neigh_panel(axes[1], nd(p), "ND", "(b) D-neighbors ND(p)")
    _neigh_panel(axes[2], n8(p), "N8", "(c) 8-neighbors N8(p)")
    fig.suptitle("Fig. 2.9  Neighborhoods of a pixel")
    written.append(finish_figure(fig, out / "fig_2_09_neighborhoods.png", args.show))

    # --- Fig. 2.10 -------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    show_matrix(axes[0], ex["fig_2_10_a"].astype(int), title=f"(a) 4-path  ({len(ex['paths_4'])} path)")
    for ax, key, title in ((axes[1], "paths_8", "(b) 8-paths"), (axes[2], "paths_m", "(c) m-path")):
        show_matrix(ax, ex["fig_2_10_bc"].astype(int), title=f"{title}  ({len(ex[key])} path(s) from (1,3) to (3,3))")
        for k, path in enumerate(ex[key]):
            pr = np.array(path)
            ax.plot(pr[:, 1] + 0.12 * k, pr[:, 0] + 0.12 * k, "-o", lw=1.5, ms=4, label=f"path {k + 1}")
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Fig. 2.10  4-, 8- and m-paths (m-adjacency removes the 8-path ambiguity)")
    written.append(finish_figure(fig, out / "fig_2_10_paths.png", args.show))

    # --- Fig. 2.11 -------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, key, n, title in ((axes[0], "labels_4", ex["n_components_4"], "(a) 4-connected components"),
                              (axes[1], "labels_8", ex["n_components_8"], "(b) 8-connected components")):
        L = ex[key]
        ax.imshow(np.where(L > 0, L, np.nan), cmap="tab10", vmin=0.5, vmax=10.5, interpolation="nearest")
        for (r, c), v in np.ndenumerate(ex["fig_2_11"].astype(int)):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=8)
        ax.set_xticks(np.arange(-0.5, L.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, L.shape[0], 1), minor=True)
        ax.grid(which="minor", color="k", lw=0.4)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_title(f"{title}: {n} components (bwlabel numbering by colour)")
    fig.suptitle("Fig. 2.11  Connected components of the same binary image")
    written.append(finish_figure(fig, out / "fig_2_11_components.png", args.show))

    # --- region boundary (§2.3.5) ----------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    show_matrix(axes[0], ex["boundary_4"].astype(int), title="Boundary pixels (4-neighbour test)")
    show_matrix(axes[1], ex["boundary_8"].astype(int), title="Boundary pixels (8-neighbour test)")
    fig.suptitle("Section 2.3.5  Boundary of a region: pixels with at least one neighbour outside R")
    written.append(finish_figure(fig, out / "region_boundary.png", args.show))

    print(f"N4/ND/N8 sizes at interior pixel {ex['p_interior']} of a {ex['shape']} image: "
          f"{len(ex['N4_interior'])}/{len(ex['ND_interior'])}/{len(ex['N8_interior'])};  at corner "
          f"{ex['p_corner']}: {len(ex['N4_corner'])}/{len(ex['ND_corner'])}/{len(ex['N8_corner'])}")
    print(f"Fig. 2.10: 8-paths = {len(ex['paths_8'])}, m-paths = {len(ex['paths_m'])} (book: several vs one)")
    print(f"Fig. 2.11: {ex['n_components_4']} components with 4-adjacency (book 5), "
          f"{ex['n_components_8']} with 8-adjacency (book 2)")
    print("labels (4-adjacency, bwlabel order):")
    print(ex["labels_4"])
    print("figures written:")
    for p_ in written:
        print("  ", p_)
    return 0


if __name__ == "__main__":
    sys.exit(main())
