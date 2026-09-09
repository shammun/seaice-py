"""Port of ``MATLAB_ROOT/ch2/chain code/chain_diff.m`` — boundary tracing and Freeman chain codes (Figs. 2.18–2.21).

Book §2.7, Eq. (2.31).  The MATLAB script hard-codes the 8×9 object of Fig. 2.19(a), traces its boundary with
``boundaries(B, 8, 'cww')`` — a typo for ``'ccw'`` that MATLAB treats as "not ccw", i.e. **clockwise**, which is
the direction shown in Figs. 2.20–2.21 — rebuilds the boundary image with ``bound2im`` and computes
``c = fchcode(b)``, then shows ``bim`` with a red "starting point" label at ``c.x0y0``.

Extras from the text: the 4-direction code of Fig. 2.19(b), the numbering diagram of Fig. 2.18, the
coordinate/code table of Fig. 2.20 and the five sequences of Fig. 2.21 (including the normalised first
difference, which ``fchcode.m`` does not compute).

Usage: ``python scripts/ch02_chain_diff.py [--out outputs/ch02] [--show]``
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seaice.ch02_preliminaries import chain_code_demo  # noqa: E402
from seaice.core import synth  # noqa: E402
from seaice.core.chaincode import STEPS_4, STEPS_8  # noqa: E402
from seaice.core.cli import chapter_argparser, resolve_dirs  # noqa: E402
from seaice.core.plotting import finish_figure, save_image, show_matrix  # noqa: E402

CH = "ch02"


def _direction_diagram(ax: plt.Axes, steps: np.ndarray, title: str) -> None:
    """Fig. 2.18: arrows from the centre labelled with their code (row axis points down like an image)."""
    for code, (dr, dc) in enumerate(steps):
        ax.annotate("", xy=(dc, dr), xytext=(0, 0), arrowprops=dict(arrowstyle="->", lw=1.5))
        ax.text(1.25 * dc, 1.25 * dr, str(code), ha="center", va="center", fontsize=12)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(1.6, -1.6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)


def main(argv: list[str] | None = None) -> int:
    args = chapter_argparser(CH, __doc__.splitlines()[0]).parse_args(argv)
    _, out = resolve_dirs(args)
    res = chain_code_demo(synth.FIG_2_19_OBJECT)
    c, c4 = res["c"], res["c4"]
    written: list[Path] = []

    # --- Fig. 2.18 numbering scheme ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    _direction_diagram(axes[0], STEPS_4, "(a) 4-direction")
    _direction_diagram(axes[1], STEPS_8, "(b) 8-direction")
    fig.suptitle("Fig. 2.18  Numbering scheme of the chain code")
    written.append(finish_figure(fig, out / "fig_2_18_direction_numbering.png", args.show))

    # --- the script's imshow(bim) + text('starting point') ---------------------------------------------------
    written.append(save_image(out / "chain_diff_bim.png", res["bim"]))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(res["bim"], cmap="gray", interpolation="nearest")
    r0, c0 = c.x0y0
    ax.text(c0, r0, "starting point", color="r", fontsize=11)
    ax.set_axis_off()
    ax.set_title("chain_diff.m: bim with starting point at (row, col) = "
                 f"({c.x0y0_matlab[0]}, {c.x0y0_matlab[1]}) [1-based]")
    written.append(finish_figure(fig, out / "chain_diff_starting_point.png", args.show))

    # --- Fig. 2.19: object, 4-direction and 8-direction boundary codes ---------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    show_matrix(axes[0], res["B"].astype(int), title="(a) Binary image with an object")
    for ax, b, cc, bim, title in ((axes[1], res["b4"], c4, res["bim4"], "(b) 4-direction"),
                                  (axes[2], res["b"], c, res["bim"], "(c) 8-direction")):
        show_matrix(ax, np.zeros_like(res["B"], dtype=int), fmt="", highlight=bim, cmap="Greys", title=title)
        pts = b[:-1]
        for (r, col), code in zip(pts, cc.fcc):
            ax.text(col, r, str(int(code)), ha="center", va="center", fontsize=9, color="b")
        ax.plot(b[:, 1], b[:, 0], "r-", lw=0.8)
        ax.plot(cc.x0y0[1], cc.x0y0[0], "ro", ms=6)
    fig.suptitle("Fig. 2.19  Chain codes of an object boundary (red dot = starting point; code at the departing pixel)")
    written.append(finish_figure(fig, out / "fig_2_19_boundary.png", args.show))

    # --- Fig. 2.20 / 2.21 as text ------------------------------------------------------------------------------
    lines = ["Fig. 2.20  Representation for a boundary (coordinates 1-based (row, col); starting point first)",
             f"Starting point: {res['x0y0_matlab']}",
             "  row  col  8-dir code"]
    lines += [f"  {r:3d}  {cc:3d}  {code}" for r, cc, code in res["table"]]
    lines += ["", "Fig. 2.21  8-directional chain codes and first differences"]
    for name, seq in res["sequences"].items():
        lines.append(f"{name:36s}: {' '.join(map(str, seq.tolist()))}")
    lines += ["", f"Fig. 2.19(b) 4-direction code ({len(c4.fcc)} codes): {' '.join(map(str, c4.fcc.tolist()))}",
              f"Fig. 2.19(c) 8-direction code ({len(c.fcc)} codes): {' '.join(map(str, c.fcc.tolist()))}"]
    txt = out / "fig_2_21_chain_codes.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(txt)
    print("\n".join(lines))

    # --- checks against the book ----------------------------------------------------------------------------
    book = synth.FIG_2_21
    checks = {
        "original == book": np.array_equal(c.fcc, book["original"]),
        "first difference == book": np.array_equal(c.diff, book["first_difference"]),
        "normalized code == book": np.array_equal(c.mm, book["normalized"]),
        "first difference of normalized == book": np.array_equal(
            c.diffmm, book["normalized_first_difference_of_normalized"]),
        "normalized first difference == book": np.array_equal(
            res["sequences"]["Normalized first difference"], book["normalized_first_difference"]),
        "starting point == (4, 2)": res["x0y0_matlab"] == synth.FIG_2_20_START_MATLAB,
        "boundary closed (first == last)": bool(np.array_equal(res["b"][0], res["b"][-1])),
        "boundary length d == 19": int(res["d"][0]) == 19,
    }
    print()
    for k, v in checks.items():
        print(f"{'OK ' if v else 'FAIL'} {k}")
    print("figures written:")
    for p in written:
        print("  ", p)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
