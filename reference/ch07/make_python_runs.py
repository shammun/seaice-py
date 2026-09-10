"""Verifier-side Python runs that are too slow for the test suite (kept as reproducible artefacts).

1. ``outputs/ch07/verify/alg3_python.npz`` — Algorithm 3 (``seaice_kmean_gvf``) on ch7's own ``sea_ice_test.jpg`` with the
   ``sea_ice_demo.m`` parameters, so the ch06-inherited ``near`` label can be **measured on this chapter's
   image** against ``reference/ch07/demo.mat``.
2. ``sensitivity_ticks.json`` — the §7.3.2 snake sweep at ``iter = 1`` (Fig. 7.26(a)) at ``--downscale`` 2 and 1,
   to settle whether the port's colour-bar ticks reproducing the printed panel is parity or coincidence.

Usage: .venv/Scripts/python.exe reference/ch07/make_python_runs.py [alg3|sens]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from seaice.ch06_gvf_snake import BOOK_PARAMS, seaice_kmean_gvf  # noqa: E402
from seaice.ch07_ice_type import colorbar_area_ticks, ice_shape_enhancement  # noqa: E402
from seaice.core.io import load_image  # noqa: E402

REF = ROOT / "reference/ch07"
#: The Algorithm-3 output is a segmentation **of the book image**, so it stays under the git-ignored
#: ``outputs/`` tree (CLAUDE.md rule 12); only the tick summary, which carries no image data, goes in reference/.
VERIFY = ROOT / "outputs/ch07/verify"
P = BOOK_PARAMS["sea_ice_demo"]


def _params(num, iters):
    return dict(kms0=P["kms0"], sigma=P["sigma"], GradientOn=P["GradientOn"], GVFOn=P["GVFOn"],
                Num=num, mu=P["mu"], iter=iters, alpha=P["alpha"], beta=P["beta"], gamma=P["gamma"],
                kappa=P["kappa"], Dmin=P["Dmin"], Dmax=P["Dmax"], Ra_min=P["Ra_min"], Ra=P["Ra"],
                Rc=P["Rc"], Rl=P["Rl"], se_radius=P["se_radius"], timer=P["timer"], keep_history=False)


def alg3() -> None:
    rgb, _ = load_image("ch07", "sea_ice_test.jpg", allow_fallback=False, verbose=False)
    t0 = time.time()
    g = seaice_kmean_gvf(rgb, **_params(P["Num"], P["iter"]))
    dt = time.time() - t0
    VERIFY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VERIFY / "alg3_python.npz", seg=g.out, bk=g.bk, bw=g.bw, seconds=np.array(dt))
    print(f"alg3: {dt:.1f} s -> {VERIFY / 'alg3_python.npz'}")


def sens() -> None:
    rgb, _ = load_image("ch07", "sea_ice_test.jpg", allow_fallback=False, verbose=False)
    rows = []
    for ds in (2, 1):
        img = rgb[::ds, ::ds] if ds > 1 else rgb
        t0 = time.time()
        g = seaice_kmean_gvf(img, **_params(P["Num"], 1))
        e = ice_shape_enhancement(g.bk, g.out, min_floe=P["min_floe"], min_brash=P["min_brash"],
                                  se_th=P["se_th"], nbins=0)
        colours = np.concatenate([e.color_floe, e.color_brash])
        _, ticks = colorbar_area_ticks(colours, 6)
        rows.append({"downscale": ds, "shape": list(img.shape[:2]), "snake_iter": 1, "gvf_iter": P["Num"],
                     "n_floe": len(e.ice_floe), "n_brash": len(e.brash_ice),
                     "ticks": ticks.tolist(), "seconds": round(time.time() - t0, 1),
                     "book_fig_7_26a": [2, 184, 407, 695, 1100, 1792, 8112]})
        print(json.dumps(rows[-1]))
    (REF / "sensitivity_ticks.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["alg3", "sens"]):
        {"alg3": alg3, "sens": sens}[name]()
