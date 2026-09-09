"""Parity helpers: compare a Python result with a MATLAB/Octave reference (.mat) robustly.

Import in tests:
    from tools.compare_arrays import assert_parity, label_agreement, load_ref

- Binary/logical images: exact match, report fraction of differing pixels.
- Label images: label numbering differs between MATLAB (column-major scan) and skimage (row-major), so we
  compare the *partition*: two label images agree if there is a one-to-one mapping between labels.
- Float arrays: allclose with atol/rtol, and report max abs error.
- MATLAB stores arrays column-major, but scipy.io.loadmat already returns the correct (row, col) layout.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_ref(path: str | Path, key: str) -> np.ndarray:
    d = loadmat(str(path))
    if key not in d:
        raise KeyError(f"{key} not in {path}; keys = {[k for k in d if not k.startswith('__')]}")
    return np.asarray(d[key])


def label_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels for which the partition induced by `a` agrees with that of `b` (0 = background)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    # background must agree
    bg_ok = (a == 0) == (b == 0)
    # map each a-label to the b-label it overlaps most, and vice versa
    pairs = np.stack([a, b], axis=1)[(a > 0) & (b > 0)]
    if len(pairs) == 0:
        return float(bg_ok.mean())
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    best_a = {}
    best_b = {}
    for (la, lb), c in zip(uniq, counts):
        if c > best_a.get(la, (None, -1))[1]:
            best_a[la] = (lb, c)
        if c > best_b.get(lb, (None, -1))[1]:
            best_b[lb] = (la, c)
    ok = bg_ok.copy()
    fg = (a > 0) & (b > 0)
    mapped = np.array([best_a.get(x, (None,))[0] == y and best_b.get(y, (None,))[0] == x for x, y in pairs])
    ok[fg] = mapped
    return float(ok.mean())


def assert_parity(py: np.ndarray, ref: np.ndarray, kind: str = "float", atol: float = 1e-6, rtol: float = 1e-5,
                  min_agreement: float = 1.0, name: str = "") -> dict:
    """Compare and return a metrics dict; raise AssertionError with the metrics if below tolerance."""
    py = np.asarray(py)
    ref = np.asarray(ref)
    if py.shape != ref.shape:
        # MATLAB may give (n,1) where Python gives (n,)
        if py.squeeze().shape == ref.squeeze().shape:
            py, ref = py.squeeze(), ref.squeeze()
        else:
            raise AssertionError(f"{name}: shape mismatch {py.shape} vs {ref.shape}")
    if kind == "binary":
        d = (py.astype(bool) != ref.astype(bool))
        metrics = {"differing_fraction": float(d.mean()), "n_diff": int(d.sum())}
        if metrics["differing_fraction"] > 1 - min_agreement:
            raise AssertionError(f"{name}: binary mismatch {metrics}")
    elif kind == "label":
        agr = label_agreement(py, ref)
        metrics = {"label_agreement": agr, "n_labels_py": int(py.max()), "n_labels_ref": int(ref.max())}
        if agr < min_agreement:
            raise AssertionError(f"{name}: label partition mismatch {metrics}")
    else:
        err = np.abs(py.astype(float) - ref.astype(float))
        metrics = {"max_abs_err": float(np.nanmax(err)), "mean_abs_err": float(np.nanmean(err))}
        if not np.allclose(py, ref, atol=atol, rtol=rtol, equal_nan=True):
            raise AssertionError(f"{name}: numeric mismatch {metrics} (atol={atol}, rtol={rtol})")
    return metrics
