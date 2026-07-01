"""Output helpers shared by screening modules."""

from __future__ import annotations

import json

import numpy as np


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[saved] {path}")


def finite_stat(arr, fn):
    vals = np.asarray(arr, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(fn(vals))


def summarize_fraction(mask):
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return None
    return float(np.sum(arr) / arr.size)
