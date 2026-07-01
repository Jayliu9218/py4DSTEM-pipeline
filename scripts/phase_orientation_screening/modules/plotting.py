"""Matplotlib plotting helpers for screening outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def savefig(filename, out_dir=None, **kwargs):
    import matplotlib.pyplot as plt

    filename = Path(filename)
    if out_dir is not None and not filename.is_absolute():
        filename = Path(out_dir) / filename
    filename.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("dpi", 200)
    kwargs.setdefault("bbox_inches", "tight")
    plt.savefig(filename, **kwargs)
    plt.close()
    print(f"[saved] {filename}")
    return filename


def plot_scalar_map(arr, title, filename, out_dir=None, cmap="viridis"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 4))
    im = plt.imshow(np.asarray(arr).T, origin="lower", cmap=cmap, interpolation="nearest")
    plt.title(title)
    plt.xlabel("scan x")
    plt.ylabel("scan y")
    plt.colorbar(im, shrink=0.8)
    return savefig(filename, out_dir=out_dir)


def plot_histogram(arr, title, filename, out_dir=None, bins=80):
    import matplotlib.pyplot as plt

    vals = np.asarray(arr).ravel()
    vals = vals[np.isfinite(vals)]
    plt.figure(figsize=(5, 4))
    if vals.size:
        plt.hist(vals, bins=bins)
    plt.title(title)
    plt.xlabel(title)
    plt.ylabel("count")
    return savefig(filename, out_dir=out_dir)
