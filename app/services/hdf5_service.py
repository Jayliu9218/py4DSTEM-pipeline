from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


class Hdf5Service:
    def open_file(self, file_path: str | Path) -> h5py.File:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        if path.suffix.lower() not in {".h5", ".hdf5", ".emd"}:
            raise ValueError("Please choose a .h5, .hdf5, or .emd file.")
        return h5py.File(path, "r")

    def describe_node(self, node: h5py.Group | h5py.Dataset, hdf5_path: str) -> dict[str, Any]:
        attrs = {key: self._format_attr(value) for key, value in node.attrs.items()}

        if isinstance(node, h5py.Dataset):
            return {
                "path": hdf5_path,
                "type": "dataset",
                "shape": tuple(node.shape),
                "dtype": str(node.dtype),
                "attrs": attrs,
            }

        return {
            "path": hdf5_path,
            "type": "group",
            "shape": "-",
            "dtype": "-",
            "attrs": attrs,
        }

    def read_2d_dataset(self, dataset: h5py.Dataset) -> np.ndarray:
        if dataset.ndim != 2:
            raise ValueError(f"Expected a 2D dataset, got shape {dataset.shape}.")
        self._ensure_numeric(dataset)
        return np.asarray(dataset[...])

    def read_4d_diffraction_pattern(self, dataset: h5py.Dataset, rx: int = 0, ry: int = 0) -> np.ndarray:
        if dataset.ndim != 4:
            raise ValueError(f"Expected a 4D dataset, got shape {dataset.shape}.")
        self._ensure_numeric(dataset)

        if rx < 0 or rx >= dataset.shape[0]:
            raise IndexError(f"rx={rx} is out of range for axis 0 with size {dataset.shape[0]}.")
        if ry < 0 or ry >= dataset.shape[1]:
            raise IndexError(f"ry={ry} is out of range for axis 1 with size {dataset.shape[1]}.")

        return np.asarray(dataset[rx, ry, :, :])

    def read_4d_scan_image(self, dataset: h5py.Dataset) -> np.ndarray:
        if dataset.ndim != 4:
            raise ValueError(f"Expected a 4D dataset, got shape {dataset.shape}.")
        self._ensure_numeric(dataset)
        return np.asarray(dataset[...]).sum(axis=(2, 3))

    def _ensure_numeric(self, dataset: h5py.Dataset) -> None:
        if not np.issubdtype(dataset.dtype, np.number):
            raise TypeError(f"Dataset dtype is not numeric: {dataset.dtype}")

    def _format_attr(self, value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.ndarray):
            if value.size <= 16:
                return str(value.tolist())
            return f"array shape={value.shape} dtype={value.dtype}"
        if isinstance(value, np.generic):
            return str(value.item())
        return str(value)
