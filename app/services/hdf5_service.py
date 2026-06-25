from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from app.services.array_reduction import scan_sum


class Hdf5Service:
    HDF5_SUFFIXES = {".h5", ".hdf5", ".emd"}

    def is_hdf5_like(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() in self.HDF5_SUFFIXES

    def open_file(self, file_path: str | Path) -> h5py.File:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        if not self.is_hdf5_like(path):
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

    def describe_preview(self, node: h5py.Group | h5py.Dataset) -> dict[str, object]:
        if isinstance(node, h5py.Dataset):
            shape = tuple(int(value) for value in node.shape)
            if len(shape) == 2 and np.issubdtype(node.dtype, np.number):
                return {"kind": "Diffraction slice", "shape": shape}
            if len(shape) == 4 and np.issubdtype(node.dtype, np.number):
                return {"kind": "DataCube", "shape": shape}
            return {"kind": "Not displayable", "shape": shape}
        try:
            dataset = self.resolve_4d_dataset(node)
            return {"kind": "DataCube", "shape": tuple(int(value) for value in dataset.shape)}
        except ValueError:
            return {"kind": "Not displayable", "shape": None}

    def resolve_4d_dataset(self, node: h5py.Group | h5py.Dataset) -> h5py.Dataset:
        if isinstance(node, h5py.Dataset):
            if node.ndim == 4 and np.issubdtype(node.dtype, np.number):
                return node
            raise ValueError("Selected node is not a numeric 4D dataset.")
        data = node.get("data")
        if isinstance(data, h5py.Dataset) and data.ndim == 4 and np.issubdtype(data.dtype, np.number):
            return data
        raise ValueError("Selected group does not contain a numeric 4D 'data' dataset.")

    def read_4d_diffraction_pattern(self, dataset: h5py.Dataset, rx: int = 0, ry: int = 0) -> np.ndarray:
        if dataset.ndim != 4:
            raise ValueError(f"Expected a 4D dataset, got shape {dataset.shape}.")
        self._ensure_numeric(dataset)

        if rx < 0 or rx >= dataset.shape[0]:
            raise IndexError(f"rx={rx} is out of range for axis 0 with size {dataset.shape[0]}.")
        if ry < 0 or ry >= dataset.shape[1]:
            raise IndexError(f"ry={ry} is out of range for axis 1 with size {dataset.shape[1]}.")

        return np.asarray(dataset[rx, ry, :, :])

    def read_4d_scan_image(
        self,
        dataset: h5py.Dataset,
        *,
        memory_budget_bytes: int | None = None,
        progress_callback=None,
    ) -> np.ndarray:
        if dataset.ndim != 4:
            raise ValueError(f"Expected a 4D dataset, got shape {dataset.shape}.")
        self._ensure_numeric(dataset)
        if memory_budget_bytes is None and progress_callback is None:
            return scan_sum(dataset)
        from app.services.array_reduction import DEFAULT_BLOCK_BYTES, scan_sum_with_progress

        return scan_sum_with_progress(
            dataset,
            memory_budget_bytes=memory_budget_bytes or DEFAULT_BLOCK_BYTES,
            progress_callback=progress_callback,
        )

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
