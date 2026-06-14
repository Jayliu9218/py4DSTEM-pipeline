from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class ResultRegistryError(Exception):
    """User-facing result registry error."""


@dataclass(frozen=True)
class ResultEntry:
    name: str
    category: str
    data: Any
    export_formats: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.category}/{self.name}"


class ResultRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, ResultEntry] = {}

    def register(
        self,
        name: str,
        category: str,
        data: Any,
        export_formats: tuple[str, ...] = ("npy",),
        metadata: dict[str, object] | None = None,
    ) -> ResultEntry:
        entry = ResultEntry(
            name=name,
            category=category,
            data=data,
            export_formats=tuple(export_formats),
            metadata=dict(metadata or {}),
        )
        self._entries[entry.key] = entry
        return entry

    def list_entries(self) -> list[ResultEntry]:
        return sorted(self._entries.values(), key=lambda entry: entry.key.lower())

    def get(self, key: str) -> ResultEntry:
        try:
            return self._entries[key]
        except KeyError as exc:
            raise ResultRegistryError(f"Result is not registered: {key}") from exc

    def clear(self) -> None:
        self._entries.clear()

    def export(self, key: str, path: str | Path) -> Path:
        entry = self.get(key)
        output_path = Path(path)
        suffix = output_path.suffix.lower().lstrip(".")
        if suffix in {"tif", "tiff"}:
            suffix = "tiff"
        # CSV is accepted universally regardless of export_formats, since any
        # numeric array or scalar dict can be flattened to tabular form.
        if suffix != "csv" and suffix not in entry.export_formats:
            raise ResultRegistryError(
                f"{entry.key} supports {', '.join(entry.export_formats)}, not .{output_path.suffix.lstrip('.')}"
            )

        if suffix == "npy":
            np.save(output_path, np.asarray(entry.data))
        elif suffix == "npz":
            self._export_npz(entry, output_path)
        elif suffix == "csv":
            self._export_csv(entry, output_path)
        elif suffix == "png":
            import matplotlib.pyplot as plt

            plt.imsave(output_path, np.asarray(entry.data), cmap="gray")
        elif suffix == "tiff":
            try:
                import tifffile
            except ModuleNotFoundError as exc:
                raise ResultRegistryError(
                    "TIFF export requires tifffile. Install project requirements first."
                ) from exc
            tifffile.imwrite(output_path, np.asarray(entry.data))
        else:
            raise ResultRegistryError(f"Unsupported export format: {suffix}")
        return output_path

    def _export_npz(self, entry: ResultEntry, path: Path) -> None:
        if isinstance(entry.data, dict):
            np.savez(path, **{str(key): np.asarray(value) for key, value in entry.data.items()})
            return
        np.savez(path, data=np.asarray(entry.data))

    def _export_csv(self, entry: ResultEntry, path: Path) -> None:
        """Write numeric result data to CSV in a form suited to its shape.

        - 2D array (maps): long format rows ``row, col, value``.
        - dict of scalars (calibration, aberrations): ``key, value`` rows.
        - dict of arrays (component bundles): ``component, row, col, value``.
        - 1D array (curves, profiles): ``index, value`` rows.
        - scalar: a single ``value`` row.
        Metadata is written as leading ``#`` comment lines.
        """
        import numbers

        def _is_scalar(value: Any) -> bool:
            return isinstance(value, numbers.Number) or np.isscalar(value)

        rows: list[list[str]] = []
        header: list[str]
        data = entry.data

        if isinstance(data, dict):
            arrays = {str(k): np.asarray(v) for k, v in data.items()}
            if arrays and all(_is_scalar(v) for v in data.values()):
                header = ["key", "value"]
                for key, value in data.items():
                    rows.append([key, repr(value)])
            else:
                header = ["component", "row", "col", "value"]
                for component, arr in arrays.items():
                    if arr.ndim == 2:
                        for r in range(arr.shape[0]):
                            for c in range(arr.shape[1]):
                                rows.append([component, str(r), str(c), repr(arr[r, c])])
                    elif arr.ndim == 1:
                        for i in range(arr.shape[0]):
                            rows.append([component, str(i), "", repr(arr[i])])
        else:
            arr = np.asarray(data)
            if arr.ndim == 2:
                header = ["row", "col", "value"]
                for r in range(arr.shape[0]):
                    for c in range(arr.shape[1]):
                        rows.append([str(r), str(c), repr(arr[r, c])])
            elif arr.ndim == 1:
                header = ["index", "value"]
                for i in range(arr.shape[0]):
                    rows.append([str(i), repr(arr[i])])
            elif arr.ndim == 0:
                header = ["value"]
                rows.append([repr(arr.item())])
            else:
                # >=3D: flatten with a single linear index.
                flat = arr.reshape(-1)
                header = ["flat_index", "value"]
                for i in range(flat.shape[0]):
                    rows.append([str(i), repr(flat[i])])

        with path.open("w", newline="", encoding="utf-8") as handle:
            for key, value in entry.metadata.items():
                handle.write(f"# {key}: {value}\n")
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
