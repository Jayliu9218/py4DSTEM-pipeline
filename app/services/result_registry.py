from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
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
        if suffix not in entry.export_formats:
            raise ResultRegistryError(
                f"{entry.key} supports {', '.join(entry.export_formats)}, not .{output_path.suffix.lstrip('.')}"
            )

        if suffix == "npy":
            np.save(output_path, np.asarray(entry.data))
        elif suffix == "npz":
            self._export_npz(entry, output_path)
        elif suffix == "png":
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
