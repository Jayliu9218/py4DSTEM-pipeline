from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ProjectStateError(Exception):
    """User-facing project state error."""


@dataclass(frozen=True)
class ProjectState:
    file_path: str | None = None
    selected_hdf5_path: str | None = None
    image_scaling: str = "log"
    image_cmap: str = "gray"
    cuda_enabled: bool = False
    recent_export_dir: str | None = None
    dataset_roles: dict[str, str | None] = field(default_factory=dict)
    page_params: dict[str, dict[str, object]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "file_path": self.file_path,
            "selected_hdf5_path": self.selected_hdf5_path,
            "image_scaling": self.image_scaling,
            "image_cmap": self.image_cmap,
            "cuda_enabled": self.cuda_enabled,
            "recent_export_dir": self.recent_export_dir,
            "dataset_roles": self.dataset_roles,
            "page_params": self.page_params,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectState":
        return cls(
            file_path=_optional_str(payload.get("file_path")),
            selected_hdf5_path=_optional_str(payload.get("selected_hdf5_path")),
            image_scaling=str(payload.get("image_scaling") or "log"),
            image_cmap=str(payload.get("image_cmap") or "gray"),
            cuda_enabled=bool(payload.get("cuda_enabled", False)),
            recent_export_dir=_optional_str(payload.get("recent_export_dir")),
            dataset_roles=_string_map(payload.get("dataset_roles", {})),
            page_params=_page_params(payload.get("page_params", {})),
        )


class ProjectStateService:
    def save(self, path: str | Path, state: ProjectState) -> Path:
        output_path = Path(path)
        output_path.write_text(
            json.dumps(state.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    def load(self, path: str | Path) -> ProjectState:
        input_path = Path(path)
        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ProjectStateError(f"Could not load project file: {exc}") from exc
        if not isinstance(payload, dict):
            raise ProjectStateError("Project file must contain a JSON object.")
        return ProjectState.from_dict(payload)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _string_map(value: object) -> dict[str, str | None]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _optional_str(item) for key, item in value.items()}


def _page_params(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, object]] = {}
    for page, params in value.items():
        if isinstance(params, dict):
            result[str(page)] = dict(params)
    return result
