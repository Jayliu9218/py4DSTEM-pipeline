from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


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
    result_entries: list[dict[str, object]] = field(default_factory=list)
    grid_states: dict[str, dict[str, object]] = field(default_factory=dict)
    window_state: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 4,
            "file_path": self.file_path,
            "selected_hdf5_path": self.selected_hdf5_path,
            "image_scaling": self.image_scaling,
            "image_cmap": self.image_cmap,
            "cuda_enabled": self.cuda_enabled,
            "recent_export_dir": self.recent_export_dir,
            "dataset_roles": self.dataset_roles,
            "page_params": self.page_params,
            "result_entries": self.result_entries,
            "grid_states": self.grid_states,
            "window_state": self.window_state,
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
            result_entries=payload.get("result_entries", []),
            grid_states=_page_params(payload.get("grid_states", {})),
            window_state=_optional_str(payload.get("window_state")),
        )


class ProjectStateService:
    def save_with_results(
        self,
        path: str | Path,
        state: ProjectState,
        result_data: dict[str, Any],
    ) -> Path:
        output_path = Path(path)
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        results_dir = output_path.parent / (output_path.stem + "_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        data_files: dict[str, str] = {}
        for key, data in result_data.items():
            safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
            safe_key = safe_key.replace("/", "__")
            npz_path = results_dir / f"{safe_key}.npz"
            if isinstance(data, dict):
                arrays = {str(k): np.asarray(v) for k, v in data.items()}
                np.savez(npz_path, **arrays)
            else:
                np.save(npz_path, np.asarray(data))
            data_files[key] = str(npz_path.relative_to(output_path.parent))
        state_with_files = ProjectState(
            file_path=state.file_path,
            selected_hdf5_path=state.selected_hdf5_path,
            image_scaling=state.image_scaling,
            image_cmap=state.image_cmap,
            cuda_enabled=state.cuda_enabled,
            recent_export_dir=state.recent_export_dir,
            dataset_roles=state.dataset_roles,
            page_params=state.page_params,
            result_entries=[
                {**entry, "data_file": data_files.get(entry.get("key", ""), "")}
                for entry in state.result_entries
                if entry.get("key", "") in data_files
            ],
            grid_states=state.grid_states,
        )
        output_path.write_text(
            json.dumps(state_with_files.to_dict(), indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
        return output_path

    def load_results(self, project_path: str | Path, entries: list[dict[str, object]]) -> dict[str, Any]:
        project_dir = Path(project_path).parent
        results: dict[str, Any] = {}
        for entry in entries:
            key = entry.get("key", "")
            data_file = entry.get("data_file", "")
            if not key or not data_file:
                continue
            npz_path = project_dir / data_file
            if not npz_path.exists():
                continue
            try:
                loaded = np.load(npz_path, allow_pickle=True)
                if "arr_0" in loaded:
                    results[key] = loaded["arr_0"]
                elif "data" in loaded:
                    results[key] = loaded["data"]
                else:
                    first_key = list(loaded.keys())[0] if len(loaded.keys()) > 0 else None
                    if first_key:
                        if len(loaded.keys()) == 1:
                            results[key] = loaded[first_key]
                        else:
                            results[key] = {k: loaded[k] for k in loaded.keys()}
            except Exception:
                continue
        return results

    def save(self, path: str | Path, state: ProjectState) -> Path:
        output_path = Path(path)
        output_path.write_text(
            json.dumps(state.to_dict(), indent=2, sort_keys=True, default=_json_default),
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


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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
