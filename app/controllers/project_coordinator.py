from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

from app.controllers.application_pages import ApplicationPages
from app.services.project_state_service import ProjectState, ProjectStateService
from app.services.report_service import ReportService
from app.services.result_registry import ResultRegistry, ResultRegistryError
from app.services.workflow_state import WorkflowState


class ProjectCoordinator:
    """Builds and restores application project state without owning dialogs."""

    def __init__(
        self,
        workflow_state: WorkflowState,
        pages: ApplicationPages,
        page_objects: dict[str, Any],
        dpc_pages: tuple[Any, ...],
        parallax_pages: dict[str, Any],
        state_service: ProjectStateService,
        result_registry: ResultRegistry,
        report_service: ReportService,
        log_panel: Any,
    ) -> None:
        self.workflow_state = workflow_state
        self.pages = pages
        self.page_objects = page_objects
        self.dpc_pages = dpc_pages
        self.parallax_pages = parallax_pages
        self.state_service = state_service
        self.result_registry = result_registry
        self.report_service = report_service
        self.log_panel = log_panel
        self.loaded_project_path: Path | None = None

    def snapshot(
        self,
        *,
        file_path: Path | None,
        selected_hdf5_path: str | None,
        image_scaling: str,
        image_cmap: str,
        cuda_enabled: bool,
        recent_export_dir: Path | None,
    ) -> ProjectState:
        roles = self.workflow_state.dataset_roles
        page_params = {
            key: page.params_snapshot() for key, page in self.page_objects.items()
        }
        page_params["dpc"] = self.dpc_params_snapshot()
        page_params.update(
            {
                key: page.params_snapshot()
                for key, page in zip(
                    (
                        "dpc_segmented",
                        "dpc_preprocess",
                        "dpc_review",
                        "dpc_reconstruction",
                        "dpc_legacy",
                    ),
                    self.dpc_pages,
                )
            }
        )
        page_params.update(
            {key: page.params_snapshot() for key, page in self.parallax_pages.items()}
        )
        return ProjectState(
            file_path=str(file_path) if file_path else None,
            selected_hdf5_path=selected_hdf5_path,
            image_scaling=image_scaling,
            image_cmap=image_cmap,
            cuda_enabled=cuda_enabled,
            recent_export_dir=str(recent_export_dir) if recent_export_dir else None,
            dataset_roles={
                "target_datacube": roles.target_datacube,
                "polycrystal_calibration": roles.polycrystal_calibration,
                "vacuum_probe": roles.vacuum_probe,
                "defocused_cbed": roles.defocused_cbed,
            },
            page_params=page_params,
            grid_states=self.pages.grid_states(),
        )

    def apply_page_params(self, page_params: dict[str, dict[str, object]]) -> None:
        for key, page in self.page_objects.items():
            params = page_params.get(key)
            if params and callable(getattr(page, "apply_params_snapshot", None)):
                page.apply_params_snapshot(params)
        legacy_parallax = page_params.get("parallax", {})
        for key, page in self.parallax_pages.items():
            params = page_params.get(key) or legacy_parallax
            if params:
                page.apply_params_snapshot(params)
        legacy_dpc = page_params.get("dpc", {})
        dpc_keys = (
            "dpc_segmented",
            "dpc_preprocess",
            "dpc_review",
            "dpc_reconstruction",
            "dpc_legacy",
        )
        for key, page in zip(dpc_keys, self.dpc_pages):
            params = page_params.get(key) or legacy_dpc
            if params:
                page.apply_params_snapshot(params)

    def restore_grid_states(self, grid_states: dict[str, dict[str, object]]) -> None:
        workspaces = self.pages.named_workspaces()
        for key, grid_state in grid_states.items():
            workspace = workspaces.get(key)
            if workspace is not None:
                workspace.restore_grid_state(grid_state)

    def dpc_params_snapshot(self) -> dict[str, object]:
        snapshot: dict[str, object] = {}
        for page in self.dpc_pages:
            snapshot.update(page.params_snapshot())
        return snapshot

    def export_registered_result(self, parent: Any, default_dir: Path) -> Path | None:
        entries = self.result_registry.list_entries()
        if not entries:
            QMessageBox.information(
                parent, "Export Results", "No results are available yet. Run a workflow step first."
            )
            return None
        labels = [entry.key for entry in entries]
        key, ok = QInputDialog.getItem(parent, "Export Results", "Result", labels, 0, False)
        if not ok or not key:
            return None
        entry = self.result_registry.get(key)
        path, _ = QFileDialog.getSaveFileName(
            parent,
            "Export result",
            str(default_dir / self.safe_result_filename(entry.name, entry.export_formats[0])),
            ";;".join(self.filters_for_entry(entry.export_formats)),
        )
        if not path:
            return None
        output_path = self.path_with_supported_suffix(Path(path), entry.export_formats[0])
        try:
            exported = self.result_registry.export(key, output_path)
            self.log_panel.log(f"Result exported: {exported}")
            return exported.parent
        except ResultRegistryError as exc:
            self.log_panel.log(f"Result export failed: {exc}")
            QMessageBox.warning(parent, "Export Results", str(exc))
            return None

    def save_project(
        self, parent: Any, default_dir: Path, state: ProjectState
    ) -> Path | None:
        path, _ = QFileDialog.getSaveFileName(
            parent, "Save project", str(default_dir / "project.json"), "Project JSON (*.json)"
        )
        if not path:
            return None
        output_path = Path(path)
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        try:
            result_data = {}
            result_entries = []
            for entry in self.result_registry.list_entries():
                result_data[entry.key] = entry.data
                result_entries.append({
                    "key": entry.key,
                    "name": entry.name,
                    "category": entry.category,
                    "export_formats": list(entry.export_formats),
                    "metadata": {str(k): str(v) for k, v in entry.metadata.items()},
                })
            from dataclasses import replace

            state = replace(state, result_entries=result_entries)
            if result_data:
                self.state_service.save_with_results(output_path, state, result_data)
            else:
                self.state_service.save(output_path, state)
            self.log_panel.log(f"Project saved: {output_path}")
            return output_path.parent
        except Exception as exc:
            self.log_panel.log(f"Project save failed: {exc}")
            QMessageBox.warning(parent, "Save Project", str(exc))
            return None

    def choose_and_load_project(self, parent: Any, default_dir: Path) -> ProjectState | None:
        path, _ = QFileDialog.getOpenFileName(
            parent, "Load project", str(default_dir), "Project JSON (*.json);;All files (*.*)"
        )
        if not path:
            return None
        try:
            state = self.state_service.load(path)
            self.loaded_project_path = Path(path)
            self.log_panel.log(f"Project loaded: {path}")
            return state
        except Exception as exc:
            self.log_panel.log(f"Project load failed: {exc}")
            QMessageBox.warning(parent, "Load Project", str(exc))
            return None

    def restore_loaded_results(self, state: ProjectState) -> None:
        if self.loaded_project_path is None or not state.result_entries:
            return
        results = self.state_service.load_results(self.loaded_project_path, state.result_entries)
        for entry in state.result_entries:
            key = entry.get("key", "")
            if key and key in results:
                self.result_registry.register(
                    name=entry.get("name", key),
                    category=entry.get("category", ""),
                    data=results[key],
                    export_formats=tuple(entry.get("export_formats", ("npy",))),
                    metadata=entry.get("metadata", {}),
                )

    def generate_report(
        self,
        parent: Any,
        default_dir: Path,
        state: ProjectState,
        event_log: str,
        process_log: str,
    ) -> Path | None:
        path, _ = QFileDialog.getSaveFileName(
            parent,
            "Generate report",
            str(default_dir / "py4dstem_report.md"),
            "Markdown (*.md)",
        )
        if not path:
            return None
        output_path = Path(path)
        if output_path.suffix.lower() != ".md":
            output_path = output_path.with_suffix(".md")
        try:
            self.report_service.generate_markdown(
                output_path, state, self.result_registry, event_log, process_log
            )
            self.log_panel.log(f"Report generated: {output_path}")
            return output_path.parent
        except Exception as exc:
            self.log_panel.log(f"Report generation failed: {exc}")
            QMessageBox.warning(parent, "Generate Report", str(exc))
            return None

    @staticmethod
    def default_output_dir(recent_export_dir: Path | None, current_file_path: Path | None) -> Path:
        return recent_export_dir or (current_file_path.parent if current_file_path else Path.cwd())

    @staticmethod
    def filters_for_entry(formats: tuple[str, ...]) -> list[str]:
        labels = {
            "npy": "NumPy array (*.npy)",
            "npz": "NumPy archive (*.npz)",
            "png": "PNG image (*.png)",
            "tiff": "TIFF image (*.tif *.tiff)",
        }
        return [labels[item] for item in formats if item in labels]

    @staticmethod
    def safe_result_filename(name: str, extension: str) -> str:
        safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
        suffix = "tif" if extension == "tiff" else extension
        return f"{safe}.{suffix}"

    @staticmethod
    def path_with_supported_suffix(path: Path, extension: str) -> Path:
        suffix = "tif" if extension == "tiff" else extension
        return path if path.suffix else path.with_suffix(f".{suffix}")
