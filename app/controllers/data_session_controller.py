from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from app.services.hdf5_service import Hdf5Service
from app.services.py4dstem_service import Py4DSTEMService


@dataclass(frozen=True)
class DataSelection:
    """Current HDF5 tree selection and preview/target display state."""

    selected_hdf5_path: str | None = None
    selected_node_kind: str | None = None
    preview_kind: str = "Not displayable"
    preview_shape: tuple[int, ...] | None = None
    preview_status: str = "Not rendered"
    previewed_hdf5_path: str | None = None
    active_target_path: str | None = None
    active_source: str | None = None
    last_rendered_path: str | None = None
    displayed: bool = False


class DataSessionController:
    """Owns mutable file/dataset session state and data-provider behavior."""

    def __init__(self, hdf5_service: Hdf5Service, py4dstem_service: Py4DSTEMService) -> None:
        self.hdf5_service = hdf5_service
        self.py4dstem_service = py4dstem_service
        self.current_file: h5py.File | None = None
        self.current_file_path: Path | None = None
        self.current_dataset_path: str | None = None
        self.current_dataset_shape: tuple[int, ...] | None = None
        self.current_4d_source: str | None = None
        self.selection = DataSelection()
        self.current_attrs: dict[str, object] = {}
        self.raw_scan_image_cache_path: str | None = None
        self.raw_scan_image_cache: np.ndarray | None = None
        self.diffraction_cache: OrderedDict[tuple[str, int, int], np.ndarray] = OrderedDict()
        self.diffraction_cache_limit = 16
        self.braggvectors_by_datacube: dict[str, object] = {}
        self.reference_braggvectors_cache: dict[str, object] = {}

    @property
    def selected_hdf5_path(self) -> str | None:
        return self.selection.selected_hdf5_path

    @selected_hdf5_path.setter
    def selected_hdf5_path(self, value: str | None) -> None:
        self._replace_selection(selected_hdf5_path=value)

    @property
    def selected_node_kind(self) -> str | None:
        return self.selection.selected_node_kind

    @selected_node_kind.setter
    def selected_node_kind(self, value: str | None) -> None:
        self._replace_selection(selected_node_kind=value)

    @property
    def selected_preview_kind(self) -> str:
        return self.selection.preview_kind

    @selected_preview_kind.setter
    def selected_preview_kind(self, value: str) -> None:
        self._replace_selection(preview_kind=value)

    @property
    def selected_preview_shape(self) -> tuple[int, ...] | None:
        return self.selection.preview_shape

    @selected_preview_shape.setter
    def selected_preview_shape(self, value: tuple[int, ...] | None) -> None:
        self._replace_selection(preview_shape=value)

    @property
    def preview_status(self) -> str:
        return self.selection.preview_status

    @preview_status.setter
    def preview_status(self, value: str) -> None:
        self._replace_selection(
            preview_status=value,
            displayed=value.startswith("Rendered"),
            last_rendered_path=(
                self.selection.selected_hdf5_path
                if value.startswith("Rendered")
                else self.selection.last_rendered_path
            ),
        )

    def update_selection(
        self,
        hdf5_path: str,
        node_kind: str,
        *,
        preview_kind: str = "Not displayable",
        preview_shape: tuple[int, ...] | None = None,
    ) -> DataSelection:
        self.selection = DataSelection(
            selected_hdf5_path=hdf5_path,
            selected_node_kind=node_kind,
            preview_kind=preview_kind,
            preview_shape=preview_shape,
            active_target_path=self.selection.active_target_path,
            active_source=self.selection.active_source,
        )
        return self.selection

    def clear_selection(self) -> None:
        self.selection = DataSelection()

    def mark_preview_rendered(self, status: str) -> DataSelection:
        self._replace_selection(
            preview_status=status,
            previewed_hdf5_path=self.selection.selected_hdf5_path,
            last_rendered_path=self.selection.selected_hdf5_path,
            displayed=True,
        )
        return self.selection

    def mark_preview_failed(self, status: str) -> DataSelection:
        self._replace_selection(preview_status=status, displayed=False)
        return self.selection

    def mark_rendered(self, path: str | None, status: str) -> DataSelection:
        self._replace_selection(
            preview_status=status,
            last_rendered_path=path,
            displayed=True,
        )
        return self.selection

    def mark_active_target(
        self,
        path: str,
        shape: tuple[int, ...],
        source: str,
    ) -> DataSelection:
        self.current_dataset_path = path
        self.current_dataset_shape = shape
        self.current_4d_source = source
        self._replace_selection(
            active_target_path=path,
            active_source=source,
        )
        return self.selection

    def _replace_selection(self, **changes: object) -> None:
        self.selection = replace(self.selection, **changes)

    def data_browser_selection_info(
        self,
        *,
        path: object,
        node_type: object,
        shape: object,
        dtype: object,
        rx: object,
        ry: object,
    ) -> dict[str, object]:
        return {
            "Selected node": self.selection.selected_hdf5_path or "-",
            "Node type": node_type,
            "Shape": shape,
            "Dtype": dtype,
            "Preview type": self.selection.preview_kind,
            "Preview status": self.selection.preview_status,
            "Previewed node": self.selection.previewed_hdf5_path or "-",
            "Active DataCube": self.selection.active_target_path or self.current_dataset_path or "-",
            "Active source": self.selection.active_source or self.current_4d_source or "-",
            "Last rendered": self.selection.last_rendered_path or "-",
            "Displayed": "Yes" if self.selection.displayed else "No",
            "rx": rx,
            "ry": ry,
            "Path": path,
        }

    def raw_scan_image(
        self,
        hdf5_path: str,
        dataset: h5py.Dataset,
        *,
        memory_budget_bytes: int | None = None,
        progress_callback=None,
    ) -> np.ndarray:
        if self.raw_scan_image_cache_path != hdf5_path or self.raw_scan_image_cache is None:
            self.raw_scan_image_cache = self.hdf5_service.read_4d_scan_image(
                dataset,
                memory_budget_bytes=memory_budget_bytes,
                progress_callback=progress_callback,
            )
            self.raw_scan_image_cache_path = hdf5_path
        return self.raw_scan_image_cache

    def open_file(self, file_path: str | Path) -> h5py.File:
        self.current_file = self.hdf5_service.open_file(file_path)
        self.current_file_path = Path(file_path)
        self.py4dstem_service.defer_open_file(file_path)
        return self.current_file

    def clear_raw_scan_image_cache(self) -> None:
        self.raw_scan_image_cache_path = None
        self.raw_scan_image_cache = None

    def diffraction_pattern(
        self,
        hdf5_path: str,
        dataset: h5py.Dataset,
        rx: int,
        ry: int,
    ) -> np.ndarray:
        key = (hdf5_path, rx, ry)
        cached = self.diffraction_cache.get(key)
        if cached is not None:
            self.diffraction_cache.move_to_end(key)
            return cached
        image = self.hdf5_service.read_4d_diffraction_pattern(dataset, rx=rx, ry=ry)
        self.diffraction_cache[key] = image
        self.diffraction_cache.move_to_end(key)
        while len(self.diffraction_cache) > self.diffraction_cache_limit:
            self.diffraction_cache.popitem(last=False)
        return image

    def virtual_detector_source(self) -> Any | None:
        if self.current_4d_source == "py4dstem":
            return self.py4dstem_service.datacube
        if (
            self.current_4d_source == "hdf5"
            and self.current_file is not None
            and self.current_dataset_path
        ):
            return self.current_file[self.current_dataset_path]
        return None

    def py4dstem_datacube(self) -> Any | None:
        if self.current_4d_source == "py4dstem":
            return self.py4dstem_service.datacube
        return None

    def selected_display_source(self, selected_path: str | None, target_path: str | None) -> Any | None:
        if selected_path and self.current_file is not None:
            try:
                node = self.current_file[selected_path]
                if isinstance(node, h5py.Dataset):
                    return node
            except Exception:
                pass
            if self.py4dstem_service.is_py4dstem_node_path(selected_path):
                try:
                    return self.py4dstem_service.read_datapath(selected_path)
                except Exception:
                    pass
        if target_path and target_path != selected_path:
            if self.py4dstem_service.is_py4dstem_node_path(target_path):
                try:
                    return self.py4dstem_service.read_datapath(target_path)
                except Exception:
                    pass
        return self.virtual_detector_source()

    def target_bright_field_image(self) -> np.ndarray | None:
        if self.current_dataset_path == self.raw_scan_image_cache_path:
            return self.raw_scan_image_cache
        return None

    def cache_scan_overview(self, source: Any, image: np.ndarray) -> bool:
        if self.current_dataset_path is None:
            return False
        active = self.virtual_detector_source()
        active_data = getattr(active, "data", active)
        source_data = getattr(source, "data", source)
        same_hdf5_path = (
            isinstance(source_data, h5py.Dataset)
            and source_data.name == self.current_dataset_path
        )
        if source is not active and source_data is not active_data and not same_hdf5_path:
            return False
        self.raw_scan_image_cache_path = self.current_dataset_path
        self.raw_scan_image_cache = np.asarray(image)
        return True

    def current_4d_shape(self) -> tuple[int, int, int, int] | None:
        if self.current_dataset_shape is None or len(self.current_dataset_shape) != 4:
            return None
        return self.current_dataset_shape

    def assign_role(
        self,
        role: str,
        selected_path: str,
        *,
        workflow_state,
        dpc_service,
        parallax_service,
        clear_workspaces,
        result_registry,
    ) -> None:
        previous_target = workflow_state.dataset_roles.target_datacube
        workflow_state.set_dataset_role(role, selected_path)
        if role != "target_datacube" or previous_target == selected_path:
            return
        dpc_service.reset_dpc_workflow()
        parallax_service.reset()
        if previous_target is None:
            clear_workspaces(exclude_keys={"preprocess"})
        else:
            clear_workspaces()
            result_registry.clear()

    def close_file(self) -> tuple[Path | None, Exception | None]:
        closed_path = self.current_file_path
        error: Exception | None = None
        if self.current_file is not None:
            try:
                self.current_file.close()
            except Exception as exc:
                error = exc
        self.current_file = None
        self.current_file_path = None
        self.current_dataset_path = None
        self.current_dataset_shape = None
        self.current_4d_source = None
        self.clear_selection()
        self.clear_raw_scan_image_cache()
        self.diffraction_cache.clear()
        self.py4dstem_service.close()
        self.braggvectors_by_datacube.clear()
        self.reference_braggvectors_cache.clear()
        return closed_path, error
