from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from app.services.hdf5_service import Hdf5Service
from app.services.py4dstem_service import Py4DSTEMService


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
        self.selected_hdf5_path: str | None = None
        self.selected_node_kind: str | None = None
        self.current_attrs: dict[str, object] = {}
        self.raw_scan_image_cache_path: str | None = None
        self.raw_scan_image_cache: np.ndarray | None = None
        self.diffraction_cache: OrderedDict[tuple[str, int, int], np.ndarray] = OrderedDict()
        self.diffraction_cache_limit = 16
        self.braggvectors_by_datacube: dict[str, object] = {}
        self.reference_braggvectors_cache: dict[str, object] = {}

    def raw_scan_image(self, hdf5_path: str, dataset: h5py.Dataset) -> np.ndarray:
        if self.raw_scan_image_cache_path != hdf5_path or self.raw_scan_image_cache is None:
            self.raw_scan_image_cache = self.hdf5_service.read_4d_scan_image(dataset)
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
            try:
                return self.py4dstem_service.read_datapath(selected_path)
            except Exception:
                pass
        if target_path and target_path != selected_path:
            try:
                return self.py4dstem_service.read_datapath(target_path)
            except Exception:
                pass
        return self.virtual_detector_source()

    def target_bright_field_image(self) -> np.ndarray | None:
        try:
            if self.current_4d_source == "py4dstem":
                return self.py4dstem_service.get_scan_image()
            if (
                self.current_4d_source == "hdf5"
                and self.current_file is not None
                and self.current_dataset_path
            ):
                node = self.current_file[self.current_dataset_path]
                if isinstance(node, h5py.Dataset) and len(tuple(node.shape)) == 4:
                    return self.raw_scan_image(self.current_dataset_path, node)
        except Exception:
            return None
        return None

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
        self.clear_raw_scan_image_cache()
        self.diffraction_cache.clear()
        self.py4dstem_service.close()
        self.braggvectors_by_datacube.clear()
        self.reference_braggvectors_cache.clear()
        return closed_path, error
