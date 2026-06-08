from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np


class Py4DSTEMServiceError(Exception):
    """User-facing py4DSTEM service error."""


@dataclass(frozen=True)
class DataCubeInfo:
    name: str
    datapath: str
    shape: tuple[int, int, int, int]
    scan_shape: tuple[int, int]
    diffraction_shape: tuple[int, int]


@dataclass(frozen=True)
class ProbeGeometry:
    radius: float
    center_x: float
    center_y: float


class Py4DSTEMService:
    def __init__(self) -> None:
        self.file_path: Path | None = None
        self.root: Any | None = None
        self.datacube: Any | None = None
        self.datacube_info: DataCubeInfo | None = None
        self.probe_geometry: ProbeGeometry | None = None

    def open_file(self, file_path: str | Path) -> None:
        self.close()
        py4DSTEM = self._py4dstem()
        path = Path(file_path)
        if not path.exists():
            raise Py4DSTEMServiceError(f"File does not exist: {path}")

        self.file_path = path
        try:
            self.root = py4DSTEM.read(path, tree=True, verbose=False)
        except Exception as exc:
            self.root = None
            raise Py4DSTEMServiceError(
                "py4DSTEM could not read this file as a py4DSTEM object. "
                "The raw HDF5 tree can still be browsed."
            ) from exc

    def close(self) -> None:
        self.file_path = None
        self.root = None
        self.datacube = None
        self.datacube_info = None
        self.probe_geometry = None

    def defer_open_file(self, file_path: str | Path) -> None:
        path = Path(file_path)
        if not path.exists():
            raise Py4DSTEMServiceError(f"File does not exist: {path}")
        self.file_path = path
        self.root = None
        self.datacube = None
        self.datacube_info = None
        self.probe_geometry = None

    def load_datacube(self, datapath: str) -> DataCubeInfo:
        if self.file_path is None:
            raise Py4DSTEMServiceError("No file is open.")
        py4DSTEM = self._py4dstem()

        try:
            obj = py4DSTEM.read(
                filepath=self.file_path,
                datapath=datapath,
                tree=False,
                verbose=False,
            )
        except Exception as exc:
            raise Py4DSTEMServiceError(
                "py4DSTEM could not load this node as a DataCube. "
                "Select a py4DSTEM DataCube group or a 4D HDF5 dataset."
            ) from exc

        if not self.is_datacube(obj):
            pass
            """
            raise Py4DSTEMServiceError(
                f"The selected py4DSTEM object is {type(obj).__name__}, not a DataCube."
            )
            """

        shape = self.get_datacube_shape(obj)
        info = DataCubeInfo(
            name=str(getattr(obj, "name", "") or Path(datapath).name or "DataCube"),
            datapath=datapath,
            shape=shape,
            scan_shape=shape[:2],
            diffraction_shape=shape[2:],
        )

        self.datacube = obj
        self.datacube_info = info
        self.probe_geometry = None
        return info

    def load_raw_4d_array(self, data: Any, datapath: str) -> DataCubeInfo:
        shape_value = getattr(data, "shape", None)
        if shape_value is None:
            shape_value = np.asarray(data).shape
        shape = tuple(int(dim) for dim in shape_value)
        if len(shape) != 4:
            raise Py4DSTEMServiceError(f"Expected a 4D array, got shape {shape}.")

        self.datacube = None
        self.probe_geometry = None
        info = DataCubeInfo(
            name=Path(datapath).name or "4D dataset",
            datapath=datapath,
            shape=shape,
            scan_shape=shape[:2],
            diffraction_shape=shape[2:],
        )
        self.datacube_info = info
        return info

    def is_datacube(self, obj: Any) -> bool:
        py4DSTEM = self._py4dstem()
        return isinstance(obj, py4DSTEM.DataCube) or (
            hasattr(obj, "data") and self._shape_is_4d(getattr(obj, "shape", None))
        )

    def get_datacube_shape(self, datacube: Any | None = None) -> tuple[int, int, int, int]:
        cube = datacube if datacube is not None else self.datacube
        if cube is None:
            raise Py4DSTEMServiceError("No DataCube is loaded.")

        shape = getattr(cube, "shape", None)
        if shape is None and hasattr(cube, "data"):
            shape = cube.data.shape
        if not self._shape_is_4d(shape):
            raise Py4DSTEMServiceError(f"Expected DataCube shape with 4 axes, got {shape}.")
        return tuple(int(dim) for dim in shape)

    def get_scan_image(self) -> np.ndarray:
        if self.datacube is None:
            raise Py4DSTEMServiceError("No py4DSTEM DataCube is loaded.")

        try:
            virtual_image = self.datacube.get_virtual_image(
                mode="all",
                name="scan_image",
                returncalc=True,
            )
            return np.asarray(getattr(virtual_image, "data", virtual_image))
        except Exception:
            data = np.asarray(self.datacube.data)
            return data.sum(axis=(2, 3))

    def get_diffraction_pattern(self, rx: int, ry: int) -> np.ndarray:
        if self.datacube is None:
            raise Py4DSTEMServiceError("No py4DSTEM DataCube is loaded.")

        shape = self.get_datacube_shape()
        self._validate_scan_coordinates(rx, ry, shape[:2])
        return np.asarray(self.datacube.data[rx, ry, :, :])

    def measure_probe_geometry(self) -> ProbeGeometry:
        if self.datacube is None:
            raise Py4DSTEMServiceError("No py4DSTEM DataCube is loaded.")

        try:
            dp_mean = np.asarray(self.datacube.get_dp_mean().data)
            radius, center_x, center_y = self.datacube.get_probe_size(
                dp=dp_mean,
                plot=False,
            )
        except Exception as exc:
            raise Py4DSTEMServiceError(f"Could not measure probe geometry: {exc}") from exc

        geometry = ProbeGeometry(
            radius=float(radius),
            center_x=float(center_x),
            center_y=float(center_y),
        )
        self.probe_geometry = geometry
        return geometry

    def describe_current_datacube(self) -> dict[str, object]:
        if self.datacube_info is None:
            return {}
        return {
            "name": self.datacube_info.name,
            "datapath": self.datacube_info.datapath,
            "shape": self.datacube_info.shape,
            "scan_shape": self.datacube_info.scan_shape,
            "diffraction_shape": self.datacube_info.diffraction_shape,
        }

    def _shape_is_4d(self, shape: Any) -> bool:
        return shape is not None and len(tuple(shape)) == 4

    def _validate_scan_coordinates(
        self,
        rx: int,
        ry: int,
        scan_shape: tuple[int, int],
    ) -> None:
        if rx < 0 or rx >= scan_shape[0]:
            raise Py4DSTEMServiceError(
                f"rx={rx} is out of range. Valid range is 0 to {scan_shape[0] - 1}."
            )
        if ry < 0 or ry >= scan_shape[1]:
            raise Py4DSTEMServiceError(
                f"ry={ry} is out of range. Valid range is 0 to {scan_shape[1] - 1}."
            )

    def _py4dstem(self):
        try:
            return import_module("py4DSTEM")
        except Exception as exc:
            raise Py4DSTEMServiceError(
                "py4DSTEM could not be imported in this environment. "
                "The HDF5 file is open, but py4DSTEM-specific loading is unavailable. "
                "Raw HDF5 browsing is still available."
            ) from exc
