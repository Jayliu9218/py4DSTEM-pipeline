from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


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
        except (OSError, ValueError, RuntimeError) as exc:
            self.root = None
            raise Py4DSTEMServiceError(
                "py4DSTEM could not read this file as a py4DSTEM object. "
                "The raw HDF5 tree can still be browsed."
            ) from exc
        except Exception as exc:
            self.root = None
            logger.exception("Unexpected error reading py4DSTEM file")
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
        canonical_datapath = self._resolve_datacube_path(datapath)
        if canonical_datapath is None:
            raise Py4DSTEMServiceError(
                "This node is not a py4DSTEM DataCube. "
                "Raw 4D HDF5 datasets remain available for browsing."
            )
        if (
            self.datacube is not None
            and self.datacube_info is not None
            and self.datacube_info.datapath == canonical_datapath
        ):
            return self.datacube_info
        py4DSTEM = self._py4dstem()

        try:
            obj = py4DSTEM.read(
                filepath=self.file_path,
                datapath=canonical_datapath,
                tree=False,
                verbose=False,
            )
        except (AttributeError, OSError, ValueError, RuntimeError) as exc:
            raise Py4DSTEMServiceError(
                "py4DSTEM could not load this node as a DataCube. "
                "Select a py4DSTEM DataCube group or a 4D HDF5 dataset."
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error loading py4DSTEM DataCube")
            raise Py4DSTEMServiceError(
                "py4DSTEM could not load this node as a DataCube. "
                "Select a py4DSTEM DataCube group or a 4D HDF5 dataset."
            ) from exc

        if not self.is_datacube(obj):
            logger.warning(
                "Loaded object is %s, not a DataCube. "
                "Attempting to continue, but some operations may fail.",
                type(obj).__name__,
            )

        shape = self.get_datacube_shape(obj)
        info = DataCubeInfo(
            name=str(getattr(obj, "name", "") or Path(canonical_datapath).name or "DataCube"),
            datapath=canonical_datapath,
            shape=shape,
            scan_shape=shape[:2],
            diffraction_shape=shape[2:],
        )

        self.datacube = obj
        self.datacube_info = info
        self.probe_geometry = None
        return info

    def _resolve_datacube_path(self, datapath: str) -> str | None:
        if self.file_path is None:
            return None
        try:
            with h5py.File(self.file_path, "r") as source:
                node = source[datapath]
                if self._python_class(node) == "DataCube":
                    return node.name
                if (
                    isinstance(node, h5py.Dataset)
                    and node.ndim == 4
                    and node.name.rsplit("/", 1)[-1] == "data"
                    and self._python_class(node.parent) == "DataCube"
                ):
                    return node.parent.name
        except (KeyError, OSError, ValueError):
            return None
        return None

    @staticmethod
    def _python_class(node: h5py.Group | h5py.Dataset) -> str | None:
        value = node.attrs.get("python_class")
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value) if value is not None else None

    def read_datapath(self, datapath: str) -> Any:
        if self.file_path is None:
            raise Py4DSTEMServiceError("No file is open.")
        py4DSTEM = self._py4dstem()
        try:
            return py4DSTEM.read(
                filepath=self.file_path,
                datapath=datapath,
                tree=False,
                verbose=False,
            )
        except (AttributeError, OSError, ValueError, RuntimeError) as exc:
            raise Py4DSTEMServiceError(f"py4DSTEM could not load reference node {datapath}.") from exc
        except Exception as exc:
            logger.exception("Unexpected error loading py4DSTEM reference node %s", datapath)
            raise Py4DSTEMServiceError(f"py4DSTEM could not load reference node {datapath}.") from exc

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
        datacube_type = getattr(py4DSTEM, "DataCube", None)
        return (
            isinstance(datacube_type, type)
            and isinstance(obj, datacube_type)
        ) or (
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
        except (AttributeError, TypeError, ValueError, RuntimeError):
            logger.debug("get_virtual_image failed, falling back to raw sum", exc_info=True)
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
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise Py4DSTEMServiceError(f"Could not measure probe geometry: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected error measuring probe geometry")
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
        except ImportError as exc:
            raise Py4DSTEMServiceError(
                "py4DSTEM could not be imported in this environment. "
                "The HDF5 file is open, but py4DSTEM-specific loading is unavailable. "
                "Raw HDF5 browsing is still available."
            ) from exc
