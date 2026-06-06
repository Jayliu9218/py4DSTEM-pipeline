from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import h5py
import numpy as np


class VirtualDetectorServiceError(Exception):
    """User-facing virtual detector error."""


@dataclass(frozen=True)
class VirtualDetectorParams:
    mode: str
    center_x: float
    center_y: float
    inner_radius: float
    outer_radius: float


@dataclass(frozen=True)
class VirtualDetectorResult:
    image: np.ndarray
    elapsed_seconds: float
    mode: str


class VirtualDetectorService:
    BRIGHT_FIELD = "Bright Field"
    ANNULAR_DARK_FIELD = "Annular Dark Field"
    CUSTOM_ANNULAR = "Custom Annular Detector"

    def compute(self, source: Any, params: VirtualDetectorParams) -> VirtualDetectorResult:
        start = perf_counter()
        self._validate_params(params)

        try:
            image = self._compute_with_py4dstem(source, params)
        except Exception:
            image = self._compute_with_array(source, params)

        elapsed = perf_counter() - start
        return VirtualDetectorResult(
            image=np.asarray(image),
            elapsed_seconds=elapsed,
            mode=params.mode,
        )

    def _compute_with_py4dstem(self, source: Any, params: VirtualDetectorParams) -> np.ndarray:
        if not hasattr(source, "get_virtual_image"):
            raise VirtualDetectorServiceError("Source does not expose py4DSTEM virtual imaging.")

        if params.mode == self.BRIGHT_FIELD:
            virtual_image = source.get_virtual_image(
                mode="circle",
                geometry=((params.center_x, params.center_y), params.outer_radius),
                centered=False,
                calibrated=False,
                name="virtual_bright_field",
                returncalc=True,
            )
        else:
            virtual_image = source.get_virtual_image(
                mode="annulus",
                geometry=((params.center_x, params.center_y), (params.inner_radius, params.outer_radius)),
                centered=False,
                calibrated=False,
                name="virtual_annular_dark_field",
                returncalc=True,
            )

        return np.asarray(getattr(virtual_image, "data", virtual_image))

    def _compute_with_array(self, source: Any, params: VirtualDetectorParams) -> np.ndarray:
        data = getattr(source, "data", source)
        shape = getattr(data, "shape", None)
        if shape is None or len(tuple(shape)) != 4:
            raise VirtualDetectorServiceError("A 4D DataCube or 4D HDF5 dataset is required.")

        mask = self._build_detector_mask(tuple(int(dim) for dim in shape), params)
        scan_shape = (int(shape[0]), int(shape[1]))
        image = np.zeros(scan_shape, dtype=np.float64)

        if isinstance(data, h5py.Dataset):
            for rx in range(scan_shape[0]):
                for ry in range(scan_shape[1]):
                    image[rx, ry] = np.asarray(data[rx, ry, :, :])[mask].sum()
            return image

        array = np.asarray(data)
        return array[:, :, mask].sum(axis=2)

    def _build_detector_mask(
        self,
        shape: tuple[int, int, int, int],
        params: VirtualDetectorParams,
    ) -> np.ndarray:
        qx_size, qy_size = shape[2], shape[3]
        x, y = np.ogrid[:qx_size, :qy_size]
        radius = np.sqrt((x - params.center_x) ** 2 + (y - params.center_y) ** 2)

        if params.mode == self.BRIGHT_FIELD:
            return radius <= params.outer_radius
        return (radius >= params.inner_radius) & (radius <= params.outer_radius)

    def _validate_params(self, params: VirtualDetectorParams) -> None:
        if params.mode not in {
            self.BRIGHT_FIELD,
            self.ANNULAR_DARK_FIELD,
            self.CUSTOM_ANNULAR,
        }:
            raise VirtualDetectorServiceError(f"Unsupported detector mode: {params.mode}")
        if params.outer_radius <= 0:
            raise VirtualDetectorServiceError("outer_radius must be greater than 0.")
        if params.mode != self.BRIGHT_FIELD and params.inner_radius < 0:
            raise VirtualDetectorServiceError("inner_radius must be 0 or greater.")
        if params.mode != self.BRIGHT_FIELD and params.inner_radius >= params.outer_radius:
            raise VirtualDetectorServiceError("inner_radius must be smaller than outer_radius.")
