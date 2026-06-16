from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from app.services.computation_task import ComputationTask
from app.services.array_reduction import (
    detector_sum,
    masked_scan_mean,
    mean_diffraction,
    scan_sum,
)


class VirtualDetectorServiceError(Exception):
    """User-facing virtual detector error."""


@dataclass(frozen=True)
class VirtualDetectorParams:
    mode: str
    center_x: float
    center_y: float
    inner_radius: float
    outer_radius: float
    roi_rx_start: int = 0
    roi_rx_end: int = 1
    roi_ry_start: int = 0
    roi_ry_end: int = 1
    roi_mode: str = "rectangle"
    roi_center_x: float = 0
    roi_center_y: float = 0
    roi_radius: float = 1


@dataclass(frozen=True)
class VirtualDetectorResult:
    image: np.ndarray
    elapsed_seconds: float
    mode: str
    detector_preview: np.ndarray | None = None
    detector_overlay: dict[str, float | str] | None = None
    real_space_preview: np.ndarray | None = None
    real_space_overlay: dict[str, float | str] | None = None


class VirtualDetectorService:
    BRIGHT_FIELD = "Bright Field"
    ANNULAR_DARK_FIELD = "Annular Dark Field"
    CUSTOM_ANNULAR = "Custom Annular Detector"
    OFF_AXIS_DARK_FIELD = "Off-axis Dark Field"
    VIRTUAL_DIFFRACTION = "Virtual Diffraction"

    def compute_task(self, source: Any, params: VirtualDetectorParams) -> ComputationTask:
        data = source if isinstance(source, np.ndarray) else getattr(source, "data", source)
        shape = tuple(int(dim) for dim in getattr(data, "shape", ()))
        result_key = (
            "virtual_detector:"
            f"{params.mode}:{shape}:"
            f"{params.center_x}:{params.center_y}:{params.inner_radius}:{params.outer_radius}:"
            f"{params.roi_mode}:{params.roi_rx_start}:{params.roi_rx_end}:"
            f"{params.roi_ry_start}:{params.roi_ry_end}:{params.roi_center_x}:"
            f"{params.roi_center_y}:{params.roi_radius}"
        )
        return ComputationTask(
            name="Virtual detector",
            operation=lambda _progress: self.compute(source, params),
            result_key=result_key,
            status_message=f"Preparing virtual detector: {params.mode}",
            parameters={
                "mode": params.mode,
                "shape": shape or "-",
                "center_x": params.center_x,
                "center_y": params.center_y,
                "inner_radius": params.inner_radius,
                "outer_radius": params.outer_radius,
                "roi_mode": params.roi_mode,
            },
        )

    def compute(self, source: Any, params: VirtualDetectorParams) -> VirtualDetectorResult:
        start = perf_counter()
        self._validate_params(params)
        if params.mode == self.VIRTUAL_DIFFRACTION:
            image = self.compute_virtual_diffraction(source, params)
            preview = self._real_space_preview(source)
            overlay = (
                {"kind": "circle", "x": params.roi_center_x, "y": params.roi_center_y, "r": params.roi_radius}
                if params.roi_mode == "circle"
                else {
                    "kind": "rect",
                    "x0": params.roi_rx_start,
                    "x1": params.roi_rx_end,
                    "y0": params.roi_ry_start,
                    "y1": params.roi_ry_end,
                }
            )
            return VirtualDetectorResult(
                np.asarray(image),
                perf_counter() - start,
                params.mode,
                real_space_preview=preview,
                real_space_overlay=overlay,
            )

        try:
            image = self._compute_with_py4dstem(source, params)
        except Exception:
            image = self._compute_with_array(source, params)

        elapsed = perf_counter() - start
        return VirtualDetectorResult(
            image=np.asarray(image),
            elapsed_seconds=elapsed,
            mode=params.mode,
            detector_preview=self._detector_preview(source),
            detector_overlay=self._detector_overlay(params),
        )

    def _detector_preview(self, source: Any) -> np.ndarray:
        if hasattr(source, "get_dp_mean"):
            result = source.get_dp_mean()
            return np.asarray(getattr(result, "data", result))
        data = source if isinstance(source, np.ndarray) else getattr(source, "data", source)
        if getattr(data, "ndim", None) != 4:
            raise VirtualDetectorServiceError("A 4D DataCube is required for detector preview.")
        return mean_diffraction(data)

    def _detector_overlay(self, params: VirtualDetectorParams) -> dict[str, float | str]:
        if params.mode in {self.BRIGHT_FIELD, self.OFF_AXIS_DARK_FIELD}:
            return {
                "kind": "circle",
                "x": params.center_x,
                "y": params.center_y,
                "r": params.outer_radius,
            }
        return {
            "kind": "ring",
            "x": params.center_x,
            "y": params.center_y,
            "inner_radius": params.inner_radius,
            "outer_radius": params.outer_radius,
        }

    def _compute_with_py4dstem(self, source: Any, params: VirtualDetectorParams) -> np.ndarray:
        if not hasattr(source, "get_virtual_image"):
            raise VirtualDetectorServiceError("Source does not expose py4DSTEM virtual imaging.")

        if params.mode in {self.BRIGHT_FIELD, self.OFF_AXIS_DARK_FIELD}:
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

    def compute_virtual_diffraction(self, source: Any, params: VirtualDetectorParams) -> np.ndarray:
        data = source if isinstance(source, np.ndarray) else getattr(source, "data", source)
        shape = tuple(int(dim) for dim in getattr(data, "shape", ()))
        if len(shape) != 4:
            raise VirtualDetectorServiceError("A 4D DataCube is required for virtual diffraction.")
        if params.roi_mode == "circle":
            if params.roi_radius <= 0:
                raise VirtualDetectorServiceError("Circular real-space ROI radius must be positive.")
            rx, ry = np.ogrid[:shape[0], :shape[1]]
            mask = (rx - params.roi_center_x) ** 2 + (ry - params.roi_center_y) ** 2 < params.roi_radius**2
        else:
            if not (
                0 <= params.roi_rx_start < params.roi_rx_end <= shape[0]
                and 0 <= params.roi_ry_start < params.roi_ry_end <= shape[1]
            ):
                raise VirtualDetectorServiceError(f"Real-space ROI must fit inside scan shape {shape[:2]}.")
            mask = np.zeros(shape[:2], dtype=bool)
            mask[params.roi_rx_start:params.roi_rx_end, params.roi_ry_start:params.roi_ry_end] = True
        if not mask.any():
            raise VirtualDetectorServiceError("Real-space ROI contains no scan positions.")
        if hasattr(source, "get_virtual_diffraction"):
            result = source.get_virtual_diffraction(method="mean", mask=mask, returncalc=True)
            return np.asarray(getattr(result, "data", result))
        return masked_scan_mean(data, mask)

    def _real_space_preview(self, source: Any) -> np.ndarray:
        data = source if isinstance(source, np.ndarray) else getattr(source, "data", source)
        if hasattr(source, "get_virtual_image"):
            result = source.get_virtual_image(mode="all", returncalc=True)
            return np.asarray(getattr(result, "data", result))
        return scan_sum(data, dtype=np.float64)

    def _compute_with_array(self, source: Any, params: VirtualDetectorParams) -> np.ndarray:
        data = source if isinstance(source, np.ndarray) else getattr(source, "data", source)
        shape = getattr(data, "shape", None)
        if shape is None or len(tuple(shape)) != 4:
            raise VirtualDetectorServiceError("A 4D DataCube or 4D HDF5 dataset is required.")

        mask = self._build_detector_mask(tuple(int(dim) for dim in shape), params)
        return detector_sum(data, mask, dtype=np.float64)

    def _build_detector_mask(
        self,
        shape: tuple[int, int, int, int],
        params: VirtualDetectorParams,
    ) -> np.ndarray:
        qx_size, qy_size = shape[2], shape[3]
        x, y = np.ogrid[:qx_size, :qy_size]
        radius_squared = (x - params.center_x) ** 2 + (y - params.center_y) ** 2

        if params.mode in {self.BRIGHT_FIELD, self.OFF_AXIS_DARK_FIELD}:
            return radius_squared <= params.outer_radius**2
        return (radius_squared >= params.inner_radius**2) & (
            radius_squared <= params.outer_radius**2
        )

    def _validate_params(self, params: VirtualDetectorParams) -> None:
        if params.mode not in {
            self.BRIGHT_FIELD,
            self.ANNULAR_DARK_FIELD,
            self.CUSTOM_ANNULAR,
            self.OFF_AXIS_DARK_FIELD,
            self.VIRTUAL_DIFFRACTION,
        }:
            raise VirtualDetectorServiceError(f"Unsupported detector mode: {params.mode}")
        if params.mode != self.VIRTUAL_DIFFRACTION and params.outer_radius <= 0:
            raise VirtualDetectorServiceError("outer_radius must be greater than 0.")
        if params.mode not in {self.BRIGHT_FIELD, self.OFF_AXIS_DARK_FIELD, self.VIRTUAL_DIFFRACTION} and params.inner_radius < 0:
            raise VirtualDetectorServiceError("inner_radius must be 0 or greater.")
        if params.mode not in {self.BRIGHT_FIELD, self.OFF_AXIS_DARK_FIELD, self.VIRTUAL_DIFFRACTION} and params.inner_radius >= params.outer_radius:
            raise VirtualDetectorServiceError("inner_radius must be smaller than outer_radius.")
