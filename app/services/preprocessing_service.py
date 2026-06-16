from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from app.services.computation_task import ComputationTask
from app.services.array_reduction import max_diffraction, mean_diffraction, scan_sum_with_progress


class PreprocessingServiceError(Exception):
    """User-facing preprocessing error."""


@dataclass(frozen=True)
class HotPixelParams:
    threshold: float = 8.0


@dataclass(frozen=True)
class HotPixelPreview:
    before_mean: np.ndarray
    after_mean: np.ndarray
    hot_pixel_mask: np.ndarray
    hot_pixel_count: int
    elapsed_seconds: float


class PreprocessingService:
    def show_data_task(
        self,
        source: Any,
        *,
        memory_budget_mb: int,
    ) -> ComputationTask:
        data = source if isinstance(source, np.ndarray) else getattr(source, "data", source)
        shape = tuple(int(value) for value in getattr(data, "shape", ()))
        dtype = getattr(data, "dtype", None)
        result_key = f"show_data:{shape}:{dtype}"
        return ComputationTask(
            name="Show Data",
            operation=lambda progress: self.show_data(
                source,
                memory_budget_bytes=max(int(memory_budget_mb), 1) * 1024 * 1024,
                progress_callback=progress,
            ),
            memory_budget_mb=memory_budget_mb,
            result_key=result_key,
            status_message="Preparing data display",
            parameters={
                "shape": shape or "-",
                "dtype": str(dtype) if dtype is not None else "-",
            },
        )

    def display_data(self, source: Any) -> dict[str, np.ndarray]:
        data = source if isinstance(source, np.ndarray) else getattr(source, "data", source)
        shape = getattr(data, "shape", None)
        if shape is None:
            raise PreprocessingServiceError("The selected object has no displayable array data.")
        shape = tuple(int(value) for value in shape)
        if len(shape) == 4:
            rx, ry = shape[0] // 2, shape[1] // 2
            qx, qy = shape[2] // 2, shape[3] // 2
            return {
                "Center diffraction pattern": np.asarray(data[rx, ry, :, :]),
                "Center real-space slice": np.asarray(data[:, :, qx, qy]),
            }
        if len(shape) == 2:
            return {"Selected diffraction slice": np.asarray(data[...])}
        raise PreprocessingServiceError(
            f"Selected data must be a 4D DataCube or 2D DiffractionSlice, got shape {shape}."
        )

    def show_data(
        self,
        source: Any,
        *,
        memory_budget_bytes: int,
        progress_callback=None,
    ) -> dict[str, np.ndarray]:
        data = source if isinstance(source, np.ndarray) else getattr(source, "data", source)
        shape = tuple(int(value) for value in getattr(data, "shape", ()))
        if len(shape) == 2:
            return self.display_data(source)
        if len(shape) != 4:
            raise PreprocessingServiceError(
                f"Selected data must be a 4D DataCube or 2D DiffractionSlice, got shape {shape}."
            )
        emit = progress_callback or (lambda _message, _fraction: None)
        rx, ry = shape[0] // 2, shape[1] // 2
        qx, qy = shape[2] // 2, shape[3] // 2
        results = {
            "Central diffraction pattern": np.asarray(data[rx, ry]),
            "Central real-space slice": np.asarray(data[:, :, qx, qy]),
        }
        results["Scan overview"] = scan_sum_with_progress(
            data,
            memory_budget_bytes=memory_budget_bytes,
            progress_callback=lambda message, fraction: emit(message, fraction * 0.6),
        )
        results["Mean diffraction pattern"] = mean_diffraction(
            data,
            memory_budget_bytes=memory_budget_bytes,
            progress_callback=lambda message, fraction: emit(message, 0.6 + fraction * 0.2),
        )
        results["Maximum diffraction pattern"] = max_diffraction(
            data,
            memory_budget_bytes=memory_budget_bytes,
            progress_callback=lambda message, fraction: emit(message, 0.8 + fraction * 0.2),
        )
        emit("Data display ready", 1.0)
        return results

    def preview_hot_pixels(self, source: Any, params: HotPixelParams) -> HotPixelPreview:
        data = source if isinstance(source, np.ndarray) else getattr(source, "data", source)
        shape = getattr(data, "shape", None)
        if shape is None or len(tuple(shape)) != 4:
            raise PreprocessingServiceError("A 4D DataCube is required for hot-pixel filtering.")
        if params.threshold <= 1:
            raise PreprocessingServiceError("Hot-pixel threshold must be greater than 1.")
        start = perf_counter()
        before = mean_diffraction(data)
        neighborhood = self._local_median(before)
        mask = before > params.threshold * np.maximum(neighborhood, np.finfo(float).eps)
        after = before.copy()
        after[mask] = neighborhood[mask]
        return HotPixelPreview(before, after, mask, int(mask.sum()), perf_counter() - start)

    def apply_hot_pixels(self, source: Any, preview: HotPixelPreview) -> int:
        if not preview.hot_pixel_count:
            return 0
        if hasattr(source, "filter_hot_pixels"):
            source.filter_hot_pixels(thresh=self._estimated_threshold(preview))
            return preview.hot_pixel_count
        data = source if isinstance(source, np.ndarray) else getattr(source, "data", source)
        array = np.asarray(data)
        neighborhood = self._local_median(mean_diffraction(data))
        array[..., preview.hot_pixel_mask] = neighborhood[preview.hot_pixel_mask]
        return preview.hot_pixel_count

    def basic_diagnostics(self, source: Any) -> dict[str, np.ndarray]:
        data = source if isinstance(source, np.ndarray) else getattr(source, "data", source)
        shape = tuple(int(value) for value in getattr(data, "shape", ()))
        if len(shape) != 4:
            raise PreprocessingServiceError("A 4D DataCube is required.")
        rx, ry = shape[0] // 2, shape[1] // 2
        qx, qy = shape[2] // 2, shape[3] // 2
        return {
            "Central diffraction pattern": np.asarray(data[rx, ry]),
            "Central real-space slice": np.asarray(data[:, :, qx, qy]),
            "Mean diffraction pattern": mean_diffraction(data),
            "Maximum diffraction pattern": max_diffraction(data),
        }

    def _local_median(self, image: np.ndarray) -> np.ndarray:
        padded = np.pad(image, 1, mode="reflect")
        stack = [
            padded[x : x + image.shape[0], y : y + image.shape[1]]
            for x in range(3)
            for y in range(3)
            if (x, y) != (1, 1)
        ]
        return np.median(np.stack(stack), axis=0)

    def _estimated_threshold(self, preview: HotPixelPreview) -> float:
        ratios = preview.before_mean[preview.hot_pixel_mask] / np.maximum(
            preview.after_mean[preview.hot_pixel_mask], np.finfo(float).eps
        )
        return float(np.min(ratios)) if ratios.size else 8.0
