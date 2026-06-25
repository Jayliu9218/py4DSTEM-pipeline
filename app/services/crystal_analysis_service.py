from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from app.services.phase_mapping_service import (
    PhaseMappingService,
    PhaseMappingServiceError,
    PhaseMatchParams,
    PhaseMatchResult,
    PhasePlanParams,
)


class CrystalAnalysisServiceError(PhaseMappingServiceError):
    """User-facing crystal-analysis service error."""


@dataclass(frozen=True)
class CrystalAnalysisStageResult:
    stage: str
    images: dict[str, np.ndarray] = field(default_factory=dict)
    metrics: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class CrystalAnalysisRunConfig:
    mode: str = "ROI 128x128"
    roi_size: int = 128


class CrystalAnalysisService(PhaseMappingService):
    """Unified CIF -> phase -> orientation -> grain -> strain workflow.

    The current app already owns Bragg detection and calibration elsewhere.
    This service consumes the calibrated BraggVectors and keeps the crystal
    analysis state in one place so phase, orientation, grain, and strain stages
    share the same CIF library and orientation maps.
    """

    def __init__(self) -> None:
        super().__init__()
        self.run_config = CrystalAnalysisRunConfig()
        self.grain_result: CrystalAnalysisStageResult | None = None
        self.strain_result: CrystalAnalysisStageResult | None = None

    def set_run_config(self, config: CrystalAnalysisRunConfig) -> None:
        self.run_config = config

    def analysis_roi(self, scan_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
        if self.run_config.mode == "Full Dataset":
            return None
        size = max(1, int(self.run_config.roi_size))
        rx, ry = scan_shape
        width = min(size, rx)
        height = min(size, ry)
        rx0 = max((rx - width) // 2, 0)
        ry0 = max((ry - height) // 2, 0)
        return (rx0, rx0 + width, ry0, ry0 + height)

    def calculate_structure_factors(self, params: PhasePlanParams) -> CrystalAnalysisStageResult:
        entries = self.enabled_crystals()
        if not entries:
            raise CrystalAnalysisServiceError("Add at least one crystal to the library first.")
        start = perf_counter()
        metrics: dict[str, object] = {"phases": len(entries), "k_max": params.k_max}
        warnings: list[str] = []
        for _index, entry in entries:
            try:
                crystal = entry.crystal
                setup = getattr(crystal, "setup_diffraction", None)
                if callable(setup):
                    setup(accelerating_voltage=params.accelerating_voltage)
                result = crystal.calculate_structure_factors(
                    k_max=params.k_max,
                    tol_structure_factor=1e-4,
                )
                setattr(entry, "structure_factors", result)
            except Exception as exc:  # noqa: BLE001 - keep other phases usable
                warnings.append(f"{entry.name}: structure factors failed: {exc}")
        return CrystalAnalysisStageResult(
            stage="structure_factors",
            metrics=metrics,
            warnings=tuple(warnings),
            elapsed_seconds=perf_counter() - start,
        )

    def build_orientation_libraries(self, params: PhasePlanParams) -> CrystalAnalysisStageResult:
        elapsed = self.create_multi_phase_plan(params)
        return CrystalAnalysisStageResult(
            stage="simulated_diffraction",
            metrics={
                "phases": len(self.enabled_crystals()),
                "zone_axis_step": params.angle_step_zone_axis,
                "in_plane_step": params.angle_step_in_plane,
                "mode": params.mode,
            },
            elapsed_seconds=elapsed,
        )

    def match_phases(
        self,
        braggvectors: Any | None,
        params: PhaseMatchParams,
    ) -> PhaseMatchResult:
        result = super().match_phases(braggvectors, params)
        composite = self._composite_phase_orientation(result)
        if composite is not None:
            result.images["Composite Phase + Orientation"] = composite
        return result

    def orientation_summary(self) -> CrystalAnalysisStageResult:
        result = self.result
        if result is None:
            raise CrystalAnalysisServiceError("Run phase matching before orientation review.")
        images = {
            "Composite Phase + Orientation": result.images.get(
                "Composite Phase + Orientation",
                result.phase_label_map,
            ),
            "Phase Map": result.phase_label_map,
            "Best Correlation": result.best_correlation_map,
            "Confidence Gap": result.confidence_map,
        }
        return CrystalAnalysisStageResult(
            stage="orientation_matching",
            images={key: value for key, value in images.items() if value is not None},
            metrics={"phases": ", ".join(result.phase_names)},
            warnings=result.warnings,
        )

    def run_grain_analysis(self) -> CrystalAnalysisStageResult:
        if self.result is None:
            raise CrystalAnalysisServiceError("Run phase matching before grain analysis.")
        start = perf_counter()
        warnings: list[str] = []
        metrics: dict[str, object] = {}
        for index, (_entry_index, entry) in enumerate(self.enabled_crystals()):
            orientation_map = self.result.per_phase_orientation[index]
            try:
                cluster = getattr(entry.crystal, "cluster_grains", None)
                if not callable(cluster):
                    warnings.append(f"{entry.name}: grain clustering is unavailable in this py4DSTEM version.")
                    continue
                cluster(orientation_map=orientation_map, progress_bar=False)
                metrics[entry.name] = "clustered"
            except TypeError:
                try:
                    entry.crystal.cluster_grains(progress_bar=False)
                    metrics[entry.name] = "clustered"
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"{entry.name}: grain clustering failed: {exc}")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{entry.name}: grain clustering failed: {exc}")
        self.grain_result = CrystalAnalysisStageResult(
            stage="grain_analysis",
            images={"Phase Map": self.result.phase_label_map},
            metrics=metrics,
            warnings=tuple(warnings),
            elapsed_seconds=perf_counter() - start,
        )
        return self.grain_result

    def run_strain_analysis(self, braggvectors: Any | None) -> CrystalAnalysisStageResult:
        if self.result is None:
            raise CrystalAnalysisServiceError("Run phase matching before strain analysis.")
        if braggvectors is None:
            raise CrystalAnalysisServiceError("Run full BraggVectors before strain analysis.")
        start = perf_counter()
        warnings: list[str] = []
        images: dict[str, np.ndarray] = {}
        metrics: dict[str, object] = {}
        for index, (_entry_index, entry) in enumerate(self.enabled_crystals()):
            orientation_map = self.result.per_phase_orientation[index]
            calculate = getattr(entry.crystal, "calculate_strain", None)
            if not callable(calculate):
                warnings.append(f"{entry.name}: strain calculation is unavailable in this py4DSTEM version.")
                continue
            try:
                strain_map = calculate(braggvectors, orientation_map, progress_bar=False)
                array = self._coerce_strain_array(strain_map)
                if array is not None:
                    images[f"{entry.name} Strain"] = self._phase_masked_array(
                        array,
                        self.result.phase_id_map == index,
                    )
                metrics[entry.name] = "complete"
            except Exception as exc:  # noqa: BLE001 - keep other phases usable
                warnings.append(f"{entry.name}: strain calculation failed: {exc}")
        if not images:
            images["Phase Map"] = self.result.phase_label_map
        self.strain_result = CrystalAnalysisStageResult(
            stage="strain_analysis",
            images=images,
            metrics=metrics,
            warnings=tuple(warnings),
            elapsed_seconds=perf_counter() - start,
        )
        return self.strain_result

    def _composite_phase_orientation(self, result: PhaseMatchResult) -> np.ndarray | None:
        if not any(np.asarray(rgb).ndim == 3 for rgb in result.per_phase_rgb):
            return None
        composite = np.zeros(result.phase_label_map.shape, dtype=float)
        for index, rgb in enumerate(result.per_phase_rgb):
            arr = np.asarray(rgb, dtype=float)
            if arr.ndim != 3 or arr.shape[:2] != result.phase_id_map.shape:
                continue
            mask = result.phase_id_map == index
            composite[mask] = arr[..., :3][mask]
        if not np.any(composite):
            return None
        return composite

    def _coerce_strain_array(self, strain_map: Any) -> np.ndarray | None:
        for attr in ("data", "slices", "strainmap"):
            if hasattr(strain_map, attr):
                try:
                    array = np.asarray(getattr(strain_map, attr), dtype=float)
                    if array.size:
                        return array
                except Exception:
                    pass
        try:
            array = np.asarray(strain_map, dtype=float)
            return array if array.size else None
        except Exception:
            return None

    def _phase_masked_array(self, array: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked = np.asarray(array, dtype=float).copy()
        if masked.shape[:2] != mask.shape:
            return masked
        try:
            masked[~mask] = np.nan
        except Exception:
            pass
        return masked
