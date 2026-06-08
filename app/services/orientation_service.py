from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np


class OrientationServiceError(Exception):
    """User-facing orientation analysis error."""


@dataclass(frozen=True)
class OrientationPlanParams:
    accelerating_voltage: float = 300000
    k_max: float = 1.5
    angle_step_zone_axis: float = 2
    angle_step_in_plane: float = 2
    corr_kernel_size: float = 0.08
    sigma_excitation_error: float = 0.02
    cuda: bool = False


@dataclass(frozen=True)
class OrientationMatchParams:
    num_matches_return: int = 1
    min_angle_between_matches_deg: float = 5
    min_number_peaks: int = 3
    inversion_symmetry: bool = True


@dataclass(frozen=True)
class OrientationQualityResult:
    maps: dict[str, np.ndarray]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class OrientationResult:
    orientation_map: Any
    preview: np.ndarray
    quality: OrientationQualityResult
    elapsed_seconds: float


class OrientationService:
    def __init__(self) -> None:
        self.crystal: Any | None = None
        self.orientation_map: Any | None = None

    def load_crystal(self, cif_path: str | Path) -> str:
        path = Path(cif_path)
        if not path.exists():
            raise OrientationServiceError(f"CIF file does not exist: {path}")
        try:
            py4DSTEM = self._py4dstem()
            self.crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(path)
        except Exception as exc:
            raise OrientationServiceError(f"Could not load crystal structure: {exc}") from exc
        self.orientation_map = None
        return str(path)

    def create_plan(self, params: OrientationPlanParams) -> float:
        crystal = self._require_crystal()
        start = perf_counter()
        try:
            crystal.setup_diffraction(accelerating_voltage=params.accelerating_voltage)
            crystal.calculate_structure_factors(k_max=params.k_max, tol_structure_factor=1e-4)
            crystal.orientation_plan(
                zone_axis_range="auto",
                angle_step_zone_axis=params.angle_step_zone_axis,
                angle_step_in_plane=params.angle_step_in_plane,
                accel_voltage=params.accelerating_voltage,
                corr_kernel_size=params.corr_kernel_size,
                sigma_excitation_error=params.sigma_excitation_error,
                power_radial=1.0,
                power_intensity=0.0,
                power_intensity_experiment=0.0,
                CUDA=params.cuda,
                progress_bar=False,
            )
        except Exception as exc:
            raise OrientationServiceError(f"Orientation plan failed: {exc}") from exc
        return perf_counter() - start

    def match(
        self,
        braggvectors: Any | None,
        params: OrientationMatchParams,
    ) -> OrientationResult:
        crystal = self._require_crystal()
        if braggvectors is None:
            raise OrientationServiceError("Run full BraggVectors before orientation matching.")
        calstate = getattr(braggvectors, "calstate", {})
        calibration_warning = self._calibration_warning(calstate)
        start = perf_counter()
        try:
            orientation_map = crystal.match_orientations(
                braggvectors,
                num_matches_return=params.num_matches_return,
                min_angle_between_matches_deg=params.min_angle_between_matches_deg,
                min_number_peaks=params.min_number_peaks,
                inversion_symmetry=params.inversion_symmetry,
                progress_bar=False,
            )
            images, fig, _ = crystal.plot_orientation_maps(
                orientation_map=orientation_map,
                orientation_ind=0,
                corr_normalize=True,
                show_axes=True,
                returnfig=True,
                progress_bar=False,
            )
            preview = np.asarray(images[:, :, :, 0]).copy()
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception as exc:
            raise OrientationServiceError(f"Orientation matching failed: {exc}") from exc
        self.orientation_map = orientation_map
        quality = self.orientation_quality(orientation_map, braggvectors, preview)
        if calibration_warning:
            quality = OrientationQualityResult(
                maps=quality.maps,
                warnings=(calibration_warning, *quality.warnings),
            )
        return OrientationResult(orientation_map, preview, quality, perf_counter() - start)

    def orientation_quality(
        self,
        orientation_map: Any,
        braggvectors: Any | None,
        preview: np.ndarray | None = None,
    ) -> OrientationQualityResult:
        maps: dict[str, np.ndarray] = {}
        warnings: list[str] = []
        if preview is not None:
            maps["Orientation RGB"] = np.asarray(preview)

        for label, candidates in [
            ("Correlation Score", ("corr", "corr_map", "correlation", "correlation_score")),
            ("Reliability", ("reliability", "confidence", "confidence_map")),
            ("Ambiguity", ("ambiguity", "ambiguity_map", "corr_gap", "correlation_gap")),
        ]:
            image = self._first_2d_array(orientation_map, candidates)
            if image is None:
                warnings.append(f"{label} map is not exposed by this py4DSTEM result.")
            else:
                maps[label] = image

        peak_count = self._peak_count_map(braggvectors)
        if peak_count is None:
            warnings.append("Peak Count map is not available because BraggVectors.raw is missing.")
        else:
            maps["Peak Count"] = peak_count

        return OrientationQualityResult(maps=maps, warnings=tuple(warnings))

    def _require_crystal(self) -> Any:
        if self.crystal is None:
            raise OrientationServiceError("Load a CIF crystal structure first.")
        return self.crystal

    def _calibration_warning(self, calstate: Any) -> str:
        missing = [
            label
            for name, label in [
                ("center", "origin"),
                ("ellipse", "ellipse"),
                ("pixel", "pixel"),
                ("rotate", "rotation"),
            ]
            if not bool(getattr(calstate, "get", lambda _name, _default=False: False)(name, False))
        ]
        if not missing:
            return ""
        return (
            "Calibration is incomplete; continuing orientation matching with lower expected "
            f"accuracy. Missing/applied-off corrections: {', '.join(missing)}."
        )

    def _first_2d_array(self, obj: Any, names: tuple[str, ...]) -> np.ndarray | None:
        for name in names:
            value = getattr(obj, name, None)
            if value is None and hasattr(obj, "get"):
                try:
                    value = obj.get(name)
                except Exception:
                    value = None
            if value is None:
                continue
            try:
                array = np.asarray(getattr(value, "data", value), dtype=float)
            except Exception:
                continue
            if array.ndim == 2:
                return array
            if array.ndim == 3:
                return array[:, :, 0]
        return None

    def _peak_count_map(self, braggvectors: Any | None) -> np.ndarray | None:
        raw = getattr(braggvectors, "raw", None)
        if raw is None or not hasattr(raw, "shape"):
            return None
        shape = tuple(int(dim) for dim in raw.shape[:2])
        counts = np.zeros(shape, dtype=float)
        try:
            for rx in range(shape[0]):
                for ry in range(shape[1]):
                    data = getattr(raw[rx, ry], "data", raw[rx, ry])
                    counts[rx, ry] = len(data)
        except Exception:
            return None
        return counts

    def _py4dstem(self):
        try:
            return import_module("py4DSTEM")
        except Exception as exc:
            raise OrientationServiceError(
                "py4DSTEM could not be imported in this environment."
            ) from exc
