from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from app.services.cif_utils import load_py4dstem_crystal_from_cif

logger = logging.getLogger(__name__)


class OrientationServiceError(Exception):
    """User-facing orientation analysis error."""


@dataclass(frozen=True)
class ManualCrystalParams:
    lattice_type: str = "cubic"
    lattice_parameters: tuple[float, ...] = (4.08,)
    elements: tuple[str | int, ...] = ("Au",)
    positions: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    space_group: str | int | None = None


@dataclass(frozen=True)
class OrientationPlanParams:
    accelerating_voltage: float = 300000
    k_max: float = 1.5
    angle_step_zone_axis: float = 2
    angle_step_in_plane: float = 2
    corr_kernel_size: float = 0.08
    sigma_excitation_error: float = 0.02
    mode: str = "General 3D"
    fiber_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fiber_angles: tuple[float, float] = (0.0, 360.0)
    symmetry_order: int = 6
    cuda: bool = False


@dataclass(frozen=True)
class SinglePatternMatchParams:
    scan_x: int = 0
    scan_y: int = 0
    num_matches_return: int = 3
    min_angle_between_matches_deg: float = 5
    min_number_peaks: int = 3
    inversion_symmetry: bool = True
    sigma_excitation_error: float = 0.03


@dataclass(frozen=True)
class OrientationMapParams:
    num_matches_return: int = 2
    min_angle_between_matches_deg: float = 5
    min_number_peaks: int = 3
    inversion_symmetry: bool = True
    corr_normalize: bool = True
    low_confidence_threshold: float = 0.1


# Compatibility with the original public API.
OrientationMatchParams = OrientationMapParams


@dataclass(frozen=True)
class OrientationStageResult:
    stage: str
    images: dict[str, np.ndarray] = field(default_factory=dict)
    overlays: dict[str, np.ndarray] = field(default_factory=dict)
    metrics: dict[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0


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


@dataclass
class OrientationContext:
    crystal: Any | None = None
    crystal_summary: str = "No crystal loaded"
    plan_params: OrientationPlanParams | None = None
    plan_result: OrientationStageResult | None = None
    single_orientation: Any | None = None
    single_result: OrientationStageResult | None = None
    single_review_accepted: bool = False
    orientation_map: Any | None = None
    map_result: OrientationStageResult | None = None


class OrientationService:
    def __init__(self) -> None:
        self.context = OrientationContext()

    @property
    def crystal(self) -> Any | None:
        return self.context.crystal

    @crystal.setter
    def crystal(self, value: Any | None) -> None:
        self.context.crystal = value

    @property
    def orientation_map(self) -> Any | None:
        return self.context.orientation_map

    @orientation_map.setter
    def orientation_map(self, value: Any | None) -> None:
        self.context.orientation_map = value

    def load_crystal(self, cif_path: str | Path) -> str:
        path = Path(cif_path)
        if not path.exists():
            raise OrientationServiceError(f"CIF file does not exist: {path}")
        try:
            crystal = load_py4dstem_crystal_from_cif(self._py4dstem(), path)
        except Exception as exc:
            raise OrientationServiceError(f"Could not load crystal structure: {exc}") from exc
        self._set_crystal(crystal, f"CIF: {path.name}")
        return str(path)

    def create_manual_crystal(self, params: ManualCrystalParams) -> str:
        if not params.lattice_parameters or any(float(v) <= 0 for v in params.lattice_parameters):
            raise OrientationServiceError("Manual crystal lattice parameters must be positive.")
        if not params.elements or len(params.elements) != len(params.positions):
            raise OrientationServiceError(
                "Manual crystal requires one element for every fractional position."
            )
        positions = np.asarray(params.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise OrientationServiceError("Fractional positions must be finite x, y, z triples.")
        try:
            cls = self._py4dstem().process.diffraction.Crystal
            crystal = cls.from_unitcell_parameters(
                list(params.lattice_parameters),
                list(params.elements),
                positions,
                space_group=params.space_group,
                lattice_type=params.lattice_type,
            )
        except Exception as exc:
            raise OrientationServiceError(f"Could not create manual crystal: {exc}") from exc
        summary = (
            f"Manual {params.lattice_type}: {len(params.elements)} atom(s), "
            f"lattice={','.join(f'{v:g}' for v in params.lattice_parameters)}"
        )
        self._set_crystal(crystal, summary)
        return summary

    def create_plan_stage(self, params: OrientationPlanParams) -> OrientationStageResult:
        crystal = self._require_crystal()
        start = perf_counter()
        kwargs: dict[str, Any] = {
            "zone_axis_range": "fiber" if params.mode == "Fiber" else "auto",
            "angle_step_zone_axis": params.angle_step_zone_axis,
            "angle_step_in_plane": params.angle_step_in_plane,
            "accel_voltage": params.accelerating_voltage,
            "corr_kernel_size": params.corr_kernel_size,
            "sigma_excitation_error": params.sigma_excitation_error,
            "power_radial": 1.0,
            "power_intensity": 0.0,
            "power_intensity_experiment": 0.0,
            "CUDA": params.cuda,
            "progress_bar": False,
        }
        if params.mode == "Fiber":
            if np.linalg.norm(params.fiber_axis) == 0:
                raise OrientationServiceError("Fiber axis cannot be the zero vector.")
            kwargs.update(fiber_axis=list(params.fiber_axis), fiber_angles=list(params.fiber_angles))
        try:
            setup = getattr(crystal, "setup_diffraction", None)
            if callable(setup):
                setup(accelerating_voltage=params.accelerating_voltage)
            crystal.calculate_structure_factors(k_max=params.k_max, tol_structure_factor=1e-4)
            crystal.orientation_plan(**kwargs)
        except Exception as exc:
            raise OrientationServiceError(f"Orientation plan failed: {exc}") from exc
        result = OrientationStageResult(
            stage="plan",
            metrics={
                "crystal": self.context.crystal_summary,
                "mode": params.mode,
                "k_max": params.k_max,
                "zone_axis_step": params.angle_step_zone_axis,
                "in_plane_step": params.angle_step_in_plane,
            },
            elapsed_seconds=perf_counter() - start,
        )
        self.context.plan_params = params
        self.context.plan_result = result
        self._invalidate_after_plan()
        return result

    def create_plan(self, params: OrientationPlanParams) -> float:
        return self.create_plan_stage(params).elapsed_seconds

    def review_single_pattern(
        self, braggvectors: Any | None, params: SinglePatternMatchParams
    ) -> OrientationStageResult:
        crystal = self._require_plan()
        peaks = self._calibrated_peaks(braggvectors, params.scan_x, params.scan_y)
        start = perf_counter()
        try:
            orientation = crystal.match_single_pattern(
                peaks,
                num_matches_return=params.num_matches_return,
                min_angle_between_matches_deg=params.min_angle_between_matches_deg,
                min_number_peaks=params.min_number_peaks,
                inversion_symmetry=params.inversion_symmetry,
                verbose=False,
            )
            simulated = [
                crystal.generate_diffraction_pattern(
                    orientation,
                    ind_orientation=index,
                    sigma_excitation_error=params.sigma_excitation_error,
                )
                for index in range(params.num_matches_return)
            ]
        except Exception as exc:
            raise OrientationServiceError(f"Single-pattern orientation review failed: {exc}") from exc

        experimental_xyi = self._peak_array(peaks)
        images = {"Experimental peaks": self._peak_image(experimental_xyi)}
        overlays: dict[str, np.ndarray] = {}
        for index, candidate in enumerate(simulated, start=1):
            candidate_xyi = self._peak_array(candidate)
            name = f"Candidate {index}"
            images[name] = self._peak_image(candidate_xyi)
            images[f"{name} overlay"] = self._peak_image(
                experimental_xyi, candidate_xyi
            )
            overlays[name] = candidate_xyi
        metrics = self._single_metrics(orientation, len(experimental_xyi), params.num_matches_return)
        warnings = tuple(filter(None, (self._calibration_warning(getattr(braggvectors, "calstate", {})),)))
        result = OrientationStageResult(
            stage="single_review",
            images=images,
            overlays=overlays,
            metrics=metrics,
            warnings=warnings,
            elapsed_seconds=perf_counter() - start,
        )
        self.context.single_orientation = orientation
        self.context.single_result = result
        self.context.single_review_accepted = False
        self.context.orientation_map = None
        self.context.map_result = None
        return result

    def accept_single_review(self) -> OrientationStageResult:
        if self.context.single_result is None:
            raise OrientationServiceError("Run a single-pattern match review first.")
        self.context.single_review_accepted = True
        return self.context.single_result

    def invalidate_plan(self) -> None:
        self.context.plan_params = None
        self.context.plan_result = None
        self._invalidate_after_plan()

    def invalidate_review(self) -> None:
        self.context.single_orientation = None
        self.context.single_result = None
        self.context.single_review_accepted = False
        self.context.orientation_map = None
        self.context.map_result = None

    def invalidate_map(self) -> None:
        self.context.orientation_map = None
        self.context.map_result = None

    def match_map(
        self, braggvectors: Any | None, params: OrientationMapParams
    ) -> OrientationStageResult:
        crystal = self._require_plan()
        if not self.context.single_review_accepted:
            raise OrientationServiceError(
                "Accept the single-pattern match review before full orientation mapping."
            )
        if braggvectors is None:
            raise OrientationServiceError("Run full BraggVectors before orientation matching.")
        start = perf_counter()
        try:
            orientation_map = crystal.match_orientations(
                braggvectors,
                num_matches_return=params.num_matches_return,
                min_angle_between_matches_deg=params.min_angle_between_matches_deg,
                min_number_peaks=params.min_number_peaks,
                inversion_symmetry=params.inversion_symmetry,
                progress_bar=True,
            )
            preview = self._plot_map(crystal, orientation_map, params)
        except Exception as exc:
            raise OrientationServiceError(f"Orientation matching failed: {exc}") from exc
        self.context.orientation_map = orientation_map
        quality = self.orientation_quality(orientation_map, braggvectors, preview)
        warnings = list(quality.warnings)
        calibration_warning = self._calibration_warning(getattr(braggvectors, "calstate", {}))
        if calibration_warning:
            warnings.insert(0, calibration_warning)
        if params.num_matches_return < 2:
            warnings.append("Confidence gap requires at least two returned orientation candidates.")
        maps = dict(quality.maps)
        confidence = maps.get("Confidence Gap", maps.get("Ambiguity"))
        if confidence is not None:
            maps["Low Confidence Mask"] = np.asarray(confidence < params.low_confidence_threshold)
        result = OrientationStageResult(
            stage="map",
            images=maps,
            metrics={"mode": self.context.plan_params.mode, "matches": params.num_matches_return},
            warnings=tuple(warnings),
            elapsed_seconds=perf_counter() - start,
        )
        self.context.map_result = result
        return result

    def match(self, braggvectors: Any | None, params: OrientationMapParams) -> OrientationResult:
        # Legacy direct matching remains available, while the staged UI uses match_map.
        if not self.context.single_review_accepted:
            self.context.single_review_accepted = True
        result = self.match_map(braggvectors, params)
        preview = result.images.get("Orientation RGB", np.zeros((1, 1, 3)))
        return OrientationResult(
            self.context.orientation_map,
            preview,
            OrientationQualityResult(result.images, result.warnings),
            result.elapsed_seconds,
        )

    def orientation_quality(
        self, orientation_map: Any, braggvectors: Any | None, preview: np.ndarray | None = None
    ) -> OrientationQualityResult:
        maps: dict[str, np.ndarray] = {}
        warnings: list[str] = []
        if preview is not None:
            maps["Orientation RGB"] = np.asarray(preview)
        for label, candidates in [
            ("Correlation Score", ("corr", "corr_map", "correlation", "correlation_score")),
            ("Reliability", ("reliability", "confidence", "confidence_map")),
            ("Confidence Gap", ("corr_gap", "correlation_gap", "ambiguity", "ambiguity_map")),
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
        return OrientationQualityResult(maps, tuple(warnings))

    def _plot_map(self, crystal: Any, orientation_map: Any, params: OrientationMapParams) -> np.ndarray:
        import matplotlib.pyplot as plt

        if self.context.plan_params and self.context.plan_params.mode == "Fiber":
            output = crystal.plot_fiber_orientation_maps(
                orientation_map,
                orientation_ind=0,
                symmetry_order=self.context.plan_params.symmetry_order,
                corr_normalize=params.corr_normalize,
                returnfig=True,
            )
        else:
            output = crystal.plot_orientation_maps(
                orientation_map=orientation_map,
                orientation_ind=0,
                corr_normalize=params.corr_normalize,
                show_axes=True,
                returnfig=True,
                progress_bar=False,
            )
        images, fig = output[0], output[1]
        try:
            array = np.asarray(images)
            if array.ndim == 4:
                array = array[:, :, :, 0]
            elif array.ndim > 3:
                array = np.squeeze(array)
            return array.copy()
        finally:
            plt.close(fig)

    def _set_crystal(self, crystal: Any, summary: str) -> None:
        self.context = OrientationContext(crystal=crystal, crystal_summary=summary)

    def _invalidate_after_plan(self) -> None:
        self.context.single_orientation = None
        self.context.single_result = None
        self.context.single_review_accepted = False
        self.context.orientation_map = None
        self.context.map_result = None

    def _require_crystal(self) -> Any:
        if self.context.crystal is None:
            raise OrientationServiceError("Load a CIF or create a manual crystal first.")
        return self.context.crystal

    def _require_plan(self) -> Any:
        crystal = self._require_crystal()
        if self.context.plan_result is None:
            raise OrientationServiceError("Create an orientation plan first.")
        return crystal

    def _calibrated_peaks(self, braggvectors: Any | None, x: int, y: int) -> Any:
        if braggvectors is None:
            raise OrientationServiceError("Run full BraggVectors before orientation review.")
        calibrated = getattr(braggvectors, "cal", None)
        if calibrated is None:
            raise OrientationServiceError("BraggVectors.cal is required for orientation review.")
        try:
            return calibrated[int(x), int(y)]
        except Exception as exc:
            raise OrientationServiceError(f"Invalid review scan position ({x}, {y}).") from exc

    def _single_metrics(self, orientation: Any, peak_count: int, count: int) -> dict[str, object]:
        metrics: dict[str, object] = {"experimental_peak_count": peak_count}
        corr = self._orientation_values(orientation, ("corr", "correlation", "correlation_score"))
        if corr:
            for index, value in enumerate(corr[:count], start=1):
                metrics[f"candidate_{index}_correlation"] = value
            if len(corr) > 1:
                metrics["confidence_gap"] = float(corr[0] - corr[1])
        for label, names in [
            ("zone_axis", ("zone_axis", "zone_axis_lattice")),
            ("in_plane_rotation", ("in_plane_rotation", "rotation", "angles")),
        ]:
            values = self._orientation_values(orientation, names, numeric=False)
            if values:
                metrics[label] = values[0]
        return metrics

    def _orientation_values(
        self, orientation: Any, names: tuple[str, ...], numeric: bool = True
    ) -> list[Any]:
        for name in names:
            value = getattr(orientation, name, None)
            if value is None:
                continue
            try:
                array = np.asarray(value)
                values = array.ravel().tolist()
                return [float(v) for v in values] if numeric else values
            except Exception:
                return [value]
        return []

    def _peak_array(self, peaks: Any) -> np.ndarray:
        data = getattr(peaks, "data", peaks)
        array = np.asarray(data)
        if array.size == 0:
            return np.zeros((0, 3), dtype=float)
        if array.dtype.names:
            names = array.dtype.names
            qx = next((name for name in ("qx", "q_x", "x") if name in names), names[0])
            qy = next((name for name in ("qy", "q_y", "y") if name in names), names[1])
            intensity = next((name for name in ("intensity", "I", "value") if name in names), None)
            return np.column_stack(
                (array[qx], array[qy], array[intensity] if intensity else np.ones(len(array)))
            ).astype(float)
        array = np.asarray(array, dtype=float).reshape(-1, array.shape[-1])
        return np.column_stack((array[:, :2], array[:, 2] if array.shape[1] > 2 else np.ones(len(array))))

    def _peak_image(self, primary: np.ndarray, secondary: np.ndarray | None = None) -> np.ndarray:
        arrays = [item for item in (primary, secondary) if item is not None and len(item)]
        if not arrays:
            return np.zeros((64, 64, 3), dtype=float)
        limit = max(float(np.max(np.abs(item[:, :2]))) for item in arrays)
        limit = max(limit, 1.0)
        image = np.zeros((128, 128, 3), dtype=float)
        for array, channel in ((primary, 1), (secondary, 0)):
            if array is None:
                continue
            coords = np.rint((array[:, :2] / (2 * limit) + 0.5) * 127).astype(int)
            coords = np.clip(coords, 0, 127)
            weights = array[:, 2] if array.shape[1] > 2 else np.ones(len(array))
            weights = weights / max(float(np.max(np.abs(weights))), 1e-12)
            for (x, y), weight in zip(coords, weights):
                image[max(0, x - 1):min(128, x + 2), max(0, y - 1):min(128, y + 2), channel] = max(
                    float(weight), 0.25
                )
        return image

    def _calibration_warning(self, calstate: Any) -> str:
        missing = [
            label
            for name, label in [
                ("center", "origin"), ("ellipse", "ellipse"), ("pixel", "pixel"), ("rotate", "rotation")
            ]
            if not bool(getattr(calstate, "get", lambda _name, _default=False: False)(name, False))
        ]
        return (
            "Calibration is incomplete; continuing orientation matching with lower expected "
            f"accuracy. Missing/applied-off corrections: {', '.join(missing)}."
            if missing else ""
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
            if array.ndim >= 3:
                return array[:, :, 0]
        return None

    def _peak_count_map(self, braggvectors: Any | None) -> np.ndarray | None:
        raw = getattr(braggvectors, "raw", None)
        if raw is None:
            return None
        try:
            shape = self._scan_shape(raw, braggvectors)
            counts = np.zeros(shape, dtype=float)
            for x in range(shape[0]):
                for y in range(shape[1]):
                    counts[x, y] = len(getattr(raw[x, y], "data", raw[x, y]))
            return counts
        except Exception:
            return None

    def _scan_shape(self, raw: Any, braggvectors: Any | None = None) -> tuple[int, int]:
        for obj in (braggvectors, raw, getattr(raw, "_data", None)):
            if obj is None:
                continue
            for attr in ("Rshape", "shape", "scan_shape"):
                value = getattr(obj, attr, None)
                if value is not None:
                    shape = tuple(int(dim) for dim in tuple(value)[:2])
                    if len(shape) == 2:
                        return shape
        raise OrientationServiceError("Could not determine scan shape from BraggVectors.")

    def _py4dstem(self):
        try:
            return import_module("py4DSTEM")
        except ImportError as exc:
            raise OrientationServiceError("py4DSTEM could not be imported in this environment.") from exc
