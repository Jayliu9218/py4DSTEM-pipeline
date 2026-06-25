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


class PhaseMappingServiceError(Exception):
    """User-facing phase mapping error."""


@dataclass(frozen=True)
class PhasePlanParams:
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
class PhaseMatchParams:
    num_matches_return: int = 2
    min_angle_between_matches_deg: float = 5
    min_number_peaks: int = 3
    inversion_symmetry: bool = True
    corr_normalize: bool = True
    low_confidence_threshold: float = 0.1


@dataclass(frozen=True)
class PhaseMatchResult:
    phase_id_map: np.ndarray
    phase_label_map: np.ndarray
    correlation_maps: list[np.ndarray]
    best_correlation_map: np.ndarray
    confidence_map: np.ndarray
    per_phase_orientation: list[Any]
    per_phase_rgb: list[np.ndarray]
    phase_names: list[str]
    phase_fraction: dict[str, float]
    images: dict[str, np.ndarray]
    warnings: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0


@dataclass
class CrystalEntry:
    name: str
    crystal: Any
    source: str = "CIF"
    enabled: bool = True


@dataclass
class PhaseMappingContext:
    crystals: list[CrystalEntry] = field(default_factory=list)
    plan_params: PhasePlanParams | None = None
    plan_ready: bool = False
    result: PhaseMatchResult | None = None


class PhaseMappingService:
    def __init__(self) -> None:
        self.context = PhaseMappingContext()

    @property
    def crystals(self) -> list[CrystalEntry]:
        return self.context.crystals

    @property
    def result(self) -> PhaseMatchResult | None:
        return self.context.result

    @property
    def plan_ready(self) -> bool:
        return self.context.plan_ready

    def load_crystal(self, cif_path: str | Path) -> str:
        path = Path(cif_path)
        if not path.exists():
            raise PhaseMappingServiceError(f"CIF file does not exist: {path}")
        try:
            crystal = load_py4dstem_crystal_from_cif(self._py4dstem(), path)
        except Exception as exc:
            raise PhaseMappingServiceError(f"Could not load crystal structure: {exc}") from exc
        entry = CrystalEntry(name=path.stem, crystal=crystal, source=f"CIF: {path.name}")
        self.context.crystals.append(entry)
        self._invalidate_after_library_change()
        return entry.source

    def remove_crystal(self, index: int) -> None:
        if not 0 <= index < len(self.context.crystals):
            raise PhaseMappingServiceError(f"Invalid crystal index {index}.")
        self.context.crystals.pop(index)
        self._invalidate_after_library_change()

    def set_crystal_enabled(self, index: int, enabled: bool) -> None:
        if not 0 <= index < len(self.context.crystals):
            raise PhaseMappingServiceError(f"Invalid crystal index {index}.")
        self.context.crystals[index].enabled = enabled
        self._invalidate_after_library_change()

    def crystal_summaries(self) -> list[str]:
        return [entry.source for entry in self.context.crystals]

    def enabled_crystals(self) -> list[tuple[int, CrystalEntry]]:
        return [
            (index, entry)
            for index, entry in enumerate(self.context.crystals)
            if entry.enabled
        ]

    def create_multi_phase_plan(self, params: PhasePlanParams) -> float:
        entries = self.enabled_crystals()
        if not entries:
            raise PhaseMappingServiceError("Add at least one crystal to the library first.")
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
                raise PhaseMappingServiceError("Fiber axis cannot be the zero vector.")
            kwargs.update(fiber_axis=list(params.fiber_axis), fiber_angles=list(params.fiber_angles))
        for _index, entry in entries:
            try:
                crystal = entry.crystal
                setup = getattr(crystal, "setup_diffraction", None)
                if callable(setup):
                    setup(accelerating_voltage=params.accelerating_voltage)
                crystal.calculate_structure_factors(k_max=params.k_max, tol_structure_factor=1e-4)
                crystal.orientation_plan(**kwargs)
            except Exception as exc:
                raise PhaseMappingServiceError(
                    f"Orientation plan failed for '{entry.name}': {exc}"
                ) from exc
        self.context.plan_params = params
        self.context.plan_ready = True
        self.context.result = None
        return perf_counter() - start

    def match_phases(
        self, braggvectors: Any | None, params: PhaseMatchParams
    ) -> PhaseMatchResult:
        entries = self.enabled_crystals()
        if not entries:
            raise PhaseMappingServiceError("Add at least one crystal to the library first.")
        if not self.context.plan_ready:
            raise PhaseMappingServiceError("Create the multi-phase orientation plan first.")
        if braggvectors is None:
            raise PhaseMappingServiceError("Run full BraggVectors before phase matching.")
        start = perf_counter()
        warnings: list[str] = []
        calibration_warning = self._calibration_warning(getattr(braggvectors, "calstate", {}))
        if calibration_warning:
            warnings.append(calibration_warning)
        scan_shape = self._scan_shape(braggvectors)
        correlation_maps: list[np.ndarray] = []
        per_phase_orientation: list[Any] = []
        per_phase_rgb: list[np.ndarray] = []
        phase_names: list[str] = []
        for _index, entry in entries:
            crystal = entry.crystal
            try:
                orientation_map = crystal.match_orientations(
                    braggvectors,
                    num_matches_return=params.num_matches_return,
                    min_angle_between_matches_deg=params.min_angle_between_matches_deg,
                    min_number_peaks=params.min_number_peaks,
                    inversion_symmetry=params.inversion_symmetry,
                    progress_bar=False,
                )
            except Exception as exc:
                raise PhaseMappingServiceError(
                    f"Phase matching failed for '{entry.name}': {exc}"
                ) from exc
            per_phase_orientation.append(orientation_map)
            phase_names.append(entry.name)
            corr_map = self._correlation_map(orientation_map, scan_shape)
            correlation_maps.append(corr_map)
            rgb = self._plot_map(crystal, orientation_map, params)
            per_phase_rgb.append(rgb)
        correlation_stack = np.stack(correlation_maps, axis=0)
        best_index = np.argmax(correlation_stack, axis=0)
        best_correlation = np.max(correlation_stack, axis=0)
        if len(correlation_maps) > 1:
            sorted_corr = np.sort(correlation_stack, axis=0)
            confidence = sorted_corr[-1] - sorted_corr[-2]
        else:
            confidence = np.ones(scan_shape, dtype=float)
        phase_id_map = best_index.astype(np.int32)
        cmap = self._phase_colormap(len(entries))
        phase_label_map = cmap[phase_id_map]
        phase_fraction = {
            entry.name: float(np.mean(phase_id_map == i))
            for i, (_idx, entry) in enumerate(entries)
        }
        images: dict[str, np.ndarray] = {
            "Phase Map": phase_label_map,
            "Best Correlation": best_correlation,
            "Confidence Gap": confidence,
        }
        for i, (_idx, entry) in enumerate(entries):
            images[f"{entry.name} Correlation"] = correlation_maps[i]
            images[f"{entry.name} IPF"] = per_phase_rgb[i]
        low_conf_mask = confidence < params.low_confidence_threshold
        images["Low Confidence Mask"] = low_conf_mask
        if np.any(low_conf_mask):
            count = int(np.count_nonzero(low_conf_mask))
            total = int(low_conf_mask.size)
            warnings.append(
                f"{count}/{total} scan positions are below the phase confidence threshold."
            )
        if len(correlation_maps) < 2:
            warnings.append("Phase discrimination requires at least two enabled crystals.")
        result = PhaseMatchResult(
            phase_id_map=phase_id_map,
            phase_label_map=phase_label_map,
            correlation_maps=correlation_maps,
            best_correlation_map=best_correlation,
            confidence_map=confidence,
            per_phase_orientation=per_phase_orientation,
            per_phase_rgb=per_phase_rgb,
            phase_names=phase_names,
            phase_fraction=phase_fraction,
            images=images,
            warnings=tuple(warnings),
            elapsed_seconds=perf_counter() - start,
        )
        self.context.result = result
        return result

    def invalidate_plan(self) -> None:
        self.context.plan_params = None
        self.context.plan_ready = False
        self.context.result = None

    def invalidate_result(self) -> None:
        self.context.result = None

    def _invalidate_after_library_change(self) -> None:
        self.context.plan_params = None
        self.context.plan_ready = False
        self.context.result = None

    def _scan_shape(self, braggvectors: Any) -> tuple[int, int]:
        for attr in ("Rshape", "shape", "scan_shape"):
            value = getattr(braggvectors, attr, None)
            if value is not None:
                shape = tuple(int(dim) for dim in tuple(value)[:2])
                if len(shape) == 2:
                    return shape
        raw = getattr(braggvectors, "raw", None)
        if raw is not None:
            for obj in (raw, getattr(raw, "_data", None)):
                if obj is None:
                    continue
                for attr in ("Rshape", "shape", "scan_shape"):
                    value = getattr(obj, attr, None)
                    if value is not None:
                        shape = tuple(int(dim) for dim in tuple(value)[:2])
                        if len(shape) == 2:
                            return shape
        raise PhaseMappingServiceError("Could not determine scan shape from BraggVectors.")

    def _correlation_map(self, orientation_map: Any, scan_shape: tuple[int, int]) -> np.ndarray:
        for name in ("corr", "corr_map", "correlation", "correlation_score"):
            value = getattr(orientation_map, name, None)
            if value is None and hasattr(orientation_map, "get"):
                try:
                    value = orientation_map.get(name)
                except Exception:
                    value = None
            if value is None:
                continue
            array = np.asarray(getattr(value, "data", value), dtype=float)
            if array.ndim == 2:
                return array
            if array.ndim >= 3:
                return array[:, :, 0]
        return np.zeros(scan_shape, dtype=float)

    def _plot_map(self, crystal: Any, orientation_map: Any, params: PhaseMatchParams) -> np.ndarray:
        plt = self._matplotlib()
        plan = self.context.plan_params
        try:
            if plan and plan.mode == "Fiber":
                output = crystal.plot_fiber_orientation_maps(
                    orientation_map,
                    orientation_ind=0,
                    symmetry_order=plan.symmetry_order,
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
            array = np.asarray(images)
            if array.ndim == 4:
                array = array[:, :, :, 0]
            elif array.ndim > 3:
                array = np.squeeze(array)
            return array.copy()
        except Exception as exc:
            logger.debug("Phase per-crystal plot failed: %s", exc)
            return np.zeros((1, 1, 3), dtype=float)
        finally:
            try:
                plt.close("all")
            except Exception:
                pass

    def _phase_colormap(self, count: int) -> np.ndarray:
        base = np.array(
            [
                [0.20, 0.45, 0.85],
                [0.85, 0.35, 0.25],
                [0.30, 0.70, 0.35],
                [0.85, 0.70, 0.20],
                [0.55, 0.35, 0.75],
                [0.20, 0.75, 0.80],
                [0.90, 0.50, 0.60],
                [0.50, 0.50, 0.50],
            ],
            dtype=float,
        )
        if count <= len(base):
            return base[:count]
        rng = np.random.default_rng(42)
        extra = rng.random((count - len(base), 3))
        return np.vstack([base, extra])

    def _calibration_warning(self, calstate: Any) -> str:
        missing = [
            label
            for name, label in [
                ("center", "origin"), ("ellipse", "ellipse"), ("pixel", "pixel"), ("rotate", "rotation")
            ]
            if not bool(getattr(calstate, "get", lambda _n, _d=False: False)(name, False))
        ]
        return (
            "Calibration is incomplete; phase discrimination accuracy may be reduced. "
            f"Missing/applied-off corrections: {', '.join(missing)}."
            if missing else ""
        )

    def _matplotlib(self):
        cached = globals().get("_plt")
        if cached is not None:
            return cached
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        globals()["_plt"] = plt
        return plt

    def _py4dstem(self):
        try:
            return import_module("py4DSTEM")
        except ImportError as exc:
            raise PhaseMappingServiceError("py4DSTEM could not be imported in this environment.") from exc
