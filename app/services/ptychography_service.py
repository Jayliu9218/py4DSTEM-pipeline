from __future__ import annotations

import copy
import inspect
import json
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from app.services.phase_contrast_service import PhaseContrastResult


class PtychographyServiceError(Exception):
    pass


COMPUTE_PRESETS = {
    "CPU": ("cpu", "cpu"),
    "GPU streamed": ("gpu", "cpu"),
    "GPU resident": ("gpu", "gpu"),
}


@dataclass(frozen=True)
class PtychographySetupParams:
    energy: float = 80e3
    defocus: float = 500.0
    model: str = "Single-slice"
    compute_preset: str = "CPU"
    vacuum_probe_path: str | None = None
    probe_source: str = "Ideal aperture"

    @property
    def device_storage(self) -> tuple[str, str]:
        return COMPUTE_PRESETS[self.compute_preset]


@dataclass(frozen=True)
class PtychographyGeometryParams:
    mode: str = "Auto"
    reciprocal_sampling: float | None = None
    scan_sampling: float | None = None
    com_rotation: float | None = None
    transpose: bool | None = None
    semiangle_cutoff: float | None = 20.0
    probe_roi: int | None = None


@dataclass(frozen=True)
class PtychographyPreprocessParams:
    vectorized_com_calculation: bool = False
    store_initial_arrays: bool = True
    max_batch_size: int = 512
    clear_fft_cache: bool = True

    @property
    def force_com_rotation(self) -> None:
        return None

    @property
    def force_reciprocal_sampling(self) -> None:
        return None


@dataclass(frozen=True)
class PtychographyOptimizationParams:
    method: str = "Grid search"
    parameter: str = "Reciprocal sampling"
    lower_bound: float = 0.01
    upper_bound: float = 0.04
    evaluations: int = 5
    reconstruction_iterations: int = 8


@dataclass(frozen=True)
class PtychographyReconstructionParams:
    num_iter: int = 64
    max_batch_size: int = 512
    object_type: str = "potential"
    object_positivity: bool = True
    fix_probe: bool = False
    seed_random: int = 0
    num_probe_modes: int = 2


@dataclass(frozen=True)
class PtychographyQCParams:
    amplitude_deviation_warning: float = 0.35
    probe_boundary_warning: float = 0.15
    aperture_outside_warning: float = 0.20
    grid_score_warning: float = 0.35
    error_ratio_warning: float = 0.90


@dataclass(frozen=True)
class PtychographyProfile:
    name: str
    setup: PtychographySetupParams = field(default_factory=PtychographySetupParams)
    geometry: PtychographyGeometryParams = field(default_factory=PtychographyGeometryParams)
    preprocess: PtychographyPreprocessParams = field(default_factory=PtychographyPreprocessParams)
    quick: PtychographyReconstructionParams = field(
        default_factory=lambda: PtychographyReconstructionParams(
            num_iter=16, object_type="complex", object_positivity=False
        )
    )
    optimization: PtychographyOptimizationParams = field(default_factory=PtychographyOptimizationParams)
    advanced: PtychographyReconstructionParams = field(default_factory=PtychographyReconstructionParams)
    qc: PtychographyQCParams = field(default_factory=PtychographyQCParams)


BUILTIN_PROFILES = MappingProxyType({
    "Safe CPU": PtychographyProfile("Safe CPU"),
    "GPU Streaming": PtychographyProfile(
        "GPU Streaming",
        setup=PtychographySetupParams(compute_preset="GPU streamed"),
    ),
    "Thin Weak-Phase": PtychographyProfile(
        "Thin Weak-Phase",
        advanced=PtychographyReconstructionParams(object_type="potential", object_positivity=True),
    ),
    "Constrained Probe": PtychographyProfile(
        "Constrained Probe",
        advanced=PtychographyReconstructionParams(fix_probe=True),
    ),
    "Mixed-State": PtychographyProfile(
        "Mixed-State",
        setup=PtychographySetupParams(model="Mixed-state"),
        advanced=PtychographyReconstructionParams(num_probe_modes=4),
    ),
})


@dataclass(frozen=True)
class PtychographyStageResult:
    stage: str
    images: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass
class PtychographyContext:
    datacube: Any | None = None
    vacuum_probe: np.ndarray | None = None
    setup_params: PtychographySetupParams | None = None
    geometry_params: PtychographyGeometryParams | None = None
    preprocess_params: PtychographyPreprocessParams | None = None
    data_result: PtychographyStageResult | None = None
    geometry_result: PtychographyStageResult | None = None
    preprocessed_ptycho: Any | None = None
    preprocess_result: PtychographyStageResult | None = None
    preprocessing_accepted: bool = False
    quick_ptycho: Any | None = None
    quick_result: PhaseContrastResult | None = None
    qc_result: PtychographyStageResult | None = None
    qc_accepted: bool = False
    optimization_result: PtychographyStageResult | None = None
    advanced_ptycho: Any | None = None
    advanced_result: PhaseContrastResult | None = None
    active_profile: PtychographyProfile = field(default_factory=lambda: BUILTIN_PROFILES["Safe CPU"])
    revision: int = 0

    @property
    def ptycho(self) -> Any | None:
        return self.preprocessed_ptycho

    @property
    def reconstruction_result(self) -> PhaseContrastResult | None:
        return self.advanced_result


class PtychographyAdapter:
    def __init__(self, py4dstem_provider: Callable[[], Any] | None = None) -> None:
        self._provider = py4dstem_provider

    def py4dstem(self) -> Any:
        if self._provider is not None:
            return self._provider()
        try:
            return import_module("py4DSTEM")
        except ImportError as exc:
            raise PtychographyServiceError("py4DSTEM is unavailable in this environment.") from exc

    @staticmethod
    def compatible_kwargs(callable_object: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(callable_object)
        except (TypeError, ValueError):
            return kwargs
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in signature.parameters}

    def construct(
        self, datacube: Any, setup: PtychographySetupParams, geometry: PtychographyGeometryParams,
        probe: np.ndarray | None,
    ) -> Any:
        phase = self.py4dstem().process.phase
        class_name = "MixedstatePtychography" if setup.model == "Mixed-state" else "SingleslicePtychography"
        cls = getattr(phase, class_name, None)
        if cls is None:
            raise PtychographyServiceError(f"This py4DSTEM version does not provide {class_name}.")
        device, storage = setup.device_storage
        kwargs = {
            "datacube": datacube,
            "energy": setup.energy,
            "defocus": setup.defocus,
            "device": device,
            "storage": storage,
            "vacuum_probe_intensity": probe,
            "semiangle_cutoff": geometry.semiangle_cutoff,
            "object_padding_px": geometry.probe_roi,
        }
        return cls(**self.compatible_kwargs(cls, {k: v for k, v in kwargs.items() if v is not None}))

    def preprocess(
        self, ptycho: Any, geometry: PtychographyGeometryParams, params: PtychographyPreprocessParams
    ) -> Any:
        kwargs = {
            "plot_center_of_mass": False,
            "plot_rotation": False,
            "plot_probe_overlaps": False,
            "force_com_rotation": geometry.com_rotation,
            "force_com_transpose": geometry.transpose,
            "force_reciprocal_sampling": geometry.reciprocal_sampling,
            "force_scan_sampling": geometry.scan_sampling,
            "vectorized_com_calculation": params.vectorized_com_calculation,
            "store_initial_arrays": params.store_initial_arrays,
            "max_batch_size": params.max_batch_size,
            "clear_fft_cache": params.clear_fft_cache,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return ptycho.preprocess(**self.compatible_kwargs(ptycho.preprocess, kwargs))

    def reconstruct(self, ptycho: Any, params: PtychographyReconstructionParams) -> Any:
        kwargs = {
            "reset": True,
            "seed_random": params.seed_random,
            "num_iter": params.num_iter,
            "max_batch_size": params.max_batch_size,
            "object_type": params.object_type,
            "object_positivity": params.object_positivity,
            "fix_probe": params.fix_probe,
            "num_probe_modes": params.num_probe_modes,
        }
        return ptycho.reconstruct(**self.compatible_kwargs(ptycho.reconstruct, kwargs))

    def optimize(
        self, datacube: Any, probe: np.ndarray | None, setup: PtychographySetupParams,
        geometry: PtychographyGeometryParams, preprocess: PtychographyPreprocessParams,
        params: PtychographyOptimizationParams, accepted_ptycho: Any | None = None,
    ) -> tuple[Any, float, np.ndarray | None]:
        try:
            module = import_module("py4DSTEM.process.phase.parameter_optimize")
            optimization_parameter = module.OptimizationParameter
            optimizer_class = module.PtychographyOptimizer
        except (ImportError, AttributeError) as exc:
            raise PtychographyServiceError(
                "This py4DSTEM installation does not provide PtychographyOptimizer."
            ) from exc
        phase = self.py4dstem().process.phase
        ptycho_class = getattr(
            phase, "MixedstatePtychography" if setup.model == "Mixed-state" else "SingleslicePtychography"
        )
        device, storage = setup.device_storage
        init_args: dict[str, Any] = {
            "datacube": datacube, "energy": setup.energy, "defocus": setup.defocus,
            "device": device, "storage": storage,
        }
        if probe is not None:
            init_args["vacuum_probe_intensity"] = probe
        elif geometry.semiangle_cutoff is not None:
            init_args["semiangle_cutoff"] = geometry.semiangle_cutoff
        if geometry.probe_roi is not None:
            init_args["object_padding_px"] = geometry.probe_roi
        preprocess_args: dict[str, Any] = {
            "plot_center_of_mass": False, "plot_rotation": False, "plot_probe_overlaps": False,
            "vectorized_com_calculation": preprocess.vectorized_com_calculation,
            "store_initial_arrays": preprocess.store_initial_arrays,
            "max_batch_size": preprocess.max_batch_size,
            "clear_fft_cache": preprocess.clear_fft_cache,
        }
        accepted_rotation = None
        if accepted_ptycho is not None and hasattr(accepted_ptycho, "_rotation_best_rad"):
            accepted_rotation = float(np.rad2deg(accepted_ptycho._rotation_best_rad))
        fixed_preprocess = {
            "force_reciprocal_sampling": geometry.reciprocal_sampling,
            "force_scan_sampling": geometry.scan_sampling,
            "force_com_rotation": geometry.com_rotation if geometry.com_rotation is not None else accepted_rotation,
            "force_com_transpose": geometry.transpose,
        }
        preprocess_args.update({key: value for key, value in fixed_preprocess.items() if value is not None})
        initial = {
            "Reciprocal sampling": geometry.reciprocal_sampling,
            "Defocus": setup.defocus,
            "Rotation": geometry.com_rotation,
            "Batch size": preprocess.max_batch_size,
            "Fix probe": 0.0,
            "Probe modes": 2.0,
        }.get(params.parameter)
        if initial is None or not params.lower_bound <= float(initial) <= params.upper_bound:
            initial = (params.lower_bound + params.upper_bound) / 2
        parameter_kwargs: dict[str, Any] = {}
        if params.parameter in {"Batch size", "Probe modes"}:
            parameter_kwargs["space"] = "integer"
        elif params.parameter == "Fix probe":
            parameter_kwargs["space"] = "boolean"
            initial = bool(round(float(initial)))
        parameter = optimization_parameter(
            initial, params.lower_bound, params.upper_bound,
            **self.compatible_kwargs(optimization_parameter, parameter_kwargs),
        )
        if params.parameter == "Reciprocal sampling":
            preprocess_args["force_reciprocal_sampling"] = parameter
        elif params.parameter == "Defocus":
            init_args["defocus"] = parameter
        elif params.parameter == "Rotation":
            preprocess_args["force_com_rotation"] = parameter
        reconstruction_args: dict[str, Any] = {
            "num_iter": params.reconstruction_iterations,
            "max_batch_size": preprocess.max_batch_size,
        }
        if params.parameter == "Batch size":
            reconstruction_args["max_batch_size"] = parameter
        elif params.parameter == "Fix probe":
            reconstruction_args["fix_probe"] = parameter
        elif params.parameter == "Probe modes":
            reconstruction_args["num_probe_modes"] = parameter
        optimizer = optimizer_class(
            ptycho_class, init_args=init_args, preprocess_args=preprocess_args,
            reconstruction_args=reconstruction_args,
        )
        if params.method == "Bayesian optimization":
            optimizer.optimize(n_calls=params.evaluations,
                               n_initial_points=min(max(2, params.evaluations // 3), params.evaluations),
                               error_metric="log")
        else:
            optimizer.grid_search(n_points=(params.evaluations,), error_metric="linear",
                                  plot_reconstructed_objects=False, return_reconstructed_objects=False)
        return optimizer, self._optimizer_best_value(optimizer, params), self._optimizer_errors(optimizer)

    @staticmethod
    def _optimizer_best_value(optimizer: Any, params: PtychographyOptimizationParams) -> float:
        for attr in ("_opt_result", "opt_result", "result"):
            values = getattr(getattr(optimizer, attr, None), "x", None)
            if values is not None and len(values):
                return float(values[0])
        return float((params.lower_bound + params.upper_bound) / 2)

    @staticmethod
    def _optimizer_errors(optimizer: Any) -> np.ndarray | None:
        for attr in ("_grid_search_errors", "grid_search_errors", "error_values"):
            values = getattr(optimizer, attr, None)
            if values is not None:
                return np.asarray(values, dtype=float).ravel()
        for attr in ("_opt_result", "opt_result", "result"):
            values = getattr(getattr(optimizer, attr, None), "func_vals", None)
            if values is not None:
                return np.asarray(values, dtype=float).ravel()
        return None


class PtychographyService:
    def __init__(self, adapter: PtychographyAdapter | None = None) -> None:
        self.adapter = adapter or PtychographyAdapter()
        self.context = PtychographyContext()

    def reset(self) -> None:
        self.context = PtychographyContext()

    def apply_profile(self, profile: PtychographyProfile | str) -> PtychographyProfile:
        selected = BUILTIN_PROFILES[profile] if isinstance(profile, str) else profile
        self.context.active_profile = selected
        self.context.setup_params = selected.setup
        self.context.geometry_params = selected.geometry
        self.context.preprocess_params = selected.preprocess
        self.invalidate_from("data")
        return selected

    @staticmethod
    def save_profile(profile: PtychographyProfile, path: Path) -> None:
        path.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")

    @staticmethod
    def load_profile(path: Path) -> PtychographyProfile:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return PtychographyProfile(
            name=raw["name"],
            setup=PtychographySetupParams(**raw.get("setup", {})),
            geometry=PtychographyGeometryParams(**raw.get("geometry", {})),
            preprocess=PtychographyPreprocessParams(**raw.get("preprocess", {})),
            quick=PtychographyReconstructionParams(**raw.get("quick", {})),
            optimization=PtychographyOptimizationParams(**raw.get("optimization", {})),
            advanced=PtychographyReconstructionParams(**raw.get("advanced", {})),
            qc=PtychographyQCParams(**raw.get("qc", {})),
        )

    def inspect_data_probe(
        self, datacube: Any, setup: PtychographySetupParams, vacuum_probe: Any | None = None
    ) -> PtychographyStageResult:
        start = perf_counter()
        data = np.asarray(getattr(datacube, "data", datacube))
        if data.ndim != 4:
            raise PtychographyServiceError(f"Ptychography requires 4D data, got shape {data.shape}.")
        mean_dp = data.mean(axis=(0, 1))
        images = {
            "Mean diffraction pattern": mean_dp,
            "Representative diffraction pattern": data[data.shape[0] // 2, data.shape[1] // 2],
        }
        probe = self._coerce_probe(vacuum_probe) if vacuum_probe is not None else self._load_probe(setup.vacuum_probe_path)
        if probe is not None:
            images["Vacuum probe"] = probe
        warnings: list[str] = []
        if not self._has_calibration(datacube):
            warnings.append("Calibration is missing or incomplete.")
        if float(np.mean(mean_dp)) <= 0:
            warnings.append("Signal is empty or non-positive.")
        maximum = float(np.max(data))
        if maximum > 0 and np.mean(data >= maximum) > 0.001:
            warnings.append("A notable fraction of pixels are saturated at the maximum value.")
        estimated = int(data.size * data.dtype.itemsize)
        if estimated > 4 * 1024**3:
            warnings.append("Estimated in-memory data size exceeds 4 GiB; use memory-saving options.")
        result = PtychographyStageResult(
            "data", images,
            {"shape": list(data.shape), "scan_shape": list(data.shape[:2]),
             "diffraction_shape": list(data.shape[2:]), "dtype": str(data.dtype),
             "estimated_bytes": estimated, "warnings": warnings, "probe_source": setup.probe_source,
             "model": setup.model, "compute_preset": setup.compute_preset},
            perf_counter() - start,
        )
        self.context.datacube = datacube
        self.context.vacuum_probe = probe
        self.context.setup_params = setup
        self.context.data_result = result
        self.invalidate_from("geometry")
        self.context.revision += 1
        return result

    def inspect_setup(self, datacube: Any, setup: PtychographySetupParams) -> PtychographyStageResult:
        return self.inspect_data_probe(datacube, setup)

    def set_geometry(self, params: PtychographyGeometryParams) -> PtychographyStageResult:
        if self.context.datacube is None:
            raise PtychographyServiceError("Inspect Data & Probe before accepting geometry.")
        sources = {
            name: ("manual override" if value is not None else params.mode.lower())
            for name, value in asdict(params).items() if name != "mode"
        }
        result = PtychographyStageResult(
            "geometry", metadata={"mode": params.mode, "values": asdict(params), "sources": sources}
        )
        self.context.geometry_params = params
        self.context.geometry_result = result
        self.invalidate_from("preprocess")
        self.context.revision += 1
        return result

    def preprocess(
        self, datacube: Any, setup: PtychographySetupParams, params: PtychographyPreprocessParams,
        geometry: PtychographyGeometryParams | None = None, vacuum_probe: Any | None = None,
    ) -> PtychographyStageResult:
        start = perf_counter()
        geometry = geometry or self.context.geometry_params or PtychographyGeometryParams()
        probe = self._coerce_probe(vacuum_probe) if vacuum_probe is not None else self._load_probe(setup.vacuum_probe_path)
        self._validate_probe_initialization(setup, geometry, probe)
        try:
            ptycho = self.adapter.construct(datacube, setup, geometry, probe)
            ptycho = self.adapter.preprocess(ptycho, geometry, params) or ptycho
        except Exception as exc:
            raise PtychographyServiceError(self._stage_error("preprocessing", setup, exc)) from exc
        result = self._extract_preprocess(ptycho, datacube, perf_counter() - start)
        self.context.preprocessed_ptycho = ptycho
        self.context.datacube = datacube
        self.context.vacuum_probe = probe
        self.context.setup_params = setup
        self.context.geometry_params = geometry
        self.context.preprocess_params = params
        self.context.preprocess_result = result
        self.context.preprocessing_accepted = False
        self.invalidate_from("quick")
        self.context.revision += 1
        return result

    def accept_preprocessing(self) -> None:
        if self.context.preprocessed_ptycho is None or self.context.preprocess_result is None:
            raise PtychographyServiceError("Run preprocessing before accepting it.")
        self.context.preprocessing_accepted = True
        self.context.revision += 1

    def quick_reconstruct(self, params: PtychographyReconstructionParams | None = None) -> PhaseContrastResult:
        params = params or self.context.active_profile.quick
        result, ptycho = self._run_reconstruction(params, "quick reconstruction", require_qc=False)
        self.context.quick_ptycho = ptycho
        self.context.quick_result = result
        self.context.qc_result = None
        self.context.qc_accepted = False
        self.context.revision += 1
        return result

    def review_qc(self, params: PtychographyQCParams | None = None) -> PtychographyStageResult:
        if self.context.quick_result is None:
            raise PtychographyServiceError("Run Quick Reconstruction before Review & QC.")
        params = params or self.context.active_profile.qc
        result = self._calculate_qc(self.context.quick_result, params)
        self.context.qc_result = result
        self.context.qc_accepted = False
        self.context.revision += 1
        return result

    def accept_qc(self) -> None:
        if self.context.qc_result is None:
            raise PtychographyServiceError("Run Review & QC before accepting its risks.")
        self.context.qc_accepted = True
        self.context.revision += 1

    def optimize(self, params: PtychographyOptimizationParams) -> PtychographyStageResult:
        if self.context.preprocessed_ptycho is None or self.context.datacube is None:
            raise PtychographyServiceError("Run preprocessing before parameter optimization.")
        setup = self.context.setup_params or PtychographySetupParams()
        geometry = self.context.geometry_params or PtychographyGeometryParams()
        preprocess = self.context.preprocess_params or PtychographyPreprocessParams()
        start = perf_counter()
        try:
            _optimizer, best, errors = self.adapter.optimize(
                self.context.datacube, self.context.vacuum_probe, setup, geometry, preprocess, params,
                self.context.preprocessed_ptycho,
            )
        except Exception as exc:
            raise PtychographyServiceError(self._stage_error("optimization", setup, exc)) from exc
        result = PtychographyStageResult(
            "optimization", {"Optimization error": errors[None, :]} if errors is not None else {},
            {"method": params.method, "parameter": params.parameter, "best_value": best,
             "evaluations": params.evaluations,
             "interpretation": "Best self-consistency reconstruction; not a claim of best physical parameters."},
            perf_counter() - start,
        )
        self.context.optimization_result = result
        self.context.revision += 1
        return result

    def advanced_reconstruct(self, params: PtychographyReconstructionParams) -> PhaseContrastResult:
        result, ptycho = self._run_reconstruction(params, "advanced reconstruction", require_qc=True)
        self.context.advanced_ptycho = ptycho
        self.context.advanced_result = result
        self.context.revision += 1
        return result

    def reconstruct(self, params: PtychographyReconstructionParams) -> PhaseContrastResult:
        if self.context.qc_accepted:
            return self.advanced_reconstruct(params)
        return self.quick_reconstruct(params)

    def save_package(self, output: Path) -> list[Path]:
        if self.context.advanced_result is None:
            raise PtychographyServiceError("Run Advanced Reconstruction before exporting.")
        output.mkdir(parents=True, exist_ok=True)
        arrays = output / "ptychography_results.npz"
        metadata = output / "ptychography_metadata.json"
        payload_arrays: dict[str, np.ndarray] = {}
        for prefix, result in (("preprocess", self.context.preprocess_result),
                               ("quick", self.context.quick_result),
                               ("qc", self.context.qc_result),
                               ("advanced", self.context.advanced_result)):
            if result is not None:
                payload_arrays.update({f"{prefix}_{self._safe_name(k)}": np.asarray(v)
                                       for k, v in result.images.items()})
        np.savez(arrays, **payload_arrays)
        payload = {
            "profile": asdict(self.context.active_profile),
            "setup": asdict(self.context.setup_params) if self.context.setup_params else {},
            "geometry": asdict(self.context.geometry_params) if self.context.geometry_params else {},
            "preprocess": asdict(self.context.preprocess_params) if self.context.preprocess_params else {},
            "qc": self.context.qc_result.metadata if self.context.qc_result else {},
            "advanced_result": self.context.advanced_result.metadata,
        }
        metadata.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        saved = [arrays, metadata]
        native = self._native_save(output / "ptychography_native.h5")
        if native is not None:
            saved.append(native)
        return saved

    def invalidate_from(self, stage: str) -> None:
        order = ["data", "geometry", "preprocess", "quick", "qc", "advanced", "export"]
        start = order.index(stage)
        if start <= order.index("geometry"):
            self.context.geometry_result = None
        if start <= order.index("preprocess"):
            self.context.preprocessed_ptycho = None
            self.context.preprocess_result = None
            self.context.preprocessing_accepted = False
        if start <= order.index("quick"):
            self.context.quick_ptycho = None
            self.context.quick_result = None
        if start <= order.index("qc"):
            self.context.qc_result = None
            self.context.qc_accepted = False
        if start <= order.index("advanced"):
            self.context.advanced_ptycho = None
            self.context.advanced_result = None

    def _run_reconstruction(
        self, params: PtychographyReconstructionParams, stage: str, require_qc: bool
    ) -> tuple[PhaseContrastResult, Any]:
        if self.context.preprocessed_ptycho is None or not self.context.preprocessing_accepted:
            raise PtychographyServiceError("Review and accept preprocessing before reconstruction.")
        if require_qc and not self.context.qc_accepted:
            raise PtychographyServiceError("Review and explicitly accept QC before Advanced Reconstruction.")
        setup = self.context.setup_params or PtychographySetupParams()
        start = perf_counter()
        try:
            ptycho, clone_strategy = self._independent_preprocessed_ptycho()
            ptycho = self.adapter.reconstruct(ptycho, params) or ptycho
        except Exception as exc:
            raise PtychographyServiceError(self._stage_error(stage, setup, exc)) from exc
        result = self._extract_reconstruction(ptycho, perf_counter() - start)
        result.metadata.update({
            "stage": stage,
            "parameters": asdict(params),
            "preprocessed_instance_strategy": clone_strategy,
        })
        return result, ptycho

    def _independent_preprocessed_ptycho(self) -> tuple[Any, str]:
        try:
            return copy.deepcopy(self.context.preprocessed_ptycho), "deepcopy"
        except Exception:
            pass
        if self.context.datacube is None:
            raise PtychographyServiceError(
                "The accepted preprocessing object cannot be copied and the source DataCube is unavailable."
            )
        setup = self.context.setup_params or PtychographySetupParams()
        geometry = self.context.geometry_params or PtychographyGeometryParams()
        preprocess = self.context.preprocess_params or PtychographyPreprocessParams()
        try:
            ptycho = self.adapter.construct(self.context.datacube, setup, geometry, self.context.vacuum_probe)
            ptycho = self.adapter.preprocess(ptycho, geometry, preprocess) or ptycho
        except Exception as exc:
            raise PtychographyServiceError(
                "The accepted py4DSTEM preprocessing object cannot be copied, and rebuilding an "
                f"independent preprocessing instance failed: {exc}"
            ) from exc
        return ptycho, "rebuild_from_accepted_parameters"

    def _load_probe(self, path: str | None) -> np.ndarray | None:
        if not path:
            return None
        try:
            probe = self.adapter.py4dstem().read(path)
            return np.asarray(getattr(probe, "data", probe))
        except Exception as exc:
            raise PtychographyServiceError(f"Could not load vacuum probe: {exc}") from exc

    @staticmethod
    def _coerce_probe(probe: Any) -> np.ndarray:
        data = np.asarray(getattr(probe, "data", probe))
        if data.ndim != 2:
            raise PtychographyServiceError(f"Vacuum probe must be a 2D array, got shape {data.shape}.")
        return data

    @staticmethod
    def _validate_probe_initialization(
        setup: PtychographySetupParams, geometry: PtychographyGeometryParams, probe: np.ndarray | None
    ) -> None:
        if probe is not None:
            return
        if setup.probe_source != "Ideal aperture":
            raise PtychographyServiceError(
                f"Probe source is '{setup.probe_source}', but no valid 2D vacuum probe was loaded."
            )
        if geometry.semiangle_cutoff is None or not np.isfinite(geometry.semiangle_cutoff):
            raise PtychographyServiceError(
                "Ideal aperture initialization requires a finite semiangle cutoff in mrad. "
                "Enable the semiangle override in Calibration / Geometry or load a vacuum probe."
            )
        if geometry.semiangle_cutoff <= 0:
            raise PtychographyServiceError("Ideal aperture semiangle cutoff must be greater than zero.")

    @staticmethod
    def _extract_preprocess(ptycho: Any, datacube: Any, elapsed: float) -> PtychographyStageResult:
        data = np.asarray(getattr(datacube, "data", datacube))
        images: dict[str, np.ndarray] = {"Raw mean diffraction pattern": data.mean(axis=(0, 1))}
        for attr, name in (("_com_fitted_x", "Fitted CoM X"), ("_com_fitted_y", "Fitted CoM Y")):
            if hasattr(ptycho, attr):
                images[name] = np.asarray(getattr(ptycho, attr))
        if hasattr(ptycho, "_amplitudes"):
            images["Centered mean diffraction amplitude"] = np.fft.fftshift(
                np.asarray(ptycho._amplitudes).mean(0)
            )
        PtychographyService._add_probe_images(ptycho, images, initial=True)
        metadata = {}
        for attr, name in (("sampling", "sampling"), ("_region_of_interest_shape", "roi_shape"),
                           ("_object_shape", "object_fov")):
            if hasattr(ptycho, attr):
                metadata[name] = np.asarray(getattr(ptycho, attr)).tolist()
        if hasattr(ptycho, "_rotation_best_rad"):
            metadata["rotation_degrees"] = float(np.rad2deg(ptycho._rotation_best_rad))
        return PtychographyStageResult("preprocess", images, metadata, elapsed)

    @staticmethod
    def _extract_reconstruction(ptycho: Any, elapsed: float) -> PhaseContrastResult:
        images: dict[str, np.ndarray] = {}
        obj = getattr(ptycho, "object_cropped", getattr(ptycho, "object", None))
        object_phase = object_amplitude = None
        if obj is not None:
            obj = np.asarray(obj)
            object_phase, object_amplitude = np.angle(obj), np.abs(obj)
            images.update({"Phase": object_phase, "Amplitude": object_amplitude,
                           "Object Complex": obj, "Object FFT": np.abs(np.fft.fftshift(np.fft.fft2(obj)))})
        PtychographyService._add_probe_images(ptycho, images, initial=False)
        if hasattr(ptycho, "object_fft"):
            images["Object FFT"] = np.asarray(ptycho.object_fft)
        error = getattr(ptycho, "error_iterations", None)
        if error is not None:
            images["Convergence error"] = np.asarray(error, dtype=float)[None, :]
        rotation = float(np.rad2deg(ptycho._rotation_best_rad)) if hasattr(ptycho, "_rotation_best_rad") else None
        return PhaseContrastResult(
            method="Ptychography", images=images,
            metadata={"artifact_guidance": "Inspect object FFT, probe boundary energy, and Fourier aperture leakage."},
            elapsed_seconds=elapsed, rotation_degrees=rotation, object_phase=object_phase,
            object_amplitude=object_amplitude, probe=images.get("Probe Intensity"),
            probe_fourier=images.get("Fourier Probe"),
            error_iterations=np.asarray(error) if error is not None else None,
        )

    @staticmethod
    def _calculate_qc(result: PhaseContrastResult, params: PtychographyQCParams) -> PtychographyStageResult:
        amplitude = np.asarray(result.images.get("Amplitude", np.ones((1, 1))), dtype=float)
        probe = np.asarray(result.images.get("Probe Intensity", np.zeros((1, 1))), dtype=float)
        fourier = np.asarray(result.images.get("Fourier Probe", np.zeros((1, 1))), dtype=float)
        obj_fft = np.abs(np.asarray(result.images.get("Object FFT", np.zeros((1, 1))), dtype=float))
        error = np.asarray(result.error_iterations if result.error_iterations is not None else [1.0], dtype=float)
        metrics = {
            "amplitude_deviation": float(np.mean(np.abs(amplitude - 1))),
            "probe_boundary_energy": PtychographyService._boundary_fraction(probe),
            "aperture_outside_energy": PtychographyService._outside_fraction(fourier),
            "object_fft_grid_score": PtychographyService._grid_score(obj_fft),
            "error_final_over_initial": float(error[-1] / error[0]) if error.size and error[0] else 1.0,
        }
        checks = [
            ("amplitude_deviation", params.amplitude_deviation_warning, "Check defocus and object model."),
            ("probe_boundary_energy", params.probe_boundary_warning, "Expand probe ROI."),
            ("aperture_outside_energy", params.aperture_outside_warning, "Consider fixing or constraining the probe."),
            ("object_fft_grid_score", params.grid_score_warning, "Check scan geometry and raster-grid artifacts."),
            ("error_final_over_initial", params.error_ratio_warning, "Reduce batch size or reconsider mixed-state."),
        ]
        warnings = [advice for name, limit, advice in checks if metrics[name] > limit]
        return PtychographyStageResult(
            "qc", {"QC metrics": np.asarray(list(metrics.values()), dtype=float)[None, :]},
            {"metrics": metrics, "warnings": warnings, "severity": "warning" if warnings else "pass",
             "guidance": "QC supports risk review; it does not establish physical truth."},
        )

    @staticmethod
    def _boundary_fraction(image: np.ndarray) -> float:
        if image.ndim < 2 or not np.sum(np.abs(image)):
            return 0.0
        width = max(1, min(image.shape[-2:]) // 10)
        mask = np.ones(image.shape[-2:], dtype=bool)
        mask[width:-width, width:-width] = False
        return float(np.sum(np.abs(image)[..., mask]) / np.sum(np.abs(image)))

    @staticmethod
    def _outside_fraction(image: np.ndarray) -> float:
        if image.ndim < 2 or not np.sum(np.abs(image)):
            return 0.0
        y, x = np.indices(image.shape[-2:])
        cy, cx = (np.asarray(image.shape[-2:]) - 1) / 2
        radius = np.hypot(y - cy, x - cx)
        outside = radius > 0.4 * min(image.shape[-2:])
        return float(np.sum(np.abs(image)[..., outside]) / np.sum(np.abs(image)))

    @staticmethod
    def _grid_score(image: np.ndarray) -> float:
        if image.size < 2 or not np.max(image):
            return 0.0
        flat = image.ravel()
        return float(np.percentile(flat, 99) / np.max(flat))

    @staticmethod
    def _add_probe_images(ptycho: Any, images: dict[str, np.ndarray], initial: bool) -> None:
        prefix = "Initial " if initial else ""
        probe = getattr(ptycho, "probe", getattr(ptycho, "_probe", None))
        if probe is not None:
            probe = np.asarray(probe)
            images[f"{prefix}Probe Complex"] = probe
            images[f"{prefix}Probe Intensity"] = np.abs(probe) ** 2
        fourier = getattr(ptycho, "probe_fourier", None)
        if fourier is None and probe is not None:
            fourier = np.fft.fft2(probe)
        if fourier is not None:
            images[f"{prefix}Fourier Probe"] = np.abs(np.asarray(fourier))

    def _native_save(self, path: Path) -> Path | None:
        obj = self.context.advanced_ptycho
        for name in ("save", "to_h5", "write"):
            method = getattr(obj, name, None)
            if callable(method):
                try:
                    method(str(path))
                    return path if path.exists() else None
                except Exception:
                    return None
        return None

    @staticmethod
    def _has_calibration(datacube: Any) -> bool:
        calibration = getattr(datacube, "calibration", None)
        return calibration is not None

    @staticmethod
    def _safe_name(name: str) -> str:
        return name.lower().replace(" ", "_").replace("/", "_")

    @staticmethod
    def _stage_error(stage: str, setup: PtychographySetupParams, exc: Exception) -> str:
        message = f"Ptychography {stage} failed: {exc}"
        lower = str(exc).lower()
        if setup.compute_preset != "CPU" and any(word in lower for word in ("cuda", "cupy", "memory", "gpu")):
            message += " CUDA/CuPy may be unavailable or out of memory; select the CPU preset to retry."
        elif "memory" in lower:
            message += " Reduce batch size or enable memory-saving preprocessing."
        return message
