from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np

from app.services.phase_contrast_service import PhaseContrastResult


class ParallaxServiceError(Exception):
    pass


ProgressCallback = Callable[[str, float], None]


@dataclass(frozen=True)
class BFMaskParams:
    threshold: float = 0.5
    use_circle: bool = False
    center_x: float | None = None
    center_y: float | None = None
    radius: float | None = None
    virtual_bf_count: int = 4
    finite_dose_enabled: bool = False
    finite_doses: tuple[float, ...] = (100.0, 50.0, 10.0)
    finite_dose_seed: int = 0


@dataclass(frozen=True)
class ParallaxAlignmentParams:
    energy: float = 300e3
    device: str = "cpu"
    object_padding_px: tuple[int, int] = (16, 16)
    edge_blend: int = 8
    normalize_images: bool = False
    threshold_intensity: float = 0.6
    alignment_bin_values: tuple[int, ...] = (32, 32, 32, 32, 16, 16, 8, 8)
    regularize_shifts: bool = False
    cross_correlation_upsample_factor: int = 8


@dataclass(frozen=True)
class ParallaxAdvancedParams:
    run_subpixel: bool = True
    kde_upsample_factor: int = 4
    kde_sigma_px: float = 0.125
    run_aberration_fit: bool = False
    run_aberration_correction: bool = False
    run_high_order_fit: bool = False
    run_ctf_fit: bool = False
    max_radial_order: int = 3
    max_angular_order: int = 4


@dataclass(frozen=True)
class ParallaxStageResult:
    stage: str
    images: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass
class ParallaxWorkflowContext:
    bf_result: ParallaxStageResult | None = None
    accepted_bf_mask: np.ndarray | None = None
    alignment_result: PhaseContrastResult | None = None
    alignment_accepted: bool = False
    advanced_result: PhaseContrastResult | None = None
    parallax: Any | None = None
    adapter_metadata: dict[str, object] = field(default_factory=dict)

    def reset(self) -> None:
        self.bf_result = None
        self.accepted_bf_mask = None
        self.alignment_result = None
        self.alignment_accepted = False
        self.advanced_result = None
        self.parallax = None
        self.adapter_metadata.clear()


class Py4DSTEMParallaxAdapter:
    """Version-aware boundary around py4DSTEM's Parallax API."""

    def __init__(self, py4dstem: Any | None = None) -> None:
        self.py4dstem = py4dstem

    def _module(self) -> Any:
        if self.py4dstem is None:
            try:
                self.py4dstem = import_module("py4DSTEM")
            except ImportError as exc:
                raise ParallaxServiceError("py4DSTEM could not be imported.") from exc
        return self.py4dstem

    @staticmethod
    def supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return kwargs
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in signature.parameters}

    def construct(self, datacube: Any, params: ParallaxAlignmentParams) -> Any:
        cls = self._module().process.phase.Parallax
        kwargs = self.supported_kwargs(
            cls,
            {
                "datacube": datacube,
                "energy": params.energy,
                "device": params.device,
                "object_padding_px": params.object_padding_px,
            },
        )
        return cls(**kwargs)

    def preprocess(
        self, parallax: Any, accepted_bf_mask: np.ndarray, params: ParallaxAlignmentParams
    ) -> Any:
        kwargs = self.supported_kwargs(
            parallax.preprocess,
            {
                "dp_mask": accepted_bf_mask,
                "edge_blend": params.edge_blend,
                "normalize_images": params.normalize_images,
                "threshold_intensity": params.threshold_intensity,
                "plot_average_bf": False,
            },
        )
        return parallax.preprocess(**kwargs)

    def reconstruct(self, parallax: Any, params: ParallaxAlignmentParams) -> Any:
        kwargs = self.supported_kwargs(
            parallax.reconstruct,
            {
                "alignment_bin_values": list(params.alignment_bin_values),
                "regularize_shifts": params.regularize_shifts,
                "cross_correlation_upsample_factor": params.cross_correlation_upsample_factor,
                "progress_bar": False,
                "plot_aligned_bf": False,
                "plot_convergence": False,
            },
        )
        return parallax.reconstruct(**kwargs)

    def subpixel(self, parallax: Any, params: ParallaxAdvancedParams) -> Any:
        kwargs = self.supported_kwargs(
            parallax.subpixel_alignment,
            {
                "kde_upsample_factor": params.kde_upsample_factor,
                "kde_sigma_px": params.kde_sigma_px,
                "plot_upsampled_BF_comparison": False,
                "plot_upsampled_FFT_comparison": False,
                "progress_bar": False,
            },
        )
        return parallax.subpixel_alignment(**kwargs)

    def aberration_fit(self, parallax: Any, params: ParallaxAdvancedParams) -> Any:
        kwargs = self.supported_kwargs(
            parallax.aberration_fit,
            {
                "max_radial_order": params.max_radial_order,
                "max_angular_order": params.max_angular_order,
                "plot_CTF_comparison": False,
                "plot_BF_shifts_comparison": False,
            },
        )
        return parallax.aberration_fit(**kwargs)

    def aberration_correct(self, parallax: Any) -> Any:
        kwargs = self.supported_kwargs(
            parallax.aberration_correct,
            {"plot_corrected_phase": False, "upsampled": True},
        )
        return parallax.aberration_correct(**kwargs)


class ParallaxService:
    def __init__(self, adapter: Py4DSTEMParallaxAdapter | None = None) -> None:
        self.adapter = adapter or Py4DSTEMParallaxAdapter()
        self.context = ParallaxWorkflowContext()

    def reset(self) -> None:
        self.context.reset()

    def prepare_bf(self, datacube: Any, params: BFMaskParams) -> ParallaxStageResult:
        start = perf_counter()
        data = np.asarray(getattr(datacube, "data", datacube))
        if data.ndim != 4:
            raise ParallaxServiceError(f"Parallax requires 4D data, got shape {data.shape}.")
        mean_dp = np.mean(data, axis=(0, 1))
        normalized = mean_dp / max(float(np.nanmax(mean_dp)), 1e-12)
        if params.use_circle:
            x, y = np.indices(mean_dp.shape)
            cx = params.center_x if params.center_x is not None else mean_dp.shape[0] / 2
            cy = params.center_y if params.center_y is not None else mean_dp.shape[1] / 2
            radius = params.radius if params.radius is not None else min(mean_dp.shape) / 4
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
        else:
            mask = normalized > params.threshold
        incoherent_bf = np.sum(data[..., mask], axis=-1)
        selected = np.argwhere(mask)
        virtual: dict[str, np.ndarray] = {}
        if selected.size:
            indices = np.linspace(0, len(selected) - 1, min(params.virtual_bf_count, len(selected)), dtype=int)
            for output_index, selected_index in enumerate(indices, start=1):
                qx, qy = selected[selected_index]
                virtual[f"Tilted virtual BF {output_index}"] = data[:, :, qx, qy]
        images = {
            "Mean diffraction pattern": mean_dp,
            "Accepted-mask preview": mask.astype(float),
            "Incoherent BF": incoherent_bf,
            **virtual,
        }
        if params.finite_dose_enabled:
            rng = np.random.default_rng(params.finite_dose_seed)
            normalized_bf = incoherent_bf / max(float(np.nanmean(incoherent_bf)), 1e-12)
            for dose in params.finite_doses:
                images[f"Finite dose {dose:g} e/A2"] = rng.poisson(
                    np.clip(normalized_bf * max(dose, 0.0), 0, None)
                ).astype(float)
        result = ParallaxStageResult(
            "bf",
            images,
            {
                **asdict(params),
                "mask_pixels": int(np.count_nonzero(mask)),
                "bf_mask": mask.copy(),
            },
            perf_counter() - start,
        )
        self.context.bf_result = result
        self.context.accepted_bf_mask = None
        self.context.alignment_result = None
        self.context.alignment_accepted = False
        self.context.advanced_result = None
        self.context.parallax = None
        return result

    def accept_bf_mask(self) -> np.ndarray:
        if self.context.bf_result is None:
            raise ParallaxServiceError("Prepare and review the BF disk before accepting it.")
        self.context.accepted_bf_mask = np.asarray(
            self.context.bf_result.metadata["bf_mask"], dtype=bool
        ).copy()
        return self.context.accepted_bf_mask.copy()

    def align(
        self,
        datacube: Any,
        params: ParallaxAlignmentParams,
        progress_callback: ProgressCallback | None = None,
    ) -> PhaseContrastResult:
        if self.context.accepted_bf_mask is None:
            raise ParallaxServiceError("Accept the BF disk before running Parallax alignment.")
        emit = progress_callback or (lambda _message, _fraction: None)
        start = perf_counter()
        emit("BF Preparation", 0.10)
        parallax = self.adapter.construct(datacube, params)
        emit("Preprocess", 0.35)
        parallax = self.adapter.preprocess(parallax, self.context.accepted_bf_mask.copy(), params)
        emit("Reconstruct", 0.70)
        parallax = self.adapter.reconstruct(parallax, params)
        self.context.parallax = parallax
        self.context.alignment_accepted = False
        result = self._extract(parallax, perf_counter() - start, "alignment")
        result.images["Accepted BF Mask"] = self.context.accepted_bf_mask.astype(float)
        self._add_shift_maps(result, parallax, self.context.accepted_bf_mask)
        self.context.alignment_result = result
        self.context.advanced_result = None
        self.context.adapter_metadata = {
            "adapter": type(self.adapter).__name__,
            "py4dstem_version": getattr(self.adapter._module(), "__version__", "unknown"),
            "alignment_params": asdict(params),
            "accepted_bf_mask_shape": list(self.context.accepted_bf_mask.shape),
        }
        emit("Review", 1.0)
        return result

    def accept_alignment(self) -> PhaseContrastResult:
        if self.context.alignment_result is None:
            raise ParallaxServiceError("Run Parallax alignment before accepting its review.")
        self.context.alignment_accepted = True
        return self.context.alignment_result

    def run_advanced(
        self,
        params: ParallaxAdvancedParams,
        progress_callback: ProgressCallback | None = None,
    ) -> PhaseContrastResult:
        if self.context.parallax is None or not self.context.alignment_accepted:
            raise ParallaxServiceError("Accept the alignment review before advanced reconstruction.")
        emit = progress_callback or (lambda _message, _fraction: None)
        start = perf_counter()
        parallax = self.context.parallax
        emit("Advanced Reconstruction", 0.10)
        if params.run_subpixel:
            parallax = self.adapter.subpixel(parallax, params)
        if params.run_aberration_fit or params.run_high_order_fit or params.run_ctf_fit:
            parallax = self.adapter.aberration_fit(parallax, params)
        if params.run_aberration_correction:
            parallax = self.adapter.aberration_correct(parallax)
        self.context.parallax = parallax
        result = self._extract(parallax, perf_counter() - start, "advanced")
        self._add_shift_maps(result, parallax, self.context.accepted_bf_mask)
        self.context.advanced_result = result
        self.context.adapter_metadata["advanced_params"] = asdict(params)
        emit("Advanced Reconstruction", 1.0)
        return result

    def save_package(self, directory: str | Path, save_figures: bool = False) -> list[Path]:
        if self.context.parallax is None:
            raise ParallaxServiceError("Run Parallax alignment before saving a package.")
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        h5_path = output / "parallax_reconstruction.h5"
        module = self.adapter._module()
        if hasattr(module, "save"):
            module.save(str(h5_path), self.context.parallax)
            saved.append(h5_path)
        metadata = {
            **self.context.adapter_metadata,
            "alignment_accepted": self.context.alignment_accepted,
            "has_advanced_result": self.context.advanced_result is not None,
            "save_figures": save_figures,
        }
        json_path = output / "parallax_pipeline_metadata.json"
        json_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        saved.append(json_path)
        if save_figures:
            figures = output / "figures"
            figures.mkdir(exist_ok=True)
            result = self.context.advanced_result or self.context.alignment_result
            if result is not None:
                import matplotlib.pyplot as plt

                for name, image in result.images.items():
                    path = figures / f"{name.lower().replace(' ', '_')}.png"
                    plt.imsave(path, np.asarray(image), cmap="gray")
                    saved.append(path)
        return saved

    @staticmethod
    def _extract(parallax: Any, elapsed: float, stage: str) -> PhaseContrastResult:
        images: dict[str, np.ndarray] = {}
        aligned = None
        aligned_attrs = (
            ("recon_BF_subpixel_aligned", "_recon_BF_subpixel_aligned", "recon_phase_corrected")
            if stage == "advanced"
            else ("recon_BF", "_recon_BF", "object_cropped", "object")
        )
        for attr in aligned_attrs:
            try:
                value = np.asarray(getattr(parallax, attr))
                if value.ndim >= 2:
                    aligned = value
                    images["Aligned BF" if stage == "alignment" else "Advanced BF"] = value
                    break
            except Exception:
                continue
        try:
            error = np.asarray(parallax.error_iterations, dtype=float)
            if error.size:
                images["Convergence"] = error[None, :]
        except Exception:
            pass
        try:
            corrected = np.asarray(parallax.recon_phase_corrected)
            images["Aberration Corrected BF"] = corrected
        except Exception:
            pass
        rotation = None
        try:
            rotation = float(np.rad2deg(parallax._rotation_best_rad))
        except Exception:
            pass
        return PhaseContrastResult(
            method="Parallax",
            images=images,
            elapsed_seconds=elapsed,
            rotation_degrees=rotation,
            object_phase=aligned,
        )

    @staticmethod
    def _add_shift_maps(
        result: PhaseContrastResult,
        parallax: Any,
        accepted_bf_mask: np.ndarray | None,
    ) -> None:
        if accepted_bf_mask is None:
            return
        try:
            shifts = np.asarray(parallax._xy_shifts)
            positions = np.argwhere(accepted_bf_mask)
            count = min(len(positions), len(shifts))
            shift_x = np.full(accepted_bf_mask.shape, np.nan, dtype=float)
            shift_y = np.full(accepted_bf_mask.shape, np.nan, dtype=float)
            shift_x[positions[:count, 0], positions[:count, 1]] = shifts[:count, 0]
            shift_y[positions[:count, 0], positions[:count, 1]] = shifts[:count, 1]
            result.images["Shift X"] = shift_x
            result.images["Shift Y"] = shift_y
            result.images["Shift Magnitude"] = np.hypot(shift_x, shift_y)
        except Exception:
            return
