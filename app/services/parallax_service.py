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

FAST_ALIGNMENT_BINS = (32, 32, 16, 16, 8, 8)
NOTEBOOK_ALIGNMENT_BINS = (32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 8, 8)


@dataclass(frozen=True)
class BFMaskParams:
    threshold: float = 0.5
    use_circle: bool = False
    center_x: float | None = None
    center_y: float | None = None
    radius: float | None = None
    virtual_bf_count: int = 5
    virtual_bf_crop: int = 48


@dataclass(frozen=True)
class ParallaxAlignmentParams:
    energy: float = 300e3
    device: str = "cpu"
    object_padding_px: tuple[int, int] = (16, 16)
    edge_blend: int = 8
    normalize_images: bool = False
    threshold_intensity: float = 0.6
    alignment_bin_values: tuple[int, ...] = FAST_ALIGNMENT_BINS
    regularize_shifts: bool = False
    cross_correlation_upsample_factor: int = 4


@dataclass(frozen=True)
class ParallaxAdvancedParams:
    kde_upsample_factor: int = 4
    kde_sigma_px: float = 0.125
    high_order_fit: bool = False
    max_radial_order: int = 3
    max_angular_order: int = 4
    ctf_thon_ring_fit: bool = False
    max_thon_rings: int = 5


@dataclass(frozen=True)
class FiniteDoseParams:
    doses: tuple[float, ...] = (100.0, 50.0, 10.0)
    seed: int = 1234


@dataclass(frozen=True)
class ParallaxAdapterCapabilities:
    subpixel_alignment: bool
    aberration_fit: bool
    aberration_correction: bool
    ctf_thon_ring_fit: bool = False
    derived_shift_diagnostics: bool = True
    derived_ctf_diagnostics: bool = True


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
    subpixel_result: PhaseContrastResult | None = None
    aberration_result: PhaseContrastResult | None = None
    correction_result: PhaseContrastResult | None = None
    finite_dose_result: ParallaxStageResult | None = None
    shift_vectors: np.ndarray | None = None
    aberrations_dict_polar: dict[str, object] = field(default_factory=dict)
    parallax: Any | None = None
    adapter_metadata: dict[str, object] = field(default_factory=dict)
    revision: int = 0

    def reset(self) -> None:
        self.bf_result = None
        self.accepted_bf_mask = None
        self.alignment_result = None
        self.alignment_accepted = False
        self.subpixel_result = None
        self.aberration_result = None
        self.correction_result = None
        self.finite_dose_result = None
        self.shift_vectors = None
        self.aberrations_dict_polar.clear()
        self.parallax = None
        self.adapter_metadata.clear()
        self.revision += 1


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

    def capabilities(self) -> ParallaxAdapterCapabilities:
        if self.py4dstem is None:
            return ParallaxAdapterCapabilities(True, True, True, False)
        try:
            cls = self.py4dstem.process.phase.Parallax
        except Exception:
            return ParallaxAdapterCapabilities(True, True, True, False)
        if not hasattr(cls, "aberration_fit"):
            return ParallaxAdapterCapabilities(True, True, True, False)
        try:
            fit_signature = inspect.signature(cls.aberration_fit)
            native_ctf_fit = "fit_CTF_FFT" in fit_signature.parameters
        except (TypeError, ValueError):
            native_ctf_fit = False
        if not native_ctf_fit:
            try:
                source = inspect.getsource(cls.aberration_fit)
                native_ctf_fit = (
                    "fit_CTF_FFT" in source and "fit_max_thon_rings" in source
                )
            except (OSError, TypeError):
                pass
        return ParallaxAdapterCapabilities(
            subpixel_alignment=hasattr(cls, "subpixel_alignment"),
            aberration_fit=hasattr(cls, "aberration_fit"),
            aberration_correction=hasattr(cls, "aberration_correct"),
            ctf_thon_ring_fit=native_ctf_fit,
        )

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
        radial_order = params.max_radial_order if params.high_order_fit else 3
        angular_order = params.max_angular_order if params.high_order_fit else 4
        kwargs = self.supported_kwargs(
            parallax.aberration_fit,
            {
                "max_radial_order": radial_order,
                "max_angular_order": angular_order,
                "plot_CTF_comparison": False,
                "plot_BF_shifts_comparison": False,
                "fit_CTF_FFT": params.ctf_thon_ring_fit,
                "fit_max_thon_rings": params.max_thon_rings,
            },
        )
        return parallax.aberration_fit(**kwargs)

    def aberration_correct(self, parallax: Any) -> Any:
        kwargs = self.supported_kwargs(
            parallax.aberration_correct,
            {"plot_corrected_phase": False, "upsampled": True},
        )
        return parallax.aberration_correct(**kwargs)

    def save(self, path: Path, parallax: Any) -> None:
        save = self._module().save
        kwargs = self.supported_kwargs(save, {"mode": "o"})
        save(str(path), parallax, **kwargs)

    @staticmethod
    def aberration_diagnostics(parallax: Any) -> dict[str, np.ndarray]:
        diagnostics: dict[str, np.ndarray] = {}
        try:
            corner_indices = np.asarray(parallax._xy_inds) - np.asarray(
                parallax._region_of_interest_shape
            ) // 2
            raveled = np.ravel_multi_index(
                corner_indices.T, tuple(parallax._region_of_interest_shape), mode="wrap"
            )
            gradients = np.asarray(
                (
                    np.asarray(parallax._aberrations_basis_du)[raveled],
                    np.asarray(parallax._aberrations_basis_dv)[raveled],
                )
            )
            fitted = np.tensordot(
                gradients, np.asarray(parallax._aberrations_coefs), axes=1
            ).T
            diagnostics["Fitted Shift X"] = fitted[:, 0]
            diagnostics["Fitted Shift Y"] = fitted[:, 1]
        except Exception:
            pass
        try:
            image = np.asarray(
                parallax.recon_BF_subpixel_aligned
                if hasattr(parallax, "recon_BF_subpixel_aligned")
                else parallax.recon_BF
            )
            sampling = (
                float(parallax._scan_sampling[0]) / float(getattr(parallax, "_kde_upsample_factor", 1)),
                float(parallax._scan_sampling[1]) / float(getattr(parallax, "_kde_upsample_factor", 1)),
            )
            ctf = parallax._calculate_CTF(
                image.shape,
                sampling,
                np.asarray(parallax._aberrations_mn),
                np.asarray(parallax._aberrations_coefs),
            )
            diagnostics["Fitted CTF"] = np.abs(np.sin(np.asarray(ctf)))
        except Exception:
            pass
        return diagnostics


class ParallaxService:
    def __init__(self, adapter: Py4DSTEMParallaxAdapter | None = None) -> None:
        self.adapter = adapter or Py4DSTEMParallaxAdapter()
        self.context = ParallaxWorkflowContext()

    def reset(self) -> None:
        self.context.reset()

    def prepare_bf(self, datacube: Any, params: BFMaskParams) -> ParallaxStageResult:
        start = perf_counter()
        data = getattr(datacube, "data", datacube)
        if data.ndim != 4:
            raise ParallaxServiceError(f"Parallax requires 4D data, got shape {data.shape}.")
        try:
            mean_dp = np.asarray(datacube.get_dp_mean().data)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            mean_dp = np.asarray(data).mean(axis=(0, 1))
        normalized = mean_dp / max(float(np.nanmax(mean_dp)), 1e-12)
        if params.use_circle:
            x, y = np.indices(mean_dp.shape)
            cx = params.center_x if params.center_x is not None else mean_dp.shape[0] / 2
            cy = params.center_y if params.center_y is not None else mean_dp.shape[1] / 2
            radius = params.radius if params.radius is not None else min(mean_dp.shape) / 4
            mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
        else:
            mask = normalized > params.threshold
        mask_count = int(np.count_nonzero(mask))
        if mask_count == 0:
            raise ParallaxServiceError("The BF disk mask is empty. Lower the threshold or adjust the circle.")
        incoherent_bf = np.zeros(tuple(data.shape[:2]), dtype=np.result_type(data.dtype, np.float32))
        selected = np.argwhere(mask)
        for qx, qy in selected:
            incoherent_bf += np.asarray(data[:, :, qx, qy])
        virtual: dict[str, np.ndarray] = {}
        selected_indices = self._representative_indices(selected, params.virtual_bf_count)
        crop = max(1, min(int(params.virtual_bf_crop), data.shape[0], data.shape[1]))
        for output_index, selected_index in enumerate(selected_indices, start=1):
            qx, qy = selected[selected_index]
            virtual[f"Tilted virtual BF {output_index}"] = np.asarray(
                data[:crop, :crop, qx, qy]
            )
        images = {
            "Mean diffraction pattern": mean_dp,
            "Accepted-mask preview": mask.astype(float),
            "Incoherent BF": incoherent_bf,
            **virtual,
        }
        result = ParallaxStageResult(
            "bf",
            images,
            {
                **asdict(params),
                "mask_pixels": mask_count,
                "mask_fraction": float(mask.mean()),
                "mask_acceptable": bool(mask_count <= mask.size * 0.75),
                "bf_mask": mask.copy(),
                "selected_points": selected[selected_indices].copy(),
            },
            perf_counter() - start,
        )
        self.context.bf_result = result
        self.context.accepted_bf_mask = None
        self.context.alignment_result = None
        self.context.alignment_accepted = False
        self.context.subpixel_result = None
        self.context.aberration_result = None
        self.context.correction_result = None
        self.context.parallax = None
        self.context.revision += 1
        return result

    def accept_bf_mask(self) -> np.ndarray:
        if self.context.bf_result is None:
            raise ParallaxServiceError("Prepare and review the BF disk before accepting it.")
        if not bool(self.context.bf_result.metadata.get("mask_acceptable", True)):
            raise ParallaxServiceError(
                "The BF disk mask covers more than 75% of the detector. Refine it before acceptance."
            )
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
        self.context.shift_vectors = self._shift_vectors(parallax, self.context.accepted_bf_mask)
        self.context.alignment_result = result
        self.context.subpixel_result = None
        self.context.aberration_result = None
        self.context.correction_result = None
        self.context.adapter_metadata = {
            "adapter": type(self.adapter).__name__,
            "py4dstem_version": getattr(self.adapter._module(), "__version__", "unknown"),
            "alignment_params": asdict(params),
            "accepted_bf_mask_shape": list(self.context.accepted_bf_mask.shape),
        }
        self.context.revision += 1
        emit("Review", 1.0)
        return result

    def accept_alignment(self) -> PhaseContrastResult:
        if self.context.alignment_result is None:
            raise ParallaxServiceError("Run Parallax alignment before accepting its review.")
        self.context.alignment_accepted = True
        return self.context.alignment_result

    def run_subpixel(
        self,
        params: ParallaxAdvancedParams,
        progress_callback: ProgressCallback | None = None,
    ) -> PhaseContrastResult:
        if self.context.parallax is None or not self.context.alignment_accepted:
            raise ParallaxServiceError("Accept the alignment review before advanced reconstruction.")
        emit = progress_callback or (lambda _message, _fraction: None)
        start = perf_counter()
        parallax = self.context.parallax
        emit("Subpixel Reconstruction", 0.10)
        parallax = self.adapter.subpixel(parallax, params)
        self.context.parallax = parallax
        result = self._extract(parallax, perf_counter() - start, "subpixel")
        self._add_subpixel_diagnostics(result, parallax)
        self._add_shift_maps(result, parallax, self.context.accepted_bf_mask)
        self.context.subpixel_result = result
        self.context.aberration_result = None
        self.context.correction_result = None
        self.context.adapter_metadata["subpixel_params"] = asdict(params)
        self.context.revision += 1
        emit("Subpixel Reconstruction", 1.0)
        return result

    def fit_aberrations(
        self, params: ParallaxAdvancedParams, progress_callback: ProgressCallback | None = None
    ) -> PhaseContrastResult:
        if self.context.parallax is None or not self.context.alignment_accepted:
            raise ParallaxServiceError("Accept the alignment review before fitting aberrations.")
        emit = progress_callback or (lambda _message, _fraction: None)
        emit("Aberration Fitting", 0.2)
        start = perf_counter()
        self.context.parallax = self.adapter.aberration_fit(self.context.parallax, params)
        result = self._extract(self.context.parallax, perf_counter() - start, "aberration")
        self._add_shift_maps(result, self.context.parallax, self.context.accepted_bf_mask)
        self.context.aberrations_dict_polar = dict(
            getattr(self.context.parallax, "aberrations_dict_polar", {})
        )
        diagnostics = self.adapter.aberration_diagnostics(self.context.parallax)
        fitted_x = diagnostics.pop("Fitted Shift X", None)
        fitted_y = diagnostics.pop("Fitted Shift Y", None)
        result.images.update(diagnostics)
        try:
            aligned = np.asarray(
                getattr(
                    self.context.parallax,
                    "recon_BF_subpixel_aligned",
                    self.context.parallax.recon_BF,
                )
            )
            result.images["Aligned BF FFT"] = np.fft.fftshift(np.abs(np.fft.fft2(aligned)))
        except Exception:
            pass
        if fitted_x is not None and fitted_y is not None and self.context.accepted_bf_mask is not None:
            points = np.argwhere(self.context.accepted_bf_mask)
            count = min(len(points), len(fitted_x), len(fitted_y))
            magnitude = np.full(self.context.accepted_bf_mask.shape, np.nan)
            magnitude[points[:count, 0], points[:count, 1]] = np.hypot(
                np.asarray(fitted_x)[:count], np.asarray(fitted_y)[:count]
            )
            result.images["Fitted Shift Magnitude"] = magnitude
            measured = self._shift_vectors(
                self.context.parallax, self.context.accepted_bf_mask
            )
            fitted = np.column_stack(
                (points[:count], np.asarray(fitted_x)[:count], np.asarray(fitted_y)[:count])
            )
            result.metadata["measured_shift_vectors"] = measured
            result.metadata["fitted_shift_vectors"] = fitted
        result.metadata["aberrations_dict_polar"] = dict(
            self.context.aberrations_dict_polar
        )
        result.metadata["ctf_thon_ring_native"] = self.adapter.capabilities().ctf_thon_ring_fit
        self.context.aberration_result = result
        self.context.correction_result = None
        self.context.adapter_metadata["aberration_params"] = asdict(params)
        self.context.adapter_metadata["aberrations_dict_polar"] = self.context.aberrations_dict_polar
        self.context.revision += 1
        emit("Aberration Fitting", 1.0)
        return result

    def apply_aberration_correction(
        self, progress_callback: ProgressCallback | None = None
    ) -> PhaseContrastResult:
        if self.context.parallax is None or self.context.aberration_result is None:
            raise ParallaxServiceError("Fit aberrations before applying aberration correction.")
        emit = progress_callback or (lambda _message, _fraction: None)
        emit("Aberration Correction", 0.2)
        start = perf_counter()
        self.context.parallax = self.adapter.aberration_correct(self.context.parallax)
        result = self._extract(self.context.parallax, perf_counter() - start, "correction")
        self.context.correction_result = result
        self.context.revision += 1
        emit("Aberration Correction", 1.0)
        return result

    def run_finite_dose_comparison(
        self,
        datacube: Any,
        alignment_params: ParallaxAlignmentParams,
        params: FiniteDoseParams,
        progress_callback: ProgressCallback | None = None,
    ) -> ParallaxStageResult:
        if self.context.accepted_bf_mask is None:
            raise ParallaxServiceError("Accept the BF disk before finite-dose comparison.")
        emit = progress_callback or (lambda _message, _fraction: None)
        images: dict[str, np.ndarray] = {}
        area = self._realspace_pixel_area(datacube)
        for index, dose in enumerate(params.doses):
            emit(f"Finite dose {dose:g} e/A2", (index + 1) / max(len(params.doses), 1))
            noisy = datacube.copy()
            rng = np.random.default_rng(params.seed + index)
            noisy.data = rng.poisson(
                np.clip(np.asarray(datacube.data) * dose * area, 0, None)
            ).astype(np.uint64)
            positions = self._representative_scan_positions(tuple(noisy.data.shape[:2]))
            patterns = [np.asarray(noisy.data[rx, ry]) for rx, ry in positions]
            images[f"Diffraction montage {dose:g} e/A2"] = np.concatenate(patterns, axis=1)
            parallax = self.adapter.construct(noisy, alignment_params)
            parallax = self.adapter.preprocess(
                parallax, self.context.accepted_bf_mask.copy(), alignment_params
            )
            parallax = self.adapter.reconstruct(parallax, alignment_params)
            extracted = self._extract(parallax, 0.0, "alignment")
            if "Aligned BF" in extracted.images:
                images[f"Aligned BF {dose:g} e/A2"] = extracted.images["Aligned BF"]
        result = ParallaxStageResult("finite_dose", images, asdict(params))
        self.context.finite_dose_result = result
        self.context.revision += 1
        return result

    def save_package(self, directory: str | Path, save_figures: bool = False) -> list[Path]:
        if self.context.parallax is None:
            raise ParallaxServiceError("Run Parallax alignment before saving a package.")
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        h5_path = output / "parallax_reconstruction.h5"
        self.adapter.save(h5_path, self.context.parallax)
        saved.append(h5_path)
        metadata = {
            **self.context.adapter_metadata,
            "alignment_accepted": self.context.alignment_accepted,
            "has_subpixel_result": self.context.subpixel_result is not None,
            "has_aberration_result": self.context.aberration_result is not None,
            "has_correction_result": self.context.correction_result is not None,
            "save_figures": save_figures,
        }
        json_path = output / "parallax_pipeline_metadata.json"
        json_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        saved.append(json_path)
        if save_figures:
            figures = output / "figures"
            figures.mkdir(exist_ok=True)
            result = (
                self.context.correction_result
                or self.context.aberration_result
                or self.context.subpixel_result
                or self.context.alignment_result
            )
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
            if stage in {"subpixel", "aberration", "correction"}
            else ("recon_BF", "_recon_BF", "object_cropped", "object")
        )
        for attr in aligned_attrs:
            try:
                value = np.asarray(getattr(parallax, attr))
                if value.ndim >= 2:
                    aligned = value
                    images["Aligned BF" if stage == "alignment" else "Subpixel Aligned BF"] = value
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
    def _representative_indices(points: np.ndarray, count: int) -> list[int]:
        center = points.mean(axis=0)
        candidates = [
            np.argmin(np.sum((points - center) ** 2, axis=1)),
            np.argmax(points[:, 0]),
            np.argmin(points[:, 1]),
            np.argmax(points[:, 1]),
            np.argmin(points[:, 0]),
        ]
        unique: list[int] = []
        for index in candidates:
            value = int(index)
            if value not in unique:
                unique.append(value)
        if len(unique) < count:
            for value in np.linspace(0, len(points) - 1, min(count, len(points)), dtype=int):
                if int(value) not in unique:
                    unique.append(int(value))
        return unique[:max(1, min(int(count), len(unique)))]

    @staticmethod
    def _shift_vectors(parallax: Any, mask: np.ndarray) -> np.ndarray | None:
        try:
            points = np.argwhere(mask)
            shifts = np.asarray(parallax._xy_shifts)
            count = min(len(points), len(shifts))
            return np.column_stack((points[:count], shifts[:count]))
        except Exception:
            return None

    @staticmethod
    def _add_subpixel_diagnostics(result: PhaseContrastResult, parallax: Any) -> None:
        try:
            original = np.asarray(parallax.recon_BF)
            result.images["Original Aligned BF"] = original
            result.images["Original Aligned BF FFT"] = np.fft.fftshift(
                np.abs(np.fft.fft2(original))
            )
        except Exception:
            pass
        try:
            subpixel = np.asarray(parallax.recon_BF_subpixel_aligned)
            sampling = float(parallax._scan_sampling[0]) / float(parallax._kde_upsample_factor)
            nx, ny = subpixel.shape
            kx = np.fft.fftfreq(nx, d=sampling)
            ky = np.fft.fftfreq(ny, d=sampling)
            radius = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
            cone = np.abs(np.fft.fft2(subpixel)) * radius
            result.images["Cone-weighted FFT"] = np.fft.fftshift(cone)
            result.images["Subpixel Aligned BF FFT"] = np.fft.fftshift(
                np.abs(np.fft.fft2(subpixel))
            )
            bins = np.linspace(0, float(radius.max()), max(min(nx, ny) // 2, 2))
            indices = np.digitize(radius.ravel(), bins)
            radial = np.array([
                cone.ravel()[indices == index].mean() if np.any(indices == index) else 0
                for index in range(1, len(bins))
            ])
            result.images["Radial cone-weighted FFT"] = radial[None, :]
            result.metadata["radial_cone_frequency"] = bins[1:]
            result.metadata["radial_cone_values"] = ParallaxService._median_filter_1d(radial, 5)
        except Exception:
            pass

    @staticmethod
    def _median_filter_1d(values: np.ndarray, width: int) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        radius = max(int(width) // 2, 0)
        if radius == 0 or not len(array):
            return array.copy()
        padded = np.pad(array, radius, mode="edge")
        return np.asarray([
            np.median(padded[index:index + 2 * radius + 1])
            for index in range(len(array))
        ])

    @staticmethod
    def _representative_scan_positions(shape: tuple[int, int]) -> list[tuple[int, int]]:
        nx, ny = shape
        return [
            (nx // 3, ny // 2),
            (2 * nx // 3, ny // 2),
            (nx // 3, ny // 4),
        ]

    @staticmethod
    def _realspace_pixel_area(datacube: Any) -> float:
        try:
            sampling = datacube.calibration.get_R_pixel_size()
            if np.isscalar(sampling):
                return float(sampling) ** 2
            values = tuple(float(value) for value in sampling)
            return values[0] * values[1]
        except Exception as exc:
            raise ParallaxServiceError(
                "Finite-dose comparison requires calibrated real-space pixel size."
            ) from exc

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
