from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import import_module
from time import perf_counter
from typing import Any

import numpy as np

from app.services.computation_task import ComputationTask

logger = logging.getLogger(__name__)


class PhaseContrastServiceError(Exception):
    pass


@dataclass(frozen=True)
class PtychographyParams:
    energy: float = 80e3
    defocus: float = 500.0
    vacuum_probe_path: str | None = None
    device: str = "cpu"
    storage: str = "cpu"
    object_type: str = "potential"
    object_positivity: bool = True
    num_iter: int = 64
    max_batch_size: int = 512
    seed_random: int = 0


@dataclass(frozen=True)
class ParallaxParams:
    energy: float = 300e3
    device: str = "cpu"
    object_padding_px: tuple[int, int] = (16, 16)
    edge_blend: int = 8
    alignment_bin_values: tuple[int, ...] = (32, 32, 16, 16, 8, 8)
    regularize_shifts: bool = False
    normalize_images: bool = False
    threshold_intensity: float = 0.6


@dataclass(frozen=True)
class DPCParams:
    energy: float = 200e3
    plot_center_of_mass: str = "off"
    force_com_rotation: float | None = None
    force_com_transpose: bool = False


@dataclass(frozen=True)
class DPCSegmentedParams:
    energy: float = 200e3
    sampling_x: float = 0.246570625
    sampling_y: float = 0.246570625
    rotation_offset_degrees: float = 60.0
    inner_radius_mrad: float = 10.0
    outer_radius_mrad: float = 25.0
    center_x: float | None = None
    center_y: float | None = None


@dataclass(frozen=True)
class DPCPreprocessParams:
    energy: float = 200e3
    padding_factor: float = 2.0
    rotation_start_degrees: float = -90.0
    rotation_end_degrees: float = 90.0
    rotation_step_degrees: float = 1.0
    maximize_divergence: bool = False
    fit_function: str = "plane"
    force_com_rotation: float | None = None
    force_com_transpose: bool | None = None
    force_com_shift_x: float | None = None
    force_com_shift_y: float | None = None
    vectorized_com_calculation: bool = False
    use_dp_mask: bool = False
    mask_inner_mrad: float = 0.0
    mask_outer_mrad: float = 25.0


@dataclass(frozen=True)
class DPCReconstructionParams:
    reset: bool = True
    max_iter: int = 64
    step_size: float | None = None
    stopping_criterion: float = 1e-6
    backtrack: bool = True
    gaussian_filter: bool = True
    gaussian_filter_sigma: float | None = None
    butterworth_filter: bool = True
    q_lowpass: float | None = None
    q_highpass: float | None = None
    butterworth_order: float = 2.0
    store_iterations: bool = False


@dataclass(frozen=True)
class DPCStageResult:
    stage: str
    images: dict[str, np.ndarray] = field(default_factory=dict)
    complex_images: dict[str, np.ndarray] = field(default_factory=dict)
    masks: tuple[np.ndarray, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class DPCAcceptanceState:
    preprocessing: bool = False


@dataclass(frozen=True)
class PhaseContrastResult:
    method: str
    images: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    rotation_degrees: float | None = None
    object_phase: np.ndarray | None = None
    object_amplitude: np.ndarray | None = None
    probe: np.ndarray | None = None
    probe_fourier: np.ndarray | None = None
    com_x: np.ndarray | None = None
    com_y: np.ndarray | None = None
    complex_com: np.ndarray | None = None
    reconstructed_potential: np.ndarray | None = None
    com_measured_x: np.ndarray | None = None
    com_measured_y: np.ndarray | None = None
    com_fitted_x: np.ndarray | None = None
    com_fitted_y: np.ndarray | None = None
    com_normalized_x: np.ndarray | None = None
    com_normalized_y: np.ndarray | None = None
    rotation_angles_degrees: np.ndarray | None = None
    rotation_metric: np.ndarray | None = None
    rotation_metric_transpose: np.ndarray | None = None
    transpose: bool | None = None
    error_iterations: np.ndarray | None = None
    object_iterations: tuple[np.ndarray, ...] = ()


class PhaseContrastService:
    PTYCHOGRAPHY = "Ptychography"
    PARALLAX = "Parallax"
    DPC = "DPC"

    def __init__(self) -> None:
        self.dpc: Any | None = None
        self.dpc_preprocess_result: PhaseContrastResult | None = None
        self.dpc_acceptance = DPCAcceptanceState()

    def reset_dpc_workflow(self) -> None:
        self.dpc = None
        self.dpc_preprocess_result = None
        self.dpc_acceptance = DPCAcceptanceState()

    @staticmethod
    def dpc_sto_preset() -> DPCSegmentedParams:
        return DPCSegmentedParams()

    def generate_segmented_dpc(
        self,
        datacube: Any,
        params: DPCSegmentedParams,
    ) -> DPCStageResult:
        start = perf_counter()
        data = np.asarray(getattr(datacube, "data", datacube))
        if data.ndim != 4:
            raise PhaseContrastServiceError(
                f"Segmented DPC requires 4D data, got shape {data.shape}."
            )
        masks = self._annular_segment_masks(data.shape[-2:], params)
        intensities = tuple(np.sum(data * mask, axis=(-2, -1)) for mask in masks)
        segmented_x = intensities[0] - intensities[2]
        segmented_y = intensities[1] - intensities[3]
        qx, qy = np.indices(data.shape[-2:], dtype=float)
        center = np.asarray(
            [
                params.center_x if params.center_x is not None else data.shape[-2] / 2,
                params.center_y if params.center_y is not None else data.shape[-1] / 2,
            ]
        )
        monolithic_x = np.zeros(data.shape[:2], dtype=float)
        monolithic_y = np.zeros(data.shape[:2], dtype=float)
        for mask in masks:
            selected = mask.astype(bool)
            if not selected.any():
                continue
            weight_x = float(np.mean(qx[selected]) - center[0])
            weight_y = float(np.mean(qy[selected]) - center[1])
            segment = np.sum(data * mask, axis=(-2, -1))
            monolithic_x += segment * weight_x
            monolithic_y += segment * weight_y
        mean_dp = np.mean(data, axis=(0, 1))
        images = {"Mean diffraction pattern": mean_dp}
        images.update({f"Segment {index + 1} mask": mask for index, mask in enumerate(masks)})
        images.update(
            {f"Segment {index + 1} intensity": image for index, image in enumerate(intensities)}
        )
        images.update(
            {
                "Segmented CoM X": segmented_x,
                "Segmented CoM Y": segmented_y,
                "Weighted CoM X": monolithic_x,
                "Weighted CoM Y": monolithic_y,
            }
        )
        return DPCStageResult(
            stage="segmented",
            images=images,
            complex_images={
                "Segmented complex CoM": segmented_x + 1j * segmented_y,
                "Weighted complex CoM": monolithic_x + 1j * monolithic_y,
            },
            masks=masks,
            metadata={
                "energy": params.energy,
                "rotation_offset_degrees": params.rotation_offset_degrees,
                "inner_radius_mrad": params.inner_radius_mrad,
                "outer_radius_mrad": params.outer_radius_mrad,
                "sampling": (params.sampling_x, params.sampling_y),
                "center": (
                    params.center_x if params.center_x is not None else data.shape[-2] / 2,
                    params.center_y if params.center_y is not None else data.shape[-1] / 2,
                ),
            },
            elapsed_seconds=perf_counter() - start,
        )

    def preprocess_dpc(
        self,
        datacube: Any,
        params: DPCPreprocessParams,
    ) -> PhaseContrastResult:
        py4DSTEM = self._py4dstem()
        start = perf_counter()
        step = max(float(params.rotation_step_degrees), 1e-9)
        rotation_angles = np.arange(
            params.rotation_start_degrees,
            params.rotation_end_degrees + step * 0.5,
            step,
        )
        kwargs: dict[str, Any] = {
            "padding_factor": params.padding_factor,
            "rotation_angles_deg": rotation_angles,
            "maximize_divergence": params.maximize_divergence,
            "fit_function": params.fit_function,
            "vectorized_com_calculation": params.vectorized_com_calculation,
            "plot_center_of_mass": False,
            "plot_rotation": False,
        }
        if params.force_com_rotation is not None:
            kwargs["force_com_rotation"] = params.force_com_rotation
        if params.force_com_transpose is not None:
            kwargs["force_com_transpose"] = params.force_com_transpose
        if params.force_com_shift_x is not None and params.force_com_shift_y is not None:
            kwargs["force_com_shifts"] = (params.force_com_shift_x, params.force_com_shift_y)
        if params.use_dp_mask:
            shape = tuple(np.asarray(getattr(datacube, "data", datacube)).shape[-2:])
            sampling_x, sampling_y = self._datacube_sampling(datacube)
            segmented = DPCSegmentedParams(
                energy=params.energy,
                sampling_x=sampling_x,
                sampling_y=sampling_y,
                inner_radius_mrad=params.mask_inner_mrad,
                outer_radius_mrad=params.mask_outer_mrad,
            )
            kwargs["dp_mask"] = np.sum(self._annular_segment_masks(shape, segmented), axis=0) > 0
        try:
            self.dpc = py4DSTEM.process.phase.DPC(
                energy=params.energy,
                datacube=datacube,
            ).preprocess(**kwargs)
        except Exception as exc:
            raise PhaseContrastServiceError(f"DPC preprocessing failed: {exc}") from exc
        self.dpc_acceptance = DPCAcceptanceState(False)
        result = self._extract_dpc(self.dpc, perf_counter() - start)
        self.dpc_preprocess_result = result
        return result

    def accept_dpc_preprocessing(self) -> DPCAcceptanceState:
        if self.dpc is None or self.dpc_preprocess_result is None:
            raise PhaseContrastServiceError("Run DPC preprocessing before accepting it.")
        self.dpc_acceptance = DPCAcceptanceState(True)
        return self.dpc_acceptance

    def reconstruct_dpc(self, params: DPCReconstructionParams) -> PhaseContrastResult:
        if self.dpc is None or not self.dpc_acceptance.preprocessing:
            raise PhaseContrastServiceError(
                "Review and accept pixelated CoM preprocessing before reconstruction."
            )
        start = perf_counter()
        try:
            reconstructed = self.dpc.reconstruct(
                reset=params.reset,
                max_iter=params.max_iter,
                step_size=params.step_size,
                stopping_criterion=params.stopping_criterion,
                backtrack=params.backtrack,
                progress_bar=True,
                gaussian_filter=params.gaussian_filter,
                gaussian_filter_sigma=params.gaussian_filter_sigma,
                butterworth_filter=params.butterworth_filter,
                q_lowpass=params.q_lowpass,
                q_highpass=params.q_highpass,
                butterworth_order=params.butterworth_order,
                store_iterations=params.store_iterations,
            )
            if reconstructed is not None:
                self.dpc = reconstructed
        except Exception as exc:
            raise PhaseContrastServiceError(f"DPC reconstruction failed: {exc}") from exc
        return self._extract_dpc(self.dpc, perf_counter() - start)

    def _annular_segment_masks(
        self,
        shape: tuple[int, int],
        params: DPCSegmentedParams,
    ) -> tuple[np.ndarray, ...]:
        py4DSTEM = self._py4dstem()
        wavelength = py4DSTEM.process.utils.electron_wavelength_angstrom(params.energy)
        alpha_x = np.fft.fftfreq(shape[0], params.sampling_x) * wavelength
        alpha_y = np.fft.fftfreq(shape[1], params.sampling_y) * wavelength
        alpha = np.sqrt(alpha_x[:, None] ** 2 + alpha_y[None, :] ** 2)
        radial = (params.inner_radius_mrad * 1e-3 <= alpha) & (
            alpha < params.outer_radius_mrad * 1e-3
        )
        theta = (
            np.arctan2(alpha_y[None, :], alpha_x[:, None])
            + np.deg2rad(params.rotation_offset_degrees)
        ) % (2 * np.pi)
        bins = np.floor(4 * theta / (2 * np.pi)).astype(int)
        masks = [np.fft.fftshift(((bins == index) & radial).astype(float)) for index in range(4)]
        default_center = np.asarray(shape, dtype=float) / 2
        target_center = np.asarray(
            [
                params.center_x if params.center_x is not None else default_center[0],
                params.center_y if params.center_y is not None else default_center[1],
            ]
        )
        shift = np.rint(target_center - default_center).astype(int)
        return tuple(np.roll(mask, tuple(shift), axis=(0, 1)) for mask in masks)

    @staticmethod
    def _datacube_sampling(datacube: Any) -> tuple[float, float]:
        calibration = getattr(datacube, "calibration", None)
        try:
            sampling = calibration.get_R_pixel_size()
            if np.isscalar(sampling):
                value = float(sampling)
                return value, value
            values = tuple(float(item) for item in sampling)
            if len(values) >= 2:
                return values[:2]
        except Exception:
            pass
        return DPCSegmentedParams().sampling_x, DPCSegmentedParams().sampling_y

    def compute_bf_df(
        self,
        datacube: Any,
        bf_radius: float | None = None,
        df_inner: float | None = None,
        df_outer: float | None = None,
        probe_geometry: Any | None = None,
    ) -> dict[str, np.ndarray]:
        try:
            if bf_radius is None and probe_geometry is not None:
                bf_radius = getattr(probe_geometry, "radius", 10.0)
            elif bf_radius is None:
                bf_radius = 10.0
            center_x = getattr(probe_geometry, "center_x", 0.0) if probe_geometry else 0.0
            center_y = getattr(probe_geometry, "center_y", 0.0) if probe_geometry else 0.0
            bf_result = datacube.get_virtual_image(
                mode="circle",
                geometry=((center_x, center_y), bf_radius),
                name="bright_field",
                centered=False,
                calibrated=False,
                returncalc=True,
            )
            bf_image = np.asarray(getattr(bf_result, "data", bf_result))
            results = {"Bright Field": bf_image}
            if df_inner is not None and df_outer is not None:
                df_result = datacube.get_virtual_image(
                    mode="annulus",
                    geometry=((center_x, center_y), (df_inner, df_outer)),
                    name="dark_field",
                    centered=False,
                    calibrated=False,
                    returncalc=True,
                )
                df_image = np.asarray(getattr(df_result, "data", df_result))
                results["Dark Field"] = df_image
            return results
        except Exception as exc:
            raise PhaseContrastServiceError(
                f"BF/DF computation failed: {exc}"
            ) from exc

    def compute_bf_df_task(
        self,
        datacube: Any,
        *,
        bf_radius: float | None = None,
        df_inner: float | None = None,
        df_outer: float | None = None,
        probe_geometry: Any | None = None,
    ) -> ComputationTask:
        data = getattr(datacube, "data", datacube)
        shape = tuple(int(value) for value in getattr(data, "shape", ()))
        params = {
            "bf_radius": bf_radius,
            "df_inner": df_inner,
            "df_outer": df_outer,
            "probe_center_x": getattr(probe_geometry, "center_x", None),
            "probe_center_y": getattr(probe_geometry, "center_y", None),
        }
        result_key = f"bf_df_preview:{shape}:{params}"
        return ComputationTask(
            name="BF / DF Preview",
            operation=lambda _progress: self.compute_bf_df(
                datacube,
                bf_radius=bf_radius,
                df_inner=df_inner,
                df_outer=df_outer,
                probe_geometry=probe_geometry,
            ),
            result_key=result_key,
            status_message="Preparing BF / DF preview",
            parameters={"shape": shape or "-", **params},
        )

    def run_ptychography(
        self,
        datacube: Any,
        params: PtychographyParams,
        progress_callback: Any | None = None,
    ) -> PhaseContrastResult:
        py4DSTEM = self._py4dstem()
        start = perf_counter()

        vacuum_probe = None
        if params.vacuum_probe_path:
            try:
                probe_data = py4DSTEM.read(params.vacuum_probe_path)
                vacuum_probe = np.asarray(getattr(probe_data, "data", probe_data))
            except Exception as exc:
                raise PhaseContrastServiceError(
                    f"Could not load vacuum probe: {exc}"
                ) from exc

        try:
            ptycho = py4DSTEM.process.phase.SingleslicePtychography(
                datacube=datacube,
                energy=params.energy,
                defocus=params.defocus,
                device=params.device,
                storage=params.storage,
                **({"vacuum_probe_intensity": vacuum_probe} if vacuum_probe is not None else {}),
            ).preprocess(
                plot_center_of_mass=False,
            )
        except Exception as exc:
            raise PhaseContrastServiceError(
                f"Ptychography preprocessing failed: {exc}"
            ) from exc

        try:
            ptycho = ptycho.reconstruct(
                reset=True,
                seed_random=params.seed_random,
                num_iter=params.num_iter,
                max_batch_size=params.max_batch_size,
                object_type=params.object_type,
                object_positivity=params.object_positivity,
            )
        except Exception as exc:
            raise PhaseContrastServiceError(
                f"Ptychography reconstruction failed: {exc}"
            ) from exc

        elapsed = perf_counter() - start
        result = self._extract_ptychography(ptycho, elapsed)
        return result

    def run_parallax(
        self,
        datacube: Any,
        params: ParallaxParams,
        progress_callback: Any | None = None,
    ) -> PhaseContrastResult:
        # Legacy aggregate-page compatibility. All py4DSTEM Parallax API calls
        # still pass through the version-aware adapter.
        from app.services.parallax_service import (
            BFMaskParams,
            ParallaxAlignmentParams,
            ParallaxService,
        )

        service = ParallaxService()
        try:
            service.prepare_bf(datacube, BFMaskParams(threshold=params.threshold_intensity))
            service.accept_bf_mask()
            return service.align(
                datacube,
                ParallaxAlignmentParams(
                    energy=params.energy,
                    device=params.device,
                    object_padding_px=params.object_padding_px,
                    edge_blend=params.edge_blend,
                    alignment_bin_values=params.alignment_bin_values,
                    regularize_shifts=params.regularize_shifts,
                    normalize_images=params.normalize_images,
                    threshold_intensity=params.threshold_intensity,
                ),
                progress_callback,
            )
        except Exception as exc:
            raise PhaseContrastServiceError(f"Parallax reconstruction failed: {exc}") from exc

    def run_dpc(
        self,
        datacube: Any,
        params: DPCParams,
        progress_callback: Any | None = None,
    ) -> PhaseContrastResult:
        self.preprocess_dpc(
            datacube,
            DPCPreprocessParams(
                energy=params.energy,
                force_com_rotation=params.force_com_rotation,
                force_com_transpose=True if params.force_com_transpose else None,
            ),
        )
        self.accept_dpc_preprocessing()
        return self.reconstruct_dpc(DPCReconstructionParams())

    def _extract_ptychography(self, ptycho: Any, elapsed: float) -> PhaseContrastResult:
        images: dict[str, np.ndarray] = {}
        rotation = None
        object_phase = None
        object_amplitude = None
        probe = None
        probe_fourier = None

        try:
            rotation = float(np.rad2deg(getattr(ptycho, "_rotation_best_rad", 0.0)))
        except Exception:
            pass

        try:
            object_phase = np.angle(np.asarray(ptycho.object_cropped))
            images["Phase"] = object_phase
        except Exception:
            logger.debug("Could not extract ptychography object phase", exc_info=True)

        try:
            object_amplitude = np.abs(np.asarray(ptycho.object_cropped))
            images["Amplitude"] = object_amplitude
        except Exception:
            logger.debug("Could not extract ptychography object amplitude", exc_info=True)

        try:
            probe = np.abs(np.asarray(ptycho.probe))
            images["Probe Intensity"] = probe
        except Exception:
            logger.debug("Could not extract ptychography probe", exc_info=True)

        try:
            probe_fourier = np.abs(np.asarray(ptycho.probe_fourier))
            images["Fourier Probe"] = probe_fourier
        except Exception:
            logger.debug("Could not extract ptychography Fourier probe", exc_info=True)

        return PhaseContrastResult(
            method=self.PTYCHOGRAPHY,
            images=images,
            elapsed_seconds=elapsed,
            rotation_degrees=rotation,
            object_phase=object_phase,
            object_amplitude=object_amplitude,
            probe=probe,
            probe_fourier=probe_fourier,
        )

    def _extract_parallax(self, parallax: Any, elapsed: float) -> PhaseContrastResult:
        images: dict[str, np.ndarray] = {}
        rotation = None
        object_phase = None

        try:
            rotation = float(np.rad2deg(getattr(parallax, "_rotation_best_rad", 0.0)))
        except Exception:
            pass

        try:
            aligned_bf = np.asarray(parallax.object_cropped)
            images["Aligned BF"] = aligned_bf
        except Exception:
            try:
                aligned_bf = np.asarray(parallax.object)
                images["Aligned BF"] = aligned_bf
            except Exception:
                logger.debug("Could not extract parallax aligned BF", exc_info=True)

        try:
            object_phase = np.asarray(parallax.object_cropped)
        except Exception:
            try:
                object_phase = np.asarray(parallax.object)
            except Exception:
                logger.debug("Could not extract parallax object", exc_info=True)

        try:
            shifts = np.asarray(parallax._shifts)
            images["Shift X"] = shifts[:, :, 0] if shifts.ndim == 3 and shifts.shape[2] >= 1 else shifts
            images["Shift Y"] = shifts[:, :, 1] if shifts.ndim == 3 and shifts.shape[2] >= 2 else shifts
        except Exception:
            logger.debug("Could not extract parallax shifts", exc_info=True)

        try:
            aberrations_C1 = float(parallax.aberrations_C1)
            images["Aberration C1"] = np.full_like(object_phase if object_phase is not None else np.zeros((1, 1)), aberrations_C1)
        except Exception:
            pass

        return PhaseContrastResult(
            method=self.PARALLAX,
            images=images,
            elapsed_seconds=elapsed,
            rotation_degrees=rotation,
            object_phase=object_phase,
        )

    def _extract_dpc(self, dpc: Any, elapsed: float) -> PhaseContrastResult:
        images: dict[str, np.ndarray] = {}
        rotation = None
        object_phase = None
        com_x = None
        com_y = None
        com_measured_x = None
        com_measured_y = None
        com_fitted_x = None
        com_fitted_y = None
        com_normalized_x = None
        com_normalized_y = None
        complex_com = None
        reconstructed_potential = None
        rotation_angles = None
        rotation_metric = None
        rotation_metric_transpose = None
        transpose = None
        error_iterations = None
        object_iterations: tuple[np.ndarray, ...] = ()

        try:
            rotation = float(np.rad2deg(getattr(dpc, "_rotation_best_rad", 0.0)))
        except Exception:
            pass

        try:
            object_phase = np.asarray(dpc.object_cropped)
            images["Phase"] = object_phase
        except Exception:
            try:
                object_phase = np.asarray(dpc.object)
                images["Phase"] = object_phase
            except Exception:
                logger.debug("Could not extract DPC object", exc_info=True)

        try:
            com_x = np.asarray(dpc._com_x)
            com_y = np.asarray(dpc._com_y)
            complex_com = com_x + 1j * com_y
            images["CoM X"] = com_x
            images["CoM Y"] = com_y
            images["Complex CoM"] = np.abs(complex_com)
        except Exception:
            logger.debug("Could not extract DPC center of mass", exc_info=True)

        for attr, key in [
            ("_com_measured_x", "Measured CoM X"),
            ("_com_measured_y", "Measured CoM Y"),
            ("_com_fitted_x", "Fitted CoM X"),
            ("_com_fitted_y", "Fitted CoM Y"),
            ("_com_normalized_x", "Normalized CoM X"),
            ("_com_normalized_y", "Normalized CoM Y"),
        ]:
            try:
                value = np.asarray(getattr(dpc, attr))
                images[key] = value
                if attr == "_com_measured_x":
                    com_measured_x = value
                elif attr == "_com_measured_y":
                    com_measured_y = value
                elif attr == "_com_fitted_x":
                    com_fitted_x = value
                elif attr == "_com_fitted_y":
                    com_fitted_y = value
                elif attr == "_com_normalized_x":
                    com_normalized_x = value
                elif attr == "_com_normalized_y":
                    com_normalized_y = value
            except Exception:
                logger.debug("Could not extract DPC diagnostic field %s", attr, exc_info=True)

        try:
            potential = getattr(dpc, "object_phase", None)
            if potential is None:
                potential = getattr(dpc, "object_cropped")
            reconstructed_potential = np.asarray(potential)
            images["Potential"] = reconstructed_potential
        except Exception:
            logger.debug("Could not extract DPC reconstructed potential", exc_info=True)

        try:
            rotation_angles = np.asarray(dpc._rotation_angles_deg)
            metric_name = "_rotation_div" if hasattr(dpc, "_rotation_div") else "_rotation_curl"
            transpose_name = metric_name + "_transpose"
            rotation_metric = np.asarray(getattr(dpc, metric_name))
            rotation_metric_transpose = np.asarray(getattr(dpc, transpose_name))
        except Exception:
            logger.debug("Could not extract DPC rotation-search diagnostics", exc_info=True)
        try:
            transpose = bool(dpc._rotation_best_transpose)
        except Exception:
            pass
        try:
            error_iterations = np.asarray(dpc.error_iterations, dtype=float)
            if error_iterations.size:
                images["Convergence error"] = error_iterations[None, :]
        except Exception:
            logger.debug("Could not extract DPC convergence history", exc_info=True)
        try:
            object_iterations = tuple(np.asarray(item) for item in dpc.object_phase_iterations)
            if object_iterations:
                indices = np.linspace(
                    0, len(object_iterations) - 1, min(4, len(object_iterations)), dtype=int
                )
                for index in indices:
                    images[f"Iteration {index}"] = object_iterations[index]
        except Exception:
            logger.debug("Could not extract DPC stored iterations", exc_info=True)

        return PhaseContrastResult(
            method=self.DPC,
            images=images,
            elapsed_seconds=elapsed,
            rotation_degrees=rotation,
            object_phase=object_phase,
            com_x=com_x,
            com_y=com_y,
            complex_com=complex_com,
            reconstructed_potential=reconstructed_potential,
            com_measured_x=com_measured_x,
            com_measured_y=com_measured_y,
            com_fitted_x=com_fitted_x,
            com_fitted_y=com_fitted_y,
            com_normalized_x=com_normalized_x,
            com_normalized_y=com_normalized_y,
            rotation_angles_degrees=rotation_angles,
            rotation_metric=rotation_metric,
            rotation_metric_transpose=rotation_metric_transpose,
            transpose=transpose,
            error_iterations=error_iterations,
            object_iterations=object_iterations,
        )

    def _py4dstem(self):
        try:
            return import_module("py4DSTEM")
        except ImportError as exc:
            raise PhaseContrastServiceError(
                "py4DSTEM could not be imported in this environment."
            ) from exc
