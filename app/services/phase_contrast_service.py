from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import import_module
from time import perf_counter
from typing import Any

import numpy as np

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
    alignment_bin_values: tuple[int, ...] = (32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 8, 8)
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
class PhaseContrastResult:
    method: str
    images: dict[str, np.ndarray] = field(default_factory=dict)
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
    com_normalized_x: np.ndarray | None = None
    com_normalized_y: np.ndarray | None = None


class PhaseContrastService:
    PTYCHOGRAPHY = "Ptychography"
    PARALLAX = "Parallax"
    DPC = "DPC"

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
        py4DSTEM = self._py4dstem()
        start = perf_counter()

        try:
            parallax = py4DSTEM.process.phase.Parallax(
                datacube=datacube,
                energy=params.energy,
                device=params.device,
                object_padding_px=params.object_padding_px,
            ).preprocess(
                edge_blend=params.edge_blend,
                plot_average_bf=False,
                normalize_images=params.normalize_images,
                threshold_intensity=params.threshold_intensity,
            ).reconstruct(
                alignment_bin_values=list(params.alignment_bin_values),
                regularize_shifts=params.regularize_shifts,
                progress_bar=False,
            )
        except Exception as exc:
            raise PhaseContrastServiceError(
                f"Parallax reconstruction failed: {exc}"
            ) from exc

        elapsed = perf_counter() - start
        result = self._extract_parallax(parallax, elapsed)
        return result

    def run_dpc(
        self,
        datacube: Any,
        params: DPCParams,
        progress_callback: Any | None = None,
    ) -> PhaseContrastResult:
        py4DSTEM = self._py4dstem()
        start = perf_counter()

        plot_com = params.plot_center_of_mass == "all"
        preprocess_kwargs: dict[str, Any] = {"plot_center_of_mass": plot_com}
        if params.force_com_rotation is not None:
            preprocess_kwargs["force_com_rotation"] = params.force_com_rotation
        if params.force_com_transpose:
            preprocess_kwargs["force_com_transpose"] = True

        try:
            dpc = py4DSTEM.process.phase.DPC(
                energy=params.energy,
                datacube=datacube,
            ).preprocess(
                **preprocess_kwargs,
            ).reconstruct(
                reset=True,
            ).visualize()
        except Exception as exc:
            raise PhaseContrastServiceError(
                f"DPC reconstruction failed: {exc}"
            ) from exc

        elapsed = perf_counter() - start
        result = self._extract_dpc(dpc, elapsed)
        return result

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
        com_normalized_x = None
        com_normalized_y = None
        complex_com = None
        reconstructed_potential = None

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
                elif attr == "_com_normalized_x":
                    com_normalized_x = value
                elif attr == "_com_normalized_y":
                    com_normalized_y = value
            except Exception:
                logger.debug("Could not extract DPC diagnostic field %s", attr, exc_info=True)

        try:
            reconstructed_potential = np.asarray(dpc.object_cropped)
            images["Potential"] = reconstructed_potential
        except Exception:
            logger.debug("Could not extract DPC reconstructed potential", exc_info=True)

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
            com_normalized_x=com_normalized_x,
            com_normalized_y=com_normalized_y,
        )

    def _py4dstem(self):
        try:
            return import_module("py4DSTEM")
        except ImportError as exc:
            raise PhaseContrastServiceError(
                "py4DSTEM could not be imported in this environment."
            ) from exc
