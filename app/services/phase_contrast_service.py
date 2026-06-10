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


@dataclass(frozen=True)
class DPCParams:
    energy: float = 200e3
    plot_center_of_mass: str = "off"


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


class PhaseContrastService:
    PTYCHOGRAPHY = "Ptychography"
    PARALLAX = "Parallax"
    DPC = "DPC"

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

        try:
            dpc = py4DSTEM.process.phase.DPC(
                energy=params.energy,
                datacube=datacube,
            ).preprocess(
                plot_center_of_mass=plot_com,
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
            images["CoM X"] = com_x
            images["CoM Y"] = com_y
        except Exception:
            logger.debug("Could not extract DPC center of mass", exc_info=True)

        return PhaseContrastResult(
            method=self.DPC,
            images=images,
            elapsed_seconds=elapsed,
            rotation_degrees=rotation,
            object_phase=object_phase,
        )

    def _py4dstem(self):
        try:
            return import_module("py4DSTEM")
        except ImportError as exc:
            raise PhaseContrastServiceError(
                "py4DSTEM could not be imported in this environment."
            ) from exc