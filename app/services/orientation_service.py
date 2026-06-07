from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import py4DSTEM


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


@dataclass(frozen=True)
class OrientationResult:
    orientation_map: Any
    preview: np.ndarray
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
                CUDA=False,
                progress_bar=False,
            )
        except Exception as exc:
            raise OrientationServiceError(f"Orientation plan failed: {exc}") from exc
        return perf_counter() - start

    def match(self, braggvectors: Any | None) -> OrientationResult:
        crystal = self._require_crystal()
        if braggvectors is None:
            raise OrientationServiceError("Run full BraggVectors before orientation matching.")
        start = perf_counter()
        try:
            orientation_map = crystal.match_orientations(
                braggvectors,
                num_matches_return=2,
                min_angle_between_matches_deg=5,
                min_number_peaks=3,
                inversion_symmetry=True,
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
        return OrientationResult(orientation_map, preview, perf_counter() - start)

    def _require_crystal(self) -> Any:
        if self.crystal is None:
            raise OrientationServiceError("Load a CIF crystal structure first.")
        return self.crystal
