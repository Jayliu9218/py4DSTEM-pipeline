from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import py4DSTEM
import tifffile


class BraggStrainServiceError(Exception):
    """User-facing Bragg/strain service error."""


@dataclass(frozen=True)
class BraggDetectionParams:
    min_absolute_intensity: float = 0
    min_relative_intensity: float = 0.005
    min_peak_spacing: int = 8
    edge_boundary: int = 5
    max_num_peaks: int = 70
    sigma_cc: float = 2
    template_sigma: float = 2
    subpixel: str = "multicorr"


@dataclass(frozen=True)
class PeakDetectionResult:
    diffraction_pattern: np.ndarray
    peaks: np.ndarray
    elapsed_seconds: float


@dataclass(frozen=True)
class BraggVectorsResult:
    braggvectors: Any
    peak_count: int | None
    elapsed_seconds: float


@dataclass(frozen=True)
class CalibrationStatus:
    origin: str
    ellipse: str
    pixel: str
    rotate: str
    complete: bool


@dataclass(frozen=True)
class StrainMapResult:
    components: dict[str, np.ndarray]
    elapsed_seconds: float


@dataclass(frozen=True)
class StrainMapParams:
    coordinate_rotation: float = 0
    max_peak_spacing: float = 3
    min_absolute_intensity: float = 0
    min_relative_intensity: float = 0
    min_spacing: float = 0
    edge_boundary: int = 1
    max_num_peaks: int = 10


class BraggStrainService:
    def __init__(self) -> None:
        self.braggvectors: Any | None = None
        self.strainmap: Any | None = None
        self.strain_result: StrainMapResult | None = None

    def detect_peaks(
        self,
        datacube: Any,
        rx: int,
        ry: int,
        params: BraggDetectionParams,
    ) -> PeakDetectionResult:
        self._ensure_datacube(datacube)
        self._validate_scan_position(datacube, rx, ry)
        start = perf_counter()
        dp = np.asarray(datacube.data[rx, ry, :, :])

        try:
            template = self._make_gaussian_template(dp.shape, params.template_sigma)
            qpoints = datacube.find_Bragg_disks(
                template=template,
                data=(rx, ry),
                corrPower=1,
                sigma_cc=params.sigma_cc,
                subpixel=params.subpixel,
                upsample_factor=16,
                minAbsoluteIntensity=params.min_absolute_intensity,
                minRelativeIntensity=params.min_relative_intensity,
                minPeakSpacing=params.min_peak_spacing,
                edgeBoundary=params.edge_boundary,
                maxNumPeaks=params.max_num_peaks,
                returncalc=True,
            )
            peaks = self._peaks_from_qpoints(qpoints)
        except Exception:
            peaks = self._fallback_peak_detection(dp, params)

        return PeakDetectionResult(
            diffraction_pattern=dp,
            peaks=peaks,
            elapsed_seconds=perf_counter() - start,
        )

    def compute_braggvectors(
        self,
        datacube: Any,
        params: BraggDetectionParams,
    ) -> BraggVectorsResult:
        self._ensure_datacube(datacube)
        start = perf_counter()
        dp_mean = np.asarray(datacube.get_dp_mean().data)
        template = self._make_gaussian_template(dp_mean.shape, params.template_sigma)

        try:
            braggvectors = datacube.find_Bragg_disks(
                template=template,
                corrPower=1,
                sigma_cc=params.sigma_cc,
                subpixel=params.subpixel,
                upsample_factor=16,
                minAbsoluteIntensity=params.min_absolute_intensity,
                minRelativeIntensity=params.min_relative_intensity,
                minPeakSpacing=params.min_peak_spacing,
                edgeBoundary=params.edge_boundary,
                maxNumPeaks=params.max_num_peaks,
                name="braggvectors",
                returncalc=True,
            )
        except Exception as exc:
            raise BraggStrainServiceError(f"py4DSTEM BraggVectors calculation failed: {exc}") from exc

        self.braggvectors = braggvectors
        peak_count = self._count_braggvectors(braggvectors)
        return BraggVectorsResult(
            braggvectors=braggvectors,
            peak_count=peak_count,
            elapsed_seconds=perf_counter() - start,
        )

    def calibration_status(self, source: Any | None) -> CalibrationStatus:
        calibration = getattr(source, "calibration", None)
        if calibration is None:
            return CalibrationStatus("missing", "missing", "missing", "missing", False)

        origin = self._safe_calibration_value(calibration, "get_origin")
        ellipse = self._safe_calibration_value(calibration, "get_ellipse")
        pixel = self._safe_calibration_value(calibration, "get_Q_pixel_size")
        rotate = self._safe_calibration_value(calibration, "get_QR_rotation")
        complete = all(value != "missing" for value in [origin, ellipse, pixel, rotate])
        return CalibrationStatus(origin, ellipse, pixel, rotate, complete)

    def compute_strain_map(
        self,
        braggvectors: Any | None,
        params: StrainMapParams,
    ) -> StrainMapResult:
        if braggvectors is None:
            raise BraggStrainServiceError("No BraggVectors object is available. Run full BraggVectors first.")

        status = self.calibration_status(braggvectors)
        if not status.complete:
            raise BraggStrainServiceError(
                "Calibration is incomplete. Required origin, ellipse, pixel, and rotate calibration "
                "before strain mapping."
            )

        start = perf_counter()
        try:
            strainmap = py4DSTEM.StrainMap(braggvectors=braggvectors)
            strainmap.choose_basis_vectors(
                minAbsoluteIntensity=params.min_absolute_intensity,
                minRelativeIntensity=params.min_relative_intensity,
                minSpacing=params.min_spacing,
                edgeBoundary=params.edge_boundary,
                maxNumPeaks=params.max_num_peaks,
                returncalc=True,
            )
            strainmap.set_max_peak_spacing(max_peak_spacing=params.max_peak_spacing)
            strainmap.fit_basis_vectors(returncalc=True)
            strainmap.get_strain(
                coordinate_rotation=params.coordinate_rotation,
                layout="square",
                returncalc=True,
            )
        except Exception as exc:
            raise BraggStrainServiceError(f"py4DSTEM strain map calculation failed: {exc}") from exc

        components = {
            "exx": np.asarray(strainmap.data[0]),
            "eyy": np.asarray(strainmap.data[1]),
            "exy": np.asarray(strainmap.data[2]),
            "theta": np.asarray(strainmap.data[3]),
        }
        result = StrainMapResult(components=components, elapsed_seconds=perf_counter() - start)
        self.strainmap = strainmap
        self.strain_result = result
        return result

    def export_strain_result(self, result: StrainMapResult, file_path: str | Path) -> None:
        path = Path(file_path)
        if path.suffix.lower() == ".npz":
            np.savez(path, **result.components)
        elif path.suffix.lower() == ".npy":
            stack = np.stack([result.components[k] for k in ["exx", "eyy", "exy", "theta"]])
            np.save(path, stack)
        elif path.suffix.lower() in {".tif", ".tiff"}:
            stack = np.stack([result.components[k] for k in ["exx", "eyy", "exy", "theta"]])
            tifffile.imwrite(path, stack)
        elif path.suffix.lower() == ".png":
            self._save_strain_png(result, path)
        else:
            raise BraggStrainServiceError("Supported strain exports are PNG, TIFF, NPY, and NPZ.")

    def _save_strain_png(self, result: StrainMapResult, path: Path) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
        for ax, name in zip(axes.ravel(), ["exx", "eyy", "exy", "theta"]):
            im = ax.imshow(result.components[name], cmap="RdBu_r")
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _ensure_datacube(self, datacube: Any) -> None:
        if datacube is None or not hasattr(datacube, "data"):
            raise BraggStrainServiceError("A py4DSTEM DataCube is required.")
        shape = getattr(datacube, "shape", getattr(datacube.data, "shape", None))
        if shape is None or len(tuple(shape)) != 4:
            raise BraggStrainServiceError(f"Expected a 4D DataCube, got shape {shape}.")

    def _validate_scan_position(self, datacube: Any, rx: int, ry: int) -> None:
        shape = tuple(int(dim) for dim in getattr(datacube, "shape", datacube.data.shape))
        if rx < 0 or rx >= shape[0] or ry < 0 or ry >= shape[1]:
            raise BraggStrainServiceError(f"Scan position rx={rx}, ry={ry} is outside {shape[:2]}.")

    def _make_gaussian_template(self, shape: tuple[int, int], sigma: float) -> np.ndarray:
        cx = (shape[0] - 1) / 2
        cy = (shape[1] - 1) / 2
        x, y = np.ogrid[: shape[0], : shape[1]]
        sigma = max(float(sigma), 0.5)
        template = np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)))
        return template / template.max()

    def _peaks_from_qpoints(self, qpoints: Any) -> np.ndarray:
        data = np.asarray(getattr(qpoints, "data", qpoints))
        if data.dtype.names and {"qx", "qy"}.issubset(data.dtype.names):
            intensity_name = "intensity" if "intensity" in data.dtype.names else None
            intensity = data[intensity_name] if intensity_name else np.zeros(len(data))
            return np.column_stack([data["qx"], data["qy"], intensity])
        array = np.asarray(data)
        if array.ndim == 1:
            return np.empty((0, 3))
        if array.shape[1] >= 3:
            return array[:, :3]
        if array.shape[1] == 2:
            return np.column_stack([array, np.zeros(array.shape[0])])
        return np.empty((0, 3))

    def _fallback_peak_detection(self, dp: np.ndarray, params: BraggDetectionParams) -> np.ndarray:
        threshold = max(
            float(params.min_absolute_intensity),
            float(dp.max()) * float(params.min_relative_intensity),
        )
        candidates: list[tuple[float, float, float]] = []
        edge = max(int(params.edge_boundary), 0)
        spacing = max(int(params.min_peak_spacing), 1)
        for x in range(edge, dp.shape[0] - edge):
            for y in range(edge, dp.shape[1] - edge):
                value = dp[x, y]
                if value < threshold:
                    continue
                x0 = max(x - spacing, 0)
                x1 = min(x + spacing + 1, dp.shape[0])
                y0 = max(y - spacing, 0)
                y1 = min(y + spacing + 1, dp.shape[1])
                if value >= dp[x0:x1, y0:y1].max():
                    candidates.append((float(x), float(y), float(value)))
        candidates.sort(key=lambda item: item[2], reverse=True)
        return np.asarray(candidates[: params.max_num_peaks], dtype=float)

    def _safe_calibration_value(self, calibration: Any, getter_name: str) -> str:
        getter = getattr(calibration, getter_name, None)
        if getter is None:
            return "missing"
        try:
            value = getter()
        except Exception:
            return "missing"
        if value is None:
            return "missing"
        return str(value)

    def _count_braggvectors(self, braggvectors: Any) -> int | None:
        raw = getattr(braggvectors, "raw", None)
        if raw is None:
            return None
        total = 0
        try:
            for rx in range(raw.shape[0]):
                for ry in range(raw.shape[1]):
                    total += len(raw[rx, ry].data)
            return total
        except Exception:
            return None
