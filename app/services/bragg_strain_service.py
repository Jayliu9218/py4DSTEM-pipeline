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
    sigma_cc: float = 0
    template_sigma: float = 2
    subpixel: str = "multicorr"


@dataclass(frozen=True)
class PeakDetectionResult:
    diffraction_pattern: np.ndarray
    peaks: np.ndarray
    elapsed_seconds: float


@dataclass(frozen=True)
class SelectedPeaksResult:
    positions: list[tuple[int, int]]
    peak_counts: list[int]
    patterns: list[np.ndarray]
    peaks: list[np.ndarray]
    elapsed_seconds: float


@dataclass(frozen=True)
class BraggVectorsResult:
    braggvectors: Any
    peak_count: int | None
    bragg_vector_map: np.ndarray
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
    reference_mode: str = "auto_valid"
    roi_rx_start: int = 0
    roi_rx_end: int = 1
    roi_ry_start: int = 0
    roi_ry_end: int = 1


@dataclass(frozen=True)
class CalibrationActionResult:
    message: str
    images: dict[str, np.ndarray]
    elapsed_seconds: float


@dataclass(frozen=True)
class ProbeKernelResult:
    kernel: np.ndarray
    probe_radius: float
    center_x: float
    center_y: float
    elapsed_seconds: float


class BraggStrainService:
    def __init__(self) -> None:
        self.braggvectors: Any | None = None
        self.strainmap: Any | None = None
        self.strain_result: StrainMapResult | None = None
        self.probe_kernel: np.ndarray | None = None

    def prepare_probe_kernel(
        self,
        datacube: Any,
        rx_start: int,
        rx_end: int,
        ry_start: int,
        ry_end: int,
    ) -> ProbeKernelResult:
        self._ensure_datacube(datacube)
        shape = tuple(int(dim) for dim in getattr(datacube, "shape", datacube.data.shape))
        if not (0 <= rx_start < rx_end <= shape[0] and 0 <= ry_start < ry_end <= shape[1]):
            raise BraggStrainServiceError(
                f"Vacuum ROI must fit inside scan shape {shape[:2]} and have non-zero area."
            )

        start = perf_counter()
        roi = np.zeros(shape[:2], dtype=bool)
        roi[rx_start:rx_end, ry_start:ry_end] = True
        try:
            probe = datacube.get_vacuum_probe(ROI=roi, plot=False, returncalc=True)
            radius, center_x, center_y = py4DSTEM.process.calibration.get_probe_size(probe.probe)
            probe.get_kernel(mode="sigmoid", radii=(radius, 2 * radius))
            kernel = np.asarray(probe.kernel)
        except Exception as exc:
            raise BraggStrainServiceError(f"Could not prepare vacuum-probe kernel: {exc}") from exc

        self.probe_kernel = kernel
        return ProbeKernelResult(
            kernel=kernel,
            probe_radius=float(radius),
            center_x=float(center_x),
            center_y=float(center_y),
            elapsed_seconds=perf_counter() - start,
        )

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
            template = self._detection_template(dp.shape, params.template_sigma)
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
        template = self._detection_template(dp_mean.shape, params.template_sigma)

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
        bragg_vector_map = np.asarray(braggvectors.histogram(mode="raw").data)
        return BraggVectorsResult(
            braggvectors=braggvectors,
            peak_count=peak_count,
            bragg_vector_map=bragg_vector_map,
            elapsed_seconds=perf_counter() - start,
        )

    def detect_selected_positions(
        self,
        datacube: Any,
        positions: list[tuple[int, int]],
        params: BraggDetectionParams,
    ) -> SelectedPeaksResult:
        start = perf_counter()
        results = [self.detect_peaks(datacube, rx, ry, params) for rx, ry in positions]
        return SelectedPeaksResult(
            positions,
            [len(result.peaks) for result in results],
            [result.diffraction_pattern for result in results],
            [result.peaks for result in results],
            perf_counter() - start,
        )

    def calibrate_origin(self, braggvectors: Any | None) -> CalibrationActionResult:
        source = self._require_braggvectors(braggvectors)
        previous_state = dict(source.calstate)
        start = perf_counter()
        try:
            qx_meas, qy_meas, mask = source.measure_origin()
            qx_fit, qy_fit, qx_residuals, qy_residuals = source.fit_origin(plot=False, returncalc=True)
            source.setcal(**previous_state)
        except Exception as exc:
            source.setcal(**previous_state)
            raise BraggStrainServiceError(f"Origin calibration failed: {exc}") from exc
        return CalibrationActionResult(
            "Origin measured and fitted.",
            {
                "qx measured": np.asarray(qx_meas),
                "qy measured": np.asarray(qy_meas),
                "valid mask": np.asarray(mask),
                "qx residual": np.asarray(qx_residuals),
                "qy residual": np.asarray(qy_residuals),
            },
            perf_counter() - start,
        )

    def calibrate_ellipse(
        self,
        braggvectors: Any | None,
        inner_radius: float,
        outer_radius: float,
        sampling: int,
    ) -> CalibrationActionResult:
        source = self._require_braggvectors(braggvectors)
        previous_state = dict(source.calstate)
        if inner_radius <= 0 or outer_radius <= inner_radius:
            raise BraggStrainServiceError("Ellipse fit radii must satisfy 0 < inner < outer.")
        start = perf_counter()
        try:
            bvm = source.histogram(mode="cal", sampling=sampling)
            p_ellipse = py4DSTEM.process.calibration.fit_ellipse_1D(
                bvm,
                center=bvm.origin,
                fitradii=(inner_radius, outer_radius),
            )
            source.calibration.set_p_ellipse(p_ellipse)
            source.setcal(**previous_state)
        except Exception as exc:
            source.setcal(**previous_state)
            raise BraggStrainServiceError(f"Ellipticity calibration failed: {exc}") from exc
        return CalibrationActionResult(
            f"Ellipticity fitted: {p_ellipse}",
            {"calibrated Bragg vector map": np.asarray(bvm.data)},
            perf_counter() - start,
        )

    def set_pixel_size(self, braggvectors: Any | None, pixel_size: float) -> CalibrationActionResult:
        source = self._require_braggvectors(braggvectors)
        previous_state = dict(source.calstate)
        if pixel_size <= 0:
            raise BraggStrainServiceError("Q pixel size must be greater than 0.")
        start = perf_counter()
        source.calibration.set_Q_pixel_size(pixel_size)
        source.calibration.set_Q_pixel_units("A^-1")
        source.setcal(**previous_state)
        return CalibrationActionResult(
            f"Q pixel size set to {pixel_size:g} A^-1.",
            {},
            perf_counter() - start,
        )

    def set_qr_rotation(self, braggvectors: Any | None, degrees: float) -> CalibrationActionResult:
        source = self._require_braggvectors(braggvectors)
        previous_state = dict(source.calstate)
        start = perf_counter()
        source.calibration.set_QR_rotation_degrees(degrees)
        source.setcal(**previous_state)
        return CalibrationActionResult(
            f"QR rotation set to {degrees:g} degrees.",
            {},
            perf_counter() - start,
        )

    def set_calibration_state(
        self,
        braggvectors: Any | None,
        center: bool,
        ellipse: bool,
        pixel: bool,
        rotate: bool,
    ) -> CalibrationActionResult:
        source = self._require_braggvectors(braggvectors)
        start = perf_counter()
        try:
            source.setcal(center=center, ellipse=ellipse, pixel=pixel, rotate=rotate)
        except Exception as exc:
            raise BraggStrainServiceError(f"Could not apply calibration state: {exc}") from exc
        enabled = [name for name, value in source.calstate.items() if value]
        return CalibrationActionResult(
            f"Applied corrections: {', '.join(enabled) if enabled else 'none'}.",
            {},
            perf_counter() - start,
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

        calstate = getattr(braggvectors, "calstate", {})
        if not all(calstate.get(name, False) for name in ["center", "ellipse", "pixel", "rotate"]):
            raise BraggStrainServiceError(
                "Apply origin, ellipse, pixel, and rotation corrections manually in step 6 "
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
            gvects = self._strain_reference(strainmap, params, braggvectors)
            strainmap.get_strain(
                gvects=gvects,
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

    def _strain_reference(self, strainmap: Any, params: StrainMapParams, braggvectors: Any) -> Any:
        if params.reference_mode == "auto_valid":
            valid = np.asarray(strainmap.g1g2_map.get_slice("mask").data, dtype=bool)
            if not valid.any():
                raise BraggStrainServiceError("No valid fitted g1/g2 points are available.")
            return valid

        scan_shape = tuple(int(dim) for dim in braggvectors.raw.shape)
        if not (
            0 <= params.roi_rx_start < params.roi_rx_end <= scan_shape[0]
            and 0 <= params.roi_ry_start < params.roi_ry_end <= scan_shape[1]
        ):
            raise BraggStrainServiceError(f"Reference ROI must fit inside scan shape {scan_shape}.")
        roi = np.zeros(scan_shape, dtype=bool)
        roi[params.roi_rx_start : params.roi_rx_end, params.roi_ry_start : params.roi_ry_end] = True
        if params.reference_mode == "roi_mask":
            return roi
        if params.reference_mode == "roi_vectors":
            return strainmap.get_reference_g1g2(roi)
        raise BraggStrainServiceError(f"Unsupported strain reference mode: {params.reference_mode}")

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

    def _require_braggvectors(self, braggvectors: Any | None) -> Any:
        if braggvectors is None:
            raise BraggStrainServiceError("No BraggVectors object is available.")
        return braggvectors

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

    def _detection_template(self, shape: tuple[int, int], sigma: float) -> np.ndarray:
        if self.probe_kernel is not None and self.probe_kernel.shape == shape:
            return self.probe_kernel
        return self._make_gaussian_template(shape, sigma)

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
        if isinstance(value, tuple) and any(hasattr(item, "shape") for item in value):
            shapes = [tuple(np.asarray(item).shape) for item in value]
            return f"set, shapes={shapes}"
        if hasattr(value, "shape"):
            return f"set, shape={tuple(np.asarray(value).shape)}"
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
