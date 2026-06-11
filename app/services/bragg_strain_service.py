from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


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
    cuda: bool = False


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
class BraggQualityResult:
    peak_count_map: np.ndarray
    mean_intensity_map: np.ndarray
    max_intensity_map: np.ndarray
    failure_mask: np.ndarray
    bragg_vector_map: np.ndarray


@dataclass(frozen=True)
class BraggVectorsResult:
    braggvectors: Any
    peak_count: int | None
    bragg_vector_map: np.ndarray
    quality: BraggQualityResult
    elapsed_seconds: float


@dataclass(frozen=True)
class CalibrationStatus:
    origin: str
    ellipse: str
    pixel: str
    rotate: str
    complete: bool


@dataclass(frozen=True)
class StrainQualityResult:
    principal_strain_1: np.ndarray
    principal_strain_2: np.ndarray
    valid_mask: np.ndarray | None
    fit_residual: np.ndarray | None


@dataclass(frozen=True)
class StrainMapResult:
    components: dict[str, np.ndarray]
    quality: StrainQualityResult
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
    manual_g1_x: float = 1.0
    manual_g1_y: float = 0.0
    manual_g2_x: float = 0.0
    manual_g2_y: float = 1.0


@dataclass(frozen=True)
class CrystalPixelParams:
    cif_path: str | None = None
    lattice_parameter: float = 4.08
    atomic_number: int = 79
    k_max: float = 1.5
    initial_pixel_size: float = 0.02


@dataclass(frozen=True)
class CalibrationActionResult:
    message: str
    images: dict[str, np.ndarray]
    elapsed_seconds: float
    measurements: dict[str, float] = field(default_factory=dict)
    overlays: dict[str, dict[str, float | str]] = field(default_factory=dict)


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
            py4DSTEM = self._py4dstem()
            radius, center_x, center_y = py4DSTEM.process.calibration.get_probe_size(probe.probe)
            probe.get_kernel(mode="sigmoid", radii=(radius, 2 * radius))
            kernel = np.asarray(probe.kernel)
        except (AttributeError, TypeError, ValueError) as exc:
            raise BraggStrainServiceError(f"Could not prepare vacuum-probe kernel: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected error preparing vacuum-probe kernel")
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
        except (AttributeError, TypeError, ValueError, RuntimeError):
            logger.debug("py4DSTEM peak detection failed, using fallback", exc_info=True)
            peaks = self._fallback_peak_detection(dp, params)
        except Exception:
            logger.warning("Unexpected error in py4DSTEM peak detection, using fallback", exc_info=True)
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
                CUDA=params.cuda,
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
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise BraggStrainServiceError(f"py4DSTEM BraggVectors calculation failed: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected error in BraggVectors calculation")
            raise BraggStrainServiceError(f"py4DSTEM BraggVectors calculation failed: {exc}") from exc

        self.braggvectors = braggvectors
        peak_count = self._count_braggvectors(braggvectors)
        bragg_vector_map = np.asarray(braggvectors.histogram(mode="raw").data)
        quality = self.bragg_quality(braggvectors, bragg_vector_map)
        return BraggVectorsResult(
            braggvectors=braggvectors,
            peak_count=peak_count,
            bragg_vector_map=bragg_vector_map,
            quality=quality,
            elapsed_seconds=perf_counter() - start,
        )

    def bragg_quality(
        self,
        braggvectors: Any | None,
        bragg_vector_map: np.ndarray | None = None,
    ) -> BraggQualityResult:
        source = self._require_braggvectors(braggvectors)
        raw = getattr(source, "raw", None)
        bvm = np.asarray(bragg_vector_map) if bragg_vector_map is not None else np.asarray(
            source.histogram(mode="raw").data
        )
        if raw is None or not hasattr(raw, "shape"):
            shape = self._quality_shape_from_source(source, bvm)
            peak_count = np.zeros(shape, dtype=float)
            return BraggQualityResult(
                peak_count_map=peak_count,
                mean_intensity_map=np.full(shape, np.nan, dtype=float),
                max_intensity_map=np.full(shape, np.nan, dtype=float),
                failure_mask=np.zeros(shape, dtype=bool),
                bragg_vector_map=bvm,
            )

        shape = tuple(int(dim) for dim in raw.shape[:2])
        peak_count = np.zeros(shape, dtype=float)
        mean_intensity = np.full(shape, np.nan, dtype=float)
        max_intensity = np.full(shape, np.nan, dtype=float)

        for rx in range(shape[0]):
            for ry in range(shape[1]):
                peaks = self._peaks_from_raw_cell(raw[rx, ry])
                peak_count[rx, ry] = len(peaks)
                if len(peaks):
                    intensities = peaks[:, 2]
                    mean_intensity[rx, ry] = float(np.nanmean(intensities))
                    max_intensity[rx, ry] = float(np.nanmax(intensities))

        return BraggQualityResult(
            peak_count_map=peak_count,
            mean_intensity_map=mean_intensity,
            max_intensity_map=max_intensity,
            failure_mask=peak_count == 0,
            bragg_vector_map=bvm,
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
            raw_bvm = np.asarray(source.histogram(mode="raw").data)
            measurements = self._origin_measurements(qx_fit, qy_fit, qx_residuals, qy_residuals, raw_bvm)
        except BraggStrainServiceError:
            source.setcal(**previous_state)
            raise
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            source.setcal(**previous_state)
            raise BraggStrainServiceError(f"Origin calibration failed: {exc}") from exc
        except Exception as exc:
            source.setcal(**previous_state)
            logger.exception("Unexpected error in origin calibration")
            raise BraggStrainServiceError(f"Origin calibration failed: {exc}") from exc
        return CalibrationActionResult(
            (
                "Origin measured and fitted: "
                f"x={measurements['x']:.4g}, y={measurements['y']:.4g}."
            ),
            {
                "raw Bragg vector map": raw_bvm,
                "qx measured": np.asarray(qx_meas),
                "qy measured": np.asarray(qy_meas),
                "valid mask": np.asarray(mask),
                "qx residual": np.asarray(qx_residuals),
                "qy residual": np.asarray(qy_residuals),
            },
            perf_counter() - start,
            measurements=measurements,
            overlays={"raw Bragg vector map": {"kind": "circle", **measurements}},
        )

    def calibrate_ellipse(
        self,
        braggvectors: Any | None,
        inner_radius: float,
        outer_radius: float,
        sampling: int,
        fit_source: Any | None = None,
        center: tuple[float, float] | None = None,
    ) -> CalibrationActionResult:
        target = self._require_braggvectors(braggvectors)
        source = fit_source if fit_source is not None else target
        source = self._require_braggvectors(source)
        previous_source_state = dict(getattr(source, "calstate", {}))
        previous_target_state = dict(getattr(target, "calstate", {}))
        if inner_radius <= 0 or outer_radius <= inner_radius:
            raise BraggStrainServiceError("Ellipse fit radii must satisfy 0 < inner < outer.")
        start = perf_counter()
        try:
            py4DSTEM = self._py4dstem()
            raw_bvm = source.histogram(mode="raw", sampling=sampling)
            bvm = source.histogram(mode="cal", sampling=sampling)
            fit_center = center if center is not None else bvm.origin
            p_ellipse = py4DSTEM.process.calibration.fit_ellipse_1D(
                bvm,
                center=fit_center,
                fitradii=(inner_radius, outer_radius),
            )
            source.calibration.set_p_ellipse(p_ellipse)
            if target is not source:
                target.calibration.set_p_ellipse(p_ellipse)
            source.setcal(**previous_source_state)
            if target is not source:
                target.setcal(**previous_target_state)
            measurements = self._ellipse_measurements(p_ellipse, raw_bvm, bvm)
            if center is not None:
                measurements = {**measurements, "x": float(center[0]), "y": float(center[1])}
        except BraggStrainServiceError:
            source.setcal(**previous_source_state)
            if target is not source:
                target.setcal(**previous_target_state)
            raise
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            source.setcal(**previous_source_state)
            if target is not source:
                target.setcal(**previous_target_state)
            raise BraggStrainServiceError(f"Ellipticity calibration failed: {exc}") from exc
        except Exception as exc:
            source.setcal(**previous_source_state)
            if target is not source:
                target.setcal(**previous_target_state)
            logger.exception("Unexpected error in ellipticity calibration")
            raise BraggStrainServiceError(f"Ellipticity calibration failed: {exc}") from exc
        label = "Ellipse Reference" if target is not source else "target"
        return CalibrationActionResult(
            (
                f"Ellipticity fitted from {label}: "
                f"a={measurements['a']:.4g}, b={measurements['b']:.4g}, "
                f"ellipticity={measurements['ellipticity']:.4g}."
            ),
            {
                "raw Bragg vector map": np.asarray(raw_bvm.data),
                "calibrated Bragg vector map": np.asarray(bvm.data),
            },
            perf_counter() - start,
            measurements=measurements,
            overlays={
                "raw Bragg vector map": {
                    "kind": "ring",
                    "x": measurements["x"],
                    "y": measurements["y"],
                    "inner_radius": float(min(measurements["a"], measurements["b"])),
                    "outer_radius": float(max(measurements["a"], measurements["b"])),
                    "a": measurements["a"],
                    "b": measurements["b"],
                    "theta": measurements.get("theta", 0.0),
                }
            },
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

    def fit_pixel_size_from_crystal(
        self,
        reference_braggvectors: Any | None,
        params: CrystalPixelParams,
    ) -> CalibrationActionResult:
        source = self._require_braggvectors(reference_braggvectors)
        if params.initial_pixel_size <= 0 or params.lattice_parameter <= 0 or params.k_max <= 0:
            raise BraggStrainServiceError("Crystal and initial pixel-size values must be positive.")
        start = perf_counter()
        try:
            py4DSTEM = self._py4dstem()
            if params.cif_path:
                crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(Path(params.cif_path))
            else:
                positions = np.array(
                    [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
                )
                crystal = py4DSTEM.process.diffraction.Crystal(
                    positions, params.atomic_number, params.lattice_parameter
                )
            crystal.calculate_structure_factors(params.k_max)
            source.calibration.set_Q_pixel_size(params.initial_pixel_size)
            source.calibration.set_Q_pixel_units("A^-1")
            source.setcal()
            crystal.calibrate_pixel_size(
                bragg_peaks=source,
                bragg_k_power=2.0,
                plot_result=False,
            )
            fitted = float(source.calibration.get_Q_pixel_size())
            images = {
                "reference raw Bragg vector map": np.asarray(source.histogram(mode="raw").data),
                "reference calibrated Bragg vector map": np.asarray(source.histogram(mode="cal").data),
            }
        except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as exc:
            raise BraggStrainServiceError(f"Crystal pixel-size fitting failed: {exc}") from exc
        return CalibrationActionResult(
            f"Crystal pixel-size fit complete: {fitted:g} A^-1. Transfer it to the target after review.",
            images,
            perf_counter() - start,
            measurements={"pixel_size": fitted},
        )

    def set_qr_rotation(
        self,
        braggvectors: Any | None,
        degrees: float,
        reference_image: np.ndarray | dict[str, np.ndarray] | None = None,
    ) -> CalibrationActionResult:
        source = self._require_braggvectors(braggvectors)
        previous_state = dict(source.calstate)
        start = perf_counter()
        try:
            raw_bvm = np.asarray(source.histogram(mode="raw").data)
            source.calibration.set_QR_rotation_degrees(degrees)
            source.setcal(
                center=bool(previous_state.get("center", False)),
                ellipse=bool(previous_state.get("ellipse", False)),
                pixel=bool(previous_state.get("pixel", False)),
                rotate=True,
            )
            rotated_bvm = np.asarray(source.histogram(mode="cal").data)
        finally:
            source.setcal(**previous_state)
        images = {
            "raw Bragg vector map": raw_bvm,
            "rotation-corrected Bragg vector map": rotated_bvm,
        }
        if isinstance(reference_image, dict):
            images.update({str(name): np.asarray(image) for name, image in reference_image.items()})
        elif reference_image is not None:
            images["rotation reference CBED bright-field image"] = np.asarray(reference_image)
        return CalibrationActionResult(
            f"QR rotation set to {degrees:g} degrees.",
            images,
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
        except (AttributeError, TypeError) as exc:
            raise BraggStrainServiceError(f"Could not apply calibration state: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected error applying calibration state")
            raise BraggStrainServiceError(f"Could not apply calibration state: {exc}") from exc
        enabled = [name for name, value in source.calstate.items() if value]
        return CalibrationActionResult(
            f"Applied corrections: {', '.join(enabled) if enabled else 'none'}.",
            {},
            perf_counter() - start,
        )

    def transfer_calibration_correction(
        self,
        target_braggvectors: Any | None,
        source_braggvectors: Any | None,
        correction: str,
    ) -> CalibrationActionResult:
        target = self._require_braggvectors(target_braggvectors)
        source = self._require_braggvectors(source_braggvectors)
        if target is source:
            raise BraggStrainServiceError("Choose a different BraggVectors object as the transfer source.")

        correction_key = correction.lower()
        if correction_key not in {"origin", "ellipse", "pixel", "rotate"}:
            raise BraggStrainServiceError(f"Unsupported calibration correction: {correction}.")

        start = perf_counter()
        try:
            self._copy_calibration_value(source, target, correction_key)
            previous_state = dict(getattr(target, "calstate", {}))
            next_state = {
                "center": bool(previous_state.get("center", False)),
                "ellipse": bool(previous_state.get("ellipse", False)),
                "pixel": bool(previous_state.get("pixel", False)),
                "rotate": bool(previous_state.get("rotate", False)),
            }
            next_state[self._calstate_name(correction_key)] = True
            target.setcal(**next_state)
        except BraggStrainServiceError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise BraggStrainServiceError(
                f"Could not transfer {self._correction_label(correction_key)} calibration: {exc}"
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected error transferring calibration correction")
            raise BraggStrainServiceError(
                f"Could not transfer {self._correction_label(correction_key)} calibration: {exc}"
            ) from exc

        return CalibrationActionResult(
            f"Transferred and applied {self._correction_label(correction_key)} correction.",
            {},
            perf_counter() - start,
        )

    def validate_calibration(self, braggvectors: Any | None) -> CalibrationActionResult:
        source = self._require_braggvectors(braggvectors)
        start = perf_counter()
        images: dict[str, np.ndarray] = {}
        try:
            images["raw Bragg vector map"] = np.asarray(source.histogram(mode="raw").data)
        except (AttributeError, TypeError, ValueError):
            logger.debug("Could not extract raw Bragg vector map for calibration validation", exc_info=True)
        try:
            images["calibrated Bragg vector map"] = np.asarray(source.histogram(mode="cal").data)
        except (AttributeError, TypeError, ValueError):
            logger.debug("Could not extract calibrated Bragg vector map for calibration validation", exc_info=True)
        try:
            status = self.calibration_status(source)
            applied = [name for name, value in getattr(source, "calstate", {}).items() if value]
        except (AttributeError, TypeError):
            status = CalibrationStatus("missing", "missing", "missing", "missing", False)
            applied = []
        required_applied = {"center", "ellipse", "pixel", "rotate"}
        if not status.complete or not required_applied.issubset(set(applied)):
            missing = sorted(required_applied - set(applied))
            raise BraggStrainServiceError(
                "Strain calibration is not ready. Measure and apply origin, ellipse, "
                f"pixel size, and rotation. Missing applied corrections: {', '.join(missing) or 'metadata'}."
            )
        return CalibrationActionResult(
            "Calibration validation complete: "
            f"origin={status.origin}, ellipse={status.ellipse}, pixel={status.pixel}, "
            f"rotate={status.rotate}, applied={', '.join(applied) if applied else 'none'}.",
            images,
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

        start = perf_counter()
        try:
            py4DSTEM = self._py4dstem()
            strainmap = py4DSTEM.StrainMap(braggvectors=braggvectors)
            _basis_calc, basis_figax = strainmap.choose_basis_vectors(
                minAbsoluteIntensity=params.min_absolute_intensity,
                minRelativeIntensity=params.min_relative_intensity,
                minSpacing=params.min_spacing,
                edgeBoundary=params.edge_boundary,
                maxNumPeaks=params.max_num_peaks,
                returncalc=True,
                returnfig=True,
            )
            self._close_matplotlib_result(basis_figax)
            strainmap.set_max_peak_spacing(max_peak_spacing=params.max_peak_spacing)
            strainmap.fit_basis_vectors(returncalc=True)
            gvects = self._strain_reference(strainmap, params, braggvectors)
            strainmap.get_strain(
                gvects=gvects,
                coordinate_rotation=params.coordinate_rotation,
                layout="square",
                returncalc=False,
            )
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            raise BraggStrainServiceError(f"py4DSTEM strain map calculation failed: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected error in strain map calculation")
            raise BraggStrainServiceError(f"py4DSTEM strain map calculation failed: {exc}") from exc
        finally:
            plt.close("all")

        components = {
            "exx": np.asarray(strainmap.data[0]),
            "eyy": np.asarray(strainmap.data[1]),
            "exy": np.asarray(strainmap.data[2]),
            "theta": np.asarray(strainmap.data[3]),
        }
        quality = self.strain_quality(strainmap, components)
        components = {
            **components,
            "principal strain 1": quality.principal_strain_1,
            "principal strain 2": quality.principal_strain_2,
        }
        if quality.fit_residual is not None:
            components["fit residual"] = quality.fit_residual
        elif quality.valid_mask is not None:
            components["valid mask"] = quality.valid_mask.astype(float)
        result = StrainMapResult(
            components=components,
            quality=quality,
            elapsed_seconds=perf_counter() - start,
        )
        self.strainmap = strainmap
        self.strain_result = result
        return result

    def strain_quality(
        self,
        strainmap: Any | None,
        components: dict[str, np.ndarray],
    ) -> StrainQualityResult:
        exx = np.asarray(components["exx"], dtype=float)
        eyy = np.asarray(components["eyy"], dtype=float)
        exy = np.asarray(components["exy"], dtype=float)
        mean = 0.5 * (exx + eyy)
        radius = np.sqrt(((exx - eyy) * 0.5) ** 2 + exy**2)
        valid_mask = self._strain_valid_mask(strainmap)
        return StrainQualityResult(
            principal_strain_1=mean + radius,
            principal_strain_2=mean - radius,
            valid_mask=valid_mask,
            fit_residual=self._strain_residual(strainmap),
        )

    def _strain_reference(self, strainmap: Any, params: StrainMapParams, braggvectors: Any) -> Any:
        if params.reference_mode == "auto_valid":
            valid = np.asarray(strainmap.g1g2_map.get_slice("mask").data, dtype=bool)
            if not valid.any():
                raise BraggStrainServiceError("No valid fitted g1/g2 points are available.")
            return valid
        if params.reference_mode == "manual_g1g2":
            return (
                np.asarray([params.manual_g1_x, params.manual_g1_y], dtype=float),
                np.asarray([params.manual_g2_x, params.manual_g2_y], dtype=float),
            )

        scan_shape = self._scan_shape(braggvectors.raw, braggvectors)
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
            reference = strainmap.get_reference_g1g2(roi)
            if hasattr(reference, "data"):
                return reference
            return np.asarray(reference)
        raise BraggStrainServiceError(f"Unsupported strain reference mode: {params.reference_mode}")

    def export_strain_result(self, result: StrainMapResult, file_path: str | Path) -> None:
        path = Path(file_path)
        if path.suffix.lower() == ".npz":
            np.savez(path, **result.components)
        elif path.suffix.lower() == ".npy":
            stack = np.stack([result.components[k] for k in ["exx", "eyy", "exy", "theta"]])
            np.save(path, stack)
        elif path.suffix.lower() in {".tif", ".tiff"}:
            try:
                import tifffile
            except ModuleNotFoundError as exc:
                raise BraggStrainServiceError(
                    "TIFF export requires tifffile. Install project requirements first."
                ) from exc
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

    def _close_matplotlib_result(self, result: Any) -> None:
        if result is None:
            plt.close("all")
            return
        fig = result[0] if isinstance(result, tuple) and result else result
        try:
            plt.close(fig)
        except Exception:
            logger.debug("Could not close matplotlib figure, closing all", exc_info=True)
            plt.close("all")

    def _ensure_datacube(self, datacube: Any) -> None:
        if datacube is None or not hasattr(datacube, "data"):
            raise BraggStrainServiceError("A py4DSTEM DataCube is required.")
        shape = getattr(datacube, "shape", getattr(datacube.data, "shape", None))
        if shape is None or len(tuple(shape)) != 4:
            raise BraggStrainServiceError(f"Expected a 4D DataCube, got shape {shape}.")

    def _scan_shape(self, raw: Any, braggvectors: Any | None = None) -> tuple[int, int]:
        if braggvectors is not None:
            for attr in ("Rshape", "shape", "scan_shape"):
                val = getattr(braggvectors, attr, None)
                if val is not None:
                    try:
                        return tuple(int(dim) for dim in tuple(val)[:2])
                    except Exception:
                        pass
        for obj in (raw, getattr(raw, "_data", None)):
            if obj is None:
                continue
            for attr in ("shape", "scan_shape"):
                val = getattr(obj, attr, None)
                if val is not None:
                    try:
                        return tuple(int(dim) for dim in tuple(val)[:2])
                    except Exception:
                        pass
        if braggvectors is not None:
            try:
                histogram = braggvectors.histogram(mode="raw")
                data = getattr(histogram, "data", histogram)
                shape = getattr(data, "shape", None)
                if shape is not None:
                    return tuple(int(dim) for dim in tuple(shape)[:2])
            except Exception:
                pass
            try:
                rows = len(raw)
                cols = len(raw[0]) if rows else 0
                if rows > 0 and cols > 0:
                    return rows, cols
            except Exception:
                pass
        raise BraggStrainServiceError("Could not determine scan shape from BraggVectors raw data.")

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

    def _peaks_from_raw_cell(self, cell: Any) -> np.ndarray:
        data = getattr(cell, "data", cell)
        return self._peaks_from_qpoints(data)

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
        except (AttributeError, TypeError, ValueError):
            return "missing"
        if value is None:
            return "missing"
        if isinstance(value, tuple) and any(hasattr(item, "shape") for item in value):
            shapes = [tuple(np.asarray(item).shape) for item in value]
            return f"set, shapes={shapes}"
        if hasattr(value, "shape"):
            return f"set, shape={tuple(np.asarray(value).shape)}"
        return str(value)

    def _copy_calibration_value(self, source: Any, target: Any, correction: str) -> None:
        source_calibration = getattr(source, "calibration", None)
        target_calibration = getattr(target, "calibration", None)
        if source_calibration is None or target_calibration is None:
            raise BraggStrainServiceError("Both source and target BraggVectors need calibration objects.")

        if correction == "origin":
            self._set_from_getter(
                source_calibration,
                target_calibration,
                getter_names=("get_origin",),
                setter_names=("set_origin",),
                label="origin",
            )
        elif correction == "ellipse":
            value = self._calibration_get(source_calibration, ("get_ellipse",), "ellipse")
            self._set_ellipse_calibration(target_calibration, value)
        elif correction == "pixel":
            size = self._calibration_get(source_calibration, ("get_Q_pixel_size",), "Q pixel size")
            units = self._calibration_get(
                source_calibration,
                ("get_Q_pixel_units",),
                "Q pixel units",
                required=False,
            )
            setter = getattr(target_calibration, "set_Q_pixel_size", None)
            if setter is None:
                raise BraggStrainServiceError("Target calibration does not support Q pixel size.")
            setter(size)
            units_setter = getattr(target_calibration, "set_Q_pixel_units", None)
            if units is not None and units_setter is not None:
                units_setter(units)
        elif correction == "rotate":
            value = self._calibration_get(
                source_calibration,
                ("get_QR_rotation_degrees", "get_QR_rotation", "get_QR_rotflip"),
                "QR rotation",
            )
            setter = self._first_attr(
                target_calibration,
                ("set_QR_rotation_degrees", "set_QR_rotation", "set_QR_rotflip"),
            )
            if setter is None:
                raise BraggStrainServiceError("Target calibration does not support QR rotation.")
            setter(value)

    def _set_from_getter(
        self,
        source_calibration: Any,
        target_calibration: Any,
        getter_names: tuple[str, ...],
        setter_names: tuple[str, ...],
        label: str,
    ) -> None:
        value = self._calibration_get(source_calibration, getter_names, label)
        setter = self._first_attr(target_calibration, setter_names)
        if setter is None:
            raise BraggStrainServiceError(f"Target calibration does not support {label}.")
        setter(value)

    def _set_ellipse_calibration(self, target_calibration: Any, value: Any) -> None:
        values = self._numeric_sequence(value)
        if len(values) == 3 and hasattr(target_calibration, "set_ellipse"):
            target_calibration.set_ellipse(value)
            return
        if hasattr(target_calibration, "set_p_ellipse"):
            target_calibration.set_p_ellipse(value)
            return
        if hasattr(target_calibration, "set_ellipse"):
            target_calibration.set_ellipse(value)
            return
        raise BraggStrainServiceError("Target calibration does not support ellipse.")

    def _calibration_get(
        self,
        calibration: Any,
        getter_names: tuple[str, ...],
        label: str,
        required: bool = True,
    ) -> Any:
        getter = self._first_attr(calibration, getter_names)
        if getter is None:
            if required:
                raise BraggStrainServiceError(f"Source calibration does not expose {label}.")
            return None
        value = getter()
        if value is None and required:
            raise BraggStrainServiceError(f"Source calibration has no {label} value.")
        return value

    def _first_attr(self, obj: Any, names: tuple[str, ...]):
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
        return None

    def _calstate_name(self, correction: str) -> str:
        return {"origin": "center", "ellipse": "ellipse", "pixel": "pixel", "rotate": "rotate"}[
            correction
        ]

    def _correction_label(self, correction: str) -> str:
        return {
            "origin": "origin",
            "ellipse": "ellipse",
            "pixel": "Q pixel size",
            "rotate": "QR rotation",
        }[correction]

    def _count_braggvectors(self, braggvectors: Any) -> int | None:
        raw = getattr(braggvectors, "raw", None)
        if raw is None:
            return None
        total = 0
        try:
            scan_shape = self._scan_shape(raw, braggvectors)
            for rx in range(scan_shape[0]):
                for ry in range(scan_shape[1]):
                    total += len(raw[rx, ry].data)
            return total
        except (AttributeError, IndexError, TypeError):
            return None

    def _quality_shape_from_source(self, source: Any, bvm: np.ndarray) -> tuple[int, int]:
        for name in ["shape", "scan_shape"]:
            value = getattr(source, name, None)
            if value is None:
                continue
            try:
                shape = tuple(int(dim) for dim in value[:2])
            except (TypeError, ValueError, IndexError):
                continue
            if len(shape) == 2 and shape[0] > 0 and shape[1] > 0:
                return shape
        if bvm.ndim >= 2:
            return tuple(int(dim) for dim in bvm.shape[:2])
        return (1, 1)

    def _origin_measurements(
        self,
        qx_fit: Any,
        qy_fit: Any,
        qx_residuals: Any,
        qy_residuals: Any,
        raw_bvm: np.ndarray,
    ) -> dict[str, float]:
        x = self._finite_mean(qx_fit)
        y = self._finite_mean(qy_fit)
        if not np.isfinite(x):
            x = (raw_bvm.shape[0] - 1) / 2
        if not np.isfinite(y):
            y = (raw_bvm.shape[1] - 1) / 2
        residual_radius = np.sqrt(
            np.nanmean(np.asarray(qx_residuals, dtype=float) ** 2)
            + np.nanmean(np.asarray(qy_residuals, dtype=float) ** 2)
        )
        if not np.isfinite(residual_radius) or residual_radius <= 0:
            residual_radius = max(min(raw_bvm.shape[:2]) * 0.03, 3.0)
        return {"x": float(x), "y": float(y), "r": float(residual_radius)}

    def _ellipse_measurements(self, p_ellipse: Any, raw_bvm: Any, calibrated_bvm: Any) -> dict[str, float]:
        values = self._numeric_sequence(p_ellipse)
        center = getattr(raw_bvm, "origin", getattr(calibrated_bvm, "origin", None))
        if center is not None:
            try:
                x, y = float(center[0]), float(center[1])
            except (TypeError, ValueError, IndexError):
                x, y = self._image_center(np.asarray(raw_bvm.data))
        else:
            x, y = self._image_center(np.asarray(raw_bvm.data))

        if len(values) >= 5:
            x, y, a, b, theta = values[:5]
        elif len(values) >= 3:
            a, b, theta = values[:3]
        else:
            a = b = max(min(np.asarray(raw_bvm.data).shape[:2]) * 0.25, 1.0)
            theta = 0.0

        a = abs(float(a))
        b = abs(float(b))
        small_axis = max(min(a, b), 1e-12)
        return {
            "x": float(x),
            "y": float(y),
            "a": a,
            "b": b,
            "theta": float(theta),
            "ellipticity": float(max(a, b) / small_axis),
        }

    def _finite_mean(self, value: Any) -> float:
        array = np.asarray(value, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return float("nan")
        return float(np.nanmean(finite))

    def _numeric_sequence(self, value: Any) -> list[float]:
        if isinstance(value, np.ndarray):
            candidates = value.ravel().tolist()
        elif isinstance(value, (list, tuple)):
            candidates = list(value)
        else:
            candidates = [value]
        values: list[float] = []
        for item in candidates:
            try:
                scalar = float(np.asarray(item).squeeze())
            except (TypeError, ValueError):
                continue
            if np.isfinite(scalar):
                values.append(scalar)
        return values

    def _image_center(self, image: np.ndarray) -> tuple[float, float]:
        if image.ndim < 2:
            return 0.0, 0.0
        return (image.shape[0] - 1) / 2, (image.shape[1] - 1) / 2

    def _strain_valid_mask(self, strainmap: Any | None) -> np.ndarray | None:
        if strainmap is None:
            return None
        try:
            return np.asarray(strainmap.g1g2_map.get_slice("mask").data, dtype=bool)
        except (AttributeError, TypeError, ValueError):
            return None

    def _py4dstem(self):
        try:
            return import_module("py4DSTEM")
        except Exception as exc:
            raise BraggStrainServiceError(
                "py4DSTEM could not be imported in this environment."
            ) from exc

    def _strain_residual(self, strainmap: Any | None) -> np.ndarray | None:
        if strainmap is None:
            return None
        for name in ["fit_residual", "residual", "residuals"]:
            value = getattr(strainmap, name, None)
            if value is None:
                continue
            try:
                array = np.asarray(getattr(value, "data", value), dtype=float)
            except (TypeError, ValueError):
                continue
            if array.ndim == 2:
                return array
        return None
