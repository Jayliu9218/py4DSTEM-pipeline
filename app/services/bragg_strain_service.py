from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from app.services.cif_utils import load_py4dstem_crystal_from_cif

logger = logging.getLogger(__name__)


def _ensure_matplotlib():
    """Lazily import matplotlib with the Agg backend on first use.

    matplotlib is a heavy import (~0.5-1.5s cold); deferring it keeps it off
    the application startup path, since no rendering happens until a user
    action triggers figure generation.
    """
    cached = globals().get("_plt")
    if cached is not None:
        return cached
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    globals()["_plt"] = plt
    return plt


class BraggStrainServiceError(Exception):
    """User-facing Bragg/strain service error."""


@dataclass(frozen=True)
class BraggDetectionParams:
    min_absolute_intensity: float = 2
    min_relative_intensity: float = 0
    min_peak_spacing: int = 18
    edge_boundary: int = 2
    max_num_peaks: int = 100
    sigma_cc: float = 0
    sigma_dp: float = 0
    corr_power: float = 1
    upsample_factor: int = 16
    radial_background_subtraction: bool = False
    template_sigma: float = 2
    subpixel: str = "poly"
    allow_gaussian_fallback: bool = False
    cuda: bool = False


@dataclass(frozen=True)
class CBSPreset:
    name: str
    bragg: BraggDetectionParams
    adf_inner_radius_factor: float = 3
    adf_outer_radius_factor: float = 6
    off_axis_qx_factor: float = 1
    off_axis_qy_factor: float = 1 / 3
    off_axis_radius_factor: float = 5 / 4

    @classmethod
    def au_notebook(cls) -> "CBSPreset":
        return cls("01_CBS Au", BraggDetectionParams())


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
    zero_detection_fraction: float = 0
    edge_clipped_peak_count: int = 0
    peak_count_distribution: np.ndarray = field(default_factory=lambda: np.empty(0))
    intensity_distribution: np.ndarray = field(default_factory=lambda: np.empty(0))


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
    valid_mask: np.ndarray | None
    fit_residual: np.ndarray | None


@dataclass(frozen=True)
class StrainMapResult:
    components: dict[str, np.ndarray]
    process_images: dict[str, np.ndarray]
    process_vectors: dict[str, np.ndarray]
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
    reference_mode: str = "global_none"
    roi_rx_start: int = 0
    roi_rx_end: int = 1
    roi_ry_start: int = 0
    roi_ry_end: int = 1


@dataclass(frozen=True)
class CrystalPixelParams:
    cif_path: str | None = None
    lattice_parameter: float = 4.08
    atomic_number: int = 79
    k_max: float = 1.5
    initial_pixel_size: float = 0.02


@dataclass(frozen=True)
class OriginCalibrationParams:
    center_guess_x: float | None = None
    center_guess_y: float | None = None
    center_guess_only: bool = True
    score_method: str | None = None
    find_center: str = "max"
    fit_function: str = "plane"
    robust: bool = False
    robust_steps: int = 3
    robust_threshold: float = 2


@dataclass(frozen=True)
class BasisSelectionParams:
    index_origin: int | None = None
    index_g1: int | None = None
    index_g2: int | None = None
    subpixel: str = "multicorr"
    upsample_factor: int = 16
    sigma: float = 0
    min_absolute_intensity: float = 1200
    min_relative_intensity: float = 0
    min_spacing: float = 2
    edge_boundary: int = 1
    max_num_peaks: int = 150


@dataclass(frozen=True)
class QRComparisonParams:
    real_rotation: float = 158
    real_position_x: float | None = 59
    real_position_y: float | None = 16.5
    reciprocal_position_x: float | None = 154
    reciprocal_position_y: float | None = 205
    real_length_fraction: float = 0.4
    reciprocal_length_fraction: float = 0.3


@dataclass(frozen=True)
class StrainStageResult:
    stage: str
    images: dict[str, np.ndarray]
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    circles: dict[str, np.ndarray] = field(default_factory=dict)
    quality: dict[str, float | str | bool] = field(default_factory=dict)
    message: str = ""
    reference_mode: str | None = None
    reference_roi: np.ndarray | None = None
    reference_g1: np.ndarray | None = None
    reference_g2: np.ndarray | None = None


@dataclass(frozen=True)
class CBSAcceptanceState:
    basis_selection: bool = False
    basis_fit: bool = False
    reference: bool = False


@dataclass(frozen=True)
class CalibrationActionResult:
    message: str
    images: dict[str, np.ndarray]
    elapsed_seconds: float
    measurements: dict[str, float] = field(default_factory=dict)
    overlays: dict[str, dict[str, float | str]] = field(default_factory=dict)
    quality: dict[str, float | str | bool] = field(default_factory=dict)
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    image_kinds: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeKernelResult:
    kernel: np.ndarray
    centered_kernel: np.ndarray
    profile_plot: np.ndarray
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
        self.pending_ellipse: tuple[Any, Any, Any] | None = None
        self.accepted_strain_stages: set[str] = set()

    def prepare_probe_kernel(
        self,
        datacube: Any,
        rx_start: int,
        rx_end: int,
        ry_start: int,
        ry_end: int,
        progress: Any = None,
    ) -> ProbeKernelResult:
        def _report(message: str, fraction: float = -1.0) -> None:
            logger.info("prepare_probe_kernel: %s", message)
            if progress is not None:
                try:
                    progress(message, fraction)
                except Exception:  # noqa: BLE001 - progress is best-effort
                    pass

        self._ensure_datacube(datacube)
        shape = tuple(int(dim) for dim in getattr(datacube, "shape", datacube.data.shape))
        if not (0 <= rx_start < rx_end <= shape[0] and 0 <= ry_start < ry_end <= shape[1]):
            raise BraggStrainServiceError(
                f"Vacuum ROI must fit inside scan shape {shape[:2]} and have non-zero area."
            )
        # Reject degenerate (near-zero-area) ROIs that make py4DSTEM's
        # get_vacuum_probe loop for an extremely long time or fail to converge.
        roi_area = (rx_end - rx_start) * (ry_end - ry_start)
        if roi_area < 4:
            raise BraggStrainServiceError(
                f"Vacuum ROI is too small ({roi_area} px); draw a larger region "
                f"(at least a 2x2 area) over vacuum scan positions."
            )

        _report(f"Preparing vacuum probe over ROI ({rx_start}:{rx_end}, {ry_start}:{ry_end}) "
                f"of scan shape {shape[:2]}", 0.0)
        start = perf_counter()
        roi = np.zeros(shape[:2], dtype=bool)
        roi[rx_start:rx_end, ry_start:ry_end] = True
        try:
            _report("Averaging vacuum-probe diffraction patterns...", 0.1)
            probe = datacube.get_vacuum_probe(ROI=roi, plot=False, returncalc=True)
            _report("Measuring probe size...", 0.5)
            py4DSTEM = self._py4dstem()
            radius, center_x, center_y = py4DSTEM.process.calibration.get_probe_size(probe.probe)
            _report(f"Probe radius={radius:.2f}px; building kernel...", 0.7)
            probe.get_kernel(mode="sigmoid", radii=(radius, 2 * radius))
            kernel = np.asarray(probe.kernel)
            _report("Computing diagnostics...", 0.9)
            centered_kernel, profile_plot = self._kernel_diagnostics(kernel, R=24, L=24, W=1)
        except (AttributeError, TypeError, ValueError) as exc:
            raise BraggStrainServiceError(f"Could not prepare vacuum-probe kernel: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected error preparing vacuum-probe kernel")
            raise BraggStrainServiceError(f"Could not prepare vacuum-probe kernel: {exc}") from exc

        self.probe_kernel = kernel
        _report(f"Probe kernel ready (radius={radius:.2f}px).", 1.0)
        return ProbeKernelResult(
            kernel=kernel,
            centered_kernel=centered_kernel,
            profile_plot=profile_plot,
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
            template = self._detection_template(dp.shape, params)
            qpoints = datacube.find_Bragg_disks(
                template=template,
                data=(rx, ry),
                corrPower=params.corr_power,
                radial_bksb=params.radial_background_subtraction,
                sigma_dp=params.sigma_dp,
                sigma_cc=params.sigma_cc,
                subpixel=params.subpixel,
                upsample_factor=params.upsample_factor,
                minAbsoluteIntensity=params.min_absolute_intensity,
                minRelativeIntensity=params.min_relative_intensity,
                minPeakSpacing=params.min_peak_spacing,
                edgeBoundary=params.edge_boundary,
                maxNumPeaks=params.max_num_peaks,
                returncalc=True,
            )
            peaks = self._peaks_from_qpoints(qpoints)
        except BraggStrainServiceError:
            raise
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
        template = self._detection_template(dp_mean.shape, params)

        try:
            braggvectors = datacube.find_Bragg_disks(
                template=template,
                CUDA=params.cuda,
                corrPower=params.corr_power,
                radial_bksb=params.radial_background_subtraction,
                sigma_dp=params.sigma_dp,
                sigma_cc=params.sigma_cc,
                subpixel=params.subpixel,
                upsample_factor=params.upsample_factor,
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
        quality = self.bragg_quality(braggvectors, bragg_vector_map, params.edge_boundary)
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
        edge_boundary: int = 0,
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
        all_intensities: list[float] = []
        edge_clipped = 0
        qshape = tuple(int(value) for value in getattr(source, "Qshape", bvm.shape)[:2])

        for rx in range(shape[0]):
            for ry in range(shape[1]):
                peaks = self._peaks_from_raw_cell(raw[rx, ry])
                peak_count[rx, ry] = len(peaks)
                if len(peaks):
                    intensities = peaks[:, 2]
                    all_intensities.extend(np.asarray(intensities, dtype=float).tolist())
                    mean_intensity[rx, ry] = float(np.nanmean(intensities))
                    max_intensity[rx, ry] = float(np.nanmax(intensities))
                    if edge_boundary > 0:
                        edge_clipped += int(np.count_nonzero(
                            (peaks[:, 0] < edge_boundary)
                            | (peaks[:, 1] < edge_boundary)
                            | (peaks[:, 0] >= qshape[0] - edge_boundary)
                            | (peaks[:, 1] >= qshape[1] - edge_boundary)
                        ))

        failure_mask = peak_count == 0
        return BraggQualityResult(
            peak_count_map=peak_count,
            mean_intensity_map=mean_intensity,
            max_intensity_map=max_intensity,
            failure_mask=failure_mask,
            bragg_vector_map=bvm,
            zero_detection_fraction=float(np.mean(failure_mask)),
            edge_clipped_peak_count=edge_clipped,
            peak_count_distribution=peak_count.ravel(),
            intensity_distribution=np.asarray(all_intensities, dtype=float),
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

    def calibrate_origin(
        self,
        braggvectors: Any | None,
        params: OriginCalibrationParams | None = None,
    ) -> CalibrationActionResult:
        source = self._require_braggvectors(braggvectors)
        params = params or OriginCalibrationParams()
        previous_state = dict(source.calstate)
        start = perf_counter()
        try:
            center_guess = None
            if params.center_guess_x is not None and params.center_guess_y is not None:
                center_guess = (params.center_guess_x, params.center_guess_y)
            raw_bvm = np.asarray(source.histogram(mode="raw").data)
            if params.center_guess_only:
                if center_guess is None:
                    center_guess = self._image_center(raw_bvm)
                if not hasattr(source, "calibration") or not hasattr(source.calibration, "set_origin"):
                    raise BraggStrainServiceError("BraggVectors calibration does not support origin setting.")
                source.calibration.set_origin(center_guess)
                qx_fit = np.full(raw_bvm.shape[:2], float(center_guess[0]), dtype=float)
                qy_fit = np.full(raw_bvm.shape[:2], float(center_guess[1]), dtype=float)
                qx_meas = qx_fit.copy()
                qy_meas = qy_fit.copy()
                qx_residuals = np.zeros(raw_bvm.shape[:2], dtype=float)
                qy_residuals = np.zeros(raw_bvm.shape[:2], dtype=float)
                mask = np.ones(raw_bvm.shape[:2], dtype=bool)
            else:
                try:
                    qx_meas, qy_meas, mask = source.measure_origin(
                        center_guess=center_guess,
                        score_method=params.score_method,
                        findcenter=params.find_center,
                    )
                except TypeError:
                    qx_meas, qy_meas, mask = source.measure_origin()
                try:
                    qx_fit, qy_fit, qx_residuals, qy_residuals = source.fit_origin(
                        fitfunction=params.fit_function,
                        robust=params.robust,
                        robust_steps=params.robust_steps,
                        robust_thresh=params.robust_threshold,
                        plot=False,
                        returncalc=True,
                    )
                except TypeError:
                    qx_fit, qy_fit, qx_residuals, qy_residuals = source.fit_origin(
                        plot=False, returncalc=True
                    )
            source.setcal(**previous_state)
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
                ("Origin set from center guess: " if params.center_guess_only else "Origin measured and fitted: ")
                + f"x={measurements['x']:.4g}, y={measurements['y']:.4g}."
            ),
            {
                "qx measured": self._center_origin_fit_map(qx_meas, mask),
                "qx fitted": self._center_origin_fit_map(qx_fit, mask),
                "qx residual": np.asarray(qx_residuals),
                "qy measured": self._center_origin_fit_map(qy_meas, mask),
                "qy fitted": self._center_origin_fit_map(qy_fit, mask),
                "qy residual": np.asarray(qy_residuals),
            },
            perf_counter() - start,
            measurements=measurements,
            quality={
                "valid_coverage": self._valid_fraction(mask),
                "qx_residual_rms": self._rms(qx_residuals),
                "qy_residual_rms": self._rms(qy_residuals),
            },
        )

    def compare_origin_correction(self, braggvectors: Any | None) -> CalibrationActionResult:
        source = self._require_braggvectors(braggvectors)
        previous_state = dict(source.calstate)
        start = perf_counter()
        try:
            raw_bvm = np.asarray(source.histogram(mode="raw").data)
            source.setcal(
                center=True,
                ellipse=bool(previous_state.get("ellipse", False)),
                pixel=bool(previous_state.get("pixel", False)),
                rotate=bool(previous_state.get("rotate", False)),
            )
            centered_bvm = np.asarray(source.histogram(mode="cal").data)
        finally:
            source.setcal(**previous_state)
        return CalibrationActionResult(
            "Origin correction comparison generated.",
            {"raw Bragg vector map": raw_bvm, "origin-centered Bragg vector map": centered_bvm},
            perf_counter() - start,
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
        self.pending_ellipse = None
        start = perf_counter()
        try:
            py4DSTEM = self._py4dstem()
            bvm = source.histogram(mode="cal", sampling=sampling)
            fit_center = center if center is not None else bvm.origin
            p_ellipse = py4DSTEM.process.calibration.fit_ellipse_1D(
                bvm,
                center=fit_center,
                fitradii=(inner_radius, outer_radius),
            )
            measurements = self._ellipse_measurements(p_ellipse, bvm, bvm)
            fit_residual = self._ellipse_fit_residual(
                np.asarray(bvm.data), p_ellipse, (inner_radius, outer_radius)
            )
            if center is not None:
                measurements = {**measurements, "x": float(center[0]), "y": float(center[1])}
            self.pending_ellipse = (target, source, p_ellipse)
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
                "ellipse fit Bragg vector map": np.asarray(bvm.data),
            },
            perf_counter() - start,
            measurements=measurements,
            overlays={
                "ellipse fit Bragg vector map": {
                    "kind": "ring",
                    "x": measurements["x"],
                    "y": measurements["y"],
                    "inner_radius": float(inner_radius),
                    "outer_radius": float(outer_radius),
                    "a": measurements["a"],
                    "b": measurements["b"],
                    "theta": measurements.get("theta", 0.0),
                }
            },
            quality={"fit_residual_rms": fit_residual},
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
                crystal = load_py4dstem_crystal_from_cif(py4DSTEM, params.cif_path)
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
            before_figax = crystal.plot_scattering_intensity(
                bragg_peaks=source, bragg_k_power=2.0, returnfig=True
            )
            before_overlay = self._figure_to_rgb(before_figax[0])
            crystal.calibrate_pixel_size(
                bragg_peaks=source,
                bragg_k_power=2.0,
                plot_result=False,
            )
            fitted = float(source.calibration.get_Q_pixel_size())
            after_figax = crystal.plot_scattering_intensity(
                bragg_peaks=source, bragg_k_power=2.0, returnfig=True
            )
            after_overlay = self._figure_to_rgb(after_figax[0])
            images = {
                "crystal overlay before fit": before_overlay,
                "crystal overlay after fit": after_overlay,
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
            image_kinds={
                "crystal overlay before fit": "rgb",
                "crystal overlay after fit": "rgb",
            },
        )

    def set_qr_rotation(
        self,
        braggvectors: Any | None,
        degrees: float,
        reference_image: np.ndarray | dict[str, np.ndarray] | None = None,
        comparison: QRComparisonParams | None = None,
    ) -> CalibrationActionResult:
        source = self._require_braggvectors(braggvectors)
        comparison = comparison or QRComparisonParams()
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
        except Exception:
            source.setcal(**previous_state)
            raise
        images = {
            "raw Bragg vector map": raw_bvm,
            "rotation-corrected Bragg vector map": rotated_bvm,
        }
        if isinstance(reference_image, dict):
            images.update({str(name): np.asarray(image) for name, image in reference_image.items()})
        elif reference_image is not None:
            images["rotation reference CBED bright-field image"] = np.asarray(reference_image)
        vectors: dict[str, np.ndarray] = {}
        if isinstance(reference_image, dict):
            real_name = next((name for name in images if "target" in name.lower()), None)
            reciprocal_name = next((name for name in images if "rotation reference" in name.lower()), None)
            if real_name and reciprocal_name:
                vectors[real_name] = self._rotation_arrow(
                    images[real_name], comparison.real_rotation,
                    comparison.real_position_x, comparison.real_position_y,
                    comparison.real_length_fraction,
                )
                vectors[reciprocal_name] = self._rotation_arrow(
                    images[reciprocal_name], comparison.real_rotation + degrees,
                    comparison.reciprocal_position_x, comparison.reciprocal_position_y,
                    comparison.reciprocal_length_fraction,
                )
        return CalibrationActionResult(
            f"QR rotation set to {degrees:g} degrees and applied.",
            images,
            perf_counter() - start,
            vectors=vectors,
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

        origin = self._format_origin(calibration)
        ellipse = self._format_ellipse(calibration)
        pixel = self._format_pixel(calibration)
        rotate = self._format_rotation(calibration)
        complete = all(value != "missing" for value in [origin, ellipse, pixel, rotate])
        return CalibrationStatus(origin, ellipse, pixel, rotate, complete)

    def choose_strain_basis(
        self,
        braggvectors: Any | None,
        params: BasisSelectionParams,
    ) -> StrainStageResult:
        source = self._require_braggvectors(braggvectors)
        py4DSTEM = self._py4dstem()
        strainmap = py4DSTEM.StrainMap(braggvectors=source)
        calc, figax = strainmap.choose_basis_vectors(
            index_origin=params.index_origin,
            index_g1=params.index_g1,
            index_g2=params.index_g2,
            subpixel=params.subpixel,
            upsample_factor=params.upsample_factor,
            sigma=params.sigma,
            minAbsoluteIntensity=params.min_absolute_intensity,
            minRelativeIntensity=params.min_relative_intensity,
            minSpacing=params.min_spacing,
            edgeBoundary=params.edge_boundary,
            maxNumPeaks=params.max_num_peaks,
            returncalc=True,
            returnfig=True,
        )
        self._close_matplotlib_result(figax)
        self.strainmap = strainmap
        self.accepted_strain_stages.clear()
        g0, g1, g2 = [np.asarray(value, dtype=float).ravel() for value in calc[:3]]
        vectors = np.asarray([
            [g0[0], g0[1], g1[0], g1[1]],
            [g0[0], g0[1], g2[0], g2[1]],
        ])
        bvm = np.asarray(strainmap.bvm.data)
        return StrainStageResult(
            "basis_selection",
            {"basis selection": bvm},
            vectors={"basis selection": vectors},
            quality={
                "g1_length": float(np.linalg.norm(g1)),
                "g2_length": float(np.linalg.norm(g2)),
                "basis_angle_degrees": float(np.degrees(np.arccos(np.clip(np.dot(g1, g2) / max(np.linalg.norm(g1) * np.linalg.norm(g2), 1e-12), -1, 1)))),
                "qr_rotation_applied": bool(getattr(source, "calstate", {}).get("rotate", False)),
            },
            message="Basis vectors selected; review g1/g2 before continuing.",
        )

    def set_strain_peak_spacing(self, max_peak_spacing: float) -> StrainStageResult:
        if self.strainmap is None:
            raise BraggStrainServiceError("Choose basis vectors before setting peak spacing.")
        if max_peak_spacing <= 0:
            raise BraggStrainServiceError("Maximum peak spacing must be positive.")
        figax = self.strainmap.set_max_peak_spacing(max_peak_spacing, returnfig=True)
        self._close_matplotlib_result(figax)
        self.accepted_strain_stages.discard("basis_fit")
        self.accepted_strain_stages.discard("reference")
        bvm = np.asarray(self.strainmap.bvm.data)
        origin = np.asarray(self.strainmap.origin, dtype=float)
        directions = getattr(self.strainmap, "braggdirections", None)
        points = np.column_stack([directions["qx"] + origin[0], directions["qy"] + origin[1]]) if directions is not None else np.empty((0, 2))
        return StrainStageResult(
            "peak_spacing",
            {"peak acceptance regions": bvm},
            circles={
                "peak acceptance regions": np.column_stack(
                    [points, np.full(len(points), float(max_peak_spacing))]
                )
            },
            quality={"max_peak_spacing": float(max_peak_spacing), "accepted_direction_count": int(len(points))},
            message="Peak acceptance spacing set; review indexed regions before fitting.",
        )

    def fit_strain_basis(self) -> StrainStageResult:
        if self.strainmap is None or not hasattr(self.strainmap, "max_peak_spacing"):
            raise BraggStrainServiceError("Choose basis vectors and set maximum peak spacing first.")
        self.strainmap.fit_basis_vectors(returncalc=True)
        self.accepted_strain_stages.discard("basis_fit")
        self.accepted_strain_stages.discard("reference")
        valid = self._strain_valid_mask(self.strainmap)
        residual = self._strain_residual(self.strainmap)
        images: dict[str, np.ndarray] = {}
        if valid is not None:
            images["basis fit valid mask"] = valid.astype(float)
        if residual is not None:
            images["basis fit residual"] = residual
        valid_fraction = float(np.mean(valid)) if valid is not None else 0
        return StrainStageResult(
            "basis_fit",
            images,
            quality={"valid_fraction": valid_fraction, "accepted": valid_fraction > 0},
            message=(
                "Basis fitting complete."
                if valid_fraction > 0
                else "No valid fitted points; adjust basis selection, peak spacing, or Bragg detection."
            ),
        )

    def calculate_strain_from_stages(
        self,
        braggvectors: Any,
        params: StrainMapParams,
    ) -> StrainMapResult:
        if self.strainmap is None or not hasattr(self.strainmap, "g1g2_map"):
            raise BraggStrainServiceError("Fit basis vectors before calculating strain.")
        missing = {"basis_selection", "basis_fit", "reference"} - self.accepted_strain_stages
        if missing:
            raise BraggStrainServiceError(
                "Explicitly accept the staged strain decisions before calculation: "
                + ", ".join(sorted(missing))
                + "."
            )
        return self._finish_strain_map(self.strainmap, braggvectors, params, basis_calc=None)

    def preview_strain_reference(
        self,
        braggvectors: Any,
        params: StrainMapParams,
    ) -> StrainStageResult:
        if self.strainmap is None or not hasattr(self.strainmap, "g1g2_map"):
            raise BraggStrainServiceError("Fit basis vectors before reviewing the reference.")
        gvects = self._strain_reference(self.strainmap, params, braggvectors)
        images, vectors = self._strain_process_diagnostics(
            self.strainmap, None, gvects, braggvectors
        )
        selected = {
            name: image
            for name, image in images.items()
            if name in {"reference mask", "reference directions", "basis fit valid mask"}
        }
        reference_roi = None
        reference_g1 = None
        reference_g2 = None
        if params.reference_mode == "roi_g1g2":
            reference_roi = self._strain_reference_roi(params, braggvectors)
            selected["reference ROI"] = reference_roi.astype(float)
            reference_g1, reference_g2 = gvects
        valid = self._strain_valid_mask(self.strainmap)
        reference_fraction = float(np.mean(reference_roi)) if reference_roi is not None else float("nan")
        message = (
            "Global reference prepared; py4DSTEM will derive g1/g2 from valid local basis vectors."
            if params.reference_mode == "global_none"
            else "ROI-derived g1/g2 prepared; review the ROI and reference directions before accepting."
        )
        return StrainStageResult(
            "reference",
            selected,
            vectors={name: value for name, value in vectors.items() if name in selected},
            quality={
                "reference_mode": params.reference_mode,
                "reference_fraction": reference_fraction,
                "valid_fraction": float(np.mean(valid)) if valid is not None else 0,
            },
            message=message,
            reference_mode=params.reference_mode,
            reference_roi=reference_roi,
            reference_g1=reference_g1,
            reference_g2=reference_g2,
        )

    def accept_strain_stage(self, stage: str) -> CBSAcceptanceState:
        if stage not in {"basis_selection", "basis_fit", "reference"}:
            raise BraggStrainServiceError(f"Unsupported strain acceptance stage: {stage}.")
        if stage == "basis_selection" and self.strainmap is None:
            raise BraggStrainServiceError("Choose basis vectors before accepting the selection.")
        if stage == "basis_fit":
            if "basis_selection" not in self.accepted_strain_stages:
                raise BraggStrainServiceError("Accept basis selection before accepting the basis fit.")
            if self.strainmap is None or not hasattr(self.strainmap, "g1g2_map"):
                raise BraggStrainServiceError("Fit basis vectors before accepting the fit.")
        if stage == "reference" and "basis_fit" not in self.accepted_strain_stages:
            raise BraggStrainServiceError("Accept the basis fit before accepting the reference.")
        self.accepted_strain_stages.add(stage)
        return CBSAcceptanceState(
            basis_selection="basis_selection" in self.accepted_strain_stages,
            basis_fit="basis_fit" in self.accepted_strain_stages,
            reference="reference" in self.accepted_strain_stages,
        )

    def compute_strain_map(
        self,
        braggvectors: Any | None,
        params: StrainMapParams,
    ) -> StrainMapResult:
        if braggvectors is None:
            raise BraggStrainServiceError("No BraggVectors object is available. Run full BraggVectors first.")

        start = perf_counter()
        plt = _ensure_matplotlib()
        try:
            py4DSTEM = self._py4dstem()
            strainmap = py4DSTEM.StrainMap(braggvectors=braggvectors)
            basis_calc, basis_figax = strainmap.choose_basis_vectors(
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

        return self._finish_strain_map(strainmap, braggvectors, params, basis_calc, start)

    def _finish_strain_map(
        self,
        strainmap: Any,
        braggvectors: Any,
        params: StrainMapParams,
        basis_calc: Any = None,
        start: float | None = None,
    ) -> StrainMapResult:
        if start is None:
            start = perf_counter()
            gvects = self._strain_reference(strainmap, params, braggvectors)
            strainmap.get_strain(
                gvects=gvects,
                coordinate_rotation=params.coordinate_rotation,
                layout="square",
                returncalc=False,
            )
        else:
            gvects = self._strain_reference(strainmap, params, braggvectors)
        components = {
            "exx": np.asarray(strainmap.data[0]),
            "eyy": np.asarray(strainmap.data[1]),
            "exy": np.asarray(strainmap.data[2]),
            "theta": np.asarray(strainmap.data[3]),
        }
        quality = self.strain_quality(strainmap, components)
        process_images, process_vectors = self._strain_process_diagnostics(
            strainmap, basis_calc, gvects, braggvectors
        )
        if params.reference_mode == "roi_g1g2":
            process_images["reference ROI"] = self._strain_reference_roi(
                params, braggvectors
            ).astype(float)
        if quality.fit_residual is not None:
            process_images["basis fit residual"] = quality.fit_residual
        elif quality.valid_mask is not None:
            process_images["basis fit valid mask"] = quality.valid_mask.astype(float)

        result = StrainMapResult(
            components=components,
            process_images=process_images,
            process_vectors=process_vectors,
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
        valid_mask = self._strain_valid_mask(strainmap)
        return StrainQualityResult(
            valid_mask=valid_mask,
            fit_residual=self._strain_residual(strainmap),
        )

    def accept_pending_ellipse(self) -> CalibrationActionResult:
        if self.pending_ellipse is None:
            raise BraggStrainServiceError("Fit an ellipse before accepting it.")
        target, source, p_ellipse = self.pending_ellipse
        start = perf_counter()
        target_state = dict(getattr(target, "calstate", {}))
        raw_bvm = np.asarray(target.histogram(mode="raw").data)
        try:
            source.calibration.set_p_ellipse(p_ellipse)
            if target is not source:
                target.calibration.set_p_ellipse(p_ellipse)
            target.setcal(
                center=bool(target_state.get("center", False)),
                ellipse=True,
                pixel=bool(target_state.get("pixel", False)),
                rotate=bool(target_state.get("rotate", False)),
            )
            corrected_bvm = np.asarray(target.histogram(mode="cal").data)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            target.setcal(**target_state)
            raise BraggStrainServiceError(f"Could not accept ellipse calibration: {exc}") from exc
        self.pending_ellipse = None
        return CalibrationActionResult(
            "Ellipse calibration accepted and applied.",
            {
                "raw Bragg vector map": raw_bvm,
                "ellipse-corrected Bragg vector map": corrected_bvm,
            },
            perf_counter() - start,
        )

    def _strain_reference(self, strainmap: Any, params: StrainMapParams, braggvectors: Any) -> Any:
        if params.reference_mode == "global_none":
            valid = np.asarray(strainmap.g1g2_map.get_slice("mask").data, dtype=bool)
            if not valid.any():
                raise BraggStrainServiceError("No valid fitted g1/g2 points are available.")
            return None
        if params.reference_mode == "roi_g1g2":
            roi = self._strain_reference_roi(params, braggvectors)
            reference = strainmap.get_reference_g1g2(roi)
            if hasattr(reference, "data"):
                reference = reference.data
            try:
                g1_ref, g2_ref = reference
            except (TypeError, ValueError) as exc:
                raise BraggStrainServiceError(
                    "ROI reference did not return g1_ref and g2_ref."
                ) from exc
            return (
                np.asarray(g1_ref, dtype=float),
                np.asarray(g2_ref, dtype=float),
            )
        raise BraggStrainServiceError(f"Unsupported strain reference mode: {params.reference_mode}")

    def _strain_reference_roi(self, params: StrainMapParams, braggvectors: Any) -> np.ndarray:
        raw = getattr(braggvectors, "raw", braggvectors)
        scan_shape = self._scan_shape(raw, braggvectors)
        if not (
            0 <= params.roi_rx_start < params.roi_rx_end <= scan_shape[0]
            and 0 <= params.roi_ry_start < params.roi_ry_end <= scan_shape[1]
        ):
            raise BraggStrainServiceError(f"Reference ROI must fit inside scan shape {scan_shape}.")
        roi = np.zeros(scan_shape, dtype=bool)
        roi[params.roi_rx_start : params.roi_rx_end, params.roi_ry_start : params.roi_ry_end] = True
        return roi

    def _strain_process_diagnostics(
        self, strainmap: Any, basis_calc: Any, gvects: Any, braggvectors: Any
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        images: dict[str, np.ndarray] = {}
        vectors: dict[str, np.ndarray] = {}
        bvm = getattr(getattr(strainmap, "bvm", None), "data", None)
        if bvm is None:
            try:
                bvm = braggvectors.histogram(mode="cal").data
            except (AttributeError, TypeError, ValueError):
                bvm = braggvectors.histogram(mode="raw").data
        images["basis selection"] = np.asarray(bvm)
        values = list(basis_calc) if isinstance(basis_calc, (tuple, list)) else []
        if len(values) >= 3:
            try:
                g0 = np.asarray(values[0], dtype=float).ravel()
            except (TypeError, ValueError):
                g0 = np.asarray([])
            rows = []
            for vector in values[1:3]:
                try:
                    value = np.asarray(vector, dtype=float).ravel()
                except (TypeError, ValueError):
                    continue
                if g0.size >= 2 and value.size >= 2:
                    rows.append([g0[0], g0[1], value[0], value[1]])
            if rows:
                vectors["basis selection"] = np.asarray(rows)
        valid = self._strain_valid_mask(strainmap)
        if valid is not None:
            images["basis fit valid mask"] = valid.astype(float)
        reference = np.asarray(gvects)
        if valid is not None and reference.ndim == 2 and reference.shape == valid.shape:
            images["reference mask"] = reference.astype(float)
        elif (
            isinstance(gvects, (tuple, list))
            or (isinstance(gvects, np.ndarray) and gvects.shape == (2, 2))
        ) and len(gvects) >= 2:
            center = np.asarray(self._image_center(np.asarray(bvm)))
            rows = []
            for vector in gvects[:2]:
                value = np.asarray(vector, dtype=float).ravel()
                if value.size >= 2:
                    rows.append([center[0], center[1], value[0], value[1]])
            if rows:
                images["reference directions"] = np.asarray(bvm)
                vectors["reference directions"] = np.asarray(rows)
        return images, vectors

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
        plt = _ensure_matplotlib()
        fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
        for ax, name in zip(axes.ravel(), ["exx", "eyy", "exy", "theta"]):
            im = ax.imshow(result.components[name], cmap="PRGn" if name == "theta" else "RdBu_r")
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _close_matplotlib_result(self, result: Any) -> None:
        plt = _ensure_matplotlib()
        if result is None:
            plt.close("all")
            return
        fig = result[0] if isinstance(result, tuple) and result else result
        try:
            plt.close(fig)
        except Exception:
            logger.debug("Could not close matplotlib figure, closing all", exc_info=True)
            plt.close("all")

    def _figure_to_rgb(self, fig: Any) -> np.ndarray:
        plt = _ensure_matplotlib()
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return image

    def _kernel_diagnostics(
        self, kernel: np.ndarray, R: int, L: int, W: int
    ) -> tuple[np.ndarray, np.ndarray]:
        plt = _ensure_matplotlib()
        shifted = np.fft.fftshift(np.asarray(kernel))
        cx, cy = shifted.shape[0] // 2, shifted.shape[1] // 2
        centered = shifted[
            max(cx - R, 0) : min(cx + R, shifted.shape[0]),
            max(cy - R, 0) : min(cy + R, shifted.shape[1]),
        ]
        half_width = max(int(W), 1)
        horizontal = shifted[max(cx - half_width, 0) : cx + half_width + 1, :].mean(axis=0)
        vertical = shifted[:, max(cy - half_width, 0) : cy + half_width + 1].mean(axis=1)
        h0, h1 = max(cy - L, 0), min(cy + L, horizontal.size)
        v0, v1 = max(cx - L, 0), min(cx + L, vertical.size)
        fig, ax = plt.subplots(figsize=(5, 3), dpi=100)
        ax.plot(np.arange(h0, h1) - cy, horizontal[h0:h1], label="horizontal")
        ax.plot(np.arange(v0, v1) - cx, vertical[v0:v1], label="vertical")
        ax.set_xlabel("q offset (px)")
        ax.set_ylabel("kernel")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.canvas.draw()
        profile_plot = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return centered, profile_plot

    def _ellipse_fit_residual(
        self,
        image: np.ndarray,
        ellipse: Any,
        fit_radii: tuple[float, float],
    ) -> float:
        values = self._numeric_sequence(ellipse)
        if len(values) < 5:
            return float("nan")
        x0, y0, a, b, theta = values[:5]
        xx, yy = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij")
        dx, dy = xx - x0, yy - y0
        ct, st = np.cos(theta), np.sin(theta)
        elliptical_r = np.sqrt(((ct * dx + st * dy) / max(a, 1e-12)) ** 2 + ((-st * dx + ct * dy) / max(b, 1e-12)) ** 2)
        circular_r = np.hypot(dx, dy)
        mask = (circular_r >= fit_radii[0]) & (circular_r <= fit_radii[1])
        if not mask.any():
            return float("nan")
        weights = np.asarray(image, dtype=float)[mask]
        weights = np.maximum(weights, 0)
        if not np.any(weights):
            return float(np.sqrt(np.mean((elliptical_r[mask] - 1) ** 2)))
        return float(np.sqrt(np.average((elliptical_r[mask] - 1) ** 2, weights=weights)))

    def _rotation_arrow(
        self,
        image: np.ndarray,
        degrees: float,
        x: float | None,
        y: float | None,
        length_fraction: float,
    ) -> np.ndarray:
        shape = np.asarray(image).shape
        x0 = float(shape[0] / 2 if x is None else x)
        y0 = float(shape[1] / 2 if y is None else y)
        length = float(np.mean(shape[:2]) * length_fraction)
        radians = np.radians(degrees)
        return np.asarray([[x0, y0, np.cos(radians) * length, np.sin(radians) * length]])

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

    def _detection_template(self, shape: tuple[int, int], params: BraggDetectionParams) -> np.ndarray:
        if self.probe_kernel is not None and self.probe_kernel.shape == shape:
            return self.probe_kernel
        if params.allow_gaussian_fallback:
            logger.warning("Using explicit Gaussian fallback instead of a prepared probe kernel.")
            return self._make_gaussian_template(shape, params.template_sigma)
        if self.probe_kernel is None:
            raise BraggStrainServiceError(
                "Prepare probe.kernel before Bragg detection, or explicitly enable Gaussian fallback."
            )
        raise BraggStrainServiceError(
            f"Prepared probe kernel shape {self.probe_kernel.shape} does not match diffraction shape "
            f"{shape}; prepare a matching kernel or explicitly enable Gaussian fallback."
        )

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

    def _format_origin(self, calibration: Any) -> str:
        value = self._calibration_value(calibration, ("get_origin",))
        if value is None:
            return "missing"
        values = self._numeric_sequence(value)
        if len(values) >= 2:
            return f"x={values[0]:.6g}, y={values[1]:.6g}"
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            x = self._finite_mean(value[0])
            y = self._finite_mean(value[1])
            if np.isfinite(x) and np.isfinite(y):
                return f"x={x:.6g}, y={y:.6g}"
        return self._safe_calibration_value(calibration, "get_origin")

    def _format_ellipse(self, calibration: Any) -> str:
        value = self._calibration_value(calibration, ("get_ellipse",))
        if value is None:
            return "missing"
        values = self._numeric_sequence(value)
        if len(values) >= 5:
            _x, _y, a, b, theta = values[:5]
        elif len(values) >= 3:
            a, b, theta = values[:3]
        else:
            return self._safe_calibration_value(calibration, "get_ellipse")
        ellipticity = max(abs(a), abs(b)) / max(min(abs(a), abs(b)), 1e-12)
        return f"a={a:.6g}, b={b:.6g}, theta={theta:.6g} rad, ellipticity={ellipticity:.6g}"

    def _format_pixel(self, calibration: Any) -> str:
        value = self._calibration_value(calibration, ("get_Q_pixel_size",))
        if value is None:
            return "missing"
        units = self._calibration_value(calibration, ("get_Q_pixel_units",))
        return f"{float(value):.6g} {units or 'units unknown'}"

    def _format_rotation(self, calibration: Any) -> str:
        value = self._calibration_value(
            calibration, ("get_QR_rotation_degrees", "get_QR_rotation")
        )
        if value is None:
            return "missing"
        return f"{float(value):.6g} deg"

    def _calibration_value(self, calibration: Any, getter_names: tuple[str, ...]) -> Any | None:
        getter = self._first_attr(calibration, getter_names)
        if getter is None:
            return None
        try:
            return getter()
        except (AttributeError, TypeError, ValueError):
            return None

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
        residual_radius = np.sqrt(self._rms(qx_residuals) ** 2 + self._rms(qy_residuals) ** 2)
        if not np.isfinite(residual_radius) or residual_radius <= 0:
            residual_radius = max(min(raw_bvm.shape[:2]) * 0.03, 3.0)
        return {"x": float(x), "y": float(y), "r": float(residual_radius)}

    def _center_origin_fit_map(self, value: Any, mask: Any) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        valid = np.asarray(mask, dtype=bool)
        selected = array[valid] if valid.shape == array.shape and valid.any() else array[np.isfinite(array)]
        center = float(np.nanmedian(selected)) if selected.size else 0.0
        return array - center

    def _rms(self, value: Any) -> float:
        array = np.asarray(value, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(finite ** 2)))

    def _valid_fraction(self, mask: Any) -> float:
        array = np.asarray(mask, dtype=bool)
        if array.size == 0:
            return 0.0
        return float(np.mean(array))

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
