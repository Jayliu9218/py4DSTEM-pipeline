r"""
General py4DSTEM workflow for 4D-STEM datasets with crystal phase/orientation analysis.

Scientific logic
----------------
This script is organized around the crystal-analysis chain:

    Crystal Analysis
    ├── CIF Manager
    ├── Structure Factors
    ├── Simulated Diffraction / Orientation Library
    ├── BVM + Voronoi structural pre-classification
    ├── Phase Matching
    ├── Orientation Matching
    ├── Grain Analysis
    └── Strain Analysis

The key rule is that orientation is solved *conditional on a crystal phase*.
For a multi-phase dataset, the same calibrated Bragg peak array is matched
against each candidate phase library. The phase map is then obtained by
comparing the best correlation / match score across crystals, while the
orientation map for a pixel is taken from the winning crystal's orientation map.

Designed for py4DSTEM-readable 4D-STEM datasets, including MIB, EMD/HDF5, and NumPy arrays. Large array operations are chunked; py4DSTEM-specific imports are delayed until runtime.

Typical use
-----------
python general_py4DSTEM_workflow.py --input "D:/data/sample.mib" --scan 512 512 --roi 192 320 192 320 --output-dir ./py4dstem_output --phase-cifs ./Ti-fcc.cif ./Ti-hcp.cif --phase-names Ti-fcc Ti-hcp --voltage 300000 --angle-step 5 --in-plane-step 5

Important cautions
------------------
1. The first full run should normally use coarse angle steps or an ROI.
2. Phase/orientation confidence is meaningful only after Bragg detection and
   diffraction calibration have been checked visually.
3. Strain interpretation requires good origin, ellipse, Q pixel size, and QR
   rotation calibration, plus enough indexed peaks per probe position.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

ProgressCallback = Callable[[str, float | None], None]


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------


@dataclass
class IOConfig:
    input_path: str
    # General input handling. For MIB files, pass scan_shape=(Rx, Ry).
    # For EMD/HDF5/py4DSTEM files, scan_shape can usually be None.
    input_type: str = "auto"  # auto, mib, h5, emd, npy, py4dstem
    datacube_path: str | None = None
    scan_shape: tuple[int, int] | None = None
    output_dir: str = "py4dstem_general_output"
    mem_mode: str = "MEMMAP"
    chunk_size: int = 32
    save_figures: bool = True
    save_npy: bool = True
    save_pickle: bool = True
    overwrite: bool = True


@dataclass
class VirtualImagingConfig:
    bf_radius_px: float = 20.0
    adf_inner_px: float = 40.0
    adf_outer_px: float = 100.0


@dataclass
class BraggDetectionConfig:
    probe_radius_px: float = 18.0
    corr_power: float = 1.0
    sigma: float = 1.0
    edge_boundary: int = 20
    min_relative_intensity: float = 0.05
    min_absolute_intensity: float | None = None
    min_peak_spacing: int = 8
    max_num_peaks: int = 100
    subpixel: str = "poly"
    upsample_factor: int = 16
    cuda: bool = False
    name: str = "braggvectors"
    use_mean_dp_probe_kernel: bool = True
    external_probe_kernel: str | None = None


@dataclass
class CalibrationConfig:
    run_origin_fit: bool = True
    run_ellipse_fit: bool = True
    bvm_mode_for_calibration: str = "cal"  # raw or cal; py4DSTEM may ignore unsupported modes
    q_pixel_size_inv_angstrom: float | None = None
    qr_rotation_deg: float | None = None
    notes: str = "Use standard sample / known lattice if possible before final strain analysis."


@dataclass
class BVMClassificationConfig:
    enabled: bool = True
    bvm_mode: str = "cal"
    num_bvm_maxima: int = 128
    bvm_maxima_min_distance_px: int = 4
    bvm_threshold_rel: float = 0.04
    voronoi_max_dist_px: float | None = 8.0
    initial_class_threshold: float = 0.30
    bp_fraction_threshold: float = 0.10
    max_initial_class_iterations: int = 200
    n_corr_init: int = 2
    nmf_components: int | None = None
    nmf_max_iter: int = 500
    random_state: int = 0


@dataclass
class CrystalConfig:
    phase_cifs: list[str] = field(default_factory=list)
    phase_names: list[str] = field(default_factory=list)
    conventional_standard_structure: bool = True
    primitive: bool = True
    accelerating_voltage: float = 300_000.0
    k_max: float = 1.5
    tol_structure_factor: float = 1e-4
    zone_axis_range: str | np.ndarray = "auto"
    angle_step_zone_axis: float = 2.0
    angle_step_in_plane: float = 2.0
    angle_coarse_zone_axis: float | None = None
    angle_refine_range: float | None = None
    corr_kernel_size: float = 0.08
    radial_power: float = 1.0
    intensity_power: float = 0.0
    calculate_correlation_array: bool = True
    tol_distance: float = 0.01
    sigma_excitation_error: float = 0.02
    num_matches_return: int = 2
    min_angle_between_matches_deg: float = 5.0
    min_number_peaks: int = 3
    inversion_symmetry: bool = True
    multiple_corr_reset: bool = True
    corr_normalize: bool = True
    low_confidence_threshold: float = 0.05
    cuda: bool = False


@dataclass
class GrainConfig:
    enabled: bool = True
    threshold_add: float = 1.0
    threshold_grow: float = 0.1
    angle_tolerance_deg: float = 5.0
    stripe_width: tuple[int, int] = (2, 2)
    area_min: int = 2


@dataclass
class StrainConfig:
    enabled: bool = False
    min_num_peaks: int = 5
    robust: bool = True
    robust_thresh: float = 3.0
    intensity_weighting: bool = False
    corr_range: tuple[float, float] = (0.0, 2.0)
    mask_from_corr: bool = True
    rotation_range: float | None = None


@dataclass
class WorkflowConfig:
    io: IOConfig
    virtual: VirtualImagingConfig = field(default_factory=VirtualImagingConfig)
    bragg: BraggDetectionConfig = field(default_factory=BraggDetectionConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    bvm_classification: BVMClassificationConfig = field(default_factory=BVMClassificationConfig)
    crystal: CrystalConfig = field(default_factory=CrystalConfig)
    grain: GrainConfig = field(default_factory=GrainConfig)
    strain: StrainConfig = field(default_factory=StrainConfig)
    roi: tuple[int, int, int, int] | None = None  # rx0, rx1, ry0, ry1, for parameter tuning


@dataclass
class CrystalRecord:
    name: str
    cif_path: str
    crystal: Any
    q_structure_factors: np.ndarray | None = None
    i_structure_factors: np.ndarray | None = None
    orientation_map: Any | None = None
    orientation_rgb: np.ndarray | None = None
    score_map: np.ndarray | None = None


@dataclass
class PhaseMatchingResult:
    phase_names: list[str]
    phase_id_map: np.ndarray
    phase_score_stack: np.ndarray
    best_score_map: np.ndarray
    second_best_score_map: np.ndarray
    confidence_gap_map: np.ndarray
    low_confidence_mask: np.ndarray
    orientation_maps: list[Any]
    per_phase_rgb: list[np.ndarray | None]
    phase_fraction: dict[str, float]


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def progress(callback: ProgressCallback | None, message: str, fraction: float | None = None) -> None:
    if callback is not None:
        callback(message, fraction)
        return
    pct = "" if fraction is None else f" [{fraction * 100:.0f}%]"
    print(f"{message}{pct}", flush=True)


def prepare_matplotlib(save_figures: bool):
    import matplotlib

    if save_figures:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def save_figure(plt: Any, path: Path, save: bool = True) -> None:
    if not save:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def jsonify(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if hasattr(obj, "__dict__") and obj.__class__.__module__.startswith("py4DSTEM"):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a function while ignoring keyword arguments unsupported by this py4DSTEM version."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)
    accepted = {
        key: value
        for key, value in kwargs.items()
        if key in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    }
    return func(*args, **accepted)


def import_py4dstem() -> Any:
    try:
        import py4DSTEM
    except ImportError as exc:
        raise ImportError(
            "py4DSTEM is required for this workflow. Install it in the runtime environment, "
            "for example: conda install -c conda-forge py4dstem"
        ) from exc
    return py4DSTEM


def ensure_output_dir(config: WorkflowConfig) -> Path:
    out = Path(config.io.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_array(config: WorkflowConfig, path: Path, array: np.ndarray) -> None:
    if config.io.save_npy:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, array)


def save_pickle_obj(config: WorkflowConfig, path: Path, obj: Any) -> None:
    if config.io.save_pickle:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(obj, f)


def normalize_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    lo, hi = np.nanpercentile(arr[finite], [1, 99])
    if hi <= lo:
        lo, hi = np.nanmin(arr[finite]), np.nanmax(arr[finite])
    if hi <= lo:
        return np.zeros_like(arr)
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1)


# -----------------------------------------------------------------------------
# MIB loading + basic maps
# -----------------------------------------------------------------------------


class DataCubeLoader:
    """Load a 4D-STEM DataCube from MIB, py4DSTEM/EMD/HDF5, or NumPy files.

    The loader intentionally uses conservative duck typing because py4DSTEM
    objects and tree APIs have evolved across releases. A valid DataCube is
    any object with a 4D ``.data`` array and a ``find_Bragg_disks`` method.
    """

    def __init__(self, config: WorkflowConfig, callback: ProgressCallback | None = None):
        self.config = config
        self.callback = callback

    def load(self) -> Any:
        py4DSTEM = import_py4dstem()
        path = Path(self.config.io.input_path)
        input_type = self.config.io.input_type.lower()
        if input_type == "auto":
            input_type = self._infer_input_type(path)
        progress(self.callback, f"Loading 4D-STEM DataCube ({input_type})...")

        if input_type == "npy":
            arr = np.load(path, mmap_mode="r" if str(self.config.io.mem_mode).upper() == "MEMMAP" else None)
            dc = self._wrap_numpy_as_datacube(py4DSTEM, arr)
        else:
            root = self._import_with_py4dstem(py4DSTEM, path, input_type)
            dc = self._extract_datacube(root)

        if not self._is_datacube_like(dc):
            raise TypeError(
                "Loaded object does not look like a py4DSTEM DataCube. "
                "Use --datacube-path for HDF5/EMD files with multiple tree nodes."
            )
        progress(self.callback, f"Loaded DataCube shape = {tuple(dc.data.shape)}")
        return dc

    def _infer_input_type(self, path: Path) -> str:
        ext = path.suffix.lower().lstrip(".")
        if ext in {"mib"}:
            return "mib"
        if ext in {"npy"}:
            return "npy"
        if ext in {"h5", "hdf5"}:
            return "h5"
        if ext in {"emd"}:
            return "emd"
        return "py4dstem"

    def _import_with_py4dstem(self, py4DSTEM: Any, path: Path, input_type: str) -> Any:
        attempts: list[dict[str, Any]] = []
        base = {"mem": self.config.io.mem_mode}
        if input_type == "mib" and self.config.io.scan_shape is not None:
            attempts.append({**base, "scan": self.config.io.scan_shape})
        if self.config.io.scan_shape is not None:
            attempts.append({**base, "scan": self.config.io.scan_shape})
        attempts.append(base)
        attempts.append({})

        last_exc: Exception | None = None
        for kwargs in attempts:
            try:
                return call_with_supported_kwargs(py4DSTEM.import_file, str(path), **kwargs)
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"py4DSTEM.import_file failed for {path}") from last_exc

    def _wrap_numpy_as_datacube(self, py4DSTEM: Any, arr: np.ndarray) -> Any:
        if arr.ndim != 4:
            raise ValueError(f"NumPy input must be 4D with shape (Rx, Ry, Qx, Qy), got {arr.shape}")
        DataCube = getattr(py4DSTEM, "DataCube", None)
        if DataCube is None:
            raise RuntimeError("py4DSTEM.DataCube class is not available in this environment.")
        try:
            return DataCube(data=arr)
        except TypeError:
            return DataCube(arr)

    def _extract_datacube(self, obj: Any) -> Any:
        if self.config.io.datacube_path:
            return self._get_tree_node(obj, self.config.io.datacube_path)
        if self._is_datacube_like(obj):
            return obj
        found = self._find_first_datacube(obj)
        if found is not None:
            return found
        return obj

    def _is_datacube_like(self, obj: Any) -> bool:
        data = getattr(obj, "data", None)
        return data is not None and getattr(data, "ndim", None) == 4 and hasattr(obj, "find_Bragg_disks")

    def _get_tree_node(self, root_obj: Any, path: str) -> Any:
        # Try exact py4DSTEM/emdfile tree calls first.
        for args, kwargs in [((path,), {}), ((), {"get": path})]:
            try:
                return root_obj.tree(*args, **kwargs)
            except Exception:
                pass
        # Then walk slash-separated names through dict-like or attribute APIs.
        node = root_obj
        for part in [p for p in path.replace("\\", "/").split("/") if p]:
            next_node = None
            if isinstance(node, Mapping) and part in node:
                next_node = node[part]
            elif hasattr(node, part):
                next_node = getattr(node, part)
            else:
                try:
                    next_node = node[part]
                except Exception:
                    pass
            if next_node is None:
                raise KeyError(f"Could not resolve datacube path component {part!r} in {path!r}")
            node = next_node
        return node

    def _find_first_datacube(self, obj: Any, max_depth: int = 4) -> Any | None:
        seen: set[int] = set()

        def rec(x: Any, depth: int) -> Any | None:
            if id(x) in seen or depth > max_depth:
                return None
            seen.add(id(x))
            if self._is_datacube_like(x):
                return x
            for attr in ("_tree", "children", "_children"):
                child = getattr(x, attr, None)
                y = rec_iter(child, depth + 1)
                if y is not None:
                    return y
            for name in dir(x):
                if name.startswith("_"):
                    continue
                try:
                    val = getattr(x, name)
                except Exception:
                    continue
                if callable(val) or isinstance(val, (str, bytes, int, float, bool, np.ndarray)):
                    continue
                y = rec_iter(val, depth + 1)
                if y is not None:
                    return y
            return None

        def rec_iter(container: Any, depth: int) -> Any | None:
            if container is None:
                return None
            if self._is_datacube_like(container):
                return container
            if isinstance(container, Mapping):
                iterable = list(container.values())[:50]
            elif isinstance(container, (list, tuple)):
                iterable = list(container)[:50]
            else:
                return rec(container, depth)
            for item in iterable:
                y = rec(item, depth)
                if y is not None:
                    return y
            return None

        return rec(obj, 0)


class DiffractionPreprocessor:
    def __init__(self, config: WorkflowConfig, output_dir: Path, callback: ProgressCallback | None = None):
        self.config = config
        self.output_dir = output_dir
        self.callback = callback
        self.plt = prepare_matplotlib(config.io.save_figures)

    def compute_mean_dp(self, datacube: Any) -> np.ndarray:
        Rx, Ry, Qx, Qy = datacube.data.shape
        chunk = self.config.io.chunk_size
        mean_dp = np.zeros((Qx, Qy), dtype=np.float64)
        count = 0
        for i in range(0, Rx, chunk):
            i2 = min(i + chunk, Rx)
            block = datacube.data[i:i2]
            mean_dp += block.sum(axis=(0, 1))
            count += block.shape[0] * block.shape[1]
            progress(self.callback, f"Mean DP chunk {i2}/{Rx}", i2 / Rx)
        mean_dp /= max(count, 1)
        save_array(self.config, self.output_dir / "01_mean_dp.npy", mean_dp)
        if self.config.io.save_figures:
            self.plt.figure(figsize=(7, 7))
            self.plt.imshow(np.log1p(mean_dp), cmap="gray")
            self.plt.title("Mean diffraction pattern")
            self.plt.colorbar()
            save_figure(self.plt, self.output_dir / "01_mean_dp.png")
        return mean_dp

    def estimate_bf_center(self, mean_dp: np.ndarray) -> tuple[float, float]:
        from scipy.ndimage import center_of_mass

        cy, cx = center_of_mass(np.nan_to_num(mean_dp, nan=0.0, posinf=0.0, neginf=0.0))
        progress(self.callback, f"Estimated BF center: cx={cx:.2f}, cy={cy:.2f}")
        return float(cx), float(cy)

    def build_detector_masks(self, shape: tuple[int, int], center_xy: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        Qx, Qy = shape
        cx, cy = center_xy
        yy, xx = np.ogrid[:Qx, :Qy]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        vcfg = self.config.virtual
        bf = r <= vcfg.bf_radius_px
        adf = (r >= vcfg.adf_inner_px) & (r <= vcfg.adf_outer_px)
        return bf, adf

    def compute_virtual_images(self, datacube: Any, mask_bf: np.ndarray, mask_adf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Rx, Ry = datacube.data.shape[:2]
        chunk = self.config.io.chunk_size
        vbf = np.zeros((Rx, Ry), dtype=np.float32)
        adf = np.zeros((Rx, Ry), dtype=np.float32)
        for i in range(0, Rx, chunk):
            i2 = min(i + chunk, Rx)
            block = datacube.data[i:i2]
            vbf[i:i2] = block[..., mask_bf].sum(axis=-1)
            adf[i:i2] = block[..., mask_adf].sum(axis=-1)
            progress(self.callback, f"Virtual images chunk {i2}/{Rx}", i2 / Rx)
        save_array(self.config, self.output_dir / "02_vbf.npy", vbf)
        save_array(self.config, self.output_dir / "02_adf.npy", adf)
        if self.config.io.save_figures:
            for name, arr in [("VBF", vbf), ("ADF", adf)]:
                self.plt.figure(figsize=(7, 7))
                self.plt.imshow(arr)
                self.plt.title(name)
                self.plt.colorbar()
                save_figure(self.plt, self.output_dir / f"02_{name.lower()}.png")
        return vbf, adf

    def compute_com(self, datacube: Any) -> tuple[np.ndarray, np.ndarray]:
        Rx, Ry, Qx, Qy = datacube.data.shape
        chunk = self.config.io.chunk_size
        qx = np.arange(Qx)
        qy = np.arange(Qy)
        QX, QY = np.meshgrid(qx, qy, indexing="ij")
        comx = np.zeros((Rx, Ry), dtype=np.float32)
        comy = np.zeros((Rx, Ry), dtype=np.float32)
        for i in range(0, Rx, chunk):
            i2 = min(i + chunk, Rx)
            block = datacube.data[i:i2]
            total = block.sum(axis=(-2, -1)).astype(np.float64)
            total[total == 0] = 1.0
            comx[i:i2] = (block * QX).sum(axis=(-2, -1)) / total
            comy[i:i2] = (block * QY).sum(axis=(-2, -1)) / total
            progress(self.callback, f"CoM chunk {i2}/{Rx}", i2 / Rx)
        save_array(self.config, self.output_dir / "03_comx.npy", comx)
        save_array(self.config, self.output_dir / "03_comy.npy", comy)
        return comx, comy


# -----------------------------------------------------------------------------
# Bragg detection + calibration
# -----------------------------------------------------------------------------


class BraggDiskDetector:
    def __init__(self, config: WorkflowConfig, output_dir: Path, callback: ProgressCallback | None = None):
        self.config = config
        self.output_dir = output_dir
        self.callback = callback
        self.plt = prepare_matplotlib(config.io.save_figures)

    def make_probe_kernel(self, datacube: Any, mean_dp: np.ndarray) -> Any:
        py4DSTEM = import_py4dstem()
        bcfg = self.config.bragg
        if bcfg.external_probe_kernel:
            kernel_path = Path(bcfg.external_probe_kernel)
            if kernel_path.suffix.lower() == ".npy":
                return np.load(kernel_path)
            try:
                import imageio.v3 as iio

                return iio.imread(kernel_path)
            except Exception as exc:
                raise ValueError(f"Could not load external probe kernel: {kernel_path}") from exc
        if bcfg.use_mean_dp_probe_kernel:
            return self._make_probe_kernel_compatible(py4DSTEM, np.asarray(mean_dp), bcfg.probe_radius_px)
        return None

    def _make_probe_kernel_compatible(self, py4DSTEM: Any, mean_dp: np.ndarray, probe_radius_px: float) -> np.ndarray:
        """Build a Bragg-disk template without relying on version-specific py4DSTEM paths.

        py4DSTEM has changed its public probe API across releases.  Some old
        examples used ``py4DSTEM.process.probe.get_probe_kernel``; recent
        documentation exposes probe calibration under
        ``py4DSTEM.process.calibration.probe`` and kernel helpers through the
        ``Probe`` class.  For reproducible batch processing we therefore use a
        small, explicit soft-edged disk kernel, centered on the bright-field disk
        in the mean diffraction pattern.
        """
        dp = np.asarray(mean_dp, dtype=np.float64)
        if dp.ndim != 2:
            raise ValueError(f"mean_dp must be 2D, got shape {dp.shape}")

        # Estimate the central disk position robustly.  If py4DSTEM's current
        # probe-size helper is available, use its center; otherwise use a
        # weighted center of mass of the brightest central beam.
        cy = (dp.shape[0] - 1) / 2.0
        cx = (dp.shape[1] - 1) / 2.0
        radius = float(probe_radius_px)
        try:
            probe_mod = getattr(getattr(py4DSTEM.process, "calibration", None), "probe", None)
            get_probe_size = getattr(probe_mod, "get_probe_size", None)
            if callable(get_probe_size):
                r_est, x0_est, y0_est = get_probe_size(dp)
                if np.isfinite(r_est) and r_est > 0 and radius <= 0:
                    radius = float(r_est)
                # py4DSTEM returns x0,y0; array indexing below is row=y, col=x.
                if np.isfinite(x0_est) and np.isfinite(y0_est):
                    cx = float(x0_est)
                    cy = float(y0_est)
        except Exception:
            pass

        if not np.isfinite(radius) or radius <= 0:
            radius = 0.07 * min(dp.shape)

        # Bright central disk weighted center as a fallback / sanity check.
        try:
            shifted = dp - np.nanmin(dp)
            thr = np.nanpercentile(shifted, 99.0)
            mask = shifted >= thr
            if mask.any() and shifted[mask].sum() > 0:
                rr, cc = np.indices(dp.shape)
                w = shifted * mask
                cy_f = float((rr * w).sum() / w.sum())
                cx_f = float((cc * w).sum() / w.sum())
                # Use fallback only if py4DSTEM center was not informative.
                if abs(cx - (dp.shape[1] - 1) / 2.0) < 1e-6 and abs(cy - (dp.shape[0] - 1) / 2.0) < 1e-6:
                    cy, cx = cy_f, cx_f
        except Exception:
            pass

        rr, cc = np.indices(dp.shape)
        rad = np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2)
        edge_width = max(1.5, 0.12 * radius)
        kernel = 1.0 / (1.0 + np.exp((rad - radius) / edge_width))

        # Normalize for stable cross-correlation.
        kernel = kernel - kernel.mean()
        norm = np.sqrt(np.sum(kernel * kernel))
        if norm > 0:
            kernel = kernel / norm
        progress(self.callback, f"Using compatible soft-disk probe kernel: center=({cx:.1f}, {cy:.1f}), radius={radius:.1f} px")
        return kernel.astype(np.float32)

    def detect(self, datacube: Any, mean_dp: np.ndarray) -> Any:
        bcfg = self.config.bragg
        kernel = self.make_probe_kernel(datacube, mean_dp)
        progress(self.callback, "Detecting Bragg disks...")
        kwargs = dict(
            template=kernel,
            corrPower=bcfg.corr_power,
            sigma=bcfg.sigma,
            edgeBoundary=bcfg.edge_boundary,
            minRelativeIntensity=bcfg.min_relative_intensity,
            minPeakSpacing=bcfg.min_peak_spacing,
            maxNumPeaks=bcfg.max_num_peaks,
            subpixel=bcfg.subpixel,
            upsample_factor=bcfg.upsample_factor,
            CUDA=bcfg.cuda,
            name=bcfg.name,
            returncalc=True,
        )
        if bcfg.min_absolute_intensity is not None:
            kwargs["minAbsoluteIntensity"] = bcfg.min_absolute_intensity
        bragg = call_with_supported_kwargs(datacube.find_Bragg_disks, **kwargs)
        save_pickle_obj(self.config, self.output_dir / "04_braggvectors.pkl", bragg)
        progress(self.callback, "Bragg disk detection complete.")
        return bragg

    def histogram(self, braggvectors: Any, mode: str = "cal") -> np.ndarray:
        try:
            hist = braggvectors.histogram(mode=mode).data
        except Exception:
            hist = braggvectors.histogram(mode="raw").data
        bvm = np.asarray(hist)
        save_array(self.config, self.output_dir / f"04_bvm_{mode}.npy", bvm)
        if self.config.io.save_figures:
            self.plt.figure(figsize=(7, 7))
            self.plt.imshow(np.log1p(bvm), cmap="gray")
            self.plt.title(f"Bragg vector map ({mode})")
            self.plt.colorbar()
            save_figure(self.plt, self.output_dir / f"04_bvm_{mode}.png")
        return bvm


class CalibrationManager:
    def __init__(self, config: WorkflowConfig, output_dir: Path, callback: ProgressCallback | None = None):
        self.config = config
        self.output_dir = output_dir
        self.callback = callback

    def run(self, datacube: Any, braggvectors: Any, bvm: np.ndarray) -> dict[str, Any]:
        py4DSTEM = import_py4dstem()
        ccfg = self.config.calibration
        metadata: dict[str, Any] = {"status": "not_run", "notes": ccfg.notes}
        progress(self.callback, "Running Bragg-vector calibration...")
        try:
            if hasattr(py4DSTEM.process, "calibration") and hasattr(
                py4DSTEM.process.calibration, "origin_and_Elliptical_calibration"
            ):
                call_with_supported_kwargs(
                    py4DSTEM.process.calibration.origin_and_Elliptical_calibration,
                    bragg_peaks=braggvectors,
                    datacube=datacube,
                    bragg_vector_map=bvm,
                )
                metadata["status"] = "origin_ellipse_done"
            else:
                metadata["status"] = "skipped_no_calibration_api"
        except Exception as exc:
            metadata["status"] = "warning"
            metadata["warning"] = str(exc)
            progress(self.callback, f"Calibration warning: {exc}")

        # Optional explicit metadata for reproducibility.
        if ccfg.q_pixel_size_inv_angstrom is not None:
            metadata["q_pixel_size_inv_angstrom"] = ccfg.q_pixel_size_inv_angstrom
        if ccfg.qr_rotation_deg is not None:
            metadata["qr_rotation_deg"] = ccfg.qr_rotation_deg

        with (self.output_dir / "05_calibration_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(jsonify(metadata), f, indent=2, ensure_ascii=False)
        progress(self.callback, f"Calibration status: {metadata['status']}")
        return metadata


# -----------------------------------------------------------------------------
# BVM + Voronoi structural pre-classification
# -----------------------------------------------------------------------------


class BVMVoronoiClassifier:
    """Implements the BVM maxima -> Voronoi labels -> co-occurrence classes step.

    This is a structural pre-classification / quality-control layer. It is not a
    replacement for crystallographic phase matching from CIF libraries; instead,
    it offers a data-driven check that phase/grain/twin boundaries are separable
    before costly orientation matching.
    """

    def __init__(self, config: WorkflowConfig, output_dir: Path, callback: ProgressCallback | None = None):
        self.config = config
        self.output_dir = output_dir
        self.callback = callback
        self.plt = prepare_matplotlib(config.io.save_figures)

    def find_bvm_maxima(self, bvm: np.ndarray) -> np.ndarray:
        cfg = self.config.bvm_classification
        arr = np.asarray(bvm, dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        threshold_abs = cfg.bvm_threshold_rel * float(arr.max()) if arr.size else 0.0
        try:
            from skimage.feature import peak_local_max

            coords = peak_local_max(
                arr,
                min_distance=cfg.bvm_maxima_min_distance_px,
                threshold_abs=threshold_abs,
                num_peaks=cfg.num_bvm_maxima,
                exclude_border=False,
            )
        except Exception:
            from scipy.ndimage import maximum_filter

            size = max(1, int(cfg.bvm_maxima_min_distance_px) * 2 + 1)
            mask = (arr == maximum_filter(arr, size=size)) & (arr >= threshold_abs)
            coords = np.argwhere(mask)
            if coords.shape[0] > cfg.num_bvm_maxima:
                vals = arr[coords[:, 0], coords[:, 1]]
                order = np.argsort(vals)[::-1][: cfg.num_bvm_maxima]
                coords = coords[order]
        # Return qx, qy convention as columns; image coords are row, col.
        maxima_qxy = np.column_stack([coords[:, 1], coords[:, 0]]).astype(float)
        np.save(self.output_dir / "06_bvm_maxima_qxy.npy", maxima_qxy)
        progress(self.callback, f"BVM maxima detected: {len(maxima_qxy)}")
        return maxima_qxy

    def voronoi_label_image(self, bvm_shape: tuple[int, int], maxima_qxy: np.ndarray) -> np.ndarray:
        from scipy.spatial import cKDTree

        h, w = bvm_shape
        yy, xx = np.mgrid[:h, :w]
        points = np.column_stack([xx.ravel(), yy.ravel()])
        tree = cKDTree(maxima_qxy)
        _, labels = tree.query(points, k=1)
        label_image = labels.reshape(h, w).astype(np.int32)
        np.save(self.output_dir / "06_voronoi_label_image.npy", label_image)
        if self.config.io.save_figures:
            self.plt.figure(figsize=(7, 7))
            self.plt.imshow(label_image)
            self.plt.scatter(maxima_qxy[:, 0], maxima_qxy[:, 1], s=8)
            self.plt.title("BVM Voronoi regions")
            save_figure(self.plt, self.output_dir / "06_bvm_voronoi.png")
        return label_image

    def get_braggpeak_labels(self, braggvectors: Any, maxima_qxy: np.ndarray) -> Any:
        cfg = self.config.bvm_classification
        try:
            from py4DSTEM.process.classification.braggvectorclassification import (
                get_braggpeak_labels_by_scan_position,
            )

            return get_braggpeak_labels_by_scan_position(
                braggvectors,
                maxima_qxy[:, 0],
                maxima_qxy[:, 1],
                max_dist=cfg.voronoi_max_dist_px,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not label Bragg peaks using py4DSTEM classification utilities. "
                "This step can be disabled with --no-bvm-classification."
            ) from exc

    @staticmethod
    def labels_to_feature_matrix(labels_by_scan: Any, n_labels: int) -> tuple[np.ndarray, tuple[int, int]]:
        Rx = len(labels_by_scan)
        Ry = len(labels_by_scan[0]) if Rx > 0 else 0
        X = np.zeros((Rx * Ry, n_labels), dtype=np.float32)
        for ix in range(Rx):
            for iy in range(Ry):
                row = ix * Ry + iy
                labels = labels_by_scan[ix][iy]
                if labels is None:
                    continue
                for lab in labels:
                    if 0 <= int(lab) < n_labels:
                        X[row, int(lab)] = 1.0
        return X, (Rx, Ry)

    def initial_classes(self, labels_by_scan: Any, n_labels: int) -> list[set[int]]:
        cfg = self.config.bvm_classification
        try:
            from py4DSTEM.process.classification.braggvectorclassification import get_initial_classes

            classes = get_initial_classes(
                labels_by_scan,
                n_labels,
                thresh=cfg.initial_class_threshold,
                BP_fraction_thresh=cfg.bp_fraction_threshold,
                max_iterations=cfg.max_initial_class_iterations,
                n_corr_init=cfg.n_corr_init,
            )
            return [set(map(int, c)) for c in classes]
        except Exception:
            return []

    def nmf_refinement(self, X: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray] | None:
        cfg = self.config.bvm_classification
        if cfg.nmf_components is None or cfg.nmf_components <= 0:
            return None
        try:
            from sklearn.decomposition import NMF

            model = NMF(
                n_components=cfg.nmf_components,
                init="nndsvda",
                max_iter=cfg.nmf_max_iter,
                random_state=cfg.random_state,
            )
            W = model.fit_transform(np.maximum(X, 0))
            H = model.components_
            class_images = W.reshape(shape[0], shape[1], cfg.nmf_components)
            np.save(self.output_dir / "06_nmf_W.npy", W)
            np.save(self.output_dir / "06_nmf_H.npy", H)
            np.save(self.output_dir / "06_nmf_class_images.npy", class_images)
            if self.config.io.save_figures:
                for k in range(cfg.nmf_components):
                    self.plt.figure(figsize=(7, 7))
                    self.plt.imshow(class_images[:, :, k])
                    self.plt.title(f"NMF structural class {k}")
                    self.plt.colorbar()
                    save_figure(self.plt, self.output_dir / f"06_nmf_class_{k}.png")
            return {"W": W, "H": H, "class_images": class_images}
        except Exception as exc:
            progress(self.callback, f"NMF refinement skipped: {exc}")
            return None

    def run(self, braggvectors: Any, bvm: np.ndarray) -> dict[str, Any]:
        if not self.config.bvm_classification.enabled:
            return {"enabled": False}
        progress(self.callback, "Running BVM Voronoi structural pre-classification...")
        maxima = self.find_bvm_maxima(bvm)
        labels_img = self.voronoi_label_image(bvm.shape, maxima)
        labels_by_scan = self.get_braggpeak_labels(braggvectors, maxima)
        X, shape = self.labels_to_feature_matrix(labels_by_scan, len(maxima))
        np.save(self.output_dir / "06_bragg_label_features.npy", X)
        classes = self.initial_classes(labels_by_scan, len(maxima))
        with (self.output_dir / "06_initial_classes.json").open("w", encoding="utf-8") as f:
            json.dump([sorted(list(c)) for c in classes], f, indent=2)
        nmf = self.nmf_refinement(X, shape)
        return {
            "enabled": True,
            "num_bvm_maxima": int(len(maxima)),
            "num_initial_classes": int(len(classes)),
            "scan_shape": shape,
            "has_nmf": nmf is not None,
            "maxima_qxy": maxima,
            "voronoi_label_image": labels_img,
            "initial_classes": classes,
            "feature_matrix": X,
        }


# -----------------------------------------------------------------------------
# Crystal Analysis modules
# -----------------------------------------------------------------------------


class CIFManager:
    def __init__(self, config: WorkflowConfig, callback: ProgressCallback | None = None):
        self.config = config
        self.callback = callback

    def load_crystals(self) -> list[CrystalRecord]:
        py4DSTEM = import_py4dstem()
        c = self.config.crystal
        if not c.phase_cifs:
            raise ValueError("At least one CIF file is required for phase/orientation mapping.")
        names = c.phase_names or [Path(p).stem for p in c.phase_cifs]
        if len(names) != len(c.phase_cifs):
            raise ValueError("phase_names must have the same length as phase_cifs.")
        records: list[CrystalRecord] = []
        for name, cif in zip(names, c.phase_cifs):
            path = Path(cif)
            if not path.exists():
                raise FileNotFoundError(f"CIF file not found: {path}")
            progress(self.callback, f"Loading CIF for phase '{name}': {path}")
            crystal_cls = py4DSTEM.process.diffraction.Crystal
            crystal = call_with_supported_kwargs(
                crystal_cls.from_CIF,
                str(path),
                primitive=c.primitive,
                conventional_standard_structure=c.conventional_standard_structure,
            )
            try:
                crystal.name = name
            except Exception:
                pass
            records.append(CrystalRecord(name=name, cif_path=str(path), crystal=crystal))
        return records


class StructureFactors:
    def __init__(self, config: WorkflowConfig, output_dir: Path, callback: ProgressCallback | None = None):
        self.config = config
        self.output_dir = output_dir
        self.callback = callback

    def calculate(self, records: list[CrystalRecord]) -> list[CrystalRecord]:
        c = self.config.crystal
        for rec in records:
            progress(self.callback, f"Calculating structure factors: {rec.name}")
            if hasattr(rec.crystal, "setup_diffraction"):
                call_with_supported_kwargs(
                    rec.crystal.setup_diffraction,
                    accelerating_voltage=c.accelerating_voltage,
                )
            out = call_with_supported_kwargs(
                rec.crystal.calculate_structure_factors,
                k_max=c.k_max,
                tol_structure_factor=c.tol_structure_factor,
                return_intensities=True,
            )
            if isinstance(out, tuple) and len(out) >= 2:
                rec.q_structure_factors = np.asarray(out[0])
                rec.i_structure_factors = np.asarray(out[1])
                np.save(self.output_dir / f"07_{rec.name}_q_structure_factors.npy", rec.q_structure_factors)
                np.save(self.output_dir / f"07_{rec.name}_i_structure_factors.npy", rec.i_structure_factors)
        return records


class SimulatedDiffraction:
    def __init__(self, config: WorkflowConfig, output_dir: Path, callback: ProgressCallback | None = None):
        self.config = config
        self.output_dir = output_dir
        self.callback = callback
        self.plt = prepare_matplotlib(config.io.save_figures)

    def build_orientation_libraries(self, records: list[CrystalRecord]) -> list[CrystalRecord]:
        c = self.config.crystal
        for rec in records:
            progress(self.callback, f"Building orientation library: {rec.name}")
            t0 = time.time()
            call_with_supported_kwargs(
                rec.crystal.orientation_plan,
                zone_axis_range=c.zone_axis_range,
                angle_step_zone_axis=c.angle_step_zone_axis,
                angle_coarse_zone_axis=c.angle_coarse_zone_axis,
                angle_refine_range=c.angle_refine_range,
                angle_step_in_plane=c.angle_step_in_plane,
                accel_voltage=c.accelerating_voltage,
                corr_kernel_size=c.corr_kernel_size,
                radial_power=c.radial_power,
                intensity_power=c.intensity_power,
                calculate_correlation_array=c.calculate_correlation_array,
                tol_distance=c.tol_distance,
                CUDA=c.cuda,
                progress_bar=True,
            )
            progress(self.callback, f"Orientation library ready: {rec.name} ({time.time() - t0:.1f}s)")
            if self.config.io.save_figures and hasattr(rec.crystal, "plot_orientation_plan"):
                try:
                    out = call_with_supported_kwargs(rec.crystal.plot_orientation_plan, returnfig=True)
                    # py4DSTEM returns varying handles; just save the active figure.
                    save_figure(self.plt, self.output_dir / f"08_{rec.name}_orientation_plan.png")
                except Exception:
                    pass
        return records

    def simulate_reference_patterns(self, records: list[CrystalRecord]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        c = self.config.crystal
        for rec in records:
            if not hasattr(rec.crystal, "generate_diffraction_pattern"):
                continue
            try:
                pattern = call_with_supported_kwargs(
                    rec.crystal.generate_diffraction_pattern,
                    zone_axis_lattice=np.array([0, 0, 1]),
                    proj_x_lattice=np.array([1, 0, 0]),
                    sigma_excitation_error=c.sigma_excitation_error,
                    k_max=c.k_max,
                )
                out[rec.name] = pattern
                save_pickle_obj(self.config, self.output_dir / f"08_{rec.name}_sim_001.pkl", pattern)
            except Exception as exc:
                progress(self.callback, f"Reference diffraction simulation skipped for {rec.name}: {exc}")
        return out


class OrientationMatching:
    def __init__(self, config: WorkflowConfig, output_dir: Path, callback: ProgressCallback | None = None):
        self.config = config
        self.output_dir = output_dir
        self.callback = callback
        self.plt = prepare_matplotlib(config.io.save_figures)

    def match_each_phase(self, records: list[CrystalRecord], braggvectors: Any) -> list[CrystalRecord]:
        c = self.config.crystal
        for rec in records:
            progress(self.callback, f"Orientation matching against phase: {rec.name}")
            t0 = time.time()
            rec.orientation_map = call_with_supported_kwargs(
                rec.crystal.match_orientations,
                braggvectors,
                num_matches_return=c.num_matches_return,
                min_angle_between_matches_deg=c.min_angle_between_matches_deg,
                min_number_peaks=c.min_number_peaks,
                inversion_symmetry=c.inversion_symmetry,
                multiple_corr_reset=c.multiple_corr_reset,
                return_orientation=True,
                progress_bar=True,
            )
            rec.score_map = extract_orientation_score_map(rec.orientation_map, match_index=0)
            save_pickle_obj(self.config, self.output_dir / f"09_{rec.name}_orientation_map.pkl", rec.orientation_map)
            if rec.score_map is not None:
                save_array(self.config, self.output_dir / f"09_{rec.name}_orientation_score.npy", rec.score_map)
                if self.config.io.save_figures:
                    self.plt.figure(figsize=(7, 7))
                    self.plt.imshow(rec.score_map)
                    self.plt.title(f"{rec.name} ACOM score")
                    self.plt.colorbar()
                    save_figure(self.plt, self.output_dir / f"09_{rec.name}_orientation_score.png")
            rec.orientation_rgb = self.orientation_rgb(rec)
            progress(self.callback, f"Orientation match complete: {rec.name} ({time.time() - t0:.1f}s)")
        return records

    def orientation_rgb(self, rec: CrystalRecord) -> np.ndarray | None:
        if rec.orientation_map is None or not hasattr(rec.crystal, "plot_orientation_maps"):
            return None
        try:
            output = call_with_supported_kwargs(
                rec.crystal.plot_orientation_maps,
                orientation_map=rec.orientation_map,
                orientation_ind=0,
                corr_normalize=self.config.crystal.corr_normalize,
                returnfig=True,
                progress_bar=False,
            )
            rgb = coerce_orientation_rgb(output)
            if rgb is not None:
                save_array(self.config, self.output_dir / f"09_{rec.name}_orientation_rgb.npy", rgb)
                if self.config.io.save_figures:
                    self.plt.figure(figsize=(7, 7))
                    self.plt.imshow(rgb)
                    self.plt.title(f"{rec.name} orientation map (IPF/RGB)")
                    save_figure(self.plt, self.output_dir / f"09_{rec.name}_orientation_rgb.png")
            return rgb
        except Exception as exc:
            progress(self.callback, f"Could not render orientation RGB for {rec.name}: {exc}")
            return None


class PhaseMatching:
    def __init__(self, config: WorkflowConfig, output_dir: Path, callback: ProgressCallback | None = None):
        self.config = config
        self.output_dir = output_dir
        self.callback = callback
        self.plt = prepare_matplotlib(config.io.save_figures)

    def match(self, records: list[CrystalRecord], braggvectors: Any) -> PhaseMatchingResult:
        if len(records) == 0:
            raise ValueError("No crystal records available for phase matching.")
        if len(records) == 1:
            score = records[0].score_map
            if score is None:
                score = extract_orientation_score_map(records[0].orientation_map)
            if score is None:
                score = infer_scan_shape_from_orientation_map(records[0].orientation_map, braggvectors)
                score = np.ones(score, dtype=np.float32)
            stack = score[None, ...]
        else:
            score_maps = []
            for rec in records:
                score = rec.score_map if rec.score_map is not None else extract_orientation_score_map(rec.orientation_map)
                if score is None:
                    raise RuntimeError(
                        f"Could not extract correlation/score map for phase {rec.name}. "
                        "Inspect the OrientationMap object and extend extract_orientation_score_map()."
                    )
                score_maps.append(np.asarray(score, dtype=np.float32))
            stack = np.stack(score_maps, axis=0)

        phase_id = np.argmax(stack, axis=0).astype(np.int16)
        sorted_scores = np.sort(stack, axis=0)
        best = sorted_scores[-1]
        second = sorted_scores[-2] if stack.shape[0] > 1 else np.zeros_like(best)
        gap = best - second
        low_conf = gap < self.config.crystal.low_confidence_threshold
        names = [r.name for r in records]
        frac = {name: float(np.mean(phase_id == i)) for i, name in enumerate(names)}

        result = PhaseMatchingResult(
            phase_names=names,
            phase_id_map=phase_id,
            phase_score_stack=stack,
            best_score_map=best,
            second_best_score_map=second,
            confidence_gap_map=gap,
            low_confidence_mask=low_conf,
            orientation_maps=[r.orientation_map for r in records],
            per_phase_rgb=[r.orientation_rgb for r in records],
            phase_fraction=frac,
        )
        self.save_result(result)
        self.try_py4dstem_crystal_phase(records, braggvectors)
        return result

    def try_py4dstem_crystal_phase(self, records: list[CrystalRecord], braggvectors: Any) -> None:
        """Optional py4DSTEM Crystal_Phase quantification, when available.

        The correlation-race phase map above is deterministic and saved regardless.
        Crystal_Phase adds a py4DSTEM-native NNLS phase quantification layer.
        """
        if len(records) < 2:
            return
        try:
            py4DSTEM = import_py4dstem()
            Crystal_Phase = getattr(py4DSTEM.process.diffraction, "Crystal_Phase")
            cp = Crystal_Phase([r.crystal for r in records], [r.orientation_map for r in records], "multi_phase")
            q = call_with_supported_kwargs(
                cp.quantify_phase,
                braggvectors,
                tolerance_distance=self.config.crystal.corr_kernel_size,
                method="nnls",
                intensity_power=self.config.crystal.intensity_power,
            )
            save_pickle_obj(self.config, self.output_dir / "10_py4dstem_crystal_phase_quantification.pkl", q)
        except Exception as exc:
            progress(self.callback, f"py4DSTEM Crystal_Phase quantification skipped: {exc}")

    def save_result(self, result: PhaseMatchingResult) -> None:
        save_array(self.config, self.output_dir / "10_phase_id_map.npy", result.phase_id_map)
        save_array(self.config, self.output_dir / "10_phase_score_stack.npy", result.phase_score_stack)
        save_array(self.config, self.output_dir / "10_phase_confidence_gap.npy", result.confidence_gap_map)
        save_array(self.config, self.output_dir / "10_phase_low_confidence_mask.npy", result.low_confidence_mask.astype(np.uint8))
        with (self.output_dir / "10_phase_fraction.json").open("w", encoding="utf-8") as f:
            json.dump(result.phase_fraction, f, indent=2, ensure_ascii=False)
        if self.config.io.save_figures:
            self.plt.figure(figsize=(7, 7))
            self.plt.imshow(result.phase_id_map)
            self.plt.title("Phase ID map")
            self.plt.colorbar()
            save_figure(self.plt, self.output_dir / "10_phase_id_map.png")
            self.plt.figure(figsize=(7, 7))
            self.plt.imshow(result.confidence_gap_map)
            self.plt.title("Phase confidence gap")
            self.plt.colorbar()
            save_figure(self.plt, self.output_dir / "10_phase_confidence_gap.png")
            composite = build_composite_phase_orientation_rgb(result)
            if composite is not None:
                np.save(self.output_dir / "10_phase_orientation_composite_rgb.npy", composite)
                self.plt.figure(figsize=(7, 7))
                self.plt.imshow(composite)
                self.plt.title("Composite phase + orientation RGB")
                save_figure(self.plt, self.output_dir / "10_phase_orientation_composite_rgb.png")


class GrainAnalysis:
    def __init__(self, config: WorkflowConfig, output_dir: Path, callback: ProgressCallback | None = None):
        self.config = config
        self.output_dir = output_dir
        self.callback = callback

    def run(self, records: list[CrystalRecord]) -> dict[str, Any]:
        if not self.config.grain.enabled:
            return {"enabled": False}
        g = self.config.grain
        out: dict[str, Any] = {"enabled": True, "phases": {}}
        for rec in records:
            if rec.orientation_map is None or not hasattr(rec.crystal, "cluster_grains"):
                continue
            progress(self.callback, f"Grain clustering: {rec.name}")
            try:
                call_with_supported_kwargs(
                    rec.crystal.cluster_grains,
                    threshold_add=g.threshold_add,
                    threshold_grow=g.threshold_grow,
                    angle_tolerance_deg=g.angle_tolerance_deg,
                    progress_bar=True,
                )
                clustered = None
                if hasattr(rec.crystal, "cluster_orientation_map"):
                    clustered = call_with_supported_kwargs(
                        rec.crystal.cluster_orientation_map,
                        stripe_width=g.stripe_width,
                        area_min=g.area_min,
                    )
                    save_pickle_obj(self.config, self.output_dir / f"11_{rec.name}_clustered_orientation_map.pkl", clustered)
                out["phases"][rec.name] = "complete" if clustered is not None else "clustered_no_map"
            except Exception as exc:
                progress(self.callback, f"Grain clustering warning for {rec.name}: {exc}")
                out["phases"][rec.name] = f"warning: {exc}"
        return out


class StrainAnalysis:
    def __init__(self, config: WorkflowConfig, output_dir: Path, callback: ProgressCallback | None = None):
        self.config = config
        self.output_dir = output_dir
        self.callback = callback
        self.plt = prepare_matplotlib(config.io.save_figures)

    def run(self, records: list[CrystalRecord], braggvectors: Any, phase_result: PhaseMatchingResult | None = None) -> dict[str, Any]:
        if not self.config.strain.enabled:
            return {"enabled": False}
        s = self.config.strain
        c = self.config.crystal
        out: dict[str, Any] = {"enabled": True, "phases": {}}
        for i, rec in enumerate(records):
            if rec.orientation_map is None or not hasattr(rec.crystal, "calculate_strain"):
                continue
            progress(self.callback, f"Strain fitting: {rec.name}")
            try:
                strain_map = call_with_supported_kwargs(
                    rec.crystal.calculate_strain,
                    braggvectors,
                    rec.orientation_map,
                    corr_kernel_size=c.corr_kernel_size,
                    sigma_excitation_error=c.sigma_excitation_error,
                    k_max=c.k_max,
                    min_num_peaks=s.min_num_peaks,
                    intensity_weighting=s.intensity_weighting,
                    robust=s.robust,
                    robust_thresh=s.robust_thresh,
                    rotation_range=s.rotation_range,
                    mask_from_corr=s.mask_from_corr,
                    corr_range=s.corr_range,
                    corr_normalize=c.corr_normalize,
                    progress_bar=True,
                )
                save_pickle_obj(self.config, self.output_dir / f"12_{rec.name}_strain_map.pkl", strain_map)
                self.try_save_strain_components(rec.name, strain_map, phase_result, phase_index=i)
                out["phases"][rec.name] = "complete"
            except Exception as exc:
                progress(self.callback, f"Strain analysis warning for {rec.name}: {exc}")
                out["phases"][rec.name] = f"warning: {exc}"
        return out

    def try_save_strain_components(
        self,
        name: str,
        strain_map: Any,
        phase_result: PhaseMatchingResult | None,
        phase_index: int,
    ) -> None:
        # py4DSTEM RealSlice exposes data in different ways across versions.
        arr = None
        for attr in ("data", "slices"):
            if hasattr(strain_map, attr):
                try:
                    arr = np.asarray(getattr(strain_map, attr))
                    break
                except Exception:
                    pass
        if arr is not None and arr.size:
            np.save(self.output_dir / f"12_{name}_strain_raw.npy", arr)
        if phase_result is not None and arr is not None and arr.ndim >= 3:
            # Mask non-winning phase pixels for convenience. Assumes first two dims are scan axes.
            mask = phase_result.phase_id_map == phase_index
            masked = np.array(arr, copy=True)
            try:
                masked[~mask] = np.nan
                np.save(self.output_dir / f"12_{name}_strain_phase_masked.npy", masked)
            except Exception:
                pass


# -----------------------------------------------------------------------------
# Orientation / phase helper functions
# -----------------------------------------------------------------------------


def extract_orientation_score_map(orientation_map: Any, match_index: int = 0) -> np.ndarray | None:
    """Extract a 2D correlation/score map from a py4DSTEM OrientationMap.

    py4DSTEM versions differ in exact attribute names. This routine checks
    common names and handles arrays shaped as (Rx, Ry), (Rx, Ry, nmatch), or
    (nmatch, Rx, Ry).
    """
    if orientation_map is None:
        return None
    candidates = [
        "corr",
        "correlation",
        "corr_score",
        "corr_scores",
        "score",
        "scores",
        "intensity",
        "match_score",
    ]
    for attr in candidates:
        if hasattr(orientation_map, attr):
            try:
                arr = np.asarray(getattr(orientation_map, attr))
                coerced = coerce_score_array(arr, match_index)
                if coerced is not None:
                    return coerced.astype(np.float32)
            except Exception:
                continue
    # Some containers behave like dictionaries.
    if isinstance(orientation_map, Mapping):
        for attr in candidates:
            if attr in orientation_map:
                coerced = coerce_score_array(np.asarray(orientation_map[attr]), match_index)
                if coerced is not None:
                    return coerced.astype(np.float32)
    # Last resort: inspect public arrays and choose a plausible 2D/3D numeric one.
    for attr in dir(orientation_map):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(orientation_map, attr)
        except Exception:
            continue
        if callable(value):
            continue
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if arr.dtype.kind not in "fiu" or arr.size == 0:
            continue
        coerced = coerce_score_array(arr, match_index)
        if coerced is not None and coerced.ndim == 2:
            # Avoid choosing orientation matrices by rejecting trailing 3x3.
            if arr.ndim >= 4 and arr.shape[-2:] == (3, 3):
                continue
            return coerced.astype(np.float32)
    return None


def coerce_score_array(arr: np.ndarray, match_index: int = 0) -> np.ndarray | None:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # (Rx, Ry, nmatch) or (nmatch, Rx, Ry)
        if arr.shape[-1] <= 16:
            return arr[:, :, min(match_index, arr.shape[-1] - 1)]
        if arr.shape[0] <= 16:
            return arr[min(match_index, arr.shape[0] - 1), :, :]
    if arr.ndim == 4:
        # A common case for per-match scalar arrays with singleton channel.
        squeezed = np.squeeze(arr)
        if squeezed.ndim in (2, 3):
            return coerce_score_array(squeezed, match_index)
    return None


def coerce_orientation_rgb(output: Any) -> np.ndarray | None:
    """Coerce py4DSTEM plot_orientation_maps output to an RGB array if possible."""
    candidates: list[Any]
    if isinstance(output, tuple):
        candidates = list(output)
    else:
        candidates = [output]
    for obj in candidates:
        try:
            arr = np.asarray(obj)
        except Exception:
            continue
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            return arr[..., :3]
        if arr.ndim == 4:
            squeezed = np.squeeze(arr)
            if squeezed.ndim == 3 and squeezed.shape[-1] in (3, 4):
                return squeezed[..., :3]
            # py4DSTEM sometimes returns (Rx,Ry,3,nmatch)
            if arr.shape[2] in (3, 4):
                return arr[:, :, :3, 0]
    return None


def infer_scan_shape_from_orientation_map(orientation_map: Any, braggvectors: Any) -> tuple[int, int]:
    for obj in (orientation_map, braggvectors):
        for attr in ("shape", "Rshape", "scan_shape"):
            if hasattr(obj, attr):
                try:
                    shape = tuple(getattr(obj, attr))
                    if len(shape) >= 2:
                        return int(shape[0]), int(shape[1])
                except Exception:
                    pass
    raise RuntimeError("Could not infer scan shape from orientation map or Bragg vectors.")


def build_composite_phase_orientation_rgb(result: PhaseMatchingResult) -> np.ndarray | None:
    rgbs = [rgb for rgb in result.per_phase_rgb if rgb is not None]
    if not rgbs:
        return None
    shape = result.phase_id_map.shape
    composite = np.zeros(shape + (3,), dtype=np.float32)
    for i, rgb in enumerate(result.per_phase_rgb):
        if rgb is None:
            continue
        arr = np.asarray(rgb, dtype=np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
        if arr.shape[:2] != shape:
            continue
        mask = result.phase_id_map == i
        composite[mask] = arr[mask, :3]
    # Fade low confidence pixels.
    composite[result.low_confidence_mask] *= 0.35
    return np.clip(composite, 0, 1)


# -----------------------------------------------------------------------------
# Top-level orchestrator
# -----------------------------------------------------------------------------


class CrystalAnalysisWorkflow:
    def __init__(self, config: WorkflowConfig, callback: ProgressCallback | None = None):
        self.config = config
        self.callback = callback
        self.output_dir = ensure_output_dir(config)

    def run(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {"config": jsonify(asdict(self.config))}
        t0 = time.time()

        # Data and basic maps
        datacube = DataCubeLoader(self.config, self.callback).load()
        pre = DiffractionPreprocessor(self.config, self.output_dir, self.callback)
        mean_dp = pre.compute_mean_dp(datacube)
        center = pre.estimate_bf_center(mean_dp)
        masks = pre.build_detector_masks(mean_dp.shape, center)
        vbf, adf = pre.compute_virtual_images(datacube, *masks)
        comx, comy = pre.compute_com(datacube)
        metadata["basic_maps"] = {
            "bf_center_xy": center,
            "vbf_shape": tuple(vbf.shape),
            "adf_shape": tuple(adf.shape),
            "com_shape": tuple(comx.shape),
        }

        # Bragg disks and calibration
        bragg_detector = BraggDiskDetector(self.config, self.output_dir, self.callback)
        braggvectors = bragg_detector.detect(datacube, mean_dp)
        bvm_raw = bragg_detector.histogram(braggvectors, mode="raw")
        bvm_cal = bragg_detector.histogram(braggvectors, mode=self.config.bvm_classification.bvm_mode)
        metadata["bragg_detection"] = {"bvm_raw_shape": tuple(bvm_raw.shape), "bvm_cal_shape": tuple(bvm_cal.shape)}
        metadata["calibration"] = CalibrationManager(self.config, self.output_dir, self.callback).run(
            datacube, braggvectors, bvm_cal
        )

        # Data-driven BVM/Voronoi pre-classification
        bvm_result = BVMVoronoiClassifier(self.config, self.output_dir, self.callback).run(braggvectors, bvm_cal)
        metadata["bvm_classification"] = {
            k: jsonify(v)
            for k, v in bvm_result.items()
            if k not in {"maxima_qxy", "voronoi_label_image", "initial_classes", "feature_matrix"}
        }

        # Crystal analysis: CIF -> structure factors -> orientation libraries -> matching
        if not self.config.crystal.phase_cifs:
            progress(self.callback, "No CIF files provided; stopping before phase/orientation mapping.")
            metadata["crystal_analysis"] = "skipped_no_cif"
            self.save_metadata(metadata, t0)
            return metadata

        records = CIFManager(self.config, self.callback).load_crystals()
        records = StructureFactors(self.config, self.output_dir, self.callback).calculate(records)
        sim = SimulatedDiffraction(self.config, self.output_dir, self.callback)
        records = sim.build_orientation_libraries(records)
        sim.simulate_reference_patterns(records)
        records = OrientationMatching(self.config, self.output_dir, self.callback).match_each_phase(records, braggvectors)
        phase_result = PhaseMatching(self.config, self.output_dir, self.callback).match(records, braggvectors)
        metadata["phase_mapping"] = {
            "phase_names": phase_result.phase_names,
            "phase_fraction": phase_result.phase_fraction,
            "low_confidence_fraction": float(np.mean(phase_result.low_confidence_mask)),
        }
        metadata["grain_analysis"] = GrainAnalysis(self.config, self.output_dir, self.callback).run(records)
        metadata["strain_analysis"] = StrainAnalysis(self.config, self.output_dir, self.callback).run(
            records, braggvectors, phase_result
        )
        self.save_metadata(metadata, t0)
        return metadata

    def save_metadata(self, metadata: dict[str, Any], t0: float) -> None:
        metadata["elapsed_seconds"] = time.time() - t0
        with (self.output_dir / "workflow_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(jsonify(metadata), f, indent=2, ensure_ascii=False)
        progress(self.callback, f"Workflow metadata saved: {self.output_dir / 'workflow_metadata.json'}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="General 4D-STEM Bragg / phase / orientation workflow using py4DSTEM")
    p.add_argument("--input", required=True, help="Path to input 4D-STEM file: MIB, EMD/HDF5, py4DSTEM file, or .npy")
    p.add_argument("--input-type", default="auto", choices=["auto", "mib", "h5", "emd", "npy", "py4dstem"], help="Input reader hint. Use auto unless py4DSTEM cannot infer the file type.")
    p.add_argument("--datacube-path", default=None, help="Optional tree path/name for DataCube inside an EMD/HDF5/py4DSTEM file")
    p.add_argument("--scan", type=int, nargs=2, default=(512, 512), metavar=("RX", "RY"), help="Scan shape for raw MIB files, e.g. --scan 512 512")
    p.add_argument("--output-dir", default="py4dstem_general_output")
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--mem", default="MEMMAP")

    p.add_argument("--bf-radius", type=float, default=20.0)
    p.add_argument("--adf-inner", type=float, default=40.0)
    p.add_argument("--adf-outer", type=float, default=100.0)

    p.add_argument("--probe-radius", type=float, default=18.0)
    p.add_argument("--corr-power", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--edge-boundary", type=int, default=20)
    p.add_argument("--min-relative-intensity", type=float, default=0.05)
    p.add_argument("--min-absolute-intensity", type=float, default=None)
    p.add_argument("--min-peak-spacing", type=int, default=8)
    p.add_argument("--max-num-peaks", type=int, default=100)
    p.add_argument("--subpixel", default="poly")
    p.add_argument("--upsample-factor", type=int, default=16)
    p.add_argument("--external-probe-kernel", default=None)

    p.add_argument("--phase-cifs", nargs="*", default=[], help="Candidate CIF files")
    p.add_argument("--phase-names", nargs="*", default=[], help="Names corresponding to --phase-cifs")
    p.add_argument("--voltage", type=float, default=300000.0, help="Accelerating voltage in volts")
    p.add_argument("--k-max", type=float, default=1.5)
    p.add_argument("--angle-step", type=float, default=5.0)
    p.add_argument("--in-plane-step", type=float, default=5.0)
    p.add_argument("--zone-axis-range", default="auto")
    p.add_argument("--corr-kernel-size", type=float, default=0.08)
    p.add_argument("--radial-power", type=float, default=1.0)
    p.add_argument("--intensity-power", type=float, default=0.0)
    p.add_argument("--num-matches", type=int, default=2)
    p.add_argument("--min-number-peaks", type=int, default=3)
    p.add_argument("--low-confidence-threshold", type=float, default=0.05)

    p.add_argument("--bvm-maxima", type=int, default=128)
    p.add_argument("--bvm-maxima-min-distance", type=int, default=4)
    p.add_argument("--bvm-threshold-rel", type=float, default=0.04)
    p.add_argument("--bvm-voronoi-max-dist", type=float, default=8.0)
    p.add_argument("--bvm-nmf-components", type=int, default=None)

    p.add_argument("--cuda", action="store_true")
    p.add_argument("--run-strain", action="store_true")
    p.add_argument("--no-grain", action="store_true")
    p.add_argument("--no-bvm-classification", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--no-npy", action="store_true")
    p.add_argument("--no-pickle", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> WorkflowConfig:
    if args.phase_names and len(args.phase_names) != len(args.phase_cifs):
        raise ValueError("--phase-names must have the same number of entries as --phase-cifs")
    return WorkflowConfig(
        io=IOConfig(
            input_path=args.input,
            input_type=args.input_type,
            datacube_path=args.datacube_path,
            scan_shape=tuple(args.scan) if args.scan is not None else None,
            output_dir=args.output_dir,
            mem_mode=args.mem,
            chunk_size=args.chunk_size,
            save_figures=not args.no_figures,
            save_npy=not args.no_npy,
            save_pickle=not args.no_pickle,
        ),
        virtual=VirtualImagingConfig(
            bf_radius_px=args.bf_radius,
            adf_inner_px=args.adf_inner,
            adf_outer_px=args.adf_outer,
        ),
        bragg=BraggDetectionConfig(
            probe_radius_px=args.probe_radius,
            corr_power=args.corr_power,
            sigma=args.sigma,
            edge_boundary=args.edge_boundary,
            min_relative_intensity=args.min_relative_intensity,
            min_absolute_intensity=args.min_absolute_intensity,
            min_peak_spacing=args.min_peak_spacing,
            max_num_peaks=args.max_num_peaks,
            subpixel=args.subpixel,
            upsample_factor=args.upsample_factor,
            cuda=args.cuda,
            external_probe_kernel=args.external_probe_kernel,
        ),
        bvm_classification=BVMClassificationConfig(
            enabled=not args.no_bvm_classification,
            num_bvm_maxima=args.bvm_maxima,
            bvm_maxima_min_distance_px=args.bvm_maxima_min_distance,
            bvm_threshold_rel=args.bvm_threshold_rel,
            voronoi_max_dist_px=args.bvm_voronoi_max_dist,
            nmf_components=args.bvm_nmf_components,
        ),
        crystal=CrystalConfig(
            phase_cifs=args.phase_cifs,
            phase_names=args.phase_names,
            accelerating_voltage=args.voltage,
            k_max=args.k_max,
            zone_axis_range=args.zone_axis_range,
            angle_step_zone_axis=args.angle_step,
            angle_step_in_plane=args.in_plane_step,
            corr_kernel_size=args.corr_kernel_size,
            radial_power=args.radial_power,
            intensity_power=args.intensity_power,
            num_matches_return=args.num_matches,
            min_number_peaks=args.min_number_peaks,
            low_confidence_threshold=args.low_confidence_threshold,
            cuda=args.cuda,
        ),
        grain=GrainConfig(enabled=not args.no_grain),
        strain=StrainConfig(enabled=args.run_strain),
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    config = config_from_args(args)
    CrystalAnalysisWorkflow(config).run()


if __name__ == "__main__":
    main()
