"""
Optimized procedural multi-axis phase/orientation screening for py4DSTEM.
Version: v7 conservative Bragg QC + Ti-only phase/orientation screening, no display.

Design goals
------------
- Keep a linear, procedure-oriented workflow, close to the original tutorial script.
- Do NOT display figures interactively; save all QC/results to disk.
- Build a candidate phase map only from physically relevant Ti candidate phases.
- Run WS2.cif only as a negative/control phase and report control-failure masks.
- Keep multi-axis fiber orientation search, but expose ambiguity instead of forcing labels.
- Treat diffuse, non-zone-axis, mixed, or high-background patterns as screening data:
  conservative Bragg extraction, peak-count QC, top-candidate export, and ambiguity masks.
- Avoid saving py4DSTEM OrientationMap objects because some py4DSTEM versions cannot serialize them.

Interpretation
--------------
This script produces a screening phase/orientation map, not a confirmed phase map.
Do not treat the best-score phase as crystallographic proof unless: conservative
Bragg peaks are reliable, negative controls stay low, score margins are high, and
single-pattern QC overlays make physical sense. Mixed or diffuse patterns are marked
LOW_PEAK / MIXED / AMBIGUOUS instead of being forced into a Ti phase.
"""

from __future__ import annotations


def main(argv=None):
    from modules.reporting import generate_phase_orientation_report
    from pathlib import Path
    import contextlib
    import csv
    import hashlib
    import importlib
    import json
    import os
    import re
    import subprocess
    import sys
    import threading
    import time
    import warnings
    import numpy as np

    # Non-interactive backend: figures are saved to disk only; no GUI windows are opened.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show = lambda *args, **kwargs: None

    import argparse

    # Keep stdout unbuffered. Progress bars are left visible by default and forced
    # to ASCII after argument parsing to avoid Windows console mojibake.
    os.environ.pop("TQDM_ASCII", None)
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass


    status_log_path = None


    def log_status(message):
        nonlocal status_log_path
        line = f"[status] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        if status_log_path is not None:
            try:
                with open(status_log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass


    @contextlib.contextmanager
    def status_step(message, interval_seconds=30):
        start = time.time()
        log_status(f"{message} ...")
        stop_event = threading.Event()

        def heartbeat():
            while not stop_event.wait(max(1, int(interval_seconds))):
                elapsed = time.time() - start
                log_status(f"{message} still running ({elapsed / 60:.1f} min elapsed)")

        thread = None
        if interval_seconds and interval_seconds > 0:
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
        try:
            yield
        finally:
            stop_event.set()
            elapsed = time.time() - start
            log_status(f"{message} done in {elapsed:.1f}s")


    def configure_tqdm_progress_bars(quiet=False):
        """Force tqdm bars to ASCII; optionally hide them for log-only batch runs."""
        tqdm_classes = []
        for module_name in ("tqdm", "tqdm.auto", "tqdm.std", "tqdm.notebook"):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            cls = getattr(module, "tqdm", None)
            if cls is not None:
                tqdm_classes.append(cls)

        for cls in tqdm_classes:
            if getattr(cls, "_phase_screening_disable_patch", False):
                continue
            original_init = cls.__init__

            def configured_init(self, *args, __original_init=original_init, __quiet=quiet, **kwargs):
                kwargs.setdefault("ascii", True)
                if __quiet:
                    kwargs["disable"] = True
                return __original_init(self, *args, **kwargs)

            cls.__init__ = configured_init
            cls._phase_screening_disable_patch = True

        # emdfile.tqdmnd keeps a module-global reference named "tqdm"; make sure
        # it points at the patched tqdm class if emdfile is already importable.
        try:
            import emdfile
            import tqdm

            if hasattr(emdfile, "tqdmnd"):
                emdfile.tqdmnd.__globals__["tqdm"] = tqdm.tqdm
        except Exception:
            pass


    parser = argparse.ArgumentParser(
        description="Multi-axis Ti phase/orientation mapping with WS2 QC."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Input 4D-STEM HDF5 file, e.g. 1_0_64_0_64.h5",
    )
    parser.add_argument(
        "--mode",
        choices=["coarse", "fine"],
        default="coarse",
        help="Preset for staged screening. coarse is intended for full-map screening; fine is for ROI/refinement.",
    )
    parser.add_argument("--k-max", type=float, default=None, help="Maximum scattering vector for structure factors/matching.")
    parser.add_argument(
        "--k-max-sweep",
        type=str,
        default=None,
        help="Comma-separated K_MAX values, e.g. 1.4,1.7,2.0. Runs child jobs into separate output dirs.",
    )
    parser.add_argument("--inv-ang-per-pixel", type=float, default=None, help="Diffraction calibration in A^-1 per pixel.")
    parser.add_argument("--angle-step-zone-axis", type=float, default=None, help="py4DSTEM orientation_plan zone-axis step.")
    parser.add_argument("--angle-step-in-plane", type=float, default=None, help="py4DSTEM orientation_plan in-plane step.")
    parser.add_argument(
        "--orientation-mode",
        choices=["s2", "fiber"],
        default="fiber",
        help="Orientation library mode. s2 uses a full-sphere coarse library; fiber keeps legacy fixed zone-axis branches.",
    )
    parser.add_argument(
        "--num-matches-return",
        type=int,
        default=5,
        help="Number of py4DSTEM orientation matches to retain per phase/branch.",
    )
    parser.add_argument(
        "--allow-screening-fallback",
        action="store_true",
        help="Continue in screening mode when optional calibration steps cannot be estimated.",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=0,
        help="Seconds between heartbeat messages during long py4DSTEM operations. Use 0 to disable.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Hide py4DSTEM/tqdm progress bars and rely on status messages/logs only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths, create the output/status log, print run settings, and exit before loading py4DSTEM.",
    )
    parser.add_argument("--margin-threshold", type=float, default=0.20, help="Best-real minus second-real score threshold.")
    parser.add_argument("--max-clean-peaks-for-single", type=int, default=50, help="Diffuse/mixed peak-count threshold.")
    parser.add_argument("--min-strong-peaks-for-match", type=int, default=6, help="Minimum strong QC peaks required for orientation matching/QC.")
    parser.add_argument("--peak-count-threshold", type=int, default=None, help="Low-peak QC threshold. Defaults to --min-strong-peaks-for-match.")
    parser.add_argument("--q-min-for-qc", type=float, default=0.12, help="Minimum q in A^-1 for clean peak QC.")
    parser.add_argument("--q-max-for-qc", type=float, default=1.4, help="Maximum q in A^-1 for clean peak QC.")
    parser.add_argument("--direct-beam-mask-radius", type=float, default=None, help="Direct-beam mask radius in pixels for diagnostic preprocessing.")
    parser.add_argument("--detect-min-relative-intensity", type=float, default=0.03, help="py4DSTEM Bragg detection minRelativeIntensity.")
    parser.add_argument("--detect-min-absolute-intensity", type=float, default=0.0, help="py4DSTEM Bragg detection minAbsoluteIntensity.")
    parser.add_argument("--detect-min-peak-spacing", type=int, default=8, help="py4DSTEM Bragg detection minPeakSpacing in pixels.")
    parser.add_argument("--detect-max-num-peaks", type=int, default=80, help="py4DSTEM Bragg detection maxNumPeaks.")
    parser.add_argument("--match-radius-q", type=float, default=0.08, help="Orientation matching correlation kernel size in A^-1.")
    parser.add_argument("--run-control", dest="run_control", action="store_true", help="Run WS2 negative-control branch(es).")
    parser.add_argument("--skip-control", dest="run_control", action="store_false", help="Skip WS2 negative-control branch(es).")
    parser.set_defaults(run_control=True)
    parser.add_argument("--force-recompute-bragg", action="store_true", help="Ignore cached Bragg peaks and recompute.")
    parser.add_argument(
        "--calibration-peaks",
        type=Path,
        default=None,
        help="JSON/CSV peak table with q_pixel and q_A^-1/known_q_A^-1 columns for calibration fit.",
    )
    parser.add_argument("--branch-only", action="store_true", help="Run a single phase/fiber-axis branch and save branch score outputs.")
    parser.add_argument("--phase", type=str, default=None, help="Phase name for --branch-only, e.g. Ti-bcc or WS2-control.")
    parser.add_argument("--fiber-axis", type=str, default=None, help="Fiber axis for --branch-only, e.g. 0,1,1.")
    parser.add_argument(
        "--aggregate-branches",
        type=Path,
        default=None,
        help="Directory containing score_branch_*.npy and metadata_branch_*.json files to aggregate.",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help="Optional run-output subdirectory name. Defaults to a tag made from mode/K_MAX/steps/control.",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=None,
        help=(
            "Base directory for run outputs and Bragg caches. Defaults to the "
            "nearest data-file parent containing Ti CIFs, or the data-file parent "
            "for absolute --data-file values."
        ),
    )
    parser.add_argument(
        "--cif-dir",
        type=Path,
        default=None,
        help="Directory containing Ti-bcc.cif, Ti-hcp.cif, and optional WS2.cif.",
    )
    # =============================================================================
    # 0. User configuration
    # =============================================================================

    args = parser.parse_args(argv)
    if args.quiet_progress:
        os.environ.setdefault("TQDM_DISABLE", "1")
    else:
        os.environ.pop("TQDM_DISABLE", None)
    configure_tqdm_progress_bars(quiet=args.quiet_progress)
    DEFAULT_ROOT = Path(r"D:\Workspace\large-4dstem-analysis\data\0617-4d")
    REQUIRED_REAL_CIFS = ("Ti-bcc.cif", "Ti-hcp.cif")


    def find_nearest_cif_dir(start_path):
        """Find the closest parent that contains the required real Ti CIFs."""
        start_path = Path(start_path).resolve()
        candidates = [start_path]
        if start_path.is_file() or start_path.suffix:
            candidates = [start_path.parent]
        candidates.extend(candidates[0].parents)
        for candidate in candidates:
            if all((candidate / name).exists() for name in REQUIRED_REAL_CIFS):
                return candidate
        return None


    data_arg = Path(args.data_file)
    if data_arg.is_absolute():
        DATA_FILE = data_arg
        inferred_cif_dir = find_nearest_cif_dir(DATA_FILE)
        ROOT = Path(args.analysis_root) if args.analysis_root is not None else (inferred_cif_dir or DATA_FILE.parent)
    else:
        ROOT = Path(args.analysis_root) if args.analysis_root is not None else DEFAULT_ROOT
        DATA_FILE = ROOT / data_arg
        inferred_cif_dir = find_nearest_cif_dir(DATA_FILE) or find_nearest_cif_dir(ROOT)

    CIF_DIR = Path(args.cif_dir) if args.cif_dir is not None else (inferred_cif_dir or ROOT)


    def sanitize_tag(value):
        return str(value).replace(".", "p").replace("-", "m").replace(",", "_")


    def parse_float_list(text):
        return [float(v.strip()) for v in str(text).split(",") if v.strip()]


    def parse_axis(text):
        if text is None:
            return None
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError("--fiber-axis must have exactly 3 comma-separated values, e.g. 0,1,1")
        return [int(p) for p in parts]


    def mode_defaults(mode):
        if mode == "fine":
            return {"k_max": 1.7, "angle_step_zone_axis": 1.0, "angle_step_in_plane": 2.0}
        return {"k_max": 1.4, "angle_step_zone_axis": 2.0, "angle_step_in_plane": 8.0}


    defaults = mode_defaults(args.mode)
    K_MAX = float(args.k_max if args.k_max is not None else defaults["k_max"])
    INV_ANG_PER_PIXEL = float(args.inv_ang_per_pixel if args.inv_ang_per_pixel is not None else 0.0192)
    ANGLE_STEP_ZONE_AXIS = float(args.angle_step_zone_axis if args.angle_step_zone_axis is not None else defaults["angle_step_zone_axis"])
    ANGLE_STEP_IN_PLANE = float(args.angle_step_in_plane if args.angle_step_in_plane is not None else defaults["angle_step_in_plane"])
    ORIENTATION_MODE = args.orientation_mode
    NUM_MATCHES_RETURN = max(1, int(args.num_matches_return))
    STATUS_INTERVAL = max(0, int(args.status_interval))
    MARGIN_THRESHOLD = float(args.margin_threshold)
    MAX_CLEAN_PEAKS_FOR_SINGLE = int(args.max_clean_peaks_for_single)

    if args.output_tag is None:
        control_tag = "control" if args.run_control else "no_control"
        args.output_tag = (
            f"{args.mode}_k{sanitize_tag(K_MAX)}_za{sanitize_tag(ANGLE_STEP_ZONE_AXIS)}_"
            f"ip{sanitize_tag(ANGLE_STEP_IN_PLANE)}_{control_tag}"
        )

    OUT_DIR = ROOT / DATA_FILE.stem / args.output_tag
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status_log_path = OUT_DIR / "run_status.log"
    with open(status_log_path, "w", encoding="utf-8") as f:
        f.write(f"[status] {time.strftime('%Y-%m-%d %H:%M:%S')} Status log created\n")
        f.write(f"[status] data_file={DATA_FILE}\n")
        f.write(f"[status] out_dir={OUT_DIR}\n")
        f.write(f"[status] command={' '.join([str(x) for x in (argv if argv is not None else sys.argv[1:])])}\n")
    log_status(f"Status log: {status_log_path}")
    log_status(f"Output directory: {OUT_DIR}")
    if args.dry_run:
        log_status("Dry run requested; resolved settings and exiting before py4DSTEM import.")
        print(f"Data file: {DATA_FILE}", flush=True)
        print(f"Analysis root: {ROOT}", flush=True)
        print(f"CIF directory: {CIF_DIR}", flush=True)
        print(f"Output directory: {OUT_DIR}", flush=True)
        print(f"Status log: {status_log_path}", flush=True)
        return 0

    # -----------------------------------------------------------------------------
    # Real candidate phases: these participate in the final phase map.
    # -----------------------------------------------------------------------------
    REAL_CANDIDATE_PHASES = [
        {
            "name": "Ti-bcc",
            "cif": CIF_DIR / "Ti-bcc.cif",
            "symmetry_order": 4,
            "zone_axis_range": "fiber",
            "fiber_axes": [
                [0, 1, 1],  # strongest bcc branch in your previous test
                [0, 0, 1],
                [1, 1, 1],
            ],
            "fiber_angles": [0, 360],
        },
        {
            "name": "Ti-hcp",
            "cif": CIF_DIR / "Ti-hcp.cif",
            "symmetry_order": 6,
            "zone_axis_range": "fiber",
            "fiber_axes": [
                [1, 0, 0],  # strongest hcp branch in your previous test
                [0, 0, 1],
                [1, 1, 0],
            ],
            "fiber_angles": [0, 360],
        },
    ]

    # -----------------------------------------------------------------------------
    # Control phases: these do NOT participate in the final phase map.
    # They are used only to test whether a physically unrelated phase can spuriously
    # beat the real candidates.
    # -----------------------------------------------------------------------------
    CONTROL_PHASES = [
        {
            "name": "WS2-control",
            "cif": CIF_DIR / "WS2.cif",
            "symmetry_order": 6,
            "zone_axis_range": "fiber",
            # Keep the control conservative. Your previous test showed WS2 [100]
            # can spuriously win; do not include it in the default control map.
            "fiber_axes": [
                [0, 0, 1],
            ],
            "fiber_angles": [0, 360],
        },
    ]

    # Bragg disk detection parameters. If you change these, set USE_CACHED_BRAGG_PEAKS=False.
    DETECT_PARAMS = {
        "corrPower": 1,
        "sigma": 0,
        "edgeBoundary": 8,
        "minRelativeIntensity": float(args.detect_min_relative_intensity),
        "minAbsoluteIntensity": float(args.detect_min_absolute_intensity),
        "minPeakSpacing": int(args.detect_min_peak_spacing),
        "subpixel": "poly",
        "upsample_factor": 8,
        "maxNumPeaks": int(args.detect_max_num_peaks),
        # "CUDA": True,
    }

    # Peak-set QC for diffuse/non-zone-axis patterns. Matching still uses py4DSTEM's
    # BraggVectors, while these masks keep the final map in screening mode.
    Q_MIN_FOR_QC = float(args.q_min_for_qc)  # A^-1; excludes central beam / central diffuse region
    Q_MAX_FOR_QC = float(args.q_max_for_qc)  # A^-1; excludes outer noisy detections
    STRONG_PEAK_PERCENTILE = 70      # per-pattern intensity percentile for strong peaks
    MIN_STRONG_PEAKS_FOR_MATCH = int(args.min_strong_peaks_for_match)
    TOP_CANDIDATES_TO_SAVE = 5
    MATCH_RADIUS_Q = float(args.match_radius_q)

    # Diffraction calibration / matching settings.
    YMAX_RADIAL_PROFILE = 30

    # Orientation-search resolution is supplied by CLI/mode presets.

    # QC test pixels.
    TEST_RXS = (0, 3, 5)
    TEST_RYS = (0, 3, 5)
    SINGLE_TEST_PIXEL = (3, 3)

    # Confidence screening. Thresholds are empirical on this score scale.
    MIN_BEST_SCORE = 0.0             # optionally set to e.g. 0.5 or 1.0
    PEAK_COUNT_THRESHOLD = int(args.peak_count_threshold) if args.peak_count_threshold is not None else MIN_STRONG_PEAKS_FOR_MATCH
    CONTROL_FAIL_MARGIN = 0.0        # control score > real best score + this value => control failure

    # Runtime behavior.
    SKIP_MISSING_CIFS = True
    USE_CACHED_BRAGG_PEAKS = not args.force_recompute_bragg
    BRAGG_CACHE_TAG = "conservative_v7"
    RUN_STRAIN_FOR_GLOBALLY_BEST_REAL_BRANCH = False
    if args.k_max_sweep:
        sweep_values = parse_float_list(args.k_max_sweep)
        sweep_results = []
        for k_value in sweep_values:
            child_tag = f"sweep_k{sanitize_tag(k_value)}"
            child_cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent.parent / "main.py"),
                "--data-file",
                str(args.data_file),
                "--mode",
                args.mode,
                "--k-max",
                str(k_value),
                "--inv-ang-per-pixel",
                str(INV_ANG_PER_PIXEL),
                "--angle-step-zone-axis",
                str(ANGLE_STEP_ZONE_AXIS),
                "--angle-step-in-plane",
                str(ANGLE_STEP_IN_PLANE),
                "--orientation-mode",
                ORIENTATION_MODE,
                "--num-matches-return",
                str(NUM_MATCHES_RETURN),
                "--margin-threshold",
                str(MARGIN_THRESHOLD),
                "--max-clean-peaks-for-single",
                str(MAX_CLEAN_PEAKS_FOR_SINGLE),
                "--min-strong-peaks-for-match",
                str(MIN_STRONG_PEAKS_FOR_MATCH),
                "--peak-count-threshold",
                str(PEAK_COUNT_THRESHOLD),
                "--q-min-for-qc",
                str(Q_MIN_FOR_QC),
                "--q-max-for-qc",
                str(Q_MAX_FOR_QC),
                "--detect-min-relative-intensity",
                str(DETECT_PARAMS["minRelativeIntensity"]),
                "--detect-min-absolute-intensity",
                str(DETECT_PARAMS["minAbsoluteIntensity"]),
                "--detect-min-peak-spacing",
                str(DETECT_PARAMS["minPeakSpacing"]),
                "--detect-max-num-peaks",
                str(DETECT_PARAMS["maxNumPeaks"]),
                "--match-radius-q",
                str(MATCH_RADIUS_Q),
                "--output-tag",
                child_tag,
            ]
            if args.direct_beam_mask_radius is not None:
                child_cmd.extend(["--direct-beam-mask-radius", str(args.direct_beam_mask_radius)])
            if args.quiet_progress:
                child_cmd.append("--quiet-progress")
            if STATUS_INTERVAL:
                child_cmd.extend(["--status-interval", str(STATUS_INTERVAL)])
            child_cmd.extend(["--analysis-root", str(ROOT)])
            child_cmd.extend(["--cif-dir", str(CIF_DIR)])
            child_cmd.append("--run-control" if args.run_control else "--skip-control")
            if args.force_recompute_bragg:
                child_cmd.append("--force-recompute-bragg")
            if args.allow_screening_fallback:
                child_cmd.append("--allow-screening-fallback")
            if args.calibration_peaks is not None:
                child_cmd.extend(["--calibration-peaks", str(args.calibration_peaks)])
            print(f"[sweep] Running K_MAX={k_value}: {' '.join(child_cmd)}")
            result = subprocess.run(child_cmd, cwd=Path(__file__).resolve().parent.parent)
            summary_path = ROOT / DATA_FILE.stem / child_tag / "phase_summary_v6_optimized.json"
            item = {"k_max": k_value, "returncode": int(result.returncode), "summary_path": str(summary_path)}
            if summary_path.exists():
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_obj = json.load(f)
                item.update(summary_obj.get("confidence_summary", {}))
                item["real_phase_results"] = summary_obj.get("real_phase_results_aggregated_over_axes", [])
                item["control_phase_results"] = summary_obj.get("control_phase_results_aggregated_over_axes", [])
            sweep_results.append(item)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
        save_path = OUT_DIR / "kmax_sweep_summary.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "data_file": str(DATA_FILE),
                    "mode": args.mode,
                    "screening_mode": "fiber_axis_only",
                    "k_max_values": sweep_values,
                    "results": sweep_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
        )
        print(f"[saved] {save_path}")
        generate_phase_orientation_report(
            OUT_DIR,
            summary_path=save_path,
            title="K_MAX Sweep Phase/Orientation Screening Report",
        )
        raise SystemExit(0)

    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive.*")

    def disable_interactive_matplotlib_show():
        import matplotlib.figure
        import matplotlib.backends.backend_agg

        matplotlib.figure.Figure.show = lambda self, *args, **kwargs: None
        matplotlib.backends.backend_agg.FigureCanvasAgg.show = (
            lambda self, *args, **kwargs: None
        )


    disable_interactive_matplotlib_show()


    # =============================================================================
    # 1. Small procedural helpers
    # =============================================================================


    def savefig(filename, **kwargs):
        from pathlib import Path
        import matplotlib.pyplot as plt

        filename = Path(filename)

        if not filename.is_absolute():
            filename = OUT_DIR / filename

        filename.parent.mkdir(parents=True, exist_ok=True)

        if "dpi" not in kwargs:
            kwargs["dpi"] = 200

        # Only set default bbox_inches when user does not provide it.
        if "bbox_inches" not in kwargs:
            kwargs["bbox_inches"] = "tight"

        plt.savefig(filename, **kwargs)
        plt.close()

        print(f"[saved] {filename}")
        return filename

    def axis_to_tag(axis):
        return "za_" + "_".join(str(v).replace("-", "m") for v in axis)


    def normalize_score(score):
        score = np.asarray(score, dtype=np.float32)
        out = np.zeros_like(score, dtype=np.float32)
        finite = np.isfinite(score)
        if np.any(finite):
            lo = np.nanmin(score[finite])
            hi = np.nanmax(score[finite])
            denom = hi - lo
            if np.isfinite(denom) and denom > 0:
                np.divide(score - lo, denom, out=out, where=finite)
                out = np.clip(out, 0, 1)
        return out


    def finite_stat(arr, fn):
        vals = np.asarray(arr, dtype=np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return float(fn(vals))


    def get_orientation_score_array(orientation_map):
        """Extract best-match score map from py4DSTEM OrientationMap across versions."""
        return get_orientation_score_stack(orientation_map)[:, :, 0]


    def get_orientation_score_stack(orientation_map):
        """Extract score stack from py4DSTEM OrientationMap across versions."""
        for attr in ["corr", "correlation", "corrs", "intensity", "intensities"]:
            if hasattr(orientation_map, attr):
                arr = np.asarray(getattr(orientation_map, attr))
                if arr.ndim == 3:
                    return arr.astype(np.float32)
                if arr.ndim == 2:
                    return arr[:, :, None].astype(np.float32)
        print("[warning] Could not find score array on orientation_map. Public attributes:")
        print([a for a in dir(orientation_map) if not a.startswith("_")])
        raise AttributeError("Update get_orientation_score_array() for this py4DSTEM version.")


    def plot_scalar_map(arr, title, filename, cmap="viridis"):
        plt.figure(figsize=(5, 4))
        im = plt.imshow(np.asarray(arr).T, origin="lower", cmap=cmap, interpolation="nearest")
        plt.title(title)
        plt.xlabel("scan x")
        plt.ylabel("scan y")
        plt.colorbar(im, shrink=0.8)
        savefig(filename)


    def plot_histogram(arr, title, filename, bins=80):
        vals = np.asarray(arr).ravel()
        vals = vals[np.isfinite(vals)]
        plt.figure(figsize=(5, 4))
        if vals.size:
            plt.hist(vals, bins=bins)
        plt.title(title)
        plt.xlabel(title)
        plt.ylabel("count")
        savefig(filename)


    def plot_index_map(index_map, labels, title, filename, save_clean=False):
        from pathlib import Path
        import numpy as np
        import matplotlib.pyplot as plt

        arr = np.asarray(index_map, dtype=np.int32)

        vmin = 0
        vmax = max(len(labels) - 1, 0)

        # Use original matplotlib default colormap
        cmap = plt.rcParams["image.cmap"]

        # ------------------------------------------------------------------
        # 1. Normal annotated version
        # ------------------------------------------------------------------
        plt.figure(figsize=(5, 4))

        im = plt.imshow(
            arr.T,
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

        plt.title(title)
        plt.xlabel("scan x")
        plt.ylabel("scan y")

        cbar = plt.colorbar(im, ticks=np.arange(len(labels)))
        cbar.ax.set_yticklabels(labels)

        savefig(filename)

        # ------------------------------------------------------------------
        # 2. Clean version: 4x4, no title / axes / colorbar / white border
        # ------------------------------------------------------------------
        if save_clean:
            filename = Path(filename)
            clean_filename = filename.with_name(
                filename.stem + "_clean" + filename.suffix
            )

            fig = plt.figure(figsize=(4, 4), frameon=False)
            ax = fig.add_axes([0, 0, 1, 1], frameon=False)

            ax.imshow(
                arr.T,
                origin="lower",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                aspect="auto",   # important: prevents internal white padding
            )

            ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])

            # Important:
            # bbox_inches=None, not "tight"
            savefig(
                clean_filename,
                dpi=300,
                bbox_inches=None,
                pad_inches=0,
                facecolor="none",
                edgecolor="none",
            )

            print(f"[saved clean] {clean_filename}")


    def plot_phase_map_with_masks(best_phase_index, phase_names, invalid_mask, mixed_mask, ambiguous_mask, control_fail_mask, filename, no_valid_mask=None):
        """Phase map labels: real phases + NO_VALID_MATCH + LOW_PEAK + MIXED + AMBIGUOUS + CONTROL_FAIL."""
        display = np.asarray(best_phase_index, dtype=np.int32).copy()
        labels = list(phase_names)
        no_valid_label = len(labels); labels.append("NO_VALID_MATCH")
        low_label = len(labels); labels.append("LOW_PEAK / WEAK")
        mixed_label = len(labels); labels.append("MIXED / DIFFUSE")
        amb_label = len(labels); labels.append("AMBIGUOUS")
        ctrl_label = len(labels); labels.append("CONTROL_FAIL")

        if no_valid_mask is None:
            no_valid_mask = np.zeros_like(display, dtype=bool)

        # Priority: no valid match > control failure > low peak/weak > mixed/diffuse > ambiguous > phase.
        display[ambiguous_mask] = amb_label
        display[mixed_mask] = mixed_label
        display[invalid_mask] = low_label
        display[control_fail_mask] = ctrl_label
        display[no_valid_mask] = no_valid_label
        plot_index_map(display, labels, "QC-masked real phase map", filename)


    def radial_background_subtract(image, center):
        """Subtract median radial background around the direct beam center."""
        arr = np.asarray(image, dtype=np.float32)
        yy, xx = np.indices(arr.shape)
        radius = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
        bins = np.floor(radius).astype(np.int32)
        flat_bins = bins.ravel()
        flat_arr = arr.ravel()
        background = np.zeros(int(flat_bins.max()) + 1, dtype=np.float32)
        for idx in range(background.size):
            vals = flat_arr[flat_bins == idx]
            background[idx] = np.nanmedian(vals) if vals.size else 0.0
        corrected = arr - background[bins]
        corrected -= np.nanmin(corrected)
        return corrected


    def direct_beam_mask(shape, center, radius):
        yy, xx = np.indices(shape)
        return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius ** 2


    def compress_image(image, mode="log", gamma=0.35):
        arr = np.asarray(image, dtype=np.float32)
        arr = arr - np.nanmin(arr)
        hi = np.nanpercentile(arr, 99.7)
        if hi > 0:
            arr = np.clip(arr / hi, 0, 1)
        if mode == "gamma":
            arr = np.power(arr, gamma)
        elif mode == "log":
            arr = np.log1p(20 * arr) / np.log1p(20)
        return arr.astype(np.float32)


    def preprocess_diffraction_image(image, center, beam_radius):
        raw = np.asarray(image, dtype=np.float32)
        processed = radial_background_subtract(raw, center)
        mask = direct_beam_mask(raw.shape, center, beam_radius)
        processed = processed.copy()
        processed[mask] = 0
        return {
            "raw": raw,
            "processed": processed,
            "log": compress_image(processed, mode="log"),
            "gamma": compress_image(processed, mode="gamma"),
            "direct_beam_mask": mask,
        }


    def simple_local_peak_detection(image, center, beam_radius, max_peaks=80, min_spacing=8, threshold_percentile=99.0):
        """Small QC-only peak picker for raw/processed overlay diagnostics."""
        arr = np.asarray(image, dtype=np.float32)
        work = arr.copy()
        work[direct_beam_mask(work.shape, center, beam_radius)] = 0
        threshold = np.nanpercentile(work, threshold_percentile)
        candidates = []
        for x in range(1, work.shape[1] - 1):
            for y in range(1, work.shape[0] - 1):
                val = work[y, x]
                if val < threshold:
                    continue
                patch = work[y - 1 : y + 2, x - 1 : x + 2]
                if val >= np.nanmax(patch):
                    candidates.append((float(val), float(x), float(y)))
        candidates.sort(reverse=True)
        selected = []
        for val, x, y in candidates:
            if all((x - px) ** 2 + (y - py) ** 2 >= min_spacing ** 2 for _, px, py in selected):
                selected.append((val, x, y))
            if len(selected) >= max_peaks:
                break
        return selected


    def save_diffraction_qc_images(label, image, center, beam_radius):
        """Save raw/processed/compressed/overlay diagnostics and return metrics."""
        qc = preprocess_diffraction_image(image, center, beam_radius)
        peaks = simple_local_peak_detection(
            qc["processed"],
            center=center,
            beam_radius=beam_radius,
            max_peaks=DETECT_PARAMS["maxNumPeaks"],
            min_spacing=DETECT_PARAMS["minPeakSpacing"],
        )
        for key, arr in [("raw", qc["raw"]), ("processed", qc["processed"]), ("log", qc["log"]), ("gamma", qc["gamma"])]:
            plt.figure(figsize=(5, 5))
            plt.imshow(arr, cmap="gray", origin="lower")
            plt.title(f"{label}: {key}")
            plt.axis("off")
            savefig(f"qc_{label}_{key}.png")

        plt.figure(figsize=(5, 5))
        plt.imshow(qc["log"], cmap="gray", origin="lower")
        if peaks:
            xs = [p[1] for p in peaks]
            ys = [p[2] for p in peaks]
            plt.scatter(xs, ys, facecolors="none", edgecolors="lime", s=45, linewidths=1)
        circle = plt.Circle(center, beam_radius, color="red", fill=False, linewidth=1)
        plt.gca().add_patch(circle)
        plt.title(f"{label}: processed peak overlay, n={len(peaks)}")
        plt.axis("off")
        savefig(f"qc_{label}_detected_peak_overlay.png")
        return {
            "label": label,
            "peak_count": int(len(peaks)),
            "direct_beam_radius_px": float(beam_radius),
            "background": "median_radial_subtraction",
            "compression": ["log", "gamma"],
        }


    def compute_peak_count_map(bragg_peaks):
        """Return detected Bragg peak count per probe position, defensively across versions."""
        # Try calibration wrapper first because downstream uses bragg_peaks.cal[x,y].
        cal = getattr(bragg_peaks, "cal", None)
        # Determine scan shape.
        shape_candidates = []
        for attr in ["Rshape", "shape"]:
            if hasattr(bragg_peaks, attr):
                val = getattr(bragg_peaks, attr)
                try:
                    if len(val) >= 2:
                        shape_candidates.append((int(val[0]), int(val[1])))
                except Exception:
                    pass
        if hasattr(bragg_peaks, "_shape"):
            try:
                val = bragg_peaks._shape
                shape_candidates.append((int(val[0]), int(val[1])))
            except Exception:
                pass
        if not shape_candidates:
            # Last resort: infer by using score maps later; return None.
            print("[warning] Could not infer bragg_peaks scan shape for peak count map.")
            return None
        rx, ry = shape_candidates[0]

        counts = np.zeros((rx, ry), dtype=np.int16)
        source = cal if cal is not None else bragg_peaks
        for i in range(rx):
            for j in range(ry):
                try:
                    pl = source[i, j]
                    if hasattr(pl, "data") and "qx" in pl.data.dtype.names:
                        counts[i, j] = len(pl.data["qx"])
                    elif hasattr(pl, "data"):
                        counts[i, j] = len(pl.data)
                    else:
                        counts[i, j] = len(pl)
                except Exception:
                    counts[i, j] = 0
        return counts


    def peaklist_arrays(peaklist):
        """Extract qx, qy, intensity arrays from a py4DSTEM point list across versions."""
        if not hasattr(peaklist, "data"):
            return None, None, None

        data = peaklist.data
        names = getattr(data.dtype, "names", None) or ()
        if "qx" not in names or "qy" not in names:
            return None, None, None

        qx = np.asarray(data["qx"], dtype=np.float32)
        qy = np.asarray(data["qy"], dtype=np.float32)

        intensity = None
        for attr in ("intensity", "intensities", "I", "amplitude", "corr"):
            if attr in names:
                intensity = np.asarray(data[attr], dtype=np.float32)
                break
        if intensity is None:
            intensity = np.ones_like(qx, dtype=np.float32)

        return qx, qy, intensity


    def compute_peak_qc_maps(bragg_peaks):
        """
        Build conservative peak-set QC maps.

        The calibrated BraggVectors are left untouched for py4DSTEM matching. These
        maps decide whether a pixel is suitable for confident phase/orientation calls.
        """
        cal = getattr(bragg_peaks, "cal", None)
        source = cal if cal is not None else bragg_peaks

        peak_count = compute_peak_count_map(bragg_peaks)
        if peak_count is None:
            return None

        rx, ry = peak_count.shape
        clean_count = np.zeros((rx, ry), dtype=np.int16)
        strong_count = np.zeros((rx, ry), dtype=np.int16)
        q_median = np.full((rx, ry), np.nan, dtype=np.float32)
        q_p90 = np.full((rx, ry), np.nan, dtype=np.float32)

        for i in range(rx):
            for j in range(ry):
                try:
                    qx, qy, intensity = peaklist_arrays(source[i, j])
                    if qx is None or qx.size == 0:
                        continue

                    q = np.sqrt(qx * qx + qy * qy)
                    clean = np.isfinite(q) & (q >= Q_MIN_FOR_QC) & (q <= Q_MAX_FOR_QC)
                    clean_count[i, j] = int(np.sum(clean))
                    if np.any(clean):
                        q_clean = q[clean]
                        i_clean = intensity[clean]
                        q_median[i, j] = float(np.nanmedian(q_clean))
                        q_p90[i, j] = float(np.nanpercentile(q_clean, 90))
                        threshold = np.nanpercentile(i_clean, STRONG_PEAK_PERCENTILE)
                        strong_count[i, j] = int(np.sum(i_clean >= threshold))
                except Exception:
                    continue

        return {
            "raw_peak_count": peak_count,
            "clean_peak_count": clean_count,
            "strong_peak_count": strong_count,
            "q_median": q_median,
            "q_p90": q_p90,
        }


    def plot_peak_radius_histogram_for_pixel(bragg_peaks, xind, yind, filename):
        """QC plot: q-radius histogram for one probe position."""
        source = getattr(bragg_peaks, "cal", None) or bragg_peaks
        qx, qy, intensity = peaklist_arrays(source[xind, yind])

        plt.figure(figsize=(5, 4))
        if qx is not None and qx.size:
            q = np.sqrt(qx * qx + qy * qy)
            clean = np.isfinite(q) & (q >= Q_MIN_FOR_QC) & (q <= Q_MAX_FOR_QC)
            weights = intensity if intensity is not None else None
            plt.hist(q[clean], bins=40, weights=None if weights is None else weights[clean])
            plt.axvline(Q_MIN_FOR_QC, color="r", linestyle="--", linewidth=1, label="q_min")
            plt.axvline(Q_MAX_FOR_QC, color="r", linestyle=":", linewidth=1, label="q_max")
            plt.legend()
        plt.title(f"Peak q-radius histogram at ({xind}, {yind})")
        plt.xlabel("q (1/Ang)")
        plt.ylabel("weighted count")
        savefig(filename)


    def select_representative_test_pixels(clean_peak_count, fallback, max_points=5):
        """Pick high-quality, spatially separated pixels for single-pattern diagnostics."""
        fallback = tuple(int(v) for v in fallback)
        if clean_peak_count is None or np.size(clean_peak_count) == 0:
            return [fallback]

        arr = np.asarray(clean_peak_count)
        if arr.ndim != 2:
            return [fallback]

        coords = np.argwhere(np.isfinite(arr) & (arr > 0))
        if coords.size == 0:
            return [fallback]

        values = arr[coords[:, 0], coords[:, 1]]
        order = np.argsort(values)[::-1]
        min_spacing = max(1, min(arr.shape) // 4)
        selected = []
        for idx in order:
            xind, yind = (int(coords[idx, 0]), int(coords[idx, 1]))
            if all((xind - sx) ** 2 + (yind - sy) ** 2 >= min_spacing ** 2 for sx, sy in selected):
                selected.append((xind, yind))
            if len(selected) >= max_points:
                break

        if not selected:
            selected = [fallback]
        return selected


    def summarize_single_pattern_match_inputs(bragg_peaks, xind, yind, template_q_after):
        """Record enough q-space information to explain a failed single-pattern match."""
        source = getattr(bragg_peaks, "cal", None) or bragg_peaks
        qx = qy = intensity = None
        try:
            qx, qy, intensity = peaklist_arrays(source[xind, yind])
        except Exception:
            pass

        if qx is None:
            q = np.array([], dtype=np.float32)
            intensity = np.array([], dtype=np.float32)
        else:
            q = np.sqrt(qx * qx + qy * qy)
            intensity = np.ones_like(q, dtype=np.float32) if intensity is None else intensity

        finite_q = np.isfinite(q)
        clean = finite_q & (q >= Q_MIN_FOR_QC) & (q <= Q_MAX_FOR_QC)
        q_clean = q[clean]
        template_q_after = np.asarray(template_q_after, dtype=np.float64)
        template_q_after = template_q_after[np.isfinite(template_q_after)]

        nearest = np.array([], dtype=np.float64)
        matched_peak_count = 0
        if q_clean.size and template_q_after.size:
            nearest = np.min(np.abs(q_clean[:, None] - template_q_after[None, :]), axis=1)
            matched_peak_count = int(np.sum(nearest <= MATCH_RADIUS_Q))

        clean_map_value = None
        strong_map_value = None
        if np.size(clean_peak_count_map) and xind < clean_peak_count_map.shape[0] and yind < clean_peak_count_map.shape[1]:
            clean_map_value = int(clean_peak_count_map[xind, yind])
        if np.size(strong_peak_count_map) and xind < strong_peak_count_map.shape[0] and yind < strong_peak_count_map.shape[1]:
            strong_map_value = int(strong_peak_count_map[xind, yind])

        denominator = max(int(q_clean.size), int(template_q_after.size), 1)
        return {
            "test_pixel": [int(xind), int(yind)],
            "n_exp_peaks_test": int(q.size),
            "n_clean_peaks_test": int(q_clean.size),
            "n_clean_peaks_test_map": clean_map_value,
            "n_strong_peaks_test_map": strong_map_value,
            "n_template_reflections": int(template_q_after.size),
            "exp_q_min": None if q_clean.size == 0 else float(np.nanmin(q_clean)),
            "exp_q_max": None if q_clean.size == 0 else float(np.nanmax(q_clean)),
            "template_q_min": None if template_q_after.size == 0 else float(np.nanmin(template_q_after)),
            "template_q_max": None if template_q_after.size == 0 else float(np.nanmax(template_q_after)),
            "match_radius_q": MATCH_RADIUS_Q,
            "nearest_template_distance_min": None if nearest.size == 0 else float(np.nanmin(nearest)),
            "nearest_template_distance_median": None if nearest.size == 0 else float(np.nanmedian(nearest)),
            "matched_peak_count": matched_peak_count,
            "score_numerator": float(matched_peak_count),
            "score_denominator": float(denominator),
        }


    def save_json(path, obj):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"[saved] {path}")


    def load_calibration_peak_rows(path):
        """Load manual calibration peaks from JSON or CSV."""
        path = Path(path)
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            rows = obj.get("peaks", obj) if isinstance(obj, dict) else obj
        else:
            with open(path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

        parsed = []
        for idx, row in enumerate(rows):
            q_pixel = row.get("q_pixel", row.get("q_pix", row.get("pixel_radius")))
            q_known = row.get("q_A^-1", row.get("q_A_inv", row.get("known_q_A^-1", row.get("known_q"))))
            if q_pixel is None or q_known is None:
                raise ValueError(f"Calibration row {idx} must include q_pixel and q_A^-1/known_q_A^-1 fields.")
            parsed.append({
                "label": row.get("label", row.get("peak", f"peak_{idx}")),
                "q_pixel": float(q_pixel),
                "q_A^-1": float(q_known),
            })
        if not parsed:
            raise ValueError(f"No calibration peaks found in {path}")
        return parsed


    def write_calibration_summary(path, provided_inv_ang_per_pixel, calibration_peaks=None):
        """Fit or report the q calibration used by this run."""
        summary = {
            "mode": "provided_value_only",
            "provided_inv_ang_per_pixel": float(provided_inv_ang_per_pixel),
            "used_inv_ang_per_pixel": float(provided_inv_ang_per_pixel),
            "calibration_peaks_path": None,
            "peaks": [],
            "fit": None,
        }
        if calibration_peaks is not None:
            rows = load_calibration_peak_rows(calibration_peaks)
            q_pixel = np.asarray([r["q_pixel"] for r in rows], dtype=np.float64)
            q_known = np.asarray([r["q_A^-1"] for r in rows], dtype=np.float64)
            denom = float(np.sum(q_pixel * q_pixel))
            if denom <= 0:
                raise ValueError("Calibration q_pixel values must not all be zero.")
            fit_scale = float(np.sum(q_pixel * q_known) / denom)
            residual = q_known - q_pixel * fit_scale
            for row, pred, res in zip(rows, q_pixel * fit_scale, residual):
                row = dict(row)
                row["q_fit_A^-1"] = float(pred)
                row["residual_A^-1"] = float(res)
                row["relative_residual"] = None if row["q_A^-1"] == 0 else float(res / row["q_A^-1"])
                summary["peaks"].append(row)
            summary.update({
                "mode": "manual_peak_fit",
                "calibration_peaks_path": str(Path(calibration_peaks)),
                "used_inv_ang_per_pixel": fit_scale,
                "fit": {
                    "model": "q_A^-1 = q_pixel * inv_ang_per_pixel",
                    "rmse_A^-1": float(np.sqrt(np.mean(residual * residual))),
                    "max_abs_residual_A^-1": float(np.max(np.abs(residual))),
                    "relative_change_from_provided": float((fit_scale - provided_inv_ang_per_pixel) / provided_inv_ang_per_pixel),
                },
            })
        save_json(path, summary)
        return float(summary["used_inv_ang_per_pixel"]), summary


    def cache_param_tag():
        params_json = json.dumps(DETECT_PARAMS, sort_keys=True, default=str)
        digest = hashlib.sha1(params_json.encode("utf-8")).hexdigest()[:10]
        return f"{BRAGG_CACHE_TAG}_{digest}_inv{sanitize_tag(f'{INV_ANG_PER_PIXEL:.8g}')}"


    def load_branch_metadata(branch_dir):
        metas = []
        for path in sorted(Path(branch_dir).glob("metadata_branch_*.json")):
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            score_value = meta.get("score_path")
            score_path = None
            if score_value:
                score_path = Path(score_value)
                if not score_path.is_absolute():
                    score_path = Path(branch_dir) / score_path
                if not score_path.exists():
                    fallback = path.with_name(path.name.replace("metadata_branch_", "score_branch_").replace(".json", ".npy"))
                    score_path = fallback
            meta["score_path"] = None if score_path is None else str(score_path)
            metas.append(meta)
        return metas


    def aggregate_branch_outputs(branch_dir):
        """Aggregate branch-only score files without reloading py4DSTEM/HDF5 objects."""
        branch_dir = Path(branch_dir)
        metas = load_branch_metadata(branch_dir)
        if not metas:
            raise RuntimeError(f"No metadata_branch_*.json files found in {branch_dir}")

        failed_metas = [
            m for m in metas
            if m.get("branch_status") not in (None, "RUN") or not m.get("score_path") or not Path(m["score_path"]).exists()
        ]
        for meta in failed_metas:
            print(
                f"[warning] Skipping failed branch metadata during aggregation: "
                f"{meta.get('branch')} status={meta.get('branch_status')} reason={meta.get('failure_reason')}"
            )
        valid_metas = [m for m in metas if m not in failed_metas]
        real_metas = [m for m in valid_metas if m.get("group") == "real"]
        control_metas = [m for m in valid_metas if m.get("group") == "control"]
        if not real_metas:
            raise RuntimeError("Aggregation requires at least one real Ti branch with branch_status=RUN and an existing score file.")

        branch_score_maps = {m["branch"]: np.load(m["score_path"]) for m in valid_metas}
        phase_to_branch_names = {}
        for meta in real_metas:
            phase_to_branch_names.setdefault(meta["phase"], []).append(meta["branch"])

        real_phase_names = sorted(phase_to_branch_names.keys())
        phase_score_maps = {}
        phase_axis_maps = {}
        phase_axis_labels = {}
        for phase_name in real_phase_names:
            names = phase_to_branch_names[phase_name]
            stack = np.stack([branch_score_maps[name] for name in names], axis=0)
            phase_score_maps[phase_name] = np.nanmax(stack, axis=0)
            phase_axis_maps[phase_name] = np.nanargmax(stack, axis=0)
            phase_axis_labels[phase_name] = [m["axis_tag"] for m in real_metas if m["branch"] in names]
            plot_scalar_map(phase_score_maps[phase_name], f"Aggregated score: {phase_name}", f"score_phase_aggregated_from_branches_{phase_name}.png")

        real_phase_stack = np.stack([phase_score_maps[name] for name in real_phase_names], axis=0)
        real_best_phase_index = np.nanargmax(real_phase_stack, axis=0)
        real_best_score = np.nanmax(real_phase_stack, axis=0)
        real_phase_max_score = np.nanmax(real_phase_stack, axis=0)
        all_zero_score_mask = np.nan_to_num(real_phase_max_score, nan=-np.inf) <= 0
        no_valid_score_mask = all_zero_score_mask | (~np.isfinite(real_phase_max_score))
        argmax_tie_mask = np.sum(
            np.isclose(real_phase_stack, real_phase_max_score[None, :, :], rtol=1e-6, atol=1e-12)
            & np.isfinite(real_phase_stack),
            axis=0,
        ) > 1
        valid_score_mask = ~no_valid_score_mask
        if real_phase_stack.shape[0] >= 2:
            sorted_real_scores = np.sort(real_phase_stack, axis=0)
            real_score_margin = sorted_real_scores[-1] - sorted_real_scores[-2]
        else:
            real_score_margin = np.full_like(real_best_score, np.nan, dtype=np.float32)
        real_score_margin[no_valid_score_mask] = np.nan

        if control_metas:
            control_status = "RUN"
            control_names = [m["branch"] for m in control_metas]
            control_stack = np.stack([branch_score_maps[name] for name in control_names], axis=0)
            control_best_score = np.nanmax(control_stack, axis=0)
            control_minus_real = control_best_score - real_best_score
        else:
            control_status = "NOT_RUN"
            control_names = []
            control_stack = np.empty((0,) + real_best_score.shape, dtype=np.float32)
            control_best_score = np.full_like(real_best_score, np.nan, dtype=np.float32)
            control_minus_real = np.full_like(real_best_score, np.nan, dtype=np.float32)

        ambiguous_mask = (real_score_margin < MARGIN_THRESHOLD) & valid_score_mask
        control_failure_mask = np.isfinite(control_best_score) & (control_best_score > (real_best_score + CONTROL_FAIL_MARGIN))
        empty_mask = np.zeros_like(real_best_phase_index, dtype=bool)
        high_confidence_mask = valid_score_mask & (~ambiguous_mask) & (~control_failure_mask)

        plot_index_map(real_best_phase_index, real_phase_names, "Aggregated Ti-only best phase", "phase_map_real_ti_only_best_candidate.png", save_clean=True)
        plot_scalar_map(real_best_score, "Aggregated Ti-only best score", "phase_map_real_best_score.png")
        plot_scalar_map(real_score_margin, "Aggregated Ti-only score margin", "phase_map_real_score_margin.png")
        plot_phase_map_with_masks(
            real_best_phase_index,
            real_phase_names,
            invalid_mask=empty_mask,
            mixed_mask=empty_mask,
            ambiguous_mask=ambiguous_mask,
            control_fail_mask=control_failure_mask,
            filename="phase_map_real_ti_only_qc_masked.png",
            no_valid_mask=no_valid_score_mask,
        )

        np.savez_compressed(
            OUT_DIR / "phase_orientation_scores_v6_optimized.npz",
            real_branch_names=np.array([m["branch"] for m in real_metas]),
            real_branch_score_stack=np.stack([branch_score_maps[m["branch"]] for m in real_metas], axis=0),
            real_phase_names=np.array(real_phase_names),
            real_phase_score_stack=real_phase_stack,
            real_best_phase_index=real_best_phase_index,
            real_best_score=real_best_score,
            real_score_margin=real_score_margin,
            no_valid_score_mask=no_valid_score_mask,
            all_zero_score_mask=all_zero_score_mask,
            argmax_tie_mask=argmax_tie_mask,
            control_branch_names=np.array(control_names),
            control_branch_score_stack=control_stack,
            control_best_score=control_best_score,
            control_minus_real=control_minus_real,
            ambiguous_mask=ambiguous_mask,
            control_failure_mask=control_failure_mask,
            high_confidence_mask=high_confidence_mask,
        )

        total_pixels = int(np.prod(real_best_phase_index.shape))
        summary = {
            "settings": {
                "data_file": str(DATA_FILE),
                "out_dir": str(OUT_DIR),
                "analysis_root": str(ROOT),
                "cif_dir": str(CIF_DIR),
                "screening_mode": "fiber_axis_only",
                "aggregation_source": str(branch_dir),
                "branch_count": len(metas),
                "run_control": bool(control_metas),
                "control_status": control_status,
                "k_max": K_MAX,
                "inv_ang_per_pixel": INV_ANG_PER_PIXEL,
                "angle_step_zone_axis": ANGLE_STEP_ZONE_AXIS,
                "angle_step_in_plane": ANGLE_STEP_IN_PLANE,
                "margin_threshold": MARGIN_THRESHOLD,
            },
            "confidence_summary": {
                "ambiguous_fraction_real_margin_or_score": float(np.sum(ambiguous_mask) / total_pixels),
                "control_failure_fraction": float(np.sum(control_failure_mask) / total_pixels),
                "final_high_confidence_fraction": float(np.sum(high_confidence_mask) / total_pixels),
                "no_valid_score_fraction": float(np.sum(no_valid_score_mask) / total_pixels),
                "all_zero_score_fraction": float(np.sum(all_zero_score_mask) / total_pixels),
                "argmax_tie_fraction": float(np.sum(argmax_tie_mask) / total_pixels),
                "failure_reason_fraction": {
                    "FAILED_NO_VALID_SCORE": float(np.sum(no_valid_score_mask) / total_pixels),
                    "AMBIGUOUS_LOW_MARGIN": float(np.sum(ambiguous_mask) / total_pixels),
                    "FAILED_CONTROL": float(np.sum(control_failure_mask) / total_pixels),
                    "PASS": float(np.sum(high_confidence_mask) / total_pixels),
                },
                "real_score_margin_median": finite_stat(real_score_margin, np.median),
                "control_minus_real_median": finite_stat(control_minus_real, np.median),
            },
            "branch_metadata": metas,
        }
        summary_path = OUT_DIR / "phase_summary_v6_optimized.json"
        save_json(summary_path, summary)
        generate_phase_orientation_report(
            OUT_DIR,
            summary_path=summary_path,
            title="Aggregated Phase/Orientation Screening Report",
        )
        print("Done.")


    if args.aggregate_branches is not None:
        aggregate_branch_outputs(args.aggregate_branches)
        raise SystemExit(0)


    # =============================================================================
    # 2. Load data and save basic QC images
    # =============================================================================

    INV_ANG_PER_PIXEL, CALIBRATION_SUMMARY = write_calibration_summary(
        OUT_DIR / "calibration_summary.json",
        INV_ANG_PER_PIXEL,
        args.calibration_peaks,
    )

    log_status("Starting phase/orientation screening")
    if ORIENTATION_MODE == "s2":
        log_status("Orientation mode is s2/full-sphere; this can take substantially longer than --orientation-mode fiber.")

    with status_step("Importing py4DSTEM", STATUS_INTERVAL):
        import py4DSTEM

    print("Script: main.py")
    print(f"py4DSTEM version: {py4DSTEM.__version__}")
    print(f"Data file: {DATA_FILE}")
    print(f"Analysis root: {ROOT}")
    print(f"CIF directory: {CIF_DIR}")
    print(f"Run output: {OUT_DIR}")
    print(f"Mode: {args.mode}, K_MAX={K_MAX}, inv_ang_per_pixel={INV_ANG_PER_PIXEL}")
    print(f"angle steps: zone_axis={ANGLE_STEP_ZONE_AXIS}, in_plane={ANGLE_STEP_IN_PLANE}")
    print(f"run_control={args.run_control}, branch_only={args.branch_only}")
    with status_step("Reading HDF5 tree", STATUS_INTERVAL):
        py4DSTEM.print_h5_tree(DATA_FILE)

    with status_step("Loading 4D-STEM dataset", STATUS_INTERVAL):
        dataset = py4DSTEM.read(DATA_FILE)
    print(dataset)

    with status_step("Computing dp_max and dp_mean", STATUS_INTERVAL):
        dataset.get_dp_max()
        dataset.get_dp_mean()
    py4DSTEM.visualize.show(dataset.tree("dp_max"), figsize=(4, 4), ticks=False)
    savefig("qc_dp_max.png")
    py4DSTEM.visualize.show(dataset.tree("dp_mean"), figsize=(4, 4), ticks=False)
    savefig("qc_dp_mean.png")


    # =============================================================================
    # 3. Probe estimation, virtual ADF, template
    # =============================================================================

    probe_semiangle, probe_qx0, probe_qy0 = dataset.get_probe_size(dataset.tree("dp_mean").data)
    center = (probe_qx0, probe_qy0)
    print(f"Estimated probe radius = {probe_semiangle:.2f} pixels")
    print(f"Estimated center = ({probe_qx0:.2f}, {probe_qy0:.2f})")
    if not np.isfinite(probe_qx0) or not np.isfinite(probe_qy0) or probe_semiangle <= 0:
        raise RuntimeError("Beam center/probe radius estimation failed; cannot continue safely.")

    direct_beam_mask_radius = float(args.direct_beam_mask_radius) if args.direct_beam_mask_radius is not None else float(max(probe_semiangle * 1.25, 4.0))
    peak_detection_diagnostics = [
        save_diffraction_qc_images(
            "dp_mean",
            dataset.tree("dp_mean").data,
            center=center,
            beam_radius=direct_beam_mask_radius,
        )
    ]
    for xind, yind in zip(TEST_RXS, TEST_RYS):
        try:
            peak_detection_diagnostics.append(
                save_diffraction_qc_images(
                    f"single_x{xind}_y{yind}",
                    dataset.data[xind, yind, :, :],
                    center=center,
                    beam_radius=direct_beam_mask_radius,
                )
            )
        except Exception as exc:
            print(f"[warning] Could not save diffraction QC images for ({xind}, {yind}): {exc}")

    py4DSTEM.show(
        dataset.tree("dp_mean"),
        figsize=(4, 4),
        circle={"center": center, "R": probe_semiangle},
        ticks=False,
        returnfig=True,
        vmax=1,
    )
    savefig("qc_probe_center.png")

    ADF_RADII = (13, 24)
    dataset.position_detector(mode="annular", geometry=(center, ADF_RADII), figsize=(4, 4), ticks=False)
    savefig("qc_adf_detector.png")
    with status_step("Computing virtual dark-field image", STATUS_INTERVAL):
        dataset.get_virtual_image(mode="annulus", geometry=(center, ADF_RADII), name="dark_field")
    py4DSTEM.show(dataset.tree("dark_field"), figsize=(4, 4))
    savefig("qc_dark_field.png")

    probe = py4DSTEM.Probe.generate_synthetic_probe(
        radius=probe_semiangle,
        width=0.7,
        Qshape=dataset.Qshape,
    )
    probe.get_kernel(
        mode="sigmoid",
        origin=(dataset.Qshape[0] / 2, dataset.Qshape[1] / 2),
        radii=(probe_semiangle * 1.2, probe_semiangle * 4),
    )
    py4DSTEM.visualize.show_kernel(probe.kernel, R=30, L=30, W=1)
    savefig("qc_probe_kernel.png")


    # =============================================================================
    # 4. Bragg disk detection, origin fitting, q calibration
    # =============================================================================

    print("Testing Bragg disk detection on selected probe positions...")
    with status_step("Testing Bragg disk detection on selected probe positions", STATUS_INTERVAL):
        disks_selected = dataset.find_Bragg_disks(data=(TEST_RXS, TEST_RYS), template=probe.kernel, **DETECT_PARAMS)
    colors = ["r", "limegreen", "c"]
    py4DSTEM.visualize.show_image_grid(
        get_ar=lambda i: dataset.data[TEST_RXS[i], TEST_RYS[i], :, :],
        H=1,
        W=len(TEST_RXS),
        axsize=(3, 3),
        get_bordercolor=lambda i: colors[i % len(colors)],
        get_x=lambda i: disks_selected[i].data["qx"],
        get_y=lambda i: disks_selected[i].data["qy"],
        get_pointcolors=lambda i: colors[i % len(colors)],
        open_circles=True,
        scale=300,
    )
    savefig("qc_bragg_detection_test.png")

    bragg_cache = ROOT / DATA_FILE.stem / f"braggdisks_cal_{cache_param_tag()}.h5"
    bragg_cache_status = "miss"
    calibration_status = {
        "beam_center_px": [float(probe_qx0), float(probe_qy0)],
        "probe_radius_px": float(probe_semiangle),
        "direct_beam_mask_radius_px": float(direct_beam_mask_radius),
        "q_scale_A_inv_per_px": float(INV_ANG_PER_PIXEL),
        "q_scale_mode": CALIBRATION_SUMMARY.get("mode"),
        "origin": "NOT_RUN",
        "ellipse": "NOT_RUN",
        "rotate": "NOT_RUN",
        "warnings": [],
    }
    if USE_CACHED_BRAGG_PEAKS and bragg_cache.exists():
        print(f"Loading cached calibrated Bragg peaks: {bragg_cache}")
        with status_step("Loading cached calibrated Bragg peaks", STATUS_INTERVAL):
            bragg_peaks = py4DSTEM.read(bragg_cache)
        bragg_cache_status = "hit"
        calibration_status["origin"] = "LOADED_FROM_CACHE"
        calibration_status["ellipse"] = "LOADED_FROM_CACHE" if bragg_peaks.calstate.get("ellipse") else "NOT_AVAILABLE_IN_CACHE"
        calibration_status["rotate"] = "LOADED_FROM_CACHE" if bragg_peaks.calstate.get("rotate") else "NOT_AVAILABLE_IN_CACHE"
    else:
        print("Finding Bragg disks for all probe positions...")
        with status_step("Finding Bragg disks for all probe positions", STATUS_INTERVAL):
            bragg_peaks = dataset.find_Bragg_disks(template=probe.kernel, **DETECT_PARAMS)

        print("Measuring and fitting origin...")
        with status_step("Measuring and fitting Bragg origin", STATUS_INTERVAL):
            bragg_peaks.measure_origin()
            bragg_peaks.fit_origin(figsize=(4, 4))
        savefig("qc_origin_fit.png")
        calibration_status["origin"] = "RUN"

        bragg_vector_map_centered = bragg_peaks.get_bvm()
        py4DSTEM.show(bragg_vector_map_centered, figsize=(4, 4))
        savefig("qc_bvm_centered.png")

        ellipse_inner = max(float(probe_semiangle * 2.0), 8.0)
        ellipse_outer = min(float(Q_MAX_FOR_QC / INV_ANG_PER_PIXEL), min(dataset.Qshape) / 2 - 4)
        if ellipse_outer > ellipse_inner:
            ellipse_errors = []
            ellipse_centers = [(0, 0), (dataset.Qshape[0] / 2, dataset.Qshape[1] / 2)]
            try:
                for ellipse_center in ellipse_centers:
                    try:
                        bragg_peaks.fit_p_ellipse(
                            bragg_vector_map_centered,
                            center=ellipse_center,
                            fitradii=(ellipse_inner, ellipse_outer),
                            scaling="log",
                            figsize=(5, 5),
                        )
                        savefig("qc_elliptical_calibration_fit.png")
                        calibration_status["ellipse_center"] = [float(ellipse_center[0]), float(ellipse_center[1])]
                        break
                    except Exception as exc:
                        ellipse_errors.append(f"center={ellipse_center}: {exc}")
                else:
                    raise RuntimeError("; ".join(ellipse_errors))
                calibration_status["ellipse"] = "RUN"
                calibration_status["ellipse_fitradii_px"] = [float(ellipse_inner), float(ellipse_outer)]
            except Exception as exc:
                calibration_status["ellipse"] = "FAILED"
                calibration_status["warnings"].append(f"Elliptical calibration failed: {exc}")
                print(f"[warning] Elliptical calibration failed: {exc}")
                if not args.allow_screening_fallback:
                    print("[warning] Continuing without ellipse calibration; pass --allow-screening-fallback to silence this warning.")
        else:
            calibration_status["ellipse"] = "SKIPPED_INVALID_FIT_RADII"
            calibration_status["warnings"].append("Ellipse fit radii were invalid for this detector shape/q range.")

        q_pix, intensity_radial = py4DSTEM.process.utils.radial_integral(bragg_vector_map_centered)
        py4DSTEM.visualize.show_qprofile(q=q_pix, intensity=intensity_radial * q_pix, ymax=YMAX_RADIAL_PROFILE, color="r")
        savefig("qc_radial_profile_pixels.png")

        bragg_peaks.calibration.set_Q_pixel_size(INV_ANG_PER_PIXEL)
        bragg_peaks.calibration.set_Q_pixel_units("A^-1")
        py4DSTEM.save(bragg_cache, bragg_peaks, mode="o")
        print(f"[saved] {bragg_cache}")
        bragg_cache_status = "recomputed" if args.force_recompute_bragg else "created"

    print(bragg_peaks.calstate)
    calibration_status["calstate"] = {key: bool(value) for key, value in bragg_peaks.calstate.items()}

    # BVM/radial profile reused for q overlays.
    bragg_vector_map_centered = bragg_peaks.get_bvm()
    q_pix, intensity_radial = py4DSTEM.process.utils.radial_integral(bragg_vector_map_centered)

    peak_count_map = compute_peak_count_map(bragg_peaks)
    if peak_count_map is not None:
        plot_scalar_map(peak_count_map, "Detected Bragg peak count", "qc_peak_count_map.png")
        plot_histogram(peak_count_map, "Detected Bragg peak count", "hist_peak_count.png", bins=40)
    else:
        print("[warning] Peak-count map unavailable; peak-count QC masks will be disabled.")

    peak_qc_maps = compute_peak_qc_maps(bragg_peaks)
    if peak_qc_maps is not None:
        clean_peak_count_map = peak_qc_maps["clean_peak_count"]
        strong_peak_count_map = peak_qc_maps["strong_peak_count"]
        q_median_map = peak_qc_maps["q_median"]
        q_p90_map = peak_qc_maps["q_p90"]

        plot_scalar_map(clean_peak_count_map, "QC peak count inside q range", "qc_clean_peak_count_map.png")
        plot_scalar_map(strong_peak_count_map, "Strong QC peak count", "qc_strong_peak_count_map.png")
        plot_scalar_map(q_median_map, "Median detected peak q", "qc_peak_q_median_map.png")
        plot_scalar_map(q_p90_map, "90th percentile detected peak q", "qc_peak_q_p90_map.png")
        plot_histogram(clean_peak_count_map, "QC peak count inside q range", "hist_clean_peak_count.png", bins=40)
        plot_histogram(strong_peak_count_map, "Strong QC peak count", "hist_strong_peak_count.png", bins=40)

        for xind, yind in zip(TEST_RXS, TEST_RYS):
            plot_peak_radius_histogram_for_pixel(
                bragg_peaks,
                xind,
                yind,
                f"qc_peak_q_radius_hist_x{xind}_y{yind}.png",
            )

        low_peak_mask_base = strong_peak_count_map < MIN_STRONG_PEAKS_FOR_MATCH
        mixed_peak_mask_base = clean_peak_count_map > MAX_CLEAN_PEAKS_FOR_SINGLE
        finite_q_median = q_median_map[np.isfinite(q_median_map)]
        finite_q_p90 = q_p90_map[np.isfinite(q_p90_map)]
        EXP_Q_SUMMARY = {
            "q_min_exp": float(np.nanmin(finite_q_median)) if finite_q_median.size else None,
            "q_median_exp": float(np.nanmedian(finite_q_median)) if finite_q_median.size else None,
            "q_max_exp": float(np.nanmax(finite_q_p90)) if finite_q_p90.size else None,
            "n_clean_peaks_p50": float(np.nanmedian(clean_peak_count_map)),
            "n_clean_peaks_p95": float(np.nanpercentile(clean_peak_count_map, 95)),
            "n_strong_peaks_p50": float(np.nanmedian(strong_peak_count_map)),
            "n_strong_peaks_p95": float(np.nanpercentile(strong_peak_count_map, 95)),
        }
    else:
        clean_peak_count_map = np.array([])
        strong_peak_count_map = np.array([])
        q_median_map = np.array([])
        q_p90_map = np.array([])
        low_peak_mask_base = peak_count_map < PEAK_COUNT_THRESHOLD if peak_count_map is not None else None
        mixed_peak_mask_base = None
        EXP_Q_SUMMARY = {
            "q_min_exp": None,
            "q_median_exp": None,
            "q_max_exp": None,
            "n_clean_peaks_p50": None,
            "n_clean_peaks_p95": None,
            "n_strong_peaks_p50": None,
            "n_strong_peaks_p95": None,
        }

    TEST_MATCH_PIXELS = select_representative_test_pixels(
        clean_peak_count_map,
        fallback=SINGLE_TEST_PIXEL,
        max_points=5,
    )
    print(f"[info] Representative single-pattern test pixels: {TEST_MATCH_PIXELS}")

    # =============================================================================
    # 5. Multi-axis orientation matching for real candidates and controls
    # =============================================================================

    def run_phase_group(phases, group_name):
        """
        Run all phase + fiber-axis branches in one group.
        Returns dictionaries of branch score maps, phase aggregated score maps, etc.
        """
        print("\n" + "#" * 80)
        print(f"Running phase group: {group_name}")
        print("#" * 80)

        branch_results = []
        branch_score_maps = {}
        branch_score_stacks = {}
        branch_orientation_maps = {}
        branch_crystals = {}
        branch_phase_names = {}
        branch_axes = {}
        phase_to_branch_names = {}

        for phase in phases:
            phase_name = phase["name"]
            cif_path = Path(phase["cif"])
            if not cif_path.exists():
                msg = f"[warning] CIF not found for {phase_name}: {cif_path}"
                if SKIP_MISSING_CIFS:
                    print(msg + " -- skipped")
                    continue
                raise FileNotFoundError(msg)

            print("\n" + "=" * 80)
            print(f"Candidate phase: {phase_name} [{group_name}]")
            print(f"CIF: {cif_path}")

            with status_step(f"Loading CIF and calculating structure factors for {phase_name}", STATUS_INTERVAL):
                crystal = py4DSTEM.process.diffraction.Crystal.from_CIF(cif_path)
                crystal.calculate_structure_factors(K_MAX)
            template_q = np.asarray(getattr(crystal, "g_vec_leng", []), dtype=np.float64)
            template_q = template_q[np.isfinite(template_q)]
            template_q_after = template_q[template_q <= K_MAX] if template_q.size else np.array([], dtype=np.float64)
            template_diagnostics = {
                "n_template_reflections_before_filter": int(template_q.size),
                "n_template_reflections_after_kmax": int(template_q_after.size),
                "q_min_template": None if template_q_after.size == 0 else float(np.nanmin(template_q_after)),
                "q_median_template": None if template_q_after.size == 0 else float(np.nanmedian(template_q_after)),
                "q_max_template": None if template_q_after.size == 0 else float(np.nanmax(template_q_after)),
                "match_radius_q": MATCH_RADIUS_Q,
                **EXP_Q_SUMMARY,
            }

            # Rough [001]-like structure-factor overlay. Useful only as QC, not proof.
            try:
                q_sf = np.linspace(0, K_MAX, 250)
                i_sf = np.zeros_like(q_sf)
                for a0 in range(crystal.g_vec_leng.shape[0]):
                    if np.abs(crystal.g_vec_all[2, a0]) < 0.01:
                        idx = np.argmin(np.abs(q_sf - crystal.g_vec_leng[a0]))
                        i_sf[idx] += crystal.struct_factors_int[a0]
                if np.nanmax(i_sf) > 0:
                    i_sf /= np.nanmax(i_sf)

                fig, ax = py4DSTEM.visualize.show_qprofile(
                    q=q_pix * INV_ANG_PER_PIXEL,
                    intensity=intensity_radial * q_pix,
                    xlabel="q (1/Ang)",
                    returnfig=True,
                    ymax=YMAX_RADIAL_PROFILE,
                    color="b",
                    label="BVM radial * q",
                )
                ax.plot(q_sf, i_sf * YMAX_RADIAL_PROFILE, c="r", label=f"{phase_name} SF")
                ax.set_xlim([0, K_MAX])
                ax.legend()
                savefig(f"qc_q_calibration_{group_name}_{phase_name}.png")
            except Exception as exc:
                print(f"[warning] Could not save q-calibration overlay for {phase_name}: {exc}")

            if ORIENTATION_MODE == "s2":
                orientation_branches = [{
                    "axis": "full_s2",
                    "axis_tag": "full_s2",
                    "zone_axis_range": "full",
                    "fiber_axis": None,
                    "fiber_angles": None,
                }]
            else:
                fiber_axes = phase.get("fiber_axes") or [phase.get("fiber_axis", [0, 0, 1])]
                orientation_branches = [
                    {
                        "axis": fiber_axis,
                        "axis_tag": axis_to_tag(fiber_axis),
                        "zone_axis_range": phase.get("zone_axis_range", "fiber"),
                        "fiber_axis": fiber_axis,
                        "fiber_angles": phase.get("fiber_angles", [0, 360]),
                    }
                    for fiber_axis in fiber_axes
                ]
            phase_to_branch_names[phase_name] = []

            for branch_spec in orientation_branches:
                fiber_axis = branch_spec["axis"]
                axis_tag = branch_spec["axis_tag"]
                branch_name = f"{phase_name}_{axis_tag}"

                print("\n" + "-" * 80)
                print(f"Orientation branch: {branch_name} [{group_name}]")
                print(f"orientation_mode = {ORIENTATION_MODE}")
                print(f"axis/library = {fiber_axis}")
                print("Building orientation plan...")

                orientation_kwargs = {
                    "angle_step_zone_axis": ANGLE_STEP_ZONE_AXIS,
                    "angle_step_in_plane": ANGLE_STEP_IN_PLANE,
                    "zone_axis_range": branch_spec["zone_axis_range"],
                    "progress_bar": not args.quiet_progress,
                    "corr_kernel_size": MATCH_RADIUS_Q,
                }
                if branch_spec["fiber_axis"] is not None:
                    orientation_kwargs["fiber_axis"] = branch_spec["fiber_axis"]
                    orientation_kwargs["fiber_angles"] = branch_spec["fiber_angles"]
                with status_step(f"Building orientation library for {branch_name}", STATUS_INTERVAL):
                    crystal.orientation_plan(**orientation_kwargs)

                single_test_diagnostics = []
                single_test_valid = False
                for xind, yind in TEST_MATCH_PIXELS:
                    diag = summarize_single_pattern_match_inputs(bragg_peaks, xind, yind, template_q_after)
                    print(
                        f"Testing single-pattern match at ({xind}, {yind}) "
                        f"with n_clean={diag['n_clean_peaks_test']} "
                        f"template_n={diag['n_template_reflections']}..."
                    )
                    try:
                        orientation = crystal.match_single_pattern(
                            bragg_peaks.cal[xind, yind],
                            num_matches_return=min(NUM_MATCHES_RETURN, TOP_CANDIDATES_TO_SAVE),
                            verbose=True,
                        )
                        diag["single_pattern_match_returned"] = orientation is not None
                        bragg_peaks_fit = crystal.generate_diffraction_pattern(
                            orientation,
                            ind_orientation=0,
                            sigma_excitation_error=0.03,
                        )
                        py4DSTEM.process.diffraction.plot_diffraction_pattern(
                            bragg_peaks_fit,
                            bragg_peaks_compare=bragg_peaks.cal[xind, yind],
                            scale_markers=1000,
                            scale_markers_compare=4e4,
                            plot_range_kx_ky=np.array([K_MAX + 0.1, K_MAX + 0.1]),
                            min_marker_size=1,
                            figsize=(5, 5),
                        )
                        savefig(f"qc_single_match_{group_name}_{branch_name}_x{xind}_y{yind}.png")
                        if orientation is not None and diag["matched_peak_count"] > 0:
                            single_test_valid = True
                    except Exception as exc:
                        diag["single_pattern_match_returned"] = False
                        diag["single_pattern_error"] = str(exc)
                        print(f"[warning] Single-pattern QC failed for {branch_name} at ({xind}, {yind}): {exc}")
                    single_test_diagnostics.append(diag)

                best_test_diag = max(
                    single_test_diagnostics,
                    key=lambda item: (
                        item.get("matched_peak_count") or 0,
                        item.get("n_clean_peaks_test") or 0,
                    ),
default={},
                )

                if not single_test_valid:
                    print(
                        f"[warning] Branch {branch_name}: single-pattern QC found no valid match; "
                        "continuing to full-map matching in screening mode."
                    )

                print("Matching orientations for all probe positions...")
                with status_step(f"Matching orientations for {branch_name}", STATUS_INTERVAL):
                    orientation_map = crystal.match_orientations(
                        bragg_peaks,
                        num_matches_return=NUM_MATCHES_RETURN,
                        min_number_peaks=MIN_STRONG_PEAKS_FOR_MATCH,
                        progress_bar=not args.quiet_progress,
                    )
                score_stack = get_orientation_score_stack(orientation_map)
                score = score_stack[:, :, 0]
                score_max = finite_stat(score, np.nanmax)
                if score_max is None or score_max <= 0:
                    branch_results.append({
                        "group": group_name,
                        "branch": branch_name,
                        "phase": phase_name,
                        "fiber_axis": fiber_axis,
"cif": str(cif_path),
                        **template_diagnostics,
                        "branch_status": "FAILED_NO_VALID_MATCH",
                        "failure_reason": "full-map score max is non-finite or <= 0",
                        "single_pattern_qc_passed": bool(single_test_valid),
                        "single_pattern_test_pixels": [list(p) for p in TEST_MATCH_PIXELS],
                        "single_pattern_test_diagnostics": single_test_diagnostics,
                        **best_test_diag,
                        "score_mean": finite_stat(score, np.nanmean),
                        "score_median": finite_stat(score, np.nanmedian),
                        "score_p95": finite_stat(score, lambda vals: np.nanpercentile(vals, 95)),
                        "score_max": score_max,
                        "matched_peak_count_single_pixel": best_test_diag.get("matched_peak_count"),
                    })
                    print(f"[warning] Branch {branch_name} produced no valid full-map score and will not enter phase aggregation.")
                    continue

                phase_to_branch_names[phase_name].append(branch_name)
                branch_score_maps[branch_name] = score
                branch_score_stacks[branch_name] = score_stack
                branch_orientation_maps[branch_name] = orientation_map
                branch_crystals[branch_name] = crystal
                branch_phase_names[branch_name] = phase_name
                branch_axes[branch_name] = fiber_axis

                plot_scalar_map(score, f"Raw correlation: {branch_name}", f"score_{group_name}_{branch_name}.png")
                plot_scalar_map(normalize_score(score), f"Normalized correlation: {branch_name}", f"score_norm_{group_name}_{branch_name}.png")

                try:
                    crystal.plot_fiber_orientation_maps(
                        orientation_map,
                        symmetry_order=phase.get("symmetry_order", None),
                        corr_range=None,
                        figsize=(4, 4),
                    )
                    savefig(f"orientation_fiber_{group_name}_{branch_name}.png")
                except Exception as exc:
                    print(f"[warning] Could not plot fiber orientation map for {branch_name}: {exc}")

                branch_results.append({
                    "group": group_name,
                    "branch": branch_name,
                    "phase": phase_name,
                    "fiber_axis": fiber_axis,
                    "cif": str(cif_path),
                    **template_diagnostics,
"branch_status": "RUN",
                    "failure_reason": None,
                    "single_pattern_qc_passed": bool(single_test_valid),
                    "single_pattern_test_pixels": [list(p) for p in TEST_MATCH_PIXELS],
                    "single_pattern_test_diagnostics": single_test_diagnostics,
                    **best_test_diag,
                    "score_mean": float(np.nanmean(score)),
                    "score_median": float(np.nanmedian(score)),
                    "score_p95": float(np.nanpercentile(score, 95)),
                    "score_max": score_max,
                    "matched_peak_count_single_pixel": best_test_diag.get("matched_peak_count"),
                })

        phase_score_maps = {}
        phase_best_axis_index_maps = {}
        phase_axis_labels = {}
        phase_names = []

        for phase in phases:
            phase_name = phase["name"]
            names = phase_to_branch_names.get(phase_name, [])
            names = [name for name in names if name in branch_score_maps]
            if not names:
                continue
            phase_names.append(phase_name)
            local_stack = np.stack([branch_score_maps[name] for name in names], axis=0)
            local_best_axis_idx = np.nanargmax(local_stack, axis=0)
            local_best_score = np.nanmax(local_stack, axis=0)

            phase_score_maps[phase_name] = local_best_score
            phase_best_axis_index_maps[phase_name] = local_best_axis_idx
            phase_axis_labels[phase_name] = [axis_to_tag(branch_axes[name]) for name in names]

            plot_scalar_map(local_best_score, f"Aggregated {group_name} score: {phase_name}", f"score_phase_aggregated_{group_name}_{phase_name}.png")
            plot_index_map(local_best_axis_idx, phase_axis_labels[phase_name], f"Best axis within {phase_name}", f"best_axis_within_{group_name}_{phase_name}.png")

        return {
            "group_name": group_name,
                "branch_results": branch_results,
                "branch_score_maps": branch_score_maps,
                "branch_score_stacks": branch_score_stacks,
                "branch_orientation_maps": branch_orientation_maps,
            "branch_crystals": branch_crystals,
            "branch_phase_names": branch_phase_names,
            "branch_axes": branch_axes,
            "phase_to_branch_names": phase_to_branch_names,
            "phase_names": phase_names,
            "phase_score_maps": phase_score_maps,
            "phase_best_axis_index_maps": phase_best_axis_index_maps,
            "phase_axis_labels": phase_axis_labels,
        }


    def select_branch_phase(phase_name, fiber_axis):
        all_phases = [(p, "real") for p in REAL_CANDIDATE_PHASES] + [(p, "control") for p in CONTROL_PHASES]
        for phase, group_name in all_phases:
            if phase["name"] != phase_name:
                continue
            axes = phase.get("fiber_axes") or [phase.get("fiber_axis", [0, 0, 1])]
            if fiber_axis not in axes:
                raise ValueError(f"Fiber axis {fiber_axis} is not configured for {phase_name}. Available axes: {axes}")
            selected = dict(phase)
            selected["fiber_axes"] = [fiber_axis]
            return selected, group_name
        raise ValueError(f"Unknown phase for --branch-only: {phase_name}")


    if args.branch_only:
        if not args.phase or not args.fiber_axis:
            raise ValueError("--branch-only requires --phase and --fiber-axis.")
        selected_axis = parse_axis(args.fiber_axis)
        selected_phase, selected_group = select_branch_phase(args.phase, selected_axis)
        branch_run = run_phase_group([selected_phase], selected_group)
        branch_result = branch_run["branch_results"][0] if branch_run["branch_results"] else {}
        branch_name = branch_result.get("branch") or f"{selected_phase['name']}_{axis_to_tag(selected_axis)}"
        score_path = OUT_DIR / f"score_branch_{branch_name}.npy"
        score_path_name = None
        if branch_name in branch_run["branch_score_maps"]:
            np.save(score_path, branch_run["branch_score_maps"][branch_name])
            score_path_name = score_path.name
        else:
            print(f"[warning] Branch-only run did not produce a valid score map for {branch_name}.")
        metadata = {
            "data_file": str(DATA_FILE),
            "out_dir": str(OUT_DIR),
            "screening_mode": "fiber_axis_only",
            "group": selected_group,
            "phase": branch_result.get("phase", selected_phase["name"]),
            "fiber_axis": branch_result.get("fiber_axis", selected_axis),
            "axis_tag": axis_to_tag(branch_result.get("fiber_axis", selected_axis)),
            "branch": branch_name,
            "score_path": score_path_name,
            "branch_status": branch_result.get("branch_status", "FAILED_NO_VALID_MATCH"),
            "failure_reason": branch_result.get("failure_reason"),
            "branch_diagnostics": branch_result,
            "orientation_mode": ORIENTATION_MODE,
            "num_matches_return": NUM_MATCHES_RETURN,
            "k_max": K_MAX,
            "inv_ang_per_pixel": INV_ANG_PER_PIXEL,
            "angle_step_zone_axis": ANGLE_STEP_ZONE_AXIS,
            "angle_step_in_plane": ANGLE_STEP_IN_PLANE,
            "bragg_cache": str(bragg_cache),
            "bragg_cache_status": bragg_cache_status,
            "detect_params": DETECT_PARAMS,
        }
        save_json(OUT_DIR / f"metadata_branch_{branch_name}.json", metadata)
        print("Done.")
        raise SystemExit(0 if score_path_name else 2)


    real = run_phase_group(REAL_CANDIDATE_PHASES, "real")
    if args.run_control:
        control = run_phase_group(CONTROL_PHASES, "control")
    else:
        print("[info] Skipping WS2-control branches because --skip-control was set.")
        control = {
            "group_name": "control",
            "branch_results": [],
            "branch_score_maps": {},
            "branch_score_stacks": {},
            "branch_orientation_maps": {},
            "branch_crystals": {},
            "branch_phase_names": {},
            "branch_axes": {},
            "phase_to_branch_names": {},
            "phase_names": [],
            "phase_score_maps": {},
            "phase_best_axis_index_maps": {},
            "phase_axis_labels": {},
        }


    # =============================================================================
    # 6. Build Ti-only phase map and QC masks
    # =============================================================================

    control_status = "RUN" if args.run_control else "NOT_RUN"

    def format_summary_stat(value):
        if value is None:
            return "NA"
        try:
            if not np.isfinite(value):
                return "NA"
            return f"{float(value):.4g}"
        except Exception:
            return str(value)

    if len(real["phase_names"]) == 0:
        failure_summary = {
            "settings": {
                "data_file": str(DATA_FILE),
                "out_dir": str(OUT_DIR),
                "output_tag": args.output_tag,
                "mode": args.mode,
                "screening_mode": "failed_before_phase_aggregation",
                "orientation_mode": ORIENTATION_MODE,
                "num_matches_return": NUM_MATCHES_RETURN,
                "run_control": args.run_control,
                "control_status": control_status,
                "k_max": K_MAX,
                "inv_ang_per_pixel": INV_ANG_PER_PIXEL,
                "angle_step_zone_axis": ANGLE_STEP_ZONE_AXIS,
                "angle_step_in_plane": ANGLE_STEP_IN_PLANE,
                "detect_params": DETECT_PARAMS,
                "q_space_diagnostics": EXP_Q_SUMMARY,
                "match_radius_q": MATCH_RADIUS_Q,
                "bragg_cache_status": bragg_cache_status,
                "bragg_cache_path": str(bragg_cache),
                "calibration": CALIBRATION_SUMMARY,
                "calibration_status": calibration_status,
                "peak_detection_diagnostics": peak_detection_diagnostics,
            },
            "confidence_summary": {
                "no_valid_score_fraction": 1.0,
                "all_zero_score_fraction": 1.0,
                "argmax_tie_fraction": 0.0,
                "failure_reason_fraction": {"FAILED_NO_VALID_BRANCH": 1.0},
                "high_confidence_fraction": 0.0,
            },
            "top_orientation_candidate_summary": {},
            "distinguishability_summary": {
                "conclusion": "INSUFFICIENT_QC",
                "reason": "No real Ti branch reached branch_status=RUN.",
            },
            "real_branch_results": real["branch_results"],
            "control_branch_results": control["branch_results"],
            "real_phase_results_aggregated_over_axes": [],
            "control_phase_results_aggregated_over_axes": [],
        }
        summary_path = OUT_DIR / "phase_summary_v6_optimized.json"
        save_json(summary_path, failure_summary)
        generate_phase_orientation_report(OUT_DIR, summary_path=summary_path)
        raise RuntimeError("No real candidate phase was successfully processed; see phase_summary_v6_optimized.json for branch failure diagnostics.")

    real_phase_names = real["phase_names"]
    real_phase_stack = np.stack([real["phase_score_maps"][name] for name in real_phase_names], axis=0)
    total_pixels = int(np.prod(real_phase_stack.shape[1:]))
    real_best_phase_index = np.nanargmax(real_phase_stack, axis=0)
    real_best_score = np.nanmax(real_phase_stack, axis=0)
    real_score_finite = np.isfinite(real_phase_stack)
    real_phase_max_score = np.nanmax(real_phase_stack, axis=0)
    all_zero_score_mask = np.nan_to_num(real_phase_max_score, nan=-np.inf) <= 0
    no_valid_score_mask = all_zero_score_mask | (~np.isfinite(real_phase_max_score))
    max_score_expanded = real_phase_max_score[None, :, :]
    argmax_tie_mask = (np.sum(np.isclose(real_phase_stack, max_score_expanded, rtol=1e-6, atol=1e-12) & real_score_finite, axis=0) > 1)
    valid_score_mask = ~no_valid_score_mask

    if real_phase_stack.shape[0] >= 2:
        sorted_real_scores = np.sort(real_phase_stack, axis=0)
        real_score_margin = sorted_real_scores[-1] - sorted_real_scores[-2]
    else:
        real_score_margin = np.full_like(real_best_score, np.nan, dtype=np.float32)
    real_score_margin[no_valid_score_mask] = np.nan

    plot_index_map(real_best_phase_index, real_phase_names, "Ti-only best candidate phase", "phase_map_real_ti_only_best_candidate.png",save_clean=True)
    plot_scalar_map(real_best_score, "Ti-only best phase score", "phase_map_real_best_score.png")
    plot_scalar_map(real_score_margin, "Ti-only best-second score margin", "phase_map_real_score_margin.png")
    plot_histogram(real_best_score, "Ti-only best phase score", "hist_real_best_phase_score.png")
    plot_histogram(real_score_margin, "Ti-only score margin", "hist_real_phase_score_margin.png")

    # Winning axis inside winning real phase.
    winning_axis_global_index = np.zeros_like(real_best_phase_index, dtype=np.int32)
    winning_axis_labels = []
    label_to_global = {}
    for pidx, phase_name in enumerate(real_phase_names):
        mask = (real_best_phase_index == pidx) & valid_score_mask
        local_axis_map = real["phase_best_axis_index_maps"][phase_name]
        local_labels = real["phase_axis_labels"][phase_name]
        for local_idx, axis_label in enumerate(local_labels):
            label = f"{phase_name}_{axis_label}"
            if label not in label_to_global:
                label_to_global[label] = len(winning_axis_labels)
                winning_axis_labels.append(label)
            winning_axis_global_index[mask & (local_axis_map == local_idx)] = label_to_global[label]
    if np.any(no_valid_score_mask):
        label_to_global["NO_VALID_MATCH"] = len(winning_axis_labels)
        winning_axis_labels.append("NO_VALID_MATCH")
        winning_axis_global_index[no_valid_score_mask] = label_to_global["NO_VALID_MATCH"]
    plot_index_map(winning_axis_global_index, winning_axis_labels, "Winning Ti phase and fiber axis", "phase_map_real_winning_axis.png")

    # Control score map: controls do not enter the final phase map.
    if len(control["phase_names"]) > 0:
        control_status = "RUN"
        control_phase_names = control["phase_names"]
        control_phase_stack = np.stack([control["phase_score_maps"][name] for name in control_phase_names], axis=0)
        control_best_index = np.nanargmax(control_phase_stack, axis=0)
        control_best_score = np.nanmax(control_phase_stack, axis=0)
        plot_index_map(control_best_index, control_phase_names, "Best negative/control phase", "qc_control_best_phase.png")
        plot_scalar_map(control_best_score, "Best negative/control score", "qc_control_best_score.png")
        control_minus_real = control_best_score - real_best_score
        plot_scalar_map(control_minus_real, "Control best score - real Ti best score", "qc_control_minus_real_score.png")
        plot_histogram(control_minus_real, "Control - real score", "hist_control_minus_real_score.png")
        control_failure_mask = control_best_score > (real_best_score + CONTROL_FAIL_MARGIN)
    else:
        control_status = "FAILED" if args.run_control else "NOT_RUN"
        control_phase_names = []
        control_best_score = np.full_like(real_best_score, np.nan, dtype=np.float32)
        control_minus_real = np.full_like(real_best_score, np.nan, dtype=np.float32)
        control_failure_mask = np.zeros_like(real_best_phase_index, dtype=bool)

    # Confidence masks.
    ambiguous_mask = ((real_score_margin < MARGIN_THRESHOLD) | (real_best_score < MIN_BEST_SCORE)) & valid_score_mask
    if low_peak_mask_base is not None:
        low_peak_mask = low_peak_mask_base.copy()
    else:
        low_peak_mask = np.zeros_like(real_best_phase_index, dtype=bool)
    if mixed_peak_mask_base is not None:
        mixed_peak_mask = mixed_peak_mask_base.copy()
    else:
        mixed_peak_mask = np.zeros_like(real_best_phase_index, dtype=bool)

    # High confidence means: real phase has margin, sufficient clean peaks, not diffuse/mixed, and control does not beat it.
    high_confidence_mask = valid_score_mask & (~ambiguous_mask) & (~low_peak_mask) & (~mixed_peak_mask) & (~control_failure_mask)

    failure_reason_map = np.full(real_best_phase_index.shape, "PASS", dtype=object)
    failure_reason_map[ambiguous_mask] = "AMBIGUOUS_LOW_MARGIN"
    failure_reason_map[mixed_peak_mask] = "FAILED_MIXED_DIFFUSE"
    failure_reason_map[low_peak_mask] = "FAILED_LOW_PEAK"
    failure_reason_map[control_failure_mask] = "FAILED_CONTROL"
    failure_reason_map[no_valid_score_mask] = "FAILED_NO_VALID_SCORE"

    def mask_fraction(mask):
        return float(np.sum(mask) / total_pixels)


    failure_reason_fraction = {
        reason: float(np.sum(failure_reason_map == reason) / total_pixels)
        for reason in ["PASS", "FAILED_NO_VALID_SCORE", "FAILED_LOW_PEAK", "FAILED_MIXED_DIFFUSE", "FAILED_CONTROL", "AMBIGUOUS_LOW_MARGIN"]
    }

    plot_scalar_map(ambiguous_mask.astype(np.float32), f"Ambiguous real Ti mask, margin < {MARGIN_THRESHOLD}", "phase_map_ambiguous_mask.png", cmap="gray")
    plot_scalar_map(low_peak_mask.astype(np.float32), f"Low peak count mask, n < {PEAK_COUNT_THRESHOLD}", "phase_map_low_peak_mask.png", cmap="gray")
    plot_scalar_map(mixed_peak_mask.astype(np.float32), f"Mixed/diffuse mask, clean peaks > {MAX_CLEAN_PEAKS_FOR_SINGLE}", "phase_map_mixed_diffuse_mask.png", cmap="gray")
    plot_scalar_map(control_failure_mask.astype(np.float32), "Negative-control failure mask", "qc_control_failure_mask.png", cmap="gray")
    plot_scalar_map(high_confidence_mask.astype(np.float32), "Final high-confidence mask", "phase_map_high_confidence_mask.png", cmap="gray")
    plot_phase_map_with_masks(
        real_best_phase_index,
        real_phase_names,
        invalid_mask=low_peak_mask,
        mixed_mask=mixed_peak_mask,
        ambiguous_mask=ambiguous_mask,
        control_fail_mask=control_failure_mask,
        filename="phase_map_real_ti_only_qc_masked.png",
        no_valid_mask=no_valid_score_mask,
    )


    def save_roi_review_candidates(filename_csv, filename_json, max_per_group=100):
        """Export representative pixels for second-pass pyxem/Stage-2B review."""
        rows = []
        seen = set()

        def add_points(category, mask, sort_arr=None, descending=True, limit=max_per_group):
            coords = np.argwhere(np.asarray(mask, dtype=bool))
            if coords.size == 0:
                return
            if sort_arr is not None:
                values = np.asarray([sort_arr[x, y] for x, y in coords], dtype=np.float64)
                order = np.argsort(values)
                if descending:
                    order = order[::-1]
                coords = coords[order]
            for x, y in coords[:limit]:
                key = (category, int(x), int(y))
                if key in seen:
                    continue
                seen.add(key)
                pidx = int(real_best_phase_index[x, y])
                if no_valid_score_mask[x, y]:
                    phase = "NO_VALID_MATCH"
                    axis_label = "NO_VALID_MATCH"
                else:
                    phase = real_phase_names[pidx] if 0 <= pidx < len(real_phase_names) else ""
                    axis_idx = int(winning_axis_global_index[x, y]) if winning_axis_global_index.size else -1
                    axis_label = winning_axis_labels[axis_idx] if 0 <= axis_idx < len(winning_axis_labels) else ""
                rows.append({
                    "category": category,
                    "scan_x": int(x),
                    "scan_y": int(y),
                    "best_phase": phase,
                    "winning_axis": axis_label,
                    "best_score": float(real_best_score[x, y]),
                    "score_margin": float(real_score_margin[x, y]),
                    "control_minus_real": float(control_minus_real[x, y]) if np.isfinite(control_minus_real[x, y]) else None,
                    "control_status": control_status,
                    "raw_peak_count": None if peak_count_map is None else int(peak_count_map[x, y]),
                    "clean_peak_count": None if np.size(clean_peak_count_map) == 0 else int(clean_peak_count_map[x, y]),
                    "strong_peak_count": None if np.size(strong_peak_count_map) == 0 else int(strong_peak_count_map[x, y]),
                    "ambiguous": bool(ambiguous_mask[x, y]),
                    "low_peak": bool(low_peak_mask[x, y]),
                    "mixed_diffuse": bool(mixed_peak_mask[x, y]),
                    "control_failure": bool(control_failure_mask[x, y]),
                    "high_confidence": bool(high_confidence_mask[x, y]),
                    "failure_reason": str(failure_reason_map[x, y]),
                })

        for pidx, phase_name in enumerate(real_phase_names):
            add_points(
                f"high_confidence_{phase_name}",
                high_confidence_mask & (real_best_phase_index == pidx),
                sort_arr=real_score_margin,
                descending=True,
                limit=25,
            )

        add_points("low_margin", ambiguous_mask, sort_arr=real_score_margin, descending=False)
        add_points("no_valid_match", no_valid_score_mask, sort_arr=real_best_score, descending=False)
        if "Ti-hcp" in real_phase_names:
            hcp_idx = real_phase_names.index("Ti-hcp")
            add_points(
                "suspicious_hcp",
                (real_best_phase_index == hcp_idx) & (~high_confidence_mask),
                sort_arr=real_best_score,
                descending=True,
            )
        add_points("control_failure", control_failure_mask, sort_arr=control_minus_real, descending=True)

        csv_path = OUT_DIR / filename_csv
        json_path = OUT_DIR / filename_json
        fieldnames = [
            "category", "scan_x", "scan_y", "best_phase", "winning_axis", "best_score",
            "score_margin", "control_minus_real", "control_status", "raw_peak_count", "clean_peak_count",
            "strong_peak_count", "ambiguous", "low_peak", "mixed_diffuse",
            "control_failure", "high_confidence", "failure_reason",
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[saved] {csv_path}")
        save_json(json_path, {"data_file": str(DATA_FILE), "rows": rows})
        return rows


    roi_review_candidates = save_roi_review_candidates("roi_review_candidates.csv", "roi_review_candidates.json")


    # =============================================================================
    # 7. Save arrays and summaries
    # =============================================================================

    total_pixels = int(np.prod(real_best_phase_index.shape))

    def build_top_orientation_candidates(group, top_n):
        candidate_maps = []
        candidate_phase_labels = []
        candidate_branch_labels = []
        candidate_match_indices = []
        candidate_axis_labels = []
        for branch_name, stack in group["branch_score_stacks"].items():
            arr = np.asarray(stack, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            for match_idx in range(arr.shape[2]):
                candidate_maps.append(arr[:, :, match_idx])
                candidate_phase_labels.append(group["branch_phase_names"].get(branch_name, ""))
                candidate_branch_labels.append(branch_name)
                candidate_match_indices.append(match_idx)
                candidate_axis_labels.append(str(group["branch_axes"].get(branch_name, "")))
        if not candidate_maps:
            shape = real_best_score.shape
            return {
                "scores": np.empty((0,) + shape, dtype=np.float32),
                "indices": np.empty((0,) + shape, dtype=np.int16),
            "phase_labels": np.array([], dtype=str),
            "branch_labels": np.array([], dtype=str),
            "axis_labels": np.array([], dtype=str),
                "match_indices": np.array([], dtype=np.int16),
                "score_margin": np.full(shape, np.nan, dtype=np.float32),
            }
        stack = np.stack(candidate_maps, axis=0)
        n = min(top_n, stack.shape[0])
        order = np.flip(np.argsort(stack, axis=0), axis=0)[:n]
        scores = np.take_along_axis(stack, order, axis=0)
        if n >= 2:
            score_margin = scores[0] - scores[1]
        else:
            score_margin = np.full_like(scores[0], np.nan, dtype=np.float32)
        return {
            "scores": scores.astype(np.float32),
            "indices": order.astype(np.int16),
            "phase_labels": np.asarray(candidate_phase_labels, dtype=str),
            "branch_labels": np.asarray(candidate_branch_labels, dtype=str),
            "axis_labels": np.asarray(candidate_axis_labels, dtype=str),
            "match_indices": np.asarray(candidate_match_indices, dtype=np.int16),
            "score_margin": score_margin.astype(np.float32),
        }


    def summarize_top_candidate_phases(top_candidates, phase_names):
        if top_candidates["indices"].size == 0:
            return {}
        top0 = top_candidates["indices"][0]
        labels = top_candidates["phase_labels"]
        valid = valid_score_mask
        total = int(np.sum(valid))
        if total == 0:
            return {phase: 0.0 for phase in phase_names}
        return {
            phase: float(np.sum(labels[top0[valid]] == phase) / total)
            for phase in phase_names
        }


    def classify_bcc_hcp_distinguishability():
        high_fraction = float(np.sum(high_confidence_mask) / total_pixels)
        median_margin = finite_stat(real_score_margin, np.median)
        top_phase_fraction = summarize_top_candidate_phases(top_orientation_candidates, real_phase_names)
        bcc_high = 0.0
        hcp_high = 0.0
        if "Ti-bcc" in real_phase_names:
            bcc_idx = real_phase_names.index("Ti-bcc")
            bcc_high = float(np.sum(high_confidence_mask & (real_best_phase_index == bcc_idx)) / total_pixels)
        if "Ti-hcp" in real_phase_names:
            hcp_idx = real_phase_names.index("Ti-hcp")
            hcp_high = float(np.sum(high_confidence_mask & (real_best_phase_index == hcp_idx)) / total_pixels)

        no_valid_fraction = float(np.sum(no_valid_score_mask) / total_pixels)
        if high_fraction < 0.05 or no_valid_fraction > 0.5:
            conclusion = "INSUFFICIENT_QC"
        elif median_margin is None or median_margin < MARGIN_THRESHOLD:
            conclusion = "AMBIGUOUS"
        elif abs(bcc_high - hcp_high) < 0.05 and min(bcc_high, hcp_high) > 0.05:
            conclusion = "AMBIGUOUS"
        else:
            conclusion = "DISTINGUISHABLE"

        return {
            "conclusion": conclusion,
            "high_confidence_fraction": high_fraction,
            "median_score_margin": median_margin,
            "top1_phase_fraction": top_phase_fraction,
            "high_confidence_fraction_ti_bcc": bcc_high,
            "high_confidence_fraction_ti_hcp": hcp_high,
            "control_status": control_status,
            "control_failure_fraction": float(np.sum(control_failure_mask) / total_pixels),
            "no_valid_score_fraction": no_valid_fraction,
            "all_zero_score_fraction": float(np.sum(all_zero_score_mask) / total_pixels),
        }

    real_branch_names = list(real["branch_score_maps"].keys())
    real_branch_stack = np.stack([real["branch_score_maps"][name] for name in real_branch_names], axis=0)
    top_n = min(TOP_CANDIDATES_TO_SAVE, len(real_branch_names))
    top_branch_indices = np.flip(np.argsort(real_branch_stack, axis=0), axis=0)[:top_n]
    top_branch_scores = np.take_along_axis(real_branch_stack, top_branch_indices, axis=0)
    top_orientation_candidates = build_top_orientation_candidates(real, TOP_CANDIDATES_TO_SAVE)
    distinguishability_summary = classify_bcc_hcp_distinguishability()

    if len(control["branch_score_maps"]) > 0:
        control_branch_names = list(control["branch_score_maps"].keys())
        control_branch_stack = np.stack([control["branch_score_maps"][name] for name in control_branch_names], axis=0)
    else:
        control_branch_names = []
        control_branch_stack = np.empty((0,) + real_best_score.shape, dtype=np.float32)

    np.savez_compressed(
        OUT_DIR / "phase_orientation_scores_v6_optimized.npz",
        real_branch_names=np.array(real_branch_names),
        real_branch_phases=np.array([real["branch_phase_names"][name] for name in real_branch_names]),
        real_branch_axes=np.array([str(real["branch_axes"][name]) for name in real_branch_names]),
        real_branch_score_stack=real_branch_stack,
        top_branch_indices=top_branch_indices,
        top_branch_scores=top_branch_scores,
        top_branch_labels=np.array(real_branch_names),
        top_orientation_candidate_indices=top_orientation_candidates["indices"],
        top_orientation_candidate_scores=top_orientation_candidates["scores"],
        top_orientation_candidate_phase_labels=top_orientation_candidates["phase_labels"],
        top_orientation_candidate_branch_labels=top_orientation_candidates["branch_labels"],
        top_orientation_candidate_axis_labels=top_orientation_candidates["axis_labels"],
        top_orientation_candidate_match_indices=top_orientation_candidates["match_indices"],
        top_orientation_score_margin=top_orientation_candidates["score_margin"],
        real_phase_names=np.array(real_phase_names),
        real_phase_score_stack=real_phase_stack,
        real_best_phase_index=real_best_phase_index,
        real_best_score=real_best_score,
        real_score_margin=real_score_margin,
        no_valid_score_mask=no_valid_score_mask,
        all_zero_score_mask=all_zero_score_mask,
        argmax_tie_mask=argmax_tie_mask,
        failure_reason_map=failure_reason_map.astype(str),
        winning_axis_labels=np.array(winning_axis_labels),
        winning_axis_global_index=winning_axis_global_index,
        control_branch_names=np.array(control_branch_names),
        control_branch_phases=np.array([control["branch_phase_names"].get(name, "") for name in control_branch_names]),
        control_branch_axes=np.array([str(control["branch_axes"].get(name, "")) for name in control_branch_names]),
        control_branch_score_stack=control_branch_stack,
        control_phase_names=np.array(control_phase_names),
        control_status=np.array(control_status),
        control_best_score=control_best_score,
        control_minus_real=control_minus_real,
        ambiguous_mask=ambiguous_mask,
        low_peak_mask=low_peak_mask,
        mixed_peak_mask=mixed_peak_mask,
        control_failure_mask=control_failure_mask,
        high_confidence_mask=high_confidence_mask,
        peak_count_map=np.array([]) if peak_count_map is None else peak_count_map,
        clean_peak_count_map=clean_peak_count_map,
        strong_peak_count_map=strong_peak_count_map,
        q_median_map=q_median_map,
        q_p90_map=q_p90_map,
        roi_review_candidate_count=np.array(len(roi_review_candidates), dtype=np.int32),
        margin_threshold=np.array(MARGIN_THRESHOLD, dtype=np.float32),
        peak_count_threshold=np.array(PEAK_COUNT_THRESHOLD, dtype=np.int16),
        max_clean_peaks_for_single=np.array(MAX_CLEAN_PEAKS_FOR_SINGLE, dtype=np.int16),
        control_fail_margin=np.array(CONTROL_FAIL_MARGIN, dtype=np.float32),
        orientation_mode=np.array(ORIENTATION_MODE),
        num_matches_return=np.array(NUM_MATCHES_RETURN, dtype=np.int16),
    )
    print(f"[saved] {OUT_DIR / 'phase_orientation_scores_v6_optimized.npz'}")

    # JSON summary.
    total_pixels = int(np.prod(real_best_phase_index.shape))
    real_phase_results = []
    for pidx, phase_name in enumerate(real_phase_names):
        score = real["phase_score_maps"][phase_name]
        raw_win = (real_best_phase_index == pidx) & valid_score_mask
        high_conf_win = raw_win & high_confidence_mask
        real_phase_results.append({
            "phase": phase_name,
            "branches": real["phase_to_branch_names"].get(phase_name, []),
            "score_mean": float(np.nanmean(score)),
            "score_median": float(np.nanmedian(score)),
            "score_p95": float(np.nanpercentile(score, 95)),
            "winning_fraction_raw_ti_only": float(np.sum(raw_win) / total_pixels),
            "winning_fraction_raw_ti_only_valid_scores_only": float(np.sum(raw_win) / max(int(np.sum(valid_score_mask)), 1)),
            "winning_fraction_after_all_qc_masks": float(np.sum(high_conf_win) / total_pixels),
        })

    control_phase_results = []
    for phase_name in control_phase_names:
        score = control["phase_score_maps"][phase_name]
        control_phase_results.append({
            "phase": phase_name,
            "branches": control["phase_to_branch_names"].get(phase_name, []),
            "score_mean": float(np.nanmean(score)),
            "score_median": float(np.nanmedian(score)),
            "score_p95": float(np.nanpercentile(score, 95)),
            "control_beats_real_fraction": float(np.sum(score > (real_best_score + CONTROL_FAIL_MARGIN)) / total_pixels),
        })

    summary = {
        "settings": {
            "data_file": str(DATA_FILE),
            "out_dir": str(OUT_DIR),
            "analysis_root": str(ROOT),
            "cif_dir": str(CIF_DIR),
            "output_tag": args.output_tag,
            "mode": args.mode,
            "screening_mode": "fiber_axis_only",
            "orientation_mode": ORIENTATION_MODE,
            "num_matches_return": NUM_MATCHES_RETURN,
            "status_interval_seconds": STATUS_INTERVAL,
            "run_control": bool(args.run_control),
            "control_status": control_status,
            "k_max": K_MAX,
            "inv_ang_per_pixel": INV_ANG_PER_PIXEL,
            "calibration": CALIBRATION_SUMMARY,
            "calibration_status": calibration_status,
            "peak_detection_diagnostics": peak_detection_diagnostics,
            "angle_step_zone_axis": ANGLE_STEP_ZONE_AXIS,
            "angle_step_in_plane": ANGLE_STEP_IN_PLANE,
            "detect_params": DETECT_PARAMS,
            "q_space_diagnostics": EXP_Q_SUMMARY,
            "q_min_for_qc": Q_MIN_FOR_QC,
            "q_max_for_qc": Q_MAX_FOR_QC,
            "direct_beam_mask_radius": direct_beam_mask_radius,
            "strong_peak_percentile": STRONG_PEAK_PERCENTILE,
            "margin_threshold": MARGIN_THRESHOLD,
            "min_best_score": MIN_BEST_SCORE,
            "peak_count_threshold": PEAK_COUNT_THRESHOLD,
            "min_strong_peaks_for_match": MIN_STRONG_PEAKS_FOR_MATCH,
            "max_clean_peaks_for_single": MAX_CLEAN_PEAKS_FOR_SINGLE,
            "match_radius_q": MATCH_RADIUS_Q,
            "control_fail_margin": CONTROL_FAIL_MARGIN,
            "bragg_cache_tag": BRAGG_CACHE_TAG,
            "bragg_cache_path": str(bragg_cache),
            "bragg_cache_status": bragg_cache_status,
            "real_candidate_phases": [p["name"] for p in REAL_CANDIDATE_PHASES],
            "control_phases": [p["name"] for p in CONTROL_PHASES],
            "real_candidate_fiber_axes": {p["name"]: p.get("fiber_axes", []) for p in REAL_CANDIDATE_PHASES},
            "control_fiber_axes": {p["name"]: p.get("fiber_axes", []) for p in CONTROL_PHASES},
        },
    "confidence_summary": {
            "ambiguous_fraction_real_margin_or_score": float(np.sum(ambiguous_mask) / total_pixels),
            "low_peak_fraction": float(np.sum(low_peak_mask) / total_pixels),
            "mixed_diffuse_fraction": float(np.sum(mixed_peak_mask) / total_pixels),
            "control_failure_fraction": float(np.sum(control_failure_mask) / total_pixels),
            "final_high_confidence_fraction": float(np.sum(high_confidence_mask) / total_pixels),
            "real_score_margin_mean": float(np.nanmean(real_score_margin)),
            "real_score_margin_median": float(np.nanmedian(real_score_margin)),
            "real_score_margin_p95": float(np.nanpercentile(real_score_margin, 95)),
            "clean_peak_count_median": None if np.size(clean_peak_count_map) == 0 else float(np.nanmedian(clean_peak_count_map)),
            "strong_peak_count_median": None if np.size(strong_peak_count_map) == 0 else float(np.nanmedian(strong_peak_count_map)),
            "control_minus_real_mean": finite_stat(control_minus_real, np.mean),
            "control_minus_real_median": finite_stat(control_minus_real, np.median),
            "no_valid_score_fraction": mask_fraction(no_valid_score_mask),
            "all_zero_score_fraction": mask_fraction(all_zero_score_mask),
            "argmax_tie_fraction": mask_fraction(argmax_tie_mask),
            "failure_reason_fraction": failure_reason_fraction,
        "roi_review_candidate_count": int(len(roi_review_candidates)),
    },
    "top_orientation_candidate_summary": {
        "top_n": int(top_orientation_candidates["scores"].shape[0]),
        "top1_phase_fraction": summarize_top_candidate_phases(top_orientation_candidates, real_phase_names),
        "top_orientation_score_margin_median": finite_stat(top_orientation_candidates["score_margin"], np.median),
        "top_orientation_score_margin_p95": finite_stat(top_orientation_candidates["score_margin"], lambda v: np.percentile(v, 95)),
    },
    "distinguishability_summary": distinguishability_summary,
    "real_branch_results": real["branch_results"],
        "control_branch_results": control["branch_results"],
        "real_phase_results_aggregated_over_axes": real_phase_results,
        "control_phase_results_aggregated_over_axes": control_phase_results,
    }
    summary_path = OUT_DIR / "phase_summary_v6_optimized.json"
    save_json(summary_path, summary)
    generate_phase_orientation_report(OUT_DIR, summary_path=summary_path)

    print("\nReal branch score summary:")
    for r in real["branch_results"]:
        status = r.get("branch_status", "RUN")
        reason = r.get("failure_reason")
        print(
            f"  {r['branch']}: status={status}, "
            f"mean={format_summary_stat(r.get('score_mean'))}, "
            f"median={format_summary_stat(r.get('score_median'))}, "
            f"p95={format_summary_stat(r.get('score_p95'))}"
            + (f", reason={reason}" if reason else "")
        )

    print("\nControl branch score summary:")
    for r in control["branch_results"]:
        status = r.get("branch_status", "RUN")
        reason = r.get("failure_reason")
        print(
            f"  {r['branch']}: status={status}, "
            f"mean={format_summary_stat(r.get('score_mean'))}, "
            f"median={format_summary_stat(r.get('score_median'))}, "
            f"p95={format_summary_stat(r.get('score_p95'))}"
            + (f", reason={reason}" if reason else "")
        )

    print("\nReal Ti-only aggregated phase summary:")
    for r in real_phase_results:
        print(
            f"  {r['phase']}: mean={r['score_mean']:.4g}, median={r['score_median']:.4g}, "
            f"p95={r['score_p95']:.4g}, raw_win={r['winning_fraction_raw_ti_only']:.3f}, "
            f"qc_high_conf_win={r['winning_fraction_after_all_qc_masks']:.3f}"
        )

    print("\nControl QC summary:")
    for r in control_phase_results:
        print(
            f"  {r['phase']}: mean={r['score_mean']:.4g}, median={r['score_median']:.4g}, "
            f"p95={r['score_p95']:.4g}, beats_real={r['control_beats_real_fraction']:.3f}"
        )

    print("\nFinal confidence summary:")
    print(f"  margin_threshold = {MARGIN_THRESHOLD}")
    print(f"  peak_count_threshold = {PEAK_COUNT_THRESHOLD}")
    print(f"  min_strong_peaks_for_match = {MIN_STRONG_PEAKS_FOR_MATCH}")
    print(f"  max_clean_peaks_for_single = {MAX_CLEAN_PEAKS_FOR_SINGLE}")
    print(f"  control_fail_margin = {CONTROL_FAIL_MARGIN}")
    print(f"  ambiguous_fraction_real_margin_or_score = {np.sum(ambiguous_mask) / total_pixels:.3f}")
    print(f"  low_peak_fraction = {np.sum(low_peak_mask) / total_pixels:.3f}")
    print(f"  mixed_diffuse_fraction = {np.sum(mixed_peak_mask) / total_pixels:.3f}")
    print(f"  control_failure_fraction = {np.sum(control_failure_mask) / total_pixels:.3f}")
    print(f"  final_high_confidence_fraction = {np.sum(high_confidence_mask) / total_pixels:.3f}")
    print(f"  real_score_margin median = {np.nanmedian(real_score_margin):.4g}")
    control_median = finite_stat(control_minus_real, np.median)
    print(f"  control_minus_real median = {'NA' if control_median is None else f'{control_median:.4g}'}")


    # =============================================================================
    # 8. Optional strain on globally best real branch
    # =============================================================================

    if RUN_STRAIN_FOR_GLOBALLY_BEST_REAL_BRANCH and len(real["branch_results"]) > 0:
        best_branch_name = max(real["branch_results"], key=lambda r: r["score_median"])["branch"]
        print(f"Calculating strain for globally best real branch: {best_branch_name}")
        crystal = real["branch_crystals"][best_branch_name]
        orientation_map = real["branch_orientation_maps"][best_branch_name]
        strain_map = crystal.calculate_strain(bragg_peaks, orientation_map, rotation_range=np.pi / 3)
        py4DSTEM.visualize.show_strain(
            strain_map,
            vrange_exx=[-1.0, 1.0],
            vrange_theta=[0.0, 60.0],
            figsize=(6, 6),
        )
        savefig(f"strain_{best_branch_name}.png")

    print("Done.")

    return 0
