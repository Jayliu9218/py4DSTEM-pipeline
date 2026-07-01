# Phase/Orientation Screening

This folder contains the py4DSTEM Ti phase/orientation screening workflow with
WS2 negative-control QC. The original long script has been split into a local
Python package while keeping the same command-line entrypoint:

```powershell
conda activate 4dstem
python scripts\phase_orientation_screening\main.py --data-file 1_0_64_0_64.h5
```

If the prompt already shows `(4dstem)`, run `python` directly instead of
wrapping it in `conda run`; this streams status messages more reliably:

```powershell
python scripts\phase_orientation_screening\main.py --data-file D:\Data\4dstem\exp\0617-4d\crop\1_0_64_0_64.h5 --mode coarse
```

If you prefer `conda run`, add `--live-stream` so long py4DSTEM steps print
heartbeat messages while they are running:

```powershell
conda run --live-stream -n 4dstem python scripts\phase_orientation_screening\main.py --data-file D:\Data\4dstem\exp\0617-4d\crop\1_0_64_0_64.h5 --mode coarse
```

Every run also writes a status log immediately after path resolution:

```text
<output-directory>\run_status.log
```

You can confirm path resolution and status-log creation without starting the
expensive py4DSTEM pipeline:

```powershell
python scripts\phase_orientation_screening\main.py --data-file D:\Data\4dstem\exp\0617-4d\crop\1_0_64_0_64.h5 --mode coarse --dry-run
```

If the terminal is quiet because a parent process is buffering output, watch the
log file in another PowerShell:

```powershell
Get-Content D:\Data\4dstem\exp\0617-4d\1_0_64_0_64\coarse_k1p4_za2p0_ip8p0_control\run_status.log -Wait
```

The script is intended for screening, not final crystallographic proof. Treat
phase/orientation labels as candidates unless Bragg peak QC, score margins,
negative controls, and single-pattern overlays are all physically consistent.

## Layout

- `main.py` is the stable CLI wrapper.
- `modules/pipeline.py` contains the procedural workflow.
- `modules/reporting.py` writes the final Markdown and HTML
  reports.
- `modules/cli.py`, `config.py`, `bragg_qc.py`,
  `orientation.py`, `plotting.py`, `aggregation.py`, and `outputs.py` hold
  focused helpers for configuration, QC, plotting, aggregation, and summaries.
- `merge_orientation_phase_maps.bat` is the existing batch helper.

## Common Runs

Coarse full-map screening:

```powershell
conda run -n 4dstem python scripts\phase_orientation_screening\main.py --data-file 1_0_64_0_64.h5 --mode coarse
```

By default, the workflow uses the faster legacy fixed fiber-axis branch mode.
Use S2 only when you intentionally want a slower full-sphere orientation search:

```powershell
conda run -n 4dstem python scripts\phase_orientation_screening\main.py --data-file 1_0_64_0_64.h5 --orientation-mode s2 --num-matches-return 5
```

Progress bars are visible by default and forced to ASCII to avoid garbled
Windows console text. If you prefer log-only output, add `--quiet-progress`.
Status heartbeat messages are off by default; enable them with
`--status-interval 30`.

Explicit fixed fiber-axis branch mode:

```powershell
conda run -n 4dstem python scripts\phase_orientation_screening\main.py --data-file 1_0_64_0_64.h5 --orientation-mode fiber
```

Absolute data paths are supported. When an absolute HDF5 path is supplied, the
workflow looks upward from that file for `Ti-bcc.cif` and `Ti-hcp.cif` and uses
that directory as the default analysis root:

```powershell
conda run -n 4dstem python scripts\phase_orientation_screening\main.py --data-file D:\Data\4dstem\exp\0617-4d\crop\1_0_64_0_64.h5 --mode coarse
```

Use `--analysis-root` to choose where run outputs and Bragg caches are written,
and `--cif-dir` if the CIF files live somewhere else:

```powershell
conda run -n 4dstem python scripts\phase_orientation_screening\main.py --data-file D:\Data\4dstem\exp\0617-4d\crop\1_0_64_0_64.h5 --analysis-root D:\Data\4dstem\exp\0617-4d --cif-dir D:\Data\4dstem\exp\0617-4d
```

Fine ROI/refinement-style screening:

```powershell
conda run -n 4dstem python scripts\phase_orientation_screening\main.py --data-file 1_0_64_0_64.h5 --mode fine
```

K-max sweep:

```powershell
conda run -n 4dstem python scripts\phase_orientation_screening\main.py --data-file 1_0_64_0_64.h5 --k-max-sweep 1.4,1.7,2.0
```

Single branch run:

```powershell
conda run -n 4dstem python scripts\phase_orientation_screening\main.py --data-file 1_0_64_0_64.h5 --branch-only --phase Ti-bcc --fiber-axis 0,1,1
```

Aggregate branch outputs:

```powershell
conda run -n 4dstem python scripts\phase_orientation_screening\main.py --data-file 1_0_64_0_64.h5 --aggregate-branches D:\path\to\branch_outputs
```

## Outputs

Runs write into:

```text
D:\Workspace\large-4dstem-analysis\data\0617-4d\<data-file-stem>\<output-tag>\
```

Important outputs include:

- `phase_summary_v6_optimized.json`: machine-readable settings, QC, branch, and
  phase summaries, including calibration status, control status, top-5 candidate
  summary, invalid-score fractions, and Ti-bcc/Ti-hcp distinguishability.
- `phase_orientation_scores_v6_optimized.npz`: score maps, masks, phase labels,
  branch labels, top branch arrays, and top orientation candidate arrays.
- `phase_orientation_report.md`: human-readable run report.
- `phase_orientation_report.html`: browser-friendly report with linked figures.
- `roi_review_candidates.csv` and `.json`: representative coordinates for
  high-confidence, ambiguous, suspicious, and control-failure review.
- `phase_map_real_ti_only_best_candidate*.png`: Ti-only candidate phase map.
- `phase_map_real_ti_only_qc_masked.png`: final QC-masked phase map.
- `phase_map_real_winning_axis.png`: winning Ti phase and fiber-axis map.
- `qc_*` and `hist_*` figures: Bragg, calibration, control, and distribution QC.
- `qc_dp_mean_raw.png`, `qc_dp_mean_processed.png`, and
  `qc_dp_mean_detected_peak_overlay.png`: raw/processed/detected peak diagnostics
  for peak finding. Matching single-frame diagnostics are saved with
  `qc_single_x*_y*_*` names.

## Debugging All-Zero Scores

The workflow now blocks all-zero scores from becoming a fake phase assignment.
Pixels with non-finite or all-zero phase scores are reported as:

```text
NO_VALID_MATCH / FAILED_NO_VALID_SCORE
```

The JSON/report include:

- `no_valid_score_fraction`
- `all_zero_score_fraction`
- `argmax_tie_fraction`
- `failure_reason_fraction`

If scores are all zero or `low_peak_fraction = 1.000`, debug one branch before
running the full map:

```powershell
python scripts\phase_orientation_screening\main.py `
  --data-file D:\Data\4dstem\exp\0617-4d\crop\1_0_64_0_64.h5 `
  --mode coarse `
  --branch-only `
  --phase Ti-bcc `
  --fiber-axis 0,1,1 `
  --min-strong-peaks-for-match 3 `
  --peak-count-threshold 3 `
  --max-clean-peaks-for-single 100 `
  --direct-beam-mask-radius 15 `
  --detect-min-relative-intensity 0.02 `
  --detect-min-peak-spacing 8 `
  --detect-max-num-peaks 100
```

First inspect:

```text
qc_dp_mean_raw.png
qc_dp_mean_processed.png
qc_dp_mean_detected_peak_overlay.png
qc_single_x*_y*_detected_peak_overlay.png
```

Then check branch diagnostics in `phase_summary_v6_optimized.json`:

- `branch_status`
- `failure_reason`
- `single_pattern_test_pixels`
- `n_exp_peaks_test`
- `n_clean_peaks_test`
- `n_template_reflections_before_filter`
- `n_template_reflections_after_kmax`
- `n_template_reflections`
- `q_min_template`, `q_median_template`, `q_max_template`
- `exp_q_min`, `exp_q_max`
- `template_q_min`, `template_q_max`
- `q_min_exp`, `q_median_exp`, `q_max_exp`
- `match_radius_q`
- `nearest_template_distance_min`, `nearest_template_distance_median`
- `matched_peak_count`
- `score_numerator`, `score_denominator`
- `score_max`

Branches with `branch_status = FAILED_NO_VALID_MATCH` are not aggregated into
phase maps. This is intentional: an all-zero or non-finite branch score is a
matching failure, not evidence for the first phase returned by `argmax`.

## Notes

- The default real candidates are Ti-bcc and Ti-hcp.
- WS2 is used only as a negative/control phase and does not participate in the
  final Ti phase map.
- Progress bars are disabled by default inside the script to avoid Windows
  console encoding issues in non-interactive batch runs.
- Existing CLI options and output filenames are intentionally preserved for
  compatibility with older runs and downstream scripts.
- This folder is ignored by the repository-level `.gitignore` rule `scripts/*`,
  so Git will not show these script changes unless the ignore rule is adjusted or
  files are force-added.
