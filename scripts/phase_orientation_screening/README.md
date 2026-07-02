# Phase/Orientation Screening

This folder contains the py4DSTEM Ti phase/orientation screening workflow with
required negative-control QC. The entrypoint is `main.py`, which delegates to a
self-contained `modules/pipeline.py` workflow plus a small
`modules/reporting.py` report builder:

```powershell
conda activate 4dstem
python scripts\phase_orientation_screening\main.py --data-file 1_0_64_0_64.h5
```

If the prompt already shows `(4dstem)`, run `python` directly instead of
wrapping it in `conda run`; this streams status messages more reliably:

```powershell
python scripts\phase_orientation_screening\main.py --data-file D:\Data\4dstem\exp\0617-4d\crop\1_0_64_0_64.h5 --mode coarse
```

For long py4DSTEM steps, pass `--status-interval 30` so the script writes
heartbeat messages to stdout and `run_status.log` while it works:

```powershell
conda run -n 4dstem python scripts\phase_orientation_screening\main.py --data-file D:\Data\4dstem\exp\0617-4d\crop\1_0_64_0_64.h5 --mode coarse --status-interval 30
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
The raw Ti-only winner map is a navigation aid only; interpret only the
QC-masked phase map and pixels in `high_confidence_mask`.

## Layout

- `main.py` is the stable CLI wrapper.
- `modules/pipeline.py` contains the procedural workflow (single, self-contained
  workflow file; all helpers are local to keep the script readable end-to-end).
- `modules/reporting.py` writes the final Markdown and HTML reports from a
  saved `phase_summary_v6_optimized.json`.
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
conda run -n 4dstem python scripts\phase_orientation_screening\main.py --data-file 1_0_64_0_64.h5 --branch-only --phase Ti-bcc --fiber-axis "0,1,1"
```

Aggregate branch outputs:

```powershell
conda run -n 4dstem python scripts\phase_orientation_screening\main.py --data-file 1_0_64_0_64.h5 --aggregate-branches D:\path\to\branch_outputs
```

## Outputs

Runs write into:

```text
<analysis-root>\<data-file-stem>\<output-tag>\
```

Important outputs include:

- `phase_summary_v6_optimized.json`: machine-readable settings, QC, branch, and
  phase summaries, including calibration status, control status, top-5 candidate
  summary, Bragg peak preflight status, invalid-score fractions, and
  Ti-bcc/Ti-hcp distinguishability.
- `phase_orientation_scores_v6_optimized.npz`: score maps, masks, phase labels,
  branch labels, top branch arrays, and top orientation candidate arrays.
- `phase_orientation_report.md`: human-readable run report.
- `phase_orientation_report.html`: browser-friendly report with linked figures.
- `roi_review_candidates.csv` and `.json`: representative coordinates for
  high-confidence, ambiguous, suspicious, and control-failure review.
- `phase_map_real_ti_only_best_candidate*.png`: raw Ti-only winner map for
  screening/navigation only, not scientific interpretation.
- `phase_map_real_ti_only_qc_masked.png`: final QC-masked phase map for
  interpretation when peak preflight and controls pass.
- `phase_map_real_winning_axis.png`: winning Ti phase and fiber-axis map.
- `qc_*` and `hist_*` figures: Bragg, calibration, control, and distribution QC.
- `qc_dp_mean_raw.png`, `qc_dp_mean_processed.png`, and
  `qc_dp_mean_detected_peak_overlay.png`: raw/processed/detected peak diagnostics
  for peak finding. Matching single-frame diagnostics are saved with
  `qc_single_x*_y*_*` names.

## Bragg Peak Preflight

Before full-map orientation matching, the workflow runs an automatic Bragg peak
preflight on representative probe positions unless explicit `--detect-*`
arguments are supplied. It sweeps a small grid around the default detection
parameters and selects the first setting that reaches:

- median strong QC peak count >= `--peak-preflight-target-strong-median`
  (default `5`)
- median clean QC peak count between `--peak-preflight-clean-median-min`
  (default `10`) and `--peak-preflight-clean-median-max` (default `20`)

The selected parameters and all attempts are recorded under
`settings.peak_preflight` in `phase_summary_v6_optimized.json`. If preflight status is
`FAILED_LOW_PEAK_USABILITY`, the run continues for screening diagnostics, but
validated high-confidence interpretation fails closed.

## Required Negative Controls

With `--run-control` enabled, the following required control CIFs must be
available in `--cif-dir`:

- `TiO2-rutile.cif`
- `TiO2-anatase.cif`
- `TiO.cif`
- `Ti-hcp-decoy.cif`
- `Ti-wrong-q-scale.cif`

These produce `TiO2-rutile-control`, `TiO2-anatase-control`, `TiO-control`,
`Ti-hcp-decoy-control`, and `Ti-wrong-q-scale-control` branches. `WS2.cif` is
still used as an optional legacy control if present. If any required control CIF
is missing, or if any required control phase produces no valid branch,
`control_status` is failed and high-confidence interpretation is blocked.

## Debugging All-Zero or No-Match Branches

The workflow blocks all-zero scores from becoming a fake phase assignment.
Pixels with non-finite or all-zero phase scores are reported as:

```text
NO_VALID_MATCH / FAILED_NO_VALID_SCORE
```

The JSON/report include:

- `no_valid_score_fraction`
- `all_zero_score_fraction`
- `argmax_tie_fraction`
- `failure_reason_fraction`

### Single-pattern QC gate (softened)

The single-pattern QC test runs on a few representative probe positions before
the full-map match. If none of these test pixels produce a valid orientation
match, the branch **no longer aborts**. Instead it prints a warning and
continues to full-map matching in screening mode. The full-map `score_max <= 0`
check still aborts genuinely dead branches with `branch_status =
FAILED_NO_VALID_MATCH` and `failure_reason = "full-map score max is
non-finite or <= 0"`.

Each branch result records:

- `single_pattern_qc_passed`: `true` if at least one test pixel matched.
- `single_pattern_test_pixels`, `single_pattern_test_diagnostics`: per-pixel
  diagnostics (see field list below).

### Root cause: q-calibration mismatch

The most common reason every test pixel fails is a q-calibration mismatch:
detected Bragg peaks land far outside the QC window `[q_min_for_qc,
q_max_for_qc]` and the template reflection range. Even with `calstate['pixel']
= True`, the calibrated vector getter may return pixel-unit coordinates when
the `--inv-ang-per-pixel` value is wrong for the dataset, so every peak sits at
q ~3-6 A^-1 while the template only reaches q ~1.4 A^-1.

Quick check: load the Bragg cache and inspect the q range of all detected
peaks. If `n_clean_peaks_test = 0` and `q_min_exp`/`q_max_exp` are `null` while
the raw peak q values are in the hundreds, the `inv_ang_per_pixel` value is
wrong by roughly that factor.

### Debug one branch before running the full map

```powershell
python scripts\phase_orientation_screening\main.py `
  --data-file D:\Data\4dstem\exp\0617-4d\crop\1_0_64_0_64.h5 `
  --mode coarse `
  --branch-only `
  --phase Ti-bcc `
  --fiber-axis "0,1,1" `
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
  - `single_pattern_qc_passed`
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

When the cause is a q-calibration mismatch, recompute the Bragg cache with a
corrected `--inv-ang-per-pixel` (or supply `--calibration-peaks` to fit it
from known Ti ring radii):

```powershell
python scripts\phase_orientation_screening\main.py `
  --data-file D:\Data\4dstem\exp\0617-4d\crop\1_0_64_0_64.h5 `
  --mode coarse --branch-only --phase Ti-bcc --fiber-axis "0,1,1" `
  --inv-ang-per-pixel 0.008 --force-recompute-bragg --status-interval 30
```

`--force-recompute-bragg` is required because the cached Bragg peaks were
detected and calibrated with the old (wrong) pixel size.

## Notes

- The default real candidates are Ti-bcc and Ti-hcp.
- Required controls are used only as negative/control phases and do not
  participate in the final Ti phase map.
- Progress bars are visible by default and forced to ASCII to avoid Windows
  console encoding issues. Use `--quiet-progress` for log-only batch runs.
- Existing CLI options and output filenames are intentionally preserved for
  compatibility with older runs and downstream scripts.
- This folder is tracked by Git (the old `scripts/*` ignore rule was removed).
