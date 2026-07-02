import json
import sys
from pathlib import Path

import numpy as np


SCREENING_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "phase_orientation_screening"
sys.path.insert(0, str(SCREENING_ROOT))

from modules.pipeline import (
    apply_control_validation_gate,
    classify_control_validation_status,
    select_peak_preflight_candidate,
)
from modules.reporting import generate_phase_orientation_report


def test_report_generation_from_synthetic_summary(tmp_path):
    summary = {
        "settings": {
            "data_file": "sample.h5",
            "out_dir": str(tmp_path),
            "mode": "coarse",
            "orientation_mode": "s2",
            "num_matches_return": 5,
            "run_control": True,
            "control_status": "RUN",
            "k_max": 1.4,
            "bragg_cache_status": "hit",
            "calibration_status": {
                "origin": "RUN",
                "ellipse": "RUN",
                "q_scale_mode": "provided_value_only",
            },
            "peak_detection_diagnostics": [
                {"label": "dp_mean", "peak_count": 12, "direct_beam_radius_px": 5.0, "background": "median_radial_subtraction"}
            ],
        },
        "confidence_summary": {
            "final_high_confidence_fraction": 0.5,
            "ambiguous_fraction_real_margin_or_score": 0.25,
        },
        "real_phase_results_aggregated_over_axes": [
            {
                "phase": "Ti-bcc",
                "score_median": 2.0,
                "score_p95": 3.0,
                "winning_fraction_raw_ti_only": 0.6,
                "winning_fraction_after_all_qc_masks": 0.4,
            }
        ],
        "control_phase_results_aggregated_over_axes": [
            {
                "phase": "WS2-control",
                "score_median": 1.0,
                "score_p95": 1.5,
                "control_beats_real_fraction": 0.1,
            }
        ],
        "top_orientation_candidate_summary": {
            "top_n": 5,
            "top1_phase_fraction": {"Ti-bcc": 1.0},
            "top_orientation_score_margin_median": 0.4,
        },
        "distinguishability_summary": {
            "conclusion": "DISTINGUISHABLE",
            "control_status": "RUN",
            "median_score_margin": 0.4,
        },
    }
    summary_path = tmp_path / "phase_summary_v6_optimized.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "phase_map_real_ti_only_qc_masked.png").write_bytes(b"placeholder")

    md_path, html_path = generate_phase_orientation_report(tmp_path, summary_path=summary_path)

    markdown = md_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "Phase/Orientation Screening Report" in markdown
    assert "Ti-bcc" in markdown
    assert "DISTINGUISHABLE" in markdown
    assert "Peak Finding and Calibration" in markdown
    assert "Raw Ti-only winner maps are unvalidated screening aids" in markdown
    assert "phase_map_real_ti_only_qc_masked.png" in markdown
    assert "<html" in html


def test_report_warns_when_peak_preflight_and_controls_fail(tmp_path):
    summary = {
        "settings": {
            "data_file": "sample.h5",
            "out_dir": str(tmp_path),
            "mode": "coarse",
            "run_control": True,
            "control_status": "FAILED_MISSING_REQUIRED_CONTROL_CIF",
            "peak_preflight_status": "FAILED_LOW_PEAK_USABILITY",
        },
        "confidence_summary": {
            "final_high_confidence_fraction": 0.02759,
        },
        "real_phase_results_aggregated_over_axes": [],
        "control_phase_results_aggregated_over_axes": [],
        "top_orientation_candidate_summary": {},
        "distinguishability_summary": {},
    }
    summary_path = tmp_path / "phase_summary_v6_optimized.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    md_path, _ = generate_phase_orientation_report(tmp_path, summary_path=summary_path)

    markdown = md_path.read_text(encoding="utf-8")
    assert "FAILED_LOW_PEAK_USABILITY" in markdown
    assert "FAILED_MISSING_REQUIRED_CONTROL_CIF" in markdown
    assert "below the 0.05 interpretation threshold" in markdown
    assert "No negative/control phase rows are available" in markdown
    assert "validation failed closed" in markdown


def test_select_peak_preflight_candidate_prefers_first_target_passing_row():
    rows = [
        {"params": {"minRelativeIntensity": 0.03}, "strong_peak_count_median": 3, "clean_peak_count_median": 8},
        {"params": {"minRelativeIntensity": 0.02}, "strong_peak_count_median": 5, "clean_peak_count_median": 10},
        {"params": {"minRelativeIntensity": 0.01}, "strong_peak_count_median": 7, "clean_peak_count_median": 12},
    ]

    selected = select_peak_preflight_candidate(rows, target_strong_median=5, clean_median_min=10, clean_median_max=20)

    assert selected is rows[1]


def test_classify_control_validation_status_requires_all_required_controls():
    required = ["TiO2-rutile-control", "TiO-control"]

    assert classify_control_validation_status(False, required, []) == "NOT_RUN"
    assert (
        classify_control_validation_status(
            True,
            required,
            ["TiO2-rutile-control"],
            missing_required_cifs=[{"phase": "TiO-control", "cif": "TiO.cif"}],
        )
        == "FAILED_MISSING_REQUIRED_CONTROL_CIF"
    )
    assert classify_control_validation_status(True, required, ["TiO2-rutile-control"]) == "FAILED"
    assert classify_control_validation_status(True, required, required) == "RUN"


def test_apply_control_validation_gate_blocks_otherwise_high_confidence_pixels():
    high_confidence = np.array([[True, False], [True, True]])
    reasons = np.array([["PASS", "FAILED_LOW_PEAK"], ["PASS", "PASS"]], dtype=object)

    gated, gated_reasons = apply_control_validation_gate(
        high_confidence,
        reasons,
        "FAILED_MISSING_REQUIRED_CONTROL_CIF",
    )

    assert not np.any(gated)
    assert gated_reasons[0, 0] == "FAILED_CONTROL_VALIDATION"
    assert gated_reasons[0, 1] == "FAILED_LOW_PEAK"
    assert gated_reasons[1, 0] == "FAILED_CONTROL_VALIDATION"
