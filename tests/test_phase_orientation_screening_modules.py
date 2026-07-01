import json
import sys
from pathlib import Path


SCREENING_ROOT = Path(__file__).resolve().parents[1] / "scripts" / "phase_orientation_screening"
sys.path.insert(0, str(SCREENING_ROOT))

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
    assert "phase_map_real_ti_only_qc_masked.png" in markdown
    assert "<html" in html