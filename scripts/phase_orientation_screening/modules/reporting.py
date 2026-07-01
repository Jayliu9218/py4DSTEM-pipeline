"""End-of-run Markdown and HTML reports for phase/orientation screening."""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path


DEFAULT_TITLE = "Phase/Orientation Screening Report"


def _load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_value(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _markdown_table(headers, rows):
    if not rows:
        return "_No rows available._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(cell) for cell in row) + " |")
    return "\n".join(lines)


def _figure_exists(out_dir, filename):
    return (Path(out_dir) / filename).exists()


def _figure_section(out_dir):
    figures = [
        ("dp_mean raw", "qc_dp_mean_raw.png"),
        ("dp_mean processed", "qc_dp_mean_processed.png"),
        ("dp_mean detected peak overlay", "qc_dp_mean_detected_peak_overlay.png"),
        ("Elliptical calibration fit", "qc_elliptical_calibration_fit.png"),
        ("Ti-only best candidate phase", "phase_map_real_ti_only_best_candidate_clean.png"),
        ("QC-masked phase map", "phase_map_real_ti_only_qc_masked.png"),
        ("Winning phase and fiber axis", "phase_map_real_winning_axis.png"),
        ("Best phase score", "phase_map_real_best_score.png"),
        ("Best-second score margin", "phase_map_real_score_margin.png"),
        ("High-confidence mask", "phase_map_high_confidence_mask.png"),
        ("Ambiguous mask", "phase_map_ambiguous_mask.png"),
        ("Low-peak mask", "phase_map_low_peak_mask.png"),
        ("Mixed/diffuse mask", "phase_map_mixed_diffuse_mask.png"),
        ("Control failure mask", "qc_control_failure_mask.png"),
        ("Control best phase", "qc_control_best_phase.png"),
        ("Control minus real score", "qc_control_minus_real_score.png"),
    ]
    lines = ["## Key Figures", ""]
    found = False
    for title, filename in figures:
        if not _figure_exists(out_dir, filename):
            continue
        found = True
        lines.extend([f"### {title}", "", f"![{title}]({filename})", ""])
    if not found:
        lines.append("_No standard figure outputs were found in this run directory._")
    return "\n".join(lines)


def _settings_section(summary):
    settings = summary.get("settings", {})
    keys = [
        "data_file",
        "out_dir",
        "output_tag",
        "mode",
        "screening_mode",
        "orientation_mode",
        "num_matches_return",
        "run_control",
        "control_status",
        "k_max",
        "inv_ang_per_pixel",
        "angle_step_zone_axis",
        "angle_step_in_plane",
        "bragg_cache_status",
        "bragg_cache_path",
    ]
    rows = [(key, settings.get(key)) for key in keys if key in settings]
    return "## Run Settings\n\n" + _markdown_table(["Setting", "Value"], rows)


def _confidence_section(summary):
    confidence = summary.get("confidence_summary", {})
    rows = [(key, confidence.get(key)) for key in sorted(confidence)]
    return "## QC and Confidence Summary\n\n" + _markdown_table(["Metric", "Value"], rows)


def _calibration_section(summary):
    settings = summary.get("settings", {})
    calibration = settings.get("calibration_status", {})
    peak_diagnostics = settings.get("peak_detection_diagnostics", [])
    rows = [(key, calibration.get(key)) for key in sorted(calibration)]
    peak_rows = [
        [item.get("label"), item.get("peak_count"), item.get("direct_beam_radius_px"), item.get("background")]
        for item in peak_diagnostics
    ]
    return "\n".join([
        "## Peak Finding and Calibration",
        "",
        "### Calibration State",
        "",
        _markdown_table(["Field", "Value"], rows),
        "",
        "### Raw/Processed Peak Diagnostics",
        "",
        _markdown_table(["Pattern", "QC peak count", "Direct-beam mask radius px", "Background"], peak_rows),
    ])


def _top_candidate_section(summary):
    candidate = summary.get("top_orientation_candidate_summary", {})
    rows = [(key, candidate.get(key)) for key in sorted(candidate)]
    distinguishability = summary.get("distinguishability_summary", {})
    distinguishability_rows = [(key, distinguishability.get(key)) for key in sorted(distinguishability)]
    return "\n".join([
        "## Top Candidates and Distinguishability",
        "",
        "### Top-5 Candidate Summary",
        "",
        _markdown_table(["Metric", "Value"], rows),
        "",
        "### Ti-bcc / Ti-hcp Distinguishability",
        "",
        _markdown_table(["Metric", "Value"], distinguishability_rows),
    ])


def _branch_diagnostics_section(summary):
    rows = []
    for item in summary.get("real_branch_results", []) + summary.get("control_branch_results", []):
        rows.append([
            item.get("branch"),
            item.get("branch_status", "RUN"),
            item.get("failure_reason"),
            item.get("n_template_reflections_before_filter"),
            item.get("n_template_reflections_after_kmax"),
            item.get("n_template_reflections"),
            item.get("q_median_template"),
            item.get("q_median_exp"),
            item.get("match_radius_q"),
            item.get("n_exp_peaks_test"),
            item.get("n_clean_peaks_test"),
            item.get("matched_peak_count"),
            item.get("nearest_template_distance_median"),
            item.get("score_median"),
        ])
    return "\n".join([
        "## Template and q-Space Diagnostics",
        "",
        _markdown_table(
            [
                "Branch",
                "Status",
                "Failure reason",
                "Template n raw",
                "Template n <= K_MAX",
                "Template n test",
                "Template q median",
                "Experimental q median",
                "Match radius q",
                "Test exp peaks",
                "Test clean peaks",
                "Matched test peaks",
                "Nearest template distance median",
                "Median score",
            ],
            rows,
        ),
    ])


def _phase_section(summary):
    real_rows = []
    for item in summary.get("real_phase_results_aggregated_over_axes", []):
        real_rows.append([
            item.get("phase"),
            item.get("score_median"),
            item.get("score_p95"),
            item.get("winning_fraction_raw_ti_only"),
            item.get("winning_fraction_after_all_qc_masks"),
        ])
    control_rows = []
    for item in summary.get("control_phase_results_aggregated_over_axes", []):
        control_rows.append([
            item.get("phase"),
            item.get("score_median"),
            item.get("score_p95"),
            item.get("control_beats_real_fraction"),
        ])
    lines = [
        "## Phase and Orientation Mapping Summary",
        "",
        "### Real Ti Candidates",
        "",
        _markdown_table(
            ["Phase", "Median score", "P95 score", "Raw win fraction", "QC high-confidence fraction"],
            real_rows,
        ),
        "",
        "### Negative/Control Candidates",
        "",
        _markdown_table(
            ["Phase", "Median score", "P95 score", "Control beats real fraction"],
            control_rows,
        ),
    ]
    if summary.get("branch_metadata"):
        rows = [
            [m.get("group"), m.get("phase"), m.get("axis_tag"), m.get("branch")]
            for m in summary.get("branch_metadata", [])
        ]
        lines.extend(["", "### Aggregated Branches", "", _markdown_table(["Group", "Phase", "Axis", "Branch"], rows)])
    return "\n".join(lines)


def _roi_section(out_dir):
    csv_path = Path(out_dir) / "roi_review_candidates.csv"
    json_path = Path(out_dir) / "roi_review_candidates.json"
    lines = ["## ROI Review Candidates", ""]
    if not csv_path.exists():
        lines.append("_No ROI review candidate table was generated for this run._")
        return "\n".join(lines)
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= 12:
                break
            rows.append([
                row.get("category"),
                row.get("scan_x"),
                row.get("scan_y"),
                row.get("best_phase"),
                row.get("winning_axis"),
                row.get("score_margin"),
                row.get("high_confidence"),
            ])
    lines.extend([
        f"Full table: [{csv_path.name}]({csv_path.name})",
        f"JSON copy: [{json_path.name}]({json_path.name})" if json_path.exists() else "",
        "",
        _markdown_table(
            ["Category", "x", "y", "Best phase", "Winning axis", "Score margin", "High confidence"],
            rows,
        ),
    ])
    return "\n".join(line for line in lines if line != "")


def _sweep_section(summary):
    results = summary.get("results", [])
    rows = []
    for item in results:
        rows.append([
            item.get("k_max"),
            item.get("returncode"),
            item.get("final_high_confidence_fraction"),
            item.get("ambiguous_fraction_real_margin_or_score"),
            item.get("control_failure_fraction"),
            item.get("summary_path"),
        ])
    return "## K_MAX Sweep Summary\n\n" + _markdown_table(
        ["K_MAX", "Return code", "High confidence", "Ambiguous", "Control failure", "Summary"],
        rows,
    )


def markdown_to_basic_html(markdown_text, title=DEFAULT_TITLE):
    """Convert the limited report Markdown we generate into simple standalone HTML."""
    link_pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def inline_markup(text):
        escaped = html.escape(text)
        return link_pattern.sub(
            lambda match: f'<a href="{html.escape(match.group(2), quote=True)}">{html.escape(match.group(1))}</a>',
            escaped,
        )

    lines = []
    in_table = False
    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt = line[2:line.index("]")]
            src = line[line.index("](") + 2:-1]
            lines.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.startswith("| ") and line.endswith(" |"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(c == "---" for c in cells):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                lines.append("<table>")
                in_table = True
            lines.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
        else:
            if in_table:
                lines.append("</table>")
                in_table = False
            if line:
                lines.append(f"<p>{inline_markup(line)}</p>")
    if in_table:
        lines.append("</table>")
    body = "\n".join(lines)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; line-height: 1.45; color: #222; }}
table {{ border-collapse: collapse; margin: 1rem 0; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.35rem 0.5rem; text-align: left; }}
th {{ background: #f2f2f2; }}
img {{ max-width: 720px; width: 100%; border: 1px solid #ddd; }}
figure {{ margin: 1.5rem 0; }}
figcaption {{ color: #555; font-size: 0.9rem; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def generate_phase_orientation_report(out_dir, summary_path=None, title=DEFAULT_TITLE):
    """Write Markdown and HTML reports into ``out_dir`` and return both paths."""
    out_dir = Path(out_dir)
    summary_path = Path(summary_path) if summary_path is not None else out_dir / "phase_summary_v6_optimized.json"
    summary = _load_json(summary_path)
    is_sweep = "results" in summary and "k_max_values" in summary
    lines = [
        f"# {title}",
        "",
        "> Screening output only: treat phase/orientation labels as candidates unless Bragg peak QC, negative controls, score margins, and single-pattern overlays are all physically consistent.",
        "",
    ]
    if is_sweep:
        lines.append(_sweep_section(summary))
    else:
        lines.extend([
            _settings_section(summary),
            "",
            _phase_section(summary),
            "",
            _calibration_section(summary),
            "",
            _confidence_section(summary),
            "",
            _top_candidate_section(summary),
            "",
            _branch_diagnostics_section(summary),
            "",
            _figure_section(out_dir),
            "",
            _roi_section(out_dir),
        ])
    markdown = "\n".join(lines).rstrip() + "\n"
    md_path = out_dir / "phase_orientation_report.md"
    html_path = out_dir / "phase_orientation_report.html"
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(markdown_to_basic_html(markdown, title=title), encoding="utf-8")
    print(f"[saved] {md_path}")
    print(f"[saved] {html_path}")
    return md_path, html_path
