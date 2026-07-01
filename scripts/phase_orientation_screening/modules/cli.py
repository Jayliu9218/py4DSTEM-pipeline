"""Command-line helpers for phase/orientation screening."""

from __future__ import annotations

import argparse
from pathlib import Path


def sanitize_tag(value):
    """Convert a numeric/string setting into a filesystem-friendly tag."""
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


def build_parser():
    parser = argparse.ArgumentParser(
        description="Multi-axis Ti phase/orientation mapping with WS2 QC."
    )
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--mode", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--k-max", type=float, default=None)
    parser.add_argument("--k-max-sweep", type=str, default=None)
    parser.add_argument("--inv-ang-per-pixel", type=float, default=None)
    parser.add_argument("--angle-step-zone-axis", type=float, default=None)
    parser.add_argument("--angle-step-in-plane", type=float, default=None)
    parser.add_argument("--margin-threshold", type=float, default=0.20)
    parser.add_argument("--max-clean-peaks-for-single", type=int, default=50)
    parser.add_argument("--run-control", dest="run_control", action="store_true")
    parser.add_argument("--skip-control", dest="run_control", action="store_false")
    parser.set_defaults(run_control=True)
    parser.add_argument("--force-recompute-bragg", action="store_true")
    parser.add_argument("--calibration-peaks", type=Path, default=None)
    parser.add_argument("--branch-only", action="store_true")
    parser.add_argument("--phase", type=str, default=None)
    parser.add_argument("--fiber-axis", type=str, default=None)
    parser.add_argument("--aggregate-branches", type=Path, default=None)
    parser.add_argument("--output-tag", type=str, default=None)
    return parser


def resolve_run_settings(args, root):
    defaults = mode_defaults(args.mode)
    k_max = float(args.k_max if args.k_max is not None else defaults["k_max"])
    inv_ang_per_pixel = float(args.inv_ang_per_pixel if args.inv_ang_per_pixel is not None else 0.0192)
    angle_step_zone_axis = float(
        args.angle_step_zone_axis
        if args.angle_step_zone_axis is not None
        else defaults["angle_step_zone_axis"]
    )
    angle_step_in_plane = float(
        args.angle_step_in_plane
        if args.angle_step_in_plane is not None
        else defaults["angle_step_in_plane"]
    )
    output_tag = args.output_tag
    if output_tag is None:
        control_tag = "control" if args.run_control else "no_control"
        output_tag = (
            f"{args.mode}_k{sanitize_tag(k_max)}_za{sanitize_tag(angle_step_zone_axis)}_"
            f"ip{sanitize_tag(angle_step_in_plane)}_{control_tag}"
        )
    data_file = Path(root) / args.data_file
    out_dir = Path(root) / data_file.stem / output_tag
    return {
        "k_max": k_max,
        "inv_ang_per_pixel": inv_ang_per_pixel,
        "angle_step_zone_axis": angle_step_zone_axis,
        "angle_step_in_plane": angle_step_in_plane,
        "output_tag": output_tag,
        "data_file": data_file,
        "out_dir": out_dir,
    }
