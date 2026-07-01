"""Orientation matching helper functions."""

from __future__ import annotations

import numpy as np


def axis_to_tag(axis):
    return "za_" + "_".join(str(v).replace("-", "m") for v in axis)


def normalize_score(score):
    score = np.asarray(score, dtype=np.float32)
    out = np.zeros_like(score, dtype=np.float32)
    finite = np.isfinite(score)
    if np.any(finite):
        lo, hi = np.nanpercentile(score[finite], [1, 99])
        if hi > lo:
            out = np.clip((score - lo) / (hi - lo), 0, 1)
    return out


def get_orientation_score_array(orientation_map):
    """Extract best-match score map from py4DSTEM OrientationMap across versions."""
    for attr in ["corr", "correlation", "corrs", "intensity", "intensities"]:
        if hasattr(orientation_map, attr):
            arr = np.asarray(getattr(orientation_map, attr))
            if arr.ndim == 3:
                return arr[:, :, 0].astype(np.float32)
            if arr.ndim == 2:
                return arr.astype(np.float32)
    public = [a for a in dir(orientation_map) if not a.startswith("_")]
    raise AttributeError(f"Update get_orientation_score_array() for this py4DSTEM version. Public attributes: {public}")


def select_branch_phase(phase_name, fiber_axis, real_candidate_phases, control_phases):
    all_phases = [(p, "real") for p in real_candidate_phases] + [(p, "control") for p in control_phases]
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
