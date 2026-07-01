"""Bragg peak QC and calibration helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


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


def fit_inv_ang_per_pixel(provided_inv_ang_per_pixel, calibration_peaks=None):
    summary = {
        "mode": "provided_value_only",
        "provided_inv_ang_per_pixel": float(provided_inv_ang_per_pixel),
        "used_inv_ang_per_pixel": float(provided_inv_ang_per_pixel),
        "calibration_peaks_path": None,
        "peaks": [],
        "fit": None,
    }
    if calibration_peaks is None:
        return float(summary["used_inv_ang_per_pixel"]), summary

    rows = load_calibration_peak_rows(calibration_peaks)
    q_pixel = np.asarray([r["q_pixel"] for r in rows], dtype=np.float64)
    q_known = np.asarray([r["q_A^-1"] for r in rows], dtype=np.float64)
    denom = float(np.sum(q_pixel * q_pixel))
    if denom <= 0:
        raise ValueError("Calibration q_pixel values must not all be zero.")
    fit_scale = float(np.sum(q_pixel * q_known) / denom)
    residual = q_known - q_pixel * fit_scale
    for row, pred, res in zip(rows, q_pixel * fit_scale, residual):
        item = dict(row)
        item["q_fit_A^-1"] = float(pred)
        item["residual_A^-1"] = float(res)
        item["relative_residual"] = None if item["q_A^-1"] == 0 else float(res / item["q_A^-1"])
        summary["peaks"].append(item)
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
    return fit_scale, summary


def cache_param_tag(detect_params, bragg_cache_tag, inv_ang_per_pixel, sanitize_tag):
    params_json = json.dumps(detect_params, sort_keys=True, default=str)
    digest = hashlib.sha1(params_json.encode("utf-8")).hexdigest()[:10]
    return f"{bragg_cache_tag}_{digest}_inv{sanitize_tag(f'{inv_ang_per_pixel:.8g}')}"


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
