"""Default scientific configuration for the Ti/WS2 screening workflow."""

from __future__ import annotations


DETECT_PARAMS = {
    "corrPower": 1,
    "sigma": 0,
    "edgeBoundary": 16,
    "minRelativeIntensity": 0.08,
    "minAbsoluteIntensity": 5,
    "minPeakSpacing": 16,
    "subpixel": "poly",
    "upsample_factor": 8,
    "maxNumPeaks": 60,
}

Q_MIN_FOR_QC = 0.12
Q_MAX_FOR_QC = 1.4
STRONG_PEAK_PERCENTILE = 70
MIN_STRONG_PEAKS_FOR_MATCH = 6
TOP_CANDIDATES_TO_SAVE = 5
YMAX_RADIAL_PROFILE = 30
TEST_RXS = (0, 3, 5)
TEST_RYS = (0, 3, 5)
SINGLE_TEST_PIXEL = (3, 3)
MIN_BEST_SCORE = 0.0
PEAK_COUNT_THRESHOLD = MIN_STRONG_PEAKS_FOR_MATCH
CONTROL_FAIL_MARGIN = 0.0
SKIP_MISSING_CIFS = True
BRAGG_CACHE_TAG = "conservative_v7"
RUN_STRAIN_FOR_GLOBALLY_BEST_REAL_BRANCH = False


def real_candidate_phases(root):
    return [
        {
            "name": "Ti-bcc",
            "cif": root / "Ti-bcc.cif",
            "symmetry_order": 4,
            "zone_axis_range": "fiber",
            "fiber_axes": [[0, 1, 1], [0, 0, 1], [1, 1, 1]],
            "fiber_angles": [0, 360],
        },
        {
            "name": "Ti-hcp",
            "cif": root / "Ti-hcp.cif",
            "symmetry_order": 6,
            "zone_axis_range": "fiber",
            "fiber_axes": [[1, 0, 0], [0, 0, 1], [1, 1, 0]],
            "fiber_angles": [0, 360],
        },
    ]


def control_phases(root):
    return [
        {
            "name": "WS2-control",
            "cif": root / "WS2.cif",
            "symmetry_order": 6,
            "zone_axis_range": "fiber",
            "fiber_axes": [[0, 0, 1]],
            "fiber_angles": [0, 360],
        }
    ]
