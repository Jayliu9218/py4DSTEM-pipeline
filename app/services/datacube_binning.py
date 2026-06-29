"""Block-averaging downsampling for 4D-STEM DataCubes.

Provides memory-efficient binning that reads the source in blocks,
so multi-GB memmap/HDF5 sources never need to fit in RAM at once.

Binning levels
--------------
Level A — Fast Preview
    R_BIN = 4 (or 8),  Q_BIN = 2
    Typical output: 128×128×128×128  or  64×64×128×128
    Use for: data inspection, beam-centre tuning, mask refinement,
             Bragg-detection parameter sweeps.

Level B — Phase / Orientation Map
    R_BIN = 2,  Q_BIN = 2
    Typical output: 256×256×128×128
    Use for: full phase map, orientation map, score map, ROI selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preset levels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinningPreset:
    """Named binning configuration with a user-visible label."""

    key: str
    label: str
    r_bin: int  # binning factor for real-space (scan) axes 0 & 1
    q_bin: int  # binning factor for reciprocal-space (detector) axes 2 & 3


BINNING_PRESETS: dict[str, BinningPreset] = {
    "none": BinningPreset(key="none", label="None (full resolution)", r_bin=1, q_bin=1),
    "level_a_4": BinningPreset(
        key="level_a_4",
        label="Level A — Fast Preview (R_BIN=4, Q_BIN=2)",
        r_bin=4,
        q_bin=2,
    ),
    "level_a_8": BinningPreset(
        key="level_a_8",
        label="Level A — Fast Preview (R_BIN=8, Q_BIN=2)",
        r_bin=8,
        q_bin=2,
    ),
    "level_b": BinningPreset(
        key="level_b",
        label="Level B — Phase/Oriontation Map (R_BIN=2, Q_BIN=2)",
        r_bin=2,
        q_bin=2,
    ),
}

DEFAULT_BLOCK_BYTES = 128 * 1024 * 1024  # 128 MiB per block


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bin_4d(
    source: Any,
    r_bin: int,
    q_bin: int,
    *,
    block_bytes: int = DEFAULT_BLOCK_BYTES,
    progress_callback: Any = None,
) -> np.ndarray:
    """Block-average a 4D array by *r_bin* in scan dims and *q_bin* in detector dims.

    Parameters
    ----------
    source :
        Anything with ``.shape`` (4-tuple) and NumPy-style integer indexing.
        Typically an HDF5 dataset or a numpy memmap.
    r_bin :
        Binning factor for the first two axes (scan / real space).
    q_bin :
        Binning factor for the last two axes (diffraction / reciprocal space).
    block_bytes :
        Approximate memory budget per read-block.  Default 128 MiB.
    progress_callback :
        Optional ``callable(message: str, fraction: float)`` for progress
        reporting (fraction in [0, 1]).

    Returns
    -------
    np.ndarray
        The binned 4D array with shape
        ``(Sx//r_bin, Sy//r_bin, Qx//q_bin, Qy//q_bin)``.
        The dtype matches *source* for integer types or is float32 for
        floating-point sources.
    """
    r_bin = max(int(r_bin), 1)
    q_bin = max(int(q_bin), 1)

    shape = _validate_4d(source)
    sx, sy, qx, qy = shape

    if sx % r_bin != 0 or sy % r_bin != 0:
        raise ValueError(
            f"Scan shape {sx}×{sy} is not divisible by R_BIN={r_bin}. "
            f"Adjust the binning factor."
        )
    if qx % q_bin != 0 or qy % q_bin != 0:
        raise ValueError(
            f"Detector shape {qx}×{qy} is not divisible by Q_BIN={q_bin}. "
            f"Adjust the binning factor."
        )

    out_sx = sx // r_bin
    out_sy = sy // r_bin
    out_qx = qx // q_bin
    out_qy = qy // q_bin

    out_dtype = _output_dtype(source)
    out = np.zeros((out_sx, out_sy, out_qx, out_qy), dtype=np.float64)

    emit = progress_callback or (lambda _m, _f: None)
    emit("Starting 4D binning", 0.0)

    # Read scan blocks (rows of the scan grid), bin each, accumulate.
    scan_rows_per_block = _scan_rows_per_block(shape, r_bin, block_bytes)
    total_blocks = max((out_sx + scan_rows_per_block - 1) // scan_rows_per_block, 1)

    for block_idx in range(total_blocks):
        src_start = block_idx * scan_rows_per_block * r_bin
        src_stop = min(src_start + scan_rows_per_block * r_bin, sx)

        # Read one block of scan rows at full detector resolution
        block = np.asarray(source[src_start:src_stop, :, :, :], dtype=np.float64)
        # block shape: (B*r_bin, sy, qx, qy)  where B = number of output rows

        binned = _bin_block(block, r_bin, q_bin)
        # binned shape: (B, out_sy, out_qx, out_qy)

        out_start = block_idx * scan_rows_per_block
        out_stop = out_start + binned.shape[0]
        out[out_start:out_stop, :, :, :] = binned

        emit("Binning 4D data", (block_idx + 1) / total_blocks)

    # Finalise dtype
    if out_dtype != np.float64:
        out = out.astype(out_dtype, copy=False)

    return out


def binning_output_shape(
    shape: tuple[int, int, int, int],
    r_bin: int,
    q_bin: int,
) -> tuple[int, int, int, int]:
    """Return the shape after binning without reading any data."""
    r_bin = max(int(r_bin), 1)
    q_bin = max(int(q_bin), 1)
    return (shape[0] // r_bin, shape[1] // r_bin, shape[2] // q_bin, shape[3] // q_bin)


def preset_for_key(key: str) -> BinningPreset:
    """Look up a binning preset by key.

    Returns the ``"none"`` preset for unknown keys so callers can always
    proceed; a warning is logged.
    """
    preset = BINNING_PRESETS.get(key)
    if preset is None:
        logger.warning("Unknown binning preset %r, falling back to None.", key)
        preset = BINNING_PRESETS["none"]
    return preset


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate_4d(source: Any) -> tuple[int, int, int, int]:
    shape = tuple(int(v) for v in getattr(source, "shape", ()))
    if len(shape) != 4:
        raise ValueError(f"Expected 4D data, got shape {shape}.")
    return shape  # type: ignore[return-value]


def _output_dtype(source: Any) -> np.dtype:
    src_dtype = np.dtype(getattr(source, "dtype", np.float32))
    if np.issubdtype(src_dtype, np.integer):
        return src_dtype
    return np.dtype(np.float32)


def _scan_rows_per_block(
    shape: tuple[int, int, int, int],
    r_bin: int,
    block_bytes: int,
) -> int:
    """How many *binned* scan rows can fit in *block_bytes*."""
    itemsize = 8  # float64 during accumulation
    # Each output row requires reading r_bin source rows × full detector shape
    bytes_per_binned_row = max(
        r_bin * shape[1] * shape[2] * shape[3] * itemsize, 1
    )
    return max(int(block_bytes) // bytes_per_binned_row, 1)


def _bin_block(block: np.ndarray, r_bin: int, q_bin: int) -> np.ndarray:
    """Bin a single scan-rows block.

    Parameters
    ----------
    block : shape (B*r_bin, sy, qx, qy)
    r_bin, q_bin : binning factors

    Returns
    -------
    binned : shape (B, sy//r_bin, qx//q_bin, qy//q_bin)
    """
    B_rbin, sy, qx, qy = block.shape
    B = B_rbin // r_bin

    # --- Step 1: bin the scan (R) dimension ---
    # Reshape: (B, r_bin, sy, qx, qy) → mean over axis=1
    if r_bin > 1:
        block = block.reshape(B, r_bin, sy, qx, qy).mean(axis=1, dtype=np.float64)
    # Now shape: (B, sy, qx, qy)

    # --- Step 2: bin the second scan dimension ---
    # Reshape: (B, sy//r_bin, r_bin, qx, qy) → mean over axis=2
    if r_bin > 1:
        block = block.reshape(B, sy // r_bin, r_bin, qx, qy).mean(axis=2, dtype=np.float64)
    # Now shape: (B, sy//r_bin, qx, qy)

    # --- Step 3: bin the detector (Q) dimensions ---
    if q_bin > 1:
        block = block.reshape(
            B, sy // r_bin, qx // q_bin, q_bin, qy // q_bin, q_bin
        ).mean(axis=(3, 5), dtype=np.float64)

    return block
