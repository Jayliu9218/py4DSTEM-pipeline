from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import numpy as np


DEFAULT_BLOCK_BYTES = 32 * 1024 * 1024
ProgressCallback = Any


class ReductionBackend(Protocol):
    """Interface for interchangeable 4D reduction backends."""

    name: str

    def mean_diffraction(
        self,
        source: Any,
        *,
        memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
        scan_stride: int = 1,
        progress_callback: ProgressCallback = None,
    ) -> np.ndarray:
        ...

    def max_diffraction(
        self,
        source: Any,
        *,
        memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
        scan_stride: int = 1,
        progress_callback: ProgressCallback = None,
    ) -> np.ndarray:
        ...

    def scan_sum(
        self,
        source: Any,
        *,
        dtype: np.dtype[Any] | type | None = None,
        memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
        scan_stride: int = 1,
        progress_callback: ProgressCallback = None,
    ) -> np.ndarray:
        ...

    def virtual_detector_sum(
        self,
        source: Any,
        mask: np.ndarray,
        *,
        dtype: np.dtype[Any] | type | None = None,
        memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
    ) -> np.ndarray:
        ...

    def masked_scan_mean(
        self,
        source: Any,
        mask: np.ndarray,
        *,
        memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
    ) -> np.ndarray:
        ...


class PythonReductionBackend:
    """Default NumPy/HDF5-friendly reduction backend."""

    name = "python"

    def mean_diffraction(
        self,
        source: Any,
        *,
        memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
        scan_stride: int = 1,
        progress_callback: ProgressCallback = None,
    ) -> np.ndarray:
        return _mean_diffraction_impl(
            source,
            memory_budget_bytes=memory_budget_bytes,
            scan_stride=scan_stride,
            progress_callback=progress_callback,
        )

    def max_diffraction(
        self,
        source: Any,
        *,
        memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
        scan_stride: int = 1,
        progress_callback: ProgressCallback = None,
    ) -> np.ndarray:
        return _max_diffraction_impl(
            source,
            memory_budget_bytes=memory_budget_bytes,
            scan_stride=scan_stride,
            progress_callback=progress_callback,
        )

    def scan_sum(
        self,
        source: Any,
        *,
        dtype: np.dtype[Any] | type | None = None,
        memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
        scan_stride: int = 1,
        progress_callback: ProgressCallback = None,
    ) -> np.ndarray:
        return _scan_sum_impl(
            source,
            dtype=dtype,
            memory_budget_bytes=memory_budget_bytes,
            scan_stride=scan_stride,
            progress_callback=progress_callback,
        )

    def virtual_detector_sum(
        self,
        source: Any,
        mask: np.ndarray,
        *,
        dtype: np.dtype[Any] | type | None = None,
        memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
    ) -> np.ndarray:
        return _detector_sum_impl(
            source,
            mask,
            dtype=dtype,
            memory_budget_bytes=memory_budget_bytes,
        )

    def masked_scan_mean(
        self,
        source: Any,
        mask: np.ndarray,
        *,
        memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
    ) -> np.ndarray:
        return _masked_scan_mean_impl(source, mask, memory_budget_bytes=memory_budget_bytes)


_reduction_backend: ReductionBackend = PythonReductionBackend()


def get_reduction_backend() -> ReductionBackend:
    return _reduction_backend


def set_reduction_backend(backend: ReductionBackend | None) -> None:
    global _reduction_backend
    _reduction_backend = backend or PythonReductionBackend()


def shape_4d(source: Any) -> tuple[int, int, int, int]:
    shape = tuple(int(value) for value in getattr(source, "shape", ()))
    if len(shape) != 4:
        raise ValueError(f"Expected 4D data, got shape {shape}.")
    return shape


def iter_scan_blocks(
    source: Any,
    *,
    target_bytes: int = DEFAULT_BLOCK_BYTES,
    scan_stride: int = 1,
) -> Iterator[tuple[slice, np.ndarray]]:
    shape = shape_4d(source)
    scan_stride = _normal_scan_stride(scan_stride)
    itemsize = int(np.dtype(getattr(source, "dtype", np.float64)).itemsize)
    sampled_y = _sampled_length(shape[1], scan_stride)
    bytes_per_row = max(int(sampled_y * np.prod(shape[2:], dtype=np.int64)) * itemsize, 1)
    rows = max(int(target_bytes) // bytes_per_row, 1)
    sampled_x = _sampled_length(shape[0], scan_stride)
    if sampled_x > 1:
        # Keep disk-backed sources from ever materializing the complete 4D dataset.
        rows = min(rows, sampled_x - 1)

    for output_start in range(0, sampled_x, rows):
        output_stop = min(output_start + rows, sampled_x)
        source_start = output_start * scan_stride
        source_stop = min(output_stop * scan_stride, shape[0])
        selection = slice(source_start, source_stop, scan_stride)
        indexer = (selection, slice(None, None, scan_stride), slice(None), slice(None))
        yield selection, np.asarray(source[indexer])


def mean_diffraction(
    source: Any,
    *,
    memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
    scan_stride: int = 1,
    progress_callback=None,
) -> np.ndarray:
    return get_reduction_backend().mean_diffraction(
        source,
        memory_budget_bytes=memory_budget_bytes,
        scan_stride=scan_stride,
        progress_callback=progress_callback,
    )


def _mean_diffraction_impl(
    source: Any,
    *,
    memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
    scan_stride: int = 1,
    progress_callback=None,
) -> np.ndarray:
    shape = shape_4d(source)
    scan_stride = _normal_scan_stride(scan_stride)
    total = np.zeros(shape[2:], dtype=np.float64)
    emit = progress_callback or (lambda _message, _fraction: None)
    count = _sampled_length(shape[0], scan_stride) * _sampled_length(shape[1], scan_stride)
    for selection, block in iter_scan_blocks(
        source,
        target_bytes=max(int(memory_budget_bytes), 1),
        scan_stride=scan_stride,
    ):
        total += np.sum(block, axis=(0, 1), dtype=np.float64)
        emit("Calculating mean diffraction", selection.stop / max(shape[0], 1))
    return total / max(count, 1)


def max_diffraction(
    source: Any,
    *,
    memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
    scan_stride: int = 1,
    progress_callback=None,
) -> np.ndarray:
    return get_reduction_backend().max_diffraction(
        source,
        memory_budget_bytes=memory_budget_bytes,
        scan_stride=scan_stride,
        progress_callback=progress_callback,
    )


def _max_diffraction_impl(
    source: Any,
    *,
    memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
    scan_stride: int = 1,
    progress_callback=None,
) -> np.ndarray:
    shape = shape_4d(source)
    scan_stride = _normal_scan_stride(scan_stride)
    maximum: np.ndarray | None = None
    emit = progress_callback or (lambda _message, _fraction: None)
    for selection, block in iter_scan_blocks(
        source,
        target_bytes=max(int(memory_budget_bytes), 1),
        scan_stride=scan_stride,
    ):
        block_maximum = np.max(block, axis=(0, 1))
        maximum = block_maximum if maximum is None else np.maximum(maximum, block_maximum)
        emit("Calculating maximum diffraction", selection.stop / max(shape[0], 1))
    if maximum is None:
        raise ValueError("Cannot reduce empty 4D data.")
    return maximum


def scan_sum(source: Any, *, dtype: np.dtype[Any] | type | None = None) -> np.ndarray:
    return scan_sum_with_progress(source, dtype=dtype)


def scan_sum_with_progress(
    source: Any,
    *,
    dtype: np.dtype[Any] | type | None = None,
    memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
    scan_stride: int = 1,
    progress_callback=None,
) -> np.ndarray:
    return get_reduction_backend().scan_sum(
        source,
        dtype=dtype,
        memory_budget_bytes=memory_budget_bytes,
        scan_stride=scan_stride,
        progress_callback=progress_callback,
    )


def _scan_sum_impl(
    source: Any,
    *,
    dtype: np.dtype[Any] | type | None = None,
    memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
    scan_stride: int = 1,
    progress_callback=None,
) -> np.ndarray:
    shape = shape_4d(source)
    scan_stride = _normal_scan_stride(scan_stride)
    output_dtype = np.dtype(dtype) if dtype is not None else _sum_dtype(source)
    result_shape = (_sampled_length(shape[0], scan_stride), _sampled_length(shape[1], scan_stride))
    result = np.empty(result_shape, dtype=output_dtype)
    emit = progress_callback or (lambda _message, _fraction: None)
    emit("Preparing scan overview", 0.0)
    for selection, block in iter_scan_blocks(
        source,
        target_bytes=max(int(memory_budget_bytes), 1),
        scan_stride=scan_stride,
    ):
        output_start = selection.start // scan_stride
        output_stop = output_start + block.shape[0]
        result[output_start:output_stop, :] = np.sum(block, axis=(2, 3), dtype=output_dtype)
        emit("Reducing scan overview", selection.stop / max(shape[0], 1))
    return result


def detector_sum(
    source: Any,
    mask: np.ndarray,
    *,
    dtype: np.dtype[Any] | type | None = None,
) -> np.ndarray:
    return get_reduction_backend().virtual_detector_sum(source, mask, dtype=dtype)


def _detector_sum_impl(
    source: Any,
    mask: np.ndarray,
    *,
    dtype: np.dtype[Any] | type | None = None,
    memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
) -> np.ndarray:
    shape = shape_4d(source)
    detector_mask = np.asarray(mask, dtype=bool)
    if detector_mask.shape != shape[2:]:
        raise ValueError(
            f"Detector mask shape {detector_mask.shape} does not match diffraction shape {shape[2:]}."
        )
    output_dtype = np.dtype(dtype) if dtype is not None else _sum_dtype(source)
    result = np.empty(shape[:2], dtype=output_dtype)
    for selection, block in iter_scan_blocks(source, target_bytes=max(int(memory_budget_bytes), 1)):
        result[selection, :] = np.sum(block[:, :, detector_mask], axis=2, dtype=output_dtype)
    return result


def masked_scan_mean(source: Any, mask: np.ndarray) -> np.ndarray:
    return get_reduction_backend().masked_scan_mean(source, mask)


def _masked_scan_mean_impl(
    source: Any,
    mask: np.ndarray,
    *,
    memory_budget_bytes: int = DEFAULT_BLOCK_BYTES,
) -> np.ndarray:
    shape = shape_4d(source)
    scan_mask = np.asarray(mask, dtype=bool)
    if scan_mask.shape != shape[:2]:
        raise ValueError(
            f"Scan mask shape {scan_mask.shape} does not match scan shape {shape[:2]}."
        )
    count = int(np.count_nonzero(scan_mask))
    if count == 0:
        raise ValueError("Scan mask contains no selected positions.")

    total = np.zeros(shape[2:], dtype=np.float64)
    for selection, block in iter_scan_blocks(source, target_bytes=max(int(memory_budget_bytes), 1)):
        block_mask = scan_mask[selection, :]
        if np.any(block_mask):
            total += np.sum(block[block_mask], axis=0, dtype=np.float64)
    return total / count


def _sum_dtype(source: Any) -> np.dtype[Any]:
    dtype = np.dtype(getattr(source, "dtype", np.float64))
    if np.issubdtype(dtype, np.signedinteger):
        return np.dtype(np.int64)
    if np.issubdtype(dtype, np.unsignedinteger):
        return np.dtype(np.uint64)
    return dtype


def _normal_scan_stride(scan_stride: int) -> int:
    return max(int(scan_stride), 1)


def _sampled_length(length: int, scan_stride: int) -> int:
    scan_stride = _normal_scan_stride(scan_stride)
    return max((int(length) + scan_stride - 1) // scan_stride, 0)
