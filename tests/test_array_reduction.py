from __future__ import annotations

import unittest
from pathlib import Path

import h5py
import numpy as np

from app.services.array_reduction import (
    detector_sum,
    masked_scan_mean,
    max_diffraction,
    mean_diffraction,
    scan_sum,
)
from app.services.hdf5_service import Hdf5Service
from app.services.preprocessing_service import HotPixelParams, PreprocessingService
from app.services.virtual_detector_service import VirtualDetectorParams, VirtualDetectorService


class RecordingDataset:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.selections: list[object] = []

    def __getitem__(self, selection):
        self.selections.append(selection)
        return self.data[selection]


class ArrayReductionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.arange(5 * 4 * 3 * 6, dtype=np.float32).reshape(5, 4, 3, 6)

    def test_block_reductions_match_numpy(self) -> None:
        mask = np.zeros((3, 6), dtype=bool)
        mask[1:, 2:5] = True
        scan_mask = np.zeros((5, 4), dtype=bool)
        scan_mask[1:4, 1:3] = True

        np.testing.assert_allclose(mean_diffraction(self.data), self.data.mean(axis=(0, 1)))
        np.testing.assert_array_equal(max_diffraction(self.data), self.data.max(axis=(0, 1)))
        np.testing.assert_allclose(scan_sum(self.data), self.data.sum(axis=(2, 3)))
        np.testing.assert_allclose(
            detector_sum(self.data, mask),
            self.data[:, :, mask].sum(axis=2),
        )
        np.testing.assert_allclose(masked_scan_mean(self.data, scan_mask), self.data[scan_mask].mean(axis=0))

    def test_reductions_never_request_complete_4d_dataset(self) -> None:
        source = RecordingDataset(self.data)
        mask = np.ones(self.data.shape[2:], dtype=bool)
        scan_mask = np.ones(self.data.shape[:2], dtype=bool)

        mean_diffraction(source)
        max_diffraction(source)
        scan_sum(source)
        detector_sum(source, mask)
        masked_scan_mean(source, scan_mask)

        self.assertTrue(source.selections)
        for selection in source.selections:
            self.assertIsInstance(selection, tuple)
            first_axis = selection[0]
            self.assertIsInstance(first_axis, slice)
            self.assertLess(first_axis.stop - first_axis.start, self.data.shape[0])

    def test_hdf5_service_workflows_match_numpy(self) -> None:
        path = Path(__file__).resolve().parents[1] / ".test-output" / "block_reduction_sample.h5"
        path.parent.mkdir(exist_ok=True)
        data = self.data.copy()
        data[:, :, 1, 3] *= 20
        try:
            with h5py.File(path, "w") as output:
                output.create_dataset("data", data=data, chunks=(1, 4, 3, 6))
            with h5py.File(path, "r") as source:
                dataset = source["data"]
                scan_image = Hdf5Service().read_4d_scan_image(dataset)
                diagnostics = PreprocessingService().basic_diagnostics(dataset)
                preview = PreprocessingService().preview_hot_pixels(dataset, HotPixelParams(8))
                params = VirtualDetectorParams(
                    VirtualDetectorService.VIRTUAL_DIFFRACTION,
                    0,
                    0,
                    0,
                    1,
                    roi_rx_start=1,
                    roi_rx_end=4,
                    roi_ry_start=1,
                    roi_ry_end=3,
                )
                diffraction = VirtualDetectorService().compute_virtual_diffraction(dataset, params)

            np.testing.assert_allclose(scan_image, data.sum(axis=(2, 3)))
            np.testing.assert_allclose(diagnostics["Mean diffraction pattern"], data.mean(axis=(0, 1)))
            np.testing.assert_allclose(diagnostics["Maximum diffraction pattern"], data.max(axis=(0, 1)))
            np.testing.assert_allclose(preview.before_mean, data.mean(axis=(0, 1)))
            np.testing.assert_allclose(diffraction, data[1:4, 1:3].mean(axis=(0, 1)))
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
