from __future__ import annotations

import unittest
import sys
import types
from pathlib import Path

import h5py
import numpy as np

from app.services.array_reduction import (
    PythonReductionBackend,
    detector_sum,
    get_reduction_backend,
    masked_scan_mean,
    max_diffraction,
    mean_diffraction,
    scan_sum,
    set_reduction_backend,
    try_enable_native_backend,
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

    def test_public_functions_delegate_to_configured_backend(self) -> None:
        class RecordingBackend(PythonReductionBackend):
            name = "recording"

            def __init__(self) -> None:
                self.calls: list[str] = []

            def mean_diffraction(self, source, **kwargs):
                self.calls.append("mean")
                return super().mean_diffraction(source, **kwargs)

            def max_diffraction(self, source, **kwargs):
                self.calls.append("max")
                return super().max_diffraction(source, **kwargs)

            def scan_sum(self, source, **kwargs):
                self.calls.append("scan")
                return super().scan_sum(source, **kwargs)

            def virtual_detector_sum(self, source, mask, **kwargs):
                self.calls.append("detector")
                return super().virtual_detector_sum(source, mask, **kwargs)

            def masked_scan_mean(self, source, mask, **kwargs):
                self.calls.append("masked")
                return super().masked_scan_mean(source, mask, **kwargs)

        original = get_reduction_backend()
        backend = RecordingBackend()
        mask = np.ones(self.data.shape[2:], dtype=bool)
        scan_mask = np.ones(self.data.shape[:2], dtype=bool)
        try:
            set_reduction_backend(backend)
            mean_diffraction(self.data)
            max_diffraction(self.data)
            scan_sum(self.data)
            detector_sum(self.data, mask)
            masked_scan_mean(self.data, scan_mask)
        finally:
            set_reduction_backend(original)

        self.assertEqual(backend.calls, ["mean", "max", "scan", "detector", "masked"])

    def test_optional_native_backend_falls_back_when_unavailable(self) -> None:
        original = get_reduction_backend()
        try:
            set_reduction_backend(None)
            enabled = try_enable_native_backend("missing_native_array_reduction_for_test")

            self.assertFalse(enabled)
            self.assertEqual(get_reduction_backend().name, "python")
            np.testing.assert_allclose(scan_sum(self.data), self.data.sum(axis=(2, 3)))
        finally:
            set_reduction_backend(original)

    def test_optional_native_backend_can_be_enabled_by_factory(self) -> None:
        class StubBackend(PythonReductionBackend):
            name = "stub-native"

        module_name = "tests._stub_native_array_reduction"
        module = types.ModuleType(module_name)
        module.create_backend = lambda: StubBackend()
        original = get_reduction_backend()
        try:
            sys.modules[module_name] = module
            self.assertTrue(try_enable_native_backend(module_name))
            self.assertEqual(get_reduction_backend().name, "stub-native")
            np.testing.assert_allclose(scan_sum(self.data), self.data.sum(axis=(2, 3)))
        finally:
            sys.modules.pop(module_name, None)
            set_reduction_backend(original)

    def test_default_native_backend_stub_is_importable_and_safe(self) -> None:
        original = get_reduction_backend()
        try:
            set_reduction_backend(None)

            self.assertTrue(try_enable_native_backend())
            self.assertEqual(get_reduction_backend().name, "native-python-stub")
            np.testing.assert_allclose(scan_sum(self.data), self.data.sum(axis=(2, 3)))
            np.testing.assert_allclose(mean_diffraction(self.data), self.data.mean(axis=(0, 1)))
        finally:
            set_reduction_backend(original)

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

    def test_show_data_respects_memory_budget_and_reports_progress(self) -> None:
        source = RecordingDataset(self.data)
        progress: list[tuple[str, float]] = []

        results = PreprocessingService().show_data(
            source,
            memory_budget_bytes=1,
            progress_callback=lambda message, fraction: progress.append((message, fraction)),
        )

        self.assertIn("Scan overview", results)
        self.assertIn("Mean diffraction pattern", results)
        self.assertIn("Maximum diffraction pattern", results)
        self.assertTrue(progress)
        self.assertEqual(progress[-1], ("Data display ready", 1.0))
        self.assertTrue(all(0 <= fraction <= 1 for _message, fraction in progress))

    def test_show_data_task_carries_metadata_and_matches_show_data(self) -> None:
        source = RecordingDataset(self.data)
        progress: list[tuple[str, float]] = []
        service = PreprocessingService()

        task = service.show_data_task(source, memory_budget_mb=1)
        results = task.run(lambda message, fraction: progress.append((message, fraction)))

        self.assertEqual(task.name, "Show Data")
        self.assertEqual(task.memory_budget_mb, 1)
        self.assertIn("show_data:", task.result_key or "")
        self.assertEqual(task.parameters["shape"], self.data.shape)
        self.assertIn(("Preparing data display", 0.0), progress)
        self.assertIn("Scan overview", results)
        np.testing.assert_allclose(results["Mean diffraction pattern"], self.data.mean(axis=(0, 1)))

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
