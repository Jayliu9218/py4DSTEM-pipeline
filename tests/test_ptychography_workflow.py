import types
import unittest
import math
import sys
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEventLoop, QThread, QTimer, Slot
from PySide6.QtWidgets import QApplication

from app.controllers.route_coordinator import build_route_modules
from app.pages.ptychography_page import PtychographyPage
from app.services.ptychography_service import (
    BUILTIN_PROFILES, COMPUTE_PRESETS, PtychographyAdapter, PtychographyGeometryParams,
    PtychographyPreprocessParams, PtychographyReconstructionParams, PtychographyService,
    PtychographyServiceError, PtychographySetupParams,
)
from app.services.workflow_state import WorkflowState, WorkflowStep
from app.widgets.log_panel import LogPanel


class _FakePtychography:
    def __init__(self, datacube, energy, defocus, device, storage, vacuum_probe_intensity=None):
        self.init = locals()
        self.object_cropped = np.ones((6, 6), dtype=complex)
        self.probe = np.ones((4, 4), dtype=complex)
        self.object_fft = np.ones((6, 6))
        self.error_iterations = [2.0, 1.0]
        self._rotation_best_rad = 0.1

    def preprocess(self, plot_center_of_mass=False):
        self._com_fitted_x = np.ones((2, 2))
        self._com_fitted_y = np.ones((2, 2))
        self._amplitudes = np.ones((4, 4, 4))
        self.sampling = (0.1, 0.1)
        self._region_of_interest_shape = (4, 4)
        self._object_shape = (6, 6)
        return self

    def reconstruct(self, num_iter=1, max_batch_size=1):
        self.reconstruction_args = (num_iter, max_batch_size)
        self.error_iterations = np.linspace(2, 0.5, num_iter)
        return self


class PtychographyWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        phase = types.SimpleNamespace(
            SingleslicePtychography=_FakePtychography,
            MixedstatePtychography=_FakePtychography,
        )
        fake = types.SimpleNamespace(process=types.SimpleNamespace(phase=phase))
        self.service = PtychographyService(PtychographyAdapter(lambda: fake))
        self.datacube = types.SimpleNamespace(data=np.ones((2, 2, 4, 4)))

    def _preprocessed(self) -> None:
        self.service.inspect_data_probe(self.datacube, PtychographySetupParams())
        self.service.set_geometry(PtychographyGeometryParams())
        self.service.preprocess(self.datacube, PtychographySetupParams(), PtychographyPreprocessParams())
        self.service.accept_preprocessing()

    def _advanced(self) -> None:
        self._preprocessed()
        self.service.quick_reconstruct()
        self.service.review_qc()
        self.service.accept_qc()
        self.service.advanced_reconstruct(PtychographyReconstructionParams())

    def test_focused_route_is_standalone_eight_stage_template(self) -> None:
        modules = build_route_modules("Phase Retrieval / Ptychography", "Ptychography")
        self.assertEqual(
            [module.key for module in modules],
            [
                "data_setup", "ptychography_data", "ptychography_geometry", "ptychography_preprocess",
                "ptychography_quick", "ptychography_review", "ptychography_optimization",
                "ptychography_advanced", "export",
            ],
        )
        self.assertFalse(any("DPC" in module.title or "Parallax" in module.title for module in modules))

    def test_compute_presets_map_device_and_storage(self) -> None:
        self.assertEqual(COMPUTE_PRESETS["CPU"], ("cpu", "cpu"))
        self.assertEqual(COMPUTE_PRESETS["GPU streamed"], ("gpu", "cpu"))
        self.assertEqual(COMPUTE_PRESETS["GPU resident"], ("gpu", "gpu"))

    def test_builtin_profiles_and_json_round_trip(self) -> None:
        self.assertEqual(set(BUILTIN_PROFILES), {
            "Safe CPU", "GPU Streaming", "Thin Weak-Phase", "Constrained Probe", "Mixed-State"
        })
        path = Path(".test-output") / "ptychography-profile.json"
        path.parent.mkdir(exist_ok=True)
        self.service.save_profile(BUILTIN_PROFILES["Mixed-State"], path)
        restored = self.service.load_profile(path)
        self.assertEqual(restored, BUILTIN_PROFILES["Mixed-State"])

    def test_data_diagnostics_and_geometry_do_not_mutate_datacube(self) -> None:
        result = self.service.inspect_data_probe(self.datacube, PtychographySetupParams())
        self.assertEqual(result.metadata["shape"], [2, 2, 4, 4])
        self.assertTrue(result.metadata["warnings"])
        self.service.set_geometry(PtychographyGeometryParams(mode="Manual override", com_rotation=12.0))
        self.assertFalse(hasattr(self.datacube, "calibration"))
        self.assertEqual(self.service.context.geometry_result.metadata["sources"]["com_rotation"], "manual override")

    def test_dataset_role_probe_is_used_without_file_loading(self) -> None:
        probe = np.arange(16).reshape(4, 4)
        setup = PtychographySetupParams(probe_source="Dataset role")
        result = self.service.inspect_data_probe(self.datacube, setup, probe)
        self.assertTrue(np.array_equal(result.images["Vacuum probe"], probe))
        self.service.set_geometry(PtychographyGeometryParams())
        self.service.preprocess(
            self.datacube, setup, PtychographyPreprocessParams(), PtychographyGeometryParams(), probe
        )
        self.assertTrue(np.array_equal(self.service.context.vacuum_probe, probe))

    def test_ideal_aperture_has_safe_semiangle_and_missing_role_probe_is_clear(self) -> None:
        self.assertEqual(BUILTIN_PROFILES["Safe CPU"].geometry.semiangle_cutoff, 20.0)
        with self.assertRaisesRegex(PtychographyServiceError, "no valid 2D vacuum probe"):
            self.service.preprocess(
                self.datacube,
                PtychographySetupParams(probe_source="Dataset role"),
                PtychographyPreprocessParams(),
                PtychographyGeometryParams(),
            )

    def test_adapter_filters_unsupported_optional_arguments(self) -> None:
        self._preprocessed()
        ptycho = self.service.context.preprocessed_ptycho
        self.assertEqual(ptycho.init["device"], "cpu")
        self.assertNotIn("semiangle_cutoff", ptycho.init)

    def test_quick_and_advanced_are_independent_and_qc_is_required(self) -> None:
        self._preprocessed()
        quick = self.service.quick_reconstruct()
        with self.assertRaisesRegex(PtychographyServiceError, "accept QC"):
            self.service.advanced_reconstruct(PtychographyReconstructionParams())
        self.service.review_qc()
        self.service.accept_qc()
        advanced = self.service.advanced_reconstruct(PtychographyReconstructionParams())
        self.assertIs(self.service.context.quick_result, quick)
        self.assertIs(self.service.context.advanced_result, advanced)
        self.assertIsNot(self.service.context.quick_ptycho, self.service.context.advanced_ptycho)

    def test_unpicklable_py4dstem_object_rebuilds_from_accepted_preprocessing(self) -> None:
        self._preprocessed()
        self.service.context.preprocessed_ptycho.unpicklable_module = math

        quick = self.service.quick_reconstruct()

        self.assertEqual(
            quick.metadata["preprocessed_instance_strategy"], "rebuild_from_accepted_parameters"
        )
        self.assertIsNot(self.service.context.preprocessed_ptycho, self.service.context.quick_ptycho)

    def test_qc_metrics_and_guidance_are_extracted(self) -> None:
        self._preprocessed()
        self.service.quick_reconstruct()
        qc = self.service.review_qc()
        self.assertIn("probe_boundary_energy", qc.metadata["metrics"])
        self.assertIn("guidance", qc.metadata)

    def test_mixed_state_and_cuda_errors_are_explicit(self) -> None:
        class Broken(_FakePtychography):
            def preprocess(self, plot_center_of_mass=False):
                raise RuntimeError("CUDA out of memory")

        phase = types.SimpleNamespace(SingleslicePtychography=Broken, MixedstatePtychography=Broken)
        service = PtychographyService(PtychographyAdapter(
            lambda: types.SimpleNamespace(process=types.SimpleNamespace(phase=phase))
        ))
        with self.assertRaisesRegex(PtychographyServiceError, "select the CPU preset"):
            service.preprocess(
                self.datacube, PtychographySetupParams(model="Mixed-state", compute_preset="GPU resident"),
                PtychographyPreprocessParams(), PtychographyGeometryParams(),
            )

    def test_package_contains_quick_advanced_arrays_and_metadata(self) -> None:
        self._advanced()
        directory = Path(".test-output") / "ptychography-package"
        saved = self.service.save_package(directory)
        self.assertEqual({path.name for path in saved}, {
            "ptychography_results.npz", "ptychography_metadata.json"
        })
        arrays = np.load(directory / "ptychography_results.npz")
        self.assertTrue(any(name.startswith("quick_") for name in arrays.files))
        self.assertTrue(any(name.startswith("advanced_") for name in arrays.files))

    def test_workflow_optimization_is_optional_branch(self) -> None:
        state = WorkflowState()
        state.mark_completed(WorkflowStep.PTYCHOGRAPHY_QC_ACCEPT)
        state.mark_completed(WorkflowStep.PTYCHOGRAPHY_ADVANCED)
        state.parameters_updated(WorkflowStep.PTYCHOGRAPHY_OPTIMIZATION)
        self.assertFalse(state.is_stale(WorkflowStep.PTYCHOGRAPHY_ADVANCED))

    def test_optimization_reuses_accepted_probe_geometry_without_none_values(self) -> None:
        captured = {}

        class Parameter:
            def __init__(self, initial, lower, upper, space="real"):
                self.initial, self.lower, self.upper, self.space = initial, lower, upper, space

        class Optimizer:
            def __init__(self, _cls, init_args, preprocess_args, reconstruction_args):
                captured.update(init=init_args, preprocess=preprocess_args, reconstruction=reconstruction_args)

            def grid_search(self, **_kwargs):
                return None

        module = types.SimpleNamespace(OptimizationParameter=Parameter, PtychographyOptimizer=Optimizer)
        self._preprocessed()
        with patch("app.services.ptychography_service.import_module", return_value=module):
            self.service.optimize(self.service.context.active_profile.optimization)

        self.assertEqual(captured["init"]["semiangle_cutoff"], 20.0)
        self.assertAlmostEqual(captured["preprocess"]["force_com_rotation"], np.rad2deg(0.1))
        self.assertFalse(any(value is None for value in captured["init"].values()))
        self.assertFalse(any(value is None for value in captured["preprocess"].values()))

    def test_worker_completion_runs_on_gui_thread_and_cleans_up(self) -> None:
        class ThreadAwarePage(PtychographyPage):
            completion_thread = None

            @Slot(object)
            def _handle_worker_finished(self, result) -> None:
                self.completion_thread = QThread.currentThread()
                super()._handle_worker_finished(result)

        page = ThreadAwarePage(lambda: self.datacube, LogPanel(), WorkflowState(), stage_mode="preprocess")
        result = types.SimpleNamespace(images={"Result": np.ones((2, 2))}, metadata={}, elapsed_seconds=0.01)
        page._start("Preprocessing", lambda: result, WorkflowStep.PTYCHOGRAPHY_PREPROCESS)
        self.assertTrue(page._is_busy)
        self._run_until_thread_finishes(page)
        self.assertIs(page.completion_thread, self.app.thread())
        self.assertIsNone(page.worker)
        self.assertFalse(page._is_busy)

    def test_failed_worker_cleans_up_and_restores_controls(self) -> None:
        page = PtychographyPage(lambda: self.datacube, LogPanel(), WorkflowState(), stage_mode="preprocess")

        def fail():
            raise RuntimeError("expected failure")

        page._start("Preprocessing", fail, WorkflowStep.PTYCHOGRAPHY_PREPROCESS)
        self._run_until_thread_finishes(page)
        self.assertIsNone(page.worker)
        self.assertFalse(page._is_busy)
        self.assertIn("expected failure", page.status_label.text())

    def test_tqdm_progress_is_forwarded_to_calculation_status(self) -> None:
        class TrackingLogPanel(LogPanel):
            def __init__(self):
                super().__init__()
                self.messages = []

            def process_progress(self, message: str) -> None:
                self.messages.append(message)
                super().process_progress(message)

        log_panel = TrackingLogPanel()
        page = PtychographyPage(lambda: self.datacube, log_panel, WorkflowState(), stage_mode="quick")
        result = types.SimpleNamespace(images={}, metadata={}, elapsed_seconds=0.01)

        def operation():
            print("Reconstructing object and probe: 25%|██", file=sys.stderr, end="\r")
            print("Searching parameters: 40%|████", file=sys.stderr, end="\r")
            return result

        page._start("Quick Reconstruction", operation, WorkflowStep.PTYCHOGRAPHY_QUICK)
        self._run_until_thread_finishes(page)

        self.assertTrue(any("Reconstructing object and probe: 25%" in item for item in log_panel.messages))
        self.assertTrue(any("Searching parameters: 40%" in item for item in log_panel.messages))

    def _run_until_thread_finishes(self, page: PtychographyPage) -> None:
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        page.worker_thread.finished.connect(loop.quit)
        timer.start(1500)
        loop.exec()
        self.app.processEvents()
        self.assertIsNone(page.worker_thread, "Qt worker cleanup did not complete before hard timeout.")


if __name__ == "__main__":
    unittest.main()
