import types
import unittest

import numpy as np

from app.services.bragg_strain_service import BraggStrainService, StrainMapParams
from app.services.preprocessing_service import HotPixelParams, PreprocessingService
from app.services.virtual_detector_service import VirtualDetectorParams, VirtualDetectorService
from app.services.workflow_state import WorkflowState, WorkflowStep


class NotebookStrainWorkflowTests(unittest.TestCase):
    def test_hot_pixel_preview_requires_explicit_apply(self) -> None:
        data = np.ones((2, 2, 5, 5), dtype=float)
        data[:, :, 2, 2] = 100
        source = types.SimpleNamespace(data=data)
        service = PreprocessingService()
        preview = service.preview_hot_pixels(source, HotPixelParams(8))
        self.assertGreater(preview.hot_pixel_count, 0)
        self.assertEqual(source.data[0, 0, 2, 2], 100)
        service.apply_hot_pixels(source, preview)
        self.assertLess(source.data[0, 0, 2, 2], 100)

    def test_notebook_data_display_for_datacube_and_diffraction_slice(self) -> None:
        data = np.arange(3 * 5 * 4 * 6).reshape(3, 5, 4, 6)
        service = PreprocessingService()
        displayed = service.display_data(types.SimpleNamespace(data=data))
        self.assertEqual(displayed["Center diffraction pattern"].shape, (4, 6))
        self.assertEqual(displayed["Center real-space slice"].shape, (3, 5))
        diffraction = service.display_data(types.SimpleNamespace(data=np.ones((4, 6))))
        self.assertEqual(diffraction["Selected diffraction slice"].shape, (4, 6))

    def test_probe_kernel_uses_generated_kernel_for_detection_and_display(self) -> None:
        display_kernel = np.ones((4, 4))

        class Probe:
            probe = np.ones((4, 4))
            kernel = display_kernel

            def get_kernel(self, **_kwargs):
                return display_kernel

        datacube = types.SimpleNamespace(
            shape=(2, 2, 4, 4),
            data=np.ones((2, 2, 4, 4)),
            get_vacuum_probe=lambda **_kwargs: Probe(),
        )
        service = BraggStrainService()
        service._py4dstem = lambda: types.SimpleNamespace(
            process=types.SimpleNamespace(
                calibration=types.SimpleNamespace(get_probe_size=lambda _probe: (1.0, 2.0, 2.0))
            )
        )
        result = service.prepare_probe_kernel(datacube, 0, 1, 0, 1)
        np.testing.assert_array_equal(result.kernel, display_kernel)
        np.testing.assert_array_equal(service.probe_kernel, display_kernel)
        self.assertEqual(result.centered_kernel.ndim, 2)
        self.assertEqual(result.profile_plot.shape[-1], 3)

    def test_off_axis_df_and_virtual_diffraction(self) -> None:
        data = np.arange(2 * 2 * 4 * 4, dtype=float).reshape(2, 2, 4, 4)
        service = VirtualDetectorService()
        off_axis = service.compute(
            data,
            VirtualDetectorParams(VirtualDetectorService.OFF_AXIS_DARK_FIELD, 1, 1, 0, 1),
        )
        diffraction = service.compute(
            data,
            VirtualDetectorParams(
                VirtualDetectorService.VIRTUAL_DIFFRACTION, 0, 0, 0, 1, 0, 2, 0, 2
            ),
        )
        self.assertEqual(off_axis.image.shape, (2, 2))
        self.assertEqual(off_axis.detector_preview.shape, (4, 4))
        self.assertEqual(off_axis.detector_overlay["kind"], "circle")
        self.assertEqual(diffraction.image.shape, (4, 4))
        self.assertIsNone(diffraction.detector_preview)

    def test_manual_g1g2_reference(self) -> None:
        params = StrainMapParams(
            reference_mode="manual_g1g2",
            manual_g1_x=2,
            manual_g1_y=3,
            manual_g2_x=4,
            manual_g2_y=5,
        )
        reference = BraggStrainService()._strain_reference(object(), params, object())
        np.testing.assert_array_equal(reference[0], [2, 3])
        np.testing.assert_array_equal(reference[1], [4, 5])

    def test_preprocessing_invalidates_completed_strain(self) -> None:
        state = WorkflowState()
        state.mark_completed(WorkflowStep.STRAIN_MAP)
        state.mark_completed(WorkflowStep.PREPROCESS_APPLY)
        self.assertTrue(state.is_stale(WorkflowStep.STRAIN_MAP))


if __name__ == "__main__":
    unittest.main()
