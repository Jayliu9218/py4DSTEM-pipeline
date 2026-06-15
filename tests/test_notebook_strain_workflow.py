import types
import unittest

import numpy as np

from app.services.bragg_strain_service import (
    BraggDetectionParams,
    BraggStrainService,
    CBSPreset,
    StrainMapParams,
)
from app.services.preprocessing_service import HotPixelParams, PreprocessingService
from app.services.virtual_detector_service import VirtualDetectorParams, VirtualDetectorService
from app.services.workflow_state import WorkflowState, WorkflowStep


class NotebookStrainWorkflowTests(unittest.TestCase):
    def test_au_notebook_preset_matches_cbs_parameters(self) -> None:
        params = CBSPreset.au_notebook().bragg
        self.assertEqual(
            (
                params.min_absolute_intensity,
                params.min_relative_intensity,
                params.min_peak_spacing,
                params.edge_boundary,
                params.sigma_cc,
                params.max_num_peaks,
                params.subpixel,
                params.corr_power,
            ),
            (2, 0, 18, 2, 0, 100, "poly", 1),
        )

    def test_bragg_detection_requires_kernel_unless_fallback_is_explicit(self) -> None:
        datacube = types.SimpleNamespace(
            shape=(1, 1, 4, 4),
            data=np.ones((1, 1, 4, 4)),
        )
        with self.assertRaisesRegex(Exception, "Prepare probe.kernel"):
            BraggStrainService().detect_peaks(datacube, 0, 0, BraggDetectionParams())

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
        result = service.prepare_probe_kernel(datacube, 0, 2, 0, 2)
        np.testing.assert_array_equal(result.kernel, display_kernel)
        np.testing.assert_array_equal(service.probe_kernel, display_kernel)
        self.assertEqual(result.centered_kernel.ndim, 2)
        self.assertEqual(result.profile_plot.shape[-1], 3)

    def test_off_axis_df_and_virtual_diffraction(self) -> None:
        data = np.arange(2 * 3 * 4 * 4, dtype=float).reshape(2, 3, 4, 4)
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
        self.assertEqual(off_axis.image.shape, (2, 3))
        self.assertEqual(off_axis.detector_preview.shape, (4, 4))
        self.assertEqual(off_axis.detector_overlay["kind"], "circle")
        self.assertEqual(diffraction.image.shape, (4, 4))
        self.assertIsNone(diffraction.detector_preview)

    def test_circular_virtual_diffraction_uses_real_space_mask(self) -> None:
        data = np.arange(5 * 7 * 3 * 4, dtype=float).reshape(5, 7, 3, 4)
        result = VirtualDetectorService().compute(
            data,
            VirtualDetectorParams(
                VirtualDetectorService.VIRTUAL_DIFFRACTION,
                0, 0, 0, 1,
                roi_mode="circle",
                roi_center_x=2,
                roi_center_y=3,
                roi_radius=1.5,
            ),
        )
        self.assertEqual(result.image.shape, (3, 4))
        self.assertEqual(result.real_space_preview.shape, (5, 7))
        self.assertEqual(result.real_space_overlay["kind"], "circle")

    def test_global_reference_passes_none_after_valid_basis_fit(self) -> None:
        strainmap = types.SimpleNamespace(
            g1g2_map=types.SimpleNamespace(
                get_slice=lambda _name: types.SimpleNamespace(data=np.ones((2, 3), dtype=bool))
            )
        )
        reference = BraggStrainService()._strain_reference(
            strainmap, StrainMapParams(reference_mode="global_none"), object()
        )
        self.assertIsNone(reference)

    def test_roi_reference_uses_get_reference_g1g2(self) -> None:
        expected_g1 = np.asarray([2.0, 3.0])
        expected_g2 = np.asarray([4.0, 5.0])
        captured = {}

        def get_reference_g1g2(roi):
            captured["roi"] = roi
            return expected_g1, expected_g2

        strainmap = types.SimpleNamespace(get_reference_g1g2=get_reference_g1g2)
        braggvectors = types.SimpleNamespace(raw=types.SimpleNamespace(shape=(4, 6)))
        params = StrainMapParams(
            reference_mode="roi_g1g2",
            roi_rx_start=1,
            roi_rx_end=3,
            roi_ry_start=2,
            roi_ry_end=5,
        )
        reference = BraggStrainService()._strain_reference(strainmap, params, braggvectors)
        np.testing.assert_array_equal(reference[0], expected_g1)
        np.testing.assert_array_equal(reference[1], expected_g2)
        self.assertEqual(int(captured["roi"].sum()), 6)

    def test_final_strain_receives_only_none_or_roi_derived_vectors(self) -> None:
        class StrainMap:
            data = [np.zeros((2, 3)) for _ in range(4)]
            bvm = types.SimpleNamespace(data=np.ones((5, 7)))
            g1g2_map = types.SimpleNamespace(
                get_slice=lambda _name: types.SimpleNamespace(data=np.ones((2, 3), dtype=bool))
            )

            def __init__(self):
                self.received_gvects = "unset"

            def get_reference_g1g2(self, _roi):
                return np.asarray([2.0, 3.0]), np.asarray([4.0, 5.0])

            def get_strain(self, **kwargs):
                self.received_gvects = kwargs["gvects"]

        braggvectors = types.SimpleNamespace(raw=types.SimpleNamespace(shape=(2, 3)))
        service = BraggStrainService()
        global_map = StrainMap()
        service._finish_strain_map(global_map, braggvectors, StrainMapParams())
        self.assertIsNone(global_map.received_gvects)

        roi_map = StrainMap()
        service._finish_strain_map(
            roi_map,
            braggvectors,
            StrainMapParams(
                reference_mode="roi_g1g2",
                roi_rx_start=0,
                roi_rx_end=1,
                roi_ry_start=0,
                roi_ry_end=2,
            ),
        )
        self.assertIsInstance(roi_map.received_gvects, tuple)
        np.testing.assert_array_equal(roi_map.received_gvects[0], [2.0, 3.0])
        np.testing.assert_array_equal(roi_map.received_gvects[1], [4.0, 5.0])

    def test_preprocessing_invalidates_completed_strain(self) -> None:
        state = WorkflowState()
        state.mark_completed(WorkflowStep.STRAIN_MAP)
        state.mark_completed(WorkflowStep.PREPROCESS_APPLY)
        self.assertTrue(state.is_stale(WorkflowStep.STRAIN_MAP))


if __name__ == "__main__":
    unittest.main()
