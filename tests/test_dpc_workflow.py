import types
import unittest

import numpy as np

from app.pages.dpc_page import DPCPage
from app.services.phase_contrast_service import (
    DPCPreprocessParams,
    DPCReconstructionParams,
    DPCSegmentedParams,
    PhaseContrastService,
    PhaseContrastServiceError,
)


class _FakeDPC:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.preprocess_kwargs = None
        self.reconstruct_kwargs = None
        shape = (3, 4)
        self._com_measured_x = np.ones(shape)
        self._com_measured_y = np.ones(shape) * 2
        self._com_fitted_x = np.ones(shape) * 3
        self._com_fitted_y = np.ones(shape) * 4
        self._com_normalized_x = np.ones(shape) * 5
        self._com_normalized_y = np.ones(shape) * 6
        self._com_x = np.ones(shape) * 7
        self._com_y = np.ones(shape) * 8
        self._rotation_angles_deg = np.array([-1.0, 0.0, 1.0])
        self._rotation_curl = np.array([3.0, 1.0, 2.0])
        self._rotation_curl_transpose = np.array([4.0, 5.0, 6.0])
        self._rotation_best_rad = np.deg2rad(-15)
        self._rotation_best_transpose = False

    def preprocess(self, **kwargs):
        self.preprocess_kwargs = kwargs
        return self

    def reconstruct(self, **kwargs):
        self.reconstruct_kwargs = kwargs
        self.object_phase = np.arange(12).reshape(3, 4)
        self.error_iterations = np.array([1.0, 0.25, 0.05])
        self.object_phase_iterations = [self.object_phase * 0, self.object_phase]
        return self


class _FakeVirtualImageDataCube:
    def __init__(self) -> None:
        self.data = np.ones((2, 3, 4, 5))
        self.calls: list[dict[str, object]] = []

    def get_virtual_image(self, **kwargs):
        self.calls.append(kwargs)
        value = 1 if kwargs["mode"] == "circle" else 2
        return np.full(self.data.shape[:2], value, dtype=float)


class DPCWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.created = []

        def make_dpc(**kwargs):
            dpc = _FakeDPC(**kwargs)
            self.created.append(dpc)
            return dpc

        self.fake_py4dstem = types.SimpleNamespace(
            process=types.SimpleNamespace(
                phase=types.SimpleNamespace(DPC=make_dpc),
                utils=types.SimpleNamespace(electron_wavelength_angstrom=lambda _energy: 1.0),
            )
        )
        self.service = PhaseContrastService()
        self.service._py4dstem = lambda: self.fake_py4dstem
        self.datacube = types.SimpleNamespace(data=np.ones((3, 4, 16, 18)))

    def test_sto_preset_matches_notebook_code(self) -> None:
        preset = self.service.dpc_sto_preset()

        self.assertEqual(preset.energy, 200e3)
        self.assertEqual(preset.rotation_offset_degrees, 60)
        self.assertEqual(preset.inner_radius_mrad, 10)
        self.assertEqual(preset.outer_radius_mrad, 25)
        self.assertEqual(preset.sampling_x, 0.246570625)
        self.assertEqual(preset.sampling_y, 0.246570625)

    def test_bf_df_task_carries_metadata_and_matches_compute(self) -> None:
        datacube = _FakeVirtualImageDataCube()
        progress: list[tuple[str, float]] = []

        task = self.service.compute_bf_df_task(datacube, bf_radius=3, df_inner=4, df_outer=6)
        result = task.run(lambda message, fraction: progress.append((message, fraction)))

        self.assertEqual(task.name, "BF / DF Preview")
        self.assertIn("bf_df_preview:", task.result_key or "")
        self.assertEqual(task.parameters["shape"], datacube.data.shape)
        self.assertIn(("Preparing BF / DF preview", 0.0), progress)
        np.testing.assert_array_equal(result["Bright Field"], np.ones(datacube.data.shape[:2]))
        np.testing.assert_array_equal(result["Dark Field"], np.ones(datacube.data.shape[:2]) * 2)
        self.assertEqual(len(datacube.calls), 2)

    def test_four_segment_demonstration_produces_notebook_outputs(self) -> None:
        result = self.service.generate_segmented_dpc(
            self.datacube,
            DPCSegmentedParams(inner_radius_mrad=0, outer_radius_mrad=1000),
        )

        self.assertEqual(len(result.masks), 4)
        self.assertEqual(result.images["Segment 1 intensity"].shape, (3, 4))
        np.testing.assert_array_equal(
            result.images["Segmented CoM X"],
            result.images["Segment 1 intensity"] - result.images["Segment 3 intensity"],
        )
        self.assertIn("Segmented complex CoM", result.complex_images)
        self.assertIn("Weighted complex CoM", result.complex_images)

    def test_preprocess_forwards_supported_parameters_without_plotting(self) -> None:
        params = DPCPreprocessParams(
            padding_factor=3,
            rotation_start_degrees=-30,
            rotation_end_degrees=30,
            rotation_step_degrees=2,
            maximize_divergence=True,
            fit_function="parabola",
            force_com_rotation=-15,
            force_com_transpose=True,
            force_com_shift_x=1.5,
            force_com_shift_y=-2.5,
            vectorized_com_calculation=True,
        )

        result = self.service.preprocess_dpc(self.datacube, params)
        forwarded = self.created[0].preprocess_kwargs

        self.assertEqual(self.created[0].init_kwargs["energy"], 200e3)
        self.assertEqual(forwarded["padding_factor"], 3)
        self.assertEqual(forwarded["force_com_rotation"], -15)
        self.assertEqual(forwarded["force_com_transpose"], True)
        self.assertEqual(forwarded["force_com_shifts"], (1.5, -2.5))
        self.assertFalse(forwarded["plot_center_of_mass"])
        self.assertFalse(forwarded["plot_rotation"])
        self.assertAlmostEqual(result.rotation_degrees, -15)
        self.assertIsNotNone(result.complex_com)
        self.assertFalse(self.service.dpc_acceptance.preprocessing)

    def test_reconstruction_requires_acceptance_and_forwards_parameters(self) -> None:
        self.service.preprocess_dpc(self.datacube, DPCPreprocessParams())
        with self.assertRaises(PhaseContrastServiceError):
            self.service.reconstruct_dpc(DPCReconstructionParams())

        self.service.accept_dpc_preprocessing()
        params = DPCReconstructionParams(
            reset=False,
            max_iter=12,
            step_size=0.25,
            stopping_criterion=1e-5,
            backtrack=False,
            gaussian_filter=False,
            gaussian_filter_sigma=1.5,
            butterworth_filter=False,
            q_lowpass=0.1,
            q_highpass=0.01,
            butterworth_order=3,
            store_iterations=True,
        )
        result = self.service.reconstruct_dpc(params)
        forwarded = self.created[0].reconstruct_kwargs

        self.assertEqual(forwarded["max_iter"], 12)
        self.assertEqual(forwarded["step_size"], 0.25)
        self.assertTrue(forwarded["store_iterations"])
        np.testing.assert_array_equal(result.reconstructed_potential, self.created[0].object_phase)
        np.testing.assert_array_equal(result.error_iterations, [1.0, 0.25, 0.05])
        self.assertEqual(len(result.object_iterations), 2)

    def test_complex_rendering_does_not_modify_export_array(self) -> None:
        complex_image = np.array([[1 + 2j, 3 - 4j]])
        original = complex_image.copy()

        rendered = DPCPage._complex_rgb(complex_image)

        self.assertEqual(rendered.shape, (1, 2, 3))
        self.assertEqual(rendered.dtype, np.uint8)
        np.testing.assert_array_equal(complex_image, original)

    def test_datacube_calibration_sampling_is_used_when_available(self) -> None:
        calibration = types.SimpleNamespace(get_R_pixel_size=lambda: (0.12, 0.34))
        source = types.SimpleNamespace(calibration=calibration)

        self.assertEqual(DPCPage._source_sampling(source), (0.12, 0.34))

    def test_reset_clears_shared_dpc_state(self) -> None:
        self.service.preprocess_dpc(self.datacube, DPCPreprocessParams())
        self.service.accept_dpc_preprocessing()

        self.service.reset_dpc_workflow()

        self.assertIsNone(self.service.dpc)
        self.assertIsNone(self.service.dpc_preprocess_result)
        self.assertFalse(self.service.dpc_acceptance.preprocessing)


if __name__ == "__main__":
    unittest.main()
