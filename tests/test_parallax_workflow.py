import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.services.parallax_service import (
    BFMaskParams,
    FAST_ALIGNMENT_BINS,
    FiniteDoseParams,
    NOTEBOOK_ALIGNMENT_BINS,
    ParallaxAdvancedParams,
    ParallaxAlignmentParams,
    ParallaxService,
    Py4DSTEMParallaxAdapter,
)


class _FakeParallax:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.preprocess_kwargs = {}
        self.reconstruct_kwargs = {}
        self.subpixel_kwargs = {}
        self.fit_kwargs = {}
        self.correct_kwargs = {}
        self.object_cropped = np.ones((3, 4))
        self._shifts = np.ones((2, 3, 2))
        self._xy_shifts = np.ones((80, 2))
        self.recon_BF = np.ones((6, 7))
        self.recon_BF_subpixel_aligned = np.ones((12, 14))
        self._scan_sampling = (1.0, 1.0)
        self._kde_upsample_factor = 2

    def preprocess(self, dp_mask=None, edge_blend=0, normalize_images=False, **kwargs):
        self.preprocess_kwargs = {
            "dp_mask": dp_mask, "edge_blend": edge_blend,
            "normalize_images": normalize_images, **kwargs,
        }
        return self

    def reconstruct(self, alignment_bin_values=None, progress_bar=True, **kwargs):
        self.reconstruct_kwargs = {
            "alignment_bin_values": alignment_bin_values,
            "progress_bar": progress_bar, **kwargs,
        }
        return self

    def subpixel_alignment(self, kde_upsample_factor=None, progress_bar=True, **kwargs):
        self.subpixel_kwargs = {
            "kde_upsample_factor": kde_upsample_factor, "progress_bar": progress_bar, **kwargs,
        }
        return self

    def aberration_fit(self, max_radial_order=3, max_angular_order=4, **kwargs):
        self.fit_kwargs = {
            "max_radial_order": max_radial_order, "max_angular_order": max_angular_order, **kwargs,
        }
        return self

    def aberration_correct(self, plot_corrected_phase=True, **kwargs):
        self.correct_kwargs = {"plot_corrected_phase": plot_corrected_phase, **kwargs}
        return self


class ParallaxWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.created = []

        def construct(**kwargs):
            instance = _FakeParallax(**kwargs)
            self.created.append(instance)
            return instance

        module = types.SimpleNamespace(
            __version__="test",
            process=types.SimpleNamespace(phase=types.SimpleNamespace(Parallax=construct))
        )
        self.saved_h5_paths = []
        module.save = lambda filepath, _data: self.saved_h5_paths.append(Path(filepath))
        self.service = ParallaxService(Py4DSTEMParallaxAdapter(module))
        data = np.zeros((2, 3, 8, 10))
        data[..., 2:6, 3:8] = 1
        self.datacube = types.SimpleNamespace(data=data)

    def test_accepted_bf_mask_is_copied_and_forwarded_as_dp_mask(self):
        self.service.prepare_bf(self.datacube, BFMaskParams(threshold=0.5))
        accepted = self.service.accept_bf_mask()
        accepted[:] = False

        self.service.align(self.datacube, ParallaxAlignmentParams())

        forwarded = self.created[0].preprocess_kwargs["dp_mask"]
        self.assertTrue(forwarded.any())
        self.assertIsNot(forwarded, self.service.context.accepted_bf_mask)
        np.testing.assert_array_equal(forwarded, self.service.context.accepted_bf_mask)

    def test_adapter_forwards_supported_version_arguments(self):
        self.service.prepare_bf(self.datacube, BFMaskParams())
        self.service.accept_bf_mask()
        self.service.align(
            self.datacube,
            ParallaxAlignmentParams(edge_blend=12, cross_correlation_upsample_factor=16),
        )

        instance = self.created[0]
        self.assertEqual(instance.preprocess_kwargs["edge_blend"], 12)
        self.assertFalse(instance.reconstruct_kwargs["progress_bar"])
        self.assertIn("Shift Magnitude", self.service.context.alignment_result.images)

    def test_fast_alignment_defaults_and_notebook_quality_schedule(self):
        alignment = ParallaxAlignmentParams()
        params = ParallaxAdvancedParams()

        self.assertEqual(alignment.alignment_bin_values, FAST_ALIGNMENT_BINS)
        self.assertEqual(alignment.cross_correlation_upsample_factor, 4)
        self.assertEqual(NOTEBOOK_ALIGNMENT_BINS, (32, 32, 32, 32, 32, 32, 16, 16, 16, 16, 8, 8))
        self.assertEqual(len(FAST_ALIGNMENT_BINS), len(NOTEBOOK_ALIGNMENT_BINS) // 2)
        self.assertFalse(params.high_order_fit)

    def test_prepare_bf_prefers_datacube_mean_diffraction_api(self):
        calls = []
        datacube = types.SimpleNamespace(
            data=self.datacube.data,
            get_dp_mean=lambda: calls.append(True) or types.SimpleNamespace(
                data=self.datacube.data.mean(axis=(0, 1))
            ),
        )

        result = self.service.prepare_bf(datacube, BFMaskParams())

        self.assertEqual(calls, [True])
        self.assertIn("Incoherent BF", result.images)

    def test_progress_is_stage_based(self):
        self.service.prepare_bf(self.datacube, BFMaskParams())
        self.service.accept_bf_mask()
        messages = []

        self.service.align(
            self.datacube, ParallaxAlignmentParams(),
            lambda message, fraction: messages.append((message, fraction)),
        )

        self.assertEqual(
            [message for message, _fraction in messages],
            ["BF Preparation", "Preprocess", "Reconstruct", "Review"],
        )

    def test_package_save_is_explicit_and_writes_h5_and_metadata(self):
        self.service.prepare_bf(self.datacube, BFMaskParams())
        self.service.accept_bf_mask()
        self.service.align(self.datacube, ParallaxAlignmentParams())
        with patch.object(Path, "write_text", return_value=1):
            saved = self.service.save_package(Path.cwd())

        self.assertIn(Path.cwd() / "parallax_reconstruction.h5", saved)
        self.assertIn(Path.cwd() / "parallax_pipeline_metadata.json", saved)

    def test_representative_virtual_bf_selection_and_crop(self):
        data = np.zeros((60, 70, 8, 10))
        data[..., 2:6, 3:8] = 1
        result = self.service.prepare_bf(
            types.SimpleNamespace(data=data),
            BFMaskParams(threshold=0.5, virtual_bf_count=5, virtual_bf_crop=48),
        )

        self.assertEqual(result.metadata["selected_points"].shape[1], 2)
        self.assertEqual(len([name for name in result.images if name.startswith("Tilted virtual BF")]), 5)
        self.assertEqual(result.images["Tilted virtual BF 1"].shape, (48, 48))

    def test_excessive_bf_mask_cannot_be_accepted(self):
        result = self.service.prepare_bf(self.datacube, BFMaskParams(threshold=-1))

        self.assertFalse(result.metadata["mask_acceptable"])
        with self.assertRaisesRegex(Exception, "more than 75%"):
            self.service.accept_bf_mask()

    def test_advanced_actions_are_separate_and_dependent(self):
        self.service.prepare_bf(self.datacube, BFMaskParams())
        self.service.accept_bf_mask()
        self.service.align(self.datacube, ParallaxAlignmentParams())
        self.service.accept_alignment()

        subpixel = self.service.run_subpixel(ParallaxAdvancedParams())
        fit = self.service.fit_aberrations(ParallaxAdvancedParams())
        correction = self.service.apply_aberration_correction()

        self.assertIs(self.service.context.subpixel_result, subpixel)
        self.assertIs(self.service.context.aberration_result, fit)
        self.assertIs(self.service.context.correction_result, correction)
        self.assertFalse(self.service.adapter.capabilities().ctf_thon_ring_fit)
        self.assertIn("Original Aligned BF FFT", subpixel.images)
        self.assertIn("Subpixel Aligned BF FFT", subpixel.images)
        self.assertIn("radial_cone_values", subpixel.metadata)

    def test_finite_dose_includes_safe_three_pattern_montages(self):
        class Calibration:
            @staticmethod
            def get_R_pixel_size():
                return 1.0

        data = np.zeros((2, 3, 8, 10))
        data[..., 2:6, 3:8] = 1
        datacube = types.SimpleNamespace(
            data=data,
            calibration=Calibration(),
            copy=lambda: types.SimpleNamespace(
                data=data.copy(),
                calibration=Calibration(),
            ),
        )
        self.service.prepare_bf(datacube, BFMaskParams())
        self.service.accept_bf_mask()

        result = self.service.run_finite_dose_comparison(
            datacube,
            ParallaxAlignmentParams(),
            FiniteDoseParams(doses=(10,), seed=1),
        )

        self.assertEqual(result.images["Diffraction montage 10 e/A2"].shape, (8, 30))


if __name__ == "__main__":
    unittest.main()
