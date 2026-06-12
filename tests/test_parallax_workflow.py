import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.services.parallax_service import (
    BFMaskParams,
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
        self.datacube = types.SimpleNamespace(data=np.ones((2, 3, 8, 10)))

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

    def test_advanced_defaults_enable_only_subpixel(self):
        params = ParallaxAdvancedParams()

        self.assertTrue(params.run_subpixel)
        self.assertFalse(params.run_aberration_fit)
        self.assertFalse(params.run_aberration_correction)
        self.assertFalse(params.run_high_order_fit)
        self.assertFalse(params.run_ctf_fit)

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


if __name__ == "__main__":
    unittest.main()
