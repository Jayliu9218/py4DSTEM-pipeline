import unittest
import sys
import types

import numpy as np

sys.modules["py4DSTEM"] = types.SimpleNamespace(
    process=types.SimpleNamespace(
        calibration=types.SimpleNamespace(fit_ellipse_1D=lambda *_args, **_kwargs: (1, 2, 0.1, 4))
    )
)

from app.services.bragg_strain_service import BraggStrainService, StrainMapParams


class _PeakCell:
    def __init__(self, peaks):
        self.data = np.asarray(peaks, dtype=float)


class _RawPeaks:
    shape = (2, 2)

    def __init__(self):
        self.cells = [
            [_PeakCell([[1, 2, 10], [3, 4, 20]]), _PeakCell([])],
            [_PeakCell([[5, 6, 30]]), _PeakCell([[7, 8, 5], [9, 10, 15]])],
        ]

    def __getitem__(self, index):
        rx, ry = index
        return self.cells[rx][ry]


class _Histogram:
    def __init__(self):
        self.data = np.ones((4, 4))
        self.origin = (2, 2)


class _BraggVectors:
    def __init__(self):
        self.raw = _RawPeaks()

    def histogram(self, mode="raw"):
        return _Histogram()


class _BraggVectorsWithoutRaw:
    shape = (3, 4)

    def histogram(self, mode="raw"):
        return _Histogram()


class _Calibration:
    def __init__(self):
        self.p_ellipse = None

    def set_p_ellipse(self, value):
        self.p_ellipse = value


class _BraggVectorsForEllipse:
    calstate = {"center": True, "ellipse": False, "pixel": False, "rotate": False}

    def __init__(self):
        self.calibration = _Calibration()

    def histogram(self, mode="raw", sampling=1):
        return _Histogram()

    def setcal(self, **kwargs):
        self.calstate = kwargs


class QualityResultTests(unittest.TestCase):
    def test_bragg_quality_maps_from_raw_peak_cells(self) -> None:
        service = BraggStrainService()

        quality = service.bragg_quality(_BraggVectors())

        np.testing.assert_array_equal(quality.peak_count_map, [[2, 0], [1, 2]])
        self.assertEqual(quality.mean_intensity_map[0, 0], 15)
        self.assertEqual(quality.max_intensity_map[1, 0], 30)
        self.assertTrue(quality.failure_mask[0, 1])

    def test_bragg_quality_degrades_when_raw_peak_data_is_missing(self) -> None:
        service = BraggStrainService()

        quality = service.bragg_quality(_BraggVectorsWithoutRaw())

        self.assertEqual(quality.peak_count_map.shape, (3, 4))
        np.testing.assert_array_equal(quality.peak_count_map, np.zeros((3, 4)))

    def test_strain_quality_adds_principal_strain_components(self) -> None:
        service = BraggStrainService()
        components = {
            "exx": np.asarray([[2.0]]),
            "eyy": np.asarray([[0.0]]),
            "exy": np.asarray([[0.0]]),
        }

        quality = service.strain_quality(None, components)

        self.assertEqual(quality.principal_strain_1[0, 0], 2.0)
        self.assertEqual(quality.principal_strain_2[0, 0], 0.0)

    def test_ellipse_fit_can_use_reference_braggvectors_and_transfer_to_target(self) -> None:
        service = BraggStrainService()
        target = _BraggVectorsForEllipse()
        reference = _BraggVectorsForEllipse()

        result = service.calibrate_ellipse(target, 1, 3, 1, fit_source=reference)

        self.assertEqual(reference.calibration.p_ellipse, (1, 2, 0.1, 4))
        self.assertEqual(target.calibration.p_ellipse, (1, 2, 0.1, 4))
        self.assertIn("Ellipse Reference", result.message)

    def test_strain_map_does_not_block_on_incomplete_calibration(self) -> None:
        service = BraggStrainService()

        with self.assertRaises(Exception) as context:
            service.compute_strain_map(_BraggVectorsForEllipse(), StrainMapParams())

        self.assertNotIn("Apply origin, ellipse, pixel, and rotation corrections", str(context.exception))


if __name__ == "__main__":
    unittest.main()
