import unittest
import sys
import types

import numpy as np

sys.modules.setdefault("py4DSTEM", types.SimpleNamespace())

from app.services.bragg_strain_service import BraggStrainService


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


class _BraggVectors:
    def __init__(self):
        self.raw = _RawPeaks()

    def histogram(self, mode="raw"):
        return _Histogram()


class _BraggVectorsWithoutRaw:
    shape = (3, 4)

    def histogram(self, mode="raw"):
        return _Histogram()


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


if __name__ == "__main__":
    unittest.main()
