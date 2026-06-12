import unittest
import sys
import types

import matplotlib.pyplot as plt
import numpy as np

sys.modules["py4DSTEM"] = types.SimpleNamespace(
    process=types.SimpleNamespace(
        calibration=types.SimpleNamespace(fit_ellipse_1D=lambda *_args, **_kwargs: (1, 2, 0.1, 4))
    )
)

from app.services.bragg_strain_service import BasisSelectionParams, BraggStrainService, StrainMapParams


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
        self.origin = None
        self.q_pixel_size = None
        self.q_pixel_units = None
        self.qr_rotation = None

    def set_p_ellipse(self, value):
        self.p_ellipse = value

    def get_ellipse(self):
        return self.p_ellipse

    def set_origin(self, value):
        self.origin = value

    def get_origin(self):
        return self.origin

    def set_Q_pixel_size(self, value):
        self.q_pixel_size = value

    def get_Q_pixel_size(self):
        return self.q_pixel_size

    def set_Q_pixel_units(self, value):
        self.q_pixel_units = value

    def get_Q_pixel_units(self):
        return self.q_pixel_units

    def set_QR_rotation_degrees(self, value):
        self.qr_rotation = value

    def get_QR_rotation_degrees(self):
        return self.qr_rotation


class _BraggVectorsForEllipse:
    calstate = {"center": True, "ellipse": False, "pixel": False, "rotate": False}

    def __init__(self):
        self.calibration = _Calibration()

    def histogram(self, mode="raw", sampling=1):
        return _Histogram()

    def setcal(self, **kwargs):
        self.calstate = kwargs


class _BraggVectorsForOrigin(_BraggVectorsForEllipse):
    def measure_origin(self):
        return np.ones((2, 2)), np.ones((2, 2)) * 2, np.ones((2, 2), dtype=bool)

    def fit_origin(self, **_kwargs):
        return (
            np.ones((2, 2)) * 1.5,
            np.ones((2, 2)) * 2.5,
            np.ones((2, 2)) * 0.1,
            np.ones((2, 2)) * 0.2,
        )


class _FakeSlice:
    data = np.ones((1, 1), dtype=bool)


class _FakeG1G2Map:
    def get_slice(self, _name):
        return _FakeSlice()


class _FakeStrainMap:
    last_coordinate_rotation = None

    def __init__(self, braggvectors):
        self.braggvectors = braggvectors
        self.g1g2_map = _FakeG1G2Map()
        self.data = [
            np.asarray([[1.0]]),
            np.asarray([[2.0]]),
            np.asarray([[0.5]]),
            np.asarray([[0.0]]),
        ]
        self.returnfig_seen = False
        self.bvm = types.SimpleNamespace(data=np.ones((7, 9)))
        self.origin = (3.0, 4.0)
        self.braggdirections = np.asarray(
            [(1.0, 0.0), (0.0, 1.0)],
            dtype=[("qx", float), ("qy", float)],
        )

    def choose_basis_vectors(self, **kwargs):
        self.returnfig_seen = kwargs.get("returnfig")
        fig, ax = plt.subplots()
        return ((3.0, 4.0), (1.0, 0.0), (0.0, 1.0), self.braggdirections), (fig, ax)

    def set_max_peak_spacing(self, max_peak_spacing=None, **_kwargs):
        self.max_peak_spacing = max_peak_spacing
        if _kwargs.get("returnfig"):
            return plt.subplots()
        return None

    def fit_basis_vectors(self, **_kwargs):
        return None

    def get_strain(self, **_kwargs):
        type(self).last_coordinate_rotation = _kwargs.get("coordinate_rotation")
        if _kwargs.get("returncalc"):
            raise AttributeError("'StrainMap' object has no attribute 'strainmap'")
        return None


class QualityResultTests(unittest.TestCase):
    def test_calibration_status_formats_scientific_values_and_rotation_degrees(self) -> None:
        calibration = _Calibration()
        calibration.set_origin((4, 5))
        calibration.set_p_ellipse((3, 2, 0.1))
        calibration.set_Q_pixel_size(0.02)
        calibration.set_Q_pixel_units("A^-1")
        calibration.set_QR_rotation_degrees(-83)

        status = BraggStrainService().calibration_status(
            types.SimpleNamespace(calibration=calibration)
        )

        self.assertEqual(status.origin, "x=4, y=5")
        self.assertIn("a=3", status.ellipse)
        self.assertIn("ellipticity=1.5", status.ellipse)
        self.assertEqual(status.pixel, "0.02 A^-1")
        self.assertEqual(status.rotate, "-83 deg")

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

    def test_strain_quality_excludes_principal_strain_components(self) -> None:
        service = BraggStrainService()
        components = {
            "exx": np.asarray([[2.0]]),
            "eyy": np.asarray([[0.0]]),
            "exy": np.asarray([[0.0]]),
        }

        quality = service.strain_quality(None, components)

        self.assertFalse(hasattr(quality, "principal_strain_1"))
        self.assertFalse(hasattr(quality, "principal_strain_2"))

    def test_ellipse_fit_requires_acceptance_before_transfer_to_target(self) -> None:
        service = BraggStrainService()
        target = _BraggVectorsForEllipse()
        reference = _BraggVectorsForEllipse()

        result = service.calibrate_ellipse(target, 1, 3, 1, fit_source=reference)

        self.assertIsNone(reference.calibration.p_ellipse)
        self.assertIsNone(target.calibration.p_ellipse)
        self.assertIn("Ellipse Reference", result.message)
        overlay = result.overlays["ellipse fit Bragg vector map"]
        self.assertEqual((overlay["inner_radius"], overlay["outer_radius"]), (1.0, 3.0))
        accepted = service.accept_pending_ellipse()
        self.assertEqual(reference.calibration.p_ellipse, (1, 2, 0.1, 4))
        self.assertEqual(target.calibration.p_ellipse, (1, 2, 0.1, 4))
        self.assertTrue(target.calstate["ellipse"])
        self.assertIn("ellipse-corrected Bragg vector map", accepted.images)

    def test_single_calibration_correction_can_transfer_between_braggvectors(self) -> None:
        service = BraggStrainService()
        target = _BraggVectorsForEllipse()
        reference = _BraggVectorsForEllipse()
        reference.calibration.set_origin((4, 5))
        reference.calibration.set_Q_pixel_size(0.03)
        reference.calibration.set_Q_pixel_units("A^-1")

        origin_result = service.transfer_calibration_correction(target, reference, "origin")
        pixel_result = service.transfer_calibration_correction(target, reference, "pixel")

        self.assertEqual(target.calibration.origin, (4, 5))
        self.assertEqual(target.calibration.q_pixel_size, 0.03)
        self.assertEqual(target.calibration.q_pixel_units, "A^-1")
        self.assertTrue(target.calstate["center"])
        self.assertTrue(target.calstate["pixel"])
        self.assertIn("origin", origin_result.message)
        self.assertIn("Q pixel size", pixel_result.message)

    def test_origin_calibration_exposes_six_process_maps_and_comparison(self) -> None:
        service = BraggStrainService()
        source = _BraggVectorsForOrigin()
        previous = dict(source.calstate)

        process = service.calibrate_origin(source)
        comparison = service.compare_origin_correction(source)

        self.assertEqual(
            list(process.images),
            ["qx measured", "qx fitted", "qx residual", "qy measured", "qy fitted", "qy residual"],
        )
        self.assertEqual(
            list(comparison.images),
            ["raw Bragg vector map", "origin-centered Bragg vector map"],
        )
        self.assertEqual(source.calstate, previous)

    def test_setting_qr_rotation_leaves_rotation_applied(self) -> None:
        service = BraggStrainService()
        source = _BraggVectorsForEllipse()

        result = service.set_qr_rotation(source, -83)

        self.assertTrue(source.calstate["rotate"])
        self.assertEqual(source.calibration.get_QR_rotation_degrees(), -83)
        self.assertIn("applied", result.message)

    def test_qr_rotation_comparison_exposes_real_and_reciprocal_arrows(self) -> None:
        service = BraggStrainService()
        source = _BraggVectorsForEllipse()
        result = service.set_qr_rotation(
            source,
            -83,
            {
                "target DataCube bright-field": np.ones((80, 100)),
                "rotation reference bright-field": np.ones((220, 240)),
            },
        )

        self.assertIn("target DataCube bright-field", result.vectors)
        self.assertIn("rotation reference bright-field", result.vectors)
        self.assertEqual(result.vectors["target DataCube bright-field"].shape, (1, 4))

    def test_strain_map_does_not_block_on_incomplete_calibration(self) -> None:
        service = BraggStrainService()

        with self.assertRaises(Exception) as context:
            service.compute_strain_map(_BraggVectorsForEllipse(), StrainMapParams())

        self.assertNotIn("Apply origin, ellipse, pixel, and rotation corrections", str(context.exception))

    def test_strain_map_closes_py4dstem_basis_figure(self) -> None:
        previous = sys.modules["py4DSTEM"]
        sys.modules["py4DSTEM"] = types.SimpleNamespace(StrainMap=_FakeStrainMap)
        try:
            service = BraggStrainService()
            result = service.compute_strain_map(
                _BraggVectorsForEllipse(), StrainMapParams(coordinate_rotation=-21.5)
            )
        finally:
            sys.modules["py4DSTEM"] = previous

        self.assertEqual(result.components["exx"][0, 0], 1.0)
        self.assertEqual(_FakeStrainMap.last_coordinate_rotation, -21.5)
        self.assertEqual(plt.get_fignums(), [])

    def test_strain_stages_expose_basis_spacing_and_fit_quality(self) -> None:
        previous = sys.modules["py4DSTEM"]
        sys.modules["py4DSTEM"] = types.SimpleNamespace(StrainMap=_FakeStrainMap)
        try:
            service = BraggStrainService()
            source = _BraggVectorsForEllipse()
            basis = service.choose_strain_basis(source, BasisSelectionParams())
            spacing = service.set_strain_peak_spacing(3)
            fitted = service.fit_strain_basis()
        finally:
            sys.modules["py4DSTEM"] = previous

        self.assertIn("basis selection", basis.vectors)
        self.assertEqual(spacing.quality["max_peak_spacing"], 3)
        self.assertIn("peak acceptance regions", spacing.circles)
        self.assertEqual(fitted.quality["valid_fraction"], 1)
        state = service.accept_strain_stage("basis_selection")
        self.assertTrue(state.basis_selection)
        self.assertFalse(state.reference)


if __name__ == "__main__":
    unittest.main()
