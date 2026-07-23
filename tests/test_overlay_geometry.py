import unittest

from pyside_overlay import qt_rect_to_physical_rect, physical_rect_to_qt_rect


class OverlayGeometryTests(unittest.TestCase):
    def test_physical_target_area_is_converted_to_qt_logical_rect(self):
        self.assertEqual(
            physical_rect_to_qt_rect([125, 250, 625, 500], 1.25),
            [100, 200, 500, 400],
        )

    def test_qt_logical_rect_is_converted_back_to_physical_target_area(self):
        self.assertEqual(
            qt_rect_to_physical_rect([100, 200, 500, 400], 1.25),
            [125, 250, 625, 500],
        )

    def test_invalid_scale_falls_back_to_unscaled_coordinates(self):
        self.assertEqual(
            physical_rect_to_qt_rect([10, 20, 110, 120], 0),
            [10, 20, 110, 120],
        )


if __name__ == "__main__":
    unittest.main()
