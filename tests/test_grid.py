"""四点透视标定和 9×9 网格映射测试。"""

import unittest

import cv2
import numpy as np

from formation.grid import (
    draw_perspective_grid,
    locate_on_grid,
    person_to_grid_json,
)
from vision.coordinate import CalibrationError, StageCalibration
from vision.tracker import TrackedPerson


class StageCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = StageCalibration.from_points(
            [(10, 10), (190, 10), (190, 190), (10, 190)],
            frame_width=200,
            frame_height=200,
            columns=9,
            rows=9,
        )

    def test_rectangle_center_maps_to_stage_center(self) -> None:
        point = self.calibration.pixel_to_stage(100, 100)

        self.assertAlmostEqual(point.x, 4.5, places=5)
        self.assertAlmostEqual(point.y, 4.5, places=5)

    def test_stage_round_trip_with_trapezoid(self) -> None:
        calibration = StageCalibration.from_points(
            [(70, 20), (330, 30), (390, 280), (20, 270)],
            frame_width=400,
            frame_height=300,
        )

        pixel = calibration.stage_to_pixel(3.25, 7.4)
        stage = calibration.pixel_to_stage(*pixel)

        self.assertAlmostEqual(stage.x, 3.25, places=4)
        self.assertAlmostEqual(stage.y, 7.4, places=4)

    def test_invalid_click_order_is_rejected(self) -> None:
        with self.assertRaises(CalibrationError):
            StageCalibration.from_points(
                [(10, 10), (190, 190), (190, 10), (10, 190)],
                frame_width=200,
                frame_height=200,
            )

    def test_four_points_are_required(self) -> None:
        with self.assertRaisesRegex(CalibrationError, "当前只有 3 个"):
            StageCalibration.from_points(
                [(10, 10), (190, 10), (190, 190)],
                frame_width=200,
                frame_height=200,
            )

    def test_grid_cells_are_one_based_and_edges_are_clamped(self) -> None:
        top_left = locate_on_grid(10, 10, self.calibration)
        bottom_right = locate_on_grid(190, 190, self.calibration)
        center = locate_on_grid(100, 100, self.calibration)

        self.assertEqual((top_left.column, top_left.row), (1, 1))
        self.assertEqual((bottom_right.column, bottom_right.row), (9, 9))
        self.assertEqual((center.column, center.row), (5, 5))

    def test_outside_stage_has_no_grid_cell(self) -> None:
        position = locate_on_grid(0, 100, self.calibration)

        self.assertFalse(position.in_stage)
        self.assertIsNone(position.column)
        self.assertIsNone(position.row)

    def test_person_json_contains_all_coordinate_systems(self) -> None:
        person = TrackedPerson(3, 100.0, 100.0, (80, 40, 120, 100))

        payload = person_to_grid_json(person, self.calibration)

        self.assertEqual(payload["id"], 3)
        self.assertEqual(payload["x"], 100)
        self.assertEqual(payload["y"], 100)
        self.assertAlmostEqual(payload["stage_x"], 4.5)
        self.assertAlmostEqual(payload["stage_y"], 4.5)
        self.assertEqual(payload["grid_col"], 5)
        self.assertEqual(payload["grid_row"], 5)
        self.assertTrue(payload["in_stage"])

    def test_grid_overlay_changes_frame_without_changing_shape(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)

        rendered = draw_perspective_grid(frame, self.calibration)

        self.assertEqual(rendered.shape, frame.shape)
        self.assertGreater(int(cv2.countNonZero(cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY))), 0)


if __name__ == "__main__":
    unittest.main()
