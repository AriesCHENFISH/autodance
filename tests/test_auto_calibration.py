"""自动舞台标定测试。"""

import os
import tempfile
import unittest

import cv2
import numpy as np

from vision.auto_calibration import _fit_symmetric_trapezoid, auto_calibrate_stage


class FitSymmetricTrapezoidTests(unittest.TestCase):
    def test_recovers_trapezoid_geometry(self) -> None:
        rng = np.random.default_rng(0)
        center, top_y, bottom_y = 640.0, 300.0, 600.0
        top_w, bottom_w = 400.0, 800.0
        points = []
        for _ in range(3000):
            y = rng.uniform(top_y, bottom_y)
            frac = (y - top_y) / (bottom_y - top_y)
            width = top_w + (bottom_w - top_w) * frac
            x = rng.uniform(center - width / 2, center + width / 2)
            points.append([x, y])
        cx, ty, tw, bw = _fit_symmetric_trapezoid(np.asarray(points))
        self.assertAlmostEqual(cx, center, delta=25)
        self.assertAlmostEqual(tw, top_w, delta=80)
        self.assertAlmostEqual(bw, bottom_w, delta=80)
        self.assertLess(tw, bw)


class _FakeResult:
    def __init__(self, boxes_xyxy):
        self._boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)

        class Boxes:
            def __len__(self):
                return len(self.xyxy)

        boxes = Boxes()
        boxes.xyxy = self._boxes_xyxy
        boxes.conf = np.full(len(boxes_xyxy), 0.9, dtype=np.float32)
        self.boxes = boxes
        self.keypoints = None


class _FakeDetector:
    def __init__(self, rng):
        self.rng = rng

    def process_image(self, frame):
        boxes = []
        for _ in range(6):
            y = float(self.rng.uniform(300, 600))
            frac = (y - 300) / 300
            width = 400 + 400 * frac
            cx = 400 + float(self.rng.uniform(-width / 2, width / 2))
            boxes.append([cx - 30, y - 120, cx + 30, y])
        return _FakeResult(boxes)


class AutoCalibrateStageTests(unittest.TestCase):
    def test_returns_symmetric_four_points(self) -> None:
        width, height = 800, 600
        video_path = os.path.join(tempfile.gettempdir(), "ad_autocal.mp4")
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
        )
        for _ in range(30):
            writer.write(np.full((height, width, 3), 255, dtype=np.uint8))
        writer.release()

        try:
            points = auto_calibrate_stage(
                video_path, _FakeDetector(np.random.default_rng(1)), sample_frames=40
            )
        finally:
            if os.path.exists(video_path):
                os.remove(video_path)

        self.assertEqual(len(points), 4)
        for point in points:
            self.assertEqual(len(point), 2)
            self.assertTrue(0 <= point[0] < width)
            self.assertTrue(0 <= point[1] < height)

        top_left, top_right, bottom_right, bottom_left = points
        center_x = 400
        self.assertAlmostEqual(top_left[0] + top_right[0], 2 * center_x, delta=60)
        self.assertAlmostEqual(bottom_left[0] + bottom_right[0], 2 * center_x, delta=60)
        self.assertGreater(bottom_right[0] - bottom_left[0], top_right[0] - top_left[0])


if __name__ == "__main__":
    unittest.main()
