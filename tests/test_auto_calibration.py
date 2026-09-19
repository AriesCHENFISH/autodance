"""自动舞台标定测试。"""

import os
import tempfile
import unittest

import cv2
import numpy as np

from vision.auto_calibration import auto_calibrate_stage


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
    def __init__(self, foot_y_values):
        self.foot_y_values = list(foot_y_values)

    def process_image(self, frame):
        boxes = []
        for index, y in enumerate(self.foot_y_values):
            x = 300 + index * 80
            boxes.append([x - 30, y - 120, x + 30, y])
        return _FakeResult(boxes)


def _make_video(width=800, height=600):
    path = os.path.join(tempfile.gettempdir(), "ad_autocal.mp4")
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height)
    )
    for _ in range(5):
        writer.write(np.full((height, width, 3), 255, dtype=np.uint8))
    writer.release()
    return path


class AutoCalibrateStageTests(unittest.TestCase):
    def test_returns_max_symmetric_trapezoid(self) -> None:
        width, height = 800, 600
        video_path = _make_video(width, height)
        try:
            points = auto_calibrate_stage(
                video_path, _FakeDetector([300, 360, 420, 480, 540, 600])
            )
        finally:
            os.remove(video_path)

        self.assertEqual(len(points), 4)
        top_left, top_right, bottom_right, bottom_left = points
        for point in points:
            self.assertEqual(len(point), 2)
            self.assertTrue(0 <= point[0] < width)
            self.assertTrue(0 <= point[1] < height)

        # 左右对称：对称轴为画面水平中心
        self.assertAlmostEqual(top_left[0] + top_right[0], width, delta=1)
        self.assertAlmostEqual(bottom_left[0] + bottom_right[0], width, delta=1)
        # 上窄下宽（透视）
        self.assertGreater(
            bottom_right[0] - bottom_left[0], top_right[0] - top_left[0]
        )
        # 近端延伸到画面底部
        self.assertAlmostEqual(bottom_y(bottom_left), height * 0.97, delta=1)
        # 远端由首帧脚点估计，位于脚点最远端之上且小于底部
        self.assertLess(top_left[1], bottom_left[1])

    def test_too_few_people_raises(self) -> None:
        video_path = _make_video()
        try:
            with self.assertRaises(ValueError):
                auto_calibrate_stage(video_path, _FakeDetector([400]))
        finally:
            os.remove(video_path)


def bottom_y(point):
    return point[1]


if __name__ == "__main__":
    unittest.main()
