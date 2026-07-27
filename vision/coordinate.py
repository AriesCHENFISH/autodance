"""视频像素坐标到舞台平面坐标的四点透视转换。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np


class CalibrationError(ValueError):
    """四点标定不完整或几何形状无效。"""


@dataclass(frozen=True)
class StagePoint:
    """以一个网格单元为单位的连续舞台坐标。"""

    x: float
    y: float


@dataclass(frozen=True)
class StageCalibration:
    """从视频四边形到矩形舞台网格的单应性标定。"""

    pixel_points: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    frame_width: int
    frame_height: int
    columns: int = 9
    rows: int = 9
    matrix: np.ndarray = field(init=False, repr=False, compare=False)
    inverse_matrix: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise CalibrationError("视频画面尺寸无效")
        if self.columns <= 0 or self.rows <= 0:
            raise CalibrationError("网格行列数必须大于 0")

        source = np.asarray(self.pixel_points, dtype=np.float32)
        if source.shape != (4, 2) or not np.isfinite(source).all():
            raise CalibrationError("透视标定必须包含四个有效坐标点")
        if (
            np.any(source[:, 0] < 0)
            or np.any(source[:, 0] >= self.frame_width)
            or np.any(source[:, 1] < 0)
            or np.any(source[:, 1] >= self.frame_height)
        ):
            raise CalibrationError("标定点必须位于视频画面内")

        contour = source.reshape((-1, 1, 2))
        area = abs(float(cv2.contourArea(contour)))
        minimum_area = self.frame_width * self.frame_height * 0.005
        if area < minimum_area:
            raise CalibrationError("四个标定点围成的舞台区域过小")
        if not cv2.isContourConvex(contour):
            raise CalibrationError(
                "标定四边形发生交叉；请按左上、右上、右下、左下重新点击"
            )

        top_y = float(np.mean(source[:2, 1]))
        bottom_y = float(np.mean(source[2:, 1]))
        left_x = float(np.mean(source[[0, 3], 0]))
        right_x = float(np.mean(source[[1, 2], 0]))
        if top_y >= bottom_y or left_x >= right_x:
            raise CalibrationError(
                "标定点顺序错误；请按左上、右上、右下、左下重新点击"
            )

        destination = np.asarray(
            [
                (0.0, 0.0),
                (float(self.columns), 0.0),
                (float(self.columns), float(self.rows)),
                (0.0, float(self.rows)),
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        inverse_matrix = cv2.getPerspectiveTransform(destination, source)
        if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-12:
            raise CalibrationError("无法从这些标定点计算稳定的透视变换")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "inverse_matrix", inverse_matrix)

    @classmethod
    def from_points(
        cls,
        points: Sequence[Sequence[float]],
        frame_width: int,
        frame_height: int,
        columns: int = 9,
        rows: int = 9,
    ) -> StageCalibration:
        """从 Gradio 状态中的点列表创建并校验标定。"""

        if len(points) != 4:
            raise CalibrationError(
                f"需要依次标定四个舞台角点，当前只有 {len(points)} 个"
            )
        normalized = tuple(
            (float(point[0]), float(point[1])) for point in points
        )
        return cls(
            pixel_points=normalized,  # type: ignore[arg-type]
            frame_width=int(frame_width),
            frame_height=int(frame_height),
            columns=int(columns),
            rows=int(rows),
        )

    def pixel_to_stage(self, x: float, y: float) -> StagePoint:
        """将一个视频像素点映射到连续舞台坐标。"""

        points = np.asarray([[[float(x), float(y)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(points, self.matrix)[0, 0]
        return StagePoint(float(transformed[0]), float(transformed[1]))

    def stage_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """将舞台坐标反投影回视频画面，用于绘制透视网格。"""

        points = np.asarray([[[float(x), float(y)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(points, self.inverse_matrix)[0, 0]
        return float(transformed[0]), float(transformed[1])

    def to_json(self) -> dict:
        """返回可复现该标定的 JSON 数据。"""

        labels = ("top_left", "top_right", "bottom_right", "bottom_left")
        return {
            "version": 1,
            "point_order": list(labels),
            "pixel_points": [
                {
                    "name": label,
                    "x": round(point[0], 3),
                    "y": round(point[1], 3),
                }
                for label, point in zip(labels, self.pixel_points)
            ],
            "frame": {
                "width": self.frame_width,
                "height": self.frame_height,
            },
            "grid": {
                "columns": self.columns,
                "rows": self.rows,
                "origin": "top_left",
                "x_direction": "left_to_right",
                "y_direction": "top_to_bottom",
                "unit": "grid_cell",
            },
            "homography_pixel_to_stage": [
                [round(float(value), 10) for value in row]
                for row in self.matrix
            ],
        }
