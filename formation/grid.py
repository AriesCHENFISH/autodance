"""舞台透视网格定位、JSON 输出与视频叠加。"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from vision.coordinate import StageCalibration, StagePoint
from vision.tracker import TrackedPerson


@dataclass(frozen=True)
class GridPosition:
    """人物在连续舞台平面及离散网格中的位置。"""

    stage: StagePoint
    column: int | None
    row: int | None
    in_stage: bool


def locate_on_grid(
    x: float,
    y: float,
    calibration: StageCalibration,
) -> GridPosition:
    """将视频像素点定位到 1 起始的舞台网格行列。"""

    stage = calibration.pixel_to_stage(x, y)
    epsilon = 1e-5
    in_stage = (
        -epsilon <= stage.x <= calibration.columns + epsilon
        and -epsilon <= stage.y <= calibration.rows + epsilon
    )
    if not in_stage:
        return GridPosition(stage=stage, column=None, row=None, in_stage=False)

    bounded_x = min(max(stage.x, 0.0), calibration.columns - epsilon)
    bounded_y = min(max(stage.y, 0.0), calibration.rows - epsilon)
    return GridPosition(
        stage=stage,
        column=math.floor(bounded_x) + 1,
        row=math.floor(bounded_y) + 1,
        in_stage=True,
    )


def person_to_grid_json(
    person: TrackedPerson,
    calibration: StageCalibration,
) -> dict:
    """生成同时包含像素、舞台和网格坐标的人物记录。"""

    position = locate_on_grid(person.x, person.y, calibration)
    return {
        **person.to_json(),
        "stage_x": round(position.stage.x, 3),
        "stage_y": round(position.stage.y, 3),
        "grid_col": position.column,
        "grid_row": position.row,
        "in_stage": position.in_stage,
    }


def _pixel_point(
    calibration: StageCalibration,
    x: float,
    y: float,
) -> tuple[int, int]:
    pixel_x, pixel_y = calibration.stage_to_pixel(x, y)
    return round(pixel_x), round(pixel_y)


def draw_perspective_grid(
    frame: np.ndarray,
    calibration: StageCalibration,
) -> np.ndarray:
    """在原视频透视平面上叠加半透明 9×9 网格。"""

    overlay = frame.copy()
    grid_color = (90, 190, 255)
    border_color = (40, 230, 255)

    for column in range(calibration.columns + 1):
        start = _pixel_point(calibration, float(column), 0.0)
        end = _pixel_point(calibration, float(column), float(calibration.rows))
        is_border = column in (0, calibration.columns)
        cv2.line(
            overlay,
            start,
            end,
            border_color if is_border else grid_color,
            3 if is_border else 1,
            cv2.LINE_AA,
        )

    for row in range(calibration.rows + 1):
        start = _pixel_point(calibration, 0.0, float(row))
        end = _pixel_point(calibration, float(calibration.columns), float(row))
        is_border = row in (0, calibration.rows)
        cv2.line(
            overlay,
            start,
            end,
            border_color if is_border else grid_color,
            3 if is_border else 1,
            cv2.LINE_AA,
        )

    return cv2.addWeighted(overlay, 0.52, frame, 0.48, 0.0)


def draw_grid_positions(
    frame: np.ndarray,
    persons: list[TrackedPerson],
    calibration: StageCalibration,
) -> np.ndarray:
    """在每个人物落脚点旁标注离散网格行列。"""

    annotated = frame.copy()
    for person in persons:
        position = locate_on_grid(person.x, person.y, calibration)
        if position.in_stage:
            label = f"C{position.column} R{position.row}"
            color = (255, 255, 255)
        else:
            label = "OUT"
            color = (0, 80, 255)
        origin = (round(person.x) + 8, round(person.y) - 8)
        cv2.putText(
            annotated,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated
