"""舞台透视网格定位、JSON 输出与视频叠加。"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math

import cv2
import numpy as np

from vision.coordinate import StageCalibration, StagePoint
from vision.tracker import TrackedPerson, color_for_id


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


def _grid_fields(
    stage_x: float,
    stage_y: float,
    columns: int,
    rows: int,
) -> dict:
    """从连续舞台坐标生成离散格位字段。"""

    bounded_x = min(max(stage_x, 0.0), columns - 1e-5)
    bounded_y = min(max(stage_y, 0.0), rows - 1e-5)
    return {
        "stage_x": round(bounded_x, 3),
        "stage_y": round(bounded_y, 3),
        "grid_col": math.floor(bounded_x) + 1,
        "grid_row": math.floor(bounded_y) + 1,
        "in_stage": True,
    }


def stabilize_grid_tracks(
    frames: list[dict],
    columns: int,
    rows: int,
    max_gap_frames: int = 12,
    outlier_distance: float = 1.25,
    boundary_margin: float = 0.5,
) -> list[dict]:
    """剔除短时脚点漂移，并用前后有效位置插值补齐短缺测。"""

    stabilized = deepcopy(frames)
    frame_maps = [
        {person["id"]: person for person in frame["persons"]}
        for frame in stabilized
    ]
    identity_ids = sorted(
        {
            person_id
            for frame_map in frame_maps
            for person_id in frame_map
        }
    )

    for identity_id in identity_ids:
        observed = {
            frame_id: frame_map[identity_id]
            for frame_id, frame_map in enumerate(frame_maps)
            if identity_id in frame_map
        }
        valid: dict[int, bool] = {}
        for frame_id, person in observed.items():
            stage_x = float(person["stage_x"])
            stage_y = float(person["stage_y"])
            valid[frame_id] = (
                -boundary_margin <= stage_x <= columns + boundary_margin
                and -boundary_margin <= stage_y <= rows + boundary_margin
            )

        # 夹在正常位置之间的单帧大跳通常来自遮挡时的错误脚踝关键点。
        observed_ids = sorted(observed)
        for index, frame_id in enumerate(observed_ids):
            if not valid[frame_id] or index == 0 or index == len(observed_ids) - 1:
                continue
            previous_id = observed_ids[index - 1]
            next_id = observed_ids[index + 1]
            if (
                previous_id != frame_id - 1
                or next_id != frame_id + 1
                or not valid[previous_id]
                or not valid[next_id]
            ):
                continue
            person = observed[frame_id]
            midpoint = (
                (
                    float(observed[previous_id]["stage_x"])
                    + float(observed[next_id]["stage_x"])
                )
                / 2,
                (
                    float(observed[previous_id]["stage_y"])
                    + float(observed[next_id]["stage_y"])
                )
                / 2,
            )
            if math.dist(
                (float(person["stage_x"]), float(person["stage_y"])),
                midpoint,
            ) > outlier_distance:
                valid[frame_id] = False

        valid_ids = sorted(frame_id for frame_id, is_valid in valid.items() if is_valid)
        for left_id, right_id in zip(valid_ids, valid_ids[1:]):
            gap = right_id - left_id - 1
            if gap <= 0 or gap > max_gap_frames:
                continue
            left = observed[left_id]
            right = observed[right_id]
            for frame_id in range(left_id + 1, right_id):
                existing = observed.get(frame_id)
                if existing is not None and valid.get(frame_id, False):
                    continue
                ratio = (frame_id - left_id) / (right_id - left_id)
                stage_x = float(left["stage_x"]) + (
                    float(right["stage_x"]) - float(left["stage_x"])
                ) * ratio
                stage_y = float(left["stage_y"]) + (
                    float(right["stage_y"]) - float(left["stage_y"])
                ) * ratio
                pixel_x = round(
                    float(left["x"])
                    + (float(right["x"]) - float(left["x"])) * ratio
                )
                pixel_y = round(
                    float(left["y"])
                    + (float(right["y"]) - float(left["y"])) * ratio
                )
                if existing is None:
                    existing = {
                        "id": identity_id,
                        "x": pixel_x,
                        "y": pixel_y,
                    }
                    stabilized[frame_id]["persons"].append(existing)
                    frame_maps[frame_id][identity_id] = existing
                    observed[frame_id] = existing
                else:
                    existing["x"] = pixel_x
                    existing["y"] = pixel_y
                existing.update(_grid_fields(stage_x, stage_y, columns, rows))
                existing["interpolated"] = True

        # 对合法但略超边界的点进行半格范围内的钳制。
        for frame_id, person in observed.items():
            if valid.get(frame_id, False) and "interpolated" not in person:
                person.update(
                    _grid_fields(
                        float(person["stage_x"]),
                        float(person["stage_y"]),
                        columns,
                        rows,
                    )
                )
                person["interpolated"] = False

    for frame in stabilized:
        frame["persons"].sort(key=lambda person: person["id"])
    return stabilized


def draw_stabilized_grid_positions(
    frame: np.ndarray,
    frame_payload: dict,
    calibration: StageCalibration,
) -> np.ndarray:
    """根据稳定后的 JSON 舞台坐标绘制点位、格位和插值标记。"""

    annotated = frame.copy()
    for person in frame_payload["persons"]:
        if not person.get("in_stage", False):
            continue
        pixel = calibration.stage_to_pixel(
            float(person["stage_x"]), float(person["stage_y"])
        )
        point = (round(pixel[0]), round(pixel[1]))
        color = color_for_id(int(person["id"]))
        cv2.circle(annotated, point, 5, color, -1, cv2.LINE_AA)
        suffix = "*" if person.get("interpolated", False) else ""
        label = f"C{person['grid_col']} R{person['grid_row']}{suffix}"
        cv2.putText(
            annotated,
            label,
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            label,
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated
