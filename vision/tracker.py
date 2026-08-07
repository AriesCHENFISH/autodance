"""追踪结果解析和视频标注工具。"""

from dataclasses import dataclass
import colorsys
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackedPerson:
    """单帧中的一个已追踪人物。"""

    person_id: int
    x: float
    y: float
    box: tuple[int, int, int, int]
    confidence: float | None = None

    def to_json(self) -> dict[str, int]:
        """转换为 tracks.json 要求的精简字段。"""

        return {
            "id": self.person_id,
            "x": round(self.x),
            "y": round(self.y),
        }


def _to_numpy(value: Any) -> np.ndarray | None:
    """将 PyTorch 张量或数组安全转换为 NumPy 数组。"""

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _foot_position(
    box: np.ndarray,
    keypoints_xy: np.ndarray | None,
    keypoints_conf: np.ndarray | None,
    index: int,
    min_keypoint_confidence: float = 0.25,
) -> tuple[float, float]:
    """优先用脚踝中点作为人物位置，关键点无效时使用检测框底边中心。"""

    if keypoints_xy is not None and index < len(keypoints_xy):
        points = keypoints_xy[index]
        confidences = (
            keypoints_conf[index]
            if keypoints_conf is not None and index < len(keypoints_conf)
            else None
        )
        valid_ankles: list[np.ndarray] = []
        for ankle_index in (15, 16):
            if ankle_index >= len(points):
                continue
            point = points[ankle_index]
            has_position = bool(point[0] > 0 or point[1] > 0)
            has_confidence = (
                confidences is None
                or ankle_index >= len(confidences)
                or confidences[ankle_index] >= min_keypoint_confidence
            )
            if has_position and has_confidence:
                valid_ankles.append(point)
        if valid_ankles:
            foot = np.mean(valid_ankles, axis=0)
            return float(foot[0]), float(foot[1])

    x1, _y1, x2, y2 = box
    return float((x1 + x2) / 2), float(y2)


def extract_tracked_persons(result: Any) -> list[TrackedPerson]:
    """从 Ultralytics 单帧结果中提取拥有稳定追踪 ID 的人物。"""

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return []

    boxes_xyxy = _to_numpy(boxes.xyxy)
    track_ids = _to_numpy(boxes.id)
    confidences = _to_numpy(getattr(boxes, "conf", None))
    keypoints = getattr(result, "keypoints", None)
    keypoints_xy = _to_numpy(keypoints.xy) if keypoints is not None else None
    keypoints_conf = _to_numpy(keypoints.conf) if keypoints is not None else None

    persons: list[TrackedPerson] = []
    for index, (box, track_id) in enumerate(zip(boxes_xyxy, track_ids)):
        x, y = _foot_position(box, keypoints_xy, keypoints_conf, index)
        persons.append(
            TrackedPerson(
                person_id=int(track_id),
                x=x,
                y=y,
                box=tuple(round(float(value)) for value in box),
                confidence=(
                    float(confidences[index])
                    if confidences is not None and index < len(confidences)
                    else None
                ),
            )
        )
    return sorted(persons, key=lambda person: person.person_id)


def color_for_id(person_id: int) -> tuple[int, int, int]:
    """根据人物 ID 生成稳定且易区分的 BGR 颜色。"""

    hue = (person_id * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
    return round(blue * 255), round(green * 255), round(red * 255)


def draw_tracked_persons(
    frame: np.ndarray, persons: list[TrackedPerson]
) -> np.ndarray:
    """在视频帧上绘制人物框、落脚点和追踪编号。"""

    annotated = frame.copy()
    for person in persons:
        color = color_for_id(person.person_id)
        x1, y1, x2, y2 = person.box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        point = (round(person.x), round(person.y))
        cv2.circle(annotated, point, 5, color, -1, lineType=cv2.LINE_AA)

        label = f"ID {person.person_id}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        label_top = max(0, y1 - text_height - 10)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (x1 + text_width + 8, label_top + text_height + 10),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_top + text_height + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated
