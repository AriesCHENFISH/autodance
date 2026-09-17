"""对关键帧静止画面独立检测并用外观重识别重建跨帧身份。"""

from __future__ import annotations

import math

import cv2
import numpy as np

from vision.coordinate import StageCalibration
from vision.identity import assign_keyframe_identities
from vision.tracker import extract_detected_persons


def _scale_coordinate(value: float, source_size: int, target_size: int) -> float:
    """把 0..source_size 连续舞台坐标映射到 1..target_size。"""

    bounded = min(max(float(value), 0.0), float(source_size))
    return 1.0 + bounded / source_size * (target_size - 1)


def _read_frame(capture: cv2.VideoCapture, frame_id: int):
    """读取指定帧，失败时返回 None。"""

    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    success, frame = capture.read()
    return frame if success else None


def _person_crop(frame: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.08):
    """裁剪人物检测框，保留躯干和衣着用于外观重识别。"""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    dx = round(box_width * margin)
    dy = round(box_height * margin)
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(width, x2 + dx)
    y2 = min(height, y2 + dy)
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return frame[y1:y2, x1:x2]


def analyze_keyframe_formations(
    video_path: str,
    frame_ids: list[int],
    detector,
    reid_extractor,
    calibration: StageCalibration,
    expected_count: int,
    grid_width: int,
    grid_height: int,
    source_grid_width: int = 9,
    source_grid_height: int = 9,
) -> list[dict]:
    """对每个关键帧独立检测并跨帧匹配身份，返回逐帧 persons。

    返回列表长度与 ``frame_ids`` 一致，每个元素形如
    ``{"frame_id": int, "persons": [{"id": int, "x": int, "y": int}, ...]}``。
    人物 ``x`` / ``y`` 是从 1 开始的离散队形格位；在舞台之外或未匹配到
    身份的检测会被省略。
    """

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("无法打开视频以重建关键帧")

        features: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        stages: list[list[tuple[float, float]]] = []
        detected_counts: list[int] = []

        for frame_id in frame_ids:
            frame = _read_frame(capture, frame_id)
            if frame is None:
                features.append(np.zeros((0, 512), dtype=np.float32))
                positions.append(np.zeros((0, 2), dtype=np.float32))
                stages.append([])
                detected_counts.append(0)
                continue

            result = detector.process_image(frame)
            persons = extract_detected_persons(result)
            crops = [_person_crop(frame, person.box) for person in persons]
            frame_features = reid_extractor.extract(crops) if crops else np.zeros((0, 512), dtype=np.float32)

            frame_stages: list[tuple[float, float]] = []
            frame_positions: list[tuple[float, float]] = []
            for person in persons:
                point = calibration.pixel_to_stage(person.x, person.y)
                frame_stages.append((point.x, point.y))
                frame_positions.append((point.x, point.y))

            features.append(frame_features)
            positions.append(np.asarray(frame_positions, dtype=np.float32).reshape(-1, 2))
            stages.append(frame_stages)
            detected_counts.append(len(persons))

        if not features:
            return []

        ids_per_frame, _assignment = assign_keyframe_identities(
            features, positions, expected_count
        )

        epsilon = 1e-5
        payloads: list[dict] = []
        for index, frame_id in enumerate(frame_ids):
            persons = []
            frame_ids_for_payload = ids_per_frame[index]
            for position_index, identity_id in enumerate(frame_ids_for_payload):
                if identity_id <= 0:
                    continue
                stage_x, stage_y = stages[index][position_index]
                in_stage = (
                    -epsilon <= stage_x <= source_grid_width + epsilon
                    and -epsilon <= stage_y <= source_grid_height + epsilon
                )
                if not in_stage:
                    continue
                x = round(_scale_coordinate(stage_x, source_grid_width, grid_width))
                y = round(_scale_coordinate(stage_y, source_grid_height, grid_height))
                persons.append(
                    {
                        "id": int(identity_id),
                        "x": min(max(x, 1), grid_width),
                        "y": min(max(y, 1), grid_height),
                    }
                )
            persons.sort(key=lambda item: item["id"])
            payloads.append({"frame_id": int(frame_id), "persons": persons})
        return payloads
    finally:
        capture.release()
