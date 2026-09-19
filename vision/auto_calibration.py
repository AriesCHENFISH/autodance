"""从首帧画面自动估计最大地面四角（对称梯形）。"""

from __future__ import annotations

import cv2
import numpy as np

from .tracker import extract_detected_persons


def auto_calibrate_stage(
    video_path: str,
    detector,
    side_margin: float = 0.02,
    bottom_margin: float = 0.03,
    top_aspect: float = 0.85,
    minimum_height_ratio: float = 0.15,
    min_people: int = 2,
    max_search_frames: int = 300,
    search_step: int = 10,
) -> list[list[float]]:
    """用首个有人画面的舞者脚点估计地面远端，并把标定框扩展到最大可见地面。

    固定机位正视角下，地面是画面下部一个左右对称的梯形：对称轴取画面
    水平中心，远端（后墙）由脚点估计，近端延伸到画面底部，横向扩展到
    画面允许的最大对称宽度（上窄下宽，比例 ``top_aspect``）。首帧若无人
    （片头），则向后跳跃搜索第一个有足够人数的画面。

    返回 ``[[左上],[右上],[右下],[左下]]`` 像素点。
    """

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("无法打开视频以自动标定")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise ValueError("无法读取视频画面尺寸")

        frame = None
        persons = []
        frame_id = 0
        while frame_id <= max_search_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            success, frame = capture.read()
            if not success:
                break
            persons = extract_detected_persons(detector.process_image(frame))
            if len(persons) >= min_people:
                break
            frame_id += search_step

        if frame is None or len(persons) < min_people:
            raise ValueError(
                "未在片头找到足够人物，无法自动标定，请手动标定"
            )

        ys = np.asarray([person.y for person in persons], dtype=np.float64)

        center_x = width / 2.0
        bottom_y = height * (1.0 - bottom_margin)
        top_y = float(np.percentile(ys, 2))
        top_y = min(top_y, bottom_y - height * minimum_height_ratio)

        bottom_half = width * (0.5 - side_margin)
        top_half = bottom_half * top_aspect

        top_left = [center_x - top_half, top_y]
        top_right = [center_x + top_half, top_y]
        bottom_right = [center_x + bottom_half, bottom_y]
        bottom_left = [center_x - bottom_half, bottom_y]

        return [top_left, top_right, bottom_right, bottom_left]
    finally:
        capture.release()
