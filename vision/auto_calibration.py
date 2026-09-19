"""从舞者脚点轨迹自动估计舞台四角（对称梯形拟合）。"""

from __future__ import annotations

import cv2
import numpy as np

from .tracker import extract_detected_persons


def _collect_foot_points(
    video_path: str,
    detector,
    sample_frames: int,
) -> tuple[np.ndarray, int, int]:
    """均匀采样若干帧，检测人物并收集脚点像素坐标。"""

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("无法打开视频以自动标定")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or width <= 0 or height <= 0:
            raise ValueError("视频没有可读取的画面")

        indices = np.linspace(0, total - 1, max(1, int(sample_frames)))
        points: list[tuple[float, float]] = []
        for frame_id in np.unique(indices.astype(int)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            success, frame = capture.read()
            if not success:
                continue
            try:
                result = detector.process_image(frame)
                persons = extract_detected_persons(result)
            except Exception:
                continue
            for person in persons:
                points.append((person.x, person.y))
        return np.asarray(points, dtype=np.float64).reshape(-1, 2), width, height
    finally:
        capture.release()


def _fit_symmetric_trapezoid(
    points: np.ndarray,
    bands: int = 6,
) -> tuple[float, float, float, float]:
    """拟合左右对称、宽度随深度线性变化的梯形。

    返回 ``(center_x, top_y, top_width, bottom_width)``。
    """

    xs = points[:, 0]
    ys = points[:, 1]
    center_x = float(np.median(xs))
    top_y = float(np.percentile(ys, 2))
    bottom_y = float(np.percentile(ys, 98))

    band_edges = np.linspace(top_y, bottom_y, bands + 1)
    widths: list[float] = []
    centers: list[float] = []
    for index in range(bands):
        low, high = band_edges[index], band_edges[index + 1]
        mask = (ys >= low) & (ys <= high)
        if int(mask.sum()) < 3:
            continue
        band_xs = xs[mask]
        left = float(np.percentile(band_xs, 2))
        right = float(np.percentile(band_xs, 98))
        widths.append(right - left)
        centers.append((low + high) / 2)

    if len(widths) >= 2:
        slope, intercept = np.polyfit(centers, widths, 1)
        top_width = slope * top_y + intercept
        bottom_width = slope * bottom_y + intercept
    else:
        top_width = bottom_width = float(np.mean(widths)) if widths else 0.0

    if top_width <= 0 or bottom_width <= 0:
        average = float(np.percentile(xs, 95) - np.percentile(xs, 5))
        top_width = bottom_width = average

    ratio = bottom_width / top_width if top_width > 0 else 1.0
    if not 0.5 <= ratio <= 2.0:
        average = (top_width + bottom_width) / 2
        top_width = bottom_width = average

    return center_x, top_y, top_width, bottom_width


def auto_calibrate_stage(
    video_path: str,
    detector,
    sample_frames: int = 40,
    margin_ratio: float = 0.06,
) -> list[list[float]]:
    """自动估计舞台四角，返回 ``[[左上],[右上],[右下],[左下]]`` 像素点。"""

    points, width, height = _collect_foot_points(video_path, detector, sample_frames)
    if points.shape[0] < 8:
        raise ValueError(
            f"检测到的脚点太少（{points.shape[0]} 个），无法自动标定，请手动标定"
        )

    center_x, top_y, top_width, bottom_width = _fit_symmetric_trapezoid(points)
    bottom_y = float(np.percentile(points[:, 1], 98))

    top_width *= 1.0 + margin_ratio
    bottom_width *= 1.0 + margin_ratio

    def clamp_x(value: float) -> float:
        return float(min(max(value, 0.0), width - 1))

    def clamp_y(value: float) -> float:
        return float(min(max(value, 0.0), height - 1))

    top_left = [clamp_x(center_x - top_width / 2), clamp_y(top_y)]
    top_right = [clamp_x(center_x + top_width / 2), clamp_y(top_y)]
    bottom_right = [clamp_x(center_x + bottom_width / 2), clamp_y(bottom_y)]
    bottom_left = [clamp_x(center_x - bottom_width / 2), clamp_y(bottom_y)]

    if (top_right[0] - top_left[0]) < 8 or (bottom_right[0] - bottom_left[0]) < 8:
        raise ValueError("自动标定得到的舞台过窄，请手动标定")
    return [top_left, top_right, bottom_right, bottom_left]
