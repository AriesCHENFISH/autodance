"""基于稳定人物轨迹检测关键队形。"""

from __future__ import annotations

import math
from statistics import median
from typing import Iterable


SOURCE_GRID_WIDTH = 9
SOURCE_GRID_HEIGHT = 9


def _percentile(values: list[float], fraction: float) -> float:
    """使用线性插值计算百分位数。"""

    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] * (1 - ratio) + ordered[upper] * ratio


def _scale_coordinate(value: float, source_size: int, target_size: int) -> float:
    """把 0..source_size 连续舞台坐标映射到 1..target_size。"""

    bounded = min(max(float(value), 0.0), float(source_size))
    return 1.0 + bounded / source_size * (target_size - 1)


def _frame_positions(
    frame: dict,
    grid_width: int,
    grid_height: int,
) -> dict[int, tuple[float, float, bool]]:
    """提取一帧内有效人物的目标网格连续坐标。"""

    positions = {}
    for person in frame.get("persons", []):
        if not person.get("in_stage", False):
            continue
        try:
            identity_id = int(person["id"])
            stage_x = float(person["stage_x"])
            stage_y = float(person["stage_y"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(stage_x) or not math.isfinite(stage_y):
            continue
        positions[identity_id] = (
            _scale_coordinate(stage_x, SOURCE_GRID_WIDTH, grid_width),
            _scale_coordinate(stage_y, SOURCE_GRID_HEIGHT, grid_height),
            bool(person.get("interpolated", False)),
        )
    return positions


def _median_positions(
    frame_maps: Iterable[dict[int, tuple[float, float, bool]]],
    identity_ids: tuple[int, ...],
) -> dict[int, tuple[float, float]]:
    """计算若干帧中每个身份的连续坐标中位数。"""

    samples = {identity_id: [] for identity_id in identity_ids}
    for frame_map in frame_maps:
        for identity_id in identity_ids:
            position = frame_map.get(identity_id)
            if position is not None:
                samples[identity_id].append((position[0], position[1]))
    return {
        identity_id: (
            median(point[0] for point in points),
            median(point[1] for point in points),
        )
        for identity_id, points in samples.items()
        if points
    }


def _stable_candidate(
    window: list[dict[int, tuple[float, float, bool]]],
    identity_ids: tuple[int, ...],
    minimum_presence_ratio: float,
    maximum_interpolation_ratio: float,
    stability_radius: float,
) -> dict[int, tuple[float, float]] | None:
    """检查完整窗口是否稳定，并返回各身份的中位位置。"""

    minimum_records = math.ceil(len(window) * minimum_presence_ratio)
    candidate = _median_positions(window, identity_ids)
    if len(candidate) != len(identity_ids):
        return None
    for identity_id in identity_ids:
        records = [
            frame_map[identity_id]
            for frame_map in window
            if identity_id in frame_map
        ]
        if len(records) < minimum_records:
            return None
        if sum(record[2] for record in records) / len(records) > maximum_interpolation_ratio:
            return None
        center = candidate[identity_id]
        distances = [math.dist((record[0], record[1]), center) for record in records]
        if _percentile(distances, 0.80) > stability_radius:
            return None
    return candidate


def _moved_count(
    positions: dict[int, tuple[float, float]],
    baseline: dict[int, tuple[float, float]],
    movement_threshold: float,
) -> int:
    """统计相对基线达到位移阈值的不同身份数。"""

    return sum(
        math.dist(position, baseline[identity_id]) >= movement_threshold
        for identity_id, position in positions.items()
        if identity_id in baseline
    )


def _formation_payload(
    formation_id: int,
    frames: list[dict],
    window_start: int,
    window_end: int,
    positions: dict[int, tuple[float, float]],
    grid_width: int,
    grid_height: int,
    fps: float,
) -> dict:
    """把连续候选位置转换为可下载的离散队形。"""

    middle_index = (window_start + window_end) // 2
    middle_frame = frames[middle_index]
    end_frame = frames[window_end]

    def timestamp(frame: dict, fallback_index: int) -> float:
        try:
            return float(frame["timestamp"])
        except (KeyError, TypeError, ValueError):
            return fallback_index / fps

    persons = []
    for identity_id in sorted(positions):
        x, y = positions[identity_id]
        persons.append(
            {
                "id": identity_id,
                "x": min(max(round(x), 1), grid_width),
                "y": min(max(round(y), 1), grid_height),
            }
        )
    return {
        "formation_id": formation_id,
        "time": round(timestamp(middle_frame, middle_index), 3),
        "confirmed_at": round(timestamp(end_frame, window_end), 3),
        "frame_id": int(middle_frame.get("frame_id", middle_index)),
        "grid_width": grid_width,
        "grid_height": grid_height,
        "persons": persons,
    }


def analyze_formations(
    frames: list[dict],
    fps: float,
    grid_width: int = 20,
    grid_height: int = 20,
    stable_seconds: float = 0.75,
    minimum_presence_ratio: float = 0.80,
    maximum_interpolation_ratio: float = 0.30,
    stability_radius: float = 1.25,
    movement_threshold: float = 2.0,
    recent_frames: int = 5,
    minimum_moved_people: int = 2,
    dedup_seconds: float = 0.50,
) -> list[dict]:
    """从逐帧 9×9 舞台轨迹中提取关键队形。"""

    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("fps 必须是正数")
    if not 8 <= int(grid_width) <= 40 or not 8 <= int(grid_height) <= 40:
        raise ValueError("网格宽度和高度必须在 8 到 40 之间")
    if not frames:
        return []

    grid_width = int(grid_width)
    grid_height = int(grid_height)
    stable_window = max(1, math.ceil(fps * stable_seconds))
    dedup_frames = max(0, math.ceil(fps * dedup_seconds))
    identity_ids = tuple(sorted({
        int(person["id"])
        for frame in frames
        for person in frame.get("persons", [])
        if "id" in person
    }))
    if not identity_ids or len(frames) < stable_window:
        return []

    frame_maps = [_frame_positions(frame, grid_width, grid_height) for frame in frames]
    formations = []
    baseline = None
    transitioning = False
    transition_started = -1
    last_saved_end = -dedup_frames - 1

    for window_end in range(stable_window - 1, len(frames)):
        window_start = window_end - stable_window + 1
        candidate = _stable_candidate(
            frame_maps[window_start : window_end + 1],
            identity_ids,
            minimum_presence_ratio,
            maximum_interpolation_ratio,
            stability_radius,
        )
        if baseline is None:
            if candidate is None:
                continue
            formations.append(_formation_payload(
                1, frames, window_start, window_end, candidate,
                grid_width, grid_height, fps,
            ))
            baseline = candidate
            last_saved_end = window_end
            continue

        if not transitioning:
            recent_start = max(0, window_end - recent_frames + 1)
            recent = _median_positions(frame_maps[recent_start : window_end + 1], identity_ids)
            if (
                len(recent) == len(identity_ids)
                and _moved_count(recent, baseline, movement_threshold) >= minimum_moved_people
            ):
                transitioning = True
                transition_started = window_end
            continue

        # 必须积累转换开始后的完整窗口，避免把旧基线帧混入候选。
        if window_start <= transition_started or candidate is None:
            continue
        if _moved_count(candidate, baseline, movement_threshold) < minimum_moved_people:
            transitioning = False
            continue
        if window_end - last_saved_end < dedup_frames:
            continue
        formations.append(_formation_payload(
            len(formations) + 1, frames, window_start, window_end, candidate,
            grid_width, grid_height, fps,
        ))
        baseline = candidate
        last_saved_end = window_end
        transitioning = False

    return formations
