"""将在线跟踪产生的碎片轨迹归并为固定数量的最终人物身份。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

import cv2
import numpy as np

from .tracker import TrackedPerson


@dataclass
class _Tracklet:
    source_id: int
    observations: list[tuple[int, TrackedPerson]] = field(default_factory=list)
    descriptors: list[np.ndarray] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.observations[0][0]

    @property
    def end_frame(self) -> int:
        return self.observations[-1][0]

    @property
    def length(self) -> int:
        return len(self.observations)

    @property
    def frame_ids(self) -> set[int]:
        return {frame_id for frame_id, _person in self.observations}

    @property
    def start_point(self) -> tuple[float, float]:
        person = self.observations[0][1]
        return person.x, person.y

    @property
    def end_point(self) -> tuple[float, float]:
        person = self.observations[-1][1]
        return person.x, person.y

    @property
    def descriptor(self) -> np.ndarray | None:
        if not self.descriptors:
            return None
        descriptor = np.mean(self.descriptors, axis=0)
        norm = float(np.linalg.norm(descriptor))
        return descriptor / norm if norm > 1e-12 else None


@dataclass
class _IdentityGroup:
    final_id: int
    tracklets: list[_Tracklet]

    @property
    def frames(self) -> set[int]:
        return set().union(*(tracklet.frame_ids for tracklet in self.tracklets))

    @property
    def descriptor(self) -> np.ndarray | None:
        weighted: list[np.ndarray] = []
        weights: list[int] = []
        for tracklet in self.tracklets:
            descriptor = tracklet.descriptor
            if descriptor is not None:
                weighted.append(descriptor)
                weights.append(tracklet.length)
        if not weighted:
            return None
        value = np.average(np.stack(weighted), axis=0, weights=weights)
        norm = float(np.linalg.norm(value))
        return value / norm if norm > 1e-12 else None


@dataclass(frozen=True)
class IdentityConsolidation:
    """离线身份归并结果及诊断信息。"""

    mapping: dict[int, int]
    anchor_frame: int | None
    dropped_ids: tuple[int, ...]
    source_id_count: int
    final_id_count: int


def extract_appearance_descriptor(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray | None:
    """提取对姿态变化较稳健的衣着 HSV 直方图。"""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 - x1 < 8 or y2 - y1 < 16:
        return None

    box_width = x2 - x1
    box_height = y2 - y1
    # 去掉容易混入邻人的框边缘、头顶背景和脚下地板，重点描述躯干衣着。
    crop_x1 = x1 + round(box_width * 0.18)
    crop_x2 = x2 - round(box_width * 0.18)
    crop_y1 = y1 + round(box_height * 0.12)
    crop_y2 = y1 + round(box_height * 0.82)
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [12, 4, 4],
        [0, 180, 0, 256, 0, 256],
    ).reshape(-1)
    norm = float(np.linalg.norm(histogram))
    if norm <= 1e-12:
        return None
    return (histogram / norm).astype(np.float32)


def _build_tracklets(
    tracked_frames: list[list[TrackedPerson]],
    descriptors: dict[int, list[np.ndarray]],
) -> dict[int, _Tracklet]:
    tracklets: dict[int, _Tracklet] = {}
    for frame_id, persons in enumerate(tracked_frames):
        for person in persons:
            tracklet = tracklets.setdefault(person.person_id, _Tracklet(person.person_id))
            tracklet.observations.append((frame_id, person))
    for source_id, values in descriptors.items():
        if source_id in tracklets:
            tracklets[source_id].descriptors.extend(values)
    return tracklets


def _select_anchor_frame(
    tracked_frames: list[list[TrackedPerson]],
    tracklets: dict[int, _Tracklet],
    expected_count: int,
) -> tuple[int | None, list[int]]:
    """选择恰好人数齐全且轨迹总体最稳定的一帧作为身份锚点。"""

    candidates: list[tuple[float, int, list[int]]] = []
    for frame_id, persons in enumerate(tracked_frames):
        source_ids = list(dict.fromkeys(person.person_id for person in persons))
        if len(source_ids) != expected_count:
            continue
        stability = sum(min(tracklets[source_id].length, 180) for source_id in source_ids)
        # 轻微偏好靠近视频中点，避免第一帧尚未确认或末帧刚新建的轨迹。
        midpoint = (len(tracked_frames) - 1) / 2
        centrality = 1.0 - abs(frame_id - midpoint) / max(1.0, midpoint)
        candidates.append((stability + centrality, frame_id, source_ids))

    if candidates:
        _score, frame_id, source_ids = max(candidates, key=lambda item: item[0])
        return frame_id, source_ids

    if not tracklets:
        return None, []
    longest = sorted(tracklets.values(), key=lambda item: item.length, reverse=True)
    return None, [item.source_id for item in longest[:expected_count]]


def _appearance_distance(
    left: np.ndarray | None,
    right: np.ndarray | None,
) -> float:
    if left is None or right is None:
        return 0.55
    return float(np.clip(1.0 - np.dot(left, right), 0.0, 1.0))


def _motion_distance(tracklet: _Tracklet, group: _IdentityGroup) -> float:
    """比较碎片端点与组内时间上最近碎片端点的运动连续性。"""

    best = math.inf
    for existing in group.tracklets:
        if tracklet.end_frame < existing.start_frame:
            gap = existing.start_frame - tracklet.end_frame
            first, second = tracklet.end_point, existing.start_point
        elif existing.end_frame < tracklet.start_frame:
            gap = tracklet.start_frame - existing.end_frame
            first, second = existing.end_point, tracklet.start_point
        else:
            continue
        distance = math.dist(first, second)
        # 允许舞者随时间快速移动；短时间内的空间连续性权重更强。
        normalized = distance / (100.0 + 12.0 * gap)
        best = min(best, normalized)
    return min(best, 1.5) if math.isfinite(best) else 0.75


def _assignment_cost(tracklet: _Tracklet, group: _IdentityGroup) -> float:
    overlap = len(tracklet.frame_ids & group.frames)
    if overlap > 2:
        return math.inf
    appearance = _appearance_distance(tracklet.descriptor, group.descriptor)
    motion = _motion_distance(tracklet, group)
    overlap_penalty = overlap * 0.20
    return appearance * 0.72 + motion * 0.28 + overlap_penalty


def consolidate_track_ids(
    tracked_frames: list[list[TrackedPerson]],
    descriptors: dict[int, list[np.ndarray]],
    expected_count: int,
) -> IdentityConsolidation:
    """将碎片 ID 映射到 ``1..expected_count``，并丢弃无法容纳的重复轨迹。"""

    if expected_count < 1:
        raise ValueError("expected_count 必须大于 0")

    tracklets = _build_tracklets(tracked_frames, descriptors)
    anchor_frame, anchor_ids = _select_anchor_frame(
        tracked_frames, tracklets, expected_count
    )
    if not anchor_ids:
        return IdentityConsolidation({}, anchor_frame, (), 0, 0)

    if anchor_frame is not None:
        anchor_people = {
            person.person_id: person
            for person in tracked_frames[anchor_frame]
            if person.person_id in anchor_ids
        }
        anchor_ids.sort(key=lambda source_id: anchor_people[source_id].x)

    groups = [
        _IdentityGroup(final_id=index + 1, tracklets=[tracklets[source_id]])
        for index, source_id in enumerate(anchor_ids)
    ]
    mapping = {
        tracklet.source_id: group.final_id
        for group in groups
        for tracklet in group.tracklets
    }

    remaining = [
        tracklet
        for source_id, tracklet in tracklets.items()
        if source_id not in mapping
    ]
    # 先归并长轨迹，使组的外观模板比短暂误检更可靠。
    remaining.sort(key=lambda item: item.length, reverse=True)
    dropped: list[int] = []
    for tracklet in remaining:
        costs = [(_assignment_cost(tracklet, group), group) for group in groups]
        cost, group = min(costs, key=lambda item: item[0])
        if not math.isfinite(cost):
            dropped.append(tracklet.source_id)
            continue
        group.tracklets.append(tracklet)
        mapping[tracklet.source_id] = group.final_id

    return IdentityConsolidation(
        mapping=mapping,
        anchor_frame=anchor_frame,
        dropped_ids=tuple(sorted(dropped)),
        source_id_count=len(tracklets),
        final_id_count=len(set(mapping.values())),
    )


def relabel_tracked_frames(
    tracked_frames: list[list[TrackedPerson]],
    consolidation: IdentityConsolidation,
) -> list[list[TrackedPerson]]:
    """应用身份映射，并确保单帧内同一最终 ID 最多出现一次。"""

    source_lengths: dict[int, int] = {}
    for persons in tracked_frames:
        for person in persons:
            source_lengths[person.person_id] = source_lengths.get(person.person_id, 0) + 1

    relabeled: list[list[TrackedPerson]] = []
    for persons in tracked_frames:
        selected: dict[int, TrackedPerson] = {}
        selected_sources: dict[int, int] = {}
        for person in persons:
            final_id = consolidation.mapping.get(person.person_id)
            if final_id is None:
                continue
            existing_source = selected_sources.get(final_id)
            if (
                existing_source is None
                or source_lengths[person.person_id] > source_lengths[existing_source]
            ):
                selected[final_id] = replace(person, person_id=final_id)
                selected_sources[final_id] = person.person_id
        relabeled.append(sorted(selected.values(), key=lambda person: person.person_id))
    return relabeled
