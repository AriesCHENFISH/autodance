"""对原始在线轨迹执行固定人数逐帧身份分配并生成诊断预览。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision import (  # noqa: E402
    TrackedPerson,
    assign_fixed_identities,
    draw_tracked_persons,
)
from vision.identity import extract_appearance_descriptor  # noqa: E402


def _load_raw_frames(raw_path: Path) -> list[list[TrackedPerson]]:
    """从实验 raw_tracks.json 重建在线追踪对象。"""

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    return [
        [
            TrackedPerson(
                person_id=int(person["online_id"]),
                x=float(person["x"]),
                y=float(person["y"]),
                box=tuple(int(value) for value in person["box"]),
                confidence=(
                    float(person["confidence"])
                    if person["confidence"] is not None
                    else None
                ),
            )
            for person in frame["persons"]
        ]
        for frame in payload
    ]


def evaluate_identity(
    video_path: Path,
    raw_path: Path,
    output_directory: Path,
    expected_count: int,
) -> dict:
    """提取逐帧衣着特征，执行身份分配并输出带最终 ID 的视频。"""

    tracked_frames = _load_raw_frames(raw_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频：{video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    descriptors: list[dict] = []
    source_frames = []
    try:
        for persons in tracked_frames:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("视频帧数少于 raw_tracks.json")
            source_frames.append(frame)
            descriptors.append(
                {
                    person.person_id: descriptor
                    for person in persons
                    if (
                        descriptor := extract_appearance_descriptor(
                            frame, person.box
                        )
                    )
                    is not None
                }
            )
    finally:
        capture.release()

    final_frames, diagnostic = assign_fixed_identities(
        tracked_frames,
        descriptors,
        expected_count,
        width,
        height,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    preview_path = output_directory / "identity_preview.mp4"
    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("无法创建身份诊断预览")
    try:
        for frame, persons in zip(source_frames, final_frames):
            writer.write(draw_tracked_persons(frame, persons))
    finally:
        writer.release()

    counts = Counter(len(persons) for persons in final_frames)
    coverage = {
        identity_id: sum(
            any(person.person_id == identity_id for person in persons)
            for persons in final_frames
        )
        for identity_id in range(1, expected_count + 1)
    }
    summary = {
        "video": video_path.name,
        "frame_count": len(final_frames),
        "anchor_frame": diagnostic.anchor_frame,
        "final_id_count": diagnostic.final_id_count,
        "people_per_frame": dict(sorted(counts.items())),
        "identity_coverage": coverage,
        "source_switches": diagnostic.source_switches,
        "mean_assignment_cost": diagnostic.mean_assignment_cost,
    }
    (output_directory / "identity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    """解析参数并运行身份评估。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("raw_tracks", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--expected-count", type=int, default=4)
    arguments = parser.parse_args()
    evaluate_identity(
        arguments.video,
        arguments.raw_tracks,
        arguments.output_directory,
        arguments.expected_count,
    )


if __name__ == "__main__":
    main()
