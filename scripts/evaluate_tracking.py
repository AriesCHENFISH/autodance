"""批量运行人物检测追踪配置并保存原始在线轨迹指标。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision import PersonDetector, extract_tracked_persons  # noqa: E402


@dataclass(frozen=True)
class ExperimentConfig:
    """一组可复现的检测追踪参数。"""

    name: str
    model_name: str
    image_size: int
    iou_threshold: float
    confidence: float = 0.10
    tracker_name: str = "Dance BoT-SORT"


CONFIGS = {
    "baseline_s640_iou70": ExperimentConfig(
        "baseline_s640_iou70", "yolo11s-pose.pt", 640, 0.70
    ),
    "s960_iou70": ExperimentConfig(
        "s960_iou70", "yolo11s-pose.pt", 960, 0.70
    ),
    "s960_iou50": ExperimentConfig(
        "s960_iou50", "yolo11s-pose.pt", 960, 0.50
    ),
}


def _person_payload(person) -> dict:
    """保存诊断所需的在线 ID、脚点、检测框和置信度。"""

    return {
        "online_id": person.person_id,
        "x": round(person.x, 3),
        "y": round(person.y, 3),
        "box": list(person.box),
        "confidence": (
            round(person.confidence, 5)
            if person.confidence is not None
            else None
        ),
    }


def _summarize(frames: list[dict], elapsed_seconds: float) -> dict:
    """计算不依赖人工真值的检测覆盖率和轨迹连续性指标。"""

    expected_count = 4
    counts = [len(frame["persons"]) for frame in frames]
    ids = sorted(
        {
            person["online_id"]
            for frame in frames
            for person in frame["persons"]
        }
    )
    observations = {online_id: [] for online_id in ids}
    for frame in frames:
        for person in frame["persons"]:
            observations[person["online_id"]].append(
                (frame["frame_id"], person)
            )

    identity_metrics = {}
    for online_id, values in observations.items():
        gaps = sum(
            right[0] - left[0] > 1
            for left, right in zip(values, values[1:])
        )
        jumps = []
        for (left_frame, left), (right_frame, right) in zip(values, values[1:]):
            if right_frame == left_frame + 1:
                jumps.append(
                    math.dist((left["x"], left["y"]), (right["x"], right["y"]))
                )
        identity_metrics[str(online_id)] = {
            "observations": len(values),
            "first_frame": values[0][0],
            "last_frame": values[-1][0],
            "gaps": gaps,
            "max_pixel_jump": round(max(jumps, default=0.0), 3),
        }

    total = max(1, len(frames))
    full_frames = sum(count == expected_count for count in counts)
    return {
        "frame_count": len(frames),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "seconds_per_frame": round(elapsed_seconds / total, 4),
        "online_id_count": len(ids),
        "people_per_frame": dict(sorted(Counter(counts).items())),
        "full_detection_frames": full_frames,
        "full_detection_rate": round(full_frames / total, 4),
        "under_detection_rate": round(
            sum(count < expected_count for count in counts) / total, 4
        ),
        "over_detection_rate": round(
            sum(count > expected_count for count in counts) / total, 4
        ),
        "identity_metrics": identity_metrics,
    }


def evaluate_video(
    video_path: Path,
    config: ExperimentConfig,
    output_directory: Path,
) -> dict:
    """用指定配置处理一个视频并保存未经归并的逐帧轨迹。"""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法打开测试视频：{video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 24.0
    detector = PersonDetector(
        model_name=config.model_name,
        tracker_name=config.tracker_name,
        confidence=config.confidence,
        image_size=config.image_size,
        iou_threshold=config.iou_threshold,
    )
    frames: list[dict] = []
    started = time.perf_counter()
    try:
        frame_id = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            result = detector.process_frame(frame)
            persons = extract_tracked_persons(result)
            frames.append(
                {
                    "frame_id": frame_id,
                    "timestamp": round(frame_id / fps, 3),
                    "persons": [_person_payload(person) for person in persons],
                }
            )
            frame_id += 1
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    summary = _summarize(frames, elapsed)
    summary.update({"video": video_path.name, "config": asdict(config)})

    run_directory = output_directory / config.name / video_path.stem
    run_directory.mkdir(parents=True, exist_ok=True)
    with (run_directory / "raw_tracks.json").open("w", encoding="utf-8") as file:
        json.dump(frames, file, ensure_ascii=False, indent=2)
    with (run_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    """解析命令行参数并运行一组视频实验。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("videos", nargs="+", type=Path)
    arguments = parser.parse_args()
    config = CONFIGS[arguments.config]
    for video in arguments.videos:
        summary = evaluate_video(video, config, arguments.output_directory)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
