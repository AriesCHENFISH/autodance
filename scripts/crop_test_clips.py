"""从固定机位源视频一次解码并裁剪多个追踪测试片段。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


DEFAULT_SEGMENTS = (
    ("segment_a_15_29", 15.0, 29.0),
    ("segment_b_38_52", 38.0, 52.0),
    ("segment_c_60_74", 60.0, 74.0),
)


def crop_segments(source: Path, output_directory: Path) -> list[Path]:
    """单次顺序读取源视频，生成三个 14 秒测试片段。"""

    output_directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"无法打开源视频：{source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("源视频的 FPS 或画面尺寸无效")

    writers: dict[str, cv2.VideoWriter] = {}
    output_paths: list[Path] = []
    for name, _start, _end in DEFAULT_SEGMENTS:
        output_path = output_directory / f"{name}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            for opened_writer in writers.values():
                opened_writer.release()
            raise RuntimeError(f"无法创建测试片段：{output_path}")
        writers[name] = writer
        output_paths.append(output_path)

    frame_id = 0
    final_frame = round(max(end for _name, _start, end in DEFAULT_SEGMENTS) * fps)
    try:
        while frame_id < final_frame:
            success, frame = capture.read()
            if not success:
                break
            timestamp = frame_id / fps
            for name, start, end in DEFAULT_SEGMENTS:
                if start <= timestamp < end:
                    writers[name].write(frame)
            frame_id += 1
    finally:
        capture.release()
        for writer in writers.values():
            writer.release()
    return output_paths


def main() -> None:
    """解析命令行参数并执行视频裁剪。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    for path in crop_segments(arguments.source, arguments.output_directory):
        print(path)


if __name__ == "__main__":
    main()
