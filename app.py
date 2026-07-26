"""AutoDance Lab Phase 1 的 Gradio 启动入口。"""

from datetime import datetime
import json
import logging
from pathlib import Path
import traceback
import uuid

import cv2
import gradio as gr

from vision import PersonDetector, draw_tracked_persons, extract_tracked_persons
from vision import (
    consolidate_track_ids,
    extract_appearance_descriptor,
    relabel_tracked_frames,
)


DATA_DIR = Path("data")
RUNS_DIR = DATA_DIR / "runs"
MODEL_NAME = "yolo11s-pose.pt"


def _create_run_directory() -> Path:
    """为本次分析创建独立的相对路径输出目录。"""

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    run_directory = RUNS_DIR / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory


def _create_logger(log_path: Path) -> logging.Logger:
    """创建只写入本次任务日志文件的日志器。"""

    logger = logging.getLogger(f"autodance.{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    """关闭日志文件句柄，避免重复分析时泄漏资源。"""

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _save_tracks(
    output_path: Path,
    frames: list[dict],
) -> None:
    """以 UTF-8 JSON 保存逐帧人物位置结果。"""

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(frames, file, ensure_ascii=False, indent=2)


def _open_video_writer(
    output_path: Path, fps: float, width: int, height: int
) -> cv2.VideoWriter:
    """创建 MP4 标注视频写入器，创建失败时抛出可记录的异常。"""

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("无法创建预览视频，请检查 OpenCV 的视频编码支持")
    return writer


def analyze_video(
    video_path: str | None,
    tracker_name: str,
    confidence: float,
    expected_people: int = 5,
) -> tuple[str | None, str | None, str | None, str]:
    """分析上传视频并返回标注视频、tracks.json、日志和状态信息。"""

    if not video_path:
        return None, None, None, "请先上传一个 MP4 视频。"

    run_directory = _create_run_directory()
    tracks_path = run_directory / "tracks.json"
    preview_path = run_directory / "tracked_preview.mp4"
    log_path = run_directory / "analysis.log"
    logger = _create_logger(log_path)
    capture: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    frame_results: list[dict] = []
    tracked_frames = []
    final_tracked_frames = []
    appearance_descriptors: dict[int, list] = {}
    failed_frames = 0

    try:
        logger.info("开始分析视频：%s", video_path)
        logger.info(
            "模型：%s，追踪器：%s，检测下限：%.2f，预期人数：%d",
            MODEL_NAME,
            tracker_name,
            confidence,
            expected_people,
        )

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("无法打开上传的视频文件")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            logger.warning("视频 FPS 无效，使用默认值 25")
            fps = 25.0
        if width <= 0 or height <= 0:
            raise ValueError("无法读取视频画面尺寸")

        detector = PersonDetector(
            model_name=MODEL_NAME,
            tracker_name=tracker_name,
            confidence=float(confidence),
        )

        frame_id = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            persons = []
            try:
                result = detector.process_frame(frame)
                persons = extract_tracked_persons(result)
                # 每三帧采样一次衣着特征，兼顾外观稳定性和 CPU 性能。
                if frame_id % 3 == 0:
                    for person in persons:
                        descriptor = extract_appearance_descriptor(frame, person.box)
                        if descriptor is not None:
                            appearance_descriptors.setdefault(
                                person.person_id, []
                            ).append(descriptor)
            except Exception:
                failed_frames += 1
                logger.exception("第 %d 帧处理失败，已保留原始画面", frame_id)
            tracked_frames.append(persons)
            frame_id += 1

        if frame_id == 0:
            raise ValueError("视频中没有可读取的画面")

        raw_unique_ids = {
            person.person_id
            for persons in tracked_frames
            for person in persons
        }
        consolidation = consolidate_track_ids(
            tracked_frames,
            appearance_descriptors,
            expected_count=int(expected_people),
        )
        final_tracked_frames = relabel_tracked_frames(
            tracked_frames,
            consolidation,
        )
        logger.info(
            "身份归并：%d 个在线 ID -> %d 个最终 ID；锚点帧：%s；丢弃轨迹：%s；映射：%s",
            consolidation.source_id_count,
            consolidation.final_id_count,
            consolidation.anchor_frame,
            list(consolidation.dropped_ids),
            consolidation.mapping,
        )

        frame_results = [
            {
                "frame_id": current_frame_id,
                "timestamp": round(current_frame_id / fps, 3),
                "persons": [person.to_json() for person in persons],
            }
            for current_frame_id, persons in enumerate(final_tracked_frames)
        ]
        _save_tracks(tracks_path, frame_results)

        # 身份归并完成后重新读取原视频，确保预览展示的是最终 1..N 编号。
        capture.release()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError("身份归并后无法重新打开原视频生成预览")
        writer = _open_video_writer(preview_path, fps, width, height)
        rendered_frames = 0
        for persons in final_tracked_frames:
            success, frame = capture.read()
            if not success:
                raise RuntimeError(
                    f"生成最终预览时只能读取 {rendered_frames}/{frame_id} 帧"
                )
            writer.write(draw_tracked_persons(frame, persons))
            rendered_frames += 1

        unique_ids = {
            person["id"]
            for frame_result in frame_results
            for person in frame_result["persons"]
        }
        if len(unique_ids) != int(expected_people):
            logger.warning(
                "最终得到 %d 个身份，与预期人数 %d 不一致；请检查是否存在全程漏检。",
                len(unique_ids),
                expected_people,
            )
        logger.info(
            "分析完成：读取 %d/%d 帧，在线 ID %d 个，最终 ID %d 个，失败 %d 帧",
            frame_id,
            total_frames,
            len(raw_unique_ids),
            len(unique_ids),
            failed_frames,
        )
        status = (
            f"分析完成：共处理 {frame_id} 帧，在线跟踪产生 {len(raw_unique_ids)} 个 ID，"
            f"已归并为 {len(unique_ids)} 个最终人物 ID，"
            f"单帧处理失败 {failed_frames} 次。所有结果保存在 `{run_directory}`。"
        )
        return str(preview_path), str(tracks_path), str(log_path), status
    except Exception as error:
        logger.error("分析任务失败：%s\n%s", error, traceback.format_exc())
        if not frame_results:
            fallback_frames = final_tracked_frames or tracked_frames
            frame_results = [
                {
                    "frame_id": current_frame_id,
                    "timestamp": round(current_frame_id / fps, 3)
                    if "fps" in locals() and fps > 0
                    else 0.0,
                    "persons": [person.to_json() for person in persons],
                }
                for current_frame_id, persons in enumerate(fallback_frames)
            ]
        _save_tracks(tracks_path, frame_results)
        status = f"分析未能完成：{error}。已保留部分 JSON 和错误日志。"
        return (
            str(preview_path) if frame_results and preview_path.exists() else None,
            str(tracks_path),
            str(log_path),
            status,
        )
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        _close_logger(logger)


def build_app() -> gr.Blocks:
    """构建 AutoDance Lab Phase 1 的 Gradio 页面。"""

    with gr.Blocks(title="AutoDance Lab") as demo:
        gr.Markdown(
            """
            # AutoDance Lab
            **Phase 1：固定机位舞蹈视频的人物检测与追踪**

            上传 4–9 人 MP4 视频后，系统会输出带人物 ID 的预览视频、逐帧位置
            `tracks.json` 及运行日志。人物坐标取脚踝中点，关键点不可用时使用检测框底边中心。
            """
        )
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="上传舞蹈视频")
                tracker_input = gr.Radio(
                    choices=["Dance BoT-SORT", "ByteTrack", "BoT-SORT"],
                    value="Dance BoT-SORT",
                    label="追踪器",
                )
                confidence_input = gr.Slider(
                    minimum=0.05,
                    maximum=0.9,
                    value=0.10,
                    step=0.05,
                    label="送入追踪器的最低检测置信度",
                )
                expected_people_input = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=5,
                    step=1,
                    label="视频中的固定人数",
                )
                analyze_button = gr.Button("开始分析", variant="primary")
            with gr.Column():
                preview_output = gr.Video(label="带 ID 的追踪视频")
                status_output = gr.Markdown()
                tracks_output = gr.File(label="下载 tracks.json")
                log_output = gr.File(label="下载 analysis.log")

        analyze_button.click(
            fn=analyze_video,
            inputs=[
                video_input,
                tracker_input,
                confidence_input,
                expected_people_input,
            ],
            outputs=[preview_output, tracks_output, log_output, status_output],
        )
    return demo


if __name__ == "__main__":
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    app = build_app()
    app.queue().launch(server_name="0.0.0.0", show_error=True)
