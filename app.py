"""AutoDance Lab 人物追踪、队形检测、编辑与 SVG 导出入口。"""

from datetime import datetime
import json
import logging
from pathlib import Path
import traceback
import uuid

import cv2
import gradio as gr
import numpy as np

from formation import (
    analyze_formations,
    draw_perspective_grid,
    draw_stabilized_grid_positions,
    person_to_grid_json,
    stabilize_grid_tracks,
)
from vision import (
    CalibrationError,
    PersonDetector,
    StageCalibration,
    assign_fixed_identities,
    draw_tracked_persons,
    extract_tracked_persons,
)
from vision import (
    consolidate_track_ids,
    extract_appearance_descriptor,
    relabel_tracked_frames,
)
from visualization import (
    FormationValidationError,
    canvas_to_grid,
    delete_formation,
    duplicate_formation,
    formation_label,
    move_person,
    normalize_formations,
    persons_to_rows,
    render_editor_canvas,
    render_formation_svg,
    selected_index,
    update_formation,
    write_formation_exports,
)


DATA_DIR = Path("data")
RUNS_DIR = DATA_DIR / "runs"
MODEL_NAME = "yolo11s-pose.pt"
INFERENCE_IMAGE_SIZE = 960
INFERENCE_IOU_THRESHOLD = 0.50
GRID_COLUMNS = 9
GRID_ROWS = 9
CALIBRATION_LABELS = ("左上", "右上", "右下", "左下")


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


def _raw_person_json(person) -> dict:
    """保存离线归并前的在线 ID、脚点、检测框和置信度。"""

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


def _read_first_video_frame(video_path: str) -> np.ndarray:
    """读取视频首帧并转换成 Gradio Image 使用的 RGB 数组。"""

    capture = cv2.VideoCapture(str(video_path))
    try:
        success, frame = capture.read()
        if not success:
            raise ValueError("无法读取视频首帧")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _render_calibration_frame(
    original_rgb: np.ndarray,
    points: list[list[float]],
) -> np.ndarray:
    """显示已选角点；四点有效时同时预览完整透视网格。"""

    rendered = original_rgb.copy()
    if len(points) == 4:
        try:
            height, width = rendered.shape[:2]
            calibration = StageCalibration.from_points(
                points,
                width,
                height,
                columns=GRID_COLUMNS,
                rows=GRID_ROWS,
            )
            rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            rendered_bgr = draw_perspective_grid(rendered_bgr, calibration)
            rendered = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
        except CalibrationError:
            pass

    for index, point in enumerate(points):
        position = (round(point[0]), round(point[1]))
        cv2.circle(rendered, position, 8, (255, 90, 40), -1, cv2.LINE_AA)
        cv2.circle(rendered, position, 11, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            rendered,
            str(index + 1),
            (position[0] + 12, position[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            str(index + 1),
            (position[0] + 12, position[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 40, 30),
            1,
            cv2.LINE_AA,
        )
    if len(points) >= 2:
        polyline = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            rendered,
            [polyline],
            len(points) == 4,
            (255, 180, 50),
            2,
            cv2.LINE_AA,
        )
    return rendered


def prepare_calibration(
    video_path: str | None,
) -> tuple[np.ndarray | None, np.ndarray | None, list, str]:
    """上传视频后提取首帧并重置四点标定状态。"""

    if not video_path:
        return None, None, [], "请先上传视频。"
    try:
        first_frame = _read_first_video_frame(video_path)
        return (
            first_frame,
            first_frame.copy(),
            [],
            "请点击第 1 个点：舞台左上角。",
        )
    except Exception as error:
        return None, None, [], f"无法准备标定画面：{error}"


def add_calibration_point(
    original_rgb: np.ndarray | None,
    points: list | None,
    x: float,
    y: float,
) -> tuple[np.ndarray | None, list, str]:
    """向标定状态添加一个点，便于 UI 回调和单元测试共用。"""

    if original_rgb is None:
        return None, [], "请先上传视频。"
    selected = [
        [float(point[0]), float(point[1])] for point in (points or [])
    ]
    if len(selected) >= 4:
        return (
            _render_calibration_frame(original_rgb, selected),
            selected,
            "四点标定已完成；如需修改，请点击“重新标定”。",
        )

    selected.append([float(x), float(y)])
    if len(selected) < 4:
        next_label = CALIBRATION_LABELS[len(selected)]
        status = (
            f"已选择 {len(selected)}/4 个点；请点击第 {len(selected) + 1} 个点："
            f"舞台{next_label}角。"
        )
    else:
        height, width = original_rgb.shape[:2]
        try:
            StageCalibration.from_points(
                selected,
                width,
                height,
                columns=GRID_COLUMNS,
                rows=GRID_ROWS,
            )
            status = "四点标定有效，9×9 透视网格预览已生成。"
        except CalibrationError as error:
            status = f"标定无效：{error} 请点击“重新标定”后重试。"
    return _render_calibration_frame(original_rgb, selected), selected, status


def _select_calibration_point(
    original_rgb: np.ndarray | None,
    points: list | None,
    event: gr.SelectData,
) -> tuple[np.ndarray | None, list, str]:
    """Gradio 图片点击事件适配器。"""

    if not isinstance(event.index, (tuple, list)) or len(event.index) < 2:
        rendered = (
            _render_calibration_frame(original_rgb, points or [])
            if original_rgb is not None
            else None
        )
        return rendered, points or [], "没有读取到有效的图片坐标，请重试。"
    return add_calibration_point(
        original_rgb,
        points,
        float(event.index[0]),
        float(event.index[1]),
    )


def reset_calibration(
    original_rgb: np.ndarray | None,
) -> tuple[np.ndarray | None, list, str]:
    """清空四点并恢复未标注的首帧。"""

    if original_rgb is None:
        return None, [], "请先上传视频。"
    return original_rgb.copy(), [], "已清空；请点击第 1 个点：舞台左上角。"


def _uploaded_path(value: object) -> Path:
    """兼容 Gradio File 返回的字符串、Path 和临时文件对象。"""

    if isinstance(value, (str, Path)):
        return Path(value)
    name = getattr(value, "name", None)
    if name:
        return Path(name)
    if isinstance(value, dict):
        candidate = value.get("path") or value.get("name")
        if candidate:
            return Path(candidate)
    raise FormationValidationError("请选择 formations.json 文件")


def _editor_payload(
    formations: list[dict],
    index: int,
    status: str,
) -> tuple:
    formation = formations[index]
    labels = [formation_label(item) for item in formations]
    selected_label = labels[index]
    person_ids = [int(person["id"]) for person in formation["persons"]]
    selected_person = person_ids[0]
    return (
        gr.Dropdown(choices=labels, value=selected_label),
        formation["name"],
        formation["time"],
        gr.Dropdown(choices=person_ids, value=selected_person),
        persons_to_rows(formation),
        render_editor_canvas(formation, selected_person),
        render_formation_svg(formation),
        status,
    )


def _empty_editor_payload(status: str) -> tuple:
    return (
        gr.Dropdown(choices=[], value=None),
        "",
        0,
        gr.Dropdown(choices=[], value=None),
        [],
        None,
        "",
        status,
    )


def load_formation_editor(source: object) -> tuple:
    """加载 Phase 3 JSON，初始化 Phase 4 编辑器。"""

    try:
        path = _uploaded_path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
        formations = normalize_formations(payload)
        return (
            formations,
            *_editor_payload(
                formations,
                0,
                f"已加载 {len(formations)} 个关键队形。选择人物后可点击画布移动。",
            ),
        )
    except Exception as error:
        return [], *_empty_editor_payload(f"加载失败：{error}")


def select_editor_formation(label: str, formations: list[dict]) -> tuple:
    try:
        index = selected_index(formations, label)
        return _editor_payload(formations, index, f"正在编辑队形 {index + 1}。")
    except Exception as error:
        return _empty_editor_payload(f"选择失败：{error}")


def apply_formation_edits(
    formations: list[dict],
    label: str,
    name: str,
    time: float,
    person_table: object,
) -> tuple:
    try:
        index = selected_index(formations, label)
        updated = update_formation(formations, index, name, time, person_table)
        return updated, *_editor_payload(updated, index, "当前队形修改已应用。")
    except Exception as error:
        if not formations:
            return formations, *_empty_editor_payload(f"应用失败：{error}")
        index = selected_index(formations, label)
        return formations, *_editor_payload(formations, index, f"应用失败：{error}")


def move_selected_person(
    formations: list[dict],
    label: str,
    name: str,
    time: float,
    person_table: object,
    person_id: int,
    event: gr.SelectData,
) -> tuple:
    """先吸收表格编辑，再把所选人物移动到点击的最近格点。"""

    try:
        index = selected_index(formations, label)
        updated = update_formation(formations, index, name, time, person_table)
        if not isinstance(event.index, (tuple, list)) or len(event.index) < 2:
            raise FormationValidationError("没有读取到有效画布坐标")
        x, y = canvas_to_grid(updated[index], event.index[0], event.index[1])
        updated = move_person(updated, index, int(person_id), x, y)
        formation = updated[index]
        return (
            updated,
            persons_to_rows(formation),
            render_editor_canvas(formation, int(person_id)),
            render_formation_svg(formation),
            f"人物 {int(person_id)} 已移动到 ({x}, {y})。",
        )
    except Exception as error:
        if not formations:
            return formations, [], None, "", f"移动失败：{error}"
        index = selected_index(formations, label)
        return (
            formations,
            persons_to_rows(formations[index]),
            render_editor_canvas(formations[index], int(person_id)),
            render_formation_svg(formations[index]),
            f"移动失败：{error}",
        )


def highlight_editor_person(
    formations: list[dict], label: str, person_id: int
) -> np.ndarray | None:
    try:
        index = selected_index(formations, label)
        return render_editor_canvas(formations[index], int(person_id))
    except Exception:
        return None


def duplicate_editor_formation(
    formations: list[dict], label: str
) -> tuple:
    try:
        index = selected_index(formations, label)
        updated = duplicate_formation(formations, index)
        return updated, *_editor_payload(updated, index + 1, "已复制为新队形。")
    except Exception as error:
        if not formations:
            return formations, *_empty_editor_payload(f"复制失败：{error}")
        index = selected_index(formations, label)
        return formations, *_editor_payload(formations, index, f"复制失败：{error}")


def delete_editor_formation(
    formations: list[dict], label: str
) -> tuple:
    try:
        index = selected_index(formations, label)
        updated = delete_formation(formations, index)
        next_index = min(index, len(updated) - 1)
        return updated, *_editor_payload(updated, next_index, "当前队形已删除。")
    except Exception as error:
        if not formations:
            return formations, *_empty_editor_payload(f"删除失败：{error}")
        index = selected_index(formations, label)
        return formations, *_editor_payload(formations, index, f"删除失败：{error}")


def save_formation_editor(
    formations: list[dict],
    label: str,
    name: str,
    time: float,
    person_table: object,
) -> tuple[str | None, str | None, str | None, str]:
    """应用未保存编辑并导出 JSON、当前 SVG 和全量 ZIP。"""

    try:
        index = selected_index(formations, label)
        updated = update_formation(formations, index, name, time, person_table)
        export_directory = DATA_DIR / "outputs" / (
            datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        )
        json_path, svg_path, zip_path = write_formation_exports(
            updated, export_directory, index
        )
        return (
            str(json_path),
            str(svg_path),
            str(zip_path),
            f"已导出 {len(updated)} 个队形到 `{export_directory}`。",
        )
    except Exception as error:
        return None, None, None, f"导出失败：{error}"


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
    formation_grid_width: int = 20,
    formation_grid_height: int = 20,
    calibration_points: list | None = None,
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str,
]:
    """分析视频并返回预览、轨迹、队形、标定、日志和状态信息。"""

    if not video_path:
        return None, None, None, None, None, None, "请先上传一个 MP4 视频。"
    if calibration_points is None or len(calibration_points) != 4:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            "请先按左上、右上、右下、左下顺序完成四点透视标定。",
        )
    formation_grid_width = int(formation_grid_width)
    formation_grid_height = int(formation_grid_height)
    if not 8 <= formation_grid_width <= 40 or not 8 <= formation_grid_height <= 40:
        return None, None, None, None, None, None, "队形网格宽度和高度必须在 8 到 40 之间。"

    run_directory = _create_run_directory()
    tracks_path = run_directory / "tracks.json"
    formations_path = run_directory / "formations.json"
    raw_tracks_path = run_directory / "raw_tracks.json"
    calibration_path = run_directory / "calibration.json"
    preview_path = run_directory / "tracked_preview.mp4"
    log_path = run_directory / "analysis.log"
    logger = _create_logger(log_path)
    capture: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    frame_results: list[dict] = []
    tracked_frames = []
    final_tracked_frames = []
    frame_descriptors: list[dict] = []
    failed_frames = 0

    try:
        logger.info("开始分析视频：%s", video_path)
        logger.info(
            "模型：%s，追踪器：%s，检测下限：%.2f，预期人数：%d，"
            "队形网格：%dx%d",
            MODEL_NAME,
            tracker_name,
            confidence,
            expected_people,
            formation_grid_width,
            formation_grid_height,
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

        calibration = StageCalibration.from_points(
            calibration_points,
            width,
            height,
            columns=GRID_COLUMNS,
            rows=GRID_ROWS,
        )
        with calibration_path.open("w", encoding="utf-8") as file:
            json.dump(calibration.to_json(), file, ensure_ascii=False, indent=2)
        logger.info(
            "透视标定：%dx%d 网格，像素角点：%s",
            GRID_COLUMNS,
            GRID_ROWS,
            calibration.pixel_points,
        )

        detector = PersonDetector(
            model_name=MODEL_NAME,
            tracker_name=tracker_name,
            confidence=float(confidence),
            image_size=INFERENCE_IMAGE_SIZE,
            iou_threshold=INFERENCE_IOU_THRESHOLD,
        )

        frame_id = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            persons = []
            descriptors = {}
            try:
                result = detector.process_frame(frame)
                persons = extract_tracked_persons(result)
                for person in persons:
                    descriptor = extract_appearance_descriptor(frame, person.box)
                    if descriptor is not None:
                        descriptors[person.person_id] = descriptor
            except Exception:
                failed_frames += 1
                logger.exception("第 %d 帧处理失败，已保留原始画面", frame_id)
            tracked_frames.append(persons)
            frame_descriptors.append(descriptors)
            frame_id += 1

        if frame_id == 0:
            raise ValueError("视频中没有可读取的画面")

        raw_unique_ids = {
            person.person_id
            for persons in tracked_frames
            for person in persons
        }
        raw_frame_results = [
            {
                "frame_id": current_frame_id,
                "timestamp": round(current_frame_id / fps, 3),
                "persons": [
                    _raw_person_json(person) for person in persons
                ],
            }
            for current_frame_id, persons in enumerate(tracked_frames)
        ]
        _save_tracks(raw_tracks_path, raw_frame_results)

        final_tracked_frames, fixed_assignment = assign_fixed_identities(
            tracked_frames,
            expected_count=int(expected_people),
            frame_descriptors=frame_descriptors,
            frame_width=width,
            frame_height=height,
        )
        if fixed_assignment.final_id_count == int(expected_people):
            logger.info(
                "固定身份分配：%d 个在线 ID -> %d 个最终 ID；锚点帧：%s；"
                "在线来源切换：%s；平均代价：%.5f",
                len(raw_unique_ids),
                fixed_assignment.final_id_count,
                fixed_assignment.anchor_frame,
                fixed_assignment.source_switches,
                fixed_assignment.mean_assignment_cost,
            )
        else:
            logger.warning(
                "没有找到人数齐全的干净锚点，降级使用轨迹片段归并。"
            )
            appearance_descriptors: dict[int, list] = {}
            for descriptors in frame_descriptors:
                for source_id, descriptor in descriptors.items():
                    appearance_descriptors.setdefault(source_id, []).append(
                        descriptor
                    )
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
                "降级身份归并：%d 个在线 ID -> %d 个最终 ID；映射：%s",
                consolidation.source_id_count,
                consolidation.final_id_count,
                consolidation.mapping,
            )

        frame_results = [
            {
                "frame_id": current_frame_id,
                "timestamp": round(current_frame_id / fps, 3),
                "persons": [
                    person_to_grid_json(person, calibration)
                    for person in persons
                ],
            }
            for current_frame_id, persons in enumerate(final_tracked_frames)
        ]
        frame_results = stabilize_grid_tracks(
            frame_results,
            columns=GRID_COLUMNS,
            rows=GRID_ROWS,
        )
        interpolated_positions = sum(
            person.get("interpolated", False)
            for frame_result in frame_results
            for person in frame_result["persons"]
        )
        unresolved_positions = sum(
            not person.get("in_stage", False)
            for frame_result in frame_results
            for person in frame_result["persons"]
        )
        logger.info(
            "坐标稳定：插值修复 %d 个人次，仍越界 %d 个人次",
            interpolated_positions,
            unresolved_positions,
        )
        _save_tracks(tracks_path, frame_results)
        formations = analyze_formations(
            frame_results,
            fps,
            grid_width=formation_grid_width,
            grid_height=formation_grid_height,
        )
        _save_tracks(formations_path, formations)
        if formations:
            logger.info(
                "关键队形：识别 %d 个，时间点：%s",
                len(formations),
                ", ".join(f"{formation['time']:.2f}s" for formation in formations),
            )
        else:
            logger.warning("关键队形：全片没有满足条件的稳定窗口，输出空列表")

        # 身份归并完成后重新读取原视频，确保预览展示的是最终 1..N 编号。
        capture.release()
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError("身份归并后无法重新打开原视频生成预览")
        writer = _open_video_writer(preview_path, fps, width, height)
        rendered_frames = 0
        for current_frame_id, persons in enumerate(final_tracked_frames):
            success, frame = capture.read()
            if not success:
                raise RuntimeError(
                    f"生成最终预览时只能读取 {rendered_frames}/{frame_id} 帧"
                )
            annotated = draw_perspective_grid(frame, calibration)
            annotated = draw_tracked_persons(
                annotated, persons, draw_position=False
            )
            annotated = draw_stabilized_grid_positions(
                annotated,
                frame_results[current_frame_id],
                calibration,
            )
            writer.write(annotated)
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
        formation_summary = (
            "（"
            + ", ".join(f"{formation['time']:.2f}s" for formation in formations)
            + "）"
            if formations
            else ""
        )
        status = (
            f"分析完成：共处理 {frame_id} 帧，在线跟踪产生 {len(raw_unique_ids)} 个 ID，"
            f"已归并为 {len(unique_ids)} 个最终人物 ID，"
            f"已映射到 {GRID_COLUMNS}×{GRID_ROWS} 舞台网格，"
            f"识别 {len(formations)} 个关键队形{formation_summary}，"
            f"单帧处理失败 {failed_frames} 次。所有结果保存在 `{run_directory}`。"
        )
        return (
            str(preview_path),
            str(tracks_path),
            str(raw_tracks_path),
            str(formations_path),
            str(calibration_path),
            str(log_path),
            status,
        )
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
            str(raw_tracks_path) if raw_tracks_path.exists() else None,
            str(formations_path) if formations_path.exists() else None,
            str(calibration_path) if calibration_path.exists() else None,
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
    """构建 AutoDance Lab Phase 1–4 的 Gradio 页面。"""

    with gr.Blocks(title="AutoDance Lab") as demo:
        gr.Markdown(
            """
            # AutoDance Lab
            **Phase 1–4：人物追踪、关键队形检测、编辑与 SVG 导出**

            上传视频后，在首帧按 **左上 → 右上 → 右下 → 左下** 点击舞台四角。
            系统会输出最终人物 ID、舞台坐标、带透视网格的预览视频，以及按
            稳定窗口识别的关键队形 `formations.json`。
            """
        )
        calibration_frame_state = gr.State()
        calibration_points_state = gr.State([])
        formation_editor_state = gr.State([])
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="上传舞蹈视频")
                gr.Markdown(
                    "### 透视标定\n"
                    "依次点击舞台区域的左上、右上、右下、左下四个角点。"
                )
                calibration_image = gr.Image(
                    label="点击首帧完成四点标定",
                    type="numpy",
                    interactive=False,
                )
                calibration_status = gr.Markdown("请先上传视频。")
                reset_calibration_button = gr.Button("重新标定")
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
                with gr.Row():
                    formation_grid_width_input = gr.Number(
                        value=20,
                        minimum=8,
                        maximum=40,
                        precision=0,
                        label="队形网格宽度",
                    )
                    formation_grid_height_input = gr.Number(
                        value=20,
                        minimum=8,
                        maximum=40,
                        precision=0,
                        label="队形网格高度",
                    )
                analyze_button = gr.Button("开始分析", variant="primary")
            with gr.Column():
                preview_output = gr.Video(label="带 ID 的追踪视频")
                status_output = gr.Markdown()
                tracks_output = gr.File(label="下载 tracks.json")
                raw_tracks_output = gr.File(label="下载 raw_tracks.json（诊断）")
                formations_output = gr.File(label="下载 formations.json")
                calibration_output = gr.File(label="下载 calibration.json")
                log_output = gr.File(label="下载 analysis.log")

        gr.Markdown(
            """
            ## Phase 4：队形编辑与 SVG

            分析完成后会自动载入关键队形；也可以单独上传已有的
            `formations.json`。选择人物后点击画布可移动到最近格点，或直接修改
            坐标表。点击“应用修改”后再切换队形。
            """
        )
        with gr.Row():
            with gr.Column(scale=2):
                formation_import_input = gr.File(
                    label="载入 formations.json",
                    file_types=[".json"],
                )
                load_formations_button = gr.Button("载入队形")
                formation_selector = gr.Dropdown(
                    choices=[], label="当前队形"
                )
                with gr.Row():
                    formation_name_input = gr.Textbox(label="队形名称")
                    formation_time_input = gr.Number(
                        label="展示时间（秒）", minimum=0
                    )
                person_selector = gr.Dropdown(
                    choices=[], label="点击画布时要移动的人物 ID"
                )
                person_table = gr.Dataframe(
                    headers=["ID", "X", "Y"],
                    datatype=["number", "number", "number"],
                    row_count=(1, "dynamic"),
                    column_count=3,
                    interactive=True,
                    label="人物格位（可直接编辑）",
                )
                with gr.Row():
                    apply_formation_button = gr.Button(
                        "应用修改", variant="primary"
                    )
                    duplicate_formation_button = gr.Button("复制为新队形")
                    delete_formation_button = gr.Button(
                        "删除当前队形", variant="stop"
                    )
                editor_status = gr.Markdown("尚未载入关键队形。")
            with gr.Column(scale=3):
                editor_canvas = gr.Image(
                    label="选择人物后点击目标格点",
                    type="numpy",
                    interactive=False,
                )
                svg_preview = gr.HTML(label="SVG 预览")

        save_formations_button = gr.Button(
            "保存并导出 SVG", variant="primary"
        )
        with gr.Row():
            edited_formations_output = gr.File(
                label="下载 formations_edited.json"
            )
            selected_svg_output = gr.File(label="下载当前队形 SVG")
            all_svgs_output = gr.File(label="下载全部 SVG（ZIP）")

        video_input.change(
            fn=prepare_calibration,
            inputs=[video_input],
            outputs=[
                calibration_frame_state,
                calibration_image,
                calibration_points_state,
                calibration_status,
            ],
        )
        calibration_image.select(
            fn=_select_calibration_point,
            inputs=[calibration_frame_state, calibration_points_state],
            outputs=[
                calibration_image,
                calibration_points_state,
                calibration_status,
            ],
        )
        reset_calibration_button.click(
            fn=reset_calibration,
            inputs=[calibration_frame_state],
            outputs=[
                calibration_image,
                calibration_points_state,
                calibration_status,
            ],
        )
        editor_outputs = [
            formation_selector,
            formation_name_input,
            formation_time_input,
            person_selector,
            person_table,
            editor_canvas,
            svg_preview,
            editor_status,
        ]
        load_formations_button.click(
            fn=load_formation_editor,
            inputs=[formation_import_input],
            outputs=[formation_editor_state, *editor_outputs],
        )
        formation_selector.input(
            fn=select_editor_formation,
            inputs=[formation_selector, formation_editor_state],
            outputs=editor_outputs,
        )
        apply_formation_button.click(
            fn=apply_formation_edits,
            inputs=[
                formation_editor_state,
                formation_selector,
                formation_name_input,
                formation_time_input,
                person_table,
            ],
            outputs=[formation_editor_state, *editor_outputs],
        )
        duplicate_formation_button.click(
            fn=duplicate_editor_formation,
            inputs=[formation_editor_state, formation_selector],
            outputs=[formation_editor_state, *editor_outputs],
        )
        delete_formation_button.click(
            fn=delete_editor_formation,
            inputs=[formation_editor_state, formation_selector],
            outputs=[formation_editor_state, *editor_outputs],
        )
        person_selector.input(
            fn=highlight_editor_person,
            inputs=[formation_editor_state, formation_selector, person_selector],
            outputs=[editor_canvas],
        )
        editor_canvas.select(
            fn=move_selected_person,
            inputs=[
                formation_editor_state,
                formation_selector,
                formation_name_input,
                formation_time_input,
                person_table,
                person_selector,
            ],
            outputs=[
                formation_editor_state,
                person_table,
                editor_canvas,
                svg_preview,
                editor_status,
            ],
        )
        save_formations_button.click(
            fn=save_formation_editor,
            inputs=[
                formation_editor_state,
                formation_selector,
                formation_name_input,
                formation_time_input,
                person_table,
            ],
            outputs=[
                edited_formations_output,
                selected_svg_output,
                all_svgs_output,
                editor_status,
            ],
        )
        analysis_event = analyze_button.click(
            fn=analyze_video,
            inputs=[
                video_input,
                tracker_input,
                confidence_input,
                expected_people_input,
                formation_grid_width_input,
                formation_grid_height_input,
                calibration_points_state,
            ],
            outputs=[
                preview_output,
                tracks_output,
                raw_tracks_output,
                formations_output,
                calibration_output,
                log_output,
                status_output,
            ],
        )
        analysis_event.then(
            fn=load_formation_editor,
            inputs=[formations_output],
            outputs=[formation_editor_state, *editor_outputs],
        )
    return demo


if __name__ == "__main__":
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    app = build_app()
    app.queue().launch(server_name="0.0.0.0", show_error=True)
