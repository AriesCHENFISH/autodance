"""人物检测与在线追踪封装。"""

from pathlib import Path
from typing import Any

import numpy as np

from models import load_pose_model


class PersonDetector:
    """使用 YOLO Pose 内置的 ByteTrack 或 BoT-SORT 完成人物检测与追踪。"""

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    TRACKER_CONFIGS = {
        "Dance BoT-SORT": str(
            _PROJECT_ROOT / "trackers" / "dance_botsort.yaml"
        ),
        "ByteTrack": "bytetrack.yaml",
        "BoT-SORT": "botsort.yaml",
    }

    def __init__(
        self,
        model_name: str = "yolo11s-pose.pt",
        tracker_name: str = "Dance BoT-SORT",
        confidence: float = 0.10,
    ) -> None:
        """初始化检测器并校验追踪器名称。"""

        if tracker_name not in self.TRACKER_CONFIGS:
            raise ValueError(f"不支持的追踪器：{tracker_name}")
        self.model = load_pose_model(model_name)
        self.tracker_config = self.TRACKER_CONFIGS[tracker_name]
        self.confidence = confidence

    def process_frame(self, frame: np.ndarray) -> Any:
        """处理单帧图像，返回包含追踪 ID、检测框和关键点的结果。"""

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            classes=[0],
            conf=self.confidence,
            verbose=False,
        )
        if not results:
            raise RuntimeError("模型未返回推理结果")
        return results[0]
