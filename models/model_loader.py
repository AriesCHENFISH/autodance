"""Ultralytics YOLO Pose 模型加载工具。"""

from ultralytics import YOLO


def load_pose_model(model_name: str = "yolo11n-pose.pt") -> YOLO:
    """加载独立的 YOLO Pose 实例，权重文件不存在时自动下载并缓存到本机。"""

    return YOLO(model_name)
