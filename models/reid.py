"""基于 OSNet 的人物重识别特征提取器。

对人物裁剪图提取 512 维外观嵌入，用于跨关键帧匹配同一身份。模型权重
首次使用时从 Google Drive 下载并缓存到本机。
"""

from __future__ import annotations

from pathlib import Path
import os

import cv2
import numpy as np
import torch

from .osnet import osnet_x1_0


_IMAGE_HEIGHT = 256
_IMAGE_WIDTH = 128
_PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_PIXEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_WEIGHTS_BY_NAME = {
    "msmt17": (
        "osnet_x1_0_msmt17.pt",
        "https://drive.google.com/uc?id=1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x",
    ),
    "market1501": (
        "osnet_x1_0_market1501.pt",
        "https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA",
    ),
}


def _default_cache_dir() -> Path:
    """返回权重缓存目录，遵循 ``TORCH_HOME``，默认 ``~/.cache/torch/checkpoints``。"""
    torch_home = os.environ.get(
        "TORCH_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "torch"),
    )
    return Path(torch_home) / "checkpoints"


def _load_state_dict(
    model: torch.nn.Module, path: Path
) -> None:
    """加载权重，忽略名称或尺寸不匹配的层（例如分类头）。"""

    checkpoint = torch.load(str(path), map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_dict = model.state_dict()
    filtered = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[7:]
        if key in model_dict and model_dict[key].size() == value.size():
            filtered[key] = value
    if not filtered:
        raise RuntimeError(f"权重文件 {path} 与模型结构不匹配")
    model_dict.update(filtered)
    model.load_state_dict(model_dict)


def _preprocess(crops: list[np.ndarray]) -> torch.Tensor:
    """将 BGR 人物裁剪图批量转换为 OSNet 输入张量。"""

    batch = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (_IMAGE_WIDTH, _IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR
        )
        normalized = (resized.astype(np.float32) / 255.0 - _PIXEL_MEAN) / _PIXEL_STD
        batch.append(normalized)
    tensor = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2)
    return tensor


def _normalize_rows(features: torch.Tensor) -> np.ndarray:
    """对特征矩阵按行做 L2 归一化并转为 NumPy。"""

    normalized = torch.nn.functional.normalize(features, p=2, dim=1)
    return normalized.detach().cpu().numpy()


class ReIDExtractor:
    """OSNet 外观嵌入提取器。"""

    def __init__(
        self,
        weights_name: str = "msmt17",
        device: str | None = None,
    ) -> None:
        if weights_name not in _WEIGHTS_BY_NAME:
            raise ValueError(f"不支持的 ReID 权重：{weights_name}")
        self.weights_name = weights_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = osnet_x1_0(num_classes=1, pretrained=False)
        self._load_weights(model, weights_name)
        model.eval()
        model.to(self.device)
        self.model = model

    def _load_weights(self, model: torch.nn.Module, weights_name: str) -> None:
        filename, url = _WEIGHTS_BY_NAME[weights_name]
        cache_dir = _default_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / filename
        if not path.exists():
            try:
                import gdown

                gdown.download(url, str(path), quiet=False)
            except ImportError as error:
                raise RuntimeError(
                    "下载 OSNet 权重需要 gdown，请先运行 pip install gdown"
                ) from error
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"OSNet 权重下载失败：{path}")
        _load_state_dict(model, path)

    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """提取裁剪图的外观嵌入，返回 ``[N, 512]`` 的 L2 归一化特征。"""

        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        with torch.no_grad():
            tensor = _preprocess(crops).to(self.device)
            features = self.model(tensor)
        return _normalize_rows(features)


_default_extractor: ReIDExtractor | None = None


def get_reid_extractor(weights_name: str = "msmt17") -> ReIDExtractor:
    """返回进程级单例 ReID 提取器，避免重复加载权重。"""

    global _default_extractor
    if _default_extractor is None or _default_extractor.weights_name != weights_name:
        _default_extractor = ReIDExtractor(weights_name=weights_name)
    return _default_extractor
