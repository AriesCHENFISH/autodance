"""OSNet（Omni-Scale Network）人物重识别模型结构。

仅包含模型前向结构，权重加载与下载逻辑见 ``models.reid``。

来源：Kaiyang Zhou 等人 ``deep-person-reid``（OSNet, ICCV 2019），MIT 许可。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["osnet_x1_0"]


class ConvLayer(nn.Module):
    """卷积层（conv + bn + relu）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        IN: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        if IN:
            self.bn = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 卷积 + bn + relu。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 卷积 + bn（无非线性）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3(nn.Module):
    """3x3 卷积 + bn + relu。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3(nn.Module):
    """轻量 3x3 卷积：1x1（线性）+ depthwise 3x3（非线性）。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelGate(nn.Module):
    """根据输入张量生成逐通道门控的小网络。"""

    def __init__(
        self,
        in_channels: int,
        num_gates: int | None = None,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        if num_gates is None:
            num_gates = in_channels
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0,
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0,
        )
        self.gate_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.gate_activation(x)
        return identity * x


class OSBlock(nn.Module):
    """Omni-scale 特征学习模块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        IN: bool = False,
        bottleneck_reduction: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


class OSNet(nn.Module):
    """Omni-Scale Network。

    Reference:
        Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
    """

    def __init__(
        self,
        num_classes: int,
        blocks: list[type[nn.Module]],
        layers: list[int],
        channels: list[int],
        feature_dim: int = 512,
        loss: str = "softmax",
        IN: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            blocks[0],
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            IN=IN,
        )
        self.conv3 = self._make_layer(
            blocks[1],
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True,
        )
        self.conv4 = self._make_layer(
            blocks[2],
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False,
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(
            self.feature_dim, channels[3], dropout_p=None
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(
        self,
        block: type[nn.Module],
        layer: int,
        in_channels: int,
        out_channels: int,
        reduce_spatial_size: bool,
        IN: bool = False,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [block(in_channels, out_channels, IN=IN)]
        for _ in range(1, layer):
            layers.append(block(out_channels, out_channels, IN=IN))

        if reduce_spatial_size:
            layers.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2),
                )
            )
        return nn.Sequential(*layers)

    def _construct_fc_layer(
        self,
        fc_dims: int | None,
        input_dim: int,
        dropout_p: float | None = None,
    ) -> nn.Sequential | None:
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, fc_dims))
        layers.append(nn.BatchNorm1d(fc_dims))
        layers.append(nn.ReLU(inplace=True))
        if dropout_p is not None:
            layers.append(nn.Dropout(p=dropout_p))
        self.feature_dim = fc_dims
        return nn.Sequential(*layers)

    def _init_params(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def featuremaps(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(
        self, x: torch.Tensor, return_featuremaps: bool = False
    ) -> torch.Tensor:
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == "softmax":
            return y
        if self.loss == "triplet":
            return y, v
        raise KeyError(f"Unsupported loss: {self.loss}")


def osnet_x1_0(num_classes: int = 1000, pretrained: bool = True, **kwargs) -> OSNet:
    """标准尺寸（width x1.0）OSNet。``pretrained`` 仅用于兼容，权重在
    :class:`models.reid.ReIDExtractor` 中加载。"""
    return OSNet(
        num_classes,
        blocks=[OSBlock, OSBlock, OSBlock],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        **kwargs,
    )
