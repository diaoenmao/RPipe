"""custom_torch models from git main: linear, mlp, cnn, resnet10/18."""

from __future__ import annotations

import math

import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, data_size: tuple[int, ...], target_size: int) -> None:
        super().__init__()
        self.output_proj = nn.Linear(math.prod(data_size), target_size)

    def forward(self, x):  # noqa: ANN001
        x = x.reshape(x.size(0), -1)
        return self.output_proj(x)


class MLP(nn.Module):
    def __init__(
        self,
        data_size: tuple[int, ...],
        hidden_size: int,
        scale_factor: float,
        num_layers: int,
        activation: str,
        target_size: int,
    ) -> None:
        super().__init__()
        input_size = math.prod(data_size)
        blocks: list[nn.Module] = []
        width = int(hidden_size)
        for _ in range(int(num_layers)):
            blocks.append(nn.Linear(input_size, width))
            if activation == 'relu':
                blocks.append(nn.ReLU())
            elif activation == 'sigmoid':
                blocks.append(nn.Sigmoid())
            else:
                raise ValueError(f'unknown mlp activation: {activation}')
            input_size = width
            width = int(width * scale_factor)
        self.blocks = nn.Sequential(*blocks)
        self.output_proj = nn.Linear(input_size, target_size)

    def forward(self, x):  # noqa: ANN001
        x = x.reshape(x.size(0), -1)
        return self.output_proj(self.blocks(x))


class CNN(nn.Module):
    def __init__(self, data_size: tuple[int, ...], hidden_size: list[int], target_size: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = [
            nn.Conv2d(data_size[0], hidden_size[0], 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        for i in range(len(hidden_size) - 1):
            blocks.extend(
                [
                    nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                ]
            )
        blocks = blocks[:-1]
        blocks.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten()])
        self.blocks = nn.Sequential(*blocks)
        self.output_proj = nn.Linear(hidden_size[-1], target_size)

    def forward(self, x):  # noqa: ANN001
        return self.output_proj(self.blocks(x))


class _ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int) -> None:
        super().__init__()
        self.n1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(
                in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x):  # noqa: ANN001
        out = F.relu(self.n1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        data_size: tuple[int, ...],
        hidden_size: list[int],
        num_blocks: list[int],
        target_size: int,
        block: type[_ResBlock] = _ResBlock,
    ) -> None:
        super().__init__()
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(data_size[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2)
        self.n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion)
        self.output_proj = nn.Linear(hidden_size[3] * block.expansion, target_size)

    def _make_layer(self, block: type[_ResBlock], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for layer_stride in strides:
            layers.append(block(self.in_planes, planes, layer_stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):  # noqa: ANN001
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(self.n4(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.output_proj(x)
