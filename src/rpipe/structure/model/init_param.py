"""Weight init matching git main ``src/model/model.py`` ``init_param``."""

from __future__ import annotations

from typing import Any


def init_param(m: Any) -> Any:
    import torch.nn as nn

    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            m.bias.data.zero_()
    return m
