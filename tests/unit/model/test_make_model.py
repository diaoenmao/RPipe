"""Unit tests for model builders (native linear)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

from rpipe.model import linear as _linear  # noqa: F401
from rpipe.model.model import make_model


def _linear_cfg():
    return {
        'data_name': 'MNIST',
        'model_name': 'linear',
        'data_size': [1, 28, 28],
        'target_size': 10,
        'stats': SimpleNamespace(
            mean=torch.tensor([0.0]),
            std=torch.tensor([1.0]),
        ),
        'linear': {},
    }


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.model_layer
def test_make_linear_model_forward_shape(device):
    model = make_model(_linear_cfg()).to(device)
    model.eval()
    x = torch.randn(4, 1, 28, 28, device=device)
    y = torch.zeros(4, dtype=torch.long, device=device)
    out = model(data=x, target=y)
    assert out['pred'].shape == (4, 10)
    assert 'loss' in out


@pytest.mark.unit
@pytest.mark.physical
@pytest.mark.runtime
@pytest.mark.p2
@pytest.mark.model_layer
@pytest.mark.gpu
def test_linear_forward_runtime_budget(device):
    model = make_model(_linear_cfg()).to(device).eval()
    x = torch.randn(64, 1, 28, 28, device=device)
    y = torch.zeros(64, dtype=torch.long, device=device)
    with torch.no_grad():
        model(data=x, target=y)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            model(data=x, target=y)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f'forward too slow: {elapsed:.3f}s on {device}'
