"""End-to-end: smoke train path via experiments ResearchPipeline."""

from __future__ import annotations

import os

import pytest
import torch

from experiments.runner import ResearchPipeline


@pytest.mark.e2e
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.application_layer
@pytest.mark.system_layer
@pytest.mark.data_layer
@pytest.mark.model_layer
@pytest.mark.algorithm_layer
@pytest.mark.feature_smoke
@pytest.mark.slow
@pytest.mark.gpu
def test_e2e_smoke_pipeline_prepare_and_train(device, tmp_path):
    """Full user path: prepare stats → train smoke suite (GPU if available)."""
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    out = tmp_path / 'output'
    pipe = ResearchPipeline(
        'smoke',
        device=str(device),
        stages=['prepare', 'train'],
        force_prepare=True,
        output_root=str(out),
    )
    # shorten further via suite hyper already (4 steps); still e2e
    result = pipe.run()
    assert result['suite'] == 'smoke'
    assert (out / 'stats' / 'MNIST').exists() or list((out / 'stats').glob('*'))
    assert (out / 'exp').exists()


@pytest.mark.e2e
@pytest.mark.physical
@pytest.mark.runtime
@pytest.mark.p2
@pytest.mark.application_layer
@pytest.mark.gpu
def test_e2e_device_selection_reported(device, cuda_available):
    """Document runtime device for the report; prefer CUDA when present."""
    assert device.type in ('cpu', 'cuda')
    if cuda_available:
        assert device.type == 'cuda'
        assert torch.zeros(1, device=device).is_cuda
    else:
        assert device.type == 'cpu'
