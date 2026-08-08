"""Integration: config → providers → Trainer."""

from __future__ import annotations

import pytest

from rpipe.config import ControlConfig, ExperimentConfig, TrainConfig, build_runtime_cfg
from rpipe.provider import get_algorithm, get_provider
from rpipe.system.backend import Trainer, NativeTrainer


@pytest.mark.integration
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.config_layer
@pytest.mark.plugins_layer
@pytest.mark.system_layer
@pytest.mark.module_trainer
@pytest.mark.feature_providers
def test_runtime_wires_train_metric_system():
    exp = ExperimentConfig(
        control=ControlConfig('MNIST', 'linear'),
        device='cpu',
        data_provider='native',
        model_provider='native',
        train_algorithm='native',
        metric_algorithm='native',
        system_provider='pytorch',
        train=TrainConfig(num_steps=2, eval_period=2, batch_size=32),
    )
    runtime = build_runtime_cfg(exp, seed=0)
    assert get_provider('data', runtime.data_provider).name == 'native'
    assert get_algorithm('train', runtime.train_algorithm).algorithm_type == 'train'
    assert get_algorithm('metric', runtime.metric_algorithm).algorithm_type == 'metric'
    assert get_provider('system', runtime.system_provider).tensor_lib == 'pytorch'
    trainer = Trainer(runtime)
    assert isinstance(trainer, NativeTrainer)


@pytest.mark.integration
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.data_layer
@pytest.mark.model_layer
def test_native_data_and_model_build_chain():
    from types import SimpleNamespace
    data_p = get_provider('data', 'native')
    model_p = get_provider('model', 'native')
    cfg = {
        'data_name': 'MNIST',
        'model_name': 'linear',
        'data_size': [1, 28, 28],
        'target_size': 10,
        'stats': SimpleNamespace(
            mean=__import__('torch').tensor([0.0]),
            std=__import__('torch').tensor([1.0]),
        ),
    }
    assert model_p.build(cfg) is not None
    assert data_p.available()
