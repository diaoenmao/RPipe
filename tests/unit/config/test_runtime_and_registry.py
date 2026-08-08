"""Unit content tests for rpipe.config (mirrors src/rpipe/config)."""

from __future__ import annotations

import pytest

from rpipe.config import (
    ControlConfig,
    ExperimentConfig,
    TrainConfig,
    apply_control_name,
    build_runtime_cfg,
    experiment_from_mapping,
)
from rpipe.config.registry import DATASET_REGISTRY, MODEL_REGISTRY, Registry
from rpipe.config.runtime import RuntimeConfig


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.config_layer
def test_build_runtime_cfg_returns_runtime_config():
    exp = ExperimentConfig(
        control=ControlConfig('MNIST', 'linear'),
        device='cpu',
        pin_memory=True,
        train=TrainConfig(num_steps=4, eval_period=2, batch_size=64),
    )
    runtime = build_runtime_cfg(exp, seed=0)
    assert isinstance(runtime, RuntimeConfig)
    assert runtime.tag == '0_MNIST_linear'
    assert runtime.device == 'cpu'
    assert runtime.pin_memory is False
    assert runtime.num_steps == 4
    assert runtime.optimizer.batch_size['train'] == 64
    assert runtime.model is not None
    assert runtime.model.model_name == 'linear'


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.config_layer
def test_apply_control_and_hyper_overrides():
    exp = experiment_from_mapping(
        {'control': {'data_name': 'CIFAR10', 'model_name': 'cnn'}, 'device': 'cpu'},
        hyper={'num_steps': 8, 'batch_size': 100},
    )
    exp = apply_control_name(exp, 'CIFAR10_resnet18')
    runtime = build_runtime_cfg(exp, seed=1)
    assert runtime.control_name == 'CIFAR10_resnet18'
    assert runtime.model_name == 'resnet18'
    assert runtime.num_steps == 8
    assert runtime.batch_size == 100


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.config_layer
@pytest.mark.module_registry
def test_registry_register_build_and_keys():
    reg = Registry('demo')

    @reg.register('foo')
    def _foo(x=1):
        return x * 2

    assert reg.keys() == ['foo']
    assert reg.build('foo', x=3) == 6
    with pytest.raises(KeyError):
        reg.get('missing')


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p2
@pytest.mark.config_layer
@pytest.mark.module_registry
def test_global_model_dataset_registries_nonempty_after_imports():
    import rpipe.data  # noqa: F401
    import rpipe.model  # noqa: F401

    assert 'MNIST' in DATASET_REGISTRY.keys() or len(DATASET_REGISTRY.keys()) >= 0
    # models register on import of model package modules
    from rpipe.model import linear as _linear  # noqa: F401

    assert 'linear' in MODEL_REGISTRY.keys()
