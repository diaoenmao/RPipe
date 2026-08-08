"""Unit tests for system trainer factory."""

from __future__ import annotations

import pytest

from rpipe.config import ControlConfig, ExperimentConfig, TrainConfig, build_runtime_cfg
from rpipe.system.backend import Trainer, NativeTrainer


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.system_layer
@pytest.mark.module_trainer
def test_trainer_dispatches_pytorch_native_train():
    exp = ExperimentConfig(
        control=ControlConfig('MNIST', 'linear'),
        device='cpu',
        train_algorithm='native',
        system_provider='pytorch',
        train=TrainConfig(num_steps=2, eval_period=2),
    )
    runtime = build_runtime_cfg(exp, 0)
    assert isinstance(Trainer(runtime), NativeTrainer)


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p2
@pytest.mark.system_layer
def test_legacy_trainer_backend_maps_to_pytorch_native_train():
    exp = ExperimentConfig(
        control=ControlConfig('MNIST', 'linear'),
        device='cpu',
        trainer_backend='native',
        train=TrainConfig(num_steps=2, eval_period=2),
    )
    runtime = build_runtime_cfg(exp, 0)
    assert runtime.system_provider == 'pytorch'
    assert runtime.train_algorithm == 'native'
    assert isinstance(Trainer(runtime), NativeTrainer)
