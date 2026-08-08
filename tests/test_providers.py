"""Tests for the 4-layer provider plugin API."""

from __future__ import annotations

import rpipe.plugins  # noqa: F401
from rpipe.plugins import list_providers
from rpipe.plugins.api import PROVIDERS, get_provider, load_builtin_providers


def test_builtin_providers_registered():
    providers = list_providers()
    assert providers['data'] == sorted(['datasets', 'native']) or set(providers['data']) == {
        'datasets',
        'native',
    }
    assert 'native' in providers['data'] and 'datasets' in providers['data']
    assert set(providers['data']) == {'native', 'datasets'}

    assert set(providers['model']) == {
        'native',
        'timm',
        'transformers',
        'modelscope',
        'peft',
        'ollama',
    }
    assert set(providers['algorithm']) == {
        'native',
        'torchmetrics',
        'evaluate',
        'lm_eval',
        'opencompass',
    }
    assert set(providers['system']) == {
        'native',
        'accelerate',
        'llama_cpp',
        'diffusers',
    }


def test_provider_package_names():
    load_builtin_providers()
    assert PROVIDERS['data'].build('datasets').package == 'datasets'
    assert PROVIDERS['model'].build('modelscope').package == 'modelscope'
    assert PROVIDERS['model'].build('ollama').package == 'ollama'
    assert PROVIDERS['algorithm'].build('lm_eval').package == 'lm_eval'
    assert PROVIDERS['algorithm'].build('evaluate').package == 'evaluate'
    assert PROVIDERS['system'].build('llama_cpp').package == 'llama-cpp-python'
    assert PROVIDERS['system'].build('diffusers').package == 'diffusers'


def test_metric_modes():
    assert get_provider('algorithm', 'native').mode == 'online'
    load_builtin_providers()
    assert PROVIDERS['algorithm'].build('lm_eval').mode == 'benchmark'
    assert PROVIDERS['algorithm'].build('torchmetrics').mode == 'online'


def test_native_providers_available():
    assert get_provider('data', 'native').available()
    assert get_provider('model', 'native').available()
    assert get_provider('algorithm', 'native').available()
    assert get_provider('system', 'native').available()


def test_trainer_proxy_native():
    from rpipe.config import ExperimentConfig, ControlConfig, TrainConfig, build_runtime_cfg
    from rpipe.system.backend import Trainer, NativeTrainer

    exp = ExperimentConfig(
        control=ControlConfig('MNIST', 'linear'),
        device='cpu',
        trainer_backend='native',
        train=TrainConfig(num_steps=2, eval_period=2),
    )
    runtime = build_runtime_cfg(exp, 0)
    trainer = Trainer(runtime)
    assert isinstance(trainer, NativeTrainer)
