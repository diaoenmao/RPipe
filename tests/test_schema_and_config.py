"""Lightweight unit tests for config + artifact schemas."""

from __future__ import annotations

import copy

import pytest

from rpipe.config import (
    ControlConfig,
    ExperimentConfig,
    TrainConfig,
    apply_control_name,
    build_runtime_cfg,
    experiment_from_mapping,
)
from rpipe.config.runtime import ModelRuntime, OptimizerRuntime, RuntimeConfig
from rpipe.schema import (
    RESULT_BLOB_SCHEMA,
    RUN_MANIFEST_SCHEMA,
    assert_valid_result_blob,
    assert_valid_run_manifest,
    validate_result_blob,
    validate_run_manifest,
)


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
    assert runtime.pin_memory is False  # forced off on cpu
    assert runtime.num_steps == 4
    assert runtime.optimizer.batch_size['train'] == 64
    assert runtime.optimizer.batch_size['test'] == 64 * 4
    assert runtime.model is not None
    assert runtime.model.model_name == 'linear'


def test_apply_control_and_hyper_overrides():
    exp = experiment_from_mapping(
        {'control': {'data_name': 'CIFAR10', 'model_name': 'cnn'}, 'device': 'cpu'},
        hyper={'num_steps': 8, 'batch_size': 100},
    )
    exp = apply_control_name(exp, 'CIFAR10_resnet18')
    assert exp.control_name == 'CIFAR10_resnet18'
    runtime = build_runtime_cfg(exp, seed=1)
    assert runtime.control_name == 'CIFAR10_resnet18'
    assert runtime.model_name == 'resnet18'
    assert runtime.num_steps == 8
    assert runtime.batch_size == 100


def test_runtime_roundtrip_dict():
    runtime = RuntimeConfig(
        control_name='MNIST_linear',
        data_name='MNIST',
        model_name='linear',
        tag='0_MNIST_linear',
        seed=0,
        step=2,
        model=ModelRuntime(data_name='MNIST', model_name='linear', data_size=(1, 28, 28), target_size=10),
        optimizer=OptimizerRuntime(batch_size={'train': 250, 'test': 1000}),
    )
    restored = RuntimeConfig.from_dict(runtime.to_dict())
    assert restored.tag == runtime.tag
    assert restored.step == 2
    assert restored.model.data_size == (1, 28, 28)
    assert restored.optimizer.batch_size['train'] == 250


def _valid_manifest():
    return {
        'schema': 'rpipe.run_manifest.v1',
        'generated_at': '2026-08-08T21:00:00',
        'suite': {
            'name': 'smoke',
            'description': 'x',
            'data_names': ['MNIST'],
            'model_names': ['linear'],
            'num_experiments': 1,
            'init_seed': 0,
            'hyper': {},
        },
        'run': {'stages': ['prepare'], 'device': 'cpu', 'cwd': '/tmp'},
        'artifacts': {
            'result_paths': [],
            'plot_paths': [],
            'excel': [],
            'processed_result': 'output/result/processed_result',
            'stats_dir': 'output/stats',
            'exp_dir': 'output/exp',
        },
        'notes': [],
        'report_hint': 'hint',
    }


def _valid_result_blob():
    return {
        'schema': 'rpipe.result_blob.v1',
        'cfg': {
            'tag': '0_MNIST_linear',
            'control_name': 'MNIST_linear',
            'data_name': 'MNIST',
            'model_name': 'linear',
            'seed': 0,
        },
        'logger': {
            'train': {'mean': {}, 'history': {}},
            'test': {'mean': {'test/Accuracy': 70.0}, 'history': {}},
        },
    }


def test_manifest_schema_accepts_valid():
    assert validate_run_manifest(_valid_manifest()) == []
    assert_valid_run_manifest(_valid_manifest())


def test_manifest_schema_rejects_missing_suite_name():
    bad = copy.deepcopy(_valid_manifest())
    del bad['suite']['name']
    errors = validate_run_manifest(bad)
    assert any('name' in e for e in errors)
    with pytest.raises(ValueError):
        assert_valid_run_manifest(bad)


def test_result_blob_schema_accepts_valid():
    assert validate_result_blob(_valid_result_blob()) == []
    assert_valid_result_blob(_valid_result_blob())


def test_result_blob_schema_rejects_wrong_schema_id():
    bad = copy.deepcopy(_valid_result_blob())
    bad['schema'] = 'nope'
    errors = validate_result_blob(bad)
    assert errors
    with pytest.raises(ValueError):
        assert_valid_result_blob(bad)


def test_schema_ids_stable():
    assert RESULT_BLOB_SCHEMA['$id'] == 'rpipe.result_blob.v1'
    assert RUN_MANIFEST_SCHEMA['$id'] == 'rpipe.run_manifest.v1'
