"""Configuration loaders and typed experiment / runtime configs.

No process-global ``cfg`` bag — callers pass ``ExperimentConfig`` / ``RuntimeConfig``.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rpipe.config.runtime import ModelRuntime, OptimizerRuntime, RuntimeConfig

__all__ = [
    'ControlConfig',
    'TrainConfig',
    'ExperimentConfig',
    'ModelRuntime',
    'OptimizerRuntime',
    'RuntimeConfig',
    'package_root',
    'repo_root',
    'default_config_path',
    'load_yaml',
    'load_default_dict',
    'experiment_from_mapping',
    'apply_control_name',
    'build_runtime_cfg',
]


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / 'default.yaml'


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader) or {}


def load_default_dict(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or default_config_path())


@dataclass
class ControlConfig:
    data_name: str = 'MNIST'
    model_name: str = 'linear'

    def name(self) -> str:
        return f'{self.data_name}_{self.model_name}'


@dataclass
class TrainConfig:
    batch_size: int = 250
    step_period: int = 1
    num_steps: int = 60
    eval_period: int = 30
    save_period: int = 30
    save_checkpoint: bool = True
    optimizer_name: str = 'SGD'
    lr: float = 1e-1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    scheduler_name: str = 'CosineAnnealingLR'
    test_batch_ratio: int = 4


@dataclass
class ExperimentConfig:
    control: ControlConfig = field(default_factory=ControlConfig)
    pin_memory: bool = True
    num_workers: int = 0
    init_seed: int = 0
    num_experiments: int = 1
    log_interval: float = 0.25
    device: str = 'cuda'
    resume_mode: int = 0
    profile: bool = False
    output_root: str = 'output'
    train: TrainConfig = field(default_factory=TrainConfig)
    hyper_overrides: dict[str, Any] = field(default_factory=dict)
    data_provider: str = 'native'
    model_provider: str = 'native'
    train_algorithm: str = 'native'
    metric_algorithm: str = 'native'
    generate_algorithm: str | None = None
    system_provider: str = 'pytorch'
    mixed_precision: str | None = None
    algorithm_provider: str | None = None
    pytorch_accelerator: str | None = None
    metric_provider: str | None = None
    trainer_backend: str | None = None

    @property
    def control_name(self) -> str:
        return self.control.name()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d['control_name'] = self.control_name
        return d


def experiment_from_mapping(raw: dict[str, Any], hyper: dict | None = None) -> ExperimentConfig:
    control_raw = raw.get('control', {})
    control = ControlConfig(
        data_name=control_raw.get('data_name', 'MNIST'),
        model_name=control_raw.get('model_name', 'linear'),
    )
    train = TrainConfig()
    hyper = hyper or {}
    for key, value in hyper.items():
        if hasattr(train, key):
            setattr(train, key, value)
    return ExperimentConfig(
        control=control,
        pin_memory=bool(raw.get('pin_memory', True)),
        num_workers=int(raw.get('num_workers', 0)),
        init_seed=int(raw.get('init_seed', 0)),
        num_experiments=int(raw.get('num_experiments', 1)),
        log_interval=float(raw.get('log_interval', 0.25)),
        device=str(raw.get('device', 'cuda')),
        resume_mode=int(raw.get('resume_mode', 0)),
        profile=bool(raw.get('profile', False)),
        output_root=str(raw.get('output_root', 'output')),
        train=train,
        hyper_overrides=dict(hyper),
        data_provider=str(raw.get('data_provider', hyper.get('data_provider', 'native'))),
        model_provider=str(raw.get('model_provider', hyper.get('model_provider', 'native'))),
        train_algorithm=str(
            raw.get(
                'train_algorithm',
                hyper.get(
                    'train_algorithm',
                    _legacy_train(
                        raw.get('trainer_backend', hyper.get('trainer_backend')),
                        raw.get('pytorch_accelerator', hyper.get('pytorch_accelerator')),
                        raw.get('algorithm_provider', hyper.get('algorithm_provider')),
                    ),
                ),
            )
        ),
        metric_algorithm=str(
            raw.get(
                'metric_algorithm',
                hyper.get(
                    'metric_algorithm',
                    raw.get('metric_provider', hyper.get('metric_provider', 'native')),
                ),
            )
        ),
        generate_algorithm=raw.get(
            'generate_algorithm',
            hyper.get(
                'generate_algorithm',
                _legacy_generate(
                    raw.get('trainer_backend', hyper.get('trainer_backend')),
                    raw.get('algorithm_provider', hyper.get('algorithm_provider')),
                ),
            ),
        ),
        system_provider=str(
            raw.get(
                'system_provider',
                hyper.get(
                    'system_provider',
                    _legacy_system(raw.get('trainer_backend', hyper.get('trainer_backend'))),
                ),
            )
        ),
        mixed_precision=raw.get('mixed_precision', hyper.get('mixed_precision')),
        algorithm_provider=raw.get('algorithm_provider', hyper.get('algorithm_provider')),
        pytorch_accelerator=raw.get('pytorch_accelerator', hyper.get('pytorch_accelerator')),
        metric_provider=raw.get('metric_provider', hyper.get('metric_provider')),
        trainer_backend=raw.get('trainer_backend', hyper.get('trainer_backend')),
    )


def _legacy_system(trainer_backend: str | None) -> str:
    if trainer_backend in (None, '', 'native', 'accelerate', 'diffusers', 'pytorch'):
        return 'pytorch'
    if trainer_backend in ('llama_cpp', 'ggml'):
        return 'ggml'
    return 'pytorch'


def _legacy_train(trainer_backend, pytorch_accelerator, algorithm_provider) -> str:
    if algorithm_provider == 'accelerate' or pytorch_accelerator == 'accelerate' or trainer_backend == 'accelerate':
        return 'accelerate'
    return 'native'


def _legacy_generate(trainer_backend, algorithm_provider):
    if trainer_backend == 'llama_cpp' or algorithm_provider == 'llama_cpp':
        return 'llama_cpp'
    if trainer_backend == 'diffusers' or algorithm_provider == 'diffusers':
        return 'diffusers'
    return None


def apply_control_name(exp: ExperimentConfig, control_name: str) -> ExperimentConfig:
    data_name, model_name = control_name.split('_', 1)
    exp = copy.deepcopy(exp)
    exp.control = ControlConfig(data_name=data_name, model_name=model_name)
    return exp


def build_runtime_cfg(exp: ExperimentConfig, seed: int) -> RuntimeConfig:
    """Build an explicit ``RuntimeConfig`` for one (seed, control) run."""
    from rpipe.system.stats import make_stats

    train = exp.train
    for key, value in (exp.hyper_overrides or {}).items():
        if hasattr(train, key):
            setattr(train, key, value)

    tag = f'{seed}_{exp.control_name}'
    model = ModelRuntime(
        data_name=exp.control.data_name,
        model_name=exp.control.model_name,
        stats=make_stats(exp.control.data_name, output_root=exp.output_root),
    )
    optimizer = OptimizerRuntime(
        optimizer_name=train.optimizer_name,
        lr=train.lr,
        momentum=train.momentum,
        weight_decay=train.weight_decay,
        nesterov=train.nesterov,
        test_batch_ratio=train.test_batch_ratio,
        batch_size={
            'train': train.batch_size,
            'test': train.test_batch_ratio * train.batch_size,
        },
        step_period=train.step_period,
        num_steps=train.num_steps,
        scheduler_name=train.scheduler_name,
    )
    return RuntimeConfig(
        control_name=exp.control_name,
        data_name=exp.control.data_name,
        model_name=exp.control.model_name,
        tag=tag,
        seed=seed,
        device=exp.device,
        output_root=exp.output_root,
        pin_memory=bool(exp.pin_memory and exp.device != 'cpu'),
        num_workers=exp.num_workers,
        log_interval=exp.log_interval,
        resume_mode=exp.resume_mode,
        profile=exp.profile,
        batch_size=train.batch_size,
        step_period=train.step_period,
        num_steps=train.num_steps,
        eval_period=train.eval_period,
        save_period=train.save_period,
        save_checkpoint=train.save_checkpoint,
        log={
            'tensorboard': True,
            'profile': exp.profile,
            'schedule': {'wait': 1, 'warmup': 4, 'active': 8, 'repeat': 1},
        },
        model=model,
        optimizer=optimizer,
        data_provider=exp.data_provider,
        model_provider=exp.model_provider,
        train_algorithm=exp.train_algorithm,
        metric_algorithm=exp.metric_algorithm,
        generate_algorithm=exp.generate_algorithm,
        system_provider=exp.system_provider,
        mixed_precision=exp.mixed_precision,
        algorithm_provider=exp.algorithm_provider,
        pytorch_accelerator=exp.pytorch_accelerator,
        metric_provider=exp.metric_provider,
        trainer_backend=exp.trainer_backend,
    )
