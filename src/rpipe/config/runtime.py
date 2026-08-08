"""Explicit runtime configuration objects (no global cfg)."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any


def _default_arch() -> dict[str, Any]:
    return {
        'linear': {},
        'mlp': {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'},
        'cnn': {'hidden_size': [64, 128, 256, 512]},
        'resnet10': {'hidden_size': [64, 128, 256, 512]},
        'resnet18': {'hidden_size': [64, 128, 256, 512]},
        'wresnet28x2': {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0},
        'wresnet28x8': {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0},
    }


@dataclass
class ModelRuntime:
    data_name: str
    model_name: str
    data_size: Any = None
    target_size: Any = None
    stats: Any = None
    arch: dict[str, Any] = field(default_factory=_default_arch)

    def as_build_dict(self) -> dict[str, Any]:
        """Dict consumed by ``make_model`` / registry factories."""
        out = {
            'data_name': self.data_name,
            'model_name': self.model_name,
            'data_size': self.data_size,
            'target_size': self.target_size,
            'stats': self.stats,
        }
        out.update(copy.deepcopy(self.arch))
        return out


@dataclass
class OptimizerRuntime:
    optimizer_name: str = 'SGD'
    lr: float = 1e-1
    momentum: float = 0.9
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 5e-4
    nesterov: bool = True
    test_batch_ratio: int = 4
    batch_size: dict[str, int] = field(default_factory=lambda: {'train': 250, 'test': 1000})
    step_period: int = 1
    num_steps: int = 60
    scheduler_name: str = 'CosineAnnealingLR'

    def as_kwargs(self) -> dict[str, Any]:
        return {
            'optimizer_name': self.optimizer_name,
            'lr': self.lr,
            'momentum': self.momentum,
            'betas': self.betas,
            'weight_decay': self.weight_decay,
            'nesterov': self.nesterov,
            'test_batch_ratio': self.test_batch_ratio,
            'batch_size': dict(self.batch_size),
            'step_period': self.step_period,
            'num_steps': self.num_steps,
            'scheduler_name': self.scheduler_name,
        }


@dataclass
class RuntimeConfig:
    """Per-run config passed explicitly into data / backend."""

    control_name: str
    data_name: str
    model_name: str
    tag: str
    seed: int
    device: str = 'cpu'
    output_root: str = 'output'
    pin_memory: bool = False
    num_workers: int = 0
    log_interval: float = 0.25
    resume_mode: int = 0
    profile: bool = False
    batch_size: int = 250
    step_period: int = 1
    num_steps: int = 60
    eval_period: int = 30
    save_period: int = 30
    save_checkpoint: bool = True
    collate_mode: str = 'dict'
    num_epochs: int | None = None
    eval_num_steps: int = -1
    step: int = 0
    num_samples: dict[str, int] | None = None
    model: ModelRuntime | None = None
    optimizer: OptimizerRuntime = field(default_factory=OptimizerRuntime)
    log: dict[str, Any] = field(default_factory=lambda: {
        'tensorboard': True,
        'profile': False,
        'schedule': {'wait': 1, 'warmup': 4, 'active': 8, 'repeat': 1},
    })
    metric: dict[str, Any] = field(default_factory=lambda: {
        'metric_name': {'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']},
        'best_split': 'test',
        'best_metric_name': 'Loss',
    })
    # pluggable providers
    data_provider: str = 'native'
    model_provider: str = 'native'
    train_algorithm: str = 'native'
    metric_algorithm: str = 'native'
    generate_algorithm: str | None = None
    system_provider: str = 'pytorch'
    mixed_precision: str | None = None
    # deprecated
    algorithm_provider: str | None = None
    pytorch_accelerator: str | None = None
    metric_provider: str | None = None
    trainer_backend: str | None = None
    # filled by backend
    path: str | None = None
    tag_path: str | None = None
    checkpoint_path: str | None = None
    best_path: str | None = None
    logger_path: str | None = None
    result_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Checkpoint-friendly dict. ``model.stats`` kept by reference (not deep-copied)."""
        stats = self.model.stats if self.model is not None else None
        d = asdict(self)
        if self.model is not None:
            d['model']['stats'] = stats
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RuntimeConfig:
        raw = copy.deepcopy(raw)
        # tolerate legacy checkpoint shape: optimizer under tag key
        model_raw = raw.pop('model', None) or {}
        opt_raw = raw.pop('optimizer', None)
        if opt_raw is None:
            tag = raw.get('tag')
            if tag and isinstance(raw.get(tag), dict) and 'optimizer' in raw[tag]:
                opt_raw = raw[tag]['optimizer']
                raw.pop(tag, None)
            else:
                opt_raw = {}
        # strip legacy keys
        for k in ('control', 'init_seed', 'num_experiments', 'eval'):
            if k == 'eval' and isinstance(raw.get('eval'), dict):
                raw['eval_num_steps'] = raw['eval'].get('num_steps', -1)
            raw.pop(k, None)

        arch = model_raw.get('arch')
        if not isinstance(arch, dict):
            arch = {k: v for k, v in model_raw.items()
                    if k not in ('data_name', 'model_name', 'data_size', 'target_size', 'stats', 'arch')}
        model = ModelRuntime(
            data_name=model_raw.get('data_name', raw.get('data_name', 'MNIST')),
            model_name=model_raw.get('model_name', raw.get('model_name', 'linear')),
            data_size=model_raw.get('data_size'),
            target_size=model_raw.get('target_size'),
            stats=model_raw.get('stats'),
            arch=arch or {},
        )
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        opt_fields = {f.name for f in OptimizerRuntime.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        optimizer = OptimizerRuntime(**{k: v for k, v in opt_raw.items() if k in opt_fields})
        kwargs = {k: v for k, v in raw.items() if k in known and k not in ('model', 'optimizer')}
        kwargs['model'] = model
        kwargs['optimizer'] = optimizer
        return cls(**kwargs)
