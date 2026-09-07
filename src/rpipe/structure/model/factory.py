"""Runtime Model object, registry, and factory."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from rpipe.structure.model.config import ModelConfig
from rpipe.structure.model.init_param import init_param
from rpipe.structure.model.shape import resolve_shape


class Model:
    def __init__(
        self,
        *,
        name: str,
        source: str,
        module: Any | None,
        meta: dict[str, Any],
    ) -> None:
        self.name = name
        self.source = source
        self.module = module
        self.meta = meta

    def to_result_snapshot(self) -> dict[str, Any]:
        out = {
            'name': self.name,
            'source': self.source,
            **{k: v for k, v in self.meta.items() if k != 'module'},
        }
        return out


class ModelRegistry:
    _items: dict[tuple[str, str], Callable[..., Model]] = {}

    @classmethod
    def register(cls, name: str, source: str, builder: Callable[..., Model]) -> None:
        cls._items[(name, source)] = builder

    @classmethod
    def get(cls, name: str, source: str) -> Callable[..., Model] | None:
        return cls._items.get((name, source))

    @classmethod
    def list(cls) -> list[tuple[str, str]]:
        return sorted(cls._items)


class ModelFactory:
    @staticmethod
    def build(
        model_config: ModelConfig,
        assets_dir: Path | str,
        data_meta: dict[str, Any] | None = None,
    ) -> Model:
        name = model_config.name or 'unknown'
        source = model_config.source or 'custom_torch'
        builder = ModelRegistry.get(name, source) or ModelRegistry.get(name, 'custom_torch')
        if builder is None:
            return Model(
                name=name,
                source=source,
                module=None,
                meta={'ready': True, 'assets_dir': str(assets_dir), 'config': dict(model_config.config)},
            )
        return builder(model_config, Path(assets_dir), data_meta=data_meta)


def _wrap(name: str, model_config: ModelConfig, assets_dir: Path, module: Any, extra: dict[str, Any]) -> Model:
    module.apply(init_param)
    return Model(
        name=name,
        source=model_config.source or 'custom_torch',
        module=module,
        meta={
            'ready': True,
            'assets_dir': str(assets_dir),
            'config': dict(model_config.config),
            **extra,
        },
    )


def _build_linear(model_config: ModelConfig, assets_dir: Path, data_meta: dict[str, Any] | None = None) -> Model:
    from rpipe.structure.model.custom_torch import Linear

    cfg = dict(model_config.config)
    data_size, target_size = resolve_shape(cfg, data_meta)
    module = Linear(data_size, target_size)
    return _wrap(
        'linear',
        model_config,
        assets_dir,
        module,
        {'data_size': list(data_size), 'target_size': target_size, 'in_features': int(math.prod(data_size))},
    )


def _build_mlp(model_config: ModelConfig, assets_dir: Path, data_meta: dict[str, Any] | None = None) -> Model:
    from rpipe.structure.model.custom_torch import MLP

    cfg = dict(model_config.config)
    data_size, target_size = resolve_shape(cfg, data_meta)
    module = MLP(
        data_size,
        hidden_size=int(cfg.get('hidden_size', 128)),
        scale_factor=float(cfg.get('scale_factor', 2)),
        num_layers=int(cfg.get('num_layers', 2)),
        activation=str(cfg.get('activation', 'relu')),
        target_size=target_size,
    )
    return _wrap(
        'mlp',
        model_config,
        assets_dir,
        module,
        {'data_size': list(data_size), 'target_size': target_size},
    )


def _build_cnn(model_config: ModelConfig, assets_dir: Path, data_meta: dict[str, Any] | None = None) -> Model:
    from rpipe.structure.model.custom_torch import CNN

    cfg = dict(model_config.config)
    data_size, target_size = resolve_shape(cfg, data_meta)
    hidden = [int(x) for x in (cfg.get('hidden_size') or [64, 128, 256, 512])]
    module = CNN(data_size, hidden, target_size)
    return _wrap(
        'cnn',
        model_config,
        assets_dir,
        module,
        {'data_size': list(data_size), 'target_size': target_size, 'hidden_size': hidden},
    )


def _build_resnet(
    model_config: ModelConfig,
    assets_dir: Path,
    data_meta: dict[str, Any] | None = None,
    *,
    name: str,
    num_blocks: list[int],
) -> Model:
    from rpipe.structure.model.custom_torch import ResNet

    cfg = dict(model_config.config)
    data_size, target_size = resolve_shape(cfg, data_meta)
    hidden = [int(x) for x in (cfg.get('hidden_size') or [64, 128, 256, 512])]
    module = ResNet(data_size, hidden, num_blocks, target_size)
    return _wrap(
        name,
        model_config,
        assets_dir,
        module,
        {'data_size': list(data_size), 'target_size': target_size, 'hidden_size': hidden},
    )


def _build_resnet18(model_config: ModelConfig, assets_dir: Path, data_meta: dict[str, Any] | None = None) -> Model:
    return _build_resnet(model_config, assets_dir, data_meta, name='resnet18', num_blocks=[2, 2, 2, 2])


def _build_resnet10(model_config: ModelConfig, assets_dir: Path, data_meta: dict[str, Any] | None = None) -> Model:
    return _build_resnet(model_config, assets_dir, data_meta, name='resnet10', num_blocks=[1, 1, 1, 1])


ModelRegistry.register('linear', 'custom_torch', _build_linear)
ModelRegistry.register('mlp', 'custom_torch', _build_mlp)
ModelRegistry.register('cnn', 'custom_torch', _build_cnn)
ModelRegistry.register('resnet18', 'custom_torch', _build_resnet18)
ModelRegistry.register('resnet', 'custom_torch', _build_resnet18)
ModelRegistry.register('resnet10', 'custom_torch', _build_resnet10)
