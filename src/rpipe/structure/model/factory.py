"""Runtime Model object, registry, and factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from rpipe.structure.model.config import ModelConfig


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
    def build(model_config: ModelConfig, assets_dir: Path | str) -> Model:
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
        return builder(model_config, Path(assets_dir))


def _build_linear(model_config: ModelConfig, assets_dir: Path) -> Model:
    import torch.nn as nn

    cfg = dict(model_config.config)
    in_features = int(cfg.get('in_features', 784))
    out_features = int(cfg.get('out_features', 10))
    module = nn.Linear(in_features, out_features)
    return Model(
        name='linear',
        source=model_config.source or 'custom_torch',
        module=module,
        meta={
            'ready': True,
            'assets_dir': str(assets_dir),
            'config': cfg,
            'in_features': in_features,
            'out_features': out_features,
        },
    )


ModelRegistry.register('linear', 'custom_torch', _build_linear)
