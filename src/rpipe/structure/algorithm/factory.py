"""Algorithm factory: mode + source → implementation."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.base import Algorithm
from rpipe.structure.algorithm.config import AlgorithmConfig


def _table() -> dict[tuple[str, str], type[Algorithm]]:
    from rpipe.structure.algorithm.eval import EvalAlgorithm
    from rpipe.structure.algorithm.train import TrainAlgorithm
    from rpipe.structure.algorithm.transformers_trainer import HfEvalAlgorithm, HfTrainAlgorithm

    return {
        ('train', 'custom_torch'): TrainAlgorithm,
        ('eval', 'custom_torch'): EvalAlgorithm,
        ('train', 'transformers_trainer'): HfTrainAlgorithm,
        ('eval', 'transformers_trainer'): HfEvalAlgorithm,
    }


class AlgorithmRegistry:
    """Lookup table filled on first build. Extra sources call ``register``."""

    _items: dict[tuple[str, str], type[Algorithm]] | None = None

    @classmethod
    def _store(cls) -> dict[tuple[str, str], type[Algorithm]]:
        if cls._items is None:
            cls._items = dict(_table())
        return cls._items

    @classmethod
    def register(cls, mode: str, source: str, impl: type[Algorithm]) -> None:
        cls._store()[(mode, source)] = impl

    @classmethod
    def get(cls, mode: str, source: str) -> type[Algorithm] | None:
        return cls._store().get((mode, source))

    @classmethod
    def list(cls) -> list[tuple[str, str]]:
        return sorted(cls._store())


class AlgorithmFactory:
    @staticmethod
    def build(algorithm_config: AlgorithmConfig, **_: Any) -> Algorithm:
        mode = algorithm_config.mode or 'train'
        source = algorithm_config.source or 'custom_torch'
        impl = AlgorithmRegistry.get(mode, source)
        if impl is None:
            raise ValueError(f'unknown algorithm mode/source: {mode}/{source}')
        return impl(algorithm_config)
