"""Algorithm runtime, factory, and tracker."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.base import Algorithm
from rpipe.structure.algorithm.config import AlgorithmConfig


class AlgorithmRegistry:
    _items: dict[tuple[str, str], type[Algorithm]] = {}

    @classmethod
    def register(cls, mode: str, source: str, impl: type[Algorithm]) -> None:
        cls._items[(mode, source)] = impl

    @classmethod
    def get(cls, mode: str, source: str) -> type[Algorithm] | None:
        return cls._items.get((mode, source))

    @classmethod
    def list(cls) -> list[tuple[str, str]]:
        return sorted(cls._items)


class AlgorithmFactory:
    @staticmethod
    def build(algorithm_config: AlgorithmConfig, **_: Any) -> Algorithm:
        from rpipe.structure.algorithm.eval import EvalAlgorithm
        from rpipe.structure.algorithm.inference import InferenceAlgorithm
        from rpipe.structure.algorithm.train import TrainAlgorithm
        from rpipe.structure.algorithm.transformers_trainer import HfEvalAlgorithm, HfTrainAlgorithm

        AlgorithmRegistry.register('train', 'custom_torch', TrainAlgorithm)
        AlgorithmRegistry.register('eval', 'custom_torch', EvalAlgorithm)
        AlgorithmRegistry.register('inference', 'custom_torch', InferenceAlgorithm)
        AlgorithmRegistry.register('train', 'transformers_trainer', HfTrainAlgorithm)
        AlgorithmRegistry.register('eval', 'transformers_trainer', HfEvalAlgorithm)

        mode = algorithm_config.mode or 'train'
        source = algorithm_config.source or 'custom_torch'
        impl = AlgorithmRegistry.get(mode, source)
        if impl is None:
            mapping = {
                'train': TrainAlgorithm,
                'eval': EvalAlgorithm,
                'inference': InferenceAlgorithm,
            }
            impl = mapping.get(mode)
        if impl is None:
            raise ValueError(f'unknown algorithm mode: {mode}')
        return impl(algorithm_config)
