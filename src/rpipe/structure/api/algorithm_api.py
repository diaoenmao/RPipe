"""algorithm_api: Algorithm, AlgorithmTracker, AlgorithmFactory.build, AlgorithmConfig."""

from pathlib import Path

from rpipe.structure.algorithm import (
    Algorithm,
    AlgorithmConfig,
    AlgorithmFactory,
    AlgorithmRegistry,
    AlgorithmTracker,
)


def build(algorithm_config: AlgorithmConfig, **kwargs):
    return AlgorithmFactory.build(algorithm_config, **kwargs)


def make_tracker(assets_dir: Path | str) -> AlgorithmTracker:
    return AlgorithmTracker(assets_dir)
