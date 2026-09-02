"""Algorithm base class (no factory import)."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.hook import AlgorithmHook
from rpipe.structure.algorithm.tracker import AlgorithmTracker


class Algorithm(AlgorithmHook):
    """Computation for one mode. Loop insertion points are ``AlgorithmHook``."""

    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config
        self.mode = config.mode or 'train'

    def run(self, data: Any, model: Any, system: Any, tracker: AlgorithmTracker) -> dict[str, Any]:
        raise NotImplementedError
