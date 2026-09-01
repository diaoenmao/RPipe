"""Algorithm base class (no factory import)."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.tracker import AlgorithmTracker


class Algorithm:
    """Computation for one mode. Loop insertion points are named methods (§6.10)."""

    def __init__(self, config: AlgorithmConfig) -> None:
        self.config = config
        self.mode = config.mode or 'train'

    def run(self, data: Any, model: Any, system: Any, tracker: AlgorithmTracker) -> dict[str, Any]:
        raise NotImplementedError

    def eval_period(self) -> int:
        """How often ``on_eval_period`` runs. ``1`` = every epoch; ``0`` = once after the loop."""
        raw = self.config.setting('eval_period', 1)
        if raw is None:
            return 1
        return int(raw)

    def on_eval_period(
        self,
        tracker: AlgorithmTracker,
        logger: Any,
        data: Any,
        model: Any,
        system: Any,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Named train-loop hook. Return True to stop ``run()``.

        Not a Flow phase. Base is no-op (do not stop).
        """
        del tracker, logger, data, model, system, extra
        return False
