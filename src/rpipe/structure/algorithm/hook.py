"""AlgorithmHook: named loop insertion points (not a Flow phase)."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.progress import (
    CHECKPOINT_LATEST,
    parse_checkpoint_mode,
    parse_percents,
    resolve_budget,
)
from rpipe.structure.algorithm.tracker import AlgorithmTracker


class AlgorithmHook:
    """Contract for algorithm-loop hooks. Methods live on ``Algorithm``.

    Base is no-op / do-not-stop. Concrete modes override the points they need.
    Shared signature: ``(tracker, logger, data, model, system, extra=None)``.
    """

    def eval_period(self) -> int:
        """How often ``on_eval_period`` runs. ``1`` = every progress unit; ``0`` = once after the loop."""
        raw = self.config.setting('eval_period', 1)  # type: ignore[attr-defined]
        if raw is None:
            return 1
        return int(raw)

    def progress_unit(self) -> str:
        return resolve_budget(self.config).unit  # type: ignore[attr-defined]

    def checkpoint_mode(self) -> str:
        raw = self.config.setting('checkpoint', CHECKPOINT_LATEST)  # type: ignore[attr-defined]
        return parse_checkpoint_mode(raw)

    def checkpoint_period(self) -> int:
        raw = self.config.setting('checkpoint_period', 1)  # type: ignore[attr-defined]
        if raw is None:
            return 1
        return int(raw)

    def checkpoint_percents(self) -> tuple[float, ...]:
        return parse_percents(self.config.setting('checkpoint_percents'))  # type: ignore[attr-defined]

    def save_best(self) -> bool:
        return bool(self.config.setting('save_best', False))  # type: ignore[attr-defined]

    def on_eval_period(
        self,
        tracker: AlgorithmTracker,
        logger: Any,
        data: Any,
        model: Any,
        system: Any,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Return True to stop ``run()``. Not a Flow phase."""
        del tracker, logger, data, model, system, extra
        return False

    def on_checkpoint(
        self,
        tracker: AlgorithmTracker,
        logger: Any,
        data: Any,
        model: Any,
        system: Any,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write weights via ``system.save_checkpoint``. Not a Flow phase."""
        del tracker, logger, data, model, system, extra
