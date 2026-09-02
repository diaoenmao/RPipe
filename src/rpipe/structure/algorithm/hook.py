"""AlgorithmHook: named loop insertion points (not a Flow phase)."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.tracker import AlgorithmTracker


class AlgorithmHook:
    """Contract for algorithm-loop hooks. Methods live on ``Algorithm``.

    Base is no-op / do-not-stop. Concrete modes override the points they need.
    Shared signature: ``(tracker, logger, data, model, system, extra=None)``.
    """

    def eval_period(self) -> int:
        """How often ``on_eval_period`` runs. ``1`` = every epoch; ``0`` = once after the loop."""
        raw = self.config.setting('eval_period', 1)  # type: ignore[attr-defined]
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
        """Return True to stop ``run()``. Not a Flow phase."""
        del tracker, logger, data, model, system, extra
        return False
