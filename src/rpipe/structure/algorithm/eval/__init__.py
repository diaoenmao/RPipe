"""Eval mode."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.base import Algorithm
from rpipe.structure.algorithm.tracker import AlgorithmTracker


class EvalAlgorithm(Algorithm):
    def run(self, data: Any, model: Any, system: Any, tracker: AlgorithmTracker) -> dict[str, Any]:
        tracker.append('test', n=1, values={'Accuracy': 0.0})
        tracker.save('test')
        tracker.flush('test')
        if getattr(system, 'logger', None) is not None:
            system.logger.report(tracker, 'test')
        return {'mode': 'eval'}


def run(control_algorithm: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    algo = EvalAlgorithm(AlgorithmConfig.from_mapping(control_algorithm))
    return algo.run(state.get('data'), state.get('model'), state.get('system'), state['tracker'])
