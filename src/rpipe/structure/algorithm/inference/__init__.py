"""Inference mode."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.base import Algorithm
from rpipe.structure.algorithm.tracker import AlgorithmTracker


class InferenceAlgorithm(Algorithm):
    def run(self, data: Any, model: Any, system: Any, tracker: AlgorithmTracker) -> dict[str, Any]:
        del data, model, tracker
        if getattr(system, 'logger', None) is not None:
            system.logger.info('inference stub')
        return {'mode': 'inference', 'outputs': []}


def run(control_algorithm: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    algo = InferenceAlgorithm(AlgorithmConfig.from_mapping(control_algorithm))
    return algo.run(state.get('data'), state.get('model'), state.get('system'), state['tracker'])
