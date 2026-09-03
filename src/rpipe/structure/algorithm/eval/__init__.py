"""Eval mode: independent algorithm; resume then score."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.base import Algorithm
from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.algorithm.eval_hook import eval_test_split
from rpipe.structure.algorithm.tracker import AlgorithmTracker


class EvalAlgorithm(Algorithm):
    def run(self, data: Any, model: Any, system: Any, tracker: AlgorithmTracker) -> dict[str, Any]:
        logger = getattr(system, 'logger', None)
        extra: dict[str, Any] = {}
        if getattr(model, 'module', None) is None:
            tracker.append('test', n=1, values={'Accuracy': 0.0})
            tracker.save('test')
            tracker.flush('test')
            if logger is not None:
                logger.report(tracker, 'test')
            return {'mode': 'eval', 'stub': True}
        if getattr(system, 'place_module', None):
            model.module = system.place_module(model.module)
        restored = self.resume(data, model, system, tracker, extra)
        metrics = eval_test_split(tracker, logger, data, model, system, extra)
        accuracy = metrics.get('Accuracy', 0.0)
        return {
            'mode': 'eval',
            'accuracy': accuracy,
            'best_accuracy': (restored or {}).get('best_accuracy', accuracy),
            'resume_stem': extra.get('resume_stem'),
            'step': (restored or {}).get('step'),
            'epoch': (restored or {}).get('epoch'),
        }


def run(control_algorithm: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    algo = EvalAlgorithm(AlgorithmConfig.from_mapping(control_algorithm))
    return algo.run(state.get('data'), state.get('model'), state.get('system'), state['tracker'])
