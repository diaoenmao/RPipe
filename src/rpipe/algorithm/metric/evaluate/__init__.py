"""algorithm/metric/evaluate — PyPI ``evaluate``."""

from __future__ import annotations

from typing import Any

from rpipe.provider import register_algorithm


@register_algorithm('metric', 'evaluate')
class EvaluateMetricAlgorithm:
    name = 'evaluate'
    package = 'evaluate'
    algorithm_type = 'metric'
    mode = 'online'

    def available(self) -> bool:
        try:
            import evaluate  # noqa: F401
            return True
        except ImportError:
            return False

    def list_metrics(self) -> list[str]:
        return ['accuracy', 'f1', 'precision', 'recall', 'squad', 'bleu', 'rouge']

    def make_metric(self, metric_kwargs: dict[str, Any]):
        if not self.available():
            raise ImportError('Install package: pip install evaluate')
        from rpipe.algorithm.metrics.metric import Metric
        from rpipe.algorithm.metric._adapters.evaluate_adapter import build_hf_evaluate_bank
        metric = Metric(
            metric_kwargs['metric_name'],
            best=float('inf'),
            best_split=metric_kwargs.get('best_split', 'test'),
            best_direction=1,
            best_metric_name=metric_kwargs.get('best_metric_name', 'Loss'),
        )
        metric.metric = build_hf_evaluate_bank(metric_kwargs['metric_name'])
        return metric
