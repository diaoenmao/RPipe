from __future__ import annotations

from typing import Any

from rpipe.plugins.api import register


@register('algorithm', 'evaluate')
class EvaluateMetricProvider:
    """PyPI package: ``evaluate`` (Hugging Face Evaluate).

    Refs: https://pypi.org/project/evaluate/ · https://huggingface.co/docs/evaluate
    """

    name = 'evaluate'
    package = 'evaluate'
    mode = 'online'
    kind = 'metric'

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
        from rpipe.plugins.algorithm._hf_evaluate_adapter import build_hf_evaluate_bank

        metric = Metric(
            metric_kwargs['metric_name'],
            best=float('inf'),
            best_split=metric_kwargs.get('best_split', 'test'),
            best_direction=1,
            best_metric_name=metric_kwargs.get('best_metric_name', 'Loss'),
        )
        metric.metric = build_hf_evaluate_bank(metric_kwargs['metric_name'])
        return metric
