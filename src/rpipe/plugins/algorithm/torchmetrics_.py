from __future__ import annotations

from typing import Any

from rpipe.plugins.api import register


@register('algorithm', 'torchmetrics')
class TorchMetricsProvider:
    """PyPI package: ``torchmetrics`` — online classification / regression metrics.

    Refs: https://pypi.org/project/torchmetrics/ · https://torchmetrics.readthedocs.io/
    """

    name = 'torchmetrics'
    package = 'torchmetrics'
    mode = 'online'
    kind = 'metric'

    def available(self) -> bool:
        try:
            import torchmetrics  # noqa: F401
            return True
        except ImportError:
            return False

    def list_metrics(self) -> list[str]:
        return ['Accuracy', 'Precision', 'Recall', 'F1Score', 'MeanSquaredError']

    def make_metric(self, metric_kwargs: dict[str, Any]):
        if not self.available():
            raise ImportError('Install torchmetrics: pip install torchmetrics')
        # Reuse native Metric shell but swap Accuracy implementation via adapter module
        from rpipe.algorithm.metrics.metric import Metric
        from rpipe.plugins.algorithm._torchmetrics_adapter import build_torchmetrics_bank

        metric = Metric(
            metric_kwargs['metric_name'],
            best=float('inf'),
            best_split=metric_kwargs.get('best_split', 'test'),
            best_direction=1,
            best_metric_name=metric_kwargs.get('best_metric_name', 'Loss'),
        )
        metric.metric = build_torchmetrics_bank(metric_kwargs['metric_name'])
        return metric
