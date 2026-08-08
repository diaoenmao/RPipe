from __future__ import annotations

from typing import Any

from rpipe.algorithm.metrics.metric import make_metric as _make_metric
from rpipe.plugins.api import register


@register('algorithm', 'native')
class NativeMetricProvider:
    """Built-in Loss/Accuracy/MSE/RMSE/GLUE metrics."""

    name = 'native'
    mode = 'online'
    kind = 'metric'

    def available(self) -> bool:
        return True

    def list_metrics(self) -> list[str]:
        return ['Loss', 'Accuracy', 'MSE', 'RMSE', 'GLUE']

    def make_metric(self, metric_kwargs: dict[str, Any]):
        return _make_metric(metric_kwargs)
