"""algorithm/metric/native — built-in Loss/Accuracy metrics."""

from __future__ import annotations

from typing import Any

from rpipe.algorithm.metrics.metric import make_metric as _make_metric
from rpipe.provider import register_algorithm


@register_algorithm('metric', 'native')
class NativeMetricAlgorithm:
    name = 'native'
    package = 'rpipe'
    algorithm_type = 'metric'
    mode = 'online'

    def available(self) -> bool:
        return True

    def list_metrics(self) -> list[str]:
        return ['Loss', 'Accuracy', 'MSE', 'RMSE', 'GLUE']

    def make_metric(self, metric_kwargs: dict[str, Any]):
        return _make_metric(metric_kwargs)
