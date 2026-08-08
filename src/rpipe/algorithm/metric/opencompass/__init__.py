"""algorithm/metric/opencompass — PyPI ``opencompass``."""

from __future__ import annotations

from typing import Any

from rpipe.provider import register_algorithm


@register_algorithm('metric', 'opencompass')
class OpenCompassMetricAlgorithm:
    name = 'opencompass'
    package = 'opencompass'
    algorithm_type = 'metric'
    mode = 'benchmark'

    def available(self) -> bool:
        try:
            import opencompass  # noqa: F401
            return True
        except ImportError:
            return False

    def list_metrics(self) -> list[str]:
        return ['mmlu', 'cmmlu', 'ceval', 'gsm8k', 'humaneval']

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        if not self.available():
            raise ImportError('Install package: pip install -U opencompass')
        config = kwargs.get('config')
        if not config:
            raise ValueError('opencompass.evaluate requires config= path')
        return {
            'schema': 'rpipe.metric_benchmark.v1',
            'algorithm': self.name,
            'algorithm_type': self.algorithm_type,
            'config': config,
            'cli': f'opencompass {config}',
        }
