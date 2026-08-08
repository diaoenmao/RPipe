"""Shared protocol for algorithm-layer metric providers.

Two modes share one registry (``rpipe.algorithm``), keyed by PyPI package name:

- ``online`` — native, torchmetrics, evaluate
- ``benchmark`` — lm_eval, opencompass
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetricProvider(Protocol):
    name: str
    mode: str  # 'online' | 'benchmark'

    def available(self) -> bool: ...

    def list_metrics(self) -> list[str]: ...


@runtime_checkable
class OnlineMetricProvider(MetricProvider, Protocol):
    def make_metric(self, metric_kwargs: dict[str, Any]) -> Any: ...


@runtime_checkable
class BenchmarkMetricProvider(MetricProvider, Protocol):
    """LLM / generative benchmark harness (GSM8K, MMLU, HumanEval+, …)."""

    def evaluate(self, **kwargs: Any) -> dict[str, Any]: ...
