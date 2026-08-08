"""Unit tests for algorithm/metric/*."""

from __future__ import annotations

import pytest

from rpipe.provider import get_algorithm


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.algorithm_layer
def test_native_metric_make_metric():
    p = get_algorithm('metric', 'native')
    metric = p.make_metric({
        'metric_name': {'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']},
        'best_split': 'test',
        'best_metric_name': 'Loss',
    })
    assert metric is not None


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p2
@pytest.mark.algorithm_layer
@pytest.mark.external
def test_lm_eval_lists_gsm8k():
    p = get_algorithm('metric', 'lm_eval', require_available=False)
    assert 'gsm8k' in p.list_metrics()
    assert p.mode == 'benchmark'
