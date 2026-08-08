"""Unit tests for data/* providers."""

from __future__ import annotations

import pytest

from rpipe.provider import get_provider


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.data_layer
@pytest.mark.module_provider
def test_native_data_lists_sets():
    names = get_provider('data', 'native').list_datasets()
    assert isinstance(names, list)


@pytest.mark.unit
@pytest.mark.content
@pytest.mark.p1
@pytest.mark.data_layer
def test_datasets_provider_package():
    p = get_provider('data', 'datasets', require_available=False)
    assert p.package == 'datasets'
    assert 'openai/gsm8k' in p.list_datasets()
