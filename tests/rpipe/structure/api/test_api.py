"""Public facade locations (CONCEPT gray door)."""

from __future__ import annotations

import pytest

from rpipe.structure import api as api_pkg

pytestmark = [
    pytest.mark.unit,
    pytest.mark.location,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_api,
]


@pytest.mark.parametrize(
    'symbol',
    ['data_api', 'model_api', 'algorithm_api', 'system_api'],
)
def test_structure_api_facade_module_is_importable_from_api_package(symbol: str):
    obj = getattr(api_pkg, symbol)
    assert obj.__name__ == f'rpipe.structure.api.{symbol}'
