import pytest

from rpipe.structure.algorithm.config import AlgorithmConfig
from rpipe.structure.system.config import SystemConfig

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_control,
              pytest.mark.cost(cost_class='c1'), pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('config_type,key', [(SystemConfig, 'device'), (AlgorithmConfig, 'num_steps')])
@pytest.mark.parametrize('nested', [None, {'value': 'nested'}, {'value': None}])
def test_raw_setting_lookup_matches_runtime_config(config_type, key, nested):
    from rpipe.structure.control.mapping import get_setting

    mapping = {key: 'outer'}
    if nested is not None:
        mapping['config'] = {key: nested['value']}
    assert get_setting(mapping, key) == config_type.from_mapping(mapping).setting(key)
