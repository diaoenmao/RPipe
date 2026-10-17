import json

import pytest

from rpipe.structure.artifact.index import load_index

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


def test_index_read_exit_preserves_first_failure(tmp_path, reader_exit_case, reader_close_fault):
    case = reader_exit_case
    path, peer = tmp_path / 'index.json', tmp_path / 'peer'
    payload = b'{"experiments":[]}'
    path.write_bytes(payload)
    peer.write_bytes(b'peer untouched')
    with reader_close_fault(path, case, 'r') as (handles, closes):
        if case.expected is None:
            assert load_index(tmp_path) == {'experiments': []}
        else:
            with pytest.raises(type(case.expected)) as caught:
                load_index(tmp_path)
            assert caught.value is case.expected
    assert len(handles) == len(closes) == 1 and handles[0].closed
    assert path.read_bytes() == payload and peer.read_bytes() == b'peer untouched'


def test_index_real_json_error_survives_secondary_close(tmp_path, reader_close_fault):
    from types import SimpleNamespace
    path = tmp_path / 'index.json'
    path.write_bytes(b'{broken JSON')
    case = SimpleNamespace(primary=None, secondary=OSError('secondary close IO'))
    with reader_close_fault(path, case, 'r') as (handles, closes):
        with pytest.raises(json.JSONDecodeError):
            load_index(tmp_path)
    assert len(handles) == len(closes) == 1 and handles[0].closed
    assert path.read_bytes() == b'{broken JSON'


def test_experiment_entries_keep_first_seen_group_order_and_run_identity(tmp_path):
    from rpipe.structure.artifact.index import experiment_entries
    configs = []
    for i, model in enumerate(['b', 'a', 'b', 'c', 'a']):
        tags = [f'tag-{i}']
        configs.append((tmp_path / f'runs/r{i}/config.yaml',
                        {'id': f'r{i}', 'seed': i, 'model': {'name': model}, 'tags': tags}))
    groups = experiment_entries(configs=configs, axis_keys=['model.name'], study_dir=tmp_path)
    assert [g['factors'] for g in groups] == [{'model.name': x} for x in ['b', 'a', 'c']]
    assert [[r['id'] for r in g['runs']] for g in groups] == [['r0', 'r2'], ['r1', 'r4'], ['r3']]
    assert groups[0]['runs'][0]['tags'] is configs[0][1]['tags']
    assert groups[0]['runs'][0]['config'] == 'runs/r0/config.yaml'
    assert groups[0]['runs'][0]['log'] == 'runs/r0/assets/logs/run.log'
    assert experiment_entries(configs=[], axis_keys=[], study_dir=tmp_path) == []
