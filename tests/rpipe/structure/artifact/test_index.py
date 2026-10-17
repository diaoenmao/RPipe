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
