import json
from pathlib import Path

import pytest

from rpipe.flow.process.aggregate import load_curve_series, load_tracker_history
from rpipe.structure.artifact.asset import kinds

pytestmark = [
    pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
    pytest.mark.flow_layer, pytest.mark.module_process,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]

_BAD_ENCODING = [b'\xff\xfe\x80', b'{"splits": "\x80"}', b'\xf0\x9f']


@pytest.mark.parametrize('raw', _BAD_ENCODING)
@pytest.mark.parametrize('primary', [False, True], ids=['fallback-only', 'valid-jsonl'])
@pytest.mark.parametrize('reader', [load_curve_series, load_tracker_history])
def test_optional_state_decode_failure_keeps_valid_trajectory(tmp_path, raw, primary, reader):
    assets = tmp_path / 'runs/a/assets'
    state = assets / kinds.TRACKER_STATE
    state.parent.mkdir(parents=True)
    state.write_bytes(raw)
    scalar = assets / kinds.TRACKER_JSONL
    if primary:
        scalar.parent.mkdir(parents=True, exist_ok=True)
        scalar.write_text(json.dumps({'split': 'test', 'name': 'Accuracy', 'mean': 75.0,
                                      'optimizer_step': 1}) + '\n', encoding='utf-8')
    before = {p: p.read_bytes() for p in assets.rglob('*') if p.is_file()}
    body = reader(tmp_path, 'a')
    if primary:
        expected = [75.0] if reader is load_tracker_history else {
            'unit': 'step', 'x': [1], 'values': [75.0]}
        assert body == {'test': {'Accuracy': expected}}
    else:
        assert body == {}
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('raw', [b'{', b'[]', b'{}',
    b'{"splits":{"test":{"Accuracy":{"history":[10.0,20.0]}}}}'])
def test_optional_state_legacy_fallback_contract(tmp_path, raw):
    state = tmp_path / 'runs/a/assets' / kinds.TRACKER_STATE
    state.parent.mkdir(parents=True)
    state.write_bytes(raw)
    body = load_curve_series(tmp_path, 'a')
    expected = {'test': {'Accuracy': {'unit': 'observation', 'x': [1, 2],
                                     'values': [10.0, 20.0]}}} if b'history' in raw else {}
    assert body == expected
    assert state.read_bytes() == raw


@pytest.mark.parametrize('error', [OSError('unreadable'), RuntimeError('programming failure')])
def test_optional_state_keeps_io_and_programming_error_boundary(tmp_path, monkeypatch, error):
    state = tmp_path / 'runs/a/assets' / kinds.TRACKER_STATE
    state.parent.mkdir(parents=True)
    state.write_text('{}', encoding='utf-8')
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path == state:
            raise error
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    if isinstance(error, RuntimeError):
        with pytest.raises(RuntimeError) as caught:
            load_curve_series(tmp_path, 'a')
        assert caught.value is error
    else:
        assert load_curve_series(tmp_path, 'a') == {}



@pytest.mark.parametrize('reader', [load_curve_series, load_tracker_history])
@pytest.mark.parametrize('primary', [False, True], ids=['legacy-only', 'valid-jsonl'])
@pytest.mark.parametrize('large', [False, True], ids=['valid-integer', 'integer-limit'])
def test_optional_state_integer_decode_keeps_primary_and_legacy(tmp_path, reader, primary, large):
    import sys

    limit = getattr(sys, 'get_int_max_str_digits', lambda: 0)()
    if large and not 0 < limit < 5000:
        pytest.skip('native integer digit limit is unavailable or outside the 5000-digit probe')
    assets = tmp_path / 'runs/a/assets'
    state = assets / kinds.TRACKER_STATE
    state.parent.mkdir(parents=True)
    raw = (b'{"splits":{"test":{"Accuracy":{"history":[10.0,20.0]}}},"unused":'
           + (b'9' * 5000 if large else b'17') + b'}')
    state.write_bytes(raw)
    scalar = assets / kinds.TRACKER_JSONL
    if primary:
        scalar.parent.mkdir(parents=True, exist_ok=True)
        scalar.write_text(json.dumps({'split': 'test', 'name': 'Accuracy', 'mean': 75.0,
                                      'optimizer_step': 1}) + '\n', encoding='utf-8')
    before = {p: p.read_bytes() for p in assets.rglob('*') if p.is_file()}
    if not primary and large:
        expected = {}
    else:
        values = [75.0] if primary else [10.0, 20.0]
        series = values if reader is load_tracker_history else {
            'unit': 'step' if primary else 'observation',
            'x': [1] if primary else [1, 2], 'values': values}
        expected = {'test': {'Accuracy': series}}
    assert reader(tmp_path, 'a') == expected
    assert {p: p.read_bytes() for p in assets.rglob('*') if p.is_file()} == before
