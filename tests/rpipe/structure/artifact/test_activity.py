"""Optional activity hints cannot hide authoritative Run status."""
from pathlib import Path
import json

import pytest

from rpipe.structure.artifact.activity import read_activity

pytestmark = [pytest.mark.unit, pytest.mark.content, pytest.mark.p1,
              pytest.mark.structure_layer, pytest.mark.module_artifact,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('payload', [b'\xff', b'{invalid', b'[]', b'{}'])
def test_unusable_activity_is_ignored_without_rewriting(tmp_path, payload):
    path = tmp_path / 'activity.json'
    path.write_bytes(payload)
    assert read_activity(tmp_path) is None
    assert path.read_bytes() == payload


@pytest.mark.parametrize('operation', ['is_file', 'read_text'])
def test_unreadable_activity_is_ignored_including_existence_check(tmp_path, monkeypatch, operation):
    path = tmp_path / 'activity.json'
    path.write_bytes(b'{"phase": "make"}')
    before = path.read_bytes()
    original = getattr(Path, operation)
    def denied(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError('activity denied')
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(Path, operation, denied)
    assert read_activity(tmp_path) is None
    assert path.read_bytes() == before


def test_missing_activity_stays_absent(tmp_path):
    assert read_activity(tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_valid_unicode_activity_keeps_original_bytes(tmp_path):
    path = tmp_path / 'activity.json'
    body = {'phase': 'make', 'detail': '准备共享输入'}
    payload = json.dumps(body, ensure_ascii=False).encode('utf-8')
    path.write_bytes(payload)
    assert read_activity(tmp_path) == body
    assert path.read_bytes() == payload


def test_unexpected_activity_reader_bug_is_not_suppressed(tmp_path, monkeypatch):
    path = tmp_path / 'activity.json'
    path.write_bytes(b'{"phase": "make"}')
    original = Path.read_text
    def broken(candidate, *args, **kwargs):
        if candidate == path:
            raise RuntimeError('unexpected reader bug')
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', broken)
    with pytest.raises(RuntimeError, match='unexpected reader bug'):
        read_activity(tmp_path)



_BAD_ACTIVITY_FIELDS = [
    {'phase': True}, {'phase': 1}, {'phase': ['make']}, {'phase': {'name': 'make'}},
    {'phase': ' '}, {'phase': '\t\n'},
    {'phase': 'make', 'detail': False}, {'phase': 'make', 'detail': 0},
    {'phase': 'make', 'detail': []}, {'phase': 'make', 'detail': {}},
]


@pytest.mark.parametrize('body', _BAD_ACTIVITY_FIELDS)
def test_bad_activity_fields_are_unavailable_without_rewrite(tmp_path, body):
    path = tmp_path / 'activity.json'
    raw = json.dumps(body).encode('utf-8')
    path.write_bytes(raw)
    assert read_activity(tmp_path) is None
    assert path.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize('body', _BAD_ACTIVITY_FIELDS)
def test_bad_activity_cannot_bypass_missing_index_error(tmp_path, capsys, body):
    from rpipe.flow.cli import main

    path = tmp_path / 'activity.json'
    raw = json.dumps(body).encode('utf-8')
    path.write_bytes(raw)
    assert main(['status', str(tmp_path)]) == 2
    captured = capsys.readouterr()
    assert 'missing index' in captured.err
    assert not captured.out
    assert path.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [path]


def test_bad_activity_does_not_prefix_or_hide_authoritative_run_table(tmp_path, capsys):
    from rpipe.flow.cli import main
    from rpipe.structure.artifact.index import write_index
    from rpipe.structure.artifact.result import write_result

    write_index(tmp_path, {'study': 'tiny', 'experiments': [
        {'factors': {'algorithm.mode': 'train'}, 'runs': [{'id': 'peer', 'seed': 0}]}]})
    write_result(tmp_path / 'runs/peer/result.json',
                 {'status': 'succeeded', 'control': {}, 'metrics': {'accuracy': 1}, 'paths': {}})
    (tmp_path / 'activity.json').write_text('{"phase": true}', encoding='utf-8')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert main(['status', str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith('tiny  planned=1 succeeded=1 failed=0 pending=0')
    assert 'accuracy=1.0000' in captured.out and 'peer' in captured.out
    assert not captured.err
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('body', [
    {'phase': 'make', 'detail': '准备共享输入'}, {'phase': 'make'},
    {'phase': 'custom', 'detail': None}, {'phase': ' custom ', 'detail': ''},
])
def test_activity_keeps_custom_unicode_and_optional_detail_compatibility(tmp_path, capsys, body):
    from rpipe.flow.cli import main

    path = tmp_path / 'activity.json'
    raw = json.dumps(body, ensure_ascii=False).encode('utf-8')
    path.write_bytes(raw)
    assert read_activity(tmp_path) == body
    assert main(['status', str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith(body['phase'] + ':')
    assert not captured.err
    assert path.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [path]
