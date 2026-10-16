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



@pytest.mark.parametrize('planned', [False, True], ids=['missing-index', 'planned-peer'])
@pytest.mark.parametrize('payload', ['valid-integer', 'integer-limit', 'deep-json'])
def test_optional_activity_integer_decode_keeps_authoritative_status(tmp_path, capsys, planned, payload):
    import sys
    from rpipe.flow.cli import main
    from rpipe.structure.artifact.index import write_index
    from rpipe.structure.artifact.result import write_result

    limit = getattr(sys, 'get_int_max_str_digits', lambda: 0)()
    large = payload != 'valid-integer'
    if payload == 'integer-limit' and not 0 < limit < 5000:
        pytest.skip('native integer digit limit is unavailable or outside the 5000-digit probe')
    if planned:
        write_index(tmp_path, {'study': 'tiny', 'experiments': [
            {'factors': {'algorithm.mode': 'train'}, 'runs': [{'id': 'peer', 'seed': 0}]}]})
        write_result(tmp_path / 'runs/peer/result.json',
                     {'status': 'succeeded', 'control': {}, 'metrics': {'accuracy': 1}, 'paths': {}})
    raw = b'{"phase":"make","unused":' + (b'[' * 10000 + b'0' + b']' * 10000 if payload == 'deep-json'
                    else b'9' * 5000 if large else b'17') + b'}'
    path = tmp_path / 'activity.json'
    path.write_bytes(raw)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert read_activity(tmp_path) == (None if large else {'phase': 'make', 'unused': 17})
    assert main(['status', str(tmp_path)]) == (0 if planned or not large else 2)
    captured = capsys.readouterr()
    if planned:
        assert 'tiny  planned=1 succeeded=1 failed=0 pending=0' in captured.out
        assert 'peer' in captured.out and 'accuracy=1.0000' in captured.out
        assert not captured.err
    elif large:
        assert 'missing index' in captured.err and not captured.out
    else:
        assert captured.out.startswith('make:') and not captured.err
    assert captured.out.startswith('make:') == (not large)
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('field', ['phase', 'detail'])
@pytest.mark.parametrize('separator', ['\n', '\r\n', '\u2028', '\t'])
def test_announcement_projects_unsafe_hint_fields_to_one_line_without_changing_mapping(tmp_path, capsys, field, separator):
    from rpipe.structure.artifact.activity import announce
    body = {'phase': 'make', 'detail': 'working'}
    body[field] += separator + 'foreign'
    peer = tmp_path / 'peer.bin'
    peer.write_bytes(b'peer')
    announce(tmp_path, body['phase'], body['detail'])
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 1 and '\t' not in output
    assert all(word in output for word in ('make', 'working', 'foreign'))
    assert read_activity(tmp_path) == body
    assert json.loads((tmp_path / 'activity.json').read_text(encoding='utf-8')) == body
    assert peer.read_bytes() == b'peer'


@pytest.mark.parametrize('detail', ['', '准备共享输入'])
def test_single_line_announcement_keeps_existing_print_and_persistence(tmp_path, capsys, detail):
    from rpipe.structure.artifact.activity import announce
    announce(tmp_path, ' custom ', detail)
    assert capsys.readouterr().out == f' custom : {detail}\n'
    assert read_activity(tmp_path) == {'phase': ' custom ', 'detail': detail}


@pytest.mark.parametrize('text,expected', [
    pytest.param('left' + chr(0xD800) + 'right', r'left\ud800right', id='high'),
    pytest.param('left' + chr(0xDC00) + 'right', r'left\udc00right', id='low'),
    pytest.param('left' + chr(0xD800) + 'mid' + chr(0xDC00) + 'right', r'left\ud800mid\udc00right', id='mixed'),
    pytest.param('left-ASCII-right', 'left-ASCII-right', id='ascii'),
    pytest.param('left研究😀right', 'left研究😀right', id='unicode'),
    pytest.param(r'left\ud800right', r'left\ud800right', id='literal-backslash'),
])
@pytest.mark.parametrize('field', ['phase', 'detail'])
def test_activity_display_unicode_cannot_hide_status_table(tmp_path, field, text, expected):
    import contextlib
    import io
    from rpipe.flow.cli import main
    from rpipe.structure.artifact.index import write_index
    from rpipe.structure.artifact.result import write_result

    class StrictOutput:
        def __init__(self):
            self.buffer = io.BytesIO()
            self.stream = io.TextIOWrapper(self.buffer, encoding='utf-8', errors='strict', newline='\n')
        def write(self, value):
            return self.stream.write(value)
        def flush(self):
            self.stream.flush()

    write_index(tmp_path, {'study': 'display', 'experiments': [
        {'factors': {'algorithm.mode': 'train'}, 'runs': [{'id': 'peer', 'seed': 0}]}]})
    write_result(tmp_path / 'runs/peer/result.json',
                 {'status': 'succeeded', 'control': {}, 'metrics': {'accuracy': 1}, 'paths': {}})
    hint = {'phase': 'make', 'detail': 'ready', field: text}
    (tmp_path / 'activity.json').write_text(json.dumps(hint), encoding='utf-8')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    sink = StrictOutput()
    with contextlib.redirect_stdout(sink):
        assert main(['status', str(tmp_path)]) == 0
    sink.flush()
    view = sink.buffer.getvalue().decode('utf-8')
    assert expected in view and 'peer' in view and 'succeeded=1' in view
    assert 'accuracy=1.0000' in view
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('value', [chr(0xD800), chr(0xDC00)], ids=['high', 'low'])
def test_activity_writer_keeps_strict_encoding_despite_display_fallback(tmp_path, value):
    from rpipe.structure.artifact.activity import announce
    target = tmp_path / 'activity.json'
    target.write_bytes(b'{"phase": "make", "detail": "old evidence"}\r\n')
    old = target.read_bytes()
    with pytest.raises(UnicodeEncodeError):
        announce(tmp_path, 'make', value)
    assert target.read_bytes() == old
    assert not list(tmp_path.glob('*.tmp'))


def test_activity_whole_read_preserves_cancel_and_io_fallback(tmp_path, reader_exit_case, reader_close_fault):
    case = reader_exit_case
    path, peer = tmp_path / 'activity.json', tmp_path / 'peer'
    payload = b'{"phase":"prepare","detail":"synthetic"}'
    path.write_bytes(payload)
    peer.write_bytes(b'peer unchanged')
    expected = case.expected if case.kind in ('cancel', 'close-programming') else None
    with reader_close_fault(path, case, 'r') as (handles, closes):
        if expected is not None:
            with pytest.raises(type(expected)) as caught:
                read_activity(tmp_path)
            assert caught.value is expected
        else:
            assert read_activity(tmp_path) == ({'phase':'prepare','detail':'synthetic'}
                                              if case.kind == 'success' else None)
    assert len(handles) == len(closes) == 1 and handles[0].closed
    assert path.read_bytes() == payload and peer.read_bytes() == b'peer unchanged'
