from pathlib import Path

import pytest

from rpipe.flow.cli import main
from rpipe.structure.artifact.index import write_index

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_artifact,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]


def test_logs_prints_events_in_time_order(tmp_path: Path, capsys):
    study = tmp_path / 'demo'
    for run_id, stamp, event in (
        ('late', '2026-09-28T04:00:02.000+08:00', '[epoch] 1 [split] test [metric] Accuracy=0.2000'),
        ('early', '2026-09-28T04:00:01.000+08:00', '[flow] start phases=prepare pid=1'),
    ):
        folder = study / 'runs' / run_id / 'assets' / 'logs'
        folder.mkdir(parents=True)
        (folder / 'run.log').write_text(
            f'{stamp} INFO  {run_id} {event}\n'
            f'{stamp} ERROR {run_id} [error] Traceback (most recent call last):\n',
            encoding='utf-8',
        )
    write_index(
        study,
        {
            'study': 'demo',
            'experiments': [
                {
                    'factors': {},
                    'runs': [
                        {'id': 'late', 'run_dir': 'late', 'log': 'runs/late/assets/logs/run.log'},
                        {'id': 'early', 'run_dir': 'early', 'log': 'runs/early/assets/logs/run.log'},
                    ],
                }
            ],
        },
    )
    assert main(['logs', str(study)]) == 0
    out = capsys.readouterr().out
    assert out.index('[flow] start') < out.index('[epoch] 1')
    assert 'Traceback' not in out
    assert not (study / 'logs').exists()
    assert main(['logs', str(tmp_path / 'missing')]) == 2


@pytest.mark.parametrize('operation', ['stat', 'read'])
def test_logs_skip_only_unreadable_run_and_keep_peer_events(tmp_path, monkeypatch, operation):
    from rpipe.structure.artifact.readout.logs import event_lines
    write_index(tmp_path, {'study': 'logs', 'experiments': [{'factors': {}, 'runs': [{'id': 'bad'}, {'id': 'peer'}]}]})
    logs = [tmp_path / 'runs' / run / 'assets/logs/run.log' for run in ['bad', 'peer']]
    for path in logs:
        path.parent.mkdir(parents=True)
        path.write_text(f'[flow] {path.parts[-4]} event\n', encoding='utf-8')
    before = {p: p.read_bytes() for p in logs}
    method = 'is_file' if operation == 'stat' else 'read_text'
    original = getattr(Path, method)
    def denied(path, *args, **kwargs):
        if path == logs[0]:
            raise PermissionError('log denied')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, method, denied)
    assert event_lines(tmp_path) == ['[flow] peer event']
    assert {p: p.read_bytes() for p in logs} == before


def test_logs_unexpected_probe_error_is_not_hidden(tmp_path, monkeypatch):
    from rpipe.structure.artifact.readout.logs import event_lines
    write_index(tmp_path, {'study': 'logs', 'experiments': [{'factors': {}, 'runs': [{'id': 'bad'}]}]})
    path = tmp_path / 'runs/bad/assets/logs/run.log'
    original = Path.is_file
    def broken(candidate):
        if candidate == path:
            raise RuntimeError('unexpected log probe failure')
        return original(candidate)
    monkeypatch.setattr(Path, 'is_file', broken)
    with pytest.raises(RuntimeError, match='unexpected log probe failure'):
        event_lines(tmp_path)


@pytest.mark.parametrize('reader', ['note', 'events'])
def test_log_decoder_programming_typeerror_is_not_optional_io(tmp_path, monkeypatch, reader):
    from rpipe.structure.artifact.readout.status import _note_from_log
    from rpipe.structure.artifact.readout.logs import event_lines
    write_index(tmp_path, {'study': 'logs', 'experiments': [{'runs': [{'id': 'bad'}]}]})
    path = tmp_path / 'runs/bad/assets/logs/run.log'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'event')
    original = Path.read_text
    def broken(candidate, *args, **kwargs):
        if candidate == path:
            raise TypeError('decoder programming error')
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', broken)
    with pytest.raises(TypeError, match='decoder programming error'):
        _note_from_log(path) if reader == 'note' else event_lines(tmp_path)



_BAD_LOG_REFERENCES = [
    'outside-relative', 'outside-absolute', 'peer', 'newline', 'tab',
    'symlink-outside', 'unicode-line', 'unicode-next-line', 'false', 'zero', 'true', 'empty-list', 'empty-map', 'nonempty-map',
]
_GOOD_LOG_REFERENCES = ['missing', 'null', 'empty', 'valid', 'absolute-own', 'unicode-space', 'internal-alias', 'internal-parent', 'study-alias', 'nonexistent']
_REFERENCE_READERS = ['status', 'logs', 'report']


def _reference_study(tmp_path, variant):
    from pathlib import Path
    import json

    study = tmp_path / 'study'
    for run, label in [('bad', 'OWN-BAD'), ('peer', 'PEER-EVIDENCE')]:
        folder = study / 'runs' / run / 'assets/logs'
        folder.mkdir(parents=True)
        (folder / 'run.log').write_text(f'2026-01-01T00:00:00+00:00 INFO - [error] {label}\n', encoding='utf-8')
    outside = tmp_path / 'outside.log'
    outside.write_text('2026-01-01T00:00:00+00:00 INFO - [error] OUTSIDE-EVIDENCE\n', encoding='utf-8')
    own = 'runs/bad/assets/logs/run.log'
    raw = {'outside-relative': '../outside.log', 'outside-absolute': str(outside.resolve()),
           'peer': 'runs/peer/assets/logs/run.log', 'newline': own + '\nFAKE-ROW',
           'tab': own + '\tFAKE-CELL', 'unicode-line': own + '\u2028FAKE-ROW', 'unicode-next-line': own + '\x85FAKE-ROW', 'false': False, 'zero': 0, 'true': True,
           'empty-list': [], 'empty-map': {}, 'nonempty-map': {'path': own},
           'null': None, 'empty': '', 'valid': own, 'absolute-own': str((study / own).resolve()),
           'internal-parent': 'runs/bad/assets/logs/../logs/run.log',
           'nonexistent': 'runs/bad/assets/logs/later.log'}.get(variant)
    if variant == 'unicode-space':
        raw = 'runs/bad/assets/logs/日志 数据.log'
        (study / raw).write_bytes((study / own).read_bytes())
    elif variant in ('symlink-outside', 'internal-alias'):
        raw = 'runs/bad/assets/logs/alias.log'
        try:
            (study / raw).symlink_to(outside if variant == 'symlink-outside' else study / own)
        except OSError:
            pytest.skip('host does not allow file symlinks')
    bad = {'id': 'bad', 'run_dir': 'bad', 'seed': 0}
    if variant != 'missing':
        bad['log'] = raw
    factors = {'algorithm.mode': 'train'}
    peer = {'id': 'peer', 'run_dir': 'peer', 'seed': 1, 'log': 'runs/peer/assets/logs/run.log'}
    (study / 'index.json').write_text(json.dumps({'study': 'synthetic', 'experiments': [
        {'factors': factors, 'runs': [bad, peer]}]}), encoding='utf-8')
    process = {'study': 'synthetic', 'experiments': [{'factors': factors, 'metrics': {}, 'runs': [
        {'id': 'bad', 'seed': 0, 'status': 'succeeded', 'metrics': {}},
        {'id': 'peer', 'seed': 1, 'status': 'succeeded', 'metrics': {}},
    ]}]}
    (study / 'process.json').write_text(json.dumps(process), encoding='utf-8')
    relative = raw if variant not in ('missing', 'null', 'empty', 'absolute-own') else own
    if variant == 'study-alias':
        alias = tmp_path / 'study-alias'
        try:
            alias.symlink_to(study, target_is_directory=True)
        except OSError:
            pytest.skip('host does not allow directory symlinks')
        study, relative = alias, own
    return study, relative


def _assert_reference_view(study, variant, reader, invalid, relative):
    from urllib.parse import quote
    from rpipe.structure.artifact.readout.status import list_runs, format_status
    from rpipe.structure.artifact.readout.logs import event_lines
    from rpipe.structure.artifact.readout.report import render_numbers

    before = {p: p.readlink() if p.is_symlink() else p.read_bytes()
              for p in study.parent.rglob('*') if p.is_symlink() or p.is_file()}
    unavailable = invalid or variant == 'nonexistent'
    if reader == 'status':
        body = list_runs(study)
        bad, peer = body['runs']
        assert body['counts'] == {'planned': 2, 'succeeded': 0, 'failed': 0, 'pending': 2}
        assert bad['note'] == ('-' if unavailable else '[error] OWN-BAD')
        assert peer['note'] == '[error] PEER-EVIDENCE'
        if invalid:
            assert bad['log'] in (None, '-')
        assert len(format_status(body).splitlines()) == 4
    elif reader == 'logs':
        lines = event_lines(study)
        assert len(lines) == (1 if unavailable else 2)
        assert sum('PEER-EVIDENCE' in x for x in lines) == 1
        assert sum('OWN-BAD' in x for x in lines) == (0 if unavailable else 1)
        assert not any('OUTSIDE-EVIDENCE' in x for x in lines)
    else:
        text = render_numbers(study)
        row = next(x for x in text.splitlines() if '`bad`' in x)
        peer = next(x for x in text.splitlines() if '`peer`' in x)
        assert '[run.log](../runs/peer/assets/logs/run.log)' in peer
        if invalid:
            assert row.endswith('| - |')
        else:
            assert '[run.log](../' + quote(relative, safe='/') + ')' in row
    assert {p: p.readlink() if p.is_symlink() else p.read_bytes()
              for p in study.parent.rglob('*') if p.is_symlink() or p.is_file()} == before


@pytest.mark.parametrize('variant', _BAD_LOG_REFERENCES)
@pytest.mark.parametrize('reader', _REFERENCE_READERS)
def test_log_reference_bad_metadata_is_one_unavailable_cell(tmp_path, variant, reader):
    study, relative = _reference_study(tmp_path, variant)
    _assert_reference_view(study, variant, reader, True, relative)


@pytest.mark.parametrize('variant', _GOOD_LOG_REFERENCES)
@pytest.mark.parametrize('reader', _REFERENCE_READERS)
def test_log_reference_supported_paths_remain_usable(tmp_path, variant, reader):
    study, relative = _reference_study(tmp_path, variant)
    _assert_reference_view(study, variant, reader, False, relative)



@pytest.mark.parametrize('reader', _REFERENCE_READERS)
@pytest.mark.parametrize('fault', ['io', 'programming'])
def test_log_reference_path_resolution_error_boundary(tmp_path, monkeypatch, reader, fault):
    from pathlib import Path
    from rpipe.structure.artifact.readout.status import list_runs
    from rpipe.structure.artifact.readout.logs import event_lines
    from rpipe.structure.artifact.readout.report import render_numbers

    study, relative = _reference_study(tmp_path, 'valid')
    bad = study / 'runs/bad/assets/logs/run.log'
    original = Path.resolve
    error = PermissionError('optional reference unavailable') if fault == 'io' else RuntimeError('programming bug')

    def broken_resolve(path, *args, **kwargs):
        if path == bad:
            raise error
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'resolve', broken_resolve)
    if fault == 'io':
        _assert_reference_view(study, 'valid', reader, True, relative)
    else:
        call = {'status': list_runs, 'logs': event_lines, 'report': render_numbers}[reader]
        with pytest.raises(RuntimeError) as caught:
            call(study)
        assert caught.value is error



@pytest.mark.parametrize('reader', _REFERENCE_READERS)
def test_log_reference_cycle_is_unavailable_without_hiding_peer(tmp_path, reader):
    from pathlib import Path
    import json

    study, _ = _reference_study(tmp_path, 'valid')
    linked = study / 'runs/bad/assets/logs/cyclic.log'
    try:
        linked.symlink_to(linked)
    except OSError:
        pytest.skip('host does not allow file symlinks')
    index = study / 'index.json'
    body = json.loads(index.read_text(encoding='utf-8'))
    body['experiments'][0]['runs'][0]['log'] = 'runs/bad/assets/logs/cyclic.log'
    index.write_text(json.dumps(body), encoding='utf-8')
    target = linked.readlink()
    _assert_reference_view(study, 'cycle', reader, True, 'runs/bad/assets/logs/cyclic.log')
    assert linked.is_symlink() and linked.readlink() == target



@pytest.mark.parametrize('reader', _REFERENCE_READERS)
def test_log_reference_directory_is_unavailable(tmp_path, reader):
    import json

    study, _ = _reference_study(tmp_path, 'valid')
    index = study / 'index.json'
    body = json.loads(index.read_text(encoding='utf-8'))
    body['experiments'][0]['runs'][0]['log'] = 'runs/bad/assets/logs'
    index.write_text(json.dumps(body), encoding='utf-8')
    _assert_reference_view(study, 'directory', reader, True, 'runs/bad/assets/logs')


@pytest.mark.parametrize('reader', _REFERENCE_READERS)
def test_log_reference_extended_windows_alias_keeps_owned_input(tmp_path, reader):
    import os
    import json
    from pathlib import Path

    if os.name != 'nt':
        pytest.skip('Windows extended path alias semantics')
    study, relative = _reference_study(tmp_path, 'valid')
    extended = str(Path('//?/' + (study / relative).resolve().as_posix()))
    assert Path(extended).samefile(study / relative)
    index = study / 'index.json'
    body = json.loads(index.read_text(encoding='utf-8'))
    body['experiments'][0]['runs'][0]['log'] = extended
    index.write_text(json.dumps(body), encoding='utf-8')
    _assert_reference_view(study, 'extended', reader, False, relative)



@pytest.mark.parametrize('server,share,owned', [('SERVER', 'SHARE', True), ('server', 'other', False), ('other', 'share', False)])
def test_log_reference_windows_namespace_preserves_share_ownership(server, share, owned):
    import os
    from pathlib import Path

    if os.name != 'nt':
        pytest.skip('Windows namespace path comparison semantics')
    from rpipe.structure.artifact.readout.status import _log_path_key

    candidate = Path(f'//?/UNC/{server}/{share}/study/runs/bad/assets/logs/run.log')
    owner = Path('//server/share/study/runs/bad')
    if owned:
        assert _log_path_key(candidate).relative_to(_log_path_key(owner)) == Path('assets/logs/run.log')
    else:
        with pytest.raises(ValueError):
            _log_path_key(candidate).relative_to(_log_path_key(owner))


_LOG_TIME_PAIRS = [
    ('dst-fold', '2026-11-01T01:45:00.000-04:00', '2026-11-01T01:15:00.000-05:00'),
    ('cross-day', '2026-01-02T00:15:00.000+02:00', '2026-01-01T23:00:00.000+00:00'),
    ('cross-offset', '2026-01-01T11:00:00.000+09:00', '2026-01-01T04:00:00.000+00:00'),
    ('utc-z-fraction', '2026-01-01T04:00:00Z', '2026-01-01T04:00:00.100Z'),
    ('same-positive', '2026-01-01T04:00:01.000+08:00', '2026-01-01T04:00:02.000+08:00'),
    ('same-negative', '2026-01-01T04:00:01.000-05:00', '2026-01-01T04:00:02.000-05:00'),
    ('same-utc', '2026-01-01T04:00:01.000+00:00', '2026-01-01T04:00:02.000+00:00'),
    ('legacy-naive', '2026-01-01T04:00:01.000', '2026-01-01T04:00:02.000'),
]


def _log_time_inputs(study, early_stamp, late_stamp):
    write_index(study, {'study': 'synthetic', 'experiments': [
        {'factors': {}, 'runs': [{'id': 'late'}, {'id': 'early'}]}]})
    lines = {}
    for rid, stamp in [('early', early_stamp), ('late', late_stamp)]:
        path = study / 'runs' / rid / 'assets/logs/run.log'
        path.parent.mkdir(parents=True)
        lines[rid] = (stamp + ' ' if stamp else '') + f'INFO  {rid} [flow] {rid.upper()}'
        path.write_text(lines[rid] + '\n', encoding='utf-8')
    entries = set(study.rglob('*'))
    before = {f: f.read_bytes() for f in entries if f.is_file()}
    return lines, entries, before


def _read_log_time_entry(study, entry, capsys):
    if entry == 'library':
        from rpipe.structure.artifact.readout.logs import event_lines
        return event_lines(study)
    assert main(['logs', str(study)]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    return captured.out.splitlines()


@pytest.mark.parametrize('entry', ['library', 'cli'])
@pytest.mark.parametrize('case,early,late', _LOG_TIME_PAIRS, ids=[r[0] for r in _LOG_TIME_PAIRS])
def test_logs_aware_time_order_keeps_original_lines_and_input_bytes(tmp_path, capsys, entry, case, early, late):
    """Library/actual CLI -> indexed native-format offset events -> actual time order, original line text and bytes."""
    lines, entries, before = _log_time_inputs(tmp_path, early, late)
    assert _read_log_time_entry(tmp_path, entry, capsys) == [lines['early'], lines['late']]
    assert {f: f.read_bytes() for f in before} == before
    assert set(tmp_path.rglob('*')) == entries


@pytest.mark.parametrize('entry', ['library', 'cli'])
@pytest.mark.parametrize('case', ['missing', 'malformed', 'invalid-date', 'sub-microsecond', 'utc-overflow', 'equal-stamp'])
def test_logs_timestamp_fallback_and_ties_retain_existing_text_order(tmp_path, capsys, entry, case):
    """Unusable/legacy timestamp keys and equal stamps keep prior fallback/order, without dropping peer evidence."""
    stamps = {
        'missing': ('', ''),
        'malformed': ('2026-01-01Tbad-A', '2026-01-01Tbad-B'),
        'invalid-date': ('2026-02-30T00:00:01+00:00', '2026-02-30T00:00:02+00:00'),
        'sub-microsecond': ('2026-01-01T00:00:00.1234561+00:00', '2026-01-01T00:00:00.1234562+00:00'),
        'utc-overflow': ('0001-01-01T00:00:01+14:00', '0001-01-01T00:00:02+14:00'),
        'equal-stamp': ('2026-01-01T00:00:01.000+08:00', '2026-01-01T00:00:01.000+08:00'),
    }[case]
    lines, entries, before = _log_time_inputs(tmp_path, *stamps)
    order = ['late', 'early'] if case in ('missing', 'equal-stamp') else ['early', 'late']
    assert _read_log_time_entry(tmp_path, entry, capsys) == [lines[rid] for rid in order]
    assert {f: f.read_bytes() for f in before} == before
    assert set(tmp_path.rglob('*')) == entries


def test_logs_whole_read_preserves_cancel_and_io_fallback(tmp_path, reader_exit_case, reader_close_fault):
    from rpipe.structure.artifact.readout.logs import event_lines
    case = reader_exit_case
    path = tmp_path / 'runs/r1/assets/logs/run.log'
    path.parent.mkdir(parents=True)
    payload = b'2026-10-17T00:00:00+00:00 INFO  r1 [flow] synthetic\n'
    path.write_bytes(payload)
    (tmp_path / 'index.json').write_bytes(b'{"experiments":[{"runs":[{"id":"r1","run_dir":"r1"}]}]}')
    before = {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    baseline = event_lines(tmp_path)
    assert baseline
    expected = case.expected if case.kind in ('cancel', 'close-programming') else None
    with reader_close_fault(path, case, 'r') as (handles, closes):
        if expected is not None:
            with pytest.raises(type(expected)) as caught:
                event_lines(tmp_path)
            assert caught.value is expected
        else:
            assert event_lines(tmp_path) == (baseline if case.kind == 'success' else [])
    assert len(handles) == len(closes) == 1 and handles[0].closed
    assert {p:p.read_bytes() for p in before} == before
