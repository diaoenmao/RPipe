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
    'symlink-outside', 'false', 'zero', 'true', 'empty-list', 'empty-map', 'nonempty-map',
]
_GOOD_LOG_REFERENCES = ['missing', 'null', 'empty', 'valid', 'absolute-own', 'unicode-space', 'internal-alias', 'nonexistent']
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
           'tab': own + '\tFAKE-CELL', 'false': False, 'zero': 0, 'true': True,
           'empty-list': [], 'empty-map': {}, 'nonempty-map': {'path': own},
           'null': None, 'empty': '', 'valid': own, 'absolute-own': str((study / own).resolve()),
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
    return study, relative


def _assert_reference_view(study, variant, reader, invalid, relative):
    from urllib.parse import quote
    from rpipe.structure.artifact.readout.status import list_runs, format_status
    from rpipe.structure.artifact.readout.logs import event_lines
    from rpipe.structure.artifact.readout.report import render_numbers

    before = {p: p.read_bytes() for p in study.parent.rglob('*') if p.is_file()}
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
    assert {p: p.read_bytes() for p in study.parent.rglob('*') if p.is_file()} == before


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
