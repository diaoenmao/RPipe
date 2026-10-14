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
