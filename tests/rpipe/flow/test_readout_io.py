"""Read-only CLI status/logs keep valid evidence when one optional log is inaccessible."""
from pathlib import Path

import pytest

from rpipe.flow.cli import main
from rpipe.structure.artifact.index import write_index
from rpipe.structure.artifact.result import write_result

pytestmark = [pytest.mark.integration, pytest.mark.content, pytest.mark.p1,
              pytest.mark.flow_layer, pytest.mark.module_cli,
              pytest.mark.cost(cost_class='c1'),
              pytest.mark.result_type('categorical', detail='summary')]


@pytest.mark.parametrize('command', ['status', 'logs'])
@pytest.mark.parametrize('operation', ['stat', 'read'])
def test_native_readout_denied_log_preserves_results_and_peer_output(tmp_path, monkeypatch, capsys, command, operation):
    """CLI -> indexed Runs -> one denied log; keep the failed/succeeded results, peer output and every file byte."""
    write_index(tmp_path, {'study': 'readout', 'experiments': [{'factors': {'algorithm.mode': 'eval'},
                 'runs': [{'id': 'bad'}, {'id': 'peer'}]}]})
    for run, status in [('bad', 'failed'), ('peer', 'succeeded')]:
        write_result(tmp_path / 'runs' / run / 'result.json', {'status': status, 'control': {},
                     'paths': {}, 'metrics': {}, 'error': 'RuntimeError: original failure'})
        log = tmp_path / 'runs' / run / 'assets/logs/run.log'
        log.parent.mkdir(parents=True)
        log.write_text(f'[flow] {run} event\n', encoding='utf-8')
    entries = set(tmp_path.rglob('*'))
    files = [p for p in entries if p.is_file()]
    before = {p: p.read_bytes() for p in files}
    denied_log = tmp_path / 'runs/bad/assets/logs/run.log'
    method = 'is_file' if operation == 'stat' else 'read_text'
    original = getattr(Path, method)
    def denied(path, *args, **kwargs):
        if path == denied_log:
            raise PermissionError('optional log denied')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, method, denied)
    assert main([command, str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    if command == 'status':
        assert 'succeeded=1 failed=1 pending=0' in captured.out
        assert 'RuntimeError: original failure' in captured.out
    else:
        assert captured.out == '[flow] peer event\n'
    assert {p: p.read_bytes() for p in files} == before
    assert set(tmp_path.rglob('*')) == entries


@pytest.mark.parametrize('metrics', [[1], [['accuracy', 0.9]]])
def test_native_status_metric_shape_keeps_filtered_peer_and_bytes(tmp_path, capsys, metrics):
    """Public result writer -> CLI status filter -> unavailable cell, unchanged success/peer and all input bytes."""
    write_index(tmp_path, {'study': 'metric_shape', 'experiments': [{'factors': {'algorithm.mode': 'eval'},
                 'runs': [{'id': 'odd'}, {'id': 'peer'}]}]})
    for run, values in [('odd', metrics), ('peer', {'accuracy': 0.8})]:
        write_result(tmp_path / 'runs' / run / 'result.json', {'status': 'succeeded', 'control': {},
                     'paths': {}, 'metrics': values})
    entries = set(tmp_path.rglob('*'))
    files = [p for p in entries if p.is_file()]
    before = {p: p.read_bytes() for p in files}
    assert main(['status', str(tmp_path), '--mode', 'eval']) == 0
    captured = capsys.readouterr()
    assert not captured.err and 'planned=2 succeeded=2 failed=0 pending=0' in captured.out
    rows = {row.split('\t')[3]: row.split('\t') for row in captured.out.splitlines()[2:]}
    assert rows['odd'][0] == 'succeeded' and rows['odd'][5] == '-'
    assert rows['peer'][5] == 'accuracy=0.8000'
    assert {p: p.read_bytes() for p in files} == before
    assert set(tmp_path.rglob('*')) == entries
