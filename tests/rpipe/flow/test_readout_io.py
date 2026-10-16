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
    method = 'is_file' if operation == 'stat' else 'open'
    original = getattr(Path, method)
    def denied(path, *args, **kwargs):
        if kwargs.get('mode', args[0] if args else 'r') not in ('r', 'rt'):
            return original(path, *args, **kwargs)
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


@pytest.fixture
def strict_readout_case(tmp_path, command):
    import json

    write_index(tmp_path, {'study': 'synthetic', 'experiments': []})
    process = tmp_path / 'process.json'
    process.write_text(json.dumps({'study': 'synthetic', 'experiments': []}), encoding='utf-8')
    numbers = tmp_path / 'docs/NUMBERS.md'
    numbers.parent.mkdir(exist_ok=True)
    numbers.write_bytes(b'previous numbers evidence')
    output = tmp_path / 'comparison.json'
    output.write_bytes(b'previous comparison evidence')
    if command in ('status', 'logs'):
        bad, argv = tmp_path / 'index.json', [command, str(tmp_path)]
    elif command == 'report':
        bad, argv = process, [command, str(tmp_path)]
    else:
        import torch
        for run in ('a', 'b'):
            root = tmp_path / 'runs' / run
            write_result(root / 'result.json', {'status': 'succeeded', 'control': {}, 'metrics': {'accuracy': 42.0}, 'paths': {}})
            state = root / 'assets/tracker/tracker_state.json'
            state.parent.mkdir(parents=True)
            state.write_text('{"splits":{}}', encoding='utf-8')
            checkpoint = root / 'assets/checkpoints/latest.pt'
            checkpoint.parent.mkdir()
            torch.save({}, checkpoint)
        bad = tmp_path / 'runs/a/result.json'
        argv = [command, str(bad.parent), str(tmp_path / 'runs/b'), '--out', str(output)]
    return bad, argv, numbers, output


@pytest.mark.parametrize('command', ['status', 'logs', 'report', 'compare'])
@pytest.mark.parametrize('corruption', ['deep', 'syntax', 'nonobject'])
def test_strict_readout_parse_refusal_keeps_exit2_and_all_artifacts(tmp_path, strict_readout_case, capsys, command, corruption):
    """Actual readout CLI -> unusable authoritative JSON -> diagnostic exit2 with original inputs and prior outputs unchanged."""
    bad, argv, _, _ = strict_readout_case
    raw = b'[]' if corruption == 'nonobject' else b'{broken' if corruption == 'syntax' else b'{"unused":' + b'[' * 12000 + b'0' + b']' * 12000 + b'}'
    bad.write_bytes(raw)
    before = {f: f.read_bytes() for f in tmp_path.rglob('*') if f.is_file()}
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.err.strip() and not captured.out
    assert {f: f.read_bytes() for f in before} == before


@pytest.mark.parametrize('command', ['status', 'logs', 'report', 'compare'])
def test_strict_readout_does_not_swallow_programming_runtime_error(tmp_path, strict_readout_case, monkeypatch, command):
    """Readout CLI keeps non-parse RuntimeError identity; no prior input/output is rewritten."""
    bad, argv, _, _ = strict_readout_case
    before = {f: f.read_bytes() for f in tmp_path.rglob('*') if f.is_file()}
    error = RuntimeError('unexpected readout programming error')
    original = Path.open
    def refuse(path, *args, **kwargs):
        if path == bad:
            raise error
        return original(path, *args, **kwargs)
    with monkeypatch.context() as fault:
        fault.setattr(Path, 'open', refuse)
        with pytest.raises(RuntimeError) as caught:
            main(argv)
        assert caught.value is error
    assert {f: f.read_bytes() for f in before} == before


@pytest.mark.parametrize('command', ['status', 'logs', 'report', 'compare'])
def test_strict_readout_valid_control_keeps_success_contract(tmp_path, strict_readout_case, capsys, command):
    """Valid synthetic readout CLI -> existing output/success semantics; source JSON/checkpoint bytes retained."""
    _, argv, numbers, output = strict_readout_case
    before = {f: f.read_bytes() for f in tmp_path.rglob('*') if f.is_file() and f not in (numbers, output)}
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert not captured.err
    if command == 'status':
        assert 'planned=0' in captured.out
    elif command == 'report':
        assert numbers.read_bytes() != b'previous numbers evidence'
    elif command == 'compare':
        assert output.read_bytes() != b'previous comparison evidence' and 'passed' in captured.out
    assert {f: f.read_bytes() for f in before} == before


@pytest.mark.parametrize('payload', ['syntax', 'deep', 'valid'])
def test_optional_launch_counts_keep_parse_failure_best_effort(tmp_path, capsys, payload):
    """CLI count display -> optional index parse refusal -> no print or mutation; valid counts still print."""
    from rpipe.flow.cli import _print_launch_counts

    raw = (b'{broken' if payload == 'syntax' else b'{"unused":' + b'[' * 12000 + b'0' + b']' * 12000 + b'}'
           if payload == 'deep' else b'{"experiments":[]}')
    index = tmp_path / 'index.json'
    index.write_bytes(raw)
    assert _print_launch_counts(tmp_path) is None
    captured = capsys.readouterr()
    assert not captured.err
    assert ('planned=0' in captured.out) if payload == 'valid' else not captured.out
    assert index.read_bytes() == raw


def test_optional_launch_counts_does_not_hide_programming_error(tmp_path, monkeypatch):
    """Optional CLI count display keeps non-parse RuntimeError identity without changing index bytes."""
    from rpipe.flow.cli import _print_launch_counts

    index = tmp_path / 'index.json'
    index.write_bytes(b'{"experiments":[]}')
    error = RuntimeError('unexpected count programming error')
    original = Path.open
    def refuse(path, *args, **kwargs):
        if path == index:
            raise error
        return original(path, *args, **kwargs)
    with monkeypatch.context() as fault:
        fault.setattr(Path, 'open', refuse)
        with pytest.raises(RuntimeError) as caught:
            _print_launch_counts(tmp_path)
        assert caught.value is error
    assert index.read_bytes() == b'{"experiments":[]}'
