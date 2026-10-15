from pathlib import Path

import pytest

from rpipe.flow.cli import main
from rpipe.structure.artifact.readout import format_status, list_runs
from rpipe.structure.artifact.index import write_index
from rpipe.structure.artifact.result import write_result
from rpipe.structure.artifact.config import write_config

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_artifact,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]


def _write_study(tmp_path: Path) -> Path:
    study = tmp_path / 'demo'
    (study / 'runs' / 'ok' / 'assets' / 'logs').mkdir(parents=True)
    (study / 'runs' / 'bad' / 'assets' / 'logs').mkdir(parents=True)
    (study / 'runs' / 'wait' / 'assets' / 'logs').mkdir(parents=True)
    write_index(
        study,
        {
            'study': 'demo',
            'description': 'status',
            'experiments': [
                {
                    'factors': {
                        'data.config.train_size': 500,
                        'algorithm.mode': 'train',
                    },
                    'runs': [
                        {
                            'id': 'ok',
                            'seed': 0,
                            'run_dir': 'ok',
                            'log': 'runs/ok/assets/logs/run.log',
                        },
                        {
                            'id': 'wait',
                            'seed': 1,
                            'run_dir': 'wait',
                            'log': 'runs/wait/assets/logs/run.log',
                        },
                    ],
                },
                {
                    'factors': {
                        'data.config.train_size': 500,
                        'algorithm.mode': 'eval',
                    },
                    'runs': [
                        {
                            'id': 'bad',
                            'seed': 0,
                            'run_dir': 'bad',
                            'log': 'runs/bad/assets/logs/run.log',
                        }
                    ],
                },
            ],
        },
    )
    write_result(
        study / 'runs' / 'ok' / 'result.json',
        {
            'status': 'succeeded',
            'control': {'id': 'ok', 'algorithm': {'mode': 'train'}},
            'metrics': {'accuracy': 0.8},
            'paths': {},
        },
    )
    write_result(
        study / 'runs' / 'bad' / 'result.json',
        {
            'status': 'failed',
            'error': 'RuntimeError: boom',
            'control': {'id': 'bad', 'algorithm': {'mode': 'eval'}},
        },
    )
    return study


def test_list_runs_joins_index_and_result_status(tmp_path: Path):
    study = _write_study(tmp_path)
    body = list_runs(study)
    by_id = {row['id']: row for row in body['runs']}
    assert body['counts'] == {'planned': 3, 'succeeded': 1, 'failed': 1, 'pending': 1}
    assert by_id['ok']['status'] == 'succeeded'
    assert by_id['ok']['metric'] == 'accuracy=0.8000'
    assert by_id['bad']['status'] == 'failed'
    assert by_id['bad']['error'] == 'RuntimeError: boom'
    assert by_id['wait']['status'] == 'pending'
    assert by_id['wait']['note'] == '-'
    assert by_id['ok']['note'] == '-'
    assert by_id['bad']['note'] == '-'
    evals = list_runs(study, modes=['eval'])
    assert [row['id'] for row in evals['runs']] == ['bad']


def test_pending_note_is_last_epoch_or_error_not_traceback(tmp_path: Path):
    study = _write_study(tmp_path)
    (study / 'runs' / 'bad' / 'assets' / 'logs' / 'run.log').write_text(
        '\n'.join(
            [
                '2026-09-28T04:00:03.000+08:00 ERROR bad [error] phase=prepare RuntimeError: boom',
                '2026-09-28T04:00:03.001+08:00 ERROR bad [error] Traceback (most recent call last):',
                '2026-09-28T04:00:04.000+08:00 INFO  bad [epoch] 1 [split] test [metric] Accuracy=0.1000',
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    log = study / 'runs' / 'wait' / 'assets' / 'logs' / 'run.log'
    log.write_text(
        '\n'.join(
            [
                '2026-09-28T04:00:00.000+08:00 INFO  wait [flow] start phases=prepare pid=1',
                '2026-09-28T04:00:01.000+08:00 INFO  wait [epoch] 2 [split] train [metric] Loss=1.2000',
                '2026-09-28T04:00:02.000+08:00 ERROR wait [error] phase=execute RuntimeError: boom',
                '2026-09-28T04:00:02.001+08:00 ERROR wait [error] Traceback (most recent call last):',
                '2026-09-28T04:00:02.002+08:00 ERROR wait [error]   File "train.py", line 1, in run',
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    body = list_runs(study)
    by_id = {row['id']: row for row in body['runs']}
    assert by_id['wait']['note'] == '[error] phase=execute RuntimeError: boom'
    assert by_id['ok']['note'] == '-'
    assert by_id['bad']['note'] == '[error] phase=prepare RuntimeError: boom'
    assert by_id['bad']['error'] == 'RuntimeError: boom'
    printed = format_status(body)
    print(printed, end='')
    assert 'note' in printed.splitlines()[1]
    assert '[error] phase=execute RuntimeError: boom' in printed
    assert 'Traceback' not in printed
    assert 'File "train.py"' not in printed

    log.write_text(
        '2026-09-28T04:00:01.000+08:00 INFO  wait [flow] start phases=prepare pid=1\n'
        '2026-09-28T04:00:08.000+08:00 INFO  wait [epoch] 2 [split] train [metric] Loss=1.2000 [time] elapsed=0:00:07\n',
        encoding='utf-8',
    )
    training = format_status(list_runs(study))
    print(training, end='')
    assert '[epoch] 2 [split] train [metric] Loss=1.2000 [time] elapsed=0:00:07' in training


def test_status_cli_prints_table(tmp_path: Path, capsys):
    study = _write_study(tmp_path)
    assert main(['status', str(study)]) == 0
    out = capsys.readouterr().out
    assert 'planned=3' in out
    assert 'succeeded=1' in out
    assert 'ok' in out
    assert 'pending' in out
    assert main(['status', str(study), '--mode', 'eval']) == 0
    filtered = capsys.readouterr().out
    assert 'bad' in filtered
    assert '\tok\t' not in filtered
    assert main(['status', str(tmp_path / 'missing')]) == 2


def test_status_cli_prints_activity_while_index_is_missing(tmp_path: Path, capsys):
    study = tmp_path / 'demo'
    study.mkdir()
    (study / 'activity.json').write_text(
        '{"phase": "make", "detail": "shared CIFAR10 download"}\n',
        encoding='utf-8',
    )
    assert main(['status', str(study)]) == 0
    assert capsys.readouterr().out.strip() == 'make: shared CIFAR10 download'


def _planned_run(study, config):
    write_index(study, {'study': 'planned', 'experiments': [{'factors': {}, 'runs': [{'id': 'r0', 'seed': 0}]}]})
    write_config(study / 'runs/r0/config.yaml', config)


@pytest.mark.parametrize('mode', ['eval', 'inference'])
def test_pending_mode_comes_from_config_without_mode_axis(tmp_path, mode):
    _planned_run(tmp_path, {'algorithm': {'mode': mode}})
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    row = list_runs(tmp_path)['runs'][0]
    assert row['mode'] == mode and row['status'] == 'pending'
    assert list_runs(tmp_path, modes=[mode])['counts']['planned'] == 1
    assert list_runs(tmp_path, modes=['train'])['counts']['planned'] == 0
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('bad', ['missing', 'yaml', 'mapping', 'algorithm', 'mode', 'empty-mode'])
def test_missing_or_malformed_mode_config_is_unknown_with_diagnostic(tmp_path, bad):
    _planned_run(tmp_path, {})
    path = tmp_path / 'runs/r0/config.yaml'
    if bad == 'missing':
        path.unlink()
    elif bad == 'yaml':
        path.write_text('algorithm: [\nprivate-token: do-not-print\n', encoding='utf-8')
    elif bad == 'mapping':
        path.write_text('[1, 2]\n', encoding='utf-8')
    elif bad == 'empty-mode':
        write_config(path, {'algorithm': {'mode': ''}})
    else:
        write_config(path, {'algorithm': []} if bad == 'algorithm' else {'algorithm': {'mode': ['eval']}})
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    row = list_runs(tmp_path)['runs'][0]
    assert row['mode'] == 'unknown' and row['status'] == 'pending'
    assert 'config' in row['error'] and 'config.yaml' in row['error']
    assert 'do-not-print' not in format_status(list_runs(tmp_path))
    assert list_runs(tmp_path, modes=['train'])['counts']['planned'] == 0
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


@pytest.mark.parametrize('config', [{}, {'algorithm': {}}, {'algorithm': {'mode': None}}])
def test_readable_default_mode_still_means_train(tmp_path, config):
    _planned_run(tmp_path, config)
    assert list_runs(tmp_path)['runs'][0]['mode'] == 'train'


@pytest.mark.parametrize('source', ['axis', 'result'])
def test_known_mode_does_not_require_legacy_config(tmp_path, source):
    _planned_run(tmp_path, {})
    (tmp_path / 'runs/r0/config.yaml').unlink()
    if source == 'axis':
        write_index(tmp_path, {'study': 'planned', 'experiments': [
            {'factors': {'algorithm.mode': 'eval'}, 'runs': [{'id': 'r0', 'seed': 0}]}]})
    else:
        write_result(tmp_path / 'runs/r0/result.json', {'status': 'succeeded',
                     'control': {'algorithm': {'mode': 'eval'}}, 'metrics': {}, 'paths': {}})
    row = list_runs(tmp_path)['runs'][0]
    assert row['mode'] == 'eval' and row['error'] is None


@pytest.mark.parametrize('broken', ['{bad json', '[]'])
def test_unreadable_result_uses_planned_config_mode(tmp_path, broken):
    _planned_run(tmp_path, {'algorithm': {'mode': 'eval'}})
    (tmp_path / 'runs/r0/result.json').write_text(broken, encoding='utf-8')
    row = list_runs(tmp_path)['runs'][0]
    assert row['mode'] == 'eval' and row['status'] == 'pending'


@pytest.mark.parametrize('status', ['failed', 'succeeded'])
def test_mode_diagnostic_preserves_execution_state_and_real_failure(tmp_path, status):
    _planned_run(tmp_path, {})
    (tmp_path / 'runs/r0/config.yaml').unlink()
    result_path = tmp_path / 'runs/r0/result.json'
    write_result(result_path, {'status': status, 'control': {}, 'metrics': {'accuracy': 0.8},
                              'paths': {}, 'error': 'RuntimeError: original failure'})
    before = result_path.read_bytes()
    row = list_runs(tmp_path)['runs'][0]
    assert row['status'] == status and row['mode'] == 'unknown'
    assert 'mode config unreadable' in row['error']
    assert ('original failure' in row['error']) == (status == 'failed')
    assert result_path.read_bytes() == before


@pytest.mark.parametrize('target', ['result', 'log'])
@pytest.mark.parametrize('operation', ['stat', 'read'])
def test_status_read_denial_keeps_other_runs_and_original_bytes(tmp_path, monkeypatch, target, operation):
    study = _write_study(tmp_path)
    log = study / 'runs/bad/assets/logs/run.log'
    log.write_text('[error] original failure\n', encoding='utf-8')
    path = study / 'runs/bad/result.json' if target == 'result' else log
    files = [p for p in study.rglob('*') if p.is_file()]
    before = {p: p.read_bytes() for p in files}
    method = 'is_file' if operation == 'stat' else 'read_text'
    original = getattr(Path, method)
    def denied(candidate, *args, **kwargs):
        if candidate == path:
            raise PermissionError('readout input denied')
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(Path, method, denied)
    body = list_runs(study)
    by_id = {row['id']: row for row in body['runs']}
    assert by_id['ok']['status'] == 'succeeded' and by_id['ok']['metric'] == 'accuracy=0.8000'
    assert by_id['bad']['mode'] == 'eval'
    if target == 'result':
        assert by_id['bad']['status'] == 'pending' and body['counts']['succeeded'] == 1
    else:
        assert by_id['bad']['status'] == 'failed' and by_id['bad']['error'] == 'RuntimeError: boom'
        assert by_id['bad']['note'] == '-'
    assert {p: p.read_bytes() for p in files} == before


@pytest.mark.parametrize('target', ['result', 'log'])
def test_status_unexpected_probe_error_is_not_hidden(tmp_path, monkeypatch, target):
    study = _write_study(tmp_path)
    path = study / ('runs/bad/result.json' if target == 'result' else 'runs/bad/assets/logs/run.log')
    original = Path.is_file
    def broken(candidate):
        if candidate == path:
            raise RuntimeError('unexpected probe failure')
        return original(candidate)
    monkeypatch.setattr(Path, 'is_file', broken)
    with pytest.raises(RuntimeError, match='unexpected probe failure'):
        list_runs(study)


@pytest.mark.parametrize('metrics,expected', [
    (None, '-'), ({}, '-'), ({'accuracy': 0.8}, 'accuracy=0.8000'),
    ([1], '-'), (1, '-'), (True, '-'), ('bad metrics', '-'),
    ([['accuracy', 0.9]], '-'),
])
def test_metric_shape_fallback_preserves_run_state_and_peer(tmp_path, metrics, expected):
    study = _write_study(tmp_path)
    write_result(study / 'runs/bad/result.json', {'status': 'succeeded', 'control': {},
                 'paths': {}, 'metrics': metrics})
    files = [p for p in study.rglob('*') if p.is_file()]
    before = {p: p.read_bytes() for p in files}
    body = list_runs(study)
    rows = {row['id']: row for row in body['runs']}
    assert body['counts'] == {'planned': 3, 'succeeded': 2, 'failed': 0, 'pending': 1}
    assert rows['bad']['status'] == 'succeeded' and rows['bad']['mode'] == 'eval'
    assert rows['bad']['metric'] == expected
    assert rows['ok']['metric'] == 'accuracy=0.8000'
    assert {p: p.read_bytes() for p in files} == before


@pytest.mark.parametrize('state,payload', [('available', b'{"value": 1}'), ('missing', None),
                                        ('unreadable', b'bad json'), ('unreadable', b'\xff')])
def test_optional_read_states_keep_input_bytes_and_strict_loader(tmp_path, state, payload):
    from rpipe.structure.artifact.readout.status import _read_optional
    from rpipe.structure.artifact.result import load_result

    path = tmp_path / 'result.json'
    if payload is not None:
        path.write_bytes(payload)
    actual, value = _read_optional(path, load_result)
    assert actual == state
    assert value == ({'value': 1} if state == 'available' else None)
    if payload is not None:
        assert path.read_bytes() == payload
        if state == 'unreadable':
            with pytest.raises((ValueError, UnicodeError)):
                load_result(path)
    else:
        assert not path.exists()


@pytest.mark.parametrize('source', ['axis', 'result'])
def test_planned_mode_conflict_is_local_and_not_silently_coerced(tmp_path, source):
    study = _write_study(tmp_path)
    path = study / 'runs/bad/config.yaml'
    write_config(path, {'algorithm': {'mode': 'train'}})
    # Both legacy sources say eval; config is now present and must be checked.
    if source == 'result':
        import json
        index = json.loads((study / 'index.json').read_text(encoding='utf-8'))
        index['experiments'][1]['factors'].pop('algorithm.mode')
        write_index(study, index)
    before = {p: p.read_bytes() for p in study.rglob('*') if p.is_file()}
    rows = {r['id']: r for r in list_runs(study)['runs']}
    assert rows['bad']['mode'] == 'unknown'
    assert 'conflict' in rows['bad']['error']
    assert rows['bad']['status'] == 'failed' and 'RuntimeError: boom' in rows['bad']['error']
    assert rows['ok']['mode'] == 'train' and rows['ok']['metric'] == 'accuracy=0.8000'
    assert [r['id'] for r in list_runs(study, modes=['eval'])['runs']] == []
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('field,value', [('id', 'another'), ('seed', 7), ('seed', True)])
def test_config_identity_conflict_does_not_change_disk_result_or_hide_peer(tmp_path, field, value):
    study = _write_study(tmp_path)
    cfg = {'id': 'bad', 'seed': 0, 'algorithm': {'mode': 'eval'}}
    cfg[field] = value
    write_config(study / 'runs/bad/config.yaml', cfg)
    before = {p: p.read_bytes() for p in study.rglob('*') if p.is_file()}
    rows = {r['id']: r for r in list_runs(study)['runs']}
    assert rows['bad']['mode'] == 'unknown' and 'conflict' in rows['bad']['error']
    assert rows['bad']['status'] == 'failed'
    assert rows['ok']['status'] == 'succeeded'
    assert {p: p.read_bytes() for p in before} == before


def test_unreadable_config_cannot_be_hidden_by_legacy_mode(tmp_path):
    study = _write_study(tmp_path)
    (study / 'runs/bad/config.yaml').write_bytes(b'algorithm: [')
    row = next(r for r in list_runs(study)['runs'] if r['id'] == 'bad')
    assert row['mode'] == 'unknown' and 'config' in row['error']


def test_optional_directory_is_unreadable_not_missing_and_programming_error_propagates(tmp_path):
    from rpipe.structure.artifact.readout.status import _read_optional

    assert _read_optional(tmp_path, lambda p: p.read_text()) == ('unreadable', None)
    def broken(path):
        raise RuntimeError('decoder bug')
    path = tmp_path / 'exists'
    path.write_bytes(b'value')
    with pytest.raises(RuntimeError, match='decoder bug'):
        _read_optional(path, broken)


@pytest.mark.parametrize('seed', [None, 0])
def test_unspecified_config_seed_keeps_planned_integer_seed(tmp_path, seed):
    _planned_run(tmp_path, {'seed': seed, 'algorithm': {'mode': 'eval'}})
    row = list_runs(tmp_path)['runs'][0]
    assert row['seed'] == 0 and type(row['seed']) is int and row['mode'] == 'eval'


def test_boolean_recorded_seed_is_not_equal_to_planned_zero(tmp_path):
    _planned_run(tmp_path, {'seed': 0, 'algorithm': {'mode': 'eval'}})
    write_result(tmp_path / 'runs/r0/result.json', {'status': 'succeeded',
                 'control': {'seed': False, 'algorithm': {'mode': 'eval'}}, 'metrics': {}, 'paths': {}})
    row = list_runs(tmp_path)['runs'][0]
    assert row['mode'] == 'unknown' and 'conflict' in row['error']


@pytest.mark.parametrize('value', [10**400, float('nan'), float('inf'), -float('inf')], ids=['overflow', 'nan', 'inf', 'negative-inf'])
@pytest.mark.parametrize('fallback', [False, True])
def test_unusable_numeric_metric_is_local_and_keeps_valid_fallback(tmp_path, capsys, value, fallback):
    study = _write_study(tmp_path)
    metrics = {'accuracy': value}
    if fallback:
        metrics['best_accuracy'] = 0.7
    write_result(study / 'runs/bad/result.json', {'status': 'succeeded',
                 'control': {'algorithm': {'mode': 'eval'}}, 'metrics': metrics, 'paths': {}})
    before = {p: p.read_bytes() for p in study.rglob('*') if p.is_file()}
    rows = {row['id']: row for row in list_runs(study)['runs']}
    assert rows['bad']['metric'] == ('best_accuracy=0.7000' if fallback else '-')
    assert rows['bad']['status'] == 'succeeded' and rows['bad']['mode'] == 'eval'
    assert rows['ok']['metric'] == 'accuracy=0.8000'
    assert main(['status', str(study)]) == 0
    assert 'accuracy=0.8000' in capsys.readouterr().out
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('text', ['false', '0', '[]', "''"])
def test_falsy_non_mapping_config_is_unknown_without_hiding_peer(tmp_path, text):
    study = _write_study(tmp_path)
    path = study / 'runs/wait/config.yaml'
    path.write_text(text, encoding='utf-8')
    before = {p: p.read_bytes() for p in study.rglob('*') if p.is_file()}
    body = list_runs(study)
    rows = {row['id']: row for row in body['runs']}
    assert rows['wait']['mode'] == 'unknown'
    assert 'mode config unreadable' in rows['wait']['error']
    assert rows['wait']['status'] == 'pending'
    assert rows['ok']['mode'] == 'train'
    assert rows['ok']['metric'] == 'accuracy=0.8000'
    assert body['counts'] == {'planned': 3, 'succeeded': 1, 'failed': 1, 'pending': 1}
    assert {p: p.read_bytes() for p in study.rglob('*') if p.is_file()} == before
