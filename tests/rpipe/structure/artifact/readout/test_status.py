from pathlib import Path

import pytest

from rpipe.flow.cli import main
from rpipe.structure.artifact.readout import format_status, list_runs
from rpipe.structure.artifact.index import write_index
from rpipe.structure.artifact.result import write_result

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
