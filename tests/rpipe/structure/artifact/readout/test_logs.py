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
