import json
from pathlib import Path

import pytest

from rpipe.flow.cli import main
from rpipe.structure.artifact.readout import numbers_path

pytestmark = [
    pytest.mark.unit,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.structure_layer,
    pytest.mark.module_artifact,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]


def test_report_writes_experiment_and_run_tables(tmp_path: Path):
    study = tmp_path / 'demo'
    (study / 'docs').mkdir(parents=True)
    (study / 'docs' / 'STUDY_REPORT.md').write_text('handwritten conclusion\n', encoding='utf-8')
    (study / 'process.json').write_text(
        json.dumps(
            {
                'study': 'demo',
                'experiments': [
                    {
                        'factors': {'model.name': 'linear', 'algorithm.mode': 'eval'},
                        'n': 1,
                        'metrics': {
                            'accuracy': {'mean': 0.2984, 'std': 0.0, 'min': 0.2984, 'max': 0.2984, 'n': 1}
                        },
                        'runs': [
                            {
                                'id': 'abc',
                                'seed': 0,
                                'status': 'succeeded',
                                'metrics': {'accuracy': 0.2984},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding='utf-8',
    )
    (study / 'index.json').write_text(
        json.dumps(
            {
                'experiments': [
                    {
                        'runs': [
                            {'id': 'abc', 'log': 'runs/abc/assets/logs/run.log'},
                        ]
                    }
                ]
            }
        ),
        encoding='utf-8',
    )
    assert main(['report', str(study)]) == 0
    text = numbers_path(study).read_text(encoding='utf-8')
    assert '0.2984' in text
    assert '`abc`' in text
    assert '[run.log](../runs/abc/assets/logs/run.log)' in text
    assert '谁更好' not in text
    assert (study / 'docs' / 'STUDY_REPORT.md').read_text(encoding='utf-8') == 'handwritten conclusion\n'
    assert main(['report', str(tmp_path / 'missing')]) == 2
