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
                        'factors': {'model.name': 'linear', 'algorithm.mode': 'eval'},
                        'runs': [
                            {'id': 'abc', 'seed': 0, 'log': 'runs/abc/assets/logs/run.log'},
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


@pytest.fixture
def report_study(tmp_path):
    study = tmp_path / 'report'
    (study / 'docs').mkdir(parents=True)
    factors = {'model.name': 'linear'}
    index = {'study': 'report', 'experiments': [
        {'factors': factors, 'runs': [{'id': 'a', 'seed': 0}, {'id': 'b', 'seed': 1}]},
    ]}
    process = {'study': 'report', 'complete': False, 'experiments': [
        {'factors': factors, 'n': 1, 'n_planned': 2,
         'metrics': {'accuracy': {'mean': 90.0, 'std': 0.0, 'min': 90.0, 'max': 90.0, 'n': 1}},
         'runs': [{'id': 'a', 'seed': 0, 'status': 'succeeded', 'metrics': {'accuracy': 90.0}},
                  {'id': 'b', 'seed': 1, 'status': None, 'metrics': {}}]},
    ]}
    (study / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    (study / 'process.json').write_text(json.dumps(process), encoding='utf-8')
    numbers_path(study).write_text('previous numbers\n', encoding='utf-8')
    (study / 'docs/STUDY_REPORT.md').write_text('original conclusion\n', encoding='utf-8')
    return study


@pytest.mark.parametrize('change', ['run_id', 'add', 'remove', 'seed', 'factors', 'study',
                                  'duplicate_index', 'duplicate_process', 'missing_process_run'])
def test_report_rejects_stale_plan_without_overwriting_evidence(report_study, change, capsys):
    study = report_study
    index = json.loads((study / 'index.json').read_text())
    process = json.loads((study / 'process.json').read_text())
    runs = index['experiments'][0]['runs']
    if change == 'run_id':
        runs[0]['id'] = 'new_version'
    elif change == 'add':
        runs.append({'id': 'c', 'seed': 2})
    elif change == 'remove':
        runs.pop()
    elif change == 'seed':
        runs[0]['seed'] = 7
    elif change == 'factors':
        index['experiments'][0]['factors']['model.name'] = 'mlp'
    elif change == 'study':
        index['study'] = 'other'
    elif change == 'duplicate_index':
        runs.append(dict(runs[0]))
    elif change == 'duplicate_process':
        process['experiments'][0]['runs'].append(dict(process['experiments'][0]['runs'][0]))
    else:
        process['experiments'][0]['runs'].pop()
    (study / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    (study / 'process.json').write_text(json.dumps(process), encoding='utf-8')
    before = {name: (study / name).read_bytes() for name in
              ['index.json', 'process.json', 'docs/NUMBERS.md', 'docs/STUDY_REPORT.md']}
    assert main(['report', str(study)]) == 2
    assert 'current index' in capsys.readouterr().err
    assert {name: (study / name).read_bytes() for name in before} == before


@pytest.mark.parametrize('contents', [None, '{', '[]', '{}', '{"experiments": {}}',
                                     '{"experiments": [null]}',
                                     '{"experiments": [{"runs": {}}]}'])
def test_report_rejects_unusable_current_index(report_study, contents, capsys):
    index = report_study / 'index.json'
    if contents is None:
        index.unlink()
    else:
        index.write_text(contents, encoding='utf-8')
    before = numbers_path(report_study).read_bytes()
    assert main(['report', str(report_study)]) == 2
    assert 'current index' in capsys.readouterr().err
    assert numbers_path(report_study).read_bytes() == before


def test_report_accepts_legacy_partial_aggregation_and_reordered_runs(report_study):
    index = json.loads((report_study / 'index.json').read_text())
    index['experiments'][0]['runs'].reverse()
    (report_study / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    before = (report_study / 'process.json').read_bytes()
    assert main(['report', str(report_study)]) == 0
    text = numbers_path(report_study).read_text(encoding='utf-8')
    assert '`a`' in text and '`b`' in text and '90.0000' in text
    assert (report_study / 'process.json').read_bytes() == before


@pytest.mark.parametrize('contents', ['{', '[]', '{}', '{"experiments": {}}',
                                     '{"experiments": [null]}',
                                     '{"experiments": [{"runs": {}}]}',
                                     '{"experiments": [{"runs": [{"seed": 0}]}]}'])
def test_report_rejects_unusable_process_without_overwriting_numbers(report_study, contents):
    (report_study / 'process.json').write_text(contents, encoding='utf-8')
    before = numbers_path(report_study).read_bytes()
    assert main(['report', str(report_study)]) == 2
    assert numbers_path(report_study).read_bytes() == before


def test_report_preserves_numbers_if_atomic_replace_fails(report_study, monkeypatch):
    import rpipe.structure.artifact._atomic as atomic

    def fail_replace(*args):
        raise PermissionError('report destination is locked')

    monkeypatch.setattr(atomic, 'atomic_replace', fail_replace)
    before = numbers_path(report_study).read_bytes()
    assert main(['report', str(report_study)]) == 2
    assert numbers_path(report_study).read_bytes() == before
    assert not list((report_study / 'docs').glob('*.tmp'))
