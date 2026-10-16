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


@pytest.mark.parametrize('value', [10**400, float('nan'), float('inf'), -float('inf')], ids=['overflow', 'nan', 'inf', 'negative-inf'])
def test_unusable_number_is_a_placeholder_in_aggregate_and_run_cells(report_study, value):
    from rpipe.structure.artifact.readout.report import render_numbers

    path = report_study / 'process.json'
    body = json.loads(path.read_text())
    exp = body['experiments'][0]
    exp['metrics']['accuracy']['mean'] = value
    exp['runs'][0]['metrics']['accuracy'] = value
    exp['runs'][1]['metrics'] = {'accuracy': 0.75}
    path.write_text(json.dumps(body), encoding='utf-8')
    before = {p: p.read_bytes() for p in report_study.rglob('*') if p.is_file()}
    text = render_numbers(report_study)
    assert '| model.name=linear | accuracy | - | 0.0000 |' in text
    assert '`a` | accuracy=- |' in text and '`b` | accuracy=0.7500 |' in text
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('value', [7, True, 'bad', [['accuracy', 0.5]], []], ids=['int', 'bool', 'text', 'pairs', 'empty-list'])
@pytest.mark.parametrize('scope', ['experiment', 'run'])
def test_nonobject_metrics_do_not_hide_other_run_or_rewrite_evidence(report_study, value, scope):
    from rpipe.structure.artifact.readout.report import render_numbers

    path = report_study / 'process.json'
    body = json.loads(path.read_text())
    exp = body['experiments'][0]
    target = exp if scope == 'experiment' else exp['runs'][0]
    target['metrics'] = value
    exp['runs'][1]['metrics'] = {'accuracy': 0.75}
    path.write_text(json.dumps(body), encoding='utf-8')
    before = {p: p.read_bytes() for p in report_study.rglob('*') if p.is_file()}
    text = render_numbers(report_study)
    assert '`a`' in text and '`b` | accuracy=0.7500 |' in text
    if scope == 'experiment':
        assert '| model.name=linear | - | - | - | - | - | 1 |' in text
    else:
        assert '`a` | - |' in text
    assert {p: p.read_bytes() for p in before} == before


_TABLE_TEXT = [
    pytest.param('left\nright', id='lf'),
    pytest.param('left\rright', id='cr'),
    pytest.param('left\r\nright', id='crlf'),
    pytest.param('left\tright', id='tab'),
    pytest.param('left\x85right', id='nel'),
    pytest.param('left\u2028right', id='line-separator'),
    pytest.param('left|right', id='pipe'),
    pytest.param('left\\|right', id='one-backslash-pipe'),
    pytest.param('left\\\\|right', id='two-backslash-pipe'),
    pytest.param('left right', id='ordinary'),
]


def _set_table_label(study, field, text):
    for filename in ('index.json', 'process.json'):
        path = study / filename
        body = json.loads(path.read_text(encoding='utf-8'))
        exp = body['experiments'][0]
        exp['factors']['algorithm.mode'] = 'train'
        if field == 'factor-key':
            exp['factors'][text] = 'value'
        elif field == 'factor-value':
            exp['factors']['data.config.prompt'] = text
        elif filename == 'process.json':
            exp['metrics'][text] = exp['metrics'].pop('accuracy')
            exp['runs'][0]['metrics'] = {text: 90.0}
        path.write_text(json.dumps(body), encoding='utf-8')
    result = {'status': 'succeeded', 'metrics': {
        text if field == 'metric-key' else 'accuracy': 90.0}}
    target = study / 'runs/a/result.json'
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(result), encoding='utf-8')
    return {p: p.read_bytes() for p in study.rglob('*') if p.is_file()}


def _markdown_columns(line):
    separators = 0
    escaped = False
    for char in line:
        if char == '|' and not escaped:
            separators += 1
        escaped = char == '\\' and not escaped
    return separators - 1


@pytest.mark.parametrize('text', _TABLE_TEXT)
@pytest.mark.parametrize('field', ['factor-key', 'factor-value'])
def test_status_text_cells_keep_planned_rows(report_study, field, text):
    from rpipe.structure.artifact.readout.status import list_runs, format_status

    before = _set_table_label(report_study, field, text)
    body = list_runs(report_study)
    view = format_status(body)
    lines = view.splitlines()
    assert len(lines) == 4
    assert all(len(line.split('\t')) == 9 for line in lines[1:])
    assert 'left' in lines[2] and 'right' in lines[2]
    assert '90.0000' in lines[2]
    assert lines[3].split('\t')[3] == 'b'
    assert body['counts'] == {'planned': 2, 'succeeded': 1, 'failed': 0, 'pending': 1}
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('text', _TABLE_TEXT)
@pytest.mark.parametrize('field', ['factor-key', 'factor-value', 'metric-key'])
def test_numbers_text_cells_keep_planned_tables(report_study, field, text):
    from rpipe.structure.artifact.readout.report import render_numbers

    before = _set_table_label(report_study, field, text)
    view = render_numbers(report_study)
    tables = []
    current = []
    for line in view.splitlines() + ['']:
        if line.startswith('|'):
            current.append(line)
        elif current:
            tables.append(current)
            current = []
    assert len(tables) == 2
    assert [len(table) for table in tables] == [3, 4]
    assert [[_markdown_columns(line) for line in table] for table in tables] == [[7] * 3, [5] * 4]
    assert 'left' in tables[0][2] and 'right' in tables[0][2]
    assert '90.0000' in tables[0][2]
    assert '`b`' in tables[1][3]
    assert '[run.log](../runs/a/assets/logs/run.log)' in tables[1][2]
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('route', ['stat', 'experiment-empty', 'experiment-fallback'])
@pytest.mark.parametrize('case,value', [
    ('zero', 0), ('one', 1), ('two', 2), ('large', 10 ** 100), ('missing', None),
    ('true', True), ('false', False), ('negative', -1), ('null', None),
    ('fraction', 1.5), ('integral-float', 2.0), ('numeric-text', '2'),
    ('pipe', '2 | forged'), ('newline', '2' + chr(10) + 'forged'),
    ('unicode-line', '2' + chr(0x2028) + 'forged'), ('list', [2]),
    ('object', {'n': 2}), ('surrogate', chr(0xD800)),
    ('nan', float('nan')), ('infinity', float('inf')),
])
def test_report_count_cells_preserve_rows_and_peers(report_study, capsys, route, case, value):
    from rpipe.structure.artifact.readout.report import render_numbers

    study = report_study
    index = json.loads((study / 'index.json').read_text())
    process = json.loads((study / 'process.json').read_text())
    peer = {'factors': {'model.name': 'mlp'}, 'runs': [{'id': 'c', 'seed': 2}]}
    index['experiments'].append(peer)
    process['experiments'].append({**peer, 'n': 1, 'metrics': {
        'accuracy': {'mean': 80.0, 'std': 0.0, 'min': 80.0, 'max': 80.0, 'n': 1}}})
    exp = process['experiments'][0]
    stat = exp['metrics']['accuracy']
    if route == 'experiment-empty':
        exp['metrics'] = {}
    elif route == 'experiment-fallback':
        stat.pop('n')
    if case == 'missing':
        (stat if route == 'stat' else exp).pop('n', None)
        expected = '1' if route == 'stat' else '0'
    else:
        (stat if route == 'stat' else exp)['n'] = value
        expected = str(value) if case in ('zero', 'one', 'two', 'large') else '-'
    (study / 'index.json').write_text(json.dumps(index), encoding='utf-8')
    (study / 'process.json').write_text(json.dumps(process), encoding='utf-8')
    before = {p: p.read_bytes() for p in [study / 'index.json', study / 'process.json',
                                         numbers_path(study), study / 'docs/STUDY_REPORT.md']}
    text = render_numbers(study)
    assert {p: p.read_bytes() for p in before} == before
    block = text.split('## Experiments')[1].split('## Runs')[0]
    rows = [line for line in block.splitlines() if line.startswith('|')]
    assert len(rows) == 4 and all(line.count('|') == 8 for line in rows)
    assert rows[2].split('|')[-2].strip() == expected
    assert rows[3].split('|')[-2].strip() == '1' and '80.0000' in rows[3]
    assert all('`' + name + '`' in text for name in ('a', 'b', 'c'))
    assert main(['report', str(study)]) == 0
    assert numbers_path(study).read_text(encoding='utf-8') == text
    assert capsys.readouterr().out.strip() == str(numbers_path(study))
    assert {p: p.read_bytes() for p in before if p != numbers_path(study)} == {
        p: value for p, value in before.items() if p != numbers_path(study)}


@pytest.mark.parametrize('run_id,expected_cell', [
    ('a`b', '`` a`b ``'), ('`edge', '`` `edge ``'), ('edge`', '`` edge` ``'),
    ('`wrapped`', '`` `wrapped` ``'), ('a``b', '``` a``b ```'),
    ('`', '`` ` ``'), ('``', '``` `` ```'), ('```', '```` ``` ````'),
    ('a``x`', '``` a``x` ```'), ('plain', '`plain`'),
    ('with space', '`with space`'), ('中文', '`中文`'),
])
@pytest.mark.parametrize('publish', [False, True])
def test_numbers_code_cell_preserves_literal_run_identity(report_study, run_id, expected_cell, publish):
    from urllib.parse import quote
    from rpipe.structure.artifact.layout import validate_run_directory
    from rpipe.structure.artifact.readout.report import render_numbers, write_numbers

    validate_run_directory(run_id)
    for name in ('index.json', 'process.json'):
        path = report_study / name
        body = json.loads(path.read_text(encoding='utf-8'))
        body['experiments'][0]['runs'][0]['id'] = run_id
        path.write_text(json.dumps(body, ensure_ascii=False), encoding='utf-8')
    human = report_study / 'docs/STUDY_REPORT.md'
    human.write_text('human conclusion stays here\n', encoding='utf-8')
    before = {p: p.read_bytes() for p in report_study.rglob('*') if p.is_file()}
    if publish:
        assert write_numbers(report_study) == numbers_path(report_study)
        text = numbers_path(report_study).read_text(encoding='utf-8')
    else:
        text = render_numbers(report_study)
    run_rows = [line for line in text.splitlines() if '[run.log]' in line]
    assert len(run_rows) == 2
    assert run_rows[0].split(' | ')[2] == expected_cell
    assert '| `b` |' in run_rows[1]
    href = '../runs/' + quote(run_id, safe='') + '/assets/logs/run.log'
    assert f'[run.log]({href})' in run_rows[0]
    assert 'accuracy=90.0000' in run_rows[0]
    assert '90.0000' in text and all(_markdown_columns(line) == 5 for line in run_rows)
    protected = {p: value for p, value in before.items() if not publish or p != numbers_path(report_study)}
    assert {p: p.read_bytes() for p in protected} == protected
