import json
from pathlib import Path

from rpipe.flow import FlowContext, FlowRunner
from rpipe.flow.process.aggregate import process_path
from rpipe.flow.process.curves import learning_curves_path
from rpipe.structure.artifact import (
    artifact_layout,
    build_index,
    write_config,
    write_index,
    write_result,
)
from rpipe.structure.artifact.asset import kinds
from rpipe.structure.artifact.paths import DERIVED_NAME


def _write_succeeded(study: Path, run_id: str, seed: int, size: int, metrics: dict, tags: list[str]):
    layout = artifact_layout(study, run_id)
    control = {
        'id': run_id,
        'seed': seed,
        'tags': tags,
        'data': {'config': {'train_size': size}},
        'model': {'name': 'linear'},
        'algorithm': {'mode': 'train'},
        'system': {'device': 'cpu'},
    }
    write_config(layout.config_path, control)
    write_result(
        layout.result_path,
        {
            'status': 'succeeded',
            'control': control,
            'metrics': metrics,
            'paths': {},
        },
    )
    return layout


def test_process_aggregates_siblings_and_baseline_delta(tmp_path: Path):
    study = tmp_path / 'demo'
    a = _write_succeeded(study, 'a', 0, 500, {'accuracy': 0.8, 'train_loss': 0.5}, ['baseline'])
    _write_succeeded(study, 'b', 1, 500, {'accuracy': 0.7, 'train_loss': 0.6}, ['baseline'])
    _write_succeeded(study, 'c', 0, 2000, {'accuracy': 0.9, 'train_loss': 0.3}, [])

    experiments = [
        {
            'factors': {'data.config.train_size': 500},
            'runs': [
                {'id': 'a', 'seed': 0, 'tags': ['baseline'], 'run_dir': 'a'},
                {'id': 'b', 'seed': 1, 'tags': ['baseline'], 'run_dir': 'b'},
            ],
        },
        {
            'factors': {'data.config.train_size': 2000},
            'runs': [
                {'id': 'c', 'seed': 0, 'tags': [], 'run_dir': 'c'},
                {'id': 'd', 'seed': 1, 'tags': [], 'run_dir': 'd'},
            ],
        },
    ]
    write_index(
        study,
        build_index(study='demo', description='', experiments=experiments),
    )

    ctx = FlowContext(study_dir=study, layout=a, config={})
    ctx.control = type('C', (), {'id': 'a'})()
    ctx.state['result'] = {
        'status': 'succeeded',
        'control': {'id': 'a'},
        'metrics': {'accuracy': 0.8, 'train_loss': 0.5},
    }
    FlowRunner(phases=['process']).run(ctx)

    body = ctx.state['process']
    assert body['study'] == 'demo'
    assert body['complete'] is False
    assert process_path(study).is_file()
    assert (a.root / DERIVED_NAME).is_file()

    by_size = {exp['factors']['data.config.train_size']: exp for exp in body['experiments']}
    assert by_size[500]['baseline'] is True
    assert by_size[500]['n'] == 2
    assert by_size[500]['metrics']['accuracy']['mean'] == 0.75
    assert by_size[2000]['n'] == 1
    assert by_size[2000]['n_planned'] == 2
    assert abs(by_size[2000]['delta_vs_baseline']['accuracy'] - 0.15) < 1e-9
    assert 'paired' in body
    assert body['paired'][0]['factors']['data.config.train_size'] == 500
    assert 'train' in body['paired'][0]


def test_process_does_not_rewrite_result(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'only')
    write_result(
        layout.result_path,
        {
            'status': 'succeeded',
            'control': {'id': 'only', 'seed': 0},
            'metrics': {'accuracy': 0.5},
            'paths': {},
        },
    )
    before = layout.result_path.read_text(encoding='utf-8')
    ctx = FlowContext(study_dir=tmp_path, layout=layout, config={})
    ctx.state['result'] = {
        'status': 'succeeded',
        'control': {'id': 'only'},
        'metrics': {'accuracy': 0.5},
    }
    FlowRunner(phases=['process']).run(ctx)
    assert layout.result_path.read_text(encoding='utf-8') == before
    assert 'accuracy' in ctx.state['process']['experiments'][0]['metrics']


def test_process_writes_learning_curves_from_tracker_history(tmp_path: Path):
    study = tmp_path / 'curves'
    layout = _write_succeeded(study, 'r0', 0, 500, {'accuracy': 0.8}, ['baseline'])
    state = {
        'step': 2,
        'splits': {
            'train': {
                'Loss': {'history': [1.0, 0.5], 'last': 0.5, 'mean': 0.0, 'n': 0},
                'Accuracy': {'history': [0.4, 0.8], 'last': 0.8, 'mean': 0.0, 'n': 0},
            },
            'test': {
                'Loss': {'history': [0.9, 0.4], 'last': 0.4, 'mean': 0.0, 'n': 0},
                'Accuracy': {'history': [0.5, 0.7], 'last': 0.7, 'mean': 0.0, 'n': 0},
            },
        },
        'last_segment': {},
    }
    path = layout.assets_dir / kinds.TRACKER_STATE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding='utf-8')
    write_index(
        study,
        build_index(
            study='curves',
            description='',
            experiments=[
                {
                    'factors': {'data.config.train_size': 500},
                    'runs': [{'id': 'r0', 'seed': 0, 'run_dir': 'r0'}],
                }
            ],
        ),
    )
    ctx = FlowContext(study_dir=study, layout=layout, config={})
    ctx.control = type('C', (), {'id': 'r0'})()
    FlowRunner(phases=['process']).run(ctx)
    figure = learning_curves_path(study)
    assert figure.is_file()
    assert ctx.state['process']['figures']['learning_curves'] == 'docs/figures/learning_curves.png'
    assert figure.stat().st_size > 0
