import json
from pathlib import Path

from rpipe.flow import FlowContext, FlowRunner
from rpipe.flow.process.aggregate import process_path, run_process_path
from rpipe.flow.process.study import run_study
from rpipe.structure.artifact import (
    artifact_layout,
    build_index,
    write_config,
    write_index,
    write_result,
)
from rpipe.structure.artifact.asset import kinds


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


def test_run_process_is_individual_only(tmp_path: Path):
    study = tmp_path / 'demo'
    a = _write_succeeded(study, 'a', 0, 500, {'accuracy': 0.8, 'train_loss': 0.5}, ['baseline'])
    _write_succeeded(study, 'b', 1, 500, {'accuracy': 0.7, 'train_loss': 0.6}, ['baseline'])

    ctx = FlowContext(study_dir=study, layout=a, config={})
    ctx.control = type('C', (), {'id': 'a'})()
    ctx.state['result'] = {
        'status': 'succeeded',
        'control': {'id': 'a'},
        'metrics': {'accuracy': 0.8, 'train_loss': 0.5},
    }
    FlowRunner(phases=['process']).run(ctx)

    body = ctx.state['process']
    assert body['scope'] == 'run'
    assert body['run_id'] == 'a'
    assert body['metrics']['accuracy'] == 0.8
    assert run_process_path(a.root).is_file()
    assert not process_path(study).is_file()


def test_study_process_aggregates_history_min_max(tmp_path: Path):
    study = tmp_path / 'demo'
    a = _write_succeeded(study, 'a', 0, 500, {'accuracy': 0.8, 'train_loss': 0.5}, ['baseline'])
    b = _write_succeeded(study, 'b', 1, 500, {'accuracy': 0.7, 'train_loss': 0.6}, ['baseline'])
    _write_succeeded(study, 'c', 0, 2000, {'accuracy': 0.9, 'train_loss': 0.3}, [])

    for layout, hist in (
        (a, [0.5, 0.8]),
        (b, [0.4, 0.6]),
    ):
        state = {
            'step': 2,
            'splits': {
                'test': {'Accuracy': {'history': hist, 'last': hist[-1], 'mean': 0.0, 'n': 0}},
            },
            'last_segment': {},
        }
        path = layout.assets_dir / kinds.TRACKER_STATE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state), encoding='utf-8')

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
    write_index(study, build_index(study='demo', description='', experiments=experiments))

    body = run_study(study)
    assert body['scope'] == 'study'
    assert body['complete'] is False
    assert process_path(study).is_file()

    by_size = {exp['factors']['data.config.train_size']: exp for exp in body['experiments']}
    acc = by_size[500]['metrics']['accuracy']
    assert acc['mean'] == 0.75
    assert acc['min'] == 0.7
    assert acc['max'] == 0.8
    history = by_size[500]['history']['test']['Accuracy']
    assert history['mean'] == [0.45, 0.7]
    assert history['min'] == [0.4, 0.6]
    assert history['max'] == [0.5, 0.8]
    assert body.get('figures', {}).get('learning_curves') == 'docs/figures/learning_curves.png'
    assert abs(by_size[2000]['delta_vs_baseline']['accuracy'] - 0.15) < 1e-9


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
    assert ctx.state['process']['scope'] == 'run'
    assert ctx.state['process']['metrics']['accuracy'] == 0.5
