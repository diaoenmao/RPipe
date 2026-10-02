import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.flow_layer,
    pytest.mark.module_process,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]

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
    """One Run's process file keeps that Run's metrics and does not invent experiment stats."""
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


def test_study_process_aggregates_history_min_max(tmp_path: Path, monkeypatch):
    """Indexed results/tracker → process and PNG; percent values and single points remain visible."""
    from matplotlib.figure import Figure

    saved_figures = []
    savefig = Figure.savefig

    def capture_figure(figure, *args, **kwargs):
        saved_figures.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capture_figure)
    study = tmp_path / 'demo'
    a = _write_succeeded(study, 'a', 0, 500, {'accuracy': 80.0, 'train_loss': 0.5}, ['baseline'])
    b = _write_succeeded(study, 'b', 1, 500, {'accuracy': 70.0, 'train_loss': 0.6}, ['baseline'])
    _write_succeeded(study, 'c', 0, 2000, {'accuracy': 90.0, 'train_loss': 0.3}, [])

    for layout, hist in (
        (a, [0.0, 50.0, 100.0]),
        (b, [0.0, 40.0, 100.0]),
    ):
        state = {
            'step': 2,
            'splits': {
                'test': {
                    'Accuracy': {'history': hist, 'last': hist[-1], 'mean': 0.0, 'n': 0},
                    'Loss': {'history': [0.5]},
                },
                'train': {
                    'Accuracy': {'history': [0.5]},
                    'Loss': {'history': [0.6]},
                },
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
    assert acc['mean'] == 75.0
    assert acc['min'] == 70.0
    assert acc['max'] == 80.0
    history = by_size[500]['history']['test']['Accuracy']
    assert history['mean'] == [0.0, 45.0, 100.0]
    assert history['min'] == [0.0, 40.0, 100.0]
    assert history['max'] == [0.0, 50.0, 100.0]
    assert body.get('figures', {}).get('learning_curves') == 'docs/figures/learning_curves.png'
    assert (study / body['figures']['learning_curves']).is_file()
    assert by_size[2000]['delta_vs_baseline']['accuracy'] == 15.0

    assert len(saved_figures) == 1
    axes = saved_figures[0].axes
    for ax in axes[:2]:
        lower, upper = ax.get_ylim()
        assert lower <= 0.0 and upper >= 100.0
        assert ax.get_ylabel() == 'Accuracy (%)'
    for ax, expected in zip(axes, ([0.0, 45.0, 100.0], [0.5], [0.5], [0.6])):
        # A real 0.5% is not inferred to be legacy 50%; Loss stays unscaled too.
        assert list(ax.lines[0].get_ydata()) == expected
        assert ax.get_xlabel() == 'history point'
        if len(expected) == 1:
            assert ax.lines[0].get_marker() not in (None, '', ' ', 'None')


def test_process_does_not_rewrite_result(tmp_path: Path):
    """process reads result.json and leaves its bytes unchanged."""
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


@pytest.mark.parametrize(
    'index_bytes',
    [None, b'{', b'[]', b'{}', b'{"experiments": {}}', b'\xff'],
    ids=['missing', 'invalid-json', 'not-object', 'missing-experiments', 'not-list', 'invalid-utf8'],
)
def test_study_process_rejects_unusable_index_without_touching_artifacts(tmp_path: Path, index_bytes):
    """Invalid Study index → process error with make hint; old outputs and Runs stay intact."""
    old = _write_succeeded(tmp_path, 'historical', 0, 500, {'accuracy': 80.0}, [])
    output = process_path(tmp_path)
    output.write_bytes(b'previous study process')
    figure = tmp_path / 'docs' / 'figures' / 'learning_curves.png'
    figure.parent.mkdir(parents=True)
    figure.write_bytes(b'previous learning curves')
    preserved = {path: path.read_bytes() for path in (old.config_path, old.result_path, output, figure)}
    if index_bytes is not None:
        (tmp_path / 'index.json').write_bytes(index_bytes)

    with pytest.raises(ValueError, match=r'index\.json.*make'):
        run_study(tmp_path)

    assert {path: path.read_bytes() for path in preserved} == preserved


def test_study_process_accepts_empty_current_index(tmp_path: Path):
    """Valid empty index → empty partial process, without scanning historical Runs."""
    _write_succeeded(tmp_path, 'historical', 0, 500, {'accuracy': 80.0}, [])
    write_index(tmp_path, build_index(study='empty', description='', experiments=[]))
    body = run_study(tmp_path)
    assert body['experiments'] == []
    assert body['complete'] is False
