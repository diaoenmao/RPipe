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


def test_report_rejects_old_process_after_fresh_make_and_accepts_reprocess(tmp_path):
    """CLI make/process/report → new-version make/report: reject stale metadata, keep evidence."""
    from rpipe.flow.cli import main
    from rpipe.structure.artifact.index import load_index
    from rpipe.structure.artifact.readout import numbers_path

    study = tmp_path / 'fresh'
    study.mkdir()
    (study / 'study.yaml').write_text('study: fresh\nseeds: [0, 1]\n', encoding='utf-8')
    config = ('version: {version}\ndata: {{name: Toy, source: stub}}\n'
              'model: {{name: linear}}\nalgorithm: {{num_steps: 2}}\n'
              'system: {{device: cpu}}\n')
    (study / 'experiment_config.yaml').write_text(config.format(version='old'), encoding='utf-8')
    assert main(['make', str(study), '--num-gpus', '0']) == 0
    assert main(['process', str(study)]) == 0
    assert main(['report', str(study)]) == 0
    old_index = load_index(study)
    old_process = (study / 'process.json').read_bytes()
    old_numbers = numbers_path(study).read_bytes()
    old_configs = {p: p.read_bytes() for p in (study / 'runs').glob('*/config.json')}

    (study / 'experiment_config.yaml').write_text(config.format(version='new'), encoding='utf-8')
    assert main(['make', str(study), '--num-gpus', '0']) == 0
    assert load_index(study)['id'] != old_index['id']
    assert main(['report', str(study)]) == 2
    assert (study / 'process.json').read_bytes() == old_process
    assert numbers_path(study).read_bytes() == old_numbers
    assert {p: p.read_bytes() for p in old_configs} == old_configs

    assert main(['process', str(study)]) == 0
    assert main(['report', str(study)]) == 0
    current_ids = {r['id'] for exp in load_index(study)['experiments'] for r in exp['runs']}
    text = numbers_path(study).read_text(encoding='utf-8')
    assert all(f'`{run_id}`' in text for run_id in current_ids)


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


def test_study_progress_summary_and_plot_share_coordinates_and_counts(tmp_path, monkeypatch):
    """Indexed successful trackers → coordinate aggregates and PNG, excluding failed runs."""
    from matplotlib.figure import Figure
    from rpipe.structure.algorithm.tracker import AlgorithmTracker

    figures = []
    savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        figures.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capture)
    runs = []
    for seed, (rid, coordinates) in enumerate([
        ('a', [(10, 1.0), (20, 2.0), (40, 4.0)]),
        ('b', [(10, 3.0), (30, 6.0), (40, 8.0)]),
        ('failed', [(999, 99.0)]),
        ('epoch', [(0.5, 7.0)]),
    ]):
        layout = _write_succeeded(tmp_path, rid, seed, 500, {'accuracy': 80.0}, [])
        if rid == 'failed':
            write_result(layout.result_path, {'status': 'failed', 'control': {'id': rid}, 'error': 'injected'})
        tracker = AlgorithmTracker(layout.assets_dir)
        tracker.begin_run()
        for point, value in coordinates:
            tracker.reset('test')
            tracker.append('test', values={'Loss': value})
            tracker.flush('test', progress={'epoch' if rid == 'epoch' else 'step': point})
        runs.append({'id': rid, 'run_dir': rid, 'seed': seed})
    write_index(tmp_path, build_index(study='progress', description='', experiments=[
        {'factors': {}, 'runs': runs},
    ]))
    body = run_study(tmp_path)
    by_unit = body['experiments'][0]['history']['test']['Loss']['by_unit']
    step = by_unit['step']
    assert step['x'] == [10, 20, 30, 40]
    assert step['n_at_point'] == [2, 1, 1, 2]
    assert step['mean'] == [2.0, 2.0, 6.0, 6.0]
    assert by_unit['epoch']['x'] == [0.5]
    assert {ax.get_xlabel() for ax in figures[0].axes} == {'optimizer step', 'epoch'}
    step_axis = next(ax for ax in figures[0].axes if ax.get_title() == 'Test loss' and ax.get_xlabel() == 'optimizer step')
    assert list(step_axis.lines[0].get_xdata()) == step['x']
    assert list(step_axis.lines[0].get_ydata()) == step['mean']
    assert [text.get_text() for text in step_axis.texts] == ['n=2', 'n=1', 'n=1', 'n=2']


@pytest.mark.parametrize('raw', [b'\xff\xfe\x80', b'{"splits": "\x80"}', b'\xf0\x9f'])
@pytest.mark.parametrize('scope', ['run', 'study'])
def test_process_optional_state_decode_failure_keeps_peer_evidence(tmp_path, raw, scope):
    """Existing result/index + bad optional tracker → process; peer and source bytes stay intact."""
    from rpipe.flow.process.individual import run as process_one
    study = tmp_path / 'demo'
    a = _write_succeeded(study, 'a', 0, 500, {'accuracy': 80.0}, [])
    b = _write_succeeded(study, 'b', 1, 500, {'accuracy': 70.0}, [])
    write_index(study, {'study': 'demo', 'experiments': [
        {'factors': {'algorithm.mode': 'train'}, 'runs': [
            {'id': 'a', 'seed': 0}, {'id': 'b', 'seed': 1}]}]})
    state = a.assets_dir / kinds.TRACKER_STATE
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_bytes(raw)
    target = run_process_path(a.root) if scope == 'run' else process_path(study)
    target.write_text('previous derived process\n', encoding='utf-8')
    before = {p: p.read_bytes() for p in study.rglob('*') if p.is_file() and p != target}
    if scope == 'run':
        ctx = FlowContext(study_dir=study, layout=a, config={})
        process_one(ctx)
        body = ctx.state['process']
        assert body['metrics'] == {'accuracy': 80.0}
        assert body['history'] == {}
        assert not process_path(study).exists()
    else:
        body = run_study(study)
        exp = body['experiments'][0]
        assert exp['n'] == 2 and exp['n_planned'] == 2
        assert exp['metrics']['accuracy']['mean'] == 75.0
        assert [r['id'] for r in exp['runs']] == ['a', 'b']
        assert body['complete'] is True
        assert 'figures' not in body
        assert not run_process_path(b.root).exists()
    assert json.loads(target.read_text(encoding='utf-8')) == body
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize('failure', ['nonobject', 'deep', 'stat_permission', 'stat_io', 'read_permission', 'syntax', 'missing', 'valid'])
def test_study_optional_result_failures_keep_peer_partial_output_and_inputs(tmp_path, monkeypatch, failure):
    """Actual Study process -> one unreadable optional result -> fresh partial output with valid peer and all source bytes retained."""
    bad = _write_succeeded(tmp_path, 'bad', 0, 500, {'accuracy': 42.0}, [])
    peer = _write_succeeded(tmp_path, 'peer', 1, 500, {'accuracy': 42.0}, [])
    write_index(tmp_path, build_index(study='synthetic', description='', experiments=[
        {'factors': {}, 'runs': [{'id': 'bad', 'seed': 0}, {'id': 'peer', 'seed': 1}]}]))
    if failure == 'missing':
        bad.result_path.unlink()
    elif failure in ('nonobject', 'deep', 'syntax'):
        raw = b'[]' if failure == 'nonobject' else b'{' if failure == 'syntax' else b'{"unused":' + b'[' * 12000 + b'0' + b']' * 12000 + b'}'
        bad.result_path.write_bytes(raw)
    before = {f: f.read_bytes() for f in tmp_path.rglob('*') if f.is_file()}
    process_path(tmp_path).write_bytes(b'previous derived process')
    method = 'read_text' if failure == 'read_permission' else 'stat'
    original = getattr(Path, method)
    def refuse(path, *args, **kwargs):
        if path == bad.result_path and failure in ('stat_permission', 'stat_io', 'read_permission'):
            raise OSError(5, 'result IO') if failure == 'stat_io' else PermissionError(13, 'result denied')
        return original(path, *args, **kwargs)
    with monkeypatch.context() as fault:
        fault.setattr(Path, method, refuse)
        body = run_study(tmp_path)
    cell = body['experiments'][0]
    count = 2 if failure == 'valid' else 1
    assert body['complete'] is (failure == 'valid')
    assert cell['n'] == count and cell['n_planned'] == 2 and cell['metrics']['accuracy']['mean'] == 42.0
    assert json.loads(process_path(tmp_path).read_text(encoding='utf-8')) == body
    assert {f: f.read_bytes() for f in before} == before
