from pathlib import Path

import pytest

from rpipe.structure.artifact import artifact_layout, load_result, write_config, validate_result
from rpipe.flow import FlowContext, FlowRunner

pytestmark = [
    pytest.mark.integration,
    pytest.mark.content,
    pytest.mark.p1,
    pytest.mark.flow_layer,
    pytest.mark.module_runner,
    pytest.mark.cost(cost_class='c1'),
    pytest.mark.result_type('categorical', detail='summary'),
]


def test_full_flow_writes_succeeded_status(tmp_path: Path):
    """FlowRunner from prepare through process writes status succeeded."""
    layout = artifact_layout(tmp_path, 'seed_0')
    write_config(
        layout.config_path,
        {
            'slug': 'seed_0',
            'seed': 0,
            'data': {'name': 'Toy', 'source': 'stub'},
            'model': {'name': 'linear'},
            'algorithm': {'mode': 'train', 'num_steps': 2},
            'system': {'device': 'cpu'},
        },
    )
    ctx = FlowContext(study_dir=tmp_path, layout=layout, config={})
    result_path = FlowRunner().run(ctx)
    assert result_path.is_file()
    result = load_result(result_path)
    assert result['status'] == 'succeeded'
    assert result['structure']['data']['source'] == 'stub'
    assert result['structure']['data']['stub'] is True
    assert ctx.control is not None
    assert ctx.control.id
    assert ctx.control.seed == 0
    assert 'train_loss' in result.get('metrics', {}) or 'accuracy' in result.get('metrics', {})
    assert (tmp_path / 'runs' / 'seed_0' / 'process.json').is_file()
    assert not (tmp_path / 'process.json').is_file()


def test_failed_flow_writes_failed_result(tmp_path: Path):
    """A failing execute phase writes status failed and an [error] traceback."""
    layout = artifact_layout(tmp_path, 'bad_run')
    write_config(
        layout.config_path,
        {
            'seed': 0,
            'data': {'name': 'Toy', 'source': 'stub'},
            'model': {'name': 'linear'},
            'algorithm': {'mode': 'train'},
            'system': {'device': 'cpu'},
        },
    )
    ctx = FlowContext(study_dir=tmp_path, layout=layout, config={})

    def boom(_ctx):
        raise RuntimeError('boom')

    import rpipe.flow.execute as execute_mod

    original = execute_mod.run
    execute_mod.run = boom
    try:
        with pytest.raises(RuntimeError, match='boom'):
            FlowRunner().run(ctx)
    finally:
        execute_mod.run = original

    assert layout.result_path.is_file()
    result = load_result(layout.result_path)
    assert result['status'] == 'failed'
    assert 'boom' in result['error']
    assert validate_result(result) == []
    log = (layout.assets_dir / 'logs' / 'run.log').read_text(encoding='utf-8')
    assert '[flow] start phases=' in log
    assert '[error] phase=execute status=failed RuntimeError: boom' in log
    assert '[error] Traceback (most recent call last):' in log
    assert 'RuntimeError: boom' in log
    assert '[flow] succeeded' not in log


def test_prepare_failure_still_writes_traceback_to_run_log(tmp_path: Path):
    """A failing prepare phase still leaves the traceback in run.log."""
    layout = artifact_layout(tmp_path, 'missing_cfg')
    ctx = FlowContext(study_dir=tmp_path, layout=layout, config={})
    with pytest.raises(Exception):
        FlowRunner().run(ctx)
    log = (layout.assets_dir / 'logs' / 'run.log').read_text(encoding='utf-8')
    assert '[flow] start phases=' in log
    assert '[error] phase=prepare status=failed' in log
    assert '[error] Traceback (most recent call last):' in log
    assert '[flow] succeeded' not in log


@pytest.mark.parametrize('layer,mapping', [
    ('data', {'name': 'Typo', 'source': 'torch'}),
    ('data', {'name': 'Toy', 'source': 'torch'}),
    ('model', {'name': 'Typo', 'source': 'custom_torch'}),
    ('model', {'name': 'linear', 'source': 'not-installed-source'}),
])
def test_invalid_configuration_writes_failed_result(tmp_path, layer, mapping):
    """Invalid data/model declaration → prepare failure → failed result, never succeeded."""
    layout = artifact_layout(tmp_path, 'invalid')
    config = {
        'seed': 0,
        'data': {'name': 'Toy', 'source': 'stub'},
        'model': {'name': 'linear'},
        'algorithm': {'mode': 'train', 'num_steps': 2},
        'system': {'device': 'cpu'},
    }
    config[layer] = mapping
    write_config(layout.config_path, config)
    ctx = FlowContext(study_dir=tmp_path, layout=layout, config={})
    with pytest.raises(ValueError, match=f'unknown {layer} name/source'):
        FlowRunner().run(ctx)
    result = load_result(layout.result_path)
    assert result['status'] == 'failed'
    assert result['metrics'] == {}
    log = (layout.assets_dir / 'logs' / 'run.log').read_text(encoding='utf-8')
    assert 'phase=prepare status=failed' in log
    assert '[flow] succeeded' not in log
