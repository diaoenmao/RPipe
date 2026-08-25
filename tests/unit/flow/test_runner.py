from pathlib import Path

import pytest

from rpipe.structure.artifact import artifact_layout, load_result, write_config, write_result, validate_result
from rpipe.flow import FlowContext, FlowRunner


def test_full_flow_writes_succeeded_status(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'seed_0')
    write_config(
        layout.config_path,
        {
            'slug': 'seed_0',
            'seed': 0,
            'data': {'name': 'Toy'},
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
    assert ctx.control is not None
    assert ctx.control.id
    assert ctx.control.seed == 0
    assert 'accuracy' in result.get('metrics', {}) or 'loss' in result.get('metrics', {})


def test_failed_flow_writes_failed_result(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'bad_run')
    write_config(
        layout.config_path,
        {
            'seed': 0,
            'data': {'name': 'Toy'},
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


def test_write_result_requires_status(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'x')
    with pytest.raises(ValueError, match='status'):
        write_result(layout.result_path, {'control': {}, 'metrics': {}, 'paths': {}})
