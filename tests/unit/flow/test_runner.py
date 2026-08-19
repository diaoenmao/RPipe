from pathlib import Path

from rpipe.artifact import artifact_layout, write_config
from rpipe.flow import FlowContext, FlowRunner


def test_full_flow_writes_result(tmp_path: Path):
    layout = artifact_layout(tmp_path, 'seed_0')
    write_config(
        layout.config_path,
        {
            'slug': 'seed_0',
            'seed': 0,
            'data': {'name': 'MNIST'},
            'model': {'name': 'linear'},
            'algorithm': {'mode': 'train', 'num_steps': 2},
            'system': {'device': 'cpu'},
        },
    )
    ctx = FlowContext(experiment_dir=tmp_path, layout=layout, config={})
    result_path = FlowRunner().run(ctx)
    assert result_path.is_file()
    assert ctx.control is not None
    assert ctx.control.id
    assert ctx.control.seed == 0
    assert 'accuracy' in (ctx.state.get('result') or {}).get('metrics', {}) or 'loss' in (
        ctx.state.get('result') or {}
    ).get('metrics', {})
