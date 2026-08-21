"""summarize: assemble Result body from Control + collected metrics."""

from __future__ import annotations

from rpipe.artifact.result import STATUS_SUCCEEDED
from rpipe.flow.context import FlowContext


def run(ctx: FlowContext) -> None:
    if ctx.control is None:
        raise RuntimeError('prepare must run before summarize')
    collected = ctx.state.get('collected') or {}
    ctx.state['result_draft'] = {
        'status': STATUS_SUCCEEDED,
        'control': ctx.control.to_dict(),
        'structure': {
            'data': ctx.state.get('data'),
            'model': ctx.state.get('model'),
            'system': ctx.state.get('system'),
        },
        'metrics': collected.get('metrics') or {},
        'paths': {
            'artifact': str(ctx.layout.root),
            'config': str(ctx.layout.config_path),
            'assets': str(ctx.layout.assets_dir),
        },
        'experiment': str(ctx.experiment_dir),
    }
