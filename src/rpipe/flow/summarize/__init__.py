"""summarize: serializable result_draft only."""

from __future__ import annotations

from typing import Any

from rpipe.flow.context import FlowContext
from rpipe.structure.artifact.result import STATUS_SUCCEEDED


def _snapshot(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    method = getattr(obj, 'to_result_snapshot', None)
    if callable(method):
        return method()
    if isinstance(obj, dict):
        drop = {'train_loader', 'test_loader', 'module', 'optimizer', 'logger', 'tracker'}
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if key in drop:
                continue
            if isinstance(value, (bool, int, float, str)) or value is None:
                out[key] = value
            else:
                out[key] = f'<{type(value).__name__}>'
        return out
    return {'type': type(obj).__name__}


def run(ctx: FlowContext) -> None:
    if ctx.control is None:
        raise RuntimeError('prepare must run before summarize')
    collected = ctx.state.get('collected') or {}
    ctx.state['result_draft'] = {
        'status': STATUS_SUCCEEDED,
        'control': ctx.control.to_dict(),
        'structure': {
            'data': _snapshot(ctx.state.get('data')),
            'model': _snapshot(ctx.state.get('model')),
            'system': _snapshot(ctx.state.get('system')),
        },
        'metrics': collected.get('metrics') or {},
        'paths': {
            'artifact': str(ctx.layout.root),
            'config': str(ctx.layout.config_path),
            'assets': str(ctx.layout.assets_dir),
            'tracker': str(ctx.layout.assets_dir / 'tracker'),
            'logs': str(ctx.layout.assets_dir / 'logs'),
            'checkpoints': str(ctx.layout.assets_dir / 'checkpoints'),
        },
        'study': str(ctx.study_dir),
    }
