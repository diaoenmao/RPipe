"""summarize: assemble Result body from Control + collected metrics."""

from __future__ import annotations

from typing import Any

from rpipe.artifact.result import STATUS_SUCCEEDED
from rpipe.flow.context import FlowContext

# Runtime handles must not enter Result JSON.
_DROP_KEYS = frozenset({'train_loader', 'test_loader', 'module', 'optimizer'})


def _safe_structure_snapshot(mapping: Any) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        if key in _DROP_KEYS:
            continue
        if isinstance(value, (bool, int, float, str)) or value is None:
            out[key] = value
        elif isinstance(value, dict):
            out[key] = _safe_structure_snapshot(value)
        elif isinstance(value, (list, tuple)) and all(
            isinstance(x, (bool, int, float, str)) or x is None for x in value
        ):
            out[key] = list(value)
        else:
            out[key] = f'<{type(value).__name__}>'
    return out


def run(ctx: FlowContext) -> None:
    if ctx.control is None:
        raise RuntimeError('prepare must run before summarize')
    collected = ctx.state.get('collected') or {}
    ctx.state['result_draft'] = {
        'status': STATUS_SUCCEEDED,
        'control': ctx.control.to_dict(),
        'structure': {
            'data': _safe_structure_snapshot(ctx.state.get('data')),
            'model': _safe_structure_snapshot(ctx.state.get('model')),
            'system': _safe_structure_snapshot(ctx.state.get('system')),
        },
        'metrics': collected.get('metrics') or {},
        'paths': {
            'artifact': str(ctx.layout.root),
            'config': str(ctx.layout.config_path),
            'assets': str(ctx.layout.assets_dir),
        },
        'experiment': str(ctx.experiment_dir),
    }
