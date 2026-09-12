"""Run-scoped process: this Run only. Does not write Study-root process.json."""

from __future__ import annotations

from rpipe.flow.context import FlowContext
from rpipe.flow.process.aggregate import load_tracker_history, run_process_path
from rpipe.structure.artifact._atomic import atomic_write_text
from rpipe.structure.artifact.result import load_result
from rpipe.structure.artifact.result.format import encode_result


def run(ctx: FlowContext) -> None:
    result = ctx.state.get('result')
    if not isinstance(result, dict) and ctx.layout.result_path.is_file():
        try:
            result = load_result(ctx.layout.result_path)
        except (OSError, TypeError, ValueError):
            result = None

    run_id = None
    if ctx.control is not None:
        run_id = ctx.control.id
    elif isinstance(result, dict):
        control = result.get('control') if isinstance(result.get('control'), dict) else {}
        run_id = control.get('id') or ctx.layout.root.name
    else:
        run_id = ctx.layout.root.name

    metrics = {}
    if isinstance(result, dict):
        metrics = dict(result.get('metrics') or {})

    history = load_tracker_history(ctx.study_dir, str(run_id or ctx.layout.root.name))
    body = {
        'scope': 'run',
        'study': ctx.study_dir.name,
        'run_id': run_id,
        'metrics': metrics,
        'history': history,
    }
    atomic_write_text(run_process_path(ctx.layout.root), encode_result(body))
    ctx.state['process'] = body
