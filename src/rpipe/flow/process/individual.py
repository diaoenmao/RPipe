"""Run-scoped process: this Run only. Does not write Study-root process.json."""

from __future__ import annotations

from rpipe.flow.context import FlowContext
from rpipe.flow.process.aggregate import load_curve_series, run_process_path
from rpipe.structure.api.artifact_api import atomic_write_text
from rpipe.structure.api.artifact_api import CorruptArtifactError
from rpipe.structure.api.artifact_api import load_result
from rpipe.structure.api.artifact_api import encode_result


def run(ctx: FlowContext) -> None:
    result = ctx.state.get('result')
    if not isinstance(result, dict):
        try:
            if ctx.layout.result_path.is_file():
                result = load_result(ctx.layout.result_path)
        except (OSError, TypeError, ValueError, CorruptArtifactError, RecursionError):
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

    series = load_curve_series(ctx.study_dir, str(run_id or ctx.layout.root.name))
    history = {split: {name: row['values'] for name, row in names.items()} for split, names in series.items()}
    body = {
        'scope': 'run',
        'study': ctx.study_dir.name,
        'run_id': run_id,
        'metrics': metrics,
        'history': history,
        'history_coordinates': {
            split: {name: {'unit': row['unit'], 'x': row['x']} for name, row in names.items()}
            for split, names in series.items()
        },
    }
    atomic_write_text(run_process_path(ctx.layout.root), encode_result(body))
    ctx.state['process'] = body
