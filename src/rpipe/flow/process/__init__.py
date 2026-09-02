"""process: Study-level aggregates from sibling results (sidecar only)."""

from __future__ import annotations

from rpipe.flow.context import FlowContext
from rpipe.flow.process.aggregate import (
    collect_from_index,
    collect_from_runs,
    process_path,
    run_delta,
    summarize,
)
from rpipe.flow.process.curves import write_learning_curves
from rpipe.structure.artifact._atomic import atomic_write_text
from rpipe.structure.artifact.index import load_index
from rpipe.structure.artifact.paths import DERIVED_NAME
from rpipe.structure.artifact.result import load_result
from rpipe.structure.artifact.result.format import encode_result


def run(ctx: FlowContext) -> None:
    study_dir = ctx.study_dir
    result = ctx.state.get('result')
    if not isinstance(result, dict) and ctx.layout.result_path.is_file():
        try:
            result = load_result(ctx.layout.result_path)
        except (OSError, TypeError, ValueError):
            result = None

    index = None
    try:
        index = load_index(study_dir)
    except (OSError, TypeError):
        index = None

    if index is not None:
        groups = collect_from_index(study_dir, index)
        study_name = str(index.get('study') or study_dir.name)
    else:
        groups = collect_from_runs(study_dir)
        study_name = study_dir.name

    source_run = None
    if ctx.control is not None:
        source_run = ctx.control.id
    elif isinstance(result, dict):
        control = result.get('control') if isinstance(result.get('control'), dict) else {}
        source_run = control.get('id')

    body = summarize(groups, study=study_name, source_run=source_run)
    figure = write_learning_curves(study_dir, index, title=f'{study_name} · mean ± std')
    if figure is not None:
        body['figures'] = {
            'learning_curves': str(figure.relative_to(study_dir)).replace('\\', '/')
        }
    atomic_write_text(process_path(study_dir), encode_result(body))
    ctx.state['process'] = body

    baseline_means = None
    for exp in body.get('experiments') or []:
        if exp.get('baseline') and exp.get('n'):
            baseline_means = {
                key: stat['mean']
                for key, stat in (exp.get('metrics') or {}).items()
                if isinstance(stat, dict) and stat.get('mean') is not None
            }
            break

    this_metrics: dict[str, float] = {}
    if isinstance(result, dict):
        raw = result.get('metrics') or {}
        this_metrics = {
            key: float(value)
            for key, value in raw.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    derived = {
        'source_run': source_run,
        'vs_baseline': run_delta(this_metrics, baseline_means),
    }
    atomic_write_text(ctx.layout.root / DERIVED_NAME, encode_result(derived))
