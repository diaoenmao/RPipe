"""Read-only Study listing: index plan joined with each Run result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.structure.artifact.index import index_path, load_index
from rpipe.structure.artifact.paths import RESULT_NAME, RUNS_DIRNAME
from rpipe.structure.artifact.result import STATUS_SUCCEEDED, load_result

STATUS_PENDING = 'pending'


def _load_result(study_dir: Path, run_dir: str) -> dict[str, Any] | None:
    path = study_dir / RUNS_DIRNAME / run_dir / RESULT_NAME
    if not path.is_file():
        return None
    try:
        return load_result(path)
    except (OSError, TypeError, ValueError):
        return None


def _mode(factors: dict[str, Any], result: dict[str, Any] | None) -> str:
    raw = factors.get('algorithm.mode')
    if raw:
        return str(raw)
    if isinstance(result, dict):
        control = result.get('control')
        if isinstance(control, dict):
            algo = control.get('algorithm')
            if isinstance(algo, dict) and algo.get('mode'):
                return str(algo['mode'])
    return 'train'


def _factor_label(factors: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in sorted((factors or {}).items()):
        if key == 'algorithm.mode':
            continue
        parts.append(f'{str(key).split(".")[-1]}={value}')
    return ','.join(parts) or '-'


def _metric_cell(metrics: dict[str, Any]) -> str:
    if not isinstance(metrics, dict):
        return '-'
    for key in ('accuracy', 'best_accuracy', 'train_loss'):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f'{key}={float(value):.4f}'
    return '-'


def list_runs(
    study_dir: Path | str,
    *,
    modes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Join ``index.json`` with each Run ``result.json``. Does not write."""
    study_dir = Path(study_dir).resolve()
    index = load_index(study_dir)
    wanted = {str(mode).strip().lower() for mode in (modes or []) if str(mode).strip()}
    rows: list[dict[str, Any]] = []
    for exp in index.get('experiments') or []:
        factors = dict(exp.get('factors') or {})
        for run in exp.get('runs') or []:
            run_dir = str(run.get('run_dir') or run.get('id') or '')
            result = _load_result(study_dir, run_dir) if run_dir else None
            mode = _mode(factors, result)
            if wanted and mode not in wanted:
                continue
            status = STATUS_PENDING
            error = None
            metrics: dict[str, Any] = {}
            if result is not None:
                status = str(result.get('status') or STATUS_PENDING)
                error = result.get('error')
                if status == STATUS_SUCCEEDED:
                    metrics = dict(result.get('metrics') or {})
            rows.append(
                {
                    'id': run.get('id') or run_dir,
                    'seed': run.get('seed'),
                    'mode': mode,
                    'status': status,
                    'factors': _factor_label(factors),
                    'metric': _metric_cell(metrics),
                    'error': None if status == STATUS_SUCCEEDED else error,
                    'log': run.get('log'),
                }
            )
    counts = {
        'planned': len(rows),
        'succeeded': sum(1 for row in rows if row['status'] == STATUS_SUCCEEDED),
        'failed': sum(1 for row in rows if row['status'] == 'failed'),
        'pending': sum(1 for row in rows if row['status'] == STATUS_PENDING),
    }
    return {
        'study': index.get('study') or study_dir.name,
        'index': str(index_path(study_dir)),
        'counts': counts,
        'runs': rows,
    }


def format_status(body: dict[str, Any]) -> str:
    counts = body.get('counts') or {}
    lines = [
        f"{body.get('study')}  "
        f"planned={counts.get('planned', 0)} "
        f"succeeded={counts.get('succeeded', 0)} "
        f"failed={counts.get('failed', 0)} "
        f"pending={counts.get('pending', 0)}"
    ]
    lines.append('status\tmode\tseed\tid\tfactors\tmetric\terror\tlog')
    for row in body.get('runs') or []:
        seed = row.get('seed')
        seed_text = '' if seed is None else str(seed)
        error = row.get('error')
        error_text = '-' if not error else str(error).replace('\t', ' ').replace('\n', ' ')
        log = row.get('log') or '-'
        lines.append(
            '\t'.join(
                [
                    str(row.get('status') or '-'),
                    str(row.get('mode') or '-'),
                    seed_text,
                    str(row.get('id') or '-'),
                    str(row.get('factors') or '-'),
                    str(row.get('metric') or '-'),
                    error_text,
                    str(log),
                ]
            )
        )
    return '\n'.join(lines) + '\n'
