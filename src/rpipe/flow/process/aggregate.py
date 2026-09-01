"""Read sibling results and assemble Experiment aggregates (not a result rewrite)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from rpipe.structure.artifact.paths import PROCESS_NAME, RESULT_NAME, RUNS_DIRNAME
from rpipe.structure.artifact.result import STATUS_SUCCEEDED, load_result


def process_path(study_dir: Path | str) -> Path:
    return Path(study_dir) / PROCESS_NAME


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {'mean': None, 'std': None, 'n': 0}
    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values)) if len(values) >= 2 else 0.0
    return {'mean': mean, 'std': std, 'n': len(values)}


def _numeric_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def _load_result(study_dir: Path, run_dir: str) -> dict[str, Any] | None:
    path = study_dir / RUNS_DIRNAME / run_dir / RESULT_NAME
    if not path.is_file():
        return None
    try:
        return load_result(path)
    except (OSError, TypeError, ValueError):
        return None


def collect_from_index(study_dir: Path, index: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for exp in index.get('experiments') or []:
        planned = list(exp.get('runs') or [])
        rows: list[dict[str, Any]] = []
        for run in planned:
            run_dir = str(run.get('run_dir') or run.get('id') or '')
            result = _load_result(study_dir, run_dir) if run_dir else None
            row: dict[str, Any] = {
                'id': run.get('id'),
                'seed': run.get('seed'),
                'tags': list(run.get('tags') or []),
                'status': None,
                'metrics': {},
            }
            if result is not None:
                row['status'] = result.get('status')
                if result.get('status') == STATUS_SUCCEEDED:
                    row['metrics'] = _numeric_metrics(result.get('metrics') or {})
            rows.append(row)
        groups.append(
            {
                'factors': dict(exp.get('factors') or {}),
                'n_planned': len(planned),
                'runs': rows,
            }
        )
    return groups


def _factors_from_control(control: dict[str, Any]) -> dict[str, Any]:
    return {
        'data': control.get('data') or {},
        'model': control.get('model') or {},
        'algorithm': control.get('algorithm') or {},
        'system': control.get('system') or {},
    }


def collect_from_runs(study_dir: Path) -> list[dict[str, Any]]:
    root = study_dir / RUNS_DIRNAME
    if not root.is_dir():
        return []
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for result_path in sorted(root.glob(f'*/{RESULT_NAME}')):
        try:
            result = load_result(result_path)
        except (OSError, TypeError, ValueError):
            continue
        control = result.get('control') if isinstance(result.get('control'), dict) else {}
        factors = _factors_from_control(control)
        token = json.dumps(factors, sort_keys=True, default=str)
        if token not in grouped:
            grouped[token] = {'factors': factors, 'n_planned': 0, 'runs': []}
            order.append(token)
        grouped[token]['n_planned'] += 1
        grouped[token]['runs'].append(
            {
                'id': control.get('id') or result_path.parent.name,
                'seed': control.get('seed'),
                'tags': list(control.get('tags') or []),
                'status': result.get('status'),
                'metrics': (
                    _numeric_metrics(result.get('metrics') or {})
                    if result.get('status') == STATUS_SUCCEEDED
                    else {}
                ),
            }
        )
    return [grouped[key] for key in order]


def summarize(
    groups: list[dict[str, Any]],
    *,
    study: str,
    source_run: str | None,
) -> dict[str, Any]:
    experiments: list[dict[str, Any]] = []
    for group in groups:
        succeeded = [row for row in group['runs'] if row.get('status') == STATUS_SUCCEEDED]
        tags: list[str] = []
        for row in group['runs']:
            tags.extend(row.get('tags') or [])
        keys = sorted({key for row in succeeded for key in row.get('metrics') or {}})
        metrics = {
            key: _mean_std(
                [float(row['metrics'][key]) for row in succeeded if key in row.get('metrics', {})]
            )
            for key in keys
        }
        experiments.append(
            {
                'factors': group['factors'],
                'baseline': 'baseline' in tags,
                'n': len(succeeded),
                'n_planned': int(group.get('n_planned') or 0),
                'metrics': metrics,
                'runs': [
                    {
                        'id': row.get('id'),
                        'seed': row.get('seed'),
                        'status': row.get('status'),
                        'metrics': row.get('metrics') or {},
                    }
                    for row in group['runs']
                ],
            }
        )

    baseline_means: dict[str, float] | None = None
    for exp in experiments:
        if exp['baseline'] and exp['n']:
            baseline_means = {
                key: stat['mean']
                for key, stat in exp['metrics'].items()
                if stat.get('mean') is not None
            }
            break

    for exp in experiments:
        if baseline_means and not exp['baseline'] and exp['n']:
            exp['delta_vs_baseline'] = {
                key: stat['mean'] - baseline_means[key]
                for key, stat in exp['metrics'].items()
                if stat.get('mean') is not None and key in baseline_means
            }
        else:
            exp['delta_vs_baseline'] = None

    complete = bool(experiments) and all(
        exp['n'] == exp['n_planned'] and exp['n_planned'] > 0 for exp in experiments
    )
    return {
        'study': study,
        'source_run': source_run,
        'complete': complete,
        'experiments': experiments,
    }


def run_delta(metrics: dict[str, float], baseline_means: dict[str, float] | None) -> dict[str, float] | None:
    if not baseline_means:
        return None
    return {
        key: metrics[key] - baseline_means[key]
        for key in metrics
        if key in baseline_means
    }
