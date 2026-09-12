"""Read sibling results and assemble Experiment aggregates (not a result rewrite)."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from rpipe.structure.artifact.asset import kinds
from rpipe.structure.artifact.paths import PROCESS_NAME, RESULT_NAME, RUNS_DIRNAME
from rpipe.structure.artifact.result import STATUS_SUCCEEDED, load_result


def process_path(study_dir: Path | str) -> Path:
    return Path(study_dir) / PROCESS_NAME


def run_process_path(run_root: Path | str) -> Path:
    return Path(run_root) / PROCESS_NAME


def load_tracker_history(study_dir: Path | str, run_dir: str) -> dict[str, dict[str, list[float]]]:
    path = Path(study_dir) / RUNS_DIRNAME / run_dir / 'assets' / kinds.TRACKER_STATE
    if not path.is_file():
        return {}
    try:
        body = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    splits = body.get('splits') if isinstance(body, dict) else None
    if not isinstance(splits, dict):
        return {}
    out: dict[str, dict[str, list[float]]] = {}
    for split, meters in splits.items():
        if not isinstance(meters, dict):
            continue
        names: dict[str, list[float]] = {}
        for name, meter in meters.items():
            history = meter.get('history') if isinstance(meter, dict) else None
            if not isinstance(history, list):
                continue
            values = [
                float(value)
                for value in history
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            if values:
                names[str(name)] = values
        if names:
            out[str(split)] = names
    return out


def attach_histories(
    study_dir: Path | str,
    groups: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
) -> None:
    for group, exp in zip(groups, experiments):
        buckets: dict[str, dict[str, list[list[float]]]] = {}
        for row in group.get('runs') or []:
            if row.get('status') != STATUS_SUCCEEDED:
                continue
            run_dir = str(row.get('id') or '')
            if not run_dir:
                continue
            for split, names in load_tracker_history(study_dir, run_dir).items():
                split_map = buckets.setdefault(split, {})
                for name, series in names.items():
                    split_map.setdefault(name, []).append(series)
        history: dict[str, dict[str, Any]] = {}
        for split, names in buckets.items():
            packed = {}
            for name, series in names.items():
                summary = summarize_histories(series)
                if summary is not None:
                    packed[name] = summary
            if packed:
                history[split] = packed
        exp['history'] = history


def _mean_std(values: list[float]) -> dict[str, float | int | None]:
    return summarize_numbers(values)


def summarize_numbers(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {'mean': None, 'std': None, 'min': None, 'max': None, 'n': 0}
    return {
        'mean': float(statistics.fmean(values)),
        'std': float(statistics.stdev(values)) if len(values) >= 2 else 0.0,
        'min': float(min(values)),
        'max': float(max(values)),
        'n': len(values),
    }


def summarize_histories(series: list[list[float]]) -> dict[str, Any] | None:
    rows = [row for row in series if row]
    if not rows:
        return None
    length = min(len(row) for row in rows)
    means: list[float] = []
    stds: list[float] = []
    mins: list[float] = []
    maxs: list[float] = []
    for i in range(length):
        col = [row[i] for row in rows]
        means.append(float(statistics.fmean(col)))
        stds.append(float(statistics.stdev(col)) if len(col) >= 2 else 0.0)
        mins.append(float(min(col)))
        maxs.append(float(max(col)))
    return {
        'mean': means,
        'std': stds,
        'min': mins,
        'max': maxs,
        'n': len(rows),
        'length': length,
    }


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
        'paired': pair_train_eval(experiments),
    }


def _mode_from_factors(factors: dict[str, Any]) -> str:
    raw = factors.get('algorithm.mode')
    if raw is None:
        return 'train'
    return str(raw)


def pair_train_eval(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join train/eval Experiment cells that differ only by ``algorithm.mode``."""
    buckets: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for exp in experiments:
        factors = dict(exp.get('factors') or {})
        shared = {k: v for k, v in factors.items() if k != 'algorithm.mode'}
        token = json.dumps(shared, sort_keys=True, default=str)
        if token not in buckets:
            buckets[token] = {'factors': shared, 'train': None, 'eval': None}
            order.append(token)
        mode = _mode_from_factors(factors)
        if mode == 'eval':
            buckets[token]['eval'] = exp
        else:
            buckets[token]['train'] = exp
    out: list[dict[str, Any]] = []
    for token in order:
        item = buckets[token]
        train = item['train']
        ev = item['eval']
        if train is None and ev is None:
            continue
        row: dict[str, Any] = {'factors': item['factors']}
        if train is not None:
            row['train'] = {
                'metrics': train.get('metrics') or {},
                'n': train.get('n'),
                'runs': train.get('runs') or [],
            }
        if ev is not None:
            row['eval'] = {
                'metrics': ev.get('metrics') or {},
                'n': ev.get('n'),
                'runs': ev.get('runs') or [],
            }
        out.append(row)
    return out


def run_delta(metrics: dict[str, float], baseline_means: dict[str, float] | None) -> dict[str, float] | None:
    if not baseline_means:
        return None
    return {
        key: metrics[key] - baseline_means[key]
        for key in metrics
        if key in baseline_means
    }
