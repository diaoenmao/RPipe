"""Read-only Study listing: index plan joined with each Run result."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from rpipe.structure.artifact.asset import kinds
from rpipe.structure.artifact.index import index_path, load_index
from rpipe.structure.artifact.paths import RESULT_NAME, RUNS_DIRNAME
from rpipe.structure.artifact.result import STATUS_SUCCEEDED, load_result

STATUS_PENDING = 'pending'
_EVENT = re.compile(r'\[(epoch|error|flow)\]\s*(.*)$')


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


def log_path(study_dir: Path, run: dict[str, Any], run_dir: str) -> Path | None:
    raw = run.get('log')
    if raw:
        return study_dir / str(raw)
    if not run_dir:
        return None
    return study_dir / RUNS_DIRNAME / run_dir / 'assets' / kinds.RUN_LOG


def _note_from_log(path: Path | None, *, errors_only: bool = False) -> str:
    """Last ``[epoch]`` or ``[error]`` summary, else the last ``[flow]`` line.

    ``errors_only`` keeps the last ``[error]`` summary and ignores later epochs.
    """
    if path is None or not path.is_file():
        return '-'
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return '-'
    chosen = None
    fallback = None
    for line in lines:
        match = _EVENT.search(line)
        if match is None:
            continue
        event, payload = match.group(1), match.group(2).strip()
        if event == 'error' and (payload.startswith('Traceback') or payload.startswith('File ')):
            continue
        text = f'[{event}] {payload}'.strip()
        if event == 'error':
            chosen = text
        elif not errors_only and event == 'epoch':
            chosen = text
        elif not errors_only and event == 'flow':
            fallback = text
    note = chosen or (None if errors_only else fallback) or '-'
    return note.replace('\t', ' ').replace('\n', ' ')


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
            note = '-'
            if status == STATUS_PENDING:
                note = _note_from_log(log_path(study_dir, run, run_dir))
            elif status == 'failed':
                note = _note_from_log(log_path(study_dir, run, run_dir), errors_only=True)
            rows.append(
                {
                    'id': run.get('id') or run_dir,
                    'seed': run.get('seed'),
                    'mode': mode,
                    'status': status,
                    'factors': _factor_label(factors),
                    'metric': _metric_cell(metrics),
                    'error': None if status == STATUS_SUCCEEDED else error,
                    'note': note,
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
    lines.append('status\tmode\tseed\tid\tfactors\tmetric\terror\tnote\tlog')
    for row in body.get('runs') or []:
        seed = row.get('seed')
        seed_text = '' if seed is None else str(seed)
        error = row.get('error')
        error_text = '-' if not error else str(error).replace('\t', ' ').replace('\n', ' ')
        note = str(row.get('note') or '-').replace('\t', ' ').replace('\n', ' ')
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
                    note,
                    str(log),
                ]
            )
        )
    return '\n'.join(lines) + '\n'
