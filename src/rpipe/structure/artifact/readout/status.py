"""Read-only Study listing: index plan joined with each Run result."""

from __future__ import annotations

import re
import stat
from pathlib import Path
from typing import Any

import yaml

from rpipe.structure.artifact.asset import kinds
from rpipe.structure.artifact.config import load_config
from rpipe.structure.artifact.errors import ArtifactError
from rpipe.structure.artifact.index import index_path, load_index
from rpipe.structure.artifact.paths import CONFIG_NAME, RESULT_NAME, RUNS_DIRNAME
from rpipe.structure.artifact.readout.report import _fmt, _text_cell
from rpipe.structure.artifact.result import STATUS_SUCCEEDED, load_result
from rpipe.structure.artifact.result.format import VALID_STATUS

STATUS_PENDING = 'pending'
_EVENT = re.compile(r'\[(epoch|error|flow)\]\s*(.*)$')


def _read_optional(path: Path | None, loader) -> tuple[str, Any]:
    """Three read states for display only; strict loaders keep their contract."""
    try:
        if path is None:
            return 'missing', None
        if not path.is_file():
            return ('unreadable' if path.exists() else 'missing'), None
        return 'available', loader(path)
    except FileNotFoundError:
        return 'missing', None
    except (OSError, UnicodeError, ValueError, ArtifactError, yaml.YAMLError):
        return 'unreadable', None


def _mode(factors, result, config_path: Path, planned) -> tuple[str, Any]:
    state, config = _read_optional(config_path, load_config)
    if state == 'unreadable':
        raise ValueError('unreadable planned config')
    control = result.get('control') if isinstance(result, dict) else None
    control = control if isinstance(control, dict) else {}
    algorithm = control.get('algorithm')
    recorded = algorithm.get('mode') if isinstance(algorithm, dict) else None
    legacy = [value for value in (factors.get('algorithm.mode'), recorded) if value]
    seed = planned.get('seed')
    if config is None:
        if legacy and isinstance(legacy[0], str):
            return legacy[0], seed
        raise ValueError('missing planned config and legacy mode')
    algorithm = config.get('algorithm')
    if algorithm is not None and not isinstance(algorithm, dict):
        raise ValueError('algorithm must be a mapping')
    mode = (algorithm or {}).get('mode')
    mode = 'train' if mode is None else mode
    if not isinstance(mode, str) or mode not in ('train', 'eval', 'inference'):
        raise ValueError('invalid algorithm mode')
    if any(value != mode for value in legacy):
        raise ValueError('mode conflicts with planned config')
    for field, expected in (('id', planned.get('id')), ('seed', seed)):
        value = config.get(field)
        if value is not None and ((field == 'seed' and type(value) is not int)
                                  or (expected is not None and value != expected)):
            raise ValueError('config identity conflict')
        recorded = control.get(field)
        if recorded is not None and ((field == 'seed' and type(recorded) is not int)
                                     or (value is not None and recorded != value)):
            raise ValueError('result identity conflict')
    return mode, seed if config.get('seed') is None else config['seed']


def _factor_label(factors: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in sorted((factors or {}).items()):
        if key == 'algorithm.mode':
            continue
        parts.append(f'{str(key).split(".")[-1]}={value}')
    return ','.join(parts) or '-'


def _log_path_key(path: Path) -> Path:
    text = path.as_posix()
    if path.drive and text[:8].lower() == '//?/unc/':
        return Path('//' + text[8:])
    if len(path.drive) == 6 and text.startswith('//?/') and path.drive.endswith(':'):
        return Path(text[4:])
    return path


def log_path(study_dir: Path, run: dict[str, Any], run_dir: str) -> Path | None:
    if not run_dir:
        return None
    raw = run.get('log')
    if raw is None or raw == '':
        raw = f'{RUNS_DIRNAME}/{run_dir}/assets/{kinds.RUN_LOG}'
    if (not isinstance(raw, str) or any(ord(char) < 32 for char in raw)
            or raw.splitlines() != [raw]):
        return None
    try:
        study_dir = Path(study_dir).resolve()
        root = study_dir / RUNS_DIRNAME / run_dir
        path = study_dir / raw
        try:
            if not stat.S_ISREG(path.stat().st_mode):
                return None
        except FileNotFoundError:
            pass
        resolved = path.resolve()
        key = _log_path_key(resolved)
        key.relative_to(_log_path_key(root.resolve()))
        key.relative_to(_log_path_key(study_dir))
        try:
            _log_path_key(path).relative_to(_log_path_key(root))
        except ValueError:
            path = resolved
    except (OSError, ValueError):
        return None
    return path


def _note_from_log(path: Path | None, *, errors_only: bool = False) -> str:
    """Last ``[epoch]`` or ``[error]`` summary, else the last ``[flow]`` line.

    ``errors_only`` keeps the last ``[error]`` summary and ignores later epochs.
    """
    state, text = _read_optional(path, lambda p: p.read_text(encoding='utf-8', errors='replace'))
    if state != 'available':
        return '-'
    lines = text.splitlines()
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
        cell = _fmt(metrics.get(key))
        if cell != '-':
            return f'{key}={cell}'
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
            result_path = study_dir / RUNS_DIRNAME / run_dir / RESULT_NAME
            result_state, result = _read_optional(result_path if run_dir else None, load_result)
            recorded_status = result.get('status') if result is not None else None
            if recorded_status is not None and (
                not isinstance(recorded_status, str)
                or (recorded_status not in VALID_STATUS and recorded_status not in ('', STATUS_PENDING))
            ):
                result_state, result = 'unreadable', None
            config_path = study_dir / RUNS_DIRNAME / run_dir / CONFIG_NAME
            mode_error = None
            seed = run.get('seed')
            try:
                mode, seed = _mode(factors, result, config_path, run)
            except (OSError, TypeError, ValueError, ArtifactError, yaml.YAMLError) as exc:
                mode = 'unknown'
                mode_error = f'mode config unreadable or conflicting ({type(exc).__name__}): {config_path}'
            if wanted and mode not in wanted:
                continue
            status = STATUS_PENDING
            error = None
            metrics: dict[str, Any] = {}
            if result is not None:
                status = result.get('status') or STATUS_PENDING
                error = result.get('error')
                if status == STATUS_SUCCEEDED:
                    metrics = result.get('metrics') or {}
            run_log = log_path(study_dir, run, run_dir)
            note = '-'
            if status == STATUS_PENDING:
                note = _note_from_log(run_log)
            elif status == 'failed':
                note = _note_from_log(run_log, errors_only=True)
            error = None if status == STATUS_SUCCEEDED else error
            if result_state == 'unreadable':
                error = f'result unreadable: {result_path}'
            if mode_error:
                error = f'{error}; {mode_error}' if error else mode_error
            rows.append(
                {
                    'id': run.get('id') or run_dir,
                    'seed': seed,
                    'mode': mode,
                    'status': status,
                    'factors': _factor_label(factors),
                    'metric': _metric_cell(metrics),
                    'error': error,
                    'note': note,
                    'log': run.get('log') if run_log is not None else None,
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
        f"{_text_cell(body.get('study'))}  "
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
        error_text = '-' if not error else str(error)
        note = str(row.get('note') or '-')
        log = row.get('log') or '-'
        lines.append(
            '\t'.join(
                _text_cell(cell) for cell in [
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
