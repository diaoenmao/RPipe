"""Write Experiment and Run number tables from process.json. Does not write a conclusion."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from urllib.parse import quote

from rpipe.structure.artifact._atomic import atomic_write_text
from rpipe.structure.artifact.errors import CorruptArtifactError
from rpipe.structure.artifact.index import index_path, load_index
from rpipe.structure.artifact.paths import DOCS_DIRNAME, PROCESS_NAME
from rpipe.structure.artifact.result import load_result


def numbers_path(study_dir: Path | str) -> Path:
    return Path(study_dir) / DOCS_DIRNAME / 'NUMBERS.md'


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return '-'
    try:
        number = float(value)
    except OverflowError:
        return '-'
    return f'{number:.4f}' if math.isfinite(number) else '-'


def _text_cell(value: Any) -> str:
    return ' '.join(str(value).splitlines()).replace('\t', ' ')


def _markdown_cell(value: Any) -> str:
    return _text_cell(value).replace('\\', '\\\\').replace('|', '\\|')


def _factors(factors: dict[str, Any]) -> str:
    parts = [f'{key}={value}' for key, value in sorted((factors or {}).items())]
    return _markdown_cell(', '.join(parts)) or '-'


def _planned_runs(body: dict[str, Any]) -> dict[str, Any]:
    experiments = body.get('experiments')
    if not isinstance(experiments, list):
        raise ValueError('experiments must be a list')
    planned: dict[str, Any] = {}
    for exp in experiments:
        if not isinstance(exp, dict) or not isinstance(exp.get('runs'), list):
            raise ValueError('each experiment must contain a runs list')
        factors = exp.get('factors', {})
        if not isinstance(factors, dict):
            raise ValueError('experiment factors must be an object')
        for run in exp['runs']:
            if not isinstance(run, dict) or not isinstance(run.get('id'), str) or not run['id']:
                raise ValueError('each planned Run must have a nonempty string id')
            if run['id'] in planned:
                raise ValueError(f'duplicate planned Run: {run["id"]}')
            planned[run['id']] = {'seed': run.get('seed'), 'factors': factors}
    return planned


def _log_links(index: dict[str, Any], study_dir: Path) -> dict[str, str]:
    # status imports this module's formatter; reuse its path guard after imports finish.
    from rpipe.structure.artifact.readout.status import _log_path_key, log_path

    links: dict[str, str] = {}
    for exp in index.get('experiments') or []:
        for run in exp.get('runs') or []:
            run_id = str(run.get('id') or '')
            if not run_id:
                continue
            path = log_path(study_dir, run, str(run.get('run_dir') or run_id))
            links[run_id] = '../' + quote(_log_path_key(path).relative_to(_log_path_key(study_dir)).as_posix(), safe='/') if path is not None else ''
    return links


def render_numbers(study_dir: Path | str) -> str:
    study_dir = Path(study_dir)
    try:
        study_dir = study_dir.resolve()
        body = load_result(study_dir / PROCESS_NAME)
    except CorruptArtifactError as exc:
        raise ValueError(f'Cannot report invalid process: {exc}') from exc
    try:
        index = load_index(study_dir)
        planned = _planned_runs(index)
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f'Cannot report without a usable current index: {index_path(study_dir)}; '
                         f'check the plan and rebuild with make: {exc}') from exc
    try:
        if ((index.get('study') is not None and body.get('study') != index['study'])
                or _planned_runs(body) != planned):
            raise ValueError('Study name or planned Run identity differs')
    except (TypeError, ValueError) as exc:
        raise ValueError(f'process does not match the current index: {exc}; '
                         'check the plan and rerun process before report') from exc
    links = _log_links(index, study_dir)
    lines = [
        f'# Numbers: {_text_cell(body.get("study") or study_dir.name)}',
        '',
        'Generated from `process.json`. The conclusion stays in `STUDY_REPORT.md`.',
        '',
        '## Experiments',
        '',
        '| factors | metric | mean | std | min | max | n |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for exp in body.get('experiments') or []:
        factors = _factors(dict(exp.get('factors') or {}))
        metrics = exp.get('metrics')
        metrics = metrics if isinstance(metrics, dict) else {}
        if not metrics:
            lines.append(f'| {factors} | - | - | - | - | - | {exp.get("n", 0)} |')
            continue
        for name in sorted(metrics):
            stat = metrics[name] if isinstance(metrics[name], dict) else {}
            lines.append(
                '| '
                + ' | '.join(
                    [
                        factors,
                        _markdown_cell(name),
                        _fmt(stat.get('mean')),
                        _fmt(stat.get('std')),
                        _fmt(stat.get('min')),
                        _fmt(stat.get('max')),
                        str(stat.get('n', exp.get('n', 0))),
                    ]
                )
                + ' |'
            )
    lines.extend(
        [
            '',
            '## Runs',
            '',
            '| factors | seed | id | metrics | log |',
            '|---|---:|---|---|---|',
        ]
    )
    for exp in body.get('experiments') or []:
        factors = _factors(dict(exp.get('factors') or {}))
        for run in exp.get('runs') or []:
            run_id = str(run.get('id') or '')
            metrics = run.get('metrics')
            metrics = metrics if isinstance(metrics, dict) else {}
            metric_text = _markdown_cell(', '.join(f'{key}={_fmt(metrics[key])}' for key in sorted(metrics))) or '-'
            href = links.get(run_id, f'../runs/{run_id}/assets/logs/run.log')
            log_text = f'[run.log]({href})' if href else '-'
            seed = '' if run.get('seed') is None else _markdown_cell(run['seed'])
            lines.append(
                f'| {factors} | {seed} | `{_markdown_cell(run_id)}` | {metric_text} | {log_text} |'
            )
    lines.append('')
    return '\n'.join(lines)


def write_numbers(study_dir: Path | str) -> Path:
    return atomic_write_text(numbers_path(study_dir), render_numbers(study_dir))
