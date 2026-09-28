"""Write Experiment and Run number tables from process.json. Does not write a conclusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.structure.artifact.index import load_index
from rpipe.structure.artifact.paths import DOCS_DIRNAME, PROCESS_NAME
from rpipe.structure.artifact.result import load_result


def numbers_path(study_dir: Path | str) -> Path:
    return Path(study_dir) / DOCS_DIRNAME / 'NUMBERS.md'


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return '-'
    return f'{float(value):.4f}'


def _factors(factors: dict[str, Any]) -> str:
    parts = [f'{key}={value}' for key, value in sorted((factors or {}).items())]
    return ', '.join(parts) or '-'


def _log_links(study_dir: Path) -> dict[str, str]:
    links: dict[str, str] = {}
    try:
        index = load_index(study_dir)
    except (OSError, TypeError, ValueError):
        return links
    for exp in index.get('experiments') or []:
        for run in exp.get('runs') or []:
            run_id = str(run.get('id') or '')
            if not run_id:
                continue
            raw = run.get('log') or f'runs/{run_id}/assets/logs/run.log'
            links[run_id] = '../' + str(raw).replace('\\', '/')
    return links


def render_numbers(study_dir: Path | str) -> str:
    study_dir = Path(study_dir)
    body = load_result(study_dir / PROCESS_NAME)
    links = _log_links(study_dir)
    lines = [
        f'# Numbers: {body.get("study") or study_dir.name}',
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
        metrics = exp.get('metrics') or {}
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
                        str(name),
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
            metrics = run.get('metrics') or {}
            metric_text = ', '.join(f'{key}={_fmt(metrics[key])}' for key in sorted(metrics)) or '-'
            href = links.get(run_id) or f'../runs/{run_id}/assets/logs/run.log'
            seed = '' if run.get('seed') is None else str(run.get('seed'))
            lines.append(
                f'| {factors} | {seed} | `{run_id}` | {metric_text} | [run.log]({href}) |'
            )
    lines.append('')
    return '\n'.join(lines)


def write_numbers(study_dir: Path | str) -> Path:
    study_dir = Path(study_dir)
    path = numbers_path(study_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_numbers(study_dir), encoding='utf-8')
    return path
