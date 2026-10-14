"""Print Study run.log event lines in timestamp order. Does not write a file."""

from __future__ import annotations

import re
from pathlib import Path

from rpipe.structure.artifact.index import load_index
from rpipe.structure.artifact.readout.status import _read_optional, log_path

_STAMP = re.compile(r'^(\d{4}-\d{2}-\d{2}T\S+)')
_EVENT = re.compile(r'\[(epoch|error|flow|warn|ckpt|resume)\]\s*(.*)$')


def event_lines(study_dir: Path | str) -> list[str]:
    study_dir = Path(study_dir).resolve()
    index = load_index(study_dir)
    rows: list[tuple[str, str]] = []
    for exp in index.get('experiments') or []:
        for run in exp.get('runs') or []:
            run_dir = str(run.get('run_dir') or run.get('id') or '')
            path = log_path(study_dir, run, run_dir)
            state, text = _read_optional(path, lambda p: p.read_text(encoding='utf-8', errors='replace'))
            if state != 'available':
                continue
            for line in text.splitlines():
                match = _EVENT.search(line)
                if match is None:
                    continue
                payload = match.group(2).strip()
                if match.group(1) == 'error' and (
                    payload.startswith('Traceback') or payload.startswith('File ')
                ):
                    continue
                stamp = _STAMP.match(line)
                rows.append((stamp.group(1) if stamp else '9999', line))
    rows.sort(key=lambda item: item[0])
    return [line for _, line in rows]


def format_logs(study_dir: Path | str) -> str:
    lines = event_lines(study_dir)
    if not lines:
        return ''
    return '\n'.join(lines) + '\n'
