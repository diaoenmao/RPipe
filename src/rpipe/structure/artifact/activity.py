"""Current Study task, so a long make can be seen without waiting for the final print."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rpipe.structure.artifact._atomic import _read_text, atomic_write_text

ACTIVITY_NAME = 'activity.json'


def activity_path(study_dir: Path | str) -> Path:
    return Path(study_dir) / ACTIVITY_NAME


def read_activity(study_dir: Path | str) -> dict[str, Any] | None:
    path = activity_path(study_dir)
    try:
        if not path.is_file():
            return None
        body = json.loads(_read_text(path))
    except (OSError, UnicodeError, ValueError, RecursionError, TypeError):
        return None
    if (not isinstance(body, dict)
            or not isinstance(body.get('phase'), str)
            or not body['phase'].strip()
            or (body.get('detail') is not None and not isinstance(body['detail'], str))):
        return None
    return body


def clear_activity(study_dir: Path | str) -> None:
    path = activity_path(study_dir)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _format_activity(phase: str, detail: str) -> str:
    return ' '.join(f'{phase}: {detail}'.splitlines()).replace('\t', ' ').encode('utf-8', errors='backslashreplace').decode('utf-8')


def announce(study_dir: Path | str, phase: str, detail: str) -> None:
    """Write ``activity.json`` and print the same line immediately."""
    body = {'phase': phase, 'detail': detail}
    atomic_write_text(
        activity_path(study_dir),
        json.dumps(body, ensure_ascii=False) + '\n',
    )
    print(_format_activity(phase, detail), flush=True)
