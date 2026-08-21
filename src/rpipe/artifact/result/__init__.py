"""Result entity IO and light validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STATUS_SUCCEEDED = 'succeeded'
STATUS_FAILED = 'failed'
VALID_STATUS = frozenset({STATUS_SUCCEEDED, STATUS_FAILED})
SUCCEEDED_REQUIRED = ('control', 'metrics', 'paths')


def validate_result(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ['result must be an object']
    status = data.get('status')
    if status not in VALID_STATUS:
        errors.append(f'status must be one of {sorted(VALID_STATUS)}')
        return errors
    if status == STATUS_SUCCEEDED:
        for key in SUCCEEDED_REQUIRED:
            if key not in data:
                errors.append(f'missing field: {key}')
    elif status == STATUS_FAILED and not data.get('error'):
        errors.append('failed result should include error')
    return errors


def write_result(path: Path | str, data: dict[str, Any]) -> Path:
    errors = validate_result(data)
    if errors:
        raise ValueError('invalid result: ' + '; '.join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return target


def load_result(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError('Result must be an object')
    return data
