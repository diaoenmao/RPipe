"""Result entity IO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.structure.artifact._atomic import _read_text, atomic_write_text

import json

from rpipe.structure.artifact.layout import CorruptArtifactError

STATUS_SUCCEEDED = 'succeeded'

STATUS_FAILED = 'failed'

VALID_STATUS = frozenset({STATUS_SUCCEEDED, STATUS_FAILED})

SUCCEEDED_REQUIRED = ('control', 'metrics', 'paths')

def decode_result(text: str) -> dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise CorruptArtifactError('Result must be an object')
    return data

def encode_result(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)

def validate_result(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ['result must be an object']
    status = data.get('status')
    if not isinstance(status, str) or status not in VALID_STATUS:
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
    return atomic_write_text(path, encode_result(data))


def load_result(path: Path | str) -> dict[str, Any]:
    return decode_result(_read_text(Path(path)))


__all__ = [
    'STATUS_FAILED',
    'STATUS_SUCCEEDED',
    'load_result',
    'validate_result',
    'write_result',
]
