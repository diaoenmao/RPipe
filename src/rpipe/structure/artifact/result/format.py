"""JSON mapping encode / decode for result."""

from __future__ import annotations

import json
from typing import Any

from rpipe.structure.artifact.errors import CorruptArtifactError

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
