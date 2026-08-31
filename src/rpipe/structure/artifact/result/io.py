"""Result entity IO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.structure.artifact._atomic import atomic_write_text
from rpipe.structure.artifact.result.format import (
    STATUS_FAILED,
    STATUS_SUCCEEDED,
    decode_result,
    encode_result,
    validate_result,
)


def write_result(path: Path | str, data: dict[str, Any]) -> Path:
    errors = validate_result(data)
    if errors:
        raise ValueError('invalid result: ' + '; '.join(errors))
    return atomic_write_text(path, encode_result(data))


def load_result(path: Path | str) -> dict[str, Any]:
    return decode_result(Path(path).read_text(encoding='utf-8'))


__all__ = [
    'STATUS_FAILED',
    'STATUS_SUCCEEDED',
    'load_result',
    'validate_result',
    'write_result',
]
