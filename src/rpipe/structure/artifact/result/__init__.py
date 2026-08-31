"""Result package."""

from rpipe.structure.artifact.result.format import (
    STATUS_FAILED,
    STATUS_SUCCEEDED,
    validate_result,
)
from rpipe.structure.artifact.result.io import load_result, write_result

__all__ = [
    'STATUS_FAILED',
    'STATUS_SUCCEEDED',
    'load_result',
    'validate_result',
    'write_result',
]
