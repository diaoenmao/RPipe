"""Config entity IO (Study writes; prepare reads)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.structure.artifact._atomic import atomic_write_text
from rpipe.structure.artifact.config.format import decode_config, encode_config
from rpipe.structure.artifact.errors import MissingConfigError


def load_config(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise MissingConfigError(f'missing config: {target}')
    return decode_config(target.read_text(encoding='utf-8'))


def write_config(path: Path | str, data: dict[str, Any]) -> Path:
    return atomic_write_text(path, encode_config(data))
