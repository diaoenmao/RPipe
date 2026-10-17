"""Config entity IO (Study writes; prepare reads)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.structure.artifact._atomic import _read_text, atomic_write_text
from rpipe.structure.artifact.layout import MissingConfigError

import yaml

from rpipe.structure.artifact.layout import CorruptArtifactError

def decode_config(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text)
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise CorruptArtifactError('Config must be a mapping')
    return data

def encode_config(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def load_config(path: Path | str) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise MissingConfigError(f'missing config: {target}')
    return decode_config(_read_text(target))


def write_config(path: Path | str, data: dict[str, Any]) -> Path:
    return atomic_write_text(path, encode_config(data))
