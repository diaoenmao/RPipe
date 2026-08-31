"""YAML mapping encode / decode for Run config."""

from __future__ import annotations

from typing import Any

import yaml

from rpipe.structure.artifact.errors import CorruptArtifactError


def decode_config(text: str) -> dict[str, Any]:
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise CorruptArtifactError('Config must be a mapping')
    return data


def encode_config(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
