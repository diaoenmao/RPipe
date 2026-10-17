"""ModelConfig dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from rpipe.structure.control.mapping import emit, split_known, _layer_mapping, _layer_from_mapping


@dataclass
class ModelConfig:
    name: str | None = None
    source: str | None = None
    path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> ModelConfig:
        return _layer_from_mapping(cls, mapping, ('name', 'source', 'path'))

    def to_mapping(self) -> dict[str, Any]:
        return _layer_mapping(self, ('name', 'source', 'path'))
