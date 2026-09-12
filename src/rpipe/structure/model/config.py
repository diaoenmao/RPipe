"""ModelConfig dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from rpipe.structure.control.mapping import emit, split_known


@dataclass
class ModelConfig:
    name: str | None = None
    source: str | None = None
    path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> ModelConfig:
        known_names = {f.name for f in fields(cls) if f.name != 'extras'}
        body, extras = split_known(dict(mapping or {}), known_names)
        return cls(
            name=body.get('name'),
            source=body.get('source'),
            path=body.get('path'),
            config=dict(body.get('config') or {}),
            extras=extras,
        )

    def to_mapping(self) -> dict[str, Any]:
        return emit(
            {
                'name': self.name,
                'source': self.source,
                'path': self.path,
                'config': dict(self.config) if self.config else None,
            },
            self.extras,
        )
