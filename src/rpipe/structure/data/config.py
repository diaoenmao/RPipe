"""DataConfig dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from rpipe.structure.control.mapping import emit, split_known, _layer_mapping


@dataclass
class DataConfig:
    name: str | None = None
    source: str | None = None
    path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> DataConfig:
        known_names = {f.name for f in fields(cls) if f.name != 'extras'}
        body, extras = split_known(mapping, known_names)
        if body.get('origin') not in (None, '') or extras.get('origin') not in (None, ''):
            raise ValueError('origin is set on the Study, not under data')
        return cls(
            name=body.get('name'),
            source=body.get('source'),
            path=body.get('path'),
            config=dict(body.get('config') or {}),
            extras=extras,
        )

    def to_mapping(self) -> dict[str, Any]:
        return _layer_mapping(self, ('name', 'source', 'path'))
