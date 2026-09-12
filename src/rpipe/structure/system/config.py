"""SystemConfig dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from rpipe.structure.control.mapping import emit, split_known


@dataclass
class SystemConfig:
    source: str | None = None
    path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> SystemConfig:
        known_names = {f.name for f in fields(cls) if f.name != 'extras'}
        body, extras = split_known(dict(mapping or {}), known_names)
        return cls(
            source=body.get('source'),
            path=body.get('path'),
            config=dict(body.get('config') or {}),
            extras=extras,
        )

    def to_mapping(self) -> dict[str, Any]:
        return emit(
            {
                'source': self.source,
                'path': self.path,
                'config': dict(self.config) if self.config else None,
            },
            self.extras,
        )

    def setting(self, key: str, default: Any = None) -> Any:
        if key in self.config:
            return self.config[key]
        return self.extras.get(key, default)
