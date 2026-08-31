"""DataConfig dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any


def _split_known(mapping: dict[str, Any], known: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    body = dict(mapping or {})
    extras = {k: body.pop(k) for k in list(body) if k not in known}
    return body, extras


def _emit(known: dict[str, Any], extras: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in known.items():
        if value is None or value == {} or value == []:
            continue
        out[key] = value
    out.update(extras)
    return out


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
        body, extras = _split_known(dict(mapping or {}), known_names)
        return cls(
            name=body.get('name'),
            source=body.get('source'),
            path=body.get('path'),
            config=dict(body.get('config') or {}),
            extras=extras,
        )

    def to_mapping(self) -> dict[str, Any]:
        return _emit(
            {
                'name': self.name,
                'source': self.source,
                'path': self.path,
                'config': dict(self.config) if self.config else None,
            },
            self.extras,
        )
