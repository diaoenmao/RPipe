"""AlgorithmConfig dataclass."""

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
class AlgorithmConfig:
    mode: str | None = None
    source: str | None = None
    path: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    compat_model: Any = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> AlgorithmConfig:
        known_names = {f.name for f in fields(cls) if f.name != 'extras'}
        body, extras = _split_known(dict(mapping or {}), known_names)
        mode = body.get('mode')
        if mode is None and 'semantics' in extras:
            sem = extras.pop('semantics')
            if isinstance(sem, list) and sem:
                mode = str(sem[0])
            elif isinstance(sem, str):
                mode = sem
        return cls(
            mode=mode,
            source=body.get('source'),
            path=body.get('path'),
            config=dict(body.get('config') or {}),
            compat_model=body.get('compat_model'),
            extras=extras,
        )

    def to_mapping(self) -> dict[str, Any]:
        return _emit(
            {
                'mode': self.mode,
                'source': self.source,
                'path': self.path,
                'config': dict(self.config) if self.config else None,
                'compat_model': self.compat_model,
            },
            self.extras,
        )

    def setting(self, key: str, default: Any = None) -> Any:
        if key in self.config:
            return self.config[key]
        return self.extras.get(key, default)
