"""Split known dataclass fields from extras; omit empty values on emit."""

from __future__ import annotations

from typing import Any


def get_setting(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    """Read raw settings with the same config-before-extras precedence as runtime."""
    config = mapping.get('config') or {}
    if not isinstance(config, dict):
        raise ValueError('layer config must be a mapping')
    return config[key] if key in config else mapping.get(key, default)


def split_known(mapping: dict[str, Any], known: set[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    body = dict(mapping or {})
    extras = {k: body.pop(k) for k in list(body) if k not in known}
    return body, extras


def emit(known: dict[str, Any], extras: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in known.items():
        if value is None or value == {} or value == []:
            continue
        out[key] = value
    out.update(extras)
    return out
