"""Config mapping, deep merge, and stable content identifiers."""

from __future__ import annotations

import hashlib
import json
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


def _layer_mapping(layer: Any, names: tuple[str, ...]) -> dict[str, Any]:
    known = {name: getattr(layer, name) for name in names}
    known['config'] = dict(layer.config) if layer.config else None
    return emit(known, layer.extras)


def deep_merge(base: Any, patch: Any) -> Any:
    """Recursively merge patch onto base. Patch wins on conflicts; lists replace."""
    if not isinstance(base, dict) or not isinstance(patch, dict):
        return patch
    out = dict(base)
    for key, value in patch.items():
        out[key] = deep_merge(out[key], value) if key in out else value
    return out



# Human metadata / self id — must not change content-addressed identity.
_HASH_EXCLUDE = frozenset({'id', 'description'})


def canonical_json(mapping: dict[str, Any]) -> str:
    """Stable JSON: sorted keys, exclude id/description, compact separators."""
    body = {k: v for k, v in mapping.items() if k not in _HASH_EXCLUDE}
    return json.dumps(body, sort_keys=True, separators=(',', ':'), default=str)


def compute_content_id(mapping: dict[str, Any], *, length: int = 16) -> str:
    """SHA-256 hex digest of canonical content, truncated to ``length`` chars."""
    digest = hashlib.sha256(canonical_json(mapping).encode('utf-8')).hexdigest()
    return digest[:length]


def compute_run_id(mapping: dict[str, Any], *, length: int = 16) -> str:
    return compute_content_id(mapping, length=length)


def compute_index_id(mapping: dict[str, Any], *, length: int = 16) -> str:
    return compute_content_id(mapping, length=length)
