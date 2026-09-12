"""Split known dataclass fields from extras; omit empty values on emit."""

from __future__ import annotations

from typing import Any


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
