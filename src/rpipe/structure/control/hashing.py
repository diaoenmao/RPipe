"""Stable hash for RunConfig.id / Study index.id (hash of all fields except ignored)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Human metadata / self id — must not change content-addressed identity.
_HASH_EXCLUDE = frozenset({'id', 'description', 'tags'})


def canonical_json(mapping: dict[str, Any]) -> str:
    """Stable JSON: sorted keys, exclude id/description/tags, compact separators."""
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
