"""Stable hash for RunConfig.id (hash of all fields except id)."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(mapping: dict[str, Any]) -> str:
    """Stable JSON: sorted keys, no id field, compact separators."""
    body = {k: v for k, v in mapping.items() if k != 'id'}
    return json.dumps(body, sort_keys=True, separators=(',', ':'), default=str)


def compute_run_id(mapping: dict[str, Any], *, length: int = 16) -> str:
    """SHA-256 hex digest of canonical content, truncated to ``length`` chars."""
    digest = hashlib.sha256(canonical_json(mapping).encode('utf-8')).hexdigest()
    return digest[:length]
