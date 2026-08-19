"""Deep-merge helpers for experiment_config ⊕ patch → run_config."""

from __future__ import annotations

from typing import Any


def deep_merge(base: Any, patch: Any) -> Any:
    """Recursively merge patch onto base. Patch wins on conflicts; lists replace."""
    if not isinstance(base, dict) or not isinstance(patch, dict):
        return patch
    out = dict(base)
    for key, value in patch.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
