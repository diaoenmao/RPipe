"""Resume target resolution (algorithm-layer; system only loads files)."""

from __future__ import annotations

from typing import Any

from rpipe.structure.algorithm.config import AlgorithmConfig


def resume_stem(config: AlgorithmConfig, *, mode: str) -> str | None:
    """Return checkpoint stem to load, or None to skip resume."""
    explicit = config.setting('resume_from')
    if explicit:
        return str(explicit)
    raw = config.setting('resume')
    if raw is None:
        return 'latest' if mode == 'train' else 'best'
    if raw is False or raw is None:
        return None
    text = str(raw).strip().lower()
    if text in ('', 'false', '0', 'none', 'off'):
        return None
    return str(raw)


def apply_module_state(module: Any, payload: dict[str, Any] | None) -> None:
    if module is None or not payload:
        return
    state = payload.get('model')
    if state is None:
        return
    module.load_state_dict(state)
