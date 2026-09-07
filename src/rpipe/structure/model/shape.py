"""Shape helpers for custom_torch builders (no torch import)."""

from __future__ import annotations

from typing import Any


def resolve_shape(cfg: dict[str, Any], data_meta: dict[str, Any] | None) -> tuple[tuple[int, ...], int]:
    meta = dict(data_meta or {})
    target_size = int(cfg.get('target_size') or cfg.get('out_features') or meta.get('target_size') or 10)
    if cfg.get('data_size') is not None:
        data_size = tuple(int(x) for x in cfg['data_size'])
    elif meta.get('data_size') is not None:
        data_size = tuple(int(x) for x in meta['data_size'])
    elif cfg.get('in_features') is not None:
        data_size = (int(cfg['in_features']),)
    else:
        data_size = (1, 28, 28)
    return data_size, target_size
