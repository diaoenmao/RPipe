"""Structure model layer."""

from __future__ import annotations

from typing import Any


def prepare_model(control_model: dict[str, Any], assets_dir) -> dict[str, Any]:
    """Land model handles for prepare."""
    name = control_model.get('name', 'unknown')
    cfg = dict(control_model.get('config') or {})
    out: dict[str, Any] = {
        'name': name,
        'ready': True,
        'assets_dir': str(assets_dir),
        'config': cfg,
    }
    if name == 'linear':
        import torch.nn as nn

        in_features = int(cfg.get('in_features', 784))
        out_features = int(cfg.get('out_features', 10))
        module = nn.Linear(in_features, out_features)
        out['module'] = module
        out['in_features'] = in_features
        out['out_features'] = out_features
    return out
