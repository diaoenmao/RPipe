"""Control object: Structure-layer variable assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Control:
    """Experiment variable assignment over Structure layers."""

    slug: str
    seed: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    algorithm: dict[str, Any] = field(default_factory=dict)
    system: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'slug': self.slug,
            'seed': self.seed,
            'data': dict(self.data),
            'model': dict(self.model),
            'algorithm': dict(self.algorithm),
            'system': dict(self.system),
        }


def control_from_config(cfg: dict[str, Any], slug: str | None = None) -> Control:
    """Build Control from an Artifact Config mapping (prepare reads Config)."""
    resolved = slug or cfg.get('slug') or cfg.get('control_slug') or 'default'
    return Control(
        slug=str(resolved),
        seed=cfg.get('seed'),
        data=dict(cfg.get('data') or {}),
        model=dict(cfg.get('model') or {}),
        algorithm=dict(cfg.get('algorithm') or {}),
        system=dict(cfg.get('system') or {}),
        raw=dict(cfg),
    )
