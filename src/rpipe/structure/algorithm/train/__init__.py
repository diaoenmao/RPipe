"""Train semantics stub."""

from __future__ import annotations

from typing import Any


def run(control_algorithm: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    steps = int(control_algorithm.get('num_steps', 1))
    return {'semantic': 'train', 'steps': steps, 'loss': 0.0}
