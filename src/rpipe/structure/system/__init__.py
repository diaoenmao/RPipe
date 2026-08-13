"""Structure system layer — device / IO stubs."""

from __future__ import annotations

from typing import Any


def prepare_system(control_system: dict[str, Any], assets_dir) -> dict[str, Any]:
    device = control_system.get('device', 'cpu')
    return {'device': device, 'assets_dir': str(assets_dir), 'ready': True}
