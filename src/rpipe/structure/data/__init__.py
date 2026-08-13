"""Structure data layer — capability stubs; adapters live here as needed."""

from __future__ import annotations

from typing import Any


def prepare_data(control_data: dict[str, Any], assets_dir) -> dict[str, Any]:
    """Validate / record data layer intent during prepare."""
    return {
        'name': control_data.get('name', 'unknown'),
        'ready': True,
        'assets_dir': str(assets_dir),
    }
