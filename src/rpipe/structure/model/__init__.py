"""Structure model layer — capability stubs; adapters live here as needed."""

from __future__ import annotations

from typing import Any


def prepare_model(control_model: dict[str, Any], assets_dir) -> dict[str, Any]:
    """Validate / record model layer intent during prepare."""
    return {
        'name': control_model.get('name', 'unknown'),
        'ready': True,
        'assets_dir': str(assets_dir),
    }
