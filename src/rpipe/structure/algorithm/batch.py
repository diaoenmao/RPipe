"""Native batch prep: flatten only when the module is a linear-style head."""

from __future__ import annotations

from typing import Any


def prepare_tensors(batch: Any, module: Any, device: Any) -> tuple[Any, Any]:
    images, targets = batch
    images = images.to(device)
    targets = targets.to(device)
    if images.dim() > 2:
        flat = images.reshape(images.size(0), -1)
        in_features = getattr(module, 'in_features', None)
        if in_features is not None and int(flat.size(1)) == int(in_features):
            images = flat
    return images, targets
