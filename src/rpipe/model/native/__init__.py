"""model/native — built-in CV models (linear/cnn/resnet…)."""

from __future__ import annotations

from typing import Any

from rpipe.model.model import make_model as _make_model
from rpipe.provider import register


@register('model', 'native')
class NativeModelProvider:
    name = 'native'
    package = 'rpipe'

    def available(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        from rpipe.config.registry import MODEL_REGISTRY
        return MODEL_REGISTRY.keys()

    def build(self, model_cfg: dict[str, Any], **kwargs):
        return _make_model(model_cfg)
