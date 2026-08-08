"""model/timm — PyPI ``timm``."""

from __future__ import annotations

from typing import Any

from rpipe.provider import register


@register('model', 'timm')
class TimmModelProvider:
    name = 'timm'
    package = 'timm'

    def available(self) -> bool:
        try:
            import timm  # noqa: F401
            return True
        except ImportError:
            return False

    def list_models(self) -> list[str]:
        return ['resnet18', 'resnet50', 'efficientnet_b0', 'vit_tiny_patch16_224']

    def build(self, model_cfg: dict[str, Any], **kwargs):
        if not self.available():
            raise ImportError('Install package: pip install timm')
        import timm
        name = model_cfg.get('timm_name') or model_cfg.get('model_name')
        num_classes = int(model_cfg.get('target_size') or model_cfg.get('num_classes') or 1000)
        return timm.create_model(name, pretrained=bool(model_cfg.get('pretrained', True)), num_classes=num_classes)
