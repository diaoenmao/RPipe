from __future__ import annotations

from typing import Any

from rpipe.plugins.api import register


@register('model', 'timm')
class TimmModelProvider:
    """PyPI package: ``timm`` — PyTorch Image Models.

    Refs: https://pypi.org/project/timm/ · https://huggingface.co/docs/timm
    """

    name = 'timm'
    package = 'timm'

    def available(self) -> bool:
        try:
            import timm  # noqa: F401
            return True
        except ImportError:
            return False

    def list_models(self) -> list[str]:
        if not self.available():
            return []
        import timm
        # Keep list short for UX; full list via timm.list_models()
        return timm.list_models(pretrained=True)[:50]

    def build(self, model_cfg: dict[str, Any], **kwargs):
        if not self.available():
            raise ImportError('Install timm: pip install timm')
        import timm
        from rpipe.model.base import base

        model_name = model_cfg.get('timm_name') or model_cfg.get('model_name')
        num_classes = int(model_cfg.get('target_size') or model_cfg.get('num_classes') or 1000)
        pretrained = bool(model_cfg.get('pretrained', False))
        core = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        # Wrap with existing Base when stats/data_name present (CV aug path)
        if model_cfg.get('data_name') and model_cfg.get('stats') is not None:
            return base(core, model_cfg['data_name'], model_cfg['stats'])
        return core
