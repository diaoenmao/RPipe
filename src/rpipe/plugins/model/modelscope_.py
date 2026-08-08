from __future__ import annotations

from typing import Any

from rpipe.plugins.api import register


@register('model', 'modelscope')
class ModelScopeModelProvider:
    """PyPI package: ``modelscope`` (ModelScope hub + pipelines).

    Refs: https://pypi.org/project/modelscope/ · https://modelscope.cn · https://github.com/modelscope/modelscope
    """

    name = 'modelscope'
    package = 'modelscope'

    def available(self) -> bool:
        try:
            import modelscope  # noqa: F401
            return True
        except ImportError:
            return False

    def list_models(self) -> list[str]:
        return [
            'qwen/Qwen2.5-0.5B-Instruct',
            'LLM-Research/Meta-Llama-3.1-8B-Instruct',
            'AI-ModelScope/bert-base-chinese',
        ]

    def build(self, model_cfg: dict[str, Any], **kwargs):
        if not self.available():
            raise ImportError('Install package: pip install modelscope')
        from modelscope import AutoModel, AutoModelForCausalLM

        name = model_cfg.get('ms_name') or model_cfg.get('model_name') or model_cfg.get('hf_name')
        task = model_cfg.get('ms_task', model_cfg.get('hf_task', 'causal_lm'))
        if task in ('causal_lm', 'lm', 'generation'):
            return AutoModelForCausalLM.from_pretrained(name)
        return AutoModel.from_pretrained(name)
