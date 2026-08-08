from __future__ import annotations

from typing import Any

from rpipe.plugins.api import register


@register('model', 'transformers')
class TransformersModelProvider:
    """PyPI package: ``transformers`` — HF Auto* LLM / encoder checkpoints.

    Refs: https://pypi.org/project/transformers/ · https://huggingface.co/docs/transformers
    """

    name = 'transformers'
    package = 'transformers'

    def available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def list_models(self) -> list[str]:
        return [
            'gpt2',
            'bert-base-uncased',
            'meta-llama/Llama-3.2-1B',
            'Qwen/Qwen2.5-0.5B',
        ]

    def build(self, model_cfg: dict[str, Any], **kwargs):
        if not self.available():
            raise ImportError('Install transformers: pip install transformers')
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification

        name = model_cfg.get('hf_name') or model_cfg.get('model_name')
        task = model_cfg.get('hf_task', 'causal_lm')
        if task in ('causal_lm', 'lm', 'generation'):
            return AutoModelForCausalLM.from_pretrained(name)
        if task in ('sequence_classification', 'cls'):
            num_labels = int(model_cfg.get('target_size') or model_cfg.get('num_labels') or 2)
            return AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
        return AutoModel.from_pretrained(name)
