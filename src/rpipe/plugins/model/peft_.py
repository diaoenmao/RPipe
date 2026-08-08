from __future__ import annotations

from typing import Any

from rpipe.plugins.api import register


@register('model', 'peft')
class PeftModelProvider:
    """PyPI package: ``peft`` — LoRA/QLoRA wrappers over ``transformers`` bases.

    Refs: https://pypi.org/project/peft/ · https://huggingface.co/docs/peft
    """

    name = 'peft'
    package = 'peft'

    def available(self) -> bool:
        try:
            import peft  # noqa: F401
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def list_models(self) -> list[str]:
        return ['lora:<hf_or_ms_name>', 'qlora:<hf_or_ms_name>']

    def build(self, model_cfg: dict[str, Any], **kwargs):
        if not self.available():
            raise ImportError('Install packages: pip install peft transformers')
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        name = model_cfg.get('hf_name') or model_cfg.get('model_name')
        if isinstance(name, str) and name.startswith(('lora:', 'qlora:')):
            name = name.split(':', 1)[1]
        base = AutoModelForCausalLM.from_pretrained(name)
        lora = LoraConfig(
            r=int(model_cfg.get('lora_r', 8)),
            lora_alpha=int(model_cfg.get('lora_alpha', 16)),
            lora_dropout=float(model_cfg.get('lora_dropout', 0.05)),
            bias='none',
            task_type='CAUSAL_LM',
        )
        return get_peft_model(base, lora)
