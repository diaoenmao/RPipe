"""model/gguf — GGUF weight path handle (binds to algorithm/generate/llama_cpp + system/ggml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rpipe.provider import register
from rpipe.provider.util import cfg_get


@register('model', 'gguf')
class GgufModelProvider:
    """Local ``.gguf`` file as model identity for the llama.cpp / GGML stack."""

    name = 'gguf'
    package = 'gguf'
    binds_generate = 'llama_cpp'
    binds_system = 'ggml'

    def available(self) -> bool:
        return True  # path-based; llama-cpp-python checked at generate time

    def list_models(self) -> list[str]:
        return ['*.gguf', 'path/to/model.Q4_K_M.gguf']

    def build(self, model_cfg: dict[str, Any], **kwargs):
        path = model_cfg.get('gguf_path') or model_cfg.get('model_name') or model_cfg.get('hf_name')
        if not path:
            raise ValueError('model/gguf requires gguf_path or model_name pointing to a .gguf file')
        p = Path(str(path))
        return {'format': 'gguf', 'path': str(p), 'exists': p.exists()}
