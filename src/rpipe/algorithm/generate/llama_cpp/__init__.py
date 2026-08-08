"""algorithm/generate/llama_cpp — GGUF generate on GGML (binds model/gguf + system/ggml)."""

from __future__ import annotations

from typing import Any

from rpipe.provider import register_algorithm
from rpipe.provider.util import cfg_get


@register_algorithm('generate', 'llama_cpp')
class LlamaCppGenerateAlgorithm:
    name = 'llama_cpp'
    package = 'llama-cpp-python'
    algorithm_type = 'generate'
    requires_system = 'ggml'
    requires_model = 'gguf'

    def available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(self, runtime: Any, **kwargs) -> dict[str, Any]:
        import llama_cpp
        model_path = cfg_get(runtime, 'gguf_path', 'model_path') or getattr(runtime, 'model_name', None)
        model_path = cfg_get(runtime, 'gguf_path') or model_path
        if isinstance(model_path, dict):
            model_path = model_path.get('path')
        if not model_path:
            raise ValueError('llama_cpp requires model/gguf path (model.arch.gguf_path)')
        n_ctx = int(kwargs.get('n_ctx', cfg_get(runtime, 'n_ctx', default=2048)))
        llm = llama_cpp.Llama(model_path=str(model_path), n_ctx=n_ctx, verbose=False)
        prompt = str(kwargs.get('prompt', cfg_get(runtime, 'prompt', default='Hello')))
        max_tokens = int(kwargs.get('max_tokens', cfg_get(runtime, 'max_tokens', default=64)))
        out = llm(prompt, max_tokens=max_tokens)
        text = out['choices'][0]['text'] if isinstance(out, dict) else str(out)
        return {
            'schema': 'rpipe.inference.v1',
            'algorithm': 'llama_cpp',
            'algorithm_type': 'generate',
            'system': 'ggml',
            'model': 'gguf',
            'prompt': prompt,
            'text': text,
        }
