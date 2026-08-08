from __future__ import annotations

from typing import Any

from rpipe.plugins.api import register


def _cfg_get(runtime, *keys: str, default=None):
    """Read inference knobs from RuntimeConfig / ModelRuntime / arch extras."""
    for key in keys:
        if hasattr(runtime, key):
            val = getattr(runtime, key)
            if val is not None:
                return val
    m = getattr(runtime, 'model', None)
    if m is None:
        return default
    if isinstance(m, dict):
        for key in keys:
            if key in m and m[key] is not None:
                return m[key]
        return default
    for key in keys:
        if hasattr(m, key):
            val = getattr(m, key)
            if val is not None:
                return val
    arch = getattr(m, 'arch', None) or {}
    if isinstance(arch, dict):
        for key in keys:
            if key in arch and arch[key] is not None:
                return arch[key]
    return default


@register('system', 'llama_cpp')
class LlamaCppSystemProvider:
    """PyPI package: ``llama-cpp-python`` — local GGUF LLM inference backend.

    Import name: ``llama_cpp``. Use for quantized GGUF inference (not training).

    Refs: https://pypi.org/project/llama-cpp-python/ · https://github.com/abetlen/llama-cpp-python
          GGUF / llama.cpp: https://github.com/ggml-org/llama.cpp
    """

    name = 'llama_cpp'
    package = 'llama-cpp-python'

    def available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def build_trainer(self, runtime):
        return LlamaCppInferenceRunner(runtime)


class LlamaCppInferenceRunner:
    """Inference-only runner: ``run()`` loads a GGUF and generates once."""

    def __init__(self, runtime):
        self.runtime = runtime

    def run(self):
        import llama_cpp

        r = self.runtime
        model_path = _cfg_get(r, 'gguf_path', 'model_path') or getattr(r, 'model_name', None)
        if not model_path or not str(model_path).endswith('.gguf') and _cfg_get(r, 'gguf_path') is None:
            # allow model_name to be a .gguf path; otherwise require arch.gguf_path
            model_path = _cfg_get(r, 'gguf_path') or model_path
        if not model_path:
            raise ValueError(
                'llama_cpp requires a GGUF path via model.arch.gguf_path or model_name ending in .gguf'
            )
        n_ctx = int(_cfg_get(r, 'n_ctx', default=2048))
        llm = llama_cpp.Llama(model_path=str(model_path), n_ctx=n_ctx, verbose=False)
        prompt = str(_cfg_get(r, 'prompt', default='Hello'))
        max_tokens = int(_cfg_get(r, 'max_tokens', default=64))
        out = llm(prompt, max_tokens=max_tokens)
        text = out['choices'][0]['text'] if isinstance(out, dict) else str(out)
        return {
            'schema': 'rpipe.inference.v1',
            'provider': 'llama_cpp',
            'package': 'llama-cpp-python',
            'prompt': prompt,
            'text': text,
        }
