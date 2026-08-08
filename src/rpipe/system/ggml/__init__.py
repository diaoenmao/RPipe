"""system/ggml — GGML tensor substrate (llama.cpp / GGUF generate stack)."""

from __future__ import annotations

from rpipe.provider import get_algorithm, register
from rpipe.provider.api import apply_bindings
from rpipe.provider.util import GenerateRunner


@register('system', 'ggml')
class GgmlSystemProvider:
    name = 'ggml'
    tensor_lib = 'ggml'
    package = 'llama-cpp-python'

    def available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            return False

    def build_runner(self, runtime):
        apply_bindings(runtime)
        gen = getattr(runtime, 'generate_algorithm', None) or 'llama_cpp'
        algo = get_algorithm('generate', gen)
        if getattr(algo, 'algorithm_type', None) != 'generate':
            raise ValueError('system/ggml expects a generate algorithm (llama_cpp)')
        return GenerateRunner(runtime, algo)

    def build_trainer(self, runtime):
        return self.build_runner(runtime)
