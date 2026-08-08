"""system/pytorch — PyTorch tensor substrate (train/generate algorithms sit on top)."""

from __future__ import annotations

from rpipe.provider import get_algorithm, register
from rpipe.provider.api import apply_bindings
from rpipe.provider.util import GenerateRunner


@register('system', 'pytorch')
class PyTorchSystemProvider:
    name = 'pytorch'
    tensor_lib = 'pytorch'
    package = 'torch'

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def build_runner(self, runtime):
        apply_bindings(runtime)
        gen = getattr(runtime, 'generate_algorithm', None)
        if gen:
            return GenerateRunner(runtime, get_algorithm('generate', gen))
        train = getattr(runtime, 'train_algorithm', None) or 'native'
        return get_algorithm('train', train).build_trainer(runtime)

    def build_trainer(self, runtime):
        return self.build_runner(runtime)
