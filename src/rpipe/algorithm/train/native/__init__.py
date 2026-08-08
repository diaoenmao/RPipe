"""algorithm/train/native — PyTorch native training loop."""

from __future__ import annotations

from rpipe.provider import register_algorithm


@register_algorithm('train', 'native')
class NativeTrainAlgorithm:
    name = 'native'
    package = 'rpipe'
    algorithm_type = 'train'
    requires_system = 'pytorch'

    def available(self) -> bool:
        return True

    def build_trainer(self, runtime):
        from rpipe.system.backend.native import NativeTrainer
        return NativeTrainer(runtime)
