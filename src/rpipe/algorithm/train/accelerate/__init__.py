"""algorithm/train/accelerate — PyPI ``accelerate`` training loop (on system/pytorch)."""

from __future__ import annotations

from rpipe.provider import register_algorithm


@register_algorithm('train', 'accelerate')
class AccelerateTrainAlgorithm:
    name = 'accelerate'
    package = 'accelerate'
    algorithm_type = 'train'
    requires_system = 'pytorch'

    def available(self) -> bool:
        try:
            import accelerate  # noqa: F401
            return True
        except ImportError:
            return False

    def build_trainer(self, runtime):
        from rpipe.system.backend.accelerate_ import AccelerateTrainer
        return AccelerateTrainer(runtime)
