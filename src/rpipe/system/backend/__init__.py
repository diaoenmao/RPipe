"""Trainer / Evaluator factory with pluggable system backends."""

from __future__ import annotations

from rpipe.config import RuntimeConfig
from rpipe.system.backend.native import Evaluator, NativeTrainer


def build_trainer(runtime: RuntimeConfig):
    """Public API: ``Trainer(runtime)``-compatible factory."""
    backend = getattr(runtime, 'trainer_backend', None) or getattr(runtime, 'backend', 'native')
    if backend in ('native', '', None):
        return NativeTrainer(runtime)
    if backend == 'accelerate':
        from rpipe.system.backend.accelerate_ import AccelerateTrainer
        return AccelerateTrainer(runtime)
    # third-party system providers (llama_cpp, diffusers, …)
    from rpipe.plugins.api import get_provider
    from rpipe.plugins import load_builtin_providers
    load_builtin_providers()
    provider = get_provider('system', backend)
    return provider.build_trainer(runtime)


# Backward-compatible names
Trainer = build_trainer


class TrainerProxy:
    """Allow ``Trainer(runtime).run()`` while dispatching to the selected backend."""

    def __new__(cls, runtime: RuntimeConfig):
        return build_trainer(runtime)


Trainer = TrainerProxy  # type: ignore

__all__ = ['Trainer', 'Evaluator', 'NativeTrainer', 'build_trainer']
