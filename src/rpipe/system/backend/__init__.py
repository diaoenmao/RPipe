"""Trainer factory — system substrate + train/generate algorithms."""

from __future__ import annotations

from rpipe.config import RuntimeConfig
from rpipe.system.backend.native import Evaluator, NativeTrainer


def build_trainer(runtime: RuntimeConfig):
    from rpipe.provider import apply_bindings, get_provider, load_builtin_providers

    load_builtin_providers()
    apply_bindings(runtime)
    system_name = getattr(runtime, 'system_provider', None) or 'pytorch'
    return get_provider('system', system_name).build_runner(runtime)


class TrainerProxy:
    def __new__(cls, runtime: RuntimeConfig):
        return build_trainer(runtime)


Trainer = TrainerProxy  # type: ignore

__all__ = ['Trainer', 'Evaluator', 'NativeTrainer', 'build_trainer']
