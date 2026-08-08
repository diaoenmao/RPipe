"""Thin provider registry + bindings (not a dumping ground for third-party code).

Implementations live next to their layer::

    data/native, data/datasets
    model/native, model/timm, model/gguf, ...
    algorithm/train/native, algorithm/metric/lm_eval, algorithm/generate/llama_cpp
    system/pytorch, system/ggml
"""

from rpipe.provider.api import (
    ALGORITHM_TYPES,
    BINDINGS,
    LAYERS,
    SYSTEM_BACKENDS,
    apply_bindings,
    find_binding,
    get_algorithm,
    get_provider,
    list_algorithms,
    list_providers,
    load_builtin_providers,
    register,
    register_algorithm,
    resolve_legacy_runtime_fields,
)

__all__ = [
    'ALGORITHM_TYPES',
    'BINDINGS',
    'LAYERS',
    'SYSTEM_BACKENDS',
    'apply_bindings',
    'find_binding',
    'get_algorithm',
    'get_provider',
    'list_algorithms',
    'list_providers',
    'load_builtin_providers',
    'register',
    'register_algorithm',
    'resolve_legacy_runtime_fields',
]
