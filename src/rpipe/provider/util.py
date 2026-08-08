"""Shared helpers for providers."""

from __future__ import annotations


def cfg_get(runtime, *keys: str, default=None):
    """Read knobs from RuntimeConfig / ModelRuntime / arch extras."""
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


class GenerateRunner:
    def __init__(self, runtime, algorithm):
        self.runtime = runtime
        self.algorithm = algorithm

    def run(self):
        return self.algorithm.generate(self.runtime)
