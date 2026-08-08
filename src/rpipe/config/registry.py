from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, key: str | None = None):
        def decorator(fn: Callable[..., Any]):
            name = key or fn.__name__
            self._items[name] = fn  # allow reload / idempotent register
            return fn

        return decorator

    def get(self, key: str) -> Callable[..., Any]:
        if key not in self._items:
            raise KeyError(f'Unknown {self.name}: {key}. Available: {sorted(self._items)}')
        return self._items[key]

    def build(self, key: str, *args, **kwargs):
        return self.get(key)(*args, **kwargs)

    def keys(self):
        return sorted(self._items.keys())


DATASET_REGISTRY = Registry('dataset')
MODEL_REGISTRY = Registry('model')
