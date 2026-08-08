"""Pluggable third-party providers for the four rpipe layers.

Layers
------
- data
- model
- algorithm (metrics + eval harnesses)
- system (trainer backends)

Third parties register via ``@PROVIDERS[layer].register("name")`` or setuptools
entry points group ``rpipe.<layer>`` (loaded by ``discover_entry_points``).
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Callable, Protocol, runtime_checkable

from rpipe.config.registry import Registry

LAYERS = ('data', 'model', 'algorithm', 'system')

PROVIDERS: dict[str, Registry] = {layer: Registry(f'{layer}_provider') for layer in LAYERS}


@runtime_checkable
class DataProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    def list_datasets(self) -> list[str]: ...

    def build(self, data_name: str, *, process: bool = False, verbose: bool = True, **kwargs) -> dict: ...


@runtime_checkable
class ModelProvider(Protocol):
    name: str

    def available(self) -> bool: ...

    def list_models(self) -> list[str]: ...

    def build(self, model_cfg: dict[str, Any], **kwargs) -> Any: ...


@runtime_checkable
class AlgorithmProvider(Protocol):
    """Metric provider under the algorithm layer (online or benchmark)."""

    name: str
    mode: str  # 'online' | 'benchmark'
    kind: str  # always 'metric' (legacy field)

    def available(self) -> bool: ...

    def list_metrics(self) -> list[str]: ...


@runtime_checkable
class SystemProvider(Protocol):
    """Trainer / distributed backend provider."""

    name: str

    def available(self) -> bool: ...

    def build_trainer(self, runtime: Any) -> Any: ...


def register(layer: str, name: str | None = None) -> Callable:
    if layer not in PROVIDERS:
        raise KeyError(f'Unknown layer {layer!r}. Expected one of {LAYERS}')
    return PROVIDERS[layer].register(name)


def get_provider(layer: str, name: str) -> Any:
    load_builtin_providers()
    discover_entry_points()
    provider = PROVIDERS[layer].build(name)
    if hasattr(provider, 'available') and not provider.available():
        raise ImportError(
            f'Provider {layer}/{name} is registered but dependencies are missing. '
            f'Install the optional extra or pick another provider.'
        )
    return provider


def list_providers(layer: str | None = None) -> dict[str, list[str]]:
    load_builtin_providers()
    discover_entry_points()
    if layer is not None:
        return {layer: PROVIDERS[layer].keys()}
    return {k: v.keys() for k, v in PROVIDERS.items()}


def discover_entry_points() -> list[str]:
    """Load third-party providers from setuptools entry points ``rpipe.<layer>``."""
    if getattr(discover_entry_points, '_done', False):
        return getattr(discover_entry_points, '_loaded', [])
    loaded: list[str] = []
    try:
        eps = entry_points()
    except Exception:
        discover_entry_points._done = True  # type: ignore[attr-defined]
        discover_entry_points._loaded = loaded  # type: ignore[attr-defined]
        return loaded
    for layer in LAYERS:
        group = f'rpipe.{layer}'
        try:
            selected = eps.select(group=group) if hasattr(eps, 'select') else eps.get(group, [])
        except Exception:
            continue
        for ep in selected:
            try:
                obj = ep.load()
                instance = obj() if isinstance(obj, type) else obj
                name = getattr(instance, 'name', ep.name)

                def _factory(inst=instance, *a, **k):
                    return inst

                PROVIDERS[layer]._items[name] = _factory
                loaded.append(f'{layer}:{name}')
            except Exception:
                continue
    discover_entry_points._done = True  # type: ignore[attr-defined]
    discover_entry_points._loaded = loaded  # type: ignore[attr-defined]
    return loaded


def load_builtin_providers() -> None:
    """Import built-in provider modules so decorators register."""
    if getattr(load_builtin_providers, '_done', False):
        return
    from rpipe.plugins import data as _data  # noqa: F401
    from rpipe.plugins import model as _model  # noqa: F401
    from rpipe.plugins import algorithm as _algorithm  # noqa: F401
    from rpipe.plugins import system as _system  # noqa: F401
    load_builtin_providers._done = True  # type: ignore[attr-defined]
