"""Provider taxonomy and bindings.

Layers
------
- data
- model — includes ``gguf`` for GGUF weight files
- algorithm — **typed** slots (not one flat knob):
    - train:    native | accelerate
    - metric:   native | torchmetrics | evaluate | lm_eval | opencompass
    - generate: llama_cpp | diffusers
- system — tensor substrate only: ``pytorch`` | ``ggml`` (usually auto-bound)

Bindings (examples)
-------------------
- train/accelerate → system=pytorch
- generate/llama_cpp → system=ggml, model=gguf (llama.cpp stack is fixed together)
- generate/diffusers → system=pytorch
- model/gguf → generate=llama_cpp, system=ggml
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Callable, Protocol, runtime_checkable

from rpipe.config.registry import Registry

LAYERS = ('data', 'model', 'algorithm', 'system')
ALGORITHM_TYPES = ('train', 'metric', 'generate')
SYSTEM_BACKENDS = ('pytorch', 'ggml')

PROVIDERS: dict[str, Registry] = {
    'data': Registry('data_provider'),
    'model': Registry('model_provider'),
    'system': Registry('system_provider'),
}
# Typed algorithm registries (names unique within each type)
ALGORITHM_PROVIDERS: dict[str, Registry] = {
    t: Registry(f'algorithm_{t}') for t in ALGORITHM_TYPES
}
# Flat view for entry points / list_providers['algorithm']
PROVIDERS['algorithm'] = Registry('algorithm_provider')


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
    name: str
    algorithm_type: str  # train | metric | generate
    package: str

    def available(self) -> bool: ...


@runtime_checkable
class TrainAlgorithmProvider(AlgorithmProvider, Protocol):
    def build_trainer(self, runtime: Any) -> Any: ...


@runtime_checkable
class MetricAlgorithmProvider(AlgorithmProvider, Protocol):
    mode: str

    def list_metrics(self) -> list[str]: ...

    def make_metric(self, metric_kwargs: dict[str, Any]) -> Any: ...


@runtime_checkable
class GenerateAlgorithmProvider(AlgorithmProvider, Protocol):
    def generate(self, runtime: Any, **kwargs) -> dict[str, Any]: ...


@runtime_checkable
class SystemProvider(Protocol):
    name: str
    tensor_lib: str

    def available(self) -> bool: ...

    def build_runner(self, runtime: Any) -> Any: ...


@dataclass(frozen=True)
class Binding:
    """Declarative compatibility between providers."""

    algorithm_type: str | None = None
    algorithm: str | None = None
    system: str | None = None
    model: str | None = None
    note: str = ''


# Fixed stacks — choosing one side fills the rest when omitted.
BINDINGS: tuple[Binding, ...] = (
    Binding(algorithm_type='train', algorithm='native', system='pytorch', note='PyTorch native train loop'),
    Binding(algorithm_type='train', algorithm='accelerate', system='pytorch', note='HF Accelerate train loop'),
    Binding(algorithm_type='metric', algorithm='native', system='pytorch'),
    Binding(algorithm_type='metric', algorithm='torchmetrics', system='pytorch'),
    Binding(algorithm_type='metric', algorithm='evaluate', system='pytorch'),
    Binding(algorithm_type='metric', algorithm='lm_eval', system='pytorch'),
    Binding(algorithm_type='metric', algorithm='opencompass', system='pytorch'),
    Binding(
        algorithm_type='generate',
        algorithm='llama_cpp',
        system='ggml',
        model='gguf',
        note='llama.cpp generate is GGML+GGUF; algorithm implies llama.cpp stack',
    ),
    Binding(algorithm_type='generate', algorithm='diffusers', system='pytorch', note='Diffusers on PyTorch'),
    Binding(model='gguf', algorithm_type='generate', algorithm='llama_cpp', system='ggml', note='GGUF model binds to llama_cpp'),
)


def register(layer: str, name: str | None = None) -> Callable:
    if layer == 'algorithm':
        raise KeyError('Use register_algorithm(algorithm_type, name) for algorithm providers')
    if layer not in PROVIDERS:
        raise KeyError(f'Unknown layer {layer!r}. Expected one of {LAYERS}')
    return PROVIDERS[layer].register(name)


def register_algorithm(algorithm_type: str, name: str | None = None) -> Callable:
    if algorithm_type not in ALGORITHM_PROVIDERS:
        raise KeyError(f'Unknown algorithm_type {algorithm_type!r}. Expected {ALGORITHM_TYPES}')

    def deco(cls_or_fn):
        reg = ALGORITHM_PROVIDERS[algorithm_type].register(name)
        wrapped = reg(cls_or_fn)
        # also publish flat key "type:name" and bare name when unique
        inst_name = name or getattr(cls_or_fn, 'name', None) or getattr(cls_or_fn, '__name__', None)

        def _factory(*a, _cls=cls_or_fn, **k):
            return _cls(*a, **k) if isinstance(_cls, type) else _cls

        flat = f'{algorithm_type}:{inst_name}'
        PROVIDERS['algorithm']._items[flat] = _factory
        # Do not publish bare names — train/metric both use "native".
        return wrapped

    return deco


def get_provider(layer: str, name: str, *, require_available: bool = True) -> Any:
    load_builtin_providers()
    discover_entry_points()
    if layer == 'algorithm' and name not in PROVIDERS['algorithm']._items:
        if ':' in name:
            t, n = name.split(':', 1)
            return get_algorithm(t, n, require_available=require_available)
        for t in ALGORITHM_TYPES:
            if name in ALGORITHM_PROVIDERS[t]._items:
                return get_algorithm(t, name, require_available=require_available)
    provider = PROVIDERS[layer].build(name)
    if require_available and hasattr(provider, 'available') and not provider.available():
        raise ImportError(
            f'Provider {layer}/{name} is registered but dependencies are missing. '
            f'Install the optional extra or pick another provider.'
        )
    return provider


def get_algorithm(algorithm_type: str, name: str, *, require_available: bool = True) -> Any:
    load_builtin_providers()
    discover_entry_points()
    if algorithm_type not in ALGORITHM_PROVIDERS:
        raise KeyError(algorithm_type)
    provider = ALGORITHM_PROVIDERS[algorithm_type].build(name)
    if require_available and hasattr(provider, 'available') and not provider.available():
        raise ImportError(
            f'Algorithm {algorithm_type}/{name} is registered but dependencies are missing.'
        )
    return provider


def list_providers(layer: str | None = None) -> dict[str, Any]:
    load_builtin_providers()
    discover_entry_points()
    algo = {t: ALGORITHM_PROVIDERS[t].keys() for t in ALGORITHM_TYPES}
    full = {
        'data': PROVIDERS['data'].keys(),
        'model': PROVIDERS['model'].keys(),
        'algorithm': algo,
        'system': PROVIDERS['system'].keys(),
    }
    if layer is not None:
        return {layer: full[layer]}
    return full


def list_algorithms(*, algorithm_type: str | None = None) -> dict[str, list[str]] | list[str]:
    load_builtin_providers()
    discover_entry_points()
    if algorithm_type is not None:
        return ALGORITHM_PROVIDERS[algorithm_type].keys()
    return {t: ALGORITHM_PROVIDERS[t].keys() for t in ALGORITHM_TYPES}


def find_binding(
    *,
    algorithm_type: str | None = None,
    algorithm: str | None = None,
    model: str | None = None,
    system: str | None = None,
) -> Binding | None:
    for b in BINDINGS:
        if algorithm_type and b.algorithm_type and b.algorithm_type != algorithm_type:
            continue
        if algorithm and b.algorithm and b.algorithm != algorithm:
            continue
        if model and b.model and b.model != model:
            # allow matching bindings that specify this model
            if b.model != model:
                continue
        if algorithm and b.algorithm == algorithm and (not algorithm_type or b.algorithm_type == algorithm_type):
            return b
        if model and b.model == model and not algorithm:
            return b
    # second pass: model-driven
    if model:
        for b in BINDINGS:
            if b.model == model:
                return b
    if algorithm:
        for b in BINDINGS:
            if b.algorithm == algorithm and (algorithm_type is None or b.algorithm_type == algorithm_type):
                return b
    return None


def apply_bindings(runtime: Any) -> Any:
    """Fill system/model/generate from fixed stacks; raise on hard conflicts."""
    resolve_legacy_runtime_fields(runtime)

    gen = getattr(runtime, 'generate_algorithm', None)
    train = getattr(runtime, 'train_algorithm', None) or 'native'
    model = getattr(runtime, 'model_provider', None) or 'native'
    system = getattr(runtime, 'system_provider', None)

    # model=gguf implies llama_cpp generate + ggml
    if model == 'gguf':
        b = find_binding(model='gguf')
        if b:
            if not gen:
                runtime.generate_algorithm = b.algorithm
                gen = b.algorithm
            elif b.algorithm and gen != b.algorithm:
                raise ValueError(f'model_provider=gguf requires generate_algorithm={b.algorithm!r}, got {gen!r}')
            runtime.system_provider = b.system
            system = b.system

    # generate algorithm binds system (+ model suggestion)
    if gen:
        b = find_binding(algorithm_type='generate', algorithm=gen)
        if b:
            if system and b.system and system != b.system:
                raise ValueError(
                    f'generate_algorithm={gen!r} requires system_provider={b.system!r}, got {system!r}'
                )
            runtime.system_provider = b.system
            if b.model and model in ('native', None, '') and getattr(runtime, 'model_provider', None) in ('native', None, ''):
                runtime.model_provider = b.model
            elif b.model and model not in (b.model, 'native') and model != b.model:
                # allow explicit transformers etc. only if not gguf-bound algo — llama_cpp wants gguf
                if gen == 'llama_cpp' and model != 'gguf':
                    # force gguf for llama_cpp unless user already set gguf
                    runtime.model_provider = 'gguf'
        runtime.train_algorithm = None  # generate run
        return runtime

    # train algorithm binds system
    b = find_binding(algorithm_type='train', algorithm=train)
    if b and b.system:
        if system and system != b.system:
            raise ValueError(
                f'train_algorithm={train!r} requires system_provider={b.system!r}, got {system!r}'
            )
        runtime.system_provider = b.system
    if not getattr(runtime, 'system_provider', None):
        runtime.system_provider = 'pytorch'
    if not getattr(runtime, 'train_algorithm', None):
        runtime.train_algorithm = 'native'
    if not getattr(runtime, 'metric_algorithm', None):
        runtime.metric_algorithm = getattr(runtime, 'algorithm_provider', None) or 'native'
    return runtime


def resolve_legacy_runtime_fields(runtime: Any) -> Any:
    """Map deprecated knobs onto train/metric/generate + system."""
    # metric_algorithm from metric_provider / algorithm_provider
    if not getattr(runtime, 'metric_algorithm', None):
        legacy = getattr(runtime, 'metric_provider', None) or getattr(runtime, 'algorithm_provider', None)
        if legacy and legacy not in ('accelerate', 'llama_cpp', 'diffusers', 'train_native'):
            # if algorithm_provider was a metric name
            if legacy in ('native', 'torchmetrics', 'evaluate', 'lm_eval', 'opencompass'):
                runtime.metric_algorithm = legacy

    tb = getattr(runtime, 'trainer_backend', None)
    accel = getattr(runtime, 'pytorch_accelerator', None)
    if not getattr(runtime, 'train_algorithm', None):
        if accel == 'accelerate' or tb == 'accelerate':
            runtime.train_algorithm = 'accelerate'
        elif tb in ('native', None, '', 'pytorch') or accel == 'native':
            if tb == 'llama_cpp':
                pass
            elif tb != 'diffusers':
                runtime.train_algorithm = getattr(runtime, 'train_algorithm', None) or 'native'

    if not getattr(runtime, 'generate_algorithm', None):
        if tb == 'llama_cpp' or getattr(runtime, 'algorithm_provider', None) == 'llama_cpp':
            runtime.generate_algorithm = 'llama_cpp'
        elif tb == 'diffusers' or getattr(runtime, 'algorithm_provider', None) == 'diffusers':
            runtime.generate_algorithm = 'diffusers'

    if not getattr(runtime, 'system_provider', None):
        if tb == 'llama_cpp' or getattr(runtime, 'generate_algorithm', None) == 'llama_cpp':
            runtime.system_provider = 'ggml'
        elif tb in ('native', 'accelerate', 'diffusers', 'pytorch', None, ''):
            runtime.system_provider = 'pytorch'
        elif tb == 'ggml':
            runtime.system_provider = 'ggml'

    # single algorithm_provider legacy: if set to accelerate → train
    ap = getattr(runtime, 'algorithm_provider', None)
    if ap == 'accelerate' and not getattr(runtime, 'train_algorithm', None):
        runtime.train_algorithm = 'accelerate'
    return runtime


def discover_entry_points() -> list[str]:
    if getattr(discover_entry_points, '_done', False):
        return getattr(discover_entry_points, '_loaded', [])
    loaded: list[str] = []
    try:
        eps = entry_points()
    except Exception:
        discover_entry_points._done = True  # type: ignore[attr-defined]
        discover_entry_points._loaded = loaded  # type: ignore[attr-defined]
        return loaded
    for layer in ('data', 'model', 'system'):
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
    # algorithm entry points: rpipe.algorithm.train / .metric / .generate
    for t in ALGORITHM_TYPES:
        group = f'rpipe.algorithm.{t}'
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

                ALGORITHM_PROVIDERS[t]._items[name] = _factory
                PROVIDERS['algorithm']._items[f'{t}:{name}'] = _factory
                loaded.append(f'algorithm.{t}:{name}')
            except Exception:
                continue
    discover_entry_points._done = True  # type: ignore[attr-defined]
    discover_entry_points._loaded = loaded  # type: ignore[attr-defined]
    return loaded


def load_builtin_providers() -> None:
    if getattr(load_builtin_providers, '_done', False):
        return
    # Import layer/provider packages: data/native, model/timm, algorithm/train/..., system/pytorch, ...
    import rpipe.data.native  # noqa: F401
    import rpipe.data.datasets  # noqa: F401
    import rpipe.model.native  # noqa: F401
    import rpipe.model.timm  # noqa: F401
    import rpipe.model.transformers  # noqa: F401
    import rpipe.model.modelscope  # noqa: F401
    import rpipe.model.peft  # noqa: F401
    import rpipe.model.ollama  # noqa: F401
    import rpipe.model.gguf  # noqa: F401
    import rpipe.algorithm.train.native  # noqa: F401
    import rpipe.algorithm.train.accelerate  # noqa: F401
    import rpipe.algorithm.metric.native  # noqa: F401
    import rpipe.algorithm.metric.torchmetrics  # noqa: F401
    import rpipe.algorithm.metric.evaluate  # noqa: F401
    import rpipe.algorithm.metric.lm_eval  # noqa: F401
    import rpipe.algorithm.metric.opencompass  # noqa: F401
    import rpipe.algorithm.generate.llama_cpp  # noqa: F401
    import rpipe.algorithm.generate.diffusers  # noqa: F401
    import rpipe.system.pytorch  # noqa: F401
    import rpipe.system.ggml  # noqa: F401
    load_builtin_providers._done = True  # type: ignore[attr-defined]
