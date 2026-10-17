"""Dataset registry, factory and vision builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from rpipe.structure.data.config import DataConfig
from rpipe.structure.data.runtime import Data, _loader_options
from rpipe.structure.api.artifact_api import cache_lock, reference_tree
from rpipe.structure.api.system_api import apply_model_origin, bind_endpoint, discard_bad_archive, endpoint_for


class DataRegistry:
    _items: dict[tuple[str, str], Callable[..., Data]] = {}

    @classmethod
    def register(cls, name: str, source: str, builder: Callable[..., Data]) -> None:
        cls._items[(name, source)] = builder

    @classmethod
    def get(cls, name: str, source: str) -> Callable[..., Data] | None:
        return cls._items.get((name, source))

    @classmethod
    def list(cls) -> list[tuple[str, str]]:
        return sorted(cls._items)


class DataFactory:
    @staticmethod
    def build(
        data_config: DataConfig,
        assets_dir: Path | str,
        seed: int | None = None,
        origin: str | None = None,
        *,
        profile_path: Path | str | None = None,
    ) -> Data:
        name = data_config.name
        source = data_config.source if data_config.source is not None else 'torch'
        builder = DataRegistry.get(name, source)
        if builder is None:
            raise ValueError(f'unknown data name/source: {name!r}/{source!r}')
        if origin not in (None, ''):
            apply_model_origin(origin)
        with cache_lock(assets_dir):
            if builder is _build_torch_vision:
                return _build_torch_vision(data_config, Path(assets_dir), seed=seed, origin=origin,
                                           profile_path=profile_path)
            return builder(data_config, Path(assets_dir), seed=seed)


def _build_stub(data_config: DataConfig, assets_dir: Path, seed: int | None = None) -> Data:
    name = data_config.name or 'unknown'
    source = data_config.source or 'stub'
    cfg = dict(data_config.config)
    return Data(
        name=name,
        source=source,
        loaders={},
        meta={
            'ready': True,
            'assets_dir': str(assets_dir),
            'config': cfg,
            'stub': True,
            'seed': seed,
        },
    )


_VISION_TORCH: dict[str, dict[str, Any]] = {
    'MNIST': {
        'root': 'mnist',
        'ctor': 'MNIST',
        'train_kw': {'train': True},
        'test_kw': {'train': False},
        'data_size': (1, 28, 28),
        'target_size': 10,
        'mean': (0.1307,),
        'std': (0.3081,),
        'train_aug': None,
    },
    'FashionMNIST': {
        'root': 'fashionmnist',
        'ctor': 'FashionMNIST',
        'train_kw': {'train': True},
        'test_kw': {'train': False},
        'data_size': (1, 28, 28),
        'target_size': 10,
        'mean': (0.2860,),
        'std': (0.3530,),
        'train_aug': None,
    },
    'CIFAR10': {
        'root': 'cifar10',
        'ctor': 'CIFAR10',
        'train_kw': {'train': True},
        'test_kw': {'train': False},
        'data_size': (3, 32, 32),
        'target_size': 10,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010),
        'train_aug': 'cifar',
    },
    'CIFAR100': {
        'root': 'cifar100',
        'ctor': 'CIFAR100',
        'train_kw': {'train': True},
        'test_kw': {'train': False},
        'data_size': (3, 32, 32),
        'target_size': 100,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761),
        'train_aug': 'cifar',
    },
    'SVHN': {
        'root': 'svhn',
        'ctor': 'SVHN',
        'train_kw': {'split': 'train'},
        'test_kw': {'split': 'test'},
        'data_size': (3, 32, 32),
        'target_size': 10,
        'mean': (0.4377, 0.4438, 0.4728),
        'std': (0.1980, 0.2010, 0.1970),
        'train_aug': 'svhn',
    },
}


def _with_recorded_stats(spec: dict[str, Any], path: Path) -> dict[str, Any]:
    """Prefer mean / std from ``stats.yaml`` when a profile has been written."""
    from rpipe.structure.data.profile import load_stats

    recorded = load_stats(path)
    if recorded is None:
        return spec
    copied = dict(spec)
    copied['mean'] = tuple(float(x) for x in recorded['mean'])
    copied['std'] = tuple(float(x) for x in recorded['std'])
    return copied


def vision_root(name: str) -> str | None:
    spec = _VISION_TORCH.get(name)
    if spec is None:
        return None
    return str(spec['root'])


def _build_torch_vision(
    data_config: DataConfig,
    assets_dir: Path,
    seed: int | None = None,
    origin: str | None = None,
    profile_path: Path | str | None = None,
) -> Data:
    import torch
    from torch.utils.data import DataLoader, Subset

    from rpipe.structure.api.system_api import make_generator, worker_init_fn
    from rpipe.structure.api.system_api import import_torchvision

    import_torchvision()
    from torchvision import datasets, transforms

    name = data_config.name or 'unknown'
    spec = _VISION_TORCH[name]
    cfg = dict(data_config.config)
    root = assets_dir / str(spec['root'])
    root.mkdir(parents=True, exist_ok=True)
    canonical_raw = assets_dir / name / 'raw'
    native_raw = root / name / 'raw' if name in ('MNIST', 'FashionMNIST') else root
    if canonical_raw.is_dir():
        reference_tree(canonical_raw, native_raw)
    ctor = getattr(datasets, str(spec['ctor']))
    augment = bool(cfg.get('augment', spec.get('train_aug') is not None))
    stats = cfg.get('stats_path', profile_path if profile_path is not None else root / 'stats.yaml')
    spec = _with_recorded_stats(spec, Path(stats))
    # Pixels stay in 0–1; model-side kornia owns augmentation.
    train_tf = transforms.ToTensor()
    test_tf = transforms.ToTensor()
    endpoint = endpoint_for(name, origin)
    discard_bad_archive(root, endpoint)
    with bind_endpoint(ctor, endpoint):
        train_full = ctor(root=str(root), download=True, transform=train_tf, **dict(spec['train_kw']))
        test_ds = ctor(root=str(root), download=True, transform=test_tf, **dict(spec['test_kw']))
    reference_tree(native_raw, canonical_raw)
    train_size = cfg.get('train_size')
    if train_size is not None:
        n = min(int(train_size), len(train_full))
        train_ds = Subset(train_full, list(range(n)))
    else:
        n = len(train_full)
        train_ds = train_full
    batch_size = int(cfg.get('batch_size', 64))
    if batch_size > n:
        batch_size = n
    generator = make_generator(seed)
    opts = _loader_options(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=opts['pin_memory'],
        num_workers=opts['num_workers'],
        worker_init_fn=worker_init_fn if seed is not None else None,
    )
    test_ratio = float(cfg.get('test_batch_ratio', 1) or 1)
    test_batch_size = max(int(batch_size * test_ratio), 1)
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=opts['pin_memory'],
        num_workers=opts['num_workers'],
    )
    data = Data(
        name=name,
        source='torch',
        loaders={'train': train_loader, 'test': test_loader},
        meta={
            'ready': True,
            'assets_dir': str(assets_dir),
            'config': cfg,
            'train_size': n,
            'test_size': len(test_ds),
            'batch_size': batch_size,
            'test_batch_size': test_batch_size,
            'data_size': list(spec['data_size']),
            'target_size': int(spec['target_size']),
            'mean': tuple(spec['mean']),
            'std': tuple(spec['std']),
            'train_aug': spec.get('train_aug') if augment else None,
            'augment': augment,
            'pin_memory': opts['pin_memory'],
            'num_workers': opts['num_workers'],
            'dtype': str(torch.float32),
            'seed': seed,
        },
    )
    data._train_set = train_ds
    return data


for _name in ('Toy', *_VISION_TORCH):
    DataRegistry.register(_name, 'stub', _build_stub)
for _name in _VISION_TORCH:
    DataRegistry.register(_name, 'torch', _build_torch_vision)
