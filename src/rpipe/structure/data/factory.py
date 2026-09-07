"""Runtime Data object, registry, and factory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

from rpipe.structure.data.config import DataConfig


class Data:
    def __init__(
        self,
        *,
        name: str,
        source: str,
        loaders: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        self.name = name
        self.source = source
        self._loaders = loaders
        self.meta = meta
        self._train_set = None

    def iter_batches(self, split: str) -> Iterator[Any]:
        loader = self._loaders.get(split)
        if loader is None:
            return iter(())
        return iter(loader)

    def steps_per_epoch(self, split: str = 'train') -> int | None:
        loader = self._loaders.get(split)
        if loader is not None:
            try:
                value = int(len(loader))
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
        if split != 'train':
            return None
        train_size = self.meta.get('train_size')
        batch_size = self.meta.get('batch_size')
        if train_size and batch_size:
            import math

            return max(int(math.ceil(int(train_size) / int(batch_size))), 1)
        return None

    def rebind_train_steps(
        self,
        *,
        step: int,
        num_steps: int,
        step_period: int = 1,
    ) -> None:
        """Rebuild train loader for remaining optimizer steps (main ``make_data_loader``).

        ``num_samples = batch_size * (num_steps - step) * step_period``. Generator is
        re-seeded from ``meta['seed']`` each time, so a mid-run resume sees the
        **prefix** of the same shuffle, not the unseen suffix (same as main).
        """
        dataset = self._train_set
        if dataset is None:
            return
        batch_size = int(self.meta.get('batch_size') or getattr(self._loaders.get('train'), 'batch_size', 1) or 1)
        period = max(int(step_period), 1)
        remaining = max(int(num_steps) - int(step), 0)
        num_samples = batch_size * remaining * period
        self._loaders['train'] = _train_loader_for_samples(
            dataset,
            batch_size=batch_size,
            num_samples=num_samples,
            seed=self.meta.get('seed'),
        )

    def to_result_snapshot(self) -> dict[str, Any]:
        out = {'name': self.name, 'source': self.source, **self.meta}
        out.pop('train_loader', None)
        out.pop('test_loader', None)
        return out


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
    def build(data_config: DataConfig, assets_dir: Path | str, seed: int | None = None) -> Data:
        name = data_config.name or 'unknown'
        source = data_config.source
        if source is None and name in _VISION_TORCH:
            source = 'torch'
        if source is None:
            source = 'stub'
        builder = DataRegistry.get(name, source)
        if builder is None:
            builder = DataRegistry.get(name, 'stub')
        if builder is None:
            return _build_stub(data_config, Path(assets_dir))
        return builder(data_config, Path(assets_dir), seed=seed)


def _empty_train_loader(dataset: Any, batch_size: int) -> Any:
    class _Empty:
        def __len__(self) -> int:
            return 0

        def __iter__(self):
            return iter(())

    loader = _Empty()
    loader.dataset = dataset
    loader.batch_size = batch_size
    return loader


def _train_loader_for_samples(
    dataset: Any,
    *,
    batch_size: int,
    num_samples: int,
    seed: int | None,
) -> Any:
    from torch.utils.data import DataLoader, RandomSampler

    from rpipe.structure.system.runtime import make_generator, worker_init_fn

    if num_samples <= 0:
        return _empty_train_loader(dataset, batch_size)
    generator = make_generator(seed)
    sampler = RandomSampler(
        dataset,
        replacement=False,
        num_samples=int(num_samples),
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=sampler,
        worker_init_fn=worker_init_fn if seed is not None else None,
    )


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


def _train_transforms(spec: dict[str, Any], *, augment: bool) -> Any:
    from torchvision import transforms

    ops: list[Any] = []
    spatial = int(spec['data_size'][-1])
    if augment and spec.get('train_aug') == 'cifar':
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(spatial, padding=4, padding_mode='reflect'),
            ]
        )
    elif augment and spec.get('train_aug') == 'svhn':
        ops.append(transforms.RandomCrop(spatial, padding=4, padding_mode='reflect'))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(spec['mean'], spec['std']),
        ]
    )
    return transforms.Compose(ops)


def _eval_transforms(spec: dict[str, Any]) -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(spec['mean'], spec['std']),
        ]
    )


def _build_torch_vision(data_config: DataConfig, assets_dir: Path, seed: int | None = None) -> Data:
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets

    from rpipe.structure.system.runtime import make_generator, worker_init_fn

    name = data_config.name or 'unknown'
    spec = _VISION_TORCH[name]
    cfg = dict(data_config.config)
    root = assets_dir / str(spec['root'])
    root.mkdir(parents=True, exist_ok=True)
    ctor = getattr(datasets, str(spec['ctor']))
    augment = bool(cfg.get('augment', spec.get('train_aug') is not None))
    train_tf = _train_transforms(spec, augment=augment)
    test_tf = _eval_transforms(spec)
    train_full = ctor(root=str(root), download=True, transform=train_tf, **dict(spec['train_kw']))
    test_ds = ctor(root=str(root), download=True, transform=test_tf, **dict(spec['test_kw']))
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
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=worker_init_fn if seed is not None else None,
    )
    test_ratio = float(cfg.get('test_batch_ratio', 1) or 1)
    test_batch_size = max(int(batch_size * test_ratio), 1)
    test_loader = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False)
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
            'augment': augment,
            'dtype': str(torch.float32),
            'seed': seed,
        },
    )
    data._train_set = train_ds
    return data


DataRegistry.register('Toy', 'stub', _build_stub)
DataRegistry.register('MNIST', 'stub', _build_stub)
DataRegistry.register('CIFAR10', 'stub', _build_stub)
DataRegistry.register('SVHN', 'stub', _build_stub)
DataRegistry.register('MNIST', 'torch', _build_torch_vision)
DataRegistry.register('CIFAR10', 'torch', _build_torch_vision)
DataRegistry.register('SVHN', 'torch', _build_torch_vision)
