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
        if source is None and name == 'MNIST':
            source = 'torch'
        if source is None:
            source = 'stub'
        builder = DataRegistry.get(name, source)
        if builder is None:
            builder = DataRegistry.get(name, 'stub')
        if builder is None:
            return _build_stub(data_config, Path(assets_dir))
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


def _build_mnist(data_config: DataConfig, assets_dir: Path, seed: int | None = None) -> Data:
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    from rpipe.structure.system.runtime import make_generator, worker_init_fn

    cfg = dict(data_config.config)
    root = assets_dir / 'mnist'
    root.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_full = datasets.MNIST(root=str(root), train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=str(root), train=False, download=True, transform=transform)
    train_size = cfg.get('train_size')
    if train_size is not None:
        n = min(int(train_size), len(train_full))
        train_ds = Subset(train_full, list(range(n)))
    else:
        n = len(train_full)
        train_ds = train_full
    batch_size = int(cfg.get('batch_size', 64))
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
    return Data(
        name='MNIST',
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
            'dtype': str(torch.float32),
            'seed': seed,
        },
    )


DataRegistry.register('Toy', 'stub', _build_stub)
DataRegistry.register('MNIST', 'stub', _build_stub)
DataRegistry.register('MNIST', 'torch', _build_mnist)
