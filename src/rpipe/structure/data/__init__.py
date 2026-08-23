"""Structure data layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def prepare_data(control_data: dict[str, Any], assets_dir) -> dict[str, Any]:
    """Land data handles for prepare.

    MNIST: download/cache under Study ``shared/data`` (passed as assets root),
    optional train subset via ``data.config.train_size``.
    """
    name = control_data.get('name', 'unknown')
    cfg = dict(control_data.get('config') or {})
    out: dict[str, Any] = {
        'name': name,
        'ready': True,
        'assets_dir': str(assets_dir),
        'config': cfg,
    }
    if name == 'MNIST':
        out.update(_prepare_mnist(cfg, Path(assets_dir)))
    return out


def _prepare_mnist(cfg: dict[str, Any], assets_dir: Path) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        'train_size': n,
        'test_size': len(test_ds),
        'batch_size': batch_size,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'dtype': str(torch.float32),
    }
