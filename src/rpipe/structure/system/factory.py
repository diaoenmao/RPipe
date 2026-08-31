"""Runtime System object, registry, factory, and Logger."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from rpipe.structure.system.config import SystemConfig
from rpipe.structure.system.logger import Logger


class System:
    def __init__(
        self,
        *,
        device: str,
        logger: Logger,
        assets_dir: Path,
        meta: dict[str, Any],
    ) -> None:
        self.device = device
        self.logger = logger
        self.assets_dir = assets_dir
        self.meta = meta

    def place_module(self, module: Any) -> Any:
        if module is None:
            return None
        import torch

        return module.to(torch.device(self.device))

    def to_result_snapshot(self) -> dict[str, Any]:
        return {'device': self.device, **self.meta}


class SystemRegistry:
    _items: dict[str, Callable[..., System]] = {}

    @classmethod
    def register(cls, source: str, builder: Callable[..., System]) -> None:
        cls._items[source] = builder

    @classmethod
    def get(cls, source: str) -> Callable[..., System] | None:
        return cls._items.get(source)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._items)


class SystemFactory:
    @staticmethod
    def build(system_config: SystemConfig, assets_dir: Path | str) -> System:
        source = system_config.source or 'native'
        builder = SystemRegistry.get(source) or _build_native
        return builder(system_config, Path(assets_dir))


def _build_native(system_config: SystemConfig, assets_dir: Path) -> System:
    device = str(system_config.setting('device', 'cpu'))
    logger = Logger(assets_dir)
    return System(
        device=device,
        logger=logger,
        assets_dir=assets_dir,
        meta={'ready': True, 'source': system_config.source or 'native'},
    )


SystemRegistry.register('native', _build_native)
