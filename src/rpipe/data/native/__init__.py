"""data/native — built-in torchvision-style datasets."""

from __future__ import annotations

from rpipe.data.dataset import make_dataset as _make_dataset
from rpipe.provider import register


@register('data', 'native')
class NativeDataProvider:
    name = 'native'
    package = 'rpipe'

    def available(self) -> bool:
        return True

    def list_datasets(self) -> list[str]:
        from rpipe.config.registry import DATASET_REGISTRY
        return DATASET_REGISTRY.keys()

    def build(self, data_name: str, *, process: bool = False, verbose: bool = True, **kwargs):
        return _make_dataset(data_name, process=process, verbose=verbose, **kwargs)
