"""data_api: Data, DataFactory.build, DataConfig."""

from rpipe.structure.data import Data, DataConfig, DataFactory, DataRegistry


def build(data_config: DataConfig, assets_dir, seed=None):
    return DataFactory.build(data_config, assets_dir, seed=seed)
