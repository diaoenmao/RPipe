"""data_api: Data, DataFactory.build, DataConfig."""

from __future__ import annotations

from pathlib import Path

from rpipe.structure.data import Data, DataConfig, DataFactory, DataRegistry
from rpipe.structure.data.prepare import prepare_shared_data


def build(data_config: DataConfig, assets_dir, seed=None, origin=None):
    return DataFactory.build(data_config, assets_dir, seed=seed, origin=origin)


def prepare_shared(study_dir: Path | str, config_paths: list[Path]) -> list[str]:
    """Study-level shared data. Call from make, before any run-one."""
    return prepare_shared_data(study_dir, config_paths)
