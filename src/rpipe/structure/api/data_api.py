"""data_api: Data, DataFactory.build, DataConfig."""

from __future__ import annotations

from pathlib import Path

from rpipe.structure.data import Data, DataConfig, DataFactory, DataRegistry
from rpipe.structure.data.prepare import prepare_shared_data


def build(data_config: DataConfig, assets_dir, seed=None, origin=None, *, profile_path=None):
    return DataFactory.build(data_config, assets_dir, seed=seed, origin=origin, profile_path=profile_path)


def prepare_shared(study_dir: Path | str, config_paths: list[Path]) -> list[str]:
    """Project-shared data. Call from make, before any run-one."""
    return prepare_shared_data(study_dir, config_paths)

from rpipe.structure.api.system_api import load_hf_dataset

from rpipe.structure.data.factory import vision_root
from rpipe.structure.data.profile import profile_study
