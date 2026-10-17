"""system_api: System, Logger, SystemFactory.build, SystemConfig."""

from rpipe.structure.system import (
    Logger,
    System,
    SystemConfig,
    SystemFactory,
    apply_runtime,
    make_generator,
    worker_init_fn,
)

__all__ = [
    'DOMESTIC', 'DOMESTIC_MODEL_HUB', 'DownloadEndpoint', 'FOREIGN', 'FOREIGN_MODEL_HUB', 'Logger',
    'System', 'SystemConfig', 'SystemFactory', 'apply_model_origin', 'apply_runtime',
    'bind_endpoint', 'build', 'discard_bad_archive', 'domestic_endpoint', 'download_model',
    'endpoint_for', 'foreign_endpoint', 'import_torchvision', 'load_hf_dataset', 'make_generator',
    'model_hub', 'normalize_origin', 'worker_init_fn',
]


def build(system_config: SystemConfig, assets_dir):
    return SystemFactory.build(system_config, assets_dir)



def _checkpoint_pickle_module(assets_dir):
    from pathlib import Path
    from rpipe.structure.api.artifact_api import ASSETS_DIRNAME, RUNS_DIRNAME
    import sys

    recipe = sys.modules.get('rpipe.structure.make.recipe')
    if recipe is None:
        return None
    root = Path(assets_dir).resolve()
    if Path(root.name) != Path(ASSETS_DIRNAME) or Path(root.parent.parent.name) != Path(RUNS_DIRNAME):
        return None
    return recipe._recipe_pickle_module(root.parent.parent.parent)

from rpipe.structure.system.origin import (
    DOMESTIC, DOMESTIC_MODEL_HUB, DownloadEndpoint, FOREIGN, FOREIGN_MODEL_HUB, apply_model_origin,
    bind_endpoint, discard_bad_archive, domestic_endpoint, download_model, endpoint_for,
    foreign_endpoint, load_hf_dataset, model_hub, normalize_origin,
)
from rpipe.structure.system.torchvision_load import import_torchvision
