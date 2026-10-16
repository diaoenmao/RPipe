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
    'Logger',
    'System',
    'SystemConfig',
    'SystemFactory',
    'apply_runtime',
    'build',
    'make_generator',
    'worker_init_fn',
]


def build(system_config: SystemConfig, assets_dir):
    return SystemFactory.build(system_config, assets_dir)



def _checkpoint_pickle_module(assets_dir):
    from pathlib import Path
    from rpipe.structure.artifact.paths import ASSETS_DIRNAME, RUNS_DIRNAME
    import sys

    recipe = sys.modules.get('rpipe.structure.make.recipe')
    if recipe is None:
        return None
    root = Path(assets_dir).resolve()
    if Path(root.name) != Path(ASSETS_DIRNAME) or Path(root.parent.parent.name) != Path(RUNS_DIRNAME):
        return None
    return recipe._recipe_pickle_module(root.parent.parent.parent)
