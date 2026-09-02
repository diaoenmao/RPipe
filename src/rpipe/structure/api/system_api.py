"""system_api: System, Logger, SystemFactory.build, SystemConfig."""

from rpipe.structure.system import (
    Logger,
    System,
    SystemConfig,
    SystemFactory,
    SystemRegistry,
    apply_runtime,
    make_generator,
    worker_init_fn,
)

__all__ = [
    'Logger',
    'System',
    'SystemConfig',
    'SystemFactory',
    'SystemRegistry',
    'apply_runtime',
    'build',
    'make_generator',
    'worker_init_fn',
]


def build(system_config: SystemConfig, assets_dir):
    return SystemFactory.build(system_config, assets_dir)
