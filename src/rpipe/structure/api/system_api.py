"""system_api: System, Logger, SystemFactory.build, SystemConfig."""

from rpipe.structure.system import Logger, System, SystemConfig, SystemFactory, SystemRegistry


def build(system_config: SystemConfig, assets_dir):
    return SystemFactory.build(system_config, assets_dir)
